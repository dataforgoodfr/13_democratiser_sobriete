"""
Create maps for physicians density in Switzerland at NUTS level.

This script creates visualizations showing:
- Physicians per 100,000 inhabitants by NUTS2 regions
- Three maps: most recent year (2024), 10 years before (2014), and 20 years before (2004)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

try:
    import geopandas as gpd
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase
    import matplotlib.gridspec as gridspec
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not available - will use simplified maps")

# Add parent directories to path for shared utilities
SCRIPT_DIR = Path(__file__).parent.resolve()
REPORTS_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPORTS_DIR / "shared" / "code"))

# Directories
EXTERNAL_DATA_DIR = SCRIPT_DIR.parent / "external_data"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "graphs" / "EU-SILC"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find shapefile - check multiple locations
SHAPEFILE_CANDIDATES = [
    EXTERNAL_DATA_DIR / "0_shapefile" / "NUTS_RG_10M_2024_3035.gpkg",
    SCRIPT_DIR.parent.parent / "1_switzerland_vs_eu27_housing_energy" / "external_data" / "0_shapefile" / "NUTS_RG_10M_2024_3035.gpkg",
]

SHAPEFILE_PATH = None
for candidate in SHAPEFILE_CANDIDATES:
    if candidate.exists():
        SHAPEFILE_PATH = candidate
        print(f"✓ Found NUTS shapefile: {SHAPEFILE_PATH}")
        break

if SHAPEFILE_PATH is None:
    print("ERROR: Could not find NUTS shapefile at any expected location")
    for candidate in SHAPEFILE_CANDIDATES:
        print(f"  Checked: {candidate}")

# Data paths
PHYSICIANS_DATA_PATH = EXTERNAL_DATA_DIR / "eurostat_physicians.csv"

# Swiss regions mapping (EUROSTAT names to NUTS codes)
SWISS_REGIONS_MAPPING = {
    'Région lémanique': 'CH01',
    'Espace Mittelland': 'CH02',
    'Nordwestschweiz': 'CH03', 
    'Zürich': 'CH04',
    'Ostschweiz': 'CH05',
    'Zentralschweiz': 'CH06',
    'Ticino': 'CH07'
}

# Target years for the maps (based on available data)
TARGET_YEARS = [2023, 2013, 2003]

def load_and_process_physicians_data():
    """Load physicians data and process it for Swiss regions."""
    print("Loading physicians data...")
    
    if not PHYSICIANS_DATA_PATH.exists():
        raise FileNotFoundError(f"Physicians data not found at: {PHYSICIANS_DATA_PATH}")
    
    # Load data
    df = pd.read_csv(PHYSICIANS_DATA_PATH)
    print(f"✓ Loaded {len(df)} rows of physician data")
    
    # Check columns
    required_cols = ['geo', 'TIME_PERIOD', 'OBS_VALUE']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert TIME_PERIOD to int and OBS_VALUE to float
    df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    
    # Filter for Swiss regions and target years
    swiss_regions = list(SWISS_REGIONS_MAPPING.keys())
    
    # Check which Swiss regions are available in the data
    available_regions = df['geo'].unique()
    print(f"Available regions in data: {sorted([r for r in available_regions if any(s in r for s in ['Swiss', 'Schweiz', 'Zurich', 'Ticino', 'Basel', 'Genf', 'Bern', 'Central', 'Espace', 'Nord', 'Ost', 'Zentral'])])}")
    
    # Filter for Swiss regions (try different name patterns)
    swiss_data = df[df['geo'].isin(swiss_regions)].copy()
    
    if swiss_data.empty:
        # Try alternative Swiss region names
        print("No exact matches found, searching for Swiss regions by pattern...")
        swiss_pattern_regions = df[df['geo'].str.contains('schweiz|Swiss|Zurich|Ticino|Basel|Genf|Bern|Espace|Zentral|Ost|Nord', 
                                                         case=False, na=False)]['geo'].unique()
        print(f"Found Swiss regions by pattern: {sorted(swiss_pattern_regions)}")
        swiss_data = df[df['geo'].isin(swiss_pattern_regions)].copy()
    
    if swiss_data.empty:
        raise ValueError("No Swiss regional data found in the dataset")
    
    print(f"✓ Found data for Swiss regions: {sorted(swiss_data['geo'].unique())}")
    
    # Filter for target years
    swiss_data = swiss_data[swiss_data['TIME_PERIOD'].isin(TARGET_YEARS)].copy()
    
    if swiss_data.empty:
        print(f"No data found for target years {TARGET_YEARS}")
        # Check available years for Swiss regions
        all_swiss = df[df['geo'].isin(swiss_regions)]
        available_years = sorted(all_swiss['TIME_PERIOD'].dropna().unique())
        print(f"Available years: {available_years}")
        # Use closest years if exact years not available
        TARGET_YEARS_ACTUAL = []
        for target_year in TARGET_YEARS:
            closest_year = min(available_years, key=lambda x: abs(x - target_year))
            TARGET_YEARS_ACTUAL.append(closest_year)
        
        print(f"Using closest available years: {TARGET_YEARS_ACTUAL}")
        swiss_data = df[df['geo'].isin(swiss_regions) & df['TIME_PERIOD'].isin(TARGET_YEARS_ACTUAL)].copy()
    
    # Add NUTS codes
    swiss_data['nuts_code'] = swiss_data['geo'].map(SWISS_REGIONS_MAPPING)
    
    # Drop rows without NUTS codes (regions not in our mapping)
    swiss_data = swiss_data.dropna(subset=['nuts_code'])
    
    print(f"✓ Processed data for {len(swiss_data)} region-year combinations")
    print(f"  Years: {sorted(swiss_data['TIME_PERIOD'].unique())}")
    print(f"  Regions: {sorted(swiss_data['geo'].unique())}")
    
    return swiss_data


def create_excel_export(data, output_dir):
    """Export processed data to Excel for further analysis."""
    print("\nExporting data to Excel...")
    
    # Prepare data for Excel export
    excel_data = []
    
    for _, row in data.iterrows():
        excel_data.append({
            'country_code': 'CHE',
            'country_name': 'Switzerland', 
            'nuts_code': row['nuts_code'],
            'region_name': row['geo'],
            'indicator': 'Physicians per 100k inhabitants',
            'visual_name': f"Physicians density - {row['geo']} ({int(row['TIME_PERIOD'])})",
            'year': int(row['TIME_PERIOD']),
            'decile': None,  # NaN for regional data
            'value': row['OBS_VALUE'],
            'unit': 'Per 100,000 inhabitants'
        })
    
    # Convert to DataFrame
    excel_df = pd.DataFrame(excel_data)
    
    # Export to Excel
    excel_path = output_dir / "switzerland_physicians_density_data.xlsx"
    excel_df.to_excel(excel_path, index=False, sheet_name='Physicians_Density_Data')
    print(f"✓ Excel data exported: {excel_path}")
    
    return excel_path


def create_swiss_physicians_map_with_shapefile(data, output_path):
    """
    Create three choropleth maps for Swiss NUTS regions showing physicians density.
    Maps for 2023, 2013, and 2003 (or closest available years).
    """
    print(f"\n=== Starting map creation for physicians density ===")
    print(f"GEOPANDAS_AVAILABLE: {GEOPANDAS_AVAILABLE}")
    print(f"SHAPEFILE_PATH: {SHAPEFILE_PATH}")
    
    if not GEOPANDAS_AVAILABLE:
        print("ERROR: geopandas not available, cannot create geographic maps")
        return create_swiss_physicians_map_simplified(data, output_path)
    
    if SHAPEFILE_PATH is None or not Path(SHAPEFILE_PATH).exists():
        print(f"ERROR: Shapefile not available at {SHAPEFILE_PATH}")
        return create_swiss_physicians_map_simplified(data, output_path)
    
    print(f"✓ All checks passed, loading shapefile...")
    
    # Load NUTS shapefile
    try:
        print(f"  Loading shapefile from: {SHAPEFILE_PATH}")
        gdf = gpd.read_file(SHAPEFILE_PATH, layer='NUTS_RG_10M_2024_3035')
        print(f"  Loaded {len(gdf)} regions from shapefile")
    except Exception as e:
        print(f"  Failed to load with layer name, trying without: {e}")
        try:
            gdf = gpd.read_file(SHAPEFILE_PATH)
            print(f"  Loaded {len(gdf)} regions from shapefile (no layer)")
        except Exception as e2:
            print(f"  ERROR loading shapefile: {e2}")
            return create_swiss_physicians_map_simplified(data, output_path)
    
    # Filter for Swiss NUTS level 2
    print(f"  Filtering for Swiss NUTS-2 regions...")
    print(f"  Available columns: {gdf.columns.tolist()}")
    
    gdf_swiss = gdf[
        (gdf['NUTS_ID'].str.startswith('CH')) &
        (gdf['NUTS_ID'].str.len() == 4) &
        (gdf['LEVL_CODE'] == 2)
    ].copy()
    
    if gdf_swiss.empty:
        print("ERROR: No Swiss NUTS-2 regions found in shapefile")
        print(f"  Available NUTS codes starting with CH: {gdf[gdf['NUTS_ID'].str.startswith('CH')]['NUTS_ID'].unique()}")
        return create_swiss_physicians_map_simplified(data, output_path)
    
    print(f"✓ Found {len(gdf_swiss)} Swiss NUTS-2 regions in shapefile")
    print(f"  NUTS codes: {sorted(gdf_swiss['NUTS_ID'].tolist())}")
    
    # Convert to appropriate CRS for display
    if gdf_swiss.crs != 'EPSG:3035':
        print(f"  Converting CRS from {gdf_swiss.crs} to EPSG:3035")
        gdf_swiss = gdf_swiss.to_crs(epsg=3035)
    else:
        print(f"  CRS already EPSG:3035")
    
    print(f"\n✓ Creating choropleth maps with actual geographic boundaries...")
    
    # Create figure with 1x3 subplots + space for colorbar
    fig = plt.figure(figsize=(20, 8))
    
    # Create grid spec for subplots and colorbar
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], 
                          hspace=0.2, wspace=0.25, left=0.05, right=0.92, 
                          top=0.88, bottom=0.12)
    
    # Add title
    fig.suptitle(
        f"Physicians per 100,000 inhabitants by NUTS2 regions in Switzerland\n"
        f"(EUROSTAT data)",
        fontsize=18, fontweight='bold'
    )
    
    # Get available years from data
    available_years = sorted(data['TIME_PERIOD'].unique())
    print(f"Available years for maps: {available_years}")
    
    # Use target years if available, otherwise closest available years
    years_to_plot = []
    for target_year in TARGET_YEARS:
        if target_year in available_years:
            years_to_plot.append(target_year)
        else:
            # Find closest available year to target year
            closest_year = min(available_years, key=lambda x: abs(x - target_year))
            if closest_year not in years_to_plot:  # Avoid duplicates
                years_to_plot.append(closest_year)
    
    # Ensure we have exactly 3 unique years
    years_to_plot = list(dict.fromkeys(years_to_plot))[:3]  # Remove duplicates and limit to 3
    years_to_plot = sorted(years_to_plot)  # Sort chronologically
    
    print(f"Years to plot: {years_to_plot}")
    
    # Use a colormap suitable for healthcare data
    colormap = 'RdYlGn'  # Red (low) to Yellow to Green (high) for healthcare density
    
    # Get global min/max for consistent color scaling across all maps
    vmin = data['OBS_VALUE'].min()
    vmax = data['OBS_VALUE'].max()
    
    print(f"Value range: {vmin:.1f} to {vmax:.1f}")
    
    # Create normalization for colormap
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(colormap)
    
    # Create subplots for each year
    axes = []
    for col in range(len(years_to_plot)):
        ax = fig.add_subplot(gs[0, col])
        axes.append(ax)
    
    # Plot each year
    for idx, year in enumerate(years_to_plot):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Filter data for this year
        year_data = data[data['TIME_PERIOD'] == year].copy()
        
        if year_data.empty:
            print(f"No data for year {year}")
            continue
        
        print(f"Plotting year {year} with {len(year_data)} data points")
        
        # Merge data with shapefile
        gdf_plot = gdf_swiss.merge(
            year_data[['nuts_code', 'OBS_VALUE']],
            left_on='NUTS_ID',
            right_on='nuts_code',
            how='left'
        )
        
        print(f"  Merged data: {len(gdf_plot)} regions, {gdf_plot['OBS_VALUE'].notna().sum()} with data")
        
        # Plot the choropleth map
        gdf_plot.plot(
            column='OBS_VALUE',
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            legend=False,  # We'll add a shared colorbar
            edgecolor='black',
            linewidth=0.5,
            missing_kwds={'color': 'lightgray', 'alpha': 0.7}
        )
        
        # Set title and styling
        ax.set_title(f'{int(year)}', fontsize=14, fontweight='bold', pad=15)
        ax.set_axis_off()
        
        # Add region labels if requested
        for idx_region, row in gdf_plot.iterrows():
            if pd.notna(row['OBS_VALUE']):
                # Get centroid for labeling
                centroid = row.geometry.centroid
                ax.annotate(
                    f"{row['OBS_VALUE']:.0f}",
                    xy=(centroid.x, centroid.y),
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
                )
    
    # Add shared colorbar
    if axes:
        cbar_ax = fig.add_subplot(gs[0, -1])
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Physicians per 100,000 inhabitants', rotation=270, labelpad=20, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    
    # Add source note
    fig.text(0.02, 0.02, 
             'Source: EUROSTAT (HLTH_RS_PHYSREG)\nNote: Gray areas indicate missing data',
             fontsize=10, style='italic')
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Map saved: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return output_path


def create_swiss_physicians_map_simplified(data, output_path):
    """
    Create simplified bar charts when geographic mapping is not available.
    """
    print("Creating simplified bar chart visualization...")
    
    # Get available years
    available_years = sorted(data['TIME_PERIOD'].unique())
    
    # Create figure
    fig, axes = plt.subplots(1, len(available_years), figsize=(5*len(available_years), 8))
    if len(available_years) == 1:
        axes = [axes]
    
    fig.suptitle(
        'Physicians per 100,000 inhabitants by Swiss regions\n(EUROSTAT data)',
        fontsize=16, fontweight='bold'
    )
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(data['geo'].unique())))
    
    for idx, year in enumerate(available_years):
        ax = axes[idx]
        year_data = data[data['TIME_PERIOD'] == year].sort_values('OBS_VALUE', ascending=True)
        
        bars = ax.barh(year_data['geo'], year_data['OBS_VALUE'], color=colors[:len(year_data)])
        
        # Add value labels on bars
        for i, (region, value) in enumerate(zip(year_data['geo'], year_data['OBS_VALUE'])):
            ax.text(value + max(year_data['OBS_VALUE']) * 0.01, i, f'{value:.0f}', 
                   va='center', ha='left', fontsize=10)
        
        ax.set_title(f'{int(year)}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Physicians per 100,000 inhabitants')
        ax.grid(axis='x', alpha=0.3)
        
        # Adjust layout
        ax.tick_params(axis='y', labelsize=9)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Simplified chart saved: {output_path}")
    plt.close(fig)
    
    return output_path


def main():
    """Main function to create physicians density maps."""
    print("=== EUROSTAT Physicians Analysis for Switzerland ===\n")
    
    try:
        # Load and process data
        data = load_and_process_physicians_data()
        
        # Export to Excel
        excel_path = create_excel_export(data, OUTPUT_DIR)
        
        # Create maps
        if GEOPANDAS_AVAILABLE and SHAPEFILE_PATH:
            map_output_path = OUTPUT_DIR / "switzerland_physicians_density_maps.png"
            create_swiss_physicians_map_with_shapefile(data, map_output_path)
        else:
            map_output_path = OUTPUT_DIR / "switzerland_physicians_density_charts.png"
            create_swiss_physicians_map_simplified(data, map_output_path)
        
        print(f"\n✓ Analysis completed successfully!")
        print(f"  Data exported: {excel_path}")
        print(f"  Maps saved: {map_output_path}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
