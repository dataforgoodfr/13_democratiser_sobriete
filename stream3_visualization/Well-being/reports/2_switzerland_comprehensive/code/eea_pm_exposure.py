"""
Create maps for particulate matter and NO2 exposure in Switzerland at NUTS level.

This script creates visualizations showing:
- Attributable deaths (AD) per 100k population
- Years of Life Lost (YLL) per 100k population

For both PM2.5 and NO2 pollutants in years 2013 and 2023.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

try:
    import geopandas as gpd
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
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "graphs" / "particulate"

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
    print("✗ WARNING: NUTS shapefile not found. Searched locations:")
    for candidate in SHAPEFILE_CANDIDATES:
        print(f"  - {candidate}")
    print("  Will use simplified visualization instead of geographic maps.")

if not GEOPANDAS_AVAILABLE:
    print("✗ WARNING: geopandas not installed. Cannot create geographic maps.")
    print("  Install with: pip install geopandas")
elif SHAPEFILE_PATH:
    print("✓ Geopandas available - will create geographic choropleth maps")


def load_eea_pm_data():
    """Load the EEA particulate matter exposure data."""
    csv_path = EXTERNAL_DATA_DIR / "eea_pm_exposure.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def prepare_data_for_mapping(df, health_indicator, scenario="Baseline from WHO 2021 AQG"):
    """
    Filter and prepare data for mapping.
    
    Parameters:
    -----------
    df : DataFrame
        Raw data from EEA
    health_indicator : str
        Either 'Attributable deaths (AD)' or 'Years of Life Lost (YLL)'
    scenario : str
        The scenario to use for filtering
    
    Returns:
    --------
    DataFrame with columns: nuts_code, year, air_pollutant, value_per_100k
    """
    # Filter for the scenario and health indicator
    filtered = df[
        (df['Scenario'] == scenario) &
        (df['Health Indicator'] == health_indicator)
    ].copy()
    
    # Filter for NUTS level 2 regions (CH01, CH02, etc.) - exclude CH0 and CH
    filtered = filtered[
        (filtered['NUTS Code'].str.startswith('CH')) &
        (filtered['NUTS Code'].str.len() == 4) &
        (filtered['NUTS Code'] != 'CH0')
    ].copy()
    
    # Filter for years 2013 and 2023
    filtered = filtered[filtered['Year'].isin([2013, 2023])].copy()
    
    # Filter for PM2.5 and NO2
    filtered = filtered[filtered['Air Pollutant'].isin(['PM2.5', 'NO2'])].copy()
    
    # Select relevant columns
    result = filtered[[
        'NUTS Code', 'NUTS Name', 'Year', 'Air Pollutant',
        'Value for 100k Of Affected Population'
    ]].copy()
    
    result.columns = ['nuts_code', 'nuts_name', 'year', 'air_pollutant', 'value_per_100k']
    
    print(f"\n{health_indicator} - Filtered to {len(result)} rows")
    print(f"NUTS codes: {sorted(result['nuts_code'].unique())}")
    print(f"Years: {sorted(result['year'].unique())}")
    print(f"Pollutants: {sorted(result['air_pollutant'].unique())}")
    
    return result


def export_data_to_excel(data, health_indicator, output_dir):
    """
    Export the processed data to Excel format with standardized columns.
    
    Parameters:
    -----------
    data : DataFrame
        Processed data with columns: nuts_code, nuts_name, year, air_pollutant, value_per_100k
    health_indicator : str
        The health indicator being processed (for filename)
    output_dir : Path
        Directory to save the Excel file
    """
    # Create Excel export data with required columns
    excel_data = []
    
    for _, row in data.iterrows():
        visual_name = f"{'Premature deaths' if 'AD' in health_indicator else 'Years of Life Lost'} due to {row['air_pollutant']} exposure in {row['nuts_name']}"
        
        excel_data.append({
            'visual_number': None,  # NaN as requested
            'visual_name': visual_name,
            'year': row['year'],
            'decile': None,  # NaN as requested  
            'value': row['value_per_100k'],
            'unit': 'Per 100,000 affected population'
        })
    
    # Convert to DataFrame
    excel_df = pd.DataFrame(excel_data)
    
    # Export to Excel
    indicator_short = 'AD' if 'AD' in health_indicator else 'YLL'
    excel_path = output_dir / f"switzerland_pm_exposure_{indicator_short}_data.xlsx"
    
    excel_df.to_excel(excel_path, index=False, sheet_name='PM_Exposure_Data')
    print(f"✓ Excel data exported: {excel_path}")
    
    return excel_path


def create_swiss_nuts_map_with_shapefile(data, health_indicator, output_path):
    """
    Create a 2x2 grid of choropleth maps for Swiss NUTS regions using geopandas.
    With a single shared colorbar for all 4 maps.
    
    Grid layout:
    - Columns: Years (2013, 2023)
    - Rows: Air Pollutants (PM2.5, NO2)
    """
    print(f"\n=== Starting map creation for {health_indicator} ===")
    print(f"GEOPANDAS_AVAILABLE: {GEOPANDAS_AVAILABLE}")
    print(f"SHAPEFILE_PATH: {SHAPEFILE_PATH}")
    
    if not GEOPANDAS_AVAILABLE:
        print("ERROR: geopandas not available, using simplified visualization")
        return create_swiss_nuts_map_simplified(data, health_indicator, output_path)
    
    if SHAPEFILE_PATH is None:
        print("ERROR: SHAPEFILE_PATH is None, using simplified visualization")
        return create_swiss_nuts_map_simplified(data, health_indicator, output_path)
        
    if not Path(SHAPEFILE_PATH).exists():
        print(f"ERROR: Shapefile does not exist at {SHAPEFILE_PATH}")
        return create_swiss_nuts_map_simplified(data, health_indicator, output_path)
    
    print(f"✓ All checks passed, loading shapefile...")
    
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase
    import numpy as np
    
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
            return create_swiss_nuts_map_simplified(data, health_indicator, output_path)
    
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
        return create_swiss_nuts_map_simplified(data, health_indicator, output_path)
    
    print(f"✓ Found {len(gdf_swiss)} Swiss NUTS-2 regions in shapefile")
    print(f"  NUTS codes: {sorted(gdf_swiss['NUTS_ID'].tolist())}")
    
    # Convert to appropriate CRS for display
    if gdf_swiss.crs != 'EPSG:3035':
        print(f"  Converting CRS from {gdf_swiss.crs} to EPSG:3035")
        gdf_swiss = gdf_swiss.to_crs(epsg=3035)
    else:
        print(f"  CRS already EPSG:3035")
    
    print(f"\n✓ Creating choropleth maps with actual geographic boundaries...")
    
    # Create figure with 2x2 subplots + space for colorbar
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid spec for subplots and colorbar
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], 
                          hspace=0.2, wspace=0.2, left=0.05, right=0.92, 
                          top=0.93, bottom=0.05)
    
    # Add title
    fig.suptitle(
        f"{'Premature deaths' if 'AD' in health_indicator else 'Years of Life Lost'} "
        f"due to air pollution exposure in Switzerland\n"
        f"({health_indicator}, per 100,000 affected population)",
        fontsize=16, fontweight='bold'
    )
    
    years = [2013, 2023]
    pollutants = ['PM2.5', 'NO2']
    
    # Use a single colormap for consistency
    colormap = 'YlOrRd'  # Yellow-Orange-Red for health impacts
    
    # Get global min/max for consistent color scaling across all maps
    vmin = data['value_per_100k'].min()
    vmax = data['value_per_100k'].max()
    
    print(f"Value range: {vmin:.1f} to {vmax:.1f}")
    
    # Create normalization for colormap
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(colormap)
    
    # Create subplots
    axes = []
    for row in range(2):
        for col in range(2):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
    
    # Plot each combination
    for idx in range(4):
        row_idx = idx // 2
        col_idx = idx % 2
        year = years[col_idx]
        pollutant = pollutants[row_idx]
        
        ax = axes[idx]
        
        # Filter data for this combination
        subset = data[
            (data['year'] == year) &
            (data['air_pollutant'] == pollutant)
        ].copy()
        
        # Merge data with shapefile
        gdf_plot = gdf_swiss.merge(
            subset[['nuts_code', 'value_per_100k']],
            left_on='NUTS_ID',
            right_on='nuts_code',
            how='left'
        )
        
        print(f"  Plotting {pollutant} {year}: {len(gdf_plot)} regions, {gdf_plot['value_per_100k'].notna().sum()} with data")
        
        # Plot with consistent color scaling - THIS IS THE GEOGRAPHIC MAP
        gdf_plot.plot(
            column='value_per_100k',
            ax=ax,
            cmap=colormap,
            edgecolor='black',
            linewidth=1.5,
            legend=False,
            vmin=vmin,
            vmax=vmax,
            missing_kwds={'color': 'lightgrey', 'edgecolor': 'black', 'linewidth': 1.5}
        )
        
        # Add region labels with values
        for idx_region, region in gdf_plot.iterrows():
            centroid = region.geometry.centroid
            value = region['value_per_100k']
            
            # Try to get region name from different possible columns
            nuts_name = None
            for col in ['NUTS_NAME', 'NAME_LATN', 'NAME']:
                if col in region and pd.notna(region[col]):
                    nuts_name = region[col]
                    break
            
            # Add value
            if pd.notna(value):
                ax.text(centroid.x, centroid.y, f'{value:.1f}',
                       ha='center', va='center', fontsize=11,
                       fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='white', alpha=0.9, 
                                edgecolor='black', linewidth=0.5))
        
        # Set title
        ax.set_title(f"{pollutant} - {year}", fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
    
    # Add single colorbar on the right
    cbar_ax = fig.add_subplot(gs[:, 2])
    cb = ColorbarBase(cbar_ax, cmap=cmap_obj, norm=norm, orientation='vertical')
    cb.set_label('Per 100,000 affected population', fontsize=12, fontweight='bold')
    cb.ax.tick_params(labelsize=10)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓✓✓ GEOGRAPHIC MAP SAVED: {output_path}")
    print(f"    This is a proper choropleth map with NUTS-2 boundaries, NOT simplified rectangles!")
    plt.close()


def create_swiss_nuts_map_simplified(data, health_indicator, output_path):
    """
    Create a 2x2 grid of simplified spatial maps for Swiss NUTS regions (no shapefile needed).
    With a single shared colorbar for all 4 maps.
    
    Grid layout:
    - Columns: Years (2013, 2023)
    - Rows: Air Pollutants (PM2.5, NO2)
    """
    print(f"\n⚠ Using SIMPLIFIED visualization (rectangles, not real geographic maps)")
    
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.patches import Rectangle
    import numpy as np
    
    # Approximate positions of Swiss NUTS-2 regions (roughly geographic)
    # Format: {nuts_code: (x, y, width, height)}
    region_positions = {
        'CH01': (0, 0.6, 0.35, 0.35),      # Région Lémanique (West)
        'CH02': (0.35, 0.5, 0.35, 0.45),   # Espace Mittelland (North-Center)
        'CH03': (0.45, 0.15, 0.3, 0.35),   # Nordwestschweiz (Northwest)
        'CH04': (0.7, 0.4, 0.3, 0.45),     # Zürich (Northeast)
        'CH05': (0.6, 0, 0.4, 0.4),        # Ostschweiz (East)
        'CH06': (0.3, 0.2, 0.3, 0.35),     # Zentralschweiz (Central)
        'CH07': (0.05, 0, 0.4, 0.25),      # Ticino (South)
    }
    
    # Create figure with 2x2 subplots + space for colorbar
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid spec for subplots and colorbar
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], 
                          hspace=0.2, wspace=0.2, left=0.05, right=0.92, 
                          top=0.93, bottom=0.05)
    
    # Add title
    fig.suptitle(
        f"{'Premature deaths' if 'AD' in health_indicator else 'Years of Life Lost'} "
        f"due to air pollution exposure in Switzerland\n"
        f"({health_indicator}, per 100,000 affected population)",
        fontsize=16, fontweight='bold'
    )
    
    years = [2013, 2023]
    pollutants = ['PM2.5', 'NO2']
    
    # Use a single colormap for consistency
    colormap = 'YlOrRd'  # Yellow-Orange-Red for health impacts
    
    # Get global min/max for consistent color scaling across all maps
    vmin = data['value_per_100k'].min()
    vmax = data['value_per_100k'].max()
    
    print(f"Value range: {vmin:.1f} to {vmax:.1f}")
    
    # Create normalization for colormap
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(colormap)
    
    # Create subplots
    axes = []
    for row in range(2):
        for col in range(2):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
    
    # Plot each combination
    for idx in range(4):
        row_idx = idx // 2
        col_idx = idx % 2
        year = years[col_idx]
        pollutant = pollutants[row_idx]
        
        ax = axes[idx]
        
        # Filter data for this combination
        subset = data[
            (data['year'] == year) &
            (data['air_pollutant'] == pollutant)
        ].copy()
        
        # Create a dict for easy lookup
        value_dict = dict(zip(subset['nuts_code'], subset['value_per_100k']))
        name_dict = dict(zip(subset['nuts_code'], subset['nuts_name']))
        
        # Draw rectangles for each region
        for nuts_code, (x, y, w, h) in region_positions.items():
            value = value_dict.get(nuts_code, np.nan)
            name = name_dict.get(nuts_code, nuts_code)
            
            # Get color based on value
            if pd.notna(value):
                color = cmap_obj(norm(value))
            else:
                color = 'lightgrey'
            
            # Draw rectangle
            rect = Rectangle((x, y), w, h, 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=2)
            ax.add_patch(rect)
            
            # Add region label and value
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Region name
            ax.text(center_x, center_y + h * 0.15, name,
                   ha='center', va='center', fontsize=9,
                   style='italic', fontweight='bold')
            
            # Value
            if pd.notna(value):
                ax.text(center_x, center_y - h * 0.1, f'{value:.1f}',
                       ha='center', va='center', fontsize=12,
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='white', alpha=0.9,
                                edgecolor='black', linewidth=0.5))
        
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f"{pollutant} - {year}", fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
    
    # Add single colorbar on the right
    cbar_ax = fig.add_subplot(gs[:, 2])
    cb = ColorbarBase(cbar_ax, cmap=cmap_obj, norm=norm, orientation='vertical')
    cb.set_label('Per 100,000 affected population', fontsize=12, fontweight='bold')
    cb.ax.tick_params(labelsize=10)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("EEA Particulate Matter Exposure Visualization for Switzerland")
    print("=" * 80)
    
    # Load data
    df = load_eea_pm_data()
    
    # Check available scenarios
    print("\nAvailable scenarios:")
    for scenario in df['Scenario'].unique():
        print(f"  - {scenario}")
    
    # Process both health indicators
    health_indicators = [
        'Attributable deaths (AD)',
        'Years of Life Lost (YLL)'
    ]
    
    for health_indicator in health_indicators:
        print(f"\n{'=' * 80}")
        print(f"Processing: {health_indicator}")
        print('=' * 80)
        
        # Prepare data
        data = prepare_data_for_mapping(df, health_indicator)
        
        if data.empty:
            print(f"No data available for {health_indicator}")
            continue
        
        # Export data to Excel
        export_data_to_excel(data, health_indicator, OUTPUT_DIR)
        
        # Create output filename
        indicator_short = 'AD' if 'AD' in health_indicator else 'YLL'
        
        # Create map visualization using shapefile (will fallback to simplified if needed)
        map_path = OUTPUT_DIR / f"switzerland_pm_exposure_{indicator_short}.png"
        create_swiss_nuts_map_with_shapefile(data, health_indicator, map_path)
    
    print("\n" + "=" * 80)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
