"""
Create RCB (Remaining Carbon Budget) Comparison Maps
=====================================================

This script generates 6 maps comparing neutrality years across different
Remaining Carbon Budget (RCB) starting dates (2018, 2021, 2025).

Configuration:
- Warming scenario: 1.5°C
- Probability: 50%
- Emissions scope: Territorial only
- Distribution methods: Responsibility and Capability (2 maps each)
- RCB dates: 2018, 2021, 2025 (3 dates × 2 distribution methods = 6 maps)

All maps use the same legend scale (same color = same value across all 6 maps).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path

# Configuration
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
SENSITIVITY_DIR = os.path.join(DATA_DIR, 'Sensitivity check')
OUTPUT_DIR = os.path.join(SENSITIVITY_DIR, 'Maps')

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed parameters for this analysis
WARMING_SCENARIO = '1.5°C'
PROBABILITY = '50%'
EMISSIONS_SCOPE = 'Territory'

# RCB dates to compare
RCB_DATES = ['2018', '2021', '2025']

# Distribution methods
DISTRIBUTION_METHODS = ['Responsibility', 'Capability']


def load_data():
    """Load required data files."""
    print("Loading data...")
    
    # Load sensitivity analysis full results
    full_results = pd.read_csv(os.path.join(SENSITIVITY_DIR, 'sensitivity_analysis_full_results.csv'))
    
    # Load combined data for ISO3 mapping
    combined_data = pd.read_csv(os.path.join(DATA_DIR, 'combined_data.csv'))
    
    # Create ISO2 to ISO3 mapping
    iso_mapping = combined_data[['ISO2', 'ISO3']].drop_duplicates()
    iso2_to_iso3 = dict(zip(iso_mapping['ISO2'], iso_mapping['ISO3']))
    
    print(f"Loaded full results: {full_results.shape}")
    print(f"Created ISO mapping with {len(iso2_to_iso3)} entries")
    
    return full_results, iso2_to_iso3


def get_scenario_name(rcb_date):
    """Get the scenario name for a given RCB date."""
    if rcb_date == '2025':
        return 'Base_2025'
    else:
        return f'Alternative_{rcb_date}'


def prepare_map_data(full_results, iso2_to_iso3, distribution_method, rcb_date):
    """Prepare data for a specific map configuration."""
    scenario_name = get_scenario_name(rcb_date)
    
    # Filter data for specific parameters
    df = full_results[
        (full_results['Emissions_scope'] == EMISSIONS_SCOPE) &
        (full_results['Warming_scenario'] == WARMING_SCENARIO) &
        (full_results['Probability'] == PROBABILITY) &
        (full_results['Distribution_method'] == distribution_method) &
        (full_results['Scenario_name'] == scenario_name)
    ].copy()
    
    # Exclude aggregates (regions like 'Africa', 'Asia-Pacific Developed', etc.)
    # Keep only country-level data (ISO2 codes that are exactly 2 characters)
    df = df[df['ISO2'].str.len() == 2].copy()
    
    # Add ISO3 code for mapping
    df['ISO3'] = df['ISO2'].map(iso2_to_iso3)
    
    # Convert neutrality year to numeric
    df['Neutrality_year_numeric'] = pd.to_numeric(df['Neutrality_year'], errors='coerce')
    
    print(f"  {distribution_method} - RCB {rcb_date}: {len(df)} countries")
    
    return df


def create_map(df, distribution_method, rcb_date, color_range_min, color_range_max):
    """Create a choropleth map for the given data and configuration."""
    
    # Map title
    title = f"Zero Carbon Year - {distribution_method} (RCB {rcb_date})<br><sup>1.5°C | 50% | Territorial Emissions</sup>"
    
    # Create choropleth map
    fig = px.choropleth(
        df,
        locations="ISO3",
        locationmode='ISO-3',
        color="Neutrality_year_numeric",
        hover_name="Country",
        hover_data={
            "Neutrality_year_numeric": False,
            "ISO3": False,
            "ISO2": True,
            "Neutrality_year": True,
            "Country_budget_MtCO2": ':.1f'
        },
        color_continuous_scale=[
            [0.0, '#8B0000'],   # Dark red for 1970s
            [0.1, '#DC143C'],   # Crimson
            [0.2, '#FB8072'],   # Light red/salmon
            [0.4, '#FDB462'],   # Orange
            [0.6, '#FFFFB3'],   # Light yellow
            [0.8, '#8DD3C7'],   # Light teal
            [1.0, '#B3DE69']    # Light green
        ],
        range_color=[color_range_min, color_range_max],
        title=title,
        labels={
            'Neutrality_year_numeric': 'Zero Carbon Year',
            'Neutrality_year': 'Zero Carbon Year',
            'Country_budget_MtCO2': 'Carbon Budget (Mt)',
            'ISO2': 'Country Code'
        }
    )
    
    # Update traces for better styling
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.5,
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "Zero Carbon Year: %{customdata[1]}<br>" +
                      "Carbon Budget (Mt): %{customdata[2]:.1f}<extra></extra>",
        customdata=df[['ISO2', 'Neutrality_year', 'Country_budget_MtCO2']].values,
        colorbar=dict(
            x=0.05,
            xanchor="left",
            thickness=15,
            len=0.7,
            title=dict(
                text="Zero Carbon Year",
                font=dict(size=14, color="#2c3e50"),
                side="top"
            )
        )
    )
    
    # Update geo settings (same as app.py)
    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='white',
        projection_type='equirectangular',
        bgcolor='white',
        landcolor='lightgray',
        framecolor='white',
        showlakes=False,
        center=dict(lat=20, lon=0),
        lataxis_range=[-60, 80],
        lonaxis_range=[-180, 180]
    )
    
    # Update layout (same as app.py)
    fig.update_layout(
        height=600,
        width=1200,
        margin={"r": 100, "t": 80, "l": 100, "b": 50},
        geo=dict(
            showland=True,
            landcolor='lightgray',
            showocean=True,
            oceancolor='white',
            showlakes=False,
            showrivers=False,
            center=dict(lat=20, lon=0),
            lataxis_range=[-60, 80],
            lonaxis_range=[-180, 180],
            projection_scale=1.0
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#2c3e50"
        ),
        title=dict(
            font=dict(size=16, color="#2c3e50"),
            x=0.5
        ),
        hoverlabel=dict(
            bgcolor="rgba(245, 245, 245, 0.9)",
            bordercolor="white",
            font=dict(color="black", size=12)
        )
    )
    
    return fig


def export_to_excel(full_results, iso2_to_iso3):
    """Export neutrality years for all configurations to Excel."""
    print("\nExporting data to Excel...")
    
    # Collect data for all configurations
    all_data = []
    
    for distribution_method in DISTRIBUTION_METHODS:
        for rcb_date in RCB_DATES:
            df = prepare_map_data(full_results, iso2_to_iso3, distribution_method, rcb_date)
            if len(df) > 0:
                df_export = df[['ISO2', 'ISO3', 'Country', 'Neutrality_year', 'Country_budget_MtCO2']].copy()
                df_export['Distribution_method'] = distribution_method
                df_export['RCB_date'] = rcb_date
                all_data.append(df_export)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Pivot to create a wide format table (easier to compare)
    pivot_df = combined_df.pivot_table(
        index=['ISO2', 'ISO3', 'Country'],
        columns=['Distribution_method', 'RCB_date'],
        values='Neutrality_year',
        aggfunc='first'
    )
    
    # Flatten column names
    pivot_df.columns = [f'{method}_{rcb}' for method, rcb in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    # Reorder columns for better readability
    col_order = ['ISO2', 'ISO3', 'Country']
    for method in DISTRIBUTION_METHODS:
        for rcb in RCB_DATES:
            col_name = f'{method}_{rcb}'
            if col_name in pivot_df.columns:
                col_order.append(col_name)
    
    pivot_df = pivot_df[col_order]
    
    # Save to Excel
    excel_path = os.path.join(OUTPUT_DIR, 'rcb_neutrality_years_comparison.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Wide format (comparison view)
        pivot_df.to_excel(writer, sheet_name='Comparison', index=False)
        
        # Sheet 2: Long format (all data)
        combined_df.to_excel(writer, sheet_name='All Data', index=False)
    
    print(f"  Saved: rcb_neutrality_years_comparison.xlsx")
    print(f"    - Sheet 'Comparison': {len(pivot_df)} countries with all scenarios side by side")
    print(f"    - Sheet 'All Data': {len(combined_df)} rows (long format)")
    
    return pivot_df, combined_df


def main():
    """Main function to generate all 6 maps."""
    print("=" * 60)
    print("Creating RCB Comparison Maps")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Warming scenario: {WARMING_SCENARIO}")
    print(f"  - Probability: {PROBABILITY}")
    print(f"  - Emissions scope: {EMISSIONS_SCOPE}")
    print(f"  - RCB dates: {RCB_DATES}")
    print(f"  - Distribution methods: {DISTRIBUTION_METHODS}")
    print("=" * 60)
    
    # Load data
    full_results, iso2_to_iso3 = load_data()
    
    # Export to Excel
    export_to_excel(full_results, iso2_to_iso3)
    
    # First pass: determine global color range across all 6 maps
    print("\nDetermining global color range...")
    all_neutrality_years = []
    
    for distribution_method in DISTRIBUTION_METHODS:
        for rcb_date in RCB_DATES:
            df = prepare_map_data(full_results, iso2_to_iso3, distribution_method, rcb_date)
            if len(df) > 0:
                all_neutrality_years.extend(df['Neutrality_year_numeric'].dropna().tolist())
    
    # Set global color range
    color_range_min = min(1970, min(all_neutrality_years)) if all_neutrality_years else 1970
    color_range_max = max(2100, max(all_neutrality_years)) if all_neutrality_years else 2100
    
    print(f"Global color range: {color_range_min} - {color_range_max}")
    
    # Second pass: create all maps with consistent color scale
    print("\nGenerating maps...")
    
    for distribution_method in DISTRIBUTION_METHODS:
        for rcb_date in RCB_DATES:
            print(f"\nCreating map: {distribution_method} - RCB {rcb_date}")
            
            # Prepare data
            df = prepare_map_data(full_results, iso2_to_iso3, distribution_method, rcb_date)
            
            if len(df) == 0:
                print(f"  WARNING: No data found for this configuration!")
                continue
            
            # Create map
            fig = create_map(df, distribution_method, rcb_date, color_range_min, color_range_max)
            
            # Generate filenames
            base_filename = f"rcb_map_{distribution_method}_RCB{rcb_date}_1.5C_50pct_Territory"
            html_path = os.path.join(OUTPUT_DIR, f"{base_filename}.html")
            png_path = os.path.join(OUTPUT_DIR, f"{base_filename}.png")
            
            # Save HTML
            fig.write_html(html_path)
            print(f"  Saved: {base_filename}.html")
            
            # Save PNG
            try:
                fig.write_image(png_path, scale=2)
                print(f"  Saved: {base_filename}.png")
            except Exception as e:
                print(f"  Could not save PNG (kaleido may not be installed): {e}")
    
    print("\n" + "=" * 60)
    print("Map generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Generate summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    
    for distribution_method in DISTRIBUTION_METHODS:
        print(f"\n{distribution_method}:")
        for rcb_date in RCB_DATES:
            df = prepare_map_data(full_results, iso2_to_iso3, distribution_method, rcb_date)
            if len(df) > 0:
                median_year = df['Neutrality_year_numeric'].median()
                mean_year = df['Neutrality_year_numeric'].mean()
                min_year = df['Neutrality_year_numeric'].min()
                max_year = df['Neutrality_year_numeric'].max()
                print(f"  RCB {rcb_date}: Median={median_year:.0f}, Mean={mean_year:.0f}, Range=[{min_year:.0f}-{max_year:.0f}]")


if __name__ == "__main__":
    main()
