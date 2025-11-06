"""
swiss_vs_eu27_time_series.py - Generate time series comparison graphs for Switzerland vs EU-27

This script creates time series graphs comparing Switzerland and EU-27 values for all primary indicators.
Each graph shows both countries' values over time for a specific primary indicator.

The graphs are saved in the 'Graphs Switzerland' folder within the output directory.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import numpy as np
import sys

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import variable mapping functions from app
from variable_mapping import get_display_name

# Color palette consistent with 6_graphs.py and dashboard
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'
SWITZERLAND_COLOR = '#ffd558'  # Use yellow for Switzerland as requested

def get_app_level5_indicators():
    """Get the Level 5 indicators that are actually available in the app's dropdown by replicating app.py logic"""
    
    # Load the unified data to check what's available
    current_dir = os.path.dirname(os.path.abspath(__file__))
    well_being_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_path = os.path.join(well_being_dir, 'output', 'unified_all_levels_1_to_5_pca_weighted.csv')
    
    try:
        df = pd.read_csv(data_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}")
        return []
    
    # EU priorities that are active in the app (from app.py)
    EU_PRIORITIES = [
        'Energy and Housing', 
        'Equality',
        'Health and Animal Welfare',
        'Intergenerational Fairness, Youth, Culture and Sport',
        'Social Rights and Skills, Quality Jobs and Preparedness'
    ]
    
    # Get Level 5 indicators that would appear in app dropdowns
    # This replicates the logic from app.py update_primary_indicator_dropdown()
    all_available_indicators = set()
    
    for eu_priority in EU_PRIORITIES:
        # Get primary indicators from the unified PCA data for this EU priority
        # This matches the exact filtering logic in app.py
        primary_options = df[
            (df['EU priority'] == eu_priority) & 
            (df['Primary and raw data'].notna()) &
            (df['Level'] == 5)  # Raw data level
        ]['Primary and raw data'].unique()
        
        print(f"EU Priority '{eu_priority}': {len(primary_options)} indicators")
        for indicator in primary_options:
            display_name = get_display_name(indicator)
            print(f"  • {display_name}")
        
        all_available_indicators.update(primary_options)
    
    print(f"\nTotal unique Level 5 indicators available in app dropdowns: {len(all_available_indicators)}")
    return sorted(all_available_indicators)

def load_data():
    """Load the primary indicators data for time series analysis"""
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    well_being_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.abspath(os.path.join(well_being_dir, 'output'))
    
    # Load Level 5 indicators data (primary indicators) - use the unified dataset for both
    unified_path = os.path.join(data_dir, 'unified_all_levels_1_to_5_pca_weighted.csv')
    
    if not os.path.exists(unified_path):
        raise FileNotFoundError(f"Unified data file not found: {unified_path}")
    
    # Load the unified dataset
    unified_df = pd.read_csv(unified_path, low_memory=False)
    
    print(f"Loaded unified data with shape: {unified_df.shape}")
    
    return unified_df

def prepare_swiss_data(unified_df):
    """Extract and prepare Switzerland data for all primary indicators from Level 5, including deciles"""
    # Get Switzerland data from Level 5 (primary indicators) - median/mean data
    swiss_data = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 5) &
        (unified_df['Decile'] == 'All')
    ].copy()
    
    # Get 1st and 10th decile data for Switzerland
    swiss_decile_1 = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 5) &
        (unified_df['Decile'] == '1.0')
    ].copy()
    
    swiss_decile_10 = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 5) &
        (unified_df['Decile'] == '10.0')
    ].copy()
    
    if swiss_data.empty:
        print("No 'All' decile data found for Switzerland at Level 5. Computing country averages from decile data...")
        
        # Get all Switzerland Level 5 data and compute country averages by year and indicator
        swiss_decile_data = unified_df[
            (unified_df['Country'] == 'CH') & 
            (unified_df['Level'] == 5) &
            (unified_df['Decile'].isin(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']))
        ].copy()
        
        if swiss_decile_data.empty:
            print("Warning: No Switzerland data found at Level 5")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Compute the mean value across all deciles for each year and indicator
        swiss_data = swiss_decile_data.groupby(['Year', 'Primary and raw data']).agg({
            'Value': 'mean',
            'Country': 'first',
            'Level': 'first',
            'Type': 'first',
            'Aggregation': lambda x: 'Mean across deciles',
            'datasource': 'first'
        }).reset_index()
        
        # Add the All decile marker
        swiss_data['Decile'] = 'All'
        swiss_data['Quintile'] = ''
        swiss_data['EU priority'] = ''
        swiss_data['Secondary'] = ''
        
        print(f"Computed data for {len(swiss_data)} year-indicator combinations")
    
    # Rename columns for consistency
    swiss_data = swiss_data.rename(columns={'Primary and raw data': 'Indicator'})
    swiss_data['Country_Name'] = 'Switzerland'
    
    # Prepare decile data
    if not swiss_decile_1.empty:
        swiss_decile_1 = swiss_decile_1.rename(columns={'Primary and raw data': 'Indicator'})
        swiss_decile_1['Country_Name'] = 'Switzerland'
    
    if not swiss_decile_10.empty:
        swiss_decile_10 = swiss_decile_10.rename(columns={'Primary and raw data': 'Indicator'})
        swiss_decile_10['Country_Name'] = 'Switzerland'
    
    return swiss_data, swiss_decile_1, swiss_decile_10

def prepare_eu27_data(unified_df):
    """Extract and prepare EU-27 data for all primary indicators, including deciles"""
    # Get EU-27 data from the unified dataset at Level 5 (primary indicators) - median data
    # Try the available aggregation method first
    eu27_data = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 5) &
        (unified_df['Decile'] == 'All') &
        (unified_df['Aggregation'] == 'Median across countries')
    ].copy()
    
    # Get 1st and 10th decile data for EU-27
    eu27_decile_1 = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 5) &
        (unified_df['Decile'] == '1.0') &
        (unified_df['Aggregation'] == 'Median across countries')
    ].copy()
    
    eu27_decile_10 = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 5) &
        (unified_df['Decile'] == '10.0') &
        (unified_df['Aggregation'] == 'Median across countries')
    ].copy()
    
    if eu27_data.empty:
        print("Warning: No EU-27 data found at Level 5 with 'All' decile")
        # Try with any decile to see what's available
        eu27_sample = unified_df[
            (unified_df['Country'] == 'All Countries') & 
            (unified_df['Level'] == 5)
        ].copy()
        
        if eu27_sample.empty:
            print("No Level 5 data found for 'All Countries'")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Get available deciles
        available_deciles = eu27_sample['Decile'].unique()
        print(f"Available deciles for EU-27 at Level 5: {available_deciles}")
        
        # Compute EU-27 averages across deciles for each year and indicator
        eu27_decile_data = eu27_sample[
            eu27_sample['Decile'].isin(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0'])
        ].copy()
        
        eu27_data = eu27_decile_data.groupby(['Year', 'Primary and raw data']).agg({
            'Value': 'mean',
            'Country': 'first',
            'Level': 'first',
            'Type': 'first',
            'Aggregation': lambda x: 'Mean across deciles',
            'datasource': 'first'
        }).reset_index()
        
        # Add the All decile marker
        eu27_data['Decile'] = 'All'
        eu27_data['Quintile'] = ''
        eu27_data['EU priority'] = ''
        eu27_data['Secondary'] = ''
        
        print(f"Computed EU-27 data for {len(eu27_data)} year-indicator combinations")
    
    # Rename columns for consistency
    eu27_data = eu27_data.rename(columns={'Primary and raw data': 'Indicator'})
    eu27_data['Country_Name'] = 'EU-27'
    
    # Prepare decile data
    if not eu27_decile_1.empty:
        eu27_decile_1 = eu27_decile_1.rename(columns={'Primary and raw data': 'Indicator'})
        eu27_decile_1['Country_Name'] = 'EU-27'
    
    if not eu27_decile_10.empty:
        eu27_decile_10 = eu27_decile_10.rename(columns={'Primary and raw data': 'Indicator'})
        eu27_decile_10['Country_Name'] = 'EU-27'
    
    return eu27_data, eu27_decile_1, eu27_decile_10

def create_time_series_plot(swiss_data, eu27_data, swiss_decile_1, swiss_decile_10, 
                          eu27_decile_1, eu27_decile_10, indicator, output_dir):
    """Create a time series plot comparing Switzerland and EU-27 for a specific indicator with deciles"""
    
    # Filter data for the specific indicator
    swiss_indicator = swiss_data[swiss_data['Indicator'] == indicator].copy()
    eu27_indicator = eu27_data[eu27_data['Indicator'] == indicator].copy()
    
    # Filter decile data (handle empty dataframes)
    swiss_d1 = swiss_decile_1[swiss_decile_1['Indicator'] == indicator].copy() if not swiss_decile_1.empty and 'Indicator' in swiss_decile_1.columns else pd.DataFrame()
    swiss_d10 = swiss_decile_10[swiss_decile_10['Indicator'] == indicator].copy() if not swiss_decile_10.empty and 'Indicator' in swiss_decile_10.columns else pd.DataFrame()
    eu27_d1 = eu27_decile_1[eu27_decile_1['Indicator'] == indicator].copy() if not eu27_decile_1.empty and 'Indicator' in eu27_decile_1.columns else pd.DataFrame()
    eu27_d10 = eu27_decile_10[eu27_decile_10['Indicator'] == indicator].copy() if not eu27_decile_10.empty and 'Indicator' in eu27_decile_10.columns else pd.DataFrame()
    
    if swiss_indicator.empty and eu27_indicator.empty:
        print(f"Warning: No data available for indicator {indicator}")
        return  # Skip if neither country has data
    
    if swiss_indicator.empty:
        print(f"Note: No Swiss data for {indicator}, showing EU-27 only")
    elif eu27_indicator.empty:
        print(f"Note: No EU-27 data for {indicator}, showing Switzerland only")
    
    # Find the time range based on your requirements:
    # - Minimum year: earliest from either EU-27 or Switzerland  
    # - Maximum year: latest from Switzerland data (if available), otherwise latest from EU data
    all_years = []
    swiss_years = []
    
    if not swiss_indicator.empty:
        swiss_years.extend(swiss_indicator['Year'].tolist())
        all_years.extend(swiss_indicator['Year'].tolist())
    if not eu27_indicator.empty:
        all_years.extend(eu27_indicator['Year'].tolist())
    if not swiss_d1.empty:
        swiss_years.extend(swiss_d1['Year'].tolist())
        all_years.extend(swiss_d1['Year'].tolist())
    if not swiss_d10.empty:
        swiss_years.extend(swiss_d10['Year'].tolist())
        all_years.extend(swiss_d10['Year'].tolist())
    if not eu27_d1.empty:
        all_years.extend(eu27_d1['Year'].tolist())
    if not eu27_d10.empty:
        all_years.extend(eu27_d10['Year'].tolist())
    
    if not all_years:
        print(f"Warning: No data found for indicator {indicator}")
        return
    
    earliest_year = min(all_years)  # Earliest from either EU-27 or Switzerland
    latest_year = max(swiss_years) if swiss_years else max(all_years)  # Latest from Switzerland, or all data if no Swiss data
    
    # Filter all data to not exceed Switzerland's latest year
    swiss_indicator = swiss_indicator[swiss_indicator['Year'] <= latest_year] if not swiss_indicator.empty else swiss_indicator
    eu27_indicator = eu27_indicator[eu27_indicator['Year'] <= latest_year] if not eu27_indicator.empty else eu27_indicator
    swiss_d1 = swiss_d1[swiss_d1['Year'] <= latest_year] if not swiss_d1.empty else swiss_d1
    swiss_d10 = swiss_d10[swiss_d10['Year'] <= latest_year] if not swiss_d10.empty else swiss_d10
    eu27_d1 = eu27_d1[eu27_d1['Year'] <= latest_year] if not eu27_d1.empty else eu27_d1
    eu27_d10 = eu27_d10[eu27_d10['Year'] <= latest_year] if not eu27_d10.empty else eu27_d10
    
    # Sort by year
    swiss_indicator = swiss_indicator.sort_values('Year')
    eu27_indicator = eu27_indicator.sort_values('Year')
    if not swiss_d1.empty:
        swiss_d1 = swiss_d1.sort_values('Year')
    if not swiss_d10.empty:
        swiss_d10 = swiss_d10.sort_values('Year')
    if not eu27_d1.empty:
        eu27_d1 = eu27_d1.sort_values('Year')
    if not eu27_d10.empty:
        eu27_d10 = eu27_d10.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add Switzerland lines first (median, then deciles) - only if data exists
    # 1. Switzerland median line (solid yellow)
    if not swiss_indicator.empty:
        fig.add_trace(go.Scatter(
            x=swiss_indicator['Year'],
            y=swiss_indicator['Value'],
            mode='lines+markers',
            name='Switzerland Average',
            line=dict(color=SWITZERLAND_COLOR, width=3),
            marker=dict(color=SWITZERLAND_COLOR, size=8, symbol='circle'),
            hovertemplate='<b>Switzerland Average</b><br>' +
                         'Year: %{x}<br>' +
                         'Value: %{y:.3f}<extra></extra>'
        ))
    
    # 2. Switzerland 1st decile line (dashed yellow)
    if not swiss_d1.empty:
        fig.add_trace(go.Scatter(
            x=swiss_d1['Year'],
            y=swiss_d1['Value'],
            mode='lines+markers',
            name='Switzerland 1st Decile',
            line=dict(color=SWITZERLAND_COLOR, width=2, dash='dash'),
            marker=dict(color=SWITZERLAND_COLOR, size=6, symbol='triangle-up'),
            hovertemplate='<b>Switzerland 1st Decile</b><br>' +
                         'Year: %{x}<br>' +
                         'Value: %{y:.3f}<extra></extra>'
        ))
    
    # 3. Switzerland 10th decile line (dashdot yellow)
    if not swiss_d10.empty:
        fig.add_trace(go.Scatter(
            x=swiss_d10['Year'],
            y=swiss_d10['Value'],
            mode='lines+markers',
            name='Switzerland 10th Decile',
            line=dict(color=SWITZERLAND_COLOR, width=2, dash='dashdot'),
            marker=dict(color=SWITZERLAND_COLOR, size=6, symbol='triangle-down'),
            hovertemplate='<b>Switzerland 10th Decile</b><br>' +
                         'Year: %{x}<br>' +
                         'Value: %{y:.3f}<extra></extra>'
        ))
    
    # Add EU-27 lines (median, then deciles) - only if data exists
    # 4. EU-27 median line (solid blue)
    if not eu27_indicator.empty:
        fig.add_trace(go.Scatter(
            x=eu27_indicator['Year'],
            y=eu27_indicator['Value'],
            mode='lines+markers',
            name='EU-27 Average',
            line=dict(color=EU_27_COLOR, width=3),
            marker=dict(color=EU_27_COLOR, size=8, symbol='diamond'),
            hovertemplate='<b>EU-27 Average</b><br>' +
                         'Year: %{x}<br>' +
                         'Value: %{y:.3f}<extra></extra>'
        ))
    
    # 5. EU-27 1st decile line (dashed blue)
    if not eu27_d1.empty:
        fig.add_trace(go.Scatter(
            x=eu27_d1['Year'],
            y=eu27_d1['Value'],
            mode='lines+markers',
            name='EU-27 1st Decile',
            line=dict(color=EU_27_COLOR, width=2, dash='dash'),
            marker=dict(color=EU_27_COLOR, size=6, symbol='triangle-up'),
            hovertemplate='<b>EU-27 1st Decile</b><br>' +
                         'Year: %{x}<br>' +
                         'Value: %{y:.3f}<extra></extra>'
        ))
    
    # 6. EU-27 10th decile line (dashdot blue)
    if not eu27_d10.empty:
        fig.add_trace(go.Scatter(
            x=eu27_d10['Year'],
            y=eu27_d10['Value'],
            mode='lines+markers',
            name='EU-27 10th Decile',
            line=dict(color=EU_27_COLOR, width=2, dash='dashdot'),
            marker=dict(color=EU_27_COLOR, size=6, symbol='triangle-down'),
            hovertemplate='<b>EU-27 10th Decile</b><br>' +
                         'Year: %{x}<br>' +
                         'Value: %{y:.3f}<extra></extra>'
        ))
    
    # Get indicator description
    description = get_display_name(indicator)
    
    # Determine x-axis tick spacing based on data availability
    year_span = latest_year - earliest_year + 1
    unique_years = set()
    
    # Collect all unique years that actually have data points
    if not swiss_indicator.empty:
        unique_years.update(swiss_indicator['Year'].tolist())
    if not eu27_indicator.empty:
        unique_years.update(eu27_indicator['Year'].tolist())
    if not swiss_d1.empty:
        unique_years.update(swiss_d1['Year'].tolist())
    if not swiss_d10.empty:
        unique_years.update(swiss_d10['Year'].tolist())
    if not eu27_d1.empty:
        unique_years.update(eu27_d1['Year'].tolist())
    if not eu27_d10.empty:
        unique_years.update(eu27_d10['Year'].tolist())
    
    num_data_points = len(unique_years)
    
    # For 3 or fewer data points, show all years with data
    # For 4-5 data points, show every year
    # For more data points, show every 2 years
    if num_data_points <= 3:
        # Show all years that have actual data points
        tick_values = sorted(list(unique_years))
        dtick_value = None  # Use tickvals instead of dtick
    elif num_data_points <= 5:
        dtick_value = 1  # Show every year
        tick_values = None
    else:
        dtick_value = 2  # Show every 2 years for longer time series
        tick_values = None
    
    # Update layout - Use only the description, not the indicator code
    if tick_values is not None:
        # Use specific tick values for 3 or fewer data points
        fig.update_layout(
            title=dict(
                text=f"{description}<br><sub>Switzerland vs EU-27 Time Series ({int(earliest_year)}-{int(latest_year)})</sub>",
                font=dict(size=16, color="#2C3E50", family="Arial, sans-serif"),
                x=0.5
            ),
            xaxis=dict(
                title="Year",
                title_font=dict(size=14, family="Arial, sans-serif"),
                tickfont=dict(size=12, family="Arial, sans-serif"),
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                tickvals=tick_values,
                ticktext=[str(int(year)) for year in tick_values],
                range=[earliest_year - 0.5, latest_year + 0.5]  # Add some padding
            ),
            yaxis=dict(
                title="Share of the population",
                title_font=dict(size=14, family="Arial, sans-serif"),
                tickfont=dict(size=12, family="Arial, sans-serif"),
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif"),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
                font=dict(size=12)
            ),
            margin=dict(t=80, b=60, l=60, r=40),
            width=1000,
            height=600
        )
    else:
        # Use dtick for regular spacing
        fig.update_layout(
            title=dict(
                text=f"{description}<br><sub>Switzerland vs EU-27 Time Series ({int(earliest_year)}-{int(latest_year)})</sub>",
                font=dict(size=16, color="#2C3E50", family="Arial, sans-serif"),
                x=0.5
            ),
            xaxis=dict(
                title="Year",
                title_font=dict(size=14, family="Arial, sans-serif"),
                tickfont=dict(size=12, family="Arial, sans-serif"),
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1,
                tick0=earliest_year,
                dtick=dtick_value,  # Dynamic tick spacing based on data points
                tickmode='linear',
                range=[earliest_year - 0.5, latest_year + 0.5]  # Add some padding
            ),
            yaxis=dict(
                title="Share of the population",
                title_font=dict(size=14, family="Arial, sans-serif"),
                tickfont=dict(size=12, family="Arial, sans-serif"),
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif"),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
                font=dict(size=12)
            ),
            margin=dict(t=80, b=60, l=60, r=40),
            width=1000,
            height=600
        )
    
    # Create safe filename
    safe_indicator = indicator.replace('-', '_').replace('/', '_')
    
    # Save as HTML
    html_path = os.path.join(output_dir, f"switzerland_vs_eu27_{safe_indicator}.html")
    fig.write_html(html_path)
    print(f"Saved HTML: {html_path}")
    
    # Save as PNG using matplotlib for reliability
    png_path = os.path.join(output_dir, f"switzerland_vs_eu27_{safe_indicator}.png")
    try:
        # Use matplotlib for PNG generation (more reliable than kaleido)
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        
        # Create matplotlib figure
        plt.figure(figsize=(12, 7))
        
        # Plot in the same order as Plotly for consistent legend
        # 1. Switzerland median line (solid yellow)
        plt.plot(swiss_indicator['Year'], swiss_indicator['Value'], 
                color=SWITZERLAND_COLOR, linewidth=3, marker='o', 
                markersize=8, label='Switzerland Average', alpha=0.9)
        
        # 2. Switzerland decile lines (different dash patterns and markers)
        if not swiss_d1.empty:
            plt.plot(swiss_d1['Year'], swiss_d1['Value'], 
                    color=SWITZERLAND_COLOR, linewidth=2, linestyle='--', marker='^', markersize=6,
                    label='Switzerland 1st Decile', alpha=0.7)
        
        if not swiss_d10.empty:
            plt.plot(swiss_d10['Year'], swiss_d10['Value'],
                    color=SWITZERLAND_COLOR, linewidth=2, linestyle='-.', marker='v', markersize=6,
                    label='Switzerland 10th Decile', alpha=0.7)
        
        # 3. EU-27 median line (solid blue)
        plt.plot(eu27_indicator['Year'], eu27_indicator['Value'],
                color=EU_27_COLOR, linewidth=3, marker='s',
                markersize=8, label='EU-27 Average', alpha=0.9)
        
        # 4. EU-27 decile lines (different dash patterns and markers)
        if not eu27_d1.empty:
            plt.plot(eu27_d1['Year'], eu27_d1['Value'], 
                    color=EU_27_COLOR, linewidth=2, linestyle='--', marker='^', markersize=6,
                    label='EU-27 1st Decile', alpha=0.7)
        
        if not eu27_d10.empty:
            plt.plot(eu27_d10['Year'], eu27_d10['Value'],
                    color=EU_27_COLOR, linewidth=2, linestyle='-.', marker='v', markersize=6,
                    label='EU-27 10th Decile', alpha=0.7)
        
        # Styling
        plt.title(f"{description}\nSwitzerland vs EU-27 Time Series ({int(earliest_year)}-{int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Share of the population', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        
        # Set x-axis to show the appropriate years based on data points
        if tick_values is not None:
            # For 3 or fewer data points, show all years with data
            plt.xticks(tick_values)
        elif dtick_value == 1:
            # For 4-5 data points, show every year
            years = list(range(int(earliest_year), int(latest_year) + 1))
            plt.xticks(years)
        else:
            # For more data points, show every 2 years starting from an even year
            year_start = int(earliest_year)
            year_end = int(latest_year)
            # Ensure we start from an even year for consistency
            if year_start % 2 != 0:
                year_start -= 1
            years = list(range(year_start, year_end + 1, 2))  # Every 2 years
            plt.xticks(years)
        
        plt.xlim(earliest_year - 0.5, latest_year + 0.5)
        
        # Set background color
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        # Save PNG
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {indicator}: {str(e)[:100]}...")
        print("  Only HTML version saved.")

def create_summary_stats(swiss_data, eu27_data, all_indicators, output_dir):
    """Create summary statistics for the indicators"""
    
    # Get all unique indicators
    swiss_indicators = set(swiss_data['Indicator'].unique()) if not swiss_data.empty else set()
    eu27_indicators = set(eu27_data['Indicator'].unique()) if not eu27_data.empty else set()
    
    summary_stats = []
    
    for indicator in sorted(all_indicators):
        swiss_indicator = swiss_data[swiss_data['Indicator'] == indicator] if not swiss_data.empty else pd.DataFrame()
        eu27_indicator = eu27_data[eu27_data['Indicator'] == indicator] if not eu27_data.empty else pd.DataFrame()
        
        # Get basic info
        description = get_display_name(indicator)
        
        # Swiss stats
        swiss_available = not swiss_indicator.empty
        swiss_years = sorted(swiss_indicator['Year'].unique()) if swiss_available else []
        swiss_year_range = f"{min(swiss_years)}-{max(swiss_years)}" if swiss_years else "No data"
        
        # EU27 stats  
        eu27_available = not eu27_indicator.empty
        eu27_years = sorted(eu27_indicator['Year'].unique()) if eu27_available else []
        eu27_year_range = f"{min(eu27_years)}-{max(eu27_years)}" if eu27_years else "No data"
        
        # Overlap
        if swiss_years and eu27_years:
            overlap_years = sorted(set(swiss_years).intersection(set(eu27_years)))
            overlap_range = f"{min(overlap_years)}-{max(overlap_years)}" if overlap_years else "No overlap"
            overlap_count = len(overlap_years)
        else:
            overlap_range = "No overlap"
            overlap_count = 0
        
        if not swiss_indicator.empty and not eu27_indicator.empty:
            swiss_mean = swiss_indicator['Value'].mean()
            swiss_latest = swiss_indicator.sort_values('Year')['Value'].iloc[-1]
            swiss_years = f"{swiss_indicator['Year'].min():.0f}-{swiss_indicator['Year'].max():.0f}"
            
            eu27_mean = eu27_indicator['Value'].mean()
            eu27_latest = eu27_indicator.sort_values('Year')['Value'].iloc[-1]
            eu27_years = f"{eu27_indicator['Year'].min():.0f}-{eu27_indicator['Year'].max():.0f}"
            
            summary_stats.append({
                'Indicator': indicator,
                'Description': get_display_name(indicator),
                'Swiss_Mean': swiss_mean,
                'Swiss_Latest': swiss_latest,
                'Swiss_Years': swiss_years,
                'EU27_Mean': eu27_mean,
                'EU27_Latest': eu27_latest,
                'EU27_Years': eu27_years,
                'Difference_Latest': swiss_latest - eu27_latest
            })
    
    # Create DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, "switzerland_vs_eu27_summary_stats.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics: {summary_path}")
    
    return summary_df

def export_all_data_to_excel(swiss_data, eu27_data, swiss_decile_1, swiss_decile_10, 
                           eu27_decile_1, eu27_decile_10, all_indicators, output_dir):
    """Export all displayed data to Excel with the requested column structure"""
    
    all_data_records = []
    
    # Process each indicator
    for indicator in sorted(all_indicators):
        # Get indicator description
        description = get_display_name(indicator)
        
        # Filter data for this indicator
        swiss_median = swiss_data[swiss_data['Indicator'] == indicator].copy()
        eu27_median = eu27_data[eu27_data['Indicator'] == indicator].copy()
        
        swiss_d1 = swiss_decile_1[swiss_decile_1['Indicator'] == indicator].copy() if not swiss_decile_1.empty and 'Indicator' in swiss_decile_1.columns else pd.DataFrame()
        swiss_d10 = swiss_decile_10[swiss_decile_10['Indicator'] == indicator].copy() if not swiss_decile_10.empty and 'Indicator' in swiss_decile_10.columns else pd.DataFrame()
        eu27_d1 = eu27_decile_1[eu27_decile_1['Indicator'] == indicator].copy() if not eu27_decile_1.empty and 'Indicator' in eu27_decile_1.columns else pd.DataFrame()
        eu27_d10 = eu27_decile_10[eu27_decile_10['Indicator'] == indicator].copy() if not eu27_decile_10.empty and 'Indicator' in eu27_decile_10.columns else pd.DataFrame()
        
        # Determine time range (same logic as in plotting function)
        all_years = []
        swiss_years = []
        
        if not swiss_median.empty:
            swiss_years.extend(swiss_median['Year'].tolist())
            all_years.extend(swiss_median['Year'].tolist())
        if not eu27_median.empty:
            all_years.extend(eu27_median['Year'].tolist())
        if not swiss_d1.empty:
            swiss_years.extend(swiss_d1['Year'].tolist())
            all_years.extend(swiss_d1['Year'].tolist())
        if not swiss_d10.empty:
            swiss_years.extend(swiss_d10['Year'].tolist())
            all_years.extend(swiss_d10['Year'].tolist())
        if not eu27_d1.empty:
            all_years.extend(eu27_d1['Year'].tolist())
        if not eu27_d10.empty:
            all_years.extend(eu27_d10['Year'].tolist())
        
        if not all_years:
            continue
            
        earliest_year = min(all_years)
        latest_year = max(swiss_years) if swiss_years else max(all_years)
        
        # Filter all datasets to the same time range used in graphs
        swiss_median = swiss_median[swiss_median['Year'] <= latest_year] if not swiss_median.empty else swiss_median
        eu27_median = eu27_median[eu27_median['Year'] <= latest_year] if not eu27_median.empty else eu27_median
        swiss_d1 = swiss_d1[swiss_d1['Year'] <= latest_year] if not swiss_d1.empty else swiss_d1
        swiss_d10 = swiss_d10[swiss_d10['Year'] <= latest_year] if not swiss_d10.empty else swiss_d10
        eu27_d1 = eu27_d1[eu27_d1['Year'] <= latest_year] if not eu27_d1.empty else eu27_d1
        eu27_d10 = eu27_d10[eu27_d10['Year'] <= latest_year] if not eu27_d10.empty else eu27_d10
        
        # Add Switzerland median data
        for _, row in swiss_median.iterrows():
            all_data_records.append({
                'indicator_code': indicator,
                'indicator_name': description,
                'year': int(row['Year']),
                'geo': 'CH',
                'decile': 'Average',
                'value': row['Value']
            })
        
        # Add Switzerland 1st decile data
        for _, row in swiss_d1.iterrows():
            all_data_records.append({
                'indicator_code': indicator,
                'indicator_name': description,
                'year': int(row['Year']),
                'geo': 'CH',
                'decile': '1st',
                'value': row['Value']
            })
        
        # Add Switzerland 10th decile data
        for _, row in swiss_d10.iterrows():
            all_data_records.append({
                'indicator_code': indicator,
                'indicator_name': description,
                'year': int(row['Year']),
                'geo': 'CH',
                'decile': '10th',
                'value': row['Value']
            })
        
        # Add EU-27 median data
        for _, row in eu27_median.iterrows():
            all_data_records.append({
                'indicator_code': indicator,
                'indicator_name': description,
                'year': int(row['Year']),
                'geo': 'EU27',
                'decile': 'Average',
                'value': row['Value']
            })
        
        # Add EU-27 1st decile data
        for _, row in eu27_d1.iterrows():
            all_data_records.append({
                'indicator_code': indicator,
                'indicator_name': description,
                'year': int(row['Year']),
                'geo': 'EU27',
                'decile': '1st',
                'value': row['Value']
            })
        
        # Add EU-27 10th decile data
        for _, row in eu27_d10.iterrows():
            all_data_records.append({
                'indicator_code': indicator,
                'indicator_name': description,
                'year': int(row['Year']),
                'geo': 'EU27',
                'decile': '10th',
                'value': row['Value']
            })
    
    # Create DataFrame
    export_df = pd.DataFrame(all_data_records)
    
    if export_df.empty:
        print("Warning: No data to export")
        return
    
    # Sort by indicator code, geo, decile, year
    export_df = export_df.sort_values(['indicator_code', 'geo', 'decile', 'year'])
    
    # Export to Excel
    excel_path = os.path.join(output_dir, "switzerland_vs_eu27_all_data.xlsx")
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main data sheet
            export_df.to_excel(writer, sheet_name='All_Data', index=False)
            
            # Create a summary sheet with indicator descriptions
            indicator_summary = []
            for indicator in sorted(all_indicators):
                description = get_display_name(indicator)
                indicator_summary.append({
                    'indicator_code': indicator,
                    'indicator_name': description
                })
            
            summary_df = pd.DataFrame(indicator_summary)
            summary_df.to_excel(writer, sheet_name='Indicator_Definitions', index=False)
            
        print(f"Exported all data to Excel: {excel_path}")
        print(f"Total records: {len(export_df)}")
        print(f"Indicators: {len(all_indicators)}")
        print(f"Countries: {export_df['geo'].nunique()}")
        print(f"Years: {export_df['year'].min()}-{export_df['year'].max()}")
        
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        # Fallback to CSV
        csv_path = os.path.join(output_dir, "switzerland_vs_eu27_all_data.csv")
        export_df.to_csv(csv_path, index=False)
        print(f"Exported to CSV instead: {csv_path}")

def main():
    """Main function to generate time series graphs for app-displayed Level 5 indicators only"""
    
    print("Starting Swiss vs EU-27 App-Displayed Level 5 Indicators Time Series Analysis")
    print("=" * 80)
    
    # Get Level 5 indicators that are displayed in the app
    print("\n1. Getting Level 5 indicators displayed in the app...")
    app_indicators = get_app_level5_indicators()
    
    if not app_indicators:
        print("Error: No Level 5 indicators found in the app")
        return
    
    print(f"Found {len(app_indicators)} Level 5 indicators displayed in the app:")
    for indicator in app_indicators:
        description = get_display_name(indicator)
        print(f"   • {description}")
    
    # Load data
    try:
        unified_df = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Filter data to only include app-displayed Level 5 indicators
    print(f"\n2. Filtering data for app-displayed indicators...")
    unified_df = unified_df[unified_df['Primary and raw data'].isin(app_indicators)].copy()
    print(f"Filtered data shape: {unified_df.shape}")
    
    # Prepare data
    print("\n3. Preparing Switzerland data...")
    swiss_data, swiss_decile_1, swiss_decile_10 = prepare_swiss_data(unified_df)
    
    print("\n4. Preparing EU-27 data...")
    eu27_data, eu27_decile_1, eu27_decile_10 = prepare_eu27_data(unified_df)
    
    if swiss_data.empty or eu27_data.empty:
        print("Error: Could not load required data")
        return
    
    # Get output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    well_being_dir = os.path.abspath(os.path.join(current_dir, '..'))
    output_dir = os.path.abspath(os.path.join(well_being_dir, 'output', 'Graphs', 'Graphs Switzerland'))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process ALL app indicators, even if Switzerland or EU has no data
    swiss_indicators = set(swiss_data['Indicator'].unique()) if not swiss_data.empty else set()
    eu27_indicators = set(eu27_data['Indicator'].unique()) if not eu27_data.empty else set()
    
    # Use ALL app indicators, not just common ones
    all_indicators = set(app_indicators)
    
    print(f"\n5. Processing ALL {len(all_indicators)} app-displayed indicators:")
    print(f"   • Switzerland has data for: {len(swiss_indicators)} indicators")
    print(f"   • EU-27 has data for: {len(eu27_indicators)} indicators")
    print(f"   • Common indicators: {len(swiss_indicators.intersection(eu27_indicators))}")
    print(f"   • Will generate graphs for ALL {len(all_indicators)} app indicators")
    
    for indicator in sorted(all_indicators):
        description = get_display_name(indicator)
        has_swiss = indicator in swiss_indicators
        has_eu = indicator in eu27_indicators
        status = []
        if has_swiss:
            status.append("CH")
        if has_eu:
            status.append("EU")
        status_str = "+".join(status) if status else "No data"
        print(f"   • {description} ({status_str})")
    
    print(f"\n6. Generating time series graphs...")
    
    # Generate graphs for each indicator
    for i, indicator in enumerate(sorted(all_indicators), 1):
        description = get_display_name(indicator)
        print(f"\n6.{i}. Processing {description}...")
        
        # Check data availability for this indicator
        has_swiss_data = indicator in swiss_indicators
        has_eu_data = indicator in eu27_indicators
        
        if not has_swiss_data and not has_eu_data:
            print(f"   Warning: No data available for {indicator}. Skipping...")
            continue
        try:
            create_time_series_plot(swiss_data, eu27_data, swiss_decile_1, swiss_decile_10, 
                                  eu27_decile_1, eu27_decile_10, indicator, output_dir)
        except Exception as e:
            print(f"Error creating plot for {indicator}: {e}")
    
    # Create summary statistics
    print(f"\n7. Creating summary statistics...")
    try:
        summary_df = create_summary_stats(swiss_data, eu27_data, all_indicators, output_dir)
        print(f"Summary covers {len(summary_df)} indicators")
    except Exception as e:
        print(f"Error creating summary statistics: {e}")
    
    # Export all data to Excel
    print(f"\n8. Exporting all data to Excel...")
    try:
        export_all_data_to_excel(swiss_data, eu27_data, swiss_decile_1, swiss_decile_10, 
                                eu27_decile_1, eu27_decile_10, all_indicators, output_dir)
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
    
    print("\n" + "=" * 80)
    print("✅ All graphs for app-displayed Level 5 indicators generated successfully!")
    print(f"📁 Output directory: {output_dir}")
    print("\nGenerated files:")
    
    # List generated files
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith(('.html', '.png', '.csv'))]
        for file in sorted(files):
            print(f"  • {file}")
    
    print(f"📊 Summary:")
    print(f"  • {len(all_indicators)} app-displayed Level 5 indicators analyzed")
    print(f"  • Time period: Dynamic ranges (earliest EU/Swiss start to latest Swiss end)")
    print(f"  • Countries: Switzerland vs EU-27")
    print(f"  • Output formats: HTML (interactive), PNG (static), CSV (statistics)")
    print(f"  • Graph titles use descriptions only (no indicator codes)")
    
    print(f"\n💡 Note:")
    print(f"  • Only Level 5 indicators displayed in the dashboard app are included")
    print(f"  • Swiss data comes from individual country data (Level 5)")
    print(f"  • EU-27 data uses population-weighted geometric means (Level 5)")
    print(f"  • All values are PCA-normalized for comparability")
    print(f"  • Colors: Switzerland median and deciles ({SWITZERLAND_COLOR}), EU-27 median and deciles ({EU_27_COLOR})")
    print(f"  • Titles show descriptions only (e.g., 'Self-perceived general health' instead of 'AH-SILC-2')")
    print(f"  • Both HTML (interactive) and PNG (static) formats generated")
    print(f"  • Includes 1st and 10th decile lines (dotted) for both countries")
    print(f"  • Time ranges: Start = earliest EU/Swiss data, End = latest Swiss data")

if __name__ == "__main__":
    main()