"""
ewbi_treatment.py - Generate time series comparison graphs for France vs EU-27 focusing on Energy and Housing

This script creates time series graphs comparing France and EU-27 values for Energy and Housing indicators,
plus additional graphs for France alone showing:
- EWBI (Level 1) overall and by decile decomposition
- Energy and Housing EU priority (Level 2) overall and by decile decomposition

The graphs are saved in the output directory with focus on housing and energy indicators.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import numpy as np
import sys

# Add the current directory and shared code to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
report_dir = os.path.abspath(os.path.join(current_dir, '..'))
reports_dir = os.path.abspath(os.path.join(report_dir, '..'))
shared_code_dir = os.path.join(reports_dir, 'shared', 'code')
well_being_code_dir = os.path.abspath(os.path.join(reports_dir, '..', '..', 'Well-being', 'code'))

sys.path.insert(0, current_dir)
sys.path.insert(0, shared_code_dir)
sys.path.insert(0, well_being_code_dir)

# Import shared utilities - try with error handling
try:
    from ewbi_data_loader import load_ewbi_unified_data, get_housing_energy_indicators
    from visualization_utils import create_time_series_plot, save_plot
except ImportError:
    # Fallback: provide stub implementations
    def load_ewbi_unified_data():
        # Load directly from Well-being output
        data_path = os.path.abspath(os.path.join(well_being_code_dir, '..', 'output', 'unified_all_levels_1_to_5_pca_weighted.csv'))
        if os.path.exists(data_path):
            return pd.read_csv(data_path, low_memory=False)
        raise FileNotFoundError(f"Could not find EWBI data at {data_path}")
    
    def get_housing_energy_indicators():
        return []
    
    def create_time_series_plot(*args, **kwargs):
        pass
    
    def save_plot(*args, **kwargs):
        pass

# Import variable mapping functions from Well-being pipeline
try:
    from variable_mapping import get_display_name
except ImportError as e:
    print(f"Warning: Could not import get_display_name: {e}")
    def get_display_name(indicator):
        return indicator

# Color palette consistent with 6_graphs.py and dashboard
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'
FRANCE_COLOR = '#ffd558'  # Use yellow for France as requested

def get_app_level5_indicators():
    """Get the Level 5 indicators that are actually available in the app's dropdown by replicating app.py logic"""
    # Use the shared EWBI data loader to get the unified dataset
    try:
        df = load_ewbi_unified_data()
    except Exception as e:
        print(f"Error loading EWBI unified data for app indicators: {e}")
        return []

    # EU priority focused on Energy and Housing only
    EU_PRIORITIES = [
        'Energy and Housing'
    ]

    # Get Level 5 indicators that would appear in app dropdowns
    all_available_indicators = set()

    for eu_priority in EU_PRIORITIES:
        primary_options = df[
            (df.get('EU priority') == eu_priority) &
            (df.get('Primary and raw data').notna()) &
            (df.get('Level') == 5)
        ]['Primary and raw data'].unique()

        print(f"EU Priority '{eu_priority}': {len(primary_options)} indicators")
        for indicator in primary_options:
            display_name = get_display_name(indicator)
            print(f"  • {display_name}")

        all_available_indicators.update(primary_options)

    print(f"\nTotal unique Level 5 indicators available in app dropdowns: {len(all_available_indicators)}")
    return sorted(all_available_indicators)

def load_data():
    """Load the primary indicators data for time series analysis using shared utilities"""
    # Use shared utility to load EWBI data
    unified_df = load_ewbi_unified_data()
    return unified_df

def prepare_france_ewbi_data(unified_df):
    """Extract France EWBI (Level 1) data for overall and decile analysis"""
    # Get France EWBI overall data (Level 1, Decile='All')
    france_ewbi_overall = unified_df[
        (unified_df['Country'] == 'FR') &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] == 'All')
    ].copy()
    
    # Get France EWBI by deciles (Level 1, all deciles except 'All')
    france_ewbi_deciles = unified_df[
        (unified_df['Country'] == 'FR') &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna())
    ].copy()
    
    print(f"France EWBI overall data: {len(france_ewbi_overall)} records")
    print(f"France EWBI decile data: {len(france_ewbi_deciles)} records")
    
    return france_ewbi_overall, france_ewbi_deciles

def prepare_france_eu_priorities_data(unified_df):
    """Extract France EU priorities (Level 2) data for overall and decile analysis"""
    # Focus only on Energy and Housing EU priority
    EU_PRIORITIES = [
        'Energy and Housing'
    ]
    
    # Get France EU priorities overall data (Level 2, Decile='All')
    # For France: NO aggregation filtering (matches app.py logic)
    france_priorities_overall = unified_df[
        (unified_df['Country'] == 'FR') &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] == 'All') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Get France EU priorities by deciles (Level 2, all deciles except 'All')
    # For France: NO aggregation filtering (matches app.py logic)
    france_priorities_deciles = unified_df[
        (unified_df['Country'] == 'FR') &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna()) &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    print(f"France EU priorities overall data: {len(france_priorities_overall)} records")
    print(f"France EU priorities decile data: {len(france_priorities_deciles)} records")
    print(f"Available EU priorities: {sorted(france_priorities_overall['EU priority'].unique())}")
    
    return france_priorities_overall, france_priorities_deciles, EU_PRIORITIES

def prepare_eu27_ewbi_data(unified_df):
    """Extract EU-27 EWBI (Level 1) data for overall and decile analysis - matches app.py logic"""
    # Get EU-27 EWBI overall data (Level 1, Decile='All') - use Population-weighted geometric mean
    eu27_ewbi_overall = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] == 'All') &
        (unified_df['Aggregation'] == 'Population-weighted geometric mean')
    ].copy()
    
    # Get EU-27 EWBI by deciles (Level 1, all deciles except 'All') - use Population-weighted geometric mean
    eu27_ewbi_deciles = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna()) &
        (unified_df['Aggregation'] == 'Population-weighted geometric mean')
    ].copy()
    
    print(f"EU-27 EWBI overall data: {len(eu27_ewbi_overall)} records")
    print(f"EU-27 EWBI decile data: {len(eu27_ewbi_deciles)} records")
    
    return eu27_ewbi_overall, eu27_ewbi_deciles

def prepare_eu27_eu_priorities_data(unified_df):
    """Extract EU-27 EU priorities (Level 2) data for overall and decile analysis - matches app.py logic"""
    EU_PRIORITIES = [
        'Energy and Housing'
    ]
    
    # Get EU-27 EU priorities overall data (Level 2, Decile='All') 
    # For reference lines: Do NOT filter by aggregation (to match app.py logic)
    eu27_priorities_overall = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] == 'All')
    ].copy()
    
    # Get EU-27 EU priorities by deciles (Level 2, all deciles except 'All') - use Population-weighted geometric mean
    eu27_priorities_deciles = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna()) &
        (unified_df['Aggregation'] == 'Population-weighted geometric mean')
    ].copy()
    
    print(f"EU-27 EU priorities overall data: {len(eu27_priorities_overall)} records")
    print(f"EU-27 EU priorities decile data: {len(eu27_priorities_deciles)} records")
    
    return eu27_priorities_overall, eu27_priorities_deciles, EU_PRIORITIES

def create_ewbi_overall_comparison_plot(france_ewbi_overall, eu27_ewbi_overall, output_dir):
    """Create EWBI overall temporal graph comparing France vs EU-27"""
    if france_ewbi_overall.empty and eu27_ewbi_overall.empty:
        print("Warning: No EWBI overall data available for France or EU-27")
        return
    
    # Sort by year
    if not france_ewbi_overall.empty:
        france_ewbi_overall = france_ewbi_overall.sort_values('Year')
    if not eu27_ewbi_overall.empty:
        eu27_ewbi_overall = eu27_ewbi_overall.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add France EWBI line (solid yellow) if data exists
    if not france_ewbi_overall.empty:
        fig.add_trace(go.Scatter(
            x=france_ewbi_overall['Year'],
            y=france_ewbi_overall['Value'],
            mode='lines+markers',
            name='France EWBI',
            line=dict(color=FRANCE_COLOR, width=4),
            marker=dict(color=FRANCE_COLOR, size=10, symbol='circle'),
            hovertemplate='<b>France EWBI</b><br>' +
                         'Year: %{x}<br>' +
                         'Score: %{y:.3f}<extra></extra>'
        ))
    
    # Add EU-27 EWBI line (solid blue) if data exists
    if not eu27_ewbi_overall.empty:
        fig.add_trace(go.Scatter(
            x=eu27_ewbi_overall['Year'],
            y=eu27_ewbi_overall['Value'],
            mode='lines+markers',
            name='EU-27 EWBI',
            line=dict(color=EU_27_COLOR, width=4),
            marker=dict(color=EU_27_COLOR, size=10, symbol='circle'),
            hovertemplate='<b>EU-27 EWBI</b><br>' +
                         'Year: %{x}<br>' +
                         'Score: %{y:.3f}<extra></extra>'
        ))
    
    # Get time range
    all_years = []
    if not france_ewbi_overall.empty:
        all_years.extend(france_ewbi_overall['Year'].tolist())
    if not eu27_ewbi_overall.empty:
        all_years.extend(eu27_ewbi_overall['Year'].tolist())
    
    if all_years:
        earliest_year = min(all_years)
        latest_year = max(all_years)
    else:
        earliest_year = latest_year = 2020  # fallback
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"European Well-Being Index (EWBI)<br><sub>France vs EU-27 Comparison ({int(earliest_year)}-{int(latest_year)})</sub>",
            font=dict(size=18, color="#2C3E50", family="Arial, sans-serif"),
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
            dtick=1 if len(all_years) <= 10 else 2,
            tickmode='linear',
            range=[earliest_year - 0.5, latest_year + 0.5]
        ),
        yaxis=dict(
            title="EWBI Score",
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
    
    # Save PNG only
    png_path = os.path.join(output_dir, "France_vs_eu27_ewbi_overall.png")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        if not france_ewbi_overall.empty:
            plt.plot(france_ewbi_overall['Year'], france_ewbi_overall['Value'], 
                    color=FRANCE_COLOR, linewidth=4, marker='o', 
                    markersize=10, label='France EWBI', alpha=0.9)
        
        if not eu27_ewbi_overall.empty:
            plt.plot(eu27_ewbi_overall['Year'], eu27_ewbi_overall['Value'], 
                    color=EU_27_COLOR, linewidth=4, marker='o', 
                    markersize=10, label='EU-27 EWBI', alpha=0.9)
        
        plt.title(f"European Well-Being Index (EWBI)\nFrance vs EU-27 Comparison ({int(earliest_year)}-{int(latest_year)})", 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('EWBI Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        
        # Set x-axis ticks
        if len(all_years) <= 10:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1))
        else:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1, 2))
            
        plt.xlim(earliest_year - 0.5, latest_year + 0.5)
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for EWBI overall: {str(e)[:100]}...")

def create_ewbi_deciles_comparison_plot(france_ewbi_deciles, eu27_ewbi_deciles, output_dir):
    """Create EWBI decile decomposition bar chart comparing France vs EU-27 - latest year only"""
    if france_ewbi_deciles.empty and eu27_ewbi_deciles.empty:
        print("Warning: No EWBI decile data available for France or EU-27")
        return
    
    # Get latest year available (prefer France data if available, otherwise EU-27)
    if not france_ewbi_deciles.empty:
        latest_year = france_ewbi_deciles['Year'].max()
    else:
        latest_year = eu27_ewbi_deciles['Year'].max()
    
    # Filter data for latest year only
    france_latest = pd.DataFrame()
    eu27_latest = pd.DataFrame()
    
    if not france_ewbi_deciles.empty:
        france_latest = france_ewbi_deciles[france_ewbi_deciles['Year'] == latest_year].copy()
        # Convert decile to numeric and sort
        france_latest['Decile_num'] = pd.to_numeric(france_latest['Decile'], errors='coerce')
        france_latest = france_latest.dropna(subset=['Decile_num'])
        france_latest = france_latest.sort_values('Decile_num')
    
    if not eu27_ewbi_deciles.empty:
        eu27_latest = eu27_ewbi_deciles[eu27_ewbi_deciles['Year'] == latest_year].copy()
        # Convert decile to numeric and sort
        eu27_latest['Decile_num'] = pd.to_numeric(eu27_latest['Decile'], errors='coerce')
        eu27_latest = eu27_latest.dropna(subset=['Decile_num'])
        eu27_latest = eu27_latest.sort_values('Decile_num')
    
    if france_latest.empty and eu27_latest.empty:
        print("Warning: No valid decile data for latest year")
        return
    
    # Create the plot
    fig = go.Figure()
    
    # Get all deciles that have data
    all_deciles = set()
    if not france_latest.empty:
        all_deciles.update(france_latest['Decile_num'])
    if not eu27_latest.empty:
        all_deciles.update(eu27_latest['Decile_num'])
    all_deciles = sorted(list(all_deciles))
    
    # Create bar chart for France
    if not france_latest.empty:
        france_x = [str(int(d)) for d in france_latest['Decile_num']]
        france_y = france_latest['Value'].tolist()
        fig.add_trace(go.Bar(
            x=france_x,
            y=france_y,
            name='France EWBI',
            marker=dict(color=FRANCE_COLOR),
            text=[f'{v:.3f}' for v in france_y],
            textposition='auto',
            hovertemplate='<b>France - Decile %{x}</b><br>' +
                         'Score: %{y:.3f}<extra></extra>',
            offsetgroup=1
        ))
        
        # Add reference line for France average
        france_average = france_latest['Value'].mean()
        fig.add_hline(
            y=france_average,
            line_dash="dash",
            line_color=FRANCE_COLOR,
            line_width=2,
            annotation_text=f"France Avg: {france_average:.3f}",
            annotation_position="top right"
        )
    
    # Create bar chart for EU-27
    if not eu27_latest.empty:
        eu27_x = [str(int(d)) for d in eu27_latest['Decile_num']]
        eu27_y = eu27_latest['Value'].tolist()
        fig.add_trace(go.Bar(
            x=eu27_x,
            y=eu27_y,
            name='EU-27 EWBI',
            marker=dict(color=EU_27_COLOR),
            text=[f'{v:.3f}' for v in eu27_y],
            textposition='auto',
            hovertemplate='<b>EU-27 - Decile %{x}</b><br>' +
                         'Score: %{y:.3f}<extra></extra>',
            offsetgroup=2
        ))
        
        # Add reference line for EU-27 average
        eu27_average = eu27_latest['Value'].mean()
        fig.add_hline(
            y=eu27_average,
            line_dash="dot",
            line_color=EU_27_COLOR,
            line_width=2,
            annotation_text=f"EU-27 Avg: {eu27_average:.3f}",
            annotation_position="bottom left"
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"European Well-Being Index (EWBI)<br><sub>France vs EU-27 by Income Decile ({int(latest_year)})</sub>",
            font=dict(size=18, color="#2C3E50", family="Arial, sans-serif"),
            x=0.5
        ),
        xaxis=dict(
            title="Income Decile",
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=12, family="Arial, sans-serif"),
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            title="EWBI Score",
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
        barmode='group',
        margin=dict(t=80, b=60, l=60, r=40),
        width=1200,
        height=600
    )
    
    # Save PNG only
    png_path = os.path.join(output_dir, "France_vs_eu27_ewbi_deciles.png")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(14, 8))
        
        # Prepare data for grouped bar chart
        deciles = [str(int(d)) for d in all_deciles]
        x = np.arange(len(deciles))
        width = 0.35
        
        # Get values for each decile, use NaN for missing data
        france_values = []
        eu27_values = []
        
        for decile in all_deciles:
            # France values
            france_val = france_latest[france_latest['Decile_num'] == decile]['Value']
            france_values.append(france_val.iloc[0] if len(france_val) > 0 else np.nan)
            
            # EU-27 values
            eu27_val = eu27_latest[eu27_latest['Decile_num'] == decile]['Value']
            eu27_values.append(eu27_val.iloc[0] if len(eu27_val) > 0 else np.nan)
        
        # Create France bars only
        if not france_latest.empty:
            france_bars = plt.bar(x, france_values, width=0.4, 
                               color=FRANCE_COLOR, alpha=0.8, label='France (Deciles)', edgecolor='black')
            # Add value labels on bars
            for i, v in enumerate(france_values):
                if not np.isnan(v):
                    plt.text(i, v + 0.001, f'{v:.3f}', 
                           ha='center', va='bottom', fontsize=9)
        
        # Overlay EU-27 line
        if not eu27_latest.empty:
            eu27_x_pos = []
            eu27_line_values = []
            for i, decile in enumerate(all_deciles):
                val = eu27_latest[eu27_latest['Decile_num'] == decile]['Value']
                if len(val) > 0:
                    eu27_x_pos.append(i)
                    eu27_line_values.append(val.iloc[0])
            
            if eu27_line_values:
                plt.plot(eu27_x_pos, eu27_line_values, color=EU_27_COLOR, linewidth=3,
                        marker='o', markersize=8, label='EU-27 (Average by Decile)', alpha=0.9)
        
        # Add reference lines for overall averages
        if not france_latest.empty:
            france_avg = france_latest['Value'].mean()
            plt.axhline(y=france_avg, color=FRANCE_COLOR, linestyle='--', 
                       linewidth=2, alpha=0.6, label=f'France Overall: {france_avg:.3f}')
        
        if not eu27_latest.empty:
            eu27_avg = eu27_latest['Value'].mean()
            plt.axhline(y=eu27_avg, color=EU_27_COLOR, linestyle=':', 
                       linewidth=2, alpha=0.6, label=f'EU-27 Overall: {eu27_avg:.3f}')
        
        plt.title(f"European Well-Being Index (EWBI)\nFrance vs EU-27 by Income Decile ({int(latest_year)})", 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Income Decile', fontsize=14)
        plt.ylabel('EWBI Score', fontsize=14)
        plt.xticks(x, deciles)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=12)
        
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for EWBI deciles: {str(e)[:100]}...")

def create_eu_priority_comparison_plots(france_priorities_overall, france_priorities_deciles, 
                                       eu27_priorities_overall, eu27_priorities_deciles, 
                                       eu_priorities, output_dir):
    """Create EU priority comparison plots for France vs EU-27 (overall and deciles) for each priority"""
    
    for priority in eu_priorities:
        print(f"Creating plots for EU priority: {priority}")
        
        # Filter France data for this priority
        france_overall = france_priorities_overall[
            france_priorities_overall['EU priority'] == priority
        ].copy() if not france_priorities_overall.empty else pd.DataFrame()
        
        france_deciles = france_priorities_deciles[
            france_priorities_deciles['EU priority'] == priority
        ].copy() if not france_priorities_deciles.empty else pd.DataFrame()
        
        # Filter EU-27 data for this priority
        eu27_overall = eu27_priorities_overall[
            eu27_priorities_overall['EU priority'] == priority
        ].copy() if not eu27_priorities_overall.empty else pd.DataFrame()
        
        eu27_deciles = eu27_priorities_deciles[
            eu27_priorities_deciles['EU priority'] == priority
        ].copy() if not eu27_priorities_deciles.empty else pd.DataFrame()
        
        # Create overall comparison plot
        if not france_overall.empty or not eu27_overall.empty:
            create_eu_priority_overall_comparison_plot(france_overall, eu27_overall, priority, output_dir)
        else:
            print(f"Warning: No overall data for {priority}")
        
        # Create deciles comparison plot
        if not france_deciles.empty or not eu27_deciles.empty:
            create_eu_priority_deciles_comparison_plot(france_deciles, eu27_deciles, france_overall, eu27_overall, priority, output_dir)
        else:
            print(f"Warning: No decile data for {priority}")

def create_eu_priority_overall_comparison_plot(france_data, eu27_data, priority_name, output_dir):
    """Create overall temporal comparison graph for a specific EU priority"""
    
    # FIXED: Filter EU-27 data by aggregation method to match app.py time series logic
    if not eu27_data.empty:
        eu27_data = eu27_data[eu27_data['Aggregation'] == 'Population-weighted geometric mean'].copy()
    
    # Sort by year
    if not france_data.empty:
        france_data = france_data.sort_values('Year')
    if not eu27_data.empty:
        eu27_data = eu27_data.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add France EU priority line (solid yellow)
    if not france_data.empty:
        fig.add_trace(go.Scatter(
            x=france_data['Year'],
            y=france_data['Value'],
            mode='lines+markers',
            name=f'France {priority_name}',
            line=dict(color=FRANCE_COLOR, width=4),
            marker=dict(color=FRANCE_COLOR, size=10, symbol='circle'),
            hovertemplate=f'<b>France {priority_name}</b><br>' +
                         'Year: %{x}<br>' +
                         'Score: %{y:.3f}<extra></extra>'
        ))
    
    # Add EU-27 EU priority line (solid blue)
    if not eu27_data.empty:
        fig.add_trace(go.Scatter(
            x=eu27_data['Year'],
            y=eu27_data['Value'],
            mode='lines+markers',
            name=f'EU-27 {priority_name}',
            line=dict(color=EU_27_COLOR, width=4),
            marker=dict(color=EU_27_COLOR, size=10, symbol='circle'),
            hovertemplate=f'<b>EU-27 {priority_name}</b><br>' +
                         'Year: %{x}<br>' +
                         'Score: %{y:.3f}<extra></extra>'
        ))
    
    # Get time range
    all_years = []
    if not france_data.empty:
        all_years.extend(france_data['Year'].tolist())
    if not eu27_data.empty:
        all_years.extend(eu27_data['Year'].tolist())
    
    if all_years:
        earliest_year = min(all_years)
        latest_year = max(all_years)
    else:
        earliest_year = latest_year = 2020
    
    # Create safe filename
    safe_priority = priority_name.lower().replace(' ', '_').replace(',', '').replace('and', 'and')
    # Handle specific long names to avoid Windows path issues
    if len(safe_priority) > 30:
        if 'intergenerational' in safe_priority:
            safe_priority = 'intergenerational_fairness'
        elif 'social_rights' in safe_priority:
            safe_priority = 'social_rights_quality_jobs'
        elif 'health' in safe_priority:
            safe_priority = 'health_animal_welfare'
        elif 'energy' in safe_priority:
            safe_priority = 'energy_and_housing'
        elif 'equality' in safe_priority:
            safe_priority = 'equality'
        else:
            safe_priority = safe_priority[:25]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{priority_name}<br><sub>France vs EU-27 Comparison ({int(earliest_year)}-{int(latest_year)})</sub>",
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
            dtick=1 if len(all_years) <= 10 else 2,
            tickmode='linear',
            range=[earliest_year - 0.5, latest_year + 0.5]
        ),
        yaxis=dict(
            title="Score",
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
    
    # Save PNG only
    png_path = os.path.join(output_dir, f"France_vs_eu27_{safe_priority}_overall.png")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        if not france_data.empty:
            plt.plot(france_data['Year'], france_data['Value'], 
                    color=FRANCE_COLOR, linewidth=4, marker='o', 
                    markersize=10, label=f'France {priority_name}', alpha=0.9)
        
        if not eu27_data.empty:
            plt.plot(eu27_data['Year'], eu27_data['Value'], 
                    color=EU_27_COLOR, linewidth=4, marker='o', 
                    markersize=10, label=f'EU-27 {priority_name}', alpha=0.9)
        
        plt.title(f"{priority_name}\nFrance vs EU-27 Comparison ({int(earliest_year)}-{int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        
        if len(all_years) <= 10:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1))
        else:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1, 2))
            
        plt.xlim(earliest_year - 0.5, latest_year + 0.5)
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {priority_name}: {str(e)[:100]}...")

def create_eu_priority_deciles_comparison_plot(france_data, eu27_data, france_overall, eu27_overall, priority_name, output_dir):
    """Create deciles comparison bar chart for a specific EU priority"""
    print(f"DEBUG: Creating deciles chart for {priority_name}")
    print(f"DEBUG: France data: {len(france_data)} records")
    print(f"DEBUG: EU-27 data: {len(eu27_data)} records") 
    print(f"DEBUG: France overall: {len(france_overall)} records")
    print(f"DEBUG: EU-27 overall: {len(eu27_overall)} records")
    
    if france_data.empty and eu27_data.empty:
        print(f"DEBUG: No data for {priority_name}")
        return
    
    # Get latest year available
    if not france_data.empty:
        latest_year = france_data['Year'].max()
    else:
        latest_year = eu27_data['Year'].max()
    
    # Filter data for latest year
    france_latest = pd.DataFrame()
    eu27_latest = pd.DataFrame()
    
    if not france_data.empty:
        france_latest = france_data[france_data['Year'] == latest_year].copy()
        france_latest['Decile_num'] = pd.to_numeric(france_latest['Decile'], errors='coerce')
        france_latest = france_latest.dropna(subset=['Decile_num']).sort_values('Decile_num')
    
    if not eu27_data.empty:
        eu27_latest = eu27_data[eu27_data['Year'] == latest_year].copy()
        eu27_latest['Decile_num'] = pd.to_numeric(eu27_latest['Decile'], errors='coerce')
        eu27_latest = eu27_latest.dropna(subset=['Decile_num']).sort_values('Decile_num')
    
    if france_latest.empty and eu27_latest.empty:
        return
    
    # Get all deciles
    all_deciles = set()
    if not france_latest.empty:
        all_deciles.update(france_latest['Decile_num'])
    if not eu27_latest.empty:
        all_deciles.update(eu27_latest['Decile_num'])
    all_deciles = sorted(list(all_deciles))
    
    # Get reference values for legend
    france_reference_value = None
    eu27_reference_value = None
    
    # Get France reference value
    if not france_latest.empty:
        france_all_latest = france_overall[
            (france_overall['Year'] == latest_year) & 
            (france_overall['Decile'] == 'All')
        ]
        if not france_all_latest.empty:
            france_geom_interdecile = france_all_latest[france_all_latest['Aggregation'] == 'Geometric mean inter-decile']
            if not france_geom_interdecile.empty:
                france_reference_value = france_geom_interdecile['Value'].iloc[0]
    
    # Get EU-27 reference value  
    if not eu27_latest.empty:
        eu27_all_latest = eu27_overall[
            (eu27_overall['Year'] == latest_year) & 
            (eu27_overall['Decile'] == 'All')
        ]
        if not eu27_all_latest.empty:
            eu27_geom_interdecile = eu27_all_latest[eu27_all_latest['Aggregation'] == 'Geometric mean inter-decile']
            if not eu27_geom_interdecile.empty and not pd.isna(eu27_geom_interdecile['Value'].iloc[0]):
                eu27_reference_value = eu27_geom_interdecile['Value'].iloc[0]
    
    # Create safe filename
    safe_priority = priority_name.lower().replace(' ', '_').replace(',', '').replace('and', 'and')
    if len(safe_priority) > 30:
        if 'intergenerational' in safe_priority:
            safe_priority = 'intergenerational_fairness'
        elif 'social_rights' in safe_priority:
            safe_priority = 'social_rights_quality_jobs'
        elif 'health' in safe_priority:
            safe_priority = 'health_animal_welfare'
        elif 'energy' in safe_priority:
            safe_priority = 'energy_and_housing'
        elif 'equality' in safe_priority:
            safe_priority = 'equality'
        else:
            safe_priority = safe_priority[:25]
    
    # MATPLOTLIB VERSION - France bars with EU-27 line overlay
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 8))
        
        all_deciles_sorted = sorted(list(all_deciles))
        decile_labels = [str(int(d)) for d in all_deciles_sorted]
        x_pos = np.arange(len(decile_labels))
        
        # Get France values
        france_vals = []
        for decile in all_deciles_sorted:
            val = france_latest[france_latest['Decile_num'] == decile]['Value']
            france_vals.append(val.iloc[0] if len(val) > 0 else np.nan)
        
        # Plot France bars
        if not france_latest.empty:
            plt.bar(x_pos, france_vals, width=0.4, color=FRANCE_COLOR, alpha=0.8,
                   label='France (Deciles)', edgecolor='black')
            # Add value labels
            for i, v in enumerate(france_vals):
                if not np.isnan(v):
                    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Overlay EU-27 line
        eu27_vals = []
        eu27_x_pos = []
        for i, decile in enumerate(all_deciles_sorted):
            val = eu27_latest[eu27_latest['Decile_num'] == decile]['Value']
            if len(val) > 0:
                eu27_vals.append(val.iloc[0])
                eu27_x_pos.append(i)
        
        if eu27_vals:
            plt.plot(eu27_x_pos, eu27_vals, color=EU_27_COLOR, linewidth=3,
                    marker='o', markersize=8, label='EU-27 (Average by Decile)', alpha=0.9)
        
        # Add reference lines
        if france_reference_value is not None:
            plt.axhline(y=france_reference_value, color=FRANCE_COLOR, linestyle='--', 
                       linewidth=2, alpha=0.6, label=f'France Overall: {france_reference_value:.3f}')
        
        if eu27_reference_value is not None:
            plt.axhline(y=eu27_reference_value, color=EU_27_COLOR, linestyle=':', 
                       linewidth=2, alpha=0.6, label=f'EU-27 Overall: {eu27_reference_value:.3f}')
        
        plt.title(f"{priority_name}\\nFrance by Income Decile vs EU-27 ({int(latest_year)})", 
                 fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Income Decile', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.xticks(x_pos, decile_labels)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=11, loc='best')
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, f"France_vs_eu27_{safe_priority}_deciles.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved PNG: {png_path}")
        return
    except Exception as e:
        print(f"Warning: Could not save matplotlib PNG for {priority_name} deciles: {str(e)[:100]}...")

def create_france_eu_priority_overall_plot(priority_data, priority_name, output_dir):
    """Create overall temporal graph for a specific EU priority in France"""
    # Sort by year
    priority_data = priority_data.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add France EU priority line (solid yellow)
    fig.add_trace(go.Scatter(
        x=priority_data['Year'],
        y=priority_data['Value'],
        mode='lines+markers',
        name=f'France {priority_name}',
        line=dict(color=FRANCE_COLOR, width=4),
        marker=dict(color=FRANCE_COLOR, size=10, symbol='circle'),
        hovertemplate=f'<b>France {priority_name}</b><br>' +
                     'Year: %{x}<br>' +
                     'Score: %{y:.3f}<extra></extra>'
    ))
    
    # Get time range
    years = priority_data['Year'].tolist()
    earliest_year = min(years)
    latest_year = max(years)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{priority_name} - France<br><sub>Overall Score Evolution ({int(earliest_year)}-{int(latest_year)})</sub>",
            font=dict(size=18, color="#2C3E50", family="Arial, sans-serif"),
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
            dtick=1 if len(years) <= 10 else 2,
            tickmode='linear',
            range=[earliest_year - 0.5, latest_year + 0.5]
        ),
        yaxis=dict(
            title="Score",
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
    
    # Create safe filename - limit length and remove problematic characters
    safe_priority = priority_name.replace(' ', '_').replace(',', '').replace('&', 'and').replace('-', '_')
    # Truncate if too long to avoid Windows path length issues
    if len(safe_priority) > 30:
        safe_priority = safe_priority[:30]
    
    # Save PNG only
    png_path = os.path.join(output_dir, f"France_{safe_priority.lower()}_overall.png")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        plt.plot(priority_data['Year'], priority_data['Value'], 
                color=FRANCE_COLOR, linewidth=4, marker='o', 
                markersize=10, label=f'France {priority_name}', alpha=0.9)
        
        plt.title(f"{priority_name} - France\nOverall Score Evolution ({int(earliest_year)}-{int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        
        # Set x-axis ticks
        if len(years) <= 10:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1))
        else:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1, 2))
            
        plt.xlim(earliest_year - 0.5, latest_year + 0.5)
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {priority_name} overall: {str(e)[:100]}...")

def create_france_eu_priority_deciles_plot(priority_deciles, priority_name, output_dir):
    """Create decile decomposition bar chart for a specific EU priority in France - latest year only"""
    # Get latest year available
    latest_year = priority_deciles['Year'].max()
    
    # Filter data for latest year only
    latest_data = priority_deciles[priority_deciles['Year'] == latest_year].copy()
    
    # Convert decile to numeric and sort
    latest_data['Decile_num'] = pd.to_numeric(latest_data['Decile'], errors='coerce')
    latest_data = latest_data.dropna(subset=['Decile_num'])
    latest_data = latest_data.sort_values('Decile_num')
    
    if latest_data.empty:
        print(f"Warning: No valid decile data for {priority_name} in latest year")
        return
    
    # Create the plot
    fig = go.Figure()
    
    # Create bar chart for deciles
    fig.add_trace(go.Bar(
        x=[str(int(d)) for d in latest_data['Decile_num']],
        y=latest_data['Value'],
        name=f'France {priority_name} by Decile',
        marker=dict(color=FRANCE_COLOR),
        text=[f'{v:.3f}' for v in latest_data['Value']],
        textposition='auto',
        hovertemplate='<b>Decile %{x}</b><br>' +
                     'Score: %{y:.3f}<extra></extra>'
    ))
    
    # Add reference line for overall average (computed from deciles)
    overall_average = latest_data['Value'].mean()
    fig.add_hline(
        y=overall_average,
        line_dash="dash",
        line_color=FRANCE_COLOR,
        line_width=2,
        annotation_text=f"France Average: {overall_average:.3f}",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{priority_name} - France<br><sub>By Income Decile ({int(latest_year)})</sub>",
            font=dict(size=16, color="#2C3E50", family="Arial, sans-serif"),
            x=0.5
        ),
        xaxis=dict(
            title="Income Decile",
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=12, family="Arial, sans-serif"),
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            title="Score",
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=12, family="Arial, sans-serif"),
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        showlegend=False,
        margin=dict(t=80, b=60, l=60, r=40),
        width=1000,
        height=600
    )
    
    # Create safe filename - limit length and remove problematic characters
    safe_priority = priority_name.replace(' ', '_').replace(',', '').replace('&', 'and').replace('-', '_')
    # Truncate if too long to avoid Windows path length issues
    if len(safe_priority) > 30:
        safe_priority = safe_priority[:30]
    
    # Save PNG only
    png_path = os.path.join(output_dir, f"France_{safe_priority.lower()}_deciles.png")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        # Create bar chart
        bars = plt.bar([str(int(d)) for d in latest_data['Decile_num']], 
                      latest_data['Value'], 
                      color=FRANCE_COLOR, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, latest_data['Value']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add reference line for average
        plt.axhline(y=overall_average, color=FRANCE_COLOR, linestyle='--', 
                   linewidth=2, alpha=0.7, 
                   label=f'France Average: {overall_average:.3f}')
        
        plt.title(f"{priority_name} - France\nBy Income Decile ({int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Income Decile', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=12)
        
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {priority_name} deciles: {str(e)[:100]}...")

def prepare_france_data(unified_df):
    """Extract and prepare France data for all primary indicators from Level 5, including all deciles"""
    # Get France data from Level 5 (primary indicators) - median/mean data
    france_data = unified_df[
        (unified_df['Country'] == 'FR') &
        (unified_df['Level'] == 5) &
        (unified_df['Decile'] == 'All')
    ].copy()
    
    # Get ALL decile data for France (for bar charts)
    france_deciles_all = unified_df[
        (unified_df['Country'] == 'FR') &
        (unified_df['Level'] == 5) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna())
    ].copy()
    
    # Also keep decile 1 and 10 for backward compatibility
    france_decile_1 = france_deciles_all[france_deciles_all['Decile'] == '1.0'].copy()
    france_decile_10 = france_deciles_all[france_deciles_all['Decile'] == '10.0'].copy()
    
    if france_data.empty:
        print("No 'All' decile data found for France at Level 5. Computing country averages from decile data...")
        
        # Get all France Level 5 data and compute country averages by year and indicator
        france_decile_data = unified_df[
            (unified_df['Country'] == 'FR') & 
            (unified_df['Level'] == 5) &
            (unified_df['Decile'].isin(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']))
        ].copy()
        
        if france_decile_data.empty:
            print("Warning: No France data found at Level 5")
            return pd.DataFrame(), pd.DataFrame(), france_deciles_all
        
        # Compute the mean value across all deciles for each year and indicator
        france_data = france_decile_data.groupby(['Year', 'Primary and raw data']).agg({
            'Value': 'mean',
            'Country': 'first',
            'Level': 'first',
            'Type': 'first',
            'Aggregation': lambda x: 'Mean across deciles',
            'datasource': 'first'
        }).reset_index()
        
        # Add the All decile marker
        france_data['Decile'] = 'All'
        france_data['Quintile'] = ''
        france_data['EU priority'] = ''
        france_data['Secondary'] = ''
        
        print(f"Computed data for {len(france_data)} year-indicator combinations")
    
    # Rename columns for consistency
    france_data = france_data.rename(columns={'Primary and raw data': 'Indicator'})
    france_data['Country_Name'] = 'France'
    
    # Prepare all deciles data
    if not france_deciles_all.empty:
        france_deciles_all = france_deciles_all.rename(columns={'Primary and raw data': 'Indicator'})
        france_deciles_all['Country_Name'] = 'France'
    
    # Prepare decile 1 and 10 for backward compatibility
    if not france_decile_1.empty:
        france_decile_1 = france_decile_1.rename(columns={'Primary and raw data': 'Indicator'})
        france_decile_1['Country_Name'] = 'France'
    
    if not france_decile_10.empty:
        france_decile_10 = france_decile_10.rename(columns={'Primary and raw data': 'Indicator'})
        france_decile_10['Country_Name'] = 'France'
    
    return france_data, france_deciles_all, france_deciles_all

def prepare_eu27_data(unified_df):
    """Extract and prepare EU-27 data for all primary indicators, including all deciles"""
    # Get EU-27 data from the unified dataset at Level 5 (primary indicators) - median data
    # Try the available aggregation method first
    eu27_data = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 5) &
        (unified_df['Decile'] == 'All') &
        (unified_df['Aggregation'] == 'Median across countries')
    ].copy()
    
    # Get ALL decile data for EU-27 (for bar charts)
    eu27_deciles_all = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 5) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna()) &
        (unified_df['Aggregation'] == 'Median across countries')
    ].copy()
    
    # Also keep decile 1 and 10 for backward compatibility
    eu27_decile_1 = eu27_deciles_all[eu27_deciles_all['Decile'] == '1.0'].copy()
    eu27_decile_10 = eu27_deciles_all[eu27_deciles_all['Decile'] == '10.0'].copy()
    
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
    
    # Prepare all deciles data
    if not eu27_deciles_all.empty:
        eu27_deciles_all = eu27_deciles_all.rename(columns={'Primary and raw data': 'Indicator'})
        eu27_deciles_all['Country_Name'] = 'EU-27'
    
    # Prepare decile 1 and 10 for backward compatibility
    if not eu27_decile_1.empty:
        eu27_decile_1 = eu27_decile_1.rename(columns={'Primary and raw data': 'Indicator'})
        eu27_decile_1['Country_Name'] = 'EU-27'
    
    if not eu27_decile_10.empty:
        eu27_decile_10 = eu27_decile_10.rename(columns={'Primary and raw data': 'Indicator'})
        eu27_decile_10['Country_Name'] = 'EU-27'
    
    return eu27_data, eu27_deciles_all, eu27_deciles_all

def create_level5_overall_plot(france_data, eu27_data, indicator, output_dir):
    """Create overall time series plot for Level 5 indicator (France and EU-27 averages only)"""
    
    france_indicator = france_data[france_data['Indicator'] == indicator].copy()
    eu27_indicator = eu27_data[eu27_data['Indicator'] == indicator].copy()
    
    if france_indicator.empty and eu27_indicator.empty:
        return
    
    # Sort by year
    if not france_indicator.empty:
        france_indicator = france_indicator.sort_values('Year')
    if not eu27_indicator.empty:
        eu27_indicator = eu27_indicator.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add France line (solid yellow)
    if not france_indicator.empty:
        fig.add_trace(go.Scatter(
            x=france_indicator['Year'],
            y=france_indicator['Value'],
            mode='lines+markers',
            name='France',
            line=dict(color=FRANCE_COLOR, width=4),
            marker=dict(color=FRANCE_COLOR, size=10, symbol='circle'),
            hovertemplate='<b>France</b><br>Year: %{x}<br>Value: %{y:.3f}<extra></extra>'
        ))
    
    # Add EU-27 line (solid blue)
    if not eu27_indicator.empty:
        fig.add_trace(go.Scatter(
            x=eu27_indicator['Year'],
            y=eu27_indicator['Value'],
            mode='lines+markers',
            name='EU-27',
            line=dict(color=EU_27_COLOR, width=4),
            marker=dict(color=EU_27_COLOR, size=10, symbol='circle'),
            hovertemplate='<b>EU-27</b><br>Year: %{x}<br>Value: %{y:.3f}<extra></extra>'
        ))
    
    # Get time range
    all_years = []
    if not france_indicator.empty:
        all_years.extend(france_indicator['Year'].tolist())
    if not eu27_indicator.empty:
        all_years.extend(eu27_indicator['Year'].tolist())
    
    if all_years:
        earliest_year = min(all_years)
        latest_year = max(all_years)
    else:
        return
    
    # Get display name
    description = get_display_name(indicator)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{description}<br><sub>France vs EU-27 Over Time</sub>",
            font=dict(size=16, color="#2C3E50", family="Arial, sans-serif"),
            x=0.5
        ),
        xaxis=dict(
            title="Year",
            title_font=dict(size=12, family="Arial, sans-serif"),
            tickfont=dict(size=11, family="Arial, sans-serif"),
            showgrid=True,
            gridcolor='lightgray',
            tick0=earliest_year,
            dtick=2 if len(all_years) > 10 else 1,
            range=[earliest_year - 0.5, latest_year + 0.5]
        ),
        yaxis=dict(
            title="Value",
            title_font=dict(size=12, family="Arial, sans-serif"),
            tickfont=dict(size=11, family="Arial, sans-serif"),
            showgrid=True,
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1),
        margin=dict(t=70, b=50, l=50, r=30),
        width=1000,
        height=600
    )
    
    # Save PNG
    png_path = os.path.join(output_dir, f"level5_overall_{indicator}.png")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 7))
        
        if not france_indicator.empty:
            plt.plot(france_indicator['Year'], france_indicator['Value'], 
                    color=FRANCE_COLOR, linewidth=4, marker='o', markersize=10, label='France', alpha=0.9)
        if not eu27_indicator.empty:
            plt.plot(eu27_indicator['Year'], eu27_indicator['Value'], 
                    color=EU_27_COLOR, linewidth=4, marker='s', markersize=10, label='EU-27', alpha=0.9)
        
        plt.title(f"{description}\nFrance vs EU-27 Over Time", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Value', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"  [OK] Saved: {os.path.basename(png_path)}")
    except Exception as e:
        print(f"  [ERROR] Could not save overall plot: {str(e)[:80]}...")

def create_level5_deciles_plot(france_decile_data, eu27_decile_data, indicator, output_dir):
    """Create decile plot for Level 5 indicator - France bars with EU-27 line overlay"""
    
    # Get France decile data for latest year
    if france_decile_data.empty:
        return
    
    france_deciles = france_decile_data[france_decile_data['Indicator'] == indicator].copy()
    eu27_deciles = eu27_decile_data[eu27_decile_data['Indicator'] == indicator].copy()
    
    if france_deciles.empty:
        return
    
    # Get latest year
    latest_year = france_deciles['Year'].max()
    france_latest = france_deciles[france_deciles['Year'] == latest_year].copy()
    
    # Convert decile to numeric and sort
    france_latest['Decile_num'] = pd.to_numeric(france_latest['Decile'], errors='coerce')
    france_latest = france_latest.dropna(subset=['Decile_num'])
    france_latest = france_latest.sort_values('Decile_num')
    
    if france_latest.empty:
        return
    
    # Get EU-27 data for the same year (if available)
    if not eu27_deciles.empty:
        eu27_latest = eu27_deciles[eu27_deciles['Year'] == latest_year].copy()
        if eu27_latest.empty:
            # Try latest available EU-27 year
            eu27_latest_year = eu27_deciles['Year'].max()
            eu27_latest = eu27_deciles[eu27_deciles['Year'] == eu27_latest_year].copy()
        
        eu27_latest['Decile_num'] = pd.to_numeric(eu27_latest['Decile'], errors='coerce')
        eu27_latest = eu27_latest.dropna(subset=['Decile_num'])
        eu27_latest = eu27_latest.sort_values('Decile_num')
    else:
        eu27_latest = pd.DataFrame()
    
    # Get display name (complete indicator name) - use get_display_name which handles spacing
    description = get_display_name(indicator)
    
    # Create the plot using matplotlib
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get all deciles
        all_deciles = sorted(france_latest['Decile_num'].unique())
        decile_labels = [str(int(d)) for d in all_deciles]
        x_positions = np.arange(len(decile_labels))
        
        # Plot France bars
        france_values = france_latest.set_index('Decile_num').loc[all_deciles, 'Value'].tolist()
        ax.bar(x_positions, france_values, width=0.6, color=FRANCE_COLOR, alpha=0.85, 
               label='France', edgecolor='black', linewidth=1.2)
        
        # Add value labels on France bars
        for i, v in enumerate(france_values):
            if not np.isnan(v):
                ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Overlay EU-27 line (if available with all deciles)
        if not eu27_latest.empty and len(eu27_latest) >= len(all_deciles):
            eu27_values = eu27_latest.set_index('Decile_num').loc[all_deciles, 'Value'].tolist()
            ax.plot(x_positions, eu27_values, color=EU_27_COLOR, linewidth=3, marker='o', 
                   markersize=8, label='EU-27', zorder=5)
            
            # Add value labels on EU-27 line (FURTHER ABOVE the line)
            for i, v in enumerate(eu27_values):
                if not np.isnan(v):
                    ax.text(i, v + 0.10, f'{v:.2f}', ha='center', va='bottom', fontsize=8, color=EU_27_COLOR, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Income Decile', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f"{description}\nFrance vs EU-27 by Income Decile ({int(latest_year)})", 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(decile_labels)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=11, loc='best')
        ax.set_facecolor('white')
        
        plt.tight_layout()
        png_path = os.path.join(output_dir, f"level5_deciles_{indicator}.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"  [OK] Saved: {os.path.basename(png_path)}")
    except Exception as e:
        print(f"  [ERROR] Could not save deciles plot: {str(e)[:80]}...")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(decile_labels)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=11, loc='best')
        ax.set_facecolor('white')
        
        plt.tight_layout()
        png_path = os.path.join(output_dir, f"level5_deciles_{indicator}.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"  [OK] Saved: {os.path.basename(png_path)}")
    except Exception as e:
        print(f"  [ERROR] Could not save deciles plot: {str(e)[:80]}...")

def create_time_series_plot(france_data, eu27_data, france_decile_1, france_decile_10, 
                          eu27_decile_1, eu27_decile_10, indicator, output_dir):
    """DEPRECATED: Create a time series plot comparing France and EU-27 for a specific indicator with deciles"""
    
    # Filter data for the specific indicator
    france_indicator = france_data[france_data['Indicator'] == indicator].copy()
    eu27_indicator = eu27_data[eu27_data['Indicator'] == indicator].copy()
    
    # Filter decile data (handle empty dataframes)
    france_d1 = france_decile_1[france_decile_1['Indicator'] == indicator].copy() if not france_decile_1.empty and 'Indicator' in france_decile_1.columns else pd.DataFrame()
    france_d10 = france_decile_10[france_decile_10['Indicator'] == indicator].copy() if not france_decile_10.empty and 'Indicator' in france_decile_10.columns else pd.DataFrame()
    eu27_d1 = eu27_decile_1[eu27_decile_1['Indicator'] == indicator].copy() if not eu27_decile_1.empty and 'Indicator' in eu27_decile_1.columns else pd.DataFrame()
    eu27_d10 = eu27_decile_10[eu27_decile_10['Indicator'] == indicator].copy() if not eu27_decile_10.empty and 'Indicator' in eu27_decile_10.columns else pd.DataFrame()
    
    if france_indicator.empty and eu27_indicator.empty:
        print(f"Warning: No data available for indicator {indicator}")
        return  # Skip if neither country has data
    
    if france_indicator.empty:
        print(f"Note: No France data for {indicator}, showing EU-27 only")
    elif eu27_indicator.empty:
        print(f"Note: No EU-27 data for {indicator}, showing France only")
    
    # Find the time range based on your requirements:
    # - Minimum year: earliest from either EU-27 or France  
    # - Maximum year: latest from France data (if available), otherwise latest from EU data
    all_years = []
    france_years = []
    
    if not france_indicator.empty:
        france_years.extend(france_indicator['Year'].tolist())
        all_years.extend(france_indicator['Year'].tolist())
    if not eu27_indicator.empty:
        all_years.extend(eu27_indicator['Year'].tolist())
    if not france_d1.empty:
        france_years.extend(france_d1['Year'].tolist())
        all_years.extend(france_d1['Year'].tolist())
    if not france_d10.empty:
        france_years.extend(france_d10['Year'].tolist())
        all_years.extend(france_d10['Year'].tolist())
    if not eu27_d1.empty:
        all_years.extend(eu27_d1['Year'].tolist())
    if not eu27_d10.empty:
        all_years.extend(eu27_d10['Year'].tolist())
    
    if not all_years:
        print(f"Warning: No data found for indicator {indicator}")
        return
    
    earliest_year = min(all_years)  # Earliest from either EU-27 or France
    latest_year = max(france_years) if france_years else max(all_years)  # Latest from France, or all data if no France data
    
    # Filter all data to not exceed France's latest year
    france_indicator = france_indicator[france_indicator['Year'] <= latest_year] if not france_indicator.empty else france_indicator
    eu27_indicator = eu27_indicator[eu27_indicator['Year'] <= latest_year] if not eu27_indicator.empty else eu27_indicator
    france_d1 = france_d1[france_d1['Year'] <= latest_year] if not france_d1.empty else france_d1
    france_d10 = france_d10[france_d10['Year'] <= latest_year] if not france_d10.empty else france_d10
    eu27_d1 = eu27_d1[eu27_d1['Year'] <= latest_year] if not eu27_d1.empty else eu27_d1
    eu27_d10 = eu27_d10[eu27_d10['Year'] <= latest_year] if not eu27_d10.empty else eu27_d10
    
    # Sort by year
    france_indicator = france_indicator.sort_values('Year')
    eu27_indicator = eu27_indicator.sort_values('Year')
    if not france_d1.empty:
        france_d1 = france_d1.sort_values('Year')
    if not france_d10.empty:
        france_d10 = france_d10.sort_values('Year')
    if not eu27_d1.empty:
        eu27_d1 = eu27_d1.sort_values('Year')
    if not eu27_d10.empty:
        eu27_d10 = eu27_d10.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add France lines first (median, then deciles) - only if data exists
    # 1. France median line (solid yellow)
    if not france_indicator.empty:
        fig.add_trace(go.Scatter(
            x=france_indicator['Year'],
            y=france_indicator['Value'],
            mode='lines+markers',
            name='France Average',
            line=dict(color=FRANCE_COLOR, width=3),
            marker=dict(color=FRANCE_COLOR, size=8, symbol='circle'),
            hovertemplate='<b>France Average</b><br>' +
                         'Year: %{x}<br>' +
                         'Value: %{y:.3f}<extra></extra>'
        ))
    
    # 2. France 1st decile line (dashed yellow)
    if not france_d1.empty:
        fig.add_trace(go.Scatter(
            x=france_d1['Year'],
            y=france_d1['Value'],
            mode='lines+markers',
            name='France 1st Decile',
            line=dict(color=FRANCE_COLOR, width=2, dash='dash'),
            marker=dict(color=FRANCE_COLOR, size=6, symbol='triangle-up'),
            hovertemplate='<b>France 1st Decile</b><br>' +
                         'Year: %{x}<br>' +
                         'Value: %{y:.3f}<extra></extra>'
        ))
    
    # 3. France 10th decile line (dashdot yellow)
    if not france_d10.empty:
        fig.add_trace(go.Scatter(
            x=france_d10['Year'],
            y=france_d10['Value'],
            mode='lines+markers',
            name='France 10th Decile',
            line=dict(color=FRANCE_COLOR, width=2, dash='dashdot'),
            marker=dict(color=FRANCE_COLOR, size=6, symbol='triangle-down'),
            hovertemplate='<b>France 10th Decile</b><br>' +
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
    if not france_indicator.empty:
        unique_years.update(france_indicator['Year'].tolist())
    if not eu27_indicator.empty:
        unique_years.update(eu27_indicator['Year'].tolist())
    if not france_d1.empty:
        unique_years.update(france_d1['Year'].tolist())
    if not france_d10.empty:
        unique_years.update(france_d10['Year'].tolist())
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
                text=f"{description}<br><sub>France vs EU-27 Time Series ({int(earliest_year)}-{int(latest_year)})</sub>",
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
                text=f"{description}<br><sub>France vs EU-27 Time Series ({int(earliest_year)}-{int(latest_year)})</sub>",
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
    
    # Create safe filename - remove problematic characters and limit length
    safe_indicator = indicator.replace('-', '_').replace('/', '_').replace(' ', '_').replace(',', '')
    # Truncate if too long to avoid Windows path length issues  
    if len(safe_indicator) > 40:
        safe_indicator = safe_indicator[:40]
    
    # Save as PNG only
    
    # Save as PNG using matplotlib for reliability
    png_path = os.path.join(output_dir, f"France_vs_eu27_{safe_indicator}.png")
    try:
        # Use matplotlib for PNG generation (more reliable than kaleido)
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        
        # Create matplotlib figure
        plt.figure(figsize=(12, 7))
        
        # Plot in the same order as Plotly for consistent legend
        # 1. France median line (solid yellow)
        plt.plot(france_indicator['Year'], france_indicator['Value'], 
                color=FRANCE_COLOR, linewidth=3, marker='o', 
                markersize=8, label='France Average', alpha=0.9)
        
        # 2. France decile lines (different dash patterns and markers)
        if not france_d1.empty:
            plt.plot(france_d1['Year'], france_d1['Value'], 
                    color=FRANCE_COLOR, linewidth=2, linestyle='--', marker='^', markersize=6,
                    label='France 1st Decile', alpha=0.7)
        
        if not france_d10.empty:
            plt.plot(france_d10['Year'], france_d10['Value'],
                    color=FRANCE_COLOR, linewidth=2, linestyle='-.', marker='v', markersize=6,
                    label='France 10th Decile', alpha=0.7)
        
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
        plt.title(f"{description}\nFrance vs EU-27 Time Series ({int(earliest_year)}-{int(latest_year)})", 
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
        print("  Only PNG version saved.")

def create_summary_stats(france_data, eu27_data, all_indicators, output_dir):
    """Create summary statistics for the indicators"""
    
    # Get all unique indicators
    france_indicators = set(france_data['Indicator'].unique()) if not france_data.empty else set()
    eu27_indicators = set(eu27_data['Indicator'].unique()) if not eu27_data.empty else set()
    
    summary_stats = []
    
    for indicator in sorted(all_indicators):
        france_indicator = france_data[france_data['Indicator'] == indicator] if not france_data.empty else pd.DataFrame()
        eu27_indicator = eu27_data[eu27_data['Indicator'] == indicator] if not eu27_data.empty else pd.DataFrame()
        
        # Get basic info
        description = get_display_name(indicator)
        
        # France stats
        france_available = not france_indicator.empty
        france_years = sorted(france_indicator['Year'].unique()) if france_available else []
        france_year_range = f"{min(france_years)}-{max(france_years)}" if france_years else "No data"
        
        # EU27 stats  
        eu27_available = not eu27_indicator.empty
        eu27_years = sorted(eu27_indicator['Year'].unique()) if eu27_available else []
        eu27_year_range = f"{min(eu27_years)}-{max(eu27_years)}" if eu27_years else "No data"
        
        # Overlap
        if france_years and eu27_years:
            overlap_years = sorted(set(france_years).intersection(set(eu27_years)))
            overlap_range = f"{min(overlap_years)}-{max(overlap_years)}" if overlap_years else "No overlap"
            overlap_count = len(overlap_years)
        else:
            overlap_range = "No overlap"
            overlap_count = 0
        
        if not france_indicator.empty and not eu27_indicator.empty:
            france_mean = france_indicator['Value'].mean()
            france_latest = france_indicator.sort_values('Year')['Value'].iloc[-1]
            france_years = f"{france_indicator['Year'].min():.0f}-{france_indicator['Year'].max():.0f}"
            
            eu27_mean = eu27_indicator['Value'].mean()
            eu27_latest = eu27_indicator.sort_values('Year')['Value'].iloc[-1]
            eu27_years = f"{eu27_indicator['Year'].min():.0f}-{eu27_indicator['Year'].max():.0f}"
            
            summary_stats.append({
                'Indicator': indicator,
                'Description': get_display_name(indicator),
                'france_Mean': france_mean,
                'france_Latest': france_latest,
                'france_Years': france_years,
                'EU27_Mean': eu27_mean,
                'EU27_Latest': eu27_latest,
                'EU27_Years': eu27_years,
                'Difference_Latest': france_latest - eu27_latest
            })
    
    # Create DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, "France_vs_eu27_summary_stats.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics: {summary_path}")
    
    return summary_df

def export_all_data_to_excel(france_data, eu27_data, france_decile_1, france_decile_10, 
                           eu27_decile_1, eu27_decile_10, all_indicators, output_dir):
    """Export all displayed data to Excel with the requested column structure"""
    
    all_data_records = []
    
    # Process each indicator
    for indicator in sorted(all_indicators):
        # Get indicator description
        description = get_display_name(indicator)
        
        # Filter data for this indicator
        france_median = france_data[france_data['Indicator'] == indicator].copy()
        eu27_median = eu27_data[eu27_data['Indicator'] == indicator].copy()
        
        france_d1 = france_decile_1[france_decile_1['Indicator'] == indicator].copy() if not france_decile_1.empty and 'Indicator' in france_decile_1.columns else pd.DataFrame()
        france_d10 = france_decile_10[france_decile_10['Indicator'] == indicator].copy() if not france_decile_10.empty and 'Indicator' in france_decile_10.columns else pd.DataFrame()
        eu27_d1 = eu27_decile_1[eu27_decile_1['Indicator'] == indicator].copy() if not eu27_decile_1.empty and 'Indicator' in eu27_decile_1.columns else pd.DataFrame()
        eu27_d10 = eu27_decile_10[eu27_decile_10['Indicator'] == indicator].copy() if not eu27_decile_10.empty and 'Indicator' in eu27_decile_10.columns else pd.DataFrame()
        
        # Determine time range (same logic as in plotting function)
        all_years = []
        france_years = []
        
        if not france_median.empty:
            france_years.extend(france_median['Year'].tolist())
            all_years.extend(france_median['Year'].tolist())
        if not eu27_median.empty:
            all_years.extend(eu27_median['Year'].tolist())
        if not france_d1.empty:
            france_years.extend(france_d1['Year'].tolist())
            all_years.extend(france_d1['Year'].tolist())
        if not france_d10.empty:
            france_years.extend(france_d10['Year'].tolist())
            all_years.extend(france_d10['Year'].tolist())
        if not eu27_d1.empty:
            all_years.extend(eu27_d1['Year'].tolist())
        if not eu27_d10.empty:
            all_years.extend(eu27_d10['Year'].tolist())
        
        if not all_years:
            continue
            
        earliest_year = min(all_years)
        latest_year = max(france_years) if france_years else max(all_years)
        
        # Filter all datasets to the same time range used in graphs
        france_median = france_median[france_median['Year'] <= latest_year] if not france_median.empty else france_median
        eu27_median = eu27_median[eu27_median['Year'] <= latest_year] if not eu27_median.empty else eu27_median
        france_d1 = france_d1[france_d1['Year'] <= latest_year] if not france_d1.empty else france_d1
        france_d10 = france_d10[france_d10['Year'] <= latest_year] if not france_d10.empty else france_d10
        eu27_d1 = eu27_d1[eu27_d1['Year'] <= latest_year] if not eu27_d1.empty else eu27_d1
        eu27_d10 = eu27_d10[eu27_d10['Year'] <= latest_year] if not eu27_d10.empty else eu27_d10
        
        # Add France median data
        for _, row in france_median.iterrows():
            all_data_records.append({
                'indicator_code': indicator,
                'indicator_name': description,
                'year': int(row['Year']),
                'geo': 'FR',
                'decile': 'Average',
                'value': row['Value']
            })
        
        # Add France 1st decile data
        for _, row in france_d1.iterrows():
            all_data_records.append({
                'indicator_code': indicator,
                'indicator_name': description,
                'year': int(row['Year']),
                'geo': 'FR',
                'decile': '1st',
                'value': row['Value']
            })
        
        # Add France 10th decile data
        for _, row in france_d10.iterrows():
            all_data_records.append({
                'indicator_code': indicator,
                'indicator_name': description,
                'year': int(row['Year']),
                'geo': 'FR',
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
    excel_path = os.path.join(output_dir, "France_vs_eu27_primary_indicators.xlsx")
    
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
        csv_path = os.path.join(output_dir, "France_vs_eu27_primary_indicators.csv")
        export_df.to_csv(csv_path, index=False)
        print(f"Exported to CSV instead: {csv_path}")

def export_ewbi_level1_to_excel(unified_df, output_dir):
    """Export EWBI Level 1 data (France + EU-27) to Excel"""
    
    all_data_records = []
    
    # Get EWBI Level 1 data for both France and EU-27 - match app.py aggregation logic
    ewbi_overall = unified_df[
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] == 'All') &
        (
            (unified_df['Country'] == 'FR') |
            ((unified_df['Country'] == 'All Countries') & 
             (unified_df['Aggregation'] == 'Population-weighted geometric mean'))
        )
    ].copy()
    
    # Get EWBI Level 1 decile data for both France and EU-27 - match app.py aggregation logic
    ewbi_deciles = unified_df[
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna()) &
        (
            (unified_df['Country'] == 'FR') |
            ((unified_df['Country'] == 'All Countries') & 
             (unified_df['Aggregation'] == 'Population-weighted geometric mean'))
        )
    ].copy()
    
    # Process overall EWBI data
    for _, row in ewbi_overall.iterrows():
        country_name = 'France' if row['Country'] == 'FR' else 'EU-27'
        all_data_records.append({
            'indicator_code': 'EWBI',
            'indicator_name': 'European Well-Being Index',
            'year': int(row['Year']),
            'geo': country_name,
            'decile': 'Average',
            'value': row['Value']
        })
    
    # Process decile EWBI data
    for _, row in ewbi_deciles.iterrows():
        country_name = 'France' if row['Country'] == 'FR' else 'EU-27'
        decile_num = int(float(row['Decile']))  # Handle float to int conversion
        decile_name = f"{decile_num}{'st' if decile_num == 1 else ('nd' if decile_num == 2 else ('rd' if decile_num == 3 else 'th'))}"
        all_data_records.append({
            'indicator_code': 'EWBI',
            'indicator_name': 'European Well-Being Index',
            'year': int(row['Year']),
            'geo': country_name,
            'decile': decile_name,
            'value': row['Value']
        })
    
    # Create DataFrame
    export_df = pd.DataFrame(all_data_records)
    
    if export_df.empty:
        print("Warning: No EWBI Level 1 data to export")
        return
    
    # Sort by geo, decile, year
    export_df = export_df.sort_values(['geo', 'decile', 'year'])
    
    # Export to Excel
    excel_path = os.path.join(output_dir, "France_vs_eu27_ewbi_level1.xlsx")
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main data sheet
            export_df.to_excel(writer, sheet_name='EWBI_Level1_Data', index=False)
            
        print(f"Exported EWBI Level 1 data to Excel: {excel_path}")
        print(f"Total EWBI records: {len(export_df)}")
        print(f"Countries: {export_df['geo'].nunique()}")
        print(f"Years: {export_df['year'].min()}-{export_df['year'].max()}")
        
    except Exception as e:
        print(f"Error exporting EWBI Level 1 to Excel: {e}")

def export_eu_priorities_to_excel(unified_df, output_dir):
    """Export EU Priorities Level 2 data (France + EU-27) to Excel"""
    
    # EU priorities from the app
    EU_PRIORITIES = [
        'Energy and Housing', 
        'Equality',
        'Health and Animal Welfare',
        'Intergenerational Fairness, Youth, Culture and Sport',
        'Social Rights and Skills, Quality Jobs and Preparedness'
    ]
    
    all_data_records = []
    
    # Get EU Priorities Level 2 data for France
    priorities_overall_ch = unified_df[
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] == 'All') &
        (unified_df['Country'] == 'FR') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Get EU Priorities Level 2 data for EU-27 (with proper aggregation filtering)
    priorities_overall_eu27 = unified_df[
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] == 'All') &
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Aggregation'] == 'Population-weighted geometric mean') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Combine overall data
    priorities_overall = pd.concat([priorities_overall_ch, priorities_overall_eu27], ignore_index=True)
    
    # Get EU Priorities Level 2 decile data for France
    priorities_deciles_ch = unified_df[
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna()) &
        (unified_df['Country'] == 'FR') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Get EU Priorities Level 2 decile data for EU-27 (with proper aggregation filtering)
    priorities_deciles_eu27 = unified_df[
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna()) &
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Aggregation'] == 'Population-weighted geometric mean') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Combine decile data
    priorities_deciles = pd.concat([priorities_deciles_ch, priorities_deciles_eu27], ignore_index=True)
    
    # Process overall priorities data
    for _, row in priorities_overall.iterrows():
        country_name = 'France' if row['Country'] == 'FR' else 'EU-27'
        all_data_records.append({
            'indicator_code': row['EU priority'].replace(' ', '_').replace(',', '').replace('&', 'and'),
            'indicator_name': row['EU priority'],
            'year': int(row['Year']),
            'geo': country_name,
            'decile': 'Average',
            'value': row['Value']
        })
    
    # Process decile priorities data
    for _, row in priorities_deciles.iterrows():
        country_name = 'France' if row['Country'] == 'FR' else 'EU-27'
        decile_num = int(float(row['Decile']))  # Handle float to int conversion
        decile_name = f"{decile_num}{'st' if decile_num == 1 else ('nd' if decile_num == 2 else ('rd' if decile_num == 3 else 'th'))}"
        all_data_records.append({
            'indicator_code': row['EU priority'].replace(' ', '_').replace(',', '').replace('&', 'and'),
            'indicator_name': row['EU priority'],
            'year': int(row['Year']),
            'geo': country_name,
            'decile': decile_name,
            'value': row['Value']
        })
    
    # Create DataFrame
    export_df = pd.DataFrame(all_data_records)
    
    if export_df.empty:
        print("Warning: No EU Priorities Level 2 data to export")
        return
    
    # Sort by indicator, geo, decile, year
    export_df = export_df.sort_values(['indicator_code', 'geo', 'decile', 'year'])
    
    # Export to Excel
    excel_path = os.path.join(output_dir, "France_vs_eu27_eu_priorities_level2.xlsx")
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main data sheet
            export_df.to_excel(writer, sheet_name='EU_Priorities_Level2_Data', index=False)
            
            # Create a summary sheet with priority descriptions
            priority_summary = []
            for priority in EU_PRIORITIES:
                priority_summary.append({
                    'priority_code': priority.replace(' ', '_').replace(',', '').replace('&', 'and'),
                    'priority_name': priority
                })
            
            summary_df = pd.DataFrame(priority_summary)
            summary_df.to_excel(writer, sheet_name='Priority_Definitions', index=False)
            
        print(f"Exported EU Priorities Level 2 data to Excel: {excel_path}")
        print(f"Total EU Priorities records: {len(export_df)}")
        print(f"Priorities: {export_df['indicator_name'].nunique()}")
        print(f"Countries: {export_df['geo'].nunique()}")
        print(f"Years: {export_df['year'].min()}-{export_df['year'].max()}")
        
    except Exception as e:
        print(f"Error exporting EU Priorities Level 2 to Excel: {e}")

def main():
    """Main function to generate time series graphs for app-displayed Level 5 indicators plus France EWBI and EU priorities"""
    
    print("Starting France vs EU-27 App-Displayed Level 5 Indicators Time Series Analysis")
    print("Plus France EWBI and EU Priorities Analysis")
    print("=" * 80)
    
    # Load data first
    try:
        unified_df = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Get output directory - use the new report structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.abspath(os.path.join(current_dir, '..'))
    output_dir = os.path.abspath(os.path.join(report_dir, 'outputs', 'graphs'))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n Output directory: {output_dir}")
    # Create EWBI-specific subfolders under outputs/graphs
    ewbi_base = os.path.join(output_dir, 'EWBI')
    level1_dir = os.path.join(ewbi_base, 'Level_1_EWBI')
    level2_dir = os.path.join(ewbi_base, 'Level_2_EU_Priorities')
    level3_dir = os.path.join(ewbi_base, 'Level_3_Primary_Indicators')

    os.makedirs(level1_dir, exist_ok=True)
    os.makedirs(level2_dir, exist_ok=True)
    os.makedirs(level3_dir, exist_ok=True)

    # Create EWBI tables folder structure
    tables_dir = os.path.abspath(os.path.join(report_dir, 'outputs', 'tables'))
    tables_ewbi_dir = os.path.join(tables_dir, 'EWBI')
    os.makedirs(tables_ewbi_dir, exist_ok=True)

    OUTPUT_DIRS = {
        'level1': level1_dir,
        'level2': level2_dir,
        'level3': level3_dir,
        'generic': output_dir,
        'tables': tables_ewbi_dir
    }
    print(f"Created EWBI subfolders under: {ewbi_base}")
    print(f"Created EWBI tables folder: {tables_ewbi_dir}")
    
    # PART 1: Generate France EWBI and EU priorities graphs
    print("\n" + "="*50)
    print("PART 1: France EWBI AND EU PRIORITIES ANALYSIS")
    print("="*50)
    
    # 1.1 France vs EU-27 EWBI (Level 1) - Overall and Deciles
    print("\n1.1. Processing EWBI (Level 1) - France vs EU-27...")
    france_ewbi_overall, france_ewbi_deciles = prepare_france_ewbi_data(unified_df)
    eu27_ewbi_overall, eu27_ewbi_deciles = prepare_eu27_ewbi_data(unified_df)
    
    if not france_ewbi_overall.empty or not eu27_ewbi_overall.empty:
        print("Creating EWBI overall comparison graph...")
        create_ewbi_overall_comparison_plot(france_ewbi_overall, eu27_ewbi_overall, OUTPUT_DIRS['level1'])
    else:
        print("Warning: No EWBI overall data found for France or EU-27")
    
    if not france_ewbi_deciles.empty or not eu27_ewbi_deciles.empty:
        print("Creating EWBI deciles comparison graph...")
        create_ewbi_deciles_comparison_plot(france_ewbi_deciles, eu27_ewbi_deciles, OUTPUT_DIRS['level1'])
    else:
        print("Warning: No EWBI deciles data found for France or EU-27")
    
    # 1.2 France vs EU-27 EU Priorities (Level 2) - Overall and Deciles
    print("\n1.2. Processing EU Priorities (Level 2) - France vs EU-27...")
    france_priorities_overall, france_priorities_deciles, eu_priorities = prepare_france_eu_priorities_data(unified_df)
    eu27_priorities_overall, eu27_priorities_deciles, _ = prepare_eu27_eu_priorities_data(unified_df)
    
    if (not france_priorities_overall.empty or not france_priorities_deciles.empty or 
        not eu27_priorities_overall.empty or not eu27_priorities_deciles.empty):
        print(f"Creating comparison graphs for {len(eu_priorities)} EU priorities...")
        create_eu_priority_comparison_plots(france_priorities_overall, france_priorities_deciles, 
                                          eu27_priorities_overall, eu27_priorities_deciles,
                                          eu_priorities, OUTPUT_DIRS['level2'])
    else:
        print("Warning: No EU priorities data found for France or EU-27")
    
    # PART 1.3: Export EWBI Level 1 and EU Priorities Level 2 data to Excel
    print("\n1.3. Exporting EWBI and EU Priorities data to Excel...")
    
    # Export EWBI Level 1 data
    print("Exporting EWBI Level 1 data...")
    try:
        export_ewbi_level1_to_excel(unified_df, OUTPUT_DIRS['tables'])
    except Exception as e:
        print(f"Error exporting EWBI Level 1: {e}")
    
    # Export EU Priorities Level 2 data
    print("Exporting EU Priorities Level 2 data...")
    try:
        export_eu_priorities_to_excel(unified_df, OUTPUT_DIRS['tables'])
    except Exception as e:
        print(f"Error exporting EU Priorities Level 2: {e}")
    
    # PART 2: Original Level 5 indicators analysis
    print("\n" + "="*50)
    print("PART 2: LEVEL 5 INDICATORS ANALYSIS (ORIGINAL)")
    print("="*50)
    
    # Get Level 5 indicators that are displayed in the app
    print("\n2.1. Getting Level 5 indicators displayed in the app...")
    app_indicators = get_app_level5_indicators()
    
    if not app_indicators:
        print("Error: No Level 5 indicators found in the app")
        # Continue with Part 1 results only
        print("\n Part 1 (EWBI and EU priorities) completed successfully!")
        return
    
    print(f"Found {len(app_indicators)} Level 5 indicators displayed in the app:")
    for indicator in app_indicators:
        description = get_display_name(indicator)
        print(f"   • {description}")
    
    # Filter data to only include app-displayed Level 5 indicators
    print(f"\n2.2. Filtering data for app-displayed indicators...")
    level5_df = unified_df[unified_df['Primary and raw data'].isin(app_indicators)].copy()
    print(f"Filtered data shape: {level5_df.shape}")
    
    # Prepare data
    print("\n2.3. Preparing France data...")
    france_data, france_deciles_all, france_deciles_all_compat = prepare_france_data(level5_df)
    france_decile_1 = france_deciles_all[france_deciles_all['Decile'] == '1.0'] if not france_deciles_all.empty else pd.DataFrame()
    france_decile_10 = france_deciles_all[france_deciles_all['Decile'] == '10.0'] if not france_deciles_all.empty else pd.DataFrame()
    
    print("\n2.4. Preparing EU-27 data...")
    eu27_data, eu27_deciles_all, eu27_deciles_all_compat = prepare_eu27_data(level5_df)
    eu27_decile_1 = eu27_deciles_all[eu27_deciles_all['Decile'] == '1.0'] if not eu27_deciles_all.empty else pd.DataFrame()
    eu27_decile_10 = eu27_deciles_all[eu27_deciles_all['Decile'] == '10.0'] if not eu27_deciles_all.empty else pd.DataFrame()
    
    if france_data.empty and eu27_data.empty:
        print("Warning: Could not load required Level 5 data")
        print("Continuing with Part 1 results only...")
        print("\n Part 1 (EWBI and EU priorities) completed successfully!")
        return
    
    # Process ALL app indicators, even if France or EU has no data
    france_indicators = set(france_data['Indicator'].unique()) if not france_data.empty else set()
    eu27_indicators = set(eu27_data['Indicator'].unique()) if not eu27_data.empty else set()
    
    # Use ALL app indicators, not just common ones
    all_indicators = set(app_indicators)
    
    print(f"\n2.5. Processing ALL {len(all_indicators)} app-displayed indicators:")
    print(f"   • France has data for: {len(france_indicators)} indicators")
    print(f"   • EU-27 has data for: {len(eu27_indicators)} indicators")
    print(f"   • Common indicators: {len(france_indicators.intersection(eu27_indicators))}")
    print(f"   • Will generate graphs for ALL {len(all_indicators)} app indicators")
    
    for indicator in sorted(all_indicators):
        description = get_display_name(indicator)
        has_France = indicator in france_indicators
        has_eu = indicator in eu27_indicators
        status = []
        if has_France:
            status.append("FR")
        if has_eu:
            status.append("EU")
        status_str = "+".join(status) if status else "No data"
        print(f"   • {description} ({status_str})")
    
    print(f"\n2.6. Generating time series graphs...")
    
    # Generate graphs for each indicator (2 per indicator: overall line + deciles bar chart)
    for i, indicator in enumerate(sorted(all_indicators), 1):
        description = get_display_name(indicator)
        print(f"\n2.6.{i}. Processing {description}...")
        
        # Check data availability for this indicator
        has_france_data = indicator in france_indicators
        has_eu_data = indicator in eu27_indicators
        
        if not has_france_data and not has_eu_data:
            print(f"   Warning: No data available for {indicator}. Skipping...")
            continue
        
        try:
            # Create overall time series plot (France and EU-27 averages only)
            create_level5_overall_plot(france_data, eu27_data, indicator, OUTPUT_DIRS['level3'])
            
            # Create deciles plot (France bars vs EU-27 bars for all deciles)
            # Use ALL decile data for both countries
            create_level5_deciles_plot(france_deciles_all, eu27_deciles_all, indicator, OUTPUT_DIRS['level3'])
        except Exception as e:
            print(f"   [ERROR] Error creating plots for {indicator}: {str(e)[:80]}")
    
    # Create summary statistics
    print(f"\n2.7. Creating summary statistics...")
    try:
        summary_df = create_summary_stats(france_data, eu27_data, all_indicators, OUTPUT_DIRS['tables'])
        print(f"Summary covers {len(summary_df)} indicators")
    except Exception as e:
        print(f"Error creating summary statistics: {e}")
    
    # Export all data to Excel
    print(f"\n2.8. Exporting all data to Excel...")
    try:
        export_all_data_to_excel(france_data, eu27_data, france_decile_1, france_decile_10, 
                                eu27_decile_1, eu27_decile_10, all_indicators, OUTPUT_DIRS['tables'])
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
    
    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print(" ALL ANALYSES COMPLETED SUCCESSFULLY!")
    print(f" Output directory: {output_dir}")
    print("\nGenerated files:")
    
    # List generated files
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.csv', '.xlsx'))]
        
        # Separate files by category
        ewbi_files = [f for f in files if 'ewbi' in f.lower()]
        priority_files = [f for f in files if any(p.lower().replace(' ', '_').replace(',', '').replace('&', 'and') in f.lower() 
                                                 for p in eu_priorities if 'ewbi' not in f.lower())]
        level5_files = [f for f in files if f.startswith('France_vs_eu27_')]
        other_files = [f for f in files if f not in ewbi_files and f not in priority_files and f not in level5_files]
        
        if ewbi_files:
            print(f"\n EWBI Files ({len(ewbi_files)}):")
            for file in sorted(ewbi_files):
                print(f"  • {file}")
        
        if priority_files:
            print(f"\n EU Priority Files ({len(priority_files)}):")
            for file in sorted(priority_files):
                print(f"  • {file}")
        
        if level5_files:
            print(f"\n Level 5 Indicator Files ({len(level5_files)}):")
            for file in sorted(level5_files):
                print(f"  • {file}")
        
        if other_files:
            print(f"\n Other Files ({len(other_files)}):")
            for file in sorted(other_files):
                print(f"  • {file}")
    
    print(f"\n Analysis Summary:")
    print(f"  PART 1 - France Analysis:")
    print(f"    • EWBI (Level 1): Overall + Deciles temporal graphs")
    print(f"    • EU Priorities (Level 2): {len(eu_priorities)} priorities × (Overall + Deciles)")
    print(f"  PART 2 - Comparative Analysis:")
    print(f"    • {len(all_indicators)} Level 5 indicators: France vs EU-27")
    print(f"    • Time ranges: Dynamic (earliest EU/France start to latest France end)")
    print(f"    • Colors: France ({FRANCE_COLOR}), EU-27 ({EU_27_COLOR})")
    print(f"    • Output formats: PNG (static images)")
    
    print(f"\n Notes:")
    print(f"  • France graphs use yellow color scheme as requested")
    print(f"  • Data source: unified PCA-weighted dataset from app.py")
    print(f"  • EWBI = European Well-Being Index (Level 1)")
    print(f"  • EU Priorities = Major policy areas (Level 2)")
    print(f"  • Level 5 = Raw statistical indicators")
    print(f"  • Decile decomposition shows income-based inequality")
    print(f"  • All graphs are saved as PNG files only")

if __name__ == "__main__":
    main()
