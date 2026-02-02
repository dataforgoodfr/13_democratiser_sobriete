"""
ewbi_treatment.py - Generate time series comparison graphs for Switzerland vs EU-27 focusing on Energy and Housing

This script creates time series graphs comparing Switzerland and EU-27 values for Energy and Housing indicators,
plus additional graphs for Switzerland alone showing:
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
reports_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
shared_code_dir = os.path.join(reports_dir, 'shared', 'code')
# Go up to Well-being directory (reports_dir/..)
well_being_code_dir = os.path.abspath(os.path.join(reports_dir, '..', 'code'))

sys.path.insert(0, current_dir)
sys.path.insert(0, shared_code_dir)
sys.path.insert(0, well_being_code_dir)

# Import shared utilities (after path setup)
try:
    from ewbi_data_loader import load_ewbi_unified_data, get_housing_energy_indicators
    from visualization_utils import create_time_series_plot, save_plot
    from variable_mapping import get_display_name
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Shared code directory: {shared_code_dir}")
    print(f"Well-being code directory: {well_being_code_dir}")
    print(f"sys.path: {sys.path[:5]}")
    raise

# Color palette consistent with 6_graphs.py and dashboard
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'
SWITZERLAND_COLOR = '#ffd558'  # Use yellow for Switzerland as requested

def get_app_level3_indicators():
    """Get the Level 3 primary indicators that are actually available in the app's dropdown by replicating app.py logic"""
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

    # Get Level 3 primary indicators that would appear in app dropdowns
    all_available_indicators = set()

    for eu_priority in EU_PRIORITIES:
        primary_options = df[
            (df.get('EU priority') == eu_priority) &
            (df.get('Primary and raw data').notna()) &
            (df.get('Level') == 3)
        ]['Primary and raw data'].unique()

        print(f"EU Priority '{eu_priority}': {len(primary_options)} indicators")
        for indicator in primary_options:
            display_name = get_display_name(indicator)
            print(f"  • {display_name}")

        all_available_indicators.update(primary_options)

    print(f"\nTotal unique Level 3 primary indicators available in app dropdowns: {len(all_available_indicators)}")
    return sorted(all_available_indicators)

def load_data():
    """Load the primary indicators data for time series analysis using shared utilities"""
    # Use shared utility to load EWBI data
    unified_df = load_ewbi_unified_data()
    return unified_df

def prepare_swiss_ewbi_data(unified_df):
    """Extract Switzerland EWBI (Level 1) data for overall and decile analysis"""
    # Get Switzerland EWBI overall data (Level 1, Decile='All Deciles')
    swiss_ewbi_overall = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] == 'All Deciles')
    ].copy()
    
    # Get Switzerland EWBI by deciles (Level 1, all deciles except 'All Deciles')
    swiss_ewbi_deciles = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] != 'All Deciles') &
        (unified_df['Decile'].notna())
    ].copy()
    
    print(f"Swiss EWBI overall data: {len(swiss_ewbi_overall)} records")
    print(f"Swiss EWBI decile data: {len(swiss_ewbi_deciles)} records")
    
    return swiss_ewbi_overall, swiss_ewbi_deciles

def prepare_swiss_eu_priorities_data(unified_df):
    """Extract Switzerland EU priorities (Level 2) data for overall and decile analysis"""
    # Focus only on Energy and Housing EU priority
    EU_PRIORITIES = [
        'Energy and Housing'
    ]
    
    # Get Switzerland EU priorities overall data (Level 2, Decile='All Deciles')
    # For Switzerland: NO aggregation filtering (matches app.py logic)
    swiss_priorities_overall = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] == 'All Deciles') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Get Switzerland EU priorities by deciles (Level 2, all deciles except 'All Deciles')
    # For Switzerland: NO aggregation filtering (matches app.py logic)
    swiss_priorities_deciles = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] != 'All Deciles') &
        (unified_df['Decile'].notna()) &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    print(f"Swiss EU priorities overall data: {len(swiss_priorities_overall)} records")
    print(f"Swiss EU priorities decile data: {len(swiss_priorities_deciles)} records")
    print(f"Available EU priorities: {sorted(swiss_priorities_overall['EU priority'].unique())}")
    
    return swiss_priorities_overall, swiss_priorities_deciles, EU_PRIORITIES

def prepare_eu27_ewbi_data(unified_df):
    """Extract EU-27 EWBI (Level 1) data for overall and decile analysis - matches app.py logic"""
    # Get EU-27 EWBI overall data (Level 1, Decile='All Deciles') - use Population-weighted geometric mean
    eu27_ewbi_overall = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] == 'All Deciles') &
        (unified_df['Aggregation'] == 'Population-weighted geometric mean')
    ].copy()
    
    # Get EU-27 EWBI by deciles (Level 1, all deciles except 'All Deciles') - use Population-weighted geometric mean
    eu27_ewbi_deciles = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] != 'All Deciles') &
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
    
    # Get EU-27 EU priorities overall data (Level 2, Decile='All Deciles') 
    # For reference lines: Do NOT filter by aggregation (to match app.py logic)
    eu27_priorities_overall = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] == 'All Deciles')
    ].copy()
    
    # Get EU-27 EU priorities by deciles (Level 2, all deciles except 'All Deciles') - use Population-weighted geometric mean
    eu27_priorities_deciles = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] != 'All Deciles') &
        (unified_df['Decile'].notna()) &
        (unified_df['Aggregation'] == 'Population-weighted geometric mean')
    ].copy()
    
    print(f"EU-27 EU priorities overall data: {len(eu27_priorities_overall)} records")
    print(f"EU-27 EU priorities decile data: {len(eu27_priorities_deciles)} records")
    
    return eu27_priorities_overall, eu27_priorities_deciles, EU_PRIORITIES

def create_ewbi_overall_comparison_plot(swiss_ewbi_overall, eu27_ewbi_overall, output_dir):
    """Create EWBI overall temporal graph comparing Switzerland vs EU-27"""
    if swiss_ewbi_overall.empty and eu27_ewbi_overall.empty:
        print("Warning: No EWBI overall data available for Switzerland or EU-27")
        return
    
    # Sort by year
    if not swiss_ewbi_overall.empty:
        swiss_ewbi_overall = swiss_ewbi_overall.sort_values('Year')
    if not eu27_ewbi_overall.empty:
        eu27_ewbi_overall = eu27_ewbi_overall.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add Switzerland EWBI line (solid yellow) if data exists
    if not swiss_ewbi_overall.empty:
        fig.add_trace(go.Scatter(
            x=swiss_ewbi_overall['Year'],
            y=swiss_ewbi_overall['Value'],
            mode='lines+markers',
            name='Switzerland EWBI',
            line=dict(color=SWITZERLAND_COLOR, width=4),
            marker=dict(color=SWITZERLAND_COLOR, size=10, symbol='circle'),
            hovertemplate='<b>Switzerland EWBI</b><br>' +
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
    if not swiss_ewbi_overall.empty:
        all_years.extend(swiss_ewbi_overall['Year'].tolist())
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
            text=f"European Well-Being Index (EWBI)<br><sub>Switzerland vs EU-27 Comparison ({int(earliest_year)}-{int(latest_year)})</sub>",
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
    png_path = os.path.join(output_dir, "switzerland_vs_eu27_ewbi_overall.png")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        if not swiss_ewbi_overall.empty:
            plt.plot(swiss_ewbi_overall['Year'], swiss_ewbi_overall['Value'], 
                    color=SWITZERLAND_COLOR, linewidth=4, marker='o', 
                    markersize=10, label='Switzerland EWBI', alpha=0.9)
        
        if not eu27_ewbi_overall.empty:
            plt.plot(eu27_ewbi_overall['Year'], eu27_ewbi_overall['Value'], 
                    color=EU_27_COLOR, linewidth=4, marker='o', 
                    markersize=10, label='EU-27 EWBI', alpha=0.9)
        
        plt.title(f"European Well-Being Index (EWBI)\nSwitzerland vs EU-27 Comparison ({int(earliest_year)}-{int(latest_year)})", 
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

def create_ewbi_deciles_comparison_plot(swiss_ewbi_deciles, eu27_ewbi_deciles, output_dir):
    """Create EWBI decile decomposition bar chart comparing Switzerland vs EU-27 - latest year only"""
    if swiss_ewbi_deciles.empty and eu27_ewbi_deciles.empty:
        print("Warning: No EWBI decile data available for Switzerland or EU-27")
        return
    
    # Get latest year available (prefer Swiss data if available, otherwise EU-27)
    if not swiss_ewbi_deciles.empty:
        latest_year = swiss_ewbi_deciles['Year'].max()
    else:
        latest_year = eu27_ewbi_deciles['Year'].max()
    
    # Filter data for latest year only
    swiss_latest = pd.DataFrame()
    eu27_latest = pd.DataFrame()
    
    if not swiss_ewbi_deciles.empty:
        swiss_latest = swiss_ewbi_deciles[swiss_ewbi_deciles['Year'] == latest_year].copy()
        # Convert decile to numeric and sort
        swiss_latest['Decile_num'] = pd.to_numeric(swiss_latest['Decile'], errors='coerce')
        swiss_latest = swiss_latest.dropna(subset=['Decile_num'])
        swiss_latest = swiss_latest.sort_values('Decile_num')
    
    if not eu27_ewbi_deciles.empty:
        eu27_latest = eu27_ewbi_deciles[eu27_ewbi_deciles['Year'] == latest_year].copy()
        # Convert decile to numeric and sort
        eu27_latest['Decile_num'] = pd.to_numeric(eu27_latest['Decile'], errors='coerce')
        eu27_latest = eu27_latest.dropna(subset=['Decile_num'])
        eu27_latest = eu27_latest.sort_values('Decile_num')
    
    if swiss_latest.empty and eu27_latest.empty:
        print("Warning: No valid decile data for latest year")
        return
    
    # Create the plot
    fig = go.Figure()
    
    # Get all deciles that have data
    all_deciles = set()
    if not swiss_latest.empty:
        all_deciles.update(swiss_latest['Decile_num'])
    if not eu27_latest.empty:
        all_deciles.update(eu27_latest['Decile_num'])
    all_deciles = sorted(list(all_deciles))
    
    # Create bar chart for Switzerland
    if not swiss_latest.empty:
        swiss_x = [str(int(d)) for d in swiss_latest['Decile_num']]
        swiss_y = swiss_latest['Value'].tolist()
        fig.add_trace(go.Bar(
            x=swiss_x,
            y=swiss_y,
            name='Switzerland EWBI',
            marker=dict(color=SWITZERLAND_COLOR),
            text=[f'{v:.3f}' for v in swiss_y],
            textposition='auto',
            hovertemplate='<b>Switzerland - Decile %{x}</b><br>' +
                         'Score: %{y:.3f}<extra></extra>',
            offsetgroup=1
        ))
        
        # Add reference line for Switzerland average
        swiss_average = swiss_latest['Value'].mean()
        fig.add_hline(
            y=swiss_average,
            line_dash="dash",
            line_color=SWITZERLAND_COLOR,
            line_width=2,
            annotation_text=f"Switzerland Avg: {swiss_average:.3f}",
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
            text=f"European Well-Being Index (EWBI)<br><sub>Switzerland vs EU-27 by Income Decile ({int(latest_year)})</sub>",
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
    png_path = os.path.join(output_dir, "switzerland_vs_eu27_ewbi_deciles.png")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(14, 8))
        
        # Prepare data for grouped bar chart
        deciles = [str(int(d)) for d in all_deciles]
        x = np.arange(len(deciles))
        width = 0.35
        
        # Get values for each decile, use NaN for missing data
        swiss_values = []
        eu27_values = []
        
        for decile in all_deciles:
            # Switzerland values
            swiss_val = swiss_latest[swiss_latest['Decile_num'] == decile]['Value']
            swiss_values.append(swiss_val.iloc[0] if len(swiss_val) > 0 else np.nan)
            
            # EU-27 values
            eu27_val = eu27_latest[eu27_latest['Decile_num'] == decile]['Value']
            eu27_values.append(eu27_val.iloc[0] if len(eu27_val) > 0 else np.nan)
        
        # Create bars
        if not swiss_latest.empty:
            swiss_bars = plt.bar(x - width/2, swiss_values, width, 
                               color=SWITZERLAND_COLOR, alpha=0.8, label='Switzerland EWBI')
        
        if not eu27_latest.empty:
            eu27_bars = plt.bar(x + width/2, eu27_values, width,
                              color=EU_27_COLOR, alpha=0.8, label='EU-27 EWBI')
        
        # Add value labels on bars
        if not swiss_latest.empty:
            for i, v in enumerate(swiss_values):
                if not np.isnan(v):
                    plt.text(i - width/2, v + 0.001, f'{v:.3f}', 
                           ha='center', va='bottom', fontsize=9)
        
        if not eu27_latest.empty:
            for i, v in enumerate(eu27_values):
                if not np.isnan(v):
                    plt.text(i + width/2, v + 0.001, f'{v:.3f}', 
                           ha='center', va='bottom', fontsize=9)
        
        # Add reference lines for averages
        if not swiss_latest.empty:
            swiss_avg = swiss_latest['Value'].mean()
            plt.axhline(y=swiss_avg, color=SWITZERLAND_COLOR, linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'Switzerland Avg: {swiss_avg:.3f}')
        
        if not eu27_latest.empty:
            eu27_avg = eu27_latest['Value'].mean()
            plt.axhline(y=eu27_avg, color=EU_27_COLOR, linestyle=':', 
                       linewidth=2, alpha=0.7, label=f'EU-27 Avg: {eu27_avg:.3f}')
        
        plt.title(f"European Well-Being Index (EWBI)\nSwitzerland vs EU-27 by Income Decile ({int(latest_year)})", 
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

def create_eu_priority_comparison_plots(swiss_priorities_overall, swiss_priorities_deciles, 
                                       eu27_priorities_overall, eu27_priorities_deciles, 
                                       eu_priorities, output_dir):
    """Create EU priority comparison plots for Switzerland vs EU-27 (overall and deciles) for each priority"""
    
    for priority in eu_priorities:
        print(f"Creating plots for EU priority: {priority}")
        
        # Filter Switzerland data for this priority
        swiss_overall = swiss_priorities_overall[
            swiss_priorities_overall['EU priority'] == priority
        ].copy() if not swiss_priorities_overall.empty else pd.DataFrame()
        
        swiss_deciles = swiss_priorities_deciles[
            swiss_priorities_deciles['EU priority'] == priority
        ].copy() if not swiss_priorities_deciles.empty else pd.DataFrame()
        
        # Filter EU-27 data for this priority
        eu27_overall = eu27_priorities_overall[
            eu27_priorities_overall['EU priority'] == priority
        ].copy() if not eu27_priorities_overall.empty else pd.DataFrame()
        
        eu27_deciles = eu27_priorities_deciles[
            eu27_priorities_deciles['EU priority'] == priority
        ].copy() if not eu27_priorities_deciles.empty else pd.DataFrame()
        
        # Create overall comparison plot
        if not swiss_overall.empty or not eu27_overall.empty:
            create_eu_priority_overall_comparison_plot(swiss_overall, eu27_overall, priority, output_dir)
        else:
            print(f"Warning: No overall data for {priority}")
        
        # Create deciles comparison plot
        if not swiss_deciles.empty or not eu27_deciles.empty:
            create_eu_priority_deciles_comparison_plot(swiss_deciles, eu27_deciles, swiss_overall, eu27_overall, priority, output_dir)
        else:
            print(f"Warning: No decile data for {priority}")

def create_eu_priority_overall_comparison_plot(swiss_data, eu27_data, priority_name, output_dir):
    """Create overall temporal comparison graph for a specific EU priority"""
    
    # FIXED: Filter EU-27 data by aggregation method to match app.py time series logic
    if not eu27_data.empty:
        eu27_data = eu27_data[eu27_data['Aggregation'] == 'Population-weighted geometric mean'].copy()
    
    # Sort by year
    if not swiss_data.empty:
        swiss_data = swiss_data.sort_values('Year')
    if not eu27_data.empty:
        eu27_data = eu27_data.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add Switzerland EU priority line (solid yellow)
    if not swiss_data.empty:
        fig.add_trace(go.Scatter(
            x=swiss_data['Year'],
            y=swiss_data['Value'],
            mode='lines+markers',
            name=f'Switzerland {priority_name}',
            line=dict(color=SWITZERLAND_COLOR, width=4),
            marker=dict(color=SWITZERLAND_COLOR, size=10, symbol='circle'),
            hovertemplate=f'<b>Switzerland {priority_name}</b><br>' +
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
    if not swiss_data.empty:
        all_years.extend(swiss_data['Year'].tolist())
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
            text=f"{priority_name}<br><sub>Switzerland vs EU-27 Comparison ({int(earliest_year)}-{int(latest_year)})</sub>",
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
    png_path = os.path.join(output_dir, f"switzerland_vs_eu27_{safe_priority}_overall.png")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        if not swiss_data.empty:
            plt.plot(swiss_data['Year'], swiss_data['Value'], 
                    color=SWITZERLAND_COLOR, linewidth=4, marker='o', 
                    markersize=10, label=f'Switzerland {priority_name}', alpha=0.9)
        
        if not eu27_data.empty:
            plt.plot(eu27_data['Year'], eu27_data['Value'], 
                    color=EU_27_COLOR, linewidth=4, marker='o', 
                    markersize=10, label=f'EU-27 {priority_name}', alpha=0.9)
        
        plt.title(f"{priority_name}\nSwitzerland vs EU-27 Comparison ({int(earliest_year)}-{int(latest_year)})", 
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

def create_eu_priority_deciles_comparison_plot(swiss_data, eu27_data, swiss_overall, eu27_overall, priority_name, output_dir):
    """Create deciles comparison bar chart for a specific EU priority"""
    print(f"DEBUG: Creating deciles chart for {priority_name}")
    print(f"DEBUG: Swiss data: {len(swiss_data)} records")
    print(f"DEBUG: EU-27 data: {len(eu27_data)} records") 
    print(f"DEBUG: Swiss overall: {len(swiss_overall)} records")
    print(f"DEBUG: EU-27 overall: {len(eu27_overall)} records")
    
    if swiss_data.empty and eu27_data.empty:
        print(f"DEBUG: No data for {priority_name}")
        return
    
    # Get latest year available
    if not swiss_data.empty:
        latest_year = swiss_data['Year'].max()
    else:
        latest_year = eu27_data['Year'].max()
    
    # Filter data for latest year
    swiss_latest = pd.DataFrame()
    eu27_latest = pd.DataFrame()
    
    if not swiss_data.empty:
        swiss_latest = swiss_data[swiss_data['Year'] == latest_year].copy()
        swiss_latest['Decile_num'] = pd.to_numeric(swiss_latest['Decile'], errors='coerce')
        swiss_latest = swiss_latest.dropna(subset=['Decile_num']).sort_values('Decile_num')
    
    if not eu27_data.empty:
        eu27_latest = eu27_data[eu27_data['Year'] == latest_year].copy()
        eu27_latest['Decile_num'] = pd.to_numeric(eu27_latest['Decile'], errors='coerce')
        eu27_latest = eu27_latest.dropna(subset=['Decile_num']).sort_values('Decile_num')
    
    if swiss_latest.empty and eu27_latest.empty:
        return
    
    # Get all deciles
    all_deciles = set()
    if not swiss_latest.empty:
        all_deciles.update(swiss_latest['Decile_num'])
    if not eu27_latest.empty:
        all_deciles.update(eu27_latest['Decile_num'])
    all_deciles = sorted(list(all_deciles))
    
    # Get reference values for legend
    swiss_reference_value = None
    eu27_reference_value = None
    
    # Get Switzerland reference value
    if not swiss_latest.empty and not swiss_overall.empty:
        swiss_all_latest = swiss_overall[
            (swiss_overall['Year'] == latest_year) & 
            (swiss_overall['Decile'] == 'All Deciles')
        ]
        if not swiss_all_latest.empty:
            swiss_geom_interdecile = swiss_all_latest[swiss_all_latest['Aggregation'] == 'Geometric mean inter-decile']
            if not swiss_geom_interdecile.empty:
                swiss_reference_value = swiss_geom_interdecile['Value'].iloc[0]
    
    # Get EU-27 reference value  
    if not eu27_latest.empty and not eu27_overall.empty:
        eu27_all_latest = eu27_overall[
            (eu27_overall['Year'] == latest_year) & 
            (eu27_overall['Decile'] == 'All Deciles')
        ]
        if not eu27_all_latest.empty:
            eu27_geom_interdecile = eu27_all_latest[eu27_all_latest['Aggregation'] == 'Geometric mean inter-decile']
            if not eu27_geom_interdecile.empty and not pd.isna(eu27_geom_interdecile['Value'].iloc[0]):
                eu27_reference_value = eu27_geom_interdecile['Value'].iloc[0]
    
    # Create plot
    fig = go.Figure()
    
    # Switzerland bars
    if not swiss_latest.empty:
        swiss_name = f'Switzerland {priority_name}'
        if swiss_reference_value is not None:
            swiss_name += f' (All: {swiss_reference_value:.3f})'
            
        fig.add_trace(go.Bar(
            x=[str(int(d)) for d in swiss_latest['Decile_num']],
            y=swiss_latest['Value'].tolist(),
            name=swiss_name,
            marker=dict(color=SWITZERLAND_COLOR),
            text=[f'{v:.3f}' for v in swiss_latest['Value']],
            textposition='auto',
            hovertemplate=f'<b>Switzerland - Decile %{{x}}</b><br>Score: %{{y:.3f}}<extra></extra>',
            offsetgroup=1
        ))
        
        # Add reference line for Switzerland
        if swiss_reference_value is not None:
            print(f"DEBUG: Added Switzerland reference line at y={swiss_reference_value:.3f}")
            fig.add_hline(
                y=swiss_reference_value,
                line_dash="dash",
                line_color=SWITZERLAND_COLOR,
                line_width=2,
                annotation_text=f"Switzerland All: {swiss_reference_value:.3f}",
                annotation_position="top right"
            )
    
    # EU-27 bars
    if not eu27_latest.empty:
        eu27_name = f'EU-27 {priority_name}'
        if eu27_reference_value is not None:
            eu27_name += f' (All: {eu27_reference_value:.3f})'
            
        fig.add_trace(go.Bar(
            x=[str(int(d)) for d in eu27_latest['Decile_num']],
            y=eu27_latest['Value'].tolist(),
            name=eu27_name,
            marker=dict(color=EU_27_COLOR),
            text=[f'{v:.3f}' for v in eu27_latest['Value']],
            textposition='auto',
            hovertemplate=f'<b>EU-27 - Decile %{{x}}</b><br>Score: %{{y:.3f}}<extra></extra>',
            offsetgroup=2
        ))
        
        # Add reference line for EU-27
        if eu27_reference_value is not None:
            print(f"DEBUG: Added EU-27 reference line at y={eu27_reference_value:.3f}")
            fig.add_hline(
                y=eu27_reference_value,
                line_dash="dot",
                line_color=EU_27_COLOR,
                line_width=2,
                annotation_text=f"EU-27 All: {eu27_reference_value:.3f}",
                annotation_position="bottom left"
            )
    
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
            text=f"{priority_name}<br><sub>Switzerland vs EU-27 by Income Decile ({int(latest_year)})</sub>",
            font=dict(size=16, color="#2C3E50", family="Arial, sans-serif"),
            x=0.5
        ),
        xaxis=dict(
            title="Income Decile",
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=12, family="Arial, sans-serif")
        ),
        yaxis=dict(
            title="Score",
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=12, family="Arial, sans-serif")
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
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
    png_path = os.path.join(output_dir, f"switzerland_vs_eu27_{safe_priority}_deciles.png")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(14, 8))
        
        deciles = [str(int(d)) for d in all_deciles]
        x = np.arange(len(deciles))
        width = 0.35
        
        # Get values for plotting
        swiss_values = []
        eu27_values = []
        
        for decile in all_deciles:
            swiss_val = swiss_latest[swiss_latest['Decile_num'] == decile]['Value']
            swiss_values.append(swiss_val.iloc[0] if len(swiss_val) > 0 else np.nan)
            
            eu27_val = eu27_latest[eu27_latest['Decile_num'] == decile]['Value']
            eu27_values.append(eu27_val.iloc[0] if len(eu27_val) > 0 else np.nan)
        
        # Create bars
        if not swiss_latest.empty:
            swiss_label = f'Switzerland {priority_name}'
            if swiss_reference_value is not None:
                swiss_label += f' (All: {swiss_reference_value:.3f})'
            plt.bar(x - width/2, swiss_values, width, 
                   color=SWITZERLAND_COLOR, alpha=0.8, label=swiss_label)
        
        if not eu27_latest.empty:
            eu27_label = f'EU-27 {priority_name}'
            if eu27_reference_value is not None:
                eu27_label += f' (All: {eu27_reference_value:.3f})'
            plt.bar(x + width/2, eu27_values, width,
                   color=EU_27_COLOR, alpha=0.8, label=eu27_label)
        
        # Add reference lines (matching Plotly version)
        if swiss_reference_value is not None:
            plt.axhline(y=swiss_reference_value, color=SWITZERLAND_COLOR, linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'Switzerland All: {swiss_reference_value:.3f}')
        
        if eu27_reference_value is not None:
            plt.axhline(y=eu27_reference_value, color=EU_27_COLOR, linestyle=':', 
                       linewidth=2, alpha=0.7, label=f'EU-27 All: {eu27_reference_value:.3f}')
        
        plt.title(f"{priority_name}\nSwitzerland vs EU-27 by Income Decile ({int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Income Decile', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(x, deciles)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {priority_name}: {str(e)[:100]}...")

def create_swiss_eu_priority_overall_plot(priority_data, priority_name, output_dir):
    """Create overall temporal graph for a specific EU priority in Switzerland"""
    # Sort by year
    priority_data = priority_data.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add Switzerland EU priority line (solid yellow)
    fig.add_trace(go.Scatter(
        x=priority_data['Year'],
        y=priority_data['Value'],
        mode='lines+markers',
        name=f'Switzerland {priority_name}',
        line=dict(color=SWITZERLAND_COLOR, width=4),
        marker=dict(color=SWITZERLAND_COLOR, size=10, symbol='circle'),
        hovertemplate=f'<b>Switzerland {priority_name}</b><br>' +
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
            text=f"{priority_name} - Switzerland<br><sub>Overall Score Evolution ({int(earliest_year)}-{int(latest_year)})</sub>",
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
    png_path = os.path.join(output_dir, f"switzerland_{safe_priority.lower()}_overall.png")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        plt.plot(priority_data['Year'], priority_data['Value'], 
                color=SWITZERLAND_COLOR, linewidth=4, marker='o', 
                markersize=10, label=f'Switzerland {priority_name}', alpha=0.9)
        
        plt.title(f"{priority_name} - Switzerland\nOverall Score Evolution ({int(earliest_year)}-{int(latest_year)})", 
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

def create_swiss_eu_priority_deciles_plot(priority_deciles, priority_name, output_dir):
    """Create decile decomposition bar chart for a specific EU priority in Switzerland - latest year only"""
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
        name=f'Switzerland {priority_name} by Decile',
        marker=dict(color=SWITZERLAND_COLOR),
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
        line_color=SWITZERLAND_COLOR,
        line_width=2,
        annotation_text=f"Switzerland Average: {overall_average:.3f}",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{priority_name} - Switzerland<br><sub>By Income Decile ({int(latest_year)})</sub>",
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
    png_path = os.path.join(output_dir, f"switzerland_{safe_priority.lower()}_deciles.png")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        # Create bar chart
        bars = plt.bar([str(int(d)) for d in latest_data['Decile_num']], 
                      latest_data['Value'], 
                      color=SWITZERLAND_COLOR, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, latest_data['Value']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add reference line for average
        plt.axhline(y=overall_average, color=SWITZERLAND_COLOR, linestyle='--', 
                   linewidth=2, alpha=0.7, 
                   label=f'Switzerland Average: {overall_average:.3f}')
        
        plt.title(f"{priority_name} - Switzerland\nBy Income Decile ({int(latest_year)})", 
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
        (unified_df['Level'] == 3) &
        (unified_df['Decile'] == '1')
    ].copy()
    
    swiss_decile_10 = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 3) &
        (unified_df['Decile'] == '10')
    ].copy()
    
    if swiss_data.empty:
        print("No 'All Deciles' data found for Switzerland at Level 3. Computing country averages from decile data...")
        
        # Get all Switzerland Level 3 data and compute country averages by year and indicator
        swiss_decile_data = unified_df[
            (unified_df['Country'] == 'CH') & 
            (unified_df['Level'] == 3) &
            (unified_df['Decile'].isin(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']))
        ].copy()
        
        if swiss_decile_data.empty:
            print("Warning: No Switzerland data found at Level 3")
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
        
        # Add the All Deciles marker
        swiss_data['Decile'] = 'All Deciles'
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
    # Get EU-27 data from the unified dataset at Level 3 (primary indicators) - median data
    # Try the available aggregation method first
    eu27_data = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 3) &
        (unified_df['Decile'] == 'All Deciles') &
        (unified_df['Aggregation'] == 'Median across countries')
    ].copy()
    
    # Get 1st and 10th decile data for EU-27
    eu27_decile_1 = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 3) &
        (unified_df['Decile'] == '1') &
        (unified_df['Aggregation'] == 'Median across countries')
    ].copy()
    
    eu27_decile_10 = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 3) &
        (unified_df['Decile'] == '10') &
        (unified_df['Aggregation'] == 'Median across countries')
    ].copy()
    
    if eu27_data.empty:
        print("Warning: No EU-27 data found at Level 3 with 'All Deciles'")
        # Try with any decile to see what's available
        eu27_sample = unified_df[
            (unified_df['Country'] == 'All Countries') & 
            (unified_df['Level'] == 3)
        ].copy()
        
        if eu27_sample.empty:
            print("No Level 3 data found for 'All Countries'")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Get available deciles
        available_deciles = eu27_sample['Decile'].unique()
        print(f"Available deciles for EU-27 at Level 3: {available_deciles}")
        
        # Compute EU-27 averages across deciles for each year and indicator
        eu27_decile_data = eu27_sample[
            eu27_sample['Decile'].isin(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
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
    
    # Create safe filename - remove problematic characters and limit length
    safe_indicator = indicator.replace('-', '_').replace('/', '_').replace(' ', '_').replace(',', '')
    # Truncate if too long to avoid Windows path length issues  
    if len(safe_indicator) > 40:
        safe_indicator = safe_indicator[:40]
    
    # Save as PNG only
    
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
        print("  Only PNG version saved.")

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
    excel_path = os.path.join(output_dir, "switzerland_vs_eu27_primary_indicators.xlsx")
    
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
        csv_path = os.path.join(output_dir, "switzerland_vs_eu27_primary_indicators.csv")
        export_df.to_csv(csv_path, index=False)
        print(f"Exported to CSV instead: {csv_path}")

def export_ewbi_level1_to_excel(unified_df, output_dir):
    """Export EWBI Level 1 data (Switzerland + EU-27) to Excel"""
    
    all_data_records = []
    
    # Get EWBI Level 1 data for both Switzerland and EU-27 - match app.py aggregation logic
    ewbi_overall = unified_df[
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] == 'All') &
        (
            (unified_df['Country'] == 'CH') |
            ((unified_df['Country'] == 'All Countries') & 
             (unified_df['Aggregation'] == 'Population-weighted geometric mean'))
        )
    ].copy()
    
    # Get EWBI Level 1 decile data for both Switzerland and EU-27 - match app.py aggregation logic
    ewbi_deciles = unified_df[
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna()) &
        (
            (unified_df['Country'] == 'CH') |
            ((unified_df['Country'] == 'All Countries') & 
             (unified_df['Aggregation'] == 'Population-weighted geometric mean'))
        )
    ].copy()
    
    # Process overall EWBI data
    for _, row in ewbi_overall.iterrows():
        country_name = 'Switzerland' if row['Country'] == 'CH' else 'EU-27'
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
        country_name = 'Switzerland' if row['Country'] == 'CH' else 'EU-27'
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
    excel_path = os.path.join(output_dir, "switzerland_vs_eu27_ewbi_level1.xlsx")
    
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
    """Export EU Priorities Level 2 data (Switzerland + EU-27) to Excel"""
    
    # EU priorities from the app
    EU_PRIORITIES = [
        'Energy and Housing', 
        'Equality',
        'Health and Animal Welfare',
        'Intergenerational Fairness, Youth, Culture and Sport',
        'Social Rights and Skills, Quality Jobs and Preparedness'
    ]
    
    all_data_records = []
    
    # Get EU Priorities Level 2 data for Switzerland
    priorities_overall_ch = unified_df[
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] == 'All Deciles') &
        (unified_df['Country'] == 'CH') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Get EU Priorities Level 2 data for EU-27 (with proper aggregation filtering)
    priorities_overall_eu27 = unified_df[
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] == 'All Deciles') &
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Aggregation'] == 'Population-weighted geometric mean') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Combine overall data
    priorities_overall = pd.concat([priorities_overall_ch, priorities_overall_eu27], ignore_index=True)
    
    # Get EU Priorities Level 2 decile data for Switzerland
    priorities_deciles_ch = unified_df[
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna()) &
        (unified_df['Country'] == 'CH') &
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
        country_name = 'Switzerland' if row['Country'] == 'CH' else 'EU-27'
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
        country_name = 'Switzerland' if row['Country'] == 'CH' else 'EU-27'
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
    excel_path = os.path.join(output_dir, "switzerland_vs_eu27_eu_priorities_level2.xlsx")
    
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

def prepare_swiss_primary_indicators_data(unified_df):
    """Extract Switzerland Primary Indicators (Level 3) data for overall and decile analysis"""
    # Focus only on Energy and Housing EU priority
    EU_PRIORITIES = [
        'Energy and Housing'
    ]
    
    # Get Switzerland Primary Indicators overall data (Level 3, Decile='All Deciles')
    swiss_primary_overall = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 3) &
        (unified_df['Decile'] == 'All Deciles') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Get Switzerland Primary Indicators by deciles (Level 3, all deciles except 'All Deciles')
    swiss_primary_deciles = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 3) &
        (unified_df['Decile'] != 'All Deciles') &
        (unified_df['Decile'].notna()) &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    print(f"Swiss Primary Indicators overall data: {len(swiss_primary_overall)} records")
    print(f"Swiss Primary Indicators decile data: {len(swiss_primary_deciles)} records")
    print(f"Available Primary Indicators: {sorted(swiss_primary_overall['Primary and raw data'].dropna().unique()) if not swiss_primary_overall.empty else []}")
    
    return swiss_primary_overall, swiss_primary_deciles

def prepare_eu27_primary_indicators_data(unified_df):
    """Extract EU-27 Primary Indicators (Level 3) data for overall and decile analysis"""
    EU_PRIORITIES = [
        'Energy and Housing'
    ]
    
    # Get EU-27 Primary Indicators overall data (Level 3, Decile='All Deciles')
    eu27_primary_overall = unified_df[
        (unified_df['Country'] == 'EU-27') &
        (unified_df['Level'] == 3) &
        (unified_df['Decile'] == 'All Deciles') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Get EU-27 Primary Indicators by deciles (Level 3, individual deciles 1-10)
    eu27_primary_deciles = unified_df[
        (unified_df['Country'] == 'EU-27') &
        (unified_df['Level'] == 3) &
        (unified_df['Decile'] != 'All Deciles') &
        (unified_df['Decile'].notna()) &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    print(f"EU-27 Primary Indicators overall data: {len(eu27_primary_overall)} records")
    print(f"EU-27 Primary Indicators decile data: {len(eu27_primary_deciles)} records")
    print(f"Available EU-27 Primary Indicators: {sorted(eu27_primary_overall['Primary and raw data'].dropna().unique()) if not eu27_primary_overall.empty else []}")
    
    return eu27_primary_overall, eu27_primary_deciles

def create_primary_indicator_deciles_comparison_plot(swiss_data, eu27_data, swiss_overall, eu27_overall, indicator_name, output_dir):
    """Create deciles comparison bar chart for a specific Primary Indicator (Level 3)"""
    print(f"Creating deciles chart for Primary Indicator: {indicator_name}")
    print(f"Swiss data: {len(swiss_data)} records")
    print(f"EU-27 data: {len(eu27_data)} records")
    
    if swiss_data.empty and eu27_data.empty:
        print(f"No decile data for {indicator_name}")
        return
    
    # Get latest year available
    if not swiss_data.empty:
        latest_year = swiss_data['Year'].max()
    else:
        latest_year = eu27_data['Year'].max()
    
    # Filter data for latest year
    swiss_latest = pd.DataFrame()
    eu27_latest = pd.DataFrame()
    
    if not swiss_data.empty:
        swiss_latest = swiss_data[swiss_data['Year'] == latest_year].copy()
        swiss_latest['Decile_num'] = pd.to_numeric(swiss_latest['Decile'], errors='coerce')
        swiss_latest = swiss_latest.dropna(subset=['Decile_num']).sort_values('Decile_num')
    
    if not eu27_data.empty:
        eu27_latest = eu27_data[eu27_data['Year'] == latest_year].copy()
        eu27_latest['Decile_num'] = pd.to_numeric(eu27_latest['Decile'], errors='coerce')
        eu27_latest = eu27_latest.dropna(subset=['Decile_num']).sort_values('Decile_num')
    
    if swiss_latest.empty and eu27_latest.empty:
        print(f"No data for latest year ({latest_year}) for {indicator_name}")
        return
    
    # Get all deciles
    all_deciles = set()
    if not swiss_latest.empty and 'Decile_num' in swiss_latest.columns:
        all_deciles.update(swiss_latest['Decile_num'])
    if not eu27_latest.empty and 'Decile_num' in eu27_latest.columns:
        all_deciles.update(eu27_latest['Decile_num'])
    
    if not all_deciles:
        print(f"Warning: No valid decile data for {indicator_name}")
        return
    
    all_deciles = sorted(list(all_deciles))
    
    # Get reference values for legend
    swiss_reference_value = None
    eu27_reference_value = None
    
    # Get Switzerland reference value
    if not swiss_latest.empty and not swiss_overall.empty:
        swiss_all_latest = swiss_overall[
            (swiss_overall['Year'] == latest_year) & 
            (swiss_overall['Decile'] == 'All Deciles') &
            (swiss_overall['Primary and raw data'] == indicator_name)
        ]
        if not swiss_all_latest.empty:
            swiss_geom_interdecile = swiss_all_latest[swiss_all_latest['Aggregation'] == 'Geometric mean inter-decile']
            if not swiss_geom_interdecile.empty:
                swiss_reference_value = swiss_geom_interdecile['Value'].iloc[0]
    
    # Get EU-27 reference value
    if not eu27_latest.empty and not eu27_overall.empty:
        eu27_all_latest = eu27_overall[
            (eu27_overall['Year'] == latest_year) & 
            (eu27_overall['Decile'] == 'All Deciles') &
            (eu27_overall['Primary and raw data'] == indicator_name)
        ]
        if not eu27_all_latest.empty:
            eu27_geom_interdecile = eu27_all_latest[eu27_all_latest['Aggregation'] == 'Geometric mean inter-decile']
            if not eu27_geom_interdecile.empty and not pd.isna(eu27_geom_interdecile['Value'].iloc[0]):
                eu27_reference_value = eu27_geom_interdecile['Value'].iloc[0]
    
    # Create safe filename from indicator name
    safe_indicator = indicator_name.lower().replace(' ', '_').replace(',', '').replace('(', '').replace(')', '').replace('/', '_')
    # Limit length to avoid path issues
    if len(safe_indicator) > 50:
        safe_indicator = safe_indicator[:50]
    
    # Save PNG only
    png_path = os.path.join(output_dir, f"switzerland_vs_eu27_{safe_indicator}_deciles.png")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(14, 8))
        
        # Prepare data for grouped bar chart
        deciles = [str(int(d)) for d in all_deciles]
        x = np.arange(len(deciles))
        width = 0.35
        
        # Get values for each decile, use NaN for missing data
        swiss_values = []
        eu27_values = []
        
        for decile in all_deciles:
            # Switzerland values
            if not swiss_latest.empty and 'Decile_num' in swiss_latest.columns:
                swiss_val = swiss_latest[swiss_latest['Decile_num'] == decile]['Value']
                swiss_values.append(swiss_val.iloc[0] if len(swiss_val) > 0 else np.nan)
            else:
                swiss_values.append(np.nan)
            
            # EU-27 values
            if not eu27_latest.empty and 'Decile_num' in eu27_latest.columns:
                eu27_val = eu27_latest[eu27_latest['Decile_num'] == decile]['Value']
                eu27_values.append(eu27_val.iloc[0] if len(eu27_val) > 0 else np.nan)
            else:
                eu27_values.append(np.nan)
        
        # Create bars
        if not swiss_latest.empty:
            swiss_bars = plt.bar(x - width/2, swiss_values, width, 
                               color=SWITZERLAND_COLOR, alpha=0.8, label='Switzerland')
        
        if not eu27_latest.empty:
            eu27_bars = plt.bar(x + width/2, eu27_values, width,
                              color=EU_27_COLOR, alpha=0.8, label='EU-27')
        
        # Add value labels on bars
        if not swiss_latest.empty:
            for i, v in enumerate(swiss_values):
                if not np.isnan(v):
                    plt.text(i - width/2, v + v*0.01, f'{v:.3f}', 
                           ha='center', va='bottom', fontsize=9)
        
        if not eu27_latest.empty:
            for i, v in enumerate(eu27_values):
                if not np.isnan(v):
                    plt.text(i + width/2, v + v*0.01, f'{v:.3f}', 
                           ha='center', va='bottom', fontsize=9)
        
        # Add reference lines for averages
        if not swiss_latest.empty and swiss_reference_value is not None:
            plt.axhline(y=swiss_reference_value, color=SWITZERLAND_COLOR, linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'Switzerland All: {swiss_reference_value:.3f}')
        
        if not eu27_latest.empty and eu27_reference_value is not None:
            plt.axhline(y=eu27_reference_value, color=EU_27_COLOR, linestyle=':', 
                       linewidth=2, alpha=0.7, label=f'EU-27 All: {eu27_reference_value:.3f}')
        
        plt.title(f"{indicator_name}\nSwitzerland vs EU-27 by Income Decile ({int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Income Decile', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(x, deciles)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(fontsize=11, loc='best')
        
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {indicator_name} deciles: {str(e)[:100]}...")

def create_primary_indicators_deciles_plots(swiss_primary_overall, swiss_primary_deciles,
                                           eu27_primary_overall, eu27_primary_deciles, output_dir):
    """Create decile comparison plots for all Primary Indicators related to Energy and Housing"""
    
    # Get all unique Primary Indicators (Primary and raw data column at Level 3)
    all_indicators = set()
    if not swiss_primary_deciles.empty:
        all_indicators.update(swiss_primary_deciles['Primary and raw data'].dropna().unique())
    if not eu27_primary_deciles.empty:
        all_indicators.update(eu27_primary_deciles['Primary and raw data'].dropna().unique())
    
    if not all_indicators:
        print("Warning: No Primary Indicators found for decile analysis")
        return
    
    print(f"Creating decile plots for {len(all_indicators)} Primary Indicators")
    
    for indicator in sorted(all_indicators):
        print(f"\nProcessing Primary Indicator: {indicator}")
        
        # Filter Switzerland data for this indicator
        swiss_overall = swiss_primary_overall[
            swiss_primary_overall['Primary and raw data'] == indicator
        ].copy() if not swiss_primary_overall.empty else pd.DataFrame()
        
        swiss_deciles = swiss_primary_deciles[
            swiss_primary_deciles['Primary and raw data'] == indicator
        ].copy() if not swiss_primary_deciles.empty else pd.DataFrame()
        
        # Filter EU-27 data for this indicator
        eu27_overall = eu27_primary_overall[
            eu27_primary_overall['Primary and raw data'] == indicator
        ].copy() if not eu27_primary_overall.empty else pd.DataFrame()
        
        eu27_deciles = eu27_primary_deciles[
            eu27_primary_deciles['Primary and raw data'] == indicator
        ].copy() if not eu27_primary_deciles.empty else pd.DataFrame()
        
        # Create deciles comparison plot
        if not swiss_deciles.empty or not eu27_deciles.empty:
            create_primary_indicator_deciles_comparison_plot(
                swiss_deciles, eu27_deciles, swiss_overall, eu27_overall, indicator, output_dir
            )
        else:
            print(f"Warning: No decile data for {indicator}")

def main():
    """Main function to generate time series graphs for app-displayed Level 3 primary indicators plus Switzerland EWBI and EU priorities"""
    
    print("Starting Swiss vs EU-27 App-Displayed Level 3 Primary Indicators Time Series Analysis")
    print("Plus Switzerland EWBI and EU Priorities Analysis")
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
    
    # PART 1: Generate Switzerland EWBI and EU priorities graphs
    print("\n" + "="*50)
    print("PART 1: SWITZERLAND EWBI AND EU PRIORITIES ANALYSIS")
    print("="*50)
    
    # 1.1 Switzerland vs EU-27 EWBI (Level 1) - Overall and Deciles
    print("\n1.1. Processing EWBI (Level 1) - Switzerland vs EU-27...")
    swiss_ewbi_overall, swiss_ewbi_deciles = prepare_swiss_ewbi_data(unified_df)
    eu27_ewbi_overall, eu27_ewbi_deciles = prepare_eu27_ewbi_data(unified_df)
    
    if not swiss_ewbi_overall.empty or not eu27_ewbi_overall.empty:
        print("Creating EWBI overall comparison graph...")
        create_ewbi_overall_comparison_plot(swiss_ewbi_overall, eu27_ewbi_overall, OUTPUT_DIRS['level1'])
    else:
        print("Warning: No EWBI overall data found for Switzerland or EU-27")
    
    if not swiss_ewbi_deciles.empty or not eu27_ewbi_deciles.empty:
        print("Creating EWBI deciles comparison graph...")
        create_ewbi_deciles_comparison_plot(swiss_ewbi_deciles, eu27_ewbi_deciles, OUTPUT_DIRS['level1'])
    else:
        print("Warning: No EWBI deciles data found for Switzerland or EU-27")
    
    # 1.2 Switzerland vs EU-27 EU Priorities (Level 2) - Overall and Deciles
    print("\n1.2. Processing EU Priorities (Level 2) - Switzerland vs EU-27...")
    swiss_priorities_overall, swiss_priorities_deciles, eu_priorities = prepare_swiss_eu_priorities_data(unified_df)
    eu27_priorities_overall, eu27_priorities_deciles, _ = prepare_eu27_eu_priorities_data(unified_df)
    
    if (not swiss_priorities_overall.empty or not swiss_priorities_deciles.empty or 
        not eu27_priorities_overall.empty or not eu27_priorities_deciles.empty):
        print(f"Creating comparison graphs for {len(eu_priorities)} EU priorities...")
        create_eu_priority_comparison_plots(swiss_priorities_overall, swiss_priorities_deciles, 
                                          eu27_priorities_overall, eu27_priorities_deciles,
                                          eu_priorities, OUTPUT_DIRS['level2'])
    else:
        print("Warning: No EU priorities data found for Switzerland or EU-27")
    
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
    
    # 1.4 Switzerland vs EU-27 Primary Indicators (Level 3) - Deciles only
    print("\n1.4. Processing Primary Indicators (Level 3) - Switzerland vs EU-27 Deciles...")
    swiss_primary_overall, swiss_primary_deciles = prepare_swiss_primary_indicators_data(unified_df)
    eu27_primary_overall, eu27_primary_deciles = prepare_eu27_primary_indicators_data(unified_df)
    
    if not swiss_primary_deciles.empty or not eu27_primary_deciles.empty:
        print("Creating Primary Indicators decile comparison graphs...")
        create_primary_indicators_deciles_plots(
            swiss_primary_overall, swiss_primary_deciles,
            eu27_primary_overall, eu27_primary_deciles,
            OUTPUT_DIRS['level3']
        )
    else:
        print("Warning: No Primary Indicators decile data found for Switzerland or EU-27")
    
    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print(" ALL DECILE ANALYSES COMPLETED SUCCESSFULLY!")
    print(f" Output directory: {output_dir}")
    print("\nGenerated decile bar charts:")
    print("  • Level 1: EWBI decile comparison")
    print("  • Level 2: EU Priorities (Energy and Housing) decile comparisons")
    print("  • Level 3: Primary Indicators decile comparisons")
    print("\n Data source: ewbi_master_aggregated.csv")
    print(f" Color scheme: Switzerland ({SWITZERLAND_COLOR}), EU-27 ({EU_27_COLOR})")
    print(" Output format: PNG (static images)")
    print("\n Note: EU-27 data may be unavailable for some indicators in current dataset.")

if __name__ == "__main__":
    main()