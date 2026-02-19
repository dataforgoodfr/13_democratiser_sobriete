"""
ewbi_treatment.py - Generate comprehensive analysis graphs for Switzerland only

This script creates comprehensive analysis graphs for Switzerland showing:
- EWBI (Level 1) overall and by decile decomposition over time
- EU priorities (Level 2) overall and by decile decomposition over time
- Primary indicators (Level 5) time series analysis

Focus is on Switzerland's performance across all well-being dimensions without EU comparison.
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
well_being_code_dir = os.path.abspath(os.path.join(reports_dir, '..', 'code'))

sys.path.insert(0, current_dir)
sys.path.insert(0, shared_code_dir)
sys.path.insert(0, well_being_code_dir)

# Import shared utilities
from ewbi_data_loader import load_ewbi_unified_data, get_housing_energy_indicators
from visualization_utils import create_time_series_plot, save_plot

# Import variable mapping functions from Well-being pipeline
from variable_mapping import get_display_name

# Color palette for Switzerland analysis
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
SWITZERLAND_COLOR = '#ffd558'  # Use yellow for Switzerland

def get_app_level5_indicators():
    """Get the Level 5 indicators that are actually available in the app's dropdown by replicating app.py logic"""
    # Use the shared EWBI data loader to get the unified dataset
    try:
        df = load_ewbi_unified_data()
    except Exception as e:
        print(f"Error loading EWBI unified data for app indicators: {e}")
        return []

    # Get EU priorities dynamically from Level 2 data (same as app.py)
    available_eu_priorities = df[df['Level']==2]['EU priority'].dropna().unique()
    EU_PRIORITIES = sorted(available_eu_priorities.tolist())

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

def prepare_swiss_ewbi_data(unified_df):
    """Extract Switzerland EWBI (Level 1) data for overall and decile analysis"""
    # Get Switzerland EWBI overall data (Level 1, Decile='All')
    swiss_ewbi_overall = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] == 'All')
    ].copy()
    
    # Get Switzerland EWBI by deciles (Level 1, all deciles except 'All')
    swiss_ewbi_deciles = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna())
    ].copy()
    
    print(f"Swiss EWBI overall data: {len(swiss_ewbi_overall)} records")
    print(f"Swiss EWBI decile data: {len(swiss_ewbi_deciles)} records")
    
    return swiss_ewbi_overall, swiss_ewbi_deciles

def prepare_swiss_eu_priorities_data(unified_df):
    """Extract Switzerland EU priorities (Level 2) data for overall and decile analysis"""
    # Get EU priorities dynamically from Level 2 data (same as app.py)
    available_eu_priorities = unified_df[unified_df['Level']==2]['EU priority'].dropna().unique()
    EU_PRIORITIES = sorted(available_eu_priorities.tolist())
    
    # Get Switzerland EU priorities overall data (Level 2, Decile='All')
    # For Switzerland: NO aggregation filtering (matches app.py logic)
    swiss_priorities_overall = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] == 'All') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Get Switzerland EU priorities by deciles (Level 2, all deciles except 'All')
    # For Switzerland: NO aggregation filtering (matches app.py logic)
    swiss_priorities_deciles = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna()) &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    print(f"Swiss EU priorities overall data: {len(swiss_priorities_overall)} records")
    print(f"Swiss EU priorities decile data: {len(swiss_priorities_deciles)} records")
    print(f"Available EU priorities: {sorted(swiss_priorities_overall['EU priority'].unique())}")
    
    return swiss_priorities_overall, swiss_priorities_deciles, EU_PRIORITIES

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
    # Get EU priorities dynamically from Level 2 data (same as app.py)
    available_eu_priorities = unified_df[unified_df['Level']==2]['EU priority'].dropna().unique()
    EU_PRIORITIES = sorted(available_eu_priorities.tolist())
    
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

def create_swiss_ewbi_overall_plot(swiss_ewbi_overall, output_dir):
    """Create EWBI overall temporal graph for Switzerland only"""
    if swiss_ewbi_overall.empty:
        print("Warning: No Switzerland EWBI overall data available")
        return
    
    # Sort by year
    swiss_ewbi_overall = swiss_ewbi_overall.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add Switzerland EWBI line (solid yellow)
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
    
    # Get time range
    years = swiss_ewbi_overall['Year'].tolist()
    earliest_year = min(years)
    latest_year = max(years)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Switzerland - European Well-Being Index (EWBI)<br><sub>Overall Score Evolution ({int(earliest_year)}-{int(latest_year)})</sub>",
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
    
    # Save PNG only (no HTML)
    png_path = os.path.join(output_dir, "switzerland_ewbi_overall.png")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        plt.plot(swiss_ewbi_overall['Year'], swiss_ewbi_overall['Value'], 
                color=SWITZERLAND_COLOR, linewidth=4, marker='o', 
                markersize=10, label='Switzerland EWBI', alpha=0.9)
        
        plt.title(f"Switzerland - European Well-Being Index (EWBI)\nOverall Score Evolution ({int(earliest_year)}-{int(latest_year)})", 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('EWBI Score', fontsize=14)
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
        print(f"Warning: Could not save PNG for EWBI overall: {str(e)[:100]}...")

def create_swiss_ewbi_deciles_plot(swiss_ewbi_deciles, output_dir):
    """Create EWBI decile decomposition bar chart for Switzerland - latest year only"""
    if swiss_ewbi_deciles.empty:
        print("Warning: No Switzerland EWBI decile data available")
        return
    
    # Get latest year available
    latest_year = swiss_ewbi_deciles['Year'].max()
    
    # Filter data for latest year only
    swiss_latest = swiss_ewbi_deciles[swiss_ewbi_deciles['Year'] == latest_year].copy()
    
    # Convert decile to numeric and sort
    swiss_latest['Decile_num'] = pd.to_numeric(swiss_latest['Decile'], errors='coerce')
    swiss_latest = swiss_latest.dropna(subset=['Decile_num'])
    swiss_latest = swiss_latest.sort_values('Decile_num')
    
    if swiss_latest.empty:
        print("Warning: No valid decile data for latest year")
        return
    
    # Create the plot
    fig = go.Figure()
    
    # Create bar chart for Switzerland
    swiss_x = [str(int(d)) for d in swiss_latest['Decile_num']]
    swiss_y = swiss_latest['Value'].tolist()
    fig.add_trace(go.Bar(
        x=swiss_x,
        y=swiss_y,
        name='Switzerland EWBI by Decile',
        marker=dict(color=SWITZERLAND_COLOR),
        text=[f'{v:.3f}' for v in swiss_y],
        textposition='auto',
        hovertemplate='<b>Switzerland - Decile %{x}</b><br>' +
                     'Score: %{y:.3f}<extra></extra>'
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
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Switzerland - European Well-Being Index (EWBI)<br><sub>By Income Decile ({int(latest_year)})</sub>",
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
        showlegend=False,
        margin=dict(t=80, b=60, l=60, r=40),
        width=1000,
        height=600
    )
    
    # Save PNG only (no HTML)
    png_path = os.path.join(output_dir, "switzerland_ewbi_deciles.png")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(12, 7))
        
        # Create bar chart without contours (matching Level 1 style)
        bars = plt.bar([str(int(d)) for d in swiss_latest['Decile_num']], 
                      swiss_latest['Value'], 
                      color=SWITZERLAND_COLOR, alpha=0.8, edgecolor='none')
        
        # Add value labels on bars
        for bar, value in zip(bars, swiss_latest['Value']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add reference line for average
        plt.axhline(y=swiss_average, color=SWITZERLAND_COLOR, linestyle='--', 
                   linewidth=2, alpha=0.7, 
                   label=f'Switzerland Average: {swiss_average:.3f}')
        
        plt.title(f"Switzerland - European Well-Being Index (EWBI)\\nBy Income Decile ({int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Income Decile', fontsize=14)
        plt.ylabel('EWBI Score', fontsize=14)
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

def create_swiss_eu_priorities_plots(swiss_priorities_overall, swiss_priorities_deciles, eu_priorities, output_dir):
    """Create EU priority plots for Switzerland (overall and deciles) for each priority"""
    print(f"Creating graphs for {len(eu_priorities)} EU priorities...")
    
    for priority in eu_priorities:
        print(f"Creating plots for EU priority: {priority}")
        
        # Filter data for this priority - overall plot
        priority_overall_data = swiss_priorities_overall[
            swiss_priorities_overall['EU priority'] == priority
        ].copy()
        
        # Filter data for this priority - deciles plot
        priority_deciles_data = swiss_priorities_deciles[
            swiss_priorities_deciles['EU priority'] == priority
        ].copy()
        
        # Create overall plot if data available
        if not priority_overall_data.empty:
            create_swiss_eu_priority_overall_plot(priority_overall_data, priority, output_dir)
        else:
            print(f"Warning: No overall data for {priority}")
            
        # Create deciles plot if data available
        if not priority_deciles_data.empty:
            create_swiss_eu_priority_deciles_plot(priority_deciles_data, priority, output_dir)
        else:
            print(f"Warning: No decile data for {priority}")

def create_swiss_eu_priority_overall_plot(priority_data, priority_name, output_dir):
    """Create overall temporal graph for a specific EU priority in Switzerland"""
    # Sort by year
    priority_data = priority_data.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add Switzerland EU priority line
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
    
    # Create safe filename
    safe_priority = priority_name.lower().replace(' ', '_').replace(',', '').replace('&', 'and').replace('-', '_')
    if len(safe_priority) > 30:
        safe_priority = safe_priority[:30]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Switzerland - {priority_name}<br><sub>Overall Score Evolution ({int(earliest_year)}-{int(latest_year)})</sub>",
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
        showlegend=False,
        margin=dict(t=80, b=60, l=60, r=40),
        width=1000,
        height=600
    )
    
    # Save PNG only (no HTML)
    # Generate PNG using matplotlib
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        # Convert years to datetime objects for better plotting
        years_datetime = [datetime(int(year), 1, 1) for year in priority_data['Year']]
        
        # Create matplotlib figure
        plt.figure(figsize=(12, 8))
        
        # Plot the line
        plt.plot(years_datetime, priority_data['Value'], 
                color=SWITZERLAND_COLOR, linewidth=3, marker='o', 
                markersize=8, label=f'Switzerland {priority_name}')
        
        # Customize the plot
        plt.title(f"Switzerland - {priority_name}\nOverall Score Evolution ({int(earliest_year)}-{int(latest_year)})",
                 fontsize=16, fontweight='bold', color="#2C3E50", pad=20)
        plt.xlabel("Year", fontsize=14, fontweight='bold')
        plt.ylabel("Score", fontsize=14, fontweight='bold')
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        if len(years) <= 10:
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        else:
            plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
        plt.xticks(rotation=45)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save PNG
        png_path = os.path.join(output_dir, f"switzerland_{safe_priority}_overall.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {priority_name} overall plot: {e}")

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
    
    # Create safe filename
    safe_priority = priority_name.lower().replace(' ', '_').replace(',', '').replace('&', 'and').replace('-', '_')
    if len(safe_priority) > 30:
        safe_priority = safe_priority[:30]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Switzerland - {priority_name}<br><sub>By Income Decile ({int(latest_year)})</sub>",
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
    
    # Save PNG only (no HTML)
    # Generate PNG using matplotlib
    try:
        import matplotlib.pyplot as plt
        
        # Create matplotlib figure
        plt.figure(figsize=(12, 8))
        
        # Create bar chart without contours (matching Level 1 style)
        x_positions = [int(d) for d in latest_data['Decile_num']]
        plt.bar(x_positions, latest_data['Value'], 
               color=SWITZERLAND_COLOR, alpha=0.8, edgecolor='none')
        
        # Add value labels on bars
        for i, (pos, val) in enumerate(zip(x_positions, latest_data['Value'])):
            plt.text(pos, val + (max(latest_data['Value']) * 0.01), f'{val:.3f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add reference line for average
        plt.axhline(y=overall_average, color=SWITZERLAND_COLOR, linestyle='--', 
                   linewidth=2, alpha=0.7, 
                   label=f'Switzerland Average: {overall_average:.3f}')
        
        # Customize the plot
        plt.title(f"Switzerland - {priority_name}\nBy Income Decile ({int(latest_year)})",
                 fontsize=16, fontweight='bold', color="#2C3E50", pad=20)
        plt.xlabel("Income Decile", fontsize=14, fontweight='bold')
        plt.ylabel("Score", fontsize=14, fontweight='bold')
        
        # Set x-axis ticks
        plt.xticks(x_positions, [str(pos) for pos in x_positions])
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save PNG
        png_path = os.path.join(output_dir, f"switzerland_{safe_priority}_deciles.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {priority_name} deciles plot: {e}")

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

def prepare_swiss_level3_data_by_eu_priority(unified_df):
    """Extract Switzerland Level 3 (primary indicators) data by EU Priority for decile visualization"""
    # Get EU priorities dynamically from Level 2 data (same as app.py)
    available_eu_priorities = unified_df[unified_df['Level']==2]['EU priority'].dropna().unique()
    EU_PRIORITIES = sorted(available_eu_priorities.tolist())
    
    print(f"EU priorities from Level 2 data: {EU_PRIORITIES}")
    
    # Get Switzerland Level 3 data for all deciles (1-10)
    # Don't filter by notna() on column name - filter by actual values later
    swiss_level3_deciles = unified_df[
        (unified_df['Country'] == 'CH') &
        (unified_df['Level'] == 3) &
        (unified_df['Decile'].isin(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']))
    ].copy()
    
    print(f"Swiss Level 3 decile data (all priorities): {len(swiss_level3_deciles)} records")
    print(f"  Unique EU priorities before filtering: {sorted(swiss_level3_deciles['EU priority'].dropna().unique())}")
    print(f"  EU_PRIORITIES list from Level 2: {EU_PRIORITIES}")
    
    # Now filter by EU priorities that exist in Level 2
    swiss_level3_deciles = swiss_level3_deciles[
        swiss_level3_deciles['EU priority'].isin(EU_PRIORITIES)
    ].copy()
    
    print(f"Swiss Level 3 decile data (filtered by Level 2 EU priorities): {len(swiss_level3_deciles)} records")
    
    if not swiss_level3_deciles.empty:
        print(f"Available EU priorities in Switzerland Level 3: {sorted(swiss_level3_deciles['EU priority'].unique())}")
        for priority in sorted(swiss_level3_deciles['EU priority'].unique()):
            priority_data = swiss_level3_deciles[swiss_level3_deciles['EU priority'] == priority]
            indicators = priority_data['Primary and raw data'].unique()
            # Count how many indicators have at least some non-null values
            valid_indicators = []
            for ind in indicators:
                if pd.notna(ind):
                    ind_data = priority_data[priority_data['Primary and raw data'] == ind]
                    if ind_data['Value'].notna().any():
                        valid_indicators.append(ind)
            print(f"  {priority}: {len(indicators)} total indicators, {len(valid_indicators)} with values")
    
    return swiss_level3_deciles, EU_PRIORITIES

def create_swiss_primary_indicator_plot(swiss_data, swiss_decile_1, swiss_decile_10, indicator, output_dir):
    """Create a time series plot for a specific primary indicator in Switzerland with deciles"""
    
    # Filter data for the specific indicator
    swiss_indicator = swiss_data[swiss_data['Indicator'] == indicator].copy()
    
    # Filter decile data (handle empty dataframes)
    swiss_d1 = swiss_decile_1[swiss_decile_1['Indicator'] == indicator].copy() if not swiss_decile_1.empty and 'Indicator' in swiss_decile_1.columns else pd.DataFrame()
    swiss_d10 = swiss_decile_10[swiss_decile_10['Indicator'] == indicator].copy() if not swiss_decile_10.empty and 'Indicator' in swiss_decile_10.columns else pd.DataFrame()
    
    if swiss_indicator.empty:
        print(f"Warning: No data available for indicator {indicator}")
        return
    
    # Find the time range
    all_years = []
    swiss_years = []
    
    if not swiss_indicator.empty:
        swiss_years.extend(swiss_indicator['Year'].tolist())
        all_years.extend(swiss_indicator['Year'].tolist())
    if not swiss_d1.empty:
        swiss_years.extend(swiss_d1['Year'].tolist())
        all_years.extend(swiss_d1['Year'].tolist())
    if not swiss_d10.empty:
        swiss_years.extend(swiss_d10['Year'].tolist())
        all_years.extend(swiss_d10['Year'].tolist())
    
    if not all_years:
        print(f"Warning: No data found for indicator {indicator}")
        return
    
    earliest_year = min(all_years)
    latest_year = max(swiss_years) if swiss_years else max(all_years)
    
    # Filter all data to not exceed Switzerland's latest year
    swiss_indicator = swiss_indicator[swiss_indicator['Year'] <= latest_year]
    swiss_d1 = swiss_d1[swiss_d1['Year'] <= latest_year] if not swiss_d1.empty else swiss_d1
    swiss_d10 = swiss_d10[swiss_d10['Year'] <= latest_year] if not swiss_d10.empty else swiss_d10
    
    # Sort by year
    swiss_indicator = swiss_indicator.sort_values('Year')
    if not swiss_d1.empty:
        swiss_d1 = swiss_d1.sort_values('Year')
    if not swiss_d10.empty:
        swiss_d10 = swiss_d10.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # 1. Switzerland average line (solid yellow)
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
    
    # Get indicator description
    description = get_display_name(indicator)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Switzerland - {description}<br><sub>Time Series ({int(earliest_year)}-{int(latest_year)})</sub>",
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
            dtick=1 if (latest_year - earliest_year) <= 10 else 2,
            tickmode='linear',
            range=[earliest_year - 0.5, latest_year + 0.5]
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
    safe_indicator = indicator.replace('-', '_').replace('/', '_').replace(' ', '_').replace(',', '')
    if len(safe_indicator) > 40:
        safe_indicator = safe_indicator[:40]
    
    # Save PNG only (no HTML)
    png_path = os.path.join(output_dir, f"switzerland_{safe_indicator}.png")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        # Plot Switzerland average line
        plt.plot(swiss_indicator['Year'], swiss_indicator['Value'], 
                color=SWITZERLAND_COLOR, linewidth=3, marker='o', 
                markersize=8, label='Switzerland Average', alpha=0.9)
        
        # Plot decile lines if available
        if not swiss_d1.empty:
            plt.plot(swiss_d1['Year'], swiss_d1['Value'], 
                    color=SWITZERLAND_COLOR, linewidth=2, linestyle='--', marker='^', markersize=6,
                    label='Switzerland 1st Decile', alpha=0.7)
        
        if not swiss_d10.empty:
            plt.plot(swiss_d10['Year'], swiss_d10['Value'],
                    color=SWITZERLAND_COLOR, linewidth=2, linestyle='-.', marker='v', markersize=6,
                    label='Switzerland 10th Decile', alpha=0.7)
        
        plt.title(f"Switzerland - {description}\\nTime Series ({int(earliest_year)}-{int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Share of the population', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        
        # Set x-axis
        if (latest_year - earliest_year) <= 10:
            years = list(range(int(earliest_year), int(latest_year) + 1))
            plt.xticks(years)
        else:
            year_start = int(earliest_year)
            year_end = int(latest_year)
            if year_start % 2 != 0:
                year_start -= 1
            years = list(range(year_start, year_end + 1, 2))
            plt.xticks(years)
        
        plt.xlim(earliest_year - 0.5, latest_year + 0.5)
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {indicator}: {str(e)[:100]}...")

def create_swiss_level3_indicators_by_eu_priority(swiss_level3_deciles, eu_priorities, output_dir):
    """Create one graph per EU Priority showing all Level 3 indicators across deciles"""
    print(f"\nCreating Level 3 All Raw Indicators graphs for {len(eu_priorities)} EU priorities...")
    
    # Create output directory for Level 3 All Raw Indicators
    level3_all_dir = os.path.join(output_dir, 'EWBI', 'Level_3_All_Raw_Indicators')
    os.makedirs(level3_all_dir, exist_ok=True)
    
    # Initialize list to collect all data for Excel export
    excel_data_list = []
    graphs_created = 0
    
    if swiss_level3_deciles.empty:
        print("Warning: No Level 3 decile data available")
        return
    
    # Convert Decile to numeric for sorting
    swiss_level3_deciles['Decile_num'] = swiss_level3_deciles['Decile'].astype(float)
    
    # Process each EU Priority
    for priority in eu_priorities:
        print(f"\nProcessing EU Priority: {priority}")
        
        # Filter data for this priority - with debug output
        priority_data = swiss_level3_deciles[swiss_level3_deciles['EU priority'] == priority].copy()
        
        print(f"  Priority data shape: {priority_data.shape}")
        print(f"  Unique indicators in data: {priority_data['Primary and raw data'].unique()}")
        
        if priority_data.empty:
            print(f"  Warning: No data available for {priority}")
            continue
        
        # Get all indicators for this priority that have at least some non-null values
        all_indicators = priority_data['Primary and raw data'].unique()
        valid_indicators_data = {}
        
        print(f"  Checking {len(all_indicators)} indicators...")
        for ind in all_indicators:
            if pd.notna(ind):
                ind_data = priority_data[priority_data['Primary and raw data'] == ind]
                non_null_count = ind_data['Value'].notna().sum()
                print(f"    {ind}: {non_null_count} non-null values")
                
                # Only include indicators with at least one non-null value
                if ind_data['Value'].notna().any():
                    # For each indicator, get the latest year available
                    latest_year_for_ind = ind_data['Year'].max()
                    ind_latest = ind_data[ind_data['Year'] == latest_year_for_ind].copy()
                    ind_latest = ind_latest.sort_values('Decile_num')
                    valid_indicators_data[ind] = {
                        'data': ind_latest,
                        'year': latest_year_for_ind
                    }
                    print(f"      -> Using year {int(latest_year_for_ind)}")
        
        if not valid_indicators_data:
            print(f"  Warning: No indicators with valid values for {priority}")
            continue
        
        indicators = sorted(valid_indicators_data.keys())
        print(f"  Found {len(indicators)} indicators with values")
        
        # Determine the most common latest year for the title
        years = [valid_indicators_data[ind]['year'] for ind in indicators]
        most_common_year = max(set(years), key=years.count)
        year_range = f"{int(min(years))}-{int(max(years))}" if len(set(years)) > 1 else str(int(most_common_year))
        
        # Create matplotlib figure
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            
            print(f"  Creating figure...")
            plt.figure(figsize=(14, 8))
            
            # Use a colormap for multiple indicators
            colors = cm.get_cmap('tab20', len(indicators))
            
            lines_plotted = 0
            # Plot each indicator
            for idx, indicator in enumerate(indicators):
                try:
                    indicator_data = valid_indicators_data[indicator]['data']
                    ind_year = valid_indicators_data[indicator]['year']
                    
                    # Filter out rows with NaN values for plotting
                    indicator_data = indicator_data[indicator_data['Value'].notna()]
                    
                    if indicator_data.empty:
                        print(f"    Skipping {indicator} - no valid values after filtering")
                        continue
                    
                    print(f"    Processing {indicator}: {len(indicator_data)} rows with values")
                    
                    # Get display name for indicator (no year in legend, it's in title)
                    display_name = get_display_name(indicator)
                    
                    # Collect data for Excel export (7 columns: visual_number, visual_name, year, filter, decile, value, unit)
                    for _, row in valid_indicators_data[indicator]['data'].iterrows():
                        excel_data_list.append({
                            'visual_number': np.nan,
                            'visual_name': priority,
                            'year': int(ind_year),
                            'filter': display_name,
                            'decile': int(row['Decile_num']),
                            'value': row['Value'] if pd.notna(row['Value']) else np.nan,
                            'unit': '%'  # Default to %, could be refined per indicator
                        })
                    
                    # Plot line with markers
                    plt.plot(indicator_data['Decile_num'], indicator_data['Value'],
                            marker='o', linewidth=2, markersize=6,
                            color=colors(idx), label=display_name, alpha=0.8)
                    lines_plotted += 1
                    print(f"      [OK] Plotted {display_name}")
                    
                except Exception as ind_error:
                    print(f"    ERROR processing indicator {indicator}: {ind_error}")
                    import traceback
                    traceback.print_exc()
            
            if lines_plotted == 0:
                print(f"  WARNING: No lines plotted for {priority}, skipping graph save")
                plt.close()
                continue
            
            print(f"  Customizing plot ({lines_plotted} lines plotted)...")
            
            # Customize the plot
            plt.title(f"Switzerland - {priority}\nAll Primary Indicators by Income Decile ({year_range})",
                     fontsize=16, fontweight='bold', color="#2C3E50", pad=20)
            plt.xlabel("Income Decile", fontsize=14, fontweight='bold')
            plt.ylabel("Indicator Value", fontsize=14, fontweight='bold')
            
            # Set x-axis ticks
            plt.xticks(range(1, 11), [str(i) for i in range(1, 11)])
            
            # Add grid and legend
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.tight_layout()
            
            # Create safe filename (short to avoid Windows MAX_PATH limit of 260 chars)
            # Use abbreviated names for long EU priorities
            priority_abbrev_map = {
                'Energy and Housing': 'Energy_Housing',
                'Equality': 'Equality',
                'Health and Animal Welfare': 'Health_Animal',
                'Intergenerational Fairness, Youth, Culture and Sport': 'Intergenerational',
                'Social Rights and Skills, Quality Jobs and Preparedness': 'Social_Rights'
            }
            safe_priority = priority_abbrev_map.get(priority, priority.replace(' ', '_').replace(',', '').replace('/', '_'))
            png_path = os.path.join(level3_all_dir, f"CH_{safe_priority}_level3_deciles.png")
            
            print(f"  Saving to: {os.path.basename(png_path)}")
            # Save PNG
            plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            graphs_created += 1
            print(f"  [SUCCESS] Saved: {os.path.basename(png_path)}")
            
        except Exception as e:
            print(f"  FATAL ERROR creating plot for {priority}: {e}")
            import traceback
            traceback.print_exc()
            plt.close()  # Ensure figure is closed even if error occurs
    
    # Summary
    print(f"\n[SUMMARY] Level 3: Created {graphs_created} / {len(eu_priorities)} EU priority graphs")
    
    # Export data to Excel
    if excel_data_list:
        try:
            excel_df = pd.DataFrame(excel_data_list)
            excel_path = os.path.join(level3_all_dir, 'switzerland_level3_indicators_by_decile.xlsx')
            excel_df.to_excel(excel_path, index=False, sheet_name='Level 3 Indicators')
            print(f"\nExported Excel file: {os.path.basename(excel_path)}")
            print(f"  Total records: {len(excel_df)}")
        except Exception as e:
            print(f"\nWarning: Could not export Excel file: {e}")
            print("  Trying CSV export as fallback...")
            try:
                csv_path = os.path.join(level3_all_dir, 'switzerland_level3_indicators_by_decile.csv')
                excel_df.to_csv(csv_path, index=False)
                print(f"  Exported CSV file: {os.path.basename(csv_path)}")
            except Exception as e2:
                print(f"  CSV export also failed: {e2}")

def main():
    """Main function to generate Switzerland comprehensive analysis"""
    
    print("Starting Switzerland Comprehensive Well-Being Analysis")
    print("=" * 60)
    
    # Load data
    try:
        unified_df = load_data()
        print(f"Loaded unified dataset with {len(unified_df)} records")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Set up output directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.abspath(os.path.join(current_dir, '..'))
    output_dir = os.path.abspath(os.path.join(report_dir, 'outputs', 'graphs'))
    
    # Create EWBI-specific subfolders
    ewbi_base = os.path.join(output_dir, 'EWBI')
    level1_dir = os.path.join(ewbi_base, 'Level_1_EWBI')
    level2_dir = os.path.join(ewbi_base, 'Level_2_EU_Priorities')
    level3_dir = os.path.join(ewbi_base, 'Level_3_Primary_Indicators')
    
    os.makedirs(level1_dir, exist_ok=True)
    os.makedirs(level2_dir, exist_ok=True)
    os.makedirs(level3_dir, exist_ok=True)
    
    print(f"Output directories created:")
    print(f"  - Level 1 EWBI: {level1_dir}")
    print(f"  - Level 2 EU Priorities: {level2_dir}")
    print(f"  - Level 3 Primary Indicators: {level3_dir}")
    
    # PART 1: Switzerland EWBI Analysis (Level 1)
    print("\n" + "="*40)
    print("PART 1: SWITZERLAND EWBI ANALYSIS")
    print("="*40)
    
    print("\n1.1. Processing Switzerland EWBI (Level 1)...")
    swiss_ewbi_overall, swiss_ewbi_deciles = prepare_swiss_ewbi_data(unified_df)
    
    if not swiss_ewbi_overall.empty:
        print("Creating Switzerland EWBI overall graph...")
        create_swiss_ewbi_overall_plot(swiss_ewbi_overall, level1_dir)
    else:
        print("Warning: No Switzerland EWBI overall data found")
    
    if not swiss_ewbi_deciles.empty:
        print("Creating Switzerland EWBI deciles graph...")
        create_swiss_ewbi_deciles_plot(swiss_ewbi_deciles, level1_dir)
    else:
        print("Warning: No Switzerland EWBI deciles data found")
    
    # PART 2: Switzerland EU Priorities Analysis (Level 2)
    print("\n" + "="*40)
    print("PART 2: SWITZERLAND EU PRIORITIES ANALYSIS")
    print("="*40)
    
    print("\n2.1. Processing Switzerland EU Priorities (Level 2)...")
    swiss_priorities_overall, swiss_priorities_deciles, eu_priorities = prepare_swiss_eu_priorities_data(unified_df)
    
    if not swiss_priorities_overall.empty or not swiss_priorities_deciles.empty:
        create_swiss_eu_priorities_plots(swiss_priorities_overall, swiss_priorities_deciles, 
                                        eu_priorities, level2_dir)
    else:
        print("Warning: No Switzerland EU priorities data found")
    
    # PART 2.5: Switzerland Level 3 All Raw Indicators by EU Priority
    print("\n" + "="*40)
    print("PART 2.5: SWITZERLAND LEVEL 3 ALL RAW INDICATORS BY EU PRIORITY")
    print("="*40)
    
    print("\n2.5.1. Processing Switzerland Level 3 data by EU Priority...")
    swiss_level3_deciles, level3_priorities = prepare_swiss_level3_data_by_eu_priority(unified_df)
    
    if not swiss_level3_deciles.empty:
        print("\n2.5.2. Creating Level 3 All Raw Indicators graphs by EU Priority...")
        create_swiss_level3_indicators_by_eu_priority(swiss_level3_deciles, level3_priorities, output_dir)
    else:
        print("Warning: No Switzerland Level 3 data found")
    
    # PART 3: Switzerland Primary Indicators Analysis (Level 5)
    print("\n" + "="*40)
    print("PART 3: SWITZERLAND PRIMARY INDICATORS ANALYSIS")
    print("="*40)
    
    print("\n3.1. Getting Level 5 indicators displayed in the app...")
    app_indicators = get_app_level5_indicators()
    
    if app_indicators:
        print(f"Found {len(app_indicators)} Level 5 indicators displayed in the app")
        
        # Filter data to only include app-displayed Level 5 indicators
        print(f"\n3.2. Filtering data for app-displayed indicators...")
        level5_df = unified_df[unified_df['Primary and raw data'].isin(app_indicators)].copy()
        print(f"Filtered data shape: {level5_df.shape}")
        
        # Prepare Switzerland Level 5 data
        print("\n3.3. Preparing Switzerland Level 5 data...")
        swiss_data, swiss_decile_1, swiss_decile_10 = prepare_swiss_data(level5_df)
        
        if not swiss_data.empty:
            swiss_indicators = set(swiss_data['Indicator'].unique())
            print(f"Switzerland has data for: {len(swiss_indicators)} indicators")
            
            print(f"\n3.4. Generating primary indicator time series graphs...")
            
            for i, indicator in enumerate(sorted(swiss_indicators), 1):
                description = get_display_name(indicator)
                print(f"\n3.4.{i}. Processing {description}...")
                
                try:
                    create_swiss_primary_indicator_plot(swiss_data, swiss_decile_1, swiss_decile_10, 
                                                       indicator, level3_dir)
                except Exception as e:
                    print(f"Error creating plot for {indicator}: {e}")
        else:
            print("Warning: Could not load Switzerland Level 5 data")
    else:
        print("Warning: No Level 5 indicators found in the app")
    
    # Summary
    print("\n" + "=" * 60)
    print(" SWITZERLAND COMPREHENSIVE ANALYSIS COMPLETED!")
    print(f" Output directory: {output_dir}")
    
    # List generated files (PNG only)
    generated_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.png'):
                rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                generated_files.append(rel_path)
    
    if generated_files:
        print(f"\nGenerated {len(generated_files)} PNG files:")
        for file in sorted(generated_files):
            print(f"  • {file}")
    else:
        print("\nWarning: No output files were generated")
    
    print(f"\nAnalysis Summary:")
    print(f"  • EWBI (Level 1): Overall + Deciles temporal analysis")
    print(f"  • EU Priorities (Level 2): {len(eu_priorities)} priorities × (Overall + Deciles)")
    if 'level3_priorities' in locals():
        print(f"  • Level 3 All Raw Indicators: {len(level3_priorities)} EU priorities × All primary indicators by decile")
        print(f"    EU Priorities processed: {level3_priorities}")
    print(f"  • Primary Indicators (Level 5): Time series analysis with deciles")
    print(f"  • Focus: Switzerland comprehensive well-being assessment")
    print(f"  • Color scheme: Switzerland yellow ({SWITZERLAND_COLOR})")
    print(f"  • Output format: PNG static files only")
    print(f"\n  📁 Level 3 graphs location: {os.path.join(output_dir, 'EWBI', 'Level_3_All_Raw_Indicators')}")

if __name__ == "__main__":
    main()