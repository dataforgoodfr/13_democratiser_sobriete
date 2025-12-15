"""
ewbi_treatment.py - Generate France vs EU-27 comparative analysis

This script creates comparative analysis graphs showing France's performance against
EU-27 averages, along with data extraction for further analysis.

Analysis includes:
- EWBI (Level 1) overall and by decile decomposition
- EU priorities (Level 2) comparative analysis
- Primary indicators time series comparison
- Data export to Excel for all levels
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
well_being_code_dir = os.path.abspath(os.path.join(reports_dir, '..', 'Well-being', 'code'))

sys.path.insert(0, current_dir)
sys.path.insert(0, shared_code_dir)
sys.path.insert(0, well_being_code_dir)

# Import shared utilities
from ewbi_data_loader import load_ewbi_unified_data, get_housing_energy_indicators
from visualization_utils import create_time_series_plot, save_plot

# Import variable mapping functions from Well-being pipeline
from variable_mapping import get_display_name

# Color palette for France vs EU-27 analysis
EU_27_COLOR = '#80b1d3'  # Blue for EU-27 (matches app.py)
FRANCE_COLOR = '#ffd558'  # Yellow for France

# Selected country for comparison
COMPARISON_COUNTRIES = {
    'FR': {'name': 'France', 'color': FRANCE_COLOR, 'profile': 'Large, high-income, Western European'}
}

def get_app_level5_indicators():
    """Get the Level 5 indicators that are actually available in the app's dropdown by replicating app.py logic"""
    # Use the shared EWBI data loader to get the unified dataset
    try:
        df = load_ewbi_unified_data()
    except Exception as e:
        print(f"Error loading EWBI unified data for app indicators: {e}")
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

def prepare_country_ewbi_data(unified_df, country_code):
    """Extract specific country EWBI (Level 1) data for overall and decile analysis"""
    # Get country EWBI overall data (Level 1, Decile='All')
    country_ewbi_overall = unified_df[
        (unified_df['Country'] == country_code) &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] == 'All')
    ].copy()
    
    # Get country EWBI by deciles (Level 1, all deciles except 'All')
    country_ewbi_deciles = unified_df[
        (unified_df['Country'] == country_code) &
        (unified_df['Level'] == 1) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna())
    ].copy()
    
    country_name = COMPARISON_COUNTRIES.get(country_code, {}).get('name', country_code)
    print(f"{country_name} EWBI overall data: {len(country_ewbi_overall)} records")
    print(f"{country_name} EWBI decile data: {len(country_ewbi_deciles)} records")
    
    return country_ewbi_overall, country_ewbi_deciles

def prepare_country_eu_priorities_data(unified_df, country_code):
    """Extract specific country EU priorities (Level 2) data for overall and decile analysis"""
    EU_PRIORITIES = [
        'Energy and Housing', 
        'Equality',
        'Health and Animal Welfare',
        'Intergenerational Fairness, Youth, Culture and Sport',
        'Social Rights and Skills, Quality Jobs and Preparedness'
    ]
    
    # Get country EU priorities overall data (Level 2, Decile='All')
    country_priorities_overall = unified_df[
        (unified_df['Country'] == country_code) &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] == 'All') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Get country EU priorities by deciles (Level 2, all deciles except 'All')
    country_priorities_deciles = unified_df[
        (unified_df['Country'] == country_code) &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] != 'All') &
        (unified_df['Decile'].notna()) &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    country_name = COMPARISON_COUNTRIES.get(country_code, {}).get('name', country_code)
    print(f"{country_name} EU priorities overall data: {len(country_priorities_overall)} records")
    print(f"{country_name} EU priorities decile data: {len(country_priorities_deciles)} records")
    
    return country_priorities_overall, country_priorities_deciles

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
        'Energy and Housing',
        'Equality', 
        'Health and Animal Welfare',
        'Intergenerational Fairness, Youth, Culture and Sport',
        'Social Rights and Skills, Quality Jobs and Preparedness'
    ]
    
    # Get EU-27 EU priorities overall data (Level 2, Decile='All') 
    # Use Geometric mean inter-decile for 'All Countries' to match app.py logic
    eu27_priorities_overall = unified_df[
        (unified_df['Country'] == 'All Countries') &
        (unified_df['Level'] == 2) &
        (unified_df['Decile'] == 'All') &
        (unified_df['Aggregation'] == 'Geometric mean inter-decile')
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

def create_eu_comparative_ewbi_overall_plot(eu27_data, country_data_dict, output_dir):
    """Create EWBI overall temporal comparison between EU-27 and selected countries"""
    if eu27_data.empty:
        print("Warning: No EU-27 EWBI overall data available")
        return
    
    # Sort EU-27 data by year
    eu27_data = eu27_data.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add EU-27 line (solid blue, thick)
    fig.add_trace(go.Scatter(
        x=eu27_data['Year'],
        y=eu27_data['Value'],
        mode='lines+markers',
        name='EU-27 Average',
        line=dict(color=EU_27_COLOR, width=4),
        marker=dict(color=EU_27_COLOR, size=10, symbol='circle'),
        hovertemplate='<b>EU-27 Average</b><br>' +
                     'Year: %{x}<br>' +
                     'Score: %{y:.3f}<extra></extra>'
    ))
    
    # Add country lines
    for country_code, country_info in COMPARISON_COUNTRIES.items():
        if country_code in country_data_dict and not country_data_dict[country_code].empty:
            country_data = country_data_dict[country_code].sort_values('Year')
            fig.add_trace(go.Scatter(
                x=country_data['Year'],
                y=country_data['Value'],
                mode='lines+markers',
                name=country_info['name'],
                line=dict(color=country_info['color'], width=3),
                marker=dict(color=country_info['color'], size=8, symbol='diamond'),
                hovertemplate=f'<b>{country_info["name"]}</b><br>' +
                             'Year: %{x}<br>' +
                             'Score: %{y:.3f}<extra></extra>'
            ))
    
    # Get time range from all available data
    all_years = list(eu27_data['Year'])
    for country_data in country_data_dict.values():
        if not country_data.empty:
            all_years.extend(list(country_data['Year']))
    
    if all_years:
        earliest_year = min(all_years)
        latest_year = max(all_years)
    else:
        earliest_year = latest_year = 2020
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"European Well-Being Index (EWBI) - Comparative Analysis<br><sub>France vs EU-27 ({int(earliest_year)}-{int(latest_year)})</sub>",
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
            dtick=1 if (latest_year - earliest_year) <= 10 else 2,
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
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        # Plot EU-27 line
        plt.plot(eu27_data['Year'], eu27_data['Value'], 
                color=EU_27_COLOR, linewidth=4, marker='o', 
                markersize=10, label='EU-27 Average', alpha=0.9)
        
        # Plot country lines
        for country_code, country_info in COMPARISON_COUNTRIES.items():
            if country_code in country_data_dict and not country_data_dict[country_code].empty:
                country_data = country_data_dict[country_code].sort_values('Year')
                plt.plot(country_data['Year'], country_data['Value'], 
                        color=country_info['color'], linewidth=3, marker='d', 
                        markersize=8, label=country_info['name'], alpha=0.9)
        
        plt.title(f"European Well-Being Index (EWBI) - Comparative Analysis\nFrance vs EU-27 ({int(earliest_year)}-{int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('EWBI Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        
        # Set x-axis ticks
        if (latest_year - earliest_year) <= 10:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1))
        else:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1, 2))
            
        plt.xlim(earliest_year - 0.5, latest_year + 0.5)
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, "france_vs_eu27_ewbi_overall.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for EU comparative EWBI overall: {str(e)[:100]}...")

def create_eu_comparative_priority_deciles_plot(eu27_priorities, country_priorities_dict, priority_name, latest_year, output_dir):
    """Create EU priority decile comparison with bars for countries and line for EU-27"""
    # Filter EU-27 decile data for this priority and latest year
    eu27_priority_deciles = eu27_priorities[
        (eu27_priorities['EU priority'] == priority_name) &
        (eu27_priorities['Year'] == latest_year) &
        (eu27_priorities['Decile'] != 'All')
    ].copy()
    
    if eu27_priority_deciles.empty:
        print(f"Warning: No EU-27 decile data for {priority_name}")
        return
    
    eu27_priority_deciles['Decile_num'] = pd.to_numeric(eu27_priority_deciles['Decile'], errors='coerce')
    eu27_priority_deciles = eu27_priority_deciles.dropna(subset=['Decile_num']).sort_values('Decile_num')
    
    # Filter country decile data for this priority and latest year
    country_decile_data = {}
    for country_code in COMPARISON_COUNTRIES.keys():
        if country_code in country_priorities_dict and not country_priorities_dict[country_code].empty:
            country_priority_deciles = country_priorities_dict[country_code][
                (country_priorities_dict[country_code]['EU priority'] == priority_name) &
                (country_priorities_dict[country_code]['Year'] == latest_year) &
                (country_priorities_dict[country_code]['Decile'] != 'All')
            ].copy()
            
            if not country_priority_deciles.empty:
                country_priority_deciles['Decile_num'] = pd.to_numeric(country_priority_deciles['Decile'], errors='coerce')
                country_priority_deciles = country_priority_deciles.dropna(subset=['Decile_num']).sort_values('Decile_num')
                country_decile_data[country_code] = country_priority_deciles
    
    # Save PNG with new design: bars for countries, line for EU-27
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 8))
        
        # Get decile positions
        x_positions = [int(d) for d in eu27_priority_deciles['Decile_num']]
        n_countries = len(country_decile_data)
        
        if n_countries == 0:
            print(f"Warning: No country decile data available for {priority_name}")
            return
            
        # Calculate bar width and positions for side-by-side bars
        bar_width = 0.25
        country_idx = 0
        
        # Create bars for each country (side by side)
        for country_code, country_info in COMPARISON_COUNTRIES.items():
            if country_code in country_decile_data:
                country_data = country_decile_data[country_code]
                
                # Calculate offset for side-by-side bars
                offset = (country_idx - (n_countries-1)/2) * bar_width
                bar_positions = [x + offset for x in x_positions]
                
                plt.bar(bar_positions, country_data['Value'],
                       width=bar_width, color=country_info['color'], alpha=0.7, 
                       edgecolor='none', label=country_info['name'])
                
                country_idx += 1
        
        # Add EU-27 as a line overlay
        plt.plot(x_positions, eu27_priority_deciles['Value'],
                color=EU_27_COLOR, linewidth=4, marker='o', 
                markersize=10, label='EU-27 Average', alpha=0.9, zorder=10)
        
        # Create safe filename
        safe_priority = priority_name.lower().replace(' ', '_').replace(',', '').replace('&', 'and').replace('-', '_')
        if len(safe_priority) > 30:
            safe_priority = safe_priority[:30]
        
        plt.title(f"{priority_name} by Income Decile\nFrance vs EU-27 ({int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Income Decile', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(x_positions, [str(pos) for pos in x_positions])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, f"france_vs_eu27_{safe_priority}_deciles.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {priority_name} deciles: {str(e)[:100]}...")

def create_eu_comparative_ewbi_deciles_plot(eu27_deciles, country_deciles_dict, output_dir):
    """Create EWBI decile comparison between EU-27 and selected countries - latest year only"""
    if eu27_deciles.empty:
        print("Warning: No EU-27 EWBI decile data available")
        return
    
    # Get latest year available across all data
    all_years = list(eu27_deciles['Year'])
    for country_data in country_deciles_dict.values():
        if not country_data.empty:
            all_years.extend(list(country_data['Year']))
    
    if not all_years:
        print("Warning: No decile data available")
        return
        
    latest_year = max(all_years)
    
    # Filter EU-27 data for latest year
    eu27_latest = eu27_deciles[eu27_deciles['Year'] == latest_year].copy()
    eu27_latest['Decile_num'] = pd.to_numeric(eu27_latest['Decile'], errors='coerce')
    eu27_latest = eu27_latest.dropna(subset=['Decile_num']).sort_values('Decile_num')
    
    if eu27_latest.empty:
        print("Warning: No EU-27 decile data for latest year")
        return
    
    # Create the plot
    fig = go.Figure()
    
    # Add EU-27 bars
    x_labels = [str(int(d)) for d in eu27_latest['Decile_num']]
    fig.add_trace(go.Bar(
        x=x_labels,
        y=eu27_latest['Value'],
        name='EU-27',
        marker=dict(color=EU_27_COLOR, opacity=0.7),
        text=[f'{v:.3f}' for v in eu27_latest['Value']],
        textposition='auto',
        hovertemplate='<b>EU-27 - Decile %{x}</b><br>' +
                     'Score: %{y:.3f}<extra></extra>'
    ))
    
    # Add country data as lines overlaid on bars
    for country_code, country_info in COMPARISON_COUNTRIES.items():
        if country_code in country_deciles_dict and not country_deciles_dict[country_code].empty:
            country_data = country_deciles_dict[country_code]
            country_latest = country_data[country_data['Year'] == latest_year].copy()
            country_latest['Decile_num'] = pd.to_numeric(country_latest['Decile'], errors='coerce')
            country_latest = country_latest.dropna(subset=['Decile_num']).sort_values('Decile_num')
            
            if not country_latest.empty:
                fig.add_trace(go.Scatter(
                    x=[str(int(d)) for d in country_latest['Decile_num']],
                    y=country_latest['Value'],
                    mode='lines+markers',
                    name=country_info['name'],
                    line=dict(color=country_info['color'], width=3),
                    marker=dict(color=country_info['color'], size=8, symbol='diamond'),
                    hovertemplate=f'<b>{country_info["name"]} - Decile %{{x}}</b><br>' +
                                 'Score: %{y:.3f}<extra></extra>'
                ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"European Well-Being Index (EWBI) by Income Decile<br><sub>France vs EU-27 ({int(latest_year)})</sub>",
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
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 8))
        
        # Get decile positions
        x_positions = [int(d) for d in eu27_latest['Decile_num']]
        n_countries = len([k for k in COMPARISON_COUNTRIES.keys() if k in country_deciles_dict and not country_deciles_dict[k].empty])
        
        if n_countries == 0:
            print("Warning: No country decile data available")
            return
            
        # Calculate bar width and positions for side-by-side bars
        bar_width = 0.25
        country_idx = 0
        
        # Create bars for each country (side by side)
        for country_code, country_info in COMPARISON_COUNTRIES.items():
            if country_code in country_deciles_dict and not country_deciles_dict[country_code].empty:
                country_data = country_deciles_dict[country_code]
                country_latest = country_data[country_data['Year'] == latest_year].copy()
                country_latest['Decile_num'] = pd.to_numeric(country_latest['Decile'], errors='coerce')
                country_latest = country_latest.dropna(subset=['Decile_num']).sort_values('Decile_num')
                
                if not country_latest.empty:
                    # Calculate offset for side-by-side bars
                    offset = (country_idx - (n_countries-1)/2) * bar_width
                    bar_positions = [x + offset for x in x_positions]
                    
                    plt.bar(bar_positions, country_latest['Value'],
                           width=bar_width, color=country_info['color'], alpha=0.7, 
                           edgecolor='none', label=country_info['name'])
                    
                    country_idx += 1
        
        # Add EU-27 as a line overlay
        plt.plot(x_positions, eu27_latest['Value'],
                color=EU_27_COLOR, linewidth=4, marker='o', 
                markersize=10, label='EU-27 Average', alpha=0.9, zorder=10)
        
        plt.title(f"European Well-Being Index (EWBI) by Income Decile\\nFrance vs EU-27 ({int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Income Decile', fontsize=14)
        plt.ylabel('EWBI Score', fontsize=14)
        plt.xticks(x_positions, [str(pos) for pos in x_positions])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, "france_vs_eu27_ewbi_deciles.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for EU comparative EWBI deciles: {str(e)[:100]}...")

def create_eu_comparative_eu_priorities_plots(eu27_priorities, country_priorities_dict, eu_priorities, output_dir, eu27_priorities_deciles=None, country_priorities_deciles_dict=None, latest_year=None):
    """Create EU priority comparative plots between EU-27 and selected countries"""
    print(f"Creating comparative graphs for {len(eu_priorities)} EU priorities...")
    
    for priority in eu_priorities:
        print(f"Creating plots for EU priority: {priority}")
        
        # Filter EU-27 data for this priority
        eu27_priority_data = eu27_priorities[
            eu27_priorities['EU priority'] == priority
        ].copy()
        
        # Filter country data for this priority
        country_priority_data = {}
        for country_code in COMPARISON_COUNTRIES.keys():
            if country_code in country_priorities_dict:
                country_data = country_priorities_dict[country_code]
                country_priority_data[country_code] = country_data[
                    country_data['EU priority'] == priority
                ].copy()
            else:
                country_priority_data[country_code] = pd.DataFrame()
        
        # Create overall comparison plot
        if not eu27_priority_data.empty:
            create_eu_comparative_priority_overall_plot(
                eu27_priority_data, country_priority_data, priority, output_dir
            )
        else:
            print(f"Warning: No EU-27 data for {priority} overall")
            
        # Create decile comparison plot for this priority (using decile data)
        if (eu27_priorities_deciles is not None and 
            country_priorities_deciles_dict is not None and 
            latest_year is not None):
            create_eu_comparative_priority_deciles_plot(
                eu27_priorities_deciles, country_priorities_deciles_dict, 
                priority, latest_year, output_dir
            )

def create_eu_comparative_priority_overall_plot(eu27_data, country_data_dict, priority_name, output_dir):
    """Create overall temporal comparison for a specific EU priority"""
    # Sort EU-27 data by year
    eu27_data = eu27_data.sort_values('Year')
    
    # Create the plot
    fig = go.Figure()
    
    # Add EU-27 line
    fig.add_trace(go.Scatter(
        x=eu27_data['Year'],
        y=eu27_data['Value'],
        mode='lines+markers',
        name='EU-27 Average',
        line=dict(color=EU_27_COLOR, width=4),
        marker=dict(color=EU_27_COLOR, size=10, symbol='circle'),
        hovertemplate=f'<b>EU-27 {priority_name}</b><br>' +
                     'Year: %{x}<br>' +
                     'Score: %{y:.3f}<extra></extra>'
    ))
    
    # Add country lines
    for country_code, country_info in COMPARISON_COUNTRIES.items():
        if country_code in country_data_dict and not country_data_dict[country_code].empty:
            country_data = country_data_dict[country_code].sort_values('Year')
            fig.add_trace(go.Scatter(
                x=country_data['Year'],
                y=country_data['Value'],
                mode='lines+markers',
                name=country_info['name'],
                line=dict(color=country_info['color'], width=3),
                marker=dict(color=country_info['color'], size=8, symbol='diamond'),
                hovertemplate=f'<b>{country_info["name"]} {priority_name}</b><br>' +
                             'Year: %{x}<br>' +
                             'Score: %{y:.3f}<extra></extra>'
            ))
    
    # Get time range
    all_years = list(eu27_data['Year'])
    for country_data in country_data_dict.values():
        if not country_data.empty:
            all_years.extend(list(country_data['Year']))
    
    if all_years:
        earliest_year = min(all_years)
        latest_year = max(all_years)
    else:
        earliest_year = latest_year = 2020
    
    # Create safe filename
    safe_priority = priority_name.lower().replace(' ', '_').replace(',', '').replace('&', 'and').replace('-', '_')
    if len(safe_priority) > 30:
        safe_priority = safe_priority[:30]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{priority_name} - Comparative Analysis<br><sub>France vs EU-27 ({int(earliest_year)}-{int(latest_year)})</sub>",
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
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        # Plot EU-27 line
        plt.plot(eu27_data['Year'], eu27_data['Value'], 
                color=EU_27_COLOR, linewidth=4, marker='o', 
                markersize=10, label='EU-27 Average', alpha=0.9)
        
        # Plot country lines
        for country_code, country_info in COMPARISON_COUNTRIES.items():
            if country_code in country_data_dict and not country_data_dict[country_code].empty:
                country_data = country_data_dict[country_code].sort_values('Year')
                plt.plot(country_data['Year'], country_data['Value'], 
                        color=country_info['color'], linewidth=3, marker='d', 
                        markersize=8, label=country_info['name'], alpha=0.9)
        
        plt.title(f"{priority_name} - Comparative Analysis\nFrance vs EU-27 ({int(earliest_year)}-{int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        
        # Set x-axis ticks
        if (latest_year - earliest_year) <= 10:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1))
        else:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1, 2))
            
        plt.xlim(earliest_year - 0.5, latest_year + 0.5)
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, f"france_vs_eu27_{safe_priority}_overall.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for EU comparative priority overall: {str(e)[:100]}...")
        print(f"Warning: Could not save PNG for {priority_name} overall plot: {str(e)[:100]}...")

def prepare_comparative_data(unified_df):
    """Extract and prepare comparative data for all countries and EU-27 including decile breakdown"""
    # Get Level 5 data for countries and EU-27 - include all deciles not just 'All'
    comparative_data = unified_df[
        (unified_df['Level'] == 5)
        # Do not filter by Decile to include both 'All' and individual deciles
    ].copy()
    
    # Rename columns for consistency
    comparative_data = comparative_data.rename(columns={'Primary and raw data': 'Indicator'})
    
    return comparative_data

def create_eu_comparative_primary_indicator_plot(comparative_data, indicator, output_dir):
    """Create a comparative time series plot for a specific primary indicator"""
    
    # Filter data for the specific indicator AND Decile == 'All' (to match app.py filtering)
    indicator_data = comparative_data[
        (comparative_data['Indicator'] == indicator) &
        (comparative_data['Decile'] == 'All')
    ].copy()
    
    if indicator_data.empty:
        print(f"Warning: No data available for {indicator}")
        return
    
    # Get EU-27 data (use Population-weighted average for Level 5, matching app.py)
    eu27_data = indicator_data[
        (indicator_data['Country'] == 'All Countries') &
        (indicator_data['Aggregation'].isin(['Median across countries', 'Population-weighted average']))
    ].copy().sort_values('Year')
    
    # Get country data
    country_data_dict = {}
    for country_code in COMPARISON_COUNTRIES.keys():
        country_data = indicator_data[
            indicator_data['Country'] == country_code
        ].copy().sort_values('Year')
        country_data_dict[country_code] = country_data
    
    # Check if we have any data
    has_data = not eu27_data.empty or any(not df.empty for df in country_data_dict.values())
    if not has_data:
        print(f"Warning: No data available for any entity for {indicator}")
        return
    
    # Find the time range
    all_years = []
    if not eu27_data.empty:
        all_years.extend(list(eu27_data['Year']))
    for country_data in country_data_dict.values():
        if not country_data.empty:
            all_years.extend(list(country_data['Year']))
    
    if not all_years:
        print(f"Warning: No year data available for {indicator}")
        return
    
    earliest_year = min(all_years)
    latest_year = max(all_years)
    
    # Create the plot
    fig = go.Figure()
    
    # Add EU-27 line if available
    if not eu27_data.empty:
        fig.add_trace(go.Scatter(
            x=eu27_data['Year'],
            y=eu27_data['Value'],
            mode='lines+markers',
            name='EU-27 Average',
            line=dict(color=EU_27_COLOR, width=4),
            marker=dict(color=EU_27_COLOR, size=10, symbol='circle'),
            hovertemplate='<b>EU-27 Average</b><br>' +
                         'Year: %{x}<br>' +
                         'Value: %{y:.3f}<extra></extra>'
        ))
    
    # Add country lines
    for country_code, country_info in COMPARISON_COUNTRIES.items():
        if not country_data_dict[country_code].empty:
            country_data = country_data_dict[country_code]
            fig.add_trace(go.Scatter(
                x=country_data['Year'],
                y=country_data['Value'],
                mode='lines+markers',
                name=country_info['name'],
                line=dict(color=country_info['color'], width=3),
                marker=dict(color=country_info['color'], size=8, symbol='diamond'),
                hovertemplate=f'<b>{country_info["name"]}</b><br>' +
                             'Year: %{x}<br>' +
                             'Value: %{y:.3f}<extra></extra>'
            ))
    
    # Get indicator description
    description = get_display_name(indicator)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{description} - Comparative Analysis<br><sub>France vs EU-27 ({int(earliest_year)}-{int(latest_year)})</sub>",
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
    
    # Save PNG only
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        # Plot EU-27 line if available
        if not eu27_data.empty:
            plt.plot(eu27_data['Year'], eu27_data['Value'], 
                    color=EU_27_COLOR, linewidth=4, marker='o', 
                    markersize=10, label='EU-27 Average', alpha=0.9)
        
        # Plot country lines
        for country_code, country_info in COMPARISON_COUNTRIES.items():
            if not country_data_dict[country_code].empty:
                country_data = country_data_dict[country_code]
                plt.plot(country_data['Year'], country_data['Value'], 
                        color=country_info['color'], linewidth=3, marker='d', 
                        markersize=8, label=country_info['name'], alpha=0.9)
        
        plt.title(f"{description} - Comparative Analysis\nFrance vs EU-27 ({int(earliest_year)}-{int(latest_year)})", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Share of the population', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='best')
        
        # Set x-axis ticks
        if (latest_year - earliest_year) <= 10:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1))
        else:
            plt.xticks(range(int(earliest_year), int(latest_year) + 1, 2))
            
        plt.xlim(earliest_year - 0.5, latest_year + 0.5)
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, f"france_vs_eu27_{safe_indicator}.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {indicator}: {str(e)[:100]}...")

def create_eu_comparative_primary_indicator_deciles_plot(comparative_data, indicator, latest_year, output_dir):
    """Create primary indicator decile comparison with bars for countries and line for EU-27"""
    
    # Filter data for the specific indicator, latest year, and decile data
    indicator_decile_data = comparative_data[
        (comparative_data['Indicator'] == indicator) &
        (comparative_data['Year'] == latest_year) &
        (comparative_data['Decile'] != 'All')
    ].copy()
    
    if indicator_decile_data.empty:
        print(f"Warning: No decile data available for {indicator}")
        return
    
    # Get EU-27 decile data
    eu27_decile_data = indicator_decile_data[
        indicator_decile_data['Country'] == 'All Countries'
    ].copy()
    
    if eu27_decile_data.empty:
        print(f"Warning: No EU-27 decile data for {indicator}")
        return
    
    eu27_decile_data['Decile_num'] = pd.to_numeric(eu27_decile_data['Decile'], errors='coerce')
    eu27_decile_data = eu27_decile_data.dropna(subset=['Decile_num']).sort_values('Decile_num')
    
    # Get country decile data
    country_decile_data = {}
    for country_code in COMPARISON_COUNTRIES.keys():
        country_data = indicator_decile_data[
            indicator_decile_data['Country'] == country_code
        ].copy()
        
        if not country_data.empty:
            country_data['Decile_num'] = pd.to_numeric(country_data['Decile'], errors='coerce')
            country_data = country_data.dropna(subset=['Decile_num']).sort_values('Decile_num')
            country_decile_data[country_code] = country_data
    
    # Save PNG with new design: bars for countries, line for EU-27
    try:
        import matplotlib.pyplot as plt
        from variable_mapping import get_display_name
        
        plt.figure(figsize=(14, 8))
        
        # Get decile positions
        x_positions = [int(d) for d in eu27_decile_data['Decile_num']]
        n_countries = len(country_decile_data)
        
        if n_countries == 0:
            print(f"Warning: No country decile data available for {indicator}")
            return
            
        # Calculate bar width and positions for side-by-side bars
        bar_width = 0.25
        country_idx = 0
        
        # Create bars for each country (side by side)
        for country_code, country_info in COMPARISON_COUNTRIES.items():
            if country_code in country_decile_data:
                country_data = country_decile_data[country_code]
                
                # Calculate offset for side-by-side bars
                offset = (country_idx - (n_countries-1)/2) * bar_width
                bar_positions = [x + offset for x in x_positions]
                
                plt.bar(bar_positions, country_data['Value'],
                       width=bar_width, color=country_info['color'], alpha=0.7, 
                       edgecolor='none', label=country_info['name'])
                
                country_idx += 1
        
        # Add EU-27 as a line overlay
        plt.plot(x_positions, eu27_decile_data['Value'],
                color=EU_27_COLOR, linewidth=4, marker='o', 
                markersize=10, label='EU-27 Average', alpha=0.9, zorder=10)
        
        # Get indicator description
        description = get_display_name(indicator)
        
        plt.title(f"{description} by Income Decile\\nFrance vs EU-27 ({int(latest_year)})", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Income Decile', fontsize=14)
        plt.ylabel('Share of the population (%)', fontsize=14)
        plt.xticks(x_positions, [str(pos) for pos in x_positions])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()
        
        # Create safe filename
        safe_indicator = indicator.replace('-', '_').replace('/', '_').replace(' ', '_').replace(',', '')
        if len(safe_indicator) > 40:
            safe_indicator = safe_indicator[:40]
        
        png_path = os.path.join(output_dir, f"france_vs_eu27_{safe_indicator}_deciles.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved PNG: {png_path}")
        
    except Exception as e:
        print(f"Warning: Could not save PNG for {indicator} deciles: {str(e)[:100]}...")

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
    excel_path = os.path.join(output_dir, "france_vs_eu27_ewbi_level1.xlsx")
    
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
    priorities_overall_fr = unified_df[
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
        (unified_df['Aggregation'] == 'Geometric mean inter-decile') &
        (unified_df['EU priority'].isin(EU_PRIORITIES))
    ].copy()
    
    # Combine overall data
    priorities_overall = pd.concat([priorities_overall_fr, priorities_overall_eu27], ignore_index=True)
    
    # Get EU Priorities Level 2 decile data for France
    priorities_deciles_fr = unified_df[
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
    priorities_deciles = pd.concat([priorities_deciles_fr, priorities_deciles_eu27], ignore_index=True)
    
    # Process overall data
    for _, row in priorities_overall.iterrows():
        country_name = 'France' if row['Country'] == 'FR' else 'EU-27'
        all_data_records.append({
            'indicator_code': f"L2_{row['EU priority']}",
            'indicator_name': row['EU priority'],
            'year': int(row['Year']),
            'geo': country_name,
            'decile': 'Average',
            'value': row['Value']
        })
    
    # Process decile data
    for _, row in priorities_deciles.iterrows():
        country_name = 'France' if row['Country'] == 'FR' else 'EU-27'
        decile_num = int(float(row['Decile']))  # Handle float to int conversion
        decile_name = f"{decile_num}{'st' if decile_num == 1 else ('nd' if decile_num == 2 else ('rd' if decile_num == 3 else 'th'))}"
        all_data_records.append({
            'indicator_code': f"L2_{row['EU priority']}",
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
    excel_path = os.path.join(output_dir, "france_vs_eu27_eu_priorities_level2.xlsx")
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main data sheet
            export_df.to_excel(writer, sheet_name='EU_Priorities_Level2_Data', index=False)
            
            # Create a summary sheet with priority descriptions
            priority_summary = []
            for priority in EU_PRIORITIES:
                priority_summary.append({
                    'indicator_code': f"L2_{priority}",
                    'indicator_name': priority
                })
            
            summary_df = pd.DataFrame(priority_summary)
            summary_df.to_excel(writer, sheet_name='Priority_Definitions', index=False)
            
        print(f"Exported EU Priorities Level 2 data to Excel: {excel_path}")
        print(f"Total EU Priorities records: {len(export_df)}")
        print(f"Priorities: {len(EU_PRIORITIES)}")
        print(f"Countries: {export_df['geo'].nunique()}")
        print(f"Years: {export_df['year'].min()}-{export_df['year'].max()}")
        
    except Exception as e:
        print(f"Error exporting EU Priorities Level 2 to Excel: {e}")

def export_primary_indicators_to_excel(comparative_data, all_indicators, output_dir):
    """Export Primary Indicators Level 5 data (France + EU-27) to Excel"""
    
    all_data_records = []
    
    # Process each indicator
    for indicator in sorted(all_indicators):
        # Get indicator description
        description = get_display_name(indicator)
        
        # Filter data for this indicator - all deciles
        indicator_data = comparative_data[
            comparative_data['Indicator'] == indicator
        ].copy()
        
        if indicator_data.empty:
            continue
        
        # Process France data
        france_data = indicator_data[indicator_data['Country'] == 'FR']
        for _, row in france_data.iterrows():
            decile_str = 'Average' if row['Decile'] == 'All' else str(int(float(row['Decile']))) + ('st' if int(float(row['Decile'])) == 1 else ('nd' if int(float(row['Decile'])) == 2 else ('rd' if int(float(row['Decile'])) == 3 else 'th')))
            all_data_records.append({
                'indicator_code': indicator,
                'indicator_name': description,
                'year': int(row['Year']),
                'geo': 'France',
                'decile': decile_str,
                'value': row['Value']
            })
        
        # Process EU-27 data
        eu27_data = indicator_data[
            (indicator_data['Country'] == 'All Countries') &
            (indicator_data['Aggregation'].isin(['Median across countries', 'Population-weighted average']))
        ]
        for _, row in eu27_data.iterrows():
            decile_str = 'Average' if row['Decile'] == 'All' else str(int(float(row['Decile']))) + ('st' if int(float(row['Decile'])) == 1 else ('nd' if int(float(row['Decile'])) == 2 else ('rd' if int(float(row['Decile'])) == 3 else 'th')))
            all_data_records.append({
                'indicator_code': indicator,
                'indicator_name': description,
                'year': int(row['Year']),
                'geo': 'EU-27',
                'decile': decile_str,
                'value': row['Value']
            })
    
    # Create DataFrame
    export_df = pd.DataFrame(all_data_records)
    
    if export_df.empty:
        print("Warning: No primary indicators data to export")
        return
    
    # Sort by indicator code, geo, decile, year
    export_df = export_df.sort_values(['indicator_code', 'geo', 'decile', 'year'])
    
    # Export to Excel
    excel_path = os.path.join(output_dir, "france_vs_eu27_primary_indicators.xlsx")
    
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
            
        print(f"Exported all primary indicators data to Excel: {excel_path}")
        print(f"Total records: {len(export_df)}")
        print(f"Indicators: {len(all_indicators)}")
        print(f"Countries: {export_df['geo'].nunique()}")
        print(f"Years: {export_df['year'].min()}-{export_df['year'].max()}")
        
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        # Fallback to CSV
        csv_path = os.path.join(output_dir, "france_vs_eu27_primary_indicators.csv")
        export_df.to_csv(csv_path, index=False)
        print(f"Exported to CSV instead: {csv_path}")

def main():
    """Main function to generate France vs EU-27 comparative analysis"""
    
    print("Starting France vs EU-27 Comparative Well-Being Analysis")
    print("=" * 60)
    print("Selected country for comparison:")
    for code, info in COMPARISON_COUNTRIES.items():
        print(f"  • {info['name']} ({code}): {info['profile']}")
    print()
    
    # Load data
    try:
        unified_df = load_data()
        print(f"Loaded unified dataset with {len(unified_df)} records")
    except FileNotFoundError as e:
        print(f"Error: Could not load data: {e}")
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
    
    # Create EWBI tables folder structure
    tables_dir = os.path.abspath(os.path.join(report_dir, 'outputs', 'tables'))
    tables_ewbi_dir = os.path.join(tables_dir, 'EWBI')
    os.makedirs(tables_ewbi_dir, exist_ok=True)
    
    print(f"Output directories created:")
    print(f"  - Level 1 EWBI: {level1_dir}")
    print(f"  - Level 2 EU Priorities: {level2_dir}")
    print(f"  - Level 3 Primary Indicators: {level3_dir}")
    print(f"  - Tables: {tables_ewbi_dir}")
    
    # PART 1: EU-27 vs Countries EWBI Analysis (Level 1)
    print("\n" + "="*50)
    print("PART 1: EU-27 vs COUNTRIES EWBI ANALYSIS")
    print("="*50)
    
    print("\n1.1. Processing EU-27 EWBI data...")
    eu27_ewbi_overall, eu27_ewbi_deciles = prepare_eu27_ewbi_data(unified_df)
    
    print("\n1.2. Processing country EWBI data...")
    country_ewbi_overall_dict = {}
    country_ewbi_deciles_dict = {}
    
    for country_code in COMPARISON_COUNTRIES.keys():
        overall, deciles = prepare_country_ewbi_data(unified_df, country_code)
        country_ewbi_overall_dict[country_code] = overall
        country_ewbi_deciles_dict[country_code] = deciles
    
    # Create EWBI comparative plots
    print("\n1.3. Creating EWBI comparative graphs...")
    if not eu27_ewbi_overall.empty:
        create_eu_comparative_ewbi_overall_plot(eu27_ewbi_overall, country_ewbi_overall_dict, level1_dir)
    
    if not eu27_ewbi_deciles.empty:
        create_eu_comparative_ewbi_deciles_plot(eu27_ewbi_deciles, country_ewbi_deciles_dict, level1_dir)
    
    # 1.4. Export EWBI Level 1 data to Excel
    print("\n1.4. Exporting EWBI Level 1 data to Excel...")
    try:
        export_ewbi_level1_to_excel(unified_df, tables_ewbi_dir)
    except Exception as e:
        print(f"Error exporting EWBI Level 1: {e}")
    
    # PART 2: EU-27 vs Countries EU Priorities Analysis (Level 2)
    print("\n" + "="*50)
    print("PART 2: FRANCE vs EU-27 EU PRIORITIES ANALYSIS")
    print("="*50)
    
    print("\n2.1. Processing EU-27 EU priorities data...")
    eu27_priorities_overall, eu27_priorities_deciles, eu_priorities = prepare_eu27_eu_priorities_data(unified_df)
    
    print("\n2.2. Processing country EU priorities data...")
    country_priorities_overall_dict = {}
    country_priorities_deciles_dict = {}
    
    for country_code in COMPARISON_COUNTRIES.keys():
        overall, deciles = prepare_country_eu_priorities_data(unified_df, country_code)
        country_priorities_overall_dict[country_code] = overall
        country_priorities_deciles_dict[country_code] = deciles
    
    # Get latest year for decile analysis
    latest_year = max(eu27_priorities_overall['Year']) if not eu27_priorities_overall.empty else unified_df['Year'].max()
    
    # Create EU priorities comparative plots
    print("\n2.3. Creating EU priorities comparative graphs...")
    if not eu27_priorities_overall.empty:
        create_eu_comparative_eu_priorities_plots(
            eu27_priorities_overall, country_priorities_overall_dict, eu_priorities, level2_dir,
            eu27_priorities_deciles, country_priorities_deciles_dict, latest_year
        )
    
    # 2.4. Export EU Priorities Level 2 data to Excel
    print("\n2.4. Exporting EU Priorities Level 2 data to Excel...")
    try:
        export_eu_priorities_to_excel(unified_df, tables_ewbi_dir)
    except Exception as e:
        print(f"Error exporting EU Priorities Level 2: {e}")
    
    # PART 3: EU-27 vs Countries Primary Indicators Analysis (Level 5)
    print("\n" + "="*50)
    print("PART 3: FRANCE vs EU-27 PRIMARY INDICATORS ANALYSIS")
    print("="*50)
    
    print("\n3.1. Getting Level 5 indicators displayed in the app...")
    app_indicators = get_app_level5_indicators()
    
    if app_indicators:
        print(f"\n3.2. Found {len(app_indicators)} Level 5 indicators displayed in the app")
        
        # Filter data to only include app-displayed Level 5 indicators
        print(f"\n3.3. Filtering data for app-displayed indicators...")
        level5_df = unified_df[unified_df['Primary and raw data'].isin(app_indicators)].copy()
        print(f"Filtered data shape: {level5_df.shape}")
        
        # Prepare comparative data
        print("\n3.4. Preparing comparative data...")
        comparative_data = prepare_comparative_data(level5_df)
        
        if not comparative_data.empty:
            available_indicators = set(comparative_data['Indicator'].unique())
            print(f"Available indicators for comparison: {len(available_indicators)}")
            
            print(f"\n3.5. Generating primary indicator comparative graphs...")
            
            # Get latest year for decile analysis
            latest_year_level5 = comparative_data['Year'].max()
            
            for i, indicator in enumerate(sorted(available_indicators), 1):
                description = get_display_name(indicator)
                print(f"\n3.5.{i}. Processing {description}...")
                
                try:
                    # Create overall time series plot
                    create_eu_comparative_primary_indicator_plot(comparative_data, indicator, level3_dir)
                    
                    # Create decile comparison plot  
                    create_eu_comparative_primary_indicator_deciles_plot(comparative_data, indicator, latest_year_level5, level3_dir)
                except Exception as e:
                    print(f"Error creating plot for {indicator}: {e}")
            
            # 3.6. Export Primary Indicators Level 5 data to Excel
            print(f"\n3.6. Exporting Primary Indicators Level 5 data to Excel...")
            try:
                export_primary_indicators_to_excel(comparative_data, available_indicators, tables_ewbi_dir)
            except Exception as e:
                print(f"Error exporting Primary Indicators Level 5: {e}")
        else:
            print("Warning: Could not load comparative Level 5 data")
    else:
        print("Warning: No Level 5 indicators found in the app")
    
    # Summary
    print("\n" + "=" * 60)
    print(" FRANCE vs EU-27 COMPARATIVE ANALYSIS COMPLETED!")
    print(f" Output directory: {output_dir}")
    print(f" Tables directory: {tables_ewbi_dir}")
    
    # List generated files (PNG and Excel)
    generated_pngs = []
    generated_excels = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), output_dir)
            if file.endswith('.png'):
                generated_pngs.append(rel_path)
    
    for root, dirs, files in os.walk(tables_ewbi_dir):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), tables_ewbi_dir)
            if file.endswith(('.xlsx', '.csv')):
                generated_excels.append(rel_path)
    
    if generated_pngs:
        print(f"\nGenerated {len(generated_pngs)} PNG files:")
        for file in sorted(generated_pngs):
            print(f"  • {file}")
    else:
        print("\nWarning: No PNG output files were generated")
    
    if generated_excels:
        print(f"\nGenerated {len(generated_excels)} data export files:")
        for file in sorted(generated_excels):
            print(f"  • {file}")
    else:
        print("\nWarning: No data export files were generated")
    
    print(f"\nAnalysis Summary:")
    print(f"  • EWBI (Level 1): Overall timeline + decile comparison + data export")
    print(f"  • EU Priorities (Level 2): {len(eu_priorities)} priorities with overall timeline + decile comparisons + data export")
    print(f"  • Primary Indicators (Level 3): Individual indicators with overall timeline + decile comparisons + data export")
    print(f"  • Country: {COMPARISON_COUNTRIES['FR']['name']}")
    print(f"  • Colors: EU-27 blue ({EU_27_COLOR}), France yellow ({FRANCE_COLOR})")
    print(f"  • Output formats: PNG graphs + Excel/CSV data exports")


if __name__ == "__main__":
    main()