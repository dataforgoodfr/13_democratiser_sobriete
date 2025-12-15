"""
Visualization Utilities - Shared plotting functions for Well-being Reports

This module provides standardized visualization functions that can be used
across all reports in the Well-being Reports directory.

Features:
- Consistent color schemes and styling
- Standard plot types for EWBI data
- Export functions for HTML and PNG formats
- Swiss and EU color themes
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


# Color schemes
SWITZERLAND_COLOR = '#ffd558'  # Yellow for Switzerland
EU_27_COLOR = '#80b1d3'  # Blue for EU-27
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']

# Standard plot styling
PLOT_STYLE = {
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white',
    'font': dict(family="Arial, sans-serif"),
    'title_font': dict(size=16, color="#2C3E50", family="Arial, sans-serif"),
    'axis_title_font': dict(size=14, family="Arial, sans-serif"),
    'axis_tick_font': dict(size=12, family="Arial, sans-serif"),
    'legend_font': dict(size=12),
    'grid_color': 'lightgray',
    'margin': dict(t=80, b=60, l=60, r=40),
    'width': 1000,
    'height': 600
}


def get_display_name(indicator_code):
    """
    Convert indicator codes to display names
    This function should be imported from the Well-being code if available
    
    Args:
        indicator_code (str): The indicator code
    
    Returns:
        str: Human-readable display name
    """
    # Basic fallback - in practice, import this from the Well-being pipeline
    return indicator_code.replace('-', ' ').replace('_', ' ').title()


def create_time_series_plot(data_dict, title, output_path=None, show_deciles=True):
    """
    Create a standardized time series plot for EWBI data
    
    Args:
        data_dict (dict): Dictionary with country names as keys and DataFrames as values
        title (str): Plot title
        output_path (str, optional): Path to save the plot
        show_deciles (bool): Whether to include decile lines
    
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Define line styles for different data types
    line_styles = {
        'overall': dict(width=3, dash='solid'),
        '1st_decile': dict(width=2, dash='dash'),
        '10th_decile': dict(width=2, dash='dashdot')
    }
    
    # Add traces for each country
    for country_name, country_data in data_dict.items():
        color = SWITZERLAND_COLOR if 'Switzerland' in country_name else EU_27_COLOR
        
        # Add overall line
        if 'overall' in country_data and not country_data['overall'].empty:
            df = country_data['overall'].sort_values('Year')
            fig.add_trace(go.Scatter(
                x=df['Year'],
                y=df['Value'],
                mode='lines+markers',
                name=f'{country_name} Average',
                line=dict(color=color, **line_styles['overall']),
                marker=dict(color=color, size=8),
                hovertemplate=f'<b>{country_name} Average</b><br>' +
                             'Year: %{x}<br>Value: %{y:.3f}<extra></extra>'
            ))
        
        # Add decile lines if requested and available
        if show_deciles and 'deciles' in country_data:
            deciles_df = country_data['deciles']
            
            # 1st decile
            d1_data = deciles_df[deciles_df['Decile'] == '1.0'].sort_values('Year')
            if not d1_data.empty:
                fig.add_trace(go.Scatter(
                    x=d1_data['Year'],
                    y=d1_data['Value'],
                    mode='lines+markers',
                    name=f'{country_name} 1st Decile',
                    line=dict(color=color, **line_styles['1st_decile']),
                    marker=dict(color=color, size=6, symbol='triangle-up'),
                    hovertemplate=f'<b>{country_name} 1st Decile</b><br>' +
                                 'Year: %{x}<br>Value: %{y:.3f}<extra></extra>'
                ))
            
            # 10th decile
            d10_data = deciles_df[deciles_df['Decile'] == '10.0'].sort_values('Year')
            if not d10_data.empty:
                fig.add_trace(go.Scatter(
                    x=d10_data['Year'],
                    y=d10_data['Value'],
                    mode='lines+markers',
                    name=f'{country_name} 10th Decile',
                    line=dict(color=color, **line_styles['10th_decile']),
                    marker=dict(color=color, size=6, symbol='triangle-down'),
                    hovertemplate=f'<b>{country_name} 10th Decile</b><br>' +
                                 'Year: %{x}<br>Value: %{y:.3f}<extra></extra>'
                ))
    
    # Update layout with standard styling
    fig.update_layout(
        title=dict(text=title, x=0.5, **PLOT_STYLE['title_font']),
        xaxis=dict(
            title="Year",
            title_font=PLOT_STYLE['axis_title_font'],
            tickfont=PLOT_STYLE['axis_tick_font'],
            showgrid=True,
            gridcolor=PLOT_STYLE['grid_color'],
            gridwidth=1
        ),
        yaxis=dict(
            title="Share of the population",
            title_font=PLOT_STYLE['axis_title_font'],
            tickfont=PLOT_STYLE['axis_tick_font'],
            showgrid=True,
            gridcolor=PLOT_STYLE['grid_color'],
            gridwidth=1
        ),
        plot_bgcolor=PLOT_STYLE['plot_bgcolor'],
        paper_bgcolor=PLOT_STYLE['paper_bgcolor'],
        font=PLOT_STYLE['font'],
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=PLOT_STYLE['legend_font']
        ),
        margin=PLOT_STYLE['margin'],
        width=PLOT_STYLE['width'],
        height=PLOT_STYLE['height']
    )
    
    # Save if path provided
    if output_path:
        save_plot(fig, output_path)
    
    return fig


def create_decile_bar_chart(data, country_name, indicator_name, year, output_path=None):
    """
    Create a bar chart showing values by income decile
    
    Args:
        data (pd.DataFrame): Decile data for a specific year
        country_name (str): Country name for title
        indicator_name (str): Indicator name for title
        year (int): Year for the data
        output_path (str, optional): Path to save the plot
    
    Returns:
        go.Figure: Plotly figure object
    """
    # Sort by decile
    data = data.copy()
    data['Decile_num'] = pd.to_numeric(data['Decile'], errors='coerce')
    data = data.dropna(subset=['Decile_num']).sort_values('Decile_num')
    
    color = SWITZERLAND_COLOR if 'Switzerland' in country_name else EU_27_COLOR
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=[str(int(d)) for d in data['Decile_num']],
        y=data['Value'],
        name=f'{country_name} by Decile',
        marker=dict(color=color),
        text=[f'{v:.3f}' for v in data['Value']],
        textposition='auto',
        hovertemplate='<b>Decile %{x}</b><br>Score: %{y:.3f}<extra></extra>'
    ))
    
    # Add average line
    overall_average = data['Value'].mean()
    fig.add_hline(
        y=overall_average,
        line_dash="dash",
        line_color=color,
        line_width=2,
        annotation_text=f"{country_name} Average: {overall_average:.3f}",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{indicator_name} - {country_name}<br><sub>By Income Decile ({year})</sub>",
            x=0.5, **PLOT_STYLE['title_font']
        ),
        xaxis=dict(
            title="Income Decile",
            title_font=PLOT_STYLE['axis_title_font'],
            tickfont=PLOT_STYLE['axis_tick_font'],
            showgrid=True,
            gridcolor=PLOT_STYLE['grid_color']
        ),
        yaxis=dict(
            title="Score",
            title_font=PLOT_STYLE['axis_title_font'],
            tickfont=PLOT_STYLE['axis_tick_font'],
            showgrid=True,
            gridcolor=PLOT_STYLE['grid_color']
        ),
        plot_bgcolor=PLOT_STYLE['plot_bgcolor'],
        paper_bgcolor=PLOT_STYLE['paper_bgcolor'],
        font=PLOT_STYLE['font'],
        showlegend=False,
        margin=PLOT_STYLE['margin'],
        width=PLOT_STYLE['width'],
        height=PLOT_STYLE['height']
    )
    
    # Save if path provided
    if output_path:
        save_plot(fig, output_path)
    
    return fig


def create_country_comparison_bar_chart(data_dict, indicator_name, year, output_path=None):
    """
    Create a bar chart comparing countries for a specific indicator and year
    
    Args:
        data_dict (dict): Dictionary with country names as keys and values as numbers
        indicator_name (str): Indicator name for title
        year (int): Year for the data
        output_path (str, optional): Path to save the plot
    
    Returns:
        go.Figure: Plotly figure object
    """
    countries = list(data_dict.keys())
    values = list(data_dict.values())
    colors = [SWITZERLAND_COLOR if 'Switzerland' in country else EU_27_COLOR for country in countries]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=countries,
        y=values,
        marker=dict(color=colors),
        text=[f'{v:.3f}' for v in values],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"{indicator_name}<br><sub>Country Comparison ({year})</sub>",
            x=0.5, **PLOT_STYLE['title_font']
        ),
        xaxis=dict(
            title="Country",
            title_font=PLOT_STYLE['axis_title_font'],
            tickfont=PLOT_STYLE['axis_tick_font']
        ),
        yaxis=dict(
            title="Score",
            title_font=PLOT_STYLE['axis_title_font'],
            tickfont=PLOT_STYLE['axis_tick_font'],
            showgrid=True,
            gridcolor=PLOT_STYLE['grid_color']
        ),
        plot_bgcolor=PLOT_STYLE['plot_bgcolor'],
        paper_bgcolor=PLOT_STYLE['paper_bgcolor'],
        font=PLOT_STYLE['font'],
        showlegend=False,
        margin=PLOT_STYLE['margin'],
        width=PLOT_STYLE['width'],
        height=PLOT_STYLE['height']
    )
    
    # Save if path provided
    if output_path:
        save_plot(fig, output_path)
    
    return fig


def save_plot(fig, output_path, formats=['html', 'png']):
    """
    Save a Plotly figure in multiple formats
    
    Args:
        fig (go.Figure): Plotly figure to save
        output_path (str): Base path for output (without extension)
        formats (list): List of formats to save ('html', 'png')
    """
    output_path = Path(output_path)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save HTML if requested
    if 'html' in formats:
        html_path = output_path.with_suffix('.html')
        fig.write_html(str(html_path))
        print(f"Saved HTML: {html_path}")
    
    # Save PNG if requested
    if 'png' in formats:
        png_path = output_path.with_suffix('.png')
        try:
            # Try using matplotlib for more reliable PNG generation
            save_plot_as_png_matplotlib(fig, png_path)
        except Exception as e:
            print(f"Warning: Could not save PNG for {output_path.name}: {str(e)[:100]}...")


def save_plot_as_png_matplotlib(fig, png_path):
    """
    Save a Plotly figure as PNG using matplotlib for reliability
    
    Args:
        fig (go.Figure): Plotly figure
        png_path (Path): Path for PNG output
    """
    # This is a simplified version - in practice you'd convert Plotly data to matplotlib
    # For now, we'll use a basic approach
    try:
        fig.write_image(str(png_path), width=1000, height=600, scale=1.5)
        print(f"Saved PNG: {png_path}")
    except Exception as e:
        print(f"Could not save PNG using plotly.write_image: {e}")
        # Fallback to matplotlib if needed
        print("Consider installing kaleido: pip install kaleido")


def setup_matplotlib_style():
    """Set up consistent matplotlib styling"""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'lightgray',
        'axes.linewidth': 1,
        'grid.color': 'lightgray',
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })


if __name__ == "__main__":
    """Test the visualization utilities"""
    import numpy as np
    
    print("Testing visualization utilities...")
    
    # Create sample data
    years = list(range(2015, 2023))
    swiss_values = np.random.uniform(0.4, 0.8, len(years))
    eu_values = np.random.uniform(0.3, 0.7, len(years))
    
    # Create sample DataFrames
    swiss_df = pd.DataFrame({
        'Year': years,
        'Value': swiss_values,
        'Country': 'CH',
        'Decile': 'All'
    })
    
    eu_df = pd.DataFrame({
        'Year': years,
        'Value': eu_values,
        'Country': 'All Countries',
        'Decile': 'All'
    })
    
    # Test time series plot
    data_dict = {
        'Switzerland': {'overall': swiss_df},
        'EU-27': {'overall': eu_df}
    }
    
    fig = create_time_series_plot(
        data_dict, 
        "Test Indicator - Switzerland vs EU-27", 
        show_deciles=False
    )
    
    print("✅ Visualization utilities test completed!")
    print("  • Time series plot created successfully")
    print("  • Standard styling applied")
    print("  • Colors: Switzerland (yellow), EU-27 (blue)")