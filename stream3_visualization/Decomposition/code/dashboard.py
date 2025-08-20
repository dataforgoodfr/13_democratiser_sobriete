#!/usr/bin/env python3
"""
CO2 Decomposition Dashboard
Visualizes CO2 emission reductions by levers across scenarios, sectors, and zones
"""

import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import os

# Get the absolute path to the data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))

# Load the data
try:
    data = pd.read_csv(os.path.join(DATA_DIR, 'unified_decomposition_data.csv'))
    print("Data loaded successfully!")
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Get unique values for filters
    ZONES = sorted(data['Zone'].unique())
    SECTORS = sorted(data['Sector'].unique())
    SCENARIOS = sorted(data['Scenario'].unique())
    LEVERS = sorted([lever for lever in data['Lever'].unique() if lever != 'Total'])
    
    print(f"Found {len(ZONES)} zones: {ZONES}")
    print(f"Found {len(SECTORS)} sectors: {SECTORS}")
    print(f"Found {len(SCENARIOS)} scenarios: {SCENARIOS}")
    print(f"Found {len(LEVERS)} levers: {LEVERS}")
    
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print(f"Please ensure that the CSV files are in the '{DATA_DIR}' directory.")
    exit()

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
    # Header section with hierarchical titles
    html.Div([
        html.H1("🌍 CO2 Emissions Decomposition Dashboard", 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'marginBottom': '10px',
                    'fontSize': '32px',
                    'fontWeight': 'bold'
                }),
        html.H2("Analyzing CO2 Emission Reductions by Levers", 
                style={
                    'textAlign': 'center',
                    'color': '#34495e',
                    'marginBottom': '20px',
                    'fontSize': '20px',
                    'fontWeight': 'normal'
                }),
        
        # Filters embedded in header
        html.Div([
            html.Div([
                html.Label("Zone:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='zone-dropdown',
                    options=[{'label': zone, 'value': zone} for zone in ZONES],
                    value=ZONES[0] if ZONES else None,
                    style={'width': '150px', 'display': 'inline-block'}
                )
            ], style={'display': 'inline-block', 'marginRight': '20px'}),
            
            html.Div([
                html.Label("Sector:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='sector-dropdown',
                    options=[{'label': sector, 'value': sector} for sector in SECTORS],
                    value=SECTORS[0] if SECTORS else None,
                    style={'width': '150px', 'display': 'inline-block'}
                )
            ], style={'display': 'inline-block', 'marginRight': '20px'}),
            
            html.Div([
                html.Label("Scenario:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='scenario-dropdown',
                    options=[{'label': scenario, 'value': scenario} for scenario in SCENARIOS],
                    value=SCENARIOS[0] if SCENARIOS else None,
                    style={'width': '200px', 'display': 'inline-block'}
                )
            ], style={'display': 'inline-block'})
        ], style={
            'backgroundColor': '#fdf6e3',
            'padding': '15px',
            'borderRadius': '8px',
            'marginBottom': '20px',
            'textAlign': 'center'
        })
    ], style={
        'backgroundColor': '#f4d03f',
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '30px',
        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
    }),
    
    # Charts Section
    html.Div([
        # Bar Chart
        html.Div([
            dcc.Graph(id='bar-chart', style={'height': '500px'})
        ], style={'marginBottom': '30px'}),
        
        # Waterfall Chart
        html.Div([
            dcc.Graph(id='waterfall-chart', style={'height': '500px'})
        ])
    ], style={
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
    })
], style={
    'backgroundColor': '#f8f9fa',
    'minHeight': '100vh',
    'padding': '20px',
    'fontFamily': 'Arial, sans-serif'
})



# Callback to update bar chart
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('zone-dropdown', 'value'),
     Input('sector-dropdown', 'value'),
     Input('scenario-dropdown', 'value')]
)
def update_bar_chart(zone, sector, scenario):
    if not all([zone, sector, scenario]):
        return go.Figure()
    
    # Get data for the selected filters
    chart_data = data[
        (data['Zone'] == zone) &
        (data['Sector'] == sector) &
        (data['Scenario'] == scenario) &
        (data['Lever'] != 'Total')
    ].copy()
    
    if chart_data.empty:
        return go.Figure().add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Calculate percentage contributions
    total_reduction = chart_data['Contrib_2015_2050_abs'].sum()
    chart_data['Percentage'] = (chart_data['Contrib_2015_2050_abs'] / total_reduction) * 100
    
    # Sort by percentage (descending)
    chart_data = chart_data.sort_values('Percentage', ascending=False)
    
    # Create bar chart
    fig = go.Figure()
    
    # Define colors for levers
    lever_colors = {
        'Population': '#1f77b4',
        'Sufficiency': '#ff7f0e', 
        'Energy Efficiency': '#2ca02c',
        'Supply Side Decarbonation': '#d62728'
    }
    
    fig.add_trace(go.Bar(
        x=chart_data['Lever'],
        y=chart_data['Percentage'],
        marker_color=[lever_colors.get(lever, "#636363") for lever in chart_data['Lever']],
        text=[f"{pct:.1f}%" for pct in chart_data['Percentage']],
        textposition='outside',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Share of Planned CO2 Reduction by Lever - {scenario} ({sector}, {zone})",
        xaxis_title="Levers",
        yaxis_title="Percentage of Total Reduction (%)",
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        font=dict(size=12),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            gridcolor='lightgray',
            gridwidth=1,
            range=[-20, 100]
        )
    )
    
    return fig

# Callback to update waterfall chart
@app.callback(
    Output('waterfall-chart', 'figure'),
    [Input('zone-dropdown', 'value'),
     Input('sector-dropdown', 'value'),
     Input('scenario-dropdown', 'value')]
)
def update_waterfall_chart(zone, sector, scenario):
    if not all([zone, sector, scenario]):
        return go.Figure()
    
    # Get data for the selected scenario
    scenario_data = data[
        (data['Zone'] == zone) & 
        (data['Sector'] == sector) &
        (data['Scenario'] == scenario)
    ].copy()
    
    if scenario_data.empty:
        return go.Figure().add_annotation(
            text="No data available for the selected scenario",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Get Total lever data for starting points
    total_data = scenario_data[scenario_data['Lever'] == 'Total']
    if total_data.empty:
        return go.Figure().add_annotation(
            text="No Total data available for the selected scenario",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    total_data = total_data.iloc[0]
    co2_2015 = total_data['CO2_2015']
    co2_2040 = total_data['CO2_2040']
    co2_2050 = total_data['CO2_2050']
    
    # Get lever contributions (excluding Total)
    lever_data = scenario_data[scenario_data['Lever'] != 'Total'].copy()
    
    # Create proper waterfall chart
    fig = go.Figure()
    
    # Define colors for levers
    lever_colors = {
        'Population': '#1f77b4',
        'Sufficiency': '#ff7f0e', 
        'Energy Efficiency': '#2ca02c',
        'Supply Side Decarbonation': '#d62728'
    }
    
    # Prepare data for waterfall chart
    x_labels = ['2015']
    y_values = [co2_2015]
    measures = ['absolute']
    colors = ['#1f77b4']  # Blue for starting year
    text_values = [f"{co2_2015:.1f}"]
    
    # Add 2015-2040 period contributions
    for _, lever in lever_data.iterrows():
        lever_name = lever['Lever']
        contrib_2015_2040 = lever['Contrib_2015_2040_abs']
        x_labels.append(f"{lever_name} (1)")
        y_values.append(contrib_2015_2040)
        measures.append('relative')
        colors.append(lever_colors.get(lever_name, "#636363"))
        text_values.append(f"{contrib_2015_2040:.1f}")
    
    # Add 2040 total
    x_labels.append('2040')
    y_values.append(co2_2040)
    measures.append('total')
    colors.append('#1f77b4')  # Blue for intermediate year
    text_values.append(f"{co2_2040:.1f}")
    
    # Add 2040-2050 period contributions
    for _, lever in lever_data.iterrows():
        lever_name = lever['Lever']
        contrib_2040_2050 = lever['Contrib_2040_2050_abs']
        x_labels.append(f"{lever_name} (2)")
        y_values.append(contrib_2040_2050)
        measures.append('relative')
        colors.append(lever_colors.get(lever_name, "#636363"))
        text_values.append(f"{contrib_2040_2050:.1f}")
    
    # Add 2050 final total
    x_labels.append('2050')
    y_values.append(co2_2050)
    measures.append('total')
    colors.append('#1f77b4')  # Blue for final year
    text_values.append(f"{co2_2050:.1f}")
    
    # Create the waterfall chart
    fig.add_trace(go.Waterfall(
        name="CO2 Emissions",
        orientation="v",  # Vertical orientation like your screenshot
        measure=measures,
        x=x_labels,
        y=y_values,
        textposition="outside",
        text=text_values,
        connector={"line": {"color": "rgb(63, 63, 63)", "width": 2}},
        decreasing={"marker": {"color": "#d62728"}},  # Red for decreases
        increasing={"marker": {"color": "#27ae60"}},  # Green for increases
        totals={"marker": {"color": "#1f77b4"}},      # Blue for totals
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"CO2 Emissions Waterfall - {scenario} ({sector}, {zone})",
        xaxis_title="Time Periods and Levers",
        yaxis_title="CO2 Emissions (Million tonnes)",
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        font=dict(size=12),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            gridcolor='lightgray',
            gridwidth=1,
            range=[-10, 700]
        )
    )
    
    return fig

if __name__ == '__main__':
    print("Starting CO2 Decomposition Dashboard...")
    print(f"Dashboard will be available at: http://localhost:8051")
    app.run_server(debug=True, host='0.0.0.0', port=8051) 