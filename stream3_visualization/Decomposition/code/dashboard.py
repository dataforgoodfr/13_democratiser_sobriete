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
                    'fontSize': '2.5rem',
                    'fontWeight': 'bold',
                    'marginBottom': '10px',
                    'fontFamily': 'Arial, sans-serif'
                }),
        html.H2([
            "Analyze CO2 emission reductions by levers across scenarios, sectors, and zones"
        ], style={
            'textAlign': 'center',
            'color': '#34495e',
            'fontSize': '1.2rem',
            'fontWeight': 'normal',
            'marginBottom': '30px',
            'fontFamily': 'Arial, sans-serif',
            'lineHeight': '1.4',
            'maxWidth': '900px',
            'margin': '0 auto 30px auto'
        }),
        
        # Controls section embedded within the header
        html.Div([
            html.Div([
                html.Label("Zone", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='zone-dropdown',
                    options=[{'label': zone, 'value': zone} for zone in ZONES],
                    value=ZONES[0] if ZONES else None,
                    style={'marginTop': '8px'},
                    clearable=False
                )
            ], style={'width': '25%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.Label("Sector", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='sector-dropdown',
                    options=[{'label': sector, 'value': sector} for sector in SECTORS],
                    value=SECTORS[0] if SECTORS else None,
                    style={'marginTop': '8px'},
                    clearable=False
                )
            ], style={'width': '25%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.Label("Scenario", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='scenario-dropdown',
                    options=[{'label': scenario, 'value': scenario} for scenario in SCENARIOS],
                    value=SCENARIOS[0] if SCENARIOS else None,
                    style={'marginTop': '8px'},
                    clearable=False
                )
            ], style={'width': '25%', 'display': 'inline-block'}),
        ], style={
            'padding': '15px 20px',
            'backgroundColor': '#fdf6e3',
            'borderRadius': '8px',
            'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
            'margin': '0 20px 10px 20px'
        })
    ], style={
        'backgroundColor': '#f4d03f',  # World Sufficiency Lab yellow
        'padding': '25px 0px 15px 0px',
        'position': 'sticky',
        'top': 0,
        'zIndex': 1000,
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Selected filters display
    html.Div(id='selected-filters', style={
        'textAlign': 'center',
        'marginBottom': '20px',
        'fontSize': '1.1rem',
        'color': '#2c3e50',
        'margin': '20px 20px 20px 20px'
    }),
    
    # Visualizations
    html.Div([
        # First row: Bar chart and waterfall chart side by side
        html.Div([
            html.Div([
                html.H3("📊 Share of Planned CO2 Reduction by Lever (%)", 
                        style={
                            'color': '#f4d03f',  # Yellow title to match EWBI
                            'fontSize': '1.8rem',
                            'fontWeight': 'bold',
                            'marginBottom': '20px',
                            'textAlign': 'center'
                        }),
                dcc.Graph(id='bar-chart', style={'height': '500px'})
            ], style={'display': 'inline-block', 'width': '49%', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H3("🌊 Planned CO2 Emissions Decrease Over Time by Lever", 
                        style={
                            'color': '#f4d03f',  # Yellow title to match EWBI
                            'fontSize': '1.8rem',
                            'fontWeight': 'bold',
                            'marginBottom': '20px',
                            'textAlign': 'center'
                        }),
                dcc.Graph(id='waterfall-chart', style={'height': '500px'})
            ], style={'display': 'inline-block', 'width': '49%', 'verticalAlign': 'top'})
        ], style={'textAlign': 'center', 'margin': '0 auto'}),
        
        # Second row: Summary statistics
        html.Div([
            html.Div(id='summary-stats', style={
                'marginTop': '20px',
                'width': '100%'
            })
        ])
    ], style={
        'margin': '0 20px',
        'paddingTop': '20px'  # Extra space to account for sticky headers
    }),
    
    # Footer
    html.Div([
        html.Hr(style={'borderColor': '#ecf0f1', 'margin': '40px 0 20px 0'}),
        html.P("Dashboard created for CO2 decomposition analysis", 
               style={
                   'textAlign': 'center', 
                   'color': '#7f8c8d', 
                   'fontStyle': 'italic',
                   'fontSize': '1rem'
               })
    ])
], style={'fontFamily': 'Arial, sans-serif'})

# Callback to update sector dropdown based on zone selection
@app.callback(
    Output('sector-dropdown', 'options'),
    Output('sector-dropdown', 'value'),
    Input('zone-dropdown', 'value')
)
def update_sector_dropdown(selected_zone):
    if not selected_zone:
        return [], None
    
    sectors = sorted(data[data['Zone'] == selected_zone]['Sector'].unique())
    options = [{'label': sector, 'value': sector} for sector in sectors]
    return options, sectors[0] if sectors else None

# Callback to update scenario dropdown based on zone and sector selection
@app.callback(
    Output('scenario-dropdown', 'options'),
    Output('scenario-dropdown', 'value'),
    Input('zone-dropdown', 'value'),
    Input('sector-dropdown', 'value')
)
def update_scenario_dropdown(selected_zone, selected_sector):
    if not selected_zone or not selected_sector:
        return [], None
    
    scenarios = sorted(data[
        (data['Zone'] == selected_zone) & 
        (data['Sector'] == selected_sector)
        ]['Scenario'].unique())
    options = [{'label': scenario, 'value': scenario} for scenario in scenarios]
    return options, scenarios[0] if scenarios else None

# Callback to update selected filters display
@app.callback(
    Output('selected-filters', 'children'),
    Input('zone-dropdown', 'value'),
    Input('sector-dropdown', 'value'),
    Input('scenario-dropdown', 'value')
)
def update_selected_filters(zone, sector, scenario):
    if not all([zone, sector, scenario]):
        return ""
    
    return html.Div([
        html.Span(f"📍 Zone: {zone}", style={'marginRight': '20px', 'fontWeight': 'bold'}),
        html.Span(f"🏭 Sector: {sector}", style={'marginRight': '20px', 'fontWeight': 'bold'}),
        html.Span(f"📈 Scenario: {scenario}", style={'fontWeight': 'bold'})
    ])

# Callback to update bar chart
@app.callback(
    Output('bar-chart', 'figure'),
    Input('zone-dropdown', 'value'),
    Input('sector-dropdown', 'value')
)
def update_bar_chart(zone, sector):
    if not zone or not sector:
        return go.Figure()
        
        # Filter data for the selected zone and sector, all scenarios
    chart_data = data[
        (data['Zone'] == zone) & 
        (data['Sector'] == sector) &
        (data['Lever'] != 'Total')  # Exclude Total lever
    ].copy()
    
    if chart_data.empty:
        return go.Figure().add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        
        # Create the bar chart
        fig = px.bar(
            chart_data,
            x='Lever',
            y='Contrib_2015_2050_pct',
            color='Scenario',
            barmode='group',
            title=f"CO2 Reduction Contributions by Lever - {sector} ({zone})",
            labels={
                'Contrib_2015_2050_pct': 'Contribution (%)',
                'Lever': 'Lever',
                'Scenario': 'Scenario'
            }
        )
        
        fig.update_layout(
            xaxis_title="Lever",
            yaxis_title="Contribution to CO2 Reduction (%)",
            height=500,
        showlegend=True,
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

# Callback to update waterfall chart
@app.callback(
    Output('waterfall-chart', 'figure'),
    Input('zone-dropdown', 'value'),
    Input('sector-dropdown', 'value'),
    Input('scenario-dropdown', 'value')
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
            
        x_labels.append(f"{lever_name}\n(2015-2040)")
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
            
        x_labels.append(f"{lever_name}\n(2040-2050)")
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
            tickangle=-45,  # Rotate labels for better readability
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            gridcolor='lightgray',
            gridwidth=1
        )
    )
    
    return fig

# Callback to update summary statistics
@app.callback(
    Output('summary-stats', 'children'),
    Input('zone-dropdown', 'value'),
    Input('sector-dropdown', 'value'),
    Input('scenario-dropdown', 'value')
)
def update_summary_stats(zone, sector, scenario):
    if not all([zone, sector, scenario]):
        return ""
    
    # Get data for the selected scenario
    scenario_data = data[
        (data['Zone'] == zone) & 
        (data['Sector'] == sector) &
        (data['Scenario'] == scenario)
    ].copy()
    
    if scenario_data.empty:
        return html.P("No data available for the selected scenario")
    
    # Get Total lever data
    total_data = scenario_data[scenario_data['Lever'] == 'Total']
    if total_data.empty:
        return html.P("No Total data available for the selected scenario")
    
    total_data = total_data.iloc[0]
    co2_2015 = total_data['CO2_2015']
    co2_2040 = total_data['CO2_2040']
    co2_2050 = total_data['CO2_2050']
    
    # Get lever contributions (excluding Total)
    lever_data = scenario_data[scenario_data['Lever'] != 'Total'].copy()
    
    return html.Div([
        html.H3("📊 Summary Statistics", style={
            'color': '#2c3e50',
            'fontSize': '1.8rem',
            'fontWeight': 'bold',
            'marginBottom': '20px',
            'textAlign': 'center'
        }),
        
        # CO2 values in a row
        html.Div([
            html.Div([
                html.H4("CO2 2015", style={
                    'margin': '0 0 10px 0', 
                    'color': '#2c3e50',
                    'fontSize': '1.2rem',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                }),
                html.P(f"{co2_2015:.1f} Mt", style={
                    'fontSize': '2rem', 
                    'fontWeight': 'bold', 
                    'color': '#e74c3c', 
                    'margin': '0',
                    'textAlign': 'center'
                })
            ], style={
                'textAlign': 'center', 
                'flex': '1',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'margin': '0 10px',
                'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'
            }),
            
            html.Div([
                html.H4("CO2 2040", style={
                    'margin': '0 0 10px 0', 
                    'color': '#2c3e50',
                    'fontSize': '1.2rem',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                }),
                html.P(f"{co2_2040:.1f} Mt", style={
                    'fontSize': '2rem', 
                    'fontWeight': 'bold', 
                    'color': '#f39c12', 
                    'margin': '0',
                    'textAlign': 'center'
                })
            ], style={
                'textAlign': 'center', 
                'flex': '1',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'margin': '0 10px',
                'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'
            }),
            
            html.Div([
                html.H4("CO2 2050", style={
                    'margin': '0 0 10px 0', 
                    'color': '#2c3e50',
                    'fontSize': '1.2rem',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                }),
                html.P(f"{co2_2050:.1f} Mt", style={
                    'fontSize': '2rem', 
                    'fontWeight': 'bold', 
                    'color': '#27ae60', 
                    'margin': '0',
                    'textAlign': 'center'
                })
            ], style={
                'textAlign': 'center', 
                'flex': '1',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'margin': '0 10px',
                'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'
            })
        ], style={
            'display': 'flex', 
            'justifyContent': 'space-around', 
            'marginBottom': '30px',
            'maxWidth': '900px',
            'margin': '0 auto 30px auto'
        }),
        
        # Lever contributions table
        html.Div([
            html.H4("📋 Lever Contributions", style={
                'color': '#34495e',
                'fontSize': '1.2rem',
                'fontWeight': 'bold',
                'marginBottom': '20px',
                'textAlign': 'center'
            }),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Lever", style={
                            'padding': '15px', 
                            'border': '1px solid #ddd',
                            'backgroundColor': '#f4d03f',
                            'color': '#2c3e50',
                            'fontWeight': 'bold'
                        }),
                        html.Th("2015-2040 (Mt)", style={
                            'padding': '15px', 
                            'border': '1px solid #ddd',
                            'backgroundColor': '#f4d03f',
                            'color': '#2c3e50',
                            'fontWeight': 'bold'
                        }),
                        html.Th("2040-2050 (Mt)", style={
                            'padding': '15px', 
                            'border': '1px solid #ddd',
                            'backgroundColor': '#f4d03f',
                            'color': '#2c3e50',
                            'fontWeight': 'bold'
                        }),
                        html.Th("2015-2050 (Mt)", style={
                            'padding': '15px', 
                            'border': '1px solid #ddd',
                            'backgroundColor': '#f4d03f',
                            'color': '#2c3e50',
                            'fontWeight': 'bold'
                        })
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(lever['Lever'], style={
                            'padding': '15px', 
                            'border': '1px solid #ddd',
                            'fontWeight': 'bold',
                            'color': '#2c3e50'
                        }),
                        html.Td(f"{lever['Contrib_2015_2040_abs']:.2f}", style={
                            'padding': '15px', 
                            'border': '1px solid #ddd', 
                            'textAlign': 'center',
                            'color': '#34495e'
                        }),
                        html.Td(f"{lever['Contrib_2040_2050_abs']:.2f}", style={
                            'padding': '15px', 
                            'border': '1px solid #ddd', 
                            'textAlign': 'center',
                            'color': '#34495e'
                        }),
                        html.Td(f"{lever['Contrib_2015_2050_abs']:.2f}", style={
                            'padding': '15px', 
                            'border': '1px solid #ddd', 
                            'textAlign': 'center',
                            'color': '#34495e'
                        })
                    ]) for _, lever in lever_data.iterrows()
                ])
            ], style={
                'margin': '0 auto', 
                'borderCollapse': 'collapse', 
                'width': '100%',
                'maxWidth': '800px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'borderRadius': '8px',
                'overflow': 'hidden'
            })
        ], style={
            'maxWidth': '900px',
            'margin': '0 auto',
            'padding': '0 20px'
        })
    ])

if __name__ == '__main__':
    print("Starting CO2 Decomposition Dashboard...")
    print(f"Dashboard will be available at: http://localhost:8051")
    app.run_server(debug=True, host='0.0.0.0', port=8051) 