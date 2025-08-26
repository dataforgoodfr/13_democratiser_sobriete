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
    # Load world data
    data = pd.read_csv(os.path.join(DATA_DIR, 'world_unified_decomposition_data.csv'))
    
    print("Data loaded successfully!")
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Get unique values for filters
    ZONES = sorted(data['Zone'].unique())
    SECTORS = sorted(data['Sector'].unique())
    
    # Clean up scenario names to remove duplicates (like "Scenario 1" vs "Scenario 1 ")
    raw_scenarios = data['Scenario'].unique()
    cleaned_scenarios = []
    for scenario in raw_scenarios:
        if pd.notna(scenario):  # Check if scenario is not NaN
            cleaned_scenario = scenario.strip()  # Remove leading/trailing whitespace
            if cleaned_scenario not in cleaned_scenarios:
                cleaned_scenarios.append(cleaned_scenario)
    SCENARIOS = sorted(cleaned_scenarios)
    
    # Define specific lever order
    LEVERS = ['Population', 'Sufficiency', 'Energy Efficiency', 'Supply Side Decarbonation']
    
    print(f"Found {len(ZONES)} zones: {ZONES}")
    print(f"Found {len(SECTORS)} sectors: {SECTORS}")
    print(f"Found {len(SCENARIOS)} scenarios: {SCENARIOS}")
    print(f"Found {len(LEVERS)} levers: {LEVERS}")
    
except FileNotFoundError as e:
    print(f"Error loading world data files: {e}")
    print(f"Please ensure that the world CSV files are in the '{DATA_DIR}' directory.")
    exit()

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
    # Header section with hierarchical titles - EXACTLY like EWBI
    html.Div([
        html.H1("World CO2 Decomposition Dashboard", 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'fontSize': '2.5rem',
                    'fontWeight': 'bold',
                    'marginBottom': '10px',
                    'fontFamily': 'Arial, sans-serif'
                }),
        html.H2([
            "Analyzing the underlying levers of decarbonation scenarios across world regions and how to derisk them"
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
        
        # Controls section embedded within the header - EXACTLY like EWBI
        html.Div([
            html.Div([
                html.Label("World Region", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
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
        'backgroundColor': '#f4d03f',  # World Sufficiency Lab yellow - EXACTLY like EWBI
        'padding': '25px 0px 15px 0px',
        'position': 'sticky',  # Make header sticky like EWBI
        'top': 0,
        'zIndex': 1000,
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Visualizations - EXACTLY like EWBI layout (no intermediate section)
    html.Div([
        # Charts stacked vertically - one below the other
        html.Div([
            dcc.Graph(id='bar-chart', style={'height': '500px'}),
            # Explanatory note about the sign convention
            html.Div([
                html.P([
                    html.Strong("📊 Understanding the Chart:"),
                    html.Br(),
                    "• ",
                    html.Strong("Positive values (above 0%): "),
                    "Indicate emissions reductions - these levers are helping decrease CO2 emissions (negative imply an increase).",
                    html.Br(),
                    "• ",
                    html.Strong("Values >100%: "),
                    "Supply Side Decarbonation can exceed 100% when other levers increase emissions, requiring it to compensate for both the target reduction and emissions increases from other sources.",
                    html.Br(),
                    "• ",
                    html.Strong("Extreme values: "),
                    "Some regions show very high percentages (>300%) due to small baseline emissions or data characteristics. The charts are scaled to show 95% of the data range.",
                    html.Br(),
                    "• ",
                    html.Strong("Sufficiency showing negative values "),
                    "means these scenarios don't envision sufficiency measures - instead, they assume increases in demand or production intensity per capita.",
                    html.Br(),

                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '15px',
                    'borderRadius': '8px',
                    'borderLeft': '4px solid #f4d03f',
                    'marginTop': '15px',
                    'fontSize': '14px',
                    'lineHeight': '1.6',
                    'color': '#2c3e50'
                })
            ])
        ], style={'marginBottom': '30px'}),
        
        html.Div([
            html.Div([
                # Waterfall chart and legend side by side
                html.Div([
                    # Waterfall chart on the left
                    html.Div([
                        dcc.Graph(id='waterfall-chart', style={'height': '500px'})
                    ], style={
                        'flex': '1',
                        'marginRight': '20px'
                    }),
                                        # Color legend on the right
                    html.Div([
                        html.Div([
                            html.Span("🔵 ", style={'color': '#1f77b4', 'fontWeight': 'bold'}),
                            "Target Years (2015, 2040, 2050)",
                            html.Br(),
                            html.Span("🔴 ", style={'color': '#d62728', 'fontWeight': 'bold'}),
                            "The lever is expected to increase emissions",
                            html.Br(),
                            html.Span("🟢 ", style={'color': '#27ae60', 'fontWeight': 'bold'}),
                            "The lever is expected to decrease emissions"
                        ], style={
                            'backgroundColor': '#f8f9fa',
                            'padding': '15px',
                            'borderRadius': '8px',
                            'borderLeft': '4px solid #1f77b4',
                            'fontSize': '14px',
                            'lineHeight': '1.6',
                            'color': '#2c3e50',
                            'textAlign': 'left',
                            'minWidth': '250px'
                        })
                    ], style={
                        'flex': '0 0 auto',
                        'marginRight': '30px'
                    })
                ], style={
                    'display': 'flex',
                    'alignItems': 'flex-start',
                    'justifyContent': 'flex-start',
                    'width': '100%',
                    'marginTop': '20px'
                })
            ])
        ])
    ], style={
        'margin': '0 20px',
        'paddingTop': '20px'  # Extra space to account for sticky headers
    })
])



# Callback to update bar chart
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('zone-dropdown', 'value'),
     Input('sector-dropdown', 'value')]
)
def update_bar_chart(zone, sector):
    if not all([zone, sector]):
        return go.Figure()
    
    # Get data for the selected zone and sector, all scenarios
    chart_data = data[
        (data['Zone'] == zone) &
        (data['Sector'] == sector) &
        (data['Lever'] != 'Total')
    ].copy()
    
    if chart_data.empty:
        return go.Figure().add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Create grouped bar chart
    fig = px.bar(
        chart_data,
        x='Lever',
        y='Contrib_2015_2050_pct',
        color='Scenario',
        barmode='group',
        title=f"Share of Planned CO2 Reduction by Lever - {sector} ({zone})",
        labels={
            'Contrib_2015_2050_pct': 'Contribution (%)',
            'Lever': 'Lever',
            'Scenario': 'Scenario'
        }
    )
    
    fig.update_layout(
        title=dict(
            font=dict(size=16, color="#f4d03f", weight="bold"),
            x=0.5,
            y=0.95
        ),
        xaxis_title="Levers",
        yaxis_title="Contribution to CO2 Change (%)",
        height=500,
        showlegend=True,
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=14),
        margin=dict(t=80, b=50, l=60, r=60),
        yaxis=dict(range=[-250, 350])
    )
    
    # Remove grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    # Add data labels with 0 decimals
    fig.update_traces(
        texttemplate='%{y:.0f}%',
        textposition='outside',
        textfont=dict(size=12, color='#2c3e50')
    )
    
    # Improve hover formatting with 0 decimals and better spacing
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>" +
                     "<b>Scenario:</b> %{fullData.name}<br>" +
                     "<b>Contribution:</b> %{y:.0f}%<br>" +
                     "<extra></extra>"
    )
    
    # Update colors to group scenarios by type with different shades
    # Get unique scenarios and categorize them
    scenarios = chart_data['Scenario'].unique()
    
    # Define color schemes for different scenario types
    remind_sdp_colors = ['#1f77b4', '#3182bd', '#4292c6']  # Different shades of blue for REMIND SDP scenarios
    remind_ssp_colors = ['#2ca02c', '#31a354', '#74c476']  # Different shades of green for REMIND SSP scenarios
    remind_india_colors = ['#ff7f0e']  # Orange for REMIndia scenarios
    
    # Create color mapping
    color_map = {}
    remind_sdp_count = 0
    remind_ssp_count = 0
    remind_india_count = 0
    
    for scenario in scenarios:
        if 'REMIND SDP' in scenario:
            color_map[scenario] = remind_sdp_colors[remind_sdp_count % len(remind_sdp_colors)]
            remind_sdp_count += 1
        elif 'REMIND SSP' in scenario:
            color_map[scenario] = remind_ssp_colors[remind_ssp_count % len(remind_ssp_colors)]
            remind_ssp_count += 1
        elif 'REMIndia' in scenario:
            color_map[scenario] = remind_india_colors[remind_india_count % len(remind_india_colors)]
            remind_india_count += 1
        else:
            color_map[scenario] = '#636363'  # Default gray
    
    # Apply the color mapping
    for i, trace in enumerate(fig.data):
        scenario_name = trace.name
        if scenario_name in color_map:
            trace.marker.color = color_map[scenario_name]
    
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
    
    # Calculate baseline for percentages (2015 = 100%)
    baseline_2015 = co2_2015
    
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
    
    # Prepare data for waterfall chart (using percentages, 2015 = 100%)
    x_labels = ['2015']
    y_values = [100.0]  # 2015 = 100%
    measures = ['absolute']
    colors = ['#1f77b4']  # Blue for starting year
    text_values = ["100%"]
    
    # Add 2015-2040 period contributions
    for _, lever in lever_data.iterrows():
        lever_name = lever['Lever']
        contrib_2015_2040 = lever['Contrib_2015_2040_abs']
        # Convert to percentage relative to 2015 baseline
        contrib_2015_2040_pct = (contrib_2015_2040 / baseline_2015) * 100
        x_labels.append(f"{lever_name} 2015-40")
        y_values.append(contrib_2015_2040_pct)
        measures.append('relative')
        colors.append(lever_colors.get(lever_name, "#636363"))
        text_values.append(f"{contrib_2015_2040_pct:.1f}%")
    
    # Add 2040 total
    x_labels.append('2040')
    co2_2040_pct = (co2_2040 / baseline_2015) * 100
    y_values.append(co2_2040_pct)
    measures.append('total')
    colors.append('#1f77b4')  # Blue for intermediate year
    text_values.append(f"{co2_2040_pct:.1f}%")
    
    # Add 2040-2050 period contributions
    for _, lever in lever_data.iterrows():
        lever_name = lever['Lever']
        contrib_2040_2050 = lever['Contrib_2040_2050_abs']
        # Convert to percentage relative to 2015 baseline
        contrib_2040_2050_pct = (contrib_2040_2050 / baseline_2015) * 100
        x_labels.append(f"{lever_name} 2040-50")
        y_values.append(contrib_2040_2050_pct)
        measures.append('relative')
        colors.append(lever_colors.get(lever_name, "#636363"))
        text_values.append(f"{contrib_2040_2050_pct:.1f}%")
    
    # Add 2050 final total
    x_labels.append('2050')
    co2_2050_pct = (co2_2050 / baseline_2015) * 100
    y_values.append(co2_2050_pct)
    measures.append('total')
    colors.append('#1f77b4')  # Blue for final year
    text_values.append(f"{co2_2050_pct:.1f}%")
    
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
        decreasing={"marker": {"color": "#27ae60"}},  # Green for decreases
        increasing={"marker": {"color": "#d62728"}},  # Red for increases
        totals={"marker": {"color": "#1f77b4"}},      # Blue for totals
        showlegend=False  # Hide legend
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Planned CO2 Emissions Decrease Over Time by Lever - {sector} ({zone}) - % of 2015 Baseline",
            font=dict(size=16, color="#f4d03f", weight="bold"),  # EXACTLY like EWBI dashboard
            x=0.5,
            y=0.95  # Consistent title position for alignment
        ),
        xaxis_title="",  # Removed X-axis label
        yaxis_title="CO2 Emissions (% of 2015 baseline)",
        height=600,  # Made taller
        width=1300,  # Make chart less wide
        showlegend=False,  # Hide legend
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=14),  # EXACTLY like EWBI dashboard
        margin=dict(t=80, b=50, l=80, r=80),  # Increased left/right margins to center and make less wide
        xaxis=dict(
            tickangle=-30,  # Less tilted
            tickfont=dict(size=14)  # Same font size as first graph
        ),
        yaxis=dict(
            showgrid=False,  # Remove horizontal lines
            range=[-250, 350]  # Extended to capture 95% of data range
        )
    )
    
    return fig

# Callbacks
@app.callback(
    Output('scenario-dropdown', 'options'),
    Output('scenario-dropdown', 'value'),
    Input('zone-dropdown', 'value')
)
def update_scenario_options(selected_zone):
    # For world data, use all available scenarios dynamically
    options = [{'label': scenario, 'value': scenario} for scenario in SCENARIOS]
    value = SCENARIOS[0] if SCENARIOS else None
    return options, value

@app.callback(
    Output('sector-dropdown', 'options'),
    Output('sector-dropdown', 'value'),
    Input('zone-dropdown', 'value')
)
def update_sector_options(selected_zone):
    # For world data, use all available sectors dynamically
    options = [{'label': sector, 'value': sector} for sector in SECTORS]
    value = SECTORS[0] if SECTORS else None
    return options, value

if __name__ == '__main__':
    print("Starting CO2 Decomposition Dashboard...")
    print(f"Dashboard will be available at: http://localhost:8051")
    app.run_server(debug=True, host='0.0.0.0', port=8051) 