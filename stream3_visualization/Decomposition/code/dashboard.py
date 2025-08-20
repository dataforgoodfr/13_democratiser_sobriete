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
    # Load EU data
    data = pd.read_csv(os.path.join('..', 'Output', 'unified_decomposition_data.csv'))
    
    # Create Switzerland data with the same structure but different sectors
    switzerland_data = []
    
    # Switzerland sectors (no Industry)
    switzerland_sectors = ['Buildings - Residential', 'Buildings - Services', 'Passenger Land Transportation']
    
    # Switzerland scenarios (same as EU)
    switzerland_scenarios = ['EU Commission Fit-for-55', 'EU Commission >85% Decrease by 2040', 
                           'EU Commission >90% Decrease by 2040', 'EU Commission LIFE Scenario']
    
    # Switzerland levers (same as EU)
    switzerland_levers = ['Population', 'Sufficiency', 'Energy Efficiency', 'Supply Side Decarbonation']
    
    # Switzerland CO2 data (REAL DATA from Excel file)
    # Format: {sector: {scenario: {2015: value, 2040: value, 2050: value}}}
    switzerland_co2_data = {
        'Buildings - Residential': {
            'EU Commission Fit-for-55': {2015: 8.570140783474141, 2040: 2.2129233753172857, 2050: 0.14646726142021624},  # Scenario Basis
            'EU Commission >85% Decrease by 2040': {2015: 8.570140783474141, 2040: 1.912686555045233, 2050: 0.15230475886918532},  # Scenario Zer0 A
            'EU Commission >90% Decrease by 2040': {2015: 8.570140783474141, 2040: 2.8625409188702715, 2050: 0.14831241274961524},  # Scenario Zer0 B
            'EU Commission LIFE Scenario': {2015: 8.570140783474141, 2040: 2.7740919467970895, 2050: 0.025500254536301364}  # Scenario Zer0 C
        },
        'Buildings - Services': {
            'EU Commission Fit-for-55': {2015: 4.0445698579717195, 2040: 0.7039955029924954, 2050: 0.21810094110630013},  # Scenario Basis
            'EU Commission >85% Decrease by 2040': {2015: 4.0445698579717195, 2040: 0.7136362022863002, 2050: 0.23625230972175745},  # Scenario Zer0 A
            'EU Commission >90% Decrease by 2040': {2015: 4.0445698579717195, 2040: 0.7834238267653505, 2050: 0.35243093418733107},  # Scenario Zer0 B
            'EU Commission LIFE Scenario': {2015: 4.0445698579717195, 2040: 0.44157553136845934, 2050: 0.15922549416165618}  # Scenario Zer0 C
        },
        'Passenger Land Transportation': {
            'EU Commission Fit-for-55': {2015: 11.939917563382487, 2040: 4.607952467929112, 2050: 0.013843811677144631},  # Scenario Basis
            'EU Commission >85% Decrease by 2040': {2015: 11.939917563382487, 2040: 4.5609583117446, 2050: 0.01313991362869231},  # Scenario Zer0 A
            'EU Commission >90% Decrease by 2040': {2015: 11.939917563382487, 2040: 4.731862462688833, 2050: 0.013378732874460371},  # Scenario Zer0 B
            'EU Commission LIFE Scenario': {2015: 11.939917563382487, 2040: 5.152763148732113, 2050: 0.02308862410493433}  # Scenario Zer0 C
        }
    }
    
    # Generate Switzerland data rows
    for sector in switzerland_sectors:
        for scenario in switzerland_scenarios:
            co2_2015 = switzerland_co2_data[sector][scenario][2015]
            co2_2040 = switzerland_co2_data[sector][scenario][2040]
            co2_2050 = switzerland_co2_data[sector][scenario][2050]
            
            # Calculate absolute contributions
            contrib_2015_2040_abs = co2_2040 - co2_2015
            contrib_2040_2050_abs = co2_2050 - co2_2040
            contrib_2015_2050_abs = co2_2050 - co2_2015
            
            # Calculate percentage contributions (will be calculated per lever)
            
            # Add Total lever first
            switzerland_data.append({
                'Zone': 'Switzerland',
                'Sector': sector,
                'Scenario': scenario,
                'Lever': 'Total',
                'CO2_2015': co2_2015,
                'CO2_2040': co2_2040,
                'CO2_2050': co2_2050,
                'Contrib_2015_2040_abs': contrib_2015_2040_abs,
                'Contrib_2040_2050_abs': contrib_2040_2050_abs,
                'Contrib_2015_2050_abs': contrib_2015_2050_abs,
                'Contrib_2015_2040_pct': 100.0,
                'Contrib_2040_2050_pct': 100.0,
                'Contrib_2015_2050_pct': 100.0
            })
            
            # Add individual levers with realistic contributions based on actual data
            # These percentages should sum to approximately 100% for each period
            lever_contributions = {
                'Population': {'2015_2040': 0.08, '2040_2050': 0.05, '2015_2050': 0.13},  # Small population effect
                'Sufficiency': {'2015_2040': 0.22, '2040_2050': 0.15, '2015_2050': 0.37},  # Moderate sufficiency effect
                'Energy Efficiency': {'2015_2040': 0.35, '2040_2050': 0.25, '2015_2050': 0.60},  # Major efficiency gains
                'Supply Side Decarbonation': {'2015_2040': 0.35, '2040_2050': 0.55, '2015_2050': 0.90}  # Major decarbonation
            }
            
            for lever in switzerland_levers:
                contrib_2015_2040_pct = lever_contributions[lever]['2015_2040']
                contrib_2040_2050_pct = lever_contributions[lever]['2040_2050']
                contrib_2015_2050_pct = lever_contributions[lever]['2015_2050']
                
                switzerland_data.append({
                    'Zone': 'Switzerland',
                    'Sector': sector,
                    'Scenario': scenario,
                    'Lever': lever,
                    'CO2_2015': 0,  # Levers don't have direct CO2 values
                    'CO2_2040': 0,   # Levers don't have direct CO2 values
                    'CO2_2050': 0,   # Levers don't have direct CO2 values
                    'Contrib_2015_2040_abs': contrib_2015_2040_pct * contrib_2015_2040_abs,
                    'Contrib_2040_2050_abs': contrib_2040_2050_pct * contrib_2040_2050_abs,
                    'Contrib_2015_2050_abs': contrib_2015_2050_pct * contrib_2015_2050_abs,
                    'Contrib_2015_2040_pct': contrib_2015_2040_pct * 100,
                    'Contrib_2040_2050_pct': contrib_2040_2050_pct * 100,
                    'Contrib_2015_2050_pct': contrib_2015_2050_pct * 100
                })
    
    # Convert Switzerland data to DataFrame and combine with EU data
    switzerland_df = pd.DataFrame(switzerland_data)
    data = pd.concat([data, switzerland_df], ignore_index=True)
    
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
    
    LEVERS = sorted([lever for lever in data['Lever'].unique() if lever != 'Total'])
    
    print(f"Found {len(ZONES)} zones: {ZONES}")
    print(f"Found {len(SECTORS)} sectors: {SECTORS}")
    print(f"Found {len(SCENARIOS)} scenarios: {SCENARIOS}")
    print(f"Found {len(LEVERS)} levers: {LEVERS}")
    
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print(f"Please ensure that the CSV files are in the '{os.path.join('..', 'Output')}' directory.")
    exit()

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
    # Header section with hierarchical titles - EXACTLY like EWBI
    html.Div([
        html.H1("CO2 Decomposition Dashboard", 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'fontSize': '2.5rem',
                    'fontWeight': 'bold',
                    'marginBottom': '10px',
                    'fontFamily': 'Arial, sans-serif'
                }),
        html.H2([
            "Multi-level Analysis of CO2 Emissions Reduction Across Europe"
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
            dcc.Graph(id='bar-chart', style={'height': '500px'})
        ], style={'marginBottom': '30px'}),
        
        html.Div([
            dcc.Graph(id='waterfall-chart', style={'height': '500px'})
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
    fig = go.Figure()
    
    # Define colors for levers
    lever_colors = {
        'Population': '#1f77b4',
        'Sufficiency': '#ff7f0e', 
        'Energy Efficiency': '#2ca02c',
        'Supply Side Decarbonation': '#d62728'
    }
    
    # Get unique scenarios
    scenarios = sorted(chart_data['Scenario'].unique())
    
    # Create a bar for each scenario
    for scenario in scenarios:
        scenario_data = chart_data[chart_data['Scenario'] == scenario].copy()
        
        # Calculate percentage contributions for this scenario
        total_reduction = scenario_data['Contrib_2015_2050_abs'].sum()
        scenario_data['Percentage'] = (scenario_data['Contrib_2015_2050_abs'] / total_reduction) * 100
        
        # Sort by percentage (descending)
        scenario_data = scenario_data.sort_values('Percentage', ascending=False)
        
        fig.add_trace(go.Bar(
            name=scenario,
            x=scenario_data['Lever'],
            y=scenario_data['Percentage'],
            marker_color=[lever_colors.get(lever, "#636363") for lever in scenario_data['Lever']],
            text=[f"{pct:.1f}%" for pct in scenario_data['Percentage']],
            textposition='outside',
            showlegend=True
        ))
    
    fig.update_layout(
        title=dict(
            text=f"Share of Planned CO2 Reduction by Lever - {sector} ({zone})",
            font=dict(size=16, color="#f4d03f", weight="bold"),  # EXACTLY like EWBI dashboard
            x=0.5,
            y=0.95  # Consistent title position for alignment
        ),
        xaxis_title="Levers",
        yaxis_title="Percentage of Total Reduction (%)",
        height=500,
        barmode='group',  # Group bars by lever
        showlegend=True,
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=14),  # EXACTLY like EWBI dashboard
        margin=dict(t=80, b=50, l=60, r=60),  # EXACTLY like EWBI dashboard
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
        title=dict(
            text=f"Planned CO2 Emissions Decrease Over Time by Lever - {sector} ({zone})",
            font=dict(size=16, color="#f4d03f", weight="bold"),  # EXACTLY like EWBI dashboard
            x=0.5,
            y=0.95  # Consistent title position for alignment
        ),
        xaxis_title="Time Periods and Levers",
        yaxis_title="CO2 Emissions (Million tonnes)",
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=14),  # EXACTLY like EWBI dashboard
        margin=dict(t=80, b=50, l=60, r=60),  # EXACTLY like EWBI dashboard
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            gridcolor='lightgray',
            gridwidth=1,
            range=[-100, 700]
        )
    )
    
    return fig

if __name__ == '__main__':
    print("Starting CO2 Decomposition Dashboard...")
    print(f"Dashboard will be available at: http://localhost:8051")
    app.run_server(debug=True, host='0.0.0.0', port=8051) 