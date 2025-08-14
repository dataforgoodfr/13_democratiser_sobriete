import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- 1. Load Data ---
try:
    # Load EWBI results
    ewbi_results = pd.read_csv('output/ewbi_results.csv')
    print("Data loaded successfully.")
    
    # Load EU priorities for more detailed analysis
    eu_priorities = pd.read_csv('output/eu_priorities.csv')
    print("EU priorities loaded successfully.")
    
    # Load secondary indicators
    secondary_indicators = pd.read_csv('output/secondary_indicators.csv')
    print("Secondary indicators loaded successfully.")
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# --- 2. Data Preprocessing ---
# Get unique countries and deciles
countries = sorted(ewbi_results['country'].unique())
deciles = sorted(ewbi_results['decile'].unique())

# Get unique EU priorities
eu_priority_names = sorted(eu_priorities['eu_priority'].unique())

# Get unique secondary indicators
secondary_names = sorted(secondary_indicators['secondary_indicator'].unique())

# --- 3. Dashboard Layout ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("European Well-Being Indicators (EWBI) Dashboard", 
             style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    # Filters
    html.Div([
        html.Div([
            html.Label("Select Country", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': country} for country in countries],
                value=countries[0] if countries else None,
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("Select Decile", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='decile-dropdown',
                options=[{'label': f'Decile {d}', 'value': d} for d in deciles],
                value=deciles[0] if deciles else None,
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'})
    ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'border': '1px solid #dee2e6', 'borderRadius': '5px'}),

    # Visualizations
    html.Div([
        # Level 1: EWBI Overview by Decile
        html.Div([
            html.H3("EWBI Scores by Decile", style={'textAlign': 'center'}),
            dcc.Graph(id='ewbi-by-decile')
        ], style={'marginTop': '20px'}),
        
        # Level 2: EU Priorities Breakdown
        html.Div([
            html.H3("EU Priorities Breakdown", style={'textAlign': 'center'}),
            dcc.Graph(id='eu-priorities-breakdown')
        ], style={'marginTop': '20px'}),
        
        # Level 3: Secondary Indicators
        html.Div([
            html.H3("Secondary Indicators", style={'textAlign': 'center'}),
            dcc.Graph(id='secondary-indicators')
        ], style={'marginTop': '20px'})
    ])
])

# --- 4. Callbacks ---
@app.callback(
    dash.dependencies.Output('ewbi-by-decile', 'figure'),
    [dash.dependencies.Input('country-dropdown', 'value')]
)
def update_ewbi_by_decile(selected_country):
    if not selected_country:
        return go.Figure()
    
    # Filter data for selected country
    country_data = ewbi_results[ewbi_results['country'] == selected_country]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'Decile {d}' for d in country_data['decile']],
        y=country_data.iloc[:, 2],  # EWBI score column
        name='EWBI Score',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f'EWBI Scores by Decile for {selected_country}',
        xaxis_title='Decile',
        yaxis_title='EWBI Score',
        yaxis_range=[0, 1]
    )
    
    return fig

@app.callback(
    dash.dependencies.Output('eu-priorities-breakdown', 'figure'),
    [dash.dependencies.Input('country-dropdown', 'value'),
     dash.dependencies.Input('decile-dropdown', 'value')]
)
def update_eu_priorities(selected_country, selected_decile):
    if not selected_country or not selected_decile:
        return go.Figure()
    
    # Filter data for selected country and decile
    filtered_data = eu_priorities[
        (eu_priorities['country'] == selected_country) & 
        (eu_priorities['decile'] == selected_decile)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered_data['eu_priority'],
        y=filtered_data.iloc[:, 3],  # Score column
        name='Priority Score',
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title=f'EU Priorities for {selected_country} - Decile {selected_decile}',
        xaxis_title='EU Priority',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        xaxis_tickangle=-45
    )
    
    return fig

@app.callback(
    dash.dependencies.Output('secondary-indicators', 'figure'),
    [dash.dependencies.Input('country-dropdown', 'value'),
     dash.dependencies.Input('decile-dropdown', 'value')]
)
def update_secondary_indicators(selected_country, selected_decile):
    if not selected_country or not selected_decile:
        return go.Figure()
    
    # Filter data for selected country and decile
    filtered_data = secondary_indicators[
        (secondary_indicators['country'] == selected_country) & 
        (secondary_indicators['decile'] == selected_decile)
    ]
    
    # Group by EU priority and get average scores
    priority_avg = filtered_data.groupby('eu_priority')['0'].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=priority_avg['eu_priority'],
        y=priority_avg['0'],
        name='Average Secondary Score',
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title=f'Secondary Indicators by EU Priority for {selected_country} - Decile {selected_decile}',
        xaxis_title='EU Priority',
        yaxis_title='Average Score',
        yaxis_range=[0, 1],
        xaxis_tickangle=-45
    )
    
    return fig

# --- 5. Run the App ---
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
