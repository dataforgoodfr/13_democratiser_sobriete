import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import json
import os

# --- 1. Load Data ---
# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths to data files relative to the script's location
DATA_DIR = os.path.abspath(os.path.join(script_dir, '..', 'output'))
JSON_PATH = os.path.abspath(os.path.join(script_dir, '..', 'data', 'ewbi_indicators.json'))

# Load the datasets
try:
    ewbi_results = pd.read_csv(os.path.join(DATA_DIR, 'ewbi_results.csv'))
    eu_priorities = pd.read_csv(os.path.join(DATA_DIR, 'eu_priorities.csv'))
    secondary_indicators = pd.read_csv(os.path.join(DATA_DIR, 'secondary_indicators.csv'))
    # primary_indicators = pd.read_csv(os.path.join(DATA_DIR, 'primary_data_preprocessed.csv')) # This file is large, load if needed
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    # Initialize empty dataframes if files are not found
    ewbi_results = pd.DataFrame()
    eu_priorities = pd.DataFrame()
    secondary_indicators = pd.DataFrame()
    # primary_indicators = pd.DataFrame()

# Load the indicator hierarchy
try:
    with open(JSON_PATH, 'r') as f:
        indicator_hierarchy = json.load(f)
    print("Indicator hierarchy loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading indicator hierarchy: {e}")
    indicator_hierarchy = {}

# --- 2. Initialize the App ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- 3. Define the Layout ---
# Get years from data for slider
years = [col for col in ewbi_results.columns if col.isdigit()]
year_marks = {int(year): str(year) for year in years}

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("European Well-Being Indicators Dashboard", style={'textAlign': 'center', 'color': '#2c3e50', 'fontFamily': 'Arial, sans-serif'}),
        html.H2("Visualizing Well-Being Across EU Countries and Demographics", style={'textAlign': 'center', 'color': '#34495e', 'fontSize': '1.2rem', 'fontFamily': 'Arial, sans-serif'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px'}),

    # Filters
    html.Div([
        html.Div([
            html.Label("Select Country", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': iso} for country, iso in zip(ewbi_results['Country'], ewbi_results['ISO3'])],
                value='EU27',  # Default value
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("Select Year", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='year-slider',
                min=int(years[0]),
                max=int(years[-1]),
                value=int(years[-1]),
                marks=year_marks,
                step=None
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'})
    ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'border': '1px solid #dee2e6', 'borderRadius': '5px'}),

    # Visualizations
    html.Div([
        # Level 1: EWBI Overview
        html.Div([
            dcc.Graph(id='ewbi-map'),
            dcc.Graph(id='ewbi-timeseries')
        ], style={'display': 'flex', 'marginTop': '20px'})
    ])
])

# --- 4. Callbacks ---
@app.callback(
    dash.dependencies.Output('ewbi-map', 'figure'),
    [dash.dependencies.Input('year-slider', 'value')]
)
def update_map(selected_year):
    # Melt the dataframe to have years in a column
    df_melted = ewbi_results.melt(id_vars=['ISO3', 'Country'], 
                                  value_vars=[str(y) for y in years],
                                  var_name='Year', 
                                  value_name='EWBI')
    df_melted['Year'] = pd.to_numeric(df_melted['Year'])
    
    # Filter by selected year
    df_filtered = df_melted[df_melted['Year'] == selected_year]
    
    fig = px.choropleth(df_filtered, 
                        locations="ISO3",
                        color="EWBI", 
                        hover_name="Country",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        scope="europe",
                        title=f"EWBI Scores in {selected_year}")
    
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='mercator'
        )
    )
    
    return fig

@app.callback(
    dash.dependencies.Output('ewbi-timeseries', 'figure'),
    [dash.dependencies.Input('country-dropdown', 'value')]
)
def update_timeseries(selected_country_iso):
    # Melt the dataframe
    df_melted = ewbi_results.melt(id_vars=['ISO3', 'Country'], 
                                  value_vars=[str(y) for y in years],
                                  var_name='Year', 
                                  value_name='EWBI')
    df_melted['Year'] = pd.to_numeric(df_melted['Year'])

    # Filter for the selected country
    df_country = df_melted[df_melted['ISO3'] == selected_country_iso]
    country_name = df_country['Country'].iloc[0]

    fig = px.line(df_country, x="Year", y="EWBI", title=f"EWBI Trend for {country_name}")
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="EWBI Score"
    )

    return fig


# --- 5. Run the App ---
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
