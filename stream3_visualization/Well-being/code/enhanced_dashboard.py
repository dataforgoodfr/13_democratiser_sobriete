import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- 1. Load Enhanced Data ---
try:
    # Load the enhanced master dataframe with decile aggregates
    master_df = pd.read_csv('../output/master_dataframe_with_decile_aggregates.csv')
    print("Enhanced master data loaded successfully.")
    
    # Load time series data
    time_series_df = pd.read_csv('../output/master_dataframe_time_series_with_decile_aggregates.csv')
    print("Time series data loaded successfully.")
    
    # Filter to only "All Deciles" for country-level analysis
    all_deciles_df = master_df[master_df['decile'] == 'All Deciles'].copy()
    print(f"All Deciles data shape: {all_deciles_df.shape}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# --- 2. Data Preprocessing ---
# Get unique countries and indicators
countries = sorted(all_deciles_df['country'].unique())
primary_indicators = sorted(all_deciles_df['primary_index'].unique())

# Get unique EU priorities (extract from column names)
eu_priority_columns = [col for col in all_deciles_df.columns if col.startswith('eu_priority_')]
eu_priority_names = [col.replace('eu_priority_', '') for col in eu_priority_columns]

# Get unique secondary indicators (extract from column names)
secondary_columns = [col for col in all_deciles_df.columns if 'Agriculture' in col or 'Health' in col or 'Education' in col or 'Social' in col or 'Environment' in col or 'Governance' in col or 'Infrastructure' in col]
secondary_names = list(set([col.split('_')[0] + '_' + col.split('_')[1] for col in secondary_columns if '_' in col]))

print(f"Countries: {len(countries)}")
print(f"Primary indicators: {len(primary_indicators)}")
print(f"EU priorities: {len(eu_priority_names)}")
print(f"Secondary indicators: {len(secondary_names)}")

# --- 3. Enhanced Dashboard Layout ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Enhanced European Well-Being Indicators (EWBI) Dashboard", 
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
            html.Label("Select Year", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(year), 'value': year} for year in [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004]],
                value=2023,
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'})
    ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'border': '1px solid #dee2e6', 'borderRadius': '5px'}),

    # Visualizations
    html.Div([
        # Level 1: EWBI Overview (Map-style country comparison)
        html.Div([
            html.H3("Level 1: EWBI Scores - Country Comparison", style={'textAlign': 'center'}),
            dcc.Graph(id='ewbi-country-comparison')
        ], style={'marginTop': '20px'}),
        
        # Level 2: EU Priorities - Country Comparison Chart
        html.Div([
            html.H3("Level 2: EU Priorities - Country Comparison", style={'textAlign': 'center'}),
            dcc.Graph(id='eu-priorities-country-comparison')
        ], style={'marginTop': '20px'}),
        
        # Level 3: Secondary Indicators - Country Comparison Chart
        html.Div([
            html.H3("Level 3: Secondary Indicators - Country Comparison", style={'textAlign': 'center'}),
            dcc.Graph(id='secondary-indicators-country-comparison')
        ], style={'marginTop': '20px'}),
        
        # Level 4: Primary Indicators - Country Comparison Chart
        html.Div([
            html.H3("Level 4: Primary Indicators - Country Comparison", style={'textAlign': 'center'}),
            dcc.Graph(id='primary-indicators-country-comparison')
        ], style={'marginTop': '20px'}),
        
        # Decile Breakdown for Selected Country
        html.Div([
            html.H3("Decile Breakdown for Selected Country", style={'textAlign': 'center'}),
            dcc.Graph(id='decile-breakdown')
        ], style={'marginTop': '20px'})
    ])
])

# --- 4. Enhanced Callbacks ---

@app.callback(
    dash.dependencies.Output('ewbi-country-comparison', 'figure'),
    [dash.dependencies.Input('year-dropdown', 'value')]
)
def update_ewbi_country_comparison(selected_year):
    if not selected_year:
        return go.Figure()
    
    # Get EWBI scores for all countries for the selected year
    year_data = all_deciles_df[['country', str(selected_year)]].copy()
    year_data = year_data.dropna()
    
    # Sort by score for better visualization
    year_data = year_data.sort_values(str(selected_year), ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=year_data['country'],
        y=year_data[str(selected_year)],
        name='EWBI Score',
        marker_color='lightblue',
        text=[f'{score:.3f}' for score in year_data[str(selected_year)]],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'EWBI Scores by Country - {selected_year} (All Deciles)',
        xaxis_title='Country',
        yaxis_title='EWBI Score',
        yaxis_range=[0, 1],
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig

@app.callback(
    dash.dependencies.Output('eu-priorities-country-comparison', 'figure'),
    [dash.dependencies.Input('year-dropdown', 'value')]
)
def update_eu_priorities_country_comparison(selected_year):
    if not selected_year:
        return go.Figure()
    
    # Get EU priorities for all countries for the selected year
    eu_data = []
    for country in countries:
        country_data = all_deciles_df[all_deciles_df['country'] == country]
        if not country_data.empty:
            for priority in eu_priority_names:
                priority_col = f'eu_priority_{priority}_{selected_year}'
                if priority_col in country_data.columns:
                    value = country_data[priority_col].iloc[0]
                    if pd.notna(value):
                        eu_data.append({
                            'country': country,
                            'eu_priority': priority,
                            'value': value
                        })
    
    eu_df = pd.DataFrame(eu_data)
    
    if eu_df.empty:
        return go.Figure()
    
    # Create a heatmap-style visualization
    eu_pivot = eu_df.pivot_table(values='value', index='country', columns='eu_priority')
    
    fig = go.Figure(data=go.Heatmap(
        z=eu_pivot.values,
        x=eu_pivot.columns,
        y=eu_pivot.index,
        colorscale='Viridis',
        text=[[f'{val:.3f}' if pd.notna(val) else '' for val in row] for row in eu_pivot.values],
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=f'EU Priorities by Country - {selected_year} (All Deciles)',
        xaxis_title='EU Priority',
        yaxis_title='Country',
        height=600
    )
    
    return fig

@app.callback(
    dash.dependencies.Output('secondary-indicators-country-comparison', 'figure'),
    [dash.dependencies.Input('year-dropdown', 'value')]
)
def update_secondary_indicators_country_comparison(selected_year):
    if not selected_year:
        return go.Figure()
    
    # Get secondary indicators for all countries for the selected year
    secondary_data = []
    for country in countries:
        country_data = all_deciles_df[all_deciles_df['country'] == country]
        if not country_data.empty:
            for indicator in secondary_names:
                # Look for columns containing the indicator name and year
                matching_cols = [col for col in country_data.columns if indicator in col and str(selected_year) in col]
                if matching_cols:
                    value = country_data[matching_cols[0]].iloc[0]
                    if pd.notna(value):
                        secondary_data.append({
                            'country': country,
                            'indicator': indicator,
                            'value': value
                        })
    
    secondary_df = pd.DataFrame(secondary_data)
    
    if secondary_df.empty:
        return go.Figure()
    
    # Create a heatmap-style visualization
    secondary_pivot = secondary_df.pivot_table(values='value', index='country', columns='indicator')
    
    fig = go.Figure(data=go.Heatmap(
        z=secondary_pivot.values,
        x=secondary_pivot.columns,
        y=secondary_pivot.index,
        colorscale='Plasma',
        text=[[f'{val:.3f}' if pd.notna(val) else '' for val in row] for row in secondary_pivot.values],
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=f'Secondary Indicators by Country - {selected_year} (All Deciles)',
        xaxis_title='Secondary Indicator',
        yaxis_title='Country',
        height=600
    )
    
    return fig

@app.callback(
    dash.dependencies.Output('primary-indicators-country-comparison', 'figure'),
    [dash.dependencies.Input('year-dropdown', 'value')]
)
def update_primary_indicators_country_comparison(selected_year):
    if not selected_year:
        return go.Figure()
    
    # Get primary indicators for all countries for the selected year
    primary_data = []
    for country in countries:
        country_data = all_deciles_df[all_deciles_df['country'] == country]
        if not country_data.empty:
            for indicator in primary_indicators[:20]:  # Limit to first 20 for readability
                if str(selected_year) in country_data.columns:
                    value = country_data[str(selected_year)].iloc[0]
                    if pd.notna(value):
                        primary_data.append({
                            'country': country,
                            'indicator': indicator,
                            'value': value
                        })
    
    primary_df = pd.DataFrame(primary_data)
    
    if primary_df.empty:
        return go.Figure()
    
    # Create a heatmap-style visualization
    primary_pivot = primary_df.pivot_table(values='value', index='country', columns='indicator')
    
    fig = go.Figure(data=go.Heatmap(
        z=primary_pivot.values,
        x=primary_pivot.columns,
        y=primary_pivot.index,
        colorscale='RdYlBu',
        text=[[f'{val:.3f}' if pd.notna(val) else '' for val in row] for row in primary_pivot.values],
        texttemplate='%{text}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title=f'Primary Indicators by Country - {selected_year} (All Deciles)',
        xaxis_title='Primary Indicator',
        yaxis_title='Country',
        height=600
    )
    
    return fig

@app.callback(
    dash.dependencies.Output('decile-breakdown', 'figure'),
    [dash.dependencies.Input('country-dropdown', 'value'),
     dash.dependencies.Input('year-dropdown', 'value')]
)
def update_decile_breakdown(selected_country, selected_year):
    if not selected_country or not selected_year:
        return go.Figure()
    
    # Get all deciles for the selected country and year
    country_data = master_df[
        (master_df['country'] == selected_country) & 
        (master_df['primary_index'] == primary_indicators[0])  # Use first indicator as example
    ]
    
    if country_data.empty:
        return go.Figure()
    
    # Get decile values for the selected year
    decile_values = []
    decile_labels = []
    
    for decile in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'All Deciles']:
        decile_data = country_data[country_data['decile'] == decile]
        if not decile_data.empty and str(selected_year) in decile_data.columns:
            value = decile_data[str(selected_year)].iloc[0]
            if pd.notna(value):
                decile_values.append(value)
                decile_labels.append(f'Decile {decile}' if decile != 'All Deciles' else 'All Deciles')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=decile_labels,
        y=decile_values,
        name='Score',
        marker_color='lightcoral',
        text=[f'{score:.3f}' for score in decile_values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Decile Breakdown for {selected_country} - {selected_year}',
        xaxis_title='Decile',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        height=400
    )
    
    return fig

# --- 5. Run the Enhanced App ---
if __name__ == '__main__':
    print("Starting Enhanced EWBI Dashboard...")
    print(f"Available countries: {len(countries)}")
    print(f"Available indicators: {len(primary_indicators)}")
    print(f"Available EU priorities: {len(eu_priority_names)}")
    app.run_server(debug=True, port=8052) 