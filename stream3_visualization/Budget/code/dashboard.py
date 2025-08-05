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
    scenario_parameters = pd.read_csv(os.path.join(DATA_DIR, 'scenario_parameters.csv'))
    forecast_data = pd.read_csv(os.path.join(DATA_DIR, 'forecast_data.csv'))
    historical_data = pd.read_csv(os.path.join(DATA_DIR, 'combined_data.csv'))
    print("All data files loaded successfully.")

    # Filter out 'Population' scenario from the entire dataset
    scenario_parameters = scenario_parameters[scenario_parameters['Budget_distribution_scenario'] != 'Population'].copy()
    
    # Check column availability and create ISO mappings
    print("\nColumn check:")
    print(f"scenario_parameters columns: {scenario_parameters.columns.tolist()}")
    print(f"forecast_data columns: {forecast_data.columns.tolist()}")
    print(f"historical_data columns: {historical_data.columns.tolist()}")
    
    # Create ISO2 to ISO3 mapping from historical_data (which has both)
    if 'ISO2' in historical_data.columns and 'ISO3' in historical_data.columns:
        iso_mapping = historical_data[['ISO2', 'ISO3']].drop_duplicates()
        iso2_to_iso3 = dict(zip(iso_mapping['ISO2'], iso_mapping['ISO3']))
        iso3_to_iso2 = dict(zip(iso_mapping['ISO3'], iso_mapping['ISO2']))
        print(f"Created ISO mapping with {len(iso2_to_iso3)} entries from historical_data")
        
        # Add missing ISO3 to scenario_parameters
        if 'ISO3' not in scenario_parameters.columns and 'ISO2' in scenario_parameters.columns:
            scenario_parameters['ISO3'] = scenario_parameters['ISO2'].map(iso2_to_iso3)
            print("Added ISO3 to scenario_parameters")
            
        # Add missing ISO codes to forecast_data (it has neither)
        if 'ISO2' not in forecast_data.columns and 'ISO3' not in forecast_data.columns:
            # forecast_data doesn't have country info directly, we'll need to merge it later
            print("forecast_data has no ISO columns - will need to merge from scenario_parameters")
        elif 'ISO3' not in forecast_data.columns and 'ISO2' in forecast_data.columns:
            forecast_data['ISO3'] = forecast_data['ISO2'].map(iso2_to_iso3)
            print("Added ISO3 to forecast_data")
        elif 'ISO2' not in forecast_data.columns and 'ISO3' in forecast_data.columns:
            forecast_data['ISO2'] = forecast_data['ISO3'].map(iso3_to_iso2)
            print("Added ISO2 to forecast_data")
            
    else:
        print("WARNING: Could not create ISO mapping - historical_data missing ISO2 or ISO3 columns")
        
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
        html.H1("Zero Carbon For All: A Fair and Inclusive Timeline", 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'fontSize': '2.5rem',
                    'fontWeight': 'bold',
                    'marginBottom': '10px',
                    'fontFamily': 'Arial, sans-serif'
                }),
        html.H2("Distributing our remaining global carbon budget to stay within 1.5°C in a fair and inclusive way implies that most developed countries have overshot their budget", 
                style={
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
                html.Label("Country", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': 'All Countries', 'value': 'ALL'}] + 
                            [{'label': f"{country} ({iso2})", 'value': iso2} 
                             for country, iso2 in scenario_parameters[
                                 (~scenario_parameters['ISO2'].isin(['WLD', 'EU', 'G20'])) &
                                 (scenario_parameters['Country'] != 'All') &
                                 (scenario_parameters['ISO2'].notna())
                             ][['Country', 'ISO2']].drop_duplicates().sort_values('Country').values],
                    value='ALL',
                    style={'marginTop': '8px'}
                )
            ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),

            html.Div([
                html.Label("Budget Distribution", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='budget-distribution-dropdown',
                    options=[{'label': i, 'value': i} for i in scenario_parameters['Budget_distribution_scenario'].unique()],
                    value='Responsibility',
                    style={'marginTop': '8px'}
                )
            ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.Label("Probability", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='probability-dropdown',
                    options=[{'label': i, 'value': i} for i in scenario_parameters['Probability_of_reach'].unique()],
                    value='50%',
                    style={'marginTop': '8px'}
                )
            ], style={'width': '23%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.Label("Emissions Scope", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='emissions-scope-dropdown',
                    options=[{'label': i, 'value': i} for i in scenario_parameters['Emissions_scope'].unique()],
                    value='Territory',
                    style={'marginTop': '8px'}
                )
            ], style={'width': '23%', 'display': 'inline-block'}),
        ], style={
            'padding': '15px 20px',
            'backgroundColor': '#fdf6e3',
            'borderRadius': '8px',
            'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
            'margin': '0 20px 10px 20px' # Added some margin for spacing
        })
    ], style={
        'backgroundColor': '#f4d03f',  # World Sufficiency Lab yellow
        'padding': '25px 0px 15px 0px', # Adjusted padding
        'position': 'sticky',
        'top': 0,
        'zIndex': 1000,
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Visualizations
    html.Div([
        dcc.Graph(id='world-map'),
        html.Div([
            dcc.Graph(id='scenario-comparison-bar', style={'display': 'inline-block', 'width': '49%'}),
            dcc.Graph(id='emissions-trajectory-line', style={'display': 'inline-block', 'width': '49%'})
        ])
    ], style={
        'margin': '0 20px',
        'paddingTop': '20px'  # Extra space to account for sticky headers
    }),
    
    # Sources section
    html.Div([
        html.H3("Data Sources", style={
            'color': '#2c3e50',
            'fontSize': '1.8rem',
            'fontWeight': 'bold',
            'marginBottom': '20px',
            'textAlign': 'center'
        }),
        
        html.Div([
            html.H4("Historical CO2 Emissions", style={
                'color': '#34495e',
                'fontSize': '1.2rem',
                'fontWeight': 'bold',
                'marginBottom': '10px'
            }),
            html.P("Historical CO2 emissions from 1970 to 2022 for Consumption and from 1970 to 2023 for Territory:", style={
                'marginBottom': '10px',
                'lineHeight': '1.6'
            }),
            html.P([
                html.Strong("Territory: "),
                "Andrew, R. M., & Peters, G. P. (2024). The Global Carbon Project's fossil CO2 emissions dataset (2024v18) [Data set]. Zenodo. ",
                html.A("https://doi.org/10.5281/zenodo.14106218", 
                       href="https://doi.org/10.5281/zenodo.14106218", 
                       target="_blank",
                       style={'color': '#f39c12'})
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
            html.P([
                html.Strong("Consumption: "),
                "Fanning, A., & Hickel, J. (2023). a-fanning/compensation-atmospheric-appropriation: First release (v1.0.0). Zenodo. ",
                html.A("https://doi.org/10.5281/zenodo.7779453", 
                       href="https://doi.org/10.5281/zenodo.7779453", 
                       target="_blank",
                       style={'color': '#f39c12'})
            ], style={'marginBottom': '20px', 'lineHeight': '1.6'}),
            
            html.H4("Budget Distribution and Calculations", style={
                'color': '#34495e',
                'fontSize': '1.2rem',
                'fontWeight': 'bold',
                'marginBottom': '10px'
            }),
            html.P("Budget distribution definitions and calculations of neutrality year, years to neutrality and emissions trajectories made by the World Sufficiency Lab based on:", style={
                'marginBottom': '10px',
                'lineHeight': '1.6'
            }),
            html.P([
                html.Strong("Carbon Budgets: "),
                "Carbon budgets from 2023 to stay within 1.5°C with probabilities of 50% and 67% from Lamboll (2023) Assessing the size and uncertainty of remaining carbon budgets. ",
                html.A("https://www.nature.com/articles/s41558-023-01848-5", 
                       href="https://www.nature.com/articles/s41558-023-01848-5", 
                       target="_blank",
                       style={'color': '#f39c12'})
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
            html.P([
                html.Strong("Population Data: "),
                "Historical and forecasted population from United Nations data"
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
            html.P([
                html.Strong("GDP Data: "),
                "GDP Purchasing Power Parity based on World Bank data in constant 2021 US$"
            ], style={'marginBottom': '20px', 'lineHeight': '1.6'}),
            
            html.H4("Scenario Definitions", style={
                'color': '#34495e',
                'fontSize': '1.2rem',
                'fontWeight': 'bold',
                'marginBottom': '10px'
            }),
            html.P([
                html.Strong('"NDC Pledges" '),
                "(Nationally Determined Contributions) are official targets set by the countries, when available, compiled by Climate Analytics and NewClimate Institute in 2022: ",
                html.A("CAT net zero target evaluations | Climate Action Tracker", 
                       href="https://climateactiontracker.org/countries/", 
                       target="_blank",
                       style={'color': '#f39c12'})
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
            html.P([
                html.Strong('"Responsibility" '),
                "means taking into account each country's cumulative emissions when available and allocating the total carbon budget based on share of population from 1970 to 2050"
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
            html.P([
                html.Strong('"Capacity" '),
                "means taking into account each country's cumulative population and GDP per capita PPP from 1970 to the latest year available"
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
        ])
    ], style={
        'margin': '40px 20px 20px 20px',
        'padding': '30px',
        'backgroundColor': '#fdf6e3',  # Light cream background like filters
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'fontSize': '0.9rem',
        'color': '#2c3e50'
    })
], style={
    'backgroundColor': '#ffffff',  # Clean white background
    'minHeight': '100vh',
    'fontFamily': 'Arial, sans-serif'
})

# Callback for the world map
@app.callback(
    Output('world-map', 'figure'),
    [Input('budget-distribution-dropdown', 'value'),
     Input('probability-dropdown', 'value'),
     Input('emissions-scope-dropdown', 'value'),
     Input('country-dropdown', 'value')]
)
def update_map(budget_dist, probability, emissions_scope, selected_country):
    # Filter data for the entire scenario to establish a consistent color scale
    all_countries_data = scenario_parameters[
        (scenario_parameters['Budget_distribution_scenario'] == budget_dist) &
        (scenario_parameters['Probability_of_reach'] == probability) &
        (scenario_parameters['Emissions_scope'] == emissions_scope) &
        (scenario_parameters['Warming_scenario'] == '1.5°C') &
        (~scenario_parameters['ISO2'].isin(['WLD', 'EU', 'G20']))  # Exclude aggregates
    ].copy()

    # Convert 'Neutrality_year' to numeric for color scaling
    all_countries_data['Neutrality_year_numeric'] = pd.to_numeric(all_countries_data['Neutrality_year'], errors='coerce')
    
    # Determine the color range from the full dataset for the scenario
    color_range_min = all_countries_data['Neutrality_year_numeric'].min()
    color_range_max = all_countries_data['Neutrality_year_numeric'].max()

    # Select data for plotting based on dropdown
    if selected_country != 'ALL':
        plot_data = all_countries_data[all_countries_data['ISO2'] == selected_country]
    else:
        plot_data = all_countries_data

    fig = px.choropleth(
        plot_data,
        locations="ISO3",
        locationmode='ISO-3',
        color="Neutrality_year_numeric",
        hover_name="Country",
        hover_data={
            "Neutrality_year": False,
            "Years_to_neutrality_from_today": False,
            "Latest_annual_CO2_emissions_Mt": False,
            "Latest_cumulative_CO2_emissions_Mt": False, 
            "Latest_emissions_per_capita_t": False,
            "Country_carbon_budget": False,
            "ISO3": False
        },
        color_continuous_scale="RdYlGn",
        range_color=[color_range_min, color_range_max],  # Apply consistent color scale
        title="Year by which countries need to reach Zero Carbon by Scenario",
        labels={
            'Neutrality_year_numeric': 'Neutrality Year',
            'Neutrality_year': 'Neutrality Year',
            'Years_to_neutrality_from_today': 'Years from Today',
            'Latest_annual_CO2_emissions_Mt': 'Annual Emissions (Mt)',
            'Latest_cumulative_CO2_emissions_Mt': 'Cumulative Emissions (Mt)',
            'Latest_emissions_per_capita_t': 'Emissions Per Capita (tonnes)',
            'Country_carbon_budget': 'Carbon Budget (Mt)'
        }
    )
    
    # Update traces for better styling and custom hover with spaces
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.5,
        hovertemplate="<b>%{hovertext}</b><br>" +
                     "Neutrality Year = %{customdata[0]}<br>" +
                     "Years from Today = %{customdata[1]}<br>" +
                     "Annual Emissions (Mt) = %{customdata[2]:.1f}<br>" +
                     "Cumulative Emissions (Mt) = %{customdata[3]:.1f}<br>" +
                     "Emissions Per Capita (tonnes) = %{customdata[4]:.2f}<br>" +
                     "Carbon Budget (Mt) = %{customdata[5]:.1f}<extra></extra>",
        customdata=plot_data[[
            'Neutrality_year', 'Years_to_neutrality_from_today',
            'Latest_annual_CO2_emissions_Mt', 'Latest_cumulative_CO2_emissions_Mt',
            'Latest_emissions_per_capita_t', 'Country_carbon_budget'
        ]].values
    )
    
    # Ensure grey (no data) countries also have white borders
    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='white',  # White coastlines
        projection_type='equirectangular',
        bgcolor='white',  # Background color of the map area
        landcolor='lightgray',  # Color for countries with no data
        framecolor='white',  # Frame color
        showlakes=False  # Hide lakes for cleaner look
    )
    
    # Also update layout to ensure consistent white borders for all countries
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor='lightgray',
            showocean=True,
            oceancolor='white',
            showlakes=False,
            showrivers=False
        )
    )
    
    # Make map bigger and position it to the left
    fig.update_layout(
        height=500,  # Smaller map
        margin={"r": 100, "t": 50, "l": 50, "b": 50},  # More space on right, less on left
        geo=dict(
            projection_scale=1.2  # Slightly zoom in
        ),
        # Style to match World Sufficiency Lab theme
        paper_bgcolor='#ffffff',  # Clean white background
        plot_bgcolor='#ffffff',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#2c3e50"
        ),
        title=dict(
            font=dict(size=20, color="#f4d03f", weight="bold"),  # Bold, smaller, yellow like top ribbon
            x=0.5
        ),
        # Lighter hover box styling
        hoverlabel=dict(
            bgcolor="rgba(245, 245, 245, 0.9)",  # Light grey background for hover
            bordercolor="white",
            font=dict(color="black", size=12)
        )
    )
    
    return fig

# Callback for the scenario comparison bar chart
@app.callback(
    Output('scenario-comparison-bar', 'figure'),
    [Input('probability-dropdown', 'value'),
     Input('emissions-scope-dropdown', 'value'),
     Input('country-dropdown', 'value')]
)
def update_bar_chart(probability, emissions_scope, selected_country):
    # Filter data based on dropdown selections
    filtered_data = scenario_parameters[
        (scenario_parameters['Probability_of_reach'] == probability) &
        (scenario_parameters['Emissions_scope'] == emissions_scope) &
        (scenario_parameters['Warming_scenario'] == '1.5°C')
    ].copy()
    
    # Apply country filter
    if selected_country != 'ALL':
        filtered_data = filtered_data[filtered_data['ISO2'] == selected_country]
        chart_title = f'Years to Neutrality by CO2 Budget Distribution Scenario - {filtered_data.iloc[0]["Country"] if len(filtered_data) > 0 else selected_country}'
    else:
        # Exclude aggregates for global analysis
        filtered_data = filtered_data[~filtered_data['ISO2'].isin(['WLD', 'EU', 'G20'])]
        chart_title = 'Years to Neutrality by Global CO2 Budget Distribution Scenario'

    # Convert 'Years_to_neutrality_from_today' to numeric, coercing errors
    filtered_data['Years_to_neutrality_numeric'] = pd.to_numeric(filtered_data['Years_to_neutrality_from_today'], errors='coerce')

    if selected_country != 'ALL':
        # For single country, show individual scenario values
        fig = px.bar(
            filtered_data,
            x='Budget_distribution_scenario',
            y='Years_to_neutrality_numeric',
            title=chart_title,
            labels={'Years_to_neutrality_numeric': 'Years to Neutrality'},
            color_discrete_sequence=['#f39c12']  # World Sufficiency Lab orange accent
        )
        
        # Add consistent styling to match theme
        fig.update_layout(
            paper_bgcolor='#ffffff',  # Clean white background
            plot_bgcolor='#ffffff',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#2c3e50"
            ),
            title=dict(
                font=dict(size=24, color="#f4d03f", weight="bold"),  # Bold, bigger, yellow like top ribbon
                x=0.5
            ),
            height=500
        )
    else:
        # For all countries, show min, max, mean
        summary_data = filtered_data.groupby('Budget_distribution_scenario')['Years_to_neutrality_numeric'].agg(['min', 'max', 'mean']).reset_index()
        
        fig = px.bar(
            summary_data,
            x='Budget_distribution_scenario',
            y='mean',
            error_y=summary_data['max'] - summary_data['mean'],
            error_y_minus=summary_data['mean'] - summary_data['min'],
            title=chart_title,
            labels={'mean': 'Average Years to Neutrality'},
            color_discrete_sequence=['#f39c12']  # World Sufficiency Lab orange accent
        )
    
    # Add consistent styling to match theme
    fig.update_layout(
        paper_bgcolor='#ffffff',  # Clean white background
        plot_bgcolor='#ffffff',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#2c3e50"
        ),
        title=dict(
            font=dict(size=20, color="#f4d03f", weight="bold"),  # Bold, smaller, yellow like top ribbon
            x=0.5
        ),
        height=500,
        yaxis_title="Average, Minimum and Maximum Years to Neutrality",
        xaxis_title=None
    )
    
    # Update hover template
    if selected_country == 'ALL':
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br><br>" +
                          "Average Years to Neutrality = %{y:.0f}<br>" +
                          "Maximum Years to Neutrality = %{customdata[0]:.0f}<br>" +
                          "Minimum Years to Neutrality = %{customdata[1]:.0f}<extra></extra>",
            customdata=summary_data[['max', 'min']].values
        )
    else:
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br><br>" +
                          "Years to Neutrality = %{y:.0f}<extra></extra>"
        )
    
    return fig

# Callback for the emissions trajectory line chart
@app.callback(
    Output('emissions-trajectory-line', 'figure'),
    [Input('budget-distribution-dropdown', 'value'),
     Input('probability-dropdown', 'value'),
     Input('emissions-scope-dropdown', 'value'),
     Input('country-dropdown', 'value')]
)
def update_line_chart(budget_dist, probability, emissions_scope, selected_country):
    # Determine which country/region to show
    if selected_country == 'ALL':
        target_iso = 'WLD'
        chart_title = 'Global Historical CO2 Emissions (Mt) and Required Trajectory by Scenario'
    else:
        target_iso = selected_country
        country_name = scenario_parameters[scenario_parameters['ISO2'] == selected_country]['Country'].iloc[0] if len(scenario_parameters[scenario_parameters['ISO2'] == selected_country]) > 0 else selected_country
        chart_title = f'{country_name} - Historical CO2 Emissions (Mt) and Required Trajectory by Scenario'
    
    # Filter historical data (exclude 2050 as it's only for population data)
    hist_data = historical_data[
        (historical_data['ISO2'] == target_iso) & 
        (historical_data['Emissions_scope'] == emissions_scope) &
        (historical_data['Year'] != 2050)  # Filter out 2050
    ]
    
    # Get the correct scenario_id
    scenario_id_row = scenario_parameters[
        (scenario_parameters['Budget_distribution_scenario'] == budget_dist) &
        (scenario_parameters['Probability_of_reach'] == probability) &
        (scenario_parameters['Emissions_scope'] == emissions_scope) &
        (scenario_parameters['ISO2'] == target_iso) &
        (scenario_parameters['Warming_scenario'] == '1.5°C')
    ]
    
    if scenario_id_row.empty:
        return go.Figure().update_layout(title=f"No data available for {chart_title}")

    scenario_id = scenario_id_row.iloc[0]['scenario_id']
    
    forecast_country = forecast_data[
        (forecast_data['scenario_id'] == scenario_id)
    ]

    fig = go.Figure()

    # Historical emissions
    if not hist_data.empty:
        fig.add_trace(go.Scatter(
            x=hist_data['Year'],
            y=hist_data['Annual_CO2_emissions_Mt'],
            mode='lines',
            name='Historical Emissions',
            line=dict(color='#3498db', width=3)  # Nice blue color, thicker line
        ))

    # Forecasted emissions
    if not forecast_country.empty:
        fig.add_trace(go.Scatter(
            x=forecast_country['Year'],
            y=forecast_country['Forecasted_emissions_Mt'],
            mode='lines',
            name='Required Trajectory',
            line=dict(dash='dash', color='#e74c3c', width=3)  # Nice red color, thicker line
        ))
    
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='CO2 Emissions (Mt)',
        hovermode='x unified',
        # Add consistent styling to match theme
        paper_bgcolor='#ffffff',  # Clean white background
        plot_bgcolor='#ffffff',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#2c3e50"
        ),
        title=dict(
            text=chart_title,
            font=dict(size=20, color="#f4d03f", weight="bold"),  # Bold, smaller, yellow like top ribbon
            x=0.5
        ),
        height=500,
        # Style the grid with fixed range
        xaxis=dict(
            gridcolor='rgba(200,200,200,0.3)',  # Light gray grid on white background
            showgrid=True,
            range=[1970, 2100],  # Fixed x-axis range from 1970 to 2100
            title='Year'
        ),
        yaxis=dict(
            gridcolor='rgba(200,200,200,0.3)',  # Light gray grid on white background
            showgrid=False
        )
    )
    
    return fig


# Callback to update country dropdown from map click
@app.callback(
    Output('country-dropdown', 'value'),
    [Input('world-map', 'clickData')]
)
def update_country_from_map(clickData):
    if clickData is None:
        # Prevent update on initial load
        return dash.no_update
    
    # Get ISO3 code from the clicked point
    iso3_code = clickData['points'][0]['location']
    
    # Find the corresponding ISO2 code in the scenario_parameters dataframe
    iso2_code_row = scenario_parameters[scenario_parameters['ISO3'] == iso3_code]
    
    if not iso2_code_row.empty:
        iso2_code = iso2_code_row.iloc[0]['ISO2']
        return iso2_code
    
    # If not found, do not update the dropdown
    return dash.no_update


if __name__ == '__main__':
    app.run(debug=True, port=8057)
