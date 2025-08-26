import subprocess

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

    # Define G20 countries list
    G20_COUNTRIES = [
        'AR', 'AU', 'BR', 'CA', 'CN', 'FR', 'DE', 'IN', 'ID', 'IT',
        'JP', 'MX', 'RU', 'SA', 'ZA', 'KR', 'TR', 'GB', 'US', 'EU'
    ]

    # Filter out 'Population' scenario from the entire dataset
    scenario_parameters = scenario_parameters[
        scenario_parameters['Budget_distribution_scenario'] != 'Population'].copy()

    # Check column availability and create ISO mappings
    print("\nColumn check:")
    print(f"scenario_parameters columns: {scenario_parameters.columns.tolist()}")
    print(f"forecast_data columns: {forecast_data.columns.tolist()}")
    print(f"historical_data columns: {historical_data.columns.tolist()}")

    # Create ISO2 to ISO3 mapping from historical_data (which has both)
    if 'ISO2' in historical_data.columns and 'ISO3' in historical_data.columns:
        iso_mapping = historical_data[['ISO2', 'ISO3']].drop_duplicates()
        iso2_to_iso3 = dict(zip(iso_mapping['ISO2'], iso_mapping['ISO3'], strict=False))
        iso3_to_iso2 = dict(zip(iso_mapping['ISO3'], iso_mapping['ISO2'], strict=False))
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


# Helper function to convert emissions scope values to display labels
def get_scope_display_label(scope_value):
    """Convert internal scope values to user-friendly display labels"""
    scope_mapping = {
        'Territory': 'Territorial',
        'Consumption': 'Consumption-based'
    }
    return scope_mapping.get(scope_value, scope_value)


# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Add headers to allow iframe embedding
@app.server.after_request
def add_header(response):
    # Allow iframe embedding from any domain
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    # More permissive CSP that allows CDN resources
    response.headers['Content-Security-Policy'] = "frame-ancestors 'self' *; default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: https:; script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; style-src 'self' 'unsafe-inline' https:; connect-src 'self' https:;"
    # Additional headers for better compatibility
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# App layout
app.layout = html.Div([
    # Font Awesome for icons
    html.Link(
        rel='stylesheet',
        href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
    ),
    # Header section with hierarchical titles
    html.Div([
        html.H1("Zero Carbon for All",
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'fontSize': '1.9rem',
                    'fontWeight': 'bold',
                    'marginBottom': '6px',
                    'marginTop': '0px',
                    'fontFamily': 'Arial, sans-serif'
                }),
        html.H2([
            "A Sufficiency Timeline To Support the ",
            html.A("ICJ's July 2025 Climate Change AO",
                   href="https://www.icj-cij.org/sites/default/files/case-related/187/187-20250723-adv-01-00-en.pdf",
                   target="_blank",
                   style={'color': '#f39c12', 'textDecoration': 'underline'}),
        ], style={
            'textAlign': 'center',
            'color': '#34495e',
            'fontSize': '1.0rem',
            'fontWeight': 'normal',
            'marginBottom': '20px',
            'fontFamily': 'Arial, sans-serif',
            'lineHeight': '1.3',
            'maxWidth': '1200px',
            'margin': '0 auto 20px auto'
        }),

        # Controls section embedded within the header
        html.Div([
            html.Div([
                html.Label("Country", style={'fontWeight': 'bold', 'color': '#2c3e50', 'fontSize': '0.9rem'}),
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
                    style={'marginTop': '6px', 'fontSize': '0.85rem'},
                    clearable=False
                )
            ], style={'width': '16%', 'display': 'inline-block', 'margin-right': '2.5%'}),
            html.Div([
                html.Label("G20 Only", style={'fontWeight': 'bold', 'color': '#2c3e50', 'fontSize': '0.9rem'}),
                dcc.Dropdown(
                    id='g20-filter-dropdown',
                    options=[
                        {'label': 'No', 'value': 'No'},
                        {'label': 'Yes', 'value': 'Yes'}
                    ],
                    value='No',
                    style={'marginTop': '6px', 'fontSize': '0.85rem'},
                    clearable=False
                )
            ], style={'width': '16%', 'display': 'inline-block', 'margin-right': '2.5%'}),

            html.Div([
                html.Label("Carbon Budget Allocation", style={'fontWeight': 'bold', 'color': '#2c3e50', 'fontSize': '0.9rem'}),
                dcc.Dropdown(
                    id='budget-distribution-dropdown',
                    options=[{'label': i, 'value': i} for i in
                             scenario_parameters['Budget_distribution_scenario'].unique()],
                    value='Responsibility',
                    style={'marginTop': '6px', 'fontSize': '0.85rem'},
                    clearable=False
                )
            ], style={'width': '16%', 'display': 'inline-block', 'margin-right': '2.5%'}),
            html.Div([
                html.Label("Scope of Emissions", style={'fontWeight': 'bold', 'color': '#2c3e50', 'fontSize': '0.9rem'}),
                dcc.Dropdown(
                    id='emissions-scope-dropdown',
                    options=[
                        {'label': 'Territorial', 'value': 'Territory'},
                        {'label': 'Consumption-based', 'value': 'Consumption'}
                    ],
                    value='Territory',
                    style={'marginTop': '6px', 'fontSize': '0.85rem'},
                    clearable=False
                )
            ], style={'width': '16%', 'display': 'inline-block', 'margin-right': '2.5%'}),
            html.Div([
                html.Label("Probability of not exceeding +1.5°C",
                           style={'fontWeight': 'bold', 'color': '#2c3e50', 'whiteSpace': 'nowrap',
                                  'overflow': 'visible', 'fontSize': '0.9rem'}),
                dcc.Dropdown(
                    id='probability-dropdown',
                    options=[{'label': i, 'value': i} for i in scenario_parameters['Probability_of_reach'].unique()],
                    value='50%',
                    style={'marginTop': '6px', 'fontSize': '0.85rem'},
                    clearable=False
                )
            ], style={'width': '16%', 'display': 'inline-block'}),
        ], style={
            'padding': '12px 20px',
            'backgroundColor': '#fdf6e3',
            'borderRadius': '8px',
            'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
            'margin': '0 20px 8px 20px'  # Reduced margin for spacing
        })
    ], style={
        'backgroundColor': '#f4d03f',  # World Sufficiency Lab yellow
        'padding': '15px 0px 12px 0px',  # Further reduced top padding
        'position': 'sticky',
        'top': 0,
        'zIndex': 1000,
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),

    # Visualizations
    html.Div([
        dcc.Graph(id='world-map'),
        html.Div([
            dcc.Graph(id='scenario-comparison-bar',
                      style={'display': 'inline-block', 'width': '49%', 'verticalAlign': 'top'}),
            dcc.Graph(id='emissions-trajectory-line',
                      style={'display': 'inline-block', 'width': '49%', 'verticalAlign': 'top'})
        ], style={'textAlign': 'center', 'margin': '0 auto'}),
        # Top 20 Emitters Charts
        html.Div([
            dcc.Graph(id='top-cumulative-emitters', style={'display': 'inline-block', 'width': '49%'}),
            dcc.Graph(id='top-per-capita-emitters', style={'display': 'inline-block', 'width': '49%'})
        ], style={'marginTop': '20px'})
    ], style={
        'margin': '0 20px',
        'paddingTop': '20px'  # Extra space to account for sticky headers
    }),

    # Collaboration section
    html.Div([
        html.Div([
            # Left side - Text content
            html.Div([
                html.P([
                    "This tool was developed as a collaboration between ",
                    html.Strong("Data for Good"),
                    ", a community of 6000+ tech experts volunteering for general interest projects, and the ",
                    html.Strong("World Sufficiency Lab.")
                ], style={
                    'fontSize': '1.1rem',
                    'lineHeight': '1.6',
                    'color': '#2c3e50',
                    'margin': '0',
                    'padding': '0'
                })
            ], style={
                'display': 'inline-block',
                'width': '60%',
                'verticalAlign': 'top',
                'paddingRight': '40px'
            }),
            
            # Right side - Data for Good logo and links
            html.Div([
                html.Div([
                    html.Img(
                        src="assets/d4g-logo.png",
                        style={
                            'width': '60px',
                            'height': '60px',
                            'objectFit': 'contain'
                        }
                    )
                ], style={
                    'textAlign': 'center',
                    'marginBottom': '15px'
                }),
                html.H4([
                    html.A("Data for Good", 
                           href="https://dataforgood.fr",
                           target="_blank",
                           style={
                               'color': '#2c3e50',
                               'textDecoration': 'none'
                           })
                ], style={
                    'fontSize': '1.1rem',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'margin': '0 0 15px 0'
                }),
                # Links arranged side by side
                html.Div([
                    html.A([
                        html.I(className="fas fa-globe", style={'marginRight': '8px'}),
                        "Website"
                    ], 
                    href="https://dataforgood.fr",
                    target="_blank",
                    style={
                        'color': '#2c3e50',
                        'textDecoration': 'none',
                        'marginRight': '20px',
                        'fontSize': '0.9rem'
                    }),
                    html.A([
                        html.I(className="fab fa-linkedin", style={'marginRight': '8px'}),
                        "LinkedIn"
                    ], 
                    href="https://www.linkedin.com/company/dataforgood/",
                    target="_blank",
                    style={
                        'color': '#2c3e50',
                        'textDecoration': 'none',
                        'fontSize': '0.9rem'
                    })
                ], style={
                    'textAlign': 'center'
                })
            ], style={
                'display': 'inline-block',
                'width': '40%',
                'verticalAlign': 'top',
                'textAlign': 'center'
            })
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'space-between'
        })
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
     Input('country-dropdown', 'value'),
     Input('g20-filter-dropdown', 'value')]
)
def update_map(budget_dist, probability, emissions_scope, selected_country, g20_filter):
    # Filter data for the entire scenario to establish a consistent color scale
    all_countries_data = scenario_parameters[
        (scenario_parameters['Budget_distribution_scenario'] == budget_dist) &
        (scenario_parameters['Probability_of_reach'] == probability) &
        (scenario_parameters['Emissions_scope'] == emissions_scope) &
        (scenario_parameters['Warming_scenario'] == '1.5°C') &
        (~scenario_parameters['ISO2'].isin(['WLD', 'EU', 'G20']))  # Exclude aggregates
        ].copy()

    # Convert 'Neutrality_year' to numeric for color scaling
    all_countries_data['Neutrality_year_numeric'] = pd.to_numeric(all_countries_data['Neutrality_year'],
                                                                  errors='coerce')

    # Determine the color range from the full dataset for the scenario
    color_range_min = min(1970, all_countries_data['Neutrality_year_numeric'].min())  # Ensure we start from 1970
    color_range_max = max(2100, all_countries_data['Neutrality_year_numeric'].max())  # Ensure we go to at least 2100

    # Select data for plotting based on dropdown
    if g20_filter == 'Yes':
        # When G20 filter is active, show only G20 countries (individual countries, not aggregate)
        plot_data = all_countries_data[all_countries_data['ISO2'].isin(G20_COUNTRIES)]
    elif selected_country != 'ALL':
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
        color_continuous_scale=[[0.0, '#8B0000'], [0.1, '#DC143C'], [0.2, '#FB8072'], [0.4, '#FDB462'],
                                [0.6, '#FFFFB3'], [0.8, '#8DD3C7'], [1.0, '#B3DE69']],
        # Dark red for 1980s, red for early years, Green for late years
        range_color=[color_range_min, color_range_max],  # Apply consistent color scale
        title="Zero Carbon Timeline: Countries' Legal Obligations Under the ICJ July 2025 Ruling",
        labels={
            'Neutrality_year_numeric': 'Zero Carbon Year',
            'Neutrality_year': 'Zero Carbon Year',
            'Years_to_neutrality_from_today': 'Years to Zero Carbon',
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
                      "Scope: %{customdata[0]}<br>" +
                      "Latest Year: %{customdata[1]}<br>" +
                      "Zero Carbon Year = %{customdata[2]}<br>" +
                      "Years to Zero Carbon = %{customdata[3]}<br>" +
                      "Annual Emissions (Mt) = %{customdata[4]:.1f}<br>" +
                      "Cumulative Emissions (Mt) = %{customdata[5]:.1f}<br>" +
                      "Emissions Per Capita (tonnes) = %{customdata[6]:.2f}<br>" +
                      "Carbon Budget (Mt) = %{customdata[7]:.1f}<extra></extra>",
        customdata=plot_data[[
            'Emissions_scope', 'Latest_year',
            'Neutrality_year', 'Years_to_neutrality_from_today',
            'Latest_annual_CO2_emissions_Mt', 'Latest_cumulative_CO2_emissions_Mt',
            'Latest_emissions_per_capita_t', 'Country_carbon_budget'
        ]].apply(lambda row: [
            get_scope_display_label(row['Emissions_scope']),  # Transform scope to display label
            row['Latest_year'],
            row['Neutrality_year'],
            row['Years_to_neutrality_from_today'],
            row['Latest_annual_CO2_emissions_Mt'],
            row['Latest_cumulative_CO2_emissions_Mt'],
            row['Latest_emissions_per_capita_t'],
            row['Country_carbon_budget']
        ], axis=1).values,
        # Position colorbar to the left of centered map
        colorbar=dict(
            x=0.05,  # Position to the left but within the centered layout
            xanchor="left",
            thickness=15,
            len=0.7,  # Shorter colorbar to create more space
            title=dict(
                text="Zero Carbon Year",
                font=dict(size=14, color="#2c3e50"),
                side="top"
            )
        )
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
        showlakes=False,  # Hide lakes for cleaner look
        # Center map more north and adjust bounds to crop Antarctica
        center=dict(lat=20, lon=0),  # Center more north
        lataxis_range=[-60, 80],  # Crop Antarctica (bottom) but keep Arctic
        lonaxis_range=[-180, 180]
    )

    # Also update layout to ensure consistent white borders for all countries
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor='lightgray',
            showocean=True,
            oceancolor='white',
            showlakes=False,
            showrivers=False,
            # Center map more north and adjust bounds to crop Antarctica
            center=dict(lat=20, lon=0),  # Center more north
            lataxis_range=[-60, 80],  # Crop Antarctica (bottom) but keep Arctic
            lonaxis_range=[-180, 180]
        )
    )

    # Make map bigger, wider, and center it properly
    fig.update_layout(
        height=600,  # Made taller to reduce flatness
        margin={"r": 150, "t": 45, "l": 150, "b": 50},  # Further reduced top margin to bring title closer to map
        geo=dict(
            projection_scale=1.0  # Less zoom to show more countries
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
            font=dict(size=16, color="#2c3e50", weight="bold"),  # Bold, smaller, dark grey like World Sufficiency Lab theme
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
     Input('country-dropdown', 'value'),
     Input('g20-filter-dropdown', 'value')]
)
def update_bar_chart(probability, selected_country, g20_filter):
    # Use selected probability and 1.5°C
    filtered_data = scenario_parameters[
        (scenario_parameters['Probability_of_reach'] == probability) &
        (scenario_parameters['Warming_scenario'] == '1.5°C')
        ].copy()

    # Apply country filter with intuitive logic
    if selected_country != 'ALL':
        # When a specific country is selected, show that country's data (ignore G20 filter)
        filtered_data = filtered_data[filtered_data['ISO2'] == selected_country]
        # Get country name for the title
        country_name = scenario_parameters[scenario_parameters['ISO2'] == selected_country]['Country'].iloc[0] if len(scenario_parameters[scenario_parameters['ISO2'] == selected_country]) > 0 else selected_country
        chart_title = f'Zero Carbon Timeline by Budget Distribution Scenario - {country_name}'
    elif g20_filter == 'Yes':
        # When G20 filter is active (and no specific country), show G20 aggregate data
        filtered_data = filtered_data[filtered_data['ISO2'] == 'G20']
        chart_title = 'Zero Carbon Timeline by Budget Distribution Scenario - G20 Aggregate'
    else:
        # For "All Countries" without G20 filter, show world aggregate data
        filtered_data = filtered_data[filtered_data['ISO2'] == 'WLD']
        chart_title = 'Zero Carbon Timeline by Budget Distribution Scenario - World'

    # Convert 'Neutrality_year' to numeric
    filtered_data['Neutrality_year_numeric'] = pd.to_numeric(filtered_data['Neutrality_year'], errors='coerce')

    # Create scenario-scope combinations for x-axis
    filtered_data['Scenario_Scope'] = filtered_data['Budget_distribution_scenario'] + ' - ' + filtered_data[
        'Emissions_scope']

    # Define the desired order (NDC Pledges only has Territory)
    scenario_order = [
        'NDC Pledges - Territory',
                    'Capability - Territory',
            'Capability - Consumption',
        'Responsibility - Territory',
        'Responsibility - Consumption'
    ]

    # Color scheme - Territory vs Consumption
    color_map = {
        'Territory': '#80B1D3',  # Light blue
        'Consumption': '#FB8072'  # Light red
    }

    # Filter out Population - only show NDC Pledges, Responsibility, Capability
    filtered_data = filtered_data[
        filtered_data['Budget_distribution_scenario'].isin(['NDC Pledges', 'Responsibility', 'Capability'])]
    # Create bar heights from 2025 baseline
    filtered_data['bar_height'] = filtered_data['Neutrality_year_numeric'] - 2025

    fig = px.bar(
        filtered_data,
        x='Scenario_Scope',
        y='bar_height',
        title=chart_title,
        labels={'bar_height': 'Years from 2025', 'Scenario_Scope': 'Budget Distribution Scenario'},
        color='Emissions_scope',
        color_discrete_map=color_map,
        category_orders={'Scenario_Scope': scenario_order}
    )

    # Manually set the base to 2025
    fig.update_traces(base=2025)

    # Remove hover data - just show scenario name
    fig.update_traces(hovertemplate="<b>%{x}</b><extra></extra>")

    # Add text labels on top of bars showing the zero carbon year
    for i, row in filtered_data.iterrows():
        # Determine if the bar is negative (below 2025) or positive (above 2025)
        bar_height = row['Neutrality_year_numeric'] - 2025
        if bar_height < 0:
            # For negative bars, place label below the bar
            yshift = -15
            yanchor = "top"
        else:
            # For positive bars, place label above the bar
            yshift = 15
            yanchor = "bottom"

        fig.add_annotation(
            x=row['Scenario_Scope'],
            y=row['Neutrality_year_numeric'],
            text=str(int(row['Neutrality_year_numeric'])),
            showarrow=False,
            font=dict(size=11, color="black", weight="bold"),
            yshift=yshift,  # Dynamic positioning based on bar direction
            xanchor="center",  # Center the text horizontally
            yanchor=yanchor  # Dynamic anchor based on bar direction
        )

    # Add horizontal line at 2025 (current year baseline)
    fig.add_hline(y=2025, line_dash="dash", line_color="black", line_width=2,
                  annotation_text="2025", annotation_position="right")

    # Add consistent styling to match theme
    fig.update_layout(
        paper_bgcolor='#ffffff',  # Clean white background
        plot_bgcolor='#ffffff',
        font=dict(
            family="Arial, sans-serif",
            size=14,  # Increased from 12 to 14
            color="#2c3e50"
        ),
        title=dict(
            font=dict(size=16, color="#2c3e50", weight="bold"),  # Bold, smaller, dark grey like World Sufficiency Lab theme
            x=0.5
        ),
        height=500,
        margin=dict(t=80, b=50, l=60, r=60),  # Match the line chart margin
        showlegend=False,  # Remove legend
        yaxis=dict(
            title=None,  # Remove Y axis title
            range=[1950, 2110],  # Extended range to accommodate labels below 1970
            tickmode='linear',
            tick0=1970,
            dtick=20  # Show ticks every 20 years for better spacing
        ),
        xaxis_title=None,
        xaxis=dict(
            tickangle=0,  # Horizontal labels
            tickmode='array',
            tickvals=['NDC Pledges - Territory', 'Capability - Territory', 'Capability - Consumption',
                      'Responsibility - Territory', 'Responsibility - Consumption'],
            ticktext=['', '', '', '', ''],  # Remove default tick labels
            tickfont=dict(size=12),
            side='bottom'  # Put emissions scope labels below bars
        )
    )

    # Add emissions scope labels at y=1965
    emissions_scope_labels = [
        (0, 'Territorial'),  # NDC Pledges - Territory
        (1, 'Territorial'),  # Capability - Territory
        (2, 'Consumption-based'),  # Capability - Consumption
        (3, 'Territorial'),  # Responsibility - Territory
        (4, 'Consumption-based')  # Responsibility - Consumption
    ]

    for pos, label in emissions_scope_labels:
        fig.add_annotation(
            x=pos,
            y=1960,  # Emissions scope labels at 1960
            text=label,
            showarrow=False,
            font=dict(size=12, color="#2c3e50"),
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="bottom"
        )

    # Add scenario group labels at the bottom
    scenario_groups = [
        ('NDC Pledges', 0, 0),  # Single bar for NDC Pledges
        ('Capability', 1, 2),  # Two bars for Capability
        ('Responsibility', 3, 4)  # Two bars for Responsibility
    ]

    for group_name, start_pos, end_pos in scenario_groups:
        center_pos = (start_pos + end_pos) / 2

        # Add visual line connecting bars in the same group (like in the drawing)
        # Now include NDC Pledges with a single bar line
        fig.add_shape(
            type="line",
            x0=start_pos - 0.1, y0=1955,  # Lines at 1955
            x1=end_pos + 0.1, y1=1955,  # Lines at 1955
            line=dict(color="#2c3e50", width=2),  # Thinner line
            xref="x", yref="y"
        )

        fig.add_annotation(
            x=center_pos,
            y=1950,  # Budget distribution labels at the bottom
            text=f"<b>{group_name}</b>",
            showarrow=False,
            font=dict(size=12, color="#2c3e50"),
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="top"
        )

    # Add text labels on bars showing zero carbon year for all cases
    for i, row in filtered_data.iterrows():
        # Determine if the bar is negative (below 2025) or positive (above 2025)
        bar_height = row['Neutrality_year_numeric'] - 2025
        if bar_height < 0:
            # For negative bars, place label below the bar
            yshift = -15
            yanchor = "top"
        else:
            # For positive bars, place label above the bar
            yshift = 15
            yanchor = "bottom"

        fig.add_annotation(
            x=row['Scenario_Scope'],
            y=row['Neutrality_year_numeric'],
            text=str(int(row['Neutrality_year_numeric'])),
            showarrow=False,
            font=dict(size=11, color="black", weight="bold"),
            yshift=yshift,  # Dynamic positioning based on bar direction
            xanchor="center",
            yanchor=yanchor
        )

    return fig


# Callback for the emissions trajectory line chart
@app.callback(
    Output('emissions-trajectory-line', 'figure'),
    [Input('budget-distribution-dropdown', 'value'),
     Input('probability-dropdown', 'value'),
     Input('emissions-scope-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('g20-filter-dropdown', 'value')]
)
def update_line_chart(budget_dist, probability, emissions_scope, selected_country, g20_filter):
    # Determine which country/region to show with intuitive logic
    if selected_country != 'ALL':
        # When a specific country is selected, show that country's data (ignore G20 filter)
        target_iso = selected_country
        country_name = scenario_parameters[scenario_parameters['ISO2'] == selected_country]['Country'].iloc[0] if len(
            scenario_parameters[scenario_parameters['ISO2'] == selected_country]) > 0 else selected_country
        chart_title = f'Past Emissions & Required Trajectory Under the ICJ AO - {get_scope_display_label(emissions_scope)} - {country_name} - Mt'
    elif g20_filter == 'Yes':
        # When G20 filter is active (and no specific country), show G20 aggregate data
        target_iso = 'G20'
        chart_title = f'Past Emissions & Required Trajectory Under the ICJ AO - G20 ({get_scope_display_label(emissions_scope)}) - Mt'
    else:
        # For "All Countries" without G20 filter, show world aggregate data
        target_iso = 'WLD'
        chart_title = f'Past Emissions & Required Trajectory Under the ICJ AO - World ({get_scope_display_label(emissions_scope)}) - Mt'
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

    # Past emissions
    if not hist_data.empty:
        fig.add_trace(go.Scatter(
            x=hist_data['Year'],
            y=hist_data['Annual_CO2_emissions_Mt'],
            mode='lines',
            name='Past Emissions',
            line=dict(color='#8DD3C7', width=3),  # Cyan from palette as requested
            hovertemplate="<b>Year: %{x}</b><br>Emissions: %{y:.2f} Mt<extra></extra>"
        ))

    # Forecasted emissions
    if not forecast_country.empty:
        fig.add_trace(go.Scatter(
            x=forecast_country['Year'],
            y=forecast_country['Forecasted_emissions_Mt'],
            mode='lines',
            name='Required Trajectory',
            line=dict(dash='dash', color='#FDB462', width=3),  # Orange from palette, different from bar chart
            hovertemplate="<b>Year: %{x}</b><br>Emissions: %{y:.2f} Mt<extra></extra>"
        ))

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,  # Remove Y axis title
        hovermode='closest',
        # Add consistent styling to match theme
        paper_bgcolor='#ffffff',  # Clean white background
        plot_bgcolor='#ffffff',
        font=dict(
            family="Arial, sans-serif",
            size=14,  # Increased from 12 to 14
            color="#2c3e50"
        ),
        title=dict(
            text=chart_title,
            font=dict(size=16, color="#2c3e50", weight="bold"),  # Bold, smaller, dark grey like World Sufficiency Lab theme
            x=0.5
        ),
        height=500,
        margin=dict(t=80, b=50, l=60, r=60),  # Match the top margin of the bar chart
        # Position legend to overlap at 2080-2100 years area
        legend=dict(
            x=0.75,  # Position at around 2080 area (75% of chart width)
            y=0.95,  # Top of chart
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)"
        ),
        # Style the grid with fixed range
        xaxis=dict(
            gridcolor='rgba(200,200,200,0.3)',  # Light gray grid on white background
            showgrid=True,
            range=[1970, 2100],  # Fixed x-axis range from 1970 to 2100
            title=None
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
    [Input('world-map', 'clickData'),
     Input('g20-filter-dropdown', 'value')]
)
def update_country_from_map(clickData, g20_filter):
    if g20_filter == 'Yes':
        # When G20 filter is active, reset to ALL to show G20 aggregate
        return 'ALL'

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


# Callback for Top 20 Cumulative Emitters Chart
@app.callback(
    Output('top-cumulative-emitters', 'figure'),
    [Input('country-dropdown', 'value'),
     Input('emissions-scope-dropdown', 'value'),
     Input('g20-filter-dropdown', 'value')]
)
def update_top_cumulative_emitters(selected_country, selected_scope, g20_filter):
    try:
        # Filter data by emissions scope
        filtered_data = historical_data[historical_data['Emissions_scope'] == selected_scope].copy()

        # Get latest year data for each country (exclude aggregates based on G20 filter)
        if g20_filter == 'Yes':
            # When G20 filter is active, show only G20 countries
            filtered_data = filtered_data[filtered_data['G20_country'] == 'Yes']
        else:
            # Exclude aggregates for normal view
            filtered_data = filtered_data[~filtered_data['ISO2'].isin(['WLD', 'EU', 'G20'])]

        # Remove rows with missing data
        filtered_data = filtered_data.dropna(subset=['ISO2', 'Year', 'Cumulative_CO2_emissions_Mt'])

        if filtered_data.empty:
            # Return empty chart if no data
            fig = px.bar(title=f'No data available for {get_scope_display_label(selected_scope)}')
            return fig

        # Get the latest year for each country safely
        latest_indices = filtered_data.groupby('ISO2')['Year'].idxmax()
        latest_data = filtered_data.loc[latest_indices].copy()

        # Calculate share of cumulative emissions
        # Get world total cumulative emissions for the same scope and year
        world_data = historical_data[
            (historical_data['Emissions_scope'] == selected_scope) &
            (historical_data['ISO2'] == 'WLD') &
            (historical_data['Cumulative_CO2_emissions_Mt'] > 0)  # Only non-zero emissions
            ]
        if not world_data.empty:
            # Get the latest year with actual emissions data
            world_latest = world_data.loc[world_data['Year'].idxmax()]
            world_total = world_latest['Cumulative_CO2_emissions_Mt']
            if world_total > 0:
                latest_data['Share_of_cumulative_emissions'] = latest_data['Cumulative_CO2_emissions_Mt'] / world_total
            else:
                latest_data['Share_of_cumulative_emissions'] = 0
        else:
            latest_data['Share_of_cumulative_emissions'] = 0

        # Remove countries with zero or negative emissions
        latest_data = latest_data[latest_data['Share_of_cumulative_emissions'] > 0]

        # Sort by share of cumulative emissions and get top 20
        top_20 = latest_data.nlargest(20, 'Share_of_cumulative_emissions')

        if top_20.empty:
            # Return empty chart if no data
            fig = px.bar(title=f'No emission data available for {get_scope_display_label(selected_scope)}')
            return fig

        # Highlight selected country if it exists in top 20 (using colors from your palette)
        colors = ['#FB8072' if country == selected_country else '#80B1D3' for country in top_20['ISO2']]

        # Convert to percentage for display
        top_20_display = top_20.copy()
        top_20_display['Share_of_cumulative_emissions_pct'] = top_20_display['Share_of_cumulative_emissions'] * 100

        # Set title based on G20 filter
        if g20_filter == 'Yes':
            title_text = f'Top 20 G20 Countries by Share of Cumulative CO2 Emissions ({get_scope_display_label(selected_scope)}, 1970-{2023 if selected_scope == "Territory" else 2022})'
        else:
            title_text = f'Top 20 Countries by Share of Cumulative CO2 Emissions ({get_scope_display_label(selected_scope)}, 1970-{2023 if selected_scope == "Territory" else 2022})'

        fig = px.bar(
            top_20_display,
            x='Share_of_cumulative_emissions_pct',
            y='Country',
            orientation='h',
            color_discrete_sequence=colors,
            title=title_text
        )

        # Update traces for better hover formatting
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>" +
                          "Share of Cumulative Emissions = %{x:.1f}%<br>" +
                          "<extra></extra>"
        )

        # Add spacing between bars and y-axis labels
        fig.update_layout(bargap=0.3)
    except Exception as e:
        # Return error chart if something goes wrong
        fig = px.bar(title=f'Error loading data: {str(e)}')
        return fig

    fig.update_layout(
        height=600,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="#2c3e50"
        ),
        title=dict(
            font=dict(size=16, color="#2c3e50", weight="bold"),
            x=0.5
        ),
        showlegend=False,
        yaxis={'categoryorder': 'total ascending', 'title': None, 'ticklabelstep': 1, 'tickmode': 'auto',
               'ticklabelposition': 'outside', 'tickangle': 0, 'tickfont': {'size': 12}},
        xaxis={'title': None, 'ticksuffix': '%'},
        margin=dict(l=200, r=100, t=80, b=50, pad=10)  # Increased left margin for more spacing
    )

    return fig


# Callback for Top 20 Per Capita Emitters Chart
@app.callback(
    Output('top-per-capita-emitters', 'figure'),
    [Input('country-dropdown', 'value'),
     Input('emissions-scope-dropdown', 'value'),
     Input('g20-filter-dropdown', 'value')]
)
def update_top_per_capita_emitters(selected_country, selected_scope, g20_filter):
    try:
        # Filter data by emissions scope
        filtered_data = historical_data[historical_data['Emissions_scope'] == selected_scope].copy()

        # Get latest year data for each country (exclude aggregates based on G20 filter)
        if g20_filter == 'Yes':
            # When G20 filter is active, show only G20 countries
            filtered_data = filtered_data[filtered_data['G20_country'] == 'Yes']
        else:
            # Exclude aggregates for normal view
            filtered_data = filtered_data[~filtered_data['ISO2'].isin(['WLD', 'EU', 'G20'])]

        # Remove rows with missing data
        filtered_data = filtered_data.dropna(subset=['ISO2', 'Year', 'Emissions_per_capita_ton'])

        if filtered_data.empty:
            # Return empty chart if no data
            fig = px.bar(title=f'No data available for {get_scope_display_label(selected_scope)}')
            return fig

        # Get the latest year for each country safely
        latest_indices = filtered_data.groupby('ISO2')['Year'].idxmax()
        latest_data = filtered_data.loc[latest_indices].copy()

        # Remove countries with zero or negative emissions per capita
        latest_data = latest_data[latest_data['Emissions_per_capita_ton'] > 0]

        if latest_data.empty:
            # Return empty chart if no valid data
            fig = px.bar(title=f'No emission data available for {get_scope_display_label(selected_scope)}')
            return fig

        # Sort by emissions per capita and get top 20
        top_20 = latest_data.nlargest(20, 'Emissions_per_capita_ton')

        # Highlight selected country if it exists in top 20 (using colors from your palette)
        colors = ['#FB8072' if country == selected_country else '#B3DE69' for country in top_20['ISO2']]

        # Set title based on G20 filter
        if g20_filter == 'Yes':
            title_text = f'Top 20 G20 Countries by CO2 Emissions Per Capita - Tons ({get_scope_display_label(selected_scope)}, {2023 if selected_scope == "Territory" else 2022})'
        else:
            title_text = f'Top 20 Countries by CO2 Emissions Per Capita - Tons ({get_scope_display_label(selected_scope)}, {2023 if selected_scope == "Territory" else 2022})'

        fig = px.bar(
            top_20,
            x='Emissions_per_capita_ton',
            y='Country',
            orientation='h',
            color_discrete_sequence=colors,
            title=title_text
        )

        # Update traces for better hover formatting
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>" +
                          "CO2 Emissions Per Capita = %{x:.1f} tons<br>" +
                          "<extra></extra>"
        )

        # Add spacing between bars and y-axis labels
        fig.update_layout(bargap=0.3)
    except Exception as e:
        # Return error chart if something goes wrong
        fig = px.bar(title=f'Error loading data: {str(e)}')
        return fig

    fig.update_layout(
        height=600,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="#2c3e50"
        ),
        title=dict(
            font=dict(size=16, color="#2c3e50", weight="bold"),
            x=0.5
        ),
        showlegend=False,
        yaxis={'categoryorder': 'total ascending', 'title': None, 'ticklabelstep': 1, 'tickmode': 'auto',
               'ticklabelposition': 'outside', 'tickangle': 0, 'tickfont': {'size': 12}},
        xaxis={'title': None},
        margin=dict(l=200, r=100, t=80, b=50, pad=10)  # Increased left margin for more spacing
    )

    return fig


# For gunicorn deployment
server = app.server

# Production deployment - no local server
# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8050)
