import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import os
import json

# Get the absolute path to the data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))

# ISO-2 to ISO-3 mapping for European countries
ISO2_TO_ISO3 = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'CH': 'CHE', 'CY': 'CYP', 'CZ': 'CZE',
    'DE': 'DEU', 'DK': 'DNK', 'EE': 'EST', 'EL': 'GRC', 'ES': 'ESP', 'FI': 'FIN',
    'FR': 'FRA', 'HR': 'HRV', 'HU': 'HUN', 'IE': 'IRL', 'IS': 'ISL', 'IT': 'ITA',
    'LT': 'LTU', 'LU': 'LUX', 'LV': 'LVA', 'MT': 'MLT', 'NL': 'NLD', 'NO': 'NOR',
    'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'RS': 'SRB', 'SE': 'SWE', 'SI': 'SVN',
    'SK': 'SVK', 'UK': 'GBR'
}

# Load the data
try:
    master_df = pd.read_csv(os.path.join(DATA_DIR, 'master_dataframe.csv'))
    time_series_df = pd.read_csv(os.path.join(DATA_DIR, 'master_dataframe_time_series.csv'))
    
    # Load EWBI structure for hierarchical navigation
    with open(os.path.join(DATA_DIR, '..', 'data', 'ewbi_indicators.json'), 'r') as f:
        ewbi_structure = json.load(f)['EWBI']
    
    print("All data files loaded successfully.")
    
    # Get unique countries and sort them (excluding aggregates for individual selection)
    COUNTRIES = sorted([c for c in master_df['country'].unique() if 'Average' not in c])
    
    # Get aggregate countries
    AGGREGATE_COUNTRIES = sorted([c for c in master_df['country'].unique() if 'Average' in c])
    
    print(f"Found {len(COUNTRIES)} individual countries and {len(AGGREGATE_COUNTRIES)} aggregates")
    
    # Create EU priority options - only the 6 main priorities
    EU_PRIORITIES = [
        'Agriculture and Food',
        'Energy and Housing', 
        'Equality',
        'Health and Animal Welfare',
        'Intergenerational Fairness, Youth, Culture and Sport',
        'Social Rights and Skills, Quality Jobs and Preparedness'
    ]
    
    # Create secondary indicator options
    SECONDARY_INDICATORS = []
    for prio in ewbi_structure:
        for component in prio['components']:
            SECONDARY_INDICATORS.append({
                'label': f"{prio['name']} - {component['name']}",
                'value': f"{prio['name']}|{component['name']}",
                'eu_priority': prio['name'],
                'secondary': component['name']
            })
    
    print(f"Dashboard initialized with {len(EU_PRIORITIES)} EU priorities and {len(SECONDARY_INDICATORS)} secondary indicators")
    
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
        html.H1("European Well-Being Index (EWBI)", 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'fontSize': '2.5rem',
                    'fontWeight': 'bold',
                    'marginBottom': '10px',
                    'fontFamily': 'Arial, sans-serif'
                }),
        html.H2([
            "Multi-level Analysis of Well-being Indicators Across Europe"
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
                html.Label("Analysis Level", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='analysis-level-dropdown',
                    options=[
                        {'label': 'Overview', 'value': 'overview'},
                        {'label': 'By EU Priority', 'value': 'by_eu_priority'}
                    ],
                    value='overview',
                    style={'marginTop': '8px'},
                    clearable=False
                )
            ], style={'width': '20%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.Label("EU Priority", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='eu-priority-dropdown',
                    options=[{'label': 'ALL', 'value': 'ALL'}] + [{'label': prio, 'value': prio} for prio in EU_PRIORITIES],
                    value='ALL',
                    style={'marginTop': '8px'},
                    clearable=False,
                    disabled=True
                )
            ], style={'width': '20%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.Label("Secondary Indicator", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='secondary-indicator-dropdown',
                    options=[{'label': 'ALL', 'value': 'ALL'}],
                    value='ALL',
                    style={'marginTop': '8px'},
                    clearable=False,
                    disabled=True
                )
            ], style={'width': '20%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.Label("Primary Indicator", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='primary-indicator-dropdown',
                    options=[{'label': 'ALL', 'value': 'ALL'}],
                    value='ALL',
                    style={'marginTop': '8px'},
                    clearable=False,
                    disabled=True
                )
            ], style={'width': '20%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.Label("Countries", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                                        dcc.Dropdown(
                    id='countries-filter',
                    options=[
                        {'label': 'EU Countries Average', 'value': 'EU Countries Average'},
                        {'label': 'All Countries Average', 'value': 'All Countries Average'},
                        {'label': 'Individual Countries', 'value': 'individual'}
                    ] + [{'label': country, 'value': country} for country in COUNTRIES],
                    value=['EU Countries Average'],  # Default to EU Countries Average
                    multi=True,
                    placeholder='Select countries to display (default: EU Countries Average)',
                    style={'marginTop': '8px'}
                )
                            ], style={'width': '20%', 'display': 'inline-block'}),
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
    
    # Visualizations
    html.Div([
        # First row: European map and Time series
        html.Div([
            dcc.Graph(id='european-map-chart', style={'display': 'inline-block', 'width': '49%', 'verticalAlign': 'top'}),
            dcc.Graph(id='time-series-chart', style={'display': 'inline-block', 'width': '49%', 'verticalAlign': 'top'})
        ], style={'textAlign': 'center', 'margin': '0 auto'}),
        
        # Second row: Decile analysis and Radar chart
        html.Div([
            dcc.Graph(id='decile-analysis-chart', style={'display': 'inline-block', 'width': '49%'}),
            dcc.Graph(id='radar-chart', style={'display': 'inline-block', 'width': '49%'})
        ], style={'marginTop': '20px'})
    ], style={
        'margin': '0 20px',
        'paddingTop': '20px'
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
            html.H4("European Well-Being Index (EWBI)", style={
                'color': '#34495e',
                'fontSize': '1.2rem',
                'fontWeight': 'bold',
                'marginBottom': '10px'
            }),
            html.P("The EWBI is a composite indicator measuring well-being across European countries, constructed from multiple data sources:", style={
                'marginBottom': '10px',
                'lineHeight': '1.6'
            }),
            html.P([
                html.Strong("EU-SILC: "),
                "European Union Statistics on Income and Living Conditions - household surveys on income, living conditions, and social inclusion"
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
            html.P([
                html.Strong("EU-SILC: "),
                "European Health Interview Survey - population health and healthcare access data"
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
            html.P([
                html.Strong("HBS: "),
                "Household Budget Survey - expenditure patterns and consumption data"
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
            html.P([
                html.Strong("LFS: "),
                "Labour Force Survey - employment and working conditions data"
            ], style={'marginBottom': '20px', 'lineHeight': '1.6'}),
            
            html.H4("Indicator Structure", style={
                'color': '#34495e',
                'fontSize': '1.2rem',
                'fontWeight': 'bold',
                'marginBottom': '10px'
            }),
            html.P("The EWBI follows a hierarchical structure:", style={
                'marginBottom': '10px',
                'lineHeight': '1.6'
            }),
            html.P([
                html.Strong("Level 1: "),
                "EWBI - Overall well-being score (geometric mean of EU priorities)"
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
            html.P([
                html.Strong("Level 2: "),
                "6 EU Priorities - Major policy areas (Agriculture & Food, Energy & Housing, Equality, Health & Animal Welfare, Intergenerational Fairness, Social Rights & Skills)"
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
            html.P([
                html.Strong("Level 3: "),
                "18 Secondary Indicators - Specific well-being dimensions within each priority"
            ], style={'marginBottom': '10px', 'lineHeight': '1.6'}),
            html.P([
                html.Strong("Level 4: "),
                "58 Primary Indicators - Individual survey questions and expenditure measures"
            ], style={'marginBottom': '20px', 'lineHeight': '1.6'}),
            
            html.H4("Methodology", style={
                'color': '#34495e',
                'fontSize': '1.2rem',
                'fontWeight': 'bold',
                'marginBottom': '10px'
            }),
            html.P("All indicators are normalized on a 0-1 scale where higher values indicate better well-being. The normalization is performed intra-decile and intra-indicator to ensure fair comparison across countries and income groups.", style={
                'marginBottom': '10px',
                'lineHeight': '1.6'
            }),
            html.P("Missing values are handled using forward-fill and backward-fill strategies as recommended by EU JRC methodology.", style={
                'marginBottom': '20px',
                'lineHeight': '1.6'
            }),
        ], style={
            'maxWidth': '1200px',
            'margin': '0 auto',
            'padding': '0 20px'
        })
    ], style={
        'marginTop': '40px',
        'padding': '30px 0',
        'backgroundColor': '#f8f9fa',
        'borderTop': '1px solid #dee2e6'
    })
])

# Callback to update EU priority dropdown based on analysis level
@app.callback(
    Output('eu-priority-dropdown', 'disabled'),
    Output('eu-priority-dropdown', 'value'),
    Input('analysis-level-dropdown', 'value')
)
def update_eu_priority_dropdown(analysis_level):
    if analysis_level == 'overview':
        # In overview mode, show ALL EU priorities
        return True, 'ALL'
    else:
        # In by_eu_priority mode, allow selection of specific priority
        return False, EU_PRIORITIES[0] if EU_PRIORITIES else None

# Callback to update secondary indicator dropdown based on EU priority
@app.callback(
    Output('secondary-indicator-dropdown', 'options'),
    Output('secondary-indicator-dropdown', 'value'),
    Output('secondary-indicator-dropdown', 'disabled'),
    Input('eu-priority-dropdown', 'value'),
    Input('analysis-level-dropdown', 'value')
)
def update_secondary_indicator_dropdown(eu_priority, analysis_level):
    if analysis_level == 'overview':
        # In overview mode, show ALL secondary indicators
        return [{'label': 'ALL', 'value': 'ALL'}], 'ALL', True
    elif analysis_level == 'by_eu_priority' and eu_priority:
        # In by_eu_priority mode, show ALL secondary indicators for the selected EU priority
        filtered_indicators = [
            indicator for indicator in SECONDARY_INDICATORS 
            if indicator['eu_priority'] == eu_priority
        ]
        options = [{'label': 'ALL', 'value': 'ALL'}] + [
            {'label': indicator['secondary'], 'value': indicator['secondary']} 
            for indicator in filtered_indicators
        ]
        return options, 'ALL', False
    else:
        return [{'label': 'ALL', 'value': 'ALL'}], 'ALL', True

# Callback to update primary indicator dropdown based on secondary indicator
@app.callback(
    Output('primary-indicator-dropdown', 'options'),
    Output('primary-indicator-dropdown', 'value'),
    Output('primary-indicator-dropdown', 'disabled'),
    Input('secondary-indicator-dropdown', 'value'),
    Input('eu-priority-dropdown', 'value'),
    Input('analysis-level-dropdown', 'value')
)
def update_primary_indicator_dropdown(secondary_indicator, eu_priority, analysis_level):
    if analysis_level == 'overview':
        # In overview mode, show ALL primary indicators
        return [{'label': 'ALL', 'value': 'ALL'}], 'ALL', True
    elif analysis_level == 'by_eu_priority' and eu_priority and secondary_indicator and secondary_indicator != 'ALL':
        # In by_eu_priority mode with specific secondary indicator, show primary indicators
        # Load the EWBI structure to get the actual primary indicators
        try:
            with open(os.path.join(DATA_DIR, '..', 'data', 'ewbi_indicators.json'), 'r') as f:
                config = json.load(f)['EWBI']
            
            primary_indicators = []
            for priority in config:
                if priority['name'] == eu_priority:
                    for component in priority['components']:
                        if component['name'] == secondary_indicator:
                            for indicator in component['indicators']:
                                primary_indicators.append({
                                    'label': indicator['code'],
                                    'value': indicator['code']
                                })
                            break
                    break
            
            if primary_indicators:
                options = [{'label': 'ALL', 'value': 'ALL'}] + primary_indicators
                return options, 'ALL', False
            else:
                return [{'label': 'ALL', 'value': 'ALL'}], 'ALL', True
        except:
            # Fallback if JSON loading fails
            return [{'label': 'ALL', 'value': 'ALL'}], 'ALL', True
    else:
        return [{'label': 'ALL', 'value': 'ALL'}], 'ALL', True

# Callback to update all charts
@app.callback(
    [Output('european-map-chart', 'figure'),
     Output('decile-analysis-chart', 'figure'),
     Output('radar-chart', 'figure'),
     Output('time-series-chart', 'figure')],
    [Input('analysis-level-dropdown', 'value'),
     Input('eu-priority-dropdown', 'value'),
     Input('secondary-indicator-dropdown', 'value'),
     Input('primary-indicator-dropdown', 'value'),
     Input('countries-filter', 'value')]
)
def update_charts(analysis_level, eu_priority, secondary_indicator, primary_indicator, selected_countries):
    # For the map (Graph 1), we always want to show all individual countries
    map_df = master_df[~master_df['country'].str.contains('Average')].copy()
    
    # For analysis charts (Graphs 2, 3, 4), filter based on selection
    if not selected_countries or selected_countries == ['EU Countries Average']:
        # Default to EU Countries Average for analysis charts
        filtered_df = master_df[master_df['country'] == 'EU Countries Average'].copy()
        time_filtered_df = time_series_df[time_series_df['country'] == 'EU Countries Average'].copy()
    else:
        # Filter by selected countries (can include aggregates and individual countries)
        filtered_df = master_df[master_df['country'].isin(selected_countries)].copy()
        time_filtered_df = time_series_df[time_series_df['country'].isin(selected_countries)].copy()
    
    # Determine what to show based on analysis level and filter selections
    if analysis_level == 'overview':
        # Show EWBI and EU priorities
        return create_overview_charts(map_df, filtered_df, time_filtered_df)
    elif analysis_level == 'by_eu_priority':
        # Handle the three levels of drill-down
        if secondary_indicator and secondary_indicator != 'ALL' and primary_indicator and primary_indicator != 'ALL':
            # Level 3: Specific Primary Indicator selected
            return create_primary_indicator_charts(map_df, filtered_df, time_filtered_df, eu_priority, secondary_indicator, primary_indicator)
        elif secondary_indicator and secondary_indicator != 'ALL':
            # Level 2: Specific Secondary Indicator selected (Primary = ALL)
            return create_secondary_indicator_charts(map_df, filtered_df, time_filtered_df, eu_priority, secondary_indicator)
        else:
            # Level 1: Only EU Priority selected (Secondary = ALL, Primary = ALL)
            return create_eu_priority_charts(map_df, filtered_df, time_filtered_df, eu_priority)
    else:
        # Fallback to overview
        return create_overview_charts(map_df, filtered_df, time_filtered_df)

def create_overview_charts(map_df, analysis_df, time_df):
    """Create charts for overview level (EWBI + EU priorities)"""
    
    # 1. European map chart (EWBI scores) - always shows all countries
    european_map = go.Figure()
    
    # Get average EWBI scores by country for the map
    ewbi_by_country = map_df.groupby('country')['ewbi_score'].mean().reset_index()
    
    # Convert ISO-2 codes to ISO-3 codes for the map
    ewbi_by_country['iso3'] = ewbi_by_country['country'].map(ISO2_TO_ISO3)
    
    # Filter out countries without ISO-3 codes (like aggregates)
    ewbi_by_country = ewbi_by_country[ewbi_by_country['iso3'].notna()].copy()
    
    # Create the choropleth map
    european_map = go.Figure(data=go.Choropleth(
        locations=ewbi_by_country['iso3'],
        z=ewbi_by_country['ewbi_score'],
        locationmode='ISO-3',
        colorscale='RdYlGn',  # Red to Green scale
        colorbar_title="EWBI Score",
        text=ewbi_by_country['country'] + ': ' + ewbi_by_country['ewbi_score'].round(3).astype(str),
        hovertemplate='<b>%{text}</b><br>' +
                      'EWBI Score: %{z:.3f}<br>' +
                      '<extra></extra>'
    ))
    
    # Update traces for better styling like Budget dashboard
    european_map.update_traces(
        marker_line_color="white",
        marker_line_width=0.5,
        colorbar=dict(
            x=0.95,  # Position to the right
            xanchor="right",
            thickness=15,
            len=0.7,
            title=dict(
                text="EWBI Score",
                font=dict(size=14, color="#2c3e50"),
                side="top"
            )
        )
    )
    
    european_map.update_layout(
        height=550,  # Slightly taller like Budget dashboard
        margin={"r": 150, "t": 80, "l": 150, "b": 50},  # Increased margins for better spacing
        geo=dict(
            scope='europe',
            projection=dict(type='equirectangular'),
            showland=True,
            landcolor='lightgray',  # Same as Budget dashboard
            coastlinecolor='white',  # White coastlines like Budget dashboard
            showcountries=True,
            countrycolor='lightgray',  # Same as Budget dashboard
            showocean=True,
            oceancolor='white',  # White ocean like Budget dashboard
            showframe=False,
            showcoastlines=True,
            coastlinewidth=1,
            projection_scale=1.0,  # Less zoom to show more of Europe
            center=dict(lat=50, lon=10),  # Center on Europe
            lataxis_range=[35, 75],  # Focus on Europe (southern to northern bounds)
            lonaxis_range=[-15, 45]  # Focus on Europe (western to eastern bounds)
        ),
        # Style to match World Sufficiency Lab theme like Budget dashboard
        paper_bgcolor='#ffffff',  # Clean white background
        plot_bgcolor='#ffffff',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#2c3e50"
        ),
        title=dict(
            text='European Well-Being Index (EWBI) Map',
            font=dict(size=16, color="#f4d03f", weight="bold"),  # Bold, smaller, yellow like top ribbon
            x=0.5
        ),
        # Lighter hover box styling like Budget dashboard
        hoverlabel=dict(
            bgcolor="rgba(245, 245, 245, 0.9)",  # Light grey background for hover
            bordercolor="white",
            font=dict(color="black", size=12)
        )
    )
    
    # Set color scale from 0 to 1
    european_map.update_traces(
        zmin=0,
        zmax=1
    )
    
    # 2. Decile analysis chart (EWBI by decile for selected countries)
    decile_analysis = go.Figure()
    
    # For decile analysis, use the analysis_df (which defaults to EU Countries Average)
    countries_to_show = []
    
    # Always show EU Countries Average first if available
    if 'EU Countries Average' in analysis_df['country'].values:
        countries_to_show.append('EU Countries Average')
    
    # Then add any other selected countries (excluding EU Countries Average to avoid duplication)
    other_countries = [c for c in analysis_df['country'].unique() if c != 'EU Countries Average' and 'Average' not in c]
    countries_to_show.extend(other_countries)
    
    # Show countries in the determined order
    for country in countries_to_show:
        country_data = analysis_df[analysis_df['country'] == country].sort_values('decile')
        decile_analysis.add_trace(go.Bar(
            x=[f'Decile {d}' for d in country_data['decile']],
            y=country_data['ewbi_score'],
            name=country,
            text=country_data['ewbi_score'].round(3),
            textposition='auto'
        ))
    
    decile_analysis.update_layout(
        title='EWBI Scores by Decile - Selected Countries',
        xaxis_title='Income Decile',
        yaxis_title='EWBI Score',
        height=400,
        barmode='group',
        font=dict(family='Arial, sans-serif', size=12),
        yaxis=dict(range=[0, 1])  # Set y-axis scale from 0 to 1
    )
    
    # 3. Radar chart (EU priorities for selected countries) - Only the 6 main priorities
    radar_chart = go.Figure()
    
    # Get only the 6 main EU priority columns
    main_eu_priorities = [
        'Agriculture and Food',
        'Energy and Housing', 
        'Equality',
        'Health and Animal Welfare',
        'Intergenerational Fairness, Youth, Culture and Sport',
        'Social Rights and Skills, Quality Jobs and Preparedness'
    ]
    
    # For radar chart, prioritize EU Countries Average, then add individual countries
    countries_to_show = []
    
    # Always show EU Countries Average first if available
    if 'EU Countries Average' in analysis_df['country'].values:
        countries_to_show.append('EU Countries Average')
    
    # Then add any other selected countries (excluding EU Countries Average to avoid duplication)
    other_countries = [c for c in analysis_df['country'].unique() if c != 'EU Countries Average' and 'Average' not in c]
    countries_to_show.extend(other_countries)
    
    # Show countries in the determined order
    for country in countries_to_show:
        country_data = analysis_df[analysis_df['country'] == country].iloc[0]  # Take first row (all deciles have same values for EU priorities)
        values = [country_data[col] for col in main_eu_priorities if col in analysis_df.columns and pd.notna(country_data[col])]
        labels = [col for col in main_eu_priorities if col in analysis_df.columns and pd.notna(country_data[col])]
        
        radar_chart.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=country
        ))
    
    radar_chart.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Already set to 0-1
            )),
        showlegend=True,
        title='EU Priorities Comparison - Selected Countries',
        height=400,
        font=dict(family='Arial, sans-serif', size=12)
    )
    
    # 4. Time series chart (EWBI evolution over time) - Line chart implementation
    time_series = go.Figure()
    
    # Load the historical EWBI evolution data
    try:
        ewbi_evolution_df = pd.read_csv(os.path.join(DATA_DIR, 'ewbi_evolution_over_time.csv'))
        
        # For time series, prioritize EU Countries Average, then add individual countries
        countries_to_show = []
        
        # Always show EU Countries Average first if available
        if 'EU Countries Average' in analysis_df['country'].values:
            countries_to_show.append('EU Countries Average')
        
        # Then add any other selected countries (excluding EU Countries Average to avoid duplication)
        other_countries = [c for c in analysis_df['country'].values if c != 'EU Countries Average' and 'Average' not in c]
        countries_to_show.extend(other_countries)
        
        # Show countries in the determined order
        for country in countries_to_show:
            if country in ewbi_evolution_df['country'].values:
                country_data = ewbi_evolution_df[ewbi_evolution_df['country'] == country].sort_values('year')
                time_series.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=country_data['ewbi_score'],
                    mode='lines+markers',
                    name=country,
                    line=dict(width=2),
                    hovertemplate=f'<b>{country}</b><br>EWBI Score: %{{y:.3f}}<br>Year: %{{x}}<extra></extra>'
                ))
        
        # Update the title to reflect what we're actually showing
        time_series.update_layout(
            title='EWBI Score Evolution Over Time - Selected Countries',
            xaxis_title='Year',
            yaxis_title='EWBI Score',
            height=400,
            font=dict(family='Arial, sans-serif', size=12),
            yaxis=dict(range=[0, 1])  # Set y-axis scale from 0 to 1
        )
        
    except FileNotFoundError:
        # Fallback if the file doesn't exist
        time_series.add_annotation(
            text="Historical EWBI data not found. Please run create_master_dataframe.py first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        
        time_series.update_layout(
            title='EWBI Score Evolution Over Time - Selected Countries',
            xaxis_title='Year',
            yaxis_title='EWBI Score',
            height=400,
            font=dict(family='Arial, sans-serif', size=12)
        )
    
    return european_map, decile_analysis, radar_chart, time_series

def create_eu_priority_charts(map_df, analysis_df, time_df, eu_priority):
    """Create charts for EU priority level"""
    
    # Get secondary indicators for this EU priority
    # The naming pattern is: "PriorityName_SecondaryIndicatorName"
    # We need to handle spaces and special characters properly
    priority_name_clean = eu_priority.replace(' ', '_').replace(',', '').replace(' and ', '_and_')
    secondary_cols = [col for col in analysis_df.columns if col.startswith(f"{priority_name_clean}_")]
    
    print(f"Looking for secondary indicators for '{eu_priority}'")
    print(f"Clean priority name: '{priority_name_clean}'")
    print(f"Found secondary columns: {secondary_cols}")
    
    # 1. European map chart (EU priority scores)
    european_map = go.Figure()
    
    # Get average EU priority scores by country for the map (always show all countries)
    if eu_priority in map_df.columns:
        priority_by_country = map_df.groupby('country')[eu_priority].mean().reset_index()
        
        # Convert ISO-2 codes to ISO-3 codes for the map
        priority_by_country['iso3'] = priority_by_country['country'].map(ISO2_TO_ISO3)
        
        # Filter out countries without ISO-3 codes (like aggregates)
        priority_by_country = priority_by_country[priority_by_country['iso3'].notna()].copy()
        
        # Create the choropleth map
        european_map = go.Figure(data=go.Choropleth(
            locations=priority_by_country['iso3'],
            z=priority_by_country[eu_priority],
            locationmode='ISO-3',
            colorscale='RdYlGn',  # Red to Green scale
            colorbar_title=f"{eu_priority} Score",
            text=priority_by_country['country'] + ': ' + priority_by_country[eu_priority].round(3).astype(str),
            hovertemplate='<b>%{text}</b><br>' +
                          f'{eu_priority} Score: %{{z:.3f}}<br>' +
                          '<extra></extra>'
        ))
        
        # Update traces for better styling like Budget dashboard
        european_map.update_traces(
            marker_line_color="white",
            marker_line_width=0.5,
            colorbar=dict(
                x=0.95,  # Position to the right
                xanchor="right",
                thickness=15,
                len=0.7,
                title=dict(
                    text=f"{eu_priority} Score",
                    font=dict(size=14, color="#2c3e50"),
                    side="top"
                )
            )
        )
        
        european_map.update_layout(
            height=550,  # Slightly taller like Budget dashboard
            margin={"r": 150, "t": 80, "l": 150, "b": 50},  # Increased margins for better spacing
            geo=dict(
                scope='europe',
                projection=dict(type='equirectangular'),
                showland=True,
                landcolor='lightgray',  # Same as Budget dashboard
                coastlinecolor='white',  # White coastlines like Budget dashboard
                showcountries=True,
                countrycolor='lightgray',  # Same as Budget dashboard
                showocean=True,
                oceancolor='white',  # White ocean like Budget dashboard
                showframe=False,
                showcoastlines=True,
                coastlinewidth=1,
                projection_scale=1.0,  # Less zoom to show more of Europe
                center=dict(lat=50, lon=10),  # Center on Europe
                lataxis_range=[35, 75],  # Focus on Europe (southern to northern bounds)
                lonaxis_range=[-15, 45]  # Focus on Europe (western to eastern bounds)
            ),
            # Style to match World Sufficiency Lab theme like Budget dashboard
            paper_bgcolor='#ffffff',  # Clean white background
            plot_bgcolor='#ffffff',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#2c3e50"
            ),
            title=dict(
                text=f'{eu_priority} Scores Map',
                font=dict(size=16, color="#f4d03f", weight="bold"),  # Bold, smaller, yellow like top ribbon
                x=0.5
            ),
                    # Lighter hover box styling like Budget dashboard
        hoverlabel=dict(
            bgcolor="rgba(245, 245, 245, 0.9)",  # Light grey background for hover
            bordercolor="white",
            font=dict(color="black", size=12)
        )
    )
    
    # Set color scale from 0 to 1
    european_map.update_traces(
        zmin=0,
        zmax=1
    )
    
    # 2. Decile analysis chart
    decile_analysis = go.Figure()
    
    # For decile analysis, prioritize EU Countries Average, then add individual countries
    countries_to_show = []
    
    # Always show EU Countries Average first if available
    if 'EU Countries Average' in analysis_df['country'].values:
        countries_to_show.append('EU Countries Average')
    
    # Then add any other selected countries (excluding EU Countries Average to avoid duplication)
    other_countries = [c for c in analysis_df['country'].unique() if c != 'EU Countries Average' and 'Average' not in c]
    countries_to_show.extend(other_countries)
    
    # Show countries in the determined order
    for country in countries_to_show:
        if eu_priority in analysis_df.columns:
            country_data = analysis_df[analysis_df['country'] == country].sort_values('decile')
            decile_analysis.add_trace(go.Bar(
                x=[f'Decile {d}' for d in country_data['decile']],
                y=country_data[eu_priority],
                name=country,
                text=country_data[eu_priority].round(3),
                textposition='auto'
            ))
    
    decile_analysis.update_layout(
        title=f'{eu_priority} Scores by Decile - Selected Countries',
        xaxis_title='Income Decile',
        yaxis_title='Score',
        height=400,
        barmode='group',
        font=dict(family='Arial, sans-serif', size=12),
        yaxis=dict(range=[0, 1])  # Set y-axis scale from 0 to 1
    )
    
    # 3. Country comparison chart (EU priority vs secondary indicators)
    radar_chart = go.Figure()
    
    # Get all individual countries (excluding aggregates) for the comparison
    individual_countries = [c for c in map_df['country'].unique() if 'Average' not in c]
    individual_countries.sort()  # Sort alphabetically
    
    # Get the EU priority score for each country
    eu_priority_scores = []
    for country in individual_countries:
        if eu_priority in map_df.columns:
            country_score = map_df[map_df['country'] == country][eu_priority].mean()
            eu_priority_scores.append(country_score)
        else:
            eu_priority_scores.append(0)
    
    # Add the main EU priority line
    radar_chart.add_trace(
        go.Scatter(
            x=individual_countries,
            y=eu_priority_scores,
            name=eu_priority,
            mode='lines+markers',
            line=dict(width=4, color='#1f77b4'),
            marker=dict(
                size=10,
                color='white',
                line=dict(color='#1f77b4', width=2)
            ),
            hovertemplate='%{y:.3f}'
        )
    )
    
    # Add secondary indicator lines
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']  # Different colors for each line
    
    for i, secondary_col in enumerate(secondary_cols):
        if i < len(colors):  # Make sure we don't run out of colors
            secondary_scores = []
            for country in individual_countries:
                if secondary_col in map_df.columns:
                    country_score = map_df[map_df['country'] == country][secondary_col].mean()
                    secondary_scores.append(country_score)
                else:
                    secondary_scores.append(0)
            
            # Clean up the secondary indicator name for display
            secondary_name = secondary_col.replace(f"{priority_name_clean}_", "")
            
            radar_chart.add_trace(
                go.Scatter(
                    x=individual_countries,
                    y=secondary_scores,
                    name=secondary_name,
                    mode='lines+markers',
                    line=dict(color=colors[i]),
                    marker=dict(
                        size=8,
                        color='white',
                        line=dict(color=colors[i], width=2)
                    ),
                    hovertemplate='%{y:.3f}'
                )
            )
    
    # Update layout to match your styling
    radar_chart.update_layout(
        title=dict(
            text=f'{eu_priority} and Secondary Indicators by Country',
            y=0.9,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=16, color="#2c3e50", weight="bold")
        ),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            range=[0, 1],
            tickformat='.1f',
            gridwidth=0.2,
            title='Score'
        ),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top'
        ),
        height=400,
        font=dict(family='Arial, sans-serif', size=12)
    )
    
    # Set y-axis scale from 0 to 1
    radar_chart.update_layout(yaxis=dict(range=[0, 1]))
    
    # 4. Time series chart - Bar chart implementation
    time_series = go.Figure()
    
    # For time series, we'll use the pre-calculated historical EU priority scores
    # Load the historical EWBI data which contains EU priority scores over time
    try:
        historical_ewbi_df = pd.read_csv(os.path.join(DATA_DIR, 'historical_ewbi_scores.csv'))
        
        # Check if the EU priority exists in the historical data
        if eu_priority in historical_ewbi_df.columns:
            # Create time series data by averaging across deciles for each country and year
            eu_priority_time_data = historical_ewbi_df.groupby(['country', 'year'])[eu_priority].mean().reset_index()
            
            # For time series, prioritize EU Countries Average, then add individual countries
            countries_to_show = []
            
            # Always show EU Countries Average first if available
            if 'EU Countries Average' in analysis_df['country'].values:
                countries_to_show.append('EU Countries Average')
            
            # Then add any other selected countries (excluding EU Countries Average to avoid duplication)
            other_countries = [c for c in analysis_df['country'].values if c != 'EU Countries Average' and 'Average' not in c]
            countries_to_show.extend(other_countries)
            
            # Show countries in the determined order
            for country in countries_to_show:
                if country in eu_priority_time_data['country'].values:
                    country_data = eu_priority_time_data[eu_priority_time_data['country'] == country].sort_values('year')
                    time_series.add_trace(go.Scatter(
                        x=country_data['year'],
                        y=country_data[eu_priority],
                        mode='lines+markers',
                        name=country,
                        line=dict(width=2),
                        hovertemplate=f'<b>{country}</b><br>{eu_priority} Score: %{{y:.3f}}<br>Year: %{{x}}<extra></extra>'
                    ))
        else:
            # EU priority not found in historical data
            time_series.add_annotation(
                text=f"Historical data for {eu_priority} not found in the dataset",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            
    except FileNotFoundError:
        # Historical data file not found
        time_series.add_annotation(
            text=f"Historical EWBI data not found. Please run create_master_dataframe.py first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
    
    time_series.update_layout(
        title=f'{eu_priority} Evolution Over Time',
        xaxis_title='Year',
        yaxis_title='Score',
        height=400,
        barmode='group',
        font=dict(family='Arial, sans-serif', size=12),
        yaxis=dict(range=[0, 1])  # Set y-axis scale from 0 to 1
    )
    
    return european_map, decile_analysis, radar_chart, time_series

def create_primary_indicator_charts(map_df, analysis_df, time_df, eu_priority, secondary_indicator, primary_indicator):
    """Create charts for primary indicator level"""
    
    # 1. European map chart (primary indicator scores)
    european_map = go.Figure()
    
    # Get the primary indicator column name
    primary_col = f"primary_{primary_indicator}"
    
    if primary_col in map_df.columns:
        # Create choropleth map for the primary indicator
        european_map.add_trace(
            go.Choropleth(
                locations=[ISO2_TO_ISO3.get(country, country) for country in map_df['country']],
                z=map_df[primary_col],
                text=map_df['country'],
                locationmode='ISO-3',
                colorscale='Viridis',
                zmin=0,
                zmax=1,
                colorbar=dict(title='Score', tickformat='.3f'),
                hovertemplate='<b>%{text}</b><br>Score: %{z:.3f}<extra></extra>'
            )
        )
        
        european_map.update_layout(
            title=f'{primary_indicator} Scores by Country',
            geo=dict(
                scope='europe',
                projection=dict(type='natural earth'),
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(204, 204, 204)',
                showocean=True,
                oceancolor='rgb(230, 230, 250)',
                showcountries=True,
                countrycolor='rgb(128, 128, 128)',
                showframe=False
            ),
            height=550,
            font=dict(family='Arial, sans-serif', size=12)
        )
    else:
        european_map.add_annotation(
            text=f"Primary indicator {primary_indicator} not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    
    # 2. Decile analysis chart
    decile_analysis = go.Figure()
    
    if primary_col in analysis_df.columns:
        # Get decile data for the selected countries
        decile_data = analysis_df.groupby('decile')[primary_col].mean().reset_index()
        
        decile_analysis.add_trace(
            go.Bar(
                x=decile_data['decile'],
                y=decile_data[primary_col],
                marker_color='#3498db',
                hovertemplate='Decile %{x}: %{y:.3f}<extra></extra>'
            )
        )
        
        decile_analysis.update_layout(
            title=f'{primary_indicator} Scores by Income Decile',
            xaxis_title='Income Decile',
            yaxis_title='Score',
            height=400,
            font=dict(family='Arial, sans-serif', size=12),
            yaxis=dict(range=[0, 1])
        )
    else:
        decile_analysis.add_annotation(
            text=f"Primary indicator {primary_indicator} not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    
    # 3. Country comparison chart (primary indicator vs other primary indicators in same secondary)
    radar_chart = go.Figure()
    
    # Get all primary indicators for the same secondary indicator
    if primary_col in map_df.columns:
        # Find other primary indicators in the same secondary group
        # For now, we'll show the selected primary indicator vs a few others
        other_primary_cols = [col for col in map_df.columns if col.startswith('primary_') and col != primary_col][:3]
        
        # Get individual countries (excluding aggregates)
        individual_countries = [c for c in map_df['country'].unique() if 'Average' not in c]
        individual_countries.sort()
        
        # Add the main primary indicator
        main_scores = []
        for country in individual_countries:
            if country in map_df['country'].values:
                score = map_df[map_df['country'] == country][primary_col].mean()
                main_scores.append(score)
            else:
                main_scores.append(0)
        
        radar_chart.add_trace(
            go.Scatter(
                x=individual_countries,
                y=main_scores,
                name=primary_indicator,
                mode='lines+markers',
                line=dict(width=4, color='#1f77b4'),
                marker=dict(
                    size=10,
                    color='white',
                    line=dict(color='#1f77b4', width=2)
                ),
                hovertemplate='%{y:.3f}'
            )
        )
        
        # Add other primary indicators
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        for i, col in enumerate(other_primary_cols):
            if i < len(colors):
                other_scores = []
                for country in individual_countries:
                    if country in map_df['country'].values:
                        score = map_df[map_df['country'] == country][col].mean()
                        other_scores.append(score)
                    else:
                        other_scores.append(0)
                
                other_name = col.replace('primary_', '')
                
                radar_chart.add_trace(
                    go.Scatter(
                        x=individual_countries,
                        y=other_scores,
                        name=other_name,
                        mode='lines+markers',
                        line=dict(color=colors[i]),
                        marker=dict(
                            size=8,
                            color='white',
                            line=dict(color=colors[i], width=2)
                        ),
                        hovertemplate='%{y:.3f}'
                    )
                )
        
        radar_chart.update_layout(
            title=f'{primary_indicator} vs Other Primary Indicators',
            xaxis=dict(tickangle=45, tickfont=dict(size=12)),
            yaxis=dict(range=[0, 1], tickformat='.1f', gridwidth=0.2, title='Score'),
            hovermode='closest',
            showlegend=True,
            legend=dict(x=1, y=1, xanchor='right', yanchor='top'),
            height=400,
            font=dict(family='Arial, sans-serif', size=12)
        )
    else:
        radar_chart.add_annotation(
            text=f"Primary indicator {primary_indicator} not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    
    # 4. Time series chart
    time_series = go.Figure()
    
    # For time series, we'll use the time series dataframe
    try:
        time_series_df = pd.read_csv(os.path.join(DATA_DIR, 'master_dataframe_time_series.csv'))
        
        # Filter for the specific primary indicator
        if 'primary_index' in time_series_df.columns:
            indicator_time_data = time_series_df[time_series_df['primary_index'] == primary_indicator]
            
            if not indicator_time_data.empty:
                # Group by country and year, average across deciles
                country_year_data = indicator_time_data.groupby(['country', 'year'])['value'].mean().reset_index()
                
                # For time series, prioritize EU Countries Average, then add individual countries
                countries_to_show = []
                if 'EU Countries Average' in country_year_data['country'].values:
                    countries_to_show.append('EU Countries Average')
                
                # Add individual countries from the filter
                individual_countries = [c for c in analysis_df['country'].unique() if 'Average' not in c]
                countries_to_show.extend(individual_countries[:5])  # Limit to 5 for readability
                
                for country in countries_to_show:
                    if country in country_year_data['country'].values:
                        country_data = country_year_data[country_year_data['country'] == country]
                        country_data = country_data.sort_values('year')
                        
                        time_series.add_trace(
                            go.Scatter(
                                x=country_data['year'],
                                y=country_data['value'],
                                name=country,
                                mode='lines+markers',
                                hovertemplate='%{y:.3f}<extra></extra>'
                            )
                        )
                
                time_series.update_layout(
                    title=f'{primary_indicator} Evolution Over Time',
                    xaxis_title='Year',
                    yaxis_title='Score',
                    height=400,
                    font=dict(family='Arial, sans-serif', size=12),
                    yaxis=dict(range=[0, 1])
                )
            else:
                time_series.add_annotation(
                    text=f"No time series data found for {primary_indicator}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
        else:
            time_series.add_annotation(
                text=f"Time series data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
    except:
        time_series.add_annotation(
            text=f"Error loading time series data for {primary_indicator}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    
    return european_map, decile_analysis, radar_chart, time_series

def create_secondary_indicator_charts(map_df, analysis_df, time_df, eu_priority, secondary_indicator):
    """Create charts for secondary indicator level (EU Priority + Specific Secondary)"""
    
    # 1. European map chart (secondary indicator scores)
    european_map = go.Figure()
    
    # Get the secondary indicator column name
    secondary_col = f"secondary_{eu_priority.replace(' ', '_and_')}_{secondary_indicator.replace(' ', '_and_')}"
    
    if secondary_col in map_df.columns:
        # Create choropleth map for the secondary indicator
        european_map.add_trace(
            go.Choropleth(
                locations=[ISO2_TO_ISO3.get(country, country) for country in map_df['country']],
                z=map_df[secondary_col],
                text=map_df['country'],
                locationmode='ISO-3',
                colorscale='Viridis',
                zmin=0,
                zmax=1,
                colorbar=dict(title='Score', tickformat='.3f'),
                hovertemplate='<b>%{text}</b><br>Score: %{z:.3f}<extra></extra>'
            )
        )
        
        european_map.update_layout(
            title=f'{secondary_indicator} Scores by Country',
            geo=dict(
                scope='europe',
                projection=dict(type='natural earth'),
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(204, 204, 204)',
                showocean=True,
                oceancolor='rgb(230, 230, 250)',
                showcountries=True,
                countrycolor='rgb(128, 128, 128)',
                showframe=False
            ),
            height=550,
            font=dict(family='Arial, sans-serif', size=12)
        )
    else:
        european_map.add_annotation(
            text=f"Secondary indicator {secondary_indicator} not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    
    # 2. Decile analysis chart
    decile_analysis = go.Figure()
    
    if secondary_col in analysis_df.columns:
        # Get decile data for the selected countries
        decile_data = analysis_df.groupby('decile')[secondary_col].mean().reset_index()
        
        decile_analysis.add_trace(
            go.Bar(
                x=decile_data['decile'],
                y=decile_data[secondary_col],
                marker_color='#3498db',
                hovertemplate='Decile %{x}: %{y:.3f}<extra></extra>'
            )
        )
        
        decile_analysis.update_layout(
            title=f'{secondary_indicator} Scores by Income Decile',
            xaxis_title='Income Decile',
            yaxis_title='Score',
            height=400,
            font=dict(family='Arial, sans-serif', size=12),
            yaxis=dict(range=[0, 1])
        )
    else:
        decile_analysis.add_annotation(
            text=f"Secondary indicator {secondary_indicator} not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    
    # 3. Country comparison chart (secondary indicator vs primary indicators)
    radar_chart = go.Figure()
    
    # Get all primary indicators for this secondary indicator
    if secondary_col in map_df.columns:
        # Find primary indicators that belong to this secondary indicator
        # We need to get this from the EWBI structure
        try:
            with open(os.path.join(DATA_DIR, '..', 'data', 'ewbi_indicators.json'), 'r') as f:
                config = json.load(f)['EWBI']
            
            primary_indicators = []
            for priority in config:
                if priority['name'] == eu_priority:
                    for component in priority['components']:
                        if component['name'] == secondary_indicator:
                            for indicator in component['indicators']:
                                primary_indicators.append(indicator['code'])
                            break
                    break
            
            # Get individual countries (excluding aggregates)
            individual_countries = [c for c in map_df['country'].unique() if 'Average' not in c]
            individual_countries.sort()
            
            # Add the secondary indicator line
            secondary_scores = []
            for country in individual_countries:
                if country in map_df['country'].values:
                    score = map_df[map_df['country'] == country][secondary_col].mean()
                    secondary_scores.append(score)
                else:
                    secondary_scores.append(0)
            
            radar_chart.add_trace(
                go.Scatter(
                    x=individual_countries,
                    y=secondary_scores,
                    name=secondary_indicator,
                    mode='lines+markers',
                    line=dict(width=4, color='#1f77b4'),
                    marker=dict(
                        size=10,
                        color='white',
                        line=dict(color='#1f77b4', width=2)
                    ),
                    hovertemplate='%{y:.3f}'
                )
            )
            
            # Add primary indicator lines
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            for i, primary_code in enumerate(primary_indicators):
                if i < len(colors):
                    primary_col = f"primary_{primary_code}"
                    if primary_col in map_df.columns:
                        primary_scores = []
                        for country in individual_countries:
                            if country in map_df['country'].values:
                                score = map_df[map_df['country'] == country][primary_col].mean()
                                primary_scores.append(score)
                            else:
                                primary_scores.append(0)
                        
                        radar_chart.add_trace(
                            go.Scatter(
                                x=individual_countries,
                                y=primary_scores,
                                name=primary_code,
                                mode='lines+markers',
                                line=dict(color=colors[i]),
                                marker=dict(
                                    size=8,
                                    color='white',
                                    line=dict(color=colors[i], width=2)
                                ),
                                hovertemplate='%{y:.3f}'
                            )
                        )
            
            radar_chart.update_layout(
                title=f'{secondary_indicator} vs Primary Indicators by Country',
                xaxis=dict(tickangle=45, tickfont=dict(size=12)),
                yaxis=dict(range=[0, 1], tickformat='.1f', gridwidth=0.2, title='Score'),
                hovermode='closest',
                showlegend=True,
                legend=dict(x=1, y=1, xanchor='right', yanchor='top'),
                height=400,
                font=dict(family='Arial, sans-serif', size=12)
            )
        except:
            radar_chart.add_annotation(
                text=f"Error loading EWBI structure for {secondary_indicator}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
    else:
        radar_chart.add_annotation(
            text=f"Secondary indicator {secondary_indicator} not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    
    # 4. Time series chart
    time_series = go.Figure()
    
    # For time series, we'll use the time series dataframe
    try:
        time_series_df = pd.read_csv(os.path.join(DATA_DIR, 'master_dataframe_time_series.csv'))
        
        # Filter for the specific secondary indicator
        if 'secondary_indicator' in time_series_df.columns:
            indicator_time_data = time_series_df[time_series_df['secondary_indicator'] == secondary_indicator]
            
            if not indicator_time_data.empty:
                # Group by country and year, average across deciles
                country_year_data = indicator_time_data.groupby(['country', 'year'])['value'].mean().reset_index()
                
                # For time series, prioritize EU Countries Average, then add individual countries
                countries_to_show = []
                if 'EU Countries Average' in country_year_data['country'].values:
                    countries_to_show.append('EU Countries Average')
                
                # Add individual countries from the filter
                individual_countries = [c for c in analysis_df['country'].unique() if 'Average' not in c]
                countries_to_show.extend(individual_countries[:5])  # Limit to 5 for readability
                
                for country in countries_to_show:
                    if country in country_year_data['country'].values:
                        country_data = country_year_data[country_year_data['country'] == country]
                        country_data = country_data.sort_values('year')
                        
                        time_series.add_trace(
                            go.Scatter(
                                x=country_data['year'],
                                y=country_data['value'],
                                name=country,
                                mode='lines+markers',
                                hovertemplate='%{y:.3f}<extra></extra>'
                            )
                        )
                
                time_series.update_layout(
                    title=f'{secondary_indicator} Evolution Over Time',
                    xaxis_title='Year',
                    yaxis_title='Score',
                    height=400,
                    font=dict(family='Arial, sans-serif', size=12),
                    yaxis=dict(range=[0, 1])
                )
            else:
                time_series.add_annotation(
                    text=f"No time series data found for {secondary_indicator}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
        else:
            time_series.add_annotation(
                text=f"Time series data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
    except:
        time_series.add_annotation(
            text=f"Error loading time series data for {secondary_indicator}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    
    return european_map, decile_analysis, radar_chart, time_series

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)