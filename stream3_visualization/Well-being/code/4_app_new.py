import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import os
import json
import sys
import numpy as np

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from variable_mapping import get_display_name, get_acronym_from_display_name

# Get the absolute path to the data directory - MODIFIED to use unified data
# Handle both local development and CleverCloud deployment paths
if os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'output')):
    # Local development path
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
else:
    # CleverCloud deployment path (from repository root)
    DATA_DIR = os.path.abspath(os.path.join('stream3_visualization', 'Well-being', 'output'))

# ISO-2 to ISO-3 mapping for European countries
ISO2_TO_ISO3 = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'CH': 'CHE', 'CY': 'CYP', 'CZ': 'CZE',
    'DE': 'DEU', 'DK': 'DNK', 'EE': 'EST', 'EL': 'GRC', 'ES': 'ESP', 'FI': 'FIN',
    'FR': 'FRA', 'HR': 'HRV', 'HU': 'HUN', 'IE': 'IRL', 'IS': 'ISL', 'IT': 'ITA',
    'LT': 'LTU', 'LU': 'LUX', 'LV': 'LVA', 'MT': 'MLT', 'NL': 'NLD', 'NO': 'NOR',
    'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'RS': 'SRB', 'SE': 'SWE', 'SI': 'SVN',
    'SK': 'SVK', 'UK': 'GBR'
}

# ISO-2 to full country names mapping
ISO2_TO_FULL_NAME = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CH': 'Switzerland', 'CY': 'Cyprus', 'CZ': 'Czech Republic',
    'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland',
    'FR': 'France', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IS': 'Iceland', 'IT': 'Italy',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta', 'NL': 'Netherlands', 'NO': 'Norway',
    'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia', 'SE': 'Sweden', 'SI': 'Slovenia',
    'SK': 'Slovakia', 'UK': 'United Kingdom'
}

# Full country names to ISO-2 mapping (reverse)
FULL_NAME_TO_ISO2 = {v: k for k, v in ISO2_TO_FULL_NAME.items()}

# EU Aggregate country name - MODIFIED to use the new aggregation name
EU_AGGREGATE = 'All Countries'

# Helper function to check if an indicator is from EHIS datasource
def is_ehis_indicator(primary_indicator):
    """Check if a primary indicator comes from EHIS datasource (uses quintiles instead of deciles)"""
    if not primary_indicator or pd.isna(primary_indicator):
        return False
    
    # Check if we have Level 5 data for this indicator
    level5_data = unified_df[
        (unified_df['Level'] == 5) & 
        (unified_df['Primary and raw data'] == primary_indicator)
    ]
    
    if level5_data.empty:
        return False
    
    # Check if this indicator comes from EHIS datasource
    return (level5_data['datasource'] == 'EHIS').any()

# Load the unified data from 3_generate_outputs.py - COMPLETELY MODIFIED DATA LOADING
try:
    unified_df = pd.read_csv(os.path.join(DATA_DIR, 'unified_all_levels_1_to_5.csv'))
    
    # Load EWBI structure for hierarchical navigation
    with open(os.path.join(DATA_DIR, '..', 'data', 'ewbi_indicators.json'), 'r') as f:
        ewbi_structure = json.load(f)['EWBI']

    print("Unified data file loaded successfully.")
    print(f"Data shape: {unified_df.shape}")
    print(f"Columns: {list(unified_df.columns)}")
    print(f"Levels: {sorted(unified_df['Level'].unique())}")
    print(f"Countries: {unified_df['Country'].nunique()}")
    print(f"Years: {unified_df['Year'].min()}-{unified_df['Year'].max()}")

    # Get unique countries and sort them (excluding aggregates for individual selection)
    COUNTRIES = sorted([c for c in unified_df['Country'].unique() 
                       if c != 'All Countries' and 'Average' not in c and 'Median' not in c])

    print(f"Found {len(COUNTRIES)} individual countries")

    # Create EU priority options
    EU_PRIORITIES = [
        'Agriculture and Food',
        'Energy and Housing', 
        'Equality',
        'Health and Animal Welfare',
        'Intergenerational Fairness, Youth, Culture and Sport',
        'Social Rights and Skills, Quality Jobs and Preparedness'
        #,        'Sustainable Transport and Tourism'
    ]

    # Create secondary indicator options - get from the unified data
    SECONDARY_INDICATORS = []
    secondary_options = unified_df[unified_df['Secondary'].notna()]['Secondary'].unique()
    
    for secondary in sorted(secondary_options):
        # Get the EU priority for this secondary
        eu_priority_for_secondary = unified_df[
            (unified_df['Secondary'] == secondary) & 
            (unified_df['EU priority'].notna())
        ]['EU priority'].iloc[0] if len(unified_df[
            (unified_df['Secondary'] == secondary) & 
            (unified_df['EU priority'].notna())
        ]) > 0 else 'Unknown'
        
        SECONDARY_INDICATORS.append({
            'label': f"{eu_priority_for_secondary} - {secondary}",
            'value': f"{eu_priority_for_secondary}|{secondary}",
            'eu_priority': eu_priority_for_secondary,
            'secondary': secondary
        })

    print(f"Dashboard initialized with {len(EU_PRIORITIES)} EU priorities and {len(SECONDARY_INDICATORS)} secondary indicators")

except FileNotFoundError as e:
    print(f"Error loading unified data file: {e}")
    print(f"Please ensure that unified_all_levels_1_to_5.csv is in the '{DATA_DIR}' directory.")
    exit()

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
    # Header section with hierarchical titles
    html.Div([
        html.H1("European Well-Being Index (EWBI)", className="dashboard-title"),
        html.H2([
            "Multi-level Analysis of Well-being Indicators Across Europe"
        ], className="dashboard-subtitle"),

        # Controls section: Data types on top, Country/Normalize below
        html.Div([
            # Top row: Data Type dropdowns side by side
            html.Div([
                html.Div([
                    html.Label("EU Priority", className="control-label"),
                    dcc.Dropdown(
                        id='eu-priority-dropdown',
                        options=[{'label': '(Select value)', 'value': 'ALL'}] + [{'label': prio, 'value': prio} for prio in EU_PRIORITIES],
                        value='ALL',
                        style={'marginTop': '0px'},
                        clearable=False
                    ),
                ], className="control-item"),
                html.Div([
                    html.Label("Secondary Indicator", className="control-label"),
                    dcc.Dropdown(
                        id='secondary-indicator-dropdown',
                        options=[{'label': '(Select value)', 'value': 'ALL'}],
                        value='ALL',
                        style={'marginTop': '0px'},
                        clearable=False,
                        disabled=True
                    ),
                ], className="control-item"),
                html.Div([
                    html.Label("Raw Data", className="control-label"),
                    dcc.Dropdown(
                        id='primary-indicator-dropdown',
                        options=[{'label': '(Select value)', 'value': 'ALL'}],
                        value='ALL',
                        style={'marginTop': '0px'},
                        clearable=False,
                        disabled=True
                    ),
                ], className="control-item"),
            ], className="controls-row"),
            # Bottom row: Country selector
            html.Div([
                html.Div([
                    html.Label("Country Selector", className="control-label"),
                    dcc.Dropdown(
                        id='countries-filter',
                        options=[{'label': ISO2_TO_FULL_NAME.get(country, country), 'value': country} for country in COUNTRIES],
                        value=[],
                        multi=True,
                        placeholder='Add countries for comparison',
                        style={'marginTop': '0px'}
                    ),
                ], style={'width': '60%', 'minWidth': '180px', 'maxWidth': '340px'}),
            ], className="controls-row"),
        ], className="controls-section")
    ], className="dashboard-header"),

    # Responsive 2x2 Visualization Grid
    html.Div([
        # Grid Item 1: European Map
        html.Div([
            dcc.Graph(id='european-map-chart', config={'responsive': True})
        ], className="grid-item"),
        
        # Grid Item 2: Time Series Chart
        html.Div([
            dcc.Graph(id='time-series-chart', config={'responsive': True})
        ], className="grid-item"),
        
        # Grid Item 3: Decile Analysis Chart
        html.Div([
            dcc.Graph(id='decile-analysis-chart', config={'responsive': True})
        ], className="grid-item"),
        
        # Grid Item 4: Radar Chart
        html.Div([
            dcc.Graph(id='radar-chart', config={'responsive': True})
        ], className="grid-item"),
    ], className="visualization-grid"),

    # Sources section
    html.Div([
        html.H3("Data Sources", className="sources-title"),
        html.Div([
            html.H4("European Well-Being Index (EWBI)", className="sources-subtitle"),
            html.P("The EWBI is a composite indicator measuring well-being across European countries, constructed from multiple data sources including EU-SILC, EHIS, and HBS surveys.", className="sources-text"),
            html.H4("Indicator Structure", className="sources-subtitle"),
            html.P("The EWBI follows a hierarchical structure:", className="sources-text"),
            html.P([html.Strong("Level 1: "), "EWBI - Overall well-being score (geometric mean of EU priorities)"], className="sources-text"),
            html.P([html.Strong("Level 2: "), "EU Priorities - Major policy areas"], className="sources-text"),
            html.P([html.Strong("Level 3: "), "Secondary Indicators - Specific well-being dimensions"], className="sources-text"),
            html.P([html.Strong("Level 5: "), "Raw Statistics - Individual survey questions and measurements"], className="sources-text"),
        ], className="sources-content")
    ], className="sources-section")
], className="dashboard-container")

# Callback to update secondary indicator dropdown based on EU priority
@app.callback(
    Output('secondary-indicator-dropdown', 'options'),
    Output('secondary-indicator-dropdown', 'value'),
    Output('secondary-indicator-dropdown', 'disabled'),
    Input('eu-priority-dropdown', 'value')
)
def update_secondary_indicator_dropdown(eu_priority):
    if eu_priority == 'ALL':
        return [{'label': '(Select value)', 'value': 'ALL'}], 'ALL', True
    elif eu_priority:
        # Special case: For 'Agriculture and Food', always show only 'Nutrition'
        if eu_priority == 'Agriculture and Food':
            options = [{'label': '(Select value)', 'value': 'ALL'}, {'label': 'Nutrition', 'value': 'Nutrition'}]
            return options, 'ALL', False
        filtered_indicators = [
            indicator for indicator in SECONDARY_INDICATORS
            if indicator['eu_priority'] == eu_priority
        ]
        options = [{'label': '(Select value)', 'value': 'ALL'}] + [
            {'label': indicator['secondary'], 'value': indicator['secondary']}
            for indicator in filtered_indicators
        ]
        return options, 'ALL', False
    else:
        return [{'label': '(Select value)', 'value': 'ALL'}], 'ALL', True

# Callback to update primary indicator dropdown based on secondary indicator
@app.callback(
    Output('primary-indicator-dropdown', 'options'),
    Output('primary-indicator-dropdown', 'value'),
    Output('primary-indicator-dropdown', 'disabled'),
    Input('secondary-indicator-dropdown', 'value'),
    Input('eu-priority-dropdown', 'value')
)
def update_primary_indicator_dropdown(secondary_indicator, eu_priority):
    if eu_priority == 'ALL':
        return [{'label': '(Select value)', 'value': 'ALL'}], 'ALL', True
    elif eu_priority and secondary_indicator and secondary_indicator != 'ALL':
        # Get primary indicators from the unified data
        primary_options = unified_df[
            (unified_df['EU priority'] == eu_priority) & 
            (unified_df['Secondary'] == secondary_indicator) &
            (unified_df['Primary and raw data'].notna()) &
            (unified_df['Level'] == 5)
        ]['Primary and raw data'].unique()
        
        if len(primary_options) > 0:
            options = [{'label': '(Select value)', 'value': 'ALL'}] + [
                {'label': get_display_name(primary), 'value': primary} for primary in sorted(primary_options)
            ]
            return options, 'ALL', False
        else:
            return [{'label': '(Select value)', 'value': 'ALL'}], 'ALL', True
    else:
        return [{'label': '(Select value)', 'value': 'ALL'}], 'ALL', True

# Main callback to update all charts
@app.callback(
    [Output('european-map-chart', 'figure'),
     Output('time-series-chart', 'figure'),
     Output('decile-analysis-chart', 'figure'),
     Output('radar-chart', 'figure')],
    [Input('eu-priority-dropdown', 'value'),
     Input('secondary-indicator-dropdown', 'value'),
     Input('primary-indicator-dropdown', 'value'),
     Input('countries-filter', 'value')]
)
def update_charts(eu_priority, secondary_indicator, primary_indicator, selected_countries):
    # Determine current level based on filters
    level_filters = create_level_filters(eu_priority, secondary_indicator, primary_indicator)
    
    # Create all charts using the unified data
    map_chart = create_map_chart(level_filters)
    time_series_chart = create_time_series_chart(level_filters, selected_countries)
    decile_chart = create_decile_chart(level_filters, selected_countries)
    radar_chart = create_radar_chart(level_filters, selected_countries)
    
    return map_chart, time_series_chart, decile_chart, radar_chart

def create_level_filters(eu_priority, secondary_indicator, primary_indicator):
    """Create level-based filters for the dashboard - MODIFIED for new level structure"""
    level_filters = {
        'eu_priority': eu_priority,
        'secondary_indicator': secondary_indicator,
        'primary_indicator': primary_indicator
    }

    # Determine the current level based on filter combinations
    if eu_priority == 'ALL' or not eu_priority:
        # Level 1: EWBI (overall)
        level_filters['current_level'] = 1
        level_filters['level_name'] = 'EWBI (Overall Well-being)'
    elif eu_priority != 'ALL' and (secondary_indicator == 'ALL' or not secondary_indicator):
        # Level 2: EU Priority selected
        level_filters['current_level'] = 2
        level_filters['level_name'] = f'EU Priority: {eu_priority}'
    elif eu_priority != 'ALL' and secondary_indicator != 'ALL' and (primary_indicator == 'ALL' or not primary_indicator):
        # Level 3: Secondary selected
        level_filters['current_level'] = 3
        level_filters['level_name'] = f'Secondary: {secondary_indicator}'
    elif primary_indicator and primary_indicator != 'ALL':
        # Level 5: Primary/Raw data selected (skip level 4 as requested)
        level_filters['current_level'] = 5
        level_filters['level_name'] = f'Raw Data: {get_display_name(primary_indicator)}'
    else:
        level_filters['current_level'] = 1
        level_filters['level_name'] = 'EWBI (Overall Well-being)'

    return level_filters

def create_map_chart(level_filters):
    """Create map chart showing latest year values per country for the selected level"""
    
    # Get latest year available - for Level 5, use latest year for the specific indicator
    if level_filters['current_level'] == 5:
        # For Level 5, get latest year available for this specific primary indicator
        indicator_data = unified_df[
            (unified_df['Level'] == 5) & 
            (unified_df['Primary and raw data'] == level_filters['primary_indicator'])
        ]
        latest_year = indicator_data['Year'].max() if not indicator_data.empty else unified_df['Year'].max()
    else:
        latest_year = unified_df['Year'].max()
    
    # Filter data based on level and get "All" deciles only (inter-decile geometric mean for levels 1-3, statistical value for level 5)
    if level_filters['current_level'] == 1:
        # Level 1: All EU priorities
        map_data = unified_df[
            (unified_df['Level'] == 1) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] == 'All') &
            (unified_df['Country'] != 'All Countries')
        ].copy()
        title = f'EWBI Overall Well-being Score by Country ({int(latest_year)})'
        
    elif level_filters['current_level'] == 2:
        # Level 2: Specific EU priority
        map_data = unified_df[
            (unified_df['Level'] == 2) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] == 'All') &
            (unified_df['EU priority'] == level_filters['eu_priority']) &
            (unified_df['Country'] != 'All Countries')
        ].copy()
        title = f'{level_filters["eu_priority"]} Score by Country ({int(latest_year)})'
        
    elif level_filters['current_level'] == 3:
        # Level 3: Specific secondary indicator
        map_data = unified_df[
            (unified_df['Level'] == 3) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] == 'All') &
            (unified_df['Secondary'] == level_filters['secondary_indicator']) &
            (unified_df['Country'] != 'All Countries')
        ].copy()
        title = f'{level_filters["secondary_indicator"]} Score by Country ({int(latest_year)})'
        
    else:  # Level 5
        # Level 5: Raw statistical values
        # Check if this is EHIS data (uses quintiles) or other data (uses deciles)
        primary_indicator = level_filters['primary_indicator']
        is_ehis_indicator_flag = is_ehis_indicator(primary_indicator)
        
        if is_ehis_indicator_flag:
            # For EHIS data, filter by Quintile == 'All'
            map_data = unified_df[
                (unified_df['Level'] == 5) & 
                (unified_df['Year'] == latest_year) &
                (unified_df['Quintile'] == 'All') &
                (unified_df['Primary and raw data'] == primary_indicator) &
                (unified_df['Country'] != 'All Countries')
            ].copy()
        else:
            # For other data sources, filter by Decile == 'All'
            map_data = unified_df[
                (unified_df['Level'] == 5) & 
                (unified_df['Year'] == latest_year) &
                (unified_df['Decile'] == 'All') &
                (unified_df['Primary and raw data'] == primary_indicator) &
                (unified_df['Country'] != 'All Countries')
            ].copy()
        title = f'{get_display_name(primary_indicator)} Raw Values by Country ({int(latest_year)})'
    
    if map_data.empty:
        # No data available
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available for {level_filters['level_name']}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold"), x=0.5)
        )
        return fig
    
    # Convert country codes to ISO-3 and full names
    map_data['iso3'] = map_data['Country'].map(ISO2_TO_ISO3)
    map_data['full_name'] = map_data['Country'].map(ISO2_TO_FULL_NAME).fillna(map_data['Country'])
    
    # Filter out countries without ISO-3 codes
    map_data = map_data[map_data['iso3'].notna()].copy()
    
    if map_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No mappable countries found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold"), x=0.5)
        )
        return fig
    
    # Create choropleth map
    # For level 5 (raw indicators), invert color scale so green = minimum (better) and red = maximum (worse)
    if level_filters['current_level'] == 5:
        colorscale = 'RdYlGn_r'  # Reversed color scale for raw indicators
    else:
        colorscale = 'RdYlGn'  # Normal color scale for aggregated levels
    
    fig = go.Figure(data=go.Choropleth(
        locations=map_data['iso3'],
        z=map_data['Value'],
        locationmode='ISO-3',
        colorscale=colorscale,
        text=map_data['full_name'],
        hovertemplate='<b>%{text}</b><br>Score: %{z:.2f}<extra></extra>',
        colorbar=dict(title="Score")
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold"), x=0.5),
        geo=dict(
            scope='europe',
            showland=True,
            landcolor='lightgray',
            coastlinecolor='white',
            showocean=True,
            oceancolor='white'
        ),
        autosize=True,
        margin=dict(t=60, b=10, l=10, r=10)
    )
    
    return fig

def create_time_series_chart(level_filters, selected_countries):
    """Create time series chart showing evolution over time"""
    
    # Get countries to show: All Countries + selected individual countries
    countries_to_show = ['All Countries']
    if selected_countries:
        countries_to_show.extend(selected_countries)
    
    # Filter data based on level - same logic as map chart but across all years
    if level_filters['current_level'] == 1:
        # Level 1: EWBI overall
        # For "All Countries": use "Median across countries", for individual countries: no aggregation filter
        ts_data = unified_df[
            (unified_df['Level'] == 1) & 
            (unified_df['Decile'] == 'All') &
            (unified_df['Country'].isin(countries_to_show)) &
            (
                (unified_df['Country'] != 'All Countries') |
                ((unified_df['Country'] == 'All Countries') & (unified_df['Aggregation'] == 'Median across countries'))
            )
        ].copy()
        title = 'EWBI Overall Well-being Evolution Over Time'
        
    elif level_filters['current_level'] == 2:
        # Level 2: EU priority
        # For "All Countries": use "Median across countries", for individual countries: no aggregation filter
        ts_data = unified_df[
            (unified_df['Level'] == 2) & 
            (unified_df['Decile'] == 'All') &
            (unified_df['EU priority'] == level_filters['eu_priority']) &
            (unified_df['Country'].isin(countries_to_show)) &
            (
                (unified_df['Country'] != 'All Countries') |
                ((unified_df['Country'] == 'All Countries') & (unified_df['Aggregation'] == 'Median across countries'))
            )
        ].copy()
        title = f'{level_filters["eu_priority"]} Evolution Over Time'
        
    elif level_filters['current_level'] == 3:
        # Level 3: Secondary indicator
        # For "All Countries": use "Median across countries", for individual countries: no aggregation filter
        ts_data = unified_df[
            (unified_df['Level'] == 3) & 
            (unified_df['Decile'] == 'All') &
            (unified_df['Secondary'] == level_filters['secondary_indicator']) &
            (unified_df['Country'].isin(countries_to_show)) &
            (
                (unified_df['Country'] != 'All Countries') |
                ((unified_df['Country'] == 'All Countries') & (unified_df['Aggregation'] == 'Median across countries'))
            )
        ].copy()
        title = f'{level_filters["secondary_indicator"]} Evolution Over Time'
        
    elif level_filters['current_level'] == 4:
        # Level 4: Primary indicator
        # For "All Countries": use "Median across countries", for individual countries: no aggregation filter
        ts_data = unified_df[
            (unified_df['Level'] == 4) & 
            (unified_df['Decile'] == 'All') &
            (unified_df['Primary and raw data'] == level_filters['primary_indicator']) &
            (unified_df['Country'].isin(countries_to_show)) &
            (
                (unified_df['Country'] != 'All Countries') |
                ((unified_df['Country'] == 'All Countries') & (unified_df['Aggregation'] == 'Median across countries'))
            )
        ].copy()
        title = f'{get_display_name(level_filters["primary_indicator"])} Evolution Over Time'
        
    else:  # Level 5
        # Level 5: Raw data
        # Check if this is EHIS data (uses quintiles) or other data (uses deciles)
        primary_indicator = level_filters['primary_indicator']
        is_ehis_indicator_flag = is_ehis_indicator(primary_indicator)
        
        if is_ehis_indicator_flag:
            # For EHIS data, filter by Quintile == 'All'
            # For "All Countries": use "Median across countries", for individual countries: no aggregation filter
            ts_data = unified_df[
                (unified_df['Level'] == 5) & 
                (unified_df['Quintile'] == 'All') &
                (unified_df['Primary and raw data'] == primary_indicator) &
                (unified_df['Country'].isin(countries_to_show)) &
                (
                    (unified_df['Country'] != 'All Countries') |
                    ((unified_df['Country'] == 'All Countries') & (unified_df['Aggregation'] == 'Median across countries'))
                )
            ].copy()
        else:
            # For other data sources, filter by Decile == 'All'
            # For "All Countries": use "Median across countries", for individual countries: no aggregation filter
            ts_data = unified_df[
                (unified_df['Level'] == 5) & 
                (unified_df['Decile'] == 'All') &
                (unified_df['Primary and raw data'] == primary_indicator) &
                (unified_df['Country'].isin(countries_to_show)) &
                (
                    (unified_df['Country'] != 'All Countries') |
                    ((unified_df['Country'] == 'All Countries') & (unified_df['Aggregation'] == 'Median across countries'))
                )
            ].copy()
        title = f'{get_display_name(primary_indicator)} Evolution Over Time'
    
    fig = go.Figure()
    
    if ts_data.empty:
        fig.add_annotation(
            text=f"No time series data available for {level_filters['level_name']}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    else:
        # Add traces for each country
        for country in countries_to_show:
            country_data = ts_data[ts_data['Country'] == country].sort_values('Year')
            if not country_data.empty:
                fig.add_trace(go.Scatter(
                    x=country_data['Year'],
                    y=country_data['Value'],
                    mode='lines+markers',
                    name=ISO2_TO_FULL_NAME.get(country, country),
                    hovertemplate='<b>Score:</b> %{y:.2f}<br><b>Year:</b> %{x}<extra></extra>'
                ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold"), x=0.5),
        autosize=True,
        margin=dict(t=60, b=50, l=50, r=50),
        yaxis=dict(title='Score'),
        xaxis=dict(title='Year')
    )
    
    return fig

def create_decile_chart(level_filters, selected_countries):
    """Create decile/quintile chart for the latest year"""
    
    # Get latest year - for Level 5, use latest year for the specific indicator
    if level_filters['current_level'] == 5:
        # For Level 5, get latest year available for this specific primary indicator
        indicator_data = unified_df[
            (unified_df['Level'] == 5) & 
            (unified_df['Primary and raw data'] == level_filters['primary_indicator'])
        ]
        latest_year = indicator_data['Year'].max() if not indicator_data.empty else unified_df['Year'].max()
    else:
        latest_year = unified_df['Year'].max()
    
    # Get countries to show
    countries_to_show = ['All Countries']
    if selected_countries:
        countries_to_show.extend(selected_countries)
    
    # Determine if we're dealing with EHIS data (Level 5 only)
    is_ehis_data = (level_filters['current_level'] == 5 and 
                    is_ehis_indicator(level_filters.get('primary_indicator')))
    
    # Filter data based on level - exclude "All" deciles/quintiles
    # Special handling for Level 2: "All Countries" only has "All" decile, individual countries have 1-10 deciles
    if level_filters['current_level'] == 1:
        decile_data = unified_df[
            (unified_df['Level'] == 1) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] != 'All') &
            (unified_df['Aggregation'] == 'Geometric mean level-1') &
            (unified_df['Country'].isin(countries_to_show))
        ].copy()
        title = f'EWBI Scores by Decile ({int(latest_year)})'
        x_axis_title = 'Income Decile'
        
    elif level_filters['current_level'] == 2:
        decile_data = unified_df[
            (unified_df['Level'] == 2) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] != 'All') &
            (unified_df['EU priority'] == level_filters['eu_priority']) &
            (unified_df['Country'].isin(countries_to_show))
        ].copy()
        title = f'{level_filters["eu_priority"]} Scores by Decile ({int(latest_year)})'
        x_axis_title = 'Income Decile'
        
    elif level_filters['current_level'] == 3:
        decile_data = unified_df[
            (unified_df['Level'] == 3) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] != 'All') &
            (unified_df['Secondary'] == level_filters['secondary_indicator']) &
            (unified_df['Country'].isin(countries_to_show))
        ].copy()  
        title = f'{level_filters["secondary_indicator"]} Scores by Decile ({int(latest_year)})'
        x_axis_title = 'Income Decile'
        
    elif level_filters['current_level'] == 4:
        decile_data = unified_df[
            (unified_df['Level'] == 4) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] != 'All') &
            (unified_df['Primary and raw data'] == level_filters['primary_indicator']) &
            (unified_df['Country'].isin(countries_to_show))
        ].copy()
        title = f'{level_filters["primary_indicator"]} Scores by Decile ({int(latest_year)})'
        x_axis_title = 'Income Decile'
        
    else:  # Level 5
        if is_ehis_data:
            # For EHIS data, use quintiles instead of deciles
            decile_data = unified_df[
                (unified_df['Level'] == 5) & 
                (unified_df['Year'] == latest_year) &
                (unified_df['Quintile'].notna()) &  # Use quintiles for EHIS
                (unified_df['Quintile'] != 'All') &
                (unified_df['Primary and raw data'] == level_filters['primary_indicator']) &
                (unified_df['Country'].isin(countries_to_show))
            ].copy()
            title = f'{get_display_name(level_filters["primary_indicator"])} Values by Quintile ({int(latest_year)})'
            x_axis_title = 'Income Quintile'
        else:
            # For other data sources, use deciles
            decile_data = unified_df[
                (unified_df['Level'] == 5) & 
                (unified_df['Year'] == latest_year) &
                (unified_df['Decile'] != 'All') &
                (unified_df['Primary and raw data'] == level_filters['primary_indicator']) &
                (unified_df['Country'].isin(countries_to_show))
            ].copy()
            title = f'{get_display_name(level_filters["primary_indicator"])} Values by Decile ({int(latest_year)})'
            x_axis_title = 'Income Decile'
    
    fig = go.Figure()
    
    if decile_data.empty:
        data_type = "quintile" if is_ehis_data else "decile"
        fig.add_annotation(
            text=f"No {data_type} data available for {level_filters['level_name']}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    else:
        # Add "All" decile/quintile values for comparison
        # For EHIS data (quintiles), use Quintile='All'; for other data, use Decile='All'
        if is_ehis_data:
            all_aggregate_data = unified_df[
                (unified_df['Level'] == level_filters['current_level']) & 
                (unified_df['Year'] == latest_year) &
                (unified_df['Quintile'] == 'All') &
                (unified_df['Country'].isin(countries_to_show))
            ]
        else:
            all_aggregate_data = unified_df[
                (unified_df['Level'] == level_filters['current_level']) & 
                (unified_df['Year'] == latest_year) &
                (unified_df['Decile'] == 'All') &
                (unified_df['Country'].isin(countries_to_show))
            ]
        
        if level_filters['current_level'] == 1:
            all_aggregate_data = all_aggregate_data[all_aggregate_data['Aggregation'] == 'Geometric mean inter-decile']
        elif level_filters['current_level'] == 2:
            all_aggregate_data = all_aggregate_data[all_aggregate_data['EU priority'] == level_filters['eu_priority']]
        elif level_filters['current_level'] == 3:
            all_aggregate_data = all_aggregate_data[all_aggregate_data['Secondary'] == level_filters['secondary_indicator']]
        elif level_filters['current_level'] == 4:
            all_aggregate_data = all_aggregate_data[all_aggregate_data['Primary and raw data'] == level_filters['primary_indicator']]
        elif level_filters['current_level'] == 5:
            all_aggregate_data = all_aggregate_data[all_aggregate_data['Primary and raw data'] == level_filters['primary_indicator']]
        
        # Create traces for each country
        for country in countries_to_show:
            country_data = decile_data[decile_data['Country'] == country].copy()
            
            if not country_data.empty:
                if is_ehis_data:
                    # For EHIS data, use quintiles
                    country_data['Quintile_num'] = pd.to_numeric(country_data['Quintile'], errors='coerce')
                    country_data = country_data.sort_values('Quintile_num')
                    
                    # Add quintile bars
                    fig.add_trace(go.Bar(
                        x=[str(int(q)) for q in country_data['Quintile_num']],
                        y=country_data['Value'],
                        name=ISO2_TO_FULL_NAME.get(country, country),
                        text=[f'{v:.2f}' for v in country_data['Value']],
                        textposition='auto'
                    ))
                else:
                    # For other data sources, use deciles
                    country_data['Decile_num'] = pd.to_numeric(country_data['Decile'], errors='coerce')
                    country_data = country_data.sort_values('Decile_num')
                    
                    # Add decile bars
                    fig.add_trace(go.Bar(
                        x=[str(int(d)) for d in country_data['Decile_num']],
                        y=country_data['Value'],
                        name=ISO2_TO_FULL_NAME.get(country, country),
                        text=[f'{v:.2f}' for v in country_data['Value']],
                        textposition='auto'
                    ))
                
                # Add "All" value as a reference line
                all_value = all_aggregate_data[all_aggregate_data['Country'] == country]
                if not all_value.empty:
                    all_val = all_value['Value'].iloc[0]
                    fig.add_hline(
                        y=all_val,
                        line_dash="dash",
                        annotation_text=f"{ISO2_TO_FULL_NAME.get(country, country)} All: {all_val:.2f}",
                        annotation_position="top right"
                    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold"), x=0.5),
        autosize=True,
        margin=dict(t=60, b=50, l=50, r=50),
        barmode='group',
        xaxis=dict(title=x_axis_title),
        yaxis=dict(title='Score')
    )
    
    return fig

def create_radar_chart(level_filters, selected_countries):
    """Create radar chart or country comparison chart"""
    
    # Get latest year - for Levels 4 and 5, use latest year for the specific indicator
    if level_filters['current_level'] in [4, 5]:
        # For Levels 4 and 5, get latest year available for this specific primary indicator
        indicator_data = unified_df[
            (unified_df['Level'] == level_filters['current_level']) & 
            (unified_df['Primary and raw data'] == level_filters['primary_indicator'])
        ]
        latest_year = indicator_data['Year'].max() if not indicator_data.empty else unified_df['Year'].max()
    else:
        latest_year = unified_df['Year'].max()
    
    if level_filters['current_level'] == 1:
        # Level 1: Radar chart showing EU priorities
        countries_to_show = ['All Countries']
        if selected_countries:
            countries_to_show.extend(selected_countries)
        
        # Get Level 2 data (EU priorities)
        radar_data = unified_df[
            (unified_df['Level'] == 2) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] == 'All') &
            (unified_df['Country'].isin(countries_to_show))
        ].copy()
        
        fig = go.Figure()
        
        if radar_data.empty:
            fig.add_annotation(
                text="No EU Priority data available for radar chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
        else:
            eu_priorities = radar_data['EU priority'].unique()
            
            for country in countries_to_show:
                country_data = radar_data[radar_data['Country'] == country]
                if not country_data.empty:
                    values = []
                    labels = []
                    for priority in eu_priorities:
                        priority_data = country_data[country_data['EU priority'] == priority]
                        if not priority_data.empty:
                            values.append(priority_data['Value'].iloc[0])
                            labels.append(priority)
                    
                    if values:
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=labels,
                            fill='toself',
                            name=ISO2_TO_FULL_NAME.get(country, country)
                        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=dict(text=f'EU Priorities Comparison ({int(latest_year)})', 
                      font=dict(size=16, color="#f4d03f", weight="bold"), x=0.5)
        )
        
    else:
        # Levels 2, 3, 4, 5: Country comparison chart
        if level_filters['current_level'] == 2:
            # Show all countries with this EU priority
            chart_data = unified_df[
                (unified_df['Level'] == 2) & 
                (unified_df['Year'] == latest_year) &
                (unified_df['Decile'] == 'All') &
                (unified_df['EU priority'] == level_filters['eu_priority']) &
                (unified_df['Country'] != 'All Countries')
            ].copy()
            title = f'{level_filters["eu_priority"]} by Country ({int(latest_year)})'
            
        elif level_filters['current_level'] == 3:
            chart_data = unified_df[
                (unified_df['Level'] == 3) & 
                (unified_df['Year'] == latest_year) &
                (unified_df['Decile'] == 'All') &
                (unified_df['Secondary'] == level_filters['secondary_indicator']) &
                (unified_df['Country'] != 'All Countries')
            ].copy()
            title = f'{level_filters["secondary_indicator"]} by Country ({int(latest_year)})'
            
        elif level_filters['current_level'] == 4:
            # Level 4: Primary indicators (aggregated, not raw data)
            chart_data = unified_df[
                (unified_df['Level'] == 4) & 
                (unified_df['Year'] == latest_year) &
                (unified_df['Decile'] == 'All') &
                (unified_df['Primary and raw data'] == level_filters['primary_indicator']) &
                (unified_df['Country'] != 'All Countries')
            ].copy()
            title = f'{level_filters["primary_indicator"]} by Country ({int(latest_year)})'
            
        else:  # Level 5
            # Check if this is EHIS data (uses quintiles) or other data (uses deciles)
            primary_indicator = level_filters['primary_indicator']
            is_ehis_indicator_flag = is_ehis_indicator(primary_indicator)
            
            if is_ehis_indicator_flag:
                # For EHIS data, filter by Quintile == 'All'
                chart_data = unified_df[
                    (unified_df['Level'] == 5) & 
                    (unified_df['Year'] == latest_year) &
                    (unified_df['Quintile'] == 'All') &
                    (unified_df['Primary and raw data'] == primary_indicator) &
                    (unified_df['Country'] != 'All Countries')
                ].copy()
            else:
                # For other data sources, filter by Decile == 'All'
                chart_data = unified_df[
                    (unified_df['Level'] == 5) & 
                    (unified_df['Year'] == latest_year) &
                    (unified_df['Decile'] == 'All') &
                    (unified_df['Primary and raw data'] == primary_indicator) &
                    (unified_df['Country'] != 'All Countries')
                ].copy()
            title = f'{primary_indicator} by Country ({int(latest_year)})'
        
        fig = go.Figure()
        
        if chart_data.empty:
            fig.add_annotation(
                text=f"No country data available for {level_filters['level_name']}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
        else:
            # Sort countries by value
            chart_data = chart_data.sort_values('Value', ascending=False)
            country_names = [ISO2_TO_FULL_NAME.get(c, c) for c in chart_data['Country']]
            
            fig.add_trace(go.Bar(
                x=country_names,
                y=chart_data['Value'],
                text=[f'{v:.2f}' for v in chart_data['Value']],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold"), x=0.5),
            xaxis=dict(tickangle=45),
            yaxis=dict(title='Score')
        )
    
    fig.update_layout(
        autosize=True,
        margin=dict(t=60, b=50, l=50, r=50)
    )
    
    return fig

# Local development server
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'
    
    print("🚀 Starting European Well-Being Index Dashboard (New Version)...")
    print(f"📊 Dashboard will be available at: http://localhost:{port}")
    print(f"🗃️  Using unified data from: {DATA_DIR}/unified_all_levels_1_to_5.csv")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    app.run(debug=debug, host='0.0.0.0', port=port)