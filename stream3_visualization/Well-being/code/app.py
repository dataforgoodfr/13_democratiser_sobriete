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

# Get the absolute path to the data directory - MODIFIED to use PCA unified data
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

# EU Aggregate country name
EU_AGGREGATE = 'EU-27'

# Color palette for consistent coloring across charts
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'

# Helper function to assign colors consistently
def get_country_color(country, all_countries):
    """Assign consistent colors to countries across all charts"""
    if country == 'All Countries':
        return EU_27_COLOR
    
    # Get all individual countries (excluding EU-27)
    individual_countries = [c for c in all_countries if c != 'All Countries']
    individual_countries.sort()  # Sort for consistent ordering
    
    if country in individual_countries:
        index = individual_countries.index(country) % len(COUNTRY_COLORS)
        return COUNTRY_COLORS[index]
    
    return COUNTRY_COLORS[0]  # Default color

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

# Load the unified PCA data from 3_generate_outputs_pca.py
try:
    # Use the population-weighted version of the data
    unified_df = pd.read_csv(os.path.join(DATA_DIR, 'unified_all_levels_1_to_5_pca_weighted.csv'), low_memory=False)
    
    # Load EWBI structure for hierarchical navigation
    with open(os.path.join(DATA_DIR, '..', 'data', 'ewbi_indicators.json'), 'r') as f:
        ewbi_structure = json.load(f)['EWBI']

    print("Unified PCA data file (population-weighted) loaded successfully.")
    
    # NEW: Validate new indicators in dashboard data
    print("\n🔍 Validating new indicators in dashboard data...")
    new_indicators = [
        'HQ-SILC-2', 'HQ-SILC-3', 'HQ-SILC-4', 'HQ-SILC-5', 
        'HQ-SILC-6', 'HQ-SILC-7', 'HQ-SILC-8',
        'IS-SILC-4', 'IS-SILC-5', 'AC-SILC-3', 'AC-SILC-4', 'EC-SILC-3',
        'RT-LFS-4', 'RT-LFS-5', 'RT-LFS-6', 'RT-LFS-7', 'RT-LFS-8', 'EL-SILC-2'
    ]
    
    available_indicators = unified_df['Primary and raw data'].unique()
    found_new_in_dashboard = [ind for ind in new_indicators if ind in available_indicators]
    
    print(f"✅ Dashboard has {len(found_new_in_dashboard)} new indicators available:")
    for ind in found_new_in_dashboard:
        count = len(unified_df[unified_df['Primary and raw data'] == ind])
        print(f"   • {ind}: {count:,} records")
    
    missing_in_dashboard = [ind for ind in new_indicators if ind not in available_indicators]
    if missing_in_dashboard:
        print(f"⚠️  Dashboard missing {len(missing_in_dashboard)} new indicators: {missing_in_dashboard}")

    print(f"Data shape: {unified_df.shape}")
    print(f"Columns: {list(unified_df.columns)}")
    print(f"Levels: {sorted(unified_df['Level'].unique())}")
    print(f"Countries: {unified_df['Country'].nunique()}")
    print(f"Years: {unified_df['Year'].min()}-{unified_df['Year'].max()}")

    # Get unique countries and sort them (excluding aggregates for individual selection)
    # Handle NaN values and ensure all values are strings
    COUNTRIES = sorted([c for c in unified_df['Country'].unique() 
                       if pd.notna(c) and isinstance(c, str) and 
                       c != 'All Countries' and 'Average' not in c and 'Median' not in c])

    print(f"Found {len(COUNTRIES)} individual countries")

    # Create EU priority options - PCA VERSION: These are the only aggregation levels available
    EU_PRIORITIES = [
        'Energy and Housing', 
        'Equality',
        'Health and Animal Welfare',
        'Intergenerational Fairness, Youth, Culture and Sport',
        'Social Rights and Skills, Quality Jobs and Preparedness'
        # Note: 'Sustainable Transport and Tourism' might be commented out in config
    ]

    # PCA VERSION: No secondary indicators - skip this section entirely
    # Secondary indicators are not used in the PCA version as we go directly from Raw -> EU Priorities -> EWBI

    print(f"PCA Dashboard initialized with {len(EU_PRIORITIES)} EU priorities")
    print("📊 PCA Version Hierarchy: Raw Data → EU Priorities → EWBI (no Secondary level)")

except FileNotFoundError as e:
    print(f"Error loading unified PCA data file: {e}")
    print(f"Please ensure that unified_all_levels_1_to_5_pca.csv is in the '{DATA_DIR}' directory.")
    print("Run 3_generate_outputs_pca.py first to generate the PCA data.")
    exit()

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Set global font for all components
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                font-family: Arial, sans-serif;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout - PCA VERSION
app.layout = html.Div([
    # Header section with hierarchical titles
    html.Div([
        html.H1("European Well-Being Index (EWBI) - PCA Version", className="dashboard-title"),
        html.H2([
            "PCA-Weighted Analysis of Well-being Indicators Across Europe"
        ], className="dashboard-subtitle"),
        html.H3([
            "Simplified Hierarchy: Raw Data → EU Priorities → EWBI"
        ], className="dashboard-subtitle", style={'color': '#f4d03f', 'font-size': '16px', 'margin-top': '10px'}),

        # Controls section: Data types on top, Country/Normalize below
        html.Div([
            # Top row: Data Type dropdowns - PCA VERSION (only EU Priority and Raw Data)
            html.Div([
                html.Div([
                    html.Label("EU Priority", className="control-label"),
                    dcc.Dropdown(
                        id='eu-priority-dropdown',
                        options=[{'label': '(EWBI Overall)', 'value': 'ALL'}] + [{'label': prio, 'value': prio} for prio in EU_PRIORITIES],
                        value='ALL',
                        style={'marginTop': '0px', 'width': '300px'},
                        clearable=False
                    ),
                ], className="control-item", style={'marginRight': '60px', 'width': '300px'}),
                html.Div([
                    html.Label("Raw Data Indicator", className="control-label"),
                    dcc.Dropdown(
                        id='primary-indicator-dropdown',
                        options=[{'label': '(Select EU Priority first)', 'value': 'ALL'}],
                        value='ALL',
                        style={'marginTop': '0px', 'width': '300px'},
                        clearable=False,
                        disabled=True
                    ),
                ], className="control-item", style={'width': '300px'}),
                # PCA VERSION: Remove Secondary indicator dropdown entirely
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
                ], style={'width': '102%', 'minWidth': '306px', 'maxWidth': '578px'}),
            ], className="controls-row"),
        ], className="controls-section")
    ], className="dashboard-header"),

    # Responsive 2x2 Visualization Grid
    html.Div([
        # Grid Item 1: European Map
        html.Div([
            dcc.Graph(id='european-map-chart', config={
                'responsive': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'ewbi_european_map',
                    'height': 500,
                    'width': 700,
                    'scale': 1.5
                }
            })
        ], className="grid-item"),
        
        # Grid Item 2: Time Series Chart
        html.Div([
            dcc.Graph(id='time-series-chart', config={
                'responsive': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'ewbi_time_series',
                    'height': 400,
                    'width': 600,
                    'scale': 1.5
                }
            })
        ], className="grid-item"),
        
        # Grid Item 3: Decile Analysis Chart
        html.Div([
            dcc.Graph(id='decile-analysis-chart', config={
                'responsive': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'ewbi_decile_analysis',
                    'height': 400,
                    'width': 600,
                    'scale': 1.5
                }
            })
        ], className="grid-item"),
        
        # Grid Item 4: Radar Chart / Country Comparison
        html.Div([
            dcc.Graph(id='radar-chart', config={
                'responsive': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'ewbi_radar_comparison',
                    'height': 400,
                    'width': 600,
                    'scale': 1.5
                }
            })
        ], className="grid-item"),
    ], className="visualization-grid"),

    # Sources section - updated for PCA version
    html.Div([
        html.H3("Data Sources - PCA Version", className="sources-title"),
        html.Div([
            html.H4("European Well-Being Index (EWBI) - PCA Enhanced", className="sources-subtitle"),
            html.P("The PCA-enhanced EWBI uses Principal Component Analysis for improved indicator weighting and a simplified hierarchical structure.", className="sources-text"),
            html.H4("PCA Version Improvements", className="sources-subtitle"),
            html.P("• Winsorization (1st-99th percentile) + Percentile scaling instead of z-score normalization", className="sources-text"),
            html.P("• PCA-based weights for EU priority aggregation ensure optimal indicator contribution", className="sources-text"),
            html.P("• Simplified hierarchy: Raw Data → EU Priorities → EWBI (no Secondary level)", className="sources-text"),
            html.P("• All values scaled to [0.1, 1] for geometric mean compatibility", className="sources-text"),
            html.P("• EU-27 aggregates use population-weighted averages (PWA) for representative statistics", className="sources-text"),
            html.H4("Indicator Structure", className="sources-subtitle"),
            html.P("The PCA EWBI follows a simplified hierarchical structure:", className="sources-text"),
            html.P([html.Strong("Level 1: "), "EWBI - Overall well-being score (geometric mean of EU priorities)"], className="sources-text"),
            html.P([html.Strong("Level 2: "), "EU Priorities - Major policy areas (PCA-weighted aggregation of raw indicators)"], className="sources-text"),
            html.P([html.Strong("Level 5: "), "Raw Statistics - Individual survey questions and measurements"], className="sources-text"),
            html.P([html.Strong("Note: "), "Level 3 (Secondary Indicators) is skipped in the PCA version for simplified analysis"], className="sources-text"),
        ], className="sources-content")
    ], className="sources-section")
], className="dashboard-container")

# PCA VERSION: Simplified callback to update primary indicator dropdown based on EU priority
@app.callback(
    Output('primary-indicator-dropdown', 'options'),
    Output('primary-indicator-dropdown', 'value'),
    Output('primary-indicator-dropdown', 'disabled'),
    Input('eu-priority-dropdown', 'value')
)
def update_primary_indicator_dropdown(eu_priority):
    if eu_priority == 'ALL':
        return [{'label': '(Select EU Priority first)', 'value': 'ALL'}], 'ALL', True
    elif eu_priority:
        # Get primary indicators from the unified PCA data for this EU priority
        primary_options = unified_df[
            (unified_df['EU priority'] == eu_priority) & 
            (unified_df['Primary and raw data'].notna()) &
            (unified_df['Level'] == 5)  # Raw data level
        ]['Primary and raw data'].unique()
        
        if len(primary_options) > 0:
            options = [{'label': '(EU Priority Overall)', 'value': 'ALL'}] + [
                {'label': get_display_name(primary), 'value': primary} for primary in sorted(primary_options)
            ]
            return options, 'ALL', False
        else:
            return [{'label': '(No data available)', 'value': 'ALL'}], 'ALL', True
    else:
        return [{'label': '(Select EU Priority first)', 'value': 'ALL'}], 'ALL', True

# Main callback to update all charts - PCA VERSION
@app.callback(
    [Output('european-map-chart', 'figure'),
     Output('time-series-chart', 'figure'),
     Output('decile-analysis-chart', 'figure'),
     Output('radar-chart', 'figure')],
    [Input('eu-priority-dropdown', 'value'),
     Input('primary-indicator-dropdown', 'value'),
     Input('countries-filter', 'value')]
)
def update_charts(eu_priority, primary_indicator, selected_countries):
    # Determine current level based on filters - PCA VERSION (no Secondary level)
    level_filters = create_level_filters_pca(eu_priority, primary_indicator)
    
    # Create all charts using the unified PCA data
    map_chart = create_map_chart_pca(level_filters)
    time_series_chart = create_time_series_chart_pca(level_filters, selected_countries)
    decile_chart = create_decile_chart_pca(level_filters, selected_countries)
    radar_chart = create_radar_chart_pca(level_filters, selected_countries)
    
    return map_chart, time_series_chart, decile_chart, radar_chart

def create_level_filters_pca(eu_priority, primary_indicator):
    """Create level-based filters for the PCA dashboard"""
    level_filters = {
        'eu_priority': eu_priority,
        'primary_indicator': primary_indicator
    }

    # Determine the current level based on filter combinations - PCA VERSION
    if eu_priority == 'ALL' or not eu_priority:
        # Level 1: EWBI (overall)
        level_filters['current_level'] = 1
        level_filters['level_name'] = 'EWBI (Overall Well-being)'
    elif eu_priority != 'ALL' and (primary_indicator == 'ALL' or not primary_indicator):
        # Level 2: EU Priority selected
        level_filters['current_level'] = 2
        level_filters['level_name'] = f'EU Priority: {eu_priority}'
    elif primary_indicator and primary_indicator != 'ALL':
        # Level 5: Primary/Raw data selected (skip levels 3 and 4 in PCA version)
        level_filters['current_level'] = 5
        level_filters['level_name'] = f'Raw Data: {get_display_name(primary_indicator)}'
    else:
        level_filters['current_level'] = 1
        level_filters['level_name'] = 'EWBI (Overall Well-being)'

    return level_filters

def create_map_chart_pca(level_filters):
    """Create map chart showing latest year values per country for the selected level - PCA VERSION"""
    
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
    
    # Filter data based on level and get "All" deciles only
    if level_filters['current_level'] == 1:
        # Level 1: EWBI overall
        map_data = unified_df[
            (unified_df['Level'] == 1) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] == 'All') &
            (unified_df['Country'] != 'All Countries')
        ].copy()
        title = f'EWBI Overall Well-being Score by Country ({int(latest_year)}) - PCA Version'
        
    elif level_filters['current_level'] == 2:
        # Level 2: Specific EU priority
        map_data = unified_df[
            (unified_df['Level'] == 2) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] == 'All') &
            (unified_df['EU priority'] == level_filters['eu_priority']) &
            (unified_df['Country'] != 'All Countries')
        ].copy()
        title = f'{level_filters["eu_priority"]} Score by Country ({int(latest_year)}) - PCA Weighted'
        
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
        title = f'{get_display_name(primary_indicator)} - Population Share by Country ({int(latest_year)})'
    
    if map_data.empty:
        # No data available
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available for {level_filters['level_name']}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, family="Arial, sans-serif")
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold", family="Arial, sans-serif"), x=0.5),
            font=dict(family="Arial, sans-serif")
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
            font=dict(size=16, family="Arial, sans-serif")
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold", family="Arial, sans-serif"), x=0.5),
            font=dict(family="Arial, sans-serif")
        )
        return fig
    
    # Create choropleth map
    # For level 5 (raw indicators), invert color scale so green = minimum (better) and red = maximum (worse)
    if level_filters['current_level'] == 5:
        colorscale = 'RdYlGn_r'  # Reversed color scale for raw indicators
        # For level 5, display as percentage and use "Population share"
        hover_template = '<b>%{text}</b><br>Population share: %{z:.1f}%<extra></extra>'
        colorbar_title = "Population share (%)"
        # Data is already in percentage format, no conversion needed
    else:
        colorscale = 'RdYlGn'  # Normal color scale for aggregated levels
        hover_template = '<b>%{text}</b><br>Score: %{z:.2f}<extra></extra>'
        colorbar_title = "Score"
    
    fig = go.Figure(data=go.Choropleth(
        locations=map_data['iso3'],
        z=map_data['Value'],
        locationmode='ISO-3',
        colorscale=colorscale,
        text=map_data['full_name'],
        hovertemplate=hover_template,
        colorbar=dict(title=colorbar_title)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold", family="Arial, sans-serif"), x=0.5),
        geo=dict(
            scope='europe',
            showland=True,
            landcolor='lightgray',
            coastlinecolor='white',
            showocean=True,
            oceancolor='white'
        ),
        autosize=True,
        margin=dict(t=60, b=10, l=10, r=10),
        font=dict(family="Arial, sans-serif")
    )
    
    return fig

def create_time_series_chart_pca(level_filters, selected_countries):
    """Create time series chart showing evolution over time - PCA VERSION"""
    
    # Get countries to show: All Countries + selected individual countries
    countries_to_show = ['All Countries']
    if selected_countries:
        countries_to_show.extend(selected_countries)
    
    # Filter data based on level - same logic as map chart but across all years
    if level_filters['current_level'] == 1:
        # Level 1: EWBI overall
        # Use Population-weighted geometric mean for 'All Countries' with Decile='All' 
        ts_data = unified_df[
            (unified_df['Level'] == 1) & 
            (unified_df['Decile'] == 'All') &
            (unified_df['Country'].isin(countries_to_show)) &
            (
                (unified_df['Country'] != 'All Countries') |
                ((unified_df['Country'] == 'All Countries') & 
                 (unified_df['Aggregation'] == 'Population-weighted geometric mean'))
            )
        ].copy()
        title = 'EWBI Overall Well-being Evolution Over Time - PCA Version (Population-Weighted)'
        
    elif level_filters['current_level'] == 2:
        # Level 2: EU priority
        # Use Population-weighted geometric mean for 'All Countries' with Decile='All'
        ts_data = unified_df[
            (unified_df['Level'] == 2) & 
            (unified_df['Decile'] == 'All') &
            (unified_df['EU priority'] == level_filters['eu_priority']) &
            (unified_df['Country'].isin(countries_to_show)) &
            (
                (unified_df['Country'] != 'All Countries') |
                ((unified_df['Country'] == 'All Countries') & 
                 (unified_df['Aggregation'] == 'Population-weighted geometric mean'))
            )
        ].copy()
        title = f'{level_filters["eu_priority"]} Evolution Over Time - Population-Weighted'
        
    else:  # Level 5
        # Level 5: Raw data
        # Check if this is EHIS data (uses quintiles) or other data (uses deciles)
        primary_indicator = level_filters['primary_indicator']
        is_ehis_indicator_flag = is_ehis_indicator(primary_indicator)
        
        if is_ehis_indicator_flag:
            # For EHIS data, filter by Quintile == 'All'
            ts_data = unified_df[
                (unified_df['Level'] == 5) & 
                (unified_df['Quintile'] == 'All') &
                (unified_df['Primary and raw data'] == primary_indicator) &
                (unified_df['Country'].isin(countries_to_show)) &
                (
                    (unified_df['Country'] != 'All Countries') |
                    ((unified_df['Country'] == 'All Countries') & 
                     (unified_df['Aggregation'].isin(['Median across countries', 'Population-weighted average'])))
                )
            ].copy()
        else:
            # For other data sources, filter by Decile == 'All'
            ts_data = unified_df[
                (unified_df['Level'] == 5) & 
                (unified_df['Decile'] == 'All') &
                (unified_df['Primary and raw data'] == primary_indicator) &
                (unified_df['Country'].isin(countries_to_show)) &
                (
                    (unified_df['Country'] != 'All Countries') |
                    ((unified_df['Country'] == 'All Countries') & 
                     (unified_df['Aggregation'].isin(['Median across countries', 'Population-weighted average'])))
                )
            ].copy()
        title = f'{get_display_name(primary_indicator)} - Population Share Over Time'
    
    fig = go.Figure()
    
    if ts_data.empty:
        fig.add_annotation(
            text=f"No time series data available for {level_filters['level_name']}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, family="Arial, sans-serif")
        )
    else:
        # Add traces for each country
        for country in countries_to_show:
            country_data = ts_data[ts_data['Country'] == country].sort_values('Year')
            if not country_data.empty:
                # Replace "All Countries" with "EU-27 (PWA)" in display name
                if country == 'All Countries':
                    display_name = 'EU-27 (PWA)'
                else:
                    display_name = ISO2_TO_FULL_NAME.get(country, country)
                
                # Adjust hover template and y-values for level 5 data
                if level_filters['current_level'] == 5:
                    y_values = country_data['Value']  # Data is already in percentage format
                    hover_template = '<b>Population share:</b> %{y:.1f}%<br><b>Year:</b> %{x}<extra></extra>'
                else:
                    y_values = country_data['Value']
                    hover_template = '<b>Score:</b> %{y:.2f}<br><b>Year:</b> %{x}<extra></extra>'
                
                fig.add_trace(go.Scatter(
                    x=country_data['Year'],
                    y=y_values,
                    mode='lines+markers',
                    name=display_name,
                    line=dict(color=get_country_color(country, countries_to_show)),
                    marker=dict(color=get_country_color(country, countries_to_show)),
                    hovertemplate=hover_template
                ))
    
    # Set y-axis title based on level
    if level_filters['current_level'] == 5:
        y_axis_title = 'Population share (%)'
    else:
        y_axis_title = 'Score'
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold", family="Arial, sans-serif"), x=0.5),
        autosize=True,
        margin=dict(t=60, b=50, l=50, r=50),
        yaxis=dict(title=y_axis_title),
        xaxis=dict(title='Year'),
        font=dict(family="Arial, sans-serif")
    )
    
    return fig

def create_decile_chart_pca(level_filters, selected_countries):
    """Create decile/quintile chart for the latest year - PCA VERSION"""
    
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
    if level_filters['current_level'] == 1:
        decile_data = unified_df[
            (unified_df['Level'] == 1) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] != 'All') &
            (unified_df['Country'].isin(countries_to_show)) &
            (
                (unified_df['Country'] != 'All Countries') |
                ((unified_df['Country'] == 'All Countries') & 
                 (unified_df['Aggregation'] == 'Population-weighted geometric mean'))
            )
        ].copy()
        title = f'EWBI Scores by Decile ({int(latest_year)}) - Population-Weighted'
        x_axis_title = 'Income Decile'
        
    elif level_filters['current_level'] == 2:
        decile_data = unified_df[
            (unified_df['Level'] == 2) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] != 'All') &
            (unified_df['EU priority'] == level_filters['eu_priority']) &
            (unified_df['Country'].isin(countries_to_show)) &
            (
                (unified_df['Country'] != 'All Countries') |
                ((unified_df['Country'] == 'All Countries') & 
                 (unified_df['Aggregation'] == 'Population-weighted geometric mean'))
            )
        ].copy()
        title = f'{level_filters["eu_priority"]} Scores by Decile ({int(latest_year)}) - Population-Weighted'
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
            title = f'{get_display_name(level_filters["primary_indicator"])} - Population Share by Quintile ({int(latest_year)})'
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
            title = f'{get_display_name(level_filters["primary_indicator"])} - Population Share by Decile ({int(latest_year)})'
            x_axis_title = 'Income Decile'
    
    fig = go.Figure()
    
    if decile_data.empty:
        data_type = "quintile" if is_ehis_data else "decile"
        fig.add_annotation(
            text=f"No {data_type} data available for {level_filters['level_name']}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, family="Arial, sans-serif")
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
        elif level_filters['current_level'] == 5:
            all_aggregate_data = all_aggregate_data[all_aggregate_data['Primary and raw data'] == level_filters['primary_indicator']]
        
        # Create traces for each country
        for country in countries_to_show:
            country_data = decile_data[decile_data['Country'] == country].copy()
            
            if not country_data.empty:
                # Replace "All Countries" with "EU-27 (PWA)" in display name
                if country == 'All Countries':
                    display_name = 'EU-27 (PWA)'
                else:
                    display_name = ISO2_TO_FULL_NAME.get(country, country)
                
                if is_ehis_data:
                    # For EHIS data, use quintiles
                    country_data['Quintile_num'] = pd.to_numeric(country_data['Quintile'], errors='coerce')
                    country_data = country_data.sort_values('Quintile_num')
                    
                    # Adjust values and text for level 5 data
                    if level_filters['current_level'] == 5:
                        y_values = country_data['Value']  # Data is already in percentage format
                        text_values = [f'{v:.1f}%' for v in y_values]
                    else:
                        y_values = country_data['Value']
                        text_values = [f'{v:.2f}' for v in y_values]
                    
                    # Add quintile bars
                    fig.add_trace(go.Bar(
                        x=[str(int(q)) for q in country_data['Quintile_num']],
                        y=y_values,
                        name=display_name,
                        text=text_values,
                        textposition='auto',
                        marker=dict(color=get_country_color(country, countries_to_show))
                    ))
                else:
                    # For other data sources, use deciles
                    country_data['Decile_num'] = pd.to_numeric(country_data['Decile'], errors='coerce')
                    country_data = country_data.sort_values('Decile_num')
                    
                    # Adjust values and text for level 5 data
                    if level_filters['current_level'] == 5:
                        y_values = country_data['Value']  # Data is already in percentage format
                        text_values = [f'{v:.1f}%' for v in y_values]
                    else:
                        y_values = country_data['Value']
                        text_values = [f'{v:.2f}' for v in y_values]
                    
                    # Add decile bars
                    fig.add_trace(go.Bar(
                        x=[str(int(d)) for d in country_data['Decile_num']],
                        y=y_values,
                        name=display_name,
                        text=text_values,
                        textposition='auto',
                        marker=dict(color=get_country_color(country, countries_to_show))
                    ))
                
                # Add "All" value as a reference line
                all_value = all_aggregate_data[all_aggregate_data['Country'] == country]
                if not all_value.empty:
                    all_val = all_value['Value'].iloc[0]
                    
                    # Replace "All Countries" with "EU-27" in annotation and adjust value for level 5
                    if country == 'All Countries':
                        annotation_country = 'EU-27'
                    else:
                        annotation_country = ISO2_TO_FULL_NAME.get(country, country)
                    
                    if level_filters['current_level'] == 5:
                        display_val = all_val  # Data is already in percentage format
                        annotation_text = f"{annotation_country} All: {display_val:.1f}%"
                    else:
                        display_val = all_val
                        annotation_text = f"{annotation_country} All: {display_val:.2f}"
                    
                    fig.add_hline(
                        y=display_val,
                        line_dash="dash",
                        annotation_text=annotation_text,
                        annotation_position="top right"
                    )
    
    # Set y-axis title based on level
    if level_filters['current_level'] == 5:
        y_axis_title = 'Population share (%)'
    else:
        y_axis_title = 'Score'
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold", family="Arial, sans-serif"), x=0.5),
        autosize=True,
        margin=dict(t=60, b=50, l=50, r=50),
        barmode='group',
        xaxis=dict(title=x_axis_title),
        yaxis=dict(title=y_axis_title),
        font=dict(family="Arial, sans-serif")
    )
    
    return fig

def create_radar_chart_pca(level_filters, selected_countries):
    """Create radar chart or country comparison chart - PCA VERSION"""
    
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
    
    if level_filters['current_level'] == 1:
        # Level 1: Radar chart showing EU priorities
        countries_to_show = ['All Countries']
        if selected_countries:
            countries_to_show.extend(selected_countries)
        
        # Get Level 2 data (EU priorities) - use Population-weighted geometric mean for 'All Countries'
        radar_data = unified_df[
            (unified_df['Level'] == 2) & 
            (unified_df['Year'] == latest_year) &
            (unified_df['Decile'] == 'All') &
            (unified_df['Country'].isin(countries_to_show)) &
            (
                (unified_df['Country'] != 'All Countries') |
                ((unified_df['Country'] == 'All Countries') & 
                 (unified_df['Aggregation'] == 'Population-weighted geometric mean'))
            )
        ].copy()
        
        fig = go.Figure()
        
        if radar_data.empty:
            fig.add_annotation(
                text="No EU Priority data available for radar chart",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, family="Arial, sans-serif")
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
                        # Replace "All Countries" with "EU-27 (PWA)" in display name
                        if country == 'All Countries':
                            display_name = 'EU-27 (PWA)'
                        else:
                            display_name = ISO2_TO_FULL_NAME.get(country, country)
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=labels,
                            fill='toself',
                            name=display_name,
                            line=dict(color=get_country_color(country, countries_to_show)),
                            fillcolor=get_country_color(country, countries_to_show)
                        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=dict(text=f'EU Priorities Comparison ({int(latest_year)}) - PCA Weighted', 
                      font=dict(size=16, color="#f4d03f", weight="bold", family="Arial, sans-serif"), x=0.5),
            font=dict(family="Arial, sans-serif")
        )
        
    else:
        # Levels 2 and 5: Country comparison chart
        if level_filters['current_level'] == 2:
            # Show all countries with this EU priority
            chart_data = unified_df[
                (unified_df['Level'] == 2) & 
                (unified_df['Year'] == latest_year) &
                (unified_df['Decile'] == 'All') &
                (unified_df['EU priority'] == level_filters['eu_priority']) &
                (unified_df['Country'] != 'All Countries')
            ].copy()
            title = f'{level_filters["eu_priority"]} by Country ({int(latest_year)}) - PCA Weighted'
            
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
            title = f'{get_display_name(primary_indicator)} - Population Share by Country ({int(latest_year)})'
        
        fig = go.Figure()
        
        if chart_data.empty:
            fig.add_annotation(
                text=f"No country data available for {level_filters['level_name']}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, family="Arial, sans-serif")
            )
        else:
            # Get EU-27 aggregate data for the same level and filters
            if level_filters['current_level'] == 2:
                # For EU Priority level
                eu_data = unified_df[
                    (unified_df['Level'] == 2) & 
                    (unified_df['Year'] == latest_year) &
                    (unified_df['Decile'] == 'All') &
                    (unified_df['EU priority'] == level_filters['eu_priority']) &
                    (unified_df['Country'] == 'All Countries') &
                    (unified_df['Aggregation'] == 'Population-weighted geometric mean')
                ].copy()
            else:  # Level 5
                primary_indicator = level_filters['primary_indicator']
                is_ehis_indicator_flag = is_ehis_indicator(primary_indicator)
                
                if is_ehis_indicator_flag:
                    # For EHIS data, filter by Quintile == 'All'
                    eu_data = unified_df[
                        (unified_df['Level'] == 5) & 
                        (unified_df['Year'] == latest_year) &
                        (unified_df['Quintile'] == 'All') &
                        (unified_df['Primary and raw data'] == primary_indicator) &
                        (unified_df['Country'] == 'All Countries') &
                        (unified_df['Aggregation'].isin(['Median across countries', 'Population-weighted average']))
                    ].copy()
                else:
                    # For other data sources, filter by Decile == 'All'
                    eu_data = unified_df[
                        (unified_df['Level'] == 5) & 
                        (unified_df['Year'] == latest_year) &
                        (unified_df['Decile'] == 'All') &
                        (unified_df['Primary and raw data'] == primary_indicator) &
                        (unified_df['Country'] == 'All Countries') &
                        (unified_df['Aggregation'].isin(['Median across countries', 'Population-weighted average']))
                    ].copy()
            
            # Combine EU-27 data with country data
            all_data = chart_data.copy()
            if not eu_data.empty:
                # Add EU-27 data
                eu_row = eu_data.iloc[0].copy()
                eu_row['Country'] = 'EU-27'
                all_data = pd.concat([all_data, pd.DataFrame([eu_row])], ignore_index=True)
            
            # Sort all data by value
            all_data = all_data.sort_values('Value', ascending=False)
            
            # Create display names and prepare data
            country_names = []
            y_values = []
            text_values = []
            colors = []
            
            for _, row in all_data.iterrows():
                country = row['Country']
                value = row['Value']
                
                # Set display name
                if country == 'EU-27':
                    display_name = 'EU-27'
                else:
                    display_name = ISO2_TO_FULL_NAME.get(country, country)
                
                country_names.append(display_name)
                y_values.append(value)
                
                # Format text based on level
                if level_filters['current_level'] == 5:
                    text_values.append(f'{value:.1f}%')
                else:
                    text_values.append(f'{value:.2f}')
                
                # Set colors: EU-27 in blue, selected countries in their colors, others in light grey
                if country == 'EU-27':
                    colors.append(EU_27_COLOR)  # EU-27 blue
                elif selected_countries and country in selected_countries:
                    colors.append(get_country_color(country, selected_countries))  # Selected country color
                else:
                    colors.append('#d3d3d3')  # Light grey for unselected countries
            
            fig.add_trace(go.Bar(
                x=country_names,
                y=y_values,
                text=text_values,
                textposition='auto',
                marker=dict(color=colors),
                showlegend=False
            ))
        
        # Set y-axis title based on level
        if level_filters['current_level'] == 5:
            y_axis_title = 'Population share (%)'
        else:
            y_axis_title = 'Score'
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold", family="Arial, sans-serif"), x=0.5),
            xaxis=dict(tickangle=45),
            yaxis=dict(title=y_axis_title),
            font=dict(family="Arial, sans-serif")
        )
    
    fig.update_layout(
        autosize=True,
        margin=dict(t=60, b=50, l=50, r=50),
        font=dict(family="Arial, sans-serif")
    )
    
    return fig

# Local development server
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8051))  # Use different port for PCA version
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'
    
    print("🚀 Starting European Well-Being Index Dashboard - PCA Version...")
    print(f"📊 Dashboard will be available at: http://localhost:{port}")
    print(f"🗃️  Using PCA unified data from: {DATA_DIR}/unified_all_levels_1_to_5_pca.csv")
    print("🔬 PCA Features:")
    print("   • Winsorization + Percentile scaling normalization")
    print("   • PCA-based weights for EU priority aggregation")
    print("   • Simplified hierarchy: Raw → EU Priorities → EWBI")
    print("   • No Secondary indicator level")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    app.run(debug=debug, host='0.0.0.0', port=port)