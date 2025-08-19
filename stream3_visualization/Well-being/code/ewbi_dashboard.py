import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import os
import json
import numpy as np

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

# Load the data
try:
    master_df = pd.read_csv(os.path.join(DATA_DIR, 'ewbi_master.csv'))
    time_series_df = pd.read_csv(os.path.join(DATA_DIR, 'ewbi_time_series.csv'))
    
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
    
    # Define secondary indicators that should be filtered out (same as in generate_outputs.py)
    secondary_indicators_to_remove = [
        'Housing expense',           # No underlying primary indicators
        'Digital Skills',            # No underlying primary indicators
        'Health cost and medical care',  # No underlying primary indicators
        'Accidents and addictive behaviour',  # No underlying primary indicators
        'Education expense',         # No underlying primary indicators
        'Leisure and culture',       # No underlying primary indicators
        'Transport',                 # No underlying primary indicators
        'Tourism'                    # No underlying primary indicators
    ]
    
    for prio in ewbi_structure:
        for component in prio['components']:
            # Filter out secondary indicators that have no underlying primary indicators
            if component['name'] not in secondary_indicators_to_remove:
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
                html.Label("EU Priority", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='eu-priority-dropdown',
                    options=[{'label': 'ALL', 'value': 'ALL'}] + [{'label': prio, 'value': prio} for prio in EU_PRIORITIES],
                    value='ALL',
                    style={'marginTop': '8px'},
                    clearable=False
                )
            ], style={'width': '25%', 'display': 'inline-block', 'margin-right': '2%'}),
            
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
                        {'label': 'EU Average', 'value': 'EU Average'}
                    ] + [{'label': ISO2_TO_FULL_NAME.get(country, country), 'value': country} for country in COUNTRIES],
                    value=['EU Average'],  # Default to EU Average
                    multi=True,
                    placeholder='Select countries to display (default: EU Average)',
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
        # First row: Map and time series side by side
        html.Div([
            dcc.Graph(id='european-map-chart', style={'display': 'inline-block', 'width': '49%', 'verticalAlign': 'top'}),
            dcc.Graph(id='time-series-chart', style={'display': 'inline-block', 'width': '49%', 'verticalAlign': 'top'})
        ], style={'textAlign': 'center', 'margin': '0 auto'}),
        
        # Second row: Decile analysis and radar chart side by side
        html.Div([
            dcc.Graph(id='decile-analysis-chart', style={'display': 'inline-block', 'width': '49%'}),
            dcc.Graph(id='radar-chart', style={'display': 'inline-block', 'width': '49%'})
        ], style={'marginTop': '20px'})
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

# Callback to update EU priority dropdown (no longer needed, but keeping for future extensibility)
# The EU priority dropdown is now always enabled and controlled directly by the user

# Callback to update secondary indicator dropdown based on EU priority
@app.callback(
    Output('secondary-indicator-dropdown', 'options'),
    Output('secondary-indicator-dropdown', 'value'),
    Output('secondary-indicator-dropdown', 'disabled'),
    Input('eu-priority-dropdown', 'value')
)
def update_secondary_indicator_dropdown(eu_priority):
    if eu_priority == 'ALL':
        # In overview mode (ALL EU priorities), show ALL secondary indicators
        return [{'label': 'ALL', 'value': 'ALL'}], 'ALL', True
    elif eu_priority:
        # When specific EU priority selected, show secondary indicators for that priority
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
    Input('eu-priority-dropdown', 'value')
)
def update_primary_indicator_dropdown(secondary_indicator, eu_priority):
    if eu_priority == 'ALL':
        # In overview mode (ALL EU priorities), show ALL primary indicators
        return [{'label': 'ALL', 'value': 'ALL'}], 'ALL', True
    elif eu_priority and secondary_indicator and secondary_indicator != 'ALL':
        # In by_eu_priority mode with specific secondary indicator, show primary indicators
        # Load the EWBI structure to get the actual primary indicators
        try:
            with open(os.path.join(DATA_DIR, '..', 'data', 'ewbi_indicators.json'), 'r') as f:
                config = json.load(f)['EWBI']
            
            # Define the indicators that should be filtered out (same as in generate_outputs.py)
            economic_indicators_to_remove = [
                'AN-SILC-1',
                'AE-HBS-1', 'AE-HBS-2',
                'HQ-SILC-2',
                'HH-SILC-1', 'HH-HBS-1', 'HH-HBS-2', 'HH-HBS-3', 'HH-HBS-4',
                'EC-HBS-1', 'EC-HBS-2',
                'ED-ICT-1', 'ED-EHIS-1',
                'AC-SILC-1', 'AC-SILC-2', 'AC-HBS-1', 'AC-HBS-2', 'AC-EHIS-1',
                'IE-HBS-1', 'IE-HBS-2',
                'IC-SILC-1', 'IC-SILC-2', 'IC-HBS-1', 'IC-HBS-2',
                'TT-SILC-1', 'TT-SILC-2', 'TT-HBS-1', 'TT-HBS-2',
                'TS-SILC-1', 'TS-HBS-1', 'TS-HBS-2'
            ]
            
            primary_indicators = []
            for priority in config:
                if priority['name'] == eu_priority:
                    for component in priority['components']:
                        if component['name'] == secondary_indicator:
                            for indicator in component['indicators']:
                                # Filter out economic indicators that should be removed
                                if indicator['code'] not in economic_indicators_to_remove:
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
     Output('time-series-chart', 'figure'),
     Output('decile-analysis-chart', 'figure'),
     Output('radar-chart', 'figure')],
    [Input('eu-priority-dropdown', 'value'),
     Input('secondary-indicator-dropdown', 'value'),
     Input('primary-indicator-dropdown', 'value'),
     Input('countries-filter', 'value')]
)
def update_charts(eu_priority, secondary_indicator, primary_indicator, selected_countries):
    # For the map (Graph 1), we always want to show all individual countries (exclude EU Average)
    map_df = master_df[(~master_df['country'].str.contains('Average')) & (master_df['decile'] == 'All')].copy()
    
    # For analysis charts (Graphs 2, 3, 4), filter based on selection
    if not selected_countries or selected_countries == ['EU Average']:
        # Default to EU Average for analysis charts
        filtered_df = master_df[master_df['country'] == 'EU Average'].copy()
        time_filtered_df = time_series_df[time_series_df['country'] == 'EU Average'].copy()
    else:
        # Filter by selected countries, only "All" deciles for country-level analysis
        filtered_df = master_df[(master_df['country'].isin(selected_countries)) & (master_df['decile'] == 'All')].copy()
        time_filtered_df = time_series_df[(time_series_df['country'].isin(selected_countries)) & (time_series_df['decile'] == 'All Deciles')].copy()
    
    # Create level filters for adaptive charts
    level_filters = create_level_filters(eu_priority, secondary_indicator, primary_indicator)
    
    # Create adaptive map chart (works for all 4 levels)
    map_chart = create_adaptive_map_chart(map_df, level_filters)
    
    # Create adaptive decile chart (works for all 4 levels)
    # For decile chart, we want ALL deciles for both EU Average and individual countries
    if not selected_countries or selected_countries == ['EU Average']:
        # Default to EU Average with all deciles
        decile_df = master_df[master_df['country'] == 'EU Average'].copy()
    else:
        # Include EU Average and selected countries with ALL deciles
        decile_df = master_df[
            (master_df['country'].isin(['EU Average'] + selected_countries))
        ].copy()
    
    decile_chart = create_adaptive_decile_chart(decile_df, level_filters)
    
    # Create adaptive radar/country comparison chart (works for all 4 levels)
    # Level 1: Use filtered_df (respects country filter for radar chart)
    # Levels 2-4: Use map_df (shows all countries for country comparison)
    if level_filters['current_level'] == 1:
        radar_country_df = filtered_df  # Use country filter for Level 1 radar
    else:
        radar_country_df = map_df  # Use all countries for Levels 2-4 country comparison
    
    radar_country_chart = create_adaptive_radar_country_chart(radar_country_df, level_filters)
    
    # Determine what to show for time series chart based on filter selections
    if eu_priority == 'ALL':
        # Level 1: Overview - Show EWBI and EU priorities
        time_series_chart, _ = create_overview_charts(map_df, filtered_df, time_filtered_df)
        return map_chart, time_series_chart, decile_chart, radar_country_chart
    else:
        # Drill down based on secondary and primary indicator selections
        if secondary_indicator and secondary_indicator != 'ALL' and primary_indicator and primary_indicator != 'ALL':
            # Level 4: Specific Primary Indicator selected
            time_series_chart, _ = create_primary_indicator_charts(map_df, filtered_df, time_filtered_df, eu_priority, secondary_indicator, primary_indicator)
            return map_chart, time_series_chart, decile_chart, radar_country_chart
        elif secondary_indicator and secondary_indicator != 'ALL':
            # Level 3: Specific Secondary Indicator selected (Primary = ALL)
            time_series_chart, _ = create_secondary_indicator_charts(map_df, filtered_df, time_filtered_df, eu_priority, secondary_indicator)
            return map_chart, time_series_chart, decile_chart, radar_country_chart
        else:
            # Level 2: Only EU Priority selected (Secondary = ALL, Primary = ALL)
            time_series_chart, _ = create_eu_priority_charts(map_df, filtered_df, time_filtered_df, eu_priority)
            return map_chart, time_series_chart, decile_chart, radar_country_chart

def create_overview_charts(map_df, analysis_df, time_df):
    """Create charts for overview level (EWBI + EU priorities) - Map is now handled separately"""
    
    # 2. Decile analysis chart (EWBI scores by decile for selected countries)
    decile_analysis = go.Figure()
    
    # Get EWBI data for selected countries (Level 1: EWBI)
    ewbi_data = analysis_df[
        (analysis_df['EU_Priority'] == 'All') & 
        (analysis_df['Secondary_indicator'] == 'All') & 
        (analysis_df['primary_index'] == 'All')
    ].copy()
    
    # Show countries in the determined order
    for country in ewbi_data['country'].unique():
        country_data = ewbi_data[ewbi_data['country'] == country].sort_values('decile')
        decile_analysis.add_trace(go.Bar(
            x=[str(d) for d in country_data['decile']],
            y=country_data['Score'],
            name=country,
            text=country_data['Score'].round(2),
            textposition='auto'
        ))
    
        decile_analysis.update_layout(
            title=dict(
                text='EWBI Scores by Decile - Selected Countries',
            font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                x=0.5
            ),
        height=500,  # Match Budget dashboard chart height
        margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
            barmode='group',
        font=dict(family='Arial, sans-serif', size=14),  # Match Budget dashboard font size
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                gridcolor='lightgrey',
                gridwidth=0.5,
                showgrid=True
            ),
            yaxis=dict(
                range=[0, 1],
                gridcolor='lightgrey',
                gridwidth=0.5,
                showgrid=True
            )
        )
    
    # 3. Radar chart (EU priorities for selected countries)
    radar_chart = go.Figure()
    
    # Get EU priority data for selected countries (Level 2: EU Priority)
    eu_priority_data = analysis_df[
        (analysis_df['EU_Priority'] != 'All') & 
        (analysis_df['Secondary_indicator'] == 'All') & 
        (analysis_df['primary_index'] == 'All')
    ].copy()
    
    # Get unique EU priorities
    eu_priorities = eu_priority_data['EU_Priority'].unique()
    
    # Show countries in the determined order
    for country in eu_priority_data['country'].unique():
        country_data = eu_priority_data[eu_priority_data['country'] == country]
        
        # Create a dictionary to map EU priority to score
        priority_scores = {}
        for _, row in country_data.iterrows():
            priority_scores[row['EU_Priority']] = row['Score']
        
        # Get values and labels for radar chart
        values = [priority_scores.get(priority, 0) for priority in eu_priorities]
        labels = eu_priorities
        
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
        title=dict(
            text='EU Priorities Comparison - Selected Countries',
            font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
            x=0.5
        ),
        height=500,  # Match Budget dashboard chart height
        margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
        font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
    )
    
    # 4. Time series chart (EWBI evolution over time)
    time_series = go.Figure()
    
    # Get EWBI time series data (Level 1: EWBI)
    ewbi_time_data = time_df[
        (time_df['EU_Priority'] == 'All') & 
        (time_df['Secondary_indicator'] == 'All') & 
        (time_df['primary_index'] == 'All')
    ].copy()
    
    if not ewbi_time_data.empty:
        # Show countries in the determined order
        for country in ewbi_time_data['country'].unique():
            country_data = ewbi_time_data[ewbi_time_data['country'] == country].sort_values('year')
            
            time_series.add_trace(
                go.Scatter(
                    x=country_data['year'],
                    y=country_data['Score'],
                    name=country,
                    mode='lines+markers',
                    hovertemplate='%{y:.3f}<extra></extra>'
                )
            )
            
            time_series.update_layout(
                title=dict(
                    text='EWBI Score Evolution Over Time',
                font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                    x=0.5
                ),
            height=500,  # Match Budget dashboard chart height
            margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
            font=dict(family='Arial, sans-serif', size=14),  # Match Budget dashboard font size
                yaxis=dict(range=[0, 1])
            )
        else:
            time_series.add_annotation(
                text="Time series data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            
            time_series.update_layout(
                title=dict(
                    text='EWBI Score Evolution Over Time',
                font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                    x=0.5
                ),
            height=500,  # Match Budget dashboard chart height
            margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
            font=dict(family='Arial, sans-serif', size=14),  # Match Budget dashboard font size
        )
    

    
    return time_series, decile_analysis

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
    
    # Map is now handled separately by create_adaptive_map_chart
    
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
                x=[str(d) for d in country_data['decile']],
                y=country_data[eu_priority],
                name=country,
                text=country_data[eu_priority].round(2),
                textposition='auto'
            ))
    
    decile_analysis.update_layout(
        title=dict(
            text=f'{eu_priority} Scores by Decile - Selected Countries',
            font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
            x=0.5
        ),
        height=500,  # Match Budget dashboard chart height
        margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
        barmode='group',
        font=dict(family='Arial, sans-serif', size=14),  # Match Budget dashboard font size
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            gridcolor='lightgrey',
            gridwidth=0.5,
            showgrid=True
        ),
        yaxis=dict(
            range=[0, 1],
            gridcolor='lightgrey',
            gridwidth=0.5,
            showgrid=True
        )
    )
    
    # 3. Country comparison chart (EU priority vs secondary indicators)
    radar_chart = go.Figure()
    
    # Get all individual countries (excluding aggregates) and their scores
    country_scores = []
    for country in map_df['country'].unique():
        if 'Average' not in country:
            if eu_priority in map_df.columns:
                country_score = map_df[map_df['country'] == country][eu_priority].mean()
                country_scores.append((country, country_score))
            else:
                country_scores.append((country, 0))
    
    # Sort by score from highest to lowest
    country_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract sorted countries and scores
    individual_countries = [country for country, score in country_scores]
    eu_priority_scores = [score for country, score in country_scores]
    
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
                    hovertemplate='%{y:.2f}'
                )
            )
    
    # Update layout to match your styling
    radar_chart.update_layout(
        title=dict(
            text=f'{eu_priority} and Secondary Indicators by Country',
            font=dict(size=18, color="#f4d03f", weight="bold"),
            x=0.5
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
            x=0.5,
            y=-0.1,
            xanchor='center',
            yanchor='top',
            orientation='h'
        ),
        height=500,  # Match Budget dashboard chart height
        margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
        font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
    )
    
    # Set y-axis scale from 0 to 1
    radar_chart.update_layout(yaxis=dict(range=[0, 1]))
    
    # 4. Time series chart - Bar chart implementation
    time_series = go.Figure()
    
    # For EU Priority level, we'll use the already-loaded time series data
    try:
        if not time_series_df.empty:
            # Use the original working logic: filter for secondary indicators that belong to this EU Priority
            try:
                with open(os.path.join(DATA_DIR, '..', 'data', 'ewbi_indicators.json'), 'r') as f:
                    config = json.load(f)['EWBI']
                
                # Find secondary indicators for this EU Priority
                secondary_indicators = []
                for priority in config:
                    if priority['name'] == eu_priority:
                        for component in priority['components']:
                            secondary_indicators.append(component['name'])
                        break
                
                if secondary_indicators:
                    # Since the current time series file doesn't have proper secondary_indicator values,
                    # we'll use the primary_score column and aggregate it like the working version did
                    # This is a workaround to restore the working functionality
                    
                    # For time series, prioritize EU Countries Average, then add individual countries
                    countries_to_show = []
                    if 'EU Countries Average' in time_series_df['country'].values:
                        countries_to_show.append('EU Countries Average')
                    
                    # Add individual countries from the filter
                    individual_countries = [c for c in analysis_df['country'].unique() if 'Average' not in c]
                    countries_to_show.extend(individual_countries[:5])  # Limit to 5 for readability
                    
                    for country in countries_to_show:
                        if country in time_series_df['country'].values:
                            # Get the primary_score data for this country over time
                            country_data = time_series_df[time_series_df['country'] == country].sort_values('year')
                            
                            if not country_data.empty:
                                time_series.add_trace(
                                    go.Scatter(
                                        x=country_data['year'],
                                        y=country_data['primary_score'],
                                        name=country,
                                        mode='lines+markers',
                                        hovertemplate='%{y:.3f}<extra></extra>'
                                    )
                                )
                    
                    time_series.update_layout(
                        title=dict(
                            text=f'{eu_priority} Evolution Over Time (Primary Indicators)',
                            font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                            x=0.5
                        ),
                        height=500,  # Match Budget dashboard chart height
                        margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
                        font=dict(family='Arial, sans-serif', size=14),  # Match Budget dashboard font size
                        yaxis=dict(range=[0, 1])
                    )
                else:
                    time_series.add_annotation(
                        text=f"No secondary indicators found for {eu_priority}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=16)
                    )
                    
                    time_series.update_layout(
                        title=dict(
                            text=f'{eu_priority} Evolution Over Time',
                            font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                            x=0.5
                        ),
                        height=500,  # Match Budget dashboard chart height
                        margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
                        font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
                    )
            except Exception as e:
                time_series.add_annotation(
                    text=f"Error loading EWBI structure for {eu_priority}: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
                
                time_series.update_layout(
                    title=dict(
                        text=f'{eu_priority} Evolution Over Time',
                        font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                        x=0.5
                    ),
                    height=500,  # Match Budget dashboard chart height
                    margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
                    font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
                )
        else:
            time_series.add_annotation(
                text="Time series data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            
            time_series.update_layout(
                title=dict(
                    text=f'{eu_priority} Evolution Over Time',
                    font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                    x=0.5
                ),
                height=500,  # Match Budget dashboard chart height
                margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
                font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
            )
    except Exception as e:
        time_series.add_annotation(
            text=f"Error loading time series data for {eu_priority}: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        
        time_series.update_layout(
            title=dict(
                text=f'{eu_priority} Evolution Over Time',
                font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                x=0.5
            ),
            height=500,  # Match Budget dashboard chart height
            margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
            font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
        )
    
    # 5. Secondary chart - Country comparison for EU priority scores
    secondary_chart = go.Figure()
    
    # Get individual countries (excluding aggregates) and their EU priority scores
    country_scores = []
    for country in map_df['country'].unique():
        if 'Average' not in country:
            if eu_priority in map_df.columns:
                country_score = map_df[map_df['country'] == country][eu_priority].mean()
                country_scores.append((country, country_score))
            else:
                country_scores.append((country, 0))
    
    # Sort by score from highest to lowest
    country_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract sorted countries and scores
    individual_countries = [country for country, score in country_scores]
    eu_priority_scores = [score for country, score in country_scores]
    
    # Add the EU priority score line
    secondary_chart.add_trace(
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
            hovertemplate='%{y:.3f}<extra></extra>'
        )
    )
    
    secondary_chart.update_layout(
        title=dict(
            text=f'{eu_priority} Scores by Country',
            font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
            x=0.5
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
        showlegend=False,  # No legend needed for single indicator
        height=500,  # Match Budget dashboard chart height
        margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
        font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
    )
    
    return time_series, decile_analysis

def create_primary_indicator_charts(map_df, analysis_df, time_df, eu_priority, secondary_indicator, primary_indicator):
    """Create charts for primary indicator level"""
    
    # Map is now handled separately by create_adaptive_map_chart
    
    # Get the primary indicator column name
    primary_col = f"primary_{primary_indicator}"
    
    # 2. Decile analysis chart
    decile_analysis = go.Figure()
    
    if primary_col in analysis_df.columns:
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
            country_data = analysis_df[analysis_df['country'] == country].sort_values('decile')
            decile_analysis.add_trace(go.Bar(
                x=[str(d) for d in country_data['decile']],
                y=country_data[primary_col],
                name=country,
                text=country_data[primary_col].round(2),
                textposition='auto'
            ))
        
        decile_analysis.update_layout(
            title=dict(
                text=f'{primary_indicator} Scores by Income Decile',
                font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                x=0.5
            ),
            height=500,  # Match Budget dashboard chart height
            margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
            font=dict(family='Arial, sans-serif', size=14),  # Match Budget dashboard font size
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                gridcolor='lightgrey',
                gridwidth=0.5,
                showgrid=True
            ),
            yaxis=dict(
                range=[0, 1],
                gridcolor='lightgrey',
                gridwidth=0.5,
                showgrid=True
            )
        )
    else:
        decile_analysis.add_annotation(
            text=f"Primary indicator {primary_indicator} not found in data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    
    # 3. Country comparison chart (primary indicator only - no underlying indicators)
    radar_chart = go.Figure()
    
    if primary_col in map_df.columns:
        # Get individual countries (excluding aggregates) and their scores
        country_scores = []
        for country in map_df['country'].unique():
            if 'Average' not in country:
                score = map_df[map_df['country'] == country][primary_col].mean()
                country_scores.append((country, score))
        
        # Sort by score from highest to lowest
        country_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract sorted countries and scores
        individual_countries = [country for country, score in country_scores]
        primary_scores = [score for country, score in country_scores]
        
        radar_chart.add_trace(
            go.Scatter(
                x=individual_countries,
                y=primary_scores,
                name=primary_indicator,
                mode='lines+markers',
                line=dict(width=4, color='#1f77b4'),
                marker=dict(
                    size=10,
                    color='white',
                    line=dict(color='#1f77b4', width=2)
                ),
                hovertemplate='%{y:.2f}'
            )
        )
        
        radar_chart.update_layout(
            title=dict(
                text=f'{primary_indicator} Scores by Country',
                font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                x=0.5
            ),
            xaxis=dict(tickangle=45, tickfont=dict(size=12)),
            yaxis=dict(range=[0, 1], tickformat='.1f', gridwidth=0.2),
            hovermode='closest',
            showlegend=False,  # No legend needed for single indicator
            height=500,  # Match Budget dashboard chart height
            margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
            font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
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
    
    # For time series, we'll use the already-loaded time series dataframe
    try:
        # Filter for the specific primary indicator
        if 'primary_index' in time_series_df.columns:
            indicator_time_data = time_series_df[time_series_df['primary_index'] == primary_indicator]
            
            if not indicator_time_data.empty:
                # Group by country and year, average across deciles
                country_year_data = indicator_time_data.groupby(['country', 'year'])['primary_score'].mean().reset_index()
                
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
                    title=dict(
                        text=f'{primary_indicator} Evolution Over Time',
                        font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                        x=0.5
                    ),
                    height=500,  # Match Budget dashboard chart height
                    margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
                    font=dict(family='Arial, sans-serif', size=14),  # Match Budget dashboard font size
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
    
    # 5. Secondary chart - Country comparison for primary indicator scores
    secondary_chart = go.Figure()
    
    # Get individual countries (excluding aggregates) and their primary indicator scores
    country_scores = []
    for country in map_df['country'].unique():
        if 'Average' not in country:
            if primary_col in map_df.columns:
                country_score = map_df[map_df['country'] == country][primary_col].mean()
                country_scores.append((country, country_score))
            else:
                country_scores.append((country, 0))
    
    # Sort by score from highest to lowest
    country_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract sorted countries and scores
    individual_countries = [country for country, score in country_scores]
    primary_scores = [score for country, score in country_scores]
    
    # Add the primary indicator score line
    secondary_chart.add_trace(
        go.Scatter(
            x=individual_countries,
            y=primary_scores,
            name=primary_indicator,
            mode='lines+markers',
            line=dict(width=4, color='#1f77b4'),
            marker=dict(
                size=10,
                color='white',
                line=dict(color='#1f77b4', width=2)
            ),
            hovertemplate='%{y:.3f}<extra></extra>'
        )
    )
    
    secondary_chart.update_layout(
        title=dict(
            text=f'{primary_indicator} Scores by Country',
            font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
            x=0.5
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
        showlegend=False,  # No legend needed for single indicator
        height=500,  # Match Budget dashboard chart height
        margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
        font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
    )
    
    return time_series, decile_analysis

def create_secondary_indicator_charts(map_df, analysis_df, time_df, eu_priority, secondary_indicator):
    """Create charts for secondary indicator level (EU Priority + Specific Secondary)"""
    
    # Map is now handled separately by create_adaptive_map_chart
    
    # Get the secondary indicator column name
    # The actual columns don't have 'secondary_' prefix
    # Fix the replacement logic to only replace ' and ' with '_and_', not every space
    secondary_col = f"{eu_priority.replace(' and ', '_and_').replace(' ', '_')}_{secondary_indicator.replace(' and ', '_and_').replace(' ', '_')}"
    
    # 2. Decile analysis chart
    decile_analysis = go.Figure()
    
    if secondary_col in analysis_df.columns:
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
            country_data = analysis_df[analysis_df['country'] == country].sort_values('decile')
            decile_analysis.add_trace(go.Bar(
                x=[str(d) for d in country_data['decile']],
                y=country_data[secondary_col],
                name=country,
                text=country_data[secondary_col].round(2),
                textposition='auto'
            ))
        
        decile_analysis.update_layout(
            title=dict(
                text=f'{secondary_indicator} Scores by Income Decile',
                font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                x=0.5
            ),
            height=500,  # Match Budget dashboard chart height
            margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
            font=dict(family='Arial, sans-serif', size=14),  # Match Budget dashboard font size
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                gridcolor='lightgrey',
                gridwidth=0.5,
                showgrid=True
            ),
            yaxis=dict(
                range=[0, 1],
                gridcolor='lightgrey',
                gridwidth=0.5,
                showgrid=True
            )
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
            
            # Get individual countries (excluding aggregates) and their scores
            country_scores = []
            for country in map_df['country'].unique():
                if 'Average' not in country:
                    score = map_df[map_df['country'] == country][secondary_col].mean()
                    country_scores.append((country, score))
            
            # Sort by score from highest to lowest
            country_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Extract sorted countries and scores
            individual_countries = [country for country, score in country_scores]
            secondary_scores = [score for country, score in country_scores]
            
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
                    hovertemplate='%{y:.2f}'
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
                title=dict(
                    text=f'{secondary_indicator} vs Primary Indicators by Country',
                    font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                    x=0.5
                ),
                xaxis=dict(tickangle=45, tickfont=dict(size=12)),
                yaxis=dict(range=[0, 1], tickformat='.1f', gridwidth=0.2),
                hovermode='closest',
                showlegend=True,
                legend=dict(
                    x=0.5,
                    y=-0.1,
                    xanchor='center',
                    yanchor='top',
                    orientation='h'
                ),
                height=500,  # Match Budget dashboard chart height
                margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
                font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
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
    
    # For secondary indicator level, we'll use the already-loaded time series data
    try:
        if not time_series_df.empty:
            # Since the current time series file doesn't have proper secondary_indicator values,
            # we'll use the primary_score column and aggregate it like the working version did
            # This is a workaround to restore the working functionality
            
            # For time series, prioritize EU Countries Average, then add individual countries
            countries_to_show = []
            if 'EU Countries Average' in time_series_df['country'].values:
                countries_to_show.append('EU Countries Average')
            
            # Add individual countries from the filter
            individual_countries = [c for c in analysis_df['country'].unique() if 'Average' not in c]
            countries_to_show.extend(individual_countries[:5])  # Limit to 5 for readability
            
            for country in countries_to_show:
                if country in time_series_df['country'].values:
                    # Get the primary_score data for this country over time
                    country_data = time_series_df[time_series_df['country'] == country].sort_values('year')
                    
                    if not country_data.empty:
                        time_series.add_trace(
                            go.Scatter(
                                x=country_data['year'],
                                y=country_data['primary_score'],
                                name=country,
                                mode='lines+markers',
                                hovertemplate='%{y:.3f}<extra></extra>'
                            )
                        )
                
            time_series.update_layout(
                title=dict(
                    text=f'{secondary_indicator} Evolution Over Time (Primary Indicators)',
                    font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                    x=0.5
                ),
                height=500,  # Match Budget dashboard chart height
                margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
                font=dict(family='Arial, sans-serif', size=14),  # Match Budget dashboard font size
                yaxis=dict(range=[0, 1])
            )
        else:
            time_series.add_annotation(
                text="Time series data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
        
        time_series.update_layout(
            title=dict(
                text=f'{secondary_indicator} Evolution Over Time',
                font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                x=0.5
            ),
            height=500,  # Match Budget dashboard chart height
            margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
            font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
        )
    except Exception as e:
        time_series.add_annotation(
            text=f"Error loading time series data for {secondary_indicator}: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    
        time_series.update_layout(
            title=dict(
                text=f'{secondary_indicator} Evolution Over Time',
                font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
                x=0.5
            ),
            height=500,  # Match Budget dashboard chart height
            margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
            font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
        )
    
    # 5. Secondary chart - Country comparison for secondary indicator scores
    secondary_chart = go.Figure()
    
    # Get individual countries (excluding aggregates) and their secondary indicator scores
    country_scores = []
    for country in map_df['country'].unique():
        if 'Average' not in country:
            if secondary_col in map_df.columns:
                country_score = map_df[map_df['country'] == country][secondary_col].mean()
                country_scores.append((country, country_score))
            else:
                country_scores.append((country, 0))
    
    # Sort by score from highest to lowest
    country_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract sorted countries and scores
    individual_countries = [country for country, score in country_scores]
    secondary_scores = [score for country, score in country_scores]
    
    # Add the secondary indicator score line
    secondary_chart.add_trace(
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
            hovertemplate='%{y:.3f}<extra></extra>'
        )
    )
    
    secondary_chart.update_layout(
        title=dict(
            text=f'{secondary_indicator} Scores by Country',
            font=dict(size=16, color="#f4d03f", weight="bold"),  # Match Budget dashboard title size
            x=0.5
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
        showlegend=False,  # No legend needed for single indicator
        height=500,  # Match Budget dashboard chart height
        margin=dict(t=80, b=50, l=60, r=60),  # Match Budget dashboard margins
        font=dict(family='Arial, sans-serif', size=14)  # Match Budget dashboard font size
    )
    
    return time_series, decile_analysis

def create_level_filters(eu_priority, secondary_indicator, primary_indicator):
    """Create level-based filters for the dashboard"""
    level_filters = {
        'eu_priority': eu_priority,
        'secondary_indicator': secondary_indicator, 
        'primary_indicator': primary_indicator
    }
    
    # Determine the current level based on filter combinations
    if eu_priority == 'ALL':
        # Level 1: EWBI (overall) - All filters are ALL
        level_filters['current_level'] = 1
        level_filters['level_name'] = 'EWBI (Overall)'
    elif eu_priority != 'ALL' and (secondary_indicator == 'ALL' or not secondary_indicator):
        # Level 2: EU Priority selected, Secondary and Primary are ALL
        level_filters['current_level'] = 2
        level_filters['level_name'] = f'EU Priority: {eu_priority}'
    elif eu_priority != 'ALL' and secondary_indicator != 'ALL' and (primary_indicator == 'ALL' or not primary_indicator):
        # Level 3: EU Priority and Secondary selected, Primary is ALL
        level_filters['current_level'] = 3
        level_filters['level_name'] = f'Secondary: {secondary_indicator}'
    elif eu_priority != 'ALL' and secondary_indicator != 'ALL' and primary_indicator != 'ALL':
        # Level 4: All three filters have specific values
        level_filters['current_level'] = 4
        level_filters['level_name'] = f'Primary: {primary_indicator}'
    else:
        # Fallback case
        level_filters['current_level'] = 1
        level_filters['level_name'] = 'EWBI (Overall)'
    
    return level_filters

def create_adaptive_map_chart(map_df, level_filters):
    """Create an adaptive map chart that works for all 4 levels"""
    
    print(f"DEBUG: Creating map for Level {level_filters['current_level']}: {level_filters['level_name']}")
    print(f"DEBUG: Filters - EU Priority: {level_filters['eu_priority']}, Secondary: {level_filters['secondary_indicator']}, Primary: {level_filters['primary_indicator']}")
    
    # Filter data based on current level
    if level_filters['current_level'] == 1:
        # Level 1: EWBI (overall)
        filtered_data = map_df[
            (map_df['EU_Priority'] == 'All') & 
            (map_df['Secondary_indicator'] == 'All') & 
            (map_df['primary_index'] == 'All')
        ].copy()
        title = 'Well-Being Score by Country - All EU Priorities'
        colorbar_title = "EWBI Score"
        
    elif level_filters['current_level'] == 2:
        # Level 2: EU Priority
        filtered_data = map_df[
            (map_df['EU_Priority'] == level_filters['eu_priority']) & 
            (map_df['Secondary_indicator'] == 'All') & 
            (map_df['primary_index'] == 'All')
        ].copy()
        title = f'Well-Being Score by Country - {level_filters["eu_priority"]} - All Secondary Indicators'
        colorbar_title = "Score"
        
    elif level_filters['current_level'] == 3:
        # Level 3: Secondary Indicator
        filtered_data = map_df[
            (map_df['EU_Priority'] == level_filters['eu_priority']) & 
            (map_df['Secondary_indicator'] == level_filters['secondary_indicator']) & 
            (map_df['primary_index'] == 'All')
        ].copy()
        title = f'Well-Being Score by Country - {level_filters["eu_priority"]} - {level_filters["secondary_indicator"]} - All Primary Indicators'
        colorbar_title = "Score"
        
    else:  # Level 4: Primary Indicator
        filtered_data = map_df[
            (map_df['EU_Priority'] == level_filters['eu_priority']) & 
            (map_df['Secondary_indicator'] == level_filters['secondary_indicator']) & 
            (map_df['primary_index'] == level_filters['primary_indicator'])
        ].copy()
        title = f'Well-Being Score by Country - {level_filters["eu_priority"]} - {level_filters["secondary_indicator"]} - {level_filters["primary_indicator"]}'
        colorbar_title = "Score"
    
    # Convert ISO-2 codes to ISO-3 codes for the map
    filtered_data['iso3'] = filtered_data['country'].map(ISO2_TO_ISO3)
    
    # Filter out countries without ISO-3 codes (like aggregates)
    filtered_data = filtered_data[filtered_data['iso3'].notna()].copy()
    
    # Create the choropleth map
    european_map = go.Figure(data=go.Choropleth(
        locations=filtered_data['iso3'],
        z=filtered_data['Score'],
        locationmode='ISO-3',
        colorscale='RdYlGn',  # Red to Green scale
        colorbar_title=colorbar_title,
        text=filtered_data['country'] + ': ' + filtered_data['Score'].round(2).astype(str),
        hovertemplate='<b>%{text}</b><br>' +
                      f'{colorbar_title}: %{{z:.2f}}<br>' +
                      '<extra></extra>'
    ))
    
    # Update traces for better styling like Budget dashboard
    european_map.update_traces(
        marker_line_color="white",
        marker_line_width=0.5,
        colorbar=dict(
            x=1.05,  # Position to the right of the map
            xanchor="left",
            thickness=15,
            len=0.7,
            title=dict(
                text=colorbar_title,
                font=dict(size=14, color="#2c3e50"),
                side="top"
            )
        )
    )
    
    european_map.update_layout(
        height=550,  # Match Budget dashboard map height
        margin={"r": 150, "t": 80, "l": 150, "b": 50},  # Match Budget dashboard margins
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
        paper_bgcolor='white',  # Clean white background
        plot_bgcolor='white',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#2c3e50"
        ),
        title=dict(
            text=title,
            font=dict(size=18, color="#f4d03f", weight="bold"),  # Same size as other titles
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
    
    return european_map

def create_adaptive_decile_chart(analysis_df, level_filters):
    """Create an adaptive decile analysis chart that works for all 4 levels"""
    
    print(f"DEBUG: Creating decile chart for Level {level_filters['current_level']}: {level_filters['level_name']}")
    
    # Define consistent color scheme for countries
    country_colors = {
        'EU Average': '#1f77b4',  # Blue
        'FR': '#ff7f0e',          # Orange/Red
        'DE': '#2ca02c',          # Green
        'IT': '#d62728',          # Red
        'ES': '#9467bd',          # Purple
        'PL': '#8c564b',          # Brown
        'RO': '#e377c2',          # Pink
        'NL': '#7f7f7f',          # Gray
        'BE': '#bcbd22',          # Olive
        'SE': '#17becf',          # Cyan
        'AT': '#ff9896',          # Light Red
        'BG': '#98df8a',          # Light Green
        'HR': '#fdd0a2',          # Light Orange
        'CY': '#c5b0d5',          # Light Purple
        'CZ': '#c49c94',          # Light Brown
        'DK': '#f7b6d2',          # Light Pink
        'EE': '#c7c7c7',          # Light Gray
        'FI': '#dbdb8d',          # Light Olive
        'EL': '#9edae5',          # Light Cyan
        'HU': '#ffed4f',          # Yellow
        'IS': '#ff9896',          # Light Red
        'IE': '#98df8a',          # Light Green
        'LV': '#fdd0a2',          # Light Orange
        'LT': '#c5b0d5',          # Light Purple
        'LU': '#c49c94',          # Light Brown
        'MT': '#f7b6d2',          # Light Pink
        'NO': '#c7c7c7',          # Light Gray
        'PT': '#dbdb8d',          # Light Olive
        'SK': '#9edae5',          # Light Cyan
        'SI': '#ffed4f',          # Yellow
        'UK': '#ff9896'           # Light Red
    }
    
    # Filter data based on current level
    if level_filters['current_level'] == 1:
        # Level 1: EWBI (overall)
        filtered_data = analysis_df[
            (analysis_df['EU_Priority'] == 'All') & 
            (analysis_df['Secondary_indicator'] == 'All') & 
            (analysis_df['primary_index'] == 'All')
        ].copy()
        title = 'Well-Being Score by Decile - All EU Priorities'
        
    elif level_filters['current_level'] == 2:
        # Level 2: EU Priority
        filtered_data = analysis_df[
            (analysis_df['EU_Priority'] == level_filters['eu_priority']) & 
            (analysis_df['Secondary_indicator'] == 'All') & 
            (analysis_df['primary_index'] == 'All')
        ].copy()
        title = f'Well-Being Score by Decile - {level_filters["eu_priority"]} - All Secondary Indicators'
        
    elif level_filters['current_level'] == 3:
        # Level 3: Secondary Indicator
        filtered_data = analysis_df[
            (analysis_df['EU_Priority'] == level_filters['eu_priority']) & 
            (analysis_df['Secondary_indicator'] == level_filters['secondary_indicator']) & 
            (analysis_df['primary_index'] == 'All')
        ].copy()
        title = f'Well-Being Score by Decile - {level_filters["eu_priority"]} - {level_filters["secondary_indicator"]} - All Primary Indicators'
        
    else:  # Level 4: Primary Indicator
        filtered_data = analysis_df[
            (analysis_df['EU_Priority'] == level_filters['eu_priority']) & 
            (analysis_df['Secondary_indicator'] == level_filters['secondary_indicator']) & 
            (analysis_df['primary_index'] == level_filters['primary_indicator'])
        ].copy()
        title = f'Well-Being Score by Decile - {level_filters["eu_priority"]} - {level_filters["secondary_indicator"]} - {level_filters["primary_indicator"]}'
    
    # Create the decile analysis chart
    decile_analysis = go.Figure()
    
    # For decile analysis, prioritize EU Average, then add individual countries
    countries_to_show = []
    
    # Always show EU Average first if available
    if 'EU Average' in filtered_data['country'].values:
        countries_to_show.append('EU Average')
    
    # Then add any other selected countries (excluding EU Average to avoid duplication)
    other_countries = [c for c in filtered_data['country'].unique() if c != 'EU Average' and 'Average' not in c]
    countries_to_show.extend(other_countries)
    
    # Show countries in the determined order
    for country in countries_to_show:
        if country in filtered_data['country'].values:
            country_data = filtered_data[filtered_data['country'] == country].copy()
            
            # Sort deciles in proper order: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, All
            decile_order = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'All']
            country_data['decile_str'] = country_data['decile'].astype(str)
            country_data = country_data.sort_values('decile_str', key=lambda x: pd.Categorical(x, categories=decile_order, ordered=True))
            
            decile_analysis.add_trace(go.Bar(
                x=country_data['decile_str'],
                y=country_data['Score'],
                name=country,
                text=country_data['Score'].round(2),
                textposition='auto',
                marker_color=country_colors.get(country, '#1f77b4')  # Use consistent colors
            ))
    
    decile_analysis.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color="#f4d03f", weight="bold"),
            x=0.5
        ),
        height=500,
        width=700,  # 20% less wide (500 * 0.8 = 400)
        margin=dict(t=80, b=80, l=60, r=60),  # Increased bottom margin for legend
        barmode='group',
        font=dict(family='Arial, sans-serif', size=14),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=False,  # Remove vertical grid lines
            categoryorder='array',
            categoryarray=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'All']
        ),
        yaxis=dict(
            range=[0, 1],
            showgrid=False  # Remove horizontal grid lines
        ),
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=-0.2,  # Position below the chart
            xanchor="center",
            x=0.5,  # Center horizontally
            bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent white background
            bordercolor='lightgray',
            borderwidth=1
        ),
        showlegend=True  # Force legend to show even with one trace
    )
    
    return decile_analysis

def create_adaptive_radar_country_chart(analysis_df, level_filters):
    """Create an adaptive radar/country comparison chart that works for all 4 levels"""
    
    print(f"DEBUG: Creating radar/country chart for Level {level_filters['current_level']}: {level_filters['level_name']}")
    
    if level_filters['current_level'] == 1:
        # Level 1: Radar chart (EU Priorities comparison)
        return create_level1_radar_chart(analysis_df)
    else:
        # Levels 2-4: Country comparison chart
        return create_levels2to4_country_chart(analysis_df, level_filters)

def create_level1_radar_chart(analysis_df):
    """Create Level 1 radar chart (EU Priorities comparison)"""
    
    radar_chart = go.Figure()
    
    # Define consistent color scheme for countries (same as decile chart)
    country_colors = {
        'EU Average': '#1f77b4',  # Blue
        'FR': '#ff7f0e',          # Orange/Red
        'DE': '#2ca02c',          # Green
        'IT': '#d62728',          # Red
        'ES': '#9467bd',          # Purple
        'PL': '#8c564b',          # Brown
        'RO': '#e377c2',          # Pink
        'NL': '#7f7f7f',          # Gray
        'BE': '#bcbd22',          # Olive
        'SE': '#17becf',          # Cyan
        'AT': '#ff9896',          # Light Red
        'BG': '#98df8a',          # Light Green
        'HR': '#fdd0a2',          # Light Orange
        'CY': '#c5b0d5',          # Light Purple
        'CZ': '#c49c94',          # Light Brown
        'DK': '#f7b6d2',          # Light Pink
        'EE': '#c7c7c7',          # Light Gray
        'FI': '#dbdb8d',          # Light Olive
        'EL': '#9edae5',          # Light Cyan
        'HU': '#ffed4f',          # Yellow
        'IS': '#ff9896',          # Light Red
        'IE': '#98df8a',          # Light Green
        'LV': '#fdd0a2',          # Light Orange
        'LT': '#c5b0d5',          # Light Purple
        'LU': '#c49c94',          # Light Brown
        'MT': '#f7b6d2',          # Light Pink
        'NO': '#c7c7c7',          # Light Gray
        'PT': '#dbdb8d',          # Light Olive
        'SK': '#9edae5',          # Light Cyan
        'SI': '#ffed4f',          # Yellow
        'UK': '#ff9896'           # Light Red
    }
    
    # Get EU priority data for selected countries (Level 2: EU Priority)
    eu_priority_data = analysis_df[
        (analysis_df['EU_Priority'] != 'All') & 
        (analysis_df['Secondary_indicator'] == 'All') & 
        (analysis_df['primary_index'] == 'All')
    ].copy()
    
    # Get unique EU priorities
    eu_priorities = eu_priority_data['EU_Priority'].unique()
    
    # Function to wrap long labels into multiple lines
    def wrap_label(label, max_length=40):
        """Break long labels into multiple lines for better readability"""
        if len(label) <= max_length:
            return label
        
        # Try to break at natural points (commas, "and", spaces)
        if ', ' in label:
            parts = label.split(', ')
            if len(parts) == 2:
                return f"{parts[0]},<br>{parts[1]}"
            elif len(parts) >= 3:
                # Always limit to 2 parts maximum
                first_part = parts[0]
                remaining_parts = ', '.join(parts[1:])
                return f"{first_part},<br>{remaining_parts}"
        elif ' and ' in label:
            parts = label.split(' and ')
            if len(parts) == 2:
                return f"{parts[0]}<br>and {parts[1]}"
        
        # If no natural break points, break at spaces
        words = label.split()
        if len(words) <= 3:
            return label
        
        # Break into roughly equal parts
        mid = len(words) // 2
        return f"{' '.join(words[:mid])}<br>{' '.join(words[mid:])}"
    
    # Wrap long labels
    wrapped_labels = [wrap_label(priority) for priority in eu_priorities]
    
    # Show countries in the determined order
    for country in eu_priority_data['country'].unique():
        country_data = eu_priority_data[eu_priority_data['country'] == country]
        
        # Create a dictionary to map EU priority to score
        priority_scores = {}
        for _, row in country_data.iterrows():
            priority_scores[row['EU_Priority']] = row['Score']
        
        # Get values and labels for radar chart
        values = [priority_scores.get(priority, 0) for priority in eu_priorities]
        labels = eu_priorities
        
        radar_chart.add_trace(go.Scatterpolar(
            r=values,
            theta=wrapped_labels,
            fill='toself',
            name=country,
            line_color=country_colors.get(country, '#1f77b4'),  # Use consistent colors
            fillcolor=country_colors.get(country, '#1f77b4')     # Use consistent colors
        ))
    
    radar_chart.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        legend=dict(
            x=0.5,  # Center legend horizontally at the bottom
            y=-0.15,  # Position below the chart
            xanchor='center',  # Center horizontally
            yanchor='top',  # Anchor to top of legend
            orientation='h',  # Horizontal layout like decile chart
            itemsizing='constant',
            itemwidth=30,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        title=dict(
            text='EU Priorities Comparison - Selected Countries',
            font=dict(size=16, color="#f4d03f", weight="bold"),
            x=0.5
        ),
        height=500,
        margin=dict(t=80, b=80, l=60, r=60),  # Increased bottom margin for legend below
        font=dict(family='Arial, sans-serif', size=14)
    )
    
    return radar_chart

def create_levels2to4_country_chart(analysis_df, level_filters):
    """Create Levels 2-4 country comparison chart"""
    
    # Country name mapping from ISO2 to full names
    country_names = {
        'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'HR': 'Croatia', 
        'CY': 'Cyprus', 'CZ': 'Czech Republic', 'DK': 'Denmark', 'EE': 'Estonia',
        'FI': 'Finland', 'FR': 'France', 'DE': 'Germany', 'EL': 'Greece', 
        'HU': 'Hungary', 'IS': 'Iceland', 'IE': 'Ireland', 'IT': 'Italy',
        'LV': 'Latvia', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'MT': 'Malta',
        'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal',
        'RO': 'Romania', 'SK': 'Slovakia', 'SI': 'Slovenia', 'ES': 'Spain',
        'SE': 'Sweden', 'UK': 'United Kingdom'
    }
    
    country_chart = go.Figure()
    
    # Filter data based on current level
    if level_filters['current_level'] == 2:
        # Level 2: EU Priority - show EU Priority + Secondary Indicators
        filtered_data = analysis_df[
            (analysis_df['EU_Priority'] == level_filters['eu_priority']) & 
            (analysis_df['Secondary_indicator'] != 'All') & 
            (analysis_df['primary_index'] == 'All')
        ].copy()
        
        # Get unique secondary indicators for this EU priority
        indicators = filtered_data['Secondary_indicator'].unique()
        title = f'{level_filters["eu_priority"]} and Secondary Indicators by Country'
        
    elif level_filters['current_level'] == 3:
        # Level 3: Secondary Indicator - show Secondary Indicator + Primary Indicators
        filtered_data = analysis_df[
            (analysis_df['EU_Priority'] == level_filters['eu_priority']) & 
            (analysis_df['Secondary_indicator'] == level_filters['secondary_indicator']) & 
            (analysis_df['primary_index'] != 'All')
        ].copy()
        
        # Get unique primary indicators for this secondary indicator
        indicators = filtered_data['primary_index'].unique()
        title = f'{level_filters["secondary_indicator"]} and Primary Indicators by Country'
        
    else:  # Level 4: Primary Indicator - show Primary Indicator only
        filtered_data = analysis_df[
            (analysis_df['EU_Priority'] == level_filters['eu_priority']) & 
            (analysis_df['Secondary_indicator'] == level_filters['secondary_indicator']) & 
            (analysis_df['primary_index'] == level_filters['primary_indicator'])
        ].copy()
        
        # For Level 4, we only have one indicator
        indicators = [level_filters['primary_indicator']]
        title = f'{level_filters["primary_indicator"]} by Country'
    
    # Get all individual countries (excluding aggregates)
    countries = [c for c in filtered_data['country'].unique() if 'Average' not in c]
    
    if not countries:
        # No individual countries found, show message
        country_chart.add_annotation(
            text="No individual country data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        country_chart.update_layout(
            title=dict(text=title, font=dict(size=16, color="#f4d03f", weight="bold"), x=0.5),
            height=500,
            margin=dict(t=80, b=50, l=60, r=60)
        )
        return country_chart
    
    # Sort countries by their score for the main indicator (first in the list)
    main_indicator = indicators[0]
    country_scores = []
    for country in countries:
        country_data = filtered_data[
            (filtered_data['country'] == country) & 
            (filtered_data['decile'] == 'All')
        ]
        if not country_data.empty:
            score = country_data['Score'].iloc[0]
            country_scores.append((country, score))
        else:
            country_scores.append((country, 0))
    
    # Sort by score from highest to lowest
    country_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_countries = [country for country, score in country_scores]
    
    # Convert ISO2 codes to full country names for display
    sorted_countries_display = [country_names.get(country, country) for country in sorted_countries]
    
    # Add the main indicator line
    main_scores = [score for country, score in country_scores]
    country_chart.add_trace(
        go.Scatter(
            x=sorted_countries_display,
            y=main_scores,
            name=main_indicator,
            mode='lines+markers',
            line=dict(width=4, color='#1f77b4'),
            marker=dict(
                size=10,
                color='white',
                line=dict(color='#1f77b4', width=2)
            ),
            hovertemplate='<b>Score:</b> %{y:.3f}<br><b>Country:</b> %{x}<br><b>Indicator:</b> %{fullData.name}<extra></extra>'
        )
    )
    
    # Add other indicators if they exist
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, indicator in enumerate(indicators[1:], 1):  # Skip first indicator (already added)
        if i < len(colors):
            indicator_scores = []
            for country in sorted_countries:
                country_data = filtered_data[
                    (filtered_data['country'] == country) & 
                    (filtered_data['decile'] == 'All')
                ]
                if not country_data.empty:
                    # Find the specific indicator data
                    if level_filters['current_level'] == 2:
                        # Level 2: secondary indicator
                        indicator_data = country_data[country_data['Secondary_indicator'] == indicator]
                    elif level_filters['current_level'] == 3:
                        # Level 3: primary indicator
                        indicator_data = country_data[country_data['primary_index'] == indicator]
                    else:
                        indicator_data = country_data
                    
                    if not indicator_data.empty:
                        score = indicator_data['Score'].iloc[0]
                        indicator_scores.append(score)
                    else:
                        indicator_scores.append(0)
                else:
                    indicator_scores.append(0)
            
            country_chart.add_trace(
                go.Scatter(
                    x=sorted_countries_display,
                    y=indicator_scores,
                    name=indicator,
                    mode='lines+markers',
                    line=dict(color=colors[i]),
                    marker=dict(
                        size=8,
                        color='white',
                        line=dict(color=colors[i], width=2)
                    ),
                    hovertemplate='<b>Score:</b> %{y:.3f}<br><b>Country:</b> %{x}<br><b>Indicator:</b> %{fullData.name}<extra></extra>'
                )
            )
    
    # Update layout
    country_chart.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color="#f4d03f", weight="bold"),
            x=0.5
        ),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            range=[0, 1],
            tickformat='.1f',
            gridwidth=0.2,
            title=''  # Remove "Score" label
        ),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.5,
            y=-0.15,  # Move legend closer to chart to reduce padding
            xanchor='center',
            yanchor='top',
            orientation='h'
        ),
        height=500,
        margin=dict(t=80, b=60, l=60, r=60),  # Reduced bottom margin
        font=dict(family='Arial, sans-serif', size=14),
        paper_bgcolor='white',  # Remove grey background
        plot_bgcolor='white'    # Remove grey background
    )
    
    return country_chart

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)