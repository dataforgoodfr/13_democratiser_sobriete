import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Load the data
file_path = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output/emissions_trajectory.csv'
df = pd.read_csv(file_path)

# Drop rows with null values in the relevant columns
df.dropna(subset=['region', 'country', 'repartition_method', 'probability_of_reach', 'temperature', 'scope'], inplace=True)

# Initialize the Dash app
app = Dash(__name__)

# Create the layout of the app
app.layout = html.Div([
    html.H1("Interactive Emissions Dashboard"),

    # Dropdowns for scenario selection
    html.Div([
        html.Label("Select Continent:"),
        dcc.Dropdown(
            id='continent-dropdown',
            options=[{'label': 'Global', 'value': 'Global'}] + [{'label': continent, 'value': continent} for continent in df['region'].unique() if continent != 'Global'],
            value='Global',
            clearable=False
        ),
    ]),

    html.Div([
        html.Label("Select Country:"),
        dcc.Dropdown(
            id='country-dropdown',
            value='All',
            clearable=False
        ),
    ]),

    html.Div([
        html.Label("Select Repartition Method:"),
        dcc.Dropdown(
            id='repartition-dropdown',
            options=[{'label': method, 'value': method} for method in df['repartition_method'].unique()],
            value='equality',
            clearable=False
        ),
    ]),

    html.Div([
        html.Label("Select Probability of Reach:"),
        dcc.Dropdown(
            id='probability-dropdown',
            options=[{'label': str(prob), 'value': prob} for prob in df['probability_of_reach'].unique()],
            value=50,
            clearable=False
        ),
    ]),

    html.Div([
        html.Label("Select Temperature:"),
        dcc.Dropdown(
            id='temperature-dropdown',
            options=[{'label': str(temp), 'value': temp} for temp in df['temperature'].unique()],
            value=1.5,
            clearable=False
        ),
    ]),

    html.Div([
        html.Label("Select Scope:"),
        dcc.Dropdown(
            id='scope-dropdown',
            options=[{'label': scope, 'value': scope} for scope in df['scope'].unique()],
            value='consumption_based',
            clearable=False
        ),
    ]),

    # Container for graphs
    html.Div([
        # World map
        html.Div(dcc.Graph(id='world-map'), style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

        # Line graph
        html.Div(dcc.Graph(id='line-graph'), style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
    ])
])

# Callback to update the country dropdown based on selected continent
@app.callback(
    Output('country-dropdown', 'options'),
    Output('country-dropdown', 'value'),
    Input('continent-dropdown', 'value')
)
def update_country_dropdown(selected_continent):
    if selected_continent == 'Global':
        countries = [{'label': 'All', 'value': 'All'}]
        default_value = 'All'
    else:
        countries = [{'label': country, 'value': country} for country in df[df['region'] == selected_continent]['country'].unique()]
        default_value = countries[0]['value'] if countries else 'All'
    return countries, default_value

# Callback to update the world map based on selected scenarios
@app.callback(
    Output('world-map', 'figure'),
    [Input('continent-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('repartition-dropdown', 'value'),
     Input('probability-dropdown', 'value'),
     Input('temperature-dropdown', 'value'),
     Input('scope-dropdown', 'value')]
)
def update_world_map(selected_continent, selected_country, selected_repartition, selected_probability, selected_temperature, selected_scope):
    if selected_country == 'All':
        filtered_df = df[
            (df['region'] == selected_continent) &
            (df['repartition_method'] == selected_repartition) &
            (df['probability_of_reach'] == selected_probability) &
            (df['temperature'] == selected_temperature) &
            (df['scope'] == selected_scope)
        ]
    else:
        filtered_df = df[
            (df['country'] == selected_country) &
            (df['region'] == selected_continent) &
            (df['repartition_method'] == selected_repartition) &
            (df['probability_of_reach'] == selected_probability) &
            (df['temperature'] == selected_temperature) &
            (df['scope'] == selected_scope)
        ]

    fig = px.choropleth(
        filtered_df,
        locations="country",
        locationmode='country names',
        color="time_to_neutrality",
        hover_name="country",
        title="Time to Neutrality by Country"
    )
    fig.update_layout(clickmode='event+select')
    return fig

# Callback to update the line graph based on selected scenarios
@app.callback(
    Output('line-graph', 'figure'),
    [Input('continent-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('repartition-dropdown', 'value'),
     Input('probability-dropdown', 'value'),
     Input('temperature-dropdown', 'value'),
     Input('scope-dropdown', 'value')]
)
def update_line_graph(selected_continent, selected_country, selected_repartition, selected_probability, selected_temperature, selected_scope):
    if selected_country == 'All':
        filtered_df = df[
            (df['region'] == selected_continent) &
            (df['repartition_method'] == selected_repartition) &
            (df['probability_of_reach'] == selected_probability) &
            (df['temperature'] == selected_temperature) &
            (df['scope'] == selected_scope)
        ]
    else:
        filtered_df = df[
            (df['country'] == selected_country) &
            (df['region'] == selected_continent) &
            (df['repartition_method'] == selected_repartition) &
            (df['probability_of_reach'] == selected_probability) &
            (df['temperature'] == selected_temperature) &
            (df['scope'] == selected_scope)
        ]

    # Melt the DataFrame for the years
    melted_df = filtered_df.melt(id_vars=["country"], value_vars=[str(year) for year in range(2023, 2101)] + ['>2100'],
                                 var_name="Year", value_name="Emissions")

    fig = px.line(
        melted_df,
        x='Year',
        y='Emissions',
        color='country',
        title="Emissions Over Time by Country"
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
