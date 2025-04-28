import pandas as pd
import numpy as np

"""
This script creates the master dataframe of all scenarios
We use the data available from 1970 to 2023 for territory emissions and from 1990 to 2022 for consumption emissions
We use the population data available from 1970 to 2050
"""

# Define the directory containing the data files
output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'
data_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data'

# Define global carbon budgets
BUDGET_GLOBAL_lamboll_2C = {"33%": 1603000, "50%": 1219000, "67%": 944000}
BUDGET_GLOBAL_foster_2C = {"33%": 1450000, "50%": 1150000, "67%": 950000}
BUDGET_GLOBAL_lamboll_15C = {"33%": 480000, "50%": 247000, "67%": 60000}
BUDGET_GLOBAL_foster_15C = {"33%": 300000, "50%": 250000, "67%": 150000}

def get_global_budget(warming_scenario, probability, budget_source):
    """Get the global carbon budget based on scenario parameters."""
    if warming_scenario == '2°C':
        if budget_source == 'Lamboll':
            return BUDGET_GLOBAL_lamboll_2C[probability]
        else:  # Foster
            return BUDGET_GLOBAL_foster_2C[probability]
    else:  # 1.5°C
        if budget_source == 'Lamboll':
            return BUDGET_GLOBAL_lamboll_15C[probability]
        else:  # Foster
            return BUDGET_GLOBAL_foster_15C[probability]

def load_current_targets():
    """Load and process current target years."""
    # Load the current targets file
    targets = pd.read_excel(f"{data_directory}/2025-04-21_Full file_Current carbon neutrality timeline per with Country ISO code.xlsx")

    # Load the ISO codes mapping file
    iso_mapping = pd.read_excel(f"{data_directory}/28-04-2025_ISO_Codes_Mapping.xlsx")
    iso_mapping.rename(columns={'Alpha-2 code': 'ISO2', 'Alpha-3 code': 'ISO3'}, inplace=True)

    # Get EU countries mapping
    eu_mapping = pd.read_excel(f"{data_directory}/2024-04-21_IPCC Regional Breakdown_ISO Country Code.xlsx",
                             sheet_name="G20_EU_Countries ",
                             header=0)

    # Merge ISO2 codes into the EU mapping
    eu_mapping = eu_mapping.merge(iso_mapping, on='ISO3', how='left')

    # Get EU countries with ISO2 codes
    eu_countries = eu_mapping[eu_mapping['EU_country'] == 'Yes']['ISO2'].tolist()

    # Create target mapping
    target_mapping = {}

    # Process each row
    for _, row in targets.iterrows():
        iso = row['ISO']
        target_year = row['Target year']

        if pd.notna(iso) and pd.notna(target_year):
            # Handle special case for NGA
            if iso == 'NGA' and target_year == '2050-2070':
                target_year = 2070
            elif isinstance(target_year, str):
                try:
                    target_year = int(target_year)
                except ValueError:
                    continue  # Skip if can't convert to integer

            if iso == 'EU27':
                # Add all EU countries with the same target year
                for eu_iso in eu_countries:
                    target_mapping[eu_iso] = int(target_year)
            else:
                target_mapping[iso] = int(target_year)

    return target_mapping

# Load the preprocessed data and current targets
combined_df = pd.read_csv(f"{output_directory}/combined_data.csv")
current_targets = load_current_targets()

# Create base dataframe with required columns
def create_base_dataframe(df):
    # Get unique countries and their regions
    base_df = df[['ISO2', 'Country', 'Region']].drop_duplicates()

    # Get population data for 2050
    pop_2050 = df[
        (df['Emissions_scope'] == 'Territory') &  # Using Territory scope since population is the same
        (df['Year'] == 2050)
    ][['ISO2', 'Population']].rename(columns={'Population': 'Population_2050'})

    # Get world population for 2050
    world_pop_2050 = df[
        (df['Emissions_scope'] == 'Territory') &  # Using Territory scope since population is the same
        (df['Year'] == 2050) &
        (df['ISO2'] == 'WLD')
    ]['Population'].iloc[0]

    # Merge population data
    base_df = base_df.merge(pop_2050, on='ISO2', how='left')

    # Calculate share of total population
    base_df['Share_of_total_population_2050'] = base_df['Population_2050'] / world_pop_2050

    # Get latest year and emissions for each scope
    emission_scopes = ['Territory', 'Consumption']
    for scope in emission_scopes:
        # Filter data for this scope and where Annual_CO2_emissions_Mt is not null and not 0
        scope_data = df[
            (df['Emissions_scope'] == scope) &
            (df['Annual_CO2_emissions_Mt'].notna()) &
            (df['Annual_CO2_emissions_Mt'] != 0) &
            (df['Year'] != 2050)  # Exclude 2050 from latest year calculation
        ]

        # For regular countries, get latest year with emissions data
        country_latest_years = scope_data[
            ~scope_data['ISO2'].isin(['WLD', 'REG', 'EU', 'G20'])
        ].groupby('ISO2')['Year'].max().reset_index()

        # For aggregates, get latest year with emissions data
        aggregate_latest_years = scope_data[
            scope_data['ISO2'].isin(['WLD', 'REG', 'EU', 'G20'])
        ].groupby('ISO2')['Year'].max().reset_index()

        # Combine the latest years
        latest_years = pd.concat([country_latest_years, aggregate_latest_years])
        latest_years.columns = ['ISO2', f'Latest_year_{scope}']
        # Convert years to integers
        latest_years[f'Latest_year_{scope}'] = latest_years[f'Latest_year_{scope}'].astype(int)

        # Get latest emissions and population for each ISO2
        latest_data = pd.merge(
            scope_data,
            latest_years,
            left_on=['ISO2', 'Year'],
            right_on=['ISO2', f'Latest_year_{scope}']
        )[['ISO2', 'Annual_CO2_emissions_Mt', 'Cumulative_CO2_emissions_Mt', 'Cumulative_population']].rename(
            columns={
                'Annual_CO2_emissions_Mt': f'Latest_annual_CO2_emissions_Mt_{scope}',
                'Cumulative_CO2_emissions_Mt': f'Latest_cumulative_CO2_emissions_Mt_{scope}',
                'Cumulative_population': f'Latest_cumulative_population_{scope}'
            }
        )

        # Merge with base dataframe
        base_df = base_df.merge(latest_years, on='ISO2', how='left')
        base_df = base_df.merge(latest_data, on='ISO2', how='left')

        # Calculate share of cumulative population for this scope
        world_cumulative_pop = base_df[
            base_df['ISO2'] == 'WLD'
        ][f'Latest_cumulative_population_{scope}'].iloc[0]

        base_df[f'Share_of_cumulative_population_{scope}'] = base_df[f'Latest_cumulative_population_{scope}'] / world_cumulative_pop

    return base_df

# Create the base dataframe
base_df = create_base_dataframe(combined_df)

# Create all scenario combinations
scenarios = []
for _, row in base_df.iterrows():
    for emissions_scope in ['Territory', 'Consumption']:
        for warming_scenario in ['1.5°C', '2°C']:
            for probability in ['33%', '50%', '67%']:
                for budget_source in ['Lamboll', 'Foster']:
                    for distribution in ['Equality', 'Responsibility', 'Current_target']:
                        # Calculate country carbon budget based on distribution scenario
                        global_budget = get_global_budget(warming_scenario, probability, budget_source)
                        if distribution == 'Equality':
                            country_budget = global_budget * row['Share_of_total_population_2050']
                        elif distribution == 'Responsibility':
                            # Get world's latest cumulative emissions
                            world_cumulative = base_df[
                                (base_df['ISO2'] == 'WLD') &
                                (base_df[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}'].notna())
                            ][f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}'].iloc[0]

                            # Calculate total available budget (global + world's historical emissions)
                            total_available = global_budget + world_cumulative

                            # Calculate country's share and subtract its historical emissions
                            country_cumulative = row[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}']
                            country_budget = (total_available * row[f'Share_of_cumulative_population_{emissions_scope}']) - country_cumulative
                        else:  # Current_target
                            country_budget = None

                        # Calculate years to neutrality and neutrality year
                        latest_annual = row[f'Latest_annual_CO2_emissions_Mt_{emissions_scope}']
                        latest_year = row[f'Latest_year_{emissions_scope}']

                        if distribution == 'Current_target':
                            # Get target year from current targets mapping
                            neutrality_year = current_targets.get(row['ISO2'])
                            if neutrality_year is not None:
                                years_to_neutrality = neutrality_year - latest_year
                                # Back-calculate Country_carbon_budget based on years_to_neutrality
                                if latest_annual > 0:
                                    country_budget = (years_to_neutrality * latest_annual) / 2
                                else:
                                    country_budget = None
                            else:
                                years_to_neutrality = "N/A"
                                neutrality_year = "N/A"
                                country_budget = None
                        elif country_budget is not None and latest_annual > 0:
                            years_to_neutrality = int(round(2 * country_budget / latest_annual))
                            if years_to_neutrality + latest_year > 2100:
                                neutrality_year = '>2100'
                            elif  years_to_neutrality + latest_year < 2023:
                                neutrality_year = '<2023'
                            else:
                                neutrality_year = int(round(latest_year + years_to_neutrality))

                        else:
                            years_to_neutrality = None
                            neutrality_year = None

                        scenario = {
                            'ISO2': row['ISO2'],
                            'Country': row['Country'],
                            'Region': row['Region'],
                            'Population_2050': row['Population_2050'],
                            'Share_of_total_population_2050': row['Share_of_total_population_2050'],
                            'Emissions_scope': emissions_scope,
                            'Latest_year': latest_year,
                            'Latest_annual_CO2_emissions_Mt': latest_annual,
                            'Latest_cumulative_CO2_emissions_Mt': row[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}'],
                            'Latest_cumulative_population': row[f'Latest_cumulative_population_{emissions_scope}'],
                            'Share_of_cumulative_population': row[f'Share_of_cumulative_population_{emissions_scope}'],
                            'Warming_scenario': warming_scenario,
                            'Probability_of_reach': probability,
                            'Budget_source': budget_source,
                            'Budget_distribution_scenario': distribution,
                            'Global_Carbon_budget': global_budget,
                            'Country_carbon_budget': country_budget,
                            'Years_to_neutrality': years_to_neutrality,
                            'Neutrality_year': neutrality_year
                        }
                        scenarios.append(scenario)

# After creating the scenarios list, create two separate dataframes
scenarios_df = pd.DataFrame(scenarios)

# 1. Create scenario parameters dataframe (one row per unique scenario)
scenario_params = scenarios_df[[
    'ISO2', 'Country', 'Region', 'Emissions_scope',
    'Warming_scenario', 'Probability_of_reach', 'Budget_source',
    'Budget_distribution_scenario', 'Years_to_neutrality', 'Neutrality_year',
    'Latest_year', 'Latest_annual_CO2_emissions_Mt',
    'Latest_cumulative_CO2_emissions_Mt', 'Latest_cumulative_population',
    'Share_of_cumulative_population', 'Population_2050',
    'Share_of_total_population_2050', 'Global_Carbon_budget',
    'Country_carbon_budget'
]].drop_duplicates()

# Create a scenario_id for each unique combination
scenario_params['scenario_id'] = range(1, len(scenario_params) + 1)

# 2. Create forecast data dataframe
forecast_data = []
for _, row in scenario_params.iterrows():
    # Skip if no latest year or emissions data
    if pd.isna(row['Latest_year']) or pd.isna(row['Latest_annual_CO2_emissions_Mt']):
        continue

    # Convert years to integers
    latest_year = int(row['Latest_year'])

    # Handle different cases for forecast
    if (row['Years_to_neutrality'] == "N/A" or
        row['Years_to_neutrality'] is None or
        (isinstance(row['Years_to_neutrality'], (int, float)) and row['Years_to_neutrality'] <= 0)):
        # For N/A or negative years_to_neutrality, drop to zero immediately
        forecast_years = pd.DataFrame({
            'Year': [latest_year + 1],
            'Forecasted_emissions_Mt': [0]
        })
    else:
        # Normal case: linear decrease to zero
        if row['Neutrality_year'] == '>2100':
            neutrality_year = 2100
        else:
            neutrality_year = int(row['Neutrality_year'])
        forecast_years = pd.DataFrame({
            'Year': range(latest_year + 1, neutrality_year + 1)
        })

        # Calculate forecasted emissions
        slope = -row['Latest_annual_CO2_emissions_Mt'] / (neutrality_year - latest_year)
        forecast_years['Forecasted_emissions_Mt'] = [
            max(0, row['Latest_annual_CO2_emissions_Mt'] + slope * (year - latest_year))
            for year in forecast_years['Year']
        ]

    # Add forecast data with scenario_id reference
    for _, year_row in forecast_years.iterrows():
        forecast_data.append({
            'scenario_id': row['scenario_id'],
            'Year': year_row['Year'],
            'Forecasted_emissions_Mt': year_row['Forecasted_emissions_Mt']
        })

# Convert forecast data to DataFrame
forecast_df = pd.DataFrame(forecast_data)

# Save both files
scenario_params.to_csv(f"{output_directory}/scenario_parameters.csv", index=False)
forecast_df.to_csv(f"{output_directory}/forecast_data.csv", index=False)

print(f"\nScenario parameters saved to {output_directory}/scenario_parameters.csv")
print(f"Forecast data saved to {output_directory}/forecast_data.csv")
