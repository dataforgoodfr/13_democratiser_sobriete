import pandas as pd
import numpy as np
from datetime import datetime

# Define the directory containing the data files
output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'
data_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data'

# Define global carbon budgets from the beginning of 2025 (in million tons of CO2)
# Source: IPCC AR6 SYR, Table 2.1, values converted from GtCO2 to MtCO2
BUDGET_GLOBAL_2C = {"33%": 1310000, "50%": 1050000, "67%": 870000, "83%": 690000}
BUDGET_GLOBAL_15C = {"33%": 200000, "50%": 130000, "67%": 80000, "83%": 30000}

def get_global_budget(warming_scenario, probability):
    """Get the global carbon budget based on scenario parameters."""
    if warming_scenario == '2°C':
        return BUDGET_GLOBAL_2C[probability]
    else:  # 1.5°C
        return BUDGET_GLOBAL_15C[probability]

def penalty_func_2(x):
    """Quadratic penalty function."""
    return x * x

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
                # Map ISO3 to ISO2 for non-EU countries
                iso2_code = iso_mapping[iso_mapping['ISO3'] == iso]['ISO2'].values
                if len(iso2_code) > 0:
                    target_mapping[iso2_code[0]] = int(target_year)
                else:
                    print(f"Warning: ISO3 code {iso} not found in ISO mapping.")

    return target_mapping

# Load the preprocessed data and current targets
combined_df = pd.read_csv(f"{output_directory}/combined_data.csv")
current_targets = load_current_targets()

# Ensure "NA" is treated as a valid ISO2 code
combined_df['ISO2'] = combined_df['ISO2'].astype(str)

# Explicitly set "NA" for Namibia
combined_df.loc[combined_df['Country'] == 'Namibia', 'ISO2'] = 'NA'

# Print to verify
print("Combined DataFrame:")
print(combined_df[['ISO2', 'Country', 'Region']].head())

# Create base dataframe with required columns
def create_base_dataframe(df):
    """Create a base dataframe with pre-calculated shares for scenarios."""
    # Get unique countries and their regions
    base_df = df[['ISO2', 'Country', 'Region']].drop_duplicates()

    # Calculate share of cumulative population from 1990 to 2050 (for Responsibility scenario)
    cum_pop_2050_df = df[
        (df['Emissions_scope'] == 'Territory') & (df['Year'] == 2050)
    ][['ISO2', 'Cumulative_population']].rename(columns={'Cumulative_population': 'Cumulative_population_2050'})
    
    world_cum_pop_2050 = cum_pop_2050_df[cum_pop_2050_df['ISO2'] == 'WLD']['Cumulative_population_2050'].iloc[0]
    
    base_df = base_df.merge(cum_pop_2050_df, on='ISO2', how='left')
    base_df['Share_of_cumulative_population_1990_to_2050'] = base_df['Cumulative_population_2050'] / world_cum_pop_2050
    base_df.drop(columns=['Cumulative_population_2050'], inplace=True)

    # Define emission_scopes within the function
    emission_scopes = ['Territory', 'Consumption']

    for scope in emission_scopes:
        scope_data = df[
            (df['Emissions_scope'] == scope) &
            (df['Annual_CO2_emissions_Mt'].notna()) &
            (df['Annual_CO2_emissions_Mt'] != 0) &
            (df['Year'] < 2050)
        ]

        latest_years = scope_data.groupby('ISO2')['Year'].max().reset_index()
        latest_years.columns = ['ISO2', f'Latest_year_{scope}']
        latest_years[f'Latest_year_{scope}'] = latest_years[f'Latest_year_{scope}'].astype(int)

        latest_data = pd.merge(
            scope_data,
            latest_years,
            left_on=['ISO2', 'Year'],
            right_on=['ISO2', f'Latest_year_{scope}']
        )[['ISO2', 'Annual_CO2_emissions_Mt', 'Cumulative_CO2_emissions_Mt', 'Cumulative_population', 'Emissions_per_capita_ton', 'GDP_PPP', 'Population', 'share_of_capacity']].rename(
            columns={
                'Annual_CO2_emissions_Mt': f'Latest_annual_CO2_emissions_Mt_{scope}',
                'Cumulative_CO2_emissions_Mt': f'Latest_cumulative_CO2_emissions_Mt_{scope}',
                'Cumulative_population': f'Latest_cumulative_population_{scope}',
                'Emissions_per_capita_ton': f'Latest_emissions_per_capita_t_{scope}',
                'GDP_PPP': f'Latest_GDP_PPP_{scope}',
                'Population': f'Latest_population_{scope}',
                'share_of_capacity': f'share_of_capacity_{scope}'
            }
        )

        base_df = base_df.merge(latest_years, on='ISO2', how='left')
        base_df = base_df.merge(latest_data, on='ISO2', how='left')

        # --- Share of Cumulative Population (Equality) ---
        world_cumulative_pop = base_df[
            base_df['ISO2'] == 'WLD'
        ][f'Latest_cumulative_population_{scope}'].iloc[0]
        base_df[f'Share_of_cumulative_population_1990_to_Latest_{scope}'] = base_df[f'Latest_cumulative_population_{scope}'] / world_cumulative_pop

        # --- Share of Cumulative Emissions (Responsibility) ---
        world_cumulative_emissions = base_df[
            base_df['ISO2'] == 'WLD'
        ][f'Latest_cumulative_CO2_emissions_Mt_{scope}'].iloc[0]
        base_df[f'Share_of_cumulative_emissions_{scope}'] = base_df[f'Latest_cumulative_CO2_emissions_Mt_{scope}'] / world_cumulative_emissions

    return base_df, emission_scopes

# Create the base dataframe
base_df, emission_scopes = create_base_dataframe(combined_df)

# Print verification for the latest years used in the 'Equality' scenario
print("\nVerifying latest years used for 'Equality' scenario calculations:")
print(base_df[['ISO2', 'Country', 'Latest_year_Territory', 'Latest_year_Consumption']].head())

# Print to verify
print("\nBase DataFrame:")
print(base_df[['ISO2', 'Country', 'Region', f'Share_of_cumulative_emissions_{emission_scopes[0]}']].head())

# Create all scenario combinations
scenarios = []
current_year = datetime.now().year
for _, row in base_df.iterrows():
    for emissions_scope in emission_scopes:
        for warming_scenario in ['1.5°C', '2°C']:
            for probability in ['33%', '50%', '67%', '83%']:
                for distribution in ['Equality', 'Responsibility', 'Current_target', 'Capacity']:
                    # Calculate country carbon budget based on distribution scenario
                    global_budget = get_global_budget(warming_scenario, probability)
                    if distribution == 'Equality':
                        country_budget = global_budget * row[f'Share_of_cumulative_population_1990_to_Latest_{emissions_scope}']
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
                        country_budget = (total_available * row['Share_of_cumulative_population_1990_to_2050']) - country_cumulative
                    elif distribution == 'Capacity':
                        country_budget = global_budget * row[f'share_of_capacity_{emissions_scope}']
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
                            if pd.notna(latest_annual) and latest_annual > 0:
                                country_budget = (years_to_neutrality * latest_annual) / 2
                            else:
                                country_budget = None
                        else:
                            years_to_neutrality = "N/A"
                            neutrality_year = "N/A"
                            country_budget = None
                    # using integers for buckets to ensure it can be visualized on the map
                    elif pd.notna(country_budget) and pd.notna(latest_annual) and latest_annual > 0:
                        years_to_neutrality = int(round(2 * country_budget / latest_annual))
                        if years_to_neutrality + latest_year > 2100:
                            neutrality_year = 2100
                        else:
                            neutrality_year = int(round(latest_year + years_to_neutrality))

                    else:
                        years_to_neutrality = "N/A"
                        neutrality_year = "N/A"

                    # Ensure years_to_neutrality_from_latest_available and neutrality_year are integers or "N/A"
                    if isinstance(years_to_neutrality, (int, float)) and pd.notna(years_to_neutrality):
                        years_to_neutrality = int(years_to_neutrality)
                    else:
                        years_to_neutrality = "N/A"
                    if isinstance(neutrality_year, (int, float)) and pd.notna(neutrality_year):
                        neutrality_year = int(neutrality_year)
                    else:
                        neutrality_year = "N/A"

                    # Calculate Years_to_neutrality_from_today
                    if isinstance(neutrality_year, int):
                        years_to_neutrality_from_today = neutrality_year - current_year
                    else:
                        years_to_neutrality_from_today = "N/A"

                    scenario = {
                        'ISO2': row['ISO2'],
                        'Country': row['Country'],
                        'Region': row['Region'],
                        'Share_of_cumulative_population_1990_to_2050': row['Share_of_cumulative_population_1990_to_2050'],
                        'share_of_capacity': row[f'share_of_capacity_{emissions_scope}'],
                        'Emissions_scope': emissions_scope,
                        'Latest_year': latest_year,
                        'Latest_annual_CO2_emissions_Mt': latest_annual,
                        'Latest_cumulative_CO2_emissions_Mt': row[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}'],
                        'Latest_cumulative_population': row[f'Latest_cumulative_population_{emissions_scope}'],
                        'Latest_emissions_per_capita_t': row[f'Latest_emissions_per_capita_t_{emissions_scope}'],
                        'Share_of_cumulative_population_1990_to_Latest': row[f'Share_of_cumulative_population_1990_to_Latest_{emissions_scope}'],
                        'Share_of_cumulative_emissions': row[f'Share_of_cumulative_emissions_{emissions_scope}'],
                        'Warming_scenario': warming_scenario,
                        'Probability_of_reach': probability,
                        'Budget_distribution_scenario': distribution,
                        'Global_Carbon_budget': global_budget,
                        'Country_carbon_budget': country_budget,
                        'Years_to_neutrality_from_latest_available': years_to_neutrality,
                        'Neutrality_year': neutrality_year,
                        'Years_to_neutrality_from_today': years_to_neutrality_from_today
                    }
                    scenarios.append(scenario)

# After creating the scenarios list, create two separate dataframes
scenarios_df = pd.DataFrame(scenarios)

# 1. Create scenario parameters dataframe (one row per unique scenario)
scenario_params = scenarios_df[[
    'ISO2', 'Country', 'Region', 'Emissions_scope',
    'Warming_scenario', 'Probability_of_reach',
    'Budget_distribution_scenario', 'Years_to_neutrality_from_latest_available', 'Years_to_neutrality_from_today', 'Neutrality_year',
    'Latest_year', 'Latest_annual_CO2_emissions_Mt',
    'Latest_cumulative_CO2_emissions_Mt','Latest_emissions_per_capita_t', 'Latest_cumulative_population',
    'Share_of_cumulative_population_1990_to_Latest',
    'Share_of_cumulative_population_1990_to_2050',
    'share_of_capacity', 'Global_Carbon_budget',
    'Country_carbon_budget', 'Share_of_cumulative_emissions'
]].drop_duplicates()

# Filter out rows where neutrality could not be calculated
original_rows = len(scenario_params)
scenario_params = scenario_params[scenario_params['Years_to_neutrality_from_latest_available'] != "N/A"].copy()
print(f"\nFiltered out {original_rows - len(scenario_params)} rows with 'N/A' neutrality years from scenario parameters.")

# Create a scenario_id for each unique combination
scenario_params['scenario_id'] = range(1, len(scenario_params) + 1)

# Ensure NA, TR, and US are included
required_isos = ['NA', 'TR', 'US']
for iso in required_isos:
    if iso not in scenario_params['ISO2'].values:
        print(f"Warning: ISO2 code {iso} is missing in scenario parameters.")

# 2. Create forecast data dataframe
forecast_data = []
for _, row in scenario_params.iterrows():
    # Skip if no latest year or emissions data
    if pd.isna(row['Latest_year']) or pd.isna(row['Latest_annual_CO2_emissions_Mt']):
        continue

    # Convert years to integers
    latest_year = int(row['Latest_year'])

    # Handle different cases for forecast
    if (row['Years_to_neutrality_from_latest_available'] == "N/A" or
        row['Years_to_neutrality_from_latest_available'] is None or
        (isinstance(row['Years_to_neutrality_from_latest_available'], (int, float)) and row['Years_to_neutrality_from_latest_available'] <= 0)):
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

# Add Data_type column to the forecast_df dataframe
forecast_df['Data_type'] = 'Forecast'

# Save both files
scenario_params.to_csv(f"{output_directory}/scenario_parameters.csv", index=False)
forecast_df.to_csv(f"{output_directory}/forecast_data.csv", index=False)

print(f"\nScenario parameters saved to {output_directory}/scenario_parameters.csv")
print(f"Forecast data saved to {output_directory}/forecast_data.csv")
