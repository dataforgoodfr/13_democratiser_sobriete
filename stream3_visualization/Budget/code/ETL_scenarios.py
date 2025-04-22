import pandas as pd
import numpy as np

# Define the directory containing the data files
output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'

# Load the preprocessed data
combined_df = pd.read_csv(f"{output_directory}/combined_data.csv")

# Create base dataframe with required columns
def create_base_dataframe(df):
    # Get unique countries and their regions
    base_df = df[['ISO3', 'Country', 'Region']].drop_duplicates()
    
    # Get population data for 2050
    pop_2050 = df[
        (df['Measure'] == 'Population') & 
        (df['Year'] == 2050)
    ][['ISO3', 'Value']].rename(columns={'Value': 'Population_2050'})
    
    # Get world population for 2050
    world_pop_2050 = df[
        (df['Measure'] == 'Population') & 
        (df['Year'] == 2050) & 
        (df['ISO3'] == 'WLD')
    ]['Value'].iloc[0]
    
    # Merge population data
    base_df = base_df.merge(pop_2050, on='ISO3', how='left')
    
    # Calculate share of total population
    base_df['Share_of_total_population_2050'] = base_df['Population_2050'] / world_pop_2050
    
    # Get latest year and emissions for each scope
    emission_scopes = ['CO2_emissions_Mt', 'Consumption_CO2_emissions_Mt']
    for scope in emission_scopes:
        # Filter data for this scope
        scope_data = df[df['Measure'] == scope]
        
        # Get latest year for each ISO3
        latest_years = scope_data.groupby('ISO3')['Year'].max().reset_index()
        latest_years.columns = ['ISO3', f'Latest_year_{scope}']
        
        # Get latest emissions for each ISO3
        latest_emissions = pd.merge(
            scope_data, 
            latest_years, 
            left_on=['ISO3', 'Year'], 
            right_on=['ISO3', f'Latest_year_{scope}']
        )[['ISO3', 'Value']].rename(columns={'Value': f'Latest_emissions_{scope}'})
        
        # Merge with base dataframe
        base_df = base_df.merge(latest_years, on='ISO3', how='left')
        base_df = base_df.merge(latest_emissions, on='ISO3', how='left')
    
    return base_df

# Create the base dataframe
base_df = create_base_dataframe(combined_df)

# Create all scenario combinations
scenarios = []
for _, row in base_df.iterrows():
    for emissions_scope in ['CO2_emissions_Mt', 'Consumption_CO2_emissions_Mt']:
        for warming_scenario in ['1.5°C', '2°C']:
            for probability in ['33%', '50%', '67%']:
                for budget_source in ['Lamboll', 'Foster']:
                    for distribution in ['Equality', 'Responsibility', 'Current_target']:
                        scenario = {
                            'ISO3': row['ISO3'],
                            'Country': row['Country'],
                            'Region': row['Region'],
                            'Population_2050': row['Population_2050'],
                            'Share_of_total_population_2050': row['Share_of_total_population_2050'],
                            'Emissions_scope': emissions_scope,
                            'Latest_year': row[f'Latest_year_{emissions_scope}'],
                            'Latest_emissions': row[f'Latest_emissions_{emissions_scope}'],
                            'Warming_scenario': warming_scenario,
                            'Probability_of_reach': probability,
                            'Budget_source': budget_source,
                            'Budget_distribution_scenario': distribution
                        }
                        scenarios.append(scenario)

# Convert to DataFrame
scenarios_df = pd.DataFrame(scenarios)

# Print the first 50 rows to verify
print("\nFirst 50 rows of the scenarios dataframe:")
print(scenarios_df.head(50).to_string())

# Save to CSV
scenarios_df.to_csv(f"{output_directory}/Budget_scenarios.csv", index=False)
print(f"\nScenarios saved to {output_directory}/Budget_scenarios.csv")
