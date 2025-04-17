import pandas as pd
import numpy as np
import country_converter as coco

def process_emissions_data():
    # Define file paths
    file_1_path = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data/carbon_budget_CO2.xlsx'
    file_2_path = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data/carbon_budget_CBA.xlsx'

    data_frames = []

    # Load and process the first file
    df1 = pd.read_excel(file_1_path)
    df1['latest_year'] = 2023
    df1.rename(columns={'emission_2023': 'emissions_latest (m tons)', 'remaining_carbon_budget': 'remaining_carbon_budget (m tons)'}, inplace=True)
    data_frames.append(df1)

    # Load and process the second file
    df2 = pd.read_excel(file_2_path)
    df2['latest_year'] = 2022
    df2.rename(columns={'emission_2022': 'emissions_latest (m tons)', 'remaining_carbon_budget': 'remaining_carbon_budget (m tons)'}, inplace=True)
    data_frames.append(df2)

    # Combine the dataframes
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Reformat population_2050
    combined_df['population_2050'] = combined_df['population_2050'].replace(r'\.', '', regex=True).astype(float)

    # Add the region column
    combined_df['region'] = coco.convert(names=combined_df['country'], to='continent', not_found=None)

    # Add the neutrality_year column
    combined_df['neutrality_year'] = combined_df['latest_year'] + combined_df['time_to_neutrality']

    # Handle NaN or infinite values in neutrality_year
    combined_df['neutrality_year'] = combined_df['neutrality_year'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Ensure neutrality_year is an integer
    combined_df['neutrality_year'] = combined_df['neutrality_year'].round().astype(int)

    # Handle NaN or infinite values in time_to_neutrality
    combined_df['time_to_neutrality'] = combined_df['time_to_neutrality'].replace([np.inf, -np.inf], np.nan).fillna(0)
    combined_df['time_to_neutrality'] = combined_df['time_to_neutrality'].round().astype(int)

    # Define the aggregation function
    def aggregate_scenarios(group):
        aggregated_data = {
            'population_2050': group['population_2050'].sum(),
            'remaining_carbon_budget (m tons)': group['remaining_carbon_budget (m tons)'].sum(),
            'time_to_neutrality': group['time_to_neutrality'].mean(),
            'neutrality_year': group['neutrality_year'].mean(),
            'emissions_latest (m tons)': group['emissions_latest (m tons)'].sum()
        }
        return pd.Series(aggregated_data)

    # Aggregate by region and curve_type, latest_year
    region_aggregated_df = combined_df.groupby(['region', 'curve_type', 'latest_year', 'repartition_method', 'source_of_budget', 'probability_of_reach', 'temperature', 'scope']).apply(aggregate_scenarios).reset_index()
    region_aggregated_df['country'] = 'All'

    # Aggregate globally and by curve_type, latest_year
    global_aggregated_df = combined_df.groupby(['curve_type', 'latest_year', 'repartition_method', 'source_of_budget', 'probability_of_reach', 'temperature', 'scope']).apply(aggregate_scenarios).reset_index()
    global_aggregated_df['country'] = 'All'
    global_aggregated_df['region'] = 'Global'

    # Combine the original data with the region and global aggregated data
    final_df = pd.concat([combined_df, region_aggregated_df, global_aggregated_df], ignore_index=True)

    return final_df

# Process the data
final_df = process_emissions_data()

# Save the result to a new Excel file
output_path = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output/emissions_trajectory_processed.xlsx'
final_df.to_excel(output_path, index=False)
print(f"Data processed and saved successfully to {output_path}")
