import pandas as pd
import numpy as np
import country_converter as coco

try:
    # Load the Excel files
    file_1_path = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data/carbon_budget_CO2.xlsx'
    file_2_path = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data/carbon_budget_CBA.xlsx'

    df1 = pd.read_excel(file_1_path)
    df2 = pd.read_excel(file_2_path)

    # Add a 'latest_year' column to each dataframe
    df1['latest_year'] = 2023
    df2['latest_year'] = 2022

    # Rename emission columns to a common name
    df1.rename(columns={'emission_2023': 'emissions_latest'}, inplace=True)
    df2.rename(columns={'emission_2022': 'emissions_latest'}, inplace=True)

    # Append the two dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Use country_converter to map countries to continents
    combined_df['region'] = coco.convert(names=combined_df['country'], to='continent', not_found=None)

    # Handle NaN or infinite values in 'time_to_neutrality'
    combined_df['time_to_neutrality'] = combined_df['time_to_neutrality'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Ensure 'time_to_neutrality' is of integer type
    combined_df['time_to_neutrality'] = combined_df['time_to_neutrality'].round().astype(int)

    # Recalculate 'neutrality_year' based on 'latest_year' + 'time_to_neutrality'
    combined_df['neutrality_year'] = combined_df['latest_year'] + combined_df['time_to_neutrality']

    # Function to calculate linear decrease
    def linear_decrease(latest_emissions, latest_year, neutrality_year):
        emissions_per_year = {}
        for year in range(latest_year + 1, min(neutrality_year, 2101)):  # Start from latest_year + 1
            if year == neutrality_year:
                emissions = 0
            else:
                emissions = latest_emissions * (neutrality_year - year) / (neutrality_year - latest_year)
            emissions_per_year[year] = emissions
        if neutrality_year > 2100:
            emissions_per_year['>2100'] = 0  # Emissions will be 0 by 2100
        return emissions_per_year

    # Function to calculate exponential decrease
    def exponential_decrease(latest_emissions, latest_year, neutrality_year):
        emissions_per_year = {}
        if neutrality_year == latest_year:
            # Handle division by zero
            emissions_per_year[latest_year] = latest_emissions
            emissions_per_year['>2100'] = 0
            return emissions_per_year
        decay_rate = np.log(0.01) / (neutrality_year - latest_year)  # Decay to 1% of original value
        for year in range(latest_year + 1, min(neutrality_year, 2101)):  # Start from latest_year + 1
            emissions = latest_emissions * np.exp(decay_rate * (year - latest_year))
            emissions_per_year[year] = emissions
        if neutrality_year > 2100:
            emissions_per_year['>2100'] = latest_emissions * np.exp(decay_rate * (2100 - latest_year))
        return emissions_per_year

    # Create a list to store the results
    emissions_data = []

    # Apply the functions based on curve_type
    for index, row in combined_df.iterrows():
        country = row['country']
        latest_emissions = row['emissions_latest']
        latest_year = row['latest_year']
        neutrality_year = row['neutrality_year']
        curve_type = row['curve_type']
        region = row['region']

        # Include all original columns except '2022'
        original_columns = {col: row[col] for col in row.index if col not in ['emissions_latest', 'latest_year', 'neutrality_year', 'curve_type', 'region', '2022']}

        # Initialize emissions data with the latest year's emissions
        emissions_per_year = {2023: latest_emissions if latest_year == 2023 else np.nan}

        if curve_type == 'linear':
            emissions_per_year.update(linear_decrease(latest_emissions, latest_year, neutrality_year))
        elif curve_type == 'exponential':
            emissions_per_year.update(exponential_decrease(latest_emissions, latest_year, neutrality_year))
        else:
            continue

        # Add the emissions data for each year as separate columns
        yearly_data = {**original_columns, 'emissions_latest': latest_emissions, 'region': region, 'curve_type': curve_type}
        yearly_data.update(emissions_per_year)
        emissions_data.append(yearly_data)

    # Create a DataFrame from the emissions data
    emissions_df = pd.DataFrame(emissions_data)

    # Print the first 100 rows and the shape of the DataFrame
    print(emissions_df.head(100))
    print("Shape of the DataFrame:", emissions_df.shape)

    # Save the DataFrame to a CSV file in the specified directory
    output_path = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output/emissions_trajectory.csv'
    emissions_df.to_csv(output_path, index=False)
    print(f"Data processed and saved successfully to {output_path}.")

except FileNotFoundError as e:
    print(f"Error: {e}. Please check the file paths.")
except ImportError as e:
    print(f"Error: {e}. Please install the required libraries.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
