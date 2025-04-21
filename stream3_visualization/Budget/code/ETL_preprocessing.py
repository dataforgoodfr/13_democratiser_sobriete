import pandas as pd
import numpy as np

# Define the directory containing the data files
data_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data'
output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'

# Step 1: Extract required columns from each file
# Population data
population = pd.read_excel(f"{data_directory}/UN population projections.xlsx")
# Filter for Median PI variant only
population = population[population['Variant'] == 'Median PI']
population = population[['Region, subregion, country or area *', 2050]]

# Footprint data
footprint = pd.read_csv(f"{data_directory}/EORA_consumption_CO2_1990-2022.csv", sep=';', encoding='latin1')
footprint = footprint[['ï»¿Country', '2022']]
footprint = footprint.rename(columns={'ï»¿Country': 'Country', '2022': 2022})
# Drop any duplicates
footprint = footprint.drop_duplicates(subset=['Country'])

# Emissions data
emissions = pd.read_excel(f"{data_directory}/EDGAR- Total fossil_CO2.xlsx")
emissions = emissions[['Country', 2023]]

# Step 2: Reformat column names
population = population.rename(columns={'Region, subregion, country or area *': 'Country'})

emissions = emissions.rename(columns={2023: 'CO2_emissions_latest (m tons)'})
emissions['latest_year'] = 2023

footprint = footprint.rename(columns={2022: 'CO2_emissions_latest (m tons)'})
footprint['latest_year'] = 2022

# Step 3: Reformat values
def clean_population(value):
    if isinstance(value, str):
        # Remove spaces and replace comma with dot
        value = value.replace(' ', '').replace(',', '.')
        # Convert to float and multiply by 1000 to get actual population
        return float(value) * 1000
    return float(value) * 1000

def clean_emissions(value):
    if isinstance(value, str):
        # Replace comma with dot for decimal separator
        value = value.replace(',', '.')
    return float(value)

# Clean population numbers
population[2050] = population[2050].apply(clean_population)

# Clean footprint numbers and convert to million tons
footprint['CO2_emissions_latest (m tons)'] = footprint['CO2_emissions_latest (m tons)'].apply(clean_emissions)
footprint['CO2_emissions_latest (m tons)'] = footprint['CO2_emissions_latest (m tons)'] / 1000

# Special handling for France/Monaco and Spain/Andorra
def combine_countries(df, main_country, dependent_country):
    # Get rows for both countries
    main_row = df[df['Country'] == main_country]
    dep_row = df[df['Country'] == dependent_country]
    
    if not main_row.empty and not dep_row.empty:
        # Sum the values
        combined_values = main_row.iloc[0].copy()
        for col in df.columns:
            if col != 'Country' and col != 'latest_year':  # Don't sum the latest_year
                combined_values[col] = main_row[col].iloc[0] + dep_row[col].iloc[0]
        
        # Remove both original rows
        df = df[~df['Country'].isin([main_country, dependent_country])]
        # Add the combined row
        df = pd.concat([df, pd.DataFrame([combined_values])], ignore_index=True)
    
    return df

# Apply special handling to population and footprint data
population = combine_countries(population, 'France', 'Monaco')
population = combine_countries(population, 'Spain', 'Andorra')

footprint = combine_countries(footprint, 'France', 'Monaco')
footprint = combine_countries(footprint, 'Spain', 'Andorra')

# Ensure no duplicates in population data
population = population.drop_duplicates(subset=['Country'])

# Ensure no duplicates in footprint data
footprint = footprint.drop_duplicates(subset=['Country'])

# Step 4: Extract sample countries for verification
sample_countries = ['France', 'Germany', 'United Kingdom', 'United States', 'China', 'Spain']

print("\nSample data for verification:")
print("\nPopulation data:")
print(population[population['Country'].isin(sample_countries)])
print("\nFootprint data:")
print(footprint[footprint['Country'].isin(sample_countries)])
print("\nEmissions data:")
print(emissions[emissions['Country'].isin(sample_countries)])

# Step 5: Country matching
# Create a mapping dictionary for country names
country_mapping = {
    'France and Monaco': 'France',
    'Monaco': 'France',
    'Spain and Andorra': 'Spain',
    'Andorra': 'Spain',
    'UK': 'United Kingdom',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'USA': 'United States',
    'United States of America': 'United States',
    'UAE': 'United Arab Emirates',
    'Czech Republic': 'Czechia',
    'Macedonia': 'North Macedonia',
    'Slovak Republic': 'Slovakia',
    'Swaziland': 'Eswatini',
    'Burma': 'Myanmar',
    'Republic of Korea': 'South Korea',
    'Korea, Republic of': 'South Korea',
    'Korea, Rep.': 'South Korea',
    'Democratic People\'s Republic of Korea': 'North Korea',
    'Korea, Dem. People\'s Rep.': 'North Korea',
    'Iran (Islamic Republic of)': 'Iran',
    'Iran, Islamic Rep.': 'Iran',
    'Russian Federation': 'Russia',
    'Viet Nam': 'Vietnam',
    'Türkiye': 'Turkey',
    'Brunei Darussalam': 'Brunei',
    'Congo, Democratic Republic of the': 'Democratic Republic of the Congo',
    'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
    'Congo': 'Republic of the Congo',
    'Congo, Rep.': 'Republic of the Congo',
    'Lao People\'s Democratic Republic': 'Laos',
    'Lao PDR': 'Laos',
    'Syrian Arab Republic': 'Syria',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Tanzania, United Republic of': 'Tanzania'
}

# Apply country mapping
population['Country'] = population['Country'].map(lambda x: country_mapping.get(x, x))
footprint['Country'] = footprint['Country'].map(lambda x: country_mapping.get(x, x))
emissions['Country'] = emissions['Country'].map(lambda x: country_mapping.get(x, x))

# Step 6: Create final dataset
# Prepare territory-based emissions
territory_based = emissions.copy()
territory_based['scope'] = 'Territory_based'

# Prepare consumption-based emissions
consumption_based = footprint.copy()
consumption_based['scope'] = 'Consumption_based'

# Merge with population data
territory_based = territory_based.merge(population, on='Country', how='left')
consumption_based = consumption_based.merge(population, on='Country', how='left')

# Rename population column
territory_based = territory_based.rename(columns={2050: 'population_2050'})
consumption_based = consumption_based.rename(columns={2050: 'population_2050'})

# Combine both datasets
final_dataset = pd.concat([territory_based, consumption_based], ignore_index=True)

# Drop any duplicates
final_dataset = final_dataset.drop_duplicates()

# Save the final dataset
final_dataset.to_csv(f"{output_directory}/Latest_emissions.csv", index=False)

# Print sample data for specific countries
print("\nSample data for key countries:")
print("=" * 120)
# Select specific countries
selected_countries = ['France', 'Russia', 'United Kingdom', 'United States', 'China', 'Turkey']
# Filter and sort the data
sample_data = final_dataset[final_dataset['Country'].isin(selected_countries)]
sample_data = sample_data.sort_values(['Country', 'scope'])
# Format the data for better display
pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 5)
# Print the data in table format
print(sample_data[['Country', 'CO2_emissions_latest (m tons)', 'latest_year', 'population_2050', 'scope']].to_string(index=False))
print("=" * 120)
