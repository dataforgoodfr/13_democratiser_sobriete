import pandas as pd
import numpy as np

# Define the directory containing the data files
data_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data'
output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'

# Load the datasets
population_file = f"{data_directory}/UN population projections_converted.csv"
emissions_file = f"{data_directory}/EDGAR- Total fossil_CO2.csv"
consumption_file = f"{data_directory}/EORA_consumption_CO2_1990-2022.csv"

# Read the files
population = pd.read_csv(population_file, sep=',', encoding='utf-8')
emissions = pd.read_csv(emissions_file, sep=';', encoding='latin1')
consumption = pd.read_csv(consumption_file, sep=';', encoding='latin1')

# Print the column names to debug
print("Population columns:", population.columns.tolist())

# Rename the population country column to a simpler name
population.rename(columns={'Region, subregion, country or area *': 'Country'}, inplace=True)

# Print the column names after renaming
print("Population columns after renaming:", population.columns.tolist())

# Rename the consumption country column to remove the BOM
consumption.rename(columns={consumption.columns[0]: 'Country'}, inplace=True)

# Print column names for all DataFrames to verify
print("Emissions columns:", emissions.columns.tolist())
print("Consumption columns:", consumption.columns.tolist())

# Create an expanded mapping dictionary that includes variations of country names
country_mapping = {
    # Original mappings
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Brunei Darussalam': 'Brunei',
    'China, Hong Kong SAR': 'Hong Kong',
    'China, Macao SAR': 'Macao',
    'China, Taiwan Province of China': 'Taiwan',
    "Côte d'Ivoire": 'Côte d\x92Ivoire',
    "Czechia": 'Czech Republic',
    "Dem. People's Republic of Korea": 'North Korea',
    'Falkland Islands (Malvinas)': 'Falkland Islands',
    'Faroe Islands': 'Faroes',
    'France': 'France and Monaco',
    'Gambia': 'The Gambia',
    'Iran (Islamic Republic of)': 'Iran',
    'Israel': 'Israel and Palestine, State of',
    'Italy': 'Italy, San Marino and the Holy See',
    "Lao People's Democratic Republic": 'Laos',
    'Liechtenstein': 'Switzerland and Liechtenstein',
    'Monaco': 'France and Monaco',
    'Montenegro': 'Serbia and Montenegro',
    'Myanmar': 'Myanmar/Burma',
    'Republic of Korea': 'South Korea',
    'Republic of Moldova': 'Moldova',
    'Russian Federation': 'Russia',
    'San Marino': 'Italy, San Marino and the Holy See',
    'Serbia': 'Serbia and Montenegro',
    'South Sudan': 'Sudan and South Sudan',
    'Spain': 'Spain and Andorra',
    'State of Palestine': 'Israel and Palestine, State of',
    'Sudan': 'Sudan and South Sudan',
    'Switzerland': 'Switzerland and Liechtenstein',
    'Syrian Arab Republic': 'Syria',
    'United Republic of Tanzania': 'Tanzania',
    'Türkiye': 'Turkey',
    'United States of America': 'United States',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    # Add new mappings for abbreviated names
    'Antigua and Barbuda': 'Antigua',
    'Saint Kitts and Nevis': 'Saint Kitts',
    'Trinidad and Tobago': 'Trinidad',
    'Saint Vincent and the Grenadines': 'Saint Vincent',
    # Add mappings for EORA consumption data
    'USA': 'United States',
    'UK': 'United Kingdom',
    'UAE': 'United Arab Emirates',
}

# Create reverse mappings for both emission and consumption countries
emissions_to_population = {v: k for k, v in country_mapping.items()}
consumption_to_population = {v: k for k, v in country_mapping.items()}

# First, create a DataFrame with all emissions countries
base_countries = emissions[~emissions['Country'].isin(["International Aviation", "International Shipping", "EU27", "GLOBAL TOTAL"])]
latest_emissions = []

# Process territory-based emissions first
for _, row in base_countries.iterrows():
    country = row['Country']
    
    # Find matching population country name
    pop_country = country
    if country in emissions_to_population:
        pop_country = emissions_to_population[country]
    
    # Get population data
    pop_match = population[population['Country'] == pop_country]
    population_2050 = pop_match['2050'].iloc[0] if not pop_match.empty else None
    
    latest_emissions.append({
        'country': country,
        'CO2_emissions_latest (m tons)': row['2023'],
        'latest_year': 2023,
        'population_2050': population_2050,
        'scope': 'Territory_based'
    })

# Process consumption-based emissions
for _, row in consumption.iterrows():
    country = row['Country']
    if country in ["International Aviation", "International Shipping", "EU27", "GLOBAL TOTAL"]:
        continue
    
    # Find matching population country name
    pop_country = country
    if country in consumption_to_population:
        pop_country = consumption_to_population[country]
    
    # Get population data
    pop_match = population[population['Country'] == pop_country]
    population_2050 = pop_match['2050'].iloc[0] if not pop_match.empty else None
    
    latest_emissions.append({
        'country': country,
        'CO2_emissions_latest (m tons)': row['2022'],
        'latest_year': 2022,
        'population_2050': population_2050,
        'scope': 'Consumption_based'
    })

# Convert to DataFrame
latest_emissions_df = pd.DataFrame(latest_emissions)

# Print statistics
print("\nData Statistics:")
print(f"Total number of records: {len(latest_emissions_df)}")
print(f"Territory-based records: {len(latest_emissions_df[latest_emissions_df['scope'] == 'Territory_based'])}")
print(f"Consumption-based records: {len(latest_emissions_df[latest_emissions_df['scope'] == 'Consumption_based'])}")
print(f"Records with population data: {len(latest_emissions_df.dropna(subset=['population_2050']))}")

# Print countries missing population data
missing_pop = latest_emissions_df[latest_emissions_df['population_2050'].isna()]['country'].unique()
missing_pop = [country for country in missing_pop if pd.notna(country)]  # Filter out NaN values
print("\nCountries missing population data:")
print(sorted(missing_pop))

# Save the final dataset
latest_emissions_df.to_csv(f"{output_directory}/Latest_emissions.csv", index=False)
print(f"\nData saved to {output_directory}/Latest_emissions.csv")
