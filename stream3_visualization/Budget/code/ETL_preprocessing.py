import pandas as pd
import numpy as np

# Define the directory containing the data files
data_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data'
output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'

# Load the datasets
emissions_file = f"{data_directory}/EDGAR- Total fossil_CO2.csv"
population_file = f"{data_directory}/UN population projections.csv"
consumption_file = f"{data_directory}/EORA_consumption_CO2_1990-2022.csv"

# Read the datasets
emissions = pd.read_csv(emissions_file, sep=';', encoding='latin1')
population = pd.read_csv(population_file, sep=';', encoding='latin1')
consumption = pd.read_csv(consumption_file, sep=';', encoding='latin1')

# Rename the consumption country column to remove the BOM
consumption.rename(columns={consumption.columns[0]: 'Country'}, inplace=True)

# Convert emission columns to numeric
emission_cols = [str(year) for year in range(1970, 2024)]
for col in emission_cols:
    emissions[col] = pd.to_numeric(emissions[col].astype(str).str.replace(',', '.'), errors='coerce')

# Convert population['2050'] to numeric safely: remove both dots and commas as these are just thousand separators
population['2050'] = pd.to_numeric(population['2050'].astype(str).str.replace('.', '').str.replace(',', ''), errors='coerce')

# Standardize column names
pop_col = "Region, subregion, country or area *"
emis_col = "Country"

# Initial data check
print("\n=== Population Data Check ===")
print("Sample of original population data:")
print(population[[pop_col, '2050']].head())

print("\n=== Population File Structure ===")
print("Column names:", population.columns.tolist())
print("\nFirst few rows of raw population data:")
print(population.head())

# Create mapping dictionary
correction_map_base = {
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Brunei Darussalam': 'Brunei',
    'China, Hong Kong SAR': 'Hong Kong',
    'China, Macao SAR': 'Macao',
    'China, Taiwan Province of China': 'Taiwan',
    "Côte d'Ivoire": 'Côte d\x92Ivoire',
    "Dem. People's Republic of Korea": 'North Korea',
    'Falkland Islands (Malvinas)': 'Falkland Islands',
    'Faroe Islands': 'Faroes',
    'France': 'France and Monaco',
    'Gambia': 'The Gambia',
    'Guam': 'Guam',
    'Guernsey': 'Guernsey',
    'Holy See': 'Italy, San Marino and the Holy See',
    'Iran (Islamic Republic of)': 'Iran',
    'Isle of Man': 'Isle of Man',
    'Israel': 'Israel and Palestine, State of',
    'Italy': 'Italy, San Marino and the Holy See',
    'Jersey': 'Jersey',
    'Kosovo (under UNSC res. 1244)': 'Kosovo',
    "Lao People's Democratic Republic": 'Laos',
    'Liechtenstein': 'Switzerland and Liechtenstein',
    'Marshall Islands': 'Marshall Islands',
    'Mayotte': 'Mayotte',
    'Micronesia (Fed. States of)': 'Micronesia',
    'Monaco': 'France and Monaco',
    'Montenegro': 'Serbia and Montenegro',
    'Montserrat': 'Montserrat',
    'Myanmar': 'Myanmar/Burma',
    'Nauru': 'Nauru',
    'Niue': 'Niue',
    'Northern Mariana Islands': 'Northern Mariana Islands',
    'Republic of Korea': 'South Korea',
    'Republic of Moldova': 'Moldova',
    'Russian Federation': 'Russia',
    'Saint Barthélemy': 'Saint Barthélemy',
    'Saint Helena': 'Saint Helena, Ascension and Tristan da Cunha',
    'Saint Martin (French part)': 'Saint Martin',
    'San Marino': 'Italy, San Marino and the Holy See',
    'Sao Tome and Principe': 'São Tomé and Príncipe',
    'Serbia': 'Serbia and Montenegro',
    'Sint Maarten (Dutch part)': 'Sint Maarten',
    'South Sudan': 'Sudan and South Sudan',
    'Spain': 'Spain and Andorra',
    'State of Palestine': 'Israel and Palestine, State of',
    'Sudan': 'Sudan and South Sudan',
    'Switzerland': 'Switzerland and Liechtenstein',
    'Syrian Arab Republic': 'Syria',
    'Tokelau': 'Tokelau',
    'Tuvalu': 'Tuvalu',
    'United Republic of Tanzania': 'Tanzania',
    'United States Virgin Islands': 'United States Virgin Islands',
    'United States of America': 'United States',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Wallis and Futuna Islands': 'Wallis and Futuna Islands'
}

# First, apply corrections to population DataFrame
population["corrected_country"] = population[pop_col].replace(correction_map_base)

# NOW we can print the corrected data
print("\nSample of population data after country corrections:")
print(population[['corrected_country', '2050']].head())

# Create the population lookup by summing population for corrected countries
population_final = population.groupby("corrected_country", as_index=False)["2050"].sum()

# After grouping, print the final data
print("\nSample of final population data:")
print(population_final.head())

# Print statistics
print("\nPopulation data statistics:")
print(f"Number of countries in original population data: {len(population[pop_col].unique())}")
print(f"Number of countries after corrections: {len(population['corrected_country'].unique())}")
print(f"Number of countries in final population data: {len(population_final)}")
print(f"Number of non-null 2050 values: {population_final['2050'].count()}")

# Example country debug
example_country = emissions['Country'].iloc[0]
print(f"\nDebugging matching for example country: {example_country}")
print("Matching population record:")
print(population_final[population_final['corrected_country'] == example_country])

# Create a reverse mapping dictionary for emissions/consumption to population names
reverse_correction_map = {v: k for k, v in correction_map_base.items()}

# Prepare the latest emissions data
latest_emissions = []

# For EDGAR data (Territory-based)
for _, row in emissions.iterrows():
    country = row['Country']
    if country in ["International Aviation", "International Shipping", "EU27", "GLOBAL TOTAL"]:
        continue  # Skip aggregated entries
    
    # Try to find the population using the original country name or its corrected version
    population_2050_match = population_final[population_final['corrected_country'] == country]['2050']
    if population_2050_match.empty and country in reverse_correction_map:
        # Try looking up using the original name from the correction map
        orig_name = reverse_correction_map[country]
        population_2050_match = population_final[population_final['corrected_country'] == orig_name]['2050']
    
    latest_year = 2023
    co2_emissions_latest = row[str(latest_year)]
    
    if not population_2050_match.empty:
        population_2050 = population_2050_match.values[0]
        latest_emissions.append({
            'country': country,
            'CO2_emissions_latest (m tons)': co2_emissions_latest,
            'latest_year': latest_year,
            'population_2050': population_2050,
            'scope': 'Territory_based'
        })

# For EORA data (Consumption-based)
for _, row in consumption.iterrows():
    country = row['Country']
    if country in ["International Aviation", "International Shipping", "EU27", "GLOBAL TOTAL"]:
        continue  # Skip aggregated entries
    
    # Try to find the population using the original country name or its corrected version
    population_2050_match = population_final[population_final['corrected_country'] == country]['2050']
    if population_2050_match.empty and country in reverse_correction_map:
        # Try looking up using the original name from the correction map
        orig_name = reverse_correction_map[country]
        population_2050_match = population_final[population_final['corrected_country'] == orig_name]['2050']
    
    latest_year = 2022
    co2_emissions_latest = row[str(latest_year)]
    
    if not population_2050_match.empty:
        population_2050 = population_2050_match.values[0]
        latest_emissions.append({
            'country': country,
            'CO2_emissions_latest (m tons)': co2_emissions_latest,
            'latest_year': latest_year,
            'population_2050': population_2050,
            'scope': 'Consumption_based'
        })

# Convert to DataFrame
latest_emissions_df = pd.DataFrame(latest_emissions)
print("\nFirst 10 rows of Latest_emissions.csv:")
print(latest_emissions_df.head(10))

# Print countries missing population data
missing_pop = latest_emissions_df[latest_emissions_df['population_2050'].isna()]['country'].unique()
print("\nCountries missing population data:")
print(sorted(missing_pop))

# Save the latest emissions data
latest_emissions_df.to_csv(f"{output_directory}/Latest_emissions.csv", index=False)
print(f"Latest emissions data saved to {output_directory}/Latest_emissions.csv")

# Prepare the historical emissions data
historical_emissions = []

# For EDGAR data (Territory-based)
for _, row in emissions.iterrows():
    country = row['Country']
    if country in ["International Aviation", "International Shipping", "EU27", "GLOBAL TOTAL"]:
        continue  # Skip aggregated entries
    for year in range(1970, 2024):
        co2_emissions = row[str(year)]
        historical_emissions.append({
            'country': country,
            'CO2_emissions_latest (m tons)': co2_emissions,
            'year': year,
            'scope': 'Territory_based'
        })

# For EORA data (Consumption-based)
for _, row in consumption.iterrows():
    country = row['Country']
    if country in ["International Aviation", "International Shipping", "EU27", "GLOBAL TOTAL"]:
        continue  # Skip aggregated entries
    for year in range(1990, 2023):
        co2_emissions = row[str(year)]
        historical_emissions.append({
            'country': country,
            'CO2_emissions_latest (m tons)': co2_emissions,
            'year': year,
            'scope': 'Consumption_based'
        })

# Convert to DataFrame
historical_emissions_df = pd.DataFrame(historical_emissions)
print("\nFirst 10 rows of Historical_emissions.csv:")
print(historical_emissions_df.head(10))
print("\nHistorical emissions data converted to dataframe")

# Save the historical emissions data
historical_emissions_df.to_csv(f"{output_directory}/Historical_emissions.csv", index=False)
print(f"Historical emissions data saved to {output_directory}/Historical_emissions.csv")
