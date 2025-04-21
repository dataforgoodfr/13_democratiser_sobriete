import pandas as pd
import numpy as np

# Define the directory containing the data files
data_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data'
output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'

# Define cleaning functions
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

def combine_country_data(df, main_country, dependent_country):
    # Get main and dependent country data
    main_data = df[df['Country'] == main_country]
    dep_data = df[df['Country'] == dependent_country]
    
    # Combine the data
    combined_data = []
    for year in df['Year'].unique():
        for scope in df['scope'].unique():
            main_year_scope = main_data[(main_data['Year'] == year) & (main_data['scope'] == scope)]
            dep_year_scope = dep_data[(dep_data['Year'] == year) & (dep_data['scope'] == scope)]
            
            if not main_year_scope.empty and not dep_year_scope.empty:
                combined_emissions = main_year_scope['CO2_emissions (m tons)'].iloc[0] + dep_year_scope['CO2_emissions (m tons)'].iloc[0]
                combined_data.append({
                    'Country': main_country,
                    'Year': year,
                    'CO2_emissions (m tons)': combined_emissions,
                    'scope': scope
                })
            elif not main_year_scope.empty:
                combined_data.append({
                    'Country': main_country,
                    'Year': year,
                    'CO2_emissions (m tons)': main_year_scope['CO2_emissions (m tons)'].iloc[0],
                    'scope': scope
                })
    
    # Remove original main and dependent country data
    df = df[~df['Country'].isin([main_country, dependent_country])]
    
    # Add combined data
    if combined_data:
        df = pd.concat([df, pd.DataFrame(combined_data)], ignore_index=True)
    
    return df

# Create a mapping dictionary for country names
country_mapping = {
    # Previous mappings
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
    'Tanzania, United Republic of': 'Tanzania',
    
    # New mappings from the image
    'Cote dIvoire': 'Côte d\'Ivoire',
    'Côte d\'Ivoire': 'Côte d\'Ivoire',
    'Falkland Islands (Malvinas)': 'Falkland Islands (Malvinas)',
    'Faroe Islands': 'Faroe Islands',
    'Hong Kong': 'Hong Kong',
    'China, Hong Kong SAR': 'Hong Kong',
    'Israel and Palestine, State of': 'Israel',
    'Italy, San Marino and the Holy See': 'Italy',
    'Macao': 'Macao',
    'Macao SAR': 'Macao',
    'China, Macao SAR': 'Macao',
    'Moldova': 'Moldova',
    'Republic of Moldova': 'Moldova',
    'Myanmar/Burma': 'Myanmar',
    'Myanmar': 'Myanmar',
    'North Korea': 'North Korea',
    'Dem. People\'s Republic of Korea': 'North Korea',
    'Saint Helena, Ascension and Tristan da Cunha': 'Saint Helena',
    'São Tomé and Príncipe': 'Sao Tome and Principe',
    'Sao Tome and Principe': 'Sao Tome and Principe',
    'Serbia and Montenegro': 'Serbia',
    'Sudan and South Sudan': 'Sudan',
    'Switzerland and Liechtenstein': 'Switzerland',
    'Taiwan': 'Taiwan',
    'China, Taiwan Province of China': 'Taiwan',
    'Tanzania': 'Tanzania',
    'United Republic of Tanzania': 'Tanzania',
    'The Gambia': 'Gambia',
    'Gambia': 'Gambia',
    'DR Congo': 'Democratic Republic of the Congo',
    'Democratic Republic of the Congo': 'Democratic Republic of the Congo',
    'Cape Verde': 'Cape Verde',
    'Cabo Verde': 'Cape Verde',
    'Antigua': 'Antigua',
    'Antigua and Barbuda': 'Antigua',
    'TFYR Macedonia': 'North Macedonia',
    'North Macedonia': 'North Macedonia'
}

# Entries to filter out
excluded_entries = [
    'International Aviation',
    'EU27',
    'GLOBAL TOTAL',
    'Gaza Strip',
    'Former USSR',
    'International Shipping',
    'Rest of World',
    'Netherlands Antilles',
    ''  # Empty country name
]

# Step 1: Extract required columns from each file
# Population data
population = pd.read_excel(f"{data_directory}/UN population projections.xlsx")
# Filter for Median PI variant only
population = population[population['Variant'] == 'Median PI']
population = population[['Region, subregion, country or area *', 2050]]

# Footprint data (for latest emissions)
footprint = pd.read_csv(f"{data_directory}/EORA_consumption_CO2_1990-2022.csv", sep=';', encoding='latin1')
footprint = footprint[['ï»¿Country', '2022']]
footprint = footprint.rename(columns={'ï»¿Country': 'Country', '2022': 2022})

# Emissions data (for latest emissions)
emissions = pd.read_excel(f"{data_directory}/EDGAR- Total fossil_CO2.xlsx")
emissions = emissions[['Country', 2023]]

# Historical data processing
# Read full historical data
emissions_full = pd.read_excel(f"{data_directory}/EDGAR- Total fossil_CO2.xlsx")
footprint_full = pd.read_csv(f"{data_directory}/EORA_consumption_CO2_1990-2022.csv", sep=';', encoding='latin1')

# Process historical emissions data
# Get all year columns (numeric columns)
emissions_years = [col for col in emissions_full.columns if isinstance(col, int)]
footprint_years = [col for col in footprint_full.columns if str(col).isdigit()]

# Melt emissions data to long format
emissions_historical = emissions_full[['Country'] + emissions_years].melt(
    id_vars=['Country'],
    var_name='Year',
    value_name='CO2_emissions (m tons)'
)
emissions_historical['scope'] = 'Territory_based'

# Melt footprint data to long format
footprint_historical = footprint_full.rename(columns={'ï»¿Country': 'Country'})
footprint_historical = footprint_historical[['Country'] + footprint_years].melt(
    id_vars=['Country'],
    var_name='Year',
    value_name='CO2_emissions (m tons)'
)
footprint_historical['scope'] = 'Consumption_based'
# Convert footprint values from thousands to millions of tons
footprint_historical['CO2_emissions (m tons)'] = footprint_historical['CO2_emissions (m tons)'].apply(clean_emissions) / 1000

# Combine historical datasets
historical_dataset = pd.concat([emissions_historical, footprint_historical], ignore_index=True)

# Step 2: Reformat column names
population = population.rename(columns={'Region, subregion, country or area *': 'Country'})

emissions = emissions.rename(columns={2023: 'CO2_emissions_latest (m tons)'})
emissions['latest_year'] = 2023

footprint = footprint.rename(columns={2022: 'CO2_emissions_latest (m tons)'})
footprint['latest_year'] = 2022

# Step 3: Clean values
# Clean population numbers
population[2050] = population[2050].apply(clean_population)

# Clean footprint numbers and convert to million tons
footprint['CO2_emissions_latest (m tons)'] = footprint['CO2_emissions_latest (m tons)'].apply(clean_emissions)
footprint['CO2_emissions_latest (m tons)'] = footprint['CO2_emissions_latest (m tons)'] / 1000

# Filter out excluded entries and NaN values
population = population[~population['Country'].isin(excluded_entries) & population['Country'].notna()]
footprint = footprint[~footprint['Country'].isin(excluded_entries) & footprint['Country'].notna()]
emissions = emissions[~emissions['Country'].isin(excluded_entries) & emissions['Country'].notna()]

# Apply country mapping
population['Country'] = population['Country'].map(lambda x: country_mapping.get(x, x))
footprint['Country'] = footprint['Country'].map(lambda x: country_mapping.get(x, x))
emissions['Country'] = emissions['Country'].map(lambda x: country_mapping.get(x, x))

# Step 4: Create latest emissions dataset
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

# Drop any duplicates and rows with missing population data
final_dataset = final_dataset.drop_duplicates()
final_dataset = final_dataset[final_dataset['population_2050'].notna()]

# Step 5: Process historical dataset
# Apply country mapping and filters
historical_dataset['Country'] = historical_dataset['Country'].map(lambda x: country_mapping.get(x, x))
historical_dataset = historical_dataset[
    ~historical_dataset['Country'].isin(excluded_entries) & 
    historical_dataset['Country'].notna()
]

# Combine France/Monaco and Spain/Andorra data
historical_dataset = combine_country_data(historical_dataset, 'France', 'Monaco')
historical_dataset = combine_country_data(historical_dataset, 'Spain', 'Andorra')

# Filter out countries without population data
countries_with_pop = final_dataset['Country'].unique()
historical_dataset = historical_dataset[historical_dataset['Country'].isin(countries_with_pop)]

# Sort the historical dataset
historical_dataset = historical_dataset.sort_values(['Country', 'Year', 'scope'])

# Save both datasets
final_dataset.to_csv(f"{output_directory}/Latest_emissions.csv", index=False)
historical_dataset.to_csv(f"{output_directory}/Historical_emissions.csv", index=False)

# Print verification data
print("\nSample data for key countries (latest emissions):")
print("=" * 120)
# Select specific countries
selected_countries = ['France', 'Spain', 'Russia', 'United Kingdom', 'United States', 'China', 'Turkey']
# Filter and sort the data
sample_data = final_dataset[final_dataset['Country'].isin(selected_countries)]
sample_data = sample_data.sort_values(['Country', 'scope'])
# Format the data for better display
pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 5)
# Print the data in table format
print(sample_data[['Country', 'CO2_emissions_latest (m tons)', 'latest_year', 'population_2050', 'scope']].to_string(index=False))
print("=" * 120)

# Print sample of historical data
print("\nSample of historical data for France and Spain:")
print("=" * 120)
france_historical = historical_dataset[historical_dataset['Country'] == 'France'].sort_values(['Year', 'scope'])
spain_historical = historical_dataset[historical_dataset['Country'] == 'Spain'].sort_values(['Year', 'scope'])
print("\nFrance:")
print(france_historical.head(10).to_string(index=False))
print("\nSpain:")
print(spain_historical.head(10).to_string(index=False))
print("=" * 120)
