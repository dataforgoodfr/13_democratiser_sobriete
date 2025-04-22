"""
This data processing script follows the following steps:
1. Load the data from the data directory from 4 source files
2. Clean the data
3. Combine the data
4. Save the data
"""


import pandas as pd
import numpy as np

# Define the directory containing the data files
data_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data'
output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'

"""
Extract relevant data from each source file:
# 2024-04-21_IPCC Regional Breakdown_ISO Country Code.xlsx needs: 
- the following columns: Intermediate level (10) as IPCC_Region_Intermediate, ISO codes as ISO3 from the tab: Full_mapping
- the following transformations: splitting the values in the ISO codes column by comma into rows mapped to their respective ISO codes values in the Intermediate level (10) column
# 2025-04-21_GDP _PPP constant 2021 US$_per country ISO Code.xlsx needs:
- the following columns from the Data tab: Country Name, Country Code, 1990 [YR1990] and all the columns to the right of this
- the following transformations: 
    - renaming the columns to remove the [YR1990] suffix
    - renaming the Country Name column to Country
    - renaming the Country Code column to ISO3
    - Transforming the years and values into 2 columns: one named Year, and one named "GDP, PPP (constant 2021 international $)" with the values
    - converting the values to float, with 2 decimal places and transforming the comma separator into a dot separator for decimals
# 2025-04-21_Population per Country ISO code_1970-2050.xlsx needs:
- the following columns from the unpopulation_dataportal_2025042 tab: Iso3 as ISO03, Location as Country, Time as Year, Value as Population
# 2025-04-22_CO2 Emissions_All Countries_ISO Code_1750-2023.xlsx needs:
- the following columns from the GCB2024v17_MtCO2_flat tab: Country, ISO 3166-1 alpha-3 as ISO3, Year, Total as CO2_emssions_Mt

Once the data is extracted as dataframes: combine into a single dataframe with the following transformations:
- ISO3,Country, Region, Year, Measure, Value where Measure is GDP, PPP (constant 2021 international $), Population, CO2_emssions_Mt and the Region column is mapped to the ISO3 code of each country
- the dataframe should be sorted by ISO3, Year, Measure
- the dataframe should be saved as a csv file in the output directory
"""

# Load the data
# 1. IPCC Regional Breakdown
ipcc_regions = pd.read_excel(f"{data_directory}/2024-04-21_IPCC Regional Breakdown_ISO Country Code.xlsx", 
                            sheet_name="Full_mapping")
# Extract relevant columns
ipcc_regions = ipcc_regions[['Intermediate level (10)', 'ISO codes']]
ipcc_regions.rename(columns={'Intermediate level (10)': 'IPCC_Region_Intermediate', 'ISO codes': 'ISO3'}, inplace=True)

# Load EU and G20 country mappings
eu_g20_mapping = pd.read_excel(f"{data_directory}/2024-04-21_IPCC Regional Breakdown_ISO Country Code.xlsx",
                              sheet_name="G20_EU_Countries ",
                              header=0)  # Explicitly set header to row 0 (first row)
eu_g20_mapping = eu_g20_mapping[['ISO3', 'EU_country', 'G20_country']]

# Split ISO codes and create separate rows
ipcc_regions_expanded = []
for _, row in ipcc_regions.iterrows():
    iso_codes = row['ISO3'].split(',')
    for iso in iso_codes:
        ipcc_regions_expanded.append({
            'IPCC_Region_Intermediate': row['IPCC_Region_Intermediate'],
            'ISO3': iso.strip()
        })
ipcc_regions = pd.DataFrame(ipcc_regions_expanded)

# Create a set of valid ISO3 codes that have region mapping
valid_iso3_codes = set(ipcc_regions['ISO3'].unique())

# 2. GDP data
gdp_data = pd.read_excel(f"{data_directory}/2025-04-21_GDP _PPP constant 2021 US$_per country ISO Code.xlsx", 
                         sheet_name="Data")
# Extract relevant columns (Country Name, Country Code, and years from 1990 onwards)
year_columns = [col for col in gdp_data.columns if '[YR' in col and int(col.split('[YR')[1].split(']')[0]) >= 1990]
gdp_data = gdp_data[['Country Name', 'Country Code'] + year_columns]

# Rename columns
gdp_data.rename(columns={'Country Name': 'Country', 'Country Code': 'ISO3'}, inplace=True)
# Remove [YR] suffix from year columns
gdp_data.columns = [col.split('[YR')[0].strip() if '[YR' in col else col for col in gdp_data.columns]

# Transform to long format
gdp_long = pd.melt(gdp_data, 
                  id_vars=['Country', 'ISO3'], 
                  var_name='Year', 
                  value_name='GDP, PPP (constant 2021 international $)')

# Convert Year to integer and clean GDP values
gdp_long['Year'] = gdp_long['Year'].astype(int)
gdp_long['GDP, PPP (constant 2021 international $)'] = gdp_long['GDP, PPP (constant 2021 international $)'].apply(
    lambda x: np.nan if pd.isna(x) or str(x).strip() in ['..', ''] else float(str(x).replace(',', '.'))
)
gdp_long['GDP, PPP (constant 2021 international $)'] = gdp_long['GDP, PPP (constant 2021 international $)'].round(2)

# 3. Population data
population_data = pd.read_excel(f"{data_directory}/2025-04-21_Population per Country ISO code_1970-2050.xlsx", 
                               sheet_name="unpopulation_dataportal_2025042")
# Extract relevant columns
population_data = population_data[['Iso3', 'Location', 'Time', 'Value']]
population_data.rename(columns={'Iso3': 'ISO3', 'Location': 'Country', 'Time': 'Year', 'Value': 'Population'}, inplace=True)

# 4. CO2 Emissions data
emissions_data = pd.read_excel(f"{data_directory}/2025-04-22_CO2 Emissions_All Countries_ISO Code_1750-2023.xlsx", 
                              sheet_name="GCB2024v17_MtCO2_flat")
# Extract relevant columns
emissions_data = emissions_data[['Country', 'ISO 3166-1 alpha-3', 'Year', 'Total']]
emissions_data.rename(columns={'ISO 3166-1 alpha-3': 'ISO3', 'Total': 'CO2_emissions_Mt'}, inplace=True)

# Combine all dataframes
# Transform GDP data to match the final format
gdp_measure = gdp_long.copy()
gdp_measure['Measure'] = 'GDP, PPP (constant 2021 international $)'
gdp_measure.rename(columns={'GDP, PPP (constant 2021 international $)': 'Value'}, inplace=True)

# Transform Population data
pop_measure = population_data.copy()
pop_measure['Measure'] = 'Population'
pop_measure.rename(columns={'Population': 'Value'}, inplace=True)

# Transform Emissions data
emissions_measure = emissions_data.copy()
emissions_measure['Measure'] = 'CO2_emissions_Mt'
emissions_measure.rename(columns={'CO2_emissions_Mt': 'Value'}, inplace=True)

# Combine all dataframes
combined_df = pd.concat([gdp_measure[['ISO3', 'Country', 'Year', 'Measure', 'Value']], 
                         pop_measure[['ISO3', 'Country', 'Year', 'Measure', 'Value']], 
                         emissions_measure[['ISO3', 'Country', 'Year', 'Measure', 'Value']]], 
                        ignore_index=True)

# Filter out ISO3 codes without region mapping
combined_df = combined_df[combined_df['ISO3'].isin(valid_iso3_codes)]

# Add Region column by mapping ISO3 codes
region_mapping = ipcc_regions.set_index('ISO3')['IPCC_Region_Intermediate'].to_dict()
combined_df['Region'] = combined_df['ISO3'].map(region_mapping)

"""
Sanity check country names:
- Names with special characters: such as Curaçao: always use without the special characters: for instance Curaçao be Curacao and Réunion be Reunion
- ISOs with different country names based on file such asKorea, Dem. People's Rep. and Dem. People's Rep. of Korea, Yemen, Rep. and Yemen: always use the Country name from the population_data file
"""
# Create a mapping of ISO3 codes to standardized country names from population data
standard_names = pop_measure[['ISO3', 'Country']].drop_duplicates().set_index('ISO3')['Country'].to_dict()

# Add specific mappings for countries with special characters
special_country_mapping = {
    'REU': 'Reunion',
    'TUR': 'Turkiye',
    'CUW': 'Curacao',
    'CIV': 'Cote d\'Ivoire',
    'BLM': 'Saint Barthelemy'
}

# Update standard_names with special mappings
standard_names.update(special_country_mapping)

# Apply standardized country names where ISO3 codes match
combined_df['Country'] = combined_df.apply(
    lambda row: standard_names.get(row['ISO3'], row['Country']) 
    if row['ISO3'] in standard_names else row['Country'], 
    axis=1
)

# Print before and after counts to verify changes
print(f"Before standardization: {combined_df['Country'].nunique()} unique country names")
print(f"After standardization: {combined_df['Country'].nunique()} unique country names")

# Print examples of countries with changed names
sample_countries = combined_df[['ISO3', 'Country']].drop_duplicates().sample(min(10, combined_df['ISO3'].nunique()))
print("\nSample of standardized country names:")
for _, row in sample_countries.iterrows():
    original = row['Country']
    standardized = standard_names.get(row['ISO3'], original)
    if original != standardized:
        print(f"ISO3: {row['ISO3']}, Original: {original}, Standardized: {standardized}")

# Add EU and G20 country flags
eu_mapping = eu_g20_mapping.set_index('ISO3')['EU_country'].to_dict()
g20_mapping = eu_g20_mapping.set_index('ISO3')['G20_country'].to_dict()

combined_df['EU_country'] = combined_df['ISO3'].map(eu_mapping).fillna('No')
combined_df['G20_country'] = combined_df['ISO3'].map(g20_mapping).fillna('No')

# Sort the dataframe
combined_df = combined_df.sort_values(['ISO3', 'Year', 'Measure'])

# Save the combined dataframe
combined_df.to_csv(f"{output_directory}/combined_data.csv", index=False)

# Print verification
print(f"Combined data saved to {output_directory}/combined_data.csv")
print(f"Total rows: {len(combined_df)}")
print(f"Unique countries: {combined_df['Country'].nunique()}")
print(f"Year range: {combined_df['Year'].min()} to {combined_df['Year'].max()}")
print(f"Measures: {', '.join(combined_df['Measure'].unique())}")
print(f"EU countries: {combined_df[combined_df['EU_country'] == 'Yes']['Country'].nunique()}")
print(f"G20 countries: {combined_df[combined_df['G20_country'] == 'Yes']['Country'].nunique()}")
print("\nFirst 50 rows of the combined dataframe:")
print(combined_df.head(50).to_string())