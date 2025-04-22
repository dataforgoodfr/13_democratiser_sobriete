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
# 2025-04-22_Consumption emissions MtCO2_ISO code.xlsx needs:
- the following columns from the Consumption GCB2024v17_MtCO2_flat tab: Country, ISO 3166-1 alpha-3 as ISO3, Year, Total as Consumption_emissions_Mt as Consumption_CO2_emissions_Mt
- the following transformations: transforming the values to float, with 2 decimal places and transforming the comma separator into a dot separator for decimals

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

# 5. Consumption CO2 Emissions data
consumption_emissions_data = pd.read_excel(f"{data_directory}/2025-04-22_Consumption emissions MtCO2_ISO code.xlsx", 
                                          sheet_name="GCB2024v17_MtCO2_flat")
# Extract relevant columns
consumption_emissions_data = consumption_emissions_data[['Country', 'ISO 3166-1 alpha-3', 'Year', 'CO2_Consumption_emissions in Mt']]
consumption_emissions_data.rename(columns={'ISO 3166-1 alpha-3': 'ISO3', 'CO2_Consumption_emissions in Mt': 'Consumption_CO2_emissions_Mt'}, inplace=True)

# Convert consumption emissions values to float with 2 decimal places
consumption_emissions_data['Consumption_CO2_emissions_Mt'] = consumption_emissions_data['Consumption_CO2_emissions_Mt'].apply(
    lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else float(x)
).round(2)

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

# Transform Consumption Emissions data
consumption_emissions_measure = consumption_emissions_data.copy()
consumption_emissions_measure['Measure'] = 'Consumption_CO2_emissions_Mt'
consumption_emissions_measure.rename(columns={'Consumption_CO2_emissions_Mt': 'Value'}, inplace=True)

# Combine all dataframes
combined_df = pd.concat([gdp_measure[['ISO3', 'Country', 'Year', 'Measure', 'Value']], 
                         pop_measure[['ISO3', 'Country', 'Year', 'Measure', 'Value']], 
                         emissions_measure[['ISO3', 'Country', 'Year', 'Measure', 'Value']],
                         consumption_emissions_measure[['ISO3', 'Country', 'Year', 'Measure', 'Value']]], 
                        ignore_index=True)

# Filter out ISO3 codes without region mapping
combined_df = combined_df[combined_df['ISO3'].isin(valid_iso3_codes)]

# Add Region column by mapping ISO3 codes
region_mapping = ipcc_regions.set_index('ISO3')['IPCC_Region_Intermediate'].to_dict()
combined_df['Region'] = combined_df['ISO3'].map(region_mapping)

# Add EU and G20 country flags
eu_mapping = eu_g20_mapping.set_index('ISO3')['EU_country'].to_dict()
g20_mapping = eu_g20_mapping.set_index('ISO3')['G20_country'].to_dict()

combined_df['EU_country'] = combined_df['ISO3'].map(eu_mapping).fillna('No')
combined_df['G20_country'] = combined_df['ISO3'].map(g20_mapping).fillna('No')

# Pivot the data to get GDP and Population as columns
pivoted_df = combined_df.pivot_table(
    index=['ISO3', 'Country', 'Region', 'Year', 'EU_country', 'G20_country'],
    columns='Measure',
    values='Value',
    aggfunc='first'
).reset_index()

# Rename the emissions columns
pivoted_df = pivoted_df.rename(columns={
    'CO2_emissions_Mt': 'CO2_emissions_Mt',
    'Consumption_CO2_emissions_Mt': 'Consumption_CO2_emissions_Mt'
})

# Calculate new metrics
pivoted_df['Emissions_per_capita_ton'] = pivoted_df['CO2_emissions_Mt'] * 1000000 / pivoted_df['Population']
pivoted_df['Emissions_per_GDP$_ton'] = pivoted_df['CO2_emissions_Mt'] * 1000000 / pivoted_df['GDP, PPP (constant 2021 international $)']

# Calculate cumulative emissions for each country and emissions scope
pivoted_df['Cumulative_CO2_emissions_Mt'] = pivoted_df.groupby(['ISO3', 'Region'])['CO2_emissions_Mt'].cumsum()
pivoted_df['Cumulative_Consumption_CO2_emissions_Mt'] = pivoted_df.groupby(['ISO3', 'Region'])['Consumption_CO2_emissions_Mt'].cumsum()

# Ensure all required columns exist before melting
required_columns = ['ISO3', 'Country', 'Region', 'Year', 'EU_country', 'G20_country', 
                   'Population', 'GDP, PPP (constant 2021 international $)',
                   'Emissions_per_capita_ton', 'Emissions_per_GDP$_ton',
                   'CO2_emissions_Mt', 'Consumption_CO2_emissions_Mt',
                   'Cumulative_CO2_emissions_Mt', 'Cumulative_Consumption_CO2_emissions_Mt']

# Check if all required columns exist
missing_columns = [col for col in required_columns if col not in pivoted_df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Melt the dataframe back to long format for emissions data
emissions_df = pivoted_df.melt(
    id_vars=['ISO3', 'Country', 'Region', 'Year', 'EU_country', 'G20_country', 
             'Population', 'GDP, PPP (constant 2021 international $)',
             'Emissions_per_capita_ton', 'Emissions_per_GDP$_ton'],
    value_vars=['CO2_emissions_Mt', 'Consumption_CO2_emissions_Mt',
                'Cumulative_CO2_emissions_Mt', 'Cumulative_Consumption_CO2_emissions_Mt'],
    var_name='Emissions_scope',
    value_name='Value'  # Keep as 'Value' during melt
)

# Rename the Value column to CO2_emissions_Mt after melting
emissions_df = emissions_df.rename(columns={'Value': 'CO2_emissions_Mt'})

# Update Emissions_scope values
emissions_scope_mapping = {
    'CO2_emissions_Mt': 'Annual Territory',
    'Consumption_CO2_emissions_Mt': 'Annual Consumption',
    'Cumulative_CO2_emissions_Mt': 'Cumulative Territory',
    'Cumulative_Consumption_CO2_emissions_Mt': 'Cumulative Consumption'
}
emissions_df['Emissions_scope'] = emissions_df['Emissions_scope'].map(emissions_scope_mapping)

# Sort the dataframe
emissions_df = emissions_df.sort_values(['ISO3', 'Year', 'Emissions_scope'])

# Create region aggregates
print("Creating region aggregates...")
region_aggregates = emissions_df.groupby(['Region', 'Year', 'Emissions_scope']).agg({
    'CO2_emissions_Mt': 'sum',
    'Population': 'sum',
    'GDP, PPP (constant 2021 international $)': 'sum'
}).reset_index()

# Calculate per capita and per GDP metrics for regions
region_aggregates['Emissions_per_capita_ton'] = region_aggregates['CO2_emissions_Mt'] * 1000000 / region_aggregates['Population']
region_aggregates['Emissions_per_GDP$_ton'] = region_aggregates['CO2_emissions_Mt'] * 1000000 / region_aggregates['GDP, PPP (constant 2021 international $)']

# Add region identifiers
region_aggregates['ISO3'] = region_aggregates['Region'].apply(lambda x: f"REG_{x[:3]}")
region_aggregates['Country'] = region_aggregates['Region'].apply(lambda x: f"Region: {x}")
region_aggregates['EU_country'] = 'N/A'
region_aggregates['G20_country'] = 'N/A'

# Create world aggregates
print("Creating world aggregates...")
world_aggregates = emissions_df.groupby(['Year', 'Emissions_scope']).agg({
    'CO2_emissions_Mt': 'sum',
    'Population': 'sum',
    'GDP, PPP (constant 2021 international $)': 'sum'
}).reset_index()

# Calculate per capita and per GDP metrics for world
world_aggregates['Emissions_per_capita_ton'] = world_aggregates['CO2_emissions_Mt'] * 1000000 / world_aggregates['Population']
world_aggregates['Emissions_per_GDP$_ton'] = world_aggregates['CO2_emissions_Mt'] * 1000000 / world_aggregates['GDP, PPP (constant 2021 international $)']

# Add world identifiers
world_aggregates['ISO3'] = 'WLD'
world_aggregates['Country'] = 'World'
world_aggregates['Region'] = 'World'
world_aggregates['EU_country'] = 'N/A'
world_aggregates['G20_country'] = 'N/A'

# Create EU aggregates
print("Creating EU aggregates...")
eu_aggregates = emissions_df[emissions_df['EU_country'] == 'Yes'].groupby(['Year', 'Emissions_scope']).agg({
    'CO2_emissions_Mt': 'sum',
    'Population': 'sum',
    'GDP, PPP (constant 2021 international $)': 'sum'
}).reset_index()

# Calculate per capita and per GDP metrics for EU
eu_aggregates['Emissions_per_capita_ton'] = eu_aggregates['CO2_emissions_Mt'] * 1000000 / eu_aggregates['Population']
eu_aggregates['Emissions_per_GDP$_ton'] = eu_aggregates['CO2_emissions_Mt'] * 1000000 / eu_aggregates['GDP, PPP (constant 2021 international $)']

# Add EU identifiers
eu_aggregates['ISO3'] = 'EU'
eu_aggregates['Country'] = 'European Union'
eu_aggregates['Region'] = 'EU'
eu_aggregates['EU_country'] = 'Yes'
eu_aggregates['G20_country'] = 'N/A'

# Create G20 aggregates
print("Creating G20 aggregates...")
g20_aggregates = emissions_df[emissions_df['G20_country'] == 'Yes'].groupby(['Year', 'Emissions_scope']).agg({
    'CO2_emissions_Mt': 'sum',
    'Population': 'sum',
    'GDP, PPP (constant 2021 international $)': 'sum'
}).reset_index()

# Calculate per capita and per GDP metrics for G20
g20_aggregates['Emissions_per_capita_ton'] = g20_aggregates['CO2_emissions_Mt'] * 1000000 / g20_aggregates['Population']
g20_aggregates['Emissions_per_GDP$_ton'] = g20_aggregates['CO2_emissions_Mt'] * 1000000 / g20_aggregates['GDP, PPP (constant 2021 international $)']

# Add G20 identifiers
g20_aggregates['ISO3'] = 'G20'
g20_aggregates['Country'] = 'G20 Countries'
g20_aggregates['Region'] = 'G20'
g20_aggregates['EU_country'] = 'N/A'
g20_aggregates['G20_country'] = 'Yes'

# Combine all dataframes
print("Combining all dataframes...")
final_df = pd.concat([
    emissions_df,
    region_aggregates,
    world_aggregates,
    eu_aggregates,
    g20_aggregates
], ignore_index=True)

# Sort the dataframe
final_df = final_df.sort_values(['ISO3', 'Year', 'Emissions_scope'])

# Save the combined dataframe
final_df.to_csv(f"{output_directory}/combined_data.csv", index=False)

# Print verification
print(f"Combined data saved to {output_directory}/combined_data.csv")
print(f"Total rows: {len(final_df)}")
print(f"Unique countries: {final_df['Country'].nunique()}")
print(f"Year range: {final_df['Year'].min()} to {final_df['Year'].max()}")
print(f"Emissions scopes: {', '.join(final_df['Emissions_scope'].unique())}")
print(f"Added {len(region_aggregates)} region aggregate rows")
print(f"Added {len(world_aggregates)} world aggregate rows")
print(f"Added {len(eu_aggregates)} EU aggregate rows")
print(f"Added {len(g20_aggregates)} G20 aggregate rows")
print("\nFirst 50 rows of the combined dataframe:")
print(final_df.head(50).to_string())