import pandas as pd
import numpy as np

# Constants
DATA_DIR = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data'
OUTPUT_DIR = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'

def load_iso_codes_mapping():
    """Load and process ISO codes mapping data."""
    iso_mapping = pd.read_excel(f"{DATA_DIR}/28-04-2025_ISO_Codes_Mapping.xlsx")
    iso_mapping.rename(columns={'Alpha-2 code': 'ISO2', 'Alpha-3 code': 'ISO3'}, inplace=True)
    # Ensure 'NA' is not interpreted as a missing value
    iso_mapping['ISO2'] = iso_mapping['ISO2'].fillna('')
    return iso_mapping[['ISO3', 'ISO2']]

def load_ipcc_regions():
    """Load and process IPCC region mapping data."""
    regions = pd.read_excel(f"{DATA_DIR}/2024-04-21_IPCC Regional Breakdown_ISO Country Code.xlsx",
                          sheet_name="Full_mapping")
    regions = regions[['Intermediate level (10)', 'ISO codes']]
    regions.rename(columns={'Intermediate level (10)': 'IPCC_Region_Intermediate', 'ISO codes': 'ISO3'}, inplace=True)

    # Split ISO codes and create separate rows
    expanded_regions = []
    for _, row in regions.iterrows():
        for iso in row['ISO3'].split(','):
            expanded_regions.append({
                'IPCC_Region_Intermediate': row['IPCC_Region_Intermediate'],
                'ISO3': iso.strip()
            })
    return pd.DataFrame(expanded_regions)

def load_eu_g20_mapping():
    """Load EU and G20 country mappings."""
    mapping = pd.read_excel(f"{DATA_DIR}/2024-04-21_IPCC Regional Breakdown_ISO Country Code.xlsx",
                          sheet_name="G20_EU_Countries ",
                          header=0)
    return mapping[['ISO3', 'EU_country', 'G20_country']]

def load_historical_population_data():
    """Load and process historical population data from emissions data."""
    emissions = pd.read_excel(f"{DATA_DIR}/2025-04-22_CO2 Emissions_All Countries_ISO Code_1750-2023.xlsx",
                            sheet_name="GCB2024v17_MtCO2_flat")
    emissions = emissions[['Country', 'ISO 3166-1 alpha-3', 'Year', 'Total', 'Per Capita']]
    emissions.rename(columns={
        'ISO 3166-1 alpha-3': 'ISO3',
        'Per Capita': 'Per_Capita',
        'Total': 'Annual_CO2_emissions_Mt'
    }, inplace=True)

    # Calculate population
    emissions['Population'] = round(((emissions['Annual_CO2_emissions_Mt'] / emissions['Per_Capita']) * 1000000), 0)

    # Select relevant columns
    return emissions[['ISO3', 'Country', 'Year', 'Population']]

def load_forecasted_population_data():
    """Load and process forecasted population data for 2050."""
    pop = pd.read_excel(f"{DATA_DIR}/2025-04-21_Population per Country ISO code_1970-2050.xlsx",
                       sheet_name="unpopulation_dataportal_2025042")
    pop = pop[['Iso3', 'Location', 'Time', 'Value']]
    pop.rename(columns={'Iso3': 'ISO3', 'Location': 'Country', 'Time': 'Year', 'Value': 'Population'}, inplace=True)

    # Filter for the year 2050
    return pop[pop['Year'] == 2050]

def load_emissions_data():
    """Load and process CO2 emissions data."""
    emissions = pd.read_excel(f"{DATA_DIR}/2025-04-22_CO2 Emissions_All Countries_ISO Code_1750-2023.xlsx",
                            sheet_name="GCB2024v17_MtCO2_flat")
    emissions = emissions[['Country', 'ISO 3166-1 alpha-3', 'Year', 'Total']]
    emissions.rename(columns={'ISO 3166-1 alpha-3': 'ISO3', 'Total': 'Annual_CO2_emissions_Mt'}, inplace=True)
    return emissions

def load_consumption_emissions_data():
    """Load and process consumption emissions data."""
    cons_emissions = pd.read_excel(f"{DATA_DIR}/2025-04-22_Consumption emissions MtCO2_ISO code.xlsx",
                                 sheet_name="GCB2024v17_MtCO2_flat")
    cons_emissions = cons_emissions[['Country', 'ISO 3166-1 alpha-3', 'Year', 'CO2_Consumption_emissions in Mt']]
    cons_emissions.rename(columns={
        'ISO 3166-1 alpha-3': 'ISO3',
        'CO2_Consumption_emissions in Mt': 'Annual_CO2_emissions_Mt'
    }, inplace=True)

    # Clean consumption emissions values
    cons_emissions['Annual_CO2_emissions_Mt'] = cons_emissions['Annual_CO2_emissions_Mt'].apply(
        lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else float(x)
    ).round(2)

    return cons_emissions

def create_measure_dataframe(df, measure_name, value_column):
    """Create a standardized measure dataframe."""
    measure_df = df.copy()
    measure_df['Measure'] = measure_name
    measure_df.rename(columns={value_column: 'Value'}, inplace=True)
    return measure_df[['ISO2', 'Country', 'Year', 'Measure', 'Value']]

def calculate_derived_metrics(df):
    """Calculate per capita metrics."""
    df['Emissions_per_capita_ton'] = df['Annual_CO2_emissions_Mt'] * 1000000 / df['Population']
    return df

def calculate_cumulative_emissions(df):
    """Calculate cumulative emissions for each country and scope."""
    df['Cumulative_CO2_emissions_Mt'] = df.groupby(['ISO2', 'Region'])['CO2_emissions_Mt'].cumsum()
    df['Cumulative_Consumption_CO2_emissions_Mt'] = df.groupby(['ISO2', 'Region'])['Consumption_CO2_emissions_Mt'].cumsum()
    return df

def create_aggregates(df, group_cols, agg_name, iso_code, region_name):
    """Create aggregates for regions, world, EU, or G20."""
    aggregates = df.groupby(group_cols).agg({
        'Annual_CO2_emissions_Mt': 'sum',
        'Cumulative_CO2_emissions_Mt': 'sum',
        'Population': 'sum',
        'Cumulative_population': 'sum'
    }).reset_index()

    # Calculate metrics
    aggregates['Emissions_per_capita_ton'] = aggregates['Annual_CO2_emissions_Mt'] * 1000000 / aggregates['Population']

    # Add identifiers
    aggregates['ISO2'] = iso_code
    aggregates['ISO3'] = iso_code
    aggregates['Country'] = agg_name
    aggregates['Region'] = region_name
    aggregates['EU_country'] = 'N/A'
    aggregates['G20_country'] = 'N/A'

    return aggregates

def main():
    # Load all data
    iso_mapping = load_iso_codes_mapping()
    ipcc_regions = load_ipcc_regions()
    eu_g20_mapping = load_eu_g20_mapping()
    historical_population_data = load_historical_population_data()
    forecasted_population_data = load_forecasted_population_data()
    emissions_data = load_emissions_data()
    consumption_emissions_data = load_consumption_emissions_data()

    # Combine historical and forecasted population data
    population_data = pd.concat([historical_population_data, forecasted_population_data], ignore_index=True)

    # Merge ISO2 codes into the dataframes
    iso2_mapping = iso_mapping.set_index('ISO3')['ISO2'].to_dict()
    population_data['ISO2'] = population_data['ISO3'].map(iso2_mapping)
    emissions_data['ISO2'] = emissions_data['ISO3'].map(iso2_mapping)
    consumption_emissions_data['ISO2'] = consumption_emissions_data['ISO3'].map(iso2_mapping)

    # Explicitly handle 'NA' for Namibia
    population_data.loc[population_data['ISO3'] == 'NAM', 'ISO2'] = 'NA'
    emissions_data.loc[emissions_data['ISO3'] == 'NAM', 'ISO2'] = 'NA'
    consumption_emissions_data.loc[consumption_emissions_data['ISO3'] == 'NAM', 'ISO2'] = 'NA'

    # Filter and add metadata to emissions data
    valid_iso3_codes = set(ipcc_regions['ISO3'].unique())
    emissions_data = emissions_data[emissions_data['ISO3'].isin(valid_iso3_codes)]
    consumption_emissions_data = consumption_emissions_data[consumption_emissions_data['ISO3'].isin(valid_iso3_codes)]

    # Add region and country flags
    region_mapping = ipcc_regions.set_index('ISO3')['IPCC_Region_Intermediate'].to_dict()
    emissions_data['Region'] = emissions_data['ISO3'].map(region_mapping)
    consumption_emissions_data['Region'] = consumption_emissions_data['ISO3'].map(region_mapping)

    eu_mapping = eu_g20_mapping.set_index('ISO3')['EU_country'].to_dict()
    g20_mapping = eu_g20_mapping.set_index('ISO3')['G20_country'].to_dict()

    emissions_data['EU_country'] = emissions_data['ISO3'].map(eu_mapping).fillna('No')
    emissions_data['G20_country'] = emissions_data['ISO3'].map(g20_mapping).fillna('No')
    consumption_emissions_data['EU_country'] = consumption_emissions_data['ISO3'].map(eu_mapping).fillna('No')
    consumption_emissions_data['G20_country'] = consumption_emissions_data['ISO3'].map(g20_mapping).fillna('No')

    # Create a base dataframe with all population data
    base_df = population_data[population_data['ISO3'].isin(valid_iso3_codes)].copy()
    base_df['Region'] = base_df['ISO3'].map(region_mapping)
    base_df['EU_country'] = base_df['ISO3'].map(eu_mapping).fillna('No')
    base_df['G20_country'] = base_df['ISO3'].map(g20_mapping).fillna('No')

    # Create territory and consumption dataframes with all years
    territory_df = base_df.copy()
    territory_df['Emissions_scope'] = 'Territory'
    territory_df = territory_df.merge(
        emissions_data[['ISO3', 'Year', 'Annual_CO2_emissions_Mt']],
        on=['ISO3', 'Year'],
        how='left'
    )

    consumption_df = base_df.copy()
    consumption_df['Emissions_scope'] = 'Consumption'
    consumption_df = consumption_df.merge(
        consumption_emissions_data[['ISO3', 'Year', 'Annual_CO2_emissions_Mt']],
        on=['ISO3', 'Year'],
        how='left'
    )

    # Calculate per capita emissions
    territory_df['Emissions_per_capita_ton'] = territory_df['Annual_CO2_emissions_Mt'] * 1000000 / territory_df['Population']
    consumption_df['Emissions_per_capita_ton'] = consumption_df['Annual_CO2_emissions_Mt'] * 1000000 / consumption_df['Population']

    # Combine the two dataframes
    emissions_df = pd.concat([territory_df, consumption_df], ignore_index=True)

    # Calculate cumulative emissions for each scope
    emissions_df['Cumulative_CO2_emissions_Mt'] = emissions_df.groupby(['ISO3', 'Region', 'Emissions_scope'])['Annual_CO2_emissions_Mt'].cumsum()

    # Calculate cumulative population for each scope
    emissions_df['Cumulative_population'] = emissions_df.groupby(['ISO3', 'Region', 'Emissions_scope'])['Population'].cumsum()

    # Create world aggregate first
    world_aggregates = create_aggregates(emissions_df, ['Year', 'Emissions_scope'], 'All', 'WLD', 'World')

    # Create aggregates for each region
    region_aggregates = []
    for region in emissions_df['Region'].unique():
        if pd.notna(region):  # Skip if region is null
            region_agg = create_aggregates(
                emissions_df[emissions_df['Region'] == region],
                ['Year', 'Emissions_scope'],
                'All',  # Set Country to "All"
                region,  # Use region name as ISO2 and ISO3
                region   # Keep region name in Region column
            )
            region_aggregates.append(region_agg)
    region_aggregates = pd.concat(region_aggregates, ignore_index=True)

    # Create EU aggregate
    eu_aggregates = create_aggregates(
        emissions_df[emissions_df['EU_country'] == 'Yes'],
        ['Year', 'Emissions_scope'],
        'All',
        'EU',
        'European Union'
    )

    # Create G20 aggregate
    g20_aggregates = create_aggregates(
        emissions_df[emissions_df['G20_country'] == 'Yes'],
        ['Year', 'Emissions_scope'],
        'All',
        'G20',
        'G20 Countries'
    )

    # Combine all dataframes
    final_df = pd.concat([
        emissions_df,
        region_aggregates,
        world_aggregates,
        eu_aggregates,
        g20_aggregates
    ], ignore_index=True)

    # Calculate share of cumulative population for each scope
    for scope in ['Territory', 'Consumption']:
        # Get world's cumulative population for this scope from the world aggregate
        world_cumulative_pop = world_aggregates[
            world_aggregates['Emissions_scope'] == scope
        ]['Cumulative_population'].iloc[0]

        # Calculate share for this scope
        final_df.loc[final_df['Emissions_scope'] == scope, 'Share_of_cumulative_population'] = (
            final_df[final_df['Emissions_scope'] == scope]['Cumulative_population'] / world_cumulative_pop
        )

    # Filter out rows where Annual_CO2_emissions_Mt is 0 or null, but keep year 2050
    final_df = final_df[
        (final_df['Annual_CO2_emissions_Mt'].notna() & final_df['Annual_CO2_emissions_Mt'] != 0) |
        (final_df['Year'] == 2050)
    ]

    # Sort and save
    final_df = final_df.sort_values(['ISO3', 'Year', 'Emissions_scope'])
    final_df.to_csv(f"{OUTPUT_DIR}/combined_data.csv", index=False)

    # Print verification
    print(f"Combined data saved to {OUTPUT_DIR}/combined_data.csv")
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

if __name__ == "__main__":
    main()
