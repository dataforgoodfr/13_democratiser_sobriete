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

    # Manually set the country name for select countries:
    iso_mapping.loc[iso_mapping['ISO2'] == 'US', 'Country'] = 'United States of America'
    iso_mapping.loc[iso_mapping['ISO3'] == 'USA', 'Country'] = 'United States of America'
    iso_mapping.loc[iso_mapping['ISO2'] == 'SO', 'Country'] = 'Somalia'
    iso_mapping.loc[iso_mapping['ISO3'] == 'SOM', 'Country'] = 'Somalia'
    iso_mapping.loc[iso_mapping['ISO2'] == 'NA', 'Country'] = 'Namibia'
    iso_mapping.loc[iso_mapping['ISO3'] == 'NAM', 'Country'] = 'Namibia'
    iso_mapping.loc[iso_mapping['ISO2'] == 'TR', 'Country'] = 'Turkey'
    iso_mapping.loc[iso_mapping['ISO3'] == 'TUR', 'Country'] = 'Turkey'
    iso_mapping.loc[iso_mapping['ISO2'] == 'SS', 'Country'] = 'South Sudan'
    iso_mapping.loc[iso_mapping['ISO3'] == 'SSD', 'Country'] = 'South Sudan'

    # Print to verify
    print("ISO Mapping:")
    print(iso_mapping[['ISO3', 'ISO2', 'Country']].head())

    return iso_mapping[['ISO3', 'ISO2', 'Country']]

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

def load_gdp_data():
    """Load and process GDP PPP data, reshaping it from wide to long format."""
    # Load the data, using the first row as the header as specified.
    gdp_wide = pd.read_excel(
        f"{DATA_DIR}/2025-04-21_GDP _PPP constant 2021 US$_per country ISO Code.xlsx",
        sheet_name='Data',
        header=0
    )

    # Clean column names to remove leading/trailing spaces.
    gdp_wide.columns = gdp_wide.columns.str.strip()

    # Rename 'Country Code' to 'ISO3'.
    if 'Country Code' in gdp_wide.columns:
        gdp_wide.rename(columns={'Country Code': 'ISO3'}, inplace=True)
    else:
        raise ValueError(f"'Country Code' not found. Available columns are: {gdp_wide.columns.tolist()}")

    # Identify the columns to keep as IDs.
    id_vars = [col for col in ['Country Name', 'ISO3', 'Series Name', 'Series Code'] if col in gdp_wide.columns]

    # Reshape the dataframe from wide to long.
    gdp_long = pd.melt(
        gdp_wide,
        id_vars=id_vars,
        var_name='Year_str',
        value_name='GDP_PPP'
    )
    
    # Extract the 4-digit year from the 'Year_str' column (e.g., from '1970 [YR1970]').
    gdp_long['Year'] = gdp_long['Year_str'].astype(str).str.extract(r'(\d{4})').astype(int)
    
    # Convert GDP values to numeric, coercing errors (like '..') to NaN, then drop rows with no GDP data.
    gdp_long['GDP_PPP'] = pd.to_numeric(gdp_long['GDP_PPP'], errors='coerce')
    gdp_long.dropna(subset=['GDP_PPP'], inplace=True)

    # Return the cleaned, final dataframe.
    return gdp_long[['ISO3', 'Year', 'GDP_PPP']]

def load_population_data():
    """Load and process population data from 1970 to 2050."""
    pop = pd.read_excel(f"{DATA_DIR}/2025-04-21_Population per Country ISO code_1970-2050.xlsx",
                       sheet_name="unpopulation_dataportal_2025042")
    pop = pop[['Iso3', 'Location', 'Time', 'Value']]
    pop.rename(columns={'Iso3': 'ISO3', 'Location': 'Country', 'Time': 'Year', 'Value': 'Population'}, inplace=True)

    # Print to verify
    print("Population Data (all years up to 2050):")
    print(pop[['ISO3', 'Country', 'Year', 'Population']].head())

    # Return all years up to 2050
    return pop[pop['Year'] <= 2050]

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
        'Cumulative_population': 'sum',
        'GDP_PPP': 'sum',
        'capacity_absolute': 'sum'
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
    population_data = load_population_data()
    emissions_data = load_emissions_data()
    consumption_emissions_data = load_consumption_emissions_data()
    gdp_data = load_gdp_data()

    # Filter data to only include years >= 1990
    population_data = population_data[population_data['Year'] >= 1990]
    emissions_data = emissions_data[emissions_data['Year'] >= 1990]
    consumption_emissions_data = consumption_emissions_data[consumption_emissions_data['Year'] >= 1990]
    gdp_data = gdp_data[gdp_data['Year'] >= 1990]

    # Print verification of year filtering
    print("\nVerifying year filtering (>= 1990):")
    print(f"Population data year range: {population_data['Year'].min()} to {population_data['Year'].max()}")
    print(f"Emissions data year range: {emissions_data['Year'].min()} to {emissions_data['Year'].max()}")
    print(f"Consumption emissions data year range: {consumption_emissions_data['Year'].min()} to {consumption_emissions_data['Year'].max()}")
    print(f"GDP data year range: {gdp_data['Year'].min()} to {gdp_data['Year'].max()}")

    # Merge ISO2 codes and country names into the dataframes
    iso2_mapping = iso_mapping.set_index('ISO3')['ISO2'].to_dict()
    country_mapping = iso_mapping.set_index('ISO3')['Country'].to_dict()

    # Ensure Namibia's ISO2 code is correctly assigned
    iso2_mapping['NAM'] = 'NA'

    # Add ISO2 mapping to GDP data before the merge
    gdp_data['ISO2'] = gdp_data['ISO3'].map(iso2_mapping)

    # Print verification of ISO mapping
    print("\nVerifying ISO mapping for population data:")
    print("Sample of ISO2 mapping dictionary:")
    sample_iso2 = dict(list(iso2_mapping.items())[:5])
    print(sample_iso2)
    
    # Apply mappings to population data
    population_data['ISO2'] = population_data['ISO3'].map(iso2_mapping)
    population_data['Country'] = population_data['ISO3'].map(country_mapping)

    # Print verification of mapped population data
    print("\nSample of mapped population data:")
    print(population_data[['ISO3', 'ISO2', 'Country', 'Year', 'Population']].head())
    
    # Check for any missing ISO2 values and print a warning
    if population_data['ISO2'].isnull().any():
        print("\nWarning: Missing ISO2 values in population data:")
        print(population_data[population_data['ISO2'].isnull()][['ISO3', 'Year', 'Population']].head())

    emissions_data['ISO2'] = emissions_data['ISO3'].map(iso2_mapping)
    emissions_data['Country'] = emissions_data['ISO3'].map(country_mapping)

    consumption_emissions_data['ISO2'] = consumption_emissions_data['ISO3'].map(iso2_mapping)
    consumption_emissions_data['Country'] = consumption_emissions_data['ISO3'].map(country_mapping)

    # Print to verify
    print("Emissions Data with ISO2 and Country:")
    print(emissions_data[['ISO3', 'ISO2', 'Country', 'Year', 'Annual_CO2_emissions_Mt']].head())

    print("Consumption Emissions Data with ISO2 and Country:")
    print(consumption_emissions_data[['ISO3', 'ISO2', 'Country', 'Year', 'Annual_CO2_emissions_Mt']].head())

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

    # Merge GDP data
    emissions_df = emissions_df.merge(gdp_data.drop(columns=['Country'], errors='ignore'), on=['ISO3', 'Year', 'ISO2'], how='left')

    # Calculate cumulative emissions for each scope
    emissions_df['Cumulative_CO2_emissions_Mt'] = emissions_df.groupby(['ISO3', 'Region', 'Emissions_scope'])['Annual_CO2_emissions_Mt'].cumsum()

    # Calculate cumulative population for each scope
    emissions_df['Cumulative_population'] = emissions_df.groupby(['ISO3', 'Region', 'Emissions_scope'])['Population'].cumsum()

    # Calculate absolute capacity, which is inversely proportional to GDP per capita (p=1).
    # This gives a larger budget share to less wealthy nations.
    emissions_df['gdp_per_capita'] = emissions_df['GDP_PPP'] / emissions_df['Population']
    emissions_df['capacity_absolute'] = emissions_df['Population'] / emissions_df['gdp_per_capita']
    
    # Handle cases with no GDP data (NaN) or infinite values by setting their capacity to 0.
    emissions_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    emissions_df['capacity_absolute'].fillna(0, inplace=True)
    emissions_df.drop(columns=['gdp_per_capita'], inplace=True)

    # Before aggregating, filter for rows with available emissions data, but ensure 2050 is kept
    # for population-based calculations.
    aggregation_df = emissions_df[
        (emissions_df['Annual_CO2_emissions_Mt'].notna()) | (emissions_df['Year'] == 2050)
    ].copy()

    # Create world aggregate first
    world_aggregates = create_aggregates(aggregation_df, ['Year', 'Emissions_scope'], 'All', 'WLD', 'World')

    # Create aggregates for each region
    region_aggregates = []
    for region in aggregation_df['Region'].unique():
        if pd.notna(region):  # Skip if region is null
            region_agg = create_aggregates(
                aggregation_df[aggregation_df['Region'] == region],
                ['Year', 'Emissions_scope'],
                'All',  # Set Country to "All"
                region,  # Use region name as ISO2 and ISO3
                region   # Keep region name in Region column
            )
            region_aggregates.append(region_agg)
    region_aggregates = pd.concat(region_aggregates, ignore_index=True)

    # Create EU aggregate
    eu_aggregates = create_aggregates(
        aggregation_df[aggregation_df['EU_country'] == 'Yes'],
        ['Year', 'Emissions_scope'],
        'All',
        'EU',
        'European Union'
    )

    # Create G20 aggregate
    g20_aggregates = create_aggregates(
        aggregation_df[aggregation_df['G20_country'] == 'Yes'],
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

    # Calculate share_of_capacity from the absolute values
    world_capacity = final_df.loc[final_df['ISO2'] == 'WLD', ['Year', 'Emissions_scope', 'capacity_absolute']].copy()
    world_capacity.rename(columns={'capacity_absolute': 'world_total_capacity'}, inplace=True)
    
    final_df = final_df.merge(world_capacity, on=['Year', 'Emissions_scope'], how='left')
    # Calculate share, handling cases where the total is zero
    final_df['share_of_capacity'] = np.where(
        final_df['world_total_capacity'] > 0,
        final_df['capacity_absolute'] / final_df['world_total_capacity'],
        0
    )
    final_df.drop(columns=['capacity_absolute', 'world_total_capacity'], inplace=True)

    # Calculate share of GDP
    world_gdp = final_df.loc[final_df['ISO2'] == 'WLD', ['Year', 'Emissions_scope', 'GDP_PPP']].copy()
    world_gdp.rename(columns={'GDP_PPP': 'world_total_gdp'}, inplace=True)
    
    final_df = final_df.merge(world_gdp, on=['Year', 'Emissions_scope'], how='left')
    final_df['share_of_GDP_PPP'] = np.where(
        final_df['world_total_gdp'] > 0,
        final_df['GDP_PPP'] / final_df['world_total_gdp'],
        0
    )
    final_df.drop(columns=['world_total_gdp'], inplace=True)

    # Calculate share of Population
    world_pop = final_df.loc[final_df['ISO2'] == 'WLD', ['Year', 'Emissions_scope', 'Population']].copy()
    world_pop.rename(columns={'Population': 'world_total_population'}, inplace=True)
    
    final_df = final_df.merge(world_pop, on=['Year', 'Emissions_scope'], how='left')
    final_df['share_of_population'] = np.where(
        final_df['world_total_population'] > 0,
        final_df['Population'] / final_df['world_total_population'],
        0
    )
    final_df.drop(columns=['world_total_population'], inplace=True)

    # Calculate share of cumulative population for each scope, per year
    final_df['Share_of_cumulative_population'] = np.nan
    for scope in ['Territory', 'Consumption']:
        # Get world cumulative population per year for this scope
        world_cum_pop = final_df[(final_df['ISO2'] == 'WLD') & (final_df['Emissions_scope'] == scope)][['Year', 'Cumulative_population']].set_index('Year')['Cumulative_population']
        # Assign share for all rows with this scope
        mask = final_df['Emissions_scope'] == scope
        final_df.loc[mask, 'Share_of_cumulative_population'] = final_df[mask].apply(
            lambda row: 1 if row['ISO2'] == 'WLD' else (row['Cumulative_population'] / world_cum_pop.get(row['Year'], np.nan)),
            axis=1
        )
    
    # --- SANITY CHECKS ---
    aggregate_iso2s = ['WLD', 'EU', 'G20'] + list(ipcc_regions['IPCC_Region_Intermediate'].unique())
    country_df = final_df[~final_df['ISO2'].isin(aggregate_iso2s)]

    # The universe for all share calculations is based on the data available in aggregation_df.
    # We must apply the same filter to the country data before running the sanity checks.
    country_df_for_checks = country_df[
        (country_df['Annual_CO2_emissions_Mt'].notna())
    ].copy()

    # Check share_of_capacity
    capacity_check = country_df_for_checks.groupby(['Year', 'Emissions_scope'])['share_of_capacity'].sum().reset_index()
    print("\n--- Sanity Check: Sum of share_of_capacity for all countries ---")
    check_failed = capacity_check[~np.isclose(capacity_check['share_of_capacity'], 1.0, atol=1e-5)]
    if not check_failed.empty:
        print("Check failed for these Year/Scope combinations:")
        print(check_failed)
    else:
        print("Check passed. All years/scopes sum to ~1.0.")
    print("----------------------------------------------------------------")

    # Check share_of_GDP_PPP
    gdp_check = country_df_for_checks.groupby(['Year', 'Emissions_scope'])['share_of_GDP_PPP'].sum().reset_index()
    print("\n--- Sanity Check: Sum of share_of_GDP_PPP for all countries ---")
    check_failed_gdp = gdp_check[~np.isclose(gdp_check['share_of_GDP_PPP'], 1.0, atol=1e-5)]
    if not check_failed_gdp.empty:
        print("Check failed for these Year/Scope combinations:")
        print(check_failed_gdp)
    else:
        print("Check passed. All years/scopes sum to ~1.0.")
    print("-------------------------------------------------------------------")
    
    # Check share_of_population
    pop_check = country_df_for_checks.groupby(['Year', 'Emissions_scope'])['share_of_population'].sum().reset_index()
    print("\n--- Sanity Check: Sum of share_of_population for all countries ---")
    check_failed_pop = pop_check[~np.isclose(pop_check['share_of_population'], 1.0, atol=1e-5)]
    if not check_failed_pop.empty:
        print("Check failed for these Year/Scope combinations:")
        print(check_failed_pop)
    else:
        print("Check passed. All years/scopes sum to ~1.0.")
    print("--------------------------------------------------------------------")

    # --- Sanity check for Share_of_cumulative_population ---
    pop_share_check = country_df_for_checks.groupby(['Year', 'Emissions_scope'])['Share_of_cumulative_population'].sum().reset_index()
    print("\n--- Sanity Check: Sum of Share_of_cumulative_population for all countries ---")
    check_failed_pop_share = pop_share_check[~np.isclose(pop_share_check['Share_of_cumulative_population'], 1.0, atol=1e-5)]
    if not check_failed_pop_share.empty:
        print("Check failed for these Year/Scope combinations:")
        print(check_failed_pop_share)
    else:
        print("Check passed. All years/scopes sum to ~1.0.")
    print("--------------------------------------------------------------------------")

    # Format as percentage string (optional, or keep as float if preferred)
    final_df['Share_of_cumulative_population'] = (final_df['Share_of_cumulative_population'].astype(float) * 100).round(2).astype(str) + '%'

    # Filter out rows where Annual_CO2_emissions_Mt is 0 or null, but keep year 2050
    final_df = final_df[
        (final_df['Annual_CO2_emissions_Mt'].notna() & final_df['Annual_CO2_emissions_Mt'] != 0) |
        (final_df['Year'] == 2050)
    ]

    # Add Data_type and Scenario_ID columns
    final_df['Data_type'] = 'Historical'
    final_df['Scenario_ID'] = None

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
