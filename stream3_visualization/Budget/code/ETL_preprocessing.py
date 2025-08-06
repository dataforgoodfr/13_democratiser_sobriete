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

def load_consumption_emissions_data_1970_1989():
    """Load and process consumption emissions data from 1970-1989."""
    print("Loading consumption emissions data from 1970-1989...")
    
    # Load the CSV file
    cons_emissions_early = pd.read_csv(f"{DATA_DIR}/2025-07-11_Consumption data scaled from 1970_edited.xlsx - eoraScaled-PRIMAPcba-19702021.csv")
    
    # Print initial info
    print(f"Original data shape: {cons_emissions_early.shape}")
    print("Original columns:", cons_emissions_early.columns.tolist())
    
    # Get year columns (all columns except 'country' and 'iso3c')
    year_columns = [col for col in cons_emissions_early.columns if col not in ['country', 'iso3c']]
    print(f"Year columns found: {year_columns[:5]}...{year_columns[-5:]}")  # Show first and last 5 years
    
    # Reshape from wide to long format
    cons_emissions_long = pd.melt(
        cons_emissions_early,
        id_vars=['country', 'iso3c'],
        value_vars=year_columns,
        var_name='Year',
        value_name='Annual_CO2_emissions_Mt'
    )
    
    # Convert Year to integer and filter for 1970-1989
    cons_emissions_long['Year'] = cons_emissions_long['Year'].astype(int)
    cons_emissions_long = cons_emissions_long[(cons_emissions_long['Year'] >= 1970) & (cons_emissions_long['Year'] <= 1989)]
    
    # Rename columns to match expected format
    cons_emissions_long.rename(columns={
        'country': 'Country',
        'iso3c': 'ISO3'
    }, inplace=True)
    
    # Convert emissions to numeric and handle any non-numeric values
    cons_emissions_long['Annual_CO2_emissions_Mt'] = pd.to_numeric(
        cons_emissions_long['Annual_CO2_emissions_Mt'], 
        errors='coerce'
    )
    
    # CRITICAL: Convert from thousand tons (kt) to million tons (MtCO2) to match 1990+ data format
    # Example: France 1989 = 507193.8005 kt -> 507.19 MtCO2
    cons_emissions_long['Annual_CO2_emissions_Mt'] = cons_emissions_long['Annual_CO2_emissions_Mt'] / 1000
    
    # Drop rows with NaN emissions
    initial_rows = len(cons_emissions_long)
    cons_emissions_long = cons_emissions_long.dropna(subset=['Annual_CO2_emissions_Mt'])
    dropped_rows = initial_rows - len(cons_emissions_long)
    
    print(f"Reshaped data shape: {cons_emissions_long.shape}")
    print(f"Dropped {dropped_rows} rows with NaN emissions")
    print(f"APPLIED SCALING: Converted from thousand tons (kt) to MtCO2 (divided by 1,000)")
    print(f"Year range: {cons_emissions_long['Year'].min()} to {cons_emissions_long['Year'].max()}")
    print(f"Unique countries: {cons_emissions_long['Country'].nunique()}")
    
    # Print sample data to verify scaling
    print("\nSample of 1970-1989 consumption emissions data (after scaling to MtCO2):")
    print(cons_emissions_long[['Country', 'ISO3', 'Year', 'Annual_CO2_emissions_Mt']].head(10))
    
    # Show some specific examples to verify scaling
    france_sample = cons_emissions_long[cons_emissions_long['Country'].str.contains('France', case=False, na=False)]
    if not france_sample.empty:
        print(f"\nFrance example (should be ~507.19 MtCO2 for 1989):")
        print(france_sample[france_sample['Year'] == 1989][['Country', 'Year', 'Annual_CO2_emissions_Mt']])
    
    return cons_emissions_long

def load_consumption_emissions_data():
    """Load and process consumption emissions data from 1990 onward and combine with 1970-1989 data, keeping only countries with complete data for the entire period."""
    print("Loading consumption emissions data from 1990 onward...")
    
    # Load 1990+ data
    cons_emissions_1990plus = pd.read_excel(f"{DATA_DIR}/2025-04-22_Consumption emissions MtCO2_ISO code.xlsx",
                                          sheet_name="GCB2024v17_MtCO2_flat")
    
    print(f"1990+ data shape: {cons_emissions_1990plus.shape}")
    print("1990+ data columns:", cons_emissions_1990plus.columns.tolist())
    
    cons_emissions_1990plus = cons_emissions_1990plus[['Country', 'ISO 3166-1 alpha-3', 'Year', 'CO2_Consumption_emissions in Mt']]
    cons_emissions_1990plus.rename(columns={
        'ISO 3166-1 alpha-3': 'ISO3',
        'CO2_Consumption_emissions in Mt': 'Annual_CO2_emissions_Mt'
    }, inplace=True)

    # Clean consumption emissions values
    cons_emissions_1990plus['Annual_CO2_emissions_Mt'] = cons_emissions_1990plus['Annual_CO2_emissions_Mt'].apply(
        lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else float(x)
    ).round(2)
    
    print(f"1990+ data year range: {cons_emissions_1990plus['Year'].min()} to {cons_emissions_1990plus['Year'].max()}")
    print(f"1990+ data unique countries: {cons_emissions_1990plus['Country'].nunique()}")
    
    # Load 1970-1989 data
    cons_emissions_1970_1989 = load_consumption_emissions_data_1970_1989()

    # --- Filter to only countries present in both datasets ---
    iso3_1970_1989 = set(cons_emissions_1970_1989['ISO3'].unique())
    iso3_1990plus = set(cons_emissions_1990plus['ISO3'].unique())
    common_iso3 = iso3_1970_1989 & iso3_1990plus
    print(f"Countries with complete data for 1970-2022: {len(common_iso3)}")
    
    cons_emissions_1970_1989 = cons_emissions_1970_1989[cons_emissions_1970_1989['ISO3'].isin(common_iso3)]
    cons_emissions_1990plus = cons_emissions_1990plus[cons_emissions_1990plus['ISO3'].isin(common_iso3)]

    # Combine both datasets
    print("\nCombining 1970-1989 and 1990+ consumption emissions data...")
    combined_cons_emissions = pd.concat([cons_emissions_1970_1989, cons_emissions_1990plus], ignore_index=True)
    
    # Sort by Country and Year
    combined_cons_emissions = combined_cons_emissions.sort_values(['Country', 'Year'])
    
    print(f"Combined data shape: {combined_cons_emissions.shape}")
    print(f"Combined data year range: {combined_cons_emissions['Year'].min()} to {combined_cons_emissions['Year'].max()}")
    print(f"Combined data unique countries: {combined_cons_emissions['Country'].nunique()}")
    
    # Check for overlapping years between datasets
    years_1970_1989 = set(cons_emissions_1970_1989['Year'].unique())
    years_1990plus = set(cons_emissions_1990plus['Year'].unique())
    overlap_years = years_1970_1989.intersection(years_1990plus)
    
    if overlap_years:
        print(f"WARNING: Overlapping years found: {sorted(overlap_years)}")
    else:
        print("No overlapping years between datasets - good!")
    
    # Print sample of combined data
    print("\nSample of combined consumption emissions data:")
    print(combined_cons_emissions[['Country', 'ISO3', 'Year', 'Annual_CO2_emissions_Mt']].head(10))
    
    print("\nTail of combined consumption emissions data:")
    print(combined_cons_emissions[['Country', 'ISO3', 'Year', 'Annual_CO2_emissions_Mt']].tail(10))

    return combined_cons_emissions

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
        'GDP_PPP': 'sum'
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

    # Filter data to only include years >= 1970 (now that we have consumption emissions from 1970)
    population_data = population_data[population_data['Year'] >= 1970]
    emissions_data = emissions_data[emissions_data['Year'] >= 1970]
    # consumption_emissions_data now includes data from 1970-1989, already filtered in the function
    gdp_data = gdp_data[gdp_data['Year'] >= 1970]

    # Print verification of year filtering
    print("\nVerifying year filtering (>= 1970):")
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

    # Calculate absolute capacity based on cumulative population (1970-2050) and latest GDP per capita
    # This gives higher capacity to countries with large cumulative populations but low current wealth
    emissions_df['gdp_per_capita'] = emissions_df['GDP_PPP'] / emissions_df['Population']
    
    # For each country, get their cumulative population from 1970 to 2050 and latest GDP per capita
    # We'll calculate this after aggregation to get the proper cumulative values
    emissions_df.drop(columns=['gdp_per_capita'], inplace=True)

    # Before aggregating, filter for rows with available emissions data, but ensure 2050 is kept
    # for population-based calculations.
    aggregation_df = emissions_df[
        (emissions_df['Annual_CO2_emissions_Mt'].notna()) | (emissions_df['Year'] == 2050)
    ].copy()

    # Create world aggregate first - use full emissions_df to include all countries with GDP data
    world_aggregates = create_aggregates(emissions_df, ['Year', 'Emissions_scope'], 'All', 'WLD', 'World')
    
    # Create capacity-specific world aggregate (only countries with both emissions and GDP data)
    capacity_world_df = emissions_df[
        (emissions_df['Annual_CO2_emissions_Mt'].notna()) &  # Has emissions data
        (emissions_df['GDP_PPP'].notna()) &  # Has GDP data
        (emissions_df['Population'].notna())  # Has population data
    ].copy()
    capacity_world_aggregates = create_aggregates(capacity_world_df, ['Year', 'Emissions_scope'], 'All', 'WLD_CAPACITY', 'World (Capacity)')

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

    # Combine all dataframes (excluding capacity world aggregate from main data)
    final_df = pd.concat([
        emissions_df,
        region_aggregates,
        world_aggregates,
        eu_aggregates,
        g20_aggregates
    ], ignore_index=True)

    # Calculate new capacity metric: cumulative population (1970-latest) / square root of latest GDP per capita
    # This gives higher capacity to countries with large cumulative populations but low current wealth
    
    # Get cumulative population from 1970 to latest available year for each country
    # For Territory: latest = 2023, for Consumption: latest = 2022
    latest_years = final_df[
        (final_df['Annual_CO2_emissions_Mt'].notna()) & 
        (~final_df['ISO2'].isin(['WLD', 'EU', 'G20'])) &  # Exclude aggregates
        (final_df['Country'] != 'All')  # Exclude region aggregates
    ].groupby(['ISO2', 'Emissions_scope'])['Year'].max().reset_index()
    
    cum_pop_latest = final_df.merge(latest_years, on=['ISO2', 'Emissions_scope', 'Year'], how='inner')
    cum_pop_latest = cum_pop_latest[['ISO2', 'Emissions_scope', 'Cumulative_population']].copy()
    cum_pop_latest.rename(columns={'Cumulative_population': 'Cumulative_population_1970_to_latest'}, inplace=True)
    
    # Get latest GDP per capita for each country (latest available year)
    latest_gdp_per_capita = final_df[
        (~final_df['ISO2'].isin(['WLD', 'EU', 'G20'])) &  # Exclude aggregates
        (final_df['Country'] != 'All') &  # Exclude region aggregates
        (final_df['GDP_PPP'].notna()) & (final_df['Population'].notna())
    ].copy()
    latest_gdp_per_capita['gdp_per_capita'] = latest_gdp_per_capita['GDP_PPP'] / latest_gdp_per_capita['Population']
    
    # Get the latest year GDP per capita for each country
    latest_year_gdp = latest_gdp_per_capita.groupby(['ISO2', 'Emissions_scope'])['Year'].max().reset_index()
    latest_gdp_per_capita = latest_gdp_per_capita.merge(latest_year_gdp, on=['ISO2', 'Emissions_scope', 'Year'], how='inner')
    latest_gdp_per_capita = latest_gdp_per_capita[['ISO2', 'Emissions_scope', 'gdp_per_capita']].copy()
    
    # Calculate capacity: cumulative population (1970-latest) / square root of latest GDP per capita
    capacity_calc = cum_pop_latest.merge(latest_gdp_per_capita, on=['ISO2', 'Emissions_scope'], how='inner')
    capacity_calc['capacity_absolute'] = capacity_calc['Cumulative_population_1970_to_latest'] / np.sqrt(capacity_calc['gdp_per_capita'])
    
    # Handle infinite values and NaN
    capacity_calc.replace([np.inf, -np.inf], np.nan, inplace=True)
    capacity_calc['capacity_absolute'].fillna(0, inplace=True)
    
    # Calculate world total capacity by summing individual country capacities
    world_capacity_total = capacity_calc.groupby('Emissions_scope')['capacity_absolute'].sum().reset_index()
    world_capacity_total.rename(columns={'capacity_absolute': 'world_total_capacity'}, inplace=True)
    
    # Calculate share of capacity
    capacity_calc = capacity_calc.merge(world_capacity_total, on='Emissions_scope', how='left')
    capacity_calc['share_of_capacity'] = np.where(
        capacity_calc['world_total_capacity'] > 0,
        capacity_calc['capacity_absolute'] / capacity_calc['world_total_capacity'],
        0
    )
    
    # Merge capacity shares back to final_df
    final_df = final_df.merge(
        capacity_calc[['ISO2', 'Emissions_scope', 'share_of_capacity']], 
        on=['ISO2', 'Emissions_scope'], 
        how='left'
    )
    
    # For world aggregate, set capacity share to 1.0 (sum of all countries)
    final_df.loc[final_df['ISO2'] == 'WLD', 'share_of_capacity'] = 1.0
    
    # Fill NaN values for other aggregates (they don't have capacity shares)
    final_df['share_of_capacity'].fillna(0, inplace=True)
    
    # Print verification of new capacity calculation
    print("\n=== New Capacity Calculation Verification ===")
    print(f"Countries with capacity data: {len(capacity_calc)}")
    print(f"Capacity calculation formula: Cumulative Population (1970-latest) / √(Latest GDP per capita)")
    
    # Show some examples
    sample_capacity = capacity_calc.head(5)
    print("\nSample capacity calculations:")
    for _, row in sample_capacity.iterrows():
        print(f"{row['ISO2']}: {row['Cumulative_population_1970_to_latest']:,.0f} people / {row['gdp_per_capita']:,.0f} $/person = {row['capacity_absolute']:,.0f} capacity units")
    
    # Verify shares sum to 1.0
    for scope in ['Territory', 'Consumption']:
        scope_sum = capacity_calc[capacity_calc['Emissions_scope'] == scope]['share_of_capacity'].sum()
        print(f"\n{scope} scope - Sum of capacity shares: {scope_sum:.6f} (should be ~1.0)")
    
    # Show world aggregate data for verification
    print("\nWorld aggregate data verification:")
    world_data = final_df[final_df['ISO2'] == 'WLD'].head(2)  # Show first 2 rows (Territory and Consumption)
    for _, row in world_data.iterrows():
        print(f"  {row['Emissions_scope']}: GDP={row['GDP_PPP']:,.0f}, Population={row['Population']:,.0f}, Capacity_share={row['share_of_capacity']:.6f}")
    
    print("=== End Capacity Verification ===\n")

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

    create_planetary_boundary_file(iso_mapping, ipcc_regions, eu_g20_mapping)

def create_planetary_boundary_file(iso_mapping, ipcc_regions, eu_g20_mapping):
    """
    This function calculates and saves a dataframe detailing when each country
    and aggregate region surpasses its share of the planetary CO2 budget.
    """
    print("\nCreating planetary boundary file...")

    # --- 1. Load and Prepare Data ---
    emissions_full = pd.read_excel(f"{DATA_DIR}/2025-04-22_CO2 Emissions_All Countries_ISO Code_1750-2023.xlsx",
                                   sheet_name="GCB2024v17_MtCO2_flat")
    emissions_full.columns = emissions_full.columns.str.strip()
    emissions_full = emissions_full[['Country', 'ISO 3166-1 alpha-3', 'Year', 'Total', 'Per Capita']]
    emissions_full.rename(columns={
        'ISO 3166-1 alpha-3': 'ISO3',
        'Total': 'Annual_CO2_emissions_Mt',
        'Per Capita': 'Per_capita_emissions'
    }, inplace=True)
    
    emissions_full['inferred_population'] = (emissions_full['Annual_CO2_emissions_Mt'] * 1000000) / emissions_full['Per_capita_emissions']
    population_full = load_population_data()
    
    pb_df = emissions_full[['ISO3', 'Country', 'Year', 'Annual_CO2_emissions_Mt', 'inferred_population']].copy()
    pb_df.rename(columns={'inferred_population': 'Population'}, inplace=True)

    # --- Use consistent mapping logic ---
    valid_iso3_codes = set(ipcc_regions['ISO3'].unique())
    pb_df = pb_df[pb_df['ISO3'].isin(valid_iso3_codes)].copy()

    iso2_mapping = iso_mapping.set_index('ISO3')['ISO2'].to_dict()
    country_mapping = iso_mapping.set_index('ISO3')['Country'].to_dict()
    iso2_mapping['NAM'] = 'NA'

    pb_df['ISO2'] = pb_df['ISO3'].map(iso2_mapping)
    pb_df['Country'] = pb_df['ISO3'].map(country_mapping).fillna(pb_df['Country'])
    pb_df = pb_df.merge(population_full[['ISO3', 'Year', 'Population']], on=['ISO3', 'Year'], how='left', suffixes=('', '_official'))
    pb_df['Population'] = pb_df['Population_official'].fillna(pb_df['Population'])
    pb_df.drop(columns=['Population_official'], inplace=True)

    region_mapping = ipcc_regions.set_index('ISO3')['IPCC_Region_Intermediate'].to_dict()
    pb_df['Region'] = pb_df['ISO3'].map(region_mapping)
    eu_mapping = eu_g20_mapping.set_index('ISO3')['EU_country'].to_dict()
    g20_mapping = eu_g20_mapping.set_index('ISO3')['G20_country'].to_dict()
    pb_df['EU_country'] = pb_df['ISO3'].map(eu_mapping).fillna('No')
    pb_df['G20_country'] = pb_df['ISO3'].map(g20_mapping).fillna('No')
    pb_df.dropna(subset=['ISO2'], inplace=True)

    # --- 2. Create Aggregates ---
    world_agg = pb_df.groupby('Year').agg({'Annual_CO2_emissions_Mt': 'sum', 'Population': 'sum'}).reset_index()
    world_agg['ISO2'], world_agg['Country'], world_agg['Region'] = 'WLD', 'All', 'World'

    region_aggs = pb_df.groupby(['Year', 'Region']).agg({'Annual_CO2_emissions_Mt': 'sum', 'Population': 'sum'}).reset_index()
    region_aggs['Country'] = 'All'
    region_aggs['ISO2'] = region_aggs['Region']

    eu_agg = pb_df[pb_df['EU_country'] == 'Yes'].groupby('Year').agg({'Annual_CO2_emissions_Mt': 'sum', 'Population': 'sum'}).reset_index()
    eu_agg['ISO2'], eu_agg['Country'], eu_agg['Region'] = 'EU', 'All', 'European Union'
    
    g20_agg = pb_df[pb_df['G20_country'] == 'Yes'].groupby('Year').agg({'Annual_CO2_emissions_Mt': 'sum', 'Population': 'sum'}).reset_index()
    g20_agg['ISO2'], g20_agg['Country'], g20_agg['Region'] = 'G20', 'All', 'G20 Countries'
    
    pb_final_df = pd.concat([pb_df, world_agg, region_aggs, eu_agg, g20_agg], ignore_index=True)
    pb_final_df.drop_duplicates(subset=['ISO2', 'Year'], keep='first', inplace=True)

    # --- 3. Calculate Cumulative Values and Budget ---
    pb_final_df.sort_values(['ISO2', 'Year'], inplace=True)
    pb_final_df['cumulative_emissions'] = pb_final_df.groupby('ISO2')['Annual_CO2_emissions_Mt'].cumsum()

    # --- Pre-1970 Stats ---
    world_emissions = pb_final_df[pb_final_df['ISO2'] == 'WLD']
    emissions_pre_1970 = world_emissions[world_emissions['Year'] < 1970]['Annual_CO2_emissions_Mt'].sum()
    total_emissions = world_emissions['Annual_CO2_emissions_Mt'].sum()
    share_pre_1970 = (emissions_pre_1970 / total_emissions) * 100
    print(f"\\n--- PRE-1970 EMISSIONS STATS ---")
    print(f"Total global emissions before 1970: {emissions_pre_1970:,.0f} MtCO2")
    print(f"Total global emissions up to latest year: {total_emissions:,.0f} MtCO2")
    print(f"Share of emissions occurring before 1970: {share_pre_1970:.2f}%")

    # --- Country Count Stats ---
    countries_with_data = pb_df[pb_df['Annual_CO2_emissions_Mt'].notna() & (pb_df['Annual_CO2_emissions_Mt'] > 0)]
    print(f"Number of countries with emissions data in 1970: {countries_with_data[countries_with_data['Year'] == 1970]['ISO2'].nunique()}")
    for decade in range(1960, 1890, -10):
        print(f"Number of countries with emissions data in {decade}: {countries_with_data[countries_with_data['Year'] == decade]['ISO2'].nunique()}")
    print(f"--- END STATS ---\\n")

    pb_final_df['cumulative_population'] = pb_final_df.groupby('ISO2')['Population'].cumsum()

    GLOBAL_BUDGET = 830000
    latest_year = pb_final_df[pb_final_df['Annual_CO2_emissions_Mt'].notna()]['Year'].max()
    
    # Use 1988 for budget allocation instead of latest year
    latest_data = pb_final_df[pb_final_df['Year'] == 1988].copy()
    world_total_cum_pop = latest_data.loc[latest_data['ISO2'] == 'WLD', 'cumulative_population'].iloc[0]

    latest_data['share_of_cumulative_population'] = latest_data['cumulative_population'] / world_total_cum_pop
    latest_data['Country_CO2_budget_Mt'] = GLOBAL_BUDGET * latest_data['share_of_cumulative_population']

    # --- 4. Find Overshoot Year and Emissions ---
    budget_map = latest_data.set_index('ISO2')['Country_CO2_budget_Mt']
    pb_final_df['Country_CO2_budget_Mt'] = pb_final_df['ISO2'].map(budget_map)
    overshoot_df = pb_final_df[pb_final_df['cumulative_emissions'] > pb_final_df['Country_CO2_budget_Mt']]
    overshoot_years = overshoot_df.groupby('ISO2')['Year'].min().reset_index().rename(columns={'Year': 'Overshoot_year'})
    
    overshoot_emissions = pd.merge(
        overshoot_years, pb_final_df,
        left_on=['ISO2', 'Overshoot_year'], right_on=['ISO2', 'Year'], how='left'
    )[['ISO2', 'cumulative_emissions']].rename(columns={'cumulative_emissions': 'overshoot_year_cumulative_emissions'})

    # --- 5. Assemble and Save Final File ---
    # Get current year data for reporting
    current_data = pb_final_df[pb_final_df['Year'] == latest_year].copy()
    # Prepare budget allocation data with proper column names
    budget_allocation = latest_data[['ISO2', 'Country_CO2_budget_Mt', 'share_of_cumulative_population']].copy()
    budget_allocation.rename(columns={
        'share_of_cumulative_population': 'share_of_cumulative_population_1988'
    }, inplace=True)
    
    # Merge budget allocation with current data
    output_df = current_data.merge(budget_allocation, on='ISO2', how='left')
    output_df = output_df.merge(overshoot_years, on='ISO2', how='left')
    output_df = output_df.merge(overshoot_emissions, on='ISO2', how='left')
    
    # Set overshoot year to "Not overshot yet" for countries that haven't overshot yet
    output_df['Overshoot_year'] = output_df['Overshoot_year'].fillna("Not overshot yet")
    
    # Convert Overshoot_year to string to handle "Not overshot yet" values
    output_df['Overshoot_year'] = output_df['Overshoot_year'].astype(str)
    
    # Filter out countries that haven't overshot yet (keep only countries with actual overshoot years)
    # First, identify which countries have overshot (have numeric overshoot years)
    output_df['has_overshot'] = pd.to_numeric(output_df['Overshoot_year'], errors='coerce').notna()
    output_df = output_df[output_df['has_overshot']].copy()
    
    # Convert Overshoot_year to integer for mapping compatibility
    output_df['Overshoot_year'] = output_df['Overshoot_year'].astype(float).astype(int)
    
    # Drop the helper column
    output_df.drop(columns=['has_overshot'], inplace=True)

    # Add Emission_scope column for dashboard filtering
    output_df['Emission_scope'] = 'Territory'

    output_df = output_df[[
        'Country', 'ISO2', 'Region', 'Emission_scope', 'cumulative_emissions', 'cumulative_population', 
        'share_of_cumulative_population_1988', 'Country_CO2_budget_Mt_y',
        'Overshoot_year', 'overshoot_year_cumulative_emissions'
    ]]
    
    # Rename the column to remove the suffix
    output_df.rename(columns={'Country_CO2_budget_Mt_y': 'Country_CO2_budget_Mt'}, inplace=True)
    output_df.rename(columns={
        'cumulative_emissions': f'Cumulative_emissions_up_to_{latest_year}',
        'cumulative_population': f'Cumulative_population_up_to_{latest_year}'
    }, inplace=True)

    output_df.to_csv(f"{OUTPUT_DIR}/planetary_boundary.csv", index=False)
    print(f"Planetary boundary data saved to {OUTPUT_DIR}/planetary_boundary.csv")
    print(f"Budget allocation based on cumulative population from 1750 to 1988")
    print(f"Overshoot calculations use full historical data up to {latest_year}")

if __name__ == "__main__":
    main()
