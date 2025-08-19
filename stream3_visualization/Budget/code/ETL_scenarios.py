import pandas as pd
import numpy as np
from datetime import datetime

# Define the directory containing the data files
output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'
data_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data'

# Configuration: Choose Responsibility methodology
# True = Use 1970 to latest emissions year (new approach)
# False = Use 1970 to 2050 (current approach)
USE_LATEST_YEAR_FOR_RESPONSIBILITY = True

# Define global carbon budgets from the beginning of 2025 (in million tons of CO2)
# Source: IPCC AR6 SYR, Table 2.1, values converted from GtCO2 to MtCO2
BUDGET_GLOBAL_2C = {"50%": 1219000, "67%": 944000}
BUDGET_GLOBAL_15C = {"50%": 247000, "67%": 60000}

def get_global_budget(warming_scenario, probability, emissions_scope=None, combined_df=None):
    """Get the global carbon budget based on scenario parameters."""
    base_budget = BUDGET_GLOBAL_2C[probability] if warming_scenario == '2°C' else BUDGET_GLOBAL_15C[probability]
    
    # For territory emissions, subtract 2023 global emissions since budgets start from 2023
    if emissions_scope == 'Territory' and combined_df is not None:
        # Get 2023 global territory emissions
        territory_2023 = combined_df[
            (combined_df['Emissions_scope'] == 'Territory') & 
            (combined_df['ISO2'] == 'WLD') & 
            (combined_df['Year'] == 2023)
        ]['Annual_CO2_emissions_Mt'].iloc[0] if len(combined_df[
            (combined_df['Emissions_scope'] == 'Territory') & 
            (combined_df['ISO2'] == 'WLD') & 
            (combined_df['Year'] == 2023)
        ]) > 0 else 0
        
        adjusted_budget = base_budget - territory_2023
        return adjusted_budget
    
    # For consumption emissions, scale the budget based on country coverage ratio
    if emissions_scope == 'Consumption' and combined_df is not None:
        # Calculate 2022 population for countries with data in each scope
        # Territory scope: get 2022 population of countries with territory emissions data
        territory_countries_2022 = combined_df[
            (combined_df['Emissions_scope'] == 'Territory') & 
            (combined_df['Year'] == 2022) &
            (combined_df['Annual_CO2_emissions_Mt'].notna()) &
            (combined_df['Annual_CO2_emissions_Mt'] > 0) &
            (combined_df['ISO2'] != 'WLD')  # Exclude world aggregate
        ]['Population'].sum()
        
        # Consumption scope: get 2022 population of countries with consumption emissions data
        consumption_countries_2022 = combined_df[
            (combined_df['Emissions_scope'] == 'Consumption') & 
            (combined_df['Year'] == 2022) &
            (combined_df['Annual_CO2_emissions_Mt'].notna()) &
            (combined_df['Annual_CO2_emissions_Mt'] > 0) &
            (combined_df['ISO2'] != 'WLD')  # Exclude world aggregate
        ]['Population'].sum()
        
        # Calculate scaling factor and apply to budget
        if territory_countries_2022 > 0 and consumption_countries_2022 > 0:
            population_ratio = consumption_countries_2022 / territory_countries_2022
            scaled_budget = base_budget * population_ratio
            
            print(f"  Consumption budget scaling: {base_budget:,.0f} × {population_ratio:.4f} = {scaled_budget:,.0f} MtCO2")
            print(f"  Territory countries 2022 population: {territory_countries_2022:,.0f}")
            print(f"  Consumption countries 2022 population: {consumption_countries_2022:,.0f}")
            
            return scaled_budget
        else:
            print(f"  Warning: Could not calculate population ratio for consumption budget scaling")
            print(f"  Territory countries 2022 population: {territory_countries_2022:,.0f}")
            print(f"  Consumption countries 2022 population: {consumption_countries_2022:,.0f}")
            return base_budget
    
    # For consumption emissions without combined_df, use the original budget
    return base_budget

def penalty_func_2(x):
    """Quadratic penalty function."""
    return x * x

def load_current_targets():
    """Load and process current target years."""
    # Load the current targets file
    targets = pd.read_excel(f"{data_directory}/2025-04-21_Full file_Current carbon neutrality timeline per with Country ISO code.xlsx")

    # Load the ISO codes mapping file
    iso_mapping = pd.read_excel(f"{data_directory}/28-04-2025_ISO_Codes_Mapping.xlsx")
    iso_mapping.rename(columns={'Alpha-2 code': 'ISO2', 'Alpha-3 code': 'ISO3'}, inplace=True)

    # Get EU countries mapping
    eu_mapping = pd.read_excel(f"{data_directory}/2024-04-21_IPCC Regional Breakdown_ISO Country Code.xlsx",
                             sheet_name="G20_EU_Countries ",
                             header=0)

    # Merge ISO2 codes into the EU mapping
    eu_mapping = eu_mapping.merge(iso_mapping, on='ISO3', how='left')

    # Get EU countries with ISO2 codes
    eu_countries = eu_mapping[eu_mapping['EU_country'] == 'Yes']['ISO2'].tolist()

    # Create target mapping
    target_mapping = {}

    # Process each row
    for _, row in targets.iterrows():
        iso = row['ISO']
        target_year = row['Target year']

        if pd.notna(iso) and pd.notna(target_year):
            # Handle special case for NGA
            if iso == 'NGA':
                target_year = 2060
            elif isinstance(target_year, str):
                try:
                    target_year = int(target_year)
                except ValueError:
                    continue  # Skip if can't convert to integer

            if iso == 'EU27':
                # Add all EU countries with the same target year
                for eu_iso in eu_countries:
                    target_mapping[eu_iso] = int(target_year)
            else:
                # Map ISO3 to ISO2 for non-EU countries
                iso2_code = iso_mapping[iso_mapping['ISO3'] == iso]['ISO2'].values
                if len(iso2_code) > 0:
                    target_mapping[iso2_code[0]] = int(target_year)
                else:
                    print(f"Warning: ISO3 code {iso} not found in ISO mapping.")

    # Add manual entries for specific countries
    # Mexico (MEX) - 2050
    mexico_iso2 = iso_mapping[iso_mapping['ISO3'] == 'MEX']['ISO2'].values
    if len(mexico_iso2) > 0:
        target_mapping[mexico_iso2[0]] = 2050
    else:
        print("Warning: MEX not found in ISO mapping.")
    
    # Ethiopia (ETH) - 2050
    ethiopia_iso2 = iso_mapping[iso_mapping['ISO3'] == 'ETH']['ISO2'].values
    if len(ethiopia_iso2) > 0:
        target_mapping[ethiopia_iso2[0]] = 2050
    else:
        print("Warning: ETH not found in ISO mapping.")

    return target_mapping

def calculate_aggregate_ndc_pledges(target_mapping, base_df):
    """Calculate NDC pledges for aggregates (G20, WLD, EU, IPCC regions) as straight average of available pledges."""
    
    # Define aggregate groups - only for aggregates that exist in the data
    aggregate_groups = {}
    
    # Check which aggregates exist in base_df
    existing_aggregates = base_df[base_df['Country'] == 'All']['ISO2'].unique()
    
    # Add G20 if it exists
    if 'G20' in existing_aggregates:
        aggregate_groups['G20'] = ['AR', 'AU', 'BR', 'CA', 'CN', 'FR', 'DE', 'IN', 'ID', 'IT', 'JP', 'MX', 'RU', 'SA', 'ZA', 'KR', 'TR', 'GB', 'US']
    
    # Add EU if it exists
    if 'EU' in existing_aggregates:
        # Get EU countries from the same source as load_current_targets
        eu_mapping = pd.read_excel(f"{data_directory}/2024-04-21_IPCC Regional Breakdown_ISO Country Code.xlsx",
                                 sheet_name="G20_EU_Countries ",
                                 header=0)
        iso_mapping = pd.read_excel(f"{data_directory}/28-04-2025_ISO_Codes_Mapping.xlsx")
        iso_mapping.rename(columns={'Alpha-2 code': 'ISO2', 'Alpha-3 code': 'ISO3'}, inplace=True)
        eu_mapping = eu_mapping.merge(iso_mapping, on='ISO3', how='left')
        eu_countries = eu_mapping[eu_mapping['EU_country'] == 'Yes']['ISO2'].tolist()
        aggregate_groups['EU'] = eu_countries
    
    # Add WLD (World) if it exists
    if 'WLD' in existing_aggregates:
        aggregate_groups['WLD'] = None  # Will be calculated differently
    
    # Add other IPCC regions if they exist
    for region in existing_aggregates:
        if region not in ['WLD', 'G20', 'EU']:
            # For now, we'll skip IPCC regions that don't have predefined country lists
            # This could be enhanced later with proper region-to-country mappings
            aggregate_groups[region] = []
    
    # Calculate aggregate NDC pledges
    aggregate_ndc = {}
    
    for aggregate_name, country_list in aggregate_groups.items():
        if aggregate_name == 'WLD':
            # For world aggregate, calculate average of all available country pledges
            available_targets = []
            for iso2, target_year in target_mapping.items():
                if iso2 in base_df['ISO2'].values:
                    available_targets.append(target_year)
            
            if available_targets:
                aggregate_ndc[aggregate_name] = int(round(sum(available_targets) / len(available_targets)))
                print(f"World (WLD) NDC pledge: {aggregate_ndc[aggregate_name]} (average of {len(available_targets)} countries)")
            else:
                aggregate_ndc[aggregate_name] = None
                print(f"World (WLD) NDC pledge: No data available")
                
        elif country_list:  # G20, EU with country lists
            available_targets = []
            for iso2 in country_list:
                if iso2 in target_mapping:
                    available_targets.append(target_mapping[iso2])
            
            if available_targets:
                aggregate_ndc[aggregate_name] = int(round(sum(available_targets) / len(available_targets)))
                print(f"{aggregate_name} NDC pledge: {aggregate_ndc[aggregate_name]} (average of {len(available_targets)} countries)")
            else:
                aggregate_ndc[aggregate_name] = None
                print(f"{aggregate_name} NDC pledge: No data available")
        else:
            # For IPCC regions without country lists, skip for now
            aggregate_ndc[aggregate_name] = None
            print(f"{aggregate_name} NDC pledge: Skipped (no country list available)")
    
    return aggregate_ndc

# Load the preprocessed data and current targets
combined_df = pd.read_csv(f"{output_directory}/combined_data.csv")
current_targets = load_current_targets()

# Ensure "NA" is treated as a valid ISO2 code
combined_df['ISO2'] = combined_df['ISO2'].astype(str)

# Explicitly set "NA" for Namibia
combined_df.loc[combined_df['Country'] == 'Namibia', 'ISO2'] = 'NA'

# Print to verify
print("Combined DataFrame:")
print(combined_df[['ISO2', 'Country', 'Region']].head())

# Create base dataframe with required columns
def create_base_dataframe(df):
    """Create a base dataframe with pre-calculated shares for scenarios."""
    # Get unique countries and their regions
    base_df = df[['ISO2', 'Country', 'Region']].drop_duplicates()

    # We'll calculate scope-specific cumulative population shares in the loop below

    # Define emission_scopes within the function
    emission_scopes = ['Territory', 'Consumption']

    for scope in emission_scopes:
        scope_data = df[
            (df['Emissions_scope'] == scope) &
            (df['Annual_CO2_emissions_Mt'].notna()) &
            (df['Annual_CO2_emissions_Mt'] != 0) &
            (df['Year'] < 2050)
        ]
        
        # FIX: Include aggregates (G20, EU, IPCC regions) even if they have GDP_PPP = 0.0
        # as long as they have valid capacity values
        # Include all aggregates (where Country == 'All' and ISO2 != WLD) that have valid capacity
        # But treat G20 and EU as regular countries with capacity data
        aggregates_with_capacity = df[
            (df['Emissions_scope'] == scope) &
            (df['Year'] < 2050) &
            (df['Country'] == 'All') &
            (~df['ISO2'].isin(['WLD'])) &  # Only exclude WLD (world aggregate)
            (df['share_of_capacity'].notna()) &
            (df['share_of_capacity'] > 0)
        ]
        
        # Combine individual countries with aggregates that have capacity
        scope_data = pd.concat([scope_data, aggregates_with_capacity], ignore_index=True).drop_duplicates()

        latest_years = scope_data.groupby('ISO2')['Year'].max().reset_index()
        latest_years.columns = ['ISO2', f'Latest_year_{scope}']
        latest_years[f'Latest_year_{scope}'] = latest_years[f'Latest_year_{scope}'].astype(int)

        latest_data = pd.merge(
            scope_data,
            latest_years,
            left_on=['ISO2', 'Year'],
            right_on=['ISO2', f'Latest_year_{scope}']
        )[['ISO2', 'Annual_CO2_emissions_Mt', 'Cumulative_CO2_emissions_Mt', 'Cumulative_population', 'Emissions_per_capita_ton', 'GDP_PPP', 'Population', 'share_of_capacity']].rename(
            columns={
                'Annual_CO2_emissions_Mt': f'Latest_annual_CO2_emissions_Mt_{scope}',
                'Cumulative_CO2_emissions_Mt': f'Latest_cumulative_CO2_emissions_Mt_{scope}',
                'Cumulative_population': f'Latest_cumulative_population_{scope}',
                'Emissions_per_capita_ton': f'Latest_emissions_per_capita_t_{scope}',
                'GDP_PPP': f'Latest_GDP_PPP_{scope}',
                'Population': f'Latest_population_{scope}',
                'share_of_capacity': f'share_of_capacity_{scope}'
            }
        )

        base_df = base_df.merge(latest_years, on='ISO2', how='left')
        base_df = base_df.merge(latest_data, on='ISO2', how='left')

        # --- Share of Cumulative Population 1970-2050 (Responsibility) ---
        # Calculate scope-specific cumulative population from 1970 to 2050
        # Only include countries that have emissions data available for this scope
        countries_with_data = scope_data['ISO2'].unique()
        cum_pop_2050_df = df[
            (df['Emissions_scope'] == scope) & 
            (df['Year'] == 2050) &
            (df['ISO2'].isin(countries_with_data))
        ][['ISO2', 'Cumulative_population']].rename(columns={'Cumulative_population': f'Cumulative_population_2050_{scope}'})
        
        # Calculate world total for this period - sum individual countries only
        # Exclude all aggregates to ensure we only sum countries with emissions data for this scope
        aggregate_iso2s = ['WLD', 'EU', 'G20'] + [iso for iso in cum_pop_2050_df['ISO2'].unique() 
                                                  if iso in df[df['Country'] == 'All']['ISO2'].unique()]
        countries_only_2050 = cum_pop_2050_df[~cum_pop_2050_df['ISO2'].isin(aggregate_iso2s)]
        world_cum_pop_2050 = countries_only_2050[f'Cumulative_population_2050_{scope}'].sum()
        
        base_df = base_df.merge(cum_pop_2050_df, on='ISO2', how='left')
        base_df[f'Share_of_cumulative_population_1970_to_2050_{scope}'] = base_df[f'Cumulative_population_2050_{scope}'] / world_cum_pop_2050
        base_df.drop(columns=[f'Cumulative_population_2050_{scope}'], inplace=True)

        # --- Share of Cumulative Population 1970-Latest (Responsibility - New Approach) ---
        # Calculate scope-specific cumulative population from 1970 to latest emissions year
        # Only include countries that have emissions data available for this scope
        countries_with_data = scope_data['ISO2'].unique()
        latest_year_for_scope = latest_years[f'Latest_year_{scope}'].max()
        
        cum_pop_latest_df = df[
            (df['Emissions_scope'] == scope) & 
            (df['Year'] == latest_year_for_scope) &
            (df['ISO2'].isin(countries_with_data))
        ].copy()
        
        # Calculate cumulative population from 1970 to latest year for each country
        cum_pop_latest_df[f'Cumulative_population_latest_{scope}'] = cum_pop_latest_df.groupby(['ISO2', 'Region', 'Emissions_scope'])['Population'].cumsum()
        
        # Get world cumulative population for this scope (sum of countries with data)
        world_cum_pop_latest = cum_pop_latest_df[f'Cumulative_population_latest_{scope}'].sum()
        
        # Merge with base dataframe
        cum_pop_latest_merge = cum_pop_latest_df[['ISO2', f'Cumulative_population_latest_{scope}']]
        base_df = base_df.merge(cum_pop_latest_merge, on='ISO2', how='left')
        
        # Calculate population share for each country using WLD total (not sum of countries with data)
        # Get the WLD cumulative population from the merged data
        wld_cum_pop_latest = cum_pop_latest_df[
            cum_pop_latest_df['ISO2'] == 'WLD'
        ][f'Cumulative_population_latest_{scope}'].iloc[0] if len(cum_pop_latest_df[
            cum_pop_latest_df['ISO2'] == 'WLD'
        ]) > 0 else world_cum_pop_latest
        
        base_df[f'Share_of_cumulative_population_1970_to_latest_{scope}'] = base_df[f'Cumulative_population_latest_{scope}'] / wld_cum_pop_latest
        
        # FIX: Ensure WLD (World) always has population share = 1.0
        base_df.loc[base_df['ISO2'] == 'WLD', f'Share_of_cumulative_population_1970_to_latest_{scope}'] = 1.0
        
        base_df.drop(columns=[f'Cumulative_population_latest_{scope}'], inplace=True)
        
        print(f"  World cumulative population 1970-{latest_year_for_scope}: {world_cum_pop_latest:,.0f}")

        # --- Share of Cumulative Population (Equality) ---
        # Calculate cumulative population from latest emission year to 2050
        latest_year_for_scope = latest_years[f'Latest_year_{scope}'].max()
        
        # Get population data from latest year to 2050 for this scope
        # Only include countries that have emissions data available for this scope
        countries_with_data = scope_data['ISO2'].unique()
        pop_future_df = df[
            (df['Emissions_scope'] == scope) & 
            (df['Year'] >= latest_year_for_scope) & 
            (df['Year'] <= 2050) &
            (df['ISO2'].isin(countries_with_data))
        ].groupby('ISO2')['Population'].sum().reset_index()
        pop_future_df.columns = ['ISO2', f'Cumulative_population_Latest_to_2050_{scope}']
        
        # Merge with base_df
        base_df = base_df.merge(pop_future_df, on='ISO2', how='left')
        
        # Calculate world total for this period - sum individual countries only
        # Exclude all aggregates to ensure we only sum countries with emissions data for this scope
        aggregate_iso2s = ['WLD', 'EU', 'G20'] + [iso for iso in pop_future_df['ISO2'].unique() 
                                                  if iso in df[df['Country'] == 'All']['ISO2'].unique()]
        countries_only_future = pop_future_df[~pop_future_df['ISO2'].isin(aggregate_iso2s)]
        world_cumulative_pop_future = countries_only_future[f'Cumulative_population_Latest_to_2050_{scope}'].sum()
        
        base_df[f'Share_of_cumulative_population_Latest_to_2050_{scope}'] = base_df[f'Cumulative_population_Latest_to_2050_{scope}'] / world_cumulative_pop_future

        # --- Share of Cumulative Emissions (Responsibility) ---
        world_cumulative_emissions = base_df[
            base_df['ISO2'] == 'WLD'
        ][f'Latest_cumulative_CO2_emissions_Mt_{scope}'].iloc[0]
        base_df[f'Share_of_cumulative_emissions_{scope}'] = base_df[f'Latest_cumulative_CO2_emissions_Mt_{scope}'] / world_cumulative_emissions

    return base_df, emission_scopes

# Create the base dataframe
base_df, emission_scopes = create_base_dataframe(combined_df)

# Calculate aggregate NDC pledges and add them to current_targets
aggregate_ndc_pledges = calculate_aggregate_ndc_pledges(current_targets, base_df)
current_targets.update(aggregate_ndc_pledges)

print("\n=== Aggregate NDC Pledges ===")
for aggregate, pledge in aggregate_ndc_pledges.items():
    if pledge is not None:
        print(f"{aggregate}: {pledge}")
    else:
        print(f"{aggregate}: No data available")
print("=== End Aggregate NDC Pledges ===\n")

# Data check: Compare sum of country cumulative emissions to world cumulative emissions for each scope
for scope in emission_scopes:
    # Get countries with actual emissions data for this scope
    scope_data = combined_df[
        (combined_df['Emissions_scope'] == scope) &
        (combined_df['Annual_CO2_emissions_Mt'].notna()) &
        (combined_df['Annual_CO2_emissions_Mt'] != 0) &
        (combined_df['Year'] < 2050)
    ]
    countries_with_data = set(scope_data['ISO2'].unique())
    
    # FIX: Include aggregates (G20, EU, IPCC regions) in the base dataframe
    # Get all aggregates (where Country == 'All' and ISO2 != WLD)
    aggregates = set(base_df[base_df['Country'] == 'All']['ISO2'].unique()) - set(['WLD'])
    all_countries_with_data = countries_with_data | aggregates
    
    # Get country rows for this scope (including aggregates)
    country_rows = base_df[
        (base_df['ISO2'].isin(all_countries_with_data))
    ]
    sum_country_cumulative = country_rows[f'Latest_cumulative_CO2_emissions_Mt_{scope}'].sum()
    world_cumulative = base_df[base_df['ISO2'] == 'WLD'][f'Latest_cumulative_CO2_emissions_Mt_{scope}'].iloc[0]
    print(f"\n[DATA CHECK] Scope: {scope}")
    print(f"Countries included in sum: {len(country_rows)}")
    print(f"Sum of country cumulative emissions: {sum_country_cumulative:,.2f} MtCO2")
    print(f"World cumulative emissions (WLD): {world_cumulative:,.2f} MtCO2")
    print(f"Difference (World - Sum Countries): {world_cumulative - sum_country_cumulative:,.2f} MtCO2")

# Print verification for the latest years used in the 'Population' scenario
print("\nVerifying latest years used for 'Population' scenario calculations:")
print(base_df[['ISO2', 'Country', 'Latest_year_Territory', 'Latest_year_Consumption']].head())

# Print to verify
print("\nBase DataFrame:")
print(base_df[['ISO2', 'Country', 'Region', f'Share_of_cumulative_emissions_{emission_scopes[0]}']].head())

# FIX: Recalculate cumulative population shares using scope-specific world totals
print("\n=== FIXING CUMULATIVE POPULATION SHARES ===")
for scope in emission_scopes:
    print(f"\nRecalculating shares for {scope} scope...")
    
    # Get countries with emissions data for this scope
    scope_data = combined_df[
        (combined_df['Emissions_scope'] == scope) &
        (combined_df['Annual_CO2_emissions_Mt'].notna()) &
        (combined_df['Annual_CO2_emissions_Mt'] != 0) &
        (combined_df['Year'] < 2050)
    ]
    countries_with_data = set(scope_data['ISO2'].unique())
    
    # Exclude aggregates
    aggregate_iso2s = set(['WLD', 'EU', 'G20']) | set(base_df[base_df['Country'] == 'All']['ISO2'].unique())
    
    # Fix Share_of_cumulative_population_1970_to_2050
    # Get cumulative population 1970-2050 for countries with emissions data only
    cum_pop_1970_2050 = combined_df[
        (combined_df['Emissions_scope'] == scope) & 
        (combined_df['Year'] == 2050) &
        (combined_df['ISO2'].isin(countries_with_data)) &
        (~combined_df['ISO2'].isin(aggregate_iso2s))
    ][['ISO2', 'Cumulative_population']].copy()
    
    # Calculate world total using only countries with emissions data
    world_cum_pop_1970_2050 = cum_pop_1970_2050['Cumulative_population'].sum()
    
    # Update shares for countries with emissions data
    for _, row in cum_pop_1970_2050.iterrows():
        mask = (base_df['ISO2'] == row['ISO2']) & (base_df[f'Share_of_cumulative_population_1970_to_2050_{scope}'].notna())
        if mask.any():
            base_df.loc[mask, f'Share_of_cumulative_population_1970_to_2050_{scope}'] = row['Cumulative_population'] / world_cum_pop_1970_2050
    
    # Set world aggregate to 1.0
    world_mask = base_df['ISO2'] == 'WLD'
    if world_mask.any():
        base_df.loc[world_mask, f'Share_of_cumulative_population_1970_to_2050_{scope}'] = 1.0
    
    print(f"  Countries with emissions data: {len(countries_with_data)}")
    print(f"  World cumulative population 1970-2050: {world_cum_pop_1970_2050:,.0f}")
    
    # Fix Share_of_cumulative_population_Latest_to_2050
    # Get latest year for this scope
    latest_year_for_scope = scope_data['Year'].max()
    
    # Get cumulative population from latest year to 2050 for countries with emissions data only
    pop_latest_to_2050 = combined_df[
        (combined_df['Emissions_scope'] == scope) & 
        (combined_df['Year'] >= latest_year_for_scope) & 
        (combined_df['Year'] <= 2050) &
        (combined_df['ISO2'].isin(countries_with_data)) &
        (~combined_df['ISO2'].isin(aggregate_iso2s))
    ].groupby('ISO2')['Population'].sum().reset_index()
    
    # Calculate world total using only countries with emissions data
    world_cum_pop_latest_to_2050 = pop_latest_to_2050['Population'].sum()
    
    # Update shares for countries with emissions data
    for _, row in pop_latest_to_2050.iterrows():
        mask = (base_df['ISO2'] == row['ISO2']) & (base_df[f'Share_of_cumulative_population_Latest_to_2050_{scope}'].notna())
        if mask.any():
            base_df.loc[mask, f'Share_of_cumulative_population_Latest_to_2050_{scope}'] = row['Population'] / world_cum_pop_latest_to_2050
    
    # Set world aggregate to 1.0
    if world_mask.any():
        base_df.loc[world_mask, f'Share_of_cumulative_population_Latest_to_2050_{scope}'] = 1.0
    
    print(f"  World cumulative population {latest_year_for_scope}-2050: {world_cum_pop_latest_to_2050:,.0f}")
    
    # Verify shares sum to 1.0
    country_shares_1970 = base_df[
        (base_df['ISO2'].isin(countries_with_data)) & 
        (~base_df['ISO2'].isin(aggregate_iso2s)) &
        (base_df[f'Share_of_cumulative_population_1970_to_2050_{scope}'].notna())
    ][f'Share_of_cumulative_population_1970_to_2050_{scope}'].sum()
    
    country_shares_latest = base_df[
        (base_df['ISO2'].isin(countries_with_data)) & 
        (~base_df['ISO2'].isin(aggregate_iso2s)) &
        (base_df[f'Share_of_cumulative_population_Latest_to_2050_{scope}'].notna())
    ][f'Share_of_cumulative_population_Latest_to_2050_{scope}'].sum()
    
    print(f"  Sum of country shares 1970-2050: {country_shares_1970:.6f}")
    print(f"  Sum of country shares {latest_year_for_scope}-2050: {country_shares_latest:.6f}")

print("=== END FIX ===\n")

# FIX: Update world aggregate values to be consistent with scope-specific calculations
print("\n=== UPDATING WORLD AGGREGATE VALUES ===")
for scope in emission_scopes:
    print(f"\nUpdating world aggregate for {scope} scope...")
    
    # Get countries with emissions data for this scope
    scope_data = combined_df[
        (combined_df['Emissions_scope'] == scope) &
        (combined_df['Annual_CO2_emissions_Mt'].notna()) &
        (combined_df['Annual_CO2_emissions_Mt'] != 0) &
        (combined_df['Year'] < 2050)
    ]
    countries_with_data = set(scope_data['ISO2'].unique())
    
    # Exclude aggregates
    aggregate_iso2s = set(['WLD', 'EU', 'G20']) | set(base_df[base_df['Country'] == 'All']['ISO2'].unique())
    
    # Get country rows for this scope
    country_rows = base_df[
        (base_df['ISO2'].isin(countries_with_data)) & 
        (~base_df['ISO2'].isin(aggregate_iso2s))
    ]
    
    # Calculate scope-specific world totals
    world_latest_emissions = country_rows[f'Latest_annual_CO2_emissions_Mt_{scope}'].sum()
    world_cumulative_emissions = country_rows[f'Latest_cumulative_CO2_emissions_Mt_{scope}'].sum()
    world_latest_population = country_rows[f'Latest_population_{scope}'].sum()
    world_cumulative_population = country_rows[f'Latest_cumulative_population_{scope}'].sum()
    
    # Update world aggregate values
    world_mask = base_df['ISO2'] == 'WLD'
    if world_mask.any():
        base_df.loc[world_mask, f'Latest_annual_CO2_emissions_Mt_{scope}'] = world_latest_emissions
        base_df.loc[world_mask, f'Latest_cumulative_CO2_emissions_Mt_{scope}'] = world_cumulative_emissions
        base_df.loc[world_mask, f'Latest_population_{scope}'] = world_latest_population
        base_df.loc[world_mask, f'Latest_cumulative_population_{scope}'] = world_cumulative_population
        base_df.loc[world_mask, f'Latest_emissions_per_capita_t_{scope}'] = (world_latest_emissions * 1000000) / world_latest_population if world_latest_population > 0 else 0
    
    print(f"  Countries with emissions data: {len(countries_with_data)}")
    print(f"  World latest emissions: {world_latest_emissions:,.2f} MtCO2")
    print(f"  World cumulative emissions: {world_cumulative_emissions:,.2f} MtCO2")
    print(f"  World latest population: {world_latest_population:,.0f}")
    print(f"  World cumulative population: {world_cumulative_population:,.0f}")

print("=== END WORLD AGGREGATE UPDATE ===\n")

# Inject aggregate responsibility shares (EU, G20, IPCC regions) by summing member-country shares
def _aggregate_members_by_scope(df: pd.DataFrame, scope: str):
    # Countries with emissions data in this scope (align with existing filters)
    members_scope = df[
        (df['Emissions_scope'] == scope) &
        (df['Annual_CO2_emissions_Mt'].notna()) &
        (df['Annual_CO2_emissions_Mt'] != 0) &
        (df['Year'] < 2050) &
        (df['Country'] != 'All')
    ][['ISO2', 'EU_country', 'G20_country', 'Region']].drop_duplicates()

    eu_members = members_scope[members_scope['EU_country'] == 'Yes']['ISO2'].unique().tolist()
    g20_members = members_scope[members_scope['G20_country'] == 'Yes']['ISO2'].unique().tolist()

    # Map Region -> list of ISO2 members (exclude aggregates)
    region_members_map = (
        members_scope[['Region', 'ISO2']]
        .drop_duplicates()
        .groupby('Region')['ISO2']
        .apply(list)
        .to_dict()
    )
    return eu_members, g20_members, region_members_map


def inject_aggregate_responsibility_shares(base_df: pd.DataFrame, combined_df: pd.DataFrame, scopes):
    print("\n=== INJECTING AGGREGATE RESPONSIBILITY SHARES ===")
    for scope in scopes:
        print(f"Calculating aggregate responsibility shares for {scope}...")
        col_1970_2050 = f'Share_of_cumulative_population_1970_to_2050_{scope}'
        col_1970_latest = f'Share_of_cumulative_population_1970_to_latest_{scope}'

        eu_members, g20_members, region_members = _aggregate_members_by_scope(combined_df, scope)

        # EU
        if 'EU' in base_df['ISO2'].values and eu_members:
            eu_share_1970_2050 = base_df[base_df['ISO2'].isin(eu_members)][col_1970_2050].sum()
            eu_share_1970_latest = base_df[base_df['ISO2'].isin(eu_members)][col_1970_latest].sum()
            base_df.loc[base_df['ISO2'] == 'EU', col_1970_2050] = eu_share_1970_2050
            base_df.loc[base_df['ISO2'] == 'EU', col_1970_latest] = eu_share_1970_latest
            print(f"  EU shares set: 1970-2050={eu_share_1970_2050:.6f}, 1970-latest={eu_share_1970_latest:.6f}")

        # G20
        if 'G20' in base_df['ISO2'].values and g20_members:
            g20_share_1970_2050 = base_df[base_df['ISO2'].isin(g20_members)][col_1970_2050].sum()
            g20_share_1970_latest = base_df[base_df['ISO2'].isin(g20_members)][col_1970_latest].sum()
            base_df.loc[base_df['ISO2'] == 'G20', col_1970_2050] = g20_share_1970_2050
            base_df.loc[base_df['ISO2'] == 'G20', col_1970_latest] = g20_share_1970_latest
            print(f"  G20 shares set: 1970-2050={g20_share_1970_2050:.6f}, 1970-latest={g20_share_1970_latest:.6f}")

        # IPCC Regions (exclude WLD/EU/G20)
        existing_aggregates = set(base_df[base_df['Country'] == 'All']['ISO2'].unique())
        for region, members in region_members.items():
            if region in existing_aggregates and region not in ['WLD', 'EU', 'G20'] and members:
                reg_share_1970_2050 = base_df[base_df['ISO2'].isin(members)][col_1970_2050].sum()
                reg_share_1970_latest = base_df[base_df['ISO2'].isin(members)][col_1970_latest].sum()
                mask = (base_df['ISO2'] == region) & (base_df['Country'] == 'All')
                base_df.loc[mask, col_1970_2050] = reg_share_1970_2050
                base_df.loc[mask, col_1970_latest] = reg_share_1970_latest
                print(f"  {region} shares set: 1970-2050={reg_share_1970_2050:.6f}, 1970-latest={reg_share_1970_latest:.6f}")

    print("=== END INJECT ===\n")


# Call injection before building scenarios
inject_aggregate_responsibility_shares(base_df, combined_df, emission_scopes)

# FINAL VERIFICATION: Check that shares sum to 1.0
print("\n=== FINAL VERIFICATION: CUMULATIVE POPULATION SHARES ===")
for scope in emission_scopes:
    print(f"\nVerifying {scope} scope:")
    
    # Get countries with emissions data for this scope
    scope_data = combined_df[
        (combined_df['Emissions_scope'] == scope) &
        (combined_df['Annual_CO2_emissions_Mt'].notna()) &
        (combined_df['Annual_CO2_emissions_Mt'] != 0) &
        (combined_df['Year'] < 2050)
    ]
    countries_with_data = set(scope_data['ISO2'].unique())
    
    # Exclude aggregates
    aggregate_iso2s = set(['WLD', 'EU', 'G20']) | set(base_df[base_df['Country'] == 'All']['ISO2'].unique())
    
    # Check 1970-2050 shares (only country shares, excluding world aggregate)
    country_shares_1970 = base_df[
        (base_df['ISO2'].isin(countries_with_data)) & 
        (~base_df['ISO2'].isin(aggregate_iso2s)) &
        (base_df[f'Share_of_cumulative_population_1970_to_2050_{scope}'].notna())
    ][f'Share_of_cumulative_population_1970_to_2050_{scope}'].sum()
    
    world_share_1970 = base_df[
        (base_df['ISO2'] == 'WLD') &
        (base_df[f'Share_of_cumulative_population_1970_to_2050_{scope}'].notna())
    ][f'Share_of_cumulative_population_1970_to_2050_{scope}'].iloc[0] if len(base_df[
        (base_df['ISO2'] == 'WLD') &
        (base_df[f'Share_of_cumulative_population_1970_to_2050_{scope}'].notna())
    ]) > 0 else 0
    
    # Check Latest-2050 shares (only country shares, excluding world aggregate)
    country_shares_latest = base_df[
        (base_df['ISO2'].isin(countries_with_data)) & 
        (~base_df['ISO2'].isin(aggregate_iso2s)) &
        (base_df[f'Share_of_cumulative_population_Latest_to_2050_{scope}'].notna())
    ][f'Share_of_cumulative_population_Latest_to_2050_{scope}'].sum()
    
    world_share_latest = base_df[
        (base_df['ISO2'] == 'WLD') &
        (base_df[f'Share_of_cumulative_population_Latest_to_2050_{scope}'].notna())
    ][f'Share_of_cumulative_population_Latest_to_2050_{scope}'].iloc[0] if len(base_df[
        (base_df['ISO2'] == 'WLD') &
        (base_df[f'Share_of_cumulative_population_Latest_to_2050_{scope}'].notna())
    ]) > 0 else 0
    
    print(f"  1970-2050 shares:")
    print(f"    Country shares sum: {country_shares_1970:.6f}")
    print(f"    World share: {world_share_1970:.6f}")
    print(f"    Country shares should sum to 1.0: {'✓' if abs(country_shares_1970 - 1.0) < 0.0001 else '✗'}")
    print(f"    World share should equal country sum: {'✓' if abs(world_share_1970 - country_shares_1970) < 0.0001 else '✗'}")
    
    print(f"  Latest-2050 shares:")
    print(f"    Country shares sum: {country_shares_latest:.6f}")
    print(f"    World share: {world_share_latest:.6f}")
    print(f"    Country shares should sum to 1.0: {'✓' if abs(country_shares_latest - 1.0) < 0.0001 else '✗'}")
    print(f"    World share should equal country sum: {'✓' if abs(world_share_latest - country_shares_latest) < 0.0001 else '✗'}")

print("=== END VERIFICATION ===\n")

# Create all scenario combinations
scenarios = []
current_year = datetime.now().year

# Print budget adjustments for territory emissions
print("\n=== Budget Adjustments for Territory Emissions ===")
for warming_scenario in ['1.5°C', '2°C']:
    for probability in ['50%', '67%']:
        base_budget = get_global_budget(warming_scenario, probability)
        territory_2023 = combined_df[
            (combined_df['Emissions_scope'] == 'Territory') & 
            (combined_df['ISO2'] == 'WLD') & 
            (combined_df['Year'] == 2023)
        ]['Annual_CO2_emissions_Mt'].iloc[0] if len(combined_df[
            (combined_df['Emissions_scope'] == 'Territory') & 
            (combined_df['ISO2'] == 'WLD') & 
            (combined_df['Year'] == 2023)
        ]) > 0 else 0
        adjusted_budget = base_budget - territory_2023
        print(f"{warming_scenario} {probability}: {base_budget:,.0f} → {adjusted_budget:,.0f} MtCO2 (subtracted {territory_2023:,.0f} MtCO2 from 2023)")
print("=== Consumption emissions use original budgets (data ends 2022) ===\n")
for _, row in base_df.iterrows():
    for emissions_scope in emission_scopes:
        for warming_scenario in ['1.5°C', '2°C']:
            for probability in ['50%', '67%']:
                for distribution in ['Population', 'Responsibility', 'NDC Pledges', 'Capacity']:
                    # Calculate country carbon budget based on distribution scenario
                    global_budget = get_global_budget(warming_scenario, probability, emissions_scope, combined_df)
                    if distribution == 'Population':
                        country_budget = global_budget * row[f'Share_of_cumulative_population_Latest_to_2050_{emissions_scope}']
                    elif distribution == 'Responsibility':
                        # Get world's latest cumulative emissions
                        world_cumulative = base_df[
                            (base_df['ISO2'] == 'WLD') &
                            (base_df[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}'].notna())
                        ][f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}'].iloc[0]

                        # Calculate total available budget (global + world's historical emissions)
                        total_available = global_budget + world_cumulative

                        # Calculate country's share and subtract its historical emissions
                        country_cumulative = row[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}']
                        
                        # Choose population share based on configuration
                        if USE_LATEST_YEAR_FOR_RESPONSIBILITY:
                            # Use 1970 to latest emissions year (new approach)
                            population_share = row[f'Share_of_cumulative_population_1970_to_latest_{emissions_scope}']
                        else:
                            # Use 1970 to 2050 (current approach)
                            population_share = row[f'Share_of_cumulative_population_1970_to_2050_{emissions_scope}']
                        
                        # Calculate theoretical budget (this maintains the fairness concept)
                        theoretical_budget = (total_available * population_share) - country_cumulative
                        
                        # For Responsibility scenarios, we need to normalize positive budgets
                        # Store the theoretical budget for normalization
                        if 'responsibility_theoretical_budgets' not in locals():
                            responsibility_theoretical_budgets = {}
                        key = (emissions_scope, warming_scenario, probability)
                        if key not in responsibility_theoretical_budgets:
                            responsibility_theoretical_budgets[key] = []
                        responsibility_theoretical_budgets[key].append({
                            'ISO2': row['ISO2'],
                            'Country': row['Country'],
                            'theoretical_budget': theoretical_budget
                        })
                        
                        # For now, use theoretical budget - we'll normalize before neutrality year calculation
                        country_budget = theoretical_budget
                    elif distribution == 'Capacity':
                        # Get world's latest cumulative emissions
                        world_cumulative = base_df[
                            (base_df['ISO2'] == 'WLD') &
                            (base_df[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}'].notna())
                        ][f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}'].iloc[0]

                        # Calculate total available budget (global + world's historical emissions)
                        total_available = global_budget + world_cumulative

                        # Calculate country's share and subtract its historical emissions
                        country_cumulative = row[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}']
                        capacity_share = row[f'share_of_capacity_{emissions_scope}']
                        
                        # FIX: Include aggregates (G20, EU, IPCC regions) even if they have GDP_PPP = 0.0
                        # as long as they have valid capacity values
                        is_aggregate = (row['ISO2'] in ['G20', 'EU']) or (row['Country'] == 'All' and row['ISO2'] not in ['WLD'])
                        

                        
                        if capacity_share == 0 or pd.isna(capacity_share):
                            # Skip this scenario for countries with missing capacity data
                            continue
                        
                        country_budget = (total_available * capacity_share) - country_cumulative
                        
                        # DEBUG: Add debug output for G20 Territory Capacity budget calculation
                        if (row['ISO2'] == 'G20' and 
                            emissions_scope == 'Territory' and 
                            warming_scenario == '1.5°C'):
                            print(f"  Global budget: {global_budget}")
                            print(f"  World cumulative: {world_cumulative}")
                            print(f"  Total available: {total_available}")
                            print(f"  Country cumulative: {country_cumulative}")
                            print(f"  Country budget: {country_budget}")
                    elif distribution == 'NDC Pledges':
                        # NDC Pledges only apply to Territory emissions, not Consumption
                        if emissions_scope == 'Territory':
                            country_budget = None
                        else:  # Consumption scope
                            # Skip NDC Pledges for Consumption emissions
                            continue
                    else:  # Capacity
                        country_budget = None

                    # Calculate years to neutrality and neutrality year
                    latest_annual = row[f'Latest_annual_CO2_emissions_Mt_{emissions_scope}']
                    latest_year = row[f'Latest_year_{emissions_scope}']

                    if distribution == 'NDC Pledges':
                        # Get target year from current targets mapping
                        ndc_neutrality_year = current_targets.get(row['ISO2'])
                        if ndc_neutrality_year is not None:
                            years_to_neutrality = ndc_neutrality_year - latest_year
                            # Back-calculate Country_carbon_budget based on years_to_neutrality
                            if pd.notna(latest_annual) and latest_annual > 0:
                                country_budget = (years_to_neutrality * latest_annual) / 2
                            else:
                                country_budget = None
                            neutrality_year = ndc_neutrality_year
                        else:
                            years_to_neutrality = "N/A"
                            neutrality_year = "N/A"
                            country_budget = None
                    # using integers for buckets to ensure it can be visualized on the map
                    elif pd.notna(country_budget) and pd.notna(latest_annual) and latest_annual > 0:
                        # Initialize neutrality_year
                        neutrality_year = None
                        
                        if country_budget < 0:
                            # For negative budgets: find the historical year when they overshot their budget
                            # This uses the same approach as the planetary boundary calculation
                            country_cumulative_emissions = row[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}']
                            allocated_budget = country_budget + country_cumulative_emissions  # This is the positive budget they should have had
                            

                            
                            # Find the exact year when cumulative emissions exceeded the allocated budget
                            # Use the same approach as the planetary boundary calculation
                            if allocated_budget > 0:
                                # Get historical emissions data for this country, excluding 2050 (which has NaN values)
                                latest_emissions_year = 2023 if emissions_scope == 'Territory' else 2022
                                country_historical = combined_df[
                                    (combined_df['ISO2'] == row['ISO2']) & 
                                    (combined_df['Emissions_scope'] == emissions_scope) &
                                    (combined_df['Cumulative_CO2_emissions_Mt'].notna()) &
                                    (combined_df['Year'] <= latest_emissions_year) &
                                    (combined_df['Year'] != 2050)  # Exclude 2050 which has NaN values
                                ].sort_values('Year')
                                

                                
                                if len(country_historical) > 0:
                                    # Find the first year when cumulative emissions exceeded the allocated budget
                                    overshoot_mask = country_historical['Cumulative_CO2_emissions_Mt'] > allocated_budget
                                    if overshoot_mask.any():
                                        overshoot_year = country_historical[overshoot_mask]['Year'].iloc[0]
                                        neutrality_year = overshoot_year
                                        

                                    else:
                                        # If no overshoot found in historical data, use 1970
                                        neutrality_year = 1970
                                        

                                else:
                                    # No historical data available, use 1970
                                    neutrality_year = 1970
                                    

                            else:
                                # If allocated budget is also negative, they never had a valid budget
                                neutrality_year = 1970

                            # Compute years to neutrality from latest available year (can be negative)
                            years_to_neutrality = int(neutrality_year - latest_year)
                                

                        else:
                            # For positive budgets: use the standard linear decrease approach
                            years_to_neutrality = int(round(2 * country_budget / latest_annual))
                            calculated_neutrality_year = int(round(latest_year + years_to_neutrality))
                            
                            # Cap neutrality year at 1970 (earliest) and 2100 (latest)
                            if calculated_neutrality_year < 1970:
                                neutrality_year = 1970
                            elif calculated_neutrality_year > 2100:
                                neutrality_year = 2100
                            else:
                                neutrality_year = calculated_neutrality_year

                    else:
                        years_to_neutrality = "N/A"
                        neutrality_year = "N/A"

                    # Ensure years_to_neutrality_from_latest_available and neutrality_year are integers or "N/A"
                    if isinstance(years_to_neutrality, (int, float)) and pd.notna(years_to_neutrality):
                        years_to_neutrality = int(years_to_neutrality)
                    else:
                        years_to_neutrality = "N/A"
                    if isinstance(neutrality_year, (int, float)) or (hasattr(neutrality_year, 'dtype') and np.issubdtype(neutrality_year.dtype, np.integer)) and neutrality_year is not None:
                        neutrality_year = int(neutrality_year)
                    else:
                        neutrality_year = "N/A"

                    # Calculate Years_to_neutrality_from_today
                    if isinstance(neutrality_year, int):
                        years_to_neutrality_from_today = neutrality_year - current_year
                    else:
                        years_to_neutrality_from_today = "N/A"

                    # Calculate sanity check columns
                    global_total_budget = global_budget + row[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}']
                    latest_cumulative_emissions_per_capita = row[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}'] / row[f'Latest_cumulative_population_{emissions_scope}'] if row[f'Latest_cumulative_population_{emissions_scope}'] > 0 else None
                    
                    # Calculate global cumulative emissions for this scope
                    world_data = combined_df[
                        (combined_df['Emissions_scope'] == emissions_scope) & 
                        (combined_df['ISO2'] == 'WLD') & 
                        (combined_df['Year'] == latest_year)
                    ]
                    
                    if len(world_data) > 0:
                        global_cumulative_emissions = world_data['Cumulative_CO2_emissions_Mt'].iloc[0]
                        # Calculate share of global cumulative emissions
                        share_of_global_cumulative_emissions = row[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}'] / global_cumulative_emissions if global_cumulative_emissions > 0 else None
                    else:
                        global_cumulative_emissions = None
                        share_of_global_cumulative_emissions = None
                    
                    scenario = {
                        'ISO2': row['ISO2'],
                        'Country': row['Country'],
                        'Region': row['Region'],
                        'Share_of_cumulative_population_1970_to_2050': row[f'Share_of_cumulative_population_1970_to_2050_{emissions_scope}'],
                        'Share_of_cumulative_population_1970_to_latest': row[f'Share_of_cumulative_population_1970_to_latest_{emissions_scope}'],
                        'share_of_capacity': row[f'share_of_capacity_{emissions_scope}'],
                        'Emissions_scope': emissions_scope,
                        'Latest_year': latest_year,
                        'Latest_population': row[f'Latest_population_{emissions_scope}'],
                        'Latest_annual_CO2_emissions_Mt': latest_annual,
                        'Latest_cumulative_CO2_emissions_Mt': row[f'Latest_cumulative_CO2_emissions_Mt_{emissions_scope}'],
                        'Latest_cumulative_population': row[f'Latest_cumulative_population_{emissions_scope}'],
                        'Latest_emissions_per_capita_t': row[f'Latest_emissions_per_capita_t_{emissions_scope}'],
                        'Share_of_cumulative_population_Latest_to_2050': row[f'Share_of_cumulative_population_Latest_to_2050_{emissions_scope}'],
                        'Share_of_cumulative_emissions': row[f'Share_of_cumulative_emissions_{emissions_scope}'],
                        'Warming_scenario': warming_scenario,
                        'Probability_of_reach': probability,
                        'Budget_distribution_scenario': distribution,
                        'Global_Carbon_budget': global_budget,
                        'Country_carbon_budget': country_budget,
                        'Country_budget_per_capita': (country_budget * 1000000) / row[f'Latest_population_{emissions_scope}'] if pd.notna(country_budget) and pd.notna(row[f'Latest_population_{emissions_scope}']) and row[f'Latest_population_{emissions_scope}'] > 0 else None,
                        'Years_to_neutrality_from_latest_available': years_to_neutrality,
                        'Neutrality_year': neutrality_year,
                        'Years_to_neutrality_from_today': years_to_neutrality_from_today,
                        'Global_Total_Budget': global_total_budget,
                        'Latest_cumulative_emissions_per_capita': latest_cumulative_emissions_per_capita,
                        'Share_of_global_cumulative_emissions': share_of_global_cumulative_emissions
                    }
                    scenarios.append(scenario)

# NORMALIZATION STEP: Fix Responsibility scenario budgets to ensure mathematical consistency
print("\n=== NORMALIZING RESPONSIBILITY SCENARIO BUDGETS ===")
print("This ensures that sum of positive country budgets = global budget")

# Group scenarios by scope, warming scenario, and probability
responsibility_scenarios = [s for s in scenarios if s['Budget_distribution_scenario'] == 'Responsibility']

if responsibility_scenarios:
    # Group by unique combinations of scope, warming, and probability
    scenario_groups = {}
    for scenario in responsibility_scenarios:
        key = (scenario['Emissions_scope'], scenario['Warming_scenario'], scenario['Probability_of_reach'])
        if key not in scenario_groups:
            scenario_groups[key] = []
        scenario_groups[key].append(scenario)
    
    # Process each group
    for (scope, warming, prob), group_scenarios in scenario_groups.items():
        print(f"\nProcessing {scope} scope, {warming}, {prob}...")
        
        # Get the global budget for this scenario
        global_budget = group_scenarios[0]['Global_Carbon_budget']
        
        # Calculate theoretical budgets and identify positive ones
        theoretical_budgets = []
        positive_scenarios = []
        
        for scenario in group_scenarios:
            theoretical_budget = scenario['Country_carbon_budget']
            theoretical_budgets.append(theoretical_budget)
            
            if theoretical_budget > 0:
                positive_scenarios.append(scenario)
        
        if positive_scenarios:
            # Calculate total theoretical positive budget
            total_theoretical_positive = sum([s['Country_carbon_budget'] for s in positive_scenarios])
            
            print(f"  Global budget: {global_budget:,.0f} MtCO2")
            print(f"  Total theoretical positive budgets: {total_theoretical_positive:,.0f} MtCO2")
            print(f"  Countries with positive budgets: {len(positive_scenarios)}")
            
            # Normalize positive budgets using the correct formula:
            # final_budget = global_budget × (theoretical_budget / sum_of_positive_theoretical)
            print(f"  Applying correct normalization formula...")
            
            # Update the scenarios list with normalized budgets
            for scenario in positive_scenarios:
                # Find the corresponding scenario in the main scenarios list
                for main_scenario in scenarios:
                    if (main_scenario['ISO2'] == scenario['ISO2'] and
                        main_scenario['Emissions_scope'] == scenario['Emissions_scope'] and
                        main_scenario['Warming_scenario'] == scenario['Warming_scenario'] and
                        main_scenario['Probability_of_reach'] == scenario['Probability_of_reach'] and
                        main_scenario['Budget_distribution_scenario'] == 'Responsibility'):
                        
                        # Apply correct normalization formula
                        theoretical_budget = main_scenario['Country_carbon_budget']
                        share_of_positive = theoretical_budget / total_theoretical_positive
                        final_budget = global_budget * share_of_positive
                        main_scenario['Country_carbon_budget'] = final_budget
                        
                        # Recalculate neutrality year based on normalized budget
                        if final_budget > 0:
                            latest_annual = main_scenario['Latest_annual_CO2_emissions_Mt']
                            latest_year = main_scenario['Latest_year']
                            
                            # Recalculate years to neutrality
                            new_years_to_neutrality = int(round(2 * final_budget / latest_annual))
                            new_neutrality_year = int(round(latest_year + new_years_to_neutrality))
                            
                            # Cap neutrality year at 1970 (earliest) and 2100 (latest)
                            if new_neutrality_year < 1970:
                                new_neutrality_year = 1970
                            elif new_neutrality_year > 2100:
                                new_neutrality_year = 2100
                            
                            # Update the scenario with new neutrality values
                            main_scenario['Years_to_neutrality_from_latest_available'] = new_years_to_neutrality
                            main_scenario['Neutrality_year'] = new_neutrality_year
                            
                            # Recalculate years from today
                            current_year = 2024
                            main_scenario['Years_to_neutrality_from_today'] = new_neutrality_year - current_year
                        
                        print(f"    {scenario['Country']}: {theoretical_budget:,.0f} → {final_budget:,.0f} MtCO2 (share: {share_of_positive:.3f})")
                        break
            
            print(f"  After normalization: sum of positive budgets = {global_budget:,.0f} MtCO2 ✓")
        else:
            print(f"  No countries with positive budgets in this scenario")

# COMPREHENSIVE SANITY CHECK: Verify normalization worked correctly
print("\n=== COMPREHENSIVE SANITY CHECK ===")
print("Verifying that sum of positive responsibility budgets equals global budgets...")

# Check each scenario group
for (scope, warming, prob), group_scenarios in scenario_groups.items():
    if warming == '1.5°C' and prob == '50%' and scope == 'Territory':
        print(f"\n🔍 SANITY CHECK: {scope} scope, {warming}, {prob}")
        
        # Get the original global budget
        original_global_budget = group_scenarios[0]['Global_Carbon_budget']
        print(f"  Original global budget: {original_global_budget:,.0f} MtCO2")
        
        # Calculate sum of all positive responsibility budgets after normalization
        positive_budgets = []
        for scenario in group_scenarios:
            if scenario['Country_carbon_budget'] > 0:
                positive_budgets.append(scenario['Country_carbon_budget'])
        
        total_positive_budgets = sum(positive_budgets)
        print(f"  Sum of positive responsibility budgets: {total_positive_budgets:,.0f} MtCO2")
        
        # Check if they match
        if abs(total_positive_budgets - original_global_budget) < 0.01:  # Allow small floating point differences
            print(f"  ✅ MATCH: Sum of positive budgets = Global budget")
        else:
            print(f"  ❌ MISMATCH: Difference = {total_positive_budgets - original_global_budget:,.0f} MtCO2")
            print(f"  This indicates a problem with normalization!")
        
        # Show top 5 countries by budget size
        top_countries = sorted(group_scenarios, key=lambda x: x['Country_carbon_budget'], reverse=True)[:5]
        print(f"  Top 5 countries by budget:")
        for i, country in enumerate(top_countries, 1):
            print(f"    {i}. {country['Country']}: {country['Country_carbon_budget']:,.0f} MtCO2")
        
        # Show budget distribution
        budget_ranges = {
            '0-1000': len([b for b in positive_budgets if 0 < b <= 1000]),
            '1000-10000': len([b for b in positive_budgets if 1000 < b <= 10000]),
            '10000-100000': len([b for b in positive_budgets if 10000 < b <= 100000]),
            '>100000': len([b for b in positive_budgets if b > 100000])
        }
        print(f"  Budget distribution:")
        for range_name, count in budget_ranges.items():
            print(f"    {range_name} MtCO2: {count} countries")

print("=== END SANITY CHECK ===\n")

# After creating the scenarios list, create two separate dataframes
scenarios_df = pd.DataFrame(scenarios)

# 1. Create scenario parameters dataframe (one row per unique scenario)
scenario_params = scenarios_df[[
    'ISO2', 'Country', 'Region', 'Emissions_scope',
    'Warming_scenario', 'Probability_of_reach',
    'Budget_distribution_scenario', 'Years_to_neutrality_from_latest_available', 'Years_to_neutrality_from_today', 'Neutrality_year',
    'Latest_year', 'Latest_population', 'Latest_annual_CO2_emissions_Mt',
    'Latest_cumulative_CO2_emissions_Mt','Latest_emissions_per_capita_t', 'Latest_cumulative_population',
    'Share_of_cumulative_population_Latest_to_2050',
    'Share_of_cumulative_population_1970_to_2050',
    'Share_of_cumulative_population_1970_to_latest',
    'share_of_capacity', 'Global_Carbon_budget',
    'Country_carbon_budget', 'Country_budget_per_capita', 'Share_of_cumulative_emissions',
    'Global_Total_Budget', 'Latest_cumulative_emissions_per_capita', 'Share_of_global_cumulative_emissions'
]].drop_duplicates()

# Filter out rows where neutrality could not be calculated
original_rows = len(scenario_params)
scenario_params = scenario_params[scenario_params['Years_to_neutrality_from_latest_available'] != "N/A"].copy()
print(f"\nFiltered out {original_rows - len(scenario_params)} rows with 'N/A' neutrality years from scenario parameters.")

# Create a scenario_id for each unique combination
scenario_params['scenario_id'] = range(1, len(scenario_params) + 1)

# Ensure NA, TR, and US are included
required_isos = ['NA', 'TR', 'US']
for iso in required_isos:
    if iso not in scenario_params['ISO2'].values:
        print(f"Warning: ISO2 code {iso} is missing in scenario parameters.")

# 2. Create forecast data dataframe
forecast_data = []
for _, row in scenario_params.iterrows():
    # Skip if no latest year or emissions data
    if pd.isna(row['Latest_year']) or pd.isna(row['Latest_annual_CO2_emissions_Mt']):
        continue

    # Convert years to integers
    latest_year = int(row['Latest_year'])
    latest_emissions = row['Latest_annual_CO2_emissions_Mt']

    # Handle different cases for forecast
    if (row['Years_to_neutrality_from_latest_available'] == "N/A" or
        row['Years_to_neutrality_from_latest_available'] is None or
        row['Country_carbon_budget'] < 0):
        # For N/A or countries with negative budgets:
        # Set the first forecast year to the latest historical emissions value for continuity
        # Then drop to 0 in the next year
        forecast_years = pd.DataFrame({
            'Year': [latest_year, latest_year + 1],
            'Forecasted_emissions_Mt': [latest_emissions, 0]
        })
    else:
        # Normal case: linear decrease to zero
        if row['Neutrality_year'] == '>2100':
            neutrality_year = 2100
        else:
            neutrality_year = int(row['Neutrality_year'])
        forecast_years = pd.DataFrame({
            'Year': range(latest_year, neutrality_year + 1)
        })

        # Calculate forecasted emissions
        if neutrality_year == latest_year:
            # If neutrality year is the same as latest year, emissions should be 0 immediately
            forecast_years['Forecasted_emissions_Mt'] = 0
        else:
            slope = -latest_emissions / (neutrality_year - latest_year)
            forecast_years['Forecasted_emissions_Mt'] = [
                max(0, latest_emissions + slope * (year - latest_year))
                for year in forecast_years['Year']
            ]

    # Add forecast data with scenario_id reference
    for _, year_row in forecast_years.iterrows():
        forecast_data.append({
            'scenario_id': row['scenario_id'],
            'Year': year_row['Year'],
            'Forecasted_emissions_Mt': year_row['Forecasted_emissions_Mt']
        })

# Convert forecast data to DataFrame
forecast_df = pd.DataFrame(forecast_data)

# Add Data_type column to the forecast_df dataframe
forecast_df['Data_type'] = 'Forecast'

# Save both files
scenario_params.to_csv(f"{output_directory}/scenario_parameters.csv", index=False)
forecast_df.to_csv(f"{output_directory}/forecast_data.csv", index=False)

print(f"\nScenario parameters saved to {output_directory}/scenario_parameters.csv")
print(f"Forecast data saved to {output_directory}/forecast_data.csv")
