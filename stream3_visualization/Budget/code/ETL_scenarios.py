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
    
    # For consumption emissions, use the original budget (data ends in 2022, budget starts from 2023)
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
        world_cum_pop_latest = cum_pop_latest_df[f'Cumulative_population_latest_{scope}'].max()
        
        # Merge with base dataframe
        cum_pop_latest_merge = cum_pop_latest_df[['ISO2', f'Cumulative_population_latest_{scope}']]
        base_df = base_df.merge(cum_pop_latest_merge, on='ISO2', how='left')
        
        # Calculate population share for each country
        base_df[f'Share_of_cumulative_population_1970_to_latest_{scope}'] = base_df[f'Cumulative_population_latest_{scope}'] / world_cum_pop_latest
        
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
    # Exclude aggregates (WLD, EU, G20, and any ISO2 where Country == 'All')
    aggregate_iso2s = set(['WLD', 'EU', 'G20']) | set(base_df[base_df['Country'] == 'All']['ISO2'].unique())
    country_rows = base_df[(base_df['ISO2'].isin(countries_with_data)) & (~base_df['ISO2'].isin(aggregate_iso2s))]
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
                        
                        country_budget = (total_available * population_share) - country_cumulative
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
                        
                        # FIX: Exclude countries with missing GDP data from Capacity scenario
                        if capacity_share == 0 or pd.isna(capacity_share):
                            # Skip this scenario for countries with missing GDP data
                            continue
                        
                        country_budget = (total_available * capacity_share) - country_cumulative
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
                        'Years_to_neutrality_from_today': years_to_neutrality_from_today
                    }
                    scenarios.append(scenario)

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
    'Country_carbon_budget', 'Country_budget_per_capita', 'Share_of_cumulative_emissions'
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
