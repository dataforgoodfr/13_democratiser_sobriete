import pandas as pd
import numpy as np
import os

# --- Constants ---
# Assumes the script is in Budget/code and data is in Budget/Data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'Data'))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'Output'))

# --- Data Loading Functions ---

def load_iso_codes_mapping():
    """Load and process ISO codes mapping data."""
    path = os.path.join(DATA_DIR, "28-04-2025_ISO_Codes_Mapping.xlsx")
    iso_mapping = pd.read_excel(path)
    iso_mapping.rename(columns={'Alpha-2 code': 'ISO2', 'Alpha-3 code': 'ISO3'}, inplace=True)
    iso_mapping['ISO2'] = iso_mapping['ISO2'].fillna('')
    # Manual corrections for consistency
    iso_mapping.loc[iso_mapping['ISO2'] == 'US', 'Country'] = 'United States of America'
    iso_mapping.loc[iso_mapping['ISO3'] == 'USA', 'Country'] = 'United States of America'
    return iso_mapping[['ISO3', 'ISO2', 'Country']]

def load_ipcc_regions():
    """Load and process IPCC region mapping data."""
    path = os.path.join(DATA_DIR, "2024-04-21_IPCC Regional Breakdown_ISO Country Code.xlsx")
    regions = pd.read_excel(path, sheet_name="Full_mapping")
    regions = regions[['Intermediate level (10)', 'ISO codes']]
    regions.rename(columns={'Intermediate level (10)': 'IPCC_Region_Intermediate', 'ISO codes': 'ISO3'}, inplace=True)
    
    expanded_regions = []
    for _, row in regions.iterrows():
        if isinstance(row['ISO3'], str):
            for iso in row['ISO3'].split(','):
                expanded_regions.append({
                    'IPCC_Region_Intermediate': row['IPCC_Region_Intermediate'],
                    'ISO3': iso.strip()
                })
    return pd.DataFrame(expanded_regions)

def load_eu_g20_mapping():
    """Load EU and G20 country mappings."""
    path = os.path.join(DATA_DIR, "2024-04-21_IPCC Regional Breakdown_ISO Country Code.xlsx")
    mapping = pd.read_excel(path, sheet_name="G20_EU_Countries ", header=0)
    return mapping[['ISO3', 'EU_country', 'G20_country']]

def load_population_data():
    """Load and process population data from 1970 to 2050."""
    path = os.path.join(DATA_DIR, "2025-04-21_Population per Country ISO code_1970-2050.xlsx")
    pop = pd.read_excel(path, sheet_name="unpopulation_dataportal_2025042")
    pop = pop[['Iso3', 'Location', 'Time', 'Value']]
    pop.rename(columns={'Iso3': 'ISO3', 'Location': 'Country', 'Time': 'Year', 'Value': 'Population'}, inplace=True)
    return pop[pop['Year'] <= 2050]

# --- Main Calculation Function ---

def calculate_historical_responsibility():
    """
    This function calculates and saves a dataframe detailing when each country
    and aggregate region surpasses its share of the planetary CO2 budget based on historical data from 1750.
    """
    print("Starting historical responsibility calculation...")

    # --- 1. Load and Prepare Data ---
    iso_mapping = load_iso_codes_mapping()
    ipcc_regions = load_ipcc_regions()
    eu_g20_mapping = load_eu_g20_mapping()

    emissions_path = os.path.join(DATA_DIR, "2025-04-22_CO2 Emissions_All Countries_ISO Code_1750-2023.xlsx")
    emissions_full = pd.read_excel(emissions_path, sheet_name="GCB2024v17_MtCO2_flat")
    emissions_full.columns = emissions_full.columns.str.strip()
    emissions_full = emissions_full[['Country', 'ISO 3166-1 alpha-3', 'Year', 'Total', 'Per Capita']]
    emissions_full.rename(columns={
        'ISO 3166-1 alpha-3': 'ISO3',
        'Total': 'Annual_CO2_emissions_Mt',
        'Per Capita': 'Per_capita_emissions'
    }, inplace=True)
    
    # Infer population for years where official data is missing, then merge with official data
    emissions_full['inferred_population'] = (emissions_full['Annual_CO2_emissions_Mt'] * 1000000) / emissions_full['Per_capita_emissions']
    population_full = load_population_data()
    
    pb_df = emissions_full[['ISO3', 'Country', 'Year', 'Annual_CO2_emissions_Mt', 'inferred_population']].copy()
    pb_df.rename(columns={'inferred_population': 'Population'}, inplace=True)

    # --- 2. Map and Clean Data ---
    valid_iso3_codes = set(ipcc_regions['ISO3'].unique())
    pb_df = pb_df[pb_df['ISO3'].isin(valid_iso3_codes)].copy()

    iso2_map = iso_mapping.set_index('ISO3')['ISO2'].to_dict()
    country_map = iso_mapping.set_index('ISO3')['Country'].to_dict()
    iso2_map['NAM'] = 'NA' # Manual correction

    pb_df['ISO2'] = pb_df['ISO3'].map(iso2_map)
    pb_df['Country'] = pb_df['ISO3'].map(country_map).fillna(pb_df['Country'])
    
    # Merge official population data where available
    pb_df = pb_df.merge(population_full[['ISO3', 'Year', 'Population']], on=['ISO3', 'Year'], how='left', suffixes=('', '_official'))
    pb_df['Population'] = pb_df['Population_official'].fillna(pb_df['Population'])
    pb_df.drop(columns=['Population_official'], inplace=True)

    region_map = ipcc_regions.set_index('ISO3')['IPCC_Region_Intermediate'].to_dict()
    pb_df['Region'] = pb_df['ISO3'].map(region_map)
    eu_map = eu_g20_mapping.set_index('ISO3')['EU_country'].to_dict()
    g20_map = eu_g20_mapping.set_index('ISO3')['G20_country'].to_dict()
    pb_df['EU_country'] = pb_df['ISO3'].map(eu_map).fillna('No')
    pb_df['G20_country'] = pb_df['ISO3'].map(g20_map).fillna('No')
    pb_df.dropna(subset=['ISO2'], inplace=True)

    # --- 3. Create Aggregates (World, Regions, etc.) ---
    world_agg = pb_df.groupby('Year').agg({'Annual_CO2_emissions_Mt': 'sum', 'Population': 'sum'}).reset_index()
    world_agg['ISO2'], world_agg['Country'], world_agg['Region'] = 'WLD', 'All', 'World'

    # Combine all data
    pb_final_df = pd.concat([pb_df, world_agg], ignore_index=True)
    pb_final_df.drop_duplicates(subset=['ISO2', 'Year'], keep='first', inplace=True)

    # --- 4. Calculate Cumulative Values and Budget ---
    pb_final_df.sort_values(['ISO2', 'Year'], inplace=True)
    pb_final_df['cumulative_emissions'] = pb_final_df.groupby('ISO2')['Annual_CO2_emissions_Mt'].cumsum()
    pb_final_df['cumulative_population'] = pb_final_df.groupby('ISO2')['Population'].cumsum()

    GLOBAL_BUDGET = 830000  # Global carbon budget in MtCO2 for 1.5C
    BUDGET_YEAR = 1988 # Year for budget allocation based on population share
    
    budget_data = pb_final_df[pb_final_df['Year'] == BUDGET_YEAR].copy()
    world_total_cum_pop = budget_data.loc[budget_data['ISO2'] == 'WLD', 'cumulative_population'].iloc[0]

    budget_data['share_of_cumulative_population'] = budget_data['cumulative_population'] / world_total_cum_pop
    budget_data['Country_CO2_budget_Mt'] = GLOBAL_BUDGET * budget_data['share_of_cumulative_population']

    # --- 5. Find Overshoot Year ---
    budget_map = budget_data.set_index('ISO2')['Country_CO2_budget_Mt']
    pb_final_df['Country_CO2_budget_Mt'] = pb_final_df['ISO2'].map(budget_map)
    
    overshoot_df = pb_final_df[pb_final_df['cumulative_emissions'] > pb_final_df['Country_CO2_budget_Mt']].copy()
    overshoot_years = overshoot_df.groupby('ISO2')['Year'].min().reset_index().rename(columns={'Year': 'Overshoot_year'})
    
    # --- 6. Assemble and Save Final File ---
    latest_year = pb_final_df[pb_final_df['Annual_CO2_emissions_Mt'].notna()]['Year'].max()
    current_data = pb_final_df[pb_final_df['Year'] == latest_year].copy()
    
    budget_allocation = budget_data[['ISO2', 'Country', 'Country_CO2_budget_Mt', 'share_of_cumulative_population']].copy()
    budget_allocation.rename(columns={'share_of_cumulative_population': f'share_of_pop_{BUDGET_YEAR}'}, inplace=True)
    
    output_df = current_data.merge(budget_allocation[['ISO2', 'Country_CO2_budget_Mt', f'share_of_pop_{BUDGET_YEAR}']], on='ISO2', how='left')
    output_df = output_df.merge(overshoot_years, on='ISO2', how='left')

    # The merge adds suffixes if columns (like 'Country') exist in both frames.
    # We need to handle the resulting column names, typically 'Country_x' from the left frame.
    if 'Country_x' not in output_df.columns:
        # If no suffix was added, it means the 'Country' column from the left frame (current_data) was kept.
        # This can happen if the right frame of the merge didn't have a 'Country' column.
        # To be safe, we check and use the correct one.
        country_col_name = 'Country'
    else:
        country_col_name = 'Country_x'

    output_df['Overshoot_year'] = output_df['Overshoot_year'].fillna("Not overshot yet")
    
    output_df = output_df[[
        country_col_name, 'ISO2', 'Region', 'cumulative_emissions', 
        f'share_of_pop_{BUDGET_YEAR}', 'Country_CO2_budget_Mt_y', 'Overshoot_year'
    ]]
    output_df.rename(columns={
        country_col_name: 'Country',
        'cumulative_emissions': f'Cumulative_emissions_up_to_{latest_year}',
        'Country_CO2_budget_Mt_y': 'Country_CO2_budget_Mt'
    }, inplace=True)
    
    # Save the file
    output_path = os.path.join(OUTPUT_DIR, "historical_responsibility_1750.csv")
    output_df.to_csv(output_path, index=False)
    
    print(f"\\nHistorical responsibility calculation complete.")
    print(f"Output saved to: {output_path}")

# --- Run the main function ---
if __name__ == "__main__":
    calculate_historical_responsibility()
