"""
FR Energy-GDP Analysis - Data Preparation
Consolidates energy, GDP, and price data to create base dataset for intensity analysis

Input files:
  - data-NRJ-GDP/FR_energy_GDP.xlsx: Final energy (TWh) and nominal GDP (1990-2024)
  - data-NRJ-GDP/price_index.xlsx: Consumer price index (monthly, base 2015)
  - data-NRJ-GDP/scenarios_energy.xlsx: Energy projections by source and scenario (2022-2050)

Output:
  - Dataset with real GDP (2015 base), final energy, and energy intensity
  - Consolidated scenarios for SNBC-3 and AME-2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_PATH = Path(__file__).parent  # stream3_visualization/Decomposition/reports/FR
DATA_PATH = BASE_PATH / 'data-NRJ-GDP'
OUTPUT_DIR = BASE_PATH / 'visuals NRJ-GDP'

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_INDEX_FILE = DATA_PATH / 'price_index.xlsx'
ENERGY_GDP_FILE = DATA_PATH / 'FR_energy_GDP.xlsx'
SCENARIOS_FILE = DATA_PATH / 'scenarios_energy.xlsx'
OUTPUT_FILE = OUTPUT_DIR / 'FR_NRJ_GDP_consolidated.xlsx'

print("="*80)
print("FR ENERGY-GDP DATA PREPARATION")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PROCESS PRICE INDEX
# ============================================================================
print("\n1. PROCESSING PRICE INDEX (Monthly -> Annual Average)")
print("-" * 80)

df_price = pd.read_excel(PRICE_INDEX_FILE)
print(f"Loaded price index: {len(df_price)} rows")
print(f"Columns: {df_price.columns.tolist()}")

# Parse the 'Période' column to extract year and month
# Expected format: "YYYY-MM" 
if 'Période' in df_price.columns:
    df_price['Période'] = df_price['Période'].astype(str)
    df_price['Year'] = df_price['Période'].str[:4].astype(int)
    df_price['Month'] = df_price['Période'].str[5:7].astype(int)
    
    # Rename index column to match
    if 'index' in df_price.columns:
        df_price = df_price.rename(columns={'index': 'Index'})
    
    # Calculate annual average price index
    annual_price_index = df_price.groupby('Year')['Index'].mean().reset_index()
    annual_price_index.columns = ['Year', 'Price_Index']
    print(f"Computed annual average price indices: {len(annual_price_index)} years")
    print(annual_price_index.head(10))
else:
    print("ERROR: Expected 'Période' column in price index file")

# ============================================================================
# STEP 2: LOAD AND CALCULATE REAL GDP
# ============================================================================
print("\n2. CALCULATING REAL GDP (Nominal GDP / Price Index * 100)")
print("-" * 80)

df_energy_gdp = pd.read_excel(ENERGY_GDP_FILE)
print(f"Loaded energy-GDP data: {len(df_energy_gdp)} rows")
print(f"Columns: {df_energy_gdp.columns.tolist()}")
print(df_energy_gdp.head())

# Rename columns to standardized names
df_energy_gdp = df_energy_gdp.rename(columns={
    'final_ener_twh': 'Final_Energy_TWh',
    'gdp': 'GDP'
})

# Merge with annual price index
if 'annual_price_index' not in locals():
    print("ERROR: Price index not successfully loaded")
else:
    df_energy_gdp = df_energy_gdp.merge(annual_price_index, on='Year', how='left')
    
    # Calculate real GDP (in 2015 prices)
    # Real GDP = Nominal GDP / (Price Index / 100)
    df_energy_gdp['GDP_Real_2015'] = df_energy_gdp['GDP'] / (df_energy_gdp['Price_Index'] / 100)
    print(f"\nCalculated real GDP for {len(df_energy_gdp)} years")
    print(f"Sample data:")
    print(df_energy_gdp[['Year', 'GDP', 'Price_Index', 'GDP_Real_2015']].head(10))

# ============================================================================
# STEP 3: LOAD SCENARIOS AND INTERPOLATE TO 2022+
# ============================================================================
print("\n3. LOADING AND INTERPOLATING SCENARIOS (Linear interpolation to 2022+)")
print("-" * 80)

df_scenarios_raw = pd.read_excel(SCENARIOS_FILE)
print(f"Loaded scenarios data: {len(df_scenarios_raw)} rows")
print(f"Columns: {df_scenarios_raw.columns.tolist()}")

# Rename columns from French
df_scenarios_raw = df_scenarios_raw.rename(columns={
    'Ann\u00e9e': 'Year',
    'Sc\u00e9nario': 'Scenario'
})

print(f"First few rows:\n{df_scenarios_raw.head()}")

# Identify energy source columns (all columns except Year and Scenario)
energy_sources = [col for col in df_scenarios_raw.columns if col not in ['Year', 'Scenario']]
print(f"\nEnergy sources identified: {energy_sources}")

# Melt the dataframe from wide to long format
df_scenarios = df_scenarios_raw.melt(
    id_vars=['Year', 'Scenario'],
    value_vars=energy_sources,
    var_name='Source',
    value_name='Energy'
)

# Remove NaN values
df_scenarios = df_scenarios.dropna(subset=['Energy'])

print(f"Melted scenarios data: {len(df_scenarios)} rows")
print(f"Years available: {sorted(df_scenarios['Year'].unique())}")
print(f"Scenarios: {df_scenarios['Scenario'].unique().tolist()}")

# Get available years in scenarios data
available_years = sorted(df_scenarios['Year'].unique())
print(f"Available years: {available_years}")

# Interpolate to 2022 if needed
if 2022 not in available_years:
    print("\nYear 2022 not directly available - interpolating...")
    
    # For each scenario and source combination, interpolate missing values
    interpolated_rows = []
    
    for scenario in df_scenarios['Scenario'].unique():
        scenario_df = df_scenarios[df_scenarios['Scenario'] == scenario]
        
        for source in scenario_df['Source'].unique():
            source_data = scenario_df[scenario_df['Source'] == source].sort_values('Year')
            
            if len(source_data) >= 2:
                years_list = sorted(source_data['Year'].unique())
                
                # Check if 2022 is between consecutive years
                for i in range(len(years_list) - 1):
                    y1, y2 = years_list[i], years_list[i + 1]
                    
                    if y1 < 2022 < y2:
                        v1 = source_data[source_data['Year'] == y1]['Energy'].iloc[0]
                        v2 = source_data[source_data['Year'] == y2]['Energy'].iloc[0]
                        
                        # Linear interpolation
                        v_2022 = v1 + (v2 - v1) * (2022 - y1) / (y2 - y1)
                        
                        interpolated_rows.append({
                            'Year': 2022,
                            'Scenario': scenario,
                            'Source': source,
                            'Energy': v_2022
                        })
                        
                        print(f"  {scenario} - {source}: {v1:.2f} ({y1}) -> {v_2022:.2f} (2022) -> {v2:.2f} ({y2})")
    
    # Add interpolated 2022 values to scenarios
    if interpolated_rows:
        df_interp_2022 = pd.DataFrame(interpolated_rows)
        df_scenarios = pd.concat([df_scenarios, df_interp_2022], ignore_index=True)
        df_scenarios = df_scenarios.sort_values(['Scenario', 'Year', 'Source'])

# ============================================================================
# STEP 4: SUM SOURCES TO GET TOTAL ENERGY BY SCENARIO
# ============================================================================
print("\n4. SUMMING SOURCES TO GET TOTAL ENERGY BY SCENARIO")
print("-" * 80)

# Filter to years >= 2022
df_scenarios = df_scenarios[df_scenarios['Year'] >= 2022].copy()

# Sum all sources by scenario and year
total_energy_scenarios = df_scenarios.groupby(['Year', 'Scenario'])['Energy'].sum().reset_index()
total_energy_scenarios.columns = ['Year', 'Scenario', 'Final_Energy_TWh']

print(f"Total energy by scenario: {len(total_energy_scenarios)} rows")
print(total_energy_scenarios.head(20))

# ============================================================================
# STEP 5: MERGE HISTORICAL DATA WITH SCENARIOS
# ============================================================================
print("\n5. CONSOLIDATING HISTORICAL DATA AND SCENARIOS")
print("-" * 80)

# Extract historical energy data (1990-2024)
df_historical = df_energy_gdp[['Year', 'Final_Energy_TWh', 'GDP_Real_2015']].copy()

# Add placeholder scenario for historical data
df_historical['Scenario'] = 'Historical'

print(f"Historical data: {len(df_historical)} years")
print(df_historical.head())

# Create consolidated dataframe
consolidated = pd.concat([df_historical, total_energy_scenarios], ignore_index=True)

# For scenario years, GDP is NaN - will add a note
consolidated = consolidated.sort_values(['Scenario', 'Year']).reset_index(drop=True)

print(f"\nConsolidated dataset: {len(consolidated)} rows")
print(consolidated.head(20))

# Calculate energy intensity for historical data
# Intensity = EUR / kWh (in EUR per kWh)
consolidated['Energy_Intensity_EUR_per_kWh'] = np.where(
    consolidated['GDP_Real_2015'].notna(),
    (consolidated['GDP_Real_2015'] * 1e9) / (consolidated['Final_Energy_TWh'] * 1e12) * 1000,
    np.nan
)

print("\nCalculated energy intensity (EUR of 2015 GDP per kWh):")
print(consolidated[consolidated['Scenario'] == 'Historical'][['Year', 'Final_Energy_TWh', 'GDP_Real_2015', 'Energy_Intensity_EUR_per_kWh']].tail(10))

# ========================================================================
# STEP 6: SAVE CONSOLIDATED DATA
# ========================================================================
print("\n" + "="*80)
print("SAVING CONSOLIDATED DATA")
print("="*80)

consolidated.to_excel(OUTPUT_FILE, index=False, sheet_name='Consolidated')
print(f"\n[OK] Consolidated data saved to: {OUTPUT_FILE}")

# Also save summary statistics
summary = consolidated.groupby('Scenario').agg({
    'Year': ['min', 'max', 'count'],
    'Final_Energy_TWh': ['min', 'max', 'mean'],
    'GDP_Real_2015': ['min', 'max', 'mean'],
    'Energy_Intensity_EUR_per_kWh': ['min', 'max', 'mean']
}).round(4)

print("\nSummary by Scenario:")
print(summary)

# Save detailed summary
with pd.ExcelWriter(OUTPUT_DIR / 'FR_NRJ_GDP_summary.xlsx', engine='openpyxl') as writer:
    consolidated.to_excel(writer, sheet_name='Data', index=False)
    summary.to_excel(writer, sheet_name='Summary')

print(f"[OK] Summary saved to: {OUTPUT_DIR / 'FR_NRJ_GDP_summary.xlsx'}")

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nNext steps:")
print("  1. Review consolidated data")
print("  2. Create visualizations (energy trends, intensity trends, scenarios)")
print("  3. Generate comparative analysis")
