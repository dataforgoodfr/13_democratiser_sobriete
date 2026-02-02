"""
FR Energy Intensity Analysis - Scenarios Visualization
Uses GDP scenario indices to project scenarios and visualize energy intensity trends

Input files:
  - visuals NRJ-GDP/FR_NRJ_GDP_consolidated.xlsx: Historical data with intensity
  - data-NRJ-GDP/gdp_scenario_index.xlsx: GDP index (base 2018) for scenarios

Output:
  - Energy intensity visualization (historical + 2 scenarios)
  - Consolidated dataset with scenario projections
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_PATH = Path(__file__).parent  # stream3_visualization/Decomposition/reports/FR
DATA_PATH = BASE_PATH / 'data-NRJ-GDP'
OUTPUT_DIR = BASE_PATH / 'visuals NRJ-GDP'

CONSOLIDATED_FILE = OUTPUT_DIR / 'FR_NRJ_GDP_consolidated.xlsx'
GDP_SCENARIO_FILE = DATA_PATH / 'gdp_scenario_index.xlsx'
OUTPUT_VISUAL = OUTPUT_DIR / 'FR_Energy_Intensity_Scenarios.png'
OUTPUT_VISUAL_GDP_ENERGY = OUTPUT_DIR / 'FR_GDP_vs_Energy_Scenarios.png'

print("="*80)
print("FR ENERGY INTENSITY - SCENARIOS VISUALIZATION")
print("="*80)

# ============================================================================
# STEP 1: LOAD CONSOLIDATED DATA
# ============================================================================
print("\n1. LOADING CONSOLIDATED DATA")
print("-" * 80)

df_consolidated = pd.read_excel(CONSOLIDATED_FILE)
print(f"Loaded consolidated data: {len(df_consolidated)} rows")
print(df_consolidated.head(10))

# Extract historical data (1990-2024)
df_historical = df_consolidated[df_consolidated['Scenario'] == 'Historical'].copy()
print(f"\nHistorical data: {len(df_historical)} years")

# Recalculate intensity for historical data using new formula: kWh / EUR
df_historical['Energy_Intensity_EUR_per_kWh'] = (
    (df_historical['Final_Energy_TWh'] * 1e12) / 
    (df_historical['GDP_Real_2015'] * 1e9) / 1000
)

# Get 2018 reference values
df_2018 = df_historical[df_historical['Year'] == 2018]
if len(df_2018) > 0:
    gdp_2018 = df_2018['GDP_Real_2015'].iloc[0]
    energy_2018 = df_2018['Final_Energy_TWh'].iloc[0]
    print(f"\n2018 Reference Values:")
    print(f"  GDP (2015 base): {gdp_2018:.2f} billion EUR")
    print(f"  Final Energy: {energy_2018:.2f} TWh")
else:
    print("ERROR: 2018 data not found")
    exit(1)

# ============================================================================
# STEP 2: LOAD GDP SCENARIO INDEX
# ============================================================================
print("\n2. LOADING GDP SCENARIO INDEX (base 2018)")
print("-" * 80)

df_gdp_scenario = pd.read_excel(GDP_SCENARIO_FILE)
print(f"Loaded GDP scenario index: {len(df_gdp_scenario)} rows")
print(f"Columns: {df_gdp_scenario.columns.tolist()}")
print(df_gdp_scenario.head(15))

# Rename columns to standardized names
df_gdp_scenario = df_gdp_scenario.rename(columns={
    'scenario': 'Scenario',
    'year': 'Year',
    'index_2018': 'GDP_Index'
})

print(f"\nUnique years: {sorted(df_gdp_scenario['Year'].unique())}")
print(f"Unique scenarios: {df_gdp_scenario['Scenario'].unique().tolist()}")

# ============================================================================
# STEP 3: MERGE SCENARIOS WITH GDP PROJECTIONS
# ============================================================================
print("\n3. MERGING SCENARIOS WITH GDP PROJECTIONS")
print("-" * 80)

# Merge consolidated data with GDP scenario index
df_scenarios_full = df_consolidated[df_consolidated['Scenario'] != 'Historical'].copy()
df_scenarios_full = df_scenarios_full.merge(
    df_gdp_scenario[['Year', 'Scenario', 'GDP_Index']],
    on=['Year', 'Scenario'],
    how='left'
)

# Fill GDP for scenarios using the index
for idx, row in df_scenarios_full.iterrows():
    if pd.isna(row['GDP_Real_2015']) and not pd.isna(row['GDP_Index']):
        gdp_projected = gdp_2018 * (row['GDP_Index'] / 100)
        df_scenarios_full.at[idx, 'GDP_Real_2015'] = gdp_projected

# Calculate intensity for scenarios
df_scenarios_full['Energy_Intensity_EUR_per_kWh'] = (
    (df_scenarios_full['Final_Energy_TWh'] * 1e12) / 
    (df_scenarios_full['GDP_Real_2015'] * 1e9) / 1000
)

print(f"Scenarios with GDP filled: {len(df_scenarios_full)} rows")
print("\n2025-2030 sample data:")
print(df_scenarios_full[(df_scenarios_full['Year'] >= 2025) & (df_scenarios_full['Year'] <= 2030)][
    ['Year', 'Scenario', 'GDP_Real_2015', 'Final_Energy_TWh', 'Energy_Intensity_EUR_per_kWh']
])

# ============================================================================
# STEP 4: COMBINE HISTORICAL AND SCENARIOS
# ============================================================================
print("\n4. COMBINING HISTORICAL AND SCENARIO DATA")
print("-" * 80)

# Only keep relevant columns
df_scenarios_full = df_scenarios_full[['Year', 'Scenario', 'GDP_Real_2015', 'Final_Energy_TWh', 'Energy_Intensity_EUR_per_kWh']]

# Create complete dataset
df_complete = pd.concat([df_historical, df_scenarios_full], ignore_index=True)
df_complete = df_complete.sort_values(['Scenario', 'Year']).reset_index(drop=True)

print(f"Combined dataset: {len(df_complete)} rows")
print(f"\nScenarios in dataset: {df_complete['Scenario'].unique().tolist()}")

for scenario in df_complete['Scenario'].unique():
    scenario_data = df_complete[df_complete['Scenario'] == scenario]
    print(f"\n{scenario}:")
    print(f"  Years: {scenario_data['Year'].min()}-{scenario_data['Year'].max()}")
    print(f"  Data points: {len(scenario_data)}")
    if len(scenario_data) > 0:
        print(f"  Intensity range: {scenario_data['Energy_Intensity_EUR_per_kWh'].min():.3f} - {scenario_data['Energy_Intensity_EUR_per_kWh'].max():.3f} GDP base 2015 / kWh")

# ============================================================================
# STEP 5: CREATE VISUALIZATION
# ============================================================================
print("\n5. CREATING VISUALIZATION")
print("-" * 80)

fig, ax = plt.subplots(figsize=(14, 8))

# Color palette
colors = {
    'Historical': '#1f77b4',      # Blue
    'SNBC-3': '#ff7f0e',          # Orange
    'AME-2024': '#2ca02c'         # Green
}

linestyles = {
    'Historical': '-',
    'SNBC-3': '--',
    'AME-2024': '-.'
}

markers = {
    'Historical': 'o',
    'SNBC-3': 's',
    'AME-2024': '^'
}

# Plot each scenario
for scenario in sorted(df_complete['Scenario'].unique()):
    scenario_data = df_complete[df_complete['Scenario'] == scenario].sort_values('Year')
    
    ax.plot(
        scenario_data['Year'],
        scenario_data['Energy_Intensity_EUR_per_kWh'],
        label=scenario,
        color=colors.get(scenario, '#000000'),
        linestyle=linestyles.get(scenario, '-'),
        linewidth=2.5,
        marker=markers.get(scenario, 'o'),
        markersize=6,
        alpha=0.8
    )

# Formatting
ax.set_xlabel('Year', fontsize=16, fontweight='bold')
ax.set_ylabel('Intensité énergétique (kWh / EUR base 2015)', fontsize=16, fontweight='bold')
ax.set_title('France : Tendances de l\'intensité énergétique (1990-2050)\nDonnées historiques et projections de scénarios', 
             fontsize=18, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.legend(fontsize=15, loc='best', framealpha=0.95)

# Add vertical line to separate historical from scenarios
ax.axvline(x=2024.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.text(2024.5, ax.get_ylim()[1] * 0.95, 'Projection', rotation=90, 
        verticalalignment='top', fontsize=14, color='gray', alpha=0.7)

# Set Y-axis to start at 0
ax.set_ylim(bottom=0)
ax.set_ylim(top=ax.get_ylim()[1])  # Keep the max

plt.tight_layout()
plt.savefig(OUTPUT_VISUAL, dpi=300, bbox_inches='tight')
print(f"\n[OK] Visualization saved to: {OUTPUT_VISUAL}")

# ============================================================================
# STEP 6: CREATE SECOND VISUALIZATION - GDP vs ENERGY
# ============================================================================
print("\n6. CREATING GDP vs ENERGY VISUALIZATION")
print("-" * 80)

fig, ax = plt.subplots(figsize=(14, 8))

colors = {
    'Historical': '#1f77b4',
    'SNBC-3': '#ff7f0e',
    'AME-2024': '#2ca02c'
}

linestyles = {
    'Historical': '-',
    'SNBC-3': '--',
    'AME-2024': '-.'
}

markers = {
    'Historical': 'o',
    'SNBC-3': 's',
    'AME-2024': '^'
}

# Plot each scenario
for scenario in sorted(df_complete['Scenario'].unique()):
    scenario_data = df_complete[df_complete['Scenario'] == scenario].sort_values('Year')
    
    ax.plot(
        scenario_data['Final_Energy_TWh'],
        scenario_data['GDP_Real_2015'],
        label=scenario,
        color=colors.get(scenario, '#000000'),
        linestyle=linestyles.get(scenario, '-'),
        linewidth=2.5,
        marker=markers.get(scenario, 'o'),
        markersize=6,
        alpha=0.8
    )

# Formatting
ax.set_xlabel('Énergie finale (TWh)', fontsize=16, fontweight='bold')
ax.set_ylabel('PIB réel en base 2015 (Milliards EUR)', fontsize=16, fontweight='bold')
ax.set_title('France : PIB vs. Consommation d\'énergie (1990-2050)\nDonnées historiques et projections de scénarios', 
             fontsize=18, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.legend(fontsize=15, loc='best', framealpha=0.95)

# Set axes to start at (0, 0)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(OUTPUT_VISUAL_GDP_ENERGY, dpi=300, bbox_inches='tight')
print(f"[OK] GDP vs Energy visualization saved to: {OUTPUT_VISUAL_GDP_ENERGY}")
plt.close(fig)

# ============================================================================
# STEP 6: SAVE EXTENDED CONSOLIDATED DATA
# ============================================================================
print("\n6. SAVING EXTENDED CONSOLIDATED DATA")
print("-" * 80)

output_extended = OUTPUT_DIR / 'FR_NRJ_GDP_with_scenarios.xlsx'
df_complete.to_excel(output_extended, index=False, sheet_name='Data')
print(f"[OK] Extended data saved to: {output_extended}")

# Create summary
summary_by_scenario = df_complete.groupby('Scenario').agg({
    'Year': ['min', 'max', 'count'],
    'Final_Energy_TWh': ['min', 'max', 'mean'],
    'GDP_Real_2015': ['min', 'max', 'mean'],
    'Energy_Intensity_EUR_per_kWh': ['min', 'max', 'mean']
}).round(3)

print("\n" + "="*80)
print("SUMMARY BY SCENARIO")
print("="*80)
print(summary_by_scenario)

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print(f"\nKey findings:")
print(f"  - Historical intensity (2024): {df_historical[df_historical['Year']==2024]['Energy_Intensity_EUR_per_kWh'].iloc[0]:.3f} GDP base 2015 / kWh")
print(f"  - Historical trend: {df_historical[df_historical['Year']==1990]['Energy_Intensity_EUR_per_kWh'].iloc[0]:.3f} (1990) -> {df_historical[df_historical['Year']==2024]['Energy_Intensity_EUR_per_kWh'].iloc[0]:.3f} (2024) GDP base 2015 / kWh")

for scenario in sorted(df_scenarios_projected['Scenario'].unique()):
    scenario_data = df_scenarios_projected[df_scenarios_projected['Scenario'] == scenario]
    if len(scenario_data) > 0:
        intensity_2030 = scenario_data[scenario_data['Year']==2030]['Energy_Intensity_EUR_per_kWh']
        if len(intensity_2030) > 0:
            print(f"  - {scenario} intensity (2030): {intensity_2030.iloc[0]:.3f} GDP base 2015 / kWh")

print(f"\nOutput files:")
print(f"  - {OUTPUT_VISUAL}")
print(f"  - {output_extended}")
