"""
FR Scenarios - Generate Compiled Data and Comprehensive Visualizations
Consolidated script that:
1. Compiles interpolated data from raw input (Activity, Final Energy, GHG Emissions)
2. Generates 5 comprehensive visualizations (3 panels each for sectors, single graph for demography)

Input: stream3_visualization/Decomposition/data/2025-12-15_FR scenarios data_before computation.xlsx
Output:
  - Data: stream3_visualization/Decomposition/data/2025-01-06_FR_scenarios_compiled.xlsx
  - Visuals: stream3_visualization/Decomposition/reports/FR/visuals raw/
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
BASE_PATH = Path(__file__).parent.parent.parent  # Navigate to stream3_visualization/Decomposition
DATA_PATH = BASE_PATH / 'data'
INPUT_FILE = DATA_PATH / '2025-12-15_FR scenarios data_before computation.xlsx'
OUTPUT_FILE = DATA_PATH / '2025-01-06_FR_scenarios_compiled.xlsx'
OUTPUT_DIR = BASE_PATH / 'reports' / 'FR' / 'visuals raw'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION - Color palette from Well-being dashboard
# ============================================================================
# Base colors from app.py
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'

CONVERSION_FACTORS = {
    'Gasoline': 0.00082,      # toe/L
    'Diesel': 0.00092,        # toe/L
    'Electric': 0.000086,     # toe/kWh
}

# Scenario colors for line plots
COLORS = {
    'SNBC-3': '#fdb462',      # Orange-warm from palette
    'AME-2024': '#8dd3c7',    # Cyan-turquoise from palette
}

# ============================================================================
# STEP 1: LOAD DATA AND DEFINE INTERPOLATION FUNCTION
# ============================================================================
print("="*80)
print("FR SCENARIOS - COMPILED DATA & VISUALIZATIONS GENERATOR")
print("="*80)
print(f"\nLoading raw data from: {INPUT_FILE}")

df_raw = pd.read_excel(INPUT_FILE)
print(f"Total rows: {len(df_raw)}")

def interpolate_to_years(data_dict):
    """Linear interpolation between raw data points only"""
    if not data_dict:
        return {}
    
    years = np.array(sorted(data_dict.keys()))
    values = np.array([data_dict[y] for y in years])
    
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 1:
        return {}
    
    years_valid = years[valid_mask]
    values_valid = values[valid_mask]
    
    min_year = years_valid.min()
    max_year = years_valid.max()
    
    interpolated = {}
    target_years = list(range(min_year, max_year + 1))
    
    for year in target_years:
        if year in data_dict:
            interpolated[year] = float(data_dict[year])
        else:
            idx = np.searchsorted(years_valid, year)
            if idx > 0 and idx < len(years_valid):
                y1, y2 = years_valid[idx-1], years_valid[idx]
                v1, v2 = values_valid[idx-1], values_valid[idx]
                interpolated[year] = float(v1 + (v2 - v1) * (year - y1) / (y2 - y1))
    
    return interpolated

# ============================================================================
# STEP 2: COMPUTE COMPILED DATA
# ============================================================================
print("\n" + "="*80)
print("COMPUTING ACTIVITY & FINAL ENERGY DATA")
print("="*80)

compiled_data = []

# 1. RESIDENTIAL FLOOR AREA
print("\n1. RESIDENTIAL FLOOR AREA")
print("-" * 80)

for scenario in ['SNBC-3', 'AME-2024']:
    print(f"  {scenario}:")
    
    house_data = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Buildings - Residential') &
        (df_raw['Type'] == 'House number') &
        (df_raw['Variable'] == 'Million')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    fa_individual = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Buildings - Residential') &
        (df_raw['Type'] == 'Floor area') &
        (df_raw['Variable'] == 'Mean floor area individual')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    fa_collective = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Buildings - Residential') &
        (df_raw['Type'] == 'Floor area') &
        (df_raw['Variable'] == 'Mean floor area collective')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    house_interp = interpolate_to_years(house_data)
    fa_ind_interp = interpolate_to_years(fa_individual)
    fa_coll_interp = interpolate_to_years(fa_collective)
    
    common_years = set(house_interp.keys()) & set(fa_ind_interp.keys()) & set(fa_coll_interp.keys())
    
    for year in sorted(common_years):
        house_count_val = house_interp[year]
        fa_ind_val = fa_ind_interp[year]
        fa_coll_val = fa_coll_interp[year]
        
        avg_fa = 0.548 * fa_ind_val + 0.452 * fa_coll_val
        total_fa = house_count_val * avg_fa
        
        compiled_data.append({
            'Geography': 'France',
            'Scenario': scenario,
            'Sector': 'Buildings - Residential',
            'Category': 'Activity',
            'Type': 'Floor area',
            'Year': year,
            'Value': total_fa,
            'Unit': 'Mm²',
        })
        
        print(f"    {year}: {total_fa:.2f} Mm²")

# 1b. SERVICES FLOOR AREA
print("\n1b. SERVICES FLOOR AREA")
print("-" * 80)

for scenario in ['SNBC-3', 'AME-2024']:
    print(f"  {scenario}:")
    
    fa_services = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Buildings - Services') &
        (df_raw['Type'] == 'Floor area') &
        (df_raw['Variable'] == 'Floor area service')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    fa_serv_interp = interpolate_to_years(fa_services)
    
    for year in sorted(fa_serv_interp.keys()):
        total_fa = fa_serv_interp[year]
        
        compiled_data.append({
            'Geography': 'France',
            'Scenario': scenario,
            'Sector': 'Buildings - Services',
            'Category': 'Activity',
            'Type': 'Floor area',
            'Year': year,
            'Value': total_fa,
            'Unit': 'Mm²',
        })
        
        print(f"    {year}: {total_fa:.2f} Mm²")

# 1c. TRANSPORT - CAR PASSENGER KILOMETERS (Activity)
print("\n1c. TRANSPORT - CAR PASSENGER KILOMETERS (Activity)")
print("-" * 80)

for scenario in ['SNBC-3', 'AME-2024']:
    print(f"  {scenario}:")
    
    car_pkm = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Passenger kilometers') &
        (df_raw['Variable'] == 'Car')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    car_pkm_interp = interpolate_to_years(car_pkm)
    
    for year in sorted(car_pkm_interp.keys()):
        pkm = car_pkm_interp[year]
        
        compiled_data.append({
            'Geography': 'France',
            'Scenario': scenario,
            'Sector': 'Transport - Passenger car and motorcycle',
            'Category': 'Activity',
            'Type': 'Car',
            'Year': year,
            'Value': pkm,
            'Unit': 'Gpkm',
        })
        
        print(f"    {year}: {pkm:.2f} Gpkm")

# 1d. TRANSPORT - MOTORCYCLE PASSENGER KILOMETERS (Activity)
print("\n1d. TRANSPORT - MOTORCYCLE PASSENGER KILOMETERS (Activity)")
print("-" * 80)

for scenario in ['SNBC-3', 'AME-2024']:
    print(f"  {scenario}:")
    
    moto_pkm = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Passenger kilometers') &
        (df_raw['Variable'] == 'Motorcycle')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    moto_pkm_interp = interpolate_to_years(moto_pkm)
    
    for year in sorted(moto_pkm_interp.keys()):
        pkm = moto_pkm_interp[year]
        
        compiled_data.append({
            'Geography': 'France',
            'Scenario': scenario,
            'Sector': 'Transport - Passenger car and motorcycle',
            'Category': 'Activity',
            'Type': 'Motorcycle',
            'Year': year,
            'Value': pkm,
            'Unit': 'Gpkm',
        })
        
        print(f"    {year}: {pkm:.2f} Gpkm")

# 2. TRANSPORT - FINAL ENERGY FOR CAR & MOTORCYCLE
print("\n2. TRANSPORT - FINAL ENERGY")
print("-" * 80)

for scenario in ['SNBC-3', 'AME-2024']:
    print(f"  {scenario}:")
    
    car_vkm = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Vehicle kilometers') &
        (df_raw['Variable'] == 'Car')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    car_diesel_share = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Diesel') &
        (df_raw['Variable'] == 'Car')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    car_gasoline_share = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Essence') &
        (df_raw['Variable'] == 'Car')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    car_electric_share = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Electric') &
        (df_raw['Variable'] == 'Car')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    car_hybrid_share = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Hybrid') &
        (df_raw['Variable'] == 'Car')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    car_diesel_consumption = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Diesel') &
        (df_raw['Variable'] == 'Car and Motorcycle')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    car_gasoline_consumption = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Gasoline') &
        (df_raw['Variable'] == 'Car and Motorcycle')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    car_electric_consumption = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Electrical vehicles') &
        (df_raw['Variable'] == 'Car and Motorcycle')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    moto_vkm = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Vehicle kilometers') &
        (df_raw['Variable'] == 'Motorcycle')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    moto_gasoline_consumption = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Transport - Passenger car and motorcycle') &
        (df_raw['Type'] == 'Gasoline') &
        (df_raw['Variable'] == 'Car and Motorcycle')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    car_vkm_interp = interpolate_to_years(car_vkm)
    car_diesel_share_interp = interpolate_to_years(car_diesel_share)
    car_gasoline_share_interp = interpolate_to_years(car_gasoline_share)
    car_electric_share_interp = interpolate_to_years(car_electric_share)
    car_hybrid_share_interp = interpolate_to_years(car_hybrid_share)
    car_diesel_cons_interp = interpolate_to_years(car_diesel_consumption)
    car_gasoline_cons_interp = interpolate_to_years(car_gasoline_consumption)
    car_electric_cons_interp = interpolate_to_years(car_electric_consumption)
    
    moto_vkm_interp = interpolate_to_years(moto_vkm)
    moto_gasoline_cons_interp = interpolate_to_years(moto_gasoline_consumption)
    
    # CAR final energy
    print("    CAR:")
    car_years = set(car_vkm_interp.keys())
    car_years = {year for year in car_years if year >= 2021}
    
    for year in sorted(car_years):
        vkm = car_vkm_interp[year]
        final_energy_toe = 0
        
        diesel_stock = car_diesel_share_interp.get(year, 0)
        gasoline_stock = car_gasoline_share_interp.get(year, 0)
        electric_stock = car_electric_share_interp.get(year, 0)
        hybrid_stock = car_hybrid_share_interp.get(year, 0)
        
        diesel_cons = car_diesel_cons_interp.get(year, 0)
        gasoline_cons = car_gasoline_cons_interp.get(year, 0)
        electric_cons = car_electric_cons_interp.get(year, 0)
        
        if diesel_stock > 0 and diesel_cons > 0:
            final_energy_toe += (vkm * 1e9) * diesel_stock * (diesel_cons / 100) * CONVERSION_FACTORS['Diesel']
        if gasoline_stock > 0 and gasoline_cons > 0:
            final_energy_toe += (vkm * 1e9) * gasoline_stock * (gasoline_cons / 100) * CONVERSION_FACTORS['Gasoline']
        if electric_stock > 0 and electric_cons > 0:
            final_energy_toe += (vkm * 1e9) * electric_stock * (electric_cons / 100) * CONVERSION_FACTORS['Electric']
        if hybrid_stock > 0 and gasoline_cons > 0:
            final_energy_toe += (vkm * 1e9) * hybrid_stock * (gasoline_cons / 100) * CONVERSION_FACTORS['Gasoline']
        
        compiled_data.append({
            'Geography': 'France',
            'Scenario': scenario,
            'Sector': 'Transport - Passenger car and motorcycle',
            'Category': 'Final energy',
            'Type': 'Car',
            'Year': year,
            'Value': final_energy_toe / 1e6,
            'Unit': 'Mtoe',
        })
        
        print(f"      {year}: {final_energy_toe / 1e6:.4f} Mtoe")
    
    # MOTORCYCLE final energy
    print("    MOTORCYCLE:")
    moto_years = set(moto_vkm_interp.keys())
    moto_years = {year for year in moto_years if year >= 2021}
    
    for year in sorted(moto_years):
        vkm = moto_vkm_interp[year]
        final_energy_toe = 0
        
        if year in moto_gasoline_cons_interp:
            gasoline_cons = moto_gasoline_cons_interp[year]
            final_energy_toe = (vkm * 1e9) * (gasoline_cons / 100) * CONVERSION_FACTORS['Gasoline']
        
        compiled_data.append({
            'Geography': 'France',
            'Scenario': scenario,
            'Sector': 'Transport - Passenger car and motorcycle',
            'Category': 'Final energy',
            'Type': 'Motorcycle',
            'Year': year,
            'Value': final_energy_toe / 1e6,
            'Unit': 'Mtoe',
        })
        
        print(f"      {year}: {final_energy_toe / 1e6:.4f} Mtoe")

# 2b. BUILDINGS - RESIDENTIAL FINAL ENERGY
print("\n2b. BUILDINGS - RESIDENTIAL FINAL ENERGY")
print("-" * 80)

for scenario in ['SNBC-3', 'AME-2024']:
    print(f"  {scenario}:")
    
    fe_residential = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Buildings - Residential') &
        (df_raw['Type'] == 'Final energy') &
        (df_raw['Variable'] == 'Residential')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    fe_res_interp = interpolate_to_years(fe_residential)
    
    for year in sorted(fe_res_interp.keys()):
        fe = fe_res_interp[year]
        
        compiled_data.append({
            'Geography': 'France',
            'Scenario': scenario,
            'Sector': 'Buildings - Residential',
            'Category': 'Final energy',
            'Type': 'Residential',
            'Year': year,
            'Value': fe,
            'Unit': 'Mtoe',
        })
        
        print(f"    {year}: {fe:.4f} Mtoe")

# 2c. BUILDINGS - SERVICES FINAL ENERGY
print("\n2c. BUILDINGS - SERVICES FINAL ENERGY")
print("-" * 80)

for scenario in ['SNBC-3', 'AME-2024']:
    print(f"  {scenario}:")
    
    fe_services = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Buildings - Services') &
        (df_raw['Type'] == 'Final energy') &
        (df_raw['Variable'] == 'Services')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    fe_srv_interp = interpolate_to_years(fe_services)
    
    for year in sorted(fe_srv_interp.keys()):
        fe = fe_srv_interp[year]
        
        compiled_data.append({
            'Geography': 'France',
            'Scenario': scenario,
            'Sector': 'Buildings - Services',
            'Category': 'Final energy',
            'Type': 'Services',
            'Year': year,
            'Value': fe,
            'Unit': 'Mtoe',
        })
        
        print(f"    {year}: {fe:.4f} Mtoe")

# 2c. AGRICULTURE - CATTLE BREEDING (Activity)
print("\n2c. AGRICULTURE - CATTLE BREEDING (Activity)")
print("-" * 80)

for scenario in ['SNBC-3', 'AME-2024']:
    print(f"  {scenario}:")
    
    cattle_data = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Agriculture - Cattle breeding') &
        (df_raw['Type'] == 'Number') &
        (df_raw['Variable'] == 'Cattle breeding')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    cattle_interp = interpolate_to_years(cattle_data)
    
    for year in sorted(cattle_interp.keys()):
        cattle = cattle_interp[year]
        
        compiled_data.append({
            'Geography': 'France',
            'Scenario': scenario,
            'Sector': 'Agriculture - Cattle breeding',
            'Category': 'Activity',
            'Type': 'Cattle breeding',
            'Year': year,
            'Value': cattle,
            'Unit': 'Million heads',
        })
        
        print(f"    {year}: {cattle:.2f} Million heads")

# 2d. AGRICULTURE - CULTURE (Activity - Agricultural land)
print("\n2d. AGRICULTURE - CULTURE (Activity - Agricultural land)")
print("-" * 80)

for scenario in ['SNBC-3', 'AME-2024']:
    print(f"  {scenario}:")
    
    culture_data = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Agriculture - Culture') &
        (df_raw['Type'] == 'Agricultural land') &
        (df_raw['Variable'] == 'Culture')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    culture_interp = interpolate_to_years(culture_data)
    
    for year in sorted(culture_interp.keys()):
        culture = culture_interp[year]
        
        compiled_data.append({
            'Geography': 'France',
            'Scenario': scenario,
            'Sector': 'Agriculture - Culture',
            'Category': 'Activity',
            'Type': 'Culture',
            'Year': year,
            'Value': culture,
            'Unit': 'kilo hectares',
        })
        
        print(f"    {year}: {culture:.2f} kilo hectares")

# 3. DEMOGRAPHY
print("\n3. DEMOGRAPHY - POPULATION")
print("-" * 80)

for scenario in ['SNBC-3', 'AME-2024']:
    pop_data = df_raw[
        (df_raw['Scenario'] == scenario) &
        (df_raw['Sector'] == 'Demography') &
        (df_raw['Type'] == 'Demography') &
        (df_raw['Variable'] == 'Population')
    ].groupby('Year')['Volume'].sum().to_dict()
    
    if not pop_data:
        print(f"  No data for {scenario}")
        continue
    
    print(f"  {scenario}:")
    
    for year in sorted(pop_data.keys()):
        compiled_data.append({
            'Geography': 'France',
            'Scenario': scenario,
            'Sector': 'Demography',
            'Category': 'Population',
            'Type': 'Population',
            'Year': year,
            'Value': pop_data[year],
            'Unit': 'Millions',
        })
        
        print(f"    {year}: {pop_data[year]:.2f} M")

# 4. GHG EMISSIONS
print("\n4. GHG EMISSIONS")
print("-" * 80)

ghg_mapping = [
    ('Buildings - Residential', 'GHG Emissions', 'Residential', 'Buildings - Residential'),
    ('Buildings - Services', 'GHG Emissions', 'Services', 'Buildings - Services'),
    ('Transport - Passenger car and motorcycle', 'GHG Emissions', 'Car', 'Transport - Car'),
    ('Transport - Passenger car and motorcycle', 'GHG Emissions', 'Motorcycle', 'Transport - Motorcycle'),
    ('Agriculture - Cattle breeding', 'GHG Emissions', 'Cattle breeding', 'Agriculture - Cattle breeding'),
    ('Agriculture - Culture', 'GHG Emissions', 'Culture', 'Agriculture - Culture'),
]

for sector, type_val, var, output_sector in ghg_mapping:
    for scenario in ['SNBC-3', 'AME-2024']:
        subset = df_raw[
            (df_raw['Scenario'] == scenario) &
            (df_raw['Sector'] == sector) &
            (df_raw['Type'] == type_val) &
            (df_raw['Variable'] == var)
        ].groupby('Year')['Volume'].sum().to_dict()
        
        if not subset:
            print(f"  No data for {scenario} - {output_sector}")
            continue
        
        print(f"  {scenario} - {output_sector}:")
        
        for year in sorted(subset.keys()):
            compiled_data.append({
                'Geography': 'France',
                'Scenario': scenario,
                'Sector': output_sector,
                'Category': 'GHG emissions',
                'Type': var,
                'Year': year,
                'Value': subset[year],
                'Unit': 'MtCO2e',
            })
            
            print(f"    {year}: {subset[year]:.2f} MtCO2e")

# Save compiled data
print("\n" + "="*80)
print("SAVING COMPILED DATA")
print("="*80)

df_compiled = pd.DataFrame(compiled_data)

# Interpolate all data to fill all years linearly
print("\nInterpolating all data to fill years linearly...")

interpolated_rows = []

for scenario in ['SNBC-3', 'AME-2024']:
    for sector in df_compiled['Sector'].unique():
        for category in df_compiled[df_compiled['Sector'] == sector]['Category'].unique():
            for type_val in df_compiled[(df_compiled['Sector'] == sector) & 
                                        (df_compiled['Category'] == category)]['Type'].unique():
                
                # Get data for this scenario/sector/category/type combination
                subset = df_compiled[
                    (df_compiled['Scenario'] == scenario) &
                    (df_compiled['Sector'] == sector) &
                    (df_compiled['Category'] == category) &
                    (df_compiled['Type'] == type_val)
                ].sort_values('Year')
                
                if len(subset) < 1:
                    continue
                
                # Get unit from first row
                unit = subset.iloc[0]['Unit']
                
                # Create year->value mapping
                year_value_dict = dict(zip(subset['Year'], subset['Value']))
                
                # Interpolate to all years
                if len(year_value_dict) > 1:
                    years = sorted(year_value_dict.keys())
                    min_year = years[0]
                    max_year = years[-1]
                    
                    for year in range(min_year, max_year + 1):
                        if year in year_value_dict:
                            value = year_value_dict[year]
                        else:
                            # Linear interpolation
                            idx = next((i for i, y in enumerate(years) if y > year), None)
                            if idx is not None and idx > 0:
                                y1, y2 = years[idx-1], years[idx]
                                v1, v2 = year_value_dict[y1], year_value_dict[y2]
                                value = v1 + (v2 - v1) * (year - y1) / (y2 - y1)
                            else:
                                continue
                        
                        interpolated_rows.append({
                            'Geography': subset.iloc[0]['Geography'],
                            'Scenario': scenario,
                            'Sector': sector,
                            'Category': category,
                            'Type': type_val,
                            'Year': year,
                            'Value': value,
                            'Unit': unit,
                        })

# Create new dataframe with interpolated data
df_compiled = pd.DataFrame(interpolated_rows).sort_values(['Scenario', 'Sector', 'Category', 'Type', 'Year'])

print(f"\nTotal rows after interpolation: {len(df_compiled)}")

df_compiled.to_excel(OUTPUT_FILE, index=False, sheet_name='Compiled Data')
print(f"[OK] Saved to: {OUTPUT_FILE}")

# ============================================================================
# STEP 3: GENERATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING COMPREHENSIVE VISUALIZATIONS")
print("="*80)

# 1. RESIDENTIAL - Activity, Final Energy, Emissions (3 panels)
print("\nGenerating Residential comprehensive view (3 panels)...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

activity_res = df_compiled[
    (df_compiled['Sector'] == 'Buildings - Residential') &
    (df_compiled['Category'] == 'Activity')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = activity_res[activity_res['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[0].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[0].set_ylabel('Activity (Mm²)', fontsize=11, fontweight='bold')
axes[0].set_title('Buildings - Residential: Activity (Floor Area)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10, loc='best')
axes[0].grid(True, alpha=0.3)

# Final Energy from compiled data
fe_res = df_compiled[
    (df_compiled['Sector'] == 'Buildings - Residential') &
    (df_compiled['Category'] == 'Final energy')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = fe_res[fe_res['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[1].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[1].set_ylabel('Final Energy (Mtoe)', fontsize=11, fontweight='bold')
axes[1].set_title('Buildings - Residential: Final Energy', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10, loc='best')
axes[1].grid(True, alpha=0.3)

emissions_res = df_compiled[
    (df_compiled['Sector'] == 'Buildings - Residential') &
    (df_compiled['Category'] == 'GHG emissions')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = emissions_res[emissions_res['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[2].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[2].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Emissions (MtCO2e)', fontsize=11, fontweight='bold')
axes[2].set_title('Buildings - Residential: GHG Emissions', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10, loc='best')
axes[2].grid(True, alpha=0.3)

fig.suptitle('Buildings - Residential: Comprehensive Overview', fontsize=14, fontweight='bold')
fig.tight_layout()
output_file = OUTPUT_DIR / 'FR_Comprehensive_Residential.png'
fig.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {output_file.name}")
plt.close(fig)

# 2. SERVICES - Activity, Final Energy, Emissions (3 panels)
print("Generating Services comprehensive view (3 panels)...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

activity_srv = df_compiled[
    (df_compiled['Sector'] == 'Buildings - Services') &
    (df_compiled['Category'] == 'Activity')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = activity_srv[activity_srv['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[0].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[0].set_ylabel('Activity (Mm²)', fontsize=11, fontweight='bold')
axes[0].set_title('Buildings - Services: Activity (Floor Area)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10, loc='best')
axes[0].grid(True, alpha=0.3)

# Final Energy from compiled data
fe_srv = df_compiled[
    (df_compiled['Sector'] == 'Buildings - Services') &
    (df_compiled['Category'] == 'Final energy')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = fe_srv[fe_srv['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[1].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[1].set_ylabel('Final Energy (Mtoe)', fontsize=11, fontweight='bold')
axes[1].set_title('Buildings - Services: Final Energy', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10, loc='best')
axes[1].grid(True, alpha=0.3)

emissions_srv = df_compiled[
    (df_compiled['Sector'] == 'Buildings - Services') &
    (df_compiled['Category'] == 'GHG emissions')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = emissions_srv[emissions_srv['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[2].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[2].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Emissions (MtCO2e)', fontsize=11, fontweight='bold')
axes[2].set_title('Buildings - Services: GHG Emissions', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10, loc='best')
axes[2].grid(True, alpha=0.3)

fig.suptitle('Buildings - Services: Comprehensive Overview', fontsize=14, fontweight='bold')
fig.tight_layout()
output_file = OUTPUT_DIR / 'FR_Comprehensive_Services.png'
fig.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {output_file.name}")
plt.close(fig)

# 3. CAR - Activity, Final Energy, Emissions (3 panels)
print("Generating Car comprehensive view (3 panels)...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Activity from compiled data
activity_car = df_compiled[
    (df_compiled['Sector'] == 'Transport - Passenger car and motorcycle') &
    (df_compiled['Category'] == 'Activity') &
    (df_compiled['Type'] == 'Car')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = activity_car[activity_car['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[0].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[0].set_ylabel('Activity (Gpkm)', fontsize=11, fontweight='bold')
axes[0].set_title('Transport - Car: Activity (Passenger Kilometers)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10, loc='best')
axes[0].grid(True, alpha=0.3)

final_energy_car = df_compiled[
    (df_compiled['Type'] == 'Car') &
    (df_compiled['Category'] == 'Final energy')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = final_energy_car[final_energy_car['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[1].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[1].set_ylabel('Final Energy (Mtoe)', fontsize=11, fontweight='bold')
axes[1].set_title('Transport - Car: Final Energy', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10, loc='best')
axes[1].grid(True, alpha=0.3)

emissions_car = df_compiled[
    (df_compiled['Type'] == 'Car') &
    (df_compiled['Category'] == 'GHG emissions')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = emissions_car[emissions_car['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[2].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[2].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Emissions (MtCO2e)', fontsize=11, fontweight='bold')
axes[2].set_title('Transport - Car: GHG Emissions', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10, loc='best')
axes[2].grid(True, alpha=0.3)

fig.suptitle('Transport - Car: Comprehensive Overview', fontsize=14, fontweight='bold')
fig.tight_layout()
output_file = OUTPUT_DIR / 'FR_Comprehensive_Car.png'
fig.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {output_file.name}")
plt.close(fig)

# 4. MOTORCYCLE - Activity, Final Energy, Emissions (3 panels)
print("Generating Motorcycle comprehensive view (3 panels)...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Activity from compiled data
activity_moto = df_compiled[
    (df_compiled['Sector'] == 'Transport - Passenger car and motorcycle') &
    (df_compiled['Category'] == 'Activity') &
    (df_compiled['Type'] == 'Motorcycle')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = activity_moto[activity_moto['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[0].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[0].set_ylabel('Activity (Gpkm)', fontsize=11, fontweight='bold')
axes[0].set_title('Transport - Motorcycle: Activity (Passenger Kilometers)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10, loc='best')
axes[0].grid(True, alpha=0.3)

# Final Energy from compiled data
final_energy_moto = df_compiled[
    (df_compiled['Type'] == 'Motorcycle') &
    (df_compiled['Category'] == 'Final energy')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = final_energy_moto[final_energy_moto['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[1].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[1].set_ylabel('Final Energy (Mtoe)', fontsize=11, fontweight='bold')
axes[1].set_title('Transport - Motorcycle: Final Energy', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10, loc='best')
axes[1].grid(True, alpha=0.3)

emissions_moto = df_compiled[
    (df_compiled['Type'] == 'Motorcycle') &
    (df_compiled['Category'] == 'GHG emissions')
]
for scenario in ['SNBC-3', 'AME-2024']:
    data = emissions_moto[emissions_moto['Scenario'] == scenario].sort_values('Year')
    if len(data) > 0:
        axes[2].plot(data['Year'], data['Value'], marker='o', label=scenario, 
                    color=COLORS[scenario], linewidth=2)

axes[2].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Emissions (MtCO2e)', fontsize=11, fontweight='bold')
axes[2].set_title('Transport - Motorcycle: GHG Emissions', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10, loc='best')
axes[2].grid(True, alpha=0.3)

fig.suptitle('Transport - Motorcycle: Comprehensive Overview', fontsize=14, fontweight='bold')
fig.tight_layout()
output_file = OUTPUT_DIR / 'FR_Comprehensive_Motorcycle.png'
fig.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {output_file.name}")
plt.close(fig)

# 5. DEMOGRAPHY - Population only
print("Generating Demography visualization (Population)...")

fig, ax = plt.subplots(figsize=(12, 6))

pop_data = df_compiled[
    (df_compiled['Category'] == 'Population')
]

# Plot AME-2024 first
ame_data = pop_data[pop_data['Scenario'] == 'AME-2024'].sort_values('Year')
if len(ame_data) > 0:
    ax.plot(ame_data['Year'], ame_data['Value'], marker='o', label='AME-2024', 
            color=COLORS['AME-2024'], linewidth=2.5, markersize=8, linestyle='-')

# Plot SNBC-3 on top with dashed line
snbc_data = pop_data[pop_data['Scenario'] == 'SNBC-3'].sort_values('Year')
if len(snbc_data) > 0:
    ax.plot(snbc_data['Year'], snbc_data['Value'], marker='s', label='SNBC-3', 
            color=COLORS['SNBC-3'], linewidth=2.5, markersize=8, linestyle='--')

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Population (Millions)', fontsize=12, fontweight='bold')
ax.set_title('Demography: Population', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

fig.tight_layout()
output_file = OUTPUT_DIR / 'FR_Comprehensive_Demography.png'
fig.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {output_file.name}")
plt.close(fig)

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"\nCompiled data saved to: {OUTPUT_FILE}")
print(f"\nVisualizations saved to: {OUTPUT_DIR}")
print(f"\nGenerated files:")
print(f"  - FR_Comprehensive_Residential.png (3 panels)")
print(f"  - FR_Comprehensive_Services.png (3 panels)")
print(f"  - FR_Comprehensive_Car.png (3 panels)")
print(f"  - FR_Comprehensive_Motorcycle.png (3 panels)")
print(f"  - FR_Comprehensive_Demography.png (Population)")
print("="*80)
