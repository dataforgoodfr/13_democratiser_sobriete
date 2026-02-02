"""
MASTER SCRIPT: Generate all 20 series visuals with all scenarios
Combines historical (with interpolation) + SNBC-3 + AME-2024 on same graphs
Output: stream3_visualization\Decomposition\reports\FR\visuals tests
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path('stream3_visualization/Decomposition')
DATA_PATH = BASE_PATH / 'data'
VISUALS_DIR = BASE_PATH / 'reports' / 'FR' / 'visuals tests'
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PATH / '2025-12-15_FR scenarios data_before computation.xlsx'

def extract_series_data(df, sector, type_val, variable, scenario):
    """Extract raw data for a series and scenario"""
    subset = df[
        (df['Sector'] == sector) &
        (df['Type'] == type_val) &
        (df['Variable'] == variable) &
        (df['Scenario'] == scenario)
    ]
    
    if subset.empty:
        return {}
    
    data_by_year = subset.groupby('Year')['Volume'].sum().to_dict()
    return data_by_year

def interpolate_data(years, values, smoothing=0.5):
    """Spline interpolation"""
    if len(years) < 4:
        return years, values
    
    try:
        spline = UnivariateSpline(years, values, s=smoothing, k=min(3, len(years)-1))
        year_range = np.arange(min(years), max(years) + 1)
        interpolated = spline(year_range)
        return year_range, interpolated
    except:
        return years, values

def plot_all_scenarios(series_config):
    """
    Plot historical (raw + interp) + scenarios (raw + interp) on same graph
    Historical format preserved exactly as original
    """
    df = pd.read_excel(INPUT_FILE)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    has_historical = False
    has_scenarios = False
    
    # ===== HISTORICAL (keep exact original format) =====
    data_dict = extract_series_data(
        df,
        series_config['sector'],
        series_config['type'],
        series_config['variable'],
        'historical'
    )
    
    if data_dict:
        has_historical = True
        years = sorted(data_dict.keys())
        values = [data_dict[y] for y in years]
        
        years_arr = np.array(years, dtype=float)
        values_arr = np.array(values, dtype=float)
        
        # Raw data points (scatter)
        ax.scatter(years_arr, values_arr, color='#1f77b4', s=80, alpha=0.5, 
                  label='Historical (raw)', zorder=3)
        
        # Interpolated curve (line)
        interp_years, interp_values = interpolate_data(years_arr, values_arr, smoothing=0.5)
        ax.plot(interp_years, interp_values, color='#1f77b4', linewidth=2.5, 
               alpha=0.8, label='Historical (interp)', zorder=2)
    
    # ===== SCENARIOS (SNBC-3 & AME-2024) =====
    colors_scenario = {'SNBC-3': '#ff7f0e', 'AME-2024': '#2ca02c'}
    markers_scenario = {'SNBC-3': 'o', 'AME-2024': 's'}
    
    for scenario in ['SNBC-3', 'AME-2024']:
        data_dict = extract_series_data(
            df,
            series_config['sector'],
            series_config['type'],
            series_config['variable'],
            scenario
        )
        
        if not data_dict:
            continue
        
        has_scenarios = True
        years = sorted(data_dict.keys())
        values = [data_dict[y] for y in years]
        
        years_arr = np.array(years, dtype=float)
        values_arr = np.array(values, dtype=float)
        
        # Raw data points (scatter)
        ax.scatter(years_arr, values_arr, color=colors_scenario[scenario], 
                  marker=markers_scenario[scenario], s=100, alpha=0.6, 
                  label=f'{scenario} (raw)', zorder=4)
        
        # Interpolated curve (line)
        interp_years, interp_values = interpolate_data(years_arr, values_arr, smoothing=0.5)
        ax.plot(interp_years, interp_values, color=colors_scenario[scenario], 
               linewidth=2.5, alpha=0.8, label=f'{scenario} (interp)', zorder=2)
    
    if not (has_historical or has_scenarios):
        return None
    
    # Formatting
    ax.set_title(f"{series_config['title']}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    filename = VISUALS_DIR / f"final_{series_config['filename']}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

# Define all 20 series
SERIES_CONFIGS = [
    # Series 1-6: Buildings Residential
    {
        'sector': 'Buildings - Residential',
        'type': 'Final energy',
        'variable': 'Residential',
        'title': '1. Buildings - Residential Final Energy',
        'filename': '1_Buildings_Final_energy'
    },
    {
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Floor area collective',
        'title': '2. Buildings - Floor Area Collective',
        'filename': '2_Buildings_Floor_area_collective'
    },
    {
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Floor area individual',
        'title': '3. Buildings - Floor Area Individual',
        'filename': '3_Buildings_Floor_area_individual'
    },
    {
        'sector': 'Buildings - Residential',
        'type': 'GHG Emissions',
        'variable': 'Residential',
        'title': '4. Buildings - Residential GHG Emissions',
        'filename': '4_Buildings_GHG_Residential'
    },
    {
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Mean floor area',
        'title': '5. Buildings - Mean Floor Area',
        'filename': '5_Buildings_Mean_floor_area'
    },
    {
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Residential',
        'title': '6. Buildings - Floor Area',
        'filename': '6_Buildings_Floor_area'
    },
    
    # Series 7-9: Buildings Services
    {
        'sector': 'Buildings - Services',
        'type': 'Final energy',
        'variable': 'Services',
        'title': '7. Buildings - Services Final Energy',
        'filename': '7_Buildings_Services_Final_energy'
    },
    {
        'sector': 'Buildings - Services',
        'type': 'Floor area',
        'variable': 'Floor area service',
        'title': '8. Buildings - Services Floor Area',
        'filename': '8_Buildings_Services_Floor_area'
    },
    {
        'sector': 'Buildings - Services',
        'type': 'GHG Emissions',
        'variable': 'Services',
        'title': '9. Buildings - Services GHG Emissions',
        'filename': '9_Buildings_Services_GHG'
    },
    
    # Series 10: Demography
    {
        'sector': 'Demography',
        'type': 'Demography',
        'variable': 'Population',
        'title': '10. Demography - Population',
        'filename': '10_Demography_Population'
    },
    
    # Series 11-12: Transport Freight
    {
        'sector': 'Transport - Freight',
        'type': 'Freight transport',
        'variable': 'Inland transport',
        'title': '11. Transport - Freight Inland',
        'filename': '11_Transport_Freight_Inland'
    },
    {
        'sector': 'Transport - Freight',
        'type': 'Passenger kilometers',
        'variable': 'Buses',
        'title': '12. Transport - Buses',
        'filename': '12_Transport_Freight_Buses'
    },
    
    # Series 13-18: Transport Passenger
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Consumption',
        'variable': 'Car',
        'title': '13. Transport - Consumption Car',
        'filename': '13_Transport_Consumption_Car'
    },
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Vehicle kilometers',
        'variable': 'Car',
        'title': '14. Transport - Vehicle km Car',
        'filename': '14_Transport_Vehicle_km_Car'
    },
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Passenger kilometers',
        'variable': 'Car',
        'title': '15. Transport - Passenger km Car',
        'filename': '15_Transport_Passenger_km_Car'
    },
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'GHG Emissions',
        'variable': 'Passenger transport',
        'title': '16. Transport - GHG Emissions Passenger',
        'filename': '16_Transport_GHG_Passenger'
    },
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Passenger kilometers',
        'variable': 'Motorcycle',
        'title': '17. Transport - Passenger km Motorcycle',
        'filename': '17_Transport_Passenger_km_Motorcycle'
    },
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Passenger kilometers',
        'variable': 'Train',
        'title': '18. Transport - Passenger km Train',
        'filename': '18_Transport_Passenger_km_Train'
    },
    
    # Series 19-20: Scenario data (fuel types for cars)
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Diesel',
        'variable': 'Car ',
        'title': '19. Transport - Diesel Percentage (Car)',
        'filename': '19_Transport_Diesel_Percentage'
    },
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Electric',
        'variable': 'Car ',
        'title': '20. Transport - Electric Percentage (Car)',
        'filename': '20_Transport_Electric_Percentage'
    },
]

def main():
    print(f"\n{'='*70}")
    print("MASTER SCRIPT: Generate All Visuals (20 series)")
    print("Historical + SNBC-3 + AME-2024 on same graphs")
    print(f"{'='*70}\n")
    
    success_count = 0
    fail_count = 0
    
    for i, config in enumerate(SERIES_CONFIGS, 1):
        print(f"[{i:2d}/20] {config['title']}...", end=" ", flush=True)
        
        try:
            result = plot_all_scenarios(config)
            if result:
                print(f"OK")
                success_count += 1
            else:
                print(f"SKIP (no data)")
                fail_count += 1
        except Exception as e:
            print(f"ERROR: {str(e)[:40]}")
            fail_count += 1
    
    print(f"\n{'='*70}")
    print(f"Summary: {success_count} visuals generated, {fail_count} skipped")
    print(f"Location: {VISUALS_DIR}")
    print(f"Format: final_*.png (all scenarios combined)")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
