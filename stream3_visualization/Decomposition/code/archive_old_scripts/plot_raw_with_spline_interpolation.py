"""
Plot raw scenario data with spline interpolation overlay
Shows both raw points and interpolated curve for SNBC-3 and AME-2024
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
TIMELINES_DIR = BASE_PATH / 'reports' / 'FR' / 'visuals tests' / 'timeline_scenarios'
TIMELINES_DIR.mkdir(parents=True, exist_ok=True)

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
        # Not enough points for spline, return as is
        return years, values
    
    try:
        spline = UnivariateSpline(years, values, s=smoothing, k=min(3, len(years)-1))
        year_range = np.arange(min(years), max(years) + 1)
        interpolated = spline(year_range)
        return year_range, interpolated
    except:
        return years, values

def plot_raw_with_interpolation(series_config):
    """
    Plot raw data + interpolated spline for both SNBC-3 and AME-2024
    """
    df = pd.read_excel(INPUT_FILE)
    
    scenarios = ['SNBC-3', 'AME-2024']
    colors = {'SNBC-3': '#ff7f0e', 'AME-2024': '#2ca02c'}
    markers = {'SNBC-3': 'o', 'AME-2024': 's'}
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    has_data = False
    
    for scenario in scenarios:
        # Extract raw data
        data_dict = extract_series_data(
            df,
            series_config['sector'],
            series_config['type'],
            series_config['variable'],
            scenario
        )
        
        if not data_dict:
            continue
        
        has_data = True
        years = sorted(data_dict.keys())
        values = [data_dict[y] for y in years]
        
        years_arr = np.array(years, dtype=float)
        values_arr = np.array(values, dtype=float)
        
        # Plot raw data points
        ax.scatter(years_arr, values_arr, color=colors[scenario], marker=markers[scenario],
                  s=150, alpha=0.7, zorder=5, label=f'{scenario} (raw)')
        
        # Interpolate and plot spline
        interp_years, interp_values = interpolate_data(years_arr, values_arr, smoothing=0.5)
        ax.plot(interp_years, interp_values, color=colors[scenario], linewidth=2.5, 
               alpha=0.6, label=f'{scenario} (interpolated)', linestyle='-')
    
    if not has_data:
        return None
    
    # Formatting
    ax.set_title(f"{series_config['title']} - Raw Data + Interpolation (SNBC-3 & AME-2024)", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    filename = TIMELINES_DIR / f"raw_interp_{series_config['filename']}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

# Define series that have scenario data
SERIES_CONFIGS = [
    # Buildings Residential
    {
        'sector': 'Buildings - Residential',
        'type': 'Final energy',
        'variable': 'Residential',
        'title': '1. Buildings - Residential Final Energy',
        'filename': '01_buildings_final_energy'
    },
    {
        'sector': 'Buildings - Residential',
        'type': 'GHG Emissions',
        'variable': 'Residential',
        'title': '4. Buildings - Residential GHG Emissions',
        'filename': '04_buildings_ghg_residential'
    },
    # Buildings Services
    {
        'sector': 'Buildings - Services',
        'type': 'Floor area',
        'variable': 'Floor area service',
        'title': '8. Buildings - Services Floor Area',
        'filename': '08_buildings_services_floor_area'
    },
    # Demography
    {
        'sector': 'Demography',
        'type': 'Demography',
        'variable': 'Population',
        'title': '10. Demography - Population',
        'filename': '10_demography_population'
    },
    # Transport
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Vehicle kilometers',
        'variable': 'Car',
        'title': '14. Transport - Vehicle km Car',
        'filename': '14_transport_vehicle_km_car'
    },
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Passenger kilometers',
        'variable': 'Car',
        'title': '15. Transport - Passenger km Car',
        'filename': '15_transport_passenger_km_car'
    },
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Passenger kilometers',
        'variable': 'Motorcycle',
        'title': '17. Transport - Passenger km Motorcycle',
        'filename': '17_transport_passenger_km_motorcycle'
    },
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Diesel',
        'variable': 'Car ',
        'title': '19. Transport - Diesel Percentage (Car)',
        'filename': '19_transport_diesel_percentage'
    },
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Electric',
        'variable': 'Car ',
        'title': '20. Transport - Electric Percentage (Car)',
        'filename': '20_transport_electric_percentage'
    },
]

def main():
    print(f"\n{'='*70}")
    print("Plotting Raw Data + Interpolation for Scenarios")
    print(f"{'='*70}\n")
    
    success_count = 0
    fail_count = 0
    
    for i, config in enumerate(SERIES_CONFIGS, 1):
        print(f"[{i:2d}/{len(SERIES_CONFIGS)}] {config['title']}...", end=" ", flush=True)
        
        try:
            result = plot_raw_with_interpolation(config)
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
    print(f"Summary: {success_count} plots generated")
    print(f"Location: {TIMELINES_DIR}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
