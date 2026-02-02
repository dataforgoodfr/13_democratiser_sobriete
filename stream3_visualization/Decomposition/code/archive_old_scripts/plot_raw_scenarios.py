"""
Plot raw data for SNBC-3 and AME-2024 scenarios (no modifications)
Follows same process as historical data treatment
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
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

def plot_raw_scenarios(series_config):
    """
    Plot raw data for both SNBC-3 and AME-2024 scenarios
    """
    df = pd.read_excel(INPUT_FILE)
    
    scenarios = ['SNBC-3', 'AME-2024']
    colors = {'SNBC-3': '#ff7f0e', 'AME-2024': '#2ca02c'}
    linestyles = {'SNBC-3': '-', 'AME-2024': '-'}  # both solid lines
    markers = {'SNBC-3': 'o', 'AME-2024': 's'}  # circles for SNBC-3, squares for AME-2024
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    has_data = False
    extracted_data = {}
    
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
        
        extracted_data[scenario] = data_dict
        has_data = True
    
    # Plot all extracted data
    for scenario in scenarios:
        if scenario not in extracted_data:
            continue
            
        data_dict = extracted_data[scenario]
        years = sorted(data_dict.keys())
        values = [data_dict[y] for y in years]
        
        # Plot raw data only (no interpolation)
        ax.plot(years, values, marker=markers[scenario], label=scenario, 
               linewidth=2.5, color=colors[scenario], linestyle=linestyles[scenario],
               alpha=0.8, markersize=7)
        ax.scatter(years, values, color=colors[scenario], s=120, alpha=0.7, zorder=5)
    
    if not has_data:
        return None
    
    # Formatting
    ax.set_title(f"{series_config['title']} - Raw Data (SNBC-3 & AME-2024)", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    filename = TIMELINES_DIR / f"raw_{series_config['filename']}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

# Define all 20 series
SERIES_CONFIGS = [
    # Series 1-6: Buildings
    {
        'sector': 'Buildings - Residential',
        'type': 'Final energy',
        'variable': 'Residential',
        'title': '1. Buildings - Residential Final Energy',
        'filename': '01_buildings_final_energy'
    },
    {
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Floor area collective',
        'title': '2. Buildings - Floor Area Collective',
        'filename': '02_buildings_floor_area_collective'
    },
    {
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Floor area individual',
        'title': '3. Buildings - Floor Area Individual',
        'filename': '03_buildings_floor_area_individual'
    },
    {
        'sector': 'Buildings - Residential',
        'type': 'GHG Emissions',
        'variable': 'Residential',
        'title': '4. Buildings - Residential GHG Emissions',
        'filename': '04_buildings_ghg_residential'
    },
    {
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Mean floor area',
        'title': '5. Buildings - Mean Floor Area',
        'filename': '05_buildings_mean_floor_area'
    },
    {
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Residential',
        'title': '6. Buildings - Floor Area',
        'filename': '06_buildings_floor_area'
    },
    
    # Series 7-9: Buildings Services
    {
        'sector': 'Buildings - Services',
        'type': 'Final energy',
        'variable': 'Services',
        'title': '7. Buildings - Services Final Energy',
        'filename': '07_buildings_services_final_energy'
    },
    {
        'sector': 'Buildings - Services',
        'type': 'Floor area',
        'variable': 'Floor area service',
        'title': '8. Buildings - Services Floor Area',
        'filename': '08_buildings_services_floor_area'
    },
    {
        'sector': 'Buildings - Services',
        'type': 'GHG Emissions',
        'variable': 'Services',
        'title': '9. Buildings - Services GHG Emissions',
        'filename': '09_buildings_services_ghg'
    },
    
    # Series 10: Demography
    {
        'sector': 'Demography',
        'type': 'Demography',
        'variable': 'Population',
        'title': '10. Demography - Population',
        'filename': '10_demography_population'
    },
    
    # Series 11-12: Transport Freight
    {
        'sector': 'Transport - Freight',
        'type': 'Freight transport',
        'variable': 'Inland transport',
        'title': '11. Transport - Freight Inland',
        'filename': '11_transport_freight_inland'
    },
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Passenger kilometers',
        'variable': 'Buses',
        'title': '12. Transport - Buses',
        'filename': '12_transport_buses'
    },
    
    # Series 13-18: Transport
    {
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Consumption',
        'variable': 'Car',
        'title': '13. Transport - Consumption Car',
        'filename': '13_transport_consumption_car'
    },
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
        'type': 'GHG Emissions',
        'variable': 'Passenger',
        'title': '16. Transport - GHG Emissions Passenger',
        'filename': '16_transport_ghg_passenger'
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
        'type': 'Essence',
        'variable': 'Car',
        'title': '18. Transport - Essence (Gasoline) %',
        'filename': '18_transport_essence_percentage'
    },
    
    # Series 19-20: Transport fuel types for cars
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
    print("Plotting Raw Data: SNBC-3 & AME-2024 Scenarios")
    print(f"{'='*70}\n")
    
    success_count = 0
    fail_count = 0
    
    for i, config in enumerate(SERIES_CONFIGS, 1):
        print(f"[{i:2d}/20] {config['title']}...", end=" ", flush=True)
        
        try:
            result = plot_raw_scenarios(config)
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
    print(f"Summary: {success_count} raw data plots generated")
    print(f"Location: {TIMELINES_DIR}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
