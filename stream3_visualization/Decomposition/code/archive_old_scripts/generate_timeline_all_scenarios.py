"""
Generate comprehensive timeline graphs for all 20 series and all 3 scenarios
Creates individual timeline plots showing all scenarios on the same chart
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
TIMELINES_DIR = VISUALS_DIR / 'timeline_scenarios'
TIMELINES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PATH / '2025-12-15_FR scenarios data_before computation.xlsx'

def extract_series_data(df, sector, type_val, variable, scenario):
    """Extract a specific data series for a given scenario"""
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

def plot_timeline_all_scenarios(series_config):
    """
    Create a timeline plot comparing all 3 scenarios for a series
    
    Args:
        series_config: Dict with 'sector', 'type', 'variable', 'title' keys
    """
    df = pd.read_excel(INPUT_FILE)
    
    scenarios = ['historical', 'SNBC-3', 'AME-2024']
    colors = {'historical': '#1f77b4', 'SNBC-3': '#ff7f0e', 'AME-2024': '#2ca02c'}
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    has_data = False
    
    for scenario in scenarios:
        # Extract data
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
        
        # Interpolate
        interp_years, interp_values = interpolate_data(
            np.array(years, dtype=float),
            np.array(values, dtype=float),
            smoothing=0.5
        )
        
        # Plot
        ax.plot(interp_years, interp_values, marker='o', label=scenario, 
               linewidth=2.5, color=colors[scenario], alpha=0.8, markersize=4)
    
    if not has_data:
        print(f"  ⚠ No data for {series_config['title']}")
        return None
    
    # Formatting
    ax.set_title(f"{series_config['title']} - All Scenarios", fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    filename = TIMELINES_DIR / f"timeline_{series_config['filename']}.png"
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
    
    # Series 19-20: Scenarios
    {
        'sector': 'Buildings - Services',
        'type': 'Final energy',
        'variable': 'Residential',
        'title': '19. Scenario - AME2024',
        'filename': '19_scenario_ame2024'
    },
    {
        'sector': 'Buildings - Services',
        'type': 'Final energy',
        'variable': 'Residential',
        'title': '20. Scenario - SNBC3',
        'filename': '20_scenario_snbc3'
    },
]

def main():
    print(f"\n{'='*70}")
    print("Generating Timeline Graphs for All Series & Scenarios")
    print(f"{'='*70}\n")
    
    success_count = 0
    fail_count = 0
    
    for i, config in enumerate(SERIES_CONFIGS, 1):
        print(f"[{i:2d}/20] {config['title']}...", end=" ")
        
        try:
            result = plot_timeline_all_scenarios(config)
            if result:
                print(f"✓ {result.name}")
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            fail_count += 1
    
    print(f"\n{'='*70}")
    print(f"Summary: {success_count} graphs generated, {fail_count} skipped")
    print(f"Location: {TIMELINES_DIR}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
