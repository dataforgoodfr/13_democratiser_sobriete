"""
Apply spline interpolation to ALL data series
Creates two visualizations for each series:
1. With modifications (moving average, data removal, etc.)
2. Direct spline on raw data (no modifications)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

# Define file paths
try:
    BASE_PATH = Path(__file__).parent.parent / "data"
except NameError:
    BASE_PATH = Path("stream3_visualization/Decomposition/data")

INPUT_FILE = BASE_PATH / "2025-12-15_FR scenarios data_before computation.xlsx"
OUTPUT_DIR = Path(__file__).parent.parent / "reports" / "FR" / "visuals tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_series_data(df, sector, type_val, variable, scenario='historical'):
    """Extract a specific data series"""
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

def apply_moving_average(data_dict, window=5):
    """Apply moving average to smooth data"""
    if not data_dict:
        return {}
    
    years = sorted(data_dict.keys())
    values = [data_dict[year] for year in years]
    
    series = pd.Series(values, index=years)
    smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
    
    return smoothed.to_dict()

def drop_years(data_dict, years_to_drop):
    """Remove specific years from data"""
    return {year: val for year, val in data_dict.items() if year not in years_to_drop}

def remove_data_point(data_dict, year_to_remove):
    """Remove a specific data point"""
    return {year: val for year, val in data_dict.items() if year != year_to_remove}

def apply_ratio_scaling_2011(data_dict):
    """Apply Method 1: Ratio Scaling to remove 2011-2012 break
    Scales pre-2012 data by ratio of 2012/2011 value"""
    if 2011 not in data_dict or 2012 not in data_dict:
        return data_dict.copy()
    
    ratio = data_dict[2012] / data_dict[2011]
    scaled_data = data_dict.copy()
    
    # Scale all years before 2012
    for year in scaled_data:
        if year < 2012:
            scaled_data[year] = scaled_data[year] * ratio
    
    return scaled_data

def interpolate_with_spline(data_dict, target_years, smoothing=0.5):
    """Spline interpolation to fill years with optional custom smoothing"""
    if not data_dict or len(data_dict) < 2:
        return {}
    
    available_years = np.array(sorted(data_dict.keys()))
    available_values = np.array([data_dict[year] for year in available_years])
    
    try:
        n_points = len(available_years)
        if n_points < 4:
            from scipy.interpolate import interp1d
            f = interp1d(available_years, available_values, kind='linear',
                        fill_value='extrapolate', bounds_error=False)
            spline = f
        else:
            spline = UnivariateSpline(available_years, available_values, k=min(3, n_points-1), s=smoothing)
    except:
        from scipy.interpolate import interp1d
        spline = interp1d(available_years, available_values, kind='linear',
                         fill_value='extrapolate', bounds_error=False)
    
    interpolated = {}
    for year in target_years:
        interpolated[year] = float(spline(year))
    
    return interpolated

def plot_superposed(title, raw_data, cleaned_data, interpolated_data, filename, series_name):
    """Create superposed plot with raw, cleaned, and interpolated on same graph"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Raw data
    raw_years = sorted(raw_data.keys())
    raw_values = [raw_data[y] for y in raw_years]
    
    ax.scatter(raw_years, raw_values, color='red', s=70, zorder=5, alpha=0.6, label='Raw Data', edgecolors='darkred', linewidth=0.5)
    ax.plot(raw_years, raw_values, color='red', linewidth=1.5, alpha=0.3, linestyle='-', marker='o', markersize=4)
    
    # Cleaned data
    cleaned_years = sorted(cleaned_data.keys())
    cleaned_values = [cleaned_data[y] for y in cleaned_years]
    
    ax.scatter(cleaned_years, cleaned_values, color='orange', s=70, zorder=4, alpha=0.7, label='Cleaned Data', edgecolors='darkorange', linewidth=0.5)
    ax.plot(cleaned_years, cleaned_values, color='orange', linewidth=2, alpha=0.5, linestyle='--', marker='s', markersize=5)
    
    # Interpolated data (spline)
    interp_years = sorted(interpolated_data.keys())
    interp_values = [interpolated_data[y] for y in interp_years]
    
    ax.plot(interp_years, interp_values, color='green', linewidth=3, alpha=0.8, label='Interpolated (Spline)', zorder=3)
    
    # Reference lines
    ax.axvline(x=2011.5, color='purple', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvspan(2019.5, 2021.5, alpha=0.05, color='gray')
    
    ax.set_title(series_name, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

# Define all 18 data series with their modification rules
ALL_SERIES = [
    {
        'num': 1,
        'sector': 'Buildings - Residential',
        'type': 'Final energy',
        'variable': 'Residential',
        'mod': 'moving_avg_5',
        'smoothing': 0.5,
        'filename': 'final_1_Buildings_Final_energy.png',
        'series_name': 'Series 1: Buildings Residential - Final Energy'
    },
    {
        'num': 2,
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Floor area collective',
        'mod': 'remove_2013',
        'smoothing': 0.2,
        'filename': 'final_2_Buildings_Floor_area_collective.png',
        'series_name': 'Series 2: Buildings Residential - Floor Area Collective'
    },
    {
        'num': 3,
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Floor area individual',
        'mod': 'none',
        'smoothing': 0.5,
        'filename': 'final_3_Buildings_Floor_area_individual.png',
        'series_name': 'Series 3: Buildings Residential - Floor Area Individual'
    },
    {
        'num': 4,
        'sector': 'Buildings - Residential',
        'type': 'GHG Emissions',
        'variable': 'Residential',
        'mod': 'none',
        'smoothing': 0.5,
        'filename': 'final_4_Buildings_GHG_Residential.png',
        'series_name': 'Series 4: Buildings Residential - GHG Emissions'
    },
    {
        'num': 5,
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Mean floor area',
        'mod': 'none',
        'smoothing': 0.5,
        'filename': 'final_5_Buildings_Mean_floor_area.png',
        'series_name': 'Series 5: Buildings Residential - Mean Floor Area'
    },
    {
        'num': 6,
        'sector': 'Buildings - Residential',
        'type': 'Floor area',
        'variable': 'Residential',
        'mod': 'none',
        'smoothing': 0.5,
        'filename': 'final_6_Buildings_Floor_area.png',
        'series_name': 'Series 6: Buildings Residential - Floor Area'
    },
    {
        'num': 7,
        'sector': 'Buildings - Services',
        'type': 'Final energy',
        'variable': 'Services',
        'mod': 'none',
        'smoothing': 0.5,
        'filename': 'final_7_Buildings_Services_Final_energy.png',
        'series_name': 'Series 7: Buildings Services - Final Energy'
    },
    {
        'num': 8,
        'sector': 'Buildings - Services',
        'type': 'Floor area',
        'variable': 'Floor area service',
        'mod': 'none',
        'smoothing': 0.5,
        'filename': 'final_8_Buildings_Services_Floor_area.png',
        'series_name': 'Series 8: Buildings Services - Floor Area'
    },
    {
        'num': 9,
        'sector': 'Buildings - Services',
        'type': 'GHG Emissions',
        'variable': 'Services',
        'mod': 'moving_avg_5',
        'smoothing': 0.5,
        'filename': 'final_9_Buildings_Services_GHG.png',
        'series_name': 'Series 9: Buildings Services - GHG Emissions'
    },
    {
        'num': 10,
        'sector': 'Demography',
        'type': 'Demography',
        'variable': 'Population',
        'mod': 'none',
        'smoothing': 0.5,
        'filename': 'final_10_Demography_Population.png',
        'series_name': 'Series 10: Demography - Population'
    },
    {
        'num': 11,
        'sector': 'Transport - Freight',
        'type': 'Freight transport',
        'variable': 'Inland transport',
        'mod': 'moving_avg_5',
        'smoothing': 0.5,
        'filename': 'final_11_Transport_Freight_Inland.png',
        'series_name': 'Series 11: Transport Freight - Inland Transport'
    },
    {
        'num': 12,
        'sector': 'Transport - Freight',
        'type': 'Passenger kilometers',
        'variable': 'Buses',
        'mod': 'drop_2020_2021',
        'smoothing': 0.5,
        'filename': 'final_12_Transport_Freight_Buses.png',
        'series_name': 'Series 12: Transport Freight - Buses'
    },
    {
        'num': 13,
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Consumption',
        'variable': 'Car',
        'mod': 'moving_avg_5',
        'smoothing': 0.5,
        'filename': 'final_13_Transport_Consumption_Car.png',
        'series_name': 'Series 13: Transport - Consumption Car'
    },
    {
        'num': 14,
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Consumption',
        'variable': 'Goods transport',
        'mod': 'moving_avg_5',
        'smoothing': 0.5,
        'filename': 'final_14_Transport_Consumption_Goods.png',
        'series_name': 'Series 14: Transport - Consumption Goods Transport'
    },
    {
        'num': 15,
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Consumption',
        'variable': 'Passenger transport',
        'mod': 'moving_avg_5',
        'smoothing': 0.5,
        'filename': 'final_15_Transport_Consumption_Passenger.png',
        'series_name': 'Series 15: Transport - Consumption Passenger Transport'
    },
    {
        'num': 16,
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'GHG Emissions',
        'variable': 'Passenger transport',
        'mod': 'drop_2020',
        'smoothing': 0.5,
        'filename': 'final_16_Transport_GHG_Passenger.png',
        'series_name': 'Series 16: Transport - GHG Emissions Passenger'
    },
    {
        'num': 17,
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Passenger kilometers',
        'variable': 'Car and Motorcycle',
        'mod': 'ratio_scaling_2011_drop_2020_2021',
        'smoothing': 0.5,
        'filename': 'final_17_Transport_Passenger_km_Car.png',
        'series_name': 'Series 17: Transport - Passenger Kilometers Car and Motorcycle'
    },
    {
        'num': 18,
        'sector': 'Transport - Passenger car and motorcycle',
        'type': 'Passenger kilometers',
        'variable': 'Train',
        'mod': 'drop_2020_2021',
        'smoothing': 0.5,
        'filename': 'final_18_Transport_Passenger_km_Train.png',
        'series_name': 'Series 18: Transport - Passenger Kilometers Train'
    },
    {
        'num': 19,
        'sector': 'Buildings - Services',
        'type': 'Number residential',
        'variable': 'Residential',
        'mod': 'none',
        'smoothing': 0.5,
        'filename': 'final_19_Scenario_AME2024.png',
        'series_name': 'Series 19: Scenario AME-2024 - Number Residential',
        'scenario': 'AME-2024'
    },
    {
        'num': 20,
        'sector': 'Buildings - Services',
        'type': 'Number residential',
        'variable': 'Residential',
        'mod': 'none',
        'smoothing': 0.5,
        'filename': 'final_20_Scenario_SNBC3.png',
        'series_name': 'Series 20: Scenario SNBC-3 - Number Residential',
        'scenario': 'SNBC-3'
    },
]

def process_all_series():
    """Process all series with modifications and direct spline"""
    print("="*80)
    print("PROCESSING ALL DATA SERIES WITH SPLINE INTERPOLATION")
    print("="*80)
    
    df = pd.read_excel(INPUT_FILE)
    
    for i, series_config in enumerate(ALL_SERIES, 1):
        print("\n[{}/{}] {}".format(i, len(ALL_SERIES), series_config['series_name']))
        
        # Get scenario (default to 'historical')
        scenario = series_config.get('scenario', 'historical')
        
        # Extract raw data
        raw_data = extract_series_data(df, series_config['sector'], series_config['type'], 
                                      series_config['variable'], scenario)
        
        if not raw_data:
            print("  -> No data found, skipping")
            continue
        
        print("  Raw data: {} years ({}-{})".format(
            len(raw_data), min(raw_data.keys()), max(raw_data.keys())))
        
        # === Graph 1: With Modifications ===
        cleaned_data = raw_data.copy()
        mod_name = series_config['mod']
        
        if mod_name == 'moving_avg_5':
            cleaned_data = apply_moving_average(raw_data, window=5)
            print("  Applied 5-year moving average")
        elif mod_name == 'drop_2020':
            cleaned_data = drop_years(raw_data, [2020])
            print("  Dropped 2020")
        elif mod_name == 'drop_2020_2021':
            cleaned_data = drop_years(raw_data, [2020, 2021])
            print("  Dropped 2020, 2021")
        elif mod_name == 'drop_2020_2021_2022':
            cleaned_data = drop_years(raw_data, [2020, 2021, 2022])
            print("  Dropped 2020, 2021, 2022")
        elif mod_name == 'remove_2013':
            cleaned_data = remove_data_point(raw_data, 2013)
            print("  Removed 2013 data point")
        elif mod_name == 'ratio_scaling_2011_drop_2020_2021':
            cleaned_data = apply_ratio_scaling_2011(raw_data)
            print("  Applied Method 1: Ratio Scaling (2011-2012 break)")
            cleaned_data = drop_years(cleaned_data, [2020, 2021])
            print("  Dropped 2020, 2021 (COVID)")
        elif mod_name == 'none':
            print("  No modifications applied")
        
        print("  Cleaned data: {} years ({}-{})".format(
            len(cleaned_data), min(cleaned_data.keys()), max(cleaned_data.keys())))
        
        # Interpolate cleaned data
        min_year = min(cleaned_data.keys())
        max_year = max(cleaned_data.keys())
        target_years = list(range(min_year, max_year + 1))
        
        smoothing = series_config.get('smoothing', 0.5)
        interpolated_data_modified = interpolate_with_spline(cleaned_data, target_years, smoothing=smoothing)
        print("  Interpolated (modified): {} years".format(len(interpolated_data_modified)))
        
        # Create visualization with modifications
        plot_superposed(
            series_config['series_name'],
            raw_data,
            cleaned_data,
            interpolated_data_modified,
            series_config['filename'],
            series_config['series_name']
        )
        print("  -> Saved: {}".format(series_config['filename']))

if __name__ == "__main__":
    process_all_series()
    
    print("\n" + "="*80)
    print("ALL SERIES PROCESSING COMPLETE!")
    print("Generated {} visualization files with modifications + spline".format(len(ALL_SERIES)))
    print("Saved to: {}".format(OUTPUT_DIR))
    print("="*80)
