"""
FR Scenarios - LMDI Waterfall Decomposition Visualizations
Decomposes CO2 = (CO2/Energy) × (Energy/Activity) × (Activity/Population) × Population
for France scenarios using compiled data

Output: stream3_visualization/Decomposition/reports/FR/visuals decomposition/
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
BASE_PATH = Path(__file__).parent.parent.parent
DATA_PATH = BASE_PATH / 'data'
COMPILED_FILE = DATA_PATH / '2025-01-06_FR_scenarios_compiled.xlsx'
OUTPUT_DIR = BASE_PATH / 'reports' / 'FR' / 'visuals decomposition'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION - Color palette from Well-being dashboard
# ============================================================================
# Base colors from app.py
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'

# Scenario colors for waterfall charts
COLORS = {
    'SNBC-3': '#fdb462',      # Orange-warm from palette
    'AME-2024': '#8dd3c7',    # Cyan-turquoise from palette
}

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("FR SCENARIOS - LMDI WATERFALL DECOMPOSITION")
print("="*80)

df = pd.read_excel(COMPILED_FILE)

print(f"\nLoaded compiled data: {len(df)} rows")

# ============================================================================
# LMDI DECOMPOSITION FUNCTION
# ============================================================================
def calculate_lmdi_contribution(co2_0, co2_t, x_0, x_t):
    """
    Calculate LMDI contribution safely
    LMDI weight = (CO2_t - CO2_0) / (ln(CO2_t) - ln(CO2_0))
    Contribution = weight × ln(x_t / x_0)
    """
    if co2_0 == co2_t or x_0 == 0 or x_t == 0 or co2_0 <= 0 or co2_t <= 0:
        return 0
    
    try:
        weight = (co2_t - co2_0) / (np.log(co2_t) - np.log(co2_0))
        contribution = weight * np.log(x_t / x_0)
        return contribution
    except:
        return 0

# ============================================================================
# AGRICULTURE DECOMPOSITION (3-LEVER MODEL)
# ============================================================================
def decompose_agriculture_sector(sector_name, activity_type=None, start_year=2021, end_year=2030):
    """
    Decompose CO2 for agriculture sectors (3-lever model without energy)
    CO2 = (CO2/Activity) × (Activity/Population) × Population
    activity_type: 'Cattle breeding' or 'Culture'
    """
    print(f"\nDecomposing {sector_name} ({start_year}-{end_year})...")
    
    results = {}
    
    for scenario in ['SNBC-3', 'AME-2024']:
        # Filter data for this scenario
        scenario_data = df[df['Scenario'] == scenario]
        
        # Get data for CO2 (from main sector)
        start_data = scenario_data[
            (scenario_data['Sector'] == sector_name) &
            (scenario_data['Year'] == start_year)
        ]
        
        end_data = scenario_data[
            (scenario_data['Sector'] == sector_name) &
            (scenario_data['Year'] == end_year)
        ]
        
        if len(start_data) == 0 or len(end_data) == 0:
            print(f"  {scenario}: Missing data for {sector_name}")
            continue
        
        # Extract Activity (either Cattle breeding or Culture)
        activity_0 = start_data[
            start_data['Category'] == 'Activity'
        ]['Value'].values
        activity_t = end_data[
            end_data['Category'] == 'Activity'
        ]['Value'].values
        
        if len(activity_0) == 0 or len(activity_t) == 0:
            print(f"  {scenario}: Missing activity data")
            continue
        
        activity_0 = float(activity_0[0])
        activity_t = float(activity_t[0])
        
        # Extract GHG Emissions
        co2_0 = start_data[
            start_data['Category'] == 'GHG emissions'
        ]['Value'].values
        co2_t = end_data[
            end_data['Category'] == 'GHG emissions'
        ]['Value'].values
        
        if len(co2_0) == 0 or len(co2_t) == 0:
            print(f"  {scenario}: Missing CO2 data")
            continue
        
        co2_0 = float(co2_0[0])
        co2_t = float(co2_t[0])
        
        # Population - extract from data
        pop_data_0 = scenario_data[
            (scenario_data['Year'] == start_year) &
            (scenario_data['Category'] == 'Population')
        ]['Value'].values
        pop_data_t = scenario_data[
            (scenario_data['Year'] == end_year) &
            (scenario_data['Category'] == 'Population')
        ]['Value'].values
        
        if len(pop_data_0) > 0 and len(pop_data_t) > 0:
            pop_0 = float(pop_data_0[0])
            pop_t = float(pop_data_t[0])
        else:
            # Fallback to constant if not in data
            pop_0 = pop_t = 67.3
        
        # Calculate intensity factors (3-lever model for agriculture)
        # Lever 1: Population
        # Lever 2: Sufficiency = Activity/Population
        # Lever 3: Decarbonation = CO2/Activity
        
        decarbonation_0 = co2_0 / activity_0 if activity_0 > 0 else 0
        decarbonation_t = co2_t / activity_t if activity_t > 0 else 0
        
        sufficiency_0 = activity_0 / pop_0 if pop_0 > 0 else 0
        sufficiency_t = activity_t / pop_t if pop_t > 0 else 0
        
        # Calculate LMDI contributions (3 levers: population, sufficiency, decarbonation)
        contrib_population = calculate_lmdi_contribution(co2_0, co2_t, pop_0, pop_t)
        contrib_sufficiency = calculate_lmdi_contribution(co2_0, co2_t, sufficiency_0, sufficiency_t)
        contrib_decarbonation = calculate_lmdi_contribution(co2_0, co2_t, decarbonation_0, decarbonation_t)
        
        # Total change
        total_change = co2_t - co2_0
        
        results[scenario] = {
            'co2_0': co2_0,
            'co2_t': co2_t,
            'total_change': total_change,
            'activity_0': activity_0,
            'activity_t': activity_t,
            'decarbonation_0': decarbonation_0,
            'decarbonation_t': decarbonation_t,
            'sufficiency_0': sufficiency_0,
            'sufficiency_t': sufficiency_t,
            'contrib_population': contrib_population,
            'contrib_sufficiency': contrib_sufficiency,
            'contrib_decarbonation': contrib_decarbonation,
        }
        
        print(f"  {scenario}:")
        print(f"    CO2: {co2_0:.2f} -> {co2_t:.2f} MtCO2e (Delta {total_change:.2f})")
        print(f"    Activity: {activity_0:.2f} -> {activity_t:.2f}")
        print(f"    LMDI Contributions (3-lever model):")
        print(f"      Demographie: {contrib_population:.2f}")
        print(f"      Sobriete (Activity/Pop): {contrib_sufficiency:.2f}")
        print(f"      Decarbonation (CO2/Activity): {contrib_decarbonation:.2f}")
    
    return results

# ============================================================================
# DECOMPOSITION FOR EACH SECTOR
# ============================================================================
def decompose_sector(sector_name, vehicle_type=None, start_year=2021, end_year=2030, activity_sector=None):
    """
    Decompose CO2 for a sector between two years
    CO2 = (CO2/Energy) × (Energy/Activity) × (Activity/Population) × Population
    vehicle_type: For transport, specify 'Car' or 'Motorcycle' (None for buildings)
    activity_sector: Different sector to get Activity/Energy from (e.g., Transport - Passenger car and motorcycle)
    """
    print(f"\nDecomposing {sector_name} ({start_year}-{end_year})...")
    
    results = {}
    
    for scenario in ['SNBC-3', 'AME-2024']:
        # Filter data for this scenario
        scenario_data = df[df['Scenario'] == scenario]
        
        # Get data for CO2 (from main sector)
        start_data = scenario_data[
            (scenario_data['Sector'] == sector_name) &
            (scenario_data['Year'] == start_year)
        ]
        
        end_data = scenario_data[
            (scenario_data['Sector'] == sector_name) &
            (scenario_data['Year'] == end_year)
        ]
        
        if len(start_data) == 0 or len(end_data) == 0:
            print(f"  {scenario}: Missing data for {sector_name}")
            continue
        
        # Get Activity and Energy from activity_sector if different
        if activity_sector:
            activity_start_data = scenario_data[
                (scenario_data['Sector'] == activity_sector) &
                (scenario_data['Year'] == start_year)
            ]
            activity_end_data = scenario_data[
                (scenario_data['Sector'] == activity_sector) &
                (scenario_data['Year'] == end_year)
            ]
        else:
            activity_start_data = start_data
            activity_end_data = end_data
        
        # Extract Activity
        if vehicle_type:
            activity_0 = activity_start_data[
                (activity_start_data['Category'] == 'Activity') & 
                (activity_start_data['Type'] == vehicle_type)
            ]['Value'].values
            activity_t = activity_end_data[
                (activity_end_data['Category'] == 'Activity') & 
                (activity_end_data['Type'] == vehicle_type)
            ]['Value'].values
        else:
            activity_0 = activity_start_data[
                activity_start_data['Category'] == 'Activity'
            ]['Value'].values
            activity_t = activity_end_data[
                activity_end_data['Category'] == 'Activity'
            ]['Value'].values
        
        if len(activity_0) == 0 or len(activity_t) == 0:
            print(f"  {scenario}: Missing activity data")
            continue
        
        activity_0 = float(activity_0[0])
        activity_t = float(activity_t[0])
        
        # Extract Final Energy
        if vehicle_type:
            energy_0 = activity_start_data[
                (activity_start_data['Category'] == 'Final energy') & 
                (activity_start_data['Type'] == vehicle_type)
            ]['Value'].values
            energy_t = activity_end_data[
                (activity_end_data['Category'] == 'Final energy') & 
                (activity_end_data['Type'] == vehicle_type)
            ]['Value'].values
        else:
            energy_0 = activity_start_data[
                activity_start_data['Category'] == 'Final energy'
            ]['Value'].values
            energy_t = activity_end_data[
                activity_end_data['Category'] == 'Final energy'
            ]['Value'].values
        
        if len(energy_0) == 0 or len(energy_t) == 0:
            print(f"  {scenario}: Missing energy data")
            continue
        
        energy_0 = float(energy_0[0])
        energy_t = float(energy_t[0])
        
        # Extract GHG Emissions (from CO2 sector - main sector)
        if vehicle_type:
            co2_0 = start_data[
                (start_data['Category'] == 'GHG emissions') & 
                (start_data['Type'] == vehicle_type)
            ]['Value'].values
            co2_t = end_data[
                (end_data['Category'] == 'GHG emissions') & 
                (end_data['Type'] == vehicle_type)
            ]['Value'].values
        else:
            co2_0 = start_data[
                start_data['Category'] == 'GHG emissions'
            ]['Value'].values
            co2_t = end_data[
                end_data['Category'] == 'GHG emissions'
            ]['Value'].values
        
        if len(co2_0) == 0 or len(co2_t) == 0:
            print(f"  {scenario}: Missing CO2 data")
            continue
        
        co2_0 = float(co2_0[0])
        co2_t = float(co2_t[0])
        
        # Population - extract from data
        pop_data_0 = scenario_data[
            (scenario_data['Year'] == start_year) &
            (scenario_data['Category'] == 'Population')
        ]['Value'].values
        pop_data_t = scenario_data[
            (scenario_data['Year'] == end_year) &
            (scenario_data['Category'] == 'Population')
        ]['Value'].values
        
        if len(pop_data_0) > 0 and len(pop_data_t) > 0:
            pop_0 = float(pop_data_0[0])
            pop_t = float(pop_data_t[0])
        else:
            # Fallback to constant if not in data
            pop_0 = pop_t = 67.3
        
        # Calculate intensity factors
        efficiency_0 = energy_0 / activity_0 if activity_0 > 0 else 0
        efficiency_t = energy_t / activity_t if activity_t > 0 else 0
        
        decarbonation_0 = co2_0 / energy_0 if energy_0 > 0 else 0
        decarbonation_t = co2_t / energy_t if energy_t > 0 else 0
        
        sufficiency_0 = activity_0 / pop_0 if pop_0 > 0 else 0
        sufficiency_t = activity_t / pop_t if pop_t > 0 else 0
        
        # Calculate LMDI contributions (all 4 levers)
        contrib_population = calculate_lmdi_contribution(co2_0, co2_t, pop_0, pop_t)
        contrib_sufficiency = calculate_lmdi_contribution(co2_0, co2_t, sufficiency_0, sufficiency_t)
        contrib_efficiency = calculate_lmdi_contribution(co2_0, co2_t, efficiency_0, efficiency_t)
        contrib_decarbonation = calculate_lmdi_contribution(co2_0, co2_t, decarbonation_0, decarbonation_t)
        
        # Total change
        total_change = co2_t - co2_0
        
        results[scenario] = {
            'co2_0': co2_0,
            'co2_t': co2_t,
            'total_change': total_change,
            'activity_0': activity_0,
            'activity_t': activity_t,
            'energy_0': energy_0,
            'energy_t': energy_t,
            'efficiency_0': efficiency_0,
            'efficiency_t': efficiency_t,
            'decarbonation_0': decarbonation_0,
            'decarbonation_t': decarbonation_t,
            'sufficiency_0': sufficiency_0,
            'sufficiency_t': sufficiency_t,
            'contrib_population': contrib_population,
            'contrib_sufficiency': contrib_sufficiency,
            'contrib_efficiency': contrib_efficiency,
            'contrib_decarbonation': contrib_decarbonation,
        }
        
        print(f"  {scenario}:")
        print(f"    CO2: {co2_0:.2f} -> {co2_t:.2f} MtCO2e (Delta {total_change:.2f})")
        print(f"    Activity: {activity_0:.2f} -> {activity_t:.2f}")
        print(f"    Energy: {energy_0:.2f} -> {energy_t:.2f} Mtoe")
        print(f"    LMDI Contributions:")
        print(f"      Population: {contrib_population:.2f}")
        print(f"      Sobriété (Activity/Pop): {contrib_sufficiency:.2f}")
        print(f"      Energy Efficiency: {contrib_efficiency:.2f}")
        print(f"      Supply Decarbonation: {contrib_decarbonation:.2f}")
    
    return results

# ============================================================================
# WATERFALL VISUALIZATION
# ============================================================================
def plot_waterfall(sector_name, decomp_results, filename):
    """Create waterfall chart for decomposition with French labels and 4 levers"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    
    # First pass: calculate all y limits to ensure same scale
    all_y_max = []
    all_y_min = []
    
    for scenario in ['SNBC-3', 'AME-2024']:
        if scenario not in decomp_results:
            continue
        
        result = decomp_results[scenario]
        
        # Calculate positions for waterfall
        co2_0 = result['co2_0']
        values = [
            co2_0,
            result['contrib_population'],
            result['contrib_sufficiency'],
            result['contrib_efficiency'],
            result['contrib_decarbonation'],
            0
        ]
        
        cumulative = co2_0
        positions = [cumulative]
        for i in range(1, 5):
            cumulative += values[i]
            positions.append(cumulative)
        positions.append(result['co2_t'])
        
        # Calculate y limits for this scenario
        all_positions = positions + [result['co2_0'], result['co2_t']]
        y_max_data = max(all_positions)
        y_min_data = min(all_positions)
        y_range = y_max_data - y_min_data if y_max_data > y_min_data else y_max_data
        
        y_max = y_max_data + max(y_range * 0.12, 2)
        y_min = max(0, y_min_data - max(y_range * 0.05, 0.5))
        
        all_y_max.append(y_max)
        all_y_min.append(y_min)
    
    # Use maximum range for both plots
    global_y_max = max(all_y_max)
    global_y_min = 0  # Always start at 0 for y-axis
    
    # Second pass: plot with same scale
    for idx, scenario in enumerate(['SNBC-3', 'AME-2024']):
        if scenario not in decomp_results:
            continue
        
        ax = axes[idx]
        result = decomp_results[scenario]
        
        # Data for waterfall - 4 levers + start/end
        categories = [
            'CO2 2021',
            'Population',
            'Sobriété',
            'Efficacité\nénergétique',
            'Décarbonation',
            'CO2 2030'
        ]
        
        values = [
            result['co2_0'],
            result['contrib_population'],
            result['contrib_sufficiency'],
            result['contrib_efficiency'],
            result['contrib_decarbonation'],
            0  # Will be calculated as final
        ]
        
        # Calculate positions for waterfall
        cumulative = result['co2_0']
        positions = [cumulative]
        
        for i in range(1, 5):  # For each lever
            cumulative += values[i]
            positions.append(cumulative)
        
        positions.append(result['co2_t'])
        
        # Create waterfall with colors from palette
        # Green (#b3de69) for negative effects (reductions), Red (#fb8072) for positive (increases)
        colors_negative = '#b3de69'  # Green from palette
        colors_positive = '#fb8072'  # Red-pink from palette
        colors_waterfall = [COLORS[scenario], 
                           colors_negative if values[1] < 0 else colors_positive,
                           colors_negative if values[2] < 0 else colors_positive,
                           colors_negative if values[3] < 0 else colors_positive,
                           colors_negative if values[4] < 0 else colors_positive,
                           COLORS[scenario]]
        
        # Plot bars
        x_pos = range(len(categories))
        
        # Starting point
        ax.bar(0, values[0], color=colors_waterfall[0], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Levers (floating)
        for i in range(1, 5):
            bottom = positions[i-1] if values[i] > 0 else positions[i]
            height = abs(values[i])
            ax.bar(i, height, bottom=min(positions[i-1], positions[i]), 
                   color=colors_waterfall[i], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Ending point
        ax.bar(5, values[5] if values[5] != 0 else result['co2_t'], 
               color=colors_waterfall[5], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Connecting lines
        for i in range(5):
            ax.plot([i+0.4, i+0.6], [positions[i], positions[i]], 'k--', alpha=0.5)
        
        # Labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax.set_ylabel('CO2 (MtCO2e)', fontsize=12, fontweight='bold')
        ax.set_title(f'{scenario}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        # Add CO2 values ON TOP of the CO2 bars (2021 and 2030)
        ax.text(0, result['co2_0'] + 0.5, f'{result["co2_0"]:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color='black', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=0.3))
        ax.text(5, result['co2_t'] + 0.5, f'{result["co2_t"]:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color='black', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=0.3))
        
        # Add value labels on top of bars (only for levers, not CO2)
        for i in range(1, 5):  # Only for levers
            val = values[i]
            y_top = max(positions[i-1], positions[i])
            ax.text(i, y_top + 0.5, f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Apply global y limits (same for both plots)
        ax.set_ylim(global_y_min, global_y_max)
    
    # French title
    plt.suptitle(f'{sector_name} - Décomposition des émissions de CO2 (LMDI)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file.name}")
    plt.close(fig)

# ============================================================================
# WATERFALL VISUALIZATION - AGRICULTURE (3 LEVERS)
# ============================================================================
def plot_waterfall_agriculture(sector_name, decomp_results, filename):
    """Create waterfall chart for agriculture decomposition with French labels and 3 levers"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    
    # First pass: calculate all y limits to ensure same scale
    all_y_max = []
    all_y_min = []
    
    for scenario in ['SNBC-3', 'AME-2024']:
        if scenario not in decomp_results:
            continue
        
        result = decomp_results[scenario]
        
        # Calculate positions for waterfall (3 levers)
        co2_0 = result['co2_0']
        values = [
            co2_0,
            result['contrib_population'],
            result['contrib_sufficiency'],
            result['contrib_decarbonation'],
            0
        ]
        
        cumulative = co2_0
        positions = [cumulative]
        for i in range(1, 4):  # 3 levers
            cumulative += values[i]
            positions.append(cumulative)
        positions.append(result['co2_t'])
        
        # Calculate y limits for this scenario
        all_positions = positions + [result['co2_0'], result['co2_t']]
        y_max_data = max(all_positions)
        y_min_data = min(all_positions)
        y_range = y_max_data - y_min_data if y_max_data > y_min_data else y_max_data
        
        y_max = y_max_data + max(y_range * 0.12, 2)
        y_min = max(0, y_min_data - max(y_range * 0.05, 0.5))
        
        all_y_max.append(y_max)
        all_y_min.append(y_min)
    
    # Use maximum range for both plots
    global_y_max = max(all_y_max)
    global_y_min = 0  # Always start at 0 for y-axis
    
    # Second pass: plot with same scale
    for idx, scenario in enumerate(['SNBC-3', 'AME-2024']):
        if scenario not in decomp_results:
            continue
        
        ax = axes[idx]
        result = decomp_results[scenario]
        
        # Data for waterfall - 3 levers + start/end
        categories = [
            'CO2 2021',
            'Démographie',
            'Sobriété',
            'Efficacité/Décarbonation',
            'CO2 2030'
        ]
        
        values = [
            result['co2_0'],
            result['contrib_population'],
            result['contrib_sufficiency'],
            result['contrib_decarbonation'],
            0  # Will be calculated as final
        ]
        
        # Calculate positions for waterfall
        cumulative = result['co2_0']
        positions = [cumulative]
        
        for i in range(1, 4):  # For each of 3 levers
            cumulative += values[i]
            positions.append(cumulative)
        
        positions.append(result['co2_t'])
        
        # Create waterfall with colors from palette
        # Green (#b3de69) for negative effects (reductions), Red (#fb8072) for positive (increases)
        colors_negative = '#b3de69'  # Green from palette
        colors_positive = '#fb8072'  # Red-pink from palette
        colors_waterfall = [COLORS[scenario], 
                           colors_negative if values[1] < 0 else colors_positive,
                           colors_negative if values[2] < 0 else colors_positive,
                           colors_negative if values[3] < 0 else colors_positive,
                           COLORS[scenario]]
        
        # Plot bars
        x_pos = range(len(categories))
        
        # Starting point
        ax.bar(0, values[0], color=colors_waterfall[0], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Levers (floating)
        for i in range(1, 4):  # 3 levers instead of 4
            bottom = positions[i-1] if values[i] > 0 else positions[i]
            height = abs(values[i])
            ax.bar(i, height, bottom=min(positions[i-1], positions[i]), 
                   color=colors_waterfall[i], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Ending point
        ax.bar(4, values[4] if values[4] != 0 else result['co2_t'], 
               color=colors_waterfall[4], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Connecting lines
        for i in range(4):  # 3 levers = 4 positions
            ax.plot([i+0.4, i+0.6], [positions[i], positions[i]], 'k--', alpha=0.5)
        
        # Labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax.set_ylabel('CO2 (MtCO2e)', fontsize=12, fontweight='bold')
        ax.set_title(f'{scenario}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        # Add CO2 values ON TOP of the CO2 bars (2021 and 2030)
        ax.text(0, result['co2_0'] + 0.5, f'{result["co2_0"]:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color='black', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=0.3))
        ax.text(4, result['co2_t'] + 0.5, f'{result["co2_t"]:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color='black', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=0.3))
        
        # Add value labels on top of bars (only for levers, not CO2)
        for i in range(1, 4):  # Only for 3 levers
            val = values[i]
            y_top = max(positions[i-1], positions[i])
            ax.text(i, y_top + 0.5, f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Apply global y limits (same for both plots)
        ax.set_ylim(global_y_min, global_y_max)
    
    # French title
    plt.suptitle(f'{sector_name} - Décomposition des émissions de CO2 (LMDI - 3 leviers)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file.name}")
    plt.close(fig)

# ============================================================================
# WATERFALL DIFF - SNBC-3 vs AME-2024
# ============================================================================
def plot_waterfall_diff(sector_name, decomp_results, filename):
    """Create waterfall chart showing difference between SNBC-3 and AME-2024 by LMDI contributions"""
    
    if 'SNBC-3' not in decomp_results or 'AME-2024' not in decomp_results:
        print(f"  Missing data for difference plot: {sector_name}")
        return
    
    snbc = decomp_results['SNBC-3']
    ame = decomp_results['AME-2024']
    
    # Calculate differences in 2021 (co2_0) and 2030 (co2_t)
    diff_2021 = snbc['co2_0'] - ame['co2_0']
    diff_2030 = snbc['co2_t'] - ame['co2_t']
    
    # Calculate differences in contributions (SNBC - AME)
    diff_population = snbc['contrib_population'] - ame['contrib_population']
    diff_sufficiency = snbc['contrib_sufficiency'] - ame['contrib_sufficiency']
    diff_efficiency = snbc['contrib_efficiency'] - ame['contrib_efficiency']
    diff_decarbonation = snbc['contrib_decarbonation'] - ame['contrib_decarbonation']
    
    # Data for waterfall (only levers, from difference in 2021 to difference in 2030)
    categories = ['Démographie\nΔ', 'Sobriété\nΔ', 'Efficacité\nΔ', 'Décarbonation\nΔ', 'Δ CO2\n2030']
    values = [
        diff_population,
        diff_sufficiency,
        diff_efficiency,
        diff_decarbonation,
        0
    ]
    
    # Calculate cumulative positions for waterfall
    cumulative = diff_2021
    positions = []
    for i in range(0, 4):  # 4 levers
        cumulative += values[i]
        positions.append(cumulative)
    
    # Colors: negative bars = #b3de69, positive bars = #fb8072, final 2030 = #80b1d3
    colors = [
        '#b3de69' if diff_population < 0 else '#fb8072',  # Démographie
        '#b3de69' if diff_sufficiency < 0 else '#fb8072',  # Sobriété
        '#b3de69' if diff_efficiency < 0 else '#fb8072',  # Efficacité
        '#b3de69' if diff_decarbonation < 0 else '#fb8072',  # Décarbonation
        '#80b1d3'  # CO2 2030 final
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(13, 8))
    
    # Plot bars
    x_pos = np.arange(len(categories))
    
    # Difference levers - each starts where the previous one ended
    for i in range(0, 4):  # 4 levers
        # Each bar starts at positions[i-1] (or diff_2021 for first) and has height values[i]
        start_pos = diff_2021 if i == 0 else positions[i-1]
        ax.bar(i, values[i], bottom=start_pos, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
        
        # Value labels on levers - positioned at the middle of each bar
        y_pos = start_pos + values[i] / 2
        ax.text(i, y_pos, f'{values[i]:.2f}', ha='center', va='center', fontweight='bold', fontsize=10, color='black')
    
    # End point (Difference in 2030) - from 0 to diff_2030
    ax.bar(4, diff_2030, bottom=0, color=colors[4], alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
    ax.text(4, diff_2030 + 0.5, f'{diff_2030:.2f}', ha='center', va='bottom', 
            fontweight='bold', fontsize=11, color='black', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=0.3))
    
    
    # Grid and labels
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.set_ylabel('Différence CO2: SNBC-3 - AME-2024 (MtCO2e)', fontsize=12, fontweight='bold')
    ax.set_title(f'{sector_name}\nDifférence SNBC-3 vs AME-2024 par contribution LMDI (2021 vs 2030)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    
    # Y-axis range with adaptive padding
    all_values = positions + [diff_2030]
    y_min_data = min(all_values)
    y_max_data = max(all_values)
    y_range = y_max_data - y_min_data
    
    # Calculate adaptive padding based on data range
    padding = max(y_range * 0.15, 0.3)  # 15% of range or 0.3, whichever is larger
    
    y_min = y_min_data - padding
    y_max = y_max_data + padding
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file.name}")
    plt.close(fig)

# ============================================================================
# WATERFALL DIFF - AGRICULTURE (3 LEVERS)
# ============================================================================
def plot_waterfall_diff_agriculture(sector_name, decomp_results, filename):
    """Create waterfall chart showing difference between SNBC-3 and AME-2024 for agriculture (3 levers)"""
    
    if 'SNBC-3' not in decomp_results or 'AME-2024' not in decomp_results:
        print(f"  Missing data for difference plot: {sector_name}")
        return
    
    snbc = decomp_results['SNBC-3']
    ame = decomp_results['AME-2024']
    
    # Calculate differences in 2021 (co2_0) and 2030 (co2_t)
    diff_2021 = snbc['co2_0'] - ame['co2_0']
    diff_2030 = snbc['co2_t'] - ame['co2_t']
    
    # Calculate differences in contributions (SNBC - AME) - 3 levers only
    diff_population = snbc['contrib_population'] - ame['contrib_population']
    diff_sufficiency = snbc['contrib_sufficiency'] - ame['contrib_sufficiency']
    diff_decarbonation = snbc['contrib_decarbonation'] - ame['contrib_decarbonation']
    
    # Data for waterfall (only levers, from difference in 2021 to difference in 2030)
    categories = ['Démographie\nΔ', 'Sobriété\nΔ', 'Efficacité/Décarbonation\nΔ', 'Δ CO2\n2030']
    values = [
        diff_population,
        diff_sufficiency,
        diff_decarbonation,
        0
    ]
    
    # Calculate cumulative positions for waterfall
    cumulative = diff_2021
    positions = []
    for i in range(0, 3):  # 3 levers
        cumulative += values[i]
        positions.append(cumulative)
    
    # Colors: negative bars = #b3de69, positive bars = #fb8072, final 2030 = #80b1d3
    colors = [
        '#b3de69' if diff_population < 0 else '#fb8072',  # Démographie
        '#b3de69' if diff_sufficiency < 0 else '#fb8072',  # Sobriété
        '#b3de69' if diff_decarbonation < 0 else '#fb8072',  # Décarbonation
        '#80b1d3'  # CO2 2030 final
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    x_pos = np.arange(len(categories))
    
    # Difference levers - each starts where the previous one ended
    for i in range(0, 3):  # 3 levers
        # Each bar starts at positions[i-1] (or diff_2021 for first) and has height values[i]
        start_pos = diff_2021 if i == 0 else positions[i-1]
        ax.bar(i, values[i], bottom=start_pos, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
        
        # Value labels on levers - positioned at the middle of each bar
        y_pos = start_pos + values[i] / 2
        ax.text(i, y_pos, f'{values[i]:.2f}', ha='center', va='center', fontweight='bold', fontsize=10, color='black')
    
    # End point (Difference in 2030) - from 0 to diff_2030
    ax.bar(3, diff_2030, bottom=0, color=colors[3], alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
    ax.text(3, diff_2030 + 0.5, f'{diff_2030:.2f}', ha='center', va='bottom', 
            fontweight='bold', fontsize=11, color='black', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=0.3))
    
    
    # Grid and labels
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.set_ylabel('Différence CO2: SNBC-3 - AME-2024 (MtCO2e)', fontsize=12, fontweight='bold')
    ax.set_title(f'{sector_name}\nDifférence SNBC-3 vs AME-2024 par contribution LMDI (2021 vs 2030)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    
    # Y-axis range with adaptive padding
    all_values = positions + [diff_2021, diff_2030]
    y_min_data = min(all_values)
    y_max_data = max(all_values)
    y_range = y_max_data - y_min_data
    
    # Calculate adaptive padding based on data range
    padding = max(y_range * 0.15, 0.3)  # 15% of range or 0.3, whichever is larger
    
    y_min = y_min_data - padding
    y_max = y_max_data + padding
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file.name}")
    plt.close(fig)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# 1. CAR DECOMPOSITION
print("\n" + "="*80)
print("1. CAR DECOMPOSITION")
print("="*80)
decomp_car = decompose_sector(
    'Transport - Car', 
    vehicle_type='Car',
    activity_sector='Transport - Passenger car and motorcycle'
)
plot_waterfall('Transport - Voiture', decomp_car, 'FR_LMDI_Car.png')
plot_waterfall_diff('Transport - Voiture', decomp_car, 'FR_LMDI_Car_Diff.png')

# 2. CAR + MOTORCYCLE DECOMPOSITION (combined)
print("\n" + "="*80)
print("2. CAR + MOTORCYCLE DECOMPOSITION (Combined)")
print("="*80)

combined_results = {}
for scenario in ['SNBC-3', 'AME-2024']:
    scenario_data = df[df['Scenario'] == scenario]
    
    # Get Activity and Energy from Transport - Passenger car and motorcycle
    activity_data_start = scenario_data[
        (scenario_data['Sector'] == 'Transport - Passenger car and motorcycle') &
        (scenario_data['Year'] == 2021)
    ]
    activity_data_end = scenario_data[
        (scenario_data['Sector'] == 'Transport - Passenger car and motorcycle') &
        (scenario_data['Year'] == 2030)
    ]
    
    # Get CO2 from Transport - Car and Transport - Motorcycle
    car_data_start = scenario_data[
        (scenario_data['Sector'] == 'Transport - Car') &
        (scenario_data['Year'] == 2021)
    ]
    car_data_end = scenario_data[
        (scenario_data['Sector'] == 'Transport - Car') &
        (scenario_data['Year'] == 2030)
    ]
    
    moto_data_start = scenario_data[
        (scenario_data['Sector'] == 'Transport - Motorcycle') &
        (scenario_data['Year'] == 2021)
    ]
    moto_data_end = scenario_data[
        (scenario_data['Sector'] == 'Transport - Motorcycle') &
        (scenario_data['Year'] == 2030)
    ]
    
    if len(activity_data_start) == 0 or len(car_data_start) == 0 or len(moto_data_start) == 0:
        print(f"  {scenario}: Missing data")
        continue
    
    # Sum car and motorcycle activity from Transport - Passenger car and motorcycle
    car_activity_0 = activity_data_start[
        (activity_data_start['Category'] == 'Activity') & 
        (activity_data_start['Type'] == 'Car')
    ]['Value'].sum()
    car_activity_t = activity_data_end[
        (activity_data_end['Category'] == 'Activity') & 
        (activity_data_end['Type'] == 'Car')
    ]['Value'].sum()
    
    moto_activity_0 = activity_data_start[
        (activity_data_start['Category'] == 'Activity') & 
        (activity_data_start['Type'] == 'Motorcycle')
    ]['Value'].sum()
    moto_activity_t = activity_data_end[
        (activity_data_end['Category'] == 'Activity') & 
        (activity_data_end['Type'] == 'Motorcycle')
    ]['Value'].sum()
    
    activity_0 = car_activity_0 + moto_activity_0
    activity_t = car_activity_t + moto_activity_t
    
    # Energy from Transport - Passenger car and motorcycle
    car_energy_0 = activity_data_start[
        (activity_data_start['Category'] == 'Final energy') & 
        (activity_data_start['Type'] == 'Car')
    ]['Value'].sum()
    car_energy_t = activity_data_end[
        (activity_data_end['Category'] == 'Final energy') & 
        (activity_data_end['Type'] == 'Car')
    ]['Value'].sum()
    
    moto_energy_0 = activity_data_start[
        (activity_data_start['Category'] == 'Final energy') & 
        (activity_data_start['Type'] == 'Motorcycle')
    ]['Value'].sum()
    moto_energy_t = activity_data_end[
        (activity_data_end['Category'] == 'Final energy') & 
        (activity_data_end['Type'] == 'Motorcycle')
    ]['Value'].sum()
    
    energy_0 = car_energy_0 + moto_energy_0
    energy_t = car_energy_t + moto_energy_t
    
    # CO2 from Transport - Car and Transport - Motorcycle
    car_co2_0 = car_data_start[
        car_data_start['Category'] == 'GHG emissions'
    ]['Value'].sum()
    car_co2_t = car_data_end[
        car_data_end['Category'] == 'GHG emissions'
    ]['Value'].sum()
    
    moto_co2_0 = moto_data_start[
        moto_data_start['Category'] == 'GHG emissions'
    ]['Value'].sum()
    moto_co2_t = moto_data_end[
        moto_data_end['Category'] == 'GHG emissions'
    ]['Value'].sum()
    
    co2_0 = car_co2_0 + moto_co2_0
    co2_t = car_co2_t + moto_co2_t
    
    # Skip if no data
    if co2_0 == 0 or activity_0 == 0 or energy_0 == 0:
        print(f"  {scenario}: Missing values (CO2={co2_0:.2f}, Activity={activity_0:.2f}, Energy={energy_0:.2f})")
        continue
    
    # Population - extract from data
    pop_data_0 = scenario_data[
        (scenario_data['Year'] == 2021) &
        (scenario_data['Category'] == 'Population')
    ]['Value'].values
    pop_data_t = scenario_data[
        (scenario_data['Year'] == 2030) &
        (scenario_data['Category'] == 'Population')
    ]['Value'].values
    
    if len(pop_data_0) > 0 and len(pop_data_t) > 0:
        pop_0 = float(pop_data_0[0])
        pop_t = float(pop_data_t[0])
    else:
        pop_0 = pop_t = 67.3
    
    efficiency_0 = energy_0 / activity_0
    efficiency_t = energy_t / activity_t
    
    decarbonation_0 = co2_0 / energy_0
    decarbonation_t = co2_t / energy_t
    
    sufficiency_0 = activity_0 / pop_0
    sufficiency_t = activity_t / pop_t
    
    contrib_population = calculate_lmdi_contribution(co2_0, co2_t, pop_0, pop_t)
    contrib_sufficiency = calculate_lmdi_contribution(co2_0, co2_t, sufficiency_0, sufficiency_t)
    contrib_efficiency = calculate_lmdi_contribution(co2_0, co2_t, efficiency_0, efficiency_t)
    contrib_decarbonation = calculate_lmdi_contribution(co2_0, co2_t, decarbonation_0, decarbonation_t)
    
    combined_results[scenario] = {
        'co2_0': co2_0,
        'co2_t': co2_t,
        'total_change': co2_t - co2_0,
        'contrib_population': contrib_population,
        'contrib_sufficiency': contrib_sufficiency,
        'contrib_efficiency': contrib_efficiency,
        'contrib_decarbonation': contrib_decarbonation,
    }
    
    print(f"  {scenario}: CO2 {co2_0:.2f} -> {co2_t:.2f} MtCO2e (Delta {co2_t - co2_0:.2f})")
    print(f"    Activity: {activity_0:.2f} -> {activity_t:.2f}")
    print(f"    Energy: {energy_0:.2f} -> {energy_t:.2f} Mtoe")
    print(f"    LMDI Contributions:")
    print(f"      Population: {contrib_population:.2f}")
    print(f"      Sobriété (Activity/Pop): {contrib_sufficiency:.2f}")
    print(f"      Energy Efficiency: {contrib_efficiency:.2f}")
    print(f"      Supply Decarbonation: {contrib_decarbonation:.2f}")

plot_waterfall('Transport - Voiture + 2RM', combined_results, 'FR_LMDI_Car_Motorcycle.png')
plot_waterfall_diff('Transport - Voiture + 2RM', combined_results, 'FR_LMDI_Car_Motorcycle_Diff.png')

# 3. RESIDENTIAL DECOMPOSITION
print("\n" + "="*80)
print("3. RESIDENTIAL DECOMPOSITION")
print("="*80)
decomp_res = decompose_sector('Buildings - Residential')
plot_waterfall('Bâtiment - Résidentiel', decomp_res, 'FR_LMDI_Residential.png')
plot_waterfall_diff('Bâtiment - Résidentiel', decomp_res, 'FR_LMDI_Residential_Diff.png')

# 4. SERVICES DECOMPOSITION
print("\n" + "="*80)
print("4. SERVICES DECOMPOSITION")
print("="*80)
decomp_srv = decompose_sector('Buildings - Services')
plot_waterfall('Bâtiment - Tertiaire', decomp_srv, 'FR_LMDI_Services.png')
plot_waterfall_diff('Bâtiment - Tertiaire', decomp_srv, 'FR_LMDI_Services_Diff.png')

# 5. AGRICULTURE - CATTLE BREEDING DECOMPOSITION
print("\n" + "="*80)
print("5. AGRICULTURE - CATTLE BREEDING DECOMPOSITION")
print("="*80)
decomp_cattle = decompose_agriculture_sector('Agriculture - Cattle breeding', activity_type='Cattle breeding')
plot_waterfall_agriculture('Agriculture - Élevage bovin', decomp_cattle, 'FR_LMDI_Agriculture_Cattle.png')
plot_waterfall_diff_agriculture('Agriculture - Élevage bovin', decomp_cattle, 'FR_LMDI_Agriculture_Cattle_Diff.png')

# 6. AGRICULTURE - CULTURE DECOMPOSITION
print("\n" + "="*80)
print("6. AGRICULTURE - CULTURE DECOMPOSITION")
print("="*80)
decomp_culture = decompose_agriculture_sector('Agriculture - Culture', activity_type='Culture')
plot_waterfall_agriculture('Agriculture - Culture', decomp_culture, 'FR_LMDI_Agriculture_Culture.png')
plot_waterfall_diff_agriculture('Agriculture - Culture', decomp_culture, 'FR_LMDI_Agriculture_Culture_Diff.png')

print("\n" + "="*80)
print("DECOMPOSITION COMPLETE")
print("="*80)
print(f"\nVisualizations saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print(f"  - FR_LMDI_Car.png")
print(f"  - FR_LMDI_Car_Diff.png")
print(f"  - FR_LMDI_Car_Motorcycle.png")
print(f"  - FR_LMDI_Car_Motorcycle_Diff.png")
print(f"  - FR_LMDI_Residential.png")
print(f"  - FR_LMDI_Residential_Diff.png")
print(f"  - FR_LMDI_Services.png")
print(f"  - FR_LMDI_Services_Diff.png")
print(f"  - FR_LMDI_Agriculture_Cattle.png")
print(f"  - FR_LMDI_Agriculture_Cattle_Diff.png")
print(f"  - FR_LMDI_Agriculture_Culture.png")
print(f"  - FR_LMDI_Agriculture_Culture_Diff.png")
