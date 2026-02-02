import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_PATH = Path('stream3_visualization/Decomposition')
TIMELINES_DIR = BASE_PATH / 'reports' / 'FR' / 'visuals tests' / 'timeline_scenarios'
TIMELINES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = BASE_PATH / 'data' / '2025-12-15_FR scenarios data_before computation.xlsx'

print("Loading data...")
df = pd.read_excel(INPUT_FILE)

print("Creating plot...")
fig, ax = plt.subplots(figsize=(14, 6))

has_data = False

for scenario in ['SNBC-3', 'AME-2024']:
    # Extract data for demography
    subset = df[
        (df['Sector'] == 'Demography') &
        (df['Type'] == 'Demography') &
        (df['Variable'] == 'Population') &
        (df['Scenario'] == scenario)
    ]
    
    if subset.empty:
        print(f"  {scenario}: NO DATA")
        continue
    
    has_data = True
    data_by_year = subset.groupby('Year')['Volume'].sum().to_dict()
    years = sorted(data_by_year.keys())
    values = [data_by_year[y] for y in years]
    
    print(f"  {scenario}: {len(years)} years = {years}")
    print(f"    Values: {values}")
    
    colors = {'SNBC-3': '#ff7f0e', 'AME-2024': '#2ca02c'}
    markers = {'SNBC-3': 'o', 'AME-2024': 's'}
    
    ax.plot(years, values, marker=markers[scenario], label=scenario, 
           linewidth=2.5, color=colors[scenario], alpha=0.8, markersize=7)
    ax.scatter(years, values, color=colors[scenario], s=120, alpha=0.7, zorder=5)

if not has_data:
    print("NO DATA FOUND FOR ANY SCENARIO!")
else:
    print("\nFormatting plot...")
    ax.set_title("10. Demography - Population - Raw Data (SNBC-3 & AME-2024)", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_file = TIMELINES_DIR / 'raw_10_demography_population.png'
    print(f"Saving to: {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SUCCESS! File saved.")
