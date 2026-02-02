"""
Visualisations du Budget Carbone Français
Génère les visualisations pour:
1. Top 15 émissions cumulées (Territorial vs Consommation)
2. Comparaison des budgets carbone restants (SNBC-3 vs scénarios d'allocation différents)
3. Comparaison des années de neutralité carbone

Sortie: fichiers PNG dans le répertoire visuals budget
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
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data-SNBC-budget'
OUTPUT_DIR = BASE_PATH / 'visuals budget'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION - Color palette from Well-being dashboard
# ============================================================================
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'

# Scenario colors - SNBC-3 always in orange, others in palette
COLORS = {
    'SNBC-3': '#fdb462',      # Orange-warm
    'Population': '#8dd3c7',  # Cyan-turquoise
    'Responsibility': '#bebada', # Mauve
    'Capability': '#b3de69',  # Green
}

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("VISUALISATIONS DU BUDGET CARBONE FRANÇAIS")
print("="*80)

# Load cumulative emissions
top15_df = pd.read_csv(DATA_PATH / 'top15_cumulative_emissions.csv')

# Load carbon budget analysis
budget_df = pd.read_csv(DATA_PATH / 'france_carbon_budget_analysis.csv')

print(f"\nChargement des données:")
print(f"  top15_cumulative_emissions.csv: {len(top15_df)} lignes")
print(f"  france_carbon_budget_analysis.csv: {len(budget_df)} lignes")

# ============================================================================
# VISUALIZATION 1: TOP 15 CUMULATIVE EMISSIONS BY SCOPE
# ============================================================================
print("\n" + "="*80)
print("1. TOP 15 ÉMISSIONS CUMULÉES (Territorial vs Consommation)")
print("="*80)

# Separate by Emissions_Type
territorial_df = top15_df[top15_df['Emissions_Type'] == 'Territorial'].copy()
consumption_df = top15_df[top15_df['Emissions_Type'] == 'Consumption'].copy()

# Find which has higher overall share for France
france_territorial = territorial_df[territorial_df['ISO3'] == 'FRA']
france_consumption = consumption_df[consumption_df['ISO3'] == 'FRA']

if not france_territorial.empty and not france_consumption.empty:
    terr_share = france_territorial['Share_of_overall_%'].values[0]
    cons_share = france_consumption['Share_of_overall_%'].values[0]
    print(f"\nPart de la France Territorial: {terr_share}%")
    print(f"Part de la France Consommation: {cons_share}%")
    
    # Decide order (top has bigger share)
    if terr_share >= cons_share:
        first_scope = 'Territorial'
        second_scope = 'Consommation'
        first_df = territorial_df
        second_df = consumption_df
    else:
        first_scope = 'Consommation'
        second_scope = 'Territorial'
        first_df = consumption_df
        second_df = territorial_df
else:
    print("Attention: France non trouvée dans les deux scopes")
    first_scope = 'Territorial'
    second_scope = 'Consommation'
    first_df = territorial_df
    second_df = consumption_df

# Create figure with two subplots (stacked vertically, larger first)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), 
                                gridspec_kw={'height_ratios': [1.1, 1]})

# Function to plot top 15
def plot_top15(ax, data_df, scope_name, period, color):
    data_df = data_df.sort_values('Share_of_overall_%', ascending=True).tail(15)
    
    # Create colors list - highlight France in orange
    colors_list = ['#fdb462' if country == 'France' else color for country in data_df['Country']]
    
    bars = ax.barh(data_df['Country'], data_df['Share_of_overall_%'], 
                    color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (idx, row) in enumerate(data_df.iterrows()):
        ax.text(row['Share_of_overall_%'] + 0.1, i, f"{row['Share_of_overall_%']:.2f}%", 
                va='center', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Part des émissions cumulées totales (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Top 15 Pays - Émissions {scope_name} ({period})', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(data_df['Share_of_overall_%']) * 1.15)

# Plot first scope (larger share)
period1 = '1970-2023' if first_scope == 'Territorial' else '1970-2022'
plot_top15(ax1, first_df, first_scope, period1, COLORS['Population'])

# Plot second scope (smaller share)
period2 = '1970-2023' if second_scope == 'Territorial' else '1970-2022'
plot_top15(ax2, second_df, second_scope, period2, COLORS['Responsibility'])

plt.tight_layout()
output_file = OUTPUT_DIR / 'FR_Top15_Emissions_Cumulees.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n[OK] Sauvegardé: {output_file.name}")
plt.close(fig)

# ============================================================================
# VISUALIZATION 2: CARBON BUDGET REMAINING COMPARISON
# ============================================================================
print("\n" + "="*80)
print("2. COMPARAISON DU BUDGET CARBONE RESTANT")
print("="*80)

# SNBC-3 budget
snbc3_budget = 5100  # Mt CO2e for 2024-2050

# Filter Territory with Population, Responsibility, and Capability
territory_alloc = budget_df[budget_df['Emissions_Scope'] == 'Territory'].copy()

# Create comparison data - SNBC-3 FIRST
budget_scenarios = {
    'SNBC-3\n(5 100 MtCO2e)': snbc3_budget,
    'Responsabilité': territory_alloc[territory_alloc['Allocation_Scenario'] == 'Responsibility']['France_Budget_MtCO2'].values[0],
    'Capacité': territory_alloc[territory_alloc['Allocation_Scenario'] == 'Capability']['France_Budget_MtCO2'].values[0],
}

print("\nBudget Carbone Restant (Mt CO2e):")
for scenario, budget in budget_scenarios.items():
    print(f"  {scenario.replace(chr(10), ' ')}: {budget:,.0f} Mt CO2e")

# Create figure
fig, ax = plt.subplots(figsize=(14, 9))

scenarios = list(budget_scenarios.keys())
budgets = list(budget_scenarios.values())
colors_list = [COLORS['SNBC-3'], COLORS['Responsibility'], COLORS['Capability']]

# Create bars
bars = ax.bar(scenarios, budgets, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels - adjust position for negative values
for bar, budget in zip(bars, budgets):
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{budget:,.0f}',
                ha='center', va='bottom',
                fontweight='bold', fontsize=12)
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height - 800,
                f'{budget:,.0f}',
                ha='center', va='top',
                fontweight='bold', fontsize=12)

# Add a horizontal line at 0
ax.axhline(y=0, color='black', linewidth=2, linestyle='-')

# Formatting
ax.set_ylabel('Budget Carbone Restant (Mt CO2e)', fontsize=12, fontweight='bold')
ax.set_title('Budget Carbone Restant pour la France: SNBC-3 vs Scénarios d\'Allocation\n(trajectoire 1,5°C)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')
y_min = min(budgets) * 1.3
y_max = max(budgets) * 1.2
ax.set_ylim(y_min, y_max)

# Add annotation
ax.text(0.5, 0.97, 'Positif = budget carbone disponible | Négatif = budget dépassé',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xticks(fontsize=11, fontweight='bold')
plt.yticks(fontsize=10)
plt.tight_layout()

output_file = OUTPUT_DIR / 'FR_Budget_Carbone_Comparaison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n[OK] Sauvegardé: {output_file.name}")
plt.close(fig)

# ============================================================================
# VISUALIZATION 3: NEUTRALITY YEAR COMPARISON
# ============================================================================
print("\n" + "="*80)
print("3. COMPARAISON DE L'ANNÉE DE NEUTRALITÉ")
print("="*80)

# Get neutrality years for Territory scenarios
neutrality_data = budget_df[budget_df['Emissions_Scope'] == 'Territory'].copy()

# Filter to exclude NDC Pledges and Population, keep only Responsibility, Capability
neutrality_data = neutrality_data[neutrality_data['Allocation_Scenario'].isin(['Responsibility', 'Capability'])]

# Prepare data - SNBC-3 first
ordered_years = {'SNBC-3': 2050}

# Add Territory scenarios (without "Territory" label)
for idx, row in neutrality_data.iterrows():
    alloc = row['Allocation_Scenario']
    year = row['Neutrality_Year']
    if alloc == 'Responsibility':
        ordered_years['Responsabilité'] = year
    elif alloc == 'Capability':
        ordered_years['Capacité'] = year

print("\nAnnées de Neutralité Carbone:")
for scenario, year in ordered_years.items():
    print(f"  {scenario}: {int(year)}")

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Reference year
ref_year = 2023

# Calculate bar heights (from 2023)
scenarios_neut = list(ordered_years.keys())
years_list = list(ordered_years.values())
heights = [year - ref_year for year in years_list]
colors_neut = [COLORS['SNBC-3'], COLORS['Responsibility'], COLORS['Capability']]

# Create bars starting from 2023
bars = ax.bar(scenarios_neut, heights, bottom=ref_year, color=colors_neut, 
              alpha=0.8, edgecolor='black', linewidth=2)

# Add year labels with small boxes (style from LMDI)
for bar, year, height in zip(bars, years_list, heights):
    x_pos = bar.get_x() + bar.get_width()/2.
    
    if height > 0:
        # Positive (future years)
        y_pos = ref_year + height - 1
        ax.text(x_pos, y_pos,
                f'{int(year)}',
                ha='center', va='top',
                fontweight='bold', fontsize=11, color='black',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=0.3))
    else:
        # Negative (past years)
        y_pos = ref_year + height + 1
        ax.text(x_pos, y_pos,
                f'{int(year)}',
                ha='center', va='bottom',
                fontweight='bold', fontsize=11, color='black',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=0.3))

# Add reference line for 2023
ax.axhline(y=ref_year, color='black', linewidth=2.5, linestyle='-', label='Année de référence (2023)')

# Formatting
ax.set_ylabel('Année', fontsize=12, fontweight='bold')
ax.set_title('Année de Neutralité Carbone pour la France: SNBC-3 vs Scénarios d\'Allocation\n(trajectoire 1,5°C)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(1970, 2060)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

plt.xticks(fontsize=11, fontweight='bold')
plt.yticks(fontsize=10)
plt.tight_layout()

output_file = OUTPUT_DIR / 'FR_Annee_Neutralite.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n[OK] Sauvegardé: {output_file.name}")
plt.close(fig)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VISUALISATIONS COMPLÈTES")
print("="*80)
print("\nFichiers générés:")
print(f"  1. FR_Top15_Emissions_Cumulees.png")
print(f"  2. FR_Budget_Carbone_Comparaison.png")
print(f"  3. FR_Annee_Neutralite.png")
print(f"\nTous sauvegardés dans: {OUTPUT_DIR}")
print("="*80)


# ============================================================================
# STACKED BAR CHART - DIRECT EMISSIONS BY SECTOR
# ============================================================================
def plot_stacked_emissions():
    """
    Create pie charts comparing emissions by sector for 2021 and 2030
    """
    # Data from user
    sectors = ['Industrie', 'Agriculture', 'Transports', 'Bâtiments', 'Déchets']
    emissions_2021 = [77.23005488, 76.4039341, 127.2180912, 74.36784578, 14.79593998]
    emissions_2030 = [45.15353526, 66.52630478, 90.378463, 34.76617683, 7.207042768]
    
    # Colors for sectors (using palette)
    sector_colors = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Totals
    total_2021 = sum(emissions_2021)
    total_2030 = sum(emissions_2030)
    reduction = total_2021 - total_2030
    reduction_pct = (reduction / total_2021) * 100
    
    # Calculate percentages
    percentages_2021 = [(val / total_2021) * 100 for val in emissions_2021]
    percentages_2030 = [(val / total_2030) * 100 for val in emissions_2030]
    
    # Pie chart 2021
    wedges1, texts1, autotexts1 = ax1.pie(emissions_2021, labels=sectors, colors=sector_colors,
                                            autopct='%1.1f%%', startangle=90, textprops={'fontweight': 'bold', 'fontsize': 11},
                                            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    # Make percentage text white/bold for better readability
    for autotext in autotexts1:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax1.set_title(f'2021\nTotal: {total_2021:.1f} MtCO2e', fontsize=13, fontweight='bold', pad=15)
    
    # Pie chart 2030
    wedges2, texts2, autotexts2 = ax2.pie(emissions_2030, labels=sectors, colors=sector_colors,
                                            autopct='%1.1f%%', startangle=90, textprops={'fontweight': 'bold', 'fontsize': 11},
                                            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    # Make percentage text white/bold for better readability
    for autotext in autotexts2:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax2.set_title(f'2030\nTotal: {total_2030:.1f} MtCO2e', fontsize=13, fontweight='bold', pad=15)
    
    # Add overall title
    fig.suptitle('Émissions Directes par Secteur - Scénario SNBC-3\n2021 vs 2030', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'FR_Emissions_Stacked_2021_2030.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: FR_Emissions_Stacked_2021_2030.png")
    plt.close()
    
    return {
        'total_2021': total_2021,
        'total_2030': total_2030,
        'reduction': reduction,
        'reduction_pct': reduction_pct
    }


# ============================================================================
# EXECUTE STACKED EMISSIONS CHART
# ============================================================================
print("\n" + "="*80)
print("4. STACKED EMISSIONS BY SECTOR (2021 vs 2030)")
print("="*80)
stacked_results = plot_stacked_emissions()
print(f"\n  Total 2021: {stacked_results['total_2021']:.1f} MtCO2e")
print(f"  Total 2030: {stacked_results['total_2030']:.1f} MtCO2e")
print(f"  Réduction: {stacked_results['reduction']:.1f} MtCO2e ({stacked_results['reduction_pct']:.1f}%)")
