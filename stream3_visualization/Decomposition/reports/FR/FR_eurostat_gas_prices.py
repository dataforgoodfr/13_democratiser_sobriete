"""
Visualisation EUROSTAT - Prix du gaz naturel en France par bande de consommation
Génère un graphe stacked bar montrant la composition des prix (Energy & supply, Taxes, VAT, etc.)

Sortie: fichiers PNG dans le répertoire visuals EUROSTAT
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
DATA_PATH = BASE_PATH / 'data-EUROSTAT'
OUTPUT_DIR = BASE_PATH / 'visuals EUROSTAT'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION - Color palette from Well-being dashboard
# ============================================================================
COLORS = {
    'NRG_SUP': '#fdb462',           # Orange-warm - Energy and supply
    'NETC': '#8dd3c7',              # Cyan - Network costs
    'TAX_FEE_LEV_CHRG': '#bebada',  # Mauve - Taxes, fees, levies and charges
    'VAT': '#b3de69',               # Green - Value added tax
    'TAX_RNW': '#ffd558',           # Yellow - Renewable taxes
    'TAX_CAP': '#fb8072',           # Red - Capacity taxes
    'TAX_ENV': '#ffffb3',           # Pale - Environmental taxes
    'TAX_NUC': '#80b1d3',           # Blue - Nuclear taxes
    'OTH': '#bc80bd',               # Purple - Other
}

# Category labels (user-provided)
CATEGORY_LABELS = {
    'NRG_SUP': 'Energy and supply',
    'NETC': 'Network costs',
    'TAX_FEE_LEV_CHRG': 'Taxes, fees, levies and charges',
    'VAT': 'Value added tax (VAT)',
    'TAX_RNW': 'Renewable taxes',
    'TAX_CAP': 'Capacity taxes',
    'TAX_ENV': 'Environmental taxes',
    'TAX_NUC': 'Nuclear taxes',
    'OTH': 'Other',
}

# Consumption band labels for Gas
CONSUMPTION_LABELS = {
    'TOT_GJ': 'All consumption bands',
    'GJ_LT20': '< 20 GJ',
    'GJ20-199': '20 - 199 GJ',
    'GJ_GE200': '≥ 200 GJ',
}

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("VISUALISATION EUROSTAT - PRIX DU GAZ NATUREL EN FRANCE")
print("="*80)

# Load EUROSTAT data
eurostat_file = DATA_PATH / 'nrg_pc_202_c__custom_19587171_gas.csv'

if not eurostat_file.exists():
    print(f"\nERREUR: Fichier non trouvé: {eurostat_file}")
    exit(1)

df = pd.read_csv(eurostat_file)

print(f"\nChargement des données EUROSTAT:")
print(f"  Fichier: nrg_pc_202_c__custom_19587171_gas.csv")
print(f"  Lignes totales: {len(df)}")
print(f"  Colonnes: {list(df.columns)}")

# ============================================================================
# FILTER DATA FOR FRANCE 2024
# ============================================================================
# Filter for France and 2024
df_france_2024 = df[(df['geo'] == 'France') & (df['TIME_PERIOD'] == 2024)].copy()

print(f"\nFiltrage pour France 2024:")
print(f"  Lignes trouvées: {len(df_france_2024)}")

if len(df_france_2024) == 0:
    print("  ERREUR: Aucune donnée trouvée pour France 2024")
    exit(1)

# Check unique values
print(f"\nValeurs uniques nrg_cons:")
print(df_france_2024['nrg_cons'].unique())

# Map nrg_cons values to codes (for Gas)
nrg_cons_mapping = {
    'Consumption of GJ - all bands': 'TOT_GJ',
    'Consumption less than 20 GJ - band D1': 'GJ_LT20',
    'Consumption from 20 GJ to 199 GJ - band D2': 'GJ20-199',
    'Consumption 200 GJ or over - band D3': 'GJ_GE200',
}

# Create mapping for nrg_prc values
nrg_prc_mapping = {
    'Energy and supply': 'NRG_SUP',
    'Network costs': 'NETC',
    'Taxes, fees, levies and charges': 'TAX_FEE_LEV_CHRG',
    'Value added tax (VAT)': 'VAT',
    'Renewable taxes': 'TAX_RNW',
    'Capacity taxes': 'TAX_CAP',
    'Environmental taxes': 'TAX_ENV',
    'Nuclear taxes': 'TAX_NUC',
    'Other': 'OTH',
}

# Add mapped columns
df_france_2024['nrg_cons_code'] = df_france_2024['nrg_cons'].map(nrg_cons_mapping)
df_france_2024['nrg_prc_code'] = df_france_2024['nrg_prc'].map(nrg_prc_mapping)

# Check for unmapped values
unmapped_cons = df_france_2024[df_france_2024['nrg_cons_code'].isna()]['nrg_cons'].unique()
unmapped_prc = df_france_2024[df_france_2024['nrg_prc_code'].isna()]['nrg_prc'].unique()

if len(unmapped_cons) > 0:
    print(f"\n  Attention: Bandes de consommation non mappées: {unmapped_cons}")
if len(unmapped_prc) > 0:
    print(f"  Attention: Catégories de prix non mappées: {unmapped_prc}")

# Remove unmapped rows
df_france_2024 = df_france_2024.dropna(subset=['nrg_cons_code', 'nrg_prc_code'])

print(f"  Après mappag: {len(df_france_2024)} lignes valides")

# ============================================================================
# PREPARE DATA FOR STACKED BAR CHART
# ============================================================================

# Desired order of consumption bands for Gas
consumption_order = ['TOT_GJ', 'GJ_LT20', 'GJ20-199', 'GJ_GE200']

# Create pivot table: rows=nrg_cons_code, columns=nrg_prc_code, values=OBS_VALUE
pivot_data = df_france_2024.pivot_table(
    index='nrg_cons_code',
    columns='nrg_prc_code',
    values='OBS_VALUE',
    aggfunc='first'
)

print(f"\nDonnées pivot:")
print(pivot_data)

# Reorder rows according to consumption_order
pivot_data = pivot_data.reindex([c for c in consumption_order if c in pivot_data.index])

# ============================================================================
# CREATE STACKED BAR CHART
# ============================================================================
print("\n" + "="*80)
print("CRÉATION DU GRAPHE STACKED BAR")
print("="*80)

fig, ax = plt.subplots(figsize=(16, 9))

# Define price components order (for stacking)
price_components = ['NRG_SUP', 'NETC', 'TAX_FEE_LEV_CHRG', 'VAT', 'TAX_RNW', 'TAX_CAP', 'TAX_ENV', 'TAX_NUC', 'OTH']
price_components = [p for p in price_components if p in pivot_data.columns]

# Create x-axis positions
x_pos = np.arange(len(pivot_data))
bar_width = 0.6

# Plot stacked bars
bottom = np.zeros(len(pivot_data))
colors_list = [COLORS.get(pc, '#cccccc') for pc in price_components]

for idx, component in enumerate(price_components):
    if component in pivot_data.columns:
        values = pivot_data[component].fillna(0).values
        ax.bar(x_pos, values, bar_width, bottom=bottom, 
               label=CATEGORY_LABELS.get(component, component),
               color=colors_list[idx], alpha=0.85, edgecolor='black', linewidth=1)
        bottom += values

# Add value labels on bars (optional - comment if too cluttered)
for i in range(len(pivot_data)):
    total = pivot_data.iloc[i].sum()
    ax.text(i, total + 0.02, f'{total:.3f}', ha='center', va='bottom', 
            fontweight='bold', fontsize=12)

# Styling
consumption_labels = [CONSUMPTION_LABELS.get(c, c) for c in pivot_data.index]
ax.set_xticks(x_pos)
ax.set_xticklabels(consumption_labels, fontsize=13, fontweight='bold', rotation=15, ha='right')

ax.set_ylabel('Prix (Purchasing Power Standard)', fontsize=13, fontweight='bold')
ax.set_title('Composition des Prix du Gaz Naturel en France (2024)\npar Bande de Consommation',
             fontsize=15, fontweight='bold', pad=20)

ax.legend(loc='upper right', fontsize=12, framealpha=0.95, ncol=1)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(bottom) * 1.15)

plt.tight_layout()

# Save figure
output_file = OUTPUT_DIR / 'FR_Gas_Prices_Composition_2024.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n[OK] Graphe sauvegardé: {output_file.name}")
print(f"     Chemin complet: {output_file}")
plt.close(fig)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VISUALISATION COMPLÈTE")
print("="*80)
print(f"\nFichier généré:")
print(f"  FR_Gas_Prices_Composition_2024.png")
print(f"\nSauvegardé dans: {OUTPUT_DIR}")
print("="*80)
