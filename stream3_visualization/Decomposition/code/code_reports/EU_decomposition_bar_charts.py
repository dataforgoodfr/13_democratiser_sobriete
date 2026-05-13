"""
EU Decomposition - Bar Charts of Lever Contributions (%)
Scenario: EU Commission >90% Decrease by 2040, period 2015-2050

Generates 4 visuals:
  1. By sector group (Buildings, Transport, Industry) with subsectors side by side
  2. By sector group, all subsectors in one graph per group
  3. All sectors on separate subplots
  4. All sectors and subsectors on separate subplots

Output: stream3_visualization/Decomposition/reports/EU/
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
BASE_PATH = Path(__file__).parent.parent.parent  # stream3_visualization/Decomposition
DATA_PATH = BASE_PATH / 'Output'
OUTPUT_DIR = BASE_PATH / 'reports' / 'EU'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
SCENARIO = 'EU Commission >90% Decrease by 2040'
PERIOD_COL = 'Contrib_2015_2050_pct'

LEVERS = ['Sufficiency', 'Energy Efficiency', 'Supply Side Decarbonation']

LEVER_COLORS = {
    'Sufficiency':               '#fdb462',  # orange
    'Energy Efficiency':         '#8dd3c7',  # teal
    'Supply Side Decarbonation': '#fb8072',  # salmon
}

LEVER_SHORT = {
    'Sufficiency': 'Sufficiency',
    'Energy Efficiency': 'Energy\nEfficiency',
    'Supply Side Decarbonation': 'Supply Side\nDecarbonation',
}

# Sector groupings
SECTOR_GROUPS = {
    'Buildings': ['Buildings - Residential', 'Buildings - Services'],
    'Transport': ['Transport - Passenger cars', 'Transport - Rail'],
    'Industry':  [
        'Industry - Steel industry',
        'Industry - Non-ferrous metal industry',
        'Industry - Chemicals industry',
        'Industry - Non-Metallic Minerals industry',
        'Industry - Pulp, Paper & Print industry',
    ],
}

# Short labels for subsectors (for axis ticks)
SECTOR_SHORT = {
    'Buildings - Residential':                  'Residential',
    'Buildings - Services':                     'Services',
    'Transport - Passenger cars':               'Passenger cars',
    'Transport - Rail':                         'Rail',
    'Industry - Steel industry':                'Steel',
    'Industry - Non-ferrous metal industry':    'Non-ferrous metals',
    'Industry - Chemicals industry':            'Chemicals',
    'Industry - Non-Metallic Minerals industry':'Non-metallic minerals',
    'Industry - Pulp, Paper & Print industry':  'Pulp, paper & print',
}


def _save(fig, name):
    """Save figure as PNG + SVG and close."""
    for ext in ('png', 'svg'):
        fig.savefig(OUTPUT_DIR / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {name}.png / .svg')


# ============================================================================
# LOAD DATA
# ============================================================================
print('=' * 70)
print('EU DECOMPOSITION – BAR CHARTS')
print('=' * 70)

df_all = pd.read_csv(DATA_PATH / 'unified_decomposition_data.csv')
df = df_all[
    (df_all['Zone'] == 'EU') &
    (df_all['Scenario'] == SCENARIO) &
    (df_all['Lever'] != 'Total') &
    (df_all['Lever'] != 'Population')
].copy()

print(f'Scenario : {SCENARIO}')
print(f'Sectors  : {sorted(df["Sector"].unique())}')
print(f'Rows     : {len(df)}')

# Export Excel file
excel_path = OUTPUT_DIR / 'eu_decomposition_90pct_2015_2050.xlsx'
pivot = df.pivot(index='Sector', columns='Lever', values=PERIOD_COL)
pivot.columns.name = None
pivot.index.name = 'Sector'
pivot = pivot[LEVERS]  # enforce lever order
pivot.to_excel(excel_path)
print(f'Excel    : {excel_path}')


# ============================================================================
# HELPER: single grouped-bar axes
# ============================================================================
def _bar_subsectors(ax, sub_df, title=None):
    """Draw grouped bars (one group per subsector, one bar per lever)."""
    sectors = sub_df['Sector'].unique()
    n_sectors = len(sectors)
    n_levers = len(LEVERS)
    x = np.arange(n_sectors)
    width = 0.8 / n_levers

    for i, lever in enumerate(LEVERS):
        vals = [sub_df.loc[sub_df['Sector'] == s, PERIOD_COL].values[0] for s in sectors]
        bars = ax.bar(x + i * width - (n_levers - 1) * width / 2, vals,
                      width, label=lever, color=LEVER_COLORS[lever], edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (1 if v >= 0 else -3),
                    f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=7, color='#2c3e50')

    ax.set_xticks(x)
    ax.set_xticklabels([SECTOR_SHORT.get(s, s) for s in sectors], fontsize=9)
    ax.set_ylabel('Contribution to CO₂ change (%)', fontsize=9)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color='#2c3e50')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _bar_levers(ax, sub_df, title=None):
    """Draw grouped bars (one group per lever, one bar per subsector)."""
    sectors = sub_df['Sector'].unique()
    n_sectors = len(sectors)
    n_levers = len(LEVERS)
    x = np.arange(n_levers)
    width = 0.8 / n_sectors

    for i, sector in enumerate(sectors):
        vals = [sub_df.loc[sub_df['Sector'] == sector, PERIOD_COL].values[0]
                if lever in sub_df.loc[sub_df['Sector'] == sector, 'Lever'].values
                else 0
                for lever in LEVERS]
        # get actual values from the filtered df
        vals = []
        for lever in LEVERS:
            row = sub_df[(sub_df['Sector'] == sector) & (sub_df['Lever'] == lever)]
            vals.append(row[PERIOD_COL].values[0] if len(row) else 0)
        bars = ax.bar(x + i * width - (n_sectors - 1) * width / 2, vals,
                      width, label=SECTOR_SHORT.get(sector, sector),
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (1 if v >= 0 else -3),
                    f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=7, color='#2c3e50')

    ax.set_xticks(x)
    ax.set_xticklabels([LEVER_SHORT[l] for l in LEVERS], fontsize=9)
    ax.set_ylabel('Contribution to CO₂ change (%)', fontsize=9)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color='#2c3e50')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================================
# VISUAL 1 – By sector group, subsectors side by side
# ============================================================================
print('\n--- Visual 1: By sector group (subsectors side by side) ---')

for group_name, sectors in SECTOR_GROUPS.items():
    n = len(sectors)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, sector in zip(axes, sectors):
        sub = df[df['Sector'] == sector]
        vals = [sub.loc[sub['Lever'] == l, PERIOD_COL].values[0] for l in LEVERS]
        colors = [LEVER_COLORS[l] for l in LEVERS]
        bars = ax.bar([LEVER_SHORT[l] for l in LEVERS], vals, color=colors,
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (1 if v >= 0 else -3),
                    f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=9, color='#2c3e50', fontweight='bold')
        ax.set_title(SECTOR_SHORT[sector], fontsize=12, fontweight='bold', color='#2c3e50')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Contribution to CO₂ change (%)', fontsize=10)
    fig.suptitle(f'{group_name} – Lever Contributions (2015-2050)\n{SCENARIO}',
                 fontsize=13, fontweight='bold', color='#2c3e50', y=1.02)
    fig.tight_layout()
    _save(fig, f'1_bar_by_group_{group_name.lower()}')


# ============================================================================
# VISUAL 2 – All subsectors in one graph (x-axis = levers, bars = subsectors)
# ============================================================================
print('\n--- Visual 2: All subsectors in one graph ---')

all_subsectors = []
for sectors in SECTOR_GROUPS.values():
    all_subsectors.extend(sectors)

sub = df[df['Sector'].isin(all_subsectors)]

fig, ax = plt.subplots(figsize=(12, 6))
_bar_levers(ax, sub, title='All Sectors – Lever Contributions (2015-2050)')
ax.legend(fontsize=8, loc='best', framealpha=0.9)
fig.suptitle(SCENARIO, fontsize=10, color='#555', y=1.0)
fig.tight_layout()
_save(fig, '2_bar_grouped_all_sectors')


# ============================================================================
# VISUAL 3 – All sectors on separate subplots
# ============================================================================
print('\n--- Visual 3: All sectors on separate subplots ---')

all_sectors = []
for sectors in SECTOR_GROUPS.values():
    all_sectors.extend(sectors)

n = len(all_sectors)
ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), sharey=True)
axes_flat = axes.flatten()

for idx, sector in enumerate(all_sectors):
    ax = axes_flat[idx]
    sub = df[df['Sector'] == sector]
    vals = [sub.loc[sub['Lever'] == l, PERIOD_COL].values[0] for l in LEVERS]
    colors = [LEVER_COLORS[l] for l in LEVERS]
    bars = ax.bar([LEVER_SHORT[l] for l in LEVERS], vals, color=colors,
                  edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (1 if v >= 0 else -3),
                f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=7, color='#2c3e50', fontweight='bold')
    ax.set_title(SECTOR_SHORT[sector], fontsize=10, fontweight='bold', color='#2c3e50')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# hide empty subplots
for idx in range(n, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle(f'All Sectors – Lever Contributions (2015-2050)\n{SCENARIO}',
             fontsize=14, fontweight='bold', color='#2c3e50')
fig.tight_layout(rect=[0, 0, 1, 0.95])
_save(fig, '3_bar_all_sectors')


# ============================================================================
# VISUAL 4 – All sectors and subsectors on separate subplots (grouped by macro sector)
# ============================================================================
print('\n--- Visual 4: All sectors & subsectors (grouped by macro sector) ---')

# One row per sector group, columns = max subsectors
max_cols = max(len(v) for v in SECTOR_GROUPS.values())
n_groups = len(SECTOR_GROUPS)

fig, axes = plt.subplots(n_groups, max_cols, figsize=(5 * max_cols, 4.5 * n_groups),
                         sharey=True)

for row, (group_name, sectors) in enumerate(SECTOR_GROUPS.items()):
    for col in range(max_cols):
        ax = axes[row, col]
        if col < len(sectors):
            sector = sectors[col]
            sub = df[df['Sector'] == sector]
            vals = [sub.loc[sub['Lever'] == l, PERIOD_COL].values[0] for l in LEVERS]
            colors = [LEVER_COLORS[l] for l in LEVERS]
            bars = ax.bar([LEVER_SHORT[l] for l in LEVERS], vals, color=colors,
                          edgecolor='white', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (1 if v >= 0 else -3),
                        f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top',
                        fontsize=7, color='#2c3e50', fontweight='bold')
            ax.set_title(SECTOR_SHORT[sector], fontsize=10, fontweight='bold', color='#2c3e50')
            ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.set_visible(False)

    # Row label on the left
    axes[row, 0].set_ylabel(f'{group_name}\nContribution (%)', fontsize=10, fontweight='bold')

fig.suptitle(f'All Sectors & Subsectors – Lever Contributions (2015-2050)\n{SCENARIO}',
             fontsize=14, fontweight='bold', color='#2c3e50')

# Add a shared legend at the bottom
handles = [plt.Rectangle((0, 0), 1, 1, color=LEVER_COLORS[l]) for l in LEVERS]
fig.legend(handles, LEVERS, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02), frameon=False)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
_save(fig, '4_bar_sectors_subsectors')

print('\nDone – all visuals saved to:', OUTPUT_DIR)
