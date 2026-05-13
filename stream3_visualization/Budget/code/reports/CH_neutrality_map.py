"""
Switzerland Report – Carbon Budget Neutrality Year Maps
Generates the 4-scenario map (shared legend) for rep_ch countries:
  CH, FR, IT, DE, AT

Output: stream3_visualization/Budget/code/reports/CH/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_PATH = Path(__file__).parent.parent.parent  # stream3_visualization/Budget
DATA_PATH = BASE_PATH / 'Output'
OUTPUT_DIR = Path(__file__).parent / 'CH'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_WELL_BEING = Path(__file__).parent.parent.parent.parent / 'Well-being'
_NUTS_GPKG = _WELL_BEING / 'reports' / '1_switzerland_vs_eu27_housing_energy' / 'external_data' / '0_shapefile' / 'NUTS_RG_10M_2024_3035.gpkg'
_WORLD_SHP = _WELL_BEING / 'reports' / '1_switzerland_vs_eu27_housing_energy' / 'external_data' / '0_shapefile' / 'ne_50m_admin_0_countries' / 'ne_50m_admin_0_countries.shp'

# ============================================================================
# CONFIGURATION
# ============================================================================
WARMING = '1.5°C'
PROBABILITY = '50%'

# rep_ch comparison group
COUNTRIES = ['CH', 'FR', 'IT', 'DE', 'AT']

SCENARIOS = [
    ('Territory',   'Responsibility'),
    ('Territory',   'Capability'),
    ('Consumption', 'Responsibility'),
    ('Consumption', 'Capability'),
]

SCENARIO_LABELS = {
    ('Territory',   'Responsibility'): 'Territorial / Responsibility',
    ('Territory',   'Capability'):     'Territorial / Capability',
    ('Consumption', 'Responsibility'): 'Consumption / Responsibility',
    ('Consumption', 'Capability'):     'Consumption / Capability',
}

SCENARIO_COLORS = {
    ('Territory',   'Responsibility'): '#fb8072',
    ('Territory',   'Capability'):     '#fdb462',
    ('Consumption', 'Responsibility'): '#8dd3c7',
    ('Consumption', 'Capability'):     '#80b1d3',
}


def _save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(OUTPUT_DIR / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {name}.png / .svg')


# ============================================================================
# LOAD DATA
# ============================================================================
print('=' * 70)
print('SWITZERLAND REPORT – NEUTRALITY YEAR MAPS')
print('=' * 70)

sp = pd.read_csv(DATA_PATH / 'scenario_parameters.csv')

base = sp[
    (sp['Warming_scenario'] == WARMING) &
    (sp['Probability_of_reach'] == PROBABILITY) &
    (sp['ISO2'].isin(COUNTRIES)) &
    (sp['ISO_Type'] == 'Country')
].copy()

print(f'Base rows ({len(COUNTRIES)} countries, 1.5°C, 50%): {len(base)}')

scenario_dfs = {}
for scope, alloc in SCENARIOS:
    key = (scope, alloc)
    sub = base[
        (base['Emissions_scope'] == scope) &
        (base['Budget_distribution_scenario'] == alloc)
    ].copy()
    sub['Scenario'] = SCENARIO_LABELS[key]
    scenario_dfs[key] = sub
    print(f'  {SCENARIO_LABELS[key]}: {len(sub)} countries, '
          f'neutrality year {sub["Neutrality_year"].min():.0f}–{sub["Neutrality_year"].max():.0f}')


# ============================================================================
# MAP HELPERS
# ============================================================================
def _load_nuts0_and_background():
    nuts = gpd.read_file(_NUTS_GPKG)
    nuts0 = nuts[nuts['LEVL_CODE'] == 0].copy()
    if nuts0.crs != 'EPSG:3035':
        nuts0 = nuts0.to_crs(epsg=3035)
    world = gpd.read_file(_WORLD_SHP)
    bg = world[(world['CONTINENT'] == 'Europe') | (world['ISO_A2'] == 'TR')].copy()
    bg = bg.to_crs(epsg=3035)
    return nuts0, bg


def _plot_neutrality_map(ax, gdf, col, cmap, norm, title, study_set, bg):
    # Background: non-study European countries
    non_study = bg[~bg['ISO_A2'].isin(study_set)]
    if not non_study.empty:
        non_study.plot(ax=ax, color='white', edgecolor='black',
                       linewidth=0.3, hatch='///', alpha=0.35)
    # Study countries without data
    no_data = gdf[gdf[col].isna()]
    if not no_data.empty:
        no_data.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.3)
    # Study countries with data
    has_data = gdf[gdf[col].notna()]
    if not has_data.empty:
        has_data.plot(column=col, cmap=cmap, norm=norm, ax=ax,
                      edgecolor='black', linewidth=0.4, legend=False)
    ax.set_title(title, fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_axis_off()
    # Crop to CH comparison group extent
    ax.set_xlim(2_800_000, 5_300_000)
    ax.set_ylim(1_500_000, 3_800_000)


# ============================================================================
# MAP: 4 scenarios side by side (shared legend)
# ============================================================================
print('\n--- Map: 4 scenarios side by side (shared legend) ---')

nuts0, bg = _load_nuts0_and_background()
country_set = set(COUNTRIES)

map_data = {}
for key in SCENARIOS:
    sub = scenario_dfs[key][['ISO2', 'Neutrality_year']].copy()
    merged = nuts0.merge(sub, left_on='CNTR_CODE', right_on='ISO2', how='left')
    merged = merged[merged['CNTR_CODE'].isin(country_set)]
    map_data[key] = merged

# Shared color scale
all_ny = pd.concat([d['Neutrality_year'] for d in map_data.values()]).dropna()
vmin, vmax = all_ny.min(), all_ny.max()
cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
layout = [
    [('Territory', 'Responsibility'), ('Territory', 'Capability')],
    [('Consumption', 'Responsibility'), ('Consumption', 'Capability')],
]
for row_idx, row_keys in enumerate(layout):
    for col_idx, key in enumerate(row_keys):
        ax = axes[row_idx, col_idx]
        _plot_neutrality_map(ax, map_data[key], 'Neutrality_year', cmap, norm,
                             SCENARIO_LABELS[key], country_set, bg)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.02])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Year to carbon neutrality', fontsize=11)

fig.suptitle(f'Year to Carbon Neutrality – Switzerland & Comparison Countries\n{WARMING} warming, {PROBABILITY} probability',
             fontsize=14, fontweight='bold', color='#2c3e50', y=0.98)
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
_save(fig, '6_map_all_scenarios_shared_legend')

print(f'\nDone – outputs in: {OUTPUT_DIR}')
