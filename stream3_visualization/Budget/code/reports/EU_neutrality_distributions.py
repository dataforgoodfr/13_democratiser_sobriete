"""
EU Carbon Budget – Year-to-Neutrality Distribution Charts
Scenario filter: 1.5°C warming, 50% probability of not exceeding
4 budget variants: Territorial×Responsibility, Territorial×Capability,
                   Consumption×Responsibility, Consumption×Capability

Generates visuals:
  1. One histogram per scenario (2×2 grid, 5-year bins)
  2. All 4 scenarios overlaid (single chart)
  3. All 4 scenarios + NDC Pledges
  4. Same as 3 but limited to 8 EWBI cluster countries (2 per cluster)
  5. Maps: 1×4 with shared legend + individual per scenario

Output: stream3_visualization/Budget/code/reports/EU/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_PATH = Path(__file__).parent.parent.parent  # stream3_visualization/Budget
DATA_PATH = BASE_PATH / 'Output'
OUTPUT_DIR = Path(__file__).parent / 'EU'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shapefile paths (relative to Well-being reports)
_WELL_BEING = Path(__file__).parent.parent.parent.parent / 'Well-being'
_NUTS_GPKG = _WELL_BEING / 'reports' / '1_switzerland_vs_eu27_housing_energy' / 'external_data' / '0_shapefile' / 'NUTS_RG_10M_2024_3035.gpkg'
_WORLD_SHP = _WELL_BEING / 'reports' / '1_switzerland_vs_eu27_housing_energy' / 'external_data' / '0_shapefile' / 'ne_50m_admin_0_countries' / 'ne_50m_admin_0_countries.shp'

# ============================================================================
# CONFIGURATION
# ============================================================================
WARMING = '1.5°C'
PROBABILITY = '50%'

EU27 = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
        'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT',
        'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']

# 4 scenarios (scope × allocation)
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
    ('Territory',   'Responsibility'): '#fb8072',  # salmon
    ('Territory',   'Capability'):     '#fdb462',  # orange
    ('Consumption', 'Responsibility'): '#8dd3c7',  # teal
    ('Consumption', 'Capability'):     '#80b1d3',  # blue
}

NDC_COLOR = '#b3de69'  # green

# EWBI clusters (12 countries, 3 per cluster)
CLUSTER_COUNTRIES = {
    'Cluster 0 – Low perf / Low EWBI':  {'countries': ['FR', 'ES', 'IT'], 'color': '#fb8072'},
    'Cluster 1 – Low perf / High EWBI': {'countries': ['BE', 'NL', 'AT'], 'color': '#fdb462'},
    'Cluster 2 – High perf / Low EWBI': {'countries': ['LT', 'HU', 'RO'], 'color': '#8dd3c7'},
    'Cluster 3 – High perf / High EWBI':{'countries': ['DE', 'PL', 'CZ'], 'color': '#80b1d3'},
}
CLUSTER_8 = [c for v in CLUSTER_COUNTRIES.values() for c in v['countries']]

# Full cluster assignments for all EU-27 countries
ALL_CLUSTERS = {
    'Cluster 0 \u2013 Low perf / Low EWBI':  {'countries': ['FR', 'ES', 'IT', 'PT', 'GR', 'FI', 'CY', 'MT'], 'color': '#fb8072'},
    'Cluster 1 \u2013 Low perf / High EWBI': {'countries': ['BE', 'NL', 'AT', 'DK', 'IE', 'LU'], 'color': '#fdb462'},
    'Cluster 2 \u2013 High perf / Low EWBI': {'countries': ['LT', 'HU', 'RO', 'BG', 'HR', 'LV', 'EE'], 'color': '#8dd3c7'},
    'Cluster 3 \u2013 High perf / High EWBI':{'countries': ['DE', 'PL', 'CZ', 'SE', 'SI', 'SK'], 'color': '#80b1d3'},
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
print('EU CARBON BUDGET – NEUTRALITY YEAR DISTRIBUTIONS')
print('=' * 70)

sp = pd.read_csv(DATA_PATH / 'scenario_parameters.csv')

# Base filter: 1.5°C, 50%, EU-27 countries
base = sp[
    (sp['Warming_scenario'] == WARMING) &
    (sp['Probability_of_reach'] == PROBABILITY) &
    (sp['ISO2'].isin(EU27)) &
    (sp['ISO_Type'] == 'Country')
].copy()

print(f'Base rows (EU-27, 1.5°C, 50%): {len(base)}')

# Build per-scenario DataFrames
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

# NDC Pledges (Territory only)
ndc = base[
    (base['Emissions_scope'] == 'Territory') &
    (base['Budget_distribution_scenario'] == 'NDC Pledges')
].copy()
ndc['Scenario'] = 'NDC Pledges'
print(f'  NDC Pledges: {len(ndc)} countries, '
      f'neutrality year {ndc["Neutrality_year"].min():.0f}–{ndc["Neutrality_year"].max():.0f}')


# ============================================================================
# VISUAL 1 – One histogram per scenario (2×2)
# ============================================================================
print('\n--- Visual 1: Individual scenario distributions (2×2) ---')

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

for ax, (key, label) in zip(axes.flatten(), SCENARIO_LABELS.items()):
    sub = scenario_dfs[key]
    ny = sub['Neutrality_year'].dropna()
    # 5-year bins
    bin_start = int(np.floor(ny.min() / 5) * 5)
    bin_end = int(np.ceil(ny.max() / 5) * 5) + 5
    bins_5y = range(bin_start, bin_end + 1, 5)
    ax.hist(ny, bins=bins_5y,
            color=SCENARIO_COLORS[key], edgecolor='white', alpha=0.85)
    ax.axvline(ny.median(), color='#2c3e50', linestyle='--', linewidth=1.5,
               label=f'Median: {ny.median():.0f}')
    ax.set_title(label, fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Year to neutrality')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0, 0].set_ylabel('Number of EU countries')
axes[1, 0].set_ylabel('Number of EU countries')
fig.suptitle('Distribution of Year to Carbon Neutrality – EU-27 Countries\n'
             f'{WARMING} warming, {PROBABILITY} probability',
             fontsize=14, fontweight='bold', color='#2c3e50')
fig.tight_layout(rect=[0, 0, 1, 0.93])
_save(fig, '1_neutrality_distributions_by_scenario')


# ============================================================================
# VISUAL 2 – All 4 scenarios overlaid
# ============================================================================
print('\n--- Visual 2: All scenarios overlaid ---')

fig, ax = plt.subplots(figsize=(12, 6))

# Determine global bin range
all_years = pd.concat([d['Neutrality_year'] for d in scenario_dfs.values()]).dropna()
bin_edges = range(int(all_years.min()) - 1, int(all_years.max()) + 3)

for key in SCENARIOS:
    ny = scenario_dfs[key]['Neutrality_year'].dropna()
    ax.hist(ny, bins=bin_edges, alpha=0.5, color=SCENARIO_COLORS[key],
            edgecolor='white', label=SCENARIO_LABELS[key])
    ax.axvline(ny.median(), color=SCENARIO_COLORS[key], linestyle='--',
               linewidth=1.5, alpha=0.8)

ax.set_xlabel('Year to neutrality', fontsize=11)
ax.set_ylabel('Number of EU countries', fontsize=11)
ax.set_title('Distribution of Year to Carbon Neutrality – EU-27\n'
             f'{WARMING} warming, {PROBABILITY} probability',
             fontsize=13, fontweight='bold', color='#2c3e50')
ax.legend(fontsize=9, loc='best')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
_save(fig, '2_neutrality_all_scenarios_overlaid')


# ============================================================================
# VISUAL 3 – All scenarios + NDC Pledges
# ============================================================================
print('\n--- Visual 3: All scenarios + NDC Pledges ---')

fig, ax = plt.subplots(figsize=(14, 6))

# Global bin range including NDC
all_years_ndc = pd.concat([all_years, ndc['Neutrality_year'].dropna()])
bin_edges_ndc = range(int(all_years_ndc.min()) - 1, int(all_years_ndc.max()) + 3)

for key in SCENARIOS:
    ny = scenario_dfs[key]['Neutrality_year'].dropna()
    ax.hist(ny, bins=bin_edges_ndc, alpha=0.45, color=SCENARIO_COLORS[key],
            edgecolor='white', label=SCENARIO_LABELS[key])
    ax.axvline(ny.median(), color=SCENARIO_COLORS[key], linestyle='--',
               linewidth=1.5, alpha=0.8)

# NDC
ny_ndc = ndc['Neutrality_year'].dropna()
ax.hist(ny_ndc, bins=bin_edges_ndc, alpha=0.6, color=NDC_COLOR,
        edgecolor='white', label='NDC Pledges')
ax.axvline(ny_ndc.median(), color=NDC_COLOR, linestyle='--',
           linewidth=2, alpha=0.9)

ax.set_xlabel('Year to neutrality', fontsize=11)
ax.set_ylabel('Number of EU countries', fontsize=11)
ax.set_title('Distribution of Year to Carbon Neutrality – EU-27 + NDC Pledges\n'
             f'{WARMING} warming, {PROBABILITY} probability',
             fontsize=13, fontweight='bold', color='#2c3e50')
ax.legend(fontsize=9, loc='best')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
_save(fig, '3_neutrality_with_ndc')


# ============================================================================
# VISUAL 4 – 8 cluster countries + NDC Pledges (dot strip / lollipop)
# ============================================================================
print('\n--- Visual 4: 8 EWBI cluster countries + NDC Pledges ---')

fig, ax = plt.subplots(figsize=(14, 7))

# Collect data for 8 countries across scenarios
y_labels = []
y_pos = []
pos = 0
gap = 0.6  # gap between clusters

for cluster_name, info in CLUSTER_COUNTRIES.items():
    for iso in info['countries']:
        country_name = base.loc[base['ISO2'] == iso, 'Country'].values[0]
        y_labels.append(f'{country_name} ({iso})')
        y_pos.append(pos)

        # Plot each scenario as a dot
        for key in SCENARIOS:
            sub = scenario_dfs[key]
            row = sub[sub['ISO2'] == iso]
            if not row.empty:
                ny = row['Neutrality_year'].values[0]
                ax.scatter(ny, pos, color=SCENARIO_COLORS[key], s=80, zorder=5,
                           edgecolors='white', linewidth=0.5)

        # NDC pledge dot
        ndc_row = ndc[ndc['ISO2'] == iso]
        if not ndc_row.empty:
            ny_ndc_val = ndc_row['Neutrality_year'].values[0]
            ax.scatter(ny_ndc_val, pos, color=NDC_COLOR, s=100, zorder=5,
                       marker='D', edgecolors='#2c3e50', linewidth=0.8)

        pos += 1
    pos += gap  # gap between clusters

ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=10)
ax.set_xlabel('Year to neutrality', fontsize=11)
ax.set_title('Year to Carbon Neutrality – 8 EWBI Cluster Countries + NDC Pledges\n'
             f'{WARMING} warming, {PROBABILITY} probability',
             fontsize=13, fontweight='bold', color='#2c3e50')

# Add cluster background shading
pos = 0
for cluster_name, info in CLUSTER_COUNTRIES.items():
    n = len(info['countries'])
    ax.axhspan(pos - 0.4, pos + n - 0.6, alpha=0.08, color=info['color'])
    # cluster label on the right
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 2050 else 2053,
            pos + (n - 1) / 2, cluster_name.split('–')[0].strip(),
            fontsize=8, color=info['color'], fontweight='bold',
            va='center', ha='left')
    pos += n + gap

ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend_handles = []
for key in SCENARIOS:
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=SCENARIO_COLORS[key],
                                     markersize=9, label=SCENARIO_LABELS[key]))
legend_handles.append(plt.Line2D([0], [0], marker='D', color='w',
                                 markerfacecolor=NDC_COLOR, markeredgecolor='#2c3e50',
                                 markersize=9, label='NDC Pledges'))
ax.legend(handles=legend_handles, fontsize=9, loc='center left',
          bbox_to_anchor=(1.02, 0.5), framealpha=0.9, edgecolor='#ccc')

fig.tight_layout(rect=[0, 0, 0.82, 1])
_save(fig, '4_neutrality_cluster_countries_ndc')


# ============================================================================
# VISUAL 5 \u2013 All EU-27 countries grouped by cluster + NDC Pledges
# ============================================================================
print('\n--- Visual 5: All EU-27 countries grouped by cluster + NDC Pledges ---')

total_countries = sum(len(v['countries']) for v in ALL_CLUSTERS.values())
fig_height = max(12, total_countries * 0.45 + 3)
fig, ax = plt.subplots(figsize=(14, fig_height))

y_labels = []
y_pos = []
pos = 0
gap = 0.8

for cluster_name, info in ALL_CLUSTERS.items():
    for iso in info['countries']:
        match = base.loc[base['ISO2'] == iso, 'Country']
        country_name = match.values[0] if len(match) > 0 else iso
        y_labels.append(f'{country_name} ({iso})')
        y_pos.append(pos)

        for key in SCENARIOS:
            sub = scenario_dfs[key]
            row = sub[sub['ISO2'] == iso]
            if not row.empty:
                ny = row['Neutrality_year'].values[0]
                ax.scatter(ny, pos, color=SCENARIO_COLORS[key], s=70, zorder=5,
                           edgecolors='white', linewidth=0.5)

        ndc_row = ndc[ndc['ISO2'] == iso]
        if not ndc_row.empty:
            ny_ndc_val = ndc_row['Neutrality_year'].values[0]
            ax.scatter(ny_ndc_val, pos, color=NDC_COLOR, s=90, zorder=5,
                       marker='D', edgecolors='#2c3e50', linewidth=0.8)

        pos += 1
    pos += gap

ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel('Year to neutrality', fontsize=11)
ax.set_title('Year to Carbon Neutrality \u2013 All EU-27 Countries by EWBI Cluster + NDC Pledges\n'
             f'{WARMING} warming, {PROBABILITY} probability',
             fontsize=13, fontweight='bold', color='#2c3e50')

# Cluster background shading
pos = 0
for cluster_name, info in ALL_CLUSTERS.items():
    n = len(info['countries'])
    ax.axhspan(pos - 0.4, pos + n - 0.6, alpha=0.08, color=info['color'])
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 2050 else 2053,
            pos + (n - 1) / 2, cluster_name.split('\u2013')[0].strip(),
            fontsize=8, color=info['color'], fontweight='bold',
            va='center', ha='left')
    pos += n + gap

ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_handles = []
for key in SCENARIOS:
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=SCENARIO_COLORS[key],
                                     markersize=9, label=SCENARIO_LABELS[key]))
legend_handles.append(plt.Line2D([0], [0], marker='D', color='w',
                                 markerfacecolor=NDC_COLOR, markeredgecolor='#2c3e50',
                                 markersize=9, label='NDC Pledges'))
ax.legend(handles=legend_handles, fontsize=9, loc='center left',
          bbox_to_anchor=(1.02, 0.5), framealpha=0.9, edgecolor='#ccc')

fig.tight_layout(rect=[0, 0, 0.82, 1])
_save(fig, '5_neutrality_all_eu27_by_cluster')


# ============================================================================
# EXCEL EXPORT
# ============================================================================
print('\n--- Excel export ---')

rows = []
for key in SCENARIOS:
    sub = scenario_dfs[key]
    for _, r in sub.iterrows():
        rows.append({
            'ISO2': r['ISO2'],
            'Country': r['Country'],
            'Emissions_scope': key[0],
            'Budget_allocation': key[1],
            'Scenario_label': SCENARIO_LABELS[key],
            'Neutrality_year': r['Neutrality_year'],
            'Country_carbon_budget': r.get('Country_carbon_budget', None),
            'Latest_annual_CO2_Mt': r.get('Latest_annual_CO2_emissions_Mt', None),
            'Latest_emissions_per_capita_t': r.get('Latest_emissions_per_capita_t', None),
        })

# Add NDC
for _, r in ndc.iterrows():
    rows.append({
        'ISO2': r['ISO2'],
        'Country': r['Country'],
        'Emissions_scope': 'Territory',
        'Budget_allocation': 'NDC Pledges',
        'Scenario_label': 'NDC Pledges',
        'Neutrality_year': r['Neutrality_year'],
        'Country_carbon_budget': r.get('Country_carbon_budget', None),
        'Latest_annual_CO2_Mt': r.get('Latest_annual_CO2_emissions_Mt', None),
        'Latest_emissions_per_capita_t': r.get('Latest_emissions_per_capita_t', None),
    })

df_export = pd.DataFrame(rows)
xlsx_path = OUTPUT_DIR / 'EU_neutrality_year_scenarios.xlsx'
df_export.to_excel(xlsx_path, index=False, sheet_name='Neutrality Years')
print(f'  Saved {xlsx_path.name}')


# ============================================================================
# MAP HELPERS
# ============================================================================
def _load_nuts0_and_background():
    """Load NUTS level-0 (countries) and European background geometries."""
    nuts = gpd.read_file(_NUTS_GPKG)
    nuts0 = nuts[nuts['LEVL_CODE'] == 0].copy()
    if nuts0.crs != 'EPSG:3035':
        nuts0 = nuts0.to_crs(epsg=3035)
    world = gpd.read_file(_WORLD_SHP)
    bg = world[(world['CONTINENT'] == 'Europe') | (world['ISO_A2'] == 'TR')].copy()
    bg = bg.to_crs(epsg=3035)
    return nuts0, bg


def _plot_neutrality_map(ax, gdf, col, cmap, norm, title, eu27_set, bg):
    """Draw a single neutrality-year choropleth on ax."""
    # Background: non-study European countries
    non_study = bg[~bg['ISO_A2'].isin(eu27_set)]
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
    # Crop to EU extent
    ax.set_xlim(2.5e6, 6.5e6)
    ax.set_ylim(1.3e6, 5.5e6)


# ============================================================================
# VISUAL 6 – Maps: all 4 scenarios side by side (shared legend)
# ============================================================================
print('\n--- Visual 6: Maps – 4 scenarios side by side (shared legend) ---')

nuts0, bg = _load_nuts0_and_background()
eu27_set = set(EU27)

# Merge neutrality years with geometries
map_data = {}
for key in SCENARIOS:
    sub = scenario_dfs[key][['ISO2', 'Neutrality_year']].copy()
    merged = nuts0.merge(sub, left_on='CNTR_CODE', right_on='ISO2', how='left')
    # Keep only EU-27
    merged = merged[merged['CNTR_CODE'].isin(eu27_set)]
    map_data[key] = merged

# Shared color scale across all 4 scenarios
all_ny = pd.concat([d['Neutrality_year'] for d in map_data.values()]).dropna()
vmin, vmax = all_ny.min(), all_ny.max()
cmap = plt.cm.RdYlGn  # red = later (fewer years left), green = earlier
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
# Layout: rows = scope (Territorial top, Consumption bottom)
#         cols = allocation (Responsibility left, Capability right)
layout = [
    [('Territory', 'Responsibility'), ('Territory', 'Capability')],
    [('Consumption', 'Responsibility'), ('Consumption', 'Capability')],
]
for row_idx, row_keys in enumerate(layout):
    for col_idx, key in enumerate(row_keys):
        ax = axes[row_idx, col_idx]
        _plot_neutrality_map(ax, map_data[key], 'Neutrality_year', cmap, norm,
                             SCENARIO_LABELS[key], eu27_set, bg)

# Shared colorbar \u2013 placed below all 4 maps
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Year to carbon neutrality', fontsize=11)

fig.suptitle(f'Year to Carbon Neutrality \u2013 EU-27\n{WARMING} warming, {PROBABILITY} probability',
             fontsize=14, fontweight='bold', color='#2c3e50', y=0.98)
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
_save(fig, '6_map_all_scenarios_shared_legend')


# ============================================================================
# VISUAL 7 – Individual maps (one per scenario)
# ============================================================================
print('\n--- Visual 7: Individual maps per scenario ---')

for key in SCENARIOS:
    fig, ax = plt.subplots(figsize=(8, 8))
    sub_data = map_data[key]
    ny = sub_data['Neutrality_year'].dropna()
    # Per-scenario color scale
    norm_i = mcolors.Normalize(vmin=ny.min(), vmax=ny.max())
    _plot_neutrality_map(ax, sub_data, 'Neutrality_year', cmap, norm_i,
                         SCENARIO_LABELS[key], eu27_set, bg)
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_i)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04,
                        pad=0.05, shrink=0.7)
    cbar.set_label('Year to carbon neutrality', fontsize=10)
    fig.suptitle(f'{SCENARIO_LABELS[key]}\n{WARMING} warming, {PROBABILITY} probability',
                 fontsize=13, fontweight='bold', color='#2c3e50')
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    label = SCENARIO_LABELS[key].lower().replace(' / ', '_').replace(' ', '_')
    _save(fig, f'7_map_{label}')


# ============================================================================
# VISUAL 8 – Per-cluster quintiles & median for each scenario
# ============================================================================
print('\n--- Visual 8: Cluster quintiles & median by scenario ---')

cluster_names_short = []
cluster_stats = {}   # {cluster_short: {scenario_key: {stat: value}}}
for cname, info in ALL_CLUSTERS.items():
    short = cname.split('–')[0].strip()   # e.g. "Cluster 0"
    cluster_names_short.append(short)
    cluster_stats[short] = {}
    for key in SCENARIOS:
        vals = scenario_dfs[key].loc[
            scenario_dfs[key]['ISO2'].isin(info['countries']), 'Neutrality_year'
        ].dropna()
        if len(vals) > 0:
            cluster_stats[short][key] = {
                'min': vals.min(),
                'q1':  np.percentile(vals, 20),
                'q2':  np.percentile(vals, 40),
                'median': vals.median(),
                'q3':  np.percentile(vals, 60),
                'q4':  np.percentile(vals, 80),
                'max': vals.max(),
            }

n_clusters = len(cluster_names_short)
n_scenarios = len(SCENARIOS)
fig, ax = plt.subplots(figsize=(14, 7))

group_width = 0.7
bar_width = group_width / n_scenarios
x_base = np.arange(n_clusters)

for i, key in enumerate(SCENARIOS):
    x = x_base + (i - (n_scenarios - 1) / 2) * bar_width
    color = SCENARIO_COLORS[key]
    for j, cshort in enumerate(cluster_names_short):
        st = cluster_stats[cshort].get(key)
        if st is None:
            continue
        # Vertical line min→max (whisker)
        ax.plot([x[j], x[j]], [st['min'], st['max']],
                color=color, linewidth=1.5, zorder=3)
        # Whisker caps
        cap_w = bar_width * 0.3
        ax.plot([x[j] - cap_w, x[j] + cap_w], [st['min'], st['min']],
                color=color, linewidth=1.5, zorder=3)
        ax.plot([x[j] - cap_w, x[j] + cap_w], [st['max'], st['max']],
                color=color, linewidth=1.5, zorder=3)
        # Box Q1→Q4 (20th–80th pctile)
        box_h = st['q4'] - st['q1']
        rect = plt.Rectangle((x[j] - bar_width * 0.4, st['q1']),
                              bar_width * 0.8, box_h,
                              facecolor=color, alpha=0.35,
                              edgecolor=color, linewidth=1.2, zorder=4)
        ax.add_patch(rect)
        # Q2–Q3 stripe (40th–60th pctile) – darker fill
        inner_h = st['q3'] - st['q2']
        rect_inner = plt.Rectangle((x[j] - bar_width * 0.4, st['q2']),
                                   bar_width * 0.8, inner_h,
                                   facecolor=color, alpha=0.55,
                                   edgecolor=color, linewidth=0.8, zorder=5)
        ax.add_patch(rect_inner)
        # Median line
        ax.plot([x[j] - bar_width * 0.4, x[j] + bar_width * 0.4],
                [st['median'], st['median']],
                color='#2c3e50', linewidth=2, zorder=6)

ax.set_xticks(x_base)
ax.set_xticklabels([f'{s}\n({len(ALL_CLUSTERS[list(ALL_CLUSTERS.keys())[i]]["countries"])} countries)'
                     for i, s in enumerate(cluster_names_short)], fontsize=10)
ax.set_ylabel('Year to carbon neutrality', fontsize=11)
ax.set_title('Carbon Neutrality Year – Quintile Distribution by EWBI Cluster & Scenario\n'
             f'{WARMING} warming, {PROBABILITY} probability',
             fontsize=13, fontweight='bold', color='#2c3e50')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_handles = []
for key in SCENARIOS:
    legend_handles.append(Patch(facecolor=SCENARIO_COLORS[key], alpha=0.5,
                                edgecolor=SCENARIO_COLORS[key],
                                label=SCENARIO_LABELS[key]))
legend_handles.append(plt.Line2D([0], [0], color='#2c3e50', linewidth=2,
                                 label='Median'))
ax.legend(handles=legend_handles, fontsize=9, loc='upper right', framealpha=0.9,
          edgecolor='#ccc')

fig.tight_layout()
_save(fig, '8_cluster_quintiles_by_scenario')


print(f'\nDone – all outputs in: {OUTPUT_DIR}')
