"""
0_clustering.py - Country clustering visuals for two reports:
  - rep_eu   : EU-27 countries only
  - rep_ewbi : EU-27 + EFTA countries (CH, NO, IS)

Each visual is exported as PNG + SVG + Excel, prefixed by the report name.

Visuals generated:
  1. ewbi_cluster_map_mpl            - Choropleth map of cluster assignments
  2. ewbi_cluster_priority_radar     - Radar of EU priorities by cluster
  3. ewbi_cluster_radar              - Radar of cluster profiles (normalized features)
  4. ewbi_country_map_mpl            - Choropleth map of EWBI values
  5. ewbi_performance_vs_ewbi_clusters - Scatter: Performance Score vs EWBI
  6. ewbi_vs_income_by_cluster       - 4-panel EWBI vs Income (last year)
  7. ewbi_vs_income_cluster_priority_grid - Priority × Cluster grid
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import geopandas as gpd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
report_dir = os.path.abspath(os.path.join(current_dir, '..'))
well_being_dir = os.path.abspath(os.path.join(report_dir, '..', '..'))
data_path = os.path.join(well_being_dir, 'output', 'ewbi_master_aggregated.csv')

output_base = os.path.join(report_dir, 'outputs', 'graphs', 'EWBI', 'Clustering')
os.makedirs(output_base, exist_ok=True)

_NUTS_GPKG = os.path.join(
    report_dir, '..', '1_switzerland_vs_eu27_housing_energy',
    'external_data', '0_shapefile', 'NUTS_RG_10M_2024_3035.gpkg',
)
_WORLD_SHP = os.path.join(
    report_dir, '..', '1_switzerland_vs_eu27_housing_energy',
    'external_data', '0_shapefile', 'ne_50m_admin_0_countries',
    'ne_50m_admin_0_countries.shp',
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COUNTRY_NAME_MAP = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CH': 'Switzerland',
    'CY': 'Cyprus', 'CZ': 'Czechia', 'DE': 'Germany', 'DK': 'Denmark',
    'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland',
    'FR': 'France', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland',
    'IS': 'Iceland', 'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg',
    'LV': 'Latvia', 'MT': 'Malta', 'NL': 'Netherlands', 'NO': 'Norway',
    'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia',
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia', 'UK': 'United Kingdom',
}

EU27_CODES = {
    'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES',
    'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT',
    'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK',
}

EFTA_CODES = {'CH', 'NO', 'IS'}

ISO2_TO_ISO3 = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'CH': 'CHE', 'CY': 'CYP',
    'CZ': 'CZE', 'DE': 'DEU', 'DK': 'DNK', 'EE': 'EST', 'EL': 'GRC',
    'ES': 'ESP', 'FI': 'FIN', 'FR': 'FRA', 'HR': 'HRV', 'HU': 'HUN',
    'IE': 'IRL', 'IS': 'ISL', 'IT': 'ITA', 'LT': 'LTU', 'LU': 'LUX',
    'LV': 'LVA', 'MT': 'MLT', 'NL': 'NLD', 'NO': 'NOR', 'PL': 'POL',
    'PT': 'PRT', 'RO': 'ROU', 'RS': 'SRB', 'SE': 'SWE', 'SI': 'SVN',
    'SK': 'SVK', 'UK': 'GBR',
}

CLUSTER_NAMES = {
    0: 'Low performer / Low EWBI',
    1: 'Low performer / High EWBI',
    2: 'High performer / Low EWBI',
    3: 'High performer / High EWBI',
}
CLUSTER_COLORS = ['#fb8072', '#fdb462', '#8dd3c7', '#80b1d3']

EU_PRIORITY_DISPLAY_MAP = {
    'Health and Animal Welfare': 'Health',
    'Intergenerational Fairness, Youth, Culture and Sport': 'Education',
    'Social Rights and Skills, Quality Jobs and Preparedness': 'Quality of Jobs',
}

METHOD5_STEP_EUR = 5000
PERF_CUT = 0.006972  # Mean of Croatia (0.009043) and Italy (0.004900) performance scores
EWBI_CUT = 0.7

# ---------------------------------------------------------------------------
# Report configurations
# ---------------------------------------------------------------------------
REPORT_CONFIGS = {
    'rep_eu': {
        'prefix': 'rep_eu',
        'title_suffix': '(EU-27)',
        'country_filter': EU27_CODES,
        'highlight_countries': {'FR', 'ES', 'BE', 'NL', 'LT', 'HR', 'DE', 'SE'},
    },
    'rep_ewbi': {
        'prefix': 'rep_ewbi',
        'title_suffix': '(EU-27 + EFTA + UK + Serbia)',
        'country_filter': EU27_CODES | EFTA_CODES | {'UK', 'RS'},
        'highlight_countries': {'FR', 'ES', 'BE', 'NL', 'LT', 'HR', 'DE', 'SE', 'CH', 'NO', 'IS', 'UK', 'RS'},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _country_name(code):
    """Return full country name for a code."""
    return COUNTRY_NAME_MAP.get(code, code)


def _priority_label(priority_name):
    """Return display label for an EU priority name."""
    return EU_PRIORITY_DISPLAY_MAP.get(priority_name, priority_name)


def _save_fig(fig, path, dpi=150):
    """Save figure as PNG and SVG."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    svg_path = os.path.splitext(path)[0] + '.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {path}")
    print(f"  Saved: {svg_path}")


def _save_excel(data, path, sheet_name='Data'):
    """Save a DataFrame or dict of DataFrames to Excel."""
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        if isinstance(data, dict):
            for name, df in data.items():
                df.to_excel(writer, sheet_name=name, index=False)
        else:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  Saved: {path}")


def _cluster_country_text(result_df):
    """Build a text block listing countries per cluster, for legend annotation."""
    lines = []
    for cl in sorted(result_df['Cluster'].unique()):
        cdf = result_df[result_df['Cluster'] == cl].sort_values('Country_Name')
        names = cdf['Country_Name'].tolist()
        lines.append(f"Cluster {cl} – {CLUSTER_NAMES[cl]}:")
        for n in names:
            lines.append(f"  {n}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_prepare_data():
    """Load data, compute Method-5 features and assign 4 clusters (all countries)."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    median_income_path = os.path.join(report_dir, 'outputs', 'data', 'median_income_by_decile.csv')
    median_income_df = pd.read_csv(median_income_path)

    ewbi_decile = df[
        (df['Level'] == 1) &
        (df['Decile'] != 'All Deciles') &
        (~df['Country'].isin(['EU-27', 'All Countries']))
    ].copy()
    ewbi_decile['Decile'] = pd.to_numeric(ewbi_decile['Decile'], errors='coerce').astype('Int64')
    ewbi_decile['Year'] = pd.to_numeric(ewbi_decile['Year'], errors='coerce').astype('Int64')
    ewbi_decile['Value'] = pd.to_numeric(ewbi_decile['Value'], errors='coerce')
    ewbi_decile = ewbi_decile.dropna(subset=['Country', 'Year', 'Decile', 'Value'])

    ewbi_all = df[
        (df['Level'] == 1) &
        (df['Decile'] == 'All Deciles') &
        (~df['Country'].isin(['EU-27', 'All Countries']))
    ].copy()
    ewbi_all['Year'] = pd.to_numeric(ewbi_all['Year'], errors='coerce').astype('Int64')
    ewbi_all['Value'] = pd.to_numeric(ewbi_all['Value'], errors='coerce')
    ewbi_all = ewbi_all.dropna(subset=['Country', 'Year', 'Value'])

    median_income_df['year'] = pd.to_numeric(median_income_df['year'], errors='coerce').astype('Int64')
    median_income_df['decile'] = pd.to_numeric(median_income_df['decile'], errors='coerce').astype('Int64')
    median_income_df['median_equi_disp_inc'] = pd.to_numeric(
        median_income_df['median_equi_disp_inc'], errors='coerce'
    )
    median_income_df = median_income_df.dropna(subset=['country', 'year', 'decile', 'median_equi_disp_inc'])

    merged = ewbi_decile.merge(
        median_income_df,
        left_on=['Country', 'Year', 'Decile'],
        right_on=['country', 'year', 'decile'],
        how='inner',
    )
    merged = merged.dropna(subset=['median_equi_disp_inc', 'Value'])

    last_year = merged.groupby('Country')['Year'].max().reset_index()
    last_year.columns = ['Country', 'Last_Year']
    points = merged.merge(last_year, on='Country')
    points = points[points['Year'] == points['Last_Year']].copy()

    points['income_bin_center'] = (
        np.round(points['median_equi_disp_inc'] / METHOD5_STEP_EUR) * METHOD5_STEP_EUR
    )
    benchmark = (
        points.groupby('income_bin_center', as_index=False)['Value']
        .mean()
        .sort_values('income_bin_center')
        .rename(columns={'Value': 'benchmark_ewbi'})
    )
    x = benchmark['income_bin_center'].values.astype(float)
    y = benchmark['benchmark_ewbi'].values.astype(float)
    points['ewbi_expected'] = np.interp(points['median_equi_disp_inc'].values.astype(float), x, y)
    points['ewbi_residual'] = points['Value'] - points['ewbi_expected']

    decile_resid = (
        points.groupby(['Country', 'Decile'], as_index=False)
        .agg(decile_residual=('ewbi_residual', 'mean'))
    )
    perf_score = decile_resid.groupby('Country', as_index=False).agg(
        Performance_Score=('decile_residual', 'mean'),
        n_deciles=('decile_residual', 'size'),
    )

    income_last = points.groupby('Country', as_index=False).agg(
        Mean_Income_Last=('median_equi_disp_inc', 'mean'),
    )
    income_d1 = points[points['Decile'] == 1][['Country', 'median_equi_disp_inc']].rename(
        columns={'median_equi_disp_inc': 'Income_D1'}
    )
    income_d10 = points[points['Decile'] == 10][['Country', 'median_equi_disp_inc']].rename(
        columns={'median_equi_disp_inc': 'Income_D10'}
    )
    income_inter = income_d10.merge(income_d1, on='Country', how='inner')
    income_inter['Income_Interdecile_Last'] = income_inter['Income_D10'] / income_inter['Income_D1']
    income_inter = income_inter.replace([np.inf, -np.inf], np.nan).dropna(subset=['Income_Interdecile_Last'])

    ewbi_last = ewbi_all.sort_values('Year').groupby('Country').tail(1)[['Country', 'Year', 'Value']].copy()
    ewbi_last.columns = ['Country', 'Last_Year_all', 'EWBI_Last']

    ewbi_first = ewbi_all.sort_values('Year').groupby('Country').head(1)[['Country', 'Year', 'Value']].copy()
    ewbi_first.columns = ['Country', 'First_Year', 'EWBI_First']

    d1 = ewbi_decile[ewbi_decile['Decile'] == 1][['Country', 'Year', 'Value']].rename(columns={'Value': 'D1'})
    d10 = ewbi_decile[ewbi_decile['Decile'] == 10][['Country', 'Year', 'Value']].rename(columns={'Value': 'D10'})
    inter = d10.merge(d1, on=['Country', 'Year'], how='inner')
    inter['Interdecile'] = inter['D10'] / inter['D1']
    inter = inter.replace([np.inf, -np.inf], np.nan).dropna(subset=['Interdecile'])

    inter_last = inter.sort_values('Year').groupby('Country').tail(1)[['Country', 'Interdecile']].copy()
    inter_last.columns = ['Country', 'Interdecile_Last']
    inter_first = inter.sort_values('Year').groupby('Country').head(1)[['Country', 'Interdecile']].copy()
    inter_first.columns = ['Country', 'Interdecile_First']

    features_df = (
        ewbi_last
        .merge(ewbi_first, on='Country', how='inner')
        .merge(inter_last, on='Country', how='left')
        .merge(inter_first, on='Country', how='left')
        .merge(income_last, on='Country', how='left')
        .merge(income_inter[['Country', 'Income_Interdecile_Last']], on='Country', how='left')
        .merge(perf_score, on='Country', how='left')
    )
    features_df['Country_Name'] = features_df['Country'].map(COUNTRY_NAME_MAP).fillna(features_df['Country'])
    features_df['Last_Year'] = features_df['Last_Year_all'].astype(int)
    features_df.drop(columns=['Last_Year_all'], inplace=True)

    # Country-level median income (from compute_median_income_by_decile.py)
    country_median_path = os.path.join(report_dir, 'outputs', 'data', 'median_income_by_country.csv')
    if os.path.exists(country_median_path):
        cm_df = pd.read_csv(country_median_path)
        cm_df['year'] = pd.to_numeric(cm_df['year'], errors='coerce').astype('Int64')
        cm_df['median_equi_disp_inc'] = pd.to_numeric(cm_df['median_equi_disp_inc'], errors='coerce')
        cm_last = cm_df.sort_values('year').groupby('country').tail(1)[['country', 'median_equi_disp_inc']]
        cm_last.columns = ['Country', 'Median_Income']
        features_df = features_df.merge(cm_last, on='Country', how='left')
    else:
        print(f"  WARNING: {country_median_path} not found – Median_Income will be NaN")
        features_df['Median_Income'] = np.nan

    n_years = (features_df['Last_Year'] - features_df['First_Year']).replace(0, np.nan)
    features_df['Annual_Growth'] = ((features_df['EWBI_Last'] / features_df['EWBI_First']) ** (1 / n_years) - 1)
    features_df['Annual_Growth'] = features_df['Annual_Growth'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    features_df = features_df.dropna(
        subset=['EWBI_Last', 'EWBI_First', 'Interdecile_Last', 'Mean_Income_Last',
                'Income_Interdecile_Last', 'Performance_Score']
    ).copy()

    # Clustering thresholds
    features_df['Performance_Group'] = np.where(
        features_df['Performance_Score'] >= PERF_CUT, 'High performer', 'Low performer'
    )
    features_df['EWBI_Group'] = np.where(
        features_df['EWBI_Last'] >= EWBI_CUT, 'High EWBI', 'Low EWBI'
    )

    cluster_map = {
        ('Low performer', 'Low EWBI'): 0,
        ('Low performer', 'High EWBI'): 1,
        ('High performer', 'Low EWBI'): 2,
        ('High performer', 'High EWBI'): 3,
    }
    features_df['Cluster'] = features_df.apply(
        lambda r: cluster_map[(r['Performance_Group'], r['EWBI_Group'])], axis=1
    ).astype(int)

    print(f"Performance threshold: {PERF_CUT}")
    print(f"EWBI threshold: {EWBI_CUT}")
    print(f"Countries with complete data: {len(features_df)}")
    return features_df


def _load_ewbi_income_merged():
    """Load and merge EWBI per-decile with median income. Returns all years."""
    median_income_path = os.path.join(report_dir, 'outputs', 'data', 'median_income_by_decile.csv')
    if not os.path.exists(median_income_path):
        return None
    median_income_df = pd.read_csv(median_income_path)
    ewbi_df = pd.read_csv(data_path, low_memory=False)

    ewbi_decile = ewbi_df[
        (ewbi_df['Level'] == 1) &
        (ewbi_df['Decile'] != 'All Deciles') &
        (~ewbi_df['Country'].isin(['EU-27', 'All Countries']))
    ].copy()
    ewbi_decile['Decile'] = pd.to_numeric(ewbi_decile['Decile'], errors='coerce').astype('Int64')
    ewbi_decile = ewbi_decile.dropna(subset=['Decile', 'Value'])

    merged = ewbi_decile.merge(
        median_income_df,
        left_on=['Country', 'Year', 'Decile'],
        right_on=['country', 'year', 'decile'],
        how='inner',
    )
    merged = merged.dropna(subset=['Value', 'median_equi_disp_inc'])
    return merged if not merged.empty else None


def _load_priority_income_merged():
    """Load and merge Level-2 EU priorities by decile with median income."""
    median_income_path = os.path.join(report_dir, 'outputs', 'data', 'median_income_by_decile.csv')
    if not os.path.exists(median_income_path):
        return None
    median_income_df = pd.read_csv(median_income_path)
    ewbi_df = pd.read_csv(data_path, low_memory=False)

    priorities_decile = ewbi_df[
        (ewbi_df['Level'] == 2) &
        (ewbi_df['Decile'] != 'All Deciles') &
        (~ewbi_df['Country'].isin(['EU-27', 'All Countries']))
    ].copy()
    priorities_decile['Decile'] = pd.to_numeric(priorities_decile['Decile'], errors='coerce').astype('Int64')
    priorities_decile = priorities_decile.dropna(subset=['Decile', 'Value', 'EU priority'])

    merged = priorities_decile.merge(
        median_income_df,
        left_on=['Country', 'Year', 'Decile'],
        right_on=['country', 'year', 'decile'],
        how='inner',
    )
    merged = merged.dropna(subset=['Value', 'median_equi_disp_inc', 'EU priority'])
    return merged if not merged.empty else None


_DATA_CODE_TO_ISO_A2 = {'UK': 'GB', 'EL': 'GR'}


def _load_nuts0_and_background():
    """Load NUTS level-0 (countries) and background world geometries.

    Countries missing from NUTS (e.g. UK post-Brexit) are supplemented with
    geometries from the Natural Earth world shapefile.
    """
    nuts = gpd.read_file(_NUTS_GPKG)
    nuts0 = nuts[nuts['LEVL_CODE'] == 0].copy()
    if nuts0.crs != 'EPSG:3035':
        nuts0 = nuts0.to_crs(epsg=3035)
    world = gpd.read_file(_WORLD_SHP)
    bg = world[(world['CONTINENT'] == 'Europe') | (world['ISO_A2'] == 'TR')].copy()
    bg = bg.to_crs(epsg=3035)

    # Supplement NUTS0 with world-shapefile geometries for missing countries
    nuts_ids = set(nuts0['NUTS_ID'].unique())
    all_data_codes = set(COUNTRY_NAME_MAP.keys())
    missing = all_data_codes - nuts_ids
    if missing:
        world_3035 = world.to_crs(epsg=3035)
        for code in missing:
            iso_a2 = _DATA_CODE_TO_ISO_A2.get(code, code)
            match = world_3035[world_3035['ISO_A2'] == iso_a2]
            if match.empty:
                continue
            row = match.iloc[[0]].copy()
            row['NUTS_ID'] = code
            nuts0 = pd.concat([nuts0, row[['NUTS_ID', 'geometry']]], ignore_index=True)

    return nuts0, bg


def _filter_df(features_df, country_set):
    """Filter features_df to only the countries in country_set."""
    return features_df[features_df['Country'].isin(country_set)].copy()


# ===================================================================
# VISUAL 1: Cluster map (matplotlib)
# ===================================================================
def plot_cluster_map(result_df, cfg, out_dir):
    """Choropleth map of cluster assignments."""
    map_df = result_df.copy()
    nuts0, bg = _load_nuts0_and_background()
    merged_geo = nuts0.merge(map_df[['Country', 'Cluster']], left_on='NUTS_ID', right_on='Country', how='left')
    study_ids = set(map_df['Country'].tolist())

    fig, ax = plt.subplots(figsize=(12, 10))
    bg.plot(ax=ax, color='white', edgecolor='black', linewidth=0.3, alpha=0.35, hatch='///')

    no_data = merged_geo[merged_geo['Cluster'].isna() & merged_geo['NUTS_ID'].isin(study_ids)]
    if not no_data.empty:
        no_data.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.3)

    for cl in sorted(CLUSTER_NAMES.keys()):
        cl_geo = merged_geo[merged_geo['Cluster'] == cl]
        if not cl_geo.empty:
            cl_geo.plot(ax=ax, color=CLUSTER_COLORS[cl], edgecolor='black', linewidth=0.4)

    legend_handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CLUSTER_COLORS[cl],
               markersize=10, label=f'Cluster {cl} – {CLUSTER_NAMES[cl]}')
        for cl in sorted(CLUSTER_NAMES.keys())
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=9, frameon=True)
    ax.set_xlim(2.5e6, 6.5e6)
    ax.set_ylim(1.3e6, 5.5e6)
    ax.set_axis_off()
    ax.set_title(f'EWBI Cluster Map {cfg["title_suffix"]}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_cluster_map_mpl.png')
    _save_fig(fig, path, dpi=150)
    plt.close(fig)

    # Excel
    excel_df = result_df[['Country', 'Country_Name', 'Cluster', 'EWBI_Last', 'Performance_Score']].copy()
    excel_df['Cluster_Name'] = excel_df['Cluster'].map(CLUSTER_NAMES)
    excel_df = excel_df.sort_values(['Cluster', 'Country_Name'])
    _save_excel(excel_df, os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_cluster_map_mpl.xlsx'))


# ===================================================================
# VISUAL 2: Priority radar
# ===================================================================
def plot_priority_radar(result_df, cfg, out_dir):
    """Radar chart of mean EWBI per cluster across Level-2 EU priorities."""
    ewbi_df = pd.read_csv(data_path, low_memory=False)

    priorities_all = ewbi_df[
        (ewbi_df['Level'] == 2) &
        (ewbi_df['Decile'] == 'All Deciles') &
        (ewbi_df['Country'].isin(result_df['Country'].tolist()))
    ].copy()
    priorities_all['Year'] = pd.to_numeric(priorities_all['Year'], errors='coerce').astype('Int64')
    priorities_all['Value'] = pd.to_numeric(priorities_all['Value'], errors='coerce')
    priorities_all = priorities_all.dropna(subset=['Country', 'Year', 'Value', 'EU priority'])

    last_year_cp = priorities_all.groupby(['Country', 'EU priority'])['Year'].max().reset_index()
    last_year_cp.columns = ['Country', 'EU priority', 'last_year']
    plot_data = priorities_all.merge(last_year_cp, on=['Country', 'EU priority'], how='inner')
    plot_data = plot_data[plot_data['Year'] == plot_data['last_year']].copy()

    cluster_map = dict(zip(result_df['Country'], result_df['Cluster']))
    plot_data['Cluster'] = plot_data['Country'].map(cluster_map)
    plot_data = plot_data.dropna(subset=['Cluster']).copy()
    plot_data['Cluster'] = plot_data['Cluster'].astype(int)

    if plot_data.empty:
        print("  WARNING: No data for priority radar")
        return

    priorities = sorted(plot_data['EU priority'].unique().tolist())
    priority_labels = [_priority_label(p) for p in priorities]
    priority_means = plot_data.groupby(['Cluster', 'EU priority'], as_index=False)['Value'].mean()

    wide = priority_means.pivot(index='Cluster', columns='EU priority', values='Value')
    wide = wide.reindex(index=sorted(CLUSTER_NAMES.keys()), columns=priorities)

    raw_min = float(np.nanmin(wide.values))
    raw_max = float(np.nanmax(wide.values))
    radial_min = max(0.0, raw_min - 0.03)
    radial_max = min(1.0, raw_max + 0.03)

    angles = np.linspace(0, 2 * np.pi, len(priorities), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))
    for cl in sorted(CLUSTER_NAMES.keys()):
        if cl not in wide.index:
            continue
        values = wide.loc[cl, priorities].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', color=CLUSTER_COLORS[cl], linewidth=2,
                label=f'Cluster {cl} – {CLUSTER_NAMES[cl]}')
        ax.fill(angles, values, alpha=0.15, color=CLUSTER_COLORS[cl])

    ax.set_thetagrids(np.degrees(angles[:-1]), priority_labels, fontsize=10)
    ax.set_ylim(radial_min, radial_max)
    ax.set_title(f'EU Priorities Radar by Cluster {cfg["title_suffix"]}',
                 fontsize=14, fontweight='bold', pad=20)

    # Country list annotation on the right
    country_text = _cluster_country_text(result_df)
    fig.text(0.98, 0.5, country_text, fontsize=7.5, fontfamily='monospace',
             va='center', ha='left', transform=fig.transFigure,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7', edgecolor='#ccc', alpha=0.9))

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.subplots_adjust(right=0.72)

    path = os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_cluster_priority_radar.png')
    _save_fig(fig, path, dpi=160)
    plt.close(fig)

    # Excel – export the cluster-level mean values displayed on the radar
    excel_means = priority_means.copy()
    excel_means['Cluster_Name'] = excel_means['Cluster'].map(CLUSTER_NAMES)
    excel_means['EU_Priority_Label'] = excel_means['EU priority'].map(_priority_label)
    excel_means = excel_means.rename(columns={'Value': 'Mean_Value'})
    excel_means = excel_means[['Cluster', 'Cluster_Name', 'EU priority', 'EU_Priority_Label', 'Mean_Value']]
    _save_excel(
        excel_means.sort_values(['Cluster', 'EU priority']),
        os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_cluster_priority_radar.xlsx'),
    )


# ===================================================================
# VISUAL 3: Cluster profile radar
# ===================================================================
def plot_cluster_radar(result_df, cfg, out_dir):
    """Radar chart showing mean feature values per cluster (normalized)."""
    feature_cols = [
        'EWBI_Last', 'Annual_Growth', 'Interdecile_Last',
        'Interdecile_First', 'Median_Income', 'Income_Interdecile_Last',
    ]
    display_labels = [
        'EWBI\n(Last Year)', 'Annual\nGrowth', 'EWBI Interdecile\n(Last Year)',
        'EWBI Interdecile\n(First Year)', 'Median Income', 'Income Interdecile\n(Last Year)',
    ]

    df_norm = result_df.copy()
    for col in feature_cols:
        cmin, cmax = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - cmin) / max(cmax - cmin, 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))

    for cl in sorted(result_df['Cluster'].unique()):
        cluster_data = df_norm[df_norm['Cluster'] == cl][feature_cols]
        values = cluster_data.mean().tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', color=CLUSTER_COLORS[cl], linewidth=2,
                label=f'Cluster {cl} – {CLUSTER_NAMES[cl]}')
        ax.fill(angles, values, alpha=0.15, color=CLUSTER_COLORS[cl])

    ax.set_thetagrids(np.degrees(angles[:-1]), display_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f'Cluster Profiles {cfg["title_suffix"]}', fontsize=15, fontweight='bold', pad=25)

    # Country list annotation on the right
    country_text = _cluster_country_text(result_df)
    fig.text(0.98, 0.5, country_text, fontsize=7.5, fontfamily='monospace',
             va='center', ha='left', transform=fig.transFigure,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7', edgecolor='#ccc', alpha=0.9))

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.subplots_adjust(right=0.72)

    path = os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_cluster_radar.png')
    _save_fig(fig, path, dpi=150)
    plt.close(fig)

    # Excel
    rows = []
    for cl in sorted(result_df['Cluster'].unique()):
        cdf = result_df[result_df['Cluster'] == cl]
        row = {'Cluster': cl, 'Cluster_Name': CLUSTER_NAMES[cl], 'N_Countries': len(cdf),
               'Countries': ', '.join(cdf['Country_Name'].tolist())}
        for col in feature_cols:
            row[f'{col}_mean'] = cdf[col].mean()
            row[f'{col}_std'] = cdf[col].std()
        rows.append(row)
    _save_excel(pd.DataFrame(rows), os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_cluster_radar.xlsx'))

    # ------------------------------------------------------------------
    # Real-values version (one independent radial axis per feature)
    # ------------------------------------------------------------------
    # Compute raw aggregated values per cluster
    raw_rows = {}
    for cl in sorted(result_df['Cluster'].unique()):
        cdf = result_df[result_df['Cluster'] == cl]
        raw_rows[cl] = [cdf[col].mean() for col in feature_cols]

    # Determine per-axis min/max across clusters for scaling to [0,1] on the plot
    raw_matrix = np.array(list(raw_rows.values()))  # shape (n_clusters, n_features)
    axis_min = raw_matrix.min(axis=0)
    axis_max = raw_matrix.max(axis=0)
    axis_range = np.where(axis_max - axis_min < 1e-9, 1.0, axis_max - axis_min)
    # Add 10 % padding so points don't sit on the boundary
    axis_min_padded = axis_min - 0.10 * axis_range
    axis_max_padded = axis_max + 0.10 * axis_range
    axis_range_padded = axis_max_padded - axis_min_padded

    fig2, ax2 = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))

    for cl in sorted(raw_rows.keys()):
        scaled = [(v - axis_min_padded[i]) / axis_range_padded[i]
                  for i, v in enumerate(raw_rows[cl])]
        scaled += scaled[:1]
        ax2.plot(angles, scaled, 'o-', color=CLUSTER_COLORS[cl], linewidth=2,
                 label=f'Cluster {cl} – {CLUSTER_NAMES[cl]}')
        ax2.fill(angles, scaled, alpha=0.15, color=CLUSTER_COLORS[cl])

    # Build tick labels showing real values at evenly-spaced radial positions
    # Skip the very center (tp=0) where all axes converge and labels overlap
    n_ticks = 5
    tick_positions = np.linspace(0, 1, n_ticks)
    for i, col in enumerate(feature_cols):
        angle = angles[i]
        for tp in tick_positions[1:]:  # skip center
            real_val = axis_min_padded[i] + tp * axis_range_padded[i]
            if abs(real_val) >= 1000:
                label_text = f'{real_val:,.0f}'
            elif abs(real_val) >= 1:
                label_text = f'{real_val:.2f}'
            else:
                label_text = f'{real_val:.4f}'
            ax2.text(angle, tp, label_text, fontsize=6, ha='center', va='bottom',
                     color='grey', alpha=0.8)

    # Central value box: show what the center (0) represents per axis
    center_lines = ["Center values:"]
    for i, lbl in enumerate(display_labels):
        clean_lbl = lbl.replace('\n', ' ')
        real_val = axis_min_padded[i]
        if abs(real_val) >= 1000:
            center_lines.append(f"  {clean_lbl}: {real_val:,.0f}")
        elif abs(real_val) >= 1:
            center_lines.append(f"  {clean_lbl}: {real_val:.2f}")
        else:
            center_lines.append(f"  {clean_lbl}: {real_val:.4f}")
    fig2.text(0.02, 0.02, "\n".join(center_lines), fontsize=7, fontfamily='monospace',
              va='bottom', ha='left', transform=fig2.transFigure,
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f7f7', edgecolor='#ccc', alpha=0.9))

    ax2.set_thetagrids(np.degrees(angles[:-1]), display_labels, fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.set_yticklabels([])
    ax2.set_title(f'Cluster Profiles – Real Values {cfg["title_suffix"]}',
                  fontsize=15, fontweight='bold', pad=25)

    country_text2 = _cluster_country_text(result_df)
    fig2.text(0.98, 0.5, country_text2, fontsize=7.5, fontfamily='monospace',
              va='center', ha='left', transform=fig2.transFigure,
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7', edgecolor='#ccc', alpha=0.9))

    ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig2.subplots_adjust(right=0.72)

    path2 = os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_cluster_radar_real.png')
    _save_fig(fig2, path2, dpi=150)
    plt.close(fig2)

    # Excel for real-values version
    real_excel_rows = []
    for cl in sorted(raw_rows.keys()):
        row = {'Cluster': cl, 'Cluster_Name': CLUSTER_NAMES[cl]}
        for j, col in enumerate(feature_cols):
            row[f'{col}_mean'] = raw_rows[cl][j]
        real_excel_rows.append(row)
    _save_excel(pd.DataFrame(real_excel_rows),
                os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_cluster_radar_real.xlsx'))


# ===================================================================
# VISUAL 4: EWBI country map (matplotlib)
# ===================================================================
def plot_country_map(result_df, cfg, out_dir):
    """Choropleth map of EWBI last-year values."""
    map_df = result_df.copy()
    nuts0, bg = _load_nuts0_and_background()
    merged_geo = nuts0.merge(map_df[['Country', 'EWBI_Last']], left_on='NUTS_ID', right_on='Country', how='left')
    study_ids = set(map_df['Country'].tolist())

    fig, ax = plt.subplots(figsize=(12, 10))
    bg.plot(ax=ax, color='white', edgecolor='black', linewidth=0.3, alpha=0.35, hatch='///')

    no_data = merged_geo[merged_geo['EWBI_Last'].isna() & merged_geo['NUTS_ID'].isin(study_ids)]
    if not no_data.empty:
        no_data.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.3)

    has_data = merged_geo[merged_geo['EWBI_Last'].notna()]
    if not has_data.empty:
        has_data.plot(column='EWBI_Last', cmap='RdYlGn', edgecolor='black', linewidth=0.4,
                      ax=ax, legend=True, legend_kwds={'label': 'EWBI', 'shrink': 0.6})

    ax.set_xlim(2.5e6, 6.5e6)
    ax.set_ylim(1.3e6, 5.5e6)
    ax.set_axis_off()
    ax.set_title(f'EWBI Country Map {cfg["title_suffix"]}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_country_map_mpl.png')
    _save_fig(fig, path, dpi=150)
    plt.close(fig)

    # Excel
    excel_df = result_df[['Country', 'Country_Name', 'EWBI_Last', 'Cluster']].copy()
    excel_df['Cluster_Name'] = excel_df['Cluster'].map(CLUSTER_NAMES)
    excel_df = excel_df.sort_values('Country_Name')
    _save_excel(excel_df, os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_country_map_mpl.xlsx'))


# ===================================================================
# VISUAL 5: Performance vs EWBI scatter
# ===================================================================
def plot_performance_vs_ewbi(result_df, cfg, out_dir):
    """Scatter: EWBI (x) vs Performance Score (y) with cluster thresholds."""
    fig, ax = plt.subplots(figsize=(14, 10))

    x_min = result_df['EWBI_Last'].min() - 0.02
    x_max = result_df['EWBI_Last'].max() + 0.02
    y_min = result_df['Performance_Score'].min() - 0.01
    y_max = result_df['Performance_Score'].max() + 0.01

    # Shade quadrants
    perf_frac = (PERF_CUT - y_min) / (y_max - y_min)
    ax.axvspan(x_min, EWBI_CUT, ymin=0, ymax=perf_frac, alpha=0.06, color=CLUSTER_COLORS[0])
    ax.axvspan(EWBI_CUT, x_max, ymin=0, ymax=perf_frac, alpha=0.06, color=CLUSTER_COLORS[1])
    ax.axvspan(x_min, EWBI_CUT, ymin=perf_frac, ymax=1, alpha=0.06, color=CLUSTER_COLORS[2])
    ax.axvspan(EWBI_CUT, x_max, ymin=perf_frac, ymax=1, alpha=0.06, color=CLUSTER_COLORS[3])

    ax.axvline(EWBI_CUT, color='#333', linewidth=1.5, linestyle='--', alpha=0.7,
               label=f'EWBI threshold = {EWBI_CUT}')
    ax.axhline(PERF_CUT, color='#333', linewidth=1.5, linestyle='--', alpha=0.7,
               label=f'Performance threshold = {PERF_CUT}')

    for cl in sorted(result_df['Cluster'].unique()):
        cdf = result_df[result_df['Cluster'] == cl]
        ax.scatter(cdf['EWBI_Last'], cdf['Performance_Score'],
                   c=CLUSTER_COLORS[cl], s=120, edgecolors='black', linewidths=0.6,
                   zorder=3, label=f'Cluster {cl} – {CLUSTER_NAMES[cl]}')

    for _, row in result_df.iterrows():
        ax.annotate(row['Country_Name'], (row['EWBI_Last'], row['Performance_Score']),
                    fontsize=8, fontweight='bold',
                    textcoords='offset points', xytext=(6, 4), zorder=4,
                    color=CLUSTER_COLORS[int(row['Cluster'])])

    # Quadrant labels
    ax.text(x_min + 0.005, y_max - 0.003, 'High performer / Low EWBI',
            fontsize=9, fontstyle='italic', color=CLUSTER_COLORS[2], alpha=0.7, va='top')
    ax.text(EWBI_CUT + 0.005, y_max - 0.003, 'High performer / High EWBI',
            fontsize=9, fontstyle='italic', color=CLUSTER_COLORS[3], alpha=0.7, va='top')
    ax.text(x_min + 0.005, y_min + 0.003, 'Low performer / Low EWBI',
            fontsize=9, fontstyle='italic', color=CLUSTER_COLORS[0], alpha=0.7, va='bottom')
    ax.text(EWBI_CUT + 0.005, y_min + 0.003, 'Low performer / High EWBI',
            fontsize=9, fontstyle='italic', color=CLUSTER_COLORS[1], alpha=0.7, va='bottom')

    ax.set_xlabel('EWBI (Last Available Year)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance Score (deviation from benchmark)', fontsize=13, fontweight='bold')
    ax.set_title(f'Cluster Allocation: Performance Score vs EWBI {cfg["title_suffix"]}',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    plt.tight_layout()

    path = os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_performance_vs_ewbi_clusters.png')
    _save_fig(fig, path, dpi=150)
    plt.close(fig)

    # Excel
    excel_df = result_df[['Country', 'Country_Name', 'Cluster', 'EWBI_Last', 'Performance_Score']].copy()
    excel_df['Cluster_Name'] = excel_df['Cluster'].map(CLUSTER_NAMES)
    excel_df = excel_df.sort_values(['Cluster', 'Country_Name'])
    _save_excel(excel_df, os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_performance_vs_ewbi_clusters.xlsx'))


# ===================================================================
# VISUAL 6: EWBI vs Income by cluster (4-panel, last year)
# ===================================================================
def plot_ewbi_vs_income_by_cluster(result_df, cfg, out_dir):
    """4-panel EWBI vs Income scatter (last year). Non-target clusters in grey."""
    merged = _load_ewbi_income_merged()
    if merged is None:
        return

    country_set = set(result_df['Country'].tolist())
    merged = merged[merged['Country'].isin(country_set)].copy()

    last_year = merged.groupby('Country')['Year'].max().reset_index()
    last_year.columns = ['Country', 'last_year']
    plot_data = merged.merge(last_year, on='Country')
    plot_data = plot_data[plot_data['Year'] == plot_data['last_year']].copy()

    cluster_map = dict(zip(result_df['Country'], result_df['Cluster']))
    highlight_codes = cfg['highlight_countries']
    countries = sorted(plot_data['Country'].unique())

    years_used = plot_data.groupby('Country')['Year'].first()
    year_min, year_max = int(years_used.min()), int(years_used.max())
    year_label = f"{year_min}" if year_min == year_max else f"{year_min}–{year_max}"

    x_min = plot_data['median_equi_disp_inc'].min()
    x_max = plot_data['median_equi_disp_inc'].max()
    y_min = plot_data['Value'].min()
    y_max = plot_data['Value'].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.07

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, cl in enumerate(sorted(CLUSTER_NAMES.keys())):
        ax = axes[i]
        target_countries = [c for c in countries if cluster_map.get(c) == cl]
        other_countries = [c for c in countries if cluster_map.get(c) != cl]

        for country in other_countries:
            cdata = plot_data[plot_data['Country'] == country].sort_values('Decile')
            if cdata.empty:
                continue
            ax.plot(cdata['median_equi_disp_inc'], cdata['Value'],
                    color='#bdbdbd', alpha=0.35, linewidth=0.9, zorder=1)
            ax.scatter(cdata['median_equi_disp_inc'], cdata['Value'],
                       c='#bdbdbd', s=30, alpha=0.35, edgecolors='white', linewidths=0.2, zorder=1)

        for country in target_countries:
            cdata = plot_data[plot_data['Country'] == country].sort_values('Decile')
            if cdata.empty:
                continue
            is_hl = country in highlight_codes
            lw = 2.4 if is_hl else 1.4
            ms = 90 if is_hl else 55
            alpha = 0.95 if is_hl else 0.78
            ax.plot(cdata['median_equi_disp_inc'], cdata['Value'],
                    color=CLUSTER_COLORS[cl], alpha=0.75, linewidth=lw, zorder=2)
            ax.scatter(cdata['median_equi_disp_inc'], cdata['Value'],
                       c=CLUSTER_COLORS[cl], s=ms, alpha=alpha,
                       edgecolors='black' if is_hl else 'white',
                       linewidths=0.6 if is_hl else 0.3, zorder=3)

            if is_hl:
                label = _country_name(country)
                row_last = cdata.iloc[-1]
                ax.annotate(label, (row_last['median_equi_disp_inc'], row_last['Value']),
                            fontsize=8.5, fontweight='bold', color=CLUSTER_COLORS[cl],
                            textcoords='offset points', xytext=(5, 2), zorder=4)

        ax.set_title(f'Cluster {cl} – {CLUSTER_NAMES[cl]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_facecolor('white')
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    for i in [2, 3]:
        axes[i].set_xlabel('Median Equivalized Disposable Income (€)', fontsize=12)
    for i in [0, 2]:
        axes[i].set_ylabel('EWBI Score', fontsize=12)

    fig.suptitle(
        f'EWBI vs Median Income by Country and Decile {cfg["title_suffix"]}\n'
        f'(Last available year: {year_label}; other clusters in grey)',
        fontsize=16, fontweight='bold', y=0.98,
    )

    legend_handles = [
        Line2D([0], [0], color='#bdbdbd', lw=2, label='Other clusters'),
    ] + [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CLUSTER_COLORS[cl], markeredgecolor='black',
               markeredgewidth=0.7, markersize=9,
               label=f'Cluster {cl} – {CLUSTER_NAMES[cl]}')
        for cl in sorted(CLUSTER_NAMES.keys())
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.01), frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    path = os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_vs_income_by_cluster.png')
    _save_fig(fig, path, dpi=150)
    plt.close(fig)

    # Excel
    plot_data['Cluster'] = plot_data['Country'].map(cluster_map)
    plot_data['Cluster_Name'] = plot_data['Cluster'].map(CLUSTER_NAMES)
    plot_data['Country_Name'] = plot_data['Country'].map(COUNTRY_NAME_MAP).fillna(plot_data['Country'])
    export_cols = ['Country', 'Country_Name', 'Cluster', 'Cluster_Name', 'Year', 'Decile', 'Value', 'median_equi_disp_inc']
    _save_excel(
        plot_data[export_cols].sort_values(['Cluster', 'Country', 'Decile']),
        os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_vs_income_by_cluster.xlsx'),
    )


# ===================================================================
# VISUAL 7: Cluster × Priority grid
# ===================================================================
def plot_cluster_priority_grid(result_df, cfg, out_dir):
    """Grid: rows = EU priorities, columns = clusters."""
    merged = _load_priority_income_merged()
    if merged is None:
        return

    country_set = set(result_df['Country'].tolist())
    merged = merged[merged['Country'].isin(country_set)].copy()

    cluster_map = dict(zip(result_df['Country'], result_df['Cluster']))
    merged['Cluster'] = merged['Country'].map(cluster_map)
    merged = merged.dropna(subset=['Cluster']).copy()
    merged['Cluster'] = merged['Cluster'].astype(int)

    last_year_cp = merged.groupby(['Country', 'EU priority'])['Year'].max().reset_index()
    last_year_cp.columns = ['Country', 'EU priority', 'last_year']
    plot_data = merged.merge(last_year_cp, on=['Country', 'EU priority'])
    plot_data = plot_data[plot_data['Year'] == plot_data['last_year']].copy()

    if plot_data.empty:
        print("  WARNING: No data for cluster-priority grid")
        return

    cluster_ids = sorted(CLUSTER_NAMES.keys())
    priorities = sorted(plot_data['EU priority'].unique().tolist())

    x_min, x_max = plot_data['median_equi_disp_inc'].min(), plot_data['median_equi_disp_inc'].max()
    y_min, y_max = plot_data['Value'].min(), plot_data['Value'].max()
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.01

    n_rows = len(priorities)
    n_cols = len(cluster_ids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows),
                             sharex=True, sharey=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for r, priority in enumerate(priorities):
        for c, cl in enumerate(cluster_ids):
            ax = axes[r, c]
            cell = plot_data[plot_data['EU priority'] == priority].copy()

            bg = cell[cell['Cluster'] != cl]
            for country in sorted(bg['Country'].unique()):
                cdata = bg[bg['Country'] == country].sort_values('Decile')
                ax.plot(cdata['median_equi_disp_inc'], cdata['Value'],
                        color='#bfbfbf', alpha=0.35, linewidth=0.9, zorder=1)
                ax.scatter(cdata['median_equi_disp_inc'], cdata['Value'],
                           c='#bfbfbf', s=16, alpha=0.35, edgecolors='none', zorder=1)

            fg = cell[cell['Cluster'] == cl]
            for country in sorted(fg['Country'].unique()):
                cdata = fg[fg['Country'] == country].sort_values('Decile')
                ax.plot(cdata['median_equi_disp_inc'], cdata['Value'],
                        color=CLUSTER_COLORS[cl], alpha=0.85, linewidth=1.6, zorder=2)
                ax.scatter(cdata['median_equi_disp_inc'], cdata['Value'],
                           c=CLUSTER_COLORS[cl], s=24, alpha=0.9,
                           edgecolors='white', linewidths=0.25, zorder=2)

            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.grid(True, alpha=0.25)
            ax.set_facecolor('white')

            if r == 0:
                ax.set_title(f"Cluster {cl}", fontsize=11, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f"{_priority_label(priority)}\nEWBI", fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel('Income (€)', fontsize=10)

    fig.suptitle(
        f'EWBI vs Income: EU Priority × Cluster {cfg["title_suffix"]}',
        fontsize=16, fontweight='bold', y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])

    path = os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_vs_income_cluster_priority_grid.png')
    _save_fig(fig, path, dpi=180)
    plt.close(fig)

    # Excel
    plot_data['Country_Name'] = plot_data['Country'].map(COUNTRY_NAME_MAP).fillna(plot_data['Country'])
    plot_data['Cluster_Name'] = plot_data['Cluster'].map(CLUSTER_NAMES)
    export_cols = ['Country', 'Country_Name', 'Cluster', 'Cluster_Name', 'EU priority',
                   'Year', 'Decile', 'Value', 'median_equi_disp_inc']
    _save_excel(
        plot_data[export_cols].sort_values(['Cluster', 'EU priority', 'Country', 'Decile']),
        os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_vs_income_cluster_priority_grid.xlsx'),
    )


# ===================================================================
# VISUAL 8: One choropleth map per EU priority
# ===================================================================
def plot_priority_maps(result_df, cfg, out_dir):
    """Choropleth maps of last-year EU-priority EWBI values (one map per priority)."""
    ewbi_df = pd.read_csv(data_path, low_memory=False)

    priorities_all = ewbi_df[
        (ewbi_df['Level'] == 2) &
        (ewbi_df['Decile'] == 'All Deciles') &
        (~ewbi_df['Country'].isin(['EU-27', 'All Countries'])) &
        (ewbi_df['Country'].isin(result_df['Country'].tolist()))
    ].copy()
    priorities_all['Year'] = pd.to_numeric(priorities_all['Year'], errors='coerce').astype('Int64')
    priorities_all['Value'] = pd.to_numeric(priorities_all['Value'], errors='coerce')
    priorities_all = priorities_all.dropna(subset=['Country', 'Year', 'Value', 'EU priority'])

    last_year_cp = (
        priorities_all.groupby(['Country', 'EU priority'])['Year'].max().reset_index()
    )
    last_year_cp.columns = ['Country', 'EU priority', 'last_year']
    plot_data = priorities_all.merge(last_year_cp, on=['Country', 'EU priority'])
    plot_data = plot_data[plot_data['Year'] == plot_data['last_year']].copy()

    if plot_data.empty:
        print("  WARNING: No Level-2 data for priority maps")
        return

    priorities = sorted(plot_data['EU priority'].unique().tolist())
    nuts0, bg = _load_nuts0_and_background()
    study_ids = set(result_df['Country'].tolist())

    all_excel_dfs = {}

    for priority in priorities:
        slug = (
            priority.lower()
            .replace(',', '').replace('/', '_').replace(' ', '_')
        )[:45]
        pdata = plot_data[plot_data['EU priority'] == priority][['Country', 'Value']].copy()

        merged_geo = nuts0.merge(
            pdata, left_on='NUTS_ID', right_on='Country', how='left'
        )

        vmin = float(pdata['Value'].min())
        vmax = float(pdata['Value'].max())

        fig, ax = plt.subplots(figsize=(12, 10))
        bg.plot(ax=ax, color='white', edgecolor='black', linewidth=0.3, alpha=0.35, hatch='///')

        no_data = merged_geo[
            merged_geo['Value'].isna() & merged_geo['NUTS_ID'].isin(study_ids)
        ]
        if not no_data.empty:
            no_data.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.3)

        has_data = merged_geo[merged_geo['Value'].notna()]
        if not has_data.empty:
            has_data.plot(
                column='Value', cmap='RdYlGn', edgecolor='black', linewidth=0.4,
                ax=ax, legend=True,
                legend_kwds={
                    'label': f'{_priority_label(priority)} (EWBI)',
                    'shrink': 0.6,
                },
                vmin=vmin, vmax=vmax,
            )

        ax.set_xlim(2.5e6, 6.5e6)
        ax.set_ylim(1.3e6, 5.5e6)
        ax.set_axis_off()
        ax.set_title(
            f'{_priority_label(priority)} — EWBI Map\n{cfg["title_suffix"]}',
            fontsize=13, fontweight='bold',
        )
        plt.tight_layout()

        path = os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_priority_map_{slug}.png')
        _save_fig(fig, path, dpi=150)
        plt.close(fig)

        all_excel_dfs[_priority_label(priority)[:31]] = (
            pdata.rename(columns={'Value': 'EWBI_Last'})
                 .sort_values('Country')
        )

    _save_excel(
        all_excel_dfs,
        os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_priority_maps.xlsx'),
    )


# ===================================================================
# VISUAL 9: EU priorities bar grid (cluster > country > decile)
# ===================================================================
def plot_priority_bar_grid(result_df, cfg, out_dir):
    """
    Bar chart mirroring 3_housing_quality.py but for EU priorities (Level 2).
    Rows = EU priorities, columns = countries × deciles (D1–D10).
    Bars coloured by cluster.
    """
    ewbi_df = pd.read_csv(data_path, low_memory=False)

    df = ewbi_df[
        (ewbi_df['Level'] == 2) &
        (ewbi_df['Decile'] != 'All Deciles') &
        (~ewbi_df['Country'].isin(['EU-27', 'All Countries'])) &
        (ewbi_df['Country'].isin(result_df['Country'].tolist()))
    ].copy()
    df['Decile_num'] = pd.to_numeric(df['Decile'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Decile_num', 'Year', 'Value', 'EU priority'])
    df['Decile_num'] = df['Decile_num'].astype(int)
    df = df[df['Decile_num'].between(1, 10)].copy()

    # Keep latest year per country × priority
    latest = df.groupby(['Country', 'EU priority'])['Year'].max().reset_index()
    latest.columns = ['Country', 'EU priority', 'Latest_Year']
    df = df.merge(latest, on=['Country', 'EU priority'])
    df = df[df['Year'] == df['Latest_Year']].copy()

    if df.empty:
        print("  WARNING: No Level-2 decile data for priority bar grid")
        return

    priorities = sorted(df['EU priority'].unique().tolist())
    cluster_map_col = dict(zip(result_df['Country'], result_df['Cluster'].astype(int)))

    # Build country order: grouped by cluster (ascending), then alphabetically within cluster
    ordered = []
    for cl in sorted(CLUSTER_NAMES.keys()):
        cl_countries = sorted([
            cc for cc in result_df[result_df['Cluster'] == cl]['Country'].tolist()
            if cc in set(df['Country'].unique())
        ])
        ordered.extend(cl_countries)
    seen = set()
    country_order = []
    for cc in ordered:
        if cc not in seen:
            seen.add(cc)
            country_order.append(cc)

    if not country_order:
        print("  WARNING: No countries with Level-2 decile data")
        return

    n_deciles   = 10
    country_gap = 1.0
    cluster_gap = 2.5
    bar_width   = 0.7

    fig_w = max(28, len(country_order) * 1.6)
    fig, axes = plt.subplots(
        len(priorities), 1,
        figsize=(fig_w, 5 * len(priorities)),
        squeeze=False,
    )

    all_excel_rows = []

    for plot_idx, priority in enumerate(priorities):
        ax = axes[plot_idx][0]
        df_pri = df[df['EU priority'] == priority].copy()

        # Build x positions with cluster / country gaps
        x_positions          = []
        cluster_boundaries   = []
        cluster_label_pos    = []
        tick_positions       = []
        tick_labels_list     = []
        current_x            = 0.0
        prev_cluster_id      = None
        cluster_start_x      = 0.0

        for cc in country_order:
            cl_id = cluster_map_col.get(cc, -1)

            if prev_cluster_id is not None and cl_id != prev_cluster_id:
                cluster_label_pos.append(
                    ((cluster_start_x + current_x - country_gap) / 2, prev_cluster_id)
                )
                cluster_boundaries.append(current_x - country_gap / 2)
                current_x    += cluster_gap
                cluster_start_x = current_x
            elif current_x > 0:
                current_x += country_gap

            if prev_cluster_id is None:
                cluster_start_x = current_x
            prev_cluster_id = cl_id

            decile_xs = np.arange(n_deciles) * (bar_width + 0.05) + current_x
            x_positions.append((cc, decile_xs))
            tick_positions.append((decile_xs[0] + decile_xs[-1]) / 2)
            tick_labels_list.append(cc)
            current_x = decile_xs[-1] + bar_width + 0.05

        if prev_cluster_id is not None:
            cluster_label_pos.append(
                ((cluster_start_x + current_x) / 2, prev_cluster_id)
            )

        all_vals = [
            v for cc in country_order
            for v in df_pri[df_pri['Country'] == cc]['Value'].dropna().tolist()
        ]
        y_max = max(all_vals) * 1.12 if all_vals else 1.0

        for cc, decile_xs in x_positions:
            cdf    = df_pri[df_pri['Country'] == cc].sort_values('Decile_num')
            cl_id  = cluster_map_col.get(cc, -1)
            color  = CLUSTER_COLORS[cl_id] if 0 <= cl_id < len(CLUSTER_COLORS) else '#999999'

            vals = []
            for d in range(1, 11):
                row = cdf[cdf['Decile_num'] == d]['Value']
                vals.append(row.iloc[0] if len(row) else np.nan)

            bars = ax.bar(
                decile_xs, vals, width=bar_width,
                color=color, edgecolor='white', linewidth=0.4,
            )
            for i, (bar, val) in enumerate(zip(bars, vals)):
                if not pd.isna(val) and i in (0, 9):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + y_max * 0.005,
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=5, fontweight='bold',
                    )

            latest_year = int(cdf['Year'].iloc[0]) if len(cdf) else ''
            for d, val in zip(range(1, 11), vals):
                all_excel_rows.append({
                    'EU_Priority':       priority,
                    'EU_Priority_Label': _priority_label(priority),
                    'Country':           cc,
                    'Country_Name':      COUNTRY_NAME_MAP.get(cc, cc),
                    'Cluster':           cl_id,
                    'Cluster_Name':      CLUSTER_NAMES.get(cl_id, ''),
                    'Year':              latest_year,
                    'Decile':            d,
                    'Value':             round(float(val), 4) if not pd.isna(val) else None,
                })

        for bx in cluster_boundaries:
            ax.axvline(x=bx, color='#888888', linewidth=1.0, linestyle='--', alpha=0.5)

        if plot_idx == 0:
            for cx, cl_id in cluster_label_pos:
                ax.text(
                    cx, y_max * 1.08,
                    f'Cluster {cl_id} \u2013 {CLUSTER_NAMES[cl_id]}',
                    ha='center', va='bottom', fontsize=8,
                    fontweight='bold', fontstyle='italic',
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor=CLUSTER_COLORS[cl_id],
                        edgecolor='#cccccc', alpha=0.5,
                    ),
                )

        ax.set_title(_priority_label(priority), fontsize=11, fontweight='bold', pad=4)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, fontsize=8, fontweight='bold',
                           rotation=0, ha='center')
        ax.set_xlim(-1, current_x + 0.5)
        ax.set_ylim(0, y_max)
        ax.grid(axis='y', alpha=0.2)
        ax.set_facecolor('white')
        ax.set_ylabel('EWBI value', fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', pad=2)
        ax.tick_params(axis='y', pad=2)

        # Disclaimer for countries with older data
        year_by_cc = df_pri.groupby('Country')['Year'].max()
        year_by_cc = year_by_cc[year_by_cc.index.isin(country_order)]
        if not year_by_cc.empty:
            modal_year = int(year_by_cc.mode().iloc[0])
            older = year_by_cc[year_by_cc < modal_year]
            if not older.empty:
                parts = [
                    f"{COUNTRY_NAME_MAP.get(cc, cc)}: {int(yr)}"
                    for cc, yr in sorted(older.items())
                ]
                ax.text(
                    0.99, 0.97, 'Data not from ' + str(modal_year) + ': ' + ', '.join(parts),
                    transform=ax.transAxes, fontsize=6, va='top', ha='right',
                    fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6',
                              edgecolor='#ccaa00', alpha=0.85),
                )

    fig.suptitle(
        f'EU Priorities EWBI by Income Decile\n{cfg["title_suffix"]}',
        fontsize=14, fontweight='bold', y=1.0,
    )

    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            fc=CLUSTER_COLORS[cl], ec='white',
            label=f'Cluster {cl} \u2013 {CLUSTER_NAMES[cl]}',
        )
        for cl in sorted(CLUSTER_NAMES.keys())
        if any(cluster_map_col.get(cc, -1) == cl for cc in country_order)
    ]
    fig.legend(
        handles=legend_handles, loc='lower center',
        ncol=min(len(legend_handles), 4), fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    plt.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.04, hspace=0.25)

    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_priority_bar_grid.png')
    _save_fig(fig, out_png, dpi=150)
    plt.close(fig)

    # Excel
    excel_df = pd.DataFrame(all_excel_rows)
    sheets: dict = {'Long_Format': excel_df}
    if not excel_df.empty:
        wide = excel_df.pivot_table(
            index=['EU_Priority', 'EU_Priority_Label', 'Country', 'Country_Name',
                   'Cluster', 'Cluster_Name', 'Year'],
            columns='Decile', values='Value', aggfunc='first',
        )
        wide.columns = [f'D{int(c)}' for c in wide.columns]
        wide = wide.reset_index()
        sheets['Wide_Format'] = wide
    _save_excel(
        sheets,
        os.path.join(out_dir, f'{cfg["prefix"]}_ewbi_priority_bar_grid.xlsx'),
    )


# ===================================================================
# Main
# ===================================================================
def generate_report(features_df_all, report_key):
    """Generate all 9 visuals for one report configuration."""
    cfg = REPORT_CONFIGS[report_key]
    print(f"\n{'='*60}")
    print(f"Generating report: {cfg['prefix']} {cfg['title_suffix']}")
    print(f"{'='*60}")

    result_df = _filter_df(features_df_all, cfg['country_filter'])
    print(f"  Countries in report: {len(result_df)}")
    for cl in sorted(result_df['Cluster'].unique()):
        names = result_df[result_df['Cluster'] == cl].sort_values('Country_Name')['Country_Name'].tolist()
        print(f"    Cluster {cl} – {CLUSTER_NAMES[cl]}: {names}")

    out_dir = os.path.join(output_base, cfg['prefix'])
    os.makedirs(out_dir, exist_ok=True)

    print("\n  [1/9] Cluster map...")
    plot_cluster_map(result_df, cfg, out_dir)

    print("\n  [2/9] Priority radar...")
    plot_priority_radar(result_df, cfg, out_dir)

    print("\n  [3/9] Cluster radar...")
    plot_cluster_radar(result_df, cfg, out_dir)

    print("\n  [4/9] Country EWBI map...")
    plot_country_map(result_df, cfg, out_dir)

    print("\n  [5/9] Performance vs EWBI scatter...")
    plot_performance_vs_ewbi(result_df, cfg, out_dir)

    print("\n  [6/9] EWBI vs Income by cluster...")
    plot_ewbi_vs_income_by_cluster(result_df, cfg, out_dir)

    print("\n  [7/9] Cluster × Priority grid...")
    plot_cluster_priority_grid(result_df, cfg, out_dir)

    print("\n  [8/9] Priority choropleth maps (one per EU priority)...")
    plot_priority_maps(result_df, cfg, out_dir)

    print("\n  [9/9] Priority bar grid (cluster > country > decile)...")
    plot_priority_bar_grid(result_df, cfg, out_dir)

    print(f"\n  All outputs for {cfg['prefix']} saved to: {out_dir}")


def main():
    features_df = load_and_prepare_data()

    for report_key in ['rep_eu', 'rep_ewbi']:
        generate_report(features_df, report_key)

    print("\nDone.")


if __name__ == '__main__':
    main()
