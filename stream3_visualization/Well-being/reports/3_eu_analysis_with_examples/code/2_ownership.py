"""
2_ownership.py - Ownership & tenure visuals for 4 reports.

Produces exactly 2 visuals per report:
  1. Real estate by income quintile — cluster-grouped panels (from eurostat_analysis.py
     create_real_estate_cluster_quintiles), adapted country lists per report.
  2. Tenure-status countries choropleth map (from eurostat_analysis_swiss.py
     create_tenure_status_countries_map), adapted study-country scope per report.

Reports:
  rep_eu   : EU-27 countries, clusters shown, + EU-27 aggregate panel
  rep_ewbi : EU-27 + EFTA, clusters shown, + EU-27 aggregate panel
  rep_fr   : France focus, comparison countries (no cluster grouping), + EU-27 aggregate
  rep_ch   : Switzerland focus, comparison countries (no cluster grouping), + EU-27 aggregate

All outputs: PNG + SVG + Excel.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm_mod
import matplotlib.patches as mpatches
import geopandas as gpd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, 'external_data')
OUTPUT_BASE = os.path.join(BASE_DIR, 'outputs', 'graphs', 'EUROSTAT', 'Ownership')
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Shapefile for map
_WORLD_SHP = os.path.join(
    BASE_DIR, '..', '1_switzerland_vs_eu27_housing_energy',
    'external_data', '0_shapefile', 'ne_50m_admin_0_countries',
    'ne_50m_admin_0_countries.shp',
)

# ---------------------------------------------------------------------------
# Country definitions
# ---------------------------------------------------------------------------
EU27_COUNTRIES = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
    'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
    'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
    'Spain', 'Sweden',
]
EFTA_COUNTRIES = ['Iceland', 'Norway', 'Switzerland']

EUROSTAT_TO_STANDARD = {
    'Austria': 'Austria', 'Belgium': 'Belgium', 'Bulgaria': 'Bulgaria',
    'Croatia': 'Croatia', 'Cyprus': 'Cyprus', 'Czechia': 'Czech Republic',
    'Denmark': 'Denmark', 'Estonia': 'Estonia', 'Finland': 'Finland',
    'France': 'France', 'Germany': 'Germany', 'Greece': 'Greece',
    'Hungary': 'Hungary', 'Iceland': 'Iceland', 'Ireland': 'Ireland',
    'Italy': 'Italy', 'Latvia': 'Latvia', 'Lithuania': 'Lithuania',
    'Luxembourg': 'Luxembourg', 'Malta': 'Malta', 'Netherlands': 'Netherlands',
    'Norway': 'Norway', 'Poland': 'Poland', 'Portugal': 'Portugal',
    'Romania': 'Romania', 'Serbia': 'Serbia', 'Slovakia': 'Slovakia',
    'Slovenia': 'Slovenia', 'Spain': 'Spain', 'Sweden': 'Sweden',
    'Switzerland': 'Switzerland', 'United Kingdom': 'United Kingdom',
}

COUNTRY_TO_ISO = {
    'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
    'Cyprus': 'CY', 'Czech Republic': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE',
    'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'GR',
    'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE', 'Italy': 'IT',
    'Latvia': 'LV', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT',
    'Netherlands': 'NL', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT',
    'Romania': 'RO', 'Serbia': 'RS', 'Slovakia': 'SK', 'Slovenia': 'SI',
    'Spain': 'ES', 'Sweden': 'SE', 'Switzerland': 'CH',
    'United Kingdom': 'UK',
}

# Cluster definitions (same as eurostat_analysis.py / 0_clustering.py)
CLUSTERS = [
    {
        'label': 'Cluster 0 - Low performer / Low EWBI',
        'candidates': ['France', 'Spain', 'Greece', 'Italy', 'Portugal', 'Finland'],
    },
    {
        'label': 'Cluster 1 - Low performer / High EWBI',
        'candidates': ['Belgium', 'Netherlands', 'Austria', 'Denmark', 'Ireland'],
    },
    {
        'label': 'Cluster 2 - High performer / Low EWBI',
        'candidates': ['Lithuania', 'Hungary', 'Romania', 'Bulgaria', 'Latvia', 'Estonia'],
    },
    {
        'label': 'Cluster 3 - High performer / High EWBI',
        'candidates': ['Germany', 'Poland', 'Sweden', 'Czech Republic', 'Slovenia', 'Slovakia'],
    },
]
CLUSTER_COLORS = {
    'Cluster 0 - Low performer / Low EWBI': '#fb8072',
    'Cluster 1 - Low performer / High EWBI': '#fdb462',
    'Cluster 2 - High performer / Low EWBI': '#8dd3c7',
    'Cluster 3 - High performer / High EWBI': '#80b1d3',
    'EU-27': '#6a3d9a',
}
CODE_MAP = {
    'France': 'FR', 'Spain': 'ES', 'Belgium': 'BE', 'Netherlands': 'NL',
    'Lithuania': 'LT', 'Hungary': 'HU', 'Germany': 'DE', 'Poland': 'PL',
    'EU27': 'EU-27',
}

# Color palette
COMPONENT_COLORS = [
    '#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3',
    '#80b1d3', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494',
]
_all_countries = EU27_COUNTRIES + EFTA_COUNTRIES
_color_palette = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf',
    '#999999', '#66c2a5', '#e67e22', '#8da0cb', '#d946ef', '#a6d854', '#ffd92f',
    '#e5c494', '#b3b3b3', '#8dd3c7', '#ffffb3', '#bebada', '#1f77b4', '#80b1d3',
    '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f',
    '#a6cee3', '#1f78b4',
]
COUNTRY_COLOR_MAP = {c: _color_palette[i % len(_color_palette)] for i, c in enumerate(_all_countries)}

# ---------------------------------------------------------------------------
# Report configurations
# ---------------------------------------------------------------------------
REPORT_CONFIGS = {
    'rep_eu': {
        'prefix': 'rep_eu',
        'title_suffix': '(EU-27)',
        'country_list': EU27_COUNTRIES,
        'include_eu27_agg': True,
        'comparison_countries': ['France', 'Spain', 'Belgium', 'Netherlands',
                                 'Lithuania', 'Croatia', 'Germany', 'Sweden'],
        'focus_country': None,
        'show_clusters': True,
    },
    'rep_ewbi': {
        'prefix': 'rep_ewbi',
        'title_suffix': '(EU-27 + EFTA + UK + Serbia)',
        'country_list': EU27_COUNTRIES + EFTA_COUNTRIES + ['United Kingdom', 'Serbia'],
        'include_eu27_agg': True,
        'comparison_countries': ['France', 'Spain', 'Belgium', 'Netherlands',
                                 'Lithuania', 'Croatia', 'Germany', 'Sweden',
                                 'Switzerland', 'Norway', 'Iceland',
                                 'United Kingdom', 'Serbia'],
        'focus_country': None,
        'show_clusters': True,
    },
    'rep_fr': {
        'prefix': 'rep_fr',
        'title_suffix': '(France)',
        'country_list': EU27_COUNTRIES + EFTA_COUNTRIES,
        'include_eu27_agg': True,
        'comparison_countries': ['France', 'Spain', 'Italy', 'Switzerland', 'Germany', 'Belgium'],
        'focus_country': 'France',
        'show_clusters': False,
    },
    'rep_ch': {
        'prefix': 'rep_ch',
        'title_suffix': '(Switzerland)',
        'country_list': EU27_COUNTRIES + EFTA_COUNTRIES,
        'include_eu27_agg': True,
        'comparison_countries': ['Switzerland', 'France', 'Italy', 'Germany', 'Austria'],
        'focus_country': 'Switzerland',
        'show_clusters': False,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def standardize_country_name(country):
    return EUROSTAT_TO_STANDARD.get(country, None)


def _save_fig(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    svg_path = os.path.splitext(path)[0] + '.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {os.path.basename(path)}")
    print(f"  Saved: {os.path.basename(svg_path)}")


def _save_excel(data, path, sheet_name='Data'):
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        if isinstance(data, dict):
            for name, df in data.items():
                df.to_excel(writer, sheet_name=name, index=False)
        else:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  Saved: {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Data loading (same as eurostat_analysis.py)
# ---------------------------------------------------------------------------
def load_real_estate_data():
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_real estate other than main.csv')
    if not os.path.exists(file_path):
        print(f"  Missing: {file_path}")
        return None
    df = pd.read_csv(file_path)
    df = df[df['TIME_PERIOD'] == df['TIME_PERIOD'].max()]
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label] + list(EUROSTAT_TO_STANDARD.keys()))].copy()
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    df_filtered['value'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['value'])
    return df_filtered


def load_tenure_status_data():
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_tenure status.csv')
    if not os.path.exists(file_path):
        print(f"  Missing: {file_path}")
        return None
    df = pd.read_csv(file_path)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label] + list(EUROSTAT_TO_STANDARD.keys()))].copy()
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    return df_filtered


# ===================================================================
# VISUAL 1: Real estate by income quintile — cluster-grouped panels
#   (adapted from eurostat_analysis.py create_real_estate_cluster_quintiles)
# ===================================================================
def plot_real_estate_cluster_quintiles(cfg, out_dir):
    """
    Cluster reports  (show_clusters=True):  2 countries per cluster + EU-27 panel,
        with cluster labels and separators — exactly like the original.
    Focus reports    (show_clusters=False): comparison_countries side-by-side + EU-27 panel,
        no cluster grouping.
    """
    print("  [1] Real estate quintiles by cluster / comparison...")
    df = load_real_estate_data()
    if df is None or df.empty:
        print("    No data available")
        return

    latest_year = int(df['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df.columns else 'Latest'
    quintile_order = ['First quintile', 'Second quintile', 'Third quintile',
                      'Fourth quintile', 'Fifth quintile']
    short_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

    # ---- Build ordered country list & cluster metadata ----
    selected_clusters = []   # only populated when show_clusters
    ordered_countries = []

    if cfg['show_clusters']:
        for cluster in CLUSTERS:
            chosen = []
            for country in cluster['candidates']:
                df_c = df[(df['country_name'] == country) & (df['quant_inc'].isin(quintile_order))]
                if len(df_c) >= 5:
                    ordered_countries.append(country)
                    chosen.append(country)
                if len(chosen) == 2:
                    break
            if len(chosen) < 2:
                print(f"    WARNING: {cluster['label']} — only {len(chosen)} countries with data")
            selected_clusters.append({'label': cluster['label'], 'countries': chosen})
    else:
        for country in cfg['comparison_countries']:
            df_c = df[(df['country_name'] == country) & (df['quant_inc'].isin(quintile_order))]
            if len(df_c) >= 5:
                ordered_countries.append(country)

    # EU-27 aggregate panel
    if cfg.get('include_eu27_agg'):
        df_eu27 = df[(df['country_name'] == 'EU27') & (df['quant_inc'].isin(quintile_order))]
        if len(df_eu27) >= 5:
            ordered_countries.append('EU27')
        else:
            print(f"    WARNING: EU27 has insufficient quintile data — skipped")

    if not ordered_countries:
        print("    No countries with data")
        return

    n_countries = len(ordered_countries)
    fig, axes = plt.subplots(1, n_countries, figsize=(3.2 * n_countries, 5.5),
                             squeeze=False, sharey=True)

    # Global y-range
    all_vals = []
    for country in ordered_countries:
        df_c = df[(df['country_name'] == country) & (df['quant_inc'].isin(quintile_order))]
        all_vals.extend(df_c['value'].tolist())
    y_max = max(all_vals) * 1.18 if all_vals else 50

    for col_idx, country in enumerate(ordered_countries):
        ax = axes[0][col_idx]

        # Determine color
        cluster_label = None
        if cfg['show_clusters']:
            for cluster in selected_clusters:
                if country in cluster['countries']:
                    cluster_label = cluster['label']
                    break
        if country == 'EU27':
            cluster_label = 'EU-27'
        color = CLUSTER_COLORS.get(cluster_label, COUNTRY_COLOR_MAP.get(country, '#8dd3c7'))

        df_c = df[(df['country_name'] == country) & (df['quant_inc'].isin(quintile_order))].copy()
        df_c['quant_inc'] = pd.Categorical(df_c['quant_inc'], categories=quintile_order, ordered=True)
        df_c = df_c.sort_values('quant_inc')

        x = np.arange(len(df_c))
        bars = ax.bar(x, df_c['value'], color=color, edgecolor='white', linewidth=1.2, width=0.7)

        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

        code = CODE_MAP.get(country, COUNTRY_TO_ISO.get(country, country))
        ax.set_title(f'{country} ({code})', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels[:len(df_c)], fontsize=8)
        ax.set_ylim(0, y_max)
        ax.grid(axis='y', alpha=0.2)
        ax.set_facecolor('white')

        if col_idx == 0:
            ax.set_ylabel('Persons owning real estate (%)', fontsize=10, fontweight='bold')

    subtitle = 'New 4 Clusters + EU-27' if cfg['show_clusters'] else cfg['title_suffix']
    fig.suptitle(
        f'Persons Owning Real Estate Other Than Main Residence ({latest_year})\n'
        f'by Income Quintile — {subtitle}',
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # ---- Cluster labels & separators (only when show_clusters) ----
    if cfg['show_clusters']:
        col_offset = 0
        for cluster in selected_clusters:
            cols_in = [ordered_countries.index(c) for c in cluster['countries'] if c in ordered_countries]
            if not cols_in:
                col_offset += 0
                continue
            left_pos = axes[0][cols_in[0]].get_position()
            right_pos = axes[0][cols_in[-1]].get_position()
            center_x = (left_pos.x0 + right_pos.x1) / 2
            fig.text(center_x, 0.96, cluster['label'], ha='center', va='bottom',
                     fontsize=11, fontweight='bold', fontstyle='italic',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#cccccc'))

        col_offset = 0
        for i, cluster in enumerate(selected_clusters[:-1]):
            n_in = sum(1 for c in cluster['countries'] if c in ordered_countries)
            col_offset += n_in
            if col_offset < n_countries:
                right_pos = axes[0][col_offset - 1].get_position()
                left_pos = axes[0][col_offset].get_position()
                line_x = (right_pos.x1 + left_pos.x0) / 2
                fig.add_artist(plt.Line2D([line_x, line_x], [0.02, 0.93],
                                          transform=fig.transFigure, color='#888888',
                                          linewidth=1.5, linestyle='--'))

    # EU-27 panel separator
    if 'EU27' in ordered_countries and len(ordered_countries) > 1:
        eu_col = ordered_countries.index('EU27')
        if eu_col > 0:
            right_pos = axes[0][eu_col - 1].get_position()
            left_pos = axes[0][eu_col].get_position()
            line_x = (right_pos.x1 + left_pos.x0) / 2
            fig.add_artist(plt.Line2D([line_x, line_x], [0.02, 0.93],
                                      transform=fig.transFigure, color='#6a3d9a',
                                      linewidth=2.0, linestyle='-'))
        eu_pos = axes[0][eu_col].get_position()
        eu_center = (eu_pos.x0 + eu_pos.x1) / 2
        fig.text(eu_center, 0.96, 'EU-27', ha='center', va='bottom',
                 fontsize=11, fontweight='bold', fontstyle='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#efe6ff', edgecolor='#6a3d9a'))

    # Save
    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_real_estate_cluster_quintiles.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    # Excel
    rows = []
    for country in ordered_countries:
        df_c = df[(df['country_name'] == country) & (df['quant_inc'].isin(quintile_order))].copy()
        df_c['quant_inc'] = pd.Categorical(df_c['quant_inc'], categories=quintile_order, ordered=True)
        df_c = df_c.sort_values('quant_inc')
        label = 'EU-27' if country == 'EU27' else country
        row = {'Country': label}
        for sl, val in zip(short_labels, df_c['value'].tolist()):
            row[sl] = round(val, 2)
        rows.append(row)
    _save_excel(pd.DataFrame(rows),
                os.path.join(out_dir, f'{cfg["prefix"]}_real_estate_cluster_quintiles.xlsx'),
                sheet_name='Real Estate by Quintile')


# ===================================================================
# VISUAL 2: Tenure-status countries map (choropleth)
#   (adapted from eurostat_analysis_swiss.py create_tenure_status_countries_map
#    + plot_functions.py plot_europe_map)
# ===================================================================
def plot_tenure_countries_map(cfg, out_dir):
    print("  [2] Tenure status countries map...")
    df = load_tenure_status_data()
    if df is None:
        return

    df_countries = df[(df['country_name'] != 'EU27') & (df['tenure'] == 'Owner')].copy()
    if 'incgrp' in df_countries.columns:
        df_countries = df_countries[df_countries['incgrp'] == 'Total'].copy()
    if 'hhtyp' in df_countries.columns:
        df_countries = df_countries[df_countries['hhtyp'] == 'Total'].copy()
    if df_countries.empty:
        return

    df_countries['geo'] = df_countries['country_name'].map(COUNTRY_TO_ISO)
    df_countries['year'] = df_countries['TIME_PERIOD'].astype(int)
    df_countries['value'] = pd.to_numeric(df_countries['value'], errors='coerce')
    df_countries = df_countries.dropna(subset=['geo', 'year', 'value']).copy()

    # Use each country's last available year (not just the global max)
    latest_year = int(df_countries['year'].max())
    idx_last = df_countries.groupby('geo')['year'].idxmax()
    df_plot = df_countries.loc[idx_last].copy()
    df_plot = df_plot.drop_duplicates(subset=['geo'], keep='last').copy()
    if df_plot.empty:
        return

    # Track countries whose data is older than the global latest year
    older_countries = df_plot[df_plot['year'] < latest_year][['geo', 'year']].copy()
    iso_to_name = {v: k for k, v in COUNTRY_TO_ISO.items()}
    older_disclaimer = {
        row['geo']: (iso_to_name.get(row['geo'], row['geo']), int(row['year']))
        for _, row in older_countries.iterrows()
    }
    if older_disclaimer:
        print(f"    Countries with older data: "
              + ", ".join(f"{v[0]} ({v[1]})" for v in older_disclaimer.values()))

    # Study countries for this report
    study_iso = [COUNTRY_TO_ISO.get(c, '') for c in cfg['country_list']]
    study_iso = [c for c in study_iso if c]

    if not os.path.exists(_WORLD_SHP):
        print(f"    Shapefile not found: {_WORLD_SHP}")
        return

    world = gpd.read_file(_WORLD_SHP)
    european_countries = [
        'AD', 'AL', 'AT', 'BA', 'BE', 'BG', 'BY', 'CH', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES',
        'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE', 'IS', 'IT', 'LI', 'LT', 'LU', 'LV', 'MC',
        'MD', 'ME', 'MK', 'MT', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'RU', 'SE', 'SI', 'SK',
        'SM', 'TR', 'UA', 'VA', 'XK',
    ]
    europe = world[
        (world['CONTINENT'] == 'Europe') |
        (world['ISO_A2_EH'].isin(european_countries))
    ].copy()
    additional = world[
        world['NAME'].isin(['Turkey', 'Cyprus', 'Russia', 'Kazakhstan']) |
        world['ISO_A2_EH'].isin(['TR', 'CY', 'RU', 'KZ'])
    ].copy()
    europe = pd.concat([europe, additional]).drop_duplicates(subset=['ISO_A2_EH'])

    # Merge data
    df_map = df_plot[['geo', 'value', 'year']].copy()
    df_map['geo'] = df_map['geo'].replace('UK', 'GB')
    df_map.rename(columns={'geo': 'ISO_A2_EH'}, inplace=True)
    europe = europe.merge(df_map, on='ISO_A2_EH', how='left')

    study_iso_mapped = [c.replace('UK', 'GB') for c in study_iso]
    europe['is_study'] = europe['ISO_A2_EH'].isin(study_iso_mapped)
    europe = europe.to_crs(epsg=3035)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Study countries with data
    study_geo = europe[europe['is_study']].copy()
    values = study_geo.loc[study_geo['value'].notna(), 'value']
    if not values.empty:
        norm = mcolors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
        study_geo.plot(column='value', cmap='YlOrRd', linewidth=0.6, ax=ax, edgecolor='black',
                       legend=False, norm=norm,
                       missing_kwds={'color': 'lightgrey', 'label': 'Missing values'})
        sm = cm_mod.ScalarMappable(norm=norm, cmap='YlOrRd')
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(f'Owner-Occupied Dwellings (%) ({latest_year})', fontsize=10)
        cbar.ax.tick_params(labelsize=9)
    else:
        study_geo.plot(color='lightgrey', linewidth=0.6, ax=ax, edgecolor='black')

    # Non-study countries
    non_study = europe[~europe['is_study']].copy()
    non_study.plot(ax=ax, color='white', linewidth=0.6, edgecolor='black', hatch='///', alpha=0.4)

    # Legend
    legend_elements = []
    if study_geo['value'].isna().any():
        legend_elements.append((
            mpatches.Rectangle((0, 0), 1, 1, facecolor='lightgrey', edgecolor='black', linewidth=0.6),
            'Missing values'))
    legend_elements.append((
        mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch='///',
                           linewidth=0.3, alpha=0.4),
        'Non-study regions'))
    ax.legend([e[0] for e in legend_elements], [e[1] for e in legend_elements],
              loc='lower left', fontsize=9, fancybox=False, framealpha=1.0,
              edgecolor='black', facecolor='white')

    # Increase legend font size (matching original)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(11)

    ax.set_xlim(2200000, 6600000)
    ax.set_ylim(1200000, 5800000)
    ax.set_axis_off()

    # Disclaimer for countries with older data
    study_older = {k: v for k, v in older_disclaimer.items()
                   if k in study_iso or k.replace('UK', 'GB') in study_iso_mapped}
    if study_older:
        disclaimer_lines = [f"{v[0]}: {v[1]}" for v in sorted(study_older.values())]
        disclaimer_text = "Data not from " + str(latest_year) + ":\n" + "\n".join(disclaimer_lines)
        ax.text(0.99, 0.02, disclaimer_text, transform=ax.transAxes,
                fontsize=8, va='bottom', ha='right', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffbe6',
                          edgecolor='#ccaa00', alpha=0.9))

    ax.set_title(f'Owner-Occupied Dwellings ({latest_year}) {cfg["title_suffix"]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(out_dir, f'{cfg["prefix"]}_tenure_status_countries_map.png')
    _save_fig(fig, path)
    plt.close(fig)

    # Excel
    iso_to_c = {v: k for k, v in COUNTRY_TO_ISO.items()}
    df_excel = df_plot[['geo', 'value', 'year']].copy()
    df_excel['country_name'] = df_excel['geo'].map(iso_to_c)
    df_excel = df_excel.dropna(subset=['country_name']).sort_values('country_name')
    _save_excel(
        df_excel[['country_name', 'geo', 'year', 'value']].rename(
            columns={'country_name': 'Country', 'geo': 'ISO Code', 'year': 'Year', 'value': 'Owner (%)'}
        ),
        os.path.join(out_dir, f'{cfg["prefix"]}_tenure_status_countries_map.xlsx'),
    )


# ===================================================================
# Main
# ===================================================================
def generate_report(report_key):
    cfg = REPORT_CONFIGS[report_key]
    print(f"\n{'='*60}")
    print(f"Generating ownership report: {cfg['prefix']} {cfg['title_suffix']}")
    print(f"{'='*60}")

    out_dir = os.path.join(OUTPUT_BASE, cfg['prefix'])
    os.makedirs(out_dir, exist_ok=True)

    plot_real_estate_cluster_quintiles(cfg, out_dir)
    plot_tenure_countries_map(cfg, out_dir)

    print(f"\n  All outputs saved to: {out_dir}")


def main():
    for report_key in ['rep_eu', 'rep_ewbi', 'rep_fr', 'rep_ch']:
        generate_report(report_key)
    print("\nDone.")


if __name__ == '__main__':
    main()
