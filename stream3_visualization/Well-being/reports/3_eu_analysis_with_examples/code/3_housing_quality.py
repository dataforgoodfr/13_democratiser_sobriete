"""
3_housing_quality.py — Housing & energy quality indicators for 4 reports.

Produces one visual per report:
  A 2×2 grid (4 subplots, one per indicator) with bar plots showing
  hierarchy: cluster > country > decile (D1–D10).

Indicators:
  HE-SILC-2  → Behind on Utility Bills
  HQ-SILC-1  → Overcrowded Dwelling
  HQ-SILC-2  → Cannot Keep Dwelling Comfortably Warm
  HQ-SILC-3  → Cannot Keep Dwelling Comfortably Cool

Reports:
  rep_eu   : EU-27 countries, cluster grouping, + EU-27 aggregate
  rep_ewbi : EU-27 + EFTA, cluster grouping, + EU-27 aggregate
  rep_fr   : France focus, comparison countries (no cluster grouping)
  rep_ch   : Switzerland focus, comparison countries (no cluster grouping)

All outputs: PNG + SVG + Excel.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATA_PATH = os.path.join(
    BASE_DIR, '..', '..', 'output', 'ewbi_master_aggregated.csv',
)
OUTPUT_BASE = os.path.join(BASE_DIR, 'outputs', 'graphs', 'Housing_Quality')
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ---------------------------------------------------------------------------
# Country definitions
# ---------------------------------------------------------------------------
COUNTRY_NAME_MAP = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CH': 'Switzerland',
    'CY': 'Cyprus', 'CZ': 'Czech Republic', 'DE': 'Germany', 'DK': 'Denmark',
    'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland',
    'FR': 'France', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland',
    'IS': 'Iceland', 'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg',
    'LV': 'Latvia', 'MT': 'Malta', 'NL': 'Netherlands', 'NO': 'Norway',
    'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia',
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia', 'UK': 'United Kingdom',
    'EU-27': 'EU-27',
}

EU27_CODES = {
    'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES',
    'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT',
    'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK',
}
EFTA_CODES = {'CH', 'NO', 'IS'}

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
INDICATORS = {
    'HE-SILC-2': 'Behind on Utility Bills',
    'HQ-SILC-1': 'Overcrowded Dwelling',
    'HQ-SILC-2': 'Cannot Keep Dwelling Comfortably Warm',
    'HQ-SILC-3': 'Cannot Keep Dwelling Comfortably Cool',
}

# ---------------------------------------------------------------------------
# Cluster definitions (from user specification)
# ---------------------------------------------------------------------------
CLUSTERS = [
    {
        'id': 0,
        'label': 'Cluster 0 – Low performer / Low EWBI',
        'short': 'C0',
        'color': '#fb8072',
        'countries': {'CY', 'FR', 'EL', 'MT', 'PT', 'ES', 'UK'},
    },
    {
        'id': 1,
        'label': 'Cluster 1 – Low performer / High EWBI',
        'short': 'C1',
        'color': '#fdb462',
        'countries': {'AT', 'BE', 'CH', 'DK', 'FI', 'IE', 'IT', 'LU', 'NO', 'NL'},
    },
    {
        'id': 2,
        'label': 'Cluster 2 – High performer / Low EWBI',
        'short': 'C2',
        'color': '#8dd3c7',
        'countries': {'BG', 'HR', 'HU', 'LV', 'LT', 'RS', 'RO'},
    },
    {
        'id': 3,
        'label': 'Cluster 3 – High performer / High EWBI',
        'short': 'C3',
        'color': '#80b1d3',
        'countries': {'CZ', 'EE', 'DE', 'IS', 'PL', 'SK', 'SI', 'SE'},
    },
]

CLUSTER_COLOR_BY_COUNTRY = {}
CLUSTER_ID_BY_COUNTRY = {}
for cl in CLUSTERS:
    for cc in cl['countries']:
        CLUSTER_COLOR_BY_COUNTRY[cc] = cl['color']
        CLUSTER_ID_BY_COUNTRY[cc] = cl['id']
CLUSTER_COLOR_BY_COUNTRY['EU-27'] = '#6a3d9a'

# ---------------------------------------------------------------------------
# Report configurations
# ---------------------------------------------------------------------------
REPORT_CONFIGS = {
    'rep_eu': {
        'prefix': 'rep_eu',
        'title_suffix': '(EU-27)',
        'country_filter': EU27_CODES | {'EU-27'},
        'comparison_countries': ['FR', 'ES', 'BE', 'NL', 'LT', 'HR', 'DE', 'SE'],
        'include_eu27_agg': True,
        'show_clusters': True,
    },
    'rep_ewbi': {
        'prefix': 'rep_ewbi',
        'title_suffix': '(EU-27 + EFTA + UK + Serbia)',
        'country_filter': EU27_CODES | EFTA_CODES | {'UK', 'RS', 'EU-27'},
        'comparison_countries': ['FR', 'ES', 'BE', 'NL', 'LT', 'HR', 'DE', 'SE',
                                 'CH', 'NO', 'IS', 'UK', 'RS'],
        'include_eu27_agg': True,
        'show_clusters': True,
    },
    'rep_fr': {
        'prefix': 'rep_fr',
        'title_suffix': '(France)',
        'country_filter': EU27_CODES | EFTA_CODES | {'EU-27'},
        'comparison_countries': ['FR', 'ES', 'IT', 'CH', 'DE', 'BE'],
        'include_eu27_agg': True,
        'show_clusters': False,
    },
    'rep_ch': {
        'prefix': 'rep_ch',
        'title_suffix': '(Switzerland)',
        'country_filter': EU27_CODES | EFTA_CODES | {'EU-27'},
        'comparison_countries': ['CH', 'FR', 'IT', 'DE', 'AT'],
        'include_eu27_agg': True,
        'show_clusters': False,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _country_label(cc):
    name = COUNTRY_NAME_MAP.get(cc, cc)
    if cc == 'EU-27':
        return 'EU-27'
    return f'{name} ({cc})'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Filter to level 3, housing/energy indicators, numeric deciles
    df = df[
        (df['Level'] == 3) &
        (df['Primary and raw data'].isin(INDICATORS.keys()))
    ].copy()

    df['Decile_num'] = pd.to_numeric(df['Decile'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Decile_num', 'Year', 'Value'])
    df['Decile_num'] = df['Decile_num'].astype(int)
    df = df[df['Decile_num'].between(1, 10)].copy()

    # Keep latest year per country × indicator
    latest = df.groupby(['Country', 'Primary and raw data'])['Year'].max().reset_index()
    latest.columns = ['Country', 'Primary and raw data', 'Latest_Year']
    df = df.merge(latest, on=['Country', 'Primary and raw data'])
    df = df[df['Year'] == df['Latest_Year']].copy()

    print(f"  Loaded {len(df)} rows for {df['Country'].nunique()} countries, "
          f"{df['Primary and raw data'].nunique()} indicators")
    return df


# ---------------------------------------------------------------------------
# Build ordered country list for a report
# ---------------------------------------------------------------------------
def _build_country_order(cfg, df):
    """
    Returns a list of country codes in the desired order.
    - show_clusters=True  → grouped by cluster, then alphabetically within cluster,
                            then EU-27 at the end.
    - show_clusters=False → comparison_countries order, then EU-27 at the end.
    """
    comparison = cfg['comparison_countries']
    available = set(df['Country'].unique())

    if cfg['show_clusters']:
        ordered = []
        for cl in CLUSTERS:
            # Pick comparison countries that belong to this cluster
            cl_countries = [cc for cc in comparison if cc in cl['countries'] and cc in available]
            # Also include other countries from this cluster that are in scope
            scope = cfg['country_filter']
            others = sorted([
                cc for cc in cl['countries']
                if cc in available and cc in scope and cc not in cl_countries
            ])
            ordered.extend(cl_countries + others)
        # Remove duplicates while preserving order
        seen = set()
        deduped = []
        for cc in ordered:
            if cc not in seen:
                seen.add(cc)
                deduped.append(cc)
        ordered = deduped
    else:
        ordered = [cc for cc in comparison if cc in available]

    # Add EU-27 at the end if requested and available
    if cfg.get('include_eu27_agg') and 'EU-27' in available:
        if 'EU-27' not in ordered:
            ordered.append('EU-27')

    return ordered


# ---------------------------------------------------------------------------
# Main visual: 2×2 grid, bar plots, hierarchy cluster > country > decile
# ---------------------------------------------------------------------------
def plot_housing_quality_grid(cfg, df, out_dir):
    """
    Create a 2×2 figure with 4 subplots (one per indicator).
    Each subplot shows grouped bar chart: bars are grouped by country,
    country groups are ordered by cluster.

    Within each country: 10 bars (D1–D10) in the cluster's color.
    Countries separated by small gaps; cluster groups separated by larger gaps
    and annotated with labels.
    """
    print(f"  [1] Housing quality 4-indicator grid...")

    country_order = _build_country_order(cfg, df)
    if not country_order:
        print("    No countries with data")
        return

    n_countries = len(country_order)
    n_deciles = 10
    country_gap = 1.0     # gap between countries (in bar-width units)
    cluster_gap = 2.5     # extra gap at cluster boundaries
    bar_width = 0.7

    # 4 rows stacked, very wide, minimal margins
    fig_w = max(28, n_countries * 1.6)
    fig, axes = plt.subplots(4, 1, figsize=(fig_w, 24), squeeze=False)
    axes_flat = axes.flatten()

    all_excel_rows = []

    for plot_idx, (ind_code, ind_name) in enumerate(INDICATORS.items()):
        ax = axes_flat[plot_idx]
        df_ind = df[df['Primary and raw data'] == ind_code].copy()

        # Build x positions with cluster/country gaps
        x_positions = []   # list of (country_code, decile_x_array)
        cluster_boundaries = []  # x positions of cluster separators
        cluster_label_positions = []  # (center_x, label) for cluster headers
        tick_positions = []
        tick_labels_list = []
        current_x = 0
        prev_cluster_id = None
        cluster_start_x = 0

        for i_cc, cc in enumerate(country_order):
            cl_id = CLUSTER_ID_BY_COUNTRY.get(cc, -1)

            # Add cluster gap if cluster changed (only for cluster reports)
            if cfg['show_clusters'] and prev_cluster_id is not None and cl_id != prev_cluster_id:
                # Record label position for the previous cluster
                cluster_label_positions.append(
                    ((cluster_start_x + current_x - country_gap) / 2, prev_cluster_id)
                )
                cluster_boundaries.append(current_x - country_gap / 2)
                current_x += cluster_gap
                cluster_start_x = current_x
            elif current_x > 0:
                current_x += country_gap

            if prev_cluster_id is None and cfg['show_clusters']:
                cluster_start_x = current_x

            prev_cluster_id = cl_id

            decile_xs = np.arange(n_deciles) * (bar_width + 0.05) + current_x
            x_positions.append((cc, decile_xs))

            # ISO 2-letter code as x-tick label
            center = (decile_xs[0] + decile_xs[-1]) / 2
            tick_positions.append(center)
            tick_labels_list.append(cc)  # 2-letter code

            current_x = decile_xs[-1] + bar_width + 0.05

        # Final cluster label
        if cfg['show_clusters'] and prev_cluster_id is not None:
            cluster_label_positions.append(
                ((cluster_start_x + current_x) / 2, prev_cluster_id)
            )

        # Compute y-range across all countries for this indicator
        all_vals = []
        for cc in country_order:
            cdf = df_ind[df_ind['Country'] == cc]
            all_vals.extend(cdf['Value'].dropna().tolist())
        y_max = max(all_vals) * 1.12 if all_vals else 50

        # Plot bars
        for cc, decile_xs in x_positions:
            cdf = df_ind[df_ind['Country'] == cc].sort_values('Decile_num')
            color = CLUSTER_COLOR_BY_COUNTRY.get(cc, '#999999')

            vals = []
            for d in range(1, 11):
                v = cdf[cdf['Decile_num'] == d]['Value']
                val = v.iloc[0] if len(v) else np.nan
                vals.append(val)

            bars = ax.bar(decile_xs, vals, width=bar_width, color=color,
                          edgecolor='white', linewidth=0.4)

            # Value labels on D1 and D10 only
            for i, (bar, val) in enumerate(zip(bars, vals)):
                if not pd.isna(val) and i in (0, 9):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + y_max * 0.005,
                            f'{val:.1f}', ha='center', va='bottom',
                            fontsize=5, fontweight='bold')

            # Collect for Excel
            latest_year = int(cdf['Year'].iloc[0]) if len(cdf) else ''
            for d, val in zip(range(1, 11), vals):
                all_excel_rows.append({
                    'Indicator': ind_code,
                    'Indicator_Name': ind_name,
                    'Country': cc,
                    'Country_Name': COUNTRY_NAME_MAP.get(cc, cc),
                    'Year': latest_year,
                    'Decile': d,
                    'Value': round(val, 2) if not pd.isna(val) else None,
                })

        # Cluster separator lines
        if cfg['show_clusters']:
            for bx in cluster_boundaries:
                ax.axvline(x=bx, color='#888888', linewidth=1.0, linestyle='--', alpha=0.5)

        # Cluster top labels
        if cfg['show_clusters'] and plot_idx == 0:
            for cx, cl_id in cluster_label_positions:
                cl_obj = CLUSTERS[cl_id]
                ax.text(cx, y_max * 1.08, cl_obj['label'],
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        fontstyle='italic',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=cl_obj['color'],
                                  edgecolor='#cccccc', alpha=0.5))

        # EU-27 separator
        if 'EU-27' in country_order and len(country_order) > 1:
            eu_idx = country_order.index('EU-27')
            if eu_idx > 0:
                _, prev_xs = x_positions[eu_idx - 1]
                _, eu_xs = x_positions[eu_idx]
                sep_x = (prev_xs[-1] + eu_xs[0]) / 2
                ax.axvline(x=sep_x, color='#6a3d9a', linewidth=1.5, linestyle='-', alpha=0.6)

        ax.set_title(ind_name, fontsize=11, fontweight='bold', pad=4)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, fontsize=8, fontweight='bold',
                           rotation=0, ha='center')
        ax.set_xlim(-1, current_x + 0.5)
        ax.set_ylim(0, y_max)
        ax.grid(axis='y', alpha=0.2)
        ax.set_facecolor('white')
        ax.set_ylabel('% of households', fontsize=9, fontweight='bold')
        # Reduce tick padding
        ax.tick_params(axis='x', pad=2)
        ax.tick_params(axis='y', pad=2)

        # Disclaimer for countries with older data than the most common year
        year_by_cc = df_ind.groupby('Country')['Year'].max()
        year_by_cc = year_by_cc[year_by_cc.index.isin(country_order)]
        if not year_by_cc.empty:
            modal_year = int(year_by_cc.mode().iloc[0])
            older = year_by_cc[year_by_cc < modal_year]
            if not older.empty:
                parts = [f"{COUNTRY_NAME_MAP.get(cc, cc)}: {int(yr)}"
                         for cc, yr in sorted(older.items())]
                disclaimer = f"Data not from {modal_year}: " + ", ".join(parts)
                ax.text(0.99, 0.97, disclaimer, transform=ax.transAxes,
                        fontsize=6, va='top', ha='right', fontstyle='italic',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6',
                                  edgecolor='#ccaa00', alpha=0.85))

    # Suptitle
    fig.suptitle(
        f'Energy & Housing Quality Indicators by Income Decile\n'
        f'{cfg["title_suffix"]}',
        fontsize=14, fontweight='bold', y=1.0,
    )

    # Legend: cluster colors (or just country colors for focus reports)
    legend_handles = []
    if cfg['show_clusters']:
        for cl in CLUSTERS:
            if any(cc in cl['countries'] for cc in country_order):
                legend_handles.append(
                    plt.Rectangle((0, 0), 1, 1, fc=cl['color'], ec='white',
                                  label=cl['label'])
                )
        if 'EU-27' in country_order:
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, fc='#6a3d9a', ec='white', label='EU-27')
            )
    else:
        for cc in country_order:
            color = CLUSTER_COLOR_BY_COUNTRY.get(cc, '#999999')
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, fc=color, ec='white',
                              label=f'{COUNTRY_NAME_MAP.get(cc, cc)} ({cc})')
            )

    fig.legend(handles=legend_handles, loc='lower center',
               ncol=min(len(legend_handles), 6), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    plt.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.04,
                        hspace=0.22)

    # Save PNG + SVG
    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_housing_quality_4_indicators.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    # Save Excel
    excel_df = pd.DataFrame(all_excel_rows)
    sheets = {}
    # Long format
    sheets['Long_Format'] = excel_df
    # Wide: rows = (indicator, country), columns = D1–D10
    if not excel_df.empty:
        wide = excel_df.pivot_table(
            index=['Indicator', 'Indicator_Name', 'Country', 'Country_Name', 'Year'],
            columns='Decile', values='Value', aggfunc='first',
        )
        wide.columns = [f'D{int(c)}' for c in wide.columns]
        wide = wide.reset_index()
        sheets['Wide_Format'] = wide
    xlsx_path = os.path.join(out_dir, f'{cfg["prefix"]}_housing_quality_4_indicators.xlsx')
    _save_excel(sheets, xlsx_path)


# ---------------------------------------------------------------------------
# Helper: plot one row of countries with decile bars
# ---------------------------------------------------------------------------
def _plot_country_bars(ax, countries, df_ind, ind_code, ind_name, cfg, y_max,
                       bar_width=0.7, country_gap=1.0, cluster_gap=2.5):
    """Plot decile bars for a list of countries on the given axes.
    Returns list of excel row dicts."""
    n_deciles = 10
    x_positions = []
    cluster_boundaries = []
    cluster_label_positions = []
    tick_positions = []
    tick_labels_list = []
    current_x = 0
    prev_cluster_id = None
    cluster_start_x = 0
    excel_rows = []

    for cc in countries:
        cl_id = CLUSTER_ID_BY_COUNTRY.get(cc, -1)

        if cfg['show_clusters'] and prev_cluster_id is not None and cl_id != prev_cluster_id:
            cluster_label_positions.append(
                ((cluster_start_x + current_x - country_gap) / 2, prev_cluster_id)
            )
            cluster_boundaries.append(current_x - country_gap / 2)
            current_x += cluster_gap
            cluster_start_x = current_x
        elif current_x > 0:
            current_x += country_gap

        if prev_cluster_id is None and cfg['show_clusters']:
            cluster_start_x = current_x

        prev_cluster_id = cl_id

        decile_xs = np.arange(n_deciles) * (bar_width + 0.05) + current_x
        x_positions.append((cc, decile_xs))

        center = (decile_xs[0] + decile_xs[-1]) / 2
        tick_positions.append(center)
        tick_labels_list.append(cc)

        current_x = decile_xs[-1] + bar_width + 0.05

    if cfg['show_clusters'] and prev_cluster_id is not None:
        cluster_label_positions.append(
            ((cluster_start_x + current_x) / 2, prev_cluster_id)
        )

    # Plot bars
    for cc, decile_xs in x_positions:
        cdf = df_ind[df_ind['Country'] == cc].sort_values('Decile_num')
        color = CLUSTER_COLOR_BY_COUNTRY.get(cc, '#999999')

        vals = []
        for d in range(1, 11):
            v = cdf[cdf['Decile_num'] == d]['Value']
            val = v.iloc[0] if len(v) else np.nan
            vals.append(val)

        bars = ax.bar(decile_xs, vals, width=bar_width, color=color,
                      edgecolor='white', linewidth=0.4)

        for i, (bar, val) in enumerate(zip(bars, vals)):
            if not pd.isna(val) and i in (0, 9):
                ax.text(bar.get_x() + bar.get_width() / 2, val + y_max * 0.005,
                        f'{val:.1f}', ha='center', va='bottom',
                        fontsize=5, fontweight='bold')

        latest_year = int(cdf['Year'].iloc[0]) if len(cdf) else ''
        for d, val in zip(range(1, 11), vals):
            excel_rows.append({
                'Indicator': ind_code,
                'Indicator_Name': ind_name,
                'Country': cc,
                'Country_Name': COUNTRY_NAME_MAP.get(cc, cc),
                'Year': latest_year,
                'Decile': d,
                'Value': round(val, 2) if not pd.isna(val) else None,
            })

    # Cluster separators
    if cfg['show_clusters']:
        for bx in cluster_boundaries:
            ax.axvline(x=bx, color='#888888', linewidth=1.0, linestyle='--', alpha=0.5)

    # Cluster labels
    if cfg['show_clusters']:
        for cx, cl_id in cluster_label_positions:
            cl_obj = CLUSTERS[cl_id]
            ax.text(cx, y_max * 1.08, cl_obj['label'],
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=cl_obj['color'],
                              edgecolor='#cccccc', alpha=0.5))

    # EU-27 separator
    if 'EU-27' in countries and len(countries) > 1:
        eu_idx = countries.index('EU-27')
        if eu_idx > 0:
            _, prev_xs = x_positions[eu_idx - 1]
            _, eu_xs = x_positions[eu_idx]
            sep_x = (prev_xs[-1] + eu_xs[0]) / 2
            ax.axvline(x=sep_x, color='#6a3d9a', linewidth=1.5, linestyle='-', alpha=0.6)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_list, fontsize=8, fontweight='bold')
    ax.set_xlim(-1, current_x + 0.5)
    ax.set_ylim(0, y_max)
    ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')
    ax.set_ylabel('% of households', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)

    # Disclaimer for older data
    year_by_cc = df_ind.groupby('Country')['Year'].max()
    year_by_cc = year_by_cc[year_by_cc.index.isin(countries)]
    if not year_by_cc.empty:
        modal_year = int(year_by_cc.mode().iloc[0])
        older = year_by_cc[year_by_cc < modal_year]
        if not older.empty:
            parts = [f"{COUNTRY_NAME_MAP.get(cc, cc)}: {int(yr)}"
                     for cc, yr in sorted(older.items())]
            disclaimer = f"Data not from {modal_year}: " + ", ".join(parts)
            ax.text(0.99, 0.97, disclaimer, transform=ax.transAxes,
                    fontsize=6, va='top', ha='right', fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6',
                              edgecolor='#ccaa00', alpha=0.85))

    return excel_rows


# ---------------------------------------------------------------------------
# Per-indicator visual: 2 rows (C0+C1 top, C2+C3+EU-27 bottom)
# ---------------------------------------------------------------------------
def plot_indicator_two_rows(ind_code, ind_name, cfg, df, country_order, out_dir):
    """Single indicator, two subplot rows with shared y-axis."""
    print(f"  [{ind_code}] {ind_name}...")
    df_ind = df[df['Primary and raw data'] == ind_code].copy()
    if df_ind.empty:
        print(f"    No data for {ind_code}")
        return

    top_countries = [cc for cc in country_order
                     if CLUSTER_ID_BY_COUNTRY.get(cc, -1) in (0, 1)]
    bottom_countries = [cc for cc in country_order
                        if CLUSTER_ID_BY_COUNTRY.get(cc, -1) in (2, 3) or cc == 'EU-27']

    # Shared y-max across both rows
    all_vals = df_ind[df_ind['Country'].isin(country_order)]['Value'].dropna().tolist()
    y_max = max(all_vals) * 1.12 if all_vals else 50

    n_max = max(len(top_countries), len(bottom_countries))
    fig_w = max(24, n_max * 1.8)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(fig_w, 14))

    rows_top = _plot_country_bars(ax_top, top_countries, df_ind,
                                  ind_code, ind_name, cfg, y_max)
    rows_bot = _plot_country_bars(ax_bot, bottom_countries, df_ind,
                                  ind_code, ind_name, cfg, y_max)

    fig.suptitle(
        f'{ind_code} \u2014 {ind_name} by Income Decile\n{cfg["title_suffix"]}',
        fontsize=14, fontweight='bold', y=1.02,
    )

    # Legend
    legend_handles = []
    for cl in CLUSTERS:
        if any(cc in cl['countries'] for cc in country_order):
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, fc=cl['color'], ec='white',
                              label=cl['label'])
            )
    if 'EU-27' in country_order:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc='#6a3d9a', ec='white', label='EU-27')
        )
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=min(len(legend_handles), 6), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(left=0.03, right=0.99, top=0.94, bottom=0.05, hspace=0.25)

    safe_name = ind_code.lower().replace('-', '_')
    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_{safe_name}.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    # Excel
    all_excel = rows_top + rows_bot
    excel_df = pd.DataFrame(all_excel)
    sheets = {'Long_Format': excel_df}
    if not excel_df.empty:
        wide = excel_df.pivot_table(
            index=['Indicator', 'Indicator_Name', 'Country', 'Country_Name', 'Year'],
            columns='Decile', values='Value', aggfunc='first',
        )
        wide.columns = [f'D{int(c)}' for c in wide.columns]
        wide = wide.reset_index()
        sheets['Wide_Format'] = wide
    xlsx_path = os.path.join(out_dir, f'{cfg["prefix"]}_{safe_name}.xlsx')
    _save_excel(sheets, xlsx_path)


# ---------------------------------------------------------------------------
# Per-indicator visual: single row (for focus reports with few countries)
# ---------------------------------------------------------------------------
def plot_indicator_single_row(ind_code, ind_name, cfg, df, country_order, out_dir):
    """Single indicator, one subplot row — used for FR / CH focus reports."""
    print(f"  [{ind_code}] {ind_name}...")
    df_ind = df[df['Primary and raw data'] == ind_code].copy()
    if df_ind.empty:
        print(f"    No data for {ind_code}")
        return

    all_vals = df_ind[df_ind['Country'].isin(country_order)]['Value'].dropna().tolist()
    y_max = max(all_vals) * 1.12 if all_vals else 50

    fig_w = max(16, len(country_order) * 2.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 7))

    excel_rows = _plot_country_bars(ax, country_order, df_ind,
                                    ind_code, ind_name, cfg, y_max)

    fig.suptitle(
        f'{ind_code} \u2014 {ind_name} by Income Decile\n{cfg["title_suffix"]}',
        fontsize=14, fontweight='bold', y=1.02,
    )

    # Legend: one entry per country
    legend_handles = []
    for cc in country_order:
        color = CLUSTER_COLOR_BY_COUNTRY.get(cc, '#999999')
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, ec='white',
                          label=f'{COUNTRY_NAME_MAP.get(cc, cc)} ({cc})')
        )
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=min(len(legend_handles), 6), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    plt.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.08)

    safe_name = ind_code.lower().replace('-', '_')
    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_{safe_name}.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    # Excel
    excel_df = pd.DataFrame(excel_rows)
    sheets = {'Long_Format': excel_df}
    if not excel_df.empty:
        wide = excel_df.pivot_table(
            index=['Indicator', 'Indicator_Name', 'Country', 'Country_Name', 'Year'],
            columns='Decile', values='Value', aggfunc='first',
        )
        wide.columns = [f'D{int(c)}' for c in wide.columns]
        wide = wide.reset_index()
        sheets['Wide_Format'] = wide
    xlsx_path = os.path.join(out_dir, f'{cfg["prefix"]}_{safe_name}.xlsx')
    _save_excel(sheets, xlsx_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_report(report_key, df):
    cfg = REPORT_CONFIGS[report_key]
    print(f"\n{'='*60}")
    print(f"Generating housing quality report: {cfg['prefix']} {cfg['title_suffix']}")
    print(f"{'='*60}")

    out_dir = os.path.join(OUTPUT_BASE, cfg['prefix'])
    os.makedirs(out_dir, exist_ok=True)

    # Filter data to countries in scope
    df_report = df[df['Country'].isin(cfg['country_filter'])].copy()

    if cfg['show_clusters']:
        country_order = _build_country_order(cfg, df_report)
        for ind_code, ind_name in INDICATORS.items():
            plot_indicator_two_rows(ind_code, ind_name, cfg, df_report,
                                   country_order, out_dir)
    else:
        country_order = _build_country_order(cfg, df_report)
        for ind_code, ind_name in INDICATORS.items():
            plot_indicator_single_row(ind_code, ind_name, cfg, df_report,
                                     country_order, out_dir)

    print(f"\n  All outputs saved to: {out_dir}")


def main():
    df = load_data()
    for report_key in ['rep_eu', 'rep_ewbi', 'rep_fr', 'rep_ch']:
        generate_report(report_key, df)
    print("\nDone.")


if __name__ == '__main__':
    main()
