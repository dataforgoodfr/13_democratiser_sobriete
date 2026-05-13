"""
4_energy_prices.py — Energy price decomposition visuals for 4 reports.

Produces 2 visuals per report:
  1. Electricity prices by consumption band — stacked bar chart,
     countries grouped by cluster (cluster > country > consumption band).
  2. Gas prices by consumption band — same layout.

Data source: Eurostat price component CSVs (PPS).

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
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, 'external_data')
OUTPUT_BASE = os.path.join(BASE_DIR, 'outputs', 'graphs', 'Energy_Prices')
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ---------------------------------------------------------------------------
# Country mapping
# ---------------------------------------------------------------------------
GEO_TO_ISO = {
    'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
    'Cyprus': 'CY', 'Czechia': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE',
    'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'EL',
    'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE', 'Italy': 'IT',
    'Latvia': 'LV', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT',
    'Netherlands': 'NL', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT',
    'Romania': 'RO', 'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES',
    'Sweden': 'SE', 'Switzerland': 'CH', 'United Kingdom': 'UK',
    'European Union - 27 countries (from 2020)': 'EU-27',
}
ISO_TO_NAME = {v: k for k, v in GEO_TO_ISO.items()}
ISO_TO_NAME['EU-27'] = 'EU-27'
ISO_TO_NAME['EL'] = 'Greece'

EU27_CODES = {
    'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES',
    'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT',
    'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK',
}
EFTA_CODES = {'CH', 'NO', 'IS'}

# ---------------------------------------------------------------------------
# Price components
# ---------------------------------------------------------------------------
PRICE_COMPONENTS = [
    'NRG_SUP', 'NETC', 'TAX_FEE_LEV_CHRG', 'VAT',
    'TAX_RNW', 'TAX_CAP', 'TAX_ENV', 'TAX_NUC', 'OTH',
]
COMPONENT_COLORS = {
    'NRG_SUP': '#fdb462',
    'NETC': '#8dd3c7',
    'TAX_FEE_LEV_CHRG': '#bebada',
    'VAT': '#b3de69',
    'TAX_RNW': '#ffd558',
    'TAX_CAP': '#fb8072',
    'TAX_ENV': '#ffffb3',
    'TAX_NUC': '#80b1d3',
    'OTH': '#bc80bd',
}
COMPONENT_LABELS = {
    'NRG_SUP': 'Energy and supply',
    'NETC': 'Network costs',
    'TAX_FEE_LEV_CHRG': 'Taxes & charges',
    'VAT': 'Value added tax',
    'TAX_RNW': 'Renewable taxes',
    'TAX_CAP': 'Capacity taxes',
    'TAX_ENV': 'Environmental taxes',
    'TAX_NUC': 'Nuclear taxes',
    'OTH': 'Other',
}

NRG_PRC_MAP = {
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

# Consumption bands
ELEC_BANDS = ['KWH_LT1000', 'KWH1000-2499', 'KWH2500-4999',
              'KWH5000-14999', 'KWH_LE15000']
GAS_BANDS = ['GJ_LT20', 'GJ20-199', 'GJ_GE200']

CONS_MAP = {
    'Consumption of kWh - all bands': 'TOT_KWH',
    'Consumption less than 1 000 kWh - band DA': 'KWH_LT1000',
    'Consumption from 1 000 kWh to 2 499 kWh - band DB': 'KWH1000-2499',
    'Consumption from 2 500 kWh to 4 999 kWh - band DC': 'KWH2500-4999',
    'Consumption from 5 000 kWh to 14 999 kWh - band DD': 'KWH5000-14999',
    'Consumption 15 000 kWh or over - band DE': 'KWH_LE15000',
    'Consumption of GJ - all bands': 'TOT_GJ',
    'Consumption less than 20 GJ - band D1': 'GJ_LT20',
    'Consumption from 20 GJ to 199 GJ - band D2': 'GJ20-199',
    'Consumption 200 GJ or over - band D3': 'GJ_GE200',
}

BAND_SHORT_LABELS = {
    'KWH_LT1000': '<1k',
    'KWH1000-2499': '1-2.5k',
    'KWH2500-4999': '2.5-5k',
    'KWH5000-14999': '5-15k',
    'KWH_LE15000': '≥15k',
    'GJ_LT20': '<20 GJ',
    'GJ20-199': '20-199 GJ',
    'GJ_GE200': '≥200 GJ',
}

# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
CLUSTERS = [
    {
        'id': 0,
        'label': 'Cluster 0 – Low perf / Low EWBI',
        'color': '#fb8072',
        'countries': {'CY', 'FR', 'EL', 'MT', 'PT', 'ES', 'UK'},
    },
    {
        'id': 1,
        'label': 'Cluster 1 – Low perf / High EWBI',
        'color': '#fdb462',
        'countries': {'AT', 'BE', 'CH', 'DK', 'FI', 'IE', 'IT', 'LU', 'NO', 'NL'},
    },
    {
        'id': 2,
        'label': 'Cluster 2 – High perf / Low EWBI',
        'color': '#8dd3c7',
        'countries': {'BG', 'HR', 'HU', 'LV', 'LT', 'RS', 'RO'},
    },
    {
        'id': 3,
        'label': 'Cluster 3 – High perf / High EWBI',
        'color': '#80b1d3',
        'countries': {'CZ', 'EE', 'DE', 'IS', 'PL', 'SK', 'SI', 'SE'},
    },
]

CLUSTER_ID_BY_COUNTRY = {}
for cl in CLUSTERS:
    for cc in cl['countries']:
        CLUSTER_ID_BY_COUNTRY[cc] = cl['id']

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
        'title_suffix': '(EU-27 + EFTA)',
        'country_filter': EU27_CODES | EFTA_CODES | {'EU-27'},
        'comparison_countries': ['FR', 'ES', 'BE', 'NL', 'LT', 'HR', 'DE', 'SE',
                                 'CH', 'NO', 'IS'],
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
                df.to_excel(writer, sheet_name=name[:31], index=False)
        else:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  Saved: {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_energy_data(energy_type):
    """Load and prepare electricity or gas price component data."""
    file_path = os.path.join(EXTERNAL_DATA_DIR, f'price_component_{energy_type}.csv')
    if not os.path.exists(file_path):
        print(f"  ERROR: File not found: {file_path}")
        return pd.DataFrame()

    print(f"  Loading {energy_type}: {file_path}")
    df = pd.read_csv(file_path)

    # Map columns
    df['iso'] = df['geo'].map(GEO_TO_ISO)
    df['comp'] = df['nrg_prc'].map(NRG_PRC_MAP)
    df['band'] = df['nrg_cons'].map(CONS_MAP)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df['year'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')

    df = df.dropna(subset=['iso', 'comp', 'band', 'value', 'year'])

    # Keep latest year per country
    latest = df.groupby('iso')['year'].max().reset_index()
    latest.columns = ['iso', 'latest_year']
    df = df.merge(latest, on='iso')
    df = df[df['year'] == df['latest_year']].copy()

    print(f"    {len(df)} rows, {df['iso'].nunique()} countries")
    return df


# ---------------------------------------------------------------------------
# Build ordered country list
# ---------------------------------------------------------------------------
def _build_country_order(cfg, available_countries):
    comparison = cfg['comparison_countries']
    available = set(available_countries)

    if cfg['show_clusters']:
        ordered = []
        for cl in CLUSTERS:
            cl_comp = [cc for cc in comparison if cc in cl['countries'] and cc in available]
            scope = cfg['country_filter']
            others = sorted([
                cc for cc in cl['countries']
                if cc in available and cc in scope and cc not in cl_comp
            ])
            ordered.extend(cl_comp + others)
        seen = set()
        deduped = []
        for cc in ordered:
            if cc not in seen:
                seen.add(cc)
                deduped.append(cc)
        ordered = deduped
    else:
        ordered = [cc for cc in comparison if cc in available]

    if cfg.get('include_eu27_agg') and 'EU-27' in available:
        if 'EU-27' not in ordered:
            ordered.append('EU-27')

    return ordered


# ---------------------------------------------------------------------------
# Main visual: single graph, stacked bars, cluster > country > consumption
# ---------------------------------------------------------------------------
def plot_energy_prices(cfg, df, energy_type, bands, out_dir):
    """
    One graph with stacked bars.
    X-axis: country groups, within each country one bar per consumption band.
    Bars are stacked by price component.
    Countries ordered by cluster (with separators) or by comparison list.
    """
    type_label = 'Electricity' if energy_type == 'electricity' else 'Gas'
    unit_label = 'kWh' if energy_type == 'electricity' else 'GJ'
    print(f"  [{type_label}] Energy prices stacked bar chart...")

    # Filter to relevant bands
    df_plot = df[df['band'].isin(bands)].copy()
    available = df_plot['iso'].unique()
    country_order = _build_country_order(cfg, available)

    if not country_order:
        print(f"    No countries with data for {type_label}")
        return

    n_countries = len(country_order)
    n_bands = len(bands)
    band_width = 0.75
    country_gap = 1.5
    cluster_gap = 3.0

    # Build x positions
    x_all = []          # list of (iso, band_x_array)
    tick_positions = []
    tick_labels = []
    cluster_boundaries = []
    cluster_label_positions = []
    current_x = 0
    prev_cluster_id = None
    cluster_start_x = 0

    for cc in country_order:
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

        band_xs = np.arange(n_bands) * (band_width + 0.15) + current_x
        x_all.append((cc, band_xs))

        center = (band_xs[0] + band_xs[-1]) / 2
        tick_positions.append(center)
        tick_labels.append(cc)

        current_x = band_xs[-1] + band_width + 0.15

    # Final cluster label
    if cfg['show_clusters'] and prev_cluster_id is not None:
        cluster_label_positions.append(
            ((cluster_start_x + current_x) / 2, prev_cluster_id)
        )

    # Compute y-range
    y_max = 0
    for cc, band_xs in x_all:
        cc_data = df_plot[df_plot['iso'] == cc]
        for band in bands:
            bdata = cc_data[cc_data['band'] == band]
            total = bdata.loc[bdata['value'] >= 0, 'value'].sum()
            y_max = max(y_max, total)
    y_max *= 1.15

    # Figure
    fig_w = max(20, n_countries * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, 7))

    # Track which components have data (for legend)
    components_used = set()
    excel_rows = []

    for cc, band_xs in x_all:
        cc_data = df_plot[df_plot['iso'] == cc]
        latest_year = int(cc_data['year'].iloc[0]) if len(cc_data) else ''

        bottom_pos = np.zeros(n_bands)

        for comp in PRICE_COMPONENTS:
            vals = []
            for band in bands:
                r = cc_data[(cc_data['band'] == band) & (cc_data['comp'] == comp)]
                vals.append(r['value'].values[0] if len(r) else 0)
            vals = np.array(vals)
            pos_vals = np.where(vals >= 0, vals, 0)
            if np.any(pos_vals > 0):
                components_used.add(comp)
                ax.bar(band_xs, pos_vals, band_width, bottom=bottom_pos,
                       color=COMPONENT_COLORS[comp], alpha=0.85,
                       edgecolor='white', linewidth=0.3)
                bottom_pos += pos_vals

        # Total label on top of each bar group
        for i, total in enumerate(bottom_pos):
            if total > 0:
                ax.text(band_xs[i], total + y_max * 0.005,
                        f'{total:.3f}', ha='center', va='bottom',
                        fontsize=5, fontweight='bold')

        # Collect Excel data
        for band in bands:
            row = {
                'Country': cc,
                'Year': latest_year,
                'Band': BAND_SHORT_LABELS.get(band, band),
                'Band_code': band,
            }
            for comp in PRICE_COMPONENTS:
                r = cc_data[(cc_data['band'] == band) & (cc_data['comp'] == comp)]
                row[COMPONENT_LABELS[comp]] = (
                    round(r['value'].values[0], 4) if len(r) else None)
            row['Total'] = round(
                sum(v for v in row.values() if isinstance(v, (int, float)) and v is not None
                    and not isinstance(v, bool)), 4)
            excel_rows.append(row)

    # Cluster separators
    if cfg['show_clusters']:
        for bx in cluster_boundaries:
            ax.axvline(x=bx, color='#888888', linewidth=1.0, linestyle='--', alpha=0.5)

    # Cluster labels at top
    if cfg['show_clusters']:
        for cx, cl_id in cluster_label_positions:
            cl_obj = CLUSTERS[cl_id]
            ax.text(cx, y_max * 0.98, cl_obj['label'],
                    ha='center', va='top', fontsize=7, fontweight='bold',
                    fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=cl_obj['color'],
                              edgecolor='#cccccc', alpha=0.4))

    # EU-27 separator
    if 'EU-27' in country_order and len(country_order) > 1:
        eu_idx = country_order.index('EU-27')
        if eu_idx > 0:
            _, prev_xs = x_all[eu_idx - 1]
            _, eu_xs = x_all[eu_idx]
            sep_x = (prev_xs[-1] + eu_xs[0]) / 2
            ax.axvline(x=sep_x, color='#6a3d9a', linewidth=1.5, linestyle='-', alpha=0.6)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, fontweight='bold', ha='center')
    ax.set_xlim(-1, current_x + 0.5)
    ax.set_ylim(0, y_max)
    ax.set_ylabel('Price (PPS per kWh)' if energy_type == 'electricity'
                  else 'Price (PPS per GJ)', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')
    ax.tick_params(axis='x', pad=2)

    ax.set_title(
        f'{type_label} Price Decomposition by Consumption Band\n{cfg["title_suffix"]}',
        fontsize=13, fontweight='bold', pad=10)

    # Legend — only components that appear
    handles = []
    labels_leg = []
    for comp in PRICE_COMPONENTS:
        if comp in components_used:
            handles.append(plt.Rectangle((0, 0), 1, 1,
                           color=COMPONENT_COLORS[comp], alpha=0.85))
            labels_leg.append(COMPONENT_LABELS[comp])
    fig.legend(handles, labels_leg, loc='lower center',
               ncol=min(len(handles), 5), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    plt.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.08)

    # Save
    out_png = os.path.join(out_dir,
                           f'{cfg["prefix"]}_{energy_type}_prices.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    # Excel
    excel_df = pd.DataFrame(excel_rows)
    xlsx_path = os.path.join(out_dir,
                             f'{cfg["prefix"]}_{energy_type}_prices.xlsx')
    _save_excel(excel_df, xlsx_path, sheet_name=type_label)


# ---------------------------------------------------------------------------
# Helper: plot stacked-bar row on a single axes for a list of countries
# ---------------------------------------------------------------------------
def _plot_stacked_row(ax, countries, df_plot, bands, cfg, y_max, energy_type):
    """Draw stacked bars for *countries* on *ax* and return excel rows."""
    n_bands = len(bands)
    band_width = 0.75
    country_gap = 1.5
    cluster_gap = 3.0

    x_all = []
    tick_positions = []
    tick_labels = []
    cluster_boundaries = []
    cluster_label_positions = []
    current_x = 0
    prev_cluster_id = None
    cluster_start_x = 0

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

        band_xs = np.arange(n_bands) * (band_width + 0.15) + current_x
        x_all.append((cc, band_xs))

        center = (band_xs[0] + band_xs[-1]) / 2
        tick_positions.append(center)
        tick_labels.append(cc)

        current_x = band_xs[-1] + band_width + 0.15

    if cfg['show_clusters'] and prev_cluster_id is not None:
        cluster_label_positions.append(
            ((cluster_start_x + current_x) / 2, prev_cluster_id)
        )

    components_used = set()
    excel_rows = []

    for cc, band_xs in x_all:
        cc_data = df_plot[df_plot['iso'] == cc]
        latest_year = int(cc_data['year'].iloc[0]) if len(cc_data) else ''

        bottom_pos = np.zeros(n_bands)
        for comp in PRICE_COMPONENTS:
            vals = []
            for band in bands:
                r = cc_data[(cc_data['band'] == band) & (cc_data['comp'] == comp)]
                vals.append(r['value'].values[0] if len(r) else 0)
            vals = np.array(vals)
            pos_vals = np.where(vals >= 0, vals, 0)
            if np.any(pos_vals > 0):
                components_used.add(comp)
                ax.bar(band_xs, pos_vals, band_width, bottom=bottom_pos,
                       color=COMPONENT_COLORS[comp], alpha=0.85,
                       edgecolor='white', linewidth=0.3)
                bottom_pos += pos_vals

        for i, total in enumerate(bottom_pos):
            if total > 0:
                ax.text(band_xs[i], total + y_max * 0.005,
                        f'{total:.3f}', ha='center', va='bottom',
                        fontsize=5, fontweight='bold')

        for band in bands:
            row = {
                'Country': cc,
                'Year': latest_year,
                'Band': BAND_SHORT_LABELS.get(band, band),
                'Band_code': band,
            }
            for comp in PRICE_COMPONENTS:
                r = cc_data[(cc_data['band'] == band) & (cc_data['comp'] == comp)]
                row[COMPONENT_LABELS[comp]] = (
                    round(r['value'].values[0], 4) if len(r) else None)
            row['Total'] = round(
                sum(v for v in row.values() if isinstance(v, (int, float)) and v is not None
                    and not isinstance(v, bool)), 4)
            excel_rows.append(row)

    # Cluster separators
    if cfg['show_clusters']:
        for bx in cluster_boundaries:
            ax.axvline(x=bx, color='#888888', linewidth=1.0, linestyle='--', alpha=0.5)

    # Cluster labels
    if cfg['show_clusters']:
        for cx, cl_id in cluster_label_positions:
            cl_obj = CLUSTERS[cl_id]
            ax.text(cx, y_max * 0.98, cl_obj['label'],
                    ha='center', va='top', fontsize=7, fontweight='bold',
                    fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=cl_obj['color'],
                              edgecolor='#cccccc', alpha=0.4))

    # EU-27 separator
    if 'EU-27' in countries and len(countries) > 1:
        eu_idx = countries.index('EU-27')
        if eu_idx > 0:
            _, prev_xs = x_all[eu_idx - 1]
            _, eu_xs = x_all[eu_idx]
            sep_x = (prev_xs[-1] + eu_xs[0]) / 2
            ax.axvline(x=sep_x, color='#6a3d9a', linewidth=1.5, linestyle='-', alpha=0.6)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, fontweight='bold', ha='center')
    ax.set_xlim(-1, current_x + 0.5)
    ax.set_ylim(0, y_max)
    ax.set_ylabel('Price (PPS per kWh)' if energy_type == 'electricity'
                  else 'Price (PPS per GJ)', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')
    ax.tick_params(axis='x', pad=2)

    return excel_rows, components_used


# ---------------------------------------------------------------------------
# Two-row visual: C0+C1 on top, C2+C3+EU-27 on bottom (shared y-axis)
# ---------------------------------------------------------------------------
def plot_energy_prices_two_rows(cfg, df, energy_type, bands, country_order, out_dir):
    """Stacked bar chart split into two rows for readability."""
    type_label = 'Electricity' if energy_type == 'electricity' else 'Gas'
    print(f"  [{type_label}] Two-row energy prices chart...")

    df_plot = df[df['band'].isin(bands)].copy()

    top_countries = [cc for cc in country_order
                     if CLUSTER_ID_BY_COUNTRY.get(cc, -1) in (0, 1)]
    bottom_countries = [cc for cc in country_order
                        if CLUSTER_ID_BY_COUNTRY.get(cc, -1) in (2, 3) or cc == 'EU-27']

    # Shared y-max
    y_max = 0
    for cc in country_order:
        cc_data = df_plot[df_plot['iso'] == cc]
        for band in bands:
            bdata = cc_data[cc_data['band'] == band]
            total = bdata.loc[bdata['value'] >= 0, 'value'].sum()
            y_max = max(y_max, total)
    y_max *= 1.15

    n_max = max(len(top_countries), len(bottom_countries))
    fig_w = max(24, n_max * 2.0)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(fig_w, 14))

    rows_top, comps_top = _plot_stacked_row(
        ax_top, top_countries, df_plot, bands, cfg, y_max, energy_type)
    rows_bot, comps_bot = _plot_stacked_row(
        ax_bot, bottom_countries, df_plot, bands, cfg, y_max, energy_type)

    fig.suptitle(
        f'{type_label} Price Decomposition by Consumption Band\n{cfg["title_suffix"]}',
        fontsize=14, fontweight='bold', y=1.02,
    )

    # Legend
    components_used = comps_top | comps_bot
    handles = []
    labels_leg = []
    for comp in PRICE_COMPONENTS:
        if comp in components_used:
            handles.append(plt.Rectangle((0, 0), 1, 1,
                           color=COMPONENT_COLORS[comp], alpha=0.85))
            labels_leg.append(COMPONENT_LABELS[comp])
    fig.legend(handles, labels_leg, loc='lower center',
               ncol=min(len(handles), 5), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.05, hspace=0.25)

    out_png = os.path.join(out_dir,
                           f'{cfg["prefix"]}_{energy_type}_prices_detail.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    # Excel
    all_excel = rows_top + rows_bot
    excel_df = pd.DataFrame(all_excel)
    xlsx_path = os.path.join(out_dir,
                             f'{cfg["prefix"]}_{energy_type}_prices_detail.xlsx')
    _save_excel(excel_df, xlsx_path, sheet_name=type_label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_report(report_key, df_elec, df_gas):
    cfg = REPORT_CONFIGS[report_key]
    print(f"\n{'='*60}")
    print(f"Generating energy prices report: {cfg['prefix']} {cfg['title_suffix']}")
    print(f"{'='*60}")

    out_dir = os.path.join(OUTPUT_BASE, cfg['prefix'])
    os.makedirs(out_dir, exist_ok=True)

    if not df_elec.empty:
        plot_energy_prices(cfg, df_elec, 'electricity', ELEC_BANDS, out_dir)
    if not df_gas.empty:
        plot_energy_prices(cfg, df_gas, 'gas', GAS_BANDS, out_dir)

    # Detailed two-row graphs for cluster-based reports
    if cfg['show_clusters']:
        if not df_elec.empty:
            available_elec = df_elec[df_elec['band'].isin(ELEC_BANDS)]['iso'].unique()
            co_elec = _build_country_order(cfg, available_elec)
            plot_energy_prices_two_rows(cfg, df_elec, 'electricity', ELEC_BANDS,
                                        co_elec, out_dir)
        if not df_gas.empty:
            available_gas = df_gas[df_gas['band'].isin(GAS_BANDS)]['iso'].unique()
            co_gas = _build_country_order(cfg, available_gas)
            plot_energy_prices_two_rows(cfg, df_gas, 'gas', GAS_BANDS,
                                        co_gas, out_dir)

    print(f"\n  All outputs saved to: {out_dir}")


def main():
    df_elec = load_energy_data('electricity')
    df_gas = load_energy_data('gas')

    for report_key in ['rep_eu', 'rep_ewbi', 'rep_fr', 'rep_ch']:
        generate_report(report_key, df_elec, df_gas)
    print("\nDone.")


if __name__ == '__main__':
    main()
