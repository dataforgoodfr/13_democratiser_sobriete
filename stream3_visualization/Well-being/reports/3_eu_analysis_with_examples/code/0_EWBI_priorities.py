"""
0_EWBI_priorities.py — EWBI indicators visuals for the EWBI report.

Produces bar-chart grids (cluster > country > decile D1–D10) for:
  Level 1: EWBI overall index (1 chart)
  Level 2: EU priority sub-indices (5 charts)
  Level 3: All primary indicators (37 charts), grouped by EU priority

EU Priorities (Level 2):
  Energy and Housing
  Equality
  Health and Animal Welfare
  Intergenerational Fairness, Youth, Culture and Sport
  Social Rights and Skills, Quality Jobs and Preparedness

The script detects forward-filled countries by comparing the aggregated
master data with the pre-forward-fill file (raw_data_break_adjusted.csv)
and displays a disclaimer on each chart for countries whose underlying
indicators stopped before the displayed year.

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
WELL_BEING_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
DATA_PATH = os.path.join(WELL_BEING_DIR, 'output', 'ewbi_master_aggregated.csv')
PRE_FFILL_PATH = os.path.join(
    WELL_BEING_DIR, 'output', '1_missing_data_output', 'raw_data_break_adjusted.csv',
)
OUTPUT_BASE = os.path.join(BASE_DIR, 'outputs', 'graphs', 'EWBI_Priorities')
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
# EU Priorities (Level 2)
# ---------------------------------------------------------------------------
EU_PRIORITIES = [
    'Energy and Housing',
    'Equality',
    'Health and Animal Welfare',
    'Intergenerational Fairness, Youth, Culture and Sport',
    'Social Rights and Skills, Quality Jobs and Preparedness',
]

EU_PRIORITY_SHORT = {
    'Energy and Housing': 'Energy & Housing',
    'Equality': 'Equality',
    'Health and Animal Welfare': 'Health',
    'Intergenerational Fairness, Youth, Culture and Sport': 'Education',
    'Social Rights and Skills, Quality Jobs and Preparedness': 'Quality of Jobs',
}

# ---------------------------------------------------------------------------
# Level 3 — Primary indicator labels
# ---------------------------------------------------------------------------
INDICATOR_LABELS = {
    # Energy and Housing
    'HQ-SILC-1': 'Overcrowded Dwelling',
    'HQ-SILC-2': 'Cannot Keep Dwelling Comfortably Warm',
    'HQ-SILC-3': 'Cannot Keep Dwelling Comfortably Cool',
    'HQ-SILC-4': 'Dwelling Too Dark',
    'HQ-SILC-5': 'Noise from Street',
    'HQ-SILC-6': 'Leaking Roof / Damp / Rot',
    'HQ-SILC-7': 'Pollution or Crime',
    'HQ-SILC-8': 'No Renovation Measures',
    'HE-SILC-2': 'Behind on Utility Bills',
    # Equality
    'ES-SILC-1': 'Unable to Handle Unexpected Costs',
    'ES-SILC-2': 'Hard to Make Ends Meet',
    'EC-SILC-2': 'Low Trust in Others',
    'EC-SILC-3': 'Cannot Get Together with Friends/Family',
    'EC-SILC-4': 'Persons Living Alone',
    # Health and Animal Welfare
    'AH-SILC-2': 'Living with Chronic Illness',
    'AH-SILC-3': 'Limited by Health Problems',
    'AH-SILC-4': 'Unable to Work Due to Long-Term Illness',
    'AC-SILC-3': 'Unmet Need for Medical Care',
    'AC-SILC-4': 'Unmet Need for Dental Care',
    # Intergenerational Fairness, Youth, Culture and Sport
    'IS-SILC-3': 'No Formal Education',
    'IS-SILC-4': 'Not Participating in Formal Training',
    'IS-SILC-5': 'No Secondary Education',
    # Social Rights and Skills, Quality Jobs and Preparedness
    'RT-SILC-1': 'Adults on Fixed-Term Contracts',
    'RT-SILC-2': 'Adults Working Part-Time',
    'RT-LFS-1': 'Working Multiple Jobs',
    'RT-LFS-2': 'Wanting to Work More Hours',
    'RT-LFS-3': 'Doing Overtime or Extra Hours',
    'RT-LFS-4': 'No Freedom on Working Time',
    'RT-LFS-5': 'Shift Work',
    'RT-LFS-6': 'Night Work',
    'RT-LFS-7': 'Saturday Work',
    'RT-LFS-8': 'Sunday Work',
    'RU-SILC-1': 'Unemployed for Over 6 Months',
    # Additional / LFS
    'EL-LFS-2': 'No Adequate Childcare Services',
    # Cross-cutting (not assigned to standard EU priority)
    'IC-SILC-1': 'IC-SILC-1',
    'IC-SILC-2': 'IC-SILC-2',
    'TS-SILC-1': 'TS-SILC-1',
}

# Indicators to exclude from Level 3 charts
EXCLUDED_INDICATORS = {'AH-SILC-1'}  # Poor Self-Rated Health — excluded from Health grid

# Indicators expressed as % of households (the rest are share of population)
HOUSEHOLD_INDICATORS = {
    # Energy and Housing
    'HQ-SILC-1', 'HQ-SILC-2', 'HQ-SILC-3', 'HQ-SILC-4', 'HQ-SILC-5',
    'HQ-SILC-6', 'HQ-SILC-7', 'HQ-SILC-8', 'HE-SILC-2',
    # Equality (except EC-SILC-4 "Persons Living Alone" which is share of population)
    'ES-SILC-1', 'ES-SILC-2', 'EC-SILC-2', 'EC-SILC-3',
}

# ---------------------------------------------------------------------------
# Cluster definitions
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
for _cl in CLUSTERS:
    for _cc in _cl['countries']:
        CLUSTER_COLOR_BY_COUNTRY[_cc] = _cl['color']
        CLUSTER_ID_BY_COUNTRY[_cc] = _cl['id']
CLUSTER_COLOR_BY_COUNTRY['EU-27'] = '#6a3d9a'

# ---------------------------------------------------------------------------
# Report configuration (EWBI report only)
# ---------------------------------------------------------------------------
REPORT_CFG = {
    'prefix': 'rep_ewbi',
    'title_suffix': '(EU-27 + EFTA + UK + Serbia)',
    'country_filter': EU27_CODES | EFTA_CODES | {'UK', 'RS', 'EU-27'},
    'comparison_countries': [
        'FR', 'ES', 'BE', 'NL', 'LT', 'HR', 'DE', 'SE',
        'CH', 'NO', 'IS', 'UK', 'RS',
    ],
    'include_eu27_agg': True,
    'show_clusters': True,
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
            for name, df_sheet in data.items():
                df_sheet.to_excel(writer, sheet_name=name, index=False)
        else:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  Saved: {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Detect forward-filled countries
# ---------------------------------------------------------------------------
def load_last_real_year():
    """
    Load the pre-forward-fill file (raw_data_break_adjusted.csv) and compute
    the last real data year per country.  Returns a dict {country_code: year}.
    """
    if not os.path.exists(PRE_FFILL_PATH):
        print(f"  WARNING: pre-ffill file not found ({PRE_FFILL_PATH}) — no disclaimer possible")
        return {}
    bf = pd.read_csv(PRE_FFILL_PATH, usecols=['Year', 'Country'], low_memory=False)
    bf['Year'] = pd.to_numeric(bf['Year'], errors='coerce')
    last_yr = bf.groupby('Country')['Year'].max()
    return last_yr.to_dict()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load master aggregated data for Levels 1, 2 and 3."""
    print(f"Loading data from {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH, low_memory=False)

    raw['Decile_num'] = pd.to_numeric(raw['Decile'], errors='coerce')
    raw['Year'] = pd.to_numeric(raw['Year'], errors='coerce')
    raw['Value'] = pd.to_numeric(raw['Value'], errors='coerce')
    raw = raw.dropna(subset=['Decile_num', 'Year', 'Value'])
    raw['Decile_num'] = raw['Decile_num'].astype(int)
    raw = raw[raw['Decile_num'].between(1, 10)].copy()

    # --- Level 2 (EU priorities) ---
    df2 = raw[raw['Level'] == 2].copy()
    df2 = df2.dropna(subset=['EU priority'])
    lat2 = df2.groupby(['Country', 'EU priority'])['Year'].max().reset_index()
    lat2.columns = ['Country', 'EU priority', 'Latest_Year']
    df2 = df2.merge(lat2, on=['Country', 'EU priority'])
    df2 = df2[df2['Year'] == df2['Latest_Year']].copy()
    print(f"  Level 2: {len(df2)} rows, {df2['Country'].nunique()} countries, "
          f"{df2['EU priority'].nunique()} priorities")

    # --- Level 1 (EWBI overall) ---
    df1 = raw[raw['Level'] == 1].copy()
    lat1 = df1.groupby('Country')['Year'].max().reset_index()
    lat1.columns = ['Country', 'Latest_Year']
    df1 = df1.merge(lat1, on='Country')
    df1 = df1[df1['Year'] == df1['Latest_Year']].copy()
    print(f"  Level 1: {len(df1)} rows, {df1['Country'].nunique()} countries")

    # --- Level 3 (primary indicators) ---
    df3 = raw[raw['Level'] == 3].copy()
    df3 = df3.dropna(subset=['Primary and raw data'])
    lat3 = df3.groupby(['Country', 'Primary and raw data'])['Year'].max().reset_index()
    lat3.columns = ['Country', 'Primary and raw data', 'Latest_Year']
    df3 = df3.merge(lat3, on=['Country', 'Primary and raw data'])
    df3 = df3[df3['Year'] == df3['Latest_Year']].copy()
    n_ind = df3['Primary and raw data'].nunique()
    print(f"  Level 3: {len(df3)} rows, {df3['Country'].nunique()} countries, "
          f"{n_ind} indicators")

    return df1, df2, df3


# ---------------------------------------------------------------------------
# Build ordered country list
# ---------------------------------------------------------------------------
def _build_country_order(cfg, df):
    comparison = cfg['comparison_countries']
    available = set(df['Country'].unique())

    ordered = []
    for cl in CLUSTERS:
        cl_countries = [cc for cc in comparison if cc in cl['countries'] and cc in available]
        scope = cfg['country_filter']
        others = sorted([
            cc for cc in cl['countries']
            if cc in available and cc in scope and cc not in cl_countries
        ])
        ordered.extend(cl_countries + others)

    seen = set()
    deduped = []
    for cc in ordered:
        if cc not in seen:
            seen.add(cc)
            deduped.append(cc)
    ordered = deduped

    if cfg.get('include_eu27_agg') and 'EU-27' in available:
        if 'EU-27' not in ordered:
            ordered.append('EU-27')

    return ordered


# ---------------------------------------------------------------------------
# Generic bar-chart plotter
# ---------------------------------------------------------------------------
def _plot_bars(df_sub, title, short_name, y_label, cfg, country_order,
               last_real_years, out_dir, file_stem, excel_id_col, excel_id_val):
    """
    Generic bar chart: cluster > country > decile for an arbitrary subset.

    Parameters
    ----------
    df_sub : filtered DataFrame (must contain Country, Decile_num, Value, Year)
    title  : chart title
    short_name : used in suptitle
    y_label : y-axis label
    excel_id_col / excel_id_val : identifier written into the Excel export
    """
    if df_sub.empty:
        print(f"    No data for {short_name}")
        return

    n_countries = len(country_order)
    n_deciles = 10
    country_gap = 1.0
    cluster_gap = 2.5
    bar_width = 0.7

    fig_w = max(28, n_countries * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, 7))

    # --- Build x positions ---
    x_positions = []
    cluster_boundaries = []
    cluster_label_positions = []
    tick_positions = []
    tick_labels_list = []
    current_x = 0
    prev_cluster_id = None
    cluster_start_x = 0
    excel_rows = []

    for i_cc, cc in enumerate(country_order):
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

    # Final cluster label
    if cfg['show_clusters'] and prev_cluster_id is not None:
        cluster_label_positions.append(
            ((cluster_start_x + current_x) / 2, prev_cluster_id)
        )

    # Global y-range
    all_vals = []
    for cc in country_order:
        cdf = df_sub[df_sub['Country'] == cc]
        all_vals.extend(cdf['Value'].dropna().tolist())
    y_max = max(all_vals) * 1.12 if all_vals else 1.0

    # Plot bars
    for cc, decile_xs in x_positions:
        cdf = df_sub[df_sub['Country'] == cc].sort_values('Decile_num')
        color = CLUSTER_COLOR_BY_COUNTRY.get(cc, '#999999')

        vals = []
        for d in range(1, 11):
            v = cdf[cdf['Decile_num'] == d]['Value']
            val = v.iloc[0] if len(v) else np.nan
            vals.append(val)

        ax.bar(decile_xs, vals, width=bar_width, color=color,
               edgecolor='white', linewidth=0.4)

        # Value labels on D1 and D10
        for i, val in enumerate(vals):
            if not pd.isna(val) and i in (0, 9):
                ax.text(decile_xs[i] + bar_width / 2, val + y_max * 0.005,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=5, fontweight='bold')

        latest_year = int(cdf['Year'].iloc[0]) if len(cdf) else ''
        for d, val in zip(range(1, 11), vals):
            excel_rows.append({
                excel_id_col: excel_id_val,
                'Country': cc,
                'Country_Name': COUNTRY_NAME_MAP.get(cc, cc),
                'Year': latest_year,
                'Decile': d,
                'Value': round(val, 4) if not pd.isna(val) else None,
            })

    # Cluster separators
    if cfg['show_clusters']:
        for bx in cluster_boundaries:
            ax.axvline(x=bx, color='#888888', linewidth=1.0, linestyle='--', alpha=0.5)

    # Cluster top labels
    if cfg['show_clusters']:
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

    ax.set_title(title, fontsize=12, fontweight='bold', pad=4)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_list, fontsize=8, fontweight='bold')
    ax.set_xlim(-1, current_x + 0.5)
    ax.set_ylim(0, y_max)
    ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')
    ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)

    # --- Disclaimer for forward-filled countries ---
    displayed_year = df_sub.groupby('Country')['Year'].max()
    if last_real_years:
        max_real = max(last_real_years.values())
        older_parts = []
        for cc in country_order:
            if cc == 'EU-27':
                continue
            disp_yr = int(displayed_year.get(cc, 0))
            real_yr = int(last_real_years.get(cc, disp_yr))
            if real_yr < disp_yr:
                name = COUNTRY_NAME_MAP.get(cc, cc)
                older_parts.append(f"{name}: last real data {real_yr}")
        if older_parts:
            disclaimer = "Forward-filled (last value carried forward):\n" + ", ".join(older_parts)
            ax.text(0.99, 0.97, disclaimer, transform=ax.transAxes,
                    fontsize=6, va='top', ha='right', fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6',
                              edgecolor='#ccaa00', alpha=0.85))

    # Suptitle
    fig.suptitle(
        f'{short_name} by Income Decile\n'
        f'{cfg["title_suffix"]}',
        fontsize=14, fontweight='bold', y=1.04,
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
               bbox_to_anchor=(0.5, -0.03))

    plt.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.08)

    # Save
    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_{file_stem}.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    # Excel
    excel_df = pd.DataFrame(excel_rows)
    sheets = {'Long_Format': excel_df}
    if not excel_df.empty:
        wide = excel_df.pivot_table(
            index=[excel_id_col, 'Country', 'Country_Name', 'Year'],
            columns='Decile', values='Value', aggfunc='first',
        )
        wide.columns = [f'D{int(c)}' for c in wide.columns]
        wide = wide.reset_index()
        sheets['Wide_Format'] = wide
    xlsx_path = os.path.join(out_dir, f'{cfg["prefix"]}_{file_stem}.xlsx')
    _save_excel(sheets, xlsx_path)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def plot_priority(priority_name, cfg, df, country_order, last_real_years, out_dir):
    """Bar chart for a single EU priority (Level 2)."""
    short_name = EU_PRIORITY_SHORT.get(priority_name, priority_name)
    print(f"  [{short_name}] Generating bar chart...")
    df_pri = df[df['EU priority'] == priority_name].copy()
    safe = short_name.lower().replace(' ', '_').replace('&', 'and')
    _plot_bars(df_pri, priority_name, f'EWBI Sub-Index — {short_name}',
              'EWBI Sub-Index Score', cfg, country_order, last_real_years,
              out_dir, f'ewbi_priority_{safe}', 'EU_Priority', priority_name)


def plot_ewbi_overall(cfg, df1, country_order, last_real_years, out_dir):
    """Bar chart for the overall EWBI index (Level 1)."""
    print(f"  [EWBI Overall] Generating bar chart...")
    _plot_bars(df1, 'EWBI — Overall Index', 'EWBI Overall Index',
              'EWBI Score', cfg, country_order, last_real_years,
              out_dir, 'ewbi_overall', 'Indicator', 'EWBI')


def plot_priority_indicators_grid(priority_name, indicators, cfg, df3,
                                  country_order, last_real_years, out_dir):
    """
    Create a vertically stacked figure with one subplot per indicator,
    for all indicators belonging to a single EU priority.
    """
    short_name = EU_PRIORITY_SHORT.get(priority_name, priority_name)
    n_ind = len(indicators)
    print(f"  [{short_name}] Generating {n_ind}-indicator grid...")

    n_countries = len(country_order)
    n_deciles = 10
    country_gap = 1.0
    cluster_gap = 2.5
    bar_width = 0.7

    fig_w = max(28, n_countries * 1.6)
    fig, axes = plt.subplots(n_ind, 1, figsize=(fig_w, 6 * n_ind), squeeze=False)
    axes_flat = axes.flatten()

    all_excel_rows = []

    for plot_idx, ind_code in enumerate(indicators):
        ax = axes_flat[plot_idx]
        ind_label = INDICATOR_LABELS.get(ind_code, ind_code)
        df_ind = df3[df3['Primary and raw data'] == ind_code].copy()

        # Build x positions with cluster/country gaps
        x_positions = []
        cluster_boundaries = []
        cluster_label_positions = []
        tick_positions = []
        tick_labels_list = []
        current_x = 0
        prev_cluster_id = None
        cluster_start_x = 0

        for i_cc, cc in enumerate(country_order):
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

        # Final cluster label
        if cfg['show_clusters'] and prev_cluster_id is not None:
            cluster_label_positions.append(
                ((cluster_start_x + current_x) / 2, prev_cluster_id)
            )

        # Global y-range for this indicator
        all_vals = []
        for cc in country_order:
            cdf = df_ind[df_ind['Country'] == cc]
            all_vals.extend(cdf['Value'].dropna().tolist())
        y_max = max(all_vals) * 1.12 if all_vals else 1.0

        # Plot bars
        for cc, decile_xs in x_positions:
            cdf = df_ind[df_ind['Country'] == cc].sort_values('Decile_num')
            color = CLUSTER_COLOR_BY_COUNTRY.get(cc, '#999999')

            vals = []
            for d in range(1, 11):
                v = cdf[cdf['Decile_num'] == d]['Value']
                val = v.iloc[0] if len(v) else np.nan
                vals.append(val)

            ax.bar(decile_xs, vals, width=bar_width, color=color,
                   edgecolor='white', linewidth=0.4)

            # Value labels on D1 and D10
            for i, val in enumerate(vals):
                if not pd.isna(val) and i in (0, 9):
                    ax.text(decile_xs[i] + bar_width / 2, val + y_max * 0.005,
                            f'{val:.1f}', ha='center', va='bottom',
                            fontsize=5, fontweight='bold')

            latest_year = int(cdf['Year'].iloc[0]) if len(cdf) else ''
            for d, val in zip(range(1, 11), vals):
                all_excel_rows.append({
                    'Indicator': ind_code,
                    'Indicator_Name': ind_label,
                    'Country': cc,
                    'Country_Name': COUNTRY_NAME_MAP.get(cc, cc),
                    'Year': latest_year,
                    'Decile': d,
                    'Value': round(val, 4) if not pd.isna(val) else None,
                })

        # Cluster separators
        if cfg['show_clusters']:
            for bx in cluster_boundaries:
                ax.axvline(x=bx, color='#888888', linewidth=1.0, linestyle='--', alpha=0.5)

        # Cluster top labels — only on first subplot
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

        ax.set_title(f'{ind_code} — {ind_label}', fontsize=11, fontweight='bold', pad=4)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, fontsize=8, fontweight='bold')
        ax.set_xlim(-1, current_x + 0.5)
        ax.set_ylim(0, y_max)
        ax.grid(axis='y', alpha=0.2)
        ax.set_facecolor('white')
        y_label = '% of households' if ind_code in HOUSEHOLD_INDICATORS else 'Share of pop. (%)'
        ax.set_ylabel(y_label, fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', pad=2)
        ax.tick_params(axis='y', pad=2)

        # Disclaimer for forward-filled countries
        displayed_year = df_ind.groupby('Country')['Year'].max()
        if last_real_years:
            max_real = max(last_real_years.values())
            older_parts = []
            for cc in country_order:
                if cc == 'EU-27':
                    continue
                disp_yr = int(displayed_year.get(cc, 0))
                real_yr = int(last_real_years.get(cc, disp_yr))
                if real_yr < disp_yr:
                    name = COUNTRY_NAME_MAP.get(cc, cc)
                    older_parts.append(f"{name}: last real data {real_yr}")
            if older_parts:
                disclaimer = "Forward-filled (last value carried forward):\n" + ", ".join(older_parts)
                ax.text(0.99, 0.97, disclaimer, transform=ax.transAxes,
                        fontsize=6, va='top', ha='right', fontstyle='italic',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6',
                                  edgecolor='#ccaa00', alpha=0.85))

    # Suptitle
    fig.suptitle(
        f'{short_name} — Primary Indicators by Income Decile\n'
        f'{cfg["title_suffix"]}',
        fontsize=14, fontweight='bold', y=1.0,
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
               bbox_to_anchor=(0.5, -0.01))

    plt.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.04,
                        hspace=0.22)

    # Save
    safe = short_name.lower().replace(' ', '_').replace('&', 'and')
    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_indicators_{safe}.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    # Excel
    excel_df = pd.DataFrame(all_excel_rows)
    sheets = {'Long_Format': excel_df}
    if not excel_df.empty:
        wide = excel_df.pivot_table(
            index=['Indicator', 'Indicator_Name', 'Country', 'Country_Name', 'Year'],
            columns='Decile', values='Value', aggfunc='first',
        )
        wide.columns = [f'D{int(c)}' for c in wide.columns]
        wide = wide.reset_index()
        sheets['Wide_Format'] = wide
    xlsx_path = os.path.join(out_dir, f'{cfg["prefix"]}_indicators_{safe}.xlsx')
    _save_excel(sheets, xlsx_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df1, df2, df3 = load_data()
    last_real_years = load_last_real_year()

    if last_real_years:
        max_real = max(last_real_years.values())
        forward_filled = {cc: int(yr) for cc, yr in last_real_years.items() if yr < max_real}
        if forward_filled:
            print(f"\n  Forward-filled countries (real data ends before {int(max_real)}):")
            for cc, yr in sorted(forward_filled.items()):
                print(f"    {COUNTRY_NAME_MAP.get(cc, cc)} ({cc}): last real data {yr}")
        else:
            print("  All countries have data up to the same year.")

    cfg = REPORT_CFG
    df2_report = df2[df2['Country'].isin(cfg['country_filter'])].copy()
    df1_report = df1[df1['Country'].isin(cfg['country_filter'])].copy()
    df3_report = df3[df3['Country'].isin(cfg['country_filter'])].copy()
    country_order = _build_country_order(cfg, df2_report)

    out_dir = os.path.join(OUTPUT_BASE, cfg['prefix'])
    os.makedirs(out_dir, exist_ok=True)

    # ---- Level 1: EWBI overall ----
    print(f"\n{'='*60}")
    print(f"Level 1 — EWBI Overall: {cfg['prefix']} {cfg['title_suffix']}")
    print(f"  Countries: {len(country_order)}")
    print(f"{'='*60}")
    plot_ewbi_overall(cfg, df1_report, country_order, last_real_years, out_dir)

    # ---- Level 2: EU priorities ----
    print(f"\n{'='*60}")
    print(f"Level 2 — EU Priorities: {cfg['prefix']} {cfg['title_suffix']}")
    print(f"{'='*60}")
    for priority in EU_PRIORITIES:
        plot_priority(priority, cfg, df2_report, country_order, last_real_years, out_dir)

    # ---- Level 3: Primary indicators (grouped by EU priority) ----
    print(f"\n{'='*60}")
    print(f"Level 3 — Primary Indicators: {cfg['prefix']} {cfg['title_suffix']}")
    print(f"{'='*60}")

    # Group indicators by EU priority for ordered output
    ind_by_priority = {}
    for ind in sorted(df3_report['Primary and raw data'].unique()):
        prios = df3_report[df3_report['Primary and raw data'] == ind]['EU priority'].dropna().unique()
        key = prios[0] if len(prios) else 'Other'
        ind_by_priority.setdefault(key, []).append(ind)

    # Process standard EU priorities first, then any remaining
    priority_order = EU_PRIORITIES + [k for k in sorted(ind_by_priority) if k not in EU_PRIORITIES]
    for prio_key in priority_order:
        indicators = [i for i in ind_by_priority.get(prio_key, []) if i not in EXCLUDED_INDICATORS]
        if not indicators:
            continue
        plot_priority_indicators_grid(prio_key, indicators, cfg, df3_report,
                                      country_order, last_real_years, out_dir)

    print(f"\n  All outputs saved to: {out_dir}")
    print("Done.")


if __name__ == '__main__':
    main()
