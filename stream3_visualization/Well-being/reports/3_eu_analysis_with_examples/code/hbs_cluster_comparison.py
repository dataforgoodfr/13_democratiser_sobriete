"""
HBS Analysis - Consumption breakdown by cluster for 6 highlighted countries.
Stacked bar charts (absolute PPS) grouped by EWBI clusters,
all urbanization levels combined.
Housing and Transport sub-components are aggregated into single categories.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from hbs_data_loader import setup_directories, load_pps_data, calculate_consumption_in_pps
import glob
import copy

plt.style.use('default')
sns.set_palette("Set2")

# Countries and clusters --------------------------------------------------
HIGHLIGHT_COUNTRIES = {
    'FR': 'France',
    'ES': 'Spain',
    'BE': 'Belgium',
    'NL': 'Netherlands',
    'LT': 'Lithuania',
    'HR': 'Croatia',
    'DE': 'Germany',
    'AT': 'Austria',
}

CLUSTERS = [
    {'name': 'Low performer / Low EWBI',  'countries': ['FR', 'ES']},
    {'name': 'Low performer / High EWBI', 'countries': ['BE', 'NL']},
    {'name': 'High performer / Low EWBI', 'countries': ['LT', 'HR']},
    {'name': 'High performer / High EWBI','countries': ['DE', 'AT']},
]

CLUSTER_COLORS = ['#fb8072', '#fdb462', '#8dd3c7', '#80b1d3']
POLAND_SCALE_FACTOR = 12.0

# Aggregated component names shown on the chart
DISPLAY_COMPONENTS = ['Housing', 'Transport', 'Food & Beverage', 'Health', 'Education']
DISPLAY_COLORS     = ['#fc8d62', '#b3de69', '#8dd3c7', '#ffffb3', '#bebada']

REPLACEMENT_CANDIDATES_BY_TARGET = {
    # Cluster: High performer / Low EWBI
    'HU': [
        ('LV', 'Latvia'),
        ('EE', 'Estonia'),
        ('SK', 'Slovakia'),
        ('SI', 'Slovenia'),
        ('CZ', 'Czechia'),
    ],
    # Cluster: High performer / High EWBI
    'PL': [
        ('SE', 'Sweden'),
        ('AT', 'Austria'),
        ('DK', 'Denmark'),
        ('FI', 'Finland'),
        ('IE', 'Ireland'),
        ('LU', 'Luxembourg'),
    ],
}

# ── helpers ───────────────────────────────────────────────────────────────

def load_country_2020(country_code):
    """
    Load HBS household data for *country_code* in 2020.

    Cache strategy (fast path first):
      1. Local parquet in outputs/data/hbs_raw_cache/ → instant load
      2. xlsx on OneDrive via calamine engine (3-8× faster than openpyxl)
         → saves parquet for all future calls
    Run 0_preprocess_hbs_cache.py once to pre-build the full cache.
    """
    _script_dir   = os.path.dirname(os.path.abspath(__file__))
    _base_dir     = os.path.abspath(os.path.join(_script_dir, ".."))
    _raw_cache_dir = os.path.join(_base_dir, "outputs", "data", "hbs_raw_cache")
    os.makedirs(_raw_cache_dir, exist_ok=True)
    cache_file = os.path.join(_raw_cache_dir, f"{country_code}_hbs2020_raw.parquet")

    # ── Fast path: local parquet ──────────────────────────────────────────
    if os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            print(f"  OK  {country_code}: {df.shape[0]} rows  [parquet cache]")
            return df
        except Exception:
            pass  # corrupt cache → fall through to xlsx

    # ── Slow path: read xlsx from OneDrive ────────────────────────────────
    external_hbs_base = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/HBS"
    folder_2020 = os.path.join(external_hbs_base, "HBS2020/HBS2020")

    if not os.path.exists(folder_2020):
        print(f"  ERROR: directory not found: {folder_2020}")
        return pd.DataFrame()

    hh_files = glob.glob(os.path.join(folder_2020, "HBS_HH_*.xlsx"))
    country_files = [f for f in hh_files if country_code in os.path.basename(f)]

    if not country_files:
        print(f"  ERROR: no HBS file for {country_code}")
        return pd.DataFrame()

    try:
        print(f"  Loading {country_code} from xlsx (first run — will cache)…")
        try:
            df = pd.read_excel(country_files[0], engine="calamine",
                               dtype_backend="numpy_nullable")
        except Exception:
            df = pd.read_excel(country_files[0])
        df['year'] = '2020'
        # Save for all future runs
        try:
            df.to_parquet(cache_file, index=False, compression="snappy")
            print(f"  OK  {country_code}: {df.shape[0]} rows  [cached to parquet]")
        except Exception:
            print(f"  OK  {country_code}: {df.shape[0]} rows  [parquet save failed]")
        return df
    except Exception as e:
        print(f"  ERROR loading {country_code}: {e}")
        return pd.DataFrame()


def assign_simple_deciles(df, n_groups=10):
    """Weighted group assignment based on equivalised income."""
    df = df.copy()
    for c in ['EUR_HH099', 'HB061', 'HA10']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    valid = df[(df['EUR_HH099'].notna()) & (df['HB061'].notna()) & (df['HB061'] > 0)].copy()
    if len(valid) < n_groups:
        return df

    valid['eq_inc'] = valid['EUR_HH099'] / valid['HB061']
    idx = np.argsort(valid['eq_inc'].values)
    cum = np.cumsum(valid['HA10'].values[idx])
    cum_n = cum / cum[-1]
    sorted_inc = valid['eq_inc'].values[idx]

    bounds = []
    for d in range(1, n_groups):
        i = np.searchsorted(cum_n, d / n_groups)
        if i < len(sorted_inc):
            bounds.append(sorted_inc[i])

    if n_groups == 10:
        labels = ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10']
    else:
        labels = [f'Q{i}' for i in range(1, n_groups + 1)]

    df['income_decile'] = pd.cut(
        df['EUR_HH099'] / df['HB061'],
        bins=[-np.inf] + bounds + [np.inf],
        labels=labels,
        duplicates='drop',
    )
    return df


def assign_consumption_groups(df, n_groups=5):
    """Weighted group assignment based on equivalised consumption (fallback)."""
    df = df.copy()
    consumption_col = 'EUR_HE00_pps'
    for c in [consumption_col, 'HB061', 'HA10']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    valid = df[(df[consumption_col].notna()) & (df['HB061'].notna()) & (df['HB061'] > 0)].copy()
    if len(valid) < n_groups:
        return df

    valid['eq_cons'] = valid[consumption_col] / valid['HB061']
    idx = np.argsort(valid['eq_cons'].values)
    cum = np.cumsum(valid['HA10'].values[idx])
    cum_n = cum / cum[-1]
    sorted_cons = valid['eq_cons'].values[idx]

    bounds = []
    for d in range(1, n_groups):
        i = np.searchsorted(cum_n, d / n_groups)
        if i < len(sorted_cons):
            bounds.append(sorted_cons[i])

    if n_groups == 10:
        labels = [f'D{i}' for i in range(1, 11)]
    else:
        labels = [f'Q{i}' for i in range(1, n_groups + 1)]

    df['income_decile'] = pd.cut(
        df[consumption_col] / df['HB061'],
        bins=[-np.inf] + bounds + [np.inf],
        labels=labels,
        duplicates='drop',
    )
    return df


def _weighted_component(valid, code, denom):
    """Return weighted per-AE value for a single raw column code."""
    col_pps = f'{code}_pps'
    if col_pps in valid.columns:
        raw = valid[col_pps].fillna(0)
    elif code in valid.columns:
        raw = valid[code].fillna(0)
    else:
        return 0.0
    ae = raw / valid['HB061']
    return (ae * valid['HA10']).sum() / denom


def calculate_components_by_decile(df, country_code=None):
    """
    Calculate aggregated consumption components per decile
    (all urbanization combined).  Housing and Transport are each
    summed into a single value.
    """
    consumption_col = 'EUR_HE00_pps'

    # Iterate over whatever income groups exist (D1-D10 or Q1-Q5).
    # Sort numerically on the embedded integer so D10 comes after D9, not after D1.
    income_groups = sorted(
        df['income_decile'].dropna().unique(),
        key=lambda x: int("".join(c for c in str(x) if c.isdigit())) if any(c.isdigit() for c in str(x)) else str(x),
    )

    results = []
    for decile in income_groups:
        grp = df[df['income_decile'] == decile].copy()
        if grp.empty:
            continue

        grp[consumption_col] = pd.to_numeric(grp[consumption_col], errors='coerce')
        if 'EUR_HH099_pps' in grp.columns:
            grp['EUR_HH099_pps'] = pd.to_numeric(grp['EUR_HH099_pps'], errors='coerce')
        grp['HB061'] = pd.to_numeric(grp['HB061'], errors='coerce')
        grp['HA10']  = pd.to_numeric(grp['HA10'],  errors='coerce')

        valid = grp[
            grp[consumption_col].notna() & grp['HB061'].notna() &
            (grp['HB061'] > 0) & grp['HA10'].notna()
        ].copy()
        if valid.empty:
            continue

        denom = valid['HA10'].sum()
        total_ae = valid[consumption_col] / valid['HB061']
        total_val = (total_ae * valid['HA10']).sum() / denom

        income_val = np.nan
        if 'EUR_HH099_pps' in valid.columns:
            income_valid = valid[valid['EUR_HH099_pps'].notna()].copy()
            if not income_valid.empty and income_valid['HA10'].sum() > 0:
                income_denom = income_valid['HA10'].sum()
                income_ae = income_valid['EUR_HH099_pps'] / income_valid['HB061']
                income_val = (income_ae * income_valid['HA10']).sum() / income_denom

        # --- Housing (Actual Rentals + Imputed Rentals + Utilities) ---
        housing = 0.0
        housing += _weighted_component(valid, 'EUR_HE041', denom)
        housing += _weighted_component(valid, 'EUR_HE042', denom)
        for sc in ['EUR_HE043', 'EUR_HE044', 'EUR_HE045']:
            housing += _weighted_component(valid, sc, denom)

        # --- Transport (Purchase + Operation + Services) ---
        transport = 0.0
        transport += _weighted_component(valid, 'EUR_HE071', denom)
        transport += _weighted_component(valid, 'EUR_HE072', denom)
        transport += _weighted_component(valid, 'EUR_HE073', denom)

        # --- Other named components ---
        food   = _weighted_component(valid, 'EUR_HE01', denom)
        health = _weighted_component(valid, 'EUR_HE06', denom)
        educ   = _weighted_component(valid, 'EUR_HE10', denom)

        comp_sum = housing + transport + food + health + educ

        row = {
            'decile': decile,
            'n_households': len(valid),
            'total_consumption': total_val,
            'equivalized_income_pps': income_val,
            'Housing': housing,
            'Transport': transport,
            'Food & Beverage': food,
            'Health': health,
            'Education': educ,
            'Other (Residual)': max(total_val - comp_sum, 0),
        }

        # Requested normalization: Poland values must be multiplied by 12.
        if country_code == 'PL':
            for key in ['total_consumption', 'Housing', 'Transport', 'Food & Beverage', 'Health', 'Education', 'Other (Residual)']:
                row[key] = row[key] * POLAND_SCALE_FACTOR

        results.append(row)

    return pd.DataFrame(results)


def has_valid_components(comp_df):
    """Check whether a component table contains enough finite values to plot/export."""
    if comp_df is None or comp_df.empty:
        return False

    needed = ['total_consumption', 'Housing', 'Transport', 'Food & Beverage', 'Health', 'Education']
    available = [c for c in needed if c in comp_df.columns]
    if not available:
        return False

    cleaned = comp_df[available].replace([np.inf, -np.inf], np.nan)
    valid_rows = cleaned.notna().all(axis=1).sum()
    return valid_rows >= 4  # 4 = minimum for quintiles, 8 = typical for deciles


def build_country_components(cc, cname, pps_df, n_groups=10):
    """Load, convert, group and aggregate one country."""
    print(f"\n--- {cname} ({cc}) ---")
    df = load_country_2020(cc)
    if df.empty:
        return pd.DataFrame()

    df = calculate_consumption_in_pps(df, pps_df)
    if 'EUR_HE00_pps' not in df.columns:
        if 'EUR_HE00' in df.columns and 'pps_factor' in df.columns:
            df['EUR_HE00_pps'] = pd.to_numeric(df['EUR_HE00'], errors='coerce') / pd.to_numeric(df['pps_factor'], errors='coerce')
            print("  INFO: derived EUR_HE00_pps from EUR_HE00 / pps_factor")
        else:
            print("  WARN: EUR_HE00_pps unavailable; skipping country to avoid nominal values")
            return pd.DataFrame()

    if 'EUR_HH099_pps' not in df.columns:
        if 'EUR_HH099' in df.columns and 'pps_factor' in df.columns:
            df['EUR_HH099_pps'] = pd.to_numeric(df['EUR_HH099'], errors='coerce') / pd.to_numeric(df['pps_factor'], errors='coerce')
            print("  INFO: derived EUR_HH099_pps from EUR_HH099 / pps_factor")
        else:
            print("  WARN: EUR_HH099_pps unavailable; income line may be missing for this country")

    df = assign_simple_deciles(df, n_groups=n_groups)

    # Fallback 1: HB070 income quintile
    if 'income_decile' not in df.columns:
        if 'HB070' in df.columns:
            df['HB070'] = pd.to_numeric(df['HB070'], errors='coerce')
            if df['HB070'].notna().sum() > 0:
                df['income_decile'] = df['HB070'].map(
                    {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5'}
                )
                print(f"  INFO: used HB070 income quintile as fallback")

    # Fallback 2: consumption-based grouping
    if 'income_decile' not in df.columns or df['income_decile'].notna().sum() == 0:
        df = assign_consumption_groups(df, n_groups=min(n_groups, 5))
        if 'income_decile' in df.columns and df['income_decile'].notna().sum() > 0:
            print(f"  INFO: used consumption-based quintile grouping as fallback")
        else:
            print("  WARN: no income grouping possible; skipping country")
            return pd.DataFrame()

    comp = calculate_components_by_decile(df, country_code=cc)
    if has_valid_components(comp):
        print(f"  OK  {len(comp)} groups computed")
        return comp

    print("  WARN: component data has NaN/insufficient values")
    return pd.DataFrame()


def resolve_country_replacement(target_code, pps_df, used_codes):
    """Find first valid replacement country for a target country code."""
    candidates = REPLACEMENT_CANDIDATES_BY_TARGET.get(target_code, [])
    for cc, cname in candidates:
        if cc in used_codes:
            continue
        comp = build_country_components(cc, cname, pps_df)
        if has_valid_components(comp):
            print(f"  REPLACEMENT SELECTED for {target_code}: {cname} ({cc})")
            return cc, cname, comp
    return None, None, pd.DataFrame()


# ── plotting ──────────────────────────────────────────────────────────────

def plot_cluster_comparison(all_components, dirs):
    """
    One figure: 3 rows (clusters) x 2 columns (countries per cluster).
    Stacked bars D1-D10.  Uniform y-axis across all subplots.
    """
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)

    decile_order = ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10']

    bar_components = DISPLAY_COMPONENTS + ['Other (Residual)']
    bar_colors     = DISPLAY_COLORS     + ['#d3d3d3']

    n_rows = len(CLUSTERS)
    n_cols = max(len(c['countries']) for c in CLUSTERS)

    # ── find global y-max ─────────────────────────────────────────────
    global_ymax = 0
    for cc, cdf in all_components.items():
        if cdf is None or cdf.empty:
            continue
        row_totals = sum(
            cdf[comp].fillna(0).values for comp in bar_components if comp in cdf.columns
        )
        global_ymax = max(global_ymax, row_totals.max())
    global_ymax *= 1.08  # 8 % headroom

    fig = plt.figure(figsize=(8 * n_cols, 6 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.35, wspace=0.25)

    n_countries_total = sum(len(c['countries']) for c in CLUSTERS)
    fig.suptitle(
        f'Consumption Breakdown by Income Decile — {n_countries_total} Countries '
        f'Grouped by EWBI Cluster (2020)\n'
        'Housing · Transport · Food · Health · Education · Residual  —  '
        'All densities combined',
        fontsize=15, fontweight='bold', y=1.002,
    )

    first_ax = None

    for row_idx, cluster in enumerate(CLUSTERS):
        n_countries = len(cluster['countries'])
        for col_idx, cc in enumerate(cluster['countries']):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if first_ax is None:
                first_ax = ax

            cname = HIGHLIGHT_COUNTRIES[cc]
            cdf = all_components.get(cc)
            if cdf is None or cdf.empty:
                ax.text(0.5, 0.5, f'{cname}\n(no data)', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{cname} ({cc}) — {cluster["name"]}',
                             fontsize=12, fontweight='bold')
                ax.set_ylim(0, global_ymax)
                continue

            cdf['decile'] = pd.Categorical(cdf['decile'], categories=decile_order, ordered=True)
            cdf = cdf.sort_values('decile')

            x = np.arange(len(cdf))
            bottom = np.zeros(len(cdf))

            for comp, color in zip(bar_components, bar_colors):
                if comp not in cdf.columns:
                    continue
                vals = cdf[comp].fillna(0).values
                ax.bar(x, vals, bottom=bottom, label=comp,
                       color=color, edgecolor='white', linewidth=0.8, alpha=0.85)
                for j, v in enumerate(vals):
                    if v > 80:
                        ax.text(j, bottom[j] + v / 2, f'{int(v)}',
                                ha='center', va='center', fontsize=6)
                bottom += vals

            ax.set_xticks(x)
            ax.set_xticklabels(cdf['decile'].values, fontsize=9)
            ax.set_xlabel('Income Decile', fontsize=10, fontweight='bold')
            ax.set_ylabel('Mean Consumption (PPS)', fontsize=10, fontweight='bold')
            ax.set_title(f'{cname} ({cc}) — {cluster["name"]}',
                         fontsize=12, fontweight='bold',
                         color=CLUSTER_COLORS[row_idx])
            ax.set_ylim(0, global_ymax)
            ax.grid(True, alpha=0.3, axis='y')

        # hide unused cells
        for col_idx in range(n_countries, n_cols):
            ax_empty = fig.add_subplot(gs[row_idx, col_idx])
            ax_empty.set_visible(False)

    # ── shared legend ─────────────────────────────────────────────────
    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', fontsize=10,
                   framealpha=0.95, bbox_to_anchor=(1.0, 0.5), ncol=1,
                   title='Component', title_fontsize=11)

    out_png = os.path.join(graphs_dir, 'HBS_cluster_comparison_8countries.png')
    out_svg = os.path.join(graphs_dir, 'HBS_cluster_comparison_8countries.svg')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_svg}")


def plot_decile_lines(all_components, dirs):
    """
    One figure with 4x2 subplots (same country layout), lines only:
    - Basic needs (sum of 5 components)
    - Overall consumption
    - Equivalized income (PPS)
    Uses a common y-axis max across subplots with +10% headroom.
    """
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)

    decile_order = ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10']
    n_rows = len(CLUSTERS)
    n_cols = max(len(c['countries']) for c in CLUSTERS)

    # Compute a global max over all plotted series.
    global_line_max = 0.0
    for _, cdf in all_components.items():
        if cdf is None or cdf.empty:
            continue
        basic_needs = np.zeros(len(cdf))
        for comp in DISPLAY_COMPONENTS:
            if comp in cdf.columns:
                basic_needs += cdf[comp].fillna(0).values

        for col in ['total_consumption', 'equivalized_income_pps']:
            if col in cdf.columns:
                vals = pd.to_numeric(cdf[col], errors='coerce').values
                finite_vals = vals[np.isfinite(vals)]
                if finite_vals.size > 0:
                    global_line_max = max(global_line_max, finite_vals.max())

        finite_basic = basic_needs[np.isfinite(basic_needs)]
        if finite_basic.size > 0:
            global_line_max = max(global_line_max, finite_basic.max())

    if global_line_max <= 0:
        global_line_max = 1.0
    global_line_max *= 1.10

    fig = plt.figure(figsize=(8 * n_cols, 6 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25)

    n_countries_total = sum(len(c['countries']) for c in CLUSTERS)
    fig.suptitle(
        f'Decile Lines — {n_countries_total} Countries Grouped by EWBI Cluster (2020)\n'
        'Basic needs (5 components) · Overall consumption · Equivalized income (PPS)',
        fontsize=15,
        fontweight='bold',
        y=1.002,
    )

    first_ax = None
    for row_idx, cluster in enumerate(CLUSTERS):
        n_countries = len(cluster['countries'])
        for col_idx, cc in enumerate(cluster['countries']):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if first_ax is None:
                first_ax = ax

            cname = HIGHLIGHT_COUNTRIES[cc]
            cdf = all_components.get(cc)
            if cdf is None or cdf.empty:
                ax.text(0.5, 0.5, f'{cname}\n(no data)', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{cname} ({cc}) — {cluster["name"]}', fontsize=12, fontweight='bold')
                ax.set_ylim(0, global_line_max)
                continue

            cdf['decile'] = pd.Categorical(cdf['decile'], categories=decile_order, ordered=True)
            cdf = cdf.sort_values('decile')

            x = np.arange(len(cdf))
            basic_needs = np.zeros(len(cdf))
            for comp in DISPLAY_COMPONENTS:
                if comp in cdf.columns:
                    basic_needs += cdf[comp].fillna(0).values

            total_vals = pd.to_numeric(cdf.get('total_consumption', np.nan), errors='coerce').values
            income_vals = pd.to_numeric(cdf.get('equivalized_income_pps', np.nan), errors='coerce').values

            ax.plot(x, basic_needs, color='#1f78b4', linewidth=2.2, marker='o', markersize=4, label='Basic needs (5 components)')
            ax.plot(x, total_vals, color='#33a02c', linewidth=2.0, marker='s', markersize=4, label='Overall consumption')
            ax.plot(x, income_vals, color='black', linewidth=2.0, marker='^', markersize=4, label='Equivalized income (PPS)')

            ax.set_xticks(x)
            ax.set_xticklabels(cdf['decile'].values, fontsize=9)
            ax.set_xlabel('Income Decile', fontsize=10, fontweight='bold')
            ax.set_ylabel('PPS per equivalized person', fontsize=10, fontweight='bold')
            ax.set_title(f'{cname} ({cc}) — {cluster["name"]}',
                         fontsize=12, fontweight='bold', color=CLUSTER_COLORS[row_idx])
            ax.set_ylim(0, global_line_max)
            ax.grid(True, alpha=0.3, axis='both')

        for col_idx in range(n_countries, n_cols):
            ax_empty = fig.add_subplot(gs[row_idx, col_idx])
            ax_empty.set_visible(False)

    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', fontsize=10,
                   framealpha=0.95, bbox_to_anchor=(1.0, 0.5), ncol=1,
                   title='Series', title_fontsize=11)

    out_png = os.path.join(graphs_dir, 'HBS_cluster_decile_lines_8countries.png')
    out_svg = os.path.join(graphs_dir, 'HBS_cluster_decile_lines_8countries.svg')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_svg}")


# ── Excel export ──────────────────────────────────────────────────────────

def export_excel(all_components, dirs):
    """
    One sheet per country.  Rows = deciles, columns = consumption components.
    """
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)

    cols_to_export = ['decile', 'Housing', 'Transport', 'Food & Beverage',
                      'Health', 'Education', 'Other (Residual)', 'total_consumption', 'equivalized_income_pps']

    xlsx_path = os.path.join(graphs_dir, 'HBS_cluster_comparison_8countries_data.xlsx')
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for cc in HIGHLIGHT_COUNTRIES:
            cdf = all_components.get(cc)
            if cdf is None or cdf.empty:
                continue
            cname = HIGHLIGHT_COUNTRIES[cc]
            sheet = cdf[[c for c in cols_to_export if c in cdf.columns]].copy()
            sheet = sheet.set_index('decile')
            sheet = sheet.round(1)
            sheet.to_excel(writer, sheet_name=f'{cname} ({cc})')

    print(f"  Saved: {xlsx_path}")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 80)
    print("HBS CLUSTER COMPARISON — 8 COUNTRIES (4 CLUSTERS)")
    print("=" * 80)

    dirs = setup_directories()
    pps_df = load_pps_data(dirs)

    all_components = {}
    active_countries = copy.deepcopy(HIGHLIGHT_COUNTRIES)
    active_clusters = copy.deepcopy(CLUSTERS)

    # First pass for all countries except the ones requested for replacement.
    for cc, cname in HIGHLIGHT_COUNTRIES.items():
        if cc in {'HU', 'PL'}:
            continue
        comp = build_country_components(cc, cname, pps_df)
        if not comp.empty:
            all_components[cc] = comp

    # Forced replacement handling for HU and PL in their respective clusters.
    for target_code in ['HU', 'PL']:
        if target_code not in active_countries:
            continue
        repl_code, repl_name, repl_comp = resolve_country_replacement(
            target_code=target_code,
            pps_df=pps_df,
            used_codes=set(active_countries.keys())
        )
        if repl_code is not None:
            # Swap target for replacement in countries and cluster definition.
            active_countries.pop(target_code, None)
            active_countries[repl_code] = repl_name
            all_components[repl_code] = repl_comp

            for cluster in active_clusters:
                updated = []
                for c in cluster['countries']:
                    updated.append(repl_code if c == target_code else c)
                cluster['countries'] = updated
        else:
            print(f"  WARN: No valid replacement found for {target_code}. Cluster will contain one country.")
            active_countries.pop(target_code, None)
            for cluster in active_clusters:

                cluster['countries'] = [c for c in cluster['countries'] if c != target_code]

    # Apply active config globally for plotting/export functions.
    globals()['HIGHLIGHT_COUNTRIES'] = active_countries
    globals()['CLUSTERS'] = active_clusters

    if all_components:
        plot_cluster_comparison(all_components, dirs)
        plot_decile_lines(all_components, dirs)
        export_excel(all_components, dirs)
    else:
        print("ERROR: no data for any country")

    print("\nDONE.\n")


if __name__ == "__main__":
    main()
