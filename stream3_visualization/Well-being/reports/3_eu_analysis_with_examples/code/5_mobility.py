"""
5_mobility.py — Mobility visuals for 4 reports.

Produces 3 visuals per report:
  1. Vehicle fleet by motor energy type (stacked area, panel per country)
  2. Passenger transport per capita by mode (stacked area + total line, panel per country)
  3. Vehicle weight categories (multi-line, panel per country)

Data source: Eurostat (motor energy, car weight, population), EEA (pkm by mode).

Reports:
  rep_eu   : EU-27 countries, cluster grouping
  rep_ewbi : EU-27 + EFTA, cluster grouping
  rep_fr   : France focus, comparison countries (no cluster grouping)
  rep_ch   : Switzerland focus, comparison countries (no cluster grouping)

All outputs: PNG + SVG + Excel.

Special handling:
  - For `rep_fr`, CH PKM (from FSO source) has `Active mobility` set to 0
    (EEA data is empty for CH; FSO active mobility is excluded for comparability).
"""

import os
import csv
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
EXT_DATA = os.path.join(BASE_DIR, 'external_data')
OUTPUT_BASE = os.path.join(BASE_DIR, 'outputs', 'graphs', 'Mobility')
os.makedirs(OUTPUT_BASE, exist_ok=True)

MOTOR_NRG_PATH = os.path.join(
    EXT_DATA, 'eurostat_road_eqs_carpda__custom_20787497_linear.csv')
CAR_WEIGHT_PATH = os.path.join(
    EXT_DATA, 'eurostat_road_eqs_unlweig__custom_20787872_linear.csv')
POP_PATH = os.path.join(EXT_DATA, 'eurostat_population.csv')
EEA_PKM_PATH = os.path.join(
    EXT_DATA,
    'eea_fr_passenger_transport_activity_(gpkm)_for_different_transport_modes.csv')
FSO_CH_PKM_PATH = os.path.join(
    EXT_DATA, 'fso_ch_gr-e-11.04.01.02-je.csv')
FSO_CH_CAR_PATH = os.path.join(
    EXT_DATA, 'fso_car_gr-f-11.03.02.01.01-cc.csv')

# ---------------------------------------------------------------------------
# Country mapping (geo name → ISO)
# ---------------------------------------------------------------------------
GEO_TO_ISO = {
    'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
    'Cyprus': 'CY', 'Czechia': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE',
    'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'EL',
    'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE', 'Italy': 'IT',
    'Latvia': 'LV', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT',
    'Netherlands': 'NL', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT',
    'Romania': 'RO', 'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES',
    'Sweden': 'SE', 'Switzerland': 'CH',
}
ISO_TO_GEO = {v: k for k, v in GEO_TO_ISO.items()}

COUNTRY_NAME_MAP = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CH': 'Switzerland',
    'CY': 'Cyprus', 'CZ': 'Czech Republic', 'DE': 'Germany', 'DK': 'Denmark',
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

# ---------------------------------------------------------------------------
# Motor energy categories & colours
# ---------------------------------------------------------------------------
MOTOR_ENERGY_CATEGORIES = [
    'Petrol (excl. hybrids)',
    'Diesel (excl. hybrids)',
    'Hybrids (regular)',
    'Plug-in hybrids',
    'Electricity',
    'Other alternative',
]
MOTOR_ENERGY_COLORS = {
    'Petrol (excl. hybrids)': '#FF6B6B',
    'Diesel (excl. hybrids)': '#4ECDC4',
    'Hybrids (regular)':      '#45B7D1',
    'Plug-in hybrids':        '#96CEB4',
    'Electricity':            '#FFEAA7',
    'Other alternative':      '#DDA0DD',
}

# ---------------------------------------------------------------------------
# PKM mode config
# ---------------------------------------------------------------------------
PKM_MODES = [
    ('Cars',            '#1f77b4'),
    ('Buses & coaches', '#ff7f0e'),
    ('Rail',            '#2ca02c'),
    ('High-speed rail', '#17becf'),
    ('Tram & metro',    '#9467bd'),
    ('Active mobility', '#8c564b'),
]
PKM_MODE_NAMES = [m[0] for m in PKM_MODES]
PKM_MODE_COLORS = {m[0]: m[1] for m in PKM_MODES}

# ---------------------------------------------------------------------------
# Weight categories
# ---------------------------------------------------------------------------
WEIGHT_CATS = [
    ('Less than 1 000 kg',    'KG_LT1000 < 1,000 kg',       '#2E86C1'),
    ('From 1 000 to 1 249 kg','KG1000-1249 1,000-1,249 kg',  '#28B463'),
    ('From 1 250 to 1 499 kg','KG1250-1499 1,250-1,499 kg',  '#F39C12'),
    ('1 500 kg or over',      'KG_GE1500 >= 1,500 kg',       '#E74C3C'),
]
WEIGHT_LABELS = [lbl for _, lbl, _ in WEIGHT_CATS]
WEIGHT_COLORS = {lbl: c for _, lbl, c in WEIGHT_CATS}

# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
CLUSTERS = [
    {
        'id': 0, 'label': 'Low performer / Low EWBI', 'color': '#fb8072',
        'countries': {'CY', 'FR', 'EL', 'MT', 'PT', 'ES', 'UK'},
    },
    {
        'id': 1, 'label': 'Low performer / High EWBI', 'color': '#fdb462',
        'countries': {'AT', 'BE', 'CH', 'DK', 'FI', 'IE', 'IT', 'LU', 'NO', 'NL'},
    },
    {
        'id': 2, 'label': 'High performer / Low EWBI', 'color': '#8dd3c7',
        'countries': {'BG', 'HR', 'HU', 'LV', 'LT', 'RS', 'RO'},
    },
    {
        'id': 3, 'label': 'High performer / High EWBI', 'color': '#80b1d3',
        'countries': {'CZ', 'EE', 'DE', 'IS', 'PL', 'SK', 'SI', 'SE'},
    },
]

CLUSTER_COLOR_BY_COUNTRY = {}
CLUSTER_ID_BY_COUNTRY = {}
for cl in CLUSTERS:
    for cc in cl['countries']:
        CLUSTER_COLOR_BY_COUNTRY[cc] = cl['color']
        CLUSTER_ID_BY_COUNTRY[cc] = cl['id']

# ---------------------------------------------------------------------------
# Report configurations
# ---------------------------------------------------------------------------
REPORT_CONFIGS = {
    'rep_eu': {
        'prefix': 'rep_eu',
        'title_suffix': '(EU-27)',
        'country_filter': EU27_CODES,
        'comparison_countries': ['FR', 'ES', 'BE', 'NL', 'LT', 'HR', 'DE', 'SE'],
        'car_weight_comparison_countries': ['FR', 'ES', 'DK', 'NL', 'HU', 'HR', 'DE', 'SE'],
        'show_clusters': True,
    },
    'rep_ewbi': {
        'prefix': 'rep_ewbi',
        'title_suffix': '(EU-27 + EFTA)',
        'country_filter': EU27_CODES | EFTA_CODES,
        'comparison_countries': ['FR', 'ES', 'BE', 'NL', 'LT', 'HR', 'DE', 'SE',
                                 'CH', 'NO', 'IS'],
        'car_weight_comparison_countries': ['FR', 'ES', 'DK', 'NL', 'HU', 'HR', 'DE', 'SE',
                                            'CH', 'NO', 'IS'],
        'show_clusters': True,
    },
    'rep_fr': {
        'prefix': 'rep_fr',
        'title_suffix': '(France)',
        'country_filter': EU27_CODES | EFTA_CODES,
        'comparison_countries': ['FR', 'ES', 'IT', 'CH', 'DE', 'BE'],
        'pkm_country_overrides': {
            'CH': {'drop_modes': ['Active mobility']},
        },
        'show_clusters': False,
    },
    'rep_ch': {
        'prefix': 'rep_ch',
        'title_suffix': '(Switzerland)',
        'country_filter': EU27_CODES | EFTA_CODES,
        'comparison_countries': ['CH', 'FR', 'IT', 'DE', 'AT'],
        'cars_per_capita_comparison_countries': ['CH', 'FR', 'IT', 'DE', 'AT', 'LT'],
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


def _build_country_order(cfg, available_set):
    comparison = cfg['comparison_countries']
    if cfg['show_clusters']:
        # Only comparison countries, ordered by cluster
        ordered = []
        for cl in CLUSTERS:
            cl_comp = [cc for cc in comparison if cc in cl['countries'] and cc in available_set]
            ordered.extend(cl_comp)
        return ordered
    else:
        return [cc for cc in comparison if cc in available_set]


def _cluster_label_for(cc):
    cl_id = CLUSTER_ID_BY_COUNTRY.get(cc)
    if cl_id is not None:
        return CLUSTERS[cl_id]['label']
    return ''


def _subplot_grid(country_order, show_clusters):
    """Determine grid dimensions: 2 columns, as many rows as needed."""
    n = len(country_order)
    n_cols = min(2, n)
    n_rows = (n + n_cols - 1) // n_cols
    return n_rows, n_cols


# ============================================================================
# DATA LOADING — Motor energy
# ============================================================================
def _build_motor_energy_pct(df_filtered):
    df_filtered['OBS_VALUE'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    pivot = df_filtered.pivot_table(
        index='TIME_PERIOD', columns='mot_nrg',
        values='OBS_VALUE', aggfunc='first').fillna(0)

    cats = pd.DataFrame(index=pivot.index)
    cats['Petrol (excl. hybrids)'] = pivot.get('Petrol (excluding hybrids) \xa0', 0)
    cats['Diesel (excl. hybrids)'] = pivot.get('Diesel (excluding hybrids) \xa0', 0)
    cats['Hybrids (regular)'] = (
        pivot.get('Hybrid diesel-electric', 0) +
        pivot.get('Hybrid electric-petrol', 0))
    cats['Plug-in hybrids'] = (
        pivot.get('Plug-in hybrid diesel-electric \xa0', 0) +
        pivot.get('Plug-in hybrid petrol-electric \xa0', 0))
    cats['Electricity'] = pivot.get('Electricity', 0)
    other_keys = ['Bi-fuel', 'Biodiesel', 'Bioethanol', 'Natural gas',
                  'Hydrogen and fuel cells\xa0', 'Liquefied petroleum gases (LPG)', 'Other']
    cats['Other alternative'] = sum(pivot.get(k, 0) for k in other_keys)

    total = pivot.get('Total', cats.sum(axis=1))
    pct = cats.div(total, axis=0) * 100
    return pct.sort_index()


def load_all_motor_energy():
    """Load motor energy % for all available countries. Returns {iso: DataFrame}."""
    df = pd.read_csv(MOTOR_NRG_PATH)
    result = {}
    for geo_name, iso in GEO_TO_ISO.items():
        sub = df[df['geo'] == geo_name].copy()
        if sub.empty:
            continue
        pct = _build_motor_energy_pct(sub)
        # Drop rows where sum < 95% (incomplete)
        row_sums = pct[MOTOR_ENERGY_CATEGORIES].sum(axis=1)
        pct = pct[row_sums >= 95].copy()
        if not pct.empty:
            result[iso] = pct
    return result


# ============================================================================
# DATA LOADING — PKM per capita
# ============================================================================
def _load_country_mode_gpkm(country_code):
    """Load country-level Gpkm by mode from EEA CSV."""
    if not os.path.exists(EEA_PKM_PATH):
        return pd.DataFrame()

    with open(EEA_PKM_PATH, mode='r', encoding='utf-8', newline='') as f:
        rows = list(csv.reader(f))

    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == 'Years - 1':
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()

    df = pd.read_csv(EEA_PKM_PATH, skiprows=header_idx)
    first_col = df.columns[0]
    years = pd.to_numeric(df[first_col], errors='coerce')
    df = df[years.notna()].copy()
    df['year'] = years[years.notna()].astype(int)
    df = df.set_index('year').sort_index()

    col_map = {
        'Cars': f'{country_code} Cars passangers (Gpkm)',
        'Buses & coaches': f'{country_code} Buses and coaches passangers (Gpkm)',
        'Rail': f'{country_code} Rail passangers (Gpkm)',
        'High-speed rail': f'{country_code} High-speed rail passangers (Gpkm)',
        'Tram & metro': f'{country_code} Tram and metro passangers (Gpkm)',
    }

    out = pd.DataFrame(index=df.index)
    for label, src_col in col_map.items():
        if src_col in df.columns:
            out[label] = pd.to_numeric(df[src_col], errors='coerce').fillna(0)
        else:
            out[label] = 0.0
    out.index.name = 'year'
    return out


def _load_population(country_name):
    """Load population series for a country from Eurostat CSV."""
    df = pd.read_csv(POP_PATH)
    sub = df[df['geo'] == country_name].copy()
    sub['year'] = pd.to_numeric(sub['TIME_PERIOD'], errors='coerce').astype('Int64')
    sub['pop'] = pd.to_numeric(sub['OBS_VALUE'], errors='coerce')
    return sub.set_index('year')['pop'].dropna()


def _load_ch_fso_gpkm():
    """Load Swiss FSO passenger-km data (Gpkm) from transposed CSV."""
    if not os.path.exists(FSO_CH_PKM_PATH):
        return pd.DataFrame()
    df = pd.read_csv(FSO_CH_PKM_PATH, index_col=0)
    # Transpose: rows=modes, cols=years → rows=years, cols=modes
    df.columns = df.columns.astype(int)
    df = df.T
    df.index.name = 'year'
    out = pd.DataFrame(index=df.index)
    out['Cars'] = pd.to_numeric(
        df.get('private motorised road transport', 0), errors='coerce').fillna(0)
    out['Buses & coaches'] = pd.to_numeric(
        df.get('public road transport', 0), errors='coerce').fillna(0)
    out['Rail'] = pd.to_numeric(
        df.get('rail and cable cars', 0), errors='coerce').fillna(0)
    out['High-speed rail'] = 0.0
    out['Tram & metro'] = 0.0
    out['Active mobility'] = pd.to_numeric(
        df.get('active mobility', 0), errors='coerce').fillna(0)
    return out


def load_all_pkm_per_capita():
    """Load pkm per capita for all available countries. Returns {iso: DataFrame}."""
    result = {}
    for iso, geo_name in ISO_TO_GEO.items():
        gpkm = _load_country_mode_gpkm(iso)
        if gpkm.empty:
            continue
        # Ensure Active mobility column exists for EEA-sourced data
        if 'Active mobility' not in gpkm.columns:
            gpkm['Active mobility'] = 0.0
        pop = _load_population(geo_name).reindex(gpkm.index)
        valid = pop.dropna().index
        data = gpkm.loc[gpkm.index.isin(valid)].copy()
        pop = pop.loc[data.index]
        for mode in PKM_MODE_NAMES:
            data[mode] = data[mode].fillna(0) * 1e9 / pop
        if not data.empty:
            result[iso] = data
    # Override CH with Swiss FSO data (EEA has no CH values)
    ch_gpkm = _load_ch_fso_gpkm()
    if not ch_gpkm.empty:
        ch_pop = _load_population('Switzerland').reindex(ch_gpkm.index)
        valid = ch_pop.dropna().index
        ch_data = ch_gpkm.loc[ch_gpkm.index.isin(valid)].copy()
        ch_pop = ch_pop.loc[ch_data.index]
        for mode in PKM_MODE_NAMES:
            ch_data[mode] = ch_data[mode].fillna(0) * 1e9 / ch_pop
        if not ch_data.empty:
            result['CH'] = ch_data
    return result


def _apply_pkm_overrides(cfg, all_data):
    """Apply optional PKM country overrides for a specific report config."""
    overrides = cfg.get('pkm_country_overrides', {})
    if not overrides:
        return all_data

    out = dict(all_data)
    for target_iso, rule in overrides.items():
        source_iso = rule.get('source_iso', target_iso)
        src = all_data.get(source_iso)
        if src is None or src.empty:
            print(f"    PKM override skipped: no data for '{source_iso}'")
            continue

        data = src.copy()
        for mode in rule.get('drop_modes', []):
            if mode in data.columns:
                data[mode] = 0.0

        out[target_iso] = data
        dropped = rule.get('drop_modes', [])
        print(f"    PKM override applied: {target_iso} (drop {dropped})")

    return out


# ============================================================================
# DATA LOADING — Car weight
# ============================================================================
def _build_weight_pct(df_filtered):
    df_filtered['OBS_VALUE'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    pivot = df_filtered.pivot_table(
        index='TIME_PERIOD', columns='weight',
        values='OBS_VALUE', aggfunc='first').fillna(0)

    total = pivot.get('Total', 0)
    pct = pd.DataFrame(index=pivot.index)
    for raw_name, label, _ in WEIGHT_CATS:
        if raw_name in pivot.columns:
            pct[label] = pivot[raw_name] / total.replace(0, np.nan) * 100
        else:
            pct[label] = np.nan
    return pct.sort_index()


def load_all_car_weight():
    """Load car weight % for all available countries. Returns {iso: DataFrame}."""
    df = pd.read_csv(CAR_WEIGHT_PATH)
    result = {}
    for geo_name, iso in GEO_TO_ISO.items():
        sub = df[df['geo'] == geo_name].copy()
        if sub.empty:
            continue
        pct = _build_weight_pct(sub)
        pct = pct.replace(0, np.nan)
        pct = pct.dropna(subset=[l for l in WEIGHT_LABELS if l in pct.columns], how='all')
        if not pct.empty:
            result[iso] = pct
    return result


# ============================================================================
# VISUAL 1: Motor energy — stacked area, panel per country
# ============================================================================
def plot_motor_energy(cfg, all_data, out_dir):
    print(f"  [1] Motor energy fleet composition...")
    available = {cc for cc in all_data if cc in cfg['country_filter'] or
                 cc in cfg['comparison_countries']}
    country_order = _build_country_order(cfg, available)
    if not country_order:
        print("    No countries with data")
        return

    n_rows, n_cols = _subplot_grid(country_order, cfg['show_clusters'])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    excel_sheets = {}
    for idx, cc in enumerate(country_order):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        data = all_data.get(cc)
        cname = COUNTRY_NAME_MAP.get(cc, cc)
        cl_label = _cluster_label_for(cc)
        title_color = CLUSTER_COLOR_BY_COUNTRY.get(cc, '#333333')

        if cfg['show_clusters'] and cl_label:
            ax.set_title(f'{cname} ({cc}) — {cl_label}',
                         fontsize=11, fontweight='bold', color=title_color)
        else:
            ax.set_title(f'{cname} ({cc})', fontsize=11, fontweight='bold')

        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='grey')
        else:
            years = data.index
            bottom = np.zeros(len(years))
            for cat in MOTOR_ENERGY_CATEGORIES:
                if cat not in data.columns:
                    continue
                vals = data[cat].fillna(0).values
                ax.fill_between(years, bottom, bottom + vals,
                                color=MOTOR_ENERGY_COLORS.get(cat, '#999'),
                                alpha=0.8, label=cat)
                bottom += vals

            # Excel data
            edf = data[MOTOR_ENERGY_CATEGORIES].copy()
            edf.index.name = 'Year'
            excel_sheets[f'{cname} ({cc})'] = edf.reset_index()

        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.25)
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Fleet share (%)', fontsize=9)

    # Hide unused axes
    total_slots = n_rows * n_cols
    for idx in range(len(country_order), total_slots):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    # Legend from first axis
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=9, title='Motor energy', title_fontsize=10)

    fig.suptitle(f'Vehicle Fleet by Motor Energy Type\n{cfg["title_suffix"]}',
                 fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_motor_energy.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    if excel_sheets:
        xlsx = os.path.join(out_dir, f'{cfg["prefix"]}_motor_energy.xlsx')
        _save_excel(excel_sheets, xlsx)


# ============================================================================
# VISUAL 2: PKM per capita — stacked area + dashed total, panel per country
# ============================================================================
def plot_pkm_per_capita(cfg, all_data, out_dir):
    print(f"  [2] PKM per capita by mode...")
    available = {cc for cc in all_data if cc in cfg['country_filter'] or
                 cc in cfg['comparison_countries']}
    country_order = _build_country_order(cfg, available)
    if not country_order:
        print("    No countries with data")
        return

    # Global y-max
    ymax = 0
    for cc in country_order:
        d = all_data.get(cc)
        if d is not None and not d.empty:
            ymax = max(ymax, d[PKM_MODE_NAMES].sum(axis=1).max())
    ymax = ymax * 1.10 if ymax > 0 else 1

    n_rows, n_cols = _subplot_grid(country_order, cfg['show_clusters'])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    excel_sheets = {}
    for idx, cc in enumerate(country_order):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        data = all_data.get(cc)
        cname = COUNTRY_NAME_MAP.get(cc, cc)
        cl_label = _cluster_label_for(cc)
        title_color = CLUSTER_COLOR_BY_COUNTRY.get(cc, '#333333')

        if cfg['show_clusters'] and cl_label:
            ax.set_title(f'{cname} ({cc}) — {cl_label}',
                         fontsize=11, fontweight='bold', color=title_color)
        else:
            ax.set_title(f'{cname} ({cc})', fontsize=11, fontweight='bold')

        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='grey')
        else:
            ax.stackplot(
                data.index,
                *[data[m].fillna(0) for m in PKM_MODE_NAMES],
                labels=PKM_MODE_NAMES,
                colors=[PKM_MODE_COLORS[m] for m in PKM_MODE_NAMES],
                alpha=0.85,
            )
            total = data[PKM_MODE_NAMES].sum(axis=1)
            ax.plot(data.index, total, color='black', linewidth=1.2,
                    linestyle='--', label='_nolegend_')

            edf = data[PKM_MODE_NAMES].copy()
            edf['Total'] = total
            edf.index.name = 'Year'
            excel_sheets[f'{cname} ({cc})'] = edf.reset_index()

        ax.set_ylim(0, ymax)
        ax.grid(axis='y', alpha=0.25)
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('pkm per capita', fontsize=9)

    total_slots = n_rows * n_cols
    for idx in range(len(country_order), total_slots):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=9, title='Mode', title_fontsize=10)

    fig.suptitle(
        f'Passenger Transport Activity per Capita by Mode\n{cfg["title_suffix"]}',
        fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_pkm_per_capita.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    if excel_sheets:
        xlsx = os.path.join(out_dir, f'{cfg["prefix"]}_pkm_per_capita.xlsx')
        _save_excel(excel_sheets, xlsx)


# ============================================================================
# DATA LOADING — Cars per capita
# ============================================================================
def load_all_cars_per_capita():
    """Load cars per 1,000 inhabitants for all available countries.

    Eurostat 'Total' motor energy rows give total car stock for EU/EFTA.
    Switzerland uses the FSO 'Voitures de tourisme' column.
    Returns {iso: Series(index=year, values=cars per 1000 inhabitants)}.
    """
    # --- Eurostat car stock (Total motor energy) ---
    df = pd.read_csv(MOTOR_NRG_PATH)
    result = {}
    for geo_name, iso in GEO_TO_ISO.items():
        sub = df[(df['geo'] == geo_name) & (df['mot_nrg'] == 'Total')].copy()
        if sub.empty:
            continue
        sub['year'] = pd.to_numeric(sub['TIME_PERIOD'], errors='coerce').astype('Int64')
        sub['cars'] = pd.to_numeric(sub['OBS_VALUE'], errors='coerce')
        sub = sub.dropna(subset=['year', 'cars']).set_index('year').sort_index()
        pop = _load_population(geo_name).reindex(sub.index)
        valid = pop.dropna().index
        if valid.empty:
            continue
        per_k = (sub.loc[valid, 'cars'] / pop.loc[valid]) * 1000
        per_k = per_k.dropna()
        if not per_k.empty:
            result[iso] = per_k

    # --- Switzerland: FSO data ---
    if os.path.exists(FSO_CH_CAR_PATH):
        fso = pd.read_csv(FSO_CH_CAR_PATH)
        year_col = fso.columns[0]  # 'Année'
        fso['year'] = pd.to_numeric(fso[year_col], errors='coerce').astype('Int64')
        fso['cars'] = pd.to_numeric(fso['Voitures de tourisme'], errors='coerce')
        fso = fso.dropna(subset=['year', 'cars']).set_index('year').sort_index()
        ch_pop = _load_population('Switzerland').reindex(fso.index)
        valid = ch_pop.dropna().index
        if not valid.empty:
            per_k = (fso.loc[valid, 'cars'] / ch_pop.loc[valid]) * 1000
            per_k = per_k.dropna()
            if not per_k.empty:
                result['CH'] = per_k

    return result


# ============================================================================
# VISUAL 4: Cars per capita — single line chart, all comparison countries
# ============================================================================
def plot_cars_per_capita(cfg, all_data, out_dir):
    print(f"  [4] Cars per 1,000 inhabitants...")
    cpc_cfg = dict(cfg)
    if 'cars_per_capita_comparison_countries' in cfg:
        cpc_cfg['comparison_countries'] = cfg['cars_per_capita_comparison_countries']
    available = {cc for cc in all_data if cc in cpc_cfg['country_filter'] or
                 cc in cpc_cfg['comparison_countries']}
    country_order = _build_country_order(cpc_cfg, available)
    if not country_order:
        print("    No countries with data")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Use distinct colors for non-cluster reports
    DISTINCT_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3',
                       '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62']

    excel_rows = []
    for idx, cc in enumerate(country_order):
        s = all_data.get(cc)
        if s is None or s.empty:
            continue
        cname = COUNTRY_NAME_MAP.get(cc, cc)
        if cfg['show_clusters']:
            color = CLUSTER_COLOR_BY_COUNTRY.get(cc, DISTINCT_COLORS[idx % len(DISTINCT_COLORS)])
        else:
            color = DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]
        lw = 2.5 if cc == cfg['comparison_countries'][0] else 1.8
        ax.plot(s.index, s.values, linewidth=lw, label=f'{cname} ({cc})',
                color=color, marker='o', markersize=3)
        for yr, val in s.items():
            excel_rows.append({'Country': cc, 'Year': yr,
                               'Cars_per_1000': val})

    ax.set_title(f'Passenger Cars per 1,000 Inhabitants\n{cfg["title_suffix"]}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Cars per 1,000 inhabitants', fontsize=11)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_cars_per_capita.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    if excel_rows:
        xlsx = os.path.join(out_dir, f'{cfg["prefix"]}_cars_per_capita.xlsx')
        _save_excel(pd.DataFrame(excel_rows), xlsx, sheet_name='Cars per 1000')


# ============================================================================
# VISUAL 3: Car weight — multi-line, panel per country
# ============================================================================
def plot_car_weight(cfg, all_data, out_dir):
    print(f"  [3] Car weight categories...")
    # Use car-weight-specific comparison list when available
    cw_cfg = dict(cfg)
    if 'car_weight_comparison_countries' in cfg:
        cw_cfg['comparison_countries'] = cfg['car_weight_comparison_countries']
    available = {cc for cc in all_data if cc in cw_cfg['country_filter'] or
                 cc in cw_cfg['comparison_countries']}
    country_order = _build_country_order(cw_cfg, available)
    if not country_order:
        print("    No countries with data")
        return

    # Global y-max
    global_max = 0
    for cc in country_order:
        d = all_data.get(cc)
        if d is not None and not d.empty:
            cols = [l for l in WEIGHT_LABELS if l in d.columns]
            if cols:
                global_max = max(global_max, d[cols].max().max())
    global_max = global_max * 1.15 if global_max > 0 else 60

    n_rows, n_cols = _subplot_grid(country_order, cfg['show_clusters'])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    excel_sheets = {}
    for idx, cc in enumerate(country_order):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        data = all_data.get(cc)
        cname = COUNTRY_NAME_MAP.get(cc, cc)
        cl_label = _cluster_label_for(cc)
        title_color = CLUSTER_COLOR_BY_COUNTRY.get(cc, '#333333')

        if cfg['show_clusters'] and cl_label:
            ax.set_title(f'{cname} ({cc}) — {cl_label}',
                         fontsize=11, fontweight='bold', color=title_color)
        else:
            ax.set_title(f'{cname} ({cc})', fontsize=11, fontweight='bold')

        if data is None or data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='grey')
        else:
            for lbl in WEIGHT_LABELS:
                if lbl in data.columns:
                    ax.plot(data.index, data[lbl], color=WEIGHT_COLORS[lbl],
                            linewidth=2.0, marker='o', markersize=4, label=lbl)

            edf = data[[l for l in WEIGHT_LABELS if l in data.columns]].copy()
            edf.index.name = 'Year'
            excel_sheets[f'{cname} ({cc})'] = edf.reset_index()

        ax.set_ylim(0, global_max)
        ax.grid(axis='y', alpha=0.25)
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Fleet share (%)', fontsize=9)

    total_slots = n_rows * n_cols
    for idx in range(len(country_order), total_slots):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=9, title='Weight class', title_fontsize=10)

    fig.suptitle(f'Vehicle Weight Categories\n{cfg["title_suffix"]}',
                 fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_car_weight.png')
    _save_fig(fig, out_png)
    plt.close(fig)

    if excel_sheets:
        xlsx = os.path.join(out_dir, f'{cfg["prefix"]}_car_weight.xlsx')
        _save_excel(excel_sheets, xlsx)


# ============================================================================
# Main
# ============================================================================
def generate_report(report_key, motor_data, pkm_data, weight_data, car_capita_data):
    cfg = REPORT_CONFIGS[report_key]
    print(f"\n{'='*60}")
    print(f"Generating mobility report: {cfg['prefix']} {cfg['title_suffix']}")
    print(f"{'='*60}")

    out_dir = os.path.join(OUTPUT_BASE, cfg['prefix'])
    os.makedirs(out_dir, exist_ok=True)

    plot_motor_energy(cfg, motor_data, out_dir)
    pkm_data_for_report = _apply_pkm_overrides(cfg, pkm_data)
    plot_pkm_per_capita(cfg, pkm_data_for_report, out_dir)
    plot_car_weight(cfg, weight_data, out_dir)
    plot_cars_per_capita(cfg, car_capita_data, out_dir)

    print(f"\n  All outputs saved to: {out_dir}")


def main():
    print("Loading motor energy data...")
    motor_data = load_all_motor_energy()
    print(f"  {len(motor_data)} countries loaded")

    print("Loading PKM per capita data...")
    pkm_data = load_all_pkm_per_capita()
    print(f"  {len(pkm_data)} countries loaded")

    print("Loading car weight data...")
    weight_data = load_all_car_weight()
    print(f"  {len(weight_data)} countries loaded")

    print("Loading cars per capita data...")
    car_capita_data = load_all_cars_per_capita()
    print(f"  {len(car_capita_data)} countries loaded")

    for report_key in ['rep_eu', 'rep_ewbi', 'rep_fr', 'rep_ch']:
        generate_report(report_key, motor_data, pkm_data, weight_data, car_capita_data)
    print("\nDone.")


if __name__ == '__main__':
    main()
