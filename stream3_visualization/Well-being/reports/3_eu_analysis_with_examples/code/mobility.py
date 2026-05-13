"""
EU-27 Passenger Transport Activity (Gpkm)
Stacked area charts: absolute and per-capita, by transport mode.

Sources
-------
- EEA: EU-27 aggregate Gpkm for all modes (1995-2022)
- Eurostat: rail per country (MIO_PKM, 2008-2024) — used for sub-breakdown
- Eurostat: road per country (MIO_PKM, 2008-2024) — sparse, not used for aggregate
- Eurostat: population (EU-27, annual)

Note: Eurostat road data covers only 7-13 EU-27 countries per year and Eurostat
rail has coverage gaps for Germany/France/Belgium in several years.  The EEA
EU-27 aggregate is the only consistent source for the full union.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import csv

plt.rcParams['font.family'] = 'Arial'

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_DATA = os.path.join(BASE_DIR, 'external_data')
OUTPUT_GRAPH = os.path.join(BASE_DIR, 'outputs', 'graphs', 'mobility')
OUTPUT_DATA = os.path.join(BASE_DIR, 'outputs', 'data')
os.makedirs(OUTPUT_GRAPH, exist_ok=True)
os.makedirs(OUTPUT_DATA, exist_ok=True)

POCKETBOOK_PATH = os.path.join(EXT_DATA, 'ee_pocketbook.xlsx')
RAIL_PATH = os.path.join(
    EXT_DATA, 'eurostat_rail_pa_speed__custom_20786010_linear.csv')
ROAD_PATH = os.path.join(
    EXT_DATA, 'eurostat_road_pa_mov__custom_20786054_linear.csv')
POP_PATH = os.path.join(EXT_DATA, 'eurostat_population.csv')
MOTOR_NRG_PATH = os.path.join(
    EXT_DATA, 'eurostat_road_eqs_carpda__custom_20787497_linear.csv')
CAR_WEIGHT_PATH = os.path.join(
    EXT_DATA, 'eurostat_road_eqs_unlweig__custom_20787872_linear.csv')

# EU-27 member states
EU27 = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
    'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
    'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia',
    'Slovenia', 'Spain', 'Sweden',
]

# Modes to plot (label, Excel column, colour)  — stacking order bottom to top
MODE_CONFIG = [
    ('Cars',             'Pass -enger Cars',  '#1f77b4'),
    ('Buses & coaches',  'Bus & Coach',       '#ff7f0e'),
    ('Powered 2W',       'P2W',               '#d62728'),
    ('Rail',             'Rail -way',         '#2ca02c'),
    ('Tram & metro',     'Tram & Metro',      '#9467bd'),
    ('Aviation',         'Air',               '#8c564b'),
    ('Maritime',         'Sea',               '#7f7f7f'),
]

# France modes (from pocketbook 'fr' sheet)
FR_MODE_CONFIG = [
    ('Cars',             'car',         '#1f77b4'),
    ('Buses & coaches',  'bus_coach',   '#ff7f0e'),
    ('Rail',             'train',       '#2ca02c'),
    ('High-speed rail',  'high-speed',  '#17becf'),
    ('Tram & metro',     'tram_metro',  '#9467bd'),
]

# 8-country cluster layout used across report visuals
CLUSTER_LAYOUT = [
    ('Low performer / Low EWBI',  ['FR', 'ES']),
    ('Low performer / High EWBI', ['BE', 'NL']),
    ('High performer / Low EWBI', ['LT', 'HR']),
    ('High performer / High EWBI', ['DE', 'AT']),
]

COUNTRY_NAMES = {
    'FR': 'France',
    'ES': 'Spain',
    'BE': 'Belgium',
    'NL': 'Netherlands',
    'LT': 'Lithuania',
    'HR': 'Croatia',
    'DE': 'Germany',
    'AT': 'Austria',
}

FALLBACK_COUNTRY_NAMES = {
    'LU': 'Luxembourg',
    'IE': 'Ireland',
    'DK': 'Denmark',
    'SE': 'Sweden',
    'LV': 'Latvia',
    'EE': 'Estonia',
    'HU': 'Hungary',
    'RO': 'Romania',
}

COUNTRY_CLUSTER_COLOR = {
    'FR': '#fb8072', 'ES': '#fb8072',
    'BE': '#fdb462', 'NL': '#fdb462',
    'LT': '#8dd3c7', 'HR': '#8dd3c7',
    'DE': '#80b1d3', 'AT': '#80b1d3',
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_eea():
    """Load EU-27 passenger transport data (Gpkm) from EC Pocketbook Excel."""
    df = pd.read_excel(POCKETBOOK_PATH, sheet_name='eu-27')
    df = df.set_index('year').sort_index()

    result = pd.DataFrame(index=df.index)
    for label, col, _ in MODE_CONFIG:
        if col in df.columns:
            result[label] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"  Warning: Pocketbook column '{col}' not found")
            result[label] = np.nan

    result.index.name = 'year'
    return result


def load_population():
    """Load EU-27 aggregate population from Eurostat."""
    df = pd.read_csv(POP_PATH)
    eu27 = df[df['geo'] == 'European Union - 27 countries (from 2020)'].copy()
    eu27['year'] = pd.to_numeric(eu27['TIME_PERIOD'], errors='coerce').astype('Int64')
    eu27['pop'] = pd.to_numeric(eu27['OBS_VALUE'], errors='coerce')
    return eu27.set_index('year')['pop'].dropna()


def load_eea_france():
    """Load France passenger transport data (Gpkm) from EC Pocketbook Excel."""
    df = pd.read_excel(POCKETBOOK_PATH, sheet_name='fr')
    df = df.set_index('year').sort_index()

    result = pd.DataFrame(index=df.index)
    for label, col, _ in FR_MODE_CONFIG:
        if col in df.columns:
            result[label] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"  Warning: Pocketbook column '{col}' not found")
            result[label] = np.nan

    result.index.name = 'year'
    return result


def load_population_france():
    """Load France population from Eurostat."""
    df = pd.read_csv(POP_PATH)
    fr = df[df['geo'] == 'France'].copy()
    fr['year'] = pd.to_numeric(fr['TIME_PERIOD'], errors='coerce').astype('Int64')
    fr['pop'] = pd.to_numeric(fr['OBS_VALUE'], errors='coerce')
    return fr.set_index('year')['pop'].dropna()


def load_population_country(country_name):
    """Load population series for one country name from Eurostat CSV."""
    df = pd.read_csv(POP_PATH)
    country = df[df['geo'] == country_name].copy()
    country['year'] = pd.to_numeric(country['TIME_PERIOD'], errors='coerce').astype('Int64')
    country['pop'] = pd.to_numeric(country['OBS_VALUE'], errors='coerce')
    return country.set_index('year')['pop'].dropna()


def _load_country_mode_gpkm_from_eea_csv(country_code):
    """Load country mode-level Gpkm from EEA CSV (same source used for France)."""
    with open(os.path.join(
            EXT_DATA, 'eea_fr_passenger_transport_activity_(gpkm)_for_different_transport_modes.csv'),
            mode='r', encoding='utf-8', newline='') as f:
        rows = list(csv.reader(f))

    header_idx = None
    for i, row in enumerate(rows):
        if row and len(row) > 0 and row[0].strip() == 'Years - 1':
            header_idx = i
            break

    if header_idx is None:
        raise ValueError('EEA country transport CSV header row not found (Years - 1).')

    df = pd.read_csv(
        os.path.join(EXT_DATA, 'eea_fr_passenger_transport_activity_(gpkm)_for_different_transport_modes.csv'),
        skiprows=header_idx)

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


def load_eurostat_rail():
    """Load Eurostat rail data for coverage analysis."""
    df = pd.read_csv(RAIL_PATH)
    df_eu = df[df['geo'].isin(EU27)].copy()
    df_eu['year'] = pd.to_numeric(df_eu['TIME_PERIOD'], errors='coerce')
    df_eu['value'] = pd.to_numeric(df_eu['OBS_VALUE'], errors='coerce')
    return df_eu


def load_eurostat_road():
    """Load Eurostat road data for coverage analysis."""
    df = pd.read_csv(ROAD_PATH)
    df_eu = df[df['geo'].isin(EU27)].copy()
    df_eu['year'] = pd.to_numeric(df_eu['TIME_PERIOD'], errors='coerce')
    df_eu['value'] = pd.to_numeric(df_eu['OBS_VALUE'], errors='coerce')
    return df_eu


# ============================================================================
# MOTOR ENERGY & CAR WEIGHT — DATA LOADING
# ============================================================================

# 6 grouped categories (same as Switzerland chart)
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

# Weight chart categories
WEIGHT_CATS = [
    ('Less than 1 000 kg',    'KG_LT1000 < 1,000 kg',       '#2E86C1'),
    ('From 1 000 to 1 249 kg','KG1000-1249 1,000-1,249 kg',  '#28B463'),
    ('From 1 250 to 1 499 kg','KG1250-1499 1,250-1,499 kg',  '#F39C12'),
    ('1 500 kg or over',      'KG_GE1500 >= 1,500 kg',       '#E74C3C'),
]


def _build_motor_energy_pct(df_filtered):
    """From a filtered DataFrame (single geo or sum), build 6-category percentages."""
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


def load_motor_energy(geo_filter):
    """Load motor energy percentages for a single geo string."""
    df = pd.read_csv(MOTOR_NRG_PATH)
    return _build_motor_energy_pct(df[df['geo'] == geo_filter].copy())


def _motor_energy_country_coverage(df_eu, geo, year):
    """Compute sum-of-6-categories / Total for one country-year."""
    gc = df_eu[(df_eu['TIME_PERIOD'] == year) & (df_eu['geo'] == geo)]
    gc = gc.set_index('mot_nrg')['OBS_VALUE']
    total = gc.get('Total', 0)
    if pd.isna(total) or total == 0:
        return 0.0

    hp = gc.get('Hybrid electric-petrol', 0) or 0
    hd = gc.get('Hybrid diesel-electric', 0) or 0
    pp = gc.get('Plug-in hybrid petrol-electric \xa0', 0) or 0
    pd2 = gc.get('Plug-in hybrid diesel-electric \xa0', 0) or 0

    if 'Petrol (excluding hybrids) \xa0' in gc.index:
        pet = gc['Petrol (excluding hybrids) \xa0'] or 0
    elif 'Petrol' in gc.index:
        pet = (gc['Petrol'] or 0) - hp - pp
    else:
        pet = 0
    if 'Diesel (excluding hybrids) \xa0' in gc.index:
        die = gc['Diesel (excluding hybrids) \xa0'] or 0
    elif 'Diesel' in gc.index:
        die = (gc['Diesel'] or 0) - hd - pd2
    else:
        die = 0

    elec = gc.get('Electricity', 0) or 0
    other_keys = ['Bi-fuel', 'Biodiesel', 'Bioethanol', 'Natural gas',
                  'Hydrogen and fuel cells\xa0', 'Liquefied petroleum gases (LPG)', 'Other']
    other = sum((gc.get(k, 0) or 0) for k in other_keys)

    cat_sum = pet + die + hp + hd + pp + pd2 + elec + other
    return cat_sum / total * 100


def load_motor_energy_eu27():
    """Load motor energy for EU-27: consistent country panel, normalised to 100 %.

    1. For each start year, find countries with >= 95 % category coverage
       in ALL years up to 2024.
    2. Pick the start year that maximises country count.
    3. Sum those countries and normalise to sum of 6 categories (= 100 %).
    """
    df = pd.read_csv(MOTOR_NRG_PATH)
    df_eu = df[df['geo'].isin(EU27)].copy()
    df_eu['OBS_VALUE'] = pd.to_numeric(df_eu['OBS_VALUE'], errors='coerce')

    all_years = sorted(df_eu['TIME_PERIOD'].unique())

    # Per country-year coverage
    cov = {}
    for y in all_years:
        for geo in EU27:
            cov[(geo, y)] = _motor_energy_country_coverage(df_eu, geo, y)

    # Find best start year (maximise countries with >= 95 % in ALL years)
    best_start, best_geos = all_years[0], set()
    for start in all_years:
        rng = [y for y in all_years if y >= start]
        good = {g for g in EU27 if all(cov.get((g, y), 0) >= 95 for y in rng)}
        if len(good) > len(best_geos) or (
                len(good) == len(best_geos) and start < best_start):
            best_start, best_geos = start, good

    selected = sorted(best_geos)
    use_years = [y for y in all_years if y >= best_start]
    print(f'Motor energy EU-27: using {len(selected)}/27 countries from '
          f'{best_start}-{use_years[-1]}')
    print(f'  Excluded: {sorted(set(EU27) - best_geos)}')

    # Sum selected countries per year
    result_rows = []
    for y in use_years:
        yr = df_eu[(df_eu['TIME_PERIOD'] == y) & (df_eu['geo'].isin(selected))]
        totals = {}
        for geo in selected:
            gc = yr[yr['geo'] == geo].set_index('mot_nrg')['OBS_VALUE']

            hp = gc.get('Hybrid electric-petrol', 0) or 0
            hd = gc.get('Hybrid diesel-electric', 0) or 0
            pp = gc.get('Plug-in hybrid petrol-electric \xa0', 0) or 0
            pd2 = gc.get('Plug-in hybrid diesel-electric \xa0', 0) or 0

            if 'Petrol (excluding hybrids) \xa0' in gc.index:
                pet = gc['Petrol (excluding hybrids) \xa0'] or 0
            elif 'Petrol' in gc.index:
                pet = (gc['Petrol'] or 0) - hp - pp
            else:
                pet = 0
            if 'Diesel (excluding hybrids) \xa0' in gc.index:
                die = gc['Diesel (excluding hybrids) \xa0'] or 0
            elif 'Diesel' in gc.index:
                die = (gc['Diesel'] or 0) - hd - pd2
            else:
                die = 0

            elec = gc.get('Electricity', 0) or 0
            other_keys = ['Bi-fuel', 'Biodiesel', 'Bioethanol', 'Natural gas',
                          'Hydrogen and fuel cells\xa0',
                          'Liquefied petroleum gases (LPG)', 'Other']
            other_alt = sum((gc.get(k, 0) or 0) for k in other_keys)

            for cat, val in [
                ('Petrol (excl. hybrids)', pet),
                ('Diesel (excl. hybrids)', die),
                ('Hybrids (regular)', hp + hd),
                ('Plug-in hybrids', pp + pd2),
                ('Electricity', elec),
                ('Other alternative', other_alt),
            ]:
                totals[cat] = totals.get(cat, 0) + (val if pd.notna(val) else 0)

        row = {'TIME_PERIOD': y}
        row.update(totals)
        result_rows.append(row)

    rdf = pd.DataFrame(result_rows).set_index('TIME_PERIOD').sort_index()
    # Normalise to sum of categories (= 100 %)
    row_sum = rdf[MOTOR_ENERGY_CATEGORIES].sum(axis=1)
    pct = rdf[MOTOR_ENERGY_CATEGORIES].div(row_sum, axis=0) * 100
    n_countries = len(selected)
    excluded = sorted(set(EU27) - best_geos)
    return pct, n_countries, excluded


def _build_weight_pct(df_filtered):
    """Build weight-category percentages from filtered DataFrame."""
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


def load_car_weight(geo_filter):
    """Load lighter-weight-category percentages for a single geo."""
    df = pd.read_csv(CAR_WEIGHT_PATH)
    return _build_weight_pct(df[df['geo'] == geo_filter].copy())


def load_car_weight_eu27():
    """Load weight-category percentages for EU-27.

    Only countries with weight data for ALL years 2013-2024 are included,
    so the panel is consistent over time.
    """
    df = pd.read_csv(CAR_WEIGHT_PATH)
    df_eu = df[df['geo'].isin(EU27)].copy()
    df_eu['OBS_VALUE'] = pd.to_numeric(df_eu['OBS_VALUE'], errors='coerce')

    detail_key = 'Less than 1 000 kg'
    target_years = list(range(2013, 2025))

    # Find countries with weight breakdown in every target year
    year_geos = {}
    for y in target_years:
        yr = df_eu[(df_eu['TIME_PERIOD'] == y) & (df_eu['weight'] == detail_key)]
        year_geos[y] = set(yr['geo'].unique())

    consistent = set(EU27)
    for geos in year_geos.values():
        consistent &= geos
    selected = sorted(consistent)

    print(f'Car weight EU-27: using {len(selected)}/27 countries for 2013-2024')
    print(f'  Excluded: {sorted(set(EU27) - consistent)}')

    # Sum selected countries per year
    df_sel = df_eu[(df_eu['geo'].isin(selected)) &
                   (df_eu['TIME_PERIOD'].isin(target_years))]
    rows = []
    for y in target_years:
        agg = df_sel[df_sel['TIME_PERIOD'] == y].groupby('weight')['OBS_VALUE'].sum().reset_index()
        agg['TIME_PERIOD'] = y
        rows.append(agg)

    agg_df = pd.concat(rows, ignore_index=True)
    pct = _build_weight_pct(agg_df)
    n_countries = len(selected)
    excluded = sorted(set(EU27) - consistent)
    return pct, n_countries, excluded


# ============================================================================
# MOTOR ENERGY CHART
# ============================================================================

def chart_motor_energy(data, region='EU-27', filename='eu27_motor_energy.png',
                       source_note='Source: Eurostat road_eqs_carpda',
                       coverage=None):
    """100 % stacked area chart of vehicle fleet by motor energy type."""
    cats = [c for c in MOTOR_ENERGY_CATEGORIES if c in data.columns]

    fig, ax = plt.subplots(figsize=(12, 8))
    years = data.index
    bottom = np.zeros(len(years))

    for cat in cats:
        vals = data[cat].values
        ax.fill_between(years, bottom, bottom + vals,
                        color=MOTOR_ENERGY_COLORS.get(cat, '#999'),
                        alpha=0.8, label=cat)
        bottom += vals

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Total Vehicle Fleet (%)', fontsize=12, fontweight='bold')
    yr_min, yr_max = int(years.min()), int(years.max())
    ax.set_title(f'{region} Vehicle Fleet by Motor Energy Type\n({yr_min}-{yr_max})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.set_xlim(yr_min, yr_max)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
              frameon=True, fancybox=True, shadow=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    note = source_note
    if isinstance(coverage, int):
        note += f'\n({coverage}/27 EU member states with consistent data)'
    elif isinstance(coverage, dict):
        cmin = min(coverage.values())
        cmax = max(coverage.values())
        note += f'\n({cmin}-{cmax}/27 EU member states reporting detailed breakdown)'
    fig.text(0.5, -0.01, note, ha='center', fontsize=8, color='#666')

    plt.tight_layout()
    out = os.path.join(OUTPUT_GRAPH, filename)
    out_svg = os.path.join(OUTPUT_GRAPH, filename.replace('.png', '.svg'))
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    print(f'Saved {out}')
    print(f'Saved {out_svg}')
    plt.close(fig)

    # Per-visual Excel
    excel_df = data[cats].copy()
    excel_df.index.name = 'Year'
    excel_out = os.path.join(OUTPUT_DATA, filename.replace('.png', '.xlsx'))
    excel_df.to_excel(excel_out, sheet_name='Motor energy %', engine='openpyxl')
    print(f'Saved {excel_out}')


# ============================================================================
# CAR WEIGHT LIGHT CATEGORIES CHART
# ============================================================================

def chart_car_weight_light(data, region='EU-27',
                           filename='eu27_car_weight_light.png',
                           source_note='Source: Eurostat road_eqs_unlweig',
                           coverage=None):
    """Line chart of weight categories as % of fleet."""
    labels = [lbl for _, lbl, _ in WEIGHT_CATS if lbl in data.columns]
    colors = {lbl: c for _, lbl, c in WEIGHT_CATS}

    # Drop years where all light categories are NaN
    data = data.dropna(subset=labels, how='any').copy()

    fig, ax = plt.subplots(figsize=(14, 8))
    for lbl in labels:
        ax.plot(data.index, data[lbl], color=colors[lbl],
                linewidth=3, marker='o', markersize=6, label=lbl)

    yr_min, yr_max = int(data.index.min()), int(data.index.max())
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Total Vehicle Fleet (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{region}: Evolution of Vehicle Weight Categories\n({yr_min}-{yr_max})',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(yr_min - 0.5, yr_max + 0.5)
    max_val = data[labels].max().max()
    ax.set_ylim(0, max_val * 1.15 if pd.notna(max_val) and max_val > 0 else 50)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.set_xticks(sorted(data.index))
    ax.set_xticklabels([str(int(y)) for y in sorted(data.index)], rotation=45)

    note = source_note
    if isinstance(coverage, int):
        note += f'\n({coverage}/27 EU member states with consistent data)'
    elif isinstance(coverage, dict):
        cmin = min(coverage.values())
        cmax = max(coverage.values())
        note += f'\n({cmin}-{cmax}/27 EU member states reporting weight breakdown)'
    fig.text(0.5, -0.01, note, ha='center', fontsize=8, color='#666')

    plt.tight_layout()
    out = os.path.join(OUTPUT_GRAPH, filename)
    out_svg = os.path.join(OUTPUT_GRAPH, filename.replace('.png', '.svg'))
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    print(f'Saved {out}')
    print(f'Saved {out_svg}')
    plt.close(fig)

    # Per-visual Excel
    excel_df = data[labels].copy()
    excel_df.index.name = 'Year'
    excel_out = os.path.join(OUTPUT_DATA, filename.replace('.png', '.xlsx'))
    excel_df.to_excel(excel_out, sheet_name='Weight %', engine='openpyxl')
    print(f'Saved {excel_out}')


# ============================================================================
# COVERAGE REPORT
# ============================================================================

def print_coverage_report(rail, road):
    """Print data availability analysis."""
    print("\n" + "=" * 60)
    print("DATA AVAILABILITY REPORT")
    print("=" * 60)

    # Rail
    rail_tot = rail[rail['vehicle'] == 'Total']
    print("\n--- Eurostat RAIL (Total, EU-27 countries) ---")
    for y in range(YEAR_MIN, YEAR_MAX + 1):
        present = sorted(rail_tot[rail_tot['year'] == y]['geo'].unique())
        missing = sorted(set(EU27) - set(present) - {'Malta', 'Cyprus'})
        print(f"  {y}: {len(present)}/25 countries"
              + (f"  missing: {', '.join(missing)}" if missing else ""))

    # Road
    road_tot = road[road['vehicle'] == 'Total']
    print("\n--- Eurostat ROAD (Total, EU-27 countries) ---")
    for y in range(YEAR_MIN, YEAR_MAX + 1):
        present = sorted(road_tot[road_tot['year'] == y]['geo'].unique())
        n = len(present)
        print(f"  {y}: {n}/27 countries")
    print("  => Road coverage too sparse for reliable EU-27 aggregate")

    print("\n--- Recommendation ---")
    print("  Using EEA EU-27 aggregate for all modes (consistent, 1995-2022)")
    print("=" * 60)


# ============================================================================
# CHARTS
# ============================================================================

def chart_pkm_absolute(eea, mode_config=None, region='EU-27',
                       filename='eu27_pkm_by_mode.png',
                       source_note='Source: EC Statistical Pocketbook 2024 — EU-27'):
    """Stacked area chart: passenger-km by mode (Gpkm)."""
    if mode_config is None:
        mode_config = MODE_CONFIG
    data = eea.copy()
    modes = [label for label, _, _ in mode_config if label in data.columns]
    colors = {label: c for label, _, c in mode_config}
    yr_min, yr_max = int(data.index.min()), int(data.index.max())

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.stackplot(data.index, *[data[m].fillna(0) for m in modes],
                 labels=modes,
                 colors=[colors[m] for m in modes],
                 alpha=0.85)

    ax.set_title(f'{region} Passenger Transport Activity by Mode',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Billion passenger-km (Gpkm)', fontsize=11)
    ax.set_xlim(yr_min, yr_max)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    # Total line
    total = data[modes].sum(axis=1)
    ax.plot(data.index, total, color='black', linewidth=1.5,
            linestyle='--', label='_nolegend_')
    for y in [yr_min, yr_max, 2020]:
        if y in total.index:
            ax.annotate(f'{total[y]:,.0f}', xy=(y, total[y]),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=8, fontweight='bold')

    fig.text(0.5, -0.01, source_note,
             ha='center', fontsize=8, color='#666')

    plt.tight_layout()
    out = os.path.join(OUTPUT_GRAPH, filename)
    out_svg = os.path.join(OUTPUT_GRAPH, filename.replace('.png', '.svg'))
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    print(f"Saved {out}")
    print(f"Saved {out_svg}")
    plt.close(fig)

    # Per-visual Excel
    excel_df = data[modes].copy()
    excel_df.index.name = 'Year'
    excel_out = os.path.join(OUTPUT_DATA, filename.replace('.png', '.xlsx'))
    excel_df.to_excel(excel_out, sheet_name='Gpkm', engine='openpyxl')
    print(f"Saved {excel_out}")


def chart_pkm_per_capita(eea, pop, mode_config=None, region='EU-27',
                         filename='eu27_pkm_per_capita_by_mode.png',
                         source_note=('Sources: EC Statistical Pocketbook 2024 (transport) · '
                                      'Eurostat DEMO_PJAN (population) — EU-27')):
    """Stacked area chart: pkm per capita by mode."""
    if mode_config is None:
        mode_config = MODE_CONFIG
    data = eea.copy()
    modes = [label for label, _, _ in mode_config if label in data.columns]
    colors = {label: c for label, _, c in mode_config}

    # Convert Gpkm to pkm/capita:  Gpkm * 1e9 / population
    pop_aligned = pop.reindex(data.index)
    # Restrict to years with population data
    valid = pop_aligned.dropna().index
    data = data.loc[data.index.isin(valid)].copy()
    pop_aligned = pop_aligned.loc[data.index]
    for m in modes:
        data[m] = data[m].fillna(0) * 1e9 / pop_aligned

    yr_min, yr_max = int(data.index.min()), int(data.index.max())

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.stackplot(data.index, *[data[m] for m in modes],
                 labels=modes,
                 colors=[colors[m] for m in modes],
                 alpha=0.85)

    ax.set_title(f'{region} Passenger Transport Activity per Capita by Mode',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Passenger-km per capita', fontsize=11)
    ax.set_xlim(yr_min, yr_max)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v / 1e3:.1f}k' if v >= 1e3 else f'{v:.0f}'))

    # Total line
    total = data[modes].sum(axis=1)
    ax.plot(data.index, total, color='black', linewidth=1.5,
            linestyle='--', label='_nolegend_')
    for y in [yr_min, yr_max, 2020]:
        if y in total.index:
            ax.annotate(f'{total[y] / 1e3:.1f}k', xy=(y, total[y]),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=8, fontweight='bold')

    fig.text(0.5, -0.01, source_note,
             ha='center', fontsize=8, color='#666')

    plt.tight_layout()
    out = os.path.join(OUTPUT_GRAPH, filename)
    out_svg = os.path.join(OUTPUT_GRAPH, filename.replace('.png', '.svg'))
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    print(f"Saved {out}")
    print(f"Saved {out_svg}")
    plt.close(fig)

    # Per-visual Excel
    excel_df = data[modes].copy()
    excel_df.index.name = 'Year'
    excel_out = os.path.join(OUTPUT_DATA, filename.replace('.png', '.xlsx'))
    excel_df.to_excel(excel_out, sheet_name='pkm per capita', engine='openpyxl')
    print(f"Saved {excel_out}")


def export_excel(eea, pop):
    """Export data underlying both charts to Excel."""
    data = eea.copy()
    modes = [label for label, _, _ in MODE_CONFIG if label in data.columns]
    pop_aligned = pop.reindex(data.index)

    # Absolute sheet
    abs_df = data[modes].copy()
    abs_df.insert(0, 'Total', abs_df.sum(axis=1))
    abs_df.columns = pd.MultiIndex.from_product(
        [['Gpkm'], abs_df.columns])

    # Per-capita sheet
    pc_df = data[modes].copy()
    for m in modes:
        pc_df[m] = pc_df[m].fillna(0) * 1e9 / pop_aligned
    pc_df.insert(0, 'Total', pc_df.sum(axis=1))
    pc_df.columns = pd.MultiIndex.from_product(
        [['pkm per capita'], pc_df.columns])

    out = os.path.join(OUTPUT_DATA, 'eu27_passenger_transport.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        abs_df.to_excel(w, sheet_name='Gpkm')
        pc_df.to_excel(w, sheet_name='pkm per capita')
        # Population
        pop_df = pop_aligned.to_frame('EU-27 population')
        pop_df.to_excel(w, sheet_name='Population')
    print(f"Saved {out}")


def chart_cluster_pkm_per_capita_by_mode(filename_base='cluster_pkm_per_capita_by_mode_8countries'):
    """Single 4x2 visual: per-capita pkm by mode for 8 countries arranged by clusters."""
    mode_order = [m[0] for m in FR_MODE_CONFIG]
    mode_colors = {m[0]: m[2] for m in FR_MODE_CONFIG}

    country_series = {}
    ymax = 0.0

    for _, countries in CLUSTER_LAYOUT:
        for cc in countries:
            name = COUNTRY_NAMES[cc]
            gpkm = _load_country_mode_gpkm_from_eea_csv(cc)
            pop = load_population_country(name).reindex(gpkm.index)

            data = gpkm.copy()
            valid = pop.dropna().index
            data = data.loc[data.index.isin(valid)].copy()
            pop = pop.loc[data.index]
            for mode in mode_order:
                data[mode] = data[mode].fillna(0) * 1e9 / pop

            country_series[cc] = data
            ymax = max(ymax, data[mode_order].sum(axis=1).max())

    if ymax <= 0 or pd.isna(ymax):
        ymax = 1
    ymax *= 1.10

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=False, sharey=True)
    fig.suptitle('Passenger Transport Activity per Capita by Mode (8 Countries, Clustered)',
                 fontsize=16, fontweight='bold', y=0.995)

    r = 0
    for cluster_name, countries in CLUSTER_LAYOUT:
        for c, cc in enumerate(countries):
            ax = axes[r, c]
            data = country_series.get(cc)
            cname = COUNTRY_NAMES[cc]

            if data is None or data.empty:
                ax.text(0.5, 0.5, f'{cname}\n(no data)', ha='center', va='center', transform=ax.transAxes)
                ax.set_ylim(0, ymax)
            else:
                ax.stackplot(
                    data.index,
                    *[data[m].fillna(0) for m in mode_order],
                    labels=mode_order,
                    colors=[mode_colors[m] for m in mode_order],
                    alpha=0.85,
                )
                total = data[mode_order].sum(axis=1)
                ax.plot(data.index, total, color='black', linewidth=1.2, linestyle='--', label='_nolegend_')
                ax.set_ylim(0, ymax)
                ax.grid(axis='y', alpha=0.25)

            ax.set_title(f'{cname} ({cc}) — {cluster_name}',
                         fontsize=11, fontweight='bold', color=COUNTRY_CLUSTER_COLOR[cc])
            ax.set_xlabel('Year', fontsize=9)
            ax.set_ylabel('pkm per capita', fontsize=9)
        r += 1

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=9, title='Mode', title_fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.9, 0.98])

    out_png = os.path.join(OUTPUT_GRAPH, f'{filename_base}.png')
    out_svg = os.path.join(OUTPUT_GRAPH, f'{filename_base}.svg')
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_png}')
    print(f'Saved {out_svg}')

    out_xlsx = os.path.join(OUTPUT_DATA, f'{filename_base}.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        for cc in COUNTRY_NAMES:
            data = country_series.get(cc)
            if data is None or data.empty:
                continue
            sheet = data[mode_order].copy()
            sheet.insert(0, 'Total', sheet.sum(axis=1))
            sheet.index.name = 'Year'
            sheet.to_excel(writer, sheet_name=f'{COUNTRY_NAMES[cc]} ({cc})')
    print(f'Saved {out_xlsx}')


def chart_cluster_motor_energy(filename_base='cluster_motor_energy_8countries'):
    """Single 4x2 visual: motor energy composition (%) for 8 countries."""
    country_series = {}
    for cc, cname in COUNTRY_NAMES.items():
        country_series[cc] = load_motor_energy(cname)

    # -- Country-specific data corrections --
    # Drop years where data is incomplete (row sum far below 100%)
    # and renormalize years that are close but don't sum exactly to 100%
    for cc in list(country_series.keys()):
        data = country_series[cc]
        if data is None or data.empty:
            continue
        row_sums = data[MOTOR_ENERGY_CATEGORIES].sum(axis=1)
        # Drop rows summing to less than 95%
        keep = row_sums >= 95
        data = data.loc[keep].copy()
        # Renormalize remaining rows to exactly 100%
        row_sums = data[MOTOR_ENERGY_CATEGORIES].sum(axis=1)
        data[MOTOR_ENERGY_CATEGORIES] = data[MOTOR_ENERGY_CATEGORIES].div(row_sums, axis=0) * 100
        country_series[cc] = data

    # Compute shared x-axis range across all countries
    all_years = [d.index for d in country_series.values() if d is not None and not d.empty]
    x_min = min(idx.min() for idx in all_years)
    x_max = max(idx.max() for idx in all_years)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True, sharey=True)
    fig.suptitle('Vehicle Fleet by Motor Energy Type (8 Countries, Clustered)',
                 fontsize=16, fontweight='bold', y=0.995)

    r = 0
    for cluster_name, countries in CLUSTER_LAYOUT:
        for c, cc in enumerate(countries):
            ax = axes[r, c]
            data = country_series.get(cc)
            cname = COUNTRY_NAMES[cc]

            if data is None or data.empty:
                ax.text(0.5, 0.5, f'{cname}\n(no data)', ha='center', va='center', transform=ax.transAxes)
                ax.set_ylim(0, 100)
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
                    bottom = bottom + vals
                ax.set_ylim(0, 100)
                ax.grid(axis='y', alpha=0.25)

            ax.set_title(f'{cname} ({cc}) — {cluster_name}',
                         fontsize=11, fontweight='bold', color=COUNTRY_CLUSTER_COLOR[cc])
            ax.set_xlabel('Year', fontsize=9)
            ax.set_ylabel('Fleet share (%)', fontsize=9)
        r += 1

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=9, title='Motor energy', title_fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.9, 0.98])

    out_png = os.path.join(OUTPUT_GRAPH, f'{filename_base}.png')
    out_svg = os.path.join(OUTPUT_GRAPH, f'{filename_base}.svg')
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_png}')
    print(f'Saved {out_svg}')

    out_xlsx = os.path.join(OUTPUT_DATA, f'{filename_base}.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        for cc, cname in COUNTRY_NAMES.items():
            data = country_series.get(cc)
            if data is None or data.empty:
                continue
            sheet = data[MOTOR_ENERGY_CATEGORIES].copy()
            sheet.index.name = 'Year'
            sheet.to_excel(writer, sheet_name=f'{cname} ({cc})')
    print(f'Saved {out_xlsx}')


def chart_cluster_car_weight_light(filename_base='cluster_car_weight_light_8countries'):
    """Single 4x2 visual: light weight-category shares (%) for 8 countries."""
    labels = [lbl for _, lbl, _ in WEIGHT_CATS]
    colors = {lbl: c for _, lbl, c in WEIGHT_CATS}

    country_series = {}
    global_max = 0.0
    for cc, cname in COUNTRY_NAMES.items():
        data = load_car_weight(cname)
        # In this dataset, zeros represent missing values for this view.
        data = data.replace(0, np.nan)
        data = data.dropna(subset=[l for l in labels if l in data.columns], how='all')
        country_series[cc] = data
        if not data.empty:
            global_max = max(global_max, data[[l for l in labels if l in data.columns]].max().max())

    # Replace countries with insufficient data by same-cluster fallback.
    display_names = COUNTRY_NAMES.copy()
    replacements = {
        'BE': ['LU', 'IE', 'DK', 'SE'],
        'LT': ['LV', 'EE', 'HU', 'RO'],
    }
    for orig_cc, fallback_codes in replacements.items():
        if orig_cc in country_series and (country_series[orig_cc].empty or len(country_series[orig_cc]) <= 3):
            for repl_code in fallback_codes:
                repl_name = FALLBACK_COUNTRY_NAMES[repl_code]
                repl_data = load_car_weight(repl_name).replace(0, np.nan)
                repl_data = repl_data.dropna(subset=[l for l in labels if l in repl_data.columns], how='all')
                if not repl_data.empty and len(repl_data) > 3:
                    country_series[orig_cc] = repl_data
                    display_names[orig_cc] = repl_name
                    global_max = max(global_max, repl_data[[l for l in labels if l in repl_data.columns]].max().max())
                    print(f"Car weight clustered: {COUNTRY_NAMES[orig_cc]} replaced by {repl_name} ({repl_code})")
                    break

    if global_max <= 0 or pd.isna(global_max):
        global_max = 1
    global_max *= 1.10

    # Compute shared x-axis range across all countries
    all_years = [d.index for d in country_series.values() if d is not None and not d.empty]
    x_min = min(idx.min() for idx in all_years)
    x_max = max(idx.max() for idx in all_years)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True, sharey=True)
    fig.suptitle('Vehicle Weight Categories (8 Countries, Clustered)',
                 fontsize=16, fontweight='bold', y=0.995)

    r = 0
    for cluster_name, countries in CLUSTER_LAYOUT:
        for c, cc in enumerate(countries):
            ax = axes[r, c]
            data = country_series.get(cc)
            cname = display_names[cc]

            if data is None or data.empty:
                ax.text(0.5, 0.5, f'{cname}\n(no data)', ha='center', va='center', transform=ax.transAxes)
                ax.set_ylim(0, global_max)
            else:
                for lbl in labels:
                    if lbl in data.columns:
                        ax.plot(data.index, data[lbl], color=colors[lbl],
                                linewidth=2.0, marker='o', markersize=4, label=lbl)
                ax.set_ylim(0, global_max)
                ax.grid(axis='y', alpha=0.25)

            ax.set_title(f'{cname} ({cc}) — {cluster_name}',
                         fontsize=11, fontweight='bold', color=COUNTRY_CLUSTER_COLOR[cc])
            ax.set_xlabel('Year', fontsize=9)
            ax.set_ylabel('Fleet share (%)', fontsize=9)
        r += 1

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=9, title='Weight class', title_fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.9, 0.98])

    out_png = os.path.join(OUTPUT_GRAPH, f'{filename_base}.png')
    out_svg = os.path.join(OUTPUT_GRAPH, f'{filename_base}.svg')
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_png}')
    print(f'Saved {out_svg}')

    out_xlsx = os.path.join(OUTPUT_DATA, f'{filename_base}.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        for cc, cname in display_names.items():
            data = country_series.get(cc)
            if data is None or data.empty:
                continue
            sheet = data[[l for l in labels if l in data.columns]].copy()
            sheet.index.name = 'Year'
            sheet.to_excel(writer, sheet_name=f'{cname} ({cc})')
    print(f'Saved {out_xlsx}')


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 50)
    print("EU-27 Passenger Transport Analysis")
    print("=" * 50)

    # Load data
    eea = load_eea()
    pop = load_population()

    print(f"Pocketbook EU-27: {len(eea)} years ({eea.index.min()}-{eea.index.max()}), "
          f"{len(eea.columns)} modes")
    print(f"Population: {len(pop)} years")

    # EU-27 Charts
    print("\nGenerating EU-27 charts...")
    chart_pkm_absolute(eea)
    chart_pkm_per_capita(eea, pop)

    # Clustered country visual for pkm per capita by mode (8 countries)
    chart_cluster_pkm_per_capita_by_mode()

    # Excel
    export_excel(eea, pop)

    # ── France ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("France Passenger Transport Analysis")
    print("=" * 50)

    eea_fr = load_eea_france()
    pop_fr = load_population_france()
    print(f"Pocketbook FR: {len(eea_fr)} years ({eea_fr.index.min()}-{eea_fr.index.max()}), "
          f"{len(eea_fr.columns)} modes")
    print(f"Population FR: {len(pop_fr)} years")

    print("\nGenerating France charts...")
    chart_pkm_absolute(
        eea_fr, mode_config=FR_MODE_CONFIG, region='France',
        filename='france_pkm_by_mode.png',
        source_note='Source: EC Statistical Pocketbook 2024 — France')
    chart_pkm_per_capita(
        eea_fr, pop_fr, mode_config=FR_MODE_CONFIG, region='France',
        filename='france_pkm_per_capita_by_mode.png',
        source_note=('Sources: EC Statistical Pocketbook 2024 (transport) · '
                     'Eurostat DEMO_PJAN (population) — France'))

    # ── Motor Energy ────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Vehicle Fleet by Motor Energy Type")
    print("=" * 50)

    mnrg_eu, mnrg_n, mnrg_excl = load_motor_energy_eu27()
    print(f"Motor energy EU-27: {len(mnrg_eu)} years "
          f"({mnrg_eu.index.min()}-{mnrg_eu.index.max()}), "
          f"{mnrg_n} countries")
    chart_motor_energy(mnrg_eu, region='EU-27',
                       filename='eu27_motor_energy.png',
                       source_note='Source: Eurostat road_eqs_carpda — EU-27 (sum of reporting countries)',
                       coverage=mnrg_n)

    mnrg_fr = load_motor_energy('France')
    print(f"Motor energy France: {len(mnrg_fr)} years "
          f"({mnrg_fr.index.min()}-{mnrg_fr.index.max()})")
    chart_motor_energy(mnrg_fr, region='France',
                       filename='france_motor_energy.png',
                       source_note='Source: Eurostat road_eqs_carpda — France')

    # Clustered country visual for motor energy (8 countries)
    chart_cluster_motor_energy()

    # ── Car Weight (lighter categories) ─────────────────────
    print("\n" + "=" * 50)
    print("Lighter Vehicle Weight Categories")
    print("=" * 50)

    wt_eu, wt_n, wt_excl = load_car_weight_eu27()
    print(f"Car weight EU-27: {len(wt_eu)} years "
          f"({wt_eu.index.min()}-{wt_eu.index.max()}), "
          f"{wt_n} countries")
    chart_car_weight_light(wt_eu, region='EU-27',
                           filename='eu27_car_weight_light.png',
                           source_note='Source: Eurostat road_eqs_unlweig — EU-27 (sum of reporting countries)',
                           coverage=wt_n)

    wt_fr = load_car_weight('France')
    print(f"Car weight France: {len(wt_fr)} years "
          f"({wt_fr.index.min()}-{wt_fr.index.max()})")
    chart_car_weight_light(wt_fr, region='France',
                           filename='france_car_weight_light.png',
                           source_note='Source: Eurostat road_eqs_unlweig — France')

    # Clustered country visual for car weight categories (8 countries)
    chart_cluster_car_weight_light()

    print("\nDone!")


if __name__ == "__main__":
    main()
