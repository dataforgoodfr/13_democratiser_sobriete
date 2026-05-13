"""
Eurostat Trade Dependency Visualization Script
Generates visualizations showing EU-27 import dependency on major exporters
for energy (mineral fuels) and raw materials.

Analyses:
1. Aggregate product analysis: Crude materials, Mineral fuels, Oils & fats
   (in value EUR and volume KG)
2. Mineral fuels disaggregated into sub-components (coal, petroleum, gas, electric)
3. Additional: HHI concentration index over time, top-partner evolution area chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# Set font
plt.rcParams['font.family'] = 'Arial'

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'external_data', 'eurostat_trade.csv')
DATA_ALL_YEAR_PATH = os.path.join(BASE_DIR, 'external_data', 'eurostat_trade_all_year.csv')
GDP_PATH = os.path.join(BASE_DIR, 'external_data', 'eurostat_gdp_current_price.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'graphs', 'EUROSTAT_trade')
EXCEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'data')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXCEL_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CONSTANTS
# ============================================================================
TARGET_YEARS = [2002, 2005, 2015, 2025]
N_PARTNERS = 12  # Top partners to show

# Aggregate product categories of interest
AGGREGATE_PRODUCTS = [
    'Crude materials, inedible, except fuels',
    'Mineral fuels, lubricants and related materials',
    'Animal and vegetable oils, fats and waxes',
]

# Sub-components of "Mineral fuels, lubricants and related materials"
MINERAL_FUEL_SUBPRODUCTS = [
    'Coal, coke and briquettes',
    'Petroleum, petroleum products and related materials',
    'Gas, natural and manufactured',
    'Electric current',
]

# Short labels for products
PRODUCT_SHORT = {
    'Crude materials, inedible, except fuels': 'Raw Materials',
    'Mineral fuels, lubricants and related materials': 'Mineral Fuels',
    'Animal and vegetable oils, fats and waxes': 'Oils & Fats',
    'Coal, coke and briquettes': 'Coal & Coke',
    'Petroleum, petroleum products and related materials': 'Petroleum',
    'Gas, natural and manufactured': 'Gas',
    'Electric current': 'Electricity',
}

# Partners to exclude (aggregates, not individual countries)
AGGREGATE_PARTNERS = [
    'All countries of the world',
    'Extra-EU27 (from 2020)',
    'Extra-EU',
    'Extra-euro area',
    'Extra-euro area - 21 countries (from 2026)',
    'Intra-EU27 (from 2020)',
    'Intra-EU',
    'Intra-euro area',
    'Intra-euro area - 21 countries (from 2026)',
    'Countries and territories not specified',
    'Countries and territories not specified for commercial or military reasons in the framework of extra-Union trade',
    'Countries and territories not specified for commercial or military reasons in the framework of intra-Union trade',
    'Countries and territories not specified within the framework of extra-Union trade',
    'Countries and territories not specified within the framework of intra-Union trade',
    'Stores and provisions',
    'Stores and provisions within the framework of extra-Union trade',
    'Stores and provisions within the framework of intra-Union trade',
    'High seas',
    'United States Minor Outlying Islands',
]

# The aggregate partner row used as denominator for % of extra-EU imports
EXTRA_EU_PARTNER = 'Extra-EU27 (from 2020)'

# France reporter label and total-partner for France-level analysis
FRANCE_REPORTER = "France (incl. Saint Barthélemy 'BL' -> 2012; incl. French Guiana 'GF', Guadeloupe 'GP', Martinique 'MQ', Réunion 'RE' from 1997; incl. Mayotte 'YT' from 2014)"
FRANCE_TOTAL_PARTNER = 'All countries of the world'
FRANCE_SUBPRODUCTS = [
    'Coal, coke and briquettes',
    'Petroleum, petroleum products and related materials',
    'Gas, natural and manufactured',
]

# New per-country imports data (all reporters, VALUE_EUR only, 3 products)
NEW_IMPORTS_PATH = os.path.join(
    BASE_DIR, 'external_data',
    'eurostat_imports_ds-059331__custom_20825822_linear.csv')

# EWBI cluster countries (code → short label used to find reporter)
CLUSTER_COUNTRIES = {
    'FR': 'France', 'ES': 'Spain', 'BE': 'Belgium', 'NL': 'Netherlands',
    'LT': 'Lithuania', 'HU': 'Hungary', 'DE': 'Germany', 'PL': 'Poland',
}

# 20 distinct colours for the unified partner colour map
CLUSTER_TOP10_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#66c2a5', '#fc8d62', '#8da0cb',
    '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#a6d854',
    '#ffd92f', '#e5c494', '#b3b3b3', '#8c564b', '#17becf',
]

# EU-27 country names (to identify intra-EU trade, which we exclude for dependency)
EU27_PARTNER_FRAGMENTS = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
    'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
    'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia',
    'Slovenia', 'Spain', 'Sweden',
]

# Country code mapping
COUNTRY_CODES = {
    'Russian Federation (Russia)': 'RU',
    'Norway (incl. Svalbard and Jan Mayen \'SJ\' -> 1994 and again from 1997)': 'NO',
    'United States (incl. Navassa Island (part of \'UM\') from 1995 -> 2000)': 'US',
    'United Kingdom': 'GB',
    'China': 'CN',
    'Saudi Arabia': 'SA',
    'Algeria': 'DZ',
    'Nigeria': 'NG',
    'Libya': 'LY',
    'Kazakhstan': 'KZ',
    'Iraq': 'IQ',
    'India': 'IN',
    'Brazil': 'BR',
    'South Africa (incl. Namibia \'NA\' -> 1989)': 'ZA',
    'Canada': 'CA',
    'Australia': 'AU',
    'Indonesia (incl. East Timor \'TP\' from 1977 -> 2000)': 'ID',
    'Malaysia': 'MY',
    'Qatar': 'QA',
    'United Arab Emirates': 'AE',
    'Kuwait': 'KW',
    'Iran, Islamic Republic of': 'IR',
    'Azerbaijan': 'AZ',
    'Colombia': 'CO',
    'Egypt': 'EG',
    'Türkiye': 'TR',
    'Switzerland (incl. Liechtenstein \'LI\' -> 1994)': 'CH',
    'Mexico': 'MX',
    'Japan': 'JP',
    'Korea, Republic of (South Korea)': 'KR',
    'Thailand': 'TH',
    'Viet Nam (incl. North Viet Nam \'VD\' from 1977)': 'VN',
    'Argentina': 'AR',
    'Chile': 'CL',
    'Ukraine': 'UA',
    'Belarus (Belorussia)': 'BY',
    'Trinidad and Tobago': 'TT',
    'Mozambique': 'MZ',
    'Côte d\'Ivoire (Ivory Coast)': 'CI',
    'Cameroon': 'CM',
    'Peru': 'PE',
    'Philippines': 'PH',
    'Singapore': 'SG',
    'Taiwan': 'TW',
    'Hong Kong': 'HK',
    'Angola': 'AO',
    'Oman': 'OM',
    'Turkmenistan': 'TM',
    'Mozambique': 'MZ',
    'Papua New Guinea': 'PG',
    'Congo': 'CG',
    'Equatorial Guinea': 'GQ',
    'Gabon': 'GA',
    'Venezuela, Bolivarian Republic of': 'VE',
    'Bahrain': 'BH',
    'Morocco': 'MA',
    'Tunisia': 'TN',
    'Georgia': 'GE',
    'Serbia': 'RS',
    'Bosnia and Herzegovina': 'BA',
    'North Macedonia': 'MK',
    'Albania': 'AL',
    'Moldova, Republic of': 'MD',
    'Montenegro': 'ME',
    'Iceland': 'IS',
    'Liechtenstein': 'LI',
    'New Zealand': 'NZ',
    'Uruguay': 'UY',
    'Paraguay': 'PY',
    'Bolivia, Plurinational State of': 'BO',
    'Ecuador': 'EC',
    'Guyana': 'GY',
    'Suriname': 'SR',
}

# Color palettes
PARTNER_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62',
    '#8da0cb', '#e78ac3',
]
PRODUCT_COLORS = {
    'Raw Materials': '#4daf4a',
    'Mineral Fuels': '#377eb8',
    'Oils & Fats': '#ff7f00',
    'Coal & Coke': '#333333',
    'Petroleum': '#984ea3',
    'Gas': '#e41a1c',
    'Electricity': '#ffff33',
}
OTHER_COLOR = '#cccccc'


def get_country_code(name):
    """Get short country code for axis labels."""
    if name in COUNTRY_CODES:
        return COUNTRY_CODES[name]
    if name == 'Other':
        return 'Other'
    return name[:3].upper()


def is_eu27_partner(partner_name):
    """Check if a partner is an EU-27 member (intra-EU trade)."""
    for fragment in EU27_PARTNER_FRAGMENTS:
        if fragment in partner_name:
            return True
    return False


# ============================================================================
# DATA LOADING
# ============================================================================

def load_trade_data():
    """Load and preprocess the Eurostat trade CSV."""
    print("Loading Eurostat trade data...")
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={'TIME_PERIOD': 'year', 'OBS_VALUE': 'value'})
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    print(f"  Loaded {len(df):,} records, years: {sorted(df['year'].unique())}")
    return df


EU27_REPORTER = 'European Union - 27 countries (AT, BE, BG, CY, CZ, DE, DK, EE, ES, FI, FR, GR, HR, HU, IE, IT, LT, LU, LV, MT, NL, PL, PT, RO, SE, SI, SK)'


def load_trade_data_all_years(reporter=None):
    """Load the all-year Eurostat trade CSV. Filter by reporter if specified."""
    print("Loading all-year Eurostat trade data...")
    df = pd.read_csv(DATA_ALL_YEAR_PATH)
    df = df.rename(columns={'TIME_PERIOD': 'year', 'OBS_VALUE': 'value'})
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    # Filter by reporter (default: EU-27)
    target = reporter if reporter is not None else EU27_REPORTER
    df = df[df['reporter'] == target].copy()
    print(f"  Loaded {len(df):,} records, years: {sorted(df['year'].unique())}")
    return df


def load_gdp_data():
    """Load EU-27 GDP data (million EUR -> EUR)."""
    print("Loading GDP data...")
    gdp_df = pd.read_csv(GDP_PATH)
    eu_label = 'European Union - 27 countries (from 2020)'
    gdp_eu = gdp_df[gdp_df['geo'] == eu_label].copy()
    gdp_eu['gdp_eur'] = gdp_eu['OBS_VALUE'] * 1_000_000
    return gdp_eu[['TIME_PERIOD', 'gdp_eur']].rename(columns={'TIME_PERIOD': 'year'})


def filter_extra_eu_imports(df, products, years=TARGET_YEARS,
                           exclude_eu_members=True):
    """
    Filter for IMPORT flow, specific products & years,
    excluding aggregate partners (and optionally EU-27 members).
    """
    mask = (
        (df['flow'] == 'IMPORT') &
        (df['product'].isin(products)) &
        (df['year'].isin(years)) &
        (~df['partner'].isin(AGGREGATE_PARTNERS))
    )
    if exclude_eu_members:
        mask = mask & (~df['partner'].apply(is_eu27_partner))
    return df[mask].copy()


def get_extra_eu_total(df, products, years=TARGET_YEARS,
                      total_partner=EXTRA_EU_PARTNER):
    """
    Get total imports using an aggregate partner row as denominator.
    """
    totals = {}
    for indicator in ['VALUE_EUR', 'QUANTITY_KG']:
        for year in years:
            for product in products:
                # Try aggregate row first
                agg = df[
                    (df['flow'] == 'IMPORT') &
                    (df['partner'] == total_partner) &
                    (df['product'] == product) &
                    (df['year'] == year) &
                    (df['indicators'] == indicator)
                ]['value'].sum()

                if agg > 0:
                    totals[(indicator, year, product)] = agg
                else:
                    # Fallback: sum non-EU individual partners
                    indiv = df[
                        (df['flow'] == 'IMPORT') &
                        (df['product'] == product) &
                        (df['year'] == year) &
                        (df['indicators'] == indicator) &
                        (~df['partner'].isin(AGGREGATE_PARTNERS)) &
                        (~df['partner'].apply(is_eu27_partner))
                    ]['value'].sum()
                    totals[(indicator, year, product)] = indiv
    return totals


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_top_partners(df_imports, indicator, product, year, n=N_PARTNERS):
    """
    Identify top N partners by import value for a given product, year, indicator.
    Returns DataFrame with partner, value, and share columns.
    """
    sub = df_imports[
        (df_imports['indicators'] == indicator) &
        (df_imports['product'] == product) &
        (df_imports['year'] == year)
    ].groupby('partner')['value'].sum().reset_index()

    sub = sub.sort_values('value', ascending=False)
    total = sub['value'].sum()
    sub['share'] = sub['value'] / total * 100 if total > 0 else 0
    return sub.head(n), total


def compute_hhi(df_imports, indicator, product, year):
    """
    Compute Herfindahl-Hirschman Index for import concentration.
    HHI = sum(share_i^2) where share_i is in [0,1].
    Returns HHI in [0, 10000] scale.
    """
    sub = df_imports[
        (df_imports['indicators'] == indicator) &
        (df_imports['product'] == product) &
        (df_imports['year'] == year)
    ].groupby('partner')['value'].sum()

    total = sub.sum()
    if total == 0:
        return 0
    shares = sub / total
    return (shares ** 2).sum() * 10000


# ============================================================================
# CHART 1: Stacked bar - Top exporters per product, per year (% of extra-EU imports)
# ============================================================================

def chart_stacked_top_partners_pct(df_imports, totals_dict, products, indicator,
                                   unit_label, filename_suffix):
    """
    For each product: one figure with subplots per year.
    Stacked bars = top N partners, share of total extra-EU imports.
    """
    years = TARGET_YEARS
    for product in products:
        short = PRODUCT_SHORT.get(product, product)
        fig, axes = plt.subplots(1, len(years), figsize=(5 * len(years), 7),
                                 sharey=True)
        if len(years) == 1:
            axes = [axes]

        # Collect union of top partners across years to keep consistent colors
        all_top = set()
        for year in years:
            top, _ = compute_top_partners(df_imports, indicator, product, year)
            all_top.update(top['partner'].tolist())

        # Assign colors
        partner_list = sorted(all_top)
        color_map = {p: PARTNER_COLORS[i % len(PARTNER_COLORS)]
                     for i, p in enumerate(partner_list)}
        color_map['Other'] = OTHER_COLOR

        for ax, year in zip(axes, years):
            top, individual_total = compute_top_partners(
                df_imports, indicator, product, year)
            extra_eu_total = totals_dict.get((indicator, year, product), individual_total)

            top = top.copy()
            top['pct'] = top['value'] / extra_eu_total * 100 if extra_eu_total > 0 else 0
            other_pct = max(100 - top['pct'].sum(), 0)

            # Stack bars
            bottom = 0
            for _, row in top.iterrows():
                code = get_country_code(row['partner'])
                ax.bar(0, row['pct'], bottom=bottom, width=0.6,
                       color=color_map.get(row['partner'], OTHER_COLOR),
                       edgecolor='white', linewidth=0.5)
                if row['pct'] > 2:
                    ax.text(0, bottom + row['pct'] / 2, f"{code}\n{row['pct']:.1f}%",
                            ha='center', va='center', fontsize=7, fontweight='bold')
                bottom += row['pct']

            if other_pct > 0.5:
                ax.bar(0, other_pct, bottom=bottom, width=0.6,
                       color=OTHER_COLOR, edgecolor='white', linewidth=0.5)
                ax.text(0, bottom + other_pct / 2, f"Other\n{other_pct:.1f}%",
                        ha='center', va='center', fontsize=7, color='#555555')

            ax.set_title(str(year), fontsize=13, fontweight='bold')
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0, 105)
            ax.set_xticks([])
            if ax == axes[0]:
                ax.set_ylabel(f'Share of extra-EU imports ({unit_label})', fontsize=11)

        fig.suptitle(f'EU-27 Import Dependency: {short}\n'
                     f'Top {N_PARTNERS} extra-EU exporters ({unit_label})',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fname = f"dependency_{short.replace(' ', '_').replace('&', 'and')}_{filename_suffix}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
        print(f"  Saved {fname}")
        plt.close(fig)


# ============================================================================
# CHART 2: Grouped bar - Top partners, absolute values across years
# ============================================================================

def chart_grouped_bars_absolute(df_imports, products, indicator, unit_label,
                                filename_suffix):
    """
    For each product: grouped bar chart, x=partner, groups=years.
    Shows absolute import values.
    """
    years = TARGET_YEARS
    for product in products:
        short = PRODUCT_SHORT.get(product, product)

        # Get union of top partners across all years
        all_partners = {}
        for year in years:
            top, _ = compute_top_partners(df_imports, indicator, product, year, n=N_PARTNERS)
            for _, row in top.iterrows():
                all_partners[row['partner']] = all_partners.get(row['partner'], 0) + row['value']

        # Sort by total and pick top N
        sorted_partners = sorted(all_partners, key=all_partners.get, reverse=True)[:N_PARTNERS]

        n_groups = len(sorted_partners)
        n_bars = len(years)
        x = np.arange(n_groups)
        width = 0.8 / n_bars

        fig, ax = plt.subplots(figsize=(14, 7))
        year_colors = ['#377eb8', '#4daf4a', '#ff7f00', '#e41a1c']

        for i, year in enumerate(years):
            vals = []
            for partner in sorted_partners:
                sub = df_imports[
                    (df_imports['indicators'] == indicator) &
                    (df_imports['product'] == product) &
                    (df_imports['year'] == year) &
                    (df_imports['partner'] == partner)
                ]['value'].sum()
                vals.append(sub)
            ax.bar(x + i * width, vals, width, label=str(year),
                   color=year_colors[i % len(year_colors)], alpha=0.85)

        codes = [get_country_code(p) for p in sorted_partners]
        ax.set_xticks(x + width * (n_bars - 1) / 2)
        ax.set_xticklabels(codes, fontsize=10)
        ax.set_ylabel(unit_label, fontsize=11)
        ax.set_title(f'EU-27 Imports of {short} — Top {N_PARTNERS} Extra-EU Partners\n({unit_label})',
                     fontsize=13, fontweight='bold')
        ax.legend(title='Year')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f'{v / 1e9:.1f}B' if v >= 1e9 else (f'{v / 1e6:.0f}M' if v >= 1e6 else f'{v:.0f}')
        ))
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fname = f"absolute_{short.replace(' ', '_').replace('&', 'and')}_{filename_suffix}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
        print(f"  Saved {fname}")
        plt.close(fig)


# ============================================================================
# CHART 3: Mineral fuels disaggregated — stacked by sub-product per partner
# ============================================================================

def chart_mineral_fuels_disaggregated(df_imports, totals_dict, indicator,
                                       unit_label, filename_suffix):
    """
    For each year: horizontal stacked bar chart.
    X = sub-product contribution, Y = top partner countries.
    Shows how each partner's fuel exports are composed.
    """
    years = TARGET_YEARS
    subproducts = MINERAL_FUEL_SUBPRODUCTS

    # Sub-product colors
    sub_colors = {
        'Coal, coke and briquettes': '#333333',
        'Petroleum, petroleum products and related materials': '#984ea3',
        'Gas, natural and manufactured': '#e41a1c',
        'Electric current': '#ffff33',
    }

    for year in years:
        # Find top partners for total mineral fuels
        agg_product = 'Mineral fuels, lubricants and related materials'
        top, total_val = compute_top_partners(
            df_imports, indicator, agg_product, year, n=N_PARTNERS)
        partners = top['partner'].tolist()

        fig, ax = plt.subplots(figsize=(12, 7))
        y_pos = np.arange(len(partners))

        for i, subp in enumerate(subproducts):
            short_sub = PRODUCT_SHORT.get(subp, subp)
            vals = []
            for partner in partners:
                v = df_imports[
                    (df_imports['indicators'] == indicator) &
                    (df_imports['product'] == subp) &
                    (df_imports['year'] == year) &
                    (df_imports['partner'] == partner)
                ]['value'].sum()
                vals.append(v)

            if i == 0:
                lefts = np.zeros(len(partners))
            ax.barh(y_pos, vals, left=lefts, height=0.6,
                    label=short_sub, color=sub_colors.get(subp, OTHER_COLOR),
                    edgecolor='white', linewidth=0.5)
            lefts = lefts + np.array(vals)

        codes = [get_country_code(p) for p in partners]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(codes, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel(unit_label, fontsize=11)
        ax.set_title(f'EU-27 Mineral Fuel Imports — Disaggregated by Type ({year})\n'
                     f'Top {N_PARTNERS} Extra-EU Partners ({unit_label})',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower right')
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f'{v / 1e9:.1f}B' if v >= 1e9 else (f'{v / 1e6:.0f}M' if v >= 1e6 else f'{v:.0f}')
        ))
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        fname = f"mineral_fuels_disagg_{year}_{filename_suffix}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
        print(f"  Saved {fname}")
        plt.close(fig)


# ============================================================================
# CHART 4: HHI concentration index over time
# ============================================================================

def chart_hhi_over_time(df_imports, products, indicator, unit_label, filename_suffix):
    """
    Line chart showing HHI concentration index for each product across years.
    Higher HHI = more concentrated (more dependent on fewer suppliers).
    """
    all_years = sorted(df_imports['year'].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    for product in products:
        short = PRODUCT_SHORT.get(product, product)
        hhis = [compute_hhi(df_imports, indicator, product, y) for y in all_years]
        color = PRODUCT_COLORS.get(short, '#333333')
        ax.plot(all_years, hhis, marker='o', linewidth=2.5, markersize=8,
                label=short, color=color)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('HHI (0–10,000)', fontsize=11)
    ax.set_title(f'EU-27 Import Concentration (HHI) — Extra-EU Partners\n({unit_label})',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(all_years)

    # Reference lines
    ax.axhline(1500, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(2500, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(all_years[0], 1550, 'Unconcentrated (< 1500)', fontsize=8, color='green')
    ax.text(all_years[0], 2550, 'Highly concentrated (> 2500)', fontsize=8, color='red')

    plt.tight_layout()
    fname = f"hhi_concentration_{filename_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
    print(f"  Saved {fname}")
    plt.close(fig)


# ============================================================================
# CHART 5: Top-5 partner share evolution (area chart)
# ============================================================================

def chart_top10_share_evolution(df_imports, df_raw, products, indicator,
                                unit_label, filename_suffix,
                                reporter_label='EU-27',
                                total_partner=EXTRA_EU_PARTNER,
                                scope_label='extra-EU',
                                file_prefix=''):
    """
    Stacked area chart showing how the share of the top 10 partners evolved.
    Uses the all-year dataset for full annual granularity (2002-2025).
    Denominator = total_partner aggregate row from df_raw.
    df_imports: filtered (countries only). df_raw: unfiltered (has aggregate rows).
    One subplot per product.
    """
    all_years = sorted(df_imports['year'].unique())
    n_top = 10

    fig, axes = plt.subplots(1, len(products), figsize=(7 * len(products), 7),
                             sharey=True)
    if len(products) == 1:
        axes = [axes]

    for ax, product in zip(axes, products):
        short = PRODUCT_SHORT.get(product, product)

        # Get union of top partners across all years
        all_partner_totals = {}
        for year in all_years:
            sub = df_imports[
                (df_imports['indicators'] == indicator) &
                (df_imports['product'] == product) &
                (df_imports['year'] == year)
            ].groupby('partner')['value'].sum()
            for p, v in sub.items():
                all_partner_totals[p] = all_partner_totals.get(p, 0) + v

        top_partners = sorted(all_partner_totals, key=all_partner_totals.get,
                              reverse=True)[:n_top]

        # Build share matrix using Extra-EU27 aggregate as denominator
        shares = {p: [] for p in top_partners}
        other_shares = []
        for year in all_years:
            # Denominator: aggregate row from raw data
            year_total = df_raw[
                (df_raw['flow'] == 'IMPORT') &
                (df_raw['indicators'] == indicator) &
                (df_raw['product'] == product) &
                (df_raw['year'] == year) &
                (df_raw['partner'] == total_partner)
            ]['value'].sum()

            top_sum = 0
            for p in top_partners:
                v = df_imports[
                    (df_imports['indicators'] == indicator) &
                    (df_imports['product'] == product) &
                    (df_imports['year'] == year) &
                    (df_imports['partner'] == p)
                ]['value'].sum()
                pct = v / year_total * 100 if year_total > 0 else 0
                shares[p].append(pct)
                top_sum += pct
            other_shares.append(max(100 - top_sum, 0))

        # Plot stacked area
        stack_data = [shares[p] for p in top_partners] + [other_shares]
        labels = [get_country_code(p) for p in top_partners] + ['Other']
        colors = PARTNER_COLORS[:n_top] + [OTHER_COLOR]

        ax.stackplot(all_years, *stack_data, labels=labels, colors=colors, alpha=0.85)
        ax.set_title(short, fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        # Show every 2nd year to avoid label crowding
        ax.set_xticks(all_years[::2])
        ax.set_xticklabels([str(y) for y in all_years[::2]], rotation=45, fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel(f'Share of {scope_label} imports (%)', fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower left', fontsize=6, ncol=2)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'{reporter_label} Import Source Evolution — Top 10 Partners\n({unit_label})',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    fname = f"{file_prefix}top10_evolution_{filename_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
    print(f"  Saved {fname}")
    plt.close(fig)


# ============================================================================
# CHART 6: HHI for mineral fuel sub-products
# ============================================================================

def chart_hhi_mineral_subproducts(df_imports, indicator, unit_label,
                                  filename_suffix, reporter_label='EU-27',
                                  file_prefix='', scope_label='Extra-EU',
                                  subproducts=None):
    """
    Line chart showing HHI for each mineral fuel sub-product over time.
    """
    if subproducts is None:
        subproducts = MINERAL_FUEL_SUBPRODUCTS
    all_years = sorted(df_imports['year'].unique())
    sub_colors = {
        'Coal, coke and briquettes': '#333333',
        'Petroleum, petroleum products and related materials': '#984ea3',
        'Gas, natural and manufactured': '#e41a1c',
        'Electric current': '#ffff33',
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for subp in subproducts:
        short = PRODUCT_SHORT.get(subp, subp)
        hhis = [compute_hhi(df_imports, indicator, subp, y) for y in all_years]
        ax.plot(all_years, hhis, marker='o', linewidth=2.5, markersize=8,
                label=short, color=sub_colors.get(subp, '#333'))

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('HHI (0–10,000)', fontsize=11)
    ax.set_title(f'{reporter_label} Mineral Fuel Import Concentration by Type\n({unit_label})',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(all_years)
    ax.axhline(1500, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(2500, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(all_years[0], 1550, 'Unconcentrated (< 1500)', fontsize=8, color='green')
    ax.text(all_years[0], 2550, 'Highly concentrated (> 2500)', fontsize=8, color='red')

    plt.tight_layout()
    fname = f"{file_prefix}hhi_mineral_sub_{filename_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
    print(f"  Saved {fname}")
    plt.close(fig)


# ============================================================================
# CHART 7: Mineral fuels — grouped by partner, stacked by sub-product, across time
# ============================================================================

def chart_mineral_fuels_grouped_stacked(df_imports, indicator, unit_label,
                                        filename_suffix,
                                        reporter_label='EU-27',
                                        scope_label='Extra-EU',
                                        file_prefix=''):
    """
    Grouped+stacked bar chart:
      X-axis  = top partner countries (grouped)
      Bars    = one per year within each partner group
      Stacked = Coal, Petroleum, Gas  (no Electricity)
      Scale   = 0 to max + 10%
    """
    years = TARGET_YEARS
    agg_product = 'Mineral fuels, lubricants and related materials'
    subproducts_no_elec = [
        'Coal, coke and briquettes',
        'Petroleum, petroleum products and related materials',
        'Gas, natural and manufactured',
    ]
    sub_colors = {
        'Coal, coke and briquettes': '#333333',
        'Petroleum, petroleum products and related materials': '#984ea3',
        'Gas, natural and manufactured': '#e41a1c',
    }

    # Union of top partners across all years, ranked by cumulated value
    all_partners = {}
    for year in years:
        top, _ = compute_top_partners(df_imports, indicator, agg_product,
                                      year, n=N_PARTNERS)
        for _, row in top.iterrows():
            all_partners[row['partner']] = (
                all_partners.get(row['partner'], 0) + row['value'])
    sorted_partners = sorted(all_partners, key=all_partners.get,
                             reverse=True)[:N_PARTNERS]

    n_partners = len(sorted_partners)
    n_years = len(years)
    bar_width = 0.8 / n_years
    x = np.arange(n_partners)

    fig, ax = plt.subplots(figsize=(16, 8))

    global_max = 0  # track max bar height for x-scale

    for yi, year in enumerate(years):
        # Collect stacked values per sub-product for every partner
        bottoms = np.zeros(n_partners)
        for si, subp in enumerate(subproducts_no_elec):
            vals = np.array([
                df_imports[
                    (df_imports['indicators'] == indicator) &
                    (df_imports['product'] == subp) &
                    (df_imports['year'] == year) &
                    (df_imports['partner'] == partner)
                ]['value'].sum()
                for partner in sorted_partners
            ])
            label = PRODUCT_SHORT.get(subp, subp) if yi == 0 else None
            ax.bar(x + yi * bar_width, vals, bar_width, bottom=bottoms,
                   color=sub_colors[subp], edgecolor='white', linewidth=0.4,
                   label=label)
            bottoms += vals

        # Year label on top of each bar group (first partner only)
        max_h = bottoms.max()
        if max_h > global_max:
            global_max = max_h

        # Add year labels above the tallest bar of each year-group
        tallest_idx = bottoms.argmax()
        ax.text(x[tallest_idx] + yi * bar_width, bottoms[tallest_idx],
                str(year), ha='center', va='bottom', fontsize=7,
                fontweight='bold', color='#444444')

    # X-axis: partner codes
    codes = [get_country_code(p) for p in sorted_partners]
    ax.set_xticks(x + bar_width * (n_years - 1) / 2)
    ax.set_xticklabels(codes, fontsize=10)

    # Y-scale: 0 to max + 10%
    ax.set_ylim(0, global_max * 1.10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v / 1e9:.1f}B' if v >= 1e9 else (
            f'{v / 1e6:.0f}M' if v >= 1e6 else f'{v:.0f}')
    ))

    ax.set_ylabel(unit_label, fontsize=11)
    ax.set_title(
        f'{reporter_label} Mineral Fuel Imports by Partner & Type (excl. Electricity)\n'
        f'Top {N_PARTNERS} {scope_label} Partners — {years[0]}–{years[-1]} ({unit_label})',
        fontsize=13, fontweight='bold')

    # Build combined legend: sub-products + year markers
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=sub_colors[s], label=PRODUCT_SHORT[s])
               for s in subproducts_no_elec]
    # Add small colored squares for years (use bar edge)
    year_colors_legend = ['#377eb8', '#4daf4a', '#ff7f00', '#e41a1c']
    for i, year in enumerate(years):
        handles.append(Line2D([0], [0], marker='s', color='w',
                              markerfacecolor=year_colors_legend[i % 4],
                              markersize=8, label=str(year)))
    ax.legend(handles=handles, loc='upper right', fontsize=9)

    # Add thin colored line at the base of each year's bars for year identification
    for yi, year in enumerate(years):
        for pi in range(n_partners):
            ax.bar(x[pi] + yi * bar_width, 0, bar_width,
                   edgecolor=year_colors_legend[yi % 4], linewidth=0)
            # Small colored tick at the bottom
            ax.plot(x[pi] + yi * bar_width, 0, marker='|', markersize=6,
                    color=year_colors_legend[yi % 4], markeredgewidth=1.5)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fname = f"{file_prefix}mineral_fuels_partner_time_{filename_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
    print(f"  Saved {fname}")
    plt.close(fig)


# ============================================================================
# CHART 8: Mineral fuels — grouped by partner, stacked sub-product, % of imports
# ============================================================================

def chart_mineral_fuels_grouped_stacked_pct(df_imports, totals_dict, indicator,
                                            unit_label, filename_suffix,
                                            reporter_label='EU-27',
                                            scope_label='Extra-EU',
                                            file_prefix=''):
    """
    Same layout as chart 7 but values expressed as % of total extra-EU
    mineral fuel imports (using the aggregate product total from totals_dict).
    """
    years = TARGET_YEARS
    agg_product = 'Mineral fuels, lubricants and related materials'
    subproducts_no_elec = [
        'Coal, coke and briquettes',
        'Petroleum, petroleum products and related materials',
        'Gas, natural and manufactured',
    ]
    sub_colors = {
        'Coal, coke and briquettes': '#333333',
        'Petroleum, petroleum products and related materials': '#984ea3',
        'Gas, natural and manufactured': '#e41a1c',
    }

    # Union of top partners across all years, ranked by cumulated value
    all_partners = {}
    for year in years:
        top, _ = compute_top_partners(df_imports, indicator, agg_product,
                                      year, n=N_PARTNERS)
        for _, row in top.iterrows():
            all_partners[row['partner']] = (
                all_partners.get(row['partner'], 0) + row['value'])
    sorted_partners = sorted(all_partners, key=all_partners.get,
                             reverse=True)[:N_PARTNERS]

    n_partners = len(sorted_partners)
    n_years = len(years)
    bar_width = 0.8 / n_years
    x = np.arange(n_partners)

    fig, ax = plt.subplots(figsize=(16, 8))

    global_max = 0

    for yi, year in enumerate(years):
        # Get the Extra-EU27 aggregate total for mineral fuels this year
        extra_eu_total = totals_dict.get((indicator, year, agg_product), 0)

        bottoms = np.zeros(n_partners)
        for si, subp in enumerate(subproducts_no_elec):
            vals_abs = np.array([
                df_imports[
                    (df_imports['indicators'] == indicator) &
                    (df_imports['product'] == subp) &
                    (df_imports['year'] == year) &
                    (df_imports['partner'] == partner)
                ]['value'].sum()
                for partner in sorted_partners
            ])
            vals_pct = vals_abs / extra_eu_total * 100 if extra_eu_total > 0 else vals_abs * 0
            label = PRODUCT_SHORT.get(subp, subp) if yi == 0 else None
            ax.bar(x + yi * bar_width, vals_pct, bar_width, bottom=bottoms,
                   color=sub_colors[subp], edgecolor='white', linewidth=0.4,
                   label=label)
            bottoms += vals_pct

        max_h = bottoms.max()
        if max_h > global_max:
            global_max = max_h

        tallest_idx = bottoms.argmax()
        ax.text(x[tallest_idx] + yi * bar_width, bottoms[tallest_idx],
                str(year), ha='center', va='bottom', fontsize=7,
                fontweight='bold', color='#444444')

    codes = [get_country_code(p) for p in sorted_partners]
    ax.set_xticks(x + bar_width * (n_years - 1) / 2)
    ax.set_xticklabels(codes, fontsize=10)

    ax.set_ylim(0, global_max * 1.10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    ax.set_ylabel(f'Share of {scope_label.lower()} mineral fuel imports (%)', fontsize=11)
    ax.set_title(
        f'{reporter_label} Mineral Fuel Imports by Partner & Type (excl. Electricity)\n'
        f'Top {N_PARTNERS} {scope_label} Partners — {years[0]}–{years[-1]} '
        f'(% of total {scope_label.lower()} imports, {unit_label}-based)',
        fontsize=13, fontweight='bold')

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=sub_colors[s], label=PRODUCT_SHORT[s])
               for s in subproducts_no_elec]
    year_colors_legend = ['#377eb8', '#4daf4a', '#ff7f00', '#e41a1c']
    for i, year in enumerate(years):
        handles.append(Line2D([0], [0], marker='s', color='w',
                              markerfacecolor=year_colors_legend[i % 4],
                              markersize=8, label=str(year)))
    ax.legend(handles=handles, loc='upper right', fontsize=9)

    for yi, year in enumerate(years):
        for pi in range(n_partners):
            ax.plot(x[pi] + yi * bar_width, 0, marker='|', markersize=6,
                    color=year_colors_legend[yi % 4], markeredgewidth=1.5)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fname = f"{file_prefix}mineral_fuels_partner_time_pct_{filename_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight')
    print(f"  Saved {fname}")
    plt.close(fig)


# ============================================================================
# EXCEL EXPORT
# ============================================================================

def export_mineral_fuels_partner_time_excel(df_imports, totals_dict,
                                            indicator, unit_label,
                                            file_prefix=''):
    """
    Export data behind mineral_fuels_partner_time charts (7 + 8) to Excel.
    One sheet per year, columns = sub-products, rows = partners.
    Includes both absolute values and % of mineral fuel imports.
    """
    agg_product = 'Mineral fuels, lubricants and related materials'
    subproducts_no_elec = [
        'Coal, coke and briquettes',
        'Petroleum, petroleum products and related materials',
        'Gas, natural and manufactured',
    ]

    # Determine top partners (same logic as the chart)
    all_partners = {}
    for year in TARGET_YEARS:
        top, _ = compute_top_partners(df_imports, indicator, agg_product,
                                      year, n=N_PARTNERS)
        for _, row in top.iterrows():
            all_partners[row['partner']] = (
                all_partners.get(row['partner'], 0) + row['value'])
    sorted_partners = sorted(all_partners, key=all_partners.get,
                             reverse=True)[:N_PARTNERS]

    suffix = 'eur' if 'EUR' in indicator else 'kg'
    fname = f"{file_prefix}mineral_fuels_partner_time_{suffix}.xlsx"
    fpath = os.path.join(EXCEL_OUTPUT_DIR, fname)

    with pd.ExcelWriter(fpath, engine='openpyxl') as writer:
        for year in TARGET_YEARS:
            extra_eu_total = totals_dict.get((indicator, year, agg_product), 0)

            rows_abs = []
            rows_pct = []
            for partner in sorted_partners:
                code = get_country_code(partner)
                row_abs = {'Partner': partner, 'Code': code}
                row_pct = {'Partner': partner, 'Code': code}
                total_abs = 0
                for subp in subproducts_no_elec:
                    short = PRODUCT_SHORT.get(subp, subp)
                    val = df_imports[
                        (df_imports['indicators'] == indicator) &
                        (df_imports['product'] == subp) &
                        (df_imports['year'] == year) &
                        (df_imports['partner'] == partner)
                    ]['value'].sum()
                    row_abs[short] = val
                    pct = val / extra_eu_total * 100 if extra_eu_total > 0 else 0
                    row_pct[short] = round(pct, 4)
                    total_abs += val
                row_abs['Total'] = total_abs
                row_pct['Total'] = round(
                    total_abs / extra_eu_total * 100 if extra_eu_total > 0 else 0, 4)
                rows_abs.append(row_abs)
                rows_pct.append(row_pct)

            df_abs = pd.DataFrame(rows_abs)
            df_pct = pd.DataFrame(rows_pct)
            df_abs.to_excel(writer, sheet_name=f'{year}_abs ({unit_label})',
                            index=False)
            df_pct.to_excel(writer, sheet_name=f'{year}_pct',
                            index=False)

    print(f"  Exported {fname}")


def export_top10_evolution_excel(df_imports, df_raw, products, indicator,
                                 unit_label, total_partner=EXTRA_EU_PARTNER,
                                 file_prefix=''):
    """
    Export data behind top10_evolution charts (5) to Excel.
    One sheet per product: rows = years, columns = partners (top 10 + Other).
    Values are share of imports (%).
    """
    all_years = sorted(df_imports['year'].unique())
    n_top = 10

    suffix = 'eur' if 'EUR' in indicator else 'kg'
    fname = f"{file_prefix}top10_evolution_{suffix}.xlsx"
    fpath = os.path.join(EXCEL_OUTPUT_DIR, fname)

    with pd.ExcelWriter(fpath, engine='openpyxl') as writer:
        for product in products:
            short = PRODUCT_SHORT.get(product, product)

            # Get union of top partners across all years
            all_partner_totals = {}
            for year in all_years:
                sub = df_imports[
                    (df_imports['indicators'] == indicator) &
                    (df_imports['product'] == product) &
                    (df_imports['year'] == year)
                ].groupby('partner')['value'].sum()
                for p, v in sub.items():
                    all_partner_totals[p] = all_partner_totals.get(p, 0) + v

            top_partners = sorted(all_partner_totals,
                                  key=all_partner_totals.get,
                                  reverse=True)[:n_top]

            rows = []
            for year in all_years:
                year_total = df_raw[
                    (df_raw['flow'] == 'IMPORT') &
                    (df_raw['indicators'] == indicator) &
                    (df_raw['product'] == product) &
                    (df_raw['year'] == year) &
                    (df_raw['partner'] == total_partner)
                ]['value'].sum()

                row = {'Year': year}
                top_sum = 0
                for p in top_partners:
                    v = df_imports[
                        (df_imports['indicators'] == indicator) &
                        (df_imports['product'] == product) &
                        (df_imports['year'] == year) &
                        (df_imports['partner'] == p)
                    ]['value'].sum()
                    pct = round(v / year_total * 100, 4) if year_total > 0 else 0
                    row[get_country_code(p)] = pct
                    top_sum += pct
                row['Other'] = round(max(100 - top_sum, 0), 4)
                rows.append(row)

            df_out = pd.DataFrame(rows)
            # Truncate sheet name to 31 chars (Excel limit)
            sheet = f'{short} ({unit_label})'[:31]
            df_out.to_excel(writer, sheet_name=sheet, index=False)

    print(f"  Exported {fname}")


def export_to_excel(df_imports, totals_dict, products, indicator, unit_label):
    """Export structured data for all products to Excel."""
    rows = []
    for year in TARGET_YEARS:
        for product in products:
            short = PRODUCT_SHORT.get(product, product)
            extra_total = totals_dict.get((indicator, year, product), 0)

            top, _ = compute_top_partners(df_imports, indicator, product, year,
                                          n=N_PARTNERS)
            for _, r in top.iterrows():
                pct = r['value'] / extra_total * 100 if extra_total > 0 else 0
                rows.append({
                    'visual_name': f'EU-27 Import Dependency — {short}',
                    'year': year,
                    'filter_1': r['partner'],
                    'filter_2': get_country_code(r['partner']),
                    'value': r['value'],
                    'share_pct': round(pct, 2),
                    'unit': unit_label,
                })

    out = pd.DataFrame(rows)
    fname = f"eu27_import_dependency_{indicator.lower()}.xlsx"
    out.to_excel(os.path.join(EXCEL_OUTPUT_DIR, fname), index=False)
    print(f"  Exported {fname} ({len(out)} rows)")
    return out


# ============================================================================
# CLUSTER ANALYSIS: Top-10 evolution for EU-27 + 8 EWBI cluster countries
# ============================================================================

def _compute_shares_for_reporter(df_reporter, product, eu_top_partners):
    """
    Compute share time-series for a single reporter and product.
    Only eu_top_partners are tracked individually; everything else → Other.
    """
    all_years = sorted(df_reporter['year'].unique())

    # Non-EU individual partners
    df_np = df_reporter[
        (df_reporter['product'] == product) &
        (~df_reporter['partner'].isin(AGGREGATE_PARTNERS)) &
        (~df_reporter['partner'].apply(is_eu27_partner))
    ]

    shares = {p: [] for p in eu_top_partners}
    other_shares = []
    for year in all_years:
        # Denominator: sum of individual non-EU partners
        # (Extra-EU27 aggregate is unreliable for some countries, e.g.
        #  DE/AT gas where individual partners cover <2% of aggregate)
        denom = df_np[df_np['year'] == year]['value'].sum()

        top_sum = 0
        for p in eu_top_partners:
            v = df_np[
                (df_np['year'] == year) & (df_np['partner'] == p)
            ]['value'].sum()
            pct = v / denom * 100 if denom > 0 else 0
            shares[p].append(pct)
            top_sum += pct
        other_shares.append(max(100 - top_sum, 0))

    return all_years, shares, other_shares


def chart_cluster_top10_evolution():
    """
    Generate top-10 partner share evolution charts for EU-27 and 8 EWBI
    cluster countries.  Uses the new per-country imports data (VALUE_EUR).

    Top 10 partners are determined at EU-27 level per product (~20 unique).
    All country charts use the SAME partner set and colours.
    Country-specific suppliers not in the EU top → "Other".

    Output: 3 figures (one per product), each with 9 subplots
    (EU-27 + 8 countries) and one shared legend.
    """
    from matplotlib.patches import Patch

    print("\n" + "=" * 60)
    print("Cluster Analysis: Top-10 Import Source Evolution (New Data)")
    print("=" * 60)

    # ── Load new data ──
    print("Loading new per-country imports data...")
    df = pd.read_csv(NEW_IMPORTS_PATH)
    df = df.rename(columns={'TIME_PERIOD': 'year', 'OBS_VALUE': 'value'})
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    print(f"  Loaded {len(df):,} records, "
          f"years: {df['year'].min()}-{df['year'].max()}")

    products = FRANCE_SUBPRODUCTS  # Coal, Petroleum, Gas

    # ── Find EU-27 reporter ──
    eu27_reporter = None
    for r in df['reporter'].unique():
        if 'European Union' in r:
            eu27_reporter = r
            break
    df_eu = df[df['reporter'] == eu27_reporter]

    # ── Build unified colour map from EU-27 top 10 per product ──
    eu_top_per_product = {}  # product → ordered list of top-10 partners
    all_top_partners = []    # deduplicated across products, in order
    for product in products:
        sub = df_eu[
            (df_eu['product'] == product) &
            (~df_eu['partner'].isin(AGGREGATE_PARTNERS)) &
            (~df_eu['partner'].apply(is_eu27_partner))
        ]
        top10 = sub.groupby('partner')['value'].sum().nlargest(10).index.tolist()
        eu_top_per_product[product] = top10
        for p in top10:
            if p not in all_top_partners:
                all_top_partners.append(p)

    color_map = {
        p: CLUSTER_TOP10_COLORS[i % len(CLUSTER_TOP10_COLORS)]
        for i, p in enumerate(all_top_partners)
    }
    print(f"  Unified colour map: {len(all_top_partners)} unique partners")

    # ── Build reporter dict: label → (code, reporter_name, df_subset) ──
    reporters = [('EU-27', 'EU27', df_eu)]
    for code, label in CLUSTER_COUNTRIES.items():
        for r in df['reporter'].unique():
            if label in r:
                reporters.append((label, code, df[df['reporter'] == r]))
                break

    # ── One figure per product, 9 subplots (3×3) ──
    for product in products:
        short = PRODUCT_SHORT.get(product, product)
        top_partners = eu_top_per_product[product]

        n = len(reporters)  # 9
        ncols = 3
        nrows = (n + ncols - 1) // ncols  # 3

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 5 * nrows),
                                 sharey=True)
        axes_flat = axes.flatten()

        for idx, (label, code, df_r) in enumerate(reporters):
            ax = axes_flat[idx]
            all_years, shares, other_shares = _compute_shares_for_reporter(
                df_r, product, top_partners)

            stack_data = [shares[p] for p in top_partners] + [other_shares]
            colors = [color_map[p] for p in top_partners] + [OTHER_COLOR]

            ax.stackplot(all_years, *stack_data, colors=colors, alpha=0.85)
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_xticks(all_years[::4])
            ax.set_xticklabels([str(y) for y in all_years[::4]],
                               rotation=45, fontsize=7)
            ax.set_ylim(0, 100)
            if idx % ncols == 0:
                ax.set_ylabel('Share of extra-EU imports (%)', fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for idx in range(n, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Shared legend below the figure
        legend_handles = [Patch(facecolor=color_map[p], alpha=0.85,
                                label=get_country_code(p))
                          for p in top_partners]
        legend_handles.append(Patch(facecolor=OTHER_COLOR, alpha=0.85,
                                    label='Other'))
        fig.legend(handles=legend_handles, loc='lower center',
                   ncol=min(len(legend_handles), 11), fontsize=9,
                   frameon=True, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            f'{short} — Import Source Evolution (Extra-EU)\n'
            f'EU-27 + 8 EWBI Cluster Countries (EUR value)',
            fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])

        fname = f'cluster_{short.replace(" ", "_").replace("&", "and")}_top10_evolution'
        fig.savefig(os.path.join(OUTPUT_DIR, fname + '.png'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, fname + '.svg'),
                    bbox_inches='tight')
        print(f"  Saved {fname}.png / .svg")
        plt.close(fig)

    # ── Excel export ──
    print("\n[Cluster Excel] Exporting top-10 evolution data")
    fname = 'cluster_top10_evolution_eur.xlsx'
    fpath = os.path.join(EXCEL_OUTPUT_DIR, fname)
    with pd.ExcelWriter(fpath, engine='openpyxl') as writer:
        for product in products:
            short = PRODUCT_SHORT.get(product, product)
            top_partners = eu_top_per_product[product]

            for label, code, df_r in reporters:
                all_years, shares, other_shares = _compute_shares_for_reporter(
                    df_r, product, top_partners)
                rows = []
                for i, year in enumerate(all_years):
                    row = {'Year': year}
                    for p in top_partners:
                        row[get_country_code(p)] = round(shares[p][i], 4)
                    row['Other'] = round(other_shares[i], 4)
                    rows.append(row)
                df_out = pd.DataFrame(rows)
                sheet = f'{code} {short}'[:31]
                df_out.to_excel(writer, sheet_name=sheet, index=False)
    print(f"  Exported {fname}")

    print("  Cluster analysis complete!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("EU-27 Trade Dependency Analysis")
    print("=" * 60)

    df = load_trade_data()
    df_all_year = load_trade_data_all_years()
    gdp = load_gdp_data()

    # Filter for extra-EU imports — aggregate products
    df_extra_agg = filter_extra_eu_imports(df, AGGREGATE_PRODUCTS)
    totals_agg = get_extra_eu_total(df, AGGREGATE_PRODUCTS)

    # Filter for extra-EU imports — mineral fuel sub-products
    df_extra_sub = filter_extra_eu_imports(df, MINERAL_FUEL_SUBPRODUCTS)
    totals_sub = get_extra_eu_total(df, MINERAL_FUEL_SUBPRODUCTS)

    # Merge to have both aggregate and sub for mineral fuels disagg chart
    all_products = AGGREGATE_PRODUCTS + MINERAL_FUEL_SUBPRODUCTS
    df_extra_all = filter_extra_eu_imports(df, all_products)
    totals_all = {**totals_agg, **totals_sub}

    # All-year data: filter extra-EU imports (all years available)
    all_year_products = [p for p in df_all_year['product'].unique()]
    df_all_year_extra = filter_extra_eu_imports(
        df_all_year, all_year_products,
        years=sorted(df_all_year['year'].unique()))

    for indicator, unit_label, suffix in [
        ('VALUE_EUR', 'EUR', 'eur'),
        ('QUANTITY_KG', 'kg', 'kg'),
    ]:
        print(f"\n{'—' * 40}")
        print(f"  Indicator: {indicator} ({unit_label})")
        print(f"{'—' * 40}")

        # Chart 1: Stacked % of extra-EU imports — aggregate products
        print("\n[Chart 1] Stacked share — aggregate products")
        chart_stacked_top_partners_pct(
            df_extra_agg, totals_agg, AGGREGATE_PRODUCTS,
            indicator, unit_label, suffix)

        # Chart 2: Grouped absolute bars — aggregate products
        print("\n[Chart 2] Grouped absolute bars — aggregate products")
        chart_grouped_bars_absolute(
            df_extra_agg, AGGREGATE_PRODUCTS, indicator, unit_label, suffix)

        # Chart 3: Mineral fuels disaggregated
        print("\n[Chart 3] Mineral fuels disaggregated")
        chart_mineral_fuels_disaggregated(
            df_extra_all, totals_all, indicator, unit_label, suffix)

        # Chart 4: HHI concentration — aggregate products
        print("\n[Chart 4] HHI concentration — aggregate products")
        chart_hhi_over_time(
            df_extra_agg, AGGREGATE_PRODUCTS, indicator, unit_label, suffix)

        # Chart 5: Top-10 share evolution — mineral fuel sub-products (all years)
        print("\n[Chart 5] Top-10 partner share evolution (all years)")
        chart_top10_share_evolution(
            df_all_year_extra, df_all_year, all_year_products, indicator,
            unit_label, suffix)

        # Chart 6: HHI for mineral fuel sub-products
        print("\n[Chart 6] HHI — mineral fuel sub-products")
        chart_hhi_mineral_subproducts(
            df_extra_all, indicator, unit_label, suffix)

        # Chart 7: Mineral fuels grouped+stacked across time
        print("\n[Chart 7] Mineral fuels — partner x time stacked")
        chart_mineral_fuels_grouped_stacked(
            df_extra_all, indicator, unit_label, suffix)

        # Chart 8: Mineral fuels grouped+stacked — share of imports
        print("\n[Chart 8] Mineral fuels — partner x time stacked (% share)")
        chart_mineral_fuels_grouped_stacked_pct(
            df_extra_all, totals_all, indicator, unit_label, suffix)

        # Excel export
        print("\n[Excel] Exporting structured data")
        export_to_excel(df_extra_agg, totals_agg, AGGREGATE_PRODUCTS,
                        indicator, unit_label)

        # Excel export — mineral fuels partner x time
        print("[Excel] Exporting mineral fuels partner x time data")
        export_mineral_fuels_partner_time_excel(
            df_extra_all, totals_all, indicator, unit_label)

        # Excel export — top-10 evolution
        print("[Excel] Exporting top-10 evolution data")
        export_top10_evolution_excel(
            df_all_year_extra, df_all_year, all_year_products,
            indicator, unit_label)

    print("\n" + "=" * 60)
    print(f"All EU-27 outputs saved to: {OUTPUT_DIR}")
    print(f"Excel files saved to: {EXCEL_OUTPUT_DIR}")

    # ==================================================================
    # FRANCE ANALYSIS
    # ==================================================================
    print("\n" + "=" * 60)
    print("France Trade Dependency Analysis (Mineral Fuels)")
    print("=" * 60)

    df_france = load_trade_data_all_years(reporter=FRANCE_REPORTER)

    # Create synthetic aggregate "Mineral fuels" rows (sum of sub-products)
    fr_sub = df_france[df_france['product'].isin(FRANCE_SUBPRODUCTS)]
    fr_agg = (fr_sub
              .groupby(['reporter', 'flow', 'partner', 'year', 'indicators'],
                       as_index=False)['value'].sum())
    fr_agg['product'] = 'Mineral fuels, lubricants and related materials'
    df_france = pd.concat([df_france, fr_agg], ignore_index=True)

    france_products = FRANCE_SUBPRODUCTS + [
        'Mineral fuels, lubricants and related materials']
    france_years = sorted(df_france['year'].unique())

    # Filter individual partners (include EU members)
    df_fr_filtered = filter_extra_eu_imports(
        df_france, france_products, years=france_years,
        exclude_eu_members=False)

    # Totals using "All countries of the world" as denominator
    totals_fr = get_extra_eu_total(
        df_france, france_products, years=france_years,
        total_partner=FRANCE_TOTAL_PARTNER)

    for indicator, unit_label, suffix in [
        ('VALUE_EUR', 'EUR', 'eur'),
        ('QUANTITY_KG', 'kg', 'kg'),
    ]:
        print(f"\n{'—' * 40}")
        print(f"  France — Indicator: {indicator} ({unit_label})")
        print(f"{'—' * 40}")

        fr_kw = dict(reporter_label='France', scope_label='Total',
                     file_prefix='france_')

        # Chart 5: Top-10 share evolution
        print("\n[FR Chart 5] Top-10 partner share evolution (all years)")
        chart_top10_share_evolution(
            df_fr_filtered, df_france, FRANCE_SUBPRODUCTS, indicator,
            unit_label, suffix,
            total_partner=FRANCE_TOTAL_PARTNER, **fr_kw)

        # Chart 6: HHI mineral sub-products
        print("\n[FR Chart 6] HHI — mineral fuel sub-products")
        chart_hhi_mineral_subproducts(
            df_fr_filtered, indicator, unit_label, suffix,
            subproducts=FRANCE_SUBPRODUCTS, **fr_kw)

        # Chart 7: Mineral fuels grouped+stacked
        print("\n[FR Chart 7] Mineral fuels — partner x time stacked")
        chart_mineral_fuels_grouped_stacked(
            df_fr_filtered, indicator, unit_label, suffix, **fr_kw)

        # Chart 8: Mineral fuels grouped+stacked % share
        print("\n[FR Chart 8] Mineral fuels — partner x time stacked (% share)")
        chart_mineral_fuels_grouped_stacked_pct(
            df_fr_filtered, totals_fr, indicator, unit_label, suffix, **fr_kw)

        # Excel export — mineral fuels partner x time
        print("[FR Excel] Exporting mineral fuels partner x time data")
        export_mineral_fuels_partner_time_excel(
            df_fr_filtered, totals_fr, indicator, unit_label,
            file_prefix='france_')

        # Excel export — top-10 evolution
        print("[FR Excel] Exporting top-10 evolution data")
        export_top10_evolution_excel(
            df_fr_filtered, df_france, FRANCE_SUBPRODUCTS,
            indicator, unit_label,
            total_partner=FRANCE_TOTAL_PARTNER, file_prefix='france_')

    # ==================================================================
    # CLUSTER ANALYSIS (new per-country imports data)
    # ==================================================================
    chart_cluster_top10_evolution()

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"Excel files saved to: {EXCEL_OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
