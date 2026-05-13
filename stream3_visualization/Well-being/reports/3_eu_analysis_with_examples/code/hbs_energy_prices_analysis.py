"""
Energy Price Components Analysis - Electricity & Gas

Analysis of electricity and gas price components for France (FR) and EU27_2020:
- Decomposition of price components for last available year
- Time series evolution of price components by consumption band

Data source: Eurostat price component data (PPS - Purchasing Power Standard)
Author: Data for Good - Well-being Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

# Colors for price components (from FR dashboard)
COLORS = {
    'NRG_SUP': '#fdb462',           # Orange - Energy and supply
    'NETC': '#8dd3c7',              # Cyan - Network costs
    'TAX_FEE_LEV_CHRG': '#bebada',  # Mauve - Taxes, fees, levies and charges
    'VAT': '#b3de69',               # Green - Value added tax
    'TAX_RNW': '#ffd558',           # Yellow - Renewable taxes
    'TAX_CAP': '#fb8072',           # Red - Capacity taxes
    'TAX_ENV': '#ffffb3',           # Pale - Environmental taxes
    'TAX_NUC': '#80b1d3',           # Blue - Nuclear taxes
    'OTH': '#bc80bd',               # Purple - Other
}

CATEGORY_LABELS = {
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

CONSUMPTION_LABELS = {
    'TOT_KWH': 'All consumption bands',
    'KWH_LT1000': '< 1,000 kWh',
    'KWH1000-2499': '1,000 - 2,499 kWh',
    'KWH2500-4999': '2,500 - 4,999 kWh',
    'KWH5000-14999': '5,000 - 14,999 kWh',
    'KWH_LE15000': '>= 15,000 kWh',
    'GJ_LT20': '< 20 GJ',
    'GJ20-199': '20 - 199 GJ',
    'GJ_GE200': '>= 200 GJ',
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories():
    """Create output directories."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_base = os.path.dirname(script_dir)
    outputs = os.path.join(report_base, 'outputs')
    os.makedirs(os.path.join(outputs, 'graphs', 'Energy_Prices'), exist_ok=True)
    
    return {
        'script_dir': script_dir,
        'outputs': outputs,
        'energy_dir': os.path.join(outputs, 'graphs', 'Energy_Prices'),
        'external_data': os.path.join(report_base, 'external_data')
    }


def load_energy_data(dirs, energy_type):
    """Load electricity or gas price data."""
    file_name = f'price_component_{energy_type}.csv'
    file_path = os.path.join(dirs['external_data'], file_name)
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return pd.DataFrame()
    
    print(f"Loading {energy_type.upper()}: {file_path}")
    df = pd.read_csv(file_path)
    print(f"  OK Loaded {len(df)} records")
    
    return df


def map_nrg_prc(nrg_prc_val):
    """Map nrg_prc text values to codes."""
    mapping = {
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
    return mapping.get(nrg_prc_val, None)


def map_nrg_cons(nrg_cons_val):
    """Map nrg_cons text values to codes."""
    mapping = {
        # Electricity
        'Consumption of kWh - all bands': 'TOT_KWH',
        'Consumption less than 1 000 kWh - band DA': 'KWH_LT1000',
        'Consumption from 1 000 kWh to 2 499 kWh - band DB': 'KWH1000-2499',
        'Consumption from 2 500 kWh to 4 999 kWh - band DC': 'KWH2500-4999',
        'Consumption from 5 000 kWh to 14 999 kWh - band DD': 'KWH5000-14999',
        'Consumption 15 000 kWh or over - band DE': 'KWH_LE15000',
        # Gas
        'Consumption of GJ - all bands': 'TOT_KWH',  # Use TOT_KWH code for all bands
        'Consumption less than 20 GJ - band D1': 'GJ_LT20',
        'Consumption from 20 GJ to 199 GJ - band D2': 'GJ20-199',
        'Consumption 200 GJ or over - band D3': 'GJ_GE200',
    }
    return mapping.get(nrg_cons_val, None)


# EWBI clusters (2 countries per cluster)
CLUSTER_COUNTRIES = {
    'Cluster 0': ['FR', 'ES'],
    'Cluster 1': ['BE', 'AT'],
    'Cluster 2': ['LT', 'HU'],
    'Cluster 3': ['DE', 'PL'],
}

CLUSTER_COLORS = {
    'Cluster 0': '#fb8072',  # salmon
    'Cluster 1': '#fdb462',  # orange
    'Cluster 2': '#8dd3c7',  # teal
    'Cluster 3': '#80b1d3',  # blue
}

ALL_CLUSTER_COUNTRIES = [c for cs in CLUSTER_COUNTRIES.values() for c in cs]


def map_country_name(geo_val):
    """Standardize country names."""
    mapping = {
        'Austria': 'AT',
        'Belgium': 'BE',
        'Bulgaria': 'BG',
        'Croatia': 'HR',
        'Cyprus': 'CY',
        'Czechia': 'CZ',
        'Denmark': 'DK',
        'Estonia': 'EE',
        'Finland': 'FI',
        'France': 'FR',
        'Germany': 'DE',
        'Greece': 'GR',
        'Hungary': 'HU',
        'Ireland': 'IE',
        'Italy': 'IT',
        'Latvia': 'LV',
        'Lithuania': 'LT',
        'Luxembourg': 'LU',
        'Malta': 'MT',
        'Netherlands': 'NL',
        'Poland': 'PL',
        'Portugal': 'PT',
        'Romania': 'RO',
        'Slovakia': 'SK',
        'Slovenia': 'SI',
        'Spain': 'ES',
        'Sweden': 'SE',
        'European Union - 27 countries (from 2020)': 'EU27_2020',
    }
    for key, val in mapping.items():
        if key.lower() in str(geo_val).lower():
            return val
    return geo_val


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def prepare_decomposition_data(df, country_code, energy_type):
    """Prepare data for price decomposition for last available year."""
    print(f"\n=== PREPARING {energy_type.upper()} DECOMPOSITION DATA ({country_code}) ===")
    
    # Add mapped columns
    df['nrg_prc_code'] = df['nrg_prc'].apply(map_nrg_prc)
    df['nrg_cons_code'] = df['nrg_cons'].apply(map_nrg_cons)
    df['country_code'] = df['geo'].apply(map_country_name)
    
    # Filter for country
    df_country = df[df['country_code'] == country_code].copy()
    print(f"OK Found {len(df_country)} records for {country_code}")
    
    if df_country.empty:
        print(f"WARNING: No data found for {country_code}")
        return pd.DataFrame()
    
    # Remove unmapped values
    df_country = df_country.dropna(subset=['nrg_prc_code', 'nrg_cons_code'])
    
    # Get last available year
    last_year = df_country['TIME_PERIOD'].max()
    print(f"OK Last available year: {last_year}")
    
    df_last_year = df_country[df_country['TIME_PERIOD'] == last_year].copy()
    
    # Convert OBS_VALUE to numeric
    df_last_year['OBS_VALUE'] = pd.to_numeric(df_last_year['OBS_VALUE'], errors='coerce')
    
    print(f"OK Extracted {len(df_last_year)} records for year {last_year}")
    
    return df_last_year


def create_decomposition_chart(df, country_code, energy_type, dirs):
    """Create stacked bar chart for price decomposition."""
    print(f"\n=== CREATING {energy_type.upper()} DECOMPOSITION CHART ({country_code}) ===")
    
    if df.empty:
        print("WARNING: No data - skipping chart")
        return
    
    # Create pivot table
    pivot_data = df.pivot_table(
        index='nrg_cons_code',
        columns='nrg_prc_code',
        values='OBS_VALUE',
        aggfunc='first'
    )
    
    # Reorder consumption bands based on energy type
    if energy_type.lower() == 'electricity':
        consumption_order = ['TOT_KWH', 'KWH_LT1000', 'KWH1000-2499', 'KWH2500-4999', 'KWH5000-14999', 'KWH_LE15000']
    else:  # gas
        consumption_order = ['TOT_KWH', 'GJ_LT20', 'GJ20-199', 'GJ_GE200']
    
    pivot_data = pivot_data.reindex([c for c in consumption_order if c in pivot_data.index])
    
    print(f"DEBUG: Pivot data shape: {pivot_data.shape}")
    print(f"DEBUG: Columns in pivot: {list(pivot_data.columns)}")
    print(f"DEBUG: Rows in pivot: {list(pivot_data.index)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define price components order
    price_components = ['NRG_SUP', 'NETC', 'TAX_FEE_LEV_CHRG', 'VAT', 'TAX_RNW', 'TAX_CAP', 'TAX_ENV', 'TAX_NUC', 'OTH']
    price_components = [p for p in price_components if p in pivot_data.columns]
    
    print(f"DEBUG: Components to plot: {price_components}")
    
    # Plot stacked bars
    x_pos = np.arange(len(pivot_data))
    bar_width = 0.6
    bottom = np.zeros(len(pivot_data))
    
    for component in price_components:
        values = pivot_data[component].fillna(0).values
        ax.bar(x_pos, values, bar_width, bottom=bottom,
               label=CATEGORY_LABELS.get(component, component),
               color=COLORS.get(component, '#cccccc'), alpha=0.85, 
               edgecolor='black', linewidth=1)
        bottom += values
    
    # Add total labels on bars
    for i in range(len(pivot_data)):
        total = pivot_data.iloc[i].sum()
        if not np.isnan(total):
            ax.text(i, total + 0.01 * max(bottom), f'{total:.3f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Styling
    consumption_labels = [CONSUMPTION_LABELS.get(c, c) for c in pivot_data.index]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(consumption_labels, fontsize=11, fontweight='bold', rotation=15, ha='right')
    
    year = df['TIME_PERIOD'].max()
    ax.set_ylabel('Price (PPS per kWh)', fontsize=12, fontweight='bold')
    ax.set_title(f'Price Decomposition - {energy_type.upper()} in {country_code} ({year})\nby Consumption Band',
                fontsize=13, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(bottom) * 1.15)
    
    plt.tight_layout()
    
    base_name = f'{country_code}_{energy_type.upper()}_decomposition_{year}'
    for ext in ('png', 'svg'):
        plt.savefig(os.path.join(dirs['energy_dir'], f'{base_name}.{ext}'),
                    dpi=300, bbox_inches='tight')
    print(f"OK Saved: {base_name}.png / .svg")
    
    # Excel export
    excel_df = pivot_data.copy()
    excel_df.index = [CONSUMPTION_LABELS.get(c, c) for c in excel_df.index]
    excel_df.columns = [CATEGORY_LABELS.get(c, c) for c in excel_df.columns]
    excel_path = os.path.join(dirs['energy_dir'], f'{base_name}.xlsx')
    excel_df.to_excel(excel_path, index_label='Consumption Band')
    print(f"OK Saved: {base_name}.xlsx")
    
    plt.close()


def prepare_timeseries_data(df, country_code, energy_type):
    """Prepare data for price components over time."""
    print(f"\n=== PREPARING {energy_type.upper()} TIMESERIES DATA ({country_code}) ===")
    
    # Add mapped columns
    df['nrg_prc_code'] = df['nrg_prc'].apply(map_nrg_prc)
    df['nrg_cons_code'] = df['nrg_cons'].apply(map_nrg_cons)
    df['country_code'] = df['geo'].apply(map_country_name)
    
    # Filter for country and TOT_KWH (all consumption bands)
    df_ts = df[
        (df['country_code'] == country_code) &
        (df['nrg_cons_code'] == 'TOT_KWH')
    ].copy()
    
    print(f"OK Found {len(df_ts)} records for {country_code} (TOT_KWH)")
    
    if df_ts.empty:
        print(f"WARNING: No timeseries data for {country_code}")
        return pd.DataFrame()
    
    # Remove unmapped values
    df_ts = df_ts.dropna(subset=['nrg_prc_code'])
    
    # Convert to numeric
    df_ts['OBS_VALUE'] = pd.to_numeric(df_ts['OBS_VALUE'], errors='coerce')
    df_ts['TIME_PERIOD'] = pd.to_numeric(df_ts['TIME_PERIOD'], errors='coerce')
    
    print(f"OK Extracted timeseries with {df_ts['TIME_PERIOD'].nunique()} years")
    
    return df_ts


def create_timeseries_chart(df, country_code, energy_type, dirs):
    """Create line chart for price components over time."""
    print(f"\n=== CREATING {energy_type.upper()} TIMESERIES CHART ({country_code}) ===")
    
    if df.empty:
        print("WARNING: No data - skipping chart")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get unique components
    components = df['nrg_prc_code'].unique()
    
    # Plot each component
    for component in sorted(components):
        if pd.isna(component):
            continue
        
        comp_data = df[df['nrg_prc_code'] == component].sort_values('TIME_PERIOD')
        
        ax.plot(comp_data['TIME_PERIOD'], comp_data['OBS_VALUE'],
               marker='o', label=CATEGORY_LABELS.get(component, component),
               color=COLORS.get(component, '#cccccc'), linewidth=2.5, markersize=6)
    
    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price Component (PPS per kWh)', fontsize=12, fontweight='bold')
    ax.set_title(f'Price Components Evolution - {energy_type.upper()} in {country_code}\n(All Consumption Bands)',
                fontsize=13, fontweight='bold', pad=20)
    
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.3f}'))
    
    plt.tight_layout()
    
    base_name = f'{country_code}_{energy_type.upper()}_timeseries'
    for ext in ('png', 'svg'):
        plt.savefig(os.path.join(dirs['energy_dir'], f'{base_name}.{ext}'),
                    dpi=300, bbox_inches='tight')
    print(f"OK Saved: {base_name}.png / .svg")
    
    # Excel export
    ts_pivot = df.pivot_table(index='TIME_PERIOD', columns='nrg_prc_code',
                              values='OBS_VALUE', aggfunc='first')
    ts_pivot.columns = [CATEGORY_LABELS.get(c, c) for c in ts_pivot.columns]
    ts_pivot.index.name = 'Year'
    excel_path = os.path.join(dirs['energy_dir'], f'{base_name}.xlsx')
    ts_pivot.to_excel(excel_path)
    print(f"OK Saved: {base_name}.xlsx")
    
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    
    dirs = setup_directories()
    
    print("\n" + "="*80)
    print("ENERGY PRICE COMPONENTS ANALYSIS")
    print("="*80)
    
    # Analysis parameters
    energy_types = ['electricity', 'gas']
    countries = ['FR', 'EU27_2020']
    
    for energy_type in energy_types:
        print(f"\n{'='*80}")
        print(f"{energy_type.upper()} ANALYSIS")
        print(f"{'='*80}")
        
        # Load data
        df = load_energy_data(dirs, energy_type)
        
        if df.empty:
            continue
        
        # Process decomposition for each country
        for country in countries:
            # Decomposition chart (last year available)
            df_decomp = prepare_decomposition_data(df.copy(), country, energy_type)
            if not df_decomp.empty:
                create_decomposition_chart(df_decomp, country, energy_type, dirs)
            
            # Timeseries chart (all years)
            df_ts = prepare_timeseries_data(df.copy(), country, energy_type)
            if not df_ts.empty:
                create_timeseries_chart(df_ts, country, energy_type, dirs)
    
    # ================================================================
    # MULTI-COUNTRY: 8 cluster countries grid (rows=countries, cols=elec/gas)
    # ================================================================
    print(f"\n{'='*80}")
    print('MULTI-COUNTRY DECOMPOSITION GRID (8 EWBI cluster countries)')
    print(f"{'='*80}")

    price_components = ['NRG_SUP', 'NETC', 'TAX_FEE_LEV_CHRG', 'VAT',
                        'TAX_RNW', 'TAX_CAP', 'TAX_ENV', 'TAX_NUC', 'OTH']
    elec_bands = ['KWH_LT1000', 'KWH1000-2499', 'KWH2500-4999',
                  'KWH5000-14999', 'KWH_LE15000']
    gas_bands = ['GJ_LT20', 'GJ20-199', 'GJ_GE200']

    # Load and prepare both datasets
    data = {}
    for etype, bands in [('electricity', elec_bands), ('gas', gas_bands)]:
        df = load_energy_data(dirs, etype)
        if df.empty:
            continue
        df['nrg_prc_code'] = df['nrg_prc'].apply(map_nrg_prc)
        df['nrg_cons_code'] = df['nrg_cons'].apply(map_nrg_cons)
        df['country_code'] = df['geo'].apply(map_country_name)
        df = df[df['country_code'].isin(ALL_CLUSTER_COUNTRIES)].copy()
        df = df.dropna(subset=['nrg_prc_code', 'nrg_cons_code'])
        df = df[df['nrg_cons_code'] != 'TOT_KWH']
        df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
        last_years = df.groupby('country_code')['TIME_PERIOD'].max()
        df = df.merge(last_years.rename('last_year'), on='country_code')
        df = df[df['TIME_PERIOD'] == df['last_year']].drop(columns='last_year')
        data[etype] = (df, bands)

    # Build flat row list: (cluster_name, country_code)
    rows = []
    for cl_name, members in CLUSTER_COUNTRIES.items():
        for cc in members:
            rows.append((cl_name, cc))
    n_rows = len(rows)  # 8

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows))

    # Pre-compute shared y-axis limits per energy type (column)
    y_limits = {}
    for col_idx, (etype, _) in enumerate([('electricity', 'Electricity'),
                                           ('gas', 'Gas')]):
        if etype not in data:
            continue
        df_e, bands = data[etype]
        col_max = 0
        col_min = 0
        for _, cc in rows:
            cc_data = df_e[df_e['country_code'] == cc]
            for band in bands:
                band_data = cc_data[cc_data['nrg_cons_code'] == band]
                if band_data.empty:
                    continue
                total = band_data['OBS_VALUE'].sum()
                neg_sum = band_data.loc[band_data['OBS_VALUE'] < 0, 'OBS_VALUE'].sum()
                col_max = max(col_max, total)
                col_min = min(col_min, neg_sum)
        y_limits[col_idx] = (col_min - 0.005 if col_min < 0 else 0,
                             col_max * 1.15)

    for col_idx, (etype, col_title) in enumerate([('electricity', 'Electricity'),
                                                   ('gas', 'Gas')]):
        if etype not in data:
            continue
        df_e, bands = data[etype]

        for row_idx, (cl_name, cc) in enumerate(rows):
            ax = axes[row_idx, col_idx]
            cc_data = df_e[df_e['country_code'] == cc]

            x_pos = np.arange(len(bands))
            bar_width = 0.6
            bottom_pos = np.zeros(len(bands))
            bottom_neg = np.zeros(len(bands))
            has_any = False

            for comp in price_components:
                vals = []
                for band in bands:
                    row = cc_data[(cc_data['nrg_cons_code'] == band) &
                                  (cc_data['nrg_prc_code'] == comp)]
                    vals.append(row['OBS_VALUE'].values[0] if len(row) else 0)
                vals = np.array(vals)
                if np.all(vals == 0):
                    continue
                has_any = True
                # Stack positive on positive, negative on negative
                pos_vals = np.where(vals >= 0, vals, 0)
                neg_vals = np.where(vals < 0, vals, 0)
                if np.any(pos_vals > 0):
                    ax.bar(x_pos, pos_vals, bar_width, bottom=bottom_pos,
                           label=CATEGORY_LABELS.get(comp, comp),
                           color=COLORS.get(comp, '#cccccc'), alpha=0.85,
                           edgecolor='black', linewidth=0.5)
                    bottom_pos += pos_vals
                if np.any(neg_vals < 0):
                    ax.bar(x_pos, neg_vals, bar_width, bottom=bottom_neg,
                           color=COLORS.get(comp, '#cccccc'), alpha=0.85,
                           edgecolor='black', linewidth=0.5)
                    bottom_neg += neg_vals

            # Total labels
            for i, total in enumerate(bottom_pos):
                if total > 0:
                    ax.text(i, total + 0.001, f'{total:.3f}', ha='center',
                            va='bottom', fontsize=7, fontweight='bold')

            band_labels = [CONSUMPTION_LABELS.get(b, b) for b in bands]
            ax.set_xticks(x_pos)
            ax.set_xticklabels(band_labels, fontsize=7, rotation=20, ha='right')
            ax.grid(True, alpha=0.2, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axhline(0, color='grey', linewidth=0.5)

            # Shared y-axis per column
            if col_idx in y_limits:
                ax.set_ylim(y_limits[col_idx])

            # Row label (country + cluster color)
            cl_color = CLUSTER_COLORS[cl_name]
            ax.set_ylabel(cc, fontsize=11, fontweight='bold', color=cl_color)

            if not has_any:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='grey')

            # Column title on top row only
            if row_idx == 0:
                ax.set_title(col_title, fontsize=13, fontweight='bold',
                             color='#2c3e50')

    # Add cluster group labels on the left margin
    row_i = 0
    for cl_name, members in CLUSTER_COUNTRIES.items():
        mid_row = row_i + len(members) / 2 - 0.5
        # Shade rows for this cluster
        for r in range(len(members)):
            for c in range(2):
                axes[row_i + r, c].set_facecolor(
                    (*plt.matplotlib.colors.to_rgb(CLUSTER_COLORS[cl_name]), 0.05))
        row_i += len(members)

    # Shared legend at the bottom
    handles, labels = [], []
    for comp in price_components:
        if comp in [h.get_label() for h in handles]:
            continue
        handles.append(plt.Rectangle((0, 0), 1, 1,
                       color=COLORS.get(comp, '#cccccc'), alpha=0.85))
        labels.append(CATEGORY_LABELS.get(comp, comp))
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Energy Price Decomposition by Consumption Band\n'
                 '8 EWBI Cluster Countries (last available year)',
                 fontsize=14, fontweight='bold', color='#2c3e50', y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    fname = 'cluster8_energy_prices_grid'
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(dirs['energy_dir'], f'{fname}.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}.png / .svg')

    # --- Excel export of the cluster grid data ---
    excel_path = os.path.join(dirs['energy_dir'], 'cluster8_energy_prices_data.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for etype in ['electricity', 'gas']:
            if etype not in data:
                continue
            df_e, bands = data[etype]
            excel_rows = []
            for cl_name, cc in rows:
                cc_data = df_e[df_e['country_code'] == cc]
                for band in bands:
                    row_dict = {
                        'Cluster': cl_name,
                        'Country': cc,
                        'Consumption_band': CONSUMPTION_LABELS.get(band, band),
                        'Band_code': band,
                    }
                    for comp in price_components:
                        r = cc_data[(cc_data['nrg_cons_code'] == band) &
                                    (cc_data['nrg_prc_code'] == comp)]
                        row_dict[CATEGORY_LABELS.get(comp, comp)] = (
                            r['OBS_VALUE'].values[0] if len(r) else np.nan)
                    # Add year
                    yr = cc_data['TIME_PERIOD'].max() if len(cc_data) else np.nan
                    row_dict['Year'] = yr
                    excel_rows.append(row_dict)
            sheet = pd.DataFrame(excel_rows)
            sheet.to_excel(writer, sheet_name=etype.capitalize(), index=False)
    print(f'  Saved Excel: {excel_path}')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"Output directory: {dirs['energy_dir']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
