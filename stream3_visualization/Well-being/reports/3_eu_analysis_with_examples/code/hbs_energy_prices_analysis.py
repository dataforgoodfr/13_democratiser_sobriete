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
    'GJ200-1999': '200 - 1,999 GJ',
    'GJ2000-69999': '2,000 - 69,999 GJ',
    'GJ_GE70000': '>= 70,000 GJ',
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
        'Consumption from 200 GJ to 1 999 GJ - band D3': 'GJ200-1999',
        'Consumption from 2 000 GJ to 69 999 GJ - band D4': 'GJ2000-69999',
        'Consumption 70 000 GJ or over - band D5': 'GJ_GE70000',
    }
    return mapping.get(nrg_cons_val, None)


def map_country_name(geo_val):
    """Standardize country names."""
    mapping = {
        'France': 'FR',
        'European Union - 27 countries (from 2020)': 'EU27_2020',
        'Germany': 'DE',
        'Spain': 'ES',
        'Italy': 'IT',
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
        consumption_order = ['TOT_KWH', 'GJ_LT20', 'GJ20-199', 'GJ200-1999', 'GJ2000-69999', 'GJ_GE70000']
    
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
    
    output_file = os.path.join(dirs['energy_dir'], f'{country_code}_{energy_type.upper()}_decomposition_{year}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output_file}")
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
    
    output_file = os.path.join(dirs['energy_dir'], f'{country_code}_{energy_type.upper()}_timeseries.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output_file}")
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
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"Output directory: {dirs['energy_dir']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
