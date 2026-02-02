"""
HBS Income Comparison Diagnostic
Creates a chart comparing equivalized income by decile for France vs Belgium.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from hbs_data_loader import setup_directories, load_pps_data, calculate_consumption_in_pps
import glob

plt.style.use('default')
sns.set_palette("Set2")


def load_country_2020(country_code):
    """Load HBS data for a specific country in 2020."""
    print(f"\n=== LOADING {country_code} 2020 DATA ===")
    
    external_hbs_base = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/HBS"
    folder_2020 = os.path.join(external_hbs_base, "HBS2020/HBS2020")
    
    if not os.path.exists(folder_2020):
        print(f"ERROR: Directory not found: {folder_2020}")
        return pd.DataFrame()
    
    hh_files = glob.glob(os.path.join(folder_2020, "HBS_HH_*.xlsx"))
    country_files = [f for f in hh_files if country_code in os.path.basename(f)]
    
    if not country_files:
        print(f"ERROR: No {country_code} household files found")
        return pd.DataFrame()
    
    filepath = country_files[0]
    
    try:
        df = pd.read_excel(filepath)
        df['year'] = '2020'
        print(f"OK Loaded: {os.path.basename(filepath)}")
        print(f"  Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        return pd.DataFrame()


def assign_simple_deciles(df):
    """Assign income deciles based on EQUIVALIZED income (EUR_HH099 / HB061)."""
    df = df.copy()
    
    df['EUR_HH099'] = pd.to_numeric(df['EUR_HH099'], errors='coerce')
    df['HB061'] = pd.to_numeric(df['HB061'], errors='coerce')
    df['HA10'] = pd.to_numeric(df['HA10'], errors='coerce')
    
    valid = df[(df['EUR_HH099'].notna()) & (df['HB061'].notna()) & (df['HB061'] > 0)].copy()
    
    if len(valid) < 10:
        print(f"ERROR: Only {len(valid)} valid records")
        return df
    
    valid['equivalized_income'] = valid['EUR_HH099'] / valid['HB061']
    
    income_vals = valid['equivalized_income'].values
    weights_vals = valid['HA10'].values
    
    total_weight = weights_vals.sum()
    sorted_idx = np.argsort(income_vals)
    sorted_income = income_vals[sorted_idx]
    sorted_weights = weights_vals[sorted_idx]
    
    cum_weights = np.cumsum(sorted_weights)
    cum_weights_norm = cum_weights / cum_weights[-1]
    
    boundaries = []
    for d in range(1, 10):
        idx = np.searchsorted(cum_weights_norm, d / 10.0)
        if idx < len(sorted_income):
            boundaries.append(sorted_income[idx])
    
    df['income_decile'] = pd.cut(
        df['EUR_HH099'] / df['HB061'],
        bins=[-np.inf] + boundaries + [np.inf],
        labels=['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
        duplicates='drop'
    )
    
    return df


def plot_income_comparison_by_decile(df_fr, df_be, dirs):
    """Create diagnostic chart comparing equivalized income by decile for FR vs BE."""
    print("\n=== CREATING INCOME COMPARISON CHART (FR vs BE) ===")
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    # Calculate mean equivalized income by decile
    data_to_plot = []
    
    for country_name, df in [('France', df_fr), ('Belgium', df_be)]:
        print(f"\nCalculating income for {country_name}...")
        df_temp = df[df['income_decile'].notna()].copy()
        df_temp['EUR_HH099'] = pd.to_numeric(df_temp['EUR_HH099'], errors='coerce')
        df_temp['HB061'] = pd.to_numeric(df_temp['HB061'], errors='coerce')
        df_temp['HA10'] = pd.to_numeric(df_temp['HA10'], errors='coerce')
        
        valid = df_temp[(df_temp['EUR_HH099'].notna()) & (df_temp['HB061'].notna()) & 
                       (df_temp['HB061'] > 0) & (df_temp['HA10'].notna())].copy()
        
        print(f"  Valid records: {len(valid)}")
        
        for decile in decile_order:
            decile_data = valid[valid['income_decile'] == decile]
            if len(decile_data) > 0:
                # Calculate weighted mean equivalized income
                equivalized = decile_data['EUR_HH099'] / decile_data['HB061']
                weighted_mean = (equivalized * decile_data['HA10']).sum() / decile_data['HA10'].sum()
                data_to_plot.append({
                    'decile': decile,
                    'country': country_name,
                    'equivalized_income': weighted_mean
                })
                print(f"    {decile}: {weighted_mean:,.0f} EUR")
    
    plot_df = pd.DataFrame(data_to_plot)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Equivalized Income Comparison: France vs Belgium (2020)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Grouped bars
    fr_data = plot_df[plot_df['country'] == 'France'].set_index('decile')['equivalized_income']
    be_data = plot_df[plot_df['country'] == 'Belgium'].set_index('decile')['equivalized_income']
    
    x = np.arange(len(decile_order))
    width = 0.35
    
    ax1.bar(x - width/2, fr_data, width, label='France', color='#8dd3c7', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.bar(x + width/2, be_data, width, label='Belgium', color='#fb8072', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Equivalized Income (EUR)', fontsize=11, fontweight='bold')
    ax1.set_title('Mean Equivalized Income by Decile', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(decile_order)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000):.0f}k'))
    
    # Plot 2: Difference (BE - FR)
    diff = be_data - fr_data
    colors = ['green' if x > 0 else 'red' for x in diff]
    
    ax2.bar(decile_order, diff, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Income Difference (EUR)', fontsize=11, fontweight='bold')
    ax2.set_title('Belgium Income - France Income\n(Green = BE higher, Red = FR higher)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000):.0f}k'))
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'DIAGNOSTIC_income_comparison_FR_BE.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"\nOK Saved: {output}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("HBS INCOME COMPARISON DIAGNOSTIC - FRANCE VS BELGIUM 2020")
    print("="*80)
    
    dirs = setup_directories()
    
    # Load France
    print("\n--- PROCESSING FRANCE ---")
    df_fr = load_country_2020('FR')
    if df_fr.empty:
        print("ERROR: Could not load France data")
        return
    
    df_fr = assign_simple_deciles(df_fr)
    
    # Load Belgium
    print("\n--- PROCESSING BELGIUM ---")
    df_be = load_country_2020('BE')
    if df_be.empty:
        print("ERROR: Could not load Belgium data")
        return
    
    df_be = assign_simple_deciles(df_be)
    
    # Create income comparison diagnostic
    print("\n--- CREATING INCOME COMPARISON DIAGNOSTIC ---")
    plot_income_comparison_by_decile(df_fr, df_be, dirs)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
