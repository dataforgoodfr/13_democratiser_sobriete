"""
HBS Housing by Urbanization - Luxembourg 2020 (Simplified Approach)

Build directly on the working components chart logic.
Simply add urbanization discrimination without changing the aggregation denominator.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from hbs_data_loader import setup_directories, load_pps_data, calculate_consumption_in_pps


def assign_simple_deciles(df):
    """Assign income deciles based on EUR_HH099 (net income)."""
    print("\n=== ASSIGNING INCOME DECILES ===")
    
    df = df.copy()
    
    # Ensure numeric columns
    df['EUR_HH099'] = pd.to_numeric(df['EUR_HH099'], errors='coerce')
    df['HA10'] = pd.to_numeric(df['HA10'], errors='coerce')
    
    # Filter valid income records
    valid = df[df['EUR_HH099'].notna()].copy()
    
    # Sort by income
    sorted_df = valid.sort_values('EUR_HH099')
    sorted_income = sorted_df['EUR_HH099'].values
    sorted_weights = sorted_df['HA10'].values
    
    # Cumulative weights
    cum_weights = np.cumsum(sorted_weights)
    cum_weights_norm = cum_weights / cum_weights[-1]
    
    # Find decile boundaries
    boundaries = []
    for d in range(1, 10):
        target = d / 10.0
        idx = np.searchsorted(cum_weights_norm, target)
        boundaries.append(sorted_income[min(idx, len(sorted_income)-1)])
    
    # Assign deciles to all records
    df['income_decile'] = pd.cut(
        df['EUR_HH099'],
        bins=[-np.inf] + boundaries + [np.inf],
        labels=['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
        duplicates='drop'
    )
    
    assigned = df['income_decile'].notna().sum()
    print(f"OK Deciles assigned to {assigned} households")
    
    return df

plt.style.use('default')
sns.set_palette("Set2")


def load_lu_2020():
    """Load HBS data for Luxembourg 2020 only."""
    print("\n=== LOADING LUXEMBOURG 2020 DATA ===")
    
    external_hbs_base = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/HBS"
    folder_2020 = os.path.join(external_hbs_base, "HBS2020/HBS2020")
    
    if not os.path.exists(folder_2020):
        print(f"ERROR: Directory not found: {folder_2020}")
        return pd.DataFrame()
    
    filepath = os.path.join(folder_2020, "HBS_HH_LU.xlsx")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(filepath, sheet_name='LU_Anonymised_HH')
        print(f"OK Loaded: {filepath}")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}")
        return pd.DataFrame()


def calculate_housing_by_urbanization(df):
    """
    Calculate housing consumption by decile AND urbanization.
    
    KEY LOGIC:
    - Filter by urbanization level
    - For each decile, calculate per-capita housing
    - Denominator = TOTAL weight for the decile (across ALL urbanization levels)
    - This ensures values are properly normalized and don't inflate
    """
    print("\n=== CALCULATING HOUSING BY DECILE AND URBANIZATION ===")
    
    urbanization_map = {
        1: 'Densely populated (≥500/km²)',
        2: 'Intermediate (100-499/km²)',
        3: 'Sparsely populated (<100/km²)'
    }
    
    has_pps = 'EUR_HE02_pps' in df.columns and df['EUR_HE02_pps'].notna().sum() > 0
    housing_col = 'EUR_HE02_pps' if has_pps else 'EUR_HE02'
    currency = 'PPS' if has_pps else 'EUR'
    
    print(f"Using column: {housing_col} ({currency})")
    
    # STEP 1: Pre-calculate TOTAL weight for each decile (across ALL urbanization)
    print("\n→ Pre-calculating total weights per decile...")
    decile_total_weights = {}
    for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
        decile_group = df[df['income_decile'] == decile].copy()
        decile_group['HB061'] = pd.to_numeric(decile_group['HB061'], errors='coerce')
        decile_group['HA10'] = pd.to_numeric(decile_group['HA10'], errors='coerce')
        
        valid_decile = decile_group[
            (decile_group['HB061'].notna()) &
            (decile_group['HB061'] > 0) &
            (decile_group['HA10'].notna())
        ]
        
        total_weight = valid_decile['HA10'].sum()
        decile_total_weights[decile] = total_weight
        print(f"  {decile}: total weight = {total_weight:,.0f}")
    
    # STEP 2: For each urbanization level, calculate housing per decile
    results = []
    
    for urban_code, urban_label in urbanization_map.items():
        print(f"\n→ Processing {urban_label}...")
        
        # Filter to just this urbanization level
        urban_group = df[df['HA09'] == urban_code].copy()
        
        if urban_group.empty:
            print(f"  WARNING: No data for urbanization code {urban_code}")
            continue
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            # Filter to this decile+urbanization combo
            decile_urban_group = urban_group[urban_group['income_decile'] == decile].copy()
            
            if decile_urban_group.empty:
                continue
            
            # Ensure numeric columns
            decile_urban_group[housing_col] = pd.to_numeric(decile_urban_group[housing_col], errors='coerce')
            decile_urban_group['HB061'] = pd.to_numeric(decile_urban_group['HB061'], errors='coerce')
            decile_urban_group['HA10'] = pd.to_numeric(decile_urban_group['HA10'], errors='coerce')
            
            # Filter to valid records
            valid_base = decile_urban_group[
                (decile_urban_group['HB061'].notna()) &
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna()) &
                (decile_urban_group[housing_col].notna()) &
                (decile_urban_group[housing_col] > 0)
            ].copy()
            
            if len(valid_base) == 0:
                continue
            
            # Calculate per-capita housing (adjusted by equivalence scale)
            valid_base['housing_ae'] = valid_base[housing_col] / valid_base['HB061']
            
            # Sum weighted values: (per-capita value * household weight)
            weighted_sum = (valid_base['housing_ae'] * valid_base['HA10']).sum()
            
            # Divide by TOTAL weight for the decile (across ALL urbanization)
            denominator = decile_total_weights[decile]
            housing_value = weighted_sum / denominator if denominator > 0 else 0
            
            results.append({
                'decile': decile,
                'urbanization': urban_label,
                'urbanization_code': urban_code,
                'housing': housing_value,
                'currency': currency,
                'sample_size': len(valid_base)
            })
            
            print(f"  {decile} @ {urban_label}: {housing_value:,.2f} {currency} (n={len(valid_base)})")
    
    result_df = pd.DataFrame(results)
    print(f"\nOK Calculated housing for {len(result_df)} decile-urbanization combinations")
    
    return result_df


def plot_housing_by_urbanization_absolute(data):
    """Create absolute housing chart by urbanization."""
    print("\n=== PLOTTING HOUSING BY URBANIZATION (ABSOLUTE) ===")
    
    pivot = data.pivot_table(index='decile', columns='urbanization', values='housing')
    
    # Ensure correct decile order
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    pivot = pivot.reindex(decile_order)
    
    # Ensure urbanization column order
    urban_cols = [
        'Densely populated (≥500/km²)',
        'Intermediate (100-499/km²)',
        'Sparsely populated (<100/km²)'
    ]
    pivot = pivot[[col for col in urban_cols if col in pivot.columns]]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind='bar', ax=ax, color=colors, width=0.8)
    
    ax.set_title('Housing Consumption by Income Decile and Urbanization Level\n(Luxembourg 2020, Absolute)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Income Decile', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Housing Consumption ({data.iloc[0]["currency"]}/adult-equivalent/year)', fontsize=12, fontweight='bold')
    ax.set_xticklabels(decile_order, rotation=0)
    ax.legend(title='Urbanization Level', fontsize=10, title_fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Format y-axis with thousand separators
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'graphs', 'HBS')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'LU_housing_by_urbanization_absolute.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output_file}")
    plt.close()


def plot_housing_by_urbanization_percentage(data):
    """Create percentage housing chart by urbanization."""
    print("\n=== PLOTTING HOUSING BY URBANIZATION (PERCENTAGE) ===")
    
    # Calculate total consumption per decile (across all urbanization)
    decile_totals = data.groupby('decile')['housing'].sum()
    
    # Calculate percentage
    data_copy = data.copy()
    data_copy['percentage'] = data_copy.apply(
        lambda row: (row['housing'] / decile_totals[row['decile']] * 100) if decile_totals[row['decile']] > 0 else 0,
        axis=1
    )
    
    pivot = data_copy.pivot_table(index='decile', columns='urbanization', values='percentage')
    
    # Ensure correct decile order
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    pivot = pivot.reindex(decile_order)
    
    # Ensure urbanization column order
    urban_cols = [
        'Densely populated (≥500/km²)',
        'Intermediate (100-499/km²)',
        'Sparsely populated (<100/km²)'
    ]
    pivot = pivot[[col for col in urban_cols if col in pivot.columns]]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind='bar', ax=ax, color=colors, width=0.8)
    
    ax.set_title('Housing Consumption Share by Income Decile and Urbanization Level\n(Luxembourg 2020, Percentage)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Income Decile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Housing Consumption (% of decile total)', fontsize=12, fontweight='bold')
    ax.set_xticklabels(decile_order, rotation=0)
    ax.legend(title='Urbanization Level', fontsize=10, title_fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Format y-axis with percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
    
    plt.tight_layout()
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'graphs', 'HBS')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'LU_housing_by_urbanization_percentage.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output_file}")
    plt.close()


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("LUXEMBOURG 2020 - HOUSING BY URBANIZATION (SIMPLIFIED APPROACH)")
    print("="*80)
    
    # Setup and load data
    dirs = setup_directories()
    
    # Load data
    df = load_lu_2020()
    if df.empty:
        print("ERROR: Failed to load data")
        return
    
    # Assign income deciles
    df = assign_simple_deciles(df)
    
    # Load PPS conversion data
    pps_df = load_pps_data(dirs)
    if pps_df is None:
        print("ERROR: Failed to load PPS data")
        return
    
    # Add PPS columns
    df = calculate_consumption_in_pps(df, pps_df)
    
    # Calculate housing by urbanization
    housing_data = calculate_housing_by_urbanization(df)
    if housing_data.empty:
        print("ERROR: No housing data calculated")
        return
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(housing_data.to_string())
    
    # Create visualizations
    plot_housing_by_urbanization_absolute(housing_data)
    plot_housing_by_urbanization_percentage(housing_data)
    
    # Save data
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'LU_housing_by_urbanization.csv')
    housing_data.to_csv(output_file, index=False)
    print(f"\nOK Saved data: {output_file}")
    
    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
