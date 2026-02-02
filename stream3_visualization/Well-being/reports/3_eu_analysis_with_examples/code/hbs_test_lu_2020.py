"""
HBS Analysis - Ultra-Fast Test (Luxembourg 2020 only)

Minimal script to test visualizations with Luxembourg data for 2020 only.
Should run in seconds instead of minutes.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from hbs_data_loader import setup_directories, load_pps_data, calculate_consumption_in_pps
import glob

# Set plotting style and color palette (from app.py)
plt.style.use('default')
sns.set_palette("Set2")

# Color palette for consistent coloring across charts (from app.py)
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'
COMPONENT_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3', 
                    '#80b1d3', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']


def load_luxembourg_2020():
    """Load HBS data for Luxembourg 2020 only."""
    print("\n=== LOADING LUXEMBOURG 2020 DATA ===")
    
    external_hbs_base = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/HBS"
    folder_2020 = os.path.join(external_hbs_base, "HBS2020/HBS2020")
    
    if not os.path.exists(folder_2020):
        print(f"ERROR: Directory not found: {folder_2020}")
        return pd.DataFrame()
    
    # Find Luxembourg files
    hh_files = glob.glob(os.path.join(folder_2020, "HBS_HH_*.xlsx"))
    lu_hh_files = [f for f in hh_files if 'LU' in os.path.basename(f)]
    
    if not lu_hh_files:
        print("ERROR: No Luxembourg household files found")
        print(f"Available files: {[os.path.basename(f) for f in hh_files[:5]]}...")
        return pd.DataFrame()
    
    filepath = lu_hh_files[0]
    
    try:
        df = pd.read_excel(filepath)
        df['year'] = '2020'
        print(f"OK Loaded: {os.path.basename(filepath)}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns (first 10): {list(df.columns[:10])}")
        return df
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        return pd.DataFrame()


def assign_simple_deciles(df):
    """Assign income deciles based on EQUIVALIZED income (EUR_HH099 / HB061).
    
    METHOD: WEIGHTED QUANTILES
    1. Sort all households by EUR_HH099 / HB061 (equivalized income)
    2. Calculate cumulative sum of HA10 (weights) in sorted order
    3. Find income VALUES where 10%, 20%, ..., 90% of weighted population is BELOW
    4. Use these income values as boundaries to assign deciles
    
    Result: Each decile represents ~10% of the weighted population population-wise,
    but the CONSUMPTION aggregated by these deciles may NOT be monotonic because:
    - Households are ranked by INCOME (EUR_HH099)
    - But aggregated CONSUMPTION (EUR_HE00) may differ
    """
    print("\n=== ASSIGNING INCOME DECILES (EQUIVALIZED, WEIGHTED QUANTILE METHOD) ===")
    print("\nDECILE ASSIGNMENT LOGIC:")
    print("  1. Sort households by EUR_HH099 / HB061 (equivalized income)")
    print("  2. Calculate cumulative weighted sum using HA10")
    print("  3. Find income THRESHOLDS where cumulative weight = 10%, 20%, ..., 90%")
    print("  4. Assign deciles based on these income thresholds")
    print("  → Each decile represents ~10% of weighted population")
    print("  → But consumption may NOT be monotonic by these deciles")
    
    df = df.copy()
    
    # Ensure numeric columns
    df['EUR_HH099'] = pd.to_numeric(df['EUR_HH099'], errors='coerce')
    df['HB061'] = pd.to_numeric(df['HB061'], errors='coerce')
    df['HA10'] = pd.to_numeric(df['HA10'], errors='coerce')
    
    # Filter valid income and equivalence scale records
    valid = df[(df['EUR_HH099'].notna()) & (df['HB061'].notna()) & (df['HB061'] > 0)].copy()
    
    print(f"\nValid records with income and HB061: {len(valid)}/{len(df)}")
    
    if len(valid) < 10:
        print(f"ERROR: Only {len(valid)} valid records")
        return df
    
    # Calculate equivalized income
    valid['equivalized_income'] = valid['EUR_HH099'] / valid['HB061']
    
    # Calculate weighted quantiles on EQUIVALIZED income
    income_vals = valid['equivalized_income'].values
    weights_vals = valid['HA10'].values
    
    total_weight = weights_vals.sum()
    print(f"Total weighted population: {total_weight:,.0f}")
    
    # Sort by equivalized income
    sorted_idx = np.argsort(income_vals)
    sorted_income = income_vals[sorted_idx]
    sorted_weights = weights_vals[sorted_idx]
    
    # Cumulative weights
    cum_weights = np.cumsum(sorted_weights)
    cum_weights_norm = cum_weights / cum_weights[-1]
    
    # Find decile boundaries
    boundaries = []
    print(f"\nDecile boundaries (income thresholds where cumulative weight crosses 10%, 20%, ... 90%):")
    for d in range(1, 10):
        target = d / 10.0
        idx = np.searchsorted(cum_weights_norm, target)
        boundary_income = sorted_income[min(idx, len(sorted_income)-1)]
        boundaries.append(boundary_income)
        print(f"  D{d} threshold (↑ {d*10}% of population): EUR_HH099/HB061 = {boundary_income:,.0f} EUR/AE")
    
    # Assign deciles to all records using equivalized income
    df['income_decile'] = pd.cut(
        df['EUR_HH099'] / df['HB061'],  # Apply equivalization to all records
        bins=[-np.inf] + boundaries + [np.inf],
        labels=['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
        duplicates='drop'
    )
    
    assigned = df['income_decile'].notna().sum()
    print(f"\nOK Deciles assigned to {assigned} households")
    
    return df


def calculate_consumption_deciles(df):
    """Calculate mean consumption by income decile with PPS fallback."""
    print("\n=== CALCULATING CONSUMPTION BY DECILE ===")
    
    # Check required columns
    print(f"Checking columns:")
    print(f"  EUR_HE00: {'EUR_HE00' in df.columns}")
    print(f"  EUR_HE00_pps: {'EUR_HE00_pps' in df.columns}")
    print(f"  HB061: {'HB061' in df.columns}")
    print(f"  HA10: {'HA10' in df.columns}")
    print(f"  income_decile: {'income_decile' in df.columns}")
    
    results = []
    
    # Ensure columns are numeric
    df['EUR_HE00'] = pd.to_numeric(df['EUR_HE00'], errors='coerce')
    df['EUR_HE00_pps'] = pd.to_numeric(df['EUR_HE00_pps'], errors='coerce')
    df['HB061'] = pd.to_numeric(df['HB061'], errors='coerce')
    df['HA10'] = pd.to_numeric(df['HA10'], errors='coerce')
    
    # Use PPS if available, otherwise use nominal
    consumption_col = 'EUR_HE00_pps' if df['EUR_HE00_pps'].notna().sum() > 0 else 'EUR_HE00'
    currency = 'PPS' if consumption_col == 'EUR_HE00_pps' else 'EUR'
    print(f"Using consumption column: {consumption_col} ({currency})")
    print(f"Non-null {consumption_col}: {df[consumption_col].notna().sum()}")
    print(f"Non-null HB061: {df['HB061'].notna().sum()}")
    print(f"Non-null income_decile: {df['income_decile'].notna().sum()}")
    
    # Calculate OVERALL MEAN directly from all households (for diagnostic)
    df_valid_overall = df[
        (df[consumption_col].notna()) &
        (df['HB061'].notna()) &
        (df['HA10'].notna())
        # NOTE: Removed filters for > 0 to test if that's the issue
    ].copy()
    
    if len(df_valid_overall) > 0:
        # Filter out zero HB061 before division
        df_valid_overall = df_valid_overall[df_valid_overall['HB061'] != 0].copy()
        
        # Calculate diagnostics
        mean_total_consumption = np.average(df_valid_overall['EUR_HE00'], weights=df_valid_overall['HA10'])
        mean_hb061 = np.average(df_valid_overall['HB061'], weights=df_valid_overall['HA10'])
        
        df_valid_overall['cons_ae'] = df_valid_overall[consumption_col] / df_valid_overall['HB061']
        overall_mean_direct = np.average(df_valid_overall['cons_ae'], weights=df_valid_overall['HA10'])
        
        print(f"\n=== DIAGNOSTIC: Consumption Analysis ===")
        print(f"Mean total consumption (EUR_HE00, not equivalized): {mean_total_consumption:,.0f} EUR")
        print(f"Mean equivalence scale (HB061): {mean_hb061:.3f}")
        print(f"Mean consumption per adult equivalent (EUR_HE00 / HB061): {mean_total_consumption / mean_hb061:,.0f} EUR")
        print(f"Overall mean (from {consumption_col}): {overall_mean_direct:,.0f} {currency}")
        print(f"Number of households: {len(df_valid_overall):,}")
        print(f"Number of valid households with income_decile: {len(df[df['income_decile'].notna()])}")
    
    # Group by decile
    for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
        group = df[df['income_decile'] == decile]
        
        if group.empty:
            continue
        
        # Filter valid records
        valid = group[
            (group[consumption_col].notna()) &
            (group['HB061'].notna()) &
            (group['HA10'].notna()) &
            (group[consumption_col] > 0) &
            (group['HB061'] > 0)
        ].copy()
        
        if len(valid) > 0:
            # Per adult equivalent consumption
            valid['cons_ae'] = valid[consumption_col] / valid['HB061']
            
            # Weighted mean
            wmean = np.average(valid['cons_ae'], weights=valid['HA10'])
            
            # Also store the sum of weights for proper overall mean calculation
            total_weight = valid['HA10'].sum()
            
            results.append({
                'decile': decile,
                'mean_consumption': wmean,
                'households': len(valid),
                'total_weight': total_weight,
                'currency': currency
            })
    
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        print(f"OK Calculated {len(result_df)} decile values")
        print(f"\nConsumption by decile:")
        for _, row in result_df.iterrows():
            print(f"  {row['decile']}: {row['currency']} {row['mean_consumption']:,.0f} ({row['households']} households)")
    else:
        print("ERROR: No consumption data calculated")
    
    return result_df


def calculate_consumption_components(df):
    """Calculate consumption by component and income decile."""
    print("\n=== CALCULATING CONSUMPTION COMPONENTS BY DECILE ===")
    
    # Consumption component mapping (HBS standard categories)
    components = {
        'EUR_HE01': 'Food & Beverage',
        'EUR_HE02': 'Alcohol & Tobacco',
        'EUR_HE03': 'Clothing & Footwear',
        'EUR_HE04': 'Housing & Utilities',
        'EUR_HE05': 'Furnishings',
        'EUR_HE06': 'Health',
        'EUR_HE07': 'Transport',
        'EUR_HE08': 'Communication',
        'EUR_HE09': 'Recreation & Culture',
        'EUR_HE10': 'Education',
        'EUR_HE11': 'Restaurants & Hotels',
        'EUR_HE12': 'Miscellaneous Goods & Services',
        'EUR_HJ08': 'Financial Services',
        'EUR_HJ90': 'Other Goods & Services'
    }
    
    # Discover all EUR_HE and EUR_HJ columns in the data
    all_eur_cols = [col for col in df.columns if col.startswith('EUR_HE') or col.startswith('EUR_HJ')]
    all_eur_cols = [col for col in all_eur_cols if not col.endswith('_pps')]  # exclude PPS versions for now
    
    # Filter to only top-level categories (HE01-HE12, HJ08, HJ90)
    top_level_cols = [col for col in all_eur_cols if col in [
        'EUR_HE01', 'EUR_HE02', 'EUR_HE03', 'EUR_HE04', 'EUR_HE05', 'EUR_HE06',
        'EUR_HE07', 'EUR_HE08', 'EUR_HE09', 'EUR_HE10', 'EUR_HE11', 'EUR_HE12',
        'EUR_HJ08', 'EUR_HJ90'
    ]]
    
    print(f"Found {len(top_level_cols)} top-level EUR components: {sorted(top_level_cols)}")
    
    # Add any top-level columns not already in components
    for col in top_level_cols:
        if col not in components:
            components[col] = col  # Use column name as label if not mapped
    
    # Check which columns exist
    available_components = {k: v for k, v in components.items() if k in df.columns}
    print(f"Using {len(available_components)} consumption components")
    
    # Determine which consumption version to use (PPS or EUR)
    has_pps = 'EUR_HE00_pps' in df.columns and df['EUR_HE00_pps'].notna().sum() > 0
    suffix = '_pps' if has_pps else ''
    currency = 'PPS' if has_pps else 'EUR'
    
    # Prepare component columns with proper suffix
    component_cols = {}
    for code, name in available_components.items():
        col_name = f'{code}{suffix}'
        if col_name in df.columns or code in df.columns:
            actual_col = col_name if col_name in df.columns else code
            component_cols[name] = actual_col
    
    # Ensure required columns are numeric
    for col in component_cols.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['HB061'] = pd.to_numeric(df['HB061'], errors='coerce')
    df['HA10'] = pd.to_numeric(df['HA10'], errors='coerce')
    
    # Calculate by decile
    results = []
    
    for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
        group = df[df['income_decile'] == decile]
        
        if group.empty:
            continue
        
        decile_data = {'decile': decile, 'currency': currency}
        
        for comp_name, comp_col in component_cols.items():
            if comp_col not in group.columns:
                continue
            
            # Filter valid records
            valid = group[
                (group[comp_col].notna()) &
                (group['HB061'].notna()) &
                (group['HA10'].notna()) &
                (group[comp_col] > 0) &
                (group['HB061'] > 0)
            ].copy()
            
            if len(valid) > 0:
                # Per adult equivalent
                valid['comp_ae'] = valid[comp_col] / valid['HB061']
                # Weighted mean
                wmean = np.average(valid['comp_ae'], weights=valid['HA10'])
                decile_data[comp_name] = wmean
            else:
                decile_data[comp_name] = 0
        
        # Add "Other Components" (residual)
        # Formula: SUM((EUR_HE00 - EUR_HE01 - ... EUR_HE12 - EUR_HJ08 - EUR_HJ90) / HB061 * HA10) / SUM(HA10)
        total_col = f'EUR_HE00{suffix}'
        if total_col in group.columns:
            valid_total = group[
                (group[total_col].notna()) &
                (group['HB061'].notna()) &
                (group['HB061'] > 0) &
                (group['HA10'].notna())
            ].copy()
            
            if len(valid_total) > 0:
                # Start with total
                residual_raw = valid_total[total_col].copy()
                
                # Subtract all identified components
                for comp_name, comp_col in component_cols.items():
                    if comp_col in valid_total.columns:
                        residual_raw = residual_raw - valid_total[comp_col].fillna(0)
                
                # Per adult equivalent
                residual_ae = residual_raw / valid_total['HB061']
                
                # Weighted mean
                wmean = np.average(residual_ae, weights=valid_total['HA10'])
                decile_data['Other Components'] = wmean
            else:
                decile_data['Other Components'] = 0
        
        results.append(decile_data)
    
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        print(f"OK Calculated {len(available_components)} components for {len(result_df)} deciles")
        
        # Add diagnostic output comparing totals
        print("\nDiagnostic: Component sums vs total consumption:")
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_row = result_df[result_df['decile'] == decile]
            if not decile_row.empty:
                component_sum = decile_row[[col for col in decile_row.columns if col not in ['decile', 'currency']]].sum(axis=1).values[0]
                print(f"  {decile}: Sum of components = {int(component_sum):,} PPS")
    
    return result_df


def plot_luxembourg_consumption(consumption_df, dirs):
    """Create visualization."""
    print("\n=== CREATING PLOTS ===")
    
    if consumption_df.empty:
        print("ERROR: No data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Sort by decile
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    consumption_df['decile'] = pd.Categorical(consumption_df['decile'], categories=decile_order, ordered=True)
    consumption_df = consumption_df.sort_values('decile')
    
    # Calculate weighted overall mean using sample weights (HA10), not household count
    overall_mean = (consumption_df['mean_consumption'] * consumption_df['total_weight']).sum() / consumption_df['total_weight'].sum()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars with EU color
    bars = ax.bar(consumption_df['decile'], consumption_df['mean_consumption'], 
                   color=EU_27_COLOR, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add overall mean line
    ax.axhline(y=overall_mean, color=COUNTRY_COLORS[0], linestyle='--', linewidth=2.5, alpha=0.9, label=f'Overall Mean: {int(overall_mean):,} PPS')
    
    # Add text label on the right side
    y_max = consumption_df['mean_consumption'].max()
    y_pos = overall_mean + (y_max - overall_mean) * 0.05  # Position slightly above the line
    ax.text(9.7, y_pos, f'Mean\n{int(overall_mean):,} PPS', 
            ha='right', va='bottom', fontsize=11, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COUNTRY_COLORS[0], alpha=0.2, edgecolor=COUNTRY_COLORS[0], linewidth=1.5))
    
    # Formatting
    ax.set_xlabel('Income Decile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Consumption per Adult Equivalent (PPS)', fontsize=12, fontweight='bold')
    ax.set_title('Luxembourg 2020: Mean Consumption Expenditure by Income Decile', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_consumption_2020_decile.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_consumption_components_stacked(components_df, dirs):
    """Create stacked bar chart showing consumption components by decile."""
    print("\n=== CREATING STACKED BAR CHART ===")
    
    if components_df.empty:
        print("ERROR: No component data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Sort by decile
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    components_df['decile'] = pd.Categorical(components_df['decile'], categories=decile_order, ordered=True)
    components_df = components_df.sort_values('decile')
    
    # Get currency and component columns (exclude decile and currency)
    currency = components_df['currency'].iloc[0] if 'currency' in components_df.columns else 'PPS'
    component_cols = [col for col in components_df.columns if col not in ['decile', 'currency']]
    
    print(f"OK Plotting {len(component_cols)} components: {', '.join(component_cols[:5])}...")
    
    # Create figure with larger size for legend
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Define color palette for components (from app.py palette extended)
    # Use predefined colors for consistent styling
    colors = COMPONENT_COLORS[:len(component_cols)] if len(component_cols) <= len(COMPONENT_COLORS) else COMPONENT_COLORS + ['#cccccc'] * (len(component_cols) - len(COMPONENT_COLORS))
    
    # Create stacked bar chart
    components_df_sorted = components_df.sort_values('decile')
    
    x_pos = np.arange(len(components_df_sorted))
    bottom = np.zeros(len(components_df_sorted))
    
    for idx, component in enumerate(component_cols):
        if component in components_df_sorted.columns:
            values = components_df_sorted[component].fillna(0).values
            ax.bar(x_pos, values, bottom=bottom, label=component, color=colors[idx], 
                   edgecolor='white', linewidth=0.5, alpha=0.85)
            bottom += values
    
    # Formatting
    ax.set_xlabel('Income Decile', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Mean Consumption per Adult Equivalent ({currency})', fontsize=12, fontweight='bold')
    ax.set_title('Luxembourg 2020: Consumption Components by Income Decile (Stacked)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(components_df_sorted['decile'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add total values on top of bars
    for i, decile in enumerate(components_df_sorted['decile']):
        total = bottom[i]
        ax.text(i, total, f'{int(total):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=1, bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_consumption_2020_components_stacked.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def calculate_housing_components_by_decile(df):
    """Calculate housing sub-components (EUR_HE041-HE045) by income decile.
    
    Uses consistent denominator: sum of weights (HA10) for the entire decile,
    applied to all components regardless of missing values.
    """
    print("\n=== CALCULATING HOUSING COMPONENTS BY DECILE ===")
    
    housing_components = {
        'EUR_HE041': 'Actual Rentals',
        'EUR_HE042': 'Imputed Rentals',
        'EUR_HE043': 'Maintenance & Repair',
        'EUR_HE044': 'Water & Services',
        'EUR_HE045': 'Electricity, Gas & Fuels'
    }
    
    # Determine if using PPS or EUR
    has_pps = 'EUR_HE041_pps' in df.columns and df['EUR_HE041_pps'].notna().sum() > 0
    suffix = '_pps' if has_pps else ''
    currency = 'PPS' if has_pps else 'EUR'
    
    # Prepare component columns with proper suffix
    component_cols = {}
    for code, name in housing_components.items():
        col_name = f'{code}{suffix}'
        if col_name not in df.columns:
            col_name = code
        component_cols[name] = col_name
    
    results = []
    
    for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
        group = df[df['income_decile'] == decile].copy()
        if group.empty:
            continue
        
        decile_data = {'decile': decile, 'currency': currency}
        
        # Ensure numeric columns
        for col in component_cols.values():
            if col in group.columns:
                group[col] = pd.to_numeric(group[col], errors='coerce')
        group['HB061'] = pd.to_numeric(group['HB061'], errors='coerce')
        group['HA10'] = pd.to_numeric(group['HA10'], errors='coerce')
        
        # Calculate denominator: sum of weights for valid records in this decile
        valid_base = group[
            (group['HB061'].notna()) &
            (group['HB061'] > 0) &
            (group['HA10'].notna())
        ].copy()
        
        if len(valid_base) == 0:
            continue
        
        denominator = valid_base['HA10'].sum()
        
        # Calculate each component using the SAME denominator
        for comp_name, col_name in component_cols.items():
            if col_name in group.columns:
                # For this component, include all valid households (treating missing/zero values as 0)
                valid_comp = valid_base.copy()
                
                # Fill missing values with 0 and calculate per adult equivalent
                valid_comp['comp_ae'] = valid_comp[col_name].fillna(0) / valid_comp['HB061']
                # Weighted sum of component per AE
                weighted_sum = (valid_comp['comp_ae'] * valid_comp['HA10']).sum()
                # Divide by the decile's total denominator (consistent across all components)
                component_value = weighted_sum / denominator
                decile_data[comp_name] = component_value
            else:
                decile_data[comp_name] = 0
        
        results.append(decile_data)
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated housing components for {len(result_df)} deciles")
    
    return result_df


def plot_housing_components_by_decile(housing_df, dirs):
    """Create stacked bar chart for housing sub-components by decile."""
    print("\n=== CREATING HOUSING COMPONENTS CHART ===")
    
    if housing_df.empty:
        print("ERROR: No housing data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Sort by decile
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    housing_df['decile'] = pd.Categorical(housing_df['decile'], categories=decile_order, ordered=True)
    housing_df = housing_df.sort_values('decile')
    
    currency = housing_df['currency'].iloc[0] if 'currency' in housing_df.columns else 'PPS'
    component_cols = [col for col in housing_df.columns if col not in ['decile', 'currency']]
    
    print(f"OK Plotting {len(component_cols)} housing components: {', '.join(component_cols)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use housing-specific colors (use first 5 from palette)
    housing_colors = COMPONENT_COLORS[:5]
    
    # Create stacked bar chart
    x_pos = np.arange(len(housing_df))
    bottom = np.zeros(len(housing_df))
    
    for idx, component in enumerate(component_cols):
        if component in housing_df.columns:
            values = housing_df[component].fillna(0).values
            ax.bar(x_pos, values, bottom=bottom, label=component, color=housing_colors[idx % len(housing_colors)],
                   edgecolor='white', linewidth=0.5, alpha=0.85)
            bottom += values
    
    # Formatting
    ax.set_xlabel('Income Decile', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Mean Housing Expenditure per Adult Equivalent ({currency})', fontsize=12, fontweight='bold')
    ax.set_title('Luxembourg 2020: Housing Expenditure Components by Income Decile', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(housing_df['decile'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add total values on top of bars
    for i, decile in enumerate(housing_df['decile']):
        total = bottom[i]
        if total > 0:
            ax.text(i, total, f'{int(total):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_housing_2020_components.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def calculate_housing_burden(df):
    """Calculate share of bottom decile spending >40% of disposable income on housing."""
    print("\n=== CALCULATING HOUSING BURDEN ===")
    
    # Housing costs = EUR_HE041 (Actual rentals) + EUR_HE042 (Imputed rentals)
    df_valid = df[df['income_decile'] == 'D1'].copy()
    
    # Ensure columns are numeric
    df_valid['EUR_HE041'] = pd.to_numeric(df_valid['EUR_HE041'], errors='coerce')
    df_valid['EUR_HE042'] = pd.to_numeric(df_valid['EUR_HE042'], errors='coerce')
    df_valid['EUR_HH099'] = pd.to_numeric(df_valid['EUR_HH099'], errors='coerce')
    df_valid['HB061'] = pd.to_numeric(df_valid['HB061'], errors='coerce')
    df_valid['HA10'] = pd.to_numeric(df_valid['HA10'], errors='coerce')
    
    # Calculate housing burden
    df_valid['housing_cost'] = (df_valid['EUR_HE041'] + df_valid['EUR_HE042'])
    df_valid['disposable_income'] = df_valid['EUR_HH099']
    df_valid['housing_burden_ratio'] = df_valid['housing_cost'] / df_valid['disposable_income']
    
    # Filter valid records
    valid = df_valid[
        (df_valid['housing_cost'].notna()) &
        (df_valid['disposable_income'].notna()) &
        (df_valid['HA10'].notna()) &
        (df_valid['disposable_income'] > 0)
    ].copy()
    
    if len(valid) == 0:
        print("ERROR: No valid housing burden data")
        return None
    
    # Count households with >40% housing burden
    high_burden = valid[valid['housing_burden_ratio'] > 0.40]
    
    # Calculate weighted share
    high_burden_weight = high_burden['HA10'].sum()
    total_weight = valid['HA10'].sum()
    share_high_burden = (high_burden_weight / total_weight * 100) if total_weight > 0 else 0
    
    # Mean housing burden
    mean_burden = np.average(valid['housing_burden_ratio'], weights=valid['HA10']) * 100
    
    print(f"OK Bottom decile analysis:")
    print(f"  Total households (D1): {len(valid)}")
    print(f"  Households with >40% housing burden: {len(high_burden)} ({share_high_burden:.1f}%)")
    print(f"  Mean housing burden ratio: {mean_burden:.1f}%")
    
    return {
        'share_high_burden': share_high_burden,
        'mean_burden': mean_burden,
        'n_households': len(valid),
        'n_high_burden': len(high_burden)
    }


def plot_housing_burden(burden_data, dirs):
    """Create visualization for housing burden in bottom decile."""
    print("\n=== CREATING HOUSING BURDEN CHART ===")
    
    if not burden_data:
        print("ERROR: No housing burden data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Share with >40% burden
    categories = ['≤40% of income', '>40% of income']
    share_values = [100 - burden_data['share_high_burden'], burden_data['share_high_burden']]
    colors_burden = [COUNTRY_COLORS[2], COUNTRY_COLORS[1]]  # green and red
    
    bars1 = ax1.bar(categories, share_values, color=colors_burden, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Share of Population (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Bottom Decile (D1): Share Spending on Housing\n(Mortgage + Rent)', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, share_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right plot: Summary statistics
    ax2.axis('off')
    summary_text = f"""
HOUSING BURDEN ANALYSIS - LUXEMBOURG 2020
Bottom Decile (D1)

Total Households: {burden_data['n_households']}

Housing Cost Definition:
  EUR_HE041: Actual rentals for housing
  EUR_HE042: Imputed rentals for housing
  Total: EUR_HE041 + EUR_HE042

Key Findings:
  Share with >40% burden: {burden_data['share_high_burden']:.1f}%
  Mean housing burden: {burden_data['mean_burden']:.1f}%

Housing cost burden is measured as
the ratio of total housing costs
(rent/mortgage) to disposable income.
    """
    
    ax2.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_housing_burden_2020.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def calculate_housing_by_decile_urbanization(df):
    """Calculate housing components by income decile and degree of urbanization.
    
    Each decile+urbanization combo is filtered separately (distinct group),
    but uses the TOTAL decile weight as denominator to keep values consistent.
    Denominator = SUM(Weight for ENTIRE decile, across all urbanization levels).
    
    CRITICAL: Must require consumption_col.notna() to match total_consumption denominator!
    """
    print("\n=== CALCULATING HOUSING COMPONENTS BY DECILE AND URBANIZATION ===")
    
    housing_components = {
        'EUR_HE041': 'Actual Rentals',
        'EUR_HE042': 'Imputed Rentals',
        'EUR_HE043_044_045': 'Utility bills'  # Combined: Maintenance, Water, Electricity
    }
    
    urbanization_map = {
        1: 'Densely populated (≥500/km²)',
        2: 'Intermediate (100-499/km²)',
        3: 'Sparsely populated (<100/km²)'
    }
    
    has_pps = 'EUR_HE041_pps' in df.columns and df['EUR_HE041_pps'].notna().sum() > 0
    suffix = '_pps' if has_pps else ''
    currency = 'PPS' if has_pps else 'EUR'
    
    # Use total consumption to determine valid households (for consistent denominators with total)
    consumption_col = 'EUR_HE00_pps' if has_pps else 'EUR_HE00'
    
    results = []
    
    for urban_code, urban_label in urbanization_map.items():
        urban_group = df[df['HA09'] == urban_code]
        
        if urban_group.empty:
            print(f"  WARNING: No data for urbanization {urban_code}")
            continue
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_urban_group = urban_group[urban_group['income_decile'] == decile].copy()
            
            if decile_urban_group.empty:
                continue
            
            decile_data = {
                'decile': decile,
                'urbanization': urban_label,
                'urbanization_code': urban_code,
                'currency': currency
            }
            
            # Ensure numeric columns
            for code in ['EUR_HE041', 'EUR_HE042', 'EUR_HE043', 'EUR_HE044', 'EUR_HE045']:
                col_name = f'{code}{suffix}'
                if col_name not in df.columns:
                    col_name = code
                if col_name in decile_urban_group.columns:
                    decile_urban_group[col_name] = pd.to_numeric(decile_urban_group[col_name], errors='coerce')
            
            decile_urban_group['HB061'] = pd.to_numeric(decile_urban_group['HB061'], errors='coerce')
            decile_urban_group['HA10'] = pd.to_numeric(decile_urban_group['HA10'], errors='coerce')
            decile_urban_group[consumption_col] = pd.to_numeric(decile_urban_group[consumption_col], errors='coerce')
            
            # Calculate denominator for THIS decile+urbanization combo
            # CRITICAL: Use same filter as total consumption for consistent percentages!
            valid_base = decile_urban_group[
                (decile_urban_group[consumption_col].notna()) &
                (decile_urban_group['HB061'].notna()) &
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna())
            ].copy()
            
            if len(valid_base) == 0:
                continue
            
            denominator = valid_base['HA10'].sum()
            
            for code, name in housing_components.items():
                if name == 'Utility bills':
                    # Sum EUR_HE043 + EUR_HE044 + EUR_HE045 per household
                    utility_cols_raw = ['EUR_HE043', 'EUR_HE044', 'EUR_HE045']
                    utility_cols = []
                    for col in utility_cols_raw:
                        col_name = f'{col}{suffix}'
                        if col_name not in df.columns:
                            col_name = col
                        if col_name in valid_base.columns:
                            utility_cols.append(col_name)
                    
                    if utility_cols:
                        utility_sum = valid_base[utility_cols].fillna(0).sum(axis=1)
                        util_ae = utility_sum / valid_base['HB061']
                        weighted_sum = (util_ae * valid_base['HA10']).sum()
                        util_value = weighted_sum / denominator if denominator > 0 else 0
                        decile_data[name] = util_value
                    else:
                        decile_data[name] = 0
                else:
                    col_name = f'{code}{suffix}'
                    if col_name not in df.columns:
                        col_name = code
                    
                    if col_name in valid_base.columns:
                        # Include all households, treating missing/zero values as 0
                        comp_ae = valid_base[col_name].fillna(0) / valid_base['HB061']
                        weighted_sum = (comp_ae * valid_base['HA10']).sum()
                        comp_value = weighted_sum / denominator if denominator > 0 else 0
                        decile_data[name] = comp_value
                    else:
                        decile_data[name] = 0
            
            results.append(decile_data)
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated housing components for {len(result_df)} decile-urbanization combinations")
    
    return result_df


def calculate_transport_by_decile_urbanization(df):
    """Calculate transport components by income decile and degree of urbanization.
    
    Each decile+urbanization combo is a DISTINCT group with its own denominator.
    Denominator = SUM(Weight for that specific decile+urbanization combination).
    
    CRITICAL: Must require consumption_col.notna() to match total_consumption denominator!
    """
    print("\n=== CALCULATING TRANSPORT COMPONENTS BY DECILE AND URBANIZATION ===")
    
    transport_components = {
        'EUR_HE071': 'Purchase of Vehicles',
        'EUR_HE072': 'Operation of Personal Transport',
        'EUR_HE073': 'Transport Services'
    }
    
    urbanization_map = {
        1: 'Densely populated (≥500/km²)',
        2: 'Intermediate (100-499/km²)',
        3: 'Sparsely populated (<100/km²)'
    }
    
    has_pps = 'EUR_HE071_pps' in df.columns and df['EUR_HE071_pps'].notna().sum() > 0
    suffix = '_pps' if has_pps else ''
    currency = 'PPS' if has_pps else 'EUR'
    
    # Use total consumption to determine valid households (for consistent denominators with total)
    consumption_col = 'EUR_HE00_pps' if has_pps else 'EUR_HE00'
    
    results = []
    
    for urban_code, urban_label in urbanization_map.items():
        urban_group = df[df['HA09'] == urban_code]
        
        if urban_group.empty:
            print(f"  WARNING: No data for urbanization {urban_code}")
            continue
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_urban_group = urban_group[urban_group['income_decile'] == decile].copy()
            
            if decile_urban_group.empty:
                continue
            
            decile_data = {
                'decile': decile,
                'urbanization': urban_label,
                'urbanization_code': urban_code,
                'currency': currency
            }
            
            # Ensure numeric columns
            for code in transport_components.keys():
                col_name = f'{code}{suffix}'
                if col_name not in df.columns:
                    col_name = code
                if col_name in decile_urban_group.columns:
                    decile_urban_group[col_name] = pd.to_numeric(decile_urban_group[col_name], errors='coerce')
            
            decile_urban_group['HB061'] = pd.to_numeric(decile_urban_group['HB061'], errors='coerce')
            decile_urban_group['HA10'] = pd.to_numeric(decile_urban_group['HA10'], errors='coerce')
            decile_urban_group[consumption_col] = pd.to_numeric(decile_urban_group[consumption_col], errors='coerce')
            
            # Calculate denominator for THIS decile+urbanization combo
            # CRITICAL: Use same filter as total consumption for consistent percentages!
            valid_base = decile_urban_group[
                (decile_urban_group[consumption_col].notna()) &
                (decile_urban_group['HB061'].notna()) &
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna())
            ].copy()
            
            if len(valid_base) == 0:
                continue
            
            denominator = valid_base['HA10'].sum()
            
            for code, name in transport_components.items():
                col_name = f'{code}{suffix}'
                if col_name not in df.columns:
                    col_name = code
                
                if col_name in valid_base.columns:
                    # Include all households, treating missing/zero values as 0
                    comp_ae = valid_base[col_name].fillna(0) / valid_base['HB061']
                    weighted_sum = (comp_ae * valid_base['HA10']).sum()
                    comp_value = weighted_sum / denominator if denominator > 0 else 0
                    decile_data[name] = comp_value
                else:
                    decile_data[name] = 0
            
            results.append(decile_data)
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated transport components for {len(result_df)} decile-urbanization combinations")
    
    return result_df


def calculate_total_consumption_by_decile_urbanization(df):
    """Calculate total EUR_HE00_pps (in PPS) per adult equivalent by decile and urbanization.
    
    Numerator: SUM(((EUR_HE00_pps / HB061) * HA10) for that decile+urbanization combo
    Denominator: SUM(HA10) for that SAME decile+urbanization combo
    
    CRITICAL: Must divide by HB061 to match component calculations for correct percentage!
    """
    print("\n=== CALCULATING TOTAL EXPENDITURE (EUR_HE00_PPS) PER ADULT EQUIVALENT BY DECILE AND URBANIZATION ===")
    
    consumption_col = 'EUR_HE00_pps'  # Use PPS version for total (must match components)
    currency = 'PPS'
    
    urbanization_map = {
        1: 'Densely populated (≥500/km²)',
        2: 'Intermediate (100-499/km²)',
        3: 'Sparsely populated (<100/km²)'
    }
    
    results = []
    
    for urban_code, urban_label in urbanization_map.items():
        urban_group = df[df['HA09'] == urban_code]
        
        if urban_group.empty:
            continue
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_urban_group = urban_group[urban_group['income_decile'] == decile].copy()
            
            if decile_urban_group.empty:
                continue
            
            # Ensure numeric columns
            decile_urban_group[consumption_col] = pd.to_numeric(decile_urban_group[consumption_col], errors='coerce')
            decile_urban_group['HB061'] = pd.to_numeric(decile_urban_group['HB061'], errors='coerce')
            decile_urban_group['HA10'] = pd.to_numeric(decile_urban_group['HA10'], errors='coerce')
            
            # Calculate denominator for THIS decile+urbanization combo
            valid_base = decile_urban_group[
                (decile_urban_group[consumption_col].notna()) &
                (decile_urban_group['HB061'].notna()) &
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna())
            ].copy()
            
            if len(valid_base) == 0:
                continue
            
            denominator = valid_base['HA10'].sum()
            
            # Per adult equivalent consumption
            consumption_ae = valid_base[consumption_col] / valid_base['HB061']
            
            # Weighted average (per adult equivalent)
            weighted_sum = (consumption_ae * valid_base['HA10']).sum()
            wmean = weighted_sum / denominator if denominator > 0 else 0
            
            results.append({
                'decile': decile,
                'urbanization': urban_label,
                'urbanization_code': urban_code,
                'total_consumption': wmean,
                'currency': currency
            })
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated total consumption for {len(result_df)} decile-urbanization combinations")
    
    return result_df


def calculate_total_consumption_raw_by_decile_urbanization(df):
    """Calculate total EUR_HE00 (raw, not PPS) per adult equivalent by decile and urbanization.
    
    Diagnostic function to verify urbanization breakdown.
    
    Numerator: SUM((EUR_HE00 / HB061) * HA10) for that decile+urbanization combo
    Denominator: SUM(HA10) for that SAME decile+urbanization combo
    """
    print("\n=== CALCULATING TOTAL RAW CONSUMPTION (EUR_HE00) BY DECILE AND URBANIZATION ===")
    
    consumption_col = 'EUR_HE00'  # Use raw EUR, not PPS
    currency = 'EUR'
    
    urbanization_map = {
        1: 'Densely populated (≥500/km²)',
        2: 'Intermediate (100-499/km²)',
        3: 'Sparsely populated (<100/km²)'
    }
    
    results = []
    
    for urban_code, urban_label in urbanization_map.items():
        urban_group = df[df['HA09'] == urban_code]
        
        if urban_group.empty:
            continue
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_urban_group = urban_group[urban_group['income_decile'] == decile].copy()
            
            if decile_urban_group.empty:
                continue
            
            # Ensure numeric columns
            decile_urban_group[consumption_col] = pd.to_numeric(decile_urban_group[consumption_col], errors='coerce')
            decile_urban_group['HB061'] = pd.to_numeric(decile_urban_group['HB061'], errors='coerce')
            decile_urban_group['HA10'] = pd.to_numeric(decile_urban_group['HA10'], errors='coerce')
            
            # Calculate denominator for THIS decile+urbanization combo
            valid_base = decile_urban_group[
                (decile_urban_group[consumption_col].notna()) &
                (decile_urban_group['HB061'].notna()) &
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna())
            ].copy()
            
            if len(valid_base) == 0:
                continue
            
            denominator = valid_base['HA10'].sum()
            
            # Per adult equivalent consumption
            consumption_ae = valid_base[consumption_col] / valid_base['HB061']
            
            # Weighted average (per adult equivalent)
            weighted_sum = (consumption_ae * valid_base['HA10']).sum()
            wmean = weighted_sum / denominator if denominator > 0 else 0
            
            results.append({
                'decile': decile,
                'urbanization': urban_label,
                'urbanization_code': urban_code,
                'total_consumption': wmean,
                'currency': currency,
                'n_households': len(valid_base),
                'sum_weights': denominator
            })
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated total raw consumption for {len(result_df)} decile-urbanization combinations")
    
    return result_df


def plot_total_consumption_raw_by_urbanization(total_df, dirs):
    """Create chart of EUR_HE00 (raw total consumption) by decile and urbanization (diagnostic)."""
    print("\n=== CREATING DIAGNOSTIC CHART: TOTAL RAW CONSUMPTION (EUR_HE00) BY DECILE AND URBANIZATION ===")
    
    if total_df.empty:
        print("ERROR: No total consumption data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    urbanization_order = [
        'Sparsely populated (<100/km²)',
        'Intermediate (100-499/km²)',
        'Densely populated (≥500/km²)'
    ]
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Total Raw Consumption (EUR_HE00) by Income Decile and Degree of Urbanization (DIAGNOSTIC)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    
    for idx, urban_label in enumerate(urbanization_order):
        ax = axes[idx]
        
        urban_data = total_df[total_df['urbanization'] == urban_label].copy()
        urban_data['decile'] = pd.Categorical(urban_data['decile'], categories=decile_order, ordered=True)
        urban_data = urban_data.sort_values('decile')
        
        if urban_data.empty:
            ax.text(0.5, 0.5, f'No data for {urban_label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(False)
            continue
        
        x_pos = np.arange(len(urban_data))
        values = urban_data['total_consumption'].values
        
        bars = ax.bar(x_pos, values, color=colors[idx], edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2., val, f'{int(val):,}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add household count and weight info
        for i, row in urban_data.iterrows():
            decile_idx = list(decile_order).index(row['decile'])
            info_text = f"n={row['n_households']}\nw={row['sum_weights']:.0f}"
            ax.text(decile_idx, values[decile_idx] * 0.5, info_text, 
                   ha='center', va='center', fontsize=7, style='italic', alpha=0.7)
        
        ax.set_ylabel('Mean Consumption EUR/AE', fontsize=10, fontweight='bold')
        ax.set_title(f'{urban_label} - Total EUR_HE00', fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(urban_data['decile'].values)
    axes[2].set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'DIAGNOSTIC_total_consumption_by_urbanization_EUR.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def calculate_total_consumption_raw_by_decile_urbanization(df):
    """Calculate total EUR_HE00_pps (in PPS) per adult equivalent by decile and urbanization.
    
    Numerator: SUM(((EUR_HE00_pps / HB061) * HA10) for that decile+urbanization combo
    Denominator: SUM(HA10) for that SAME decile+urbanization combo
    
    CRITICAL: Must divide by HB061 to match component calculations for correct percentage!
    """
    print("\n=== CALCULATING TOTAL EXPENDITURE (EUR_HE00_PPS) PER ADULT EQUIVALENT BY DECILE AND URBANIZATION ===")
    
    consumption_col = 'EUR_HE00_pps'  # Use PPS version for total (must match components)
    currency = 'PPS'
    
    urbanization_map = {
        1: 'Densely populated (≥500/km²)',
        2: 'Intermediate (100-499/km²)',
        3: 'Sparsely populated (<100/km²)'
    }
    
    results = []
    
    for urban_code, urban_label in urbanization_map.items():
        urban_group = df[df['HA09'] == urban_code]
        
        if urban_group.empty:
            continue
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_urban_group = urban_group[urban_group['income_decile'] == decile].copy()
            
            if decile_urban_group.empty:
                continue
            
            # Ensure numeric columns
            decile_urban_group[consumption_col] = pd.to_numeric(decile_urban_group[consumption_col], errors='coerce')
            decile_urban_group['HB061'] = pd.to_numeric(decile_urban_group['HB061'], errors='coerce')
            decile_urban_group['HA10'] = pd.to_numeric(decile_urban_group['HA10'], errors='coerce')
            
            # Calculate denominator for THIS decile+urbanization combo
            valid_base = decile_urban_group[
                (decile_urban_group[consumption_col].notna()) &
                (decile_urban_group['HB061'].notna()) &
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna())
            ].copy()
            
            if len(valid_base) == 0:
                continue
            
            denominator = valid_base['HA10'].sum()
            
            # Per adult equivalent consumption
            consumption_ae = valid_base[consumption_col] / valid_base['HB061']
            
            # Weighted average (per adult equivalent)
            weighted_sum = (consumption_ae * valid_base['HA10']).sum()
            wmean = weighted_sum / denominator if denominator > 0 else 0
            
            results.append({
                'decile': decile,
                'urbanization': urban_label,
                'urbanization_code': urban_code,
                'total_consumption': wmean,
                'currency': currency,
                'n_households': len(valid_base),
                'sum_weights': denominator
            })
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated total EUR_HE00_pps (per AE) for {len(result_df)} decile-urbanization combinations")
    
    return result_df


def calculate_total_consumption_combined_by_decile(df):
    """Calculate total EUR_HE00_pps (in PPS) per adult equivalent by decile ONLY (no urbanization split).
    
    Numerator: SUM(((EUR_HE00_pps / HB061) * HA10) for that decile, ALL urbanization levels
    Denominator: SUM(HA10) for that SAME decile
    """
    print("\n=== CALCULATING TOTAL EXPENDITURE (EUR_HE00_PPS) PER ADULT EQUIVALENT BY DECILE (COMBINED) ===")
    
    consumption_col = 'EUR_HE00_pps'
    currency = 'PPS'
    
    results = []
    
    for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
        decile_group = df[df['income_decile'] == decile].copy()
        
        if decile_group.empty:
            continue
        
        # Ensure numeric columns
        decile_group[consumption_col] = pd.to_numeric(decile_group[consumption_col], errors='coerce')
        decile_group['HB061'] = pd.to_numeric(decile_group['HB061'], errors='coerce')
        decile_group['HA10'] = pd.to_numeric(decile_group['HA10'], errors='coerce')
        
        # Calculate denominator for THIS decile (all urbanization levels combined)
        valid_base = decile_group[
            (decile_group[consumption_col].notna()) &
            (decile_group['HB061'].notna()) &
            (decile_group['HB061'] > 0) &
            (decile_group['HA10'].notna())
        ].copy()
        
        if len(valid_base) == 0:
            continue
        
        denominator = valid_base['HA10'].sum()
        
        # Per adult equivalent consumption
        consumption_ae = valid_base[consumption_col] / valid_base['HB061']
        
        # Weighted average (per adult equivalent)
        weighted_sum = (consumption_ae * valid_base['HA10']).sum()
        wmean = weighted_sum / denominator if denominator > 0 else 0
        
        results.append({
            'decile': decile,
            'total_consumption': wmean,
            'currency': currency,
            'n_households': len(valid_base),
            'sum_weights': denominator
        })
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated total EUR_HE00_pps (per AE) for {len(result_df)} deciles (combined)")
    
    return result_df


def plot_total_consumption_combined_by_decile(total_df, dirs):
    """Create chart of EUR_HE00_pps (combined total consumption) by decile only."""
    print("\n=== CREATING CHART: TOTAL CONSUMPTION (EUR_HE00_PPS) BY DECILE (COMBINED) ===")
    
    if total_df.empty:
        print("ERROR: No total consumption data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    total_df['decile'] = pd.Categorical(total_df['decile'], categories=decile_order, ordered=True)
    total_df = total_df.sort_values('decile')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Total Consumption (EUR_HE00_PPS) by Income Decile - All Urbanization Levels Combined', 
                 fontsize=14, fontweight='bold')
    
    x_pos = np.arange(len(total_df))
    values = total_df['total_consumption'].values
    
    bars = ax.bar(x_pos, values, color='#1f77b4', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2., val, f'{int(val):,}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add household count and weight info
    for i, row in total_df.iterrows():
        decile_idx = list(decile_order).index(row['decile'])
        info_text = f"n={row['n_households']}\nw={row['sum_weights']:.0f}"
        ax.text(decile_idx, values[decile_idx] * 0.5, info_text, 
               ha='center', va='center', fontsize=8, style='italic', alpha=0.7)
    
    ax.set_ylabel('Mean Consumption PPS/AE', fontsize=11, fontweight='bold')
    ax.set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(total_df['decile'].values)
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'DIAGNOSTIC_total_consumption_by_decile_combined.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_housing_by_urbanization(housing_df, dirs):
    """Create stacked bar chart for housing by decile and urbanization (absolute values)."""
    print("\n=== CREATING HOUSING BY URBANIZATION CHART (ABSOLUTE) ===")
    
    if housing_df.empty:
        print("ERROR: No housing data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Order: Sparsely (3), Intermediate (2), Densely (1)
    urbanization_order = [
        'Sparsely populated (<100/km²)',
        'Intermediate (100-499/km²)',
        'Densely populated (≥500/km²)'
    ]
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    component_cols = [col for col in housing_df.columns 
                     if col not in ['decile', 'urbanization', 'urbanization_code', 'currency']]
    
    # Define colors and hatches for housing components (all orange with different textures)
    housing_color = '#fc8d62'
    hatches = ['', '///', 'xxx']  # Different hatches for different components
    
    # Create figure with 3 subplots (one per urbanization level)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Housing Expenditure Components by Income Decile and Degree of Urbanization', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    for idx, urban_label in enumerate(urbanization_order):
        ax = axes[idx]
        
        # Filter data for this urbanization level
        urban_data = housing_df[housing_df['urbanization'] == urban_label].copy()
        urban_data['decile'] = pd.Categorical(urban_data['decile'], categories=decile_order, ordered=True)
        urban_data = urban_data.sort_values('decile')
        
        if urban_data.empty:
            ax.text(0.5, 0.5, f'No data for {urban_label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(False)
            continue
        
        # Prepare data
        x_pos = np.arange(len(urban_data))
        bottom = np.zeros(len(urban_data))
        
        # Create stacked bars with different hatches
        for comp_idx, component in enumerate(component_cols):
            if component in urban_data.columns:
                values = urban_data[component].fillna(0).values
                ax.bar(x_pos, values, bottom=bottom, label=component, 
                      color=housing_color, hatch=hatches[comp_idx % len(hatches)],
                      edgecolor='white', linewidth=0.8, alpha=0.85)
                bottom += values
        
        # Add total values on top
        for i in range(len(urban_data)):
            total = bottom[i]
            if total > 0:
                ax.text(i, total, f'{int(total):,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Formatting
        ax.set_ylabel(f'Mean Consumption (PPS)', fontsize=10, fontweight='bold')
        ax.set_title(urban_label, fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        if idx == 0:  # Only show legend on top subplot, positioned outside
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95, bbox_to_anchor=(1.01, 1))
    
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(urban_data['decile'].values)
    axes[2].set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_housing_by_urbanization_absolute.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_housing_by_urbanization_percentage(housing_df, total_df, dirs):
    """Create percentage stacked bar chart for housing by decile and urbanization."""
    print("\n=== CREATING HOUSING BY URBANIZATION CHART (PERCENTAGE) ===")
    
    if housing_df.empty or total_df.empty:
        print("ERROR: No housing or total consumption data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Order: Sparsely (3), Intermediate (2), Densely (1)
    urbanization_order = [
        'Sparsely populated (<100/km²)',
        'Intermediate (100-499/km²)',
        'Densely populated (≥500/km²)'
    ]
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    component_cols = [col for col in housing_df.columns 
                     if col not in ['decile', 'urbanization', 'urbanization_code', 'currency']]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Housing Expenditure as % of Total Consumption by Decile and Degree of Urbanization', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    housing_color = '#fc8d62'
    hatches = ['', '///', 'xxx']
    
    for idx, urban_label in enumerate(urbanization_order):
        ax = axes[idx]
        
        # Filter data for this urbanization level
        urban_housing = housing_df[housing_df['urbanization'] == urban_label].copy()
        urban_total = total_df[total_df['urbanization'] == urban_label].copy()
        
        urban_housing['decile'] = pd.Categorical(urban_housing['decile'], categories=decile_order, ordered=True)
        urban_housing = urban_housing.sort_values('decile')
        
        urban_total['decile'] = pd.Categorical(urban_total['decile'], categories=decile_order, ordered=True)
        urban_total = urban_total.sort_values('decile')
        
        if urban_housing.empty or urban_total.empty:
            ax.text(0.5, 0.5, f'No data for {urban_label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(False)
            continue
        
        # Calculate percentages
        x_pos = np.arange(len(urban_housing))
        bottom = np.zeros(len(urban_housing))
        
        housing_total = np.zeros(len(urban_housing))
        for component in component_cols:
            if component in urban_housing.columns:
                housing_total += urban_housing[component].fillna(0).values
        
        # Create stacked bars with percentages
        for comp_idx, component in enumerate(component_cols):
            if component in urban_housing.columns:
                values = urban_housing[component].fillna(0).values
                percentages = (values / urban_total['total_consumption'].values * 100)
                ax.bar(x_pos, percentages, bottom=bottom, label=component,
                      color=housing_color, hatch=hatches[comp_idx % len(hatches)],
                      edgecolor='white', linewidth=0.8, alpha=0.85)
                bottom += percentages
        
        # Add "other components" in grey (remaining percentage)
        other_pct = 100 - bottom
        ax.bar(x_pos, other_pct, bottom=bottom, label='Other Components',
              color='#cccccc', edgecolor='white', linewidth=0.8, alpha=0.85)
        
        # No labels on bars for percentage chart
        
        # Formatting
        ax.set_ylabel('Percentage of Total Consumption (%)', fontsize=10, fontweight='bold')
        ax.set_title(urban_label, fontsize=11, fontweight='bold', loc='left')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95, bbox_to_anchor=(1.01, 1))
    
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(urban_housing['decile'].values)
    axes[2].set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_housing_by_urbanization_percentage.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_transport_by_urbanization(transport_df, dirs):
    """Create stacked bar chart for transport by decile and urbanization (absolute values)."""
    print("\n=== CREATING TRANSPORT BY URBANIZATION CHART (ABSOLUTE) ===")
    
    if transport_df.empty:
        print("ERROR: No transport data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    urbanization_order = [
        'Sparsely populated (<100/km²)',
        'Intermediate (100-499/km²)',
        'Densely populated (≥500/km²)'
    ]
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    component_cols = [col for col in transport_df.columns 
                     if col not in ['decile', 'urbanization', 'urbanization_code', 'currency']]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Transport Expenditure Components by Income Decile and Degree of Urbanization', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    transport_color = '#b3de69'
    hatches = ['', '///', 'xxx']
    
    for idx, urban_label in enumerate(urbanization_order):
        ax = axes[idx]
        
        urban_data = transport_df[transport_df['urbanization'] == urban_label].copy()
        urban_data['decile'] = pd.Categorical(urban_data['decile'], categories=decile_order, ordered=True)
        urban_data = urban_data.sort_values('decile')
        
        if urban_data.empty:
            ax.text(0.5, 0.5, f'No data for {urban_label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(False)
            continue
        
        x_pos = np.arange(len(urban_data))
        bottom = np.zeros(len(urban_data))
        
        for comp_idx, component in enumerate(component_cols):
            if component in urban_data.columns:
                values = urban_data[component].fillna(0).values
                ax.bar(x_pos, values, bottom=bottom, label=component,
                      color=transport_color, hatch=hatches[comp_idx % len(hatches)],
                      edgecolor='white', linewidth=0.8, alpha=0.85)
                bottom += values
        
        for i in range(len(urban_data)):
            total = bottom[i]
            if total > 0:
                ax.text(i, total, f'{int(total):,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_ylabel('Mean Consumption (PPS)', fontsize=10, fontweight='bold')
        ax.set_title(urban_label, fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95, bbox_to_anchor=(1.01, 1))
    
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(urban_data['decile'].values)
    axes[2].set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_transport_by_urbanization_absolute.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_transport_by_urbanization_percentage(transport_df, total_df, dirs):
    """Create percentage stacked bar chart for transport by decile and urbanization."""
    print("\n=== CREATING TRANSPORT BY URBANIZATION CHART (PERCENTAGE) ===")
    
    if transport_df.empty or total_df.empty:
        print("ERROR: No transport or total consumption data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    urbanization_order = [
        'Sparsely populated (<100/km²)',
        'Intermediate (100-499/km²)',
        'Densely populated (≥500/km²)'
    ]
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    component_cols = [col for col in transport_df.columns 
                     if col not in ['decile', 'urbanization', 'urbanization_code', 'currency']]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Transport Expenditure as % of Total Consumption by Decile and Degree of Urbanization', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    transport_color = '#b3de69'
    hatches = ['', '///', 'xxx']
    
    for idx, urban_label in enumerate(urbanization_order):
        ax = axes[idx]
        
        urban_transport = transport_df[transport_df['urbanization'] == urban_label].copy()
        urban_total = total_df[total_df['urbanization'] == urban_label].copy()
        
        urban_transport['decile'] = pd.Categorical(urban_transport['decile'], categories=decile_order, ordered=True)
        urban_transport = urban_transport.sort_values('decile')
        
        urban_total['decile'] = pd.Categorical(urban_total['decile'], categories=decile_order, ordered=True)
        urban_total = urban_total.sort_values('decile')
        
        if urban_transport.empty or urban_total.empty:
            ax.text(0.5, 0.5, f'No data for {urban_label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(False)
            continue
        
        x_pos = np.arange(len(urban_transport))
        bottom = np.zeros(len(urban_transport))
        
        transport_total = np.zeros(len(urban_transport))
        for component in component_cols:
            if component in urban_transport.columns:
                transport_total += urban_transport[component].fillna(0).values
        
        for comp_idx, component in enumerate(component_cols):
            if component in urban_transport.columns:
                values = urban_transport[component].fillna(0).values
                percentages = (values / urban_total['total_consumption'].values * 100)
                ax.bar(x_pos, percentages, bottom=bottom, label=component,
                      color=transport_color, hatch=hatches[comp_idx % len(hatches)],
                      edgecolor='white', linewidth=0.8, alpha=0.85)
                bottom += percentages
        
        other_pct = 100 - bottom
        ax.bar(x_pos, other_pct, bottom=bottom, label='Other Components',
              color='#cccccc', edgecolor='white', linewidth=0.8, alpha=0.85)
        
        ax.set_ylabel('Percentage of Total Consumption (%)', fontsize=10, fontweight='bold')
        ax.set_title(urban_label, fontsize=11, fontweight='bold', loc='left')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95, bbox_to_anchor=(1.01, 1))
    
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(urban_transport['decile'].values)
    axes[2].set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_transport_by_urbanization_percentage.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def calculate_other_components_by_urbanization(df):
    """Calculate Food & Beverage, Health, and Education by decile+urbanization.
    
    CRITICAL: Must use same denominator as total consumption for percentage calculations!
    Denominator = SUM(HA10) for households with valid consumption_col, HB061, HA10.
    """
    print("\n=== CALCULATING OTHER COMPONENTS (FOOD, HEALTH, EDUCATION) BY DECILE AND URBANIZATION ===")
    
    other_components = {
        'EUR_HE01': 'Food & Beverage',
        'EUR_HE06': 'Health',
        'EUR_HE10': 'Education'
    }
    
    urbanization_map = {
        1: 'Densely populated (≥500/km²)',
        2: 'Intermediate (100-499/km²)',
        3: 'Sparsely populated (<100/km²)'
    }
    
    has_pps = 'EUR_HE01_pps' in df.columns and df['EUR_HE01_pps'].notna().sum() > 0
    suffix = '_pps' if has_pps else ''
    currency = 'PPS' if has_pps else 'EUR'
    
    # Use total consumption to determine valid households (for consistent denominators with total)
    consumption_col = 'EUR_HE00_pps' if has_pps else 'EUR_HE00'
    
    results = []
    
    for urban_code, urban_label in urbanization_map.items():
        urban_group = df[df['HA09'] == urban_code]
        
        if urban_group.empty:
            continue
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_urban_group = urban_group[urban_group['income_decile'] == decile].copy()
            
            if decile_urban_group.empty:
                continue
            
            decile_data = {
                'decile': decile,
                'urbanization': urban_label,
                'urbanization_code': urban_code,
                'currency': currency
            }
            
            # Ensure numeric columns
            decile_urban_group[consumption_col] = pd.to_numeric(decile_urban_group[consumption_col], errors='coerce')
            for code in other_components.keys():
                col_name = f'{code}{suffix}'
                if col_name not in df.columns:
                    col_name = code
                if col_name in decile_urban_group.columns:
                    decile_urban_group[col_name] = pd.to_numeric(decile_urban_group[col_name], errors='coerce')
            
            decile_urban_group['HB061'] = pd.to_numeric(decile_urban_group['HB061'], errors='coerce')
            decile_urban_group['HA10'] = pd.to_numeric(decile_urban_group['HA10'], errors='coerce')
            
            # Calculate denominator for THIS decile+urbanization combo
            # CRITICAL: Use same filter as total consumption for consistent percentages!
            valid_base = decile_urban_group[
                (decile_urban_group[consumption_col].notna()) &
                (decile_urban_group['HB061'].notna()) &
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna())
            ].copy()
            
            if len(valid_base) == 0:
                continue
            
            denominator = valid_base['HA10'].sum()
            
            for code, name in other_components.items():
                col_name = f'{code}{suffix}'
                if col_name not in df.columns:
                    col_name = code
                
                if col_name in valid_base.columns:
                    # Include all households, treating missing/zero values as 0
                    comp_ae = valid_base[col_name].fillna(0) / valid_base['HB061']
                    weighted_sum = (comp_ae * valid_base['HA10']).sum()
                    comp_value = weighted_sum / denominator if denominator > 0 else 0
                    decile_data[name] = comp_value
                else:
                    decile_data[name] = 0
            
            results.append(decile_data)
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated other components for {len(result_df)} decile-urbanization combinations")
    
    return result_df


def calculate_total_housing_by_decile_urbanization(df):
    """Calculate TOTAL housing (EUR_HE02) by income decile and degree of urbanization.
    
    Numerator: SUM((housing / HB061) * HA10) for that decile+urbanization combo
    Denominator: SUM(HA10) for that SAME decile+urbanization combo (with valid housing data)
    
    CRITICAL: Denominator must be LOCAL to each decile+urbanization combo, NOT global!
    """
    print("\n=== CALCULATING TOTAL HOUSING BY DECILE AND URBANIZATION ===")
    
    urbanization_map = {
        1: 'Densely populated (≥500/km²)',
        2: 'Intermediate (100-499/km²)',
        3: 'Sparsely populated (<100/km²)'
    }
    
    has_pps = 'EUR_HE02_pps' in df.columns and df['EUR_HE02_pps'].notna().sum() > 0
    housing_col = 'EUR_HE02_pps' if has_pps else 'EUR_HE02'
    currency = 'PPS' if has_pps else 'EUR'
    
    print(f"Using column: {housing_col} ({currency})")
    
    results = []
    
    for urban_code, urban_label in urbanization_map.items():
        urban_group = df[df['HA09'] == urban_code]
        
        if urban_group.empty:
            print(f"  WARNING: No data for urbanization {urban_code}")
            continue
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_urban_group = urban_group[urban_group['income_decile'] == decile].copy()
            
            if decile_urban_group.empty:
                continue
            
            # Ensure numeric columns
            decile_urban_group[housing_col] = pd.to_numeric(decile_urban_group[housing_col], errors='coerce')
            decile_urban_group['HB061'] = pd.to_numeric(decile_urban_group['HB061'], errors='coerce')
            decile_urban_group['HA10'] = pd.to_numeric(decile_urban_group['HA10'], errors='coerce')
            
            # Filter to valid records for THIS decile+urbanization combo
            valid_base = decile_urban_group[
                (decile_urban_group[housing_col].notna()) &
                (decile_urban_group['HB061'].notna()) &
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna())
            ].copy()
            
            if len(valid_base) == 0:
                continue
            
            # Use LOCAL denominator for THIS decile+urbanization combo
            denominator = valid_base['HA10'].sum()
            if denominator == 0:
                continue
            
            # Calculate per-capita housing (adjusted by equivalence scale)
            valid_base['housing_ae'] = valid_base[housing_col] / valid_base['HB061']
            
            # Sum weighted values: (per-capita value * household weight)
            weighted_sum = (valid_base['housing_ae'] * valid_base['HA10']).sum()
            
            # Divide by LOCAL weight for this urbanization+decile combo
            housing_value = weighted_sum / denominator
            
            results.append({
                'decile': decile,
                'urbanization': urban_label,
                'urbanization_code': urban_code,
                'housing': housing_value,
                'currency': currency,
                'sample_size': len(valid_base)
            })
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated total housing for {len(result_df)} decile-urbanization combinations")
    
    return result_df


def plot_housing_by_urbanization_absolute(housing_df, dirs):
    """Create absolute housing chart by urbanization."""
    print("\n=== CREATING HOUSING BY URBANIZATION CHART (ABSOLUTE) ===")
    
    if housing_df.empty:
        print("ERROR: No housing data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Pivot data: rows = deciles, columns = urbanization levels
    pivot = housing_df.pivot_table(index='decile', columns='urbanization', values='housing')
    
    # Ensure correct decile order
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    pivot = pivot.reindex(decile_order)
    
    # Ensure correct urbanization column order
    urban_cols = [
        'Densely populated (≥500/km²)',
        'Intermediate (100-499/km²)',
        'Sparsely populated (<100/km²)'
    ]
    pivot = pivot[[col for col in urban_cols if col in pivot.columns]]
    
    currency = housing_df['currency'].iloc[0] if 'currency' in housing_df.columns else 'PPS'
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind='bar', ax=ax, color=colors, width=0.8)
    
    ax.set_title('Housing Consumption by Income Decile and Urbanization Level\n(Luxembourg 2020, Absolute)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Income Decile', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Housing Consumption ({currency}/adult-equivalent/year)', fontsize=12, fontweight='bold')
    ax.set_xticklabels(decile_order, rotation=0)
    ax.legend(title='Urbanization Level', fontsize=10, title_fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Format y-axis with thousand separators
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    output_file = os.path.join(graphs_dir, 'LU_housing_by_urbanization_absolute.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output_file}")
    plt.close()


def calculate_other_components_residual_by_urbanization(df, housing_col='EUR_HE02', transport_col='EUR_HE07', other_col='EUR_HE01_06_10'):
    """Calculate residual consumption (all other components) by decile+urbanization.
    
    Residual = Total - Housing - Transport - Food - Health - Education (all equivalized)
    
    Numerator: SUM(((EUR_HE00 - housing - transport - food - health - education) / HB061) * HA10)
    Denominator: SUM(HA10) for that SAME decile+urbanization combo
    """
    print("\n=== CALCULATING OTHER COMPONENTS RESIDUAL BY DECILE AND URBANIZATION ===")
    
    urbanization_map = {
        1: 'Densely populated (≥500/km²)',
        2: 'Intermediate (100-499/km²)',
        3: 'Sparsely populated (<100/km²)'
    }
    
    has_pps = 'EUR_HE00_pps' in df.columns and df['EUR_HE00_pps'].notna().sum() > 0
    consumption_col = 'EUR_HE00_pps' if has_pps else 'EUR_HE00'
    currency = 'PPS' if has_pps else 'EUR'
    
    # Component columns to subtract from total
    housing_col_name = 'EUR_HE02_pps' if has_pps and 'EUR_HE02_pps' in df.columns else 'EUR_HE02'
    transport_col_name = 'EUR_HE07_pps' if has_pps and 'EUR_HE07_pps' in df.columns else 'EUR_HE07'
    food_col_name = 'EUR_HE01_pps' if has_pps and 'EUR_HE01_pps' in df.columns else 'EUR_HE01'
    health_col_name = 'EUR_HE06_pps' if has_pps and 'EUR_HE06_pps' in df.columns else 'EUR_HE06'
    education_col_name = 'EUR_HE10_pps' if has_pps and 'EUR_HE10_pps' in df.columns else 'EUR_HE10'
    
    results = []
    
    for urban_code, urban_label in urbanization_map.items():
        urban_group = df[df['HA09'] == urban_code]
        
        if urban_group.empty:
            continue
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_urban_group = urban_group[urban_group['income_decile'] == decile].copy()
            
            if decile_urban_group.empty:
                continue
            
            # Ensure numeric columns
            decile_urban_group[consumption_col] = pd.to_numeric(decile_urban_group[consumption_col], errors='coerce')
            decile_urban_group[housing_col_name] = pd.to_numeric(decile_urban_group[housing_col_name], errors='coerce')
            decile_urban_group[transport_col_name] = pd.to_numeric(decile_urban_group[transport_col_name], errors='coerce')
            decile_urban_group[food_col_name] = pd.to_numeric(decile_urban_group[food_col_name], errors='coerce')
            decile_urban_group[health_col_name] = pd.to_numeric(decile_urban_group[health_col_name], errors='coerce')
            decile_urban_group[education_col_name] = pd.to_numeric(decile_urban_group[education_col_name], errors='coerce')
            decile_urban_group['HB061'] = pd.to_numeric(decile_urban_group['HB061'], errors='coerce')
            decile_urban_group['HA10'] = pd.to_numeric(decile_urban_group['HA10'], errors='coerce')
            
            # Calculate denominator for THIS decile+urbanization combo
            valid_base = decile_urban_group[
                (decile_urban_group['HB061'].notna()) &
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna()) &
                (decile_urban_group[consumption_col].notna())
            ].copy()
            
            if len(valid_base) == 0:
                continue
            
            denominator = valid_base['HA10'].sum()
            
            # Calculate residual = (total - housing - transport - food - health - education) / HB061
            # All subtracted components treated as 0 if missing
            housing_sum = valid_base[housing_col_name].fillna(0)
            transport_sum = valid_base[transport_col_name].fillna(0)
            food_sum = valid_base[food_col_name].fillna(0)
            health_sum = valid_base[health_col_name].fillna(0)
            education_sum = valid_base[education_col_name].fillna(0)
            
            # Residual on raw consumption: EUR_HE00 - EUR_HE01 - EUR_HE07 - EUR_HE02 - EUR_HE06 - EUR_HE10
            residual_raw = (valid_base[consumption_col] - housing_sum - transport_sum - 
                           food_sum - health_sum - education_sum)
            
            # Per adult equivalent residual
            residual_ae = residual_raw / valid_base['HB061']
            
            # Weighted average
            weighted_sum = (residual_ae * valid_base['HA10']).sum()
            residual_value = weighted_sum / denominator if denominator > 0 else 0
            
            results.append({
                'decile': decile,
                'urbanization': urban_label,
                'urbanization_code': urban_code,
                'residual': residual_value,
                'currency': currency
            })
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated residual components for {len(result_df)} decile-urbanization combinations")
    
    return result_df


def calculate_all_components_by_decile_urbanization(df):
    """Calculate ALL components (housing, transport, food, health, education) AND residual
    using the EXACT SAME METHOD as total consumption.
    
    For each component: SUM((value / HB061) * HA10) / SUM(HA10)
    where the filter is: decile + urbanization + valid consumption data
    
    Residual is computed as: total_consumption - sum_of_all_named_components
    """
    print("\n=== CALCULATING ALL COMPONENTS BY DECILE AND URBANIZATION (CONSISTENT METHOD) ===")
    
    consumption_col = 'EUR_HE00_pps'
    
    components = {
        'EUR_HE041': 'Actual Rentals',
        'EUR_HE042': 'Imputed Rentals', 
        'EUR_HE043_044_045': 'Utility bills',
        'EUR_HE071': 'Purchase of Vehicles',
        'EUR_HE072': 'Operation of Personal Transport',
        'EUR_HE073': 'Transport Services',
        'EUR_HE01': 'Food & Beverage',
        'EUR_HE06': 'Health',
        'EUR_HE10': 'Education'
    }
    
    urbanization_map = {
        1: 'Densely populated (≥500/km²)',
        2: 'Intermediate (100-499/km²)',
        3: 'Sparsely populated (<100/km²)'
    }
    
    results = []
    
    for urban_code, urban_label in urbanization_map.items():
        urban_group = df[df['HA09'] == urban_code]
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_urban_group = urban_group[urban_group['income_decile'] == decile].copy()
            
            if decile_urban_group.empty:
                continue
            
            # Convert to numeric
            decile_urban_group[consumption_col] = pd.to_numeric(decile_urban_group[consumption_col], errors='coerce')
            decile_urban_group['HB061'] = pd.to_numeric(decile_urban_group['HB061'], errors='coerce')
            decile_urban_group['HA10'] = pd.to_numeric(decile_urban_group['HA10'], errors='coerce')
            
            # CRITICAL: Use SAME filter for ALL components - consumption must be valid
            valid_base = decile_urban_group[
                (decile_urban_group[consumption_col].notna()) &
                (decile_urban_group['HB061'].notna()) &
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna())
            ].copy()
            
            if len(valid_base) == 0:
                continue
            
            # Denominator: sum of weights for THIS decile+urbanization combo
            denominator = valid_base['HA10'].sum()
            
            # Calculate total consumption
            consumption_ae = valid_base[consumption_col] / valid_base['HB061']
            total_weighted = (consumption_ae * valid_base['HA10']).sum()
            total_value = total_weighted / denominator if denominator > 0 else 0
            
            # Calculate each component using the SAME METHOD
            component_values = {}
            component_sum = 0
            
            for code, component_name in components.items():
                # Handle combined columns (like EUR_HE043_044_045)
                if 'EUR_HE043_044_045' in code:
                    # This is utility bills: EUR_HE043 + EUR_HE044 + EUR_HE045
                    cols_to_use = []
                    for sub_code in ['EUR_HE043', 'EUR_HE044', 'EUR_HE045']:
                        if f'{sub_code}_pps' in valid_base.columns:
                            cols_to_use.append(f'{sub_code}_pps')
                        elif sub_code in valid_base.columns:
                            cols_to_use.append(sub_code)
                    
                    if cols_to_use:
                        comp_raw = valid_base[cols_to_use].fillna(0).sum(axis=1)
                    else:
                        comp_raw = pd.Series([0] * len(valid_base), index=valid_base.index)
                else:
                    # Single component code
                    col_pps = f'{code}_pps'
                    col_raw = code
                    
                    if col_pps in valid_base.columns:
                        comp_raw = valid_base[col_pps].fillna(0)
                    elif col_raw in valid_base.columns:
                        comp_raw = valid_base[col_raw].fillna(0)
                    else:
                        comp_raw = pd.Series([0] * len(valid_base), index=valid_base.index)
                
                # Calculate per-AE and weighted average (SAME METHOD AS TOTAL)
                comp_ae = comp_raw / valid_base['HB061']
                comp_weighted = (comp_ae * valid_base['HA10']).sum()
                comp_value = comp_weighted / denominator if denominator > 0 else 0
                
                component_values[component_name] = comp_value
                component_sum += comp_value
            
            # Residual: total - sum of components
            residual = total_value - component_sum
            
            results.append({
                'decile': decile,
                'urbanization': urban_label,
                'urbanization_code': urban_code,
                'total_consumption': total_value,
                'residual': residual,
                'currency': 'PPS',
                **component_values
            })
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated all components for {len(result_df)} decile-urbanization combinations")
    
    return result_df


def plot_diagnostic_total_consumption_by_decile_combined(result_df, dirs):
    """DIAGNOSTIC: Plot total consumption by decile and urbanization to identify ordering issues."""
    print("\n=== DIAGNOSTIC: TOTAL CONSUMPTION BY DECILE ===")
    
    if result_df.empty:
        print("ERROR: No data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    urbanization_order = [
        'Sparsely populated (<100/km²)',
        'Intermediate (100-499/km²)',
        'Densely populated (≥500/km²)'
    ]
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle('DIAGNOSTIC: Mean Total Consumption by Income Decile and Urbanization', 
                 fontsize=14, fontweight='bold')
    
    for idx, urban_label in enumerate(urbanization_order):
        ax = axes[idx]
        
        urban_data = result_df[result_df['urbanization'] == urban_label].copy()
        urban_data['decile'] = pd.Categorical(urban_data['decile'], categories=decile_order, ordered=True)
        urban_data = urban_data.sort_values('decile')
        
        if urban_data.empty:
            ax.text(0.5, 0.5, f'No data for {urban_label}', ha='center', va='center', transform=ax.transAxes)
            continue
        
        x_pos = np.arange(len(urban_data))
        total_values = urban_data['total_consumption'].fillna(0).values
        
        # Color bars to show if they are increasing (green) or decreasing (red)
        colors = []
        for i in range(len(total_values)):
            if i == 0:
                colors.append('#2ca02c')  # Green for first
            elif total_values[i] >= total_values[i-1]:
                colors.append('#2ca02c')  # Green for increasing
            else:
                colors.append('#d62728')  # Red for decreasing
        
        bars = ax.bar(x_pos, total_values, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, total_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(value):,}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Income Decile', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Total Consumption (PPS)', fontsize=11, fontweight='bold')
        ax.set_title(urban_label, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(urban_data['decile'].values)
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Add note if any decreases detected
        has_decreases = any(total_values[i] < total_values[i-1] for i in range(1, len(total_values)))
        if has_decreases:
            ax.text(0.5, -0.15, '⚠ RED BARS = DECREASE FROM PREVIOUS DECILE (DATA ISSUE!)', 
                   ha='center', transform=ax.transAxes, fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'DIAGNOSTIC_total_consumption_by_decile_combined.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    
    # Also print the data to console for debugging
    print("\nTotal Consumption by Decile and Urbanization:")
    for urban_label in urbanization_order:
        print(f"\n{urban_label}:")
        urban_data = result_df[result_df['urbanization'] == urban_label].copy()
        urban_data['decile'] = pd.Categorical(urban_data['decile'], categories=decile_order, ordered=True)
        urban_data = urban_data.sort_values('decile')
        for _, row in urban_data.iterrows():
            print(f"  {row['decile']}: {int(row['total_consumption']):,} PPS")
    
    plt.close()


def plot_housing_transport_combined_by_urbanization(result_df, dirs):
    """Create combined housing + transport + other stacked bar chart (absolute values) using unified data."""
    print("\n=== CREATING COMBINED HOUSING + TRANSPORT + OTHER BY URBANIZATION CHART (ABSOLUTE) ===")
    
    if result_df.empty:
        print("ERROR: No component data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    urbanization_order = [
        'Sparsely populated (<100/km²)',
        'Intermediate (100-499/km²)',
        'Densely populated (≥500/km²)'
    ]
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    # Component columns (excluding metadata)
    exclude_cols = {'decile', 'urbanization', 'urbanization_code', 'total_consumption', 'residual', 'currency'}
    component_cols = [col for col in result_df.columns if col not in exclude_cols]
    
    # Organize by category
    housing_components = ['Actual Rentals', 'Imputed Rentals', 'Utility bills']
    transport_components = ['Purchase of Vehicles', 'Operation of Personal Transport', 'Transport Services']
    other_components = ['Food & Beverage', 'Health', 'Education']
    
    # Define colors
    housing_color = '#fc8d62'
    transport_color = '#b3de69'
    other_colors = ['#8dd3c7', '#ffffb3', '#bebada']
    housing_hatches = ['', '///', 'xxx']
    transport_hatches = ['', '///', 'xxx']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Housing + Transport + Other Expenditure by Income Decile and Degree of Urbanization', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    for idx, urban_label in enumerate(urbanization_order):
        ax = axes[idx]
        
        urban_data = result_df[result_df['urbanization'] == urban_label].copy()
        urban_data['decile'] = pd.Categorical(urban_data['decile'], categories=decile_order, ordered=True)
        urban_data = urban_data.sort_values('decile')
        
        if urban_data.empty:
            ax.text(0.5, 0.5, f'No data for {urban_label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(False)
            continue
        
        x_pos = np.arange(len(urban_data))
        bottom = np.zeros(len(urban_data))
        
        # Plot housing components with hatches
        for h_idx, component in enumerate(housing_components):
            if component in urban_data.columns:
                values = urban_data[component].fillna(0).values
                ax.bar(x_pos, values, bottom=bottom, label=f'{component}',
                      color=housing_color, hatch=housing_hatches[h_idx % len(housing_hatches)],
                      edgecolor='white', linewidth=0.8, alpha=0.85)
                bottom += values
        
        # Plot transport components with hatches
        for t_idx, component in enumerate(transport_components):
            if component in urban_data.columns:
                values = urban_data[component].fillna(0).values
                ax.bar(x_pos, values, bottom=bottom, label=f'{component}',
                      color=transport_color, hatch=transport_hatches[t_idx % len(transport_hatches)],
                      edgecolor='white', linewidth=0.8, alpha=0.85)
                bottom += values
        
        # Plot other components (Food, Health, Education) without hatches
        for o_idx, component in enumerate(other_components):
            if component in urban_data.columns:
                values = urban_data[component].fillna(0).values
                ax.bar(x_pos, values, bottom=bottom, label=component,
                      color=other_colors[o_idx % len(other_colors)], edgecolor='white', linewidth=0.8, alpha=0.85)
                bottom += values
        
        # Plot residual (grey)
        residual_values = urban_data['residual'].fillna(0).values
        residual_values = np.maximum(residual_values, 0)
        
        ax.bar(x_pos, residual_values, bottom=bottom, label='Other Components (Residual)',
              color='#d3d3d3', edgecolor='white', linewidth=0.8, alpha=0.85)
        
        # Add total values on top
        bottom += residual_values
        for i in range(len(urban_data)):
            total = bottom[i]
            if total > 0:
                ax.text(i, total, f'{int(total):,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_ylabel('Mean Consumption (PPS)', fontsize=10, fontweight='bold')
        ax.set_title(urban_label, fontsize=11, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.95, bbox_to_anchor=(1.01, 1), ncol=1)
    
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(urban_data['decile'].values)
    axes[2].set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_housing_transport_by_urbanization_absolute.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_housing_transport_combined_by_urbanization_percentage(result_df, dirs):
    """Create combined housing + transport + other stacked bar chart normalized to 100% per bar using unified data."""
    print("\n=== CREATING COMBINED HOUSING + TRANSPORT + OTHER BY URBANIZATION CHART (100% STACKED) ===")
    
    if result_df.empty:
        print("ERROR: No component data to plot")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    urbanization_order = [
        'Sparsely populated (<100/km²)',
        'Intermediate (100-499/km²)',
        'Densely populated (≥500/km²)'
    ]
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    # Component columns (excluding metadata)
    exclude_cols = {'decile', 'urbanization', 'urbanization_code', 'total_consumption', 'residual', 'currency'}
    component_cols = [col for col in result_df.columns if col not in exclude_cols]
    
    # Organize by category
    housing_components = ['Actual Rentals', 'Imputed Rentals', 'Utility bills']
    transport_components = ['Purchase of Vehicles', 'Operation of Personal Transport', 'Transport Services']
    other_components = ['Food & Beverage', 'Health', 'Education']
    
    # Define colors
    housing_color = '#fc8d62'
    transport_color = '#b3de69'
    other_colors = ['#8dd3c7', '#ffffb3', '#bebada']
    housing_hatches = ['', '///', 'xxx']
    transport_hatches = ['', '///', 'xxx']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Housing + Transport + Other as % of Total Consumption by Income Decile and Degree of Urbanization', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    for idx, urban_label in enumerate(urbanization_order):
        ax = axes[idx]
        
        urban_data = result_df[result_df['urbanization'] == urban_label].copy()
        urban_data['decile'] = pd.Categorical(urban_data['decile'], categories=decile_order, ordered=True)
        urban_data = urban_data.sort_values('decile')
        
        if urban_data.empty:
            ax.text(0.5, 0.5, f'No data for {urban_label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(False)
            continue
        
        x_pos = np.arange(len(urban_data))
        bottom = np.zeros(len(urban_data))
        
        # Get total consumption for normalization (per bar)
        total_consumption = urban_data['total_consumption'].fillna(0).values
        # Avoid division by zero
        total_consumption = np.where(total_consumption > 0, total_consumption, 1)
        
        # Plot housing components with hatches (normalized to percentage)
        for h_idx, component in enumerate(housing_components):
            if component in urban_data.columns:
                values = urban_data[component].fillna(0).values
                percentages = (values / total_consumption) * 100
                ax.bar(x_pos, percentages, bottom=bottom, label=f'{component}',
                      color=housing_color, hatch=housing_hatches[h_idx % len(housing_hatches)],
                      edgecolor='white', linewidth=0.8, alpha=0.85)
                bottom += percentages
        
        # Plot transport components with hatches (normalized to percentage)
        for t_idx, component in enumerate(transport_components):
            if component in urban_data.columns:
                values = urban_data[component].fillna(0).values
                percentages = (values / total_consumption) * 100
                ax.bar(x_pos, percentages, bottom=bottom, label=f'{component}',
                      color=transport_color, hatch=transport_hatches[t_idx % len(transport_hatches)],
                      edgecolor='white', linewidth=0.8, alpha=0.85)
                bottom += percentages
        
        # Plot other components (Food, Health, Education) without hatches (normalized to percentage)
        for o_idx, component in enumerate(other_components):
            if component in urban_data.columns:
                values = urban_data[component].fillna(0).values
                percentages = (values / total_consumption) * 100
                ax.bar(x_pos, percentages, bottom=bottom, label=component,
                      color=other_colors[o_idx % len(other_colors)], edgecolor='white', linewidth=0.8, alpha=0.85)
                bottom += percentages
        
        # Plot residual (grey) normalized to percentage
        residual_values = urban_data['residual'].fillna(0).values
        residual_values = np.maximum(residual_values, 0)
        residual_percentages = (residual_values / total_consumption) * 100
        
        ax.bar(x_pos, residual_percentages, bottom=bottom, label='Other Components (Residual)',
              color='#d3d3d3', edgecolor='white', linewidth=0.8, alpha=0.85)
        
        ax.set_ylabel('Percentage of Total Consumption (%)', fontsize=10, fontweight='bold')
        ax.set_title(urban_label, fontsize=11, fontweight='bold', loc='left')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.95, bbox_to_anchor=(1.01, 1), ncol=1)
    
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(urban_data['decile'].values)
    axes[2].set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_housing_transport_by_urbanization_percentage.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("HBS ULTRA-FAST TEST - LUXEMBOURG 2020")
    print("="*80)
    
    # Step 1: Setup
    print("\nSTEP 1: Setup...")
    dirs = setup_directories()
    
    # Step 2: Load Luxembourg 2020 data
    print("\nSTEP 2: Load Luxembourg 2020 data...")
    df = load_luxembourg_2020()
    if df.empty:
        print("ERROR: No data loaded!")
        return
    
    # Step 3: Load and apply PPS conversion
    print("\nSTEP 3: PPS conversion...")
    pps_df = load_pps_data(dirs)
    
    # Keep year before conversion
    df_with_year = df.copy()
    df = calculate_consumption_in_pps(df, pps_df)
    
    # Restore year if lost
    if 'year' not in df.columns and 'year' in df_with_year.columns:
        df['year'] = df_with_year['year']
    
    # Check if PPS was applied
    if 'EUR_HE00_pps' not in df.columns:
        print("WARNING: PPS column not found, using nominal values")
        df['EUR_HE00_pps'] = df['EUR_HE00']
    
    pps_coverage = df['EUR_HE00_pps'].notna().sum() / len(df) * 100
    print(f"  PPS conversion coverage: {pps_coverage:.1f}%")
    
    # Step 4: Assign deciles
    print("\nSTEP 4: Assign income deciles...")
    df = assign_simple_deciles(df)
    
    # Step 5: Calculate consumption by decile
    print("\nSTEP 5: Calculate consumption by decile...")
    consumption_df = calculate_consumption_deciles(df)
    
    if consumption_df.empty:
        print("ERROR: No consumption data!")
        return
    
    # Step 6: Calculate consumption components
    print("\nSTEP 6: Calculate consumption components...")
    components_df = calculate_consumption_components(df)
    
    # Step 7: Create visualizations
    print("\nSTEP 7: Create visualizations...")
    plot_luxembourg_consumption(consumption_df, dirs)
    
    if not components_df.empty:
        plot_consumption_components_stacked(components_df, dirs)
    
    # Step 8: Calculate housing components by decile
    print("\nSTEP 8: Calculate housing components by decile...")
    housing_df = calculate_housing_components_by_decile(df)
    
    # Step 9: Create housing components visualization
    print("\nSTEP 9: Create housing components chart...")
    if not housing_df.empty:
        plot_housing_components_by_decile(housing_df, dirs)
    
    # Step 10: Calculate housing burden
    print("\nSTEP 10: Calculate housing burden...")
    burden_data = calculate_housing_burden(df)
    
    # Step 11: Create housing burden visualization
    print("\nSTEP 11: Create housing burden chart...")
    if burden_data:
        plot_housing_burden(burden_data, dirs)
    
    # Step 12: Calculate housing by decile and urbanization
    print("\nSTEP 12: Calculate housing by decile and urbanization...")
    housing_urban_df = calculate_housing_by_decile_urbanization(df)
    
    # Step 13: Calculate transport by decile and urbanization
    print("\nSTEP 13: Calculate transport by decile and urbanization...")
    transport_urban_df = calculate_transport_by_decile_urbanization(df)
    
    # Step 14: Calculate other components (Food, Health, Education) by decile and urbanization
    print("\nSTEP 14: Calculate other components by decile and urbanization...")
    other_urban_df = calculate_other_components_by_urbanization(df)
    
    # Step 15: Calculate total consumption by decile and urbanization
    print("\nSTEP 15: Calculate total consumption by decile and urbanization...")
    total_urban_df = calculate_total_consumption_by_decile_urbanization(df)
    
    # Step 15a: DIAGNOSTIC - Calculate total RAW consumption (EUR_HE00 only) by decile and urbanization
    print("\nSTEP 15a: DIAGNOSTIC - Calculate raw EUR_HE00 by decile and urbanization...")
    total_raw_urban_df = calculate_total_consumption_raw_by_decile_urbanization(df)
    
    # Step 15b: Create diagnostic plot
    print("\nSTEP 15b: Create diagnostic plot...")
    if not total_raw_urban_df.empty:
        plot_total_consumption_raw_by_urbanization(total_raw_urban_df, dirs)
    
    # Step 15c: Calculate combined total consumption (no urbanization split)
    print("\nSTEP 15c: Calculate total consumption by decile (combined, no urbanization split)...")
    total_combined_df = calculate_total_consumption_combined_by_decile(df)
    
    # Step 15c: Create combined diagnostic plot
    print("\nSTEP 15c: Create combined diagnostic plot...")
    if not total_combined_df.empty:
        plot_total_consumption_combined_by_decile(total_combined_df, dirs)
    
    # Step 15b: Calculate TOTAL HOUSING by decile and urbanization (NEW)
    print("\nSTEP 15b: Calculate TOTAL HOUSING by decile and urbanization...")
    total_housing_urban_df = calculate_total_housing_by_decile_urbanization(df)
    
    # Step 16: Create housing by urbanization visualizations
    print("\nSTEP 16: Create housing by urbanization visualizations...")
    if not housing_urban_df.empty:
        plot_housing_by_urbanization(housing_urban_df, dirs)
        plot_housing_by_urbanization_percentage(housing_urban_df, total_urban_df, dirs)
    
    # Step 16b: Create TOTAL housing by urbanization visualizations (NEW)
    # DISABLED: plot_housing_by_urbanization_absolute suppressed per user request
    # print("\nSTEP 16b: Create TOTAL housing by urbanization visualizations...")
    # if not total_housing_urban_df.empty:
    #     plot_housing_by_urbanization_absolute(total_housing_urban_df, dirs)
    
    # Step 17: Create transport by urbanization visualizations
    print("\nSTEP 17: Create transport by urbanization visualizations...")
    if not transport_urban_df.empty:
        plot_transport_by_urbanization(transport_urban_df, dirs)
        plot_transport_by_urbanization_percentage(transport_urban_df, total_urban_df, dirs)
    
    # Step 17b: Calculate ALL COMPONENTS (housing, transport, food, health, education) + residual by decile and urbanization
    print("\nSTEP 17b: Calculate ALL components by decile and urbanization (unified method)...")
    all_components_df = calculate_all_components_by_decile_urbanization(df)
    
    # Step 17c: DIAGNOSTIC - Check total consumption by decile
    print("\nSTEP 17c: Create diagnostic chart for total consumption...")
    if not all_components_df.empty:
        plot_diagnostic_total_consumption_by_decile_combined(all_components_df, dirs)
    
    # Step 18: Create combined housing + transport + other visualizations
    print("\nSTEP 18: Create combined housing + transport + other visualizations...")
    if not all_components_df.empty:
        plot_housing_transport_combined_by_urbanization(all_components_df, dirs)
        plot_housing_transport_combined_by_urbanization_percentage(all_components_df, dirs)
    
    # Step 19: Save data to CSV
    print("\nSTEP 19: Save data to CSV...")
    data_dir = os.path.join(dirs['outputs'], 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    if not total_housing_urban_df.empty:
        output_file = os.path.join(data_dir, 'LU_housing_by_urbanization_absolute.csv')
        total_housing_urban_df.to_csv(output_file, index=False)
        print(f"OK Saved: {output_file}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

