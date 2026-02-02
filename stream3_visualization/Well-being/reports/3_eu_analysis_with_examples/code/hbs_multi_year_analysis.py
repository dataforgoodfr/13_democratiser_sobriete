"""
Multi-Year HBS Analysis for Luxembourg

Analysis of consumption patterns across 2010, 2015, and 2020
- Changes in spending shares by income decile over time
- Home expenditure trends by decile
- Energy expense trends by decile

Author: Data for Good - Well-being Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Add hbs_data_loader to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hbs_data_loader import (
    setup_directories, 
    load_pps_data, 
    calculate_consumption_in_pps,
    assign_income_deciles,
    stack_excels
)

# ============================================================================
# COLOR CONFIGURATION (from app.py)
# ============================================================================
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'
COMPONENT_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3', 
                    '#80b1d3', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']
YEAR_COLORS = {'2010': '#d62728', '2015': '#ff7f0e', '2020': '#2ca02c'}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_luxembourg_multiple_years(dirs):
    """Load Luxembourg HBS data for 2010, 2015, and 2020."""
    print("\n=== LOADING LUXEMBOURG DATA (2010, 2015, 2020) ===")
    
    external_hbs_base = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/HBS"
    
    if not os.path.exists(external_hbs_base):
        print(f"ERROR: External HBS directory not found: {external_hbs_base}")
        return pd.DataFrame()
    
    # Paths for each year
    paths = {
        '2010': os.path.join(external_hbs_base, "HBS2010/HBS2010"),
        '2015': os.path.join(external_hbs_base, "HBS2015/HBS2015"),
        '2020': os.path.join(external_hbs_base, "HBS2020/HBS2020"),
    }
    
    patterns = {
        '2010': 'LU_HBS_hh.xlsx',
        '2015': 'LU_MFR_hh.xlsx',
        '2020': 'HBS_HH_LU.xlsx',
    }
    
    dfs = []
    
    for year, folder in paths.items():
        pattern = patterns[year]
        file_path = os.path.join(folder, pattern)
        
        if os.path.exists(file_path):
            print(f"\nLoading {year}: {file_path}")
            try:
                df = pd.read_excel(file_path)
                df['year'] = year
                dfs.append(df)
                print(f"  OK Loaded {len(df)} households")
            except Exception as e:
                print(f"  ERROR Loading {year}: {str(e)}")
        else:
            # Try wildcard search
            import glob
            wildcard_path = os.path.join(folder, pattern.replace('.xlsx', '*'))
            files = glob.glob(wildcard_path)
            if files:
                print(f"Using alternative file: {files[0]}")
                df = pd.read_excel(files[0])
                df['year'] = year
                dfs.append(df)
                print(f"  OK Loaded {len(df)} households")
            else:
                print(f"  ERROR File not found: {file_path}")
    
    if not dfs:
        print("ERROR: No data loaded for any year")
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nOK Total: {len(combined_df)} households across all years")
    print(f"Years available: {sorted(combined_df['year'].unique())}")
    
    return combined_df


def assign_simple_deciles(df):
    """Assign income deciles using weighted quantiles (HA10 weight)."""
    print("\n=== ASSIGNING INCOME DECILES ===")
    
    valid_income = df[df['EUR_HH099'].notna()].copy()
    print(f"Valid records with income: {len(valid_income)}/{len(df)}")
    
    df['income_decile'] = None
    
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        year_valid = year_data[year_data['EUR_HH099'].notna()]
        
        if year_valid.empty:
            continue
        
        # Use weighted quantiles
        sorted_data = year_valid.sort_values('EUR_HH099').copy()
        sorted_data = sorted_data[sorted_data['HA10'].notna()].copy()
        
        sorted_data['cum_weight'] = sorted_data['HA10'].cumsum()
        sorted_data['cum_pct'] = sorted_data['cum_weight'] / sorted_data['HA10'].sum()
        
        # Assign deciles
        def get_decile(pct):
            if pct < 0.1: return 'D1'
            elif pct < 0.2: return 'D2'
            elif pct < 0.3: return 'D3'
            elif pct < 0.4: return 'D4'
            elif pct < 0.5: return 'D5'
            elif pct < 0.6: return 'D6'
            elif pct < 0.7: return 'D7'
            elif pct < 0.8: return 'D8'
            elif pct < 0.9: return 'D9'
            else: return 'D10'
        
        sorted_data['decile'] = sorted_data['cum_pct'].apply(get_decile)
        
        # Map back to original dataframe
        for idx, row in sorted_data.iterrows():
            df.loc[idx, 'income_decile'] = row['decile']
    
    print(f"OK Deciles assigned to {df['income_decile'].notna().sum()} households")
    
    return df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_spending_shares_by_year(df):
    """
    Calculate spending shares (as % of total consumption) by decile and year.
    Returns the % change from 2010 to 2020.
    """
    print("\n=== CALCULATING SPENDING SHARES BY YEAR ===")
    
    # Housing and Energy components
    housing_col = 'EUR_HE04_pps' if 'EUR_HE04_pps' in df.columns else 'EUR_HE04'
    energy_col = 'EUR_HE045_pps' if 'EUR_HE045_pps' in df.columns else 'EUR_HE045'
    total_col = 'EUR_HE00_pps' if 'EUR_HE00_pps' in df.columns else 'EUR_HE00'
    
    results = []
    
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_df = year_df[year_df['income_decile'] == decile]
            
            if decile_df.empty:
                continue
            
            # Housing share
            valid_housing = decile_df[
                (decile_df[housing_col].notna()) &
                (decile_df[total_col].notna()) &
                (decile_df['HB061'].notna()) &
                (decile_df['HA10'].notna()) &
                (decile_df[total_col] > 0)
            ].copy()
            
            if len(valid_housing) > 0:
                valid_housing['housing_ae'] = valid_housing[housing_col] / valid_housing['HB061']
                valid_housing['total_ae'] = valid_housing[total_col] / valid_housing['HB061']
                valid_housing['housing_share'] = (valid_housing['housing_ae'] / valid_housing['total_ae'] * 100)
                
                housing_share = np.average(valid_housing['housing_share'], weights=valid_housing['HA10'])
            else:
                housing_share = np.nan
            
            # Energy share
            valid_energy = decile_df[
                (decile_df[energy_col].notna()) &
                (decile_df[total_col].notna()) &
                (decile_df['HB061'].notna()) &
                (decile_df['HA10'].notna()) &
                (decile_df[total_col] > 0)
            ].copy()
            
            if len(valid_energy) > 0:
                valid_energy['energy_ae'] = valid_energy[energy_col] / valid_energy['HB061']
                valid_energy['total_ae'] = valid_energy[total_col] / valid_energy['HB061']
                valid_energy['energy_share'] = (valid_energy['energy_ae'] / valid_energy['total_ae'] * 100)
                
                energy_share = np.average(valid_energy['energy_share'], weights=valid_energy['HA10'])
            else:
                energy_share = np.nan
            
            results.append({
                'year': year,
                'decile': decile,
                'housing_share': housing_share,
                'energy_share': energy_share
            })
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated spending shares for {len(result_df)} decile-year combinations")
    
    return result_df


def calculate_spending_changes(spending_df):
    """Calculate % change in spending shares from 2015 to 2020."""
    print("\n=== CALCULATING SPENDING SHARE CHANGES (2015 -> 2020) ===")
    
    # Check what years are available - convert to strings for comparison
    years_available = [str(y) for y in spending_df['year'].unique()]
    print(f"DEBUG: Years available: {years_available}")
    
    if '2015' not in years_available or '2020' not in years_available:
        print(f"WARNING: Need both 2015 and 2020 data for comparison")
        return pd.DataFrame()
    
    changes = []
    
    for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
        # Convert year to string for comparison
        d2015 = spending_df[(spending_df['decile'] == decile) & (spending_df['year'].astype(str) == '2015')]
        d2020 = spending_df[(spending_df['decile'] == decile) & (spending_df['year'].astype(str) == '2020')]
        
        if not d2015.empty and not d2020.empty:
            housing_2015 = d2015['housing_share'].values[0]
            housing_2020 = d2020['housing_share'].values[0]
            energy_2015 = d2015['energy_share'].values[0]
            energy_2020 = d2020['energy_share'].values[0]
            
            housing_change = ((housing_2020 - housing_2015) / housing_2015 * 100) if housing_2015 > 0 else np.nan
            energy_change = ((energy_2020 - energy_2015) / energy_2015 * 100) if energy_2015 > 0 else np.nan
            
            changes.append({
                'decile': decile,
                'housing_change_pct': housing_change,
                'energy_change_pct': energy_change,
                'housing_2015': housing_2015,
                'housing_2020': housing_2020,
                'energy_2015': energy_2015,
                'energy_2020': energy_2020
            })
    
    change_df = pd.DataFrame(changes)
    print(f"OK Calculated {len(change_df)} decile-level changes")
    
    return change_df


def calculate_housing_energy_by_decile_year(df):
    """Calculate housing and energy expenditure (in PPS per adult equiv) by decile and year."""
    print("\n=== CALCULATING HOUSING & ENERGY BY YEAR ===")
    
    housing_col = 'EUR_HE04_pps' if 'EUR_HE04_pps' in df.columns else 'EUR_HE04'
    energy_col = 'EUR_HE045_pps' if 'EUR_HE045_pps' in df.columns else 'EUR_HE045'
    
    results = []
    
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        
        for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
            decile_df = year_df[year_df['income_decile'] == decile]
            
            if decile_df.empty:
                continue
            
            # Housing expenditure
            valid_housing = decile_df[
                (decile_df[housing_col].notna()) &
                (decile_df['HB061'].notna()) &
                (decile_df['HA10'].notna()) &
                (decile_df['HB061'] > 0)
            ].copy()
            
            if len(valid_housing) > 0:
                valid_housing['housing_ae'] = valid_housing[housing_col] / valid_housing['HB061']
                housing_exp = np.average(valid_housing['housing_ae'], weights=valid_housing['HA10'])
            else:
                housing_exp = np.nan
            
            # Energy expenditure
            valid_energy = decile_df[
                (decile_df[energy_col].notna()) &
                (decile_df['HB061'].notna()) &
                (decile_df['HA10'].notna()) &
                (decile_df['HB061'] > 0)
            ].copy()
            
            if len(valid_energy) > 0:
                valid_energy['energy_ae'] = valid_energy[energy_col] / valid_energy['HB061']
                energy_exp = np.average(valid_energy['energy_ae'], weights=valid_energy['HA10'])
            else:
                energy_exp = np.nan
            
            results.append({
                'year': year,
                'decile': decile,
                'housing_pps': housing_exp,
                'energy_pps': energy_exp
            })
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated housing & energy for {len(result_df)} decile-year combinations")
    
    return result_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_spending_share_changes(change_df, dirs):
    """Bar chart showing % change in housing/energy shares from 2010 to 2020."""
    print("\n=== CREATING SPENDING SHARE CHANGE CHART ===")
    
    if change_df.empty:
        print("WARNING: No data to plot - skipping spending share change visualization")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Housing share change
    deciles = change_df['decile'].values
    housing_changes = change_df['housing_change_pct'].values
    
    colors_housing = [COUNTRY_COLORS[1] if x < 0 else COUNTRY_COLORS[2] for x in housing_changes]
    ax1.bar(deciles, housing_changes, color=colors_housing, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    ax1.set_ylabel('% Change in Housing Cost Share', fontsize=11, fontweight='bold')
    ax1.set_title('Change in Housing Spending Share\n2015 -> 2020', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (decile, val) in enumerate(zip(deciles, housing_changes)):
        if not np.isnan(val):
            y_pos = val + (2 if val > 0 else -5)
            ax1.text(i, y_pos, f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', 
                    fontsize=9, fontweight='bold')
    
    # Energy share change
    energy_changes = change_df['energy_change_pct'].values
    colors_energy = [COUNTRY_COLORS[1] if x < 0 else COUNTRY_COLORS[2] for x in energy_changes]
    
    ax2.bar(deciles, energy_changes, color=colors_energy, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    ax2.set_ylabel('% Change in Energy Cost Share', fontsize=11, fontweight='bold')
    ax2.set_title('Change in Energy Spending Share\n2015 -> 2020', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (decile, val) in enumerate(zip(deciles, energy_changes)):
        if not np.isnan(val):
            y_pos = val + (2 if val > 0 else -5)
            ax2.text(i, y_pos, f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', 
                    fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_spending_share_changes_2015_2020.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_housing_energy_over_time(housing_energy_df, dirs):
    """Line/bar chart showing housing and energy expenditure trends by decile."""
    print("\n=== CREATING HOUSING & ENERGY OVER TIME CHART ===")
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 5, figsize=(18, 10))
    axes = axes.flatten()
    
    deciles = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    years = sorted(housing_energy_df['year'].unique())
    year_colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
    
    for ax_idx, decile in enumerate(deciles):
        ax = axes[ax_idx]
        
        decile_data = housing_energy_df[housing_energy_df['decile'] == decile].sort_values('year')
        
        if decile_data.empty:
            ax.text(0.5, 0.5, f'No data\nfor {decile}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{decile}', fontweight='bold')
            continue
        
        x_pos = np.arange(len(years))
        width = 0.35
        
        # Housing bars
        housing = decile_data['housing_pps'].values
        ax.bar(x_pos - width/2, housing, width, label='Housing (EUR_HE04)', 
              color='#8dd3c7', alpha=0.85, edgecolor='black', linewidth=0.8)
        
        # Energy bars
        energy = decile_data['energy_pps'].values
        ax.bar(x_pos + width/2, energy, width, label='Energy (EUR_HE045)', 
              color='#fc8d62', alpha=0.85, edgecolor='black', linewidth=0.8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(years)
        ax.set_ylabel('PPS per Adult Equiv.', fontsize=9)
        ax.set_title(f'{decile}', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        if ax_idx == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    fig.suptitle('Housing and Energy Expenditure by Income Decile (2010, 2015, 2020)', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_housing_energy_over_time_by_decile.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_housing_energy_trends(housing_energy_df, dirs):
    """Combined line chart showing trends across all deciles."""
    print("\n=== CREATING HOUSING & ENERGY TRENDS CHART ===")
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    years = sorted(housing_energy_df['year'].unique())
    deciles = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    # Housing trends
    for decile_idx, decile in enumerate(deciles):
        decile_data = housing_energy_df[housing_energy_df['decile'] == decile].sort_values('year')
        if not decile_data.empty:
            housing = decile_data['housing_pps'].values
            ax1.plot(years, housing, marker='o', label=decile, 
                    color=COMPONENT_COLORS[decile_idx % len(COMPONENT_COLORS)], 
                    linewidth=2, markersize=6)
    
    ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Housing Expenditure (PPS per Adult Equiv.)', fontsize=11, fontweight='bold')
    ax1.set_title('Housing Expenditure Trends by Decile', fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Energy trends
    for decile_idx, decile in enumerate(deciles):
        decile_data = housing_energy_df[housing_energy_df['decile'] == decile].sort_values('year')
        if not decile_data.empty:
            energy = decile_data['energy_pps'].values
            ax2.plot(years, energy, marker='s', label=decile,
                    color=COMPONENT_COLORS[decile_idx % len(COMPONENT_COLORS)],
                    linewidth=2, markersize=6)
    
    ax2.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Energy Expenditure (PPS per Adult Equiv.)', fontsize=11, fontweight='bold')
    ax2.set_title('Energy Expenditure Trends by Decile', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'LU_housing_energy_trends.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline for multi-year Luxembourg HBS analysis."""
    
    dirs = setup_directories()
    
    print(f"\n{'='*80}")
    print(f"LUXEMBOURG MULTI-YEAR HBS ANALYSIS (2010, 2015, 2020)")
    print(f"{'='*80}")
    
    # STEP 1: Load data
    print("\nSTEP 1: Loading Luxembourg HBS data for multiple years...")
    df = load_luxembourg_multiple_years(dirs)
    
    if df.empty:
        print("ERROR: No data loaded!")
        return
    
    # STEP 2: Load and apply PPS conversion
    print("\nSTEP 2: Loading PPS conversion factors...")
    pps_df = load_pps_data(dirs)
    
    if not pps_df.empty:
        df = calculate_consumption_in_pps(df, pps_df)
        pps_coverage = df['EUR_HE00_pps'].notna().sum() / len(df) * 100
        print(f"  PPS conversion coverage: {pps_coverage:.1f}%")
    else:
        print("  WARNING: PPS conversion not available, using nominal values")
        df['EUR_HE00_pps'] = df['EUR_HE00']
    
    # STEP 3: Assign deciles
    print("\nSTEP 3: Assigning income deciles...")
    df = assign_simple_deciles(df)
    
    # STEP 4: Calculate spending shares
    print("\nSTEP 4: Calculating spending shares by year...")
    spending_df = calculate_spending_shares_by_year(df)
    
    # STEP 5: Calculate changes from 2010 to 2020
    print("\nSTEP 5: Calculating spending share changes...")
    change_df = calculate_spending_changes(spending_df)
    
    # STEP 6: Plot spending share changes
    print("\nSTEP 6: Creating spending share change visualizations...")
    plot_spending_share_changes(change_df, dirs)
    
    # STEP 7: Calculate housing & energy by year
    print("\nSTEP 7: Calculating housing & energy expenditure by year...")
    housing_energy_df = calculate_housing_energy_by_decile_year(df)
    
    # STEP 8: Plot housing & energy by decile (small multiples)
    print("\nSTEP 8: Creating housing & energy by decile chart...")
    plot_housing_energy_over_time(housing_energy_df, dirs)
    
    # STEP 9: Plot housing & energy trends (all deciles)
    print("\nSTEP 9: Creating housing & energy trends chart...")
    plot_housing_energy_trends(housing_energy_df, dirs)
    
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Output directory: {dirs['outputs']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
