"""
HBS Analysis - Disposable Income After Fundamental Needs
Creates scatter plots showing disposable income (after housing, transport, food, health, education)
by income decile and urbanization level for selected countries.
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

# Color palette from ewbi_visuals.py
COUNTRY_COLORS_PALETTE = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']

# Country color map (using ewbi_visuals.py palette)
_country_colors = {
    'Belgium': '#ffd558',
    'France': '#fb8072',
    'Germany': '#b3de69',
    'Poland': '#fdb462',
    'Spain': '#bebada',
    'Greece': '#8dd3c7',
    'Hungary': '#ffffb3',
    'Sweden': '#80b1d3'  # EU_27_COLOR from ewbi_visuals.py for 8th country
}

def get_country_color(country_name):
    """Get color for country"""
    return _country_colors.get(country_name, '#8dd3c7')


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


def calculate_disposable_after_needs_by_decile_urbanization(df):
    """
    Calculate disposable income after fundamental needs (housing, transport, food, health, education).
    
    Metric: EUR_HH099 (equivalized income) - (housing + transport + other components)
    """
    print("\n=== CALCULATING DISPOSABLE INCOME AFTER FUNDAMENTAL NEEDS ===")
    
    consumption_col = 'EUR_HE00_pps'
    income_col = 'EUR_HH099'
    
    components = {
        'housing': ['Actual Rentals', 'Imputed Rentals', 'Utility bills'],
        'transport': ['Purchase of Vehicles', 'Operation of Personal Transport', 'Transport Services'],
        'other': ['Food & Beverage', 'Health', 'Education']
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
            
            decile_urban_group[income_col] = pd.to_numeric(decile_urban_group[income_col], errors='coerce')
            decile_urban_group['HB061'] = pd.to_numeric(decile_urban_group['HB061'], errors='coerce')
            decile_urban_group['HA10'] = pd.to_numeric(decile_urban_group['HA10'], errors='coerce')
            
            valid = decile_urban_group[
                (decile_urban_group[income_col].notna()) & 
                (decile_urban_group['HB061'].notna()) & 
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna())
            ].copy()
            
            if len(valid) == 0:
                continue
            
            denominator = valid['HA10'].sum()
            
            # Calculate equivalized income
            equivalized_income = valid[income_col] / valid['HB061']
            weighted_income = (equivalized_income * valid['HA10']).sum() / denominator
            
            # Calculate sum of housing, transport, and other components
            total_needs = 0
            
            for component_category, component_names in components.items():
                component_sum = pd.Series([0.0] * len(valid), index=valid.index)
                
                for component_name in component_names:
                    # Try to find the component in the data
                    if component_name == 'Actual Rentals':
                        col = 'EUR_HE041_pps' if 'EUR_HE041_pps' in valid.columns else 'EUR_HE041'
                    elif component_name == 'Imputed Rentals':
                        col = 'EUR_HE042_pps' if 'EUR_HE042_pps' in valid.columns else 'EUR_HE042'
                    elif component_name == 'Utility bills':
                        # Sum EUR_HE043, EUR_HE044, EUR_HE045
                        cols = []
                        for sub_code in ['EUR_HE043', 'EUR_HE044', 'EUR_HE045']:
                            if f'{sub_code}_pps' in valid.columns:
                                cols.append(f'{sub_code}_pps')
                            elif sub_code in valid.columns:
                                cols.append(sub_code)
                        if cols:
                            component_sum += valid[cols].fillna(0).sum(axis=1)
                        continue
                    elif component_name == 'Purchase of Vehicles':
                        col = 'EUR_HE071_pps' if 'EUR_HE071_pps' in valid.columns else 'EUR_HE071'
                    elif component_name == 'Operation of Personal Transport':
                        col = 'EUR_HE072_pps' if 'EUR_HE072_pps' in valid.columns else 'EUR_HE072'
                    elif component_name == 'Transport Services':
                        col = 'EUR_HE073_pps' if 'EUR_HE073_pps' in valid.columns else 'EUR_HE073'
                    elif component_name == 'Food & Beverage':
                        col = 'EUR_HE01_pps' if 'EUR_HE01_pps' in valid.columns else 'EUR_HE01'
                    elif component_name == 'Health':
                        col = 'EUR_HE06_pps' if 'EUR_HE06_pps' in valid.columns else 'EUR_HE06'
                    elif component_name == 'Education':
                        col = 'EUR_HE10_pps' if 'EUR_HE10_pps' in valid.columns else 'EUR_HE10'
                    else:
                        continue
                    
                    if col in valid.columns:
                        component_sum += valid[col].fillna(0)
                
                # Calculate weighted mean for this component category
                comp_ae = component_sum / valid['HB061']
                comp_weighted = (comp_ae * valid['HA10']).sum() / denominator
                total_needs += comp_weighted
            
            # Calculate disposable income after needs
            disposable_after_needs = weighted_income - total_needs
            
            results.append({
                'decile': decile,
                'urbanization': urban_label,
                'urbanization_code': urban_code,
                'equivalized_income': weighted_income,
                'total_needs_spending': total_needs,
                'disposable_after_needs': disposable_after_needs,
                'currency': 'PPS'
            })
    
    result_df = pd.DataFrame(results)
    print(f"OK Calculated disposable income for {len(result_df)} decile-urbanization combinations")
    
    return result_df


def plot_disposable_income_scatter_all_countries(all_results, dirs):
    """Create scatter plot showing disposable income after needs for all countries by decile and urbanization."""
    print(f"\n=== CREATING COMBINED SCATTER PLOT FOR ALL COUNTRIES ===")
    
    if not all_results:
        print(f"ERROR: No data for any countries")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Arrange urbanization from sparse (top) to dense (bottom)
    urbanization_order = [
        'Sparsely populated (<100/km²)',
        'Intermediate (100-499/km²)',
        'Densely populated (≥500/km²)'
    ]
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    # Find global maximum y-value across all countries and urbanization levels
    all_values = []
    for country_data in all_results.values():
        all_values.extend(country_data['disposable_after_needs'].values)
    
    if all_values:
        global_max = max(all_values)
        y_max_scale = global_max * 1.10  # Add 10% buffer
    else:
        y_max_scale = 1000  # Default fallback
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Disposable Income After Fundamental Needs by Urbanization (2020)\n'
                 'Income - (Housing + Transport + Food + Health + Education)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    for urban_idx, urban_label in enumerate(urbanization_order):
        ax = axes[urban_idx]
        
        # Plot data for each country
        for country_name, result_df in all_results.items():
            country_color = get_country_color(country_name)
            
            urban_data = result_df[result_df['urbanization'] == urban_label].copy()
            urban_data['decile'] = pd.Categorical(urban_data['decile'], categories=decile_order, ordered=True)
            urban_data = urban_data.sort_values('decile')
            
            if urban_data.empty:
                continue
            
            # Scatter plot - slight offset for each country to avoid overlap
            x_pos = np.arange(len(urban_data))
            values = urban_data['disposable_after_needs'].values
            
            # Add small offset per country to avoid dot overlap
            countries_list = list(all_results.keys())
            country_idx = countries_list.index(country_name)
            x_offset = (country_idx - len(countries_list)/2 + 0.5) * 0.08
            
            ax.scatter(x_pos + x_offset, values, s=100, c=country_color, alpha=0.7, label=country_name)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Formatting
        ax.set_ylabel('Disposable Income (PPS)', fontsize=11, fontweight='bold')
        ax.set_title(f'{urban_label}', fontsize=12, fontweight='bold')
        ax.set_xticks(np.arange(len(decile_order)))
        ax.set_xticklabels(decile_order, fontsize=10)
        ax.set_xlabel('Income Decile', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000):.0f}k'))
        
        # Set y-axis with consistent scale: 0 to (max + 10%)
        ax.set_ylim(0, y_max_scale)
        
        # Add legend on first subplot
        if urban_idx == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'HBS_disposable_income_after_needs_all_countries.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("HBS DISPOSABLE INCOME AFTER FUNDAMENTAL NEEDS ANALYSIS")
    print("="*80)
    
    dirs = setup_directories()
    pps_df = load_pps_data(dirs)
    
    # Process 8 countries and collect all results
    countries = ['BE', 'FR', 'DE', 'PL', 'ES', 'EL', 'HU', 'SE']
    country_names = {
        'BE': 'Belgium',
        'FR': 'France',
        'DE': 'Germany',
        'PL': 'Poland',
        'ES': 'Spain',
        'EL': 'Greece',
        'HU': 'Hungary',
        'SE': 'Sweden'
    }
    
    all_results = {}
    
    for country_code in countries:
        country_name = country_names[country_code]
        print(f"\n--- PROCESSING {country_name.upper()} ---")
        
        df = load_country_2020(country_code)
        if df.empty:
            print(f"ERROR: Could not load {country_name} data")
            continue
        
        print(f"Initial data shape: {df.shape}")
        print(f"Data columns sample: {df.columns.tolist()[:15]}")
        
        df = calculate_consumption_in_pps(df, pps_df)
        if 'EUR_HE00_pps' not in df.columns:
            df['EUR_HE00_pps'] = df['EUR_HE00']
        
        # Diagnostic: check if PPS conversion worked
        pps_cols = [col for col in df.columns if col.endswith('_pps')]
        if pps_cols:
            sample_pps = df[pps_cols[0]].notna().sum()
            print(f"PPS columns found: {len(pps_cols)}, non-null values in first: {sample_pps}/{len(df)}")
        else:
            print("WARNING: No _pps columns found in data after PPS conversion")
        
        df = assign_simple_deciles(df)
        disposable = calculate_disposable_after_needs_by_decile_urbanization(df)
        
        if not disposable.empty:
            all_results[country_name] = disposable
        else:
            print(f"ERROR: Could not calculate disposable income for {country_name}")
            continue
    
    # Create combined plot with all countries
    if all_results:
        plot_disposable_income_scatter_all_countries(all_results, dirs)
    else:
        print("ERROR: No data for any countries")
        return
    
    print("\n" + "="*80)
    print("ALL COUNTRIES PROCESSED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
