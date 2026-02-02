"""
HBS Analysis - France vs Belgium Parallel Comparison
Ultra-fast script to load FR and BE data and create parallel visualization.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from hbs_data_loader import setup_directories, load_pps_data, calculate_consumption_in_pps
import glob

# Set plotting style
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
    
    # Find country files
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
    """Assign income deciles based on EQUIVALIZED income."""
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


def calculate_all_components_by_decile_urbanization(df):
    """Calculate all components by decile and urbanization (same method as main script)."""
    print("\n=== CALCULATING ALL COMPONENTS BY DECILE AND URBANIZATION ===")
    
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
            
            valid = decile_urban_group[
                (decile_urban_group[consumption_col].notna()) & 
                (decile_urban_group['HB061'].notna()) & 
                (decile_urban_group['HB061'] > 0) &
                (decile_urban_group['HA10'].notna())
            ].copy()
            
            if len(valid) == 0:
                continue
            
            # Denominator: sum of weights for THIS decile+urbanization combo
            denominator = valid['HA10'].sum()
            
            # Calculate total consumption
            consumption_ae = valid[consumption_col] / valid['HB061']
            total_weighted = (consumption_ae * valid['HA10']).sum()
            total_value = total_weighted / denominator if denominator > 0 else 0
            
            # Calculate each component using the SAME METHOD
            component_values = {}
            component_sum = 0
            
            for code, component_name in components.items():
                # Handle combined columns (like EUR_HE043_044_045)
                if 'EUR_HE043_044_045' in code:
                    cols_to_use = []
                    for sub_code in ['EUR_HE043', 'EUR_HE044', 'EUR_HE045']:
                        if f'{sub_code}_pps' in valid.columns:
                            cols_to_use.append(f'{sub_code}_pps')
                        elif sub_code in valid.columns:
                            cols_to_use.append(sub_code)
                    
                    if cols_to_use:
                        comp_raw = valid[cols_to_use].fillna(0).sum(axis=1)
                    else:
                        comp_raw = pd.Series([0] * len(valid), index=valid.index)
                else:
                    col_pps = f'{code}_pps'
                    col_raw = code
                    
                    if col_pps in valid.columns:
                        comp_raw = valid[col_pps].fillna(0)
                    elif col_raw in valid.columns:
                        comp_raw = valid[col_raw].fillna(0)
                    else:
                        comp_raw = pd.Series([0] * len(valid), index=valid.index)
                
                # Calculate per-AE and weighted average
                comp_ae = comp_raw / valid['HB061']
                comp_weighted = (comp_ae * valid['HA10']).sum()
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


def plot_housing_transport_parallel_fr_be(result_df_fr, result_df_be, dirs):
    """Create parallel visualization of housing+transport for France and Belgium."""
    print("\n=== CREATING PARALLEL FR/BE HOUSING + TRANSPORT VISUALIZATION ===")
    
    if result_df_fr.empty or result_df_be.empty:
        print("ERROR: Missing data for FR or BE")
        return
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    urbanization_order = [
        'Sparsely populated (<100/km²)',
        'Intermediate (100-499/km²)',
        'Densely populated (≥500/km²)'
    ]
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    housing_components = ['Actual Rentals', 'Imputed Rentals', 'Utility bills']
    transport_components = ['Purchase of Vehicles', 'Operation of Personal Transport', 'Transport Services']
    other_components = ['Food & Beverage', 'Health', 'Education']
    
    housing_color = '#fc8d62'
    transport_color = '#b3de69'
    other_colors = ['#8dd3c7', '#ffffb3', '#bebada']
    housing_hatches = ['', '///', 'xxx']
    transport_hatches = ['', '///', 'xxx']
    
    # Calculate max y-value for synchronized scaling
    max_y_absolute = 0
    for result_df in [result_df_fr, result_df_be]:
        all_urban_data = result_df[result_df['urbanization'].isin(urbanization_order)]
        if not all_urban_data.empty:
            for component in housing_components + transport_components + other_components:
                if component in all_urban_data.columns:
                    max_y_absolute = max(max_y_absolute, all_urban_data[component].max())
        if 'residual' in result_df.columns:
            max_y_absolute = max(max_y_absolute, result_df['residual'].max())
    max_y_absolute *= 1.15
    
    fig = plt.figure(figsize=(16, 10))
    
    countries = [('France (FR)', result_df_fr, 0), ('Belgium (BE)', result_df_be, 1)]
    axes_abs = []
    
    for country_name, result_df, country_idx in countries:
        for urban_idx, urban_label in enumerate(urbanization_order):
            # ABSOLUTE VALUES ROW
            ax_abs = plt.subplot(2, 6, country_idx * 3 + urban_idx + 1)
            axes_abs.append(ax_abs)
            
            urban_data = result_df[result_df['urbanization'] == urban_label].copy()
            urban_data['decile'] = pd.Categorical(urban_data['decile'], categories=decile_order, ordered=True)
            urban_data = urban_data.sort_values('decile')
            
            if urban_data.empty:
                ax_abs.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_abs.transAxes)
                ax_abs.set_visible(False)
                continue
            
            x_pos = np.arange(len(urban_data))
            bottom = np.zeros(len(urban_data))
            
            for h_idx, component in enumerate(housing_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    ax_abs.bar(x_pos, values, bottom=bottom, label=f'{component}',
                          color=housing_color, hatch=housing_hatches[h_idx % len(housing_hatches)],
                          edgecolor='white', linewidth=0.5, alpha=0.85)
                    bottom += values
            
            for t_idx, component in enumerate(transport_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    ax_abs.bar(x_pos, values, bottom=bottom, label=f'{component}',
                          color=transport_color, hatch=transport_hatches[t_idx % len(transport_hatches)],
                          edgecolor='white', linewidth=0.5, alpha=0.85)
                    bottom += values
            
            for o_idx, component in enumerate(other_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    ax_abs.bar(x_pos, values, bottom=bottom, label=component,
                          color=other_colors[o_idx % len(other_colors)], edgecolor='white', linewidth=0.5, alpha=0.85)
                    bottom += values
            
            residual_values = urban_data['residual'].fillna(0).values
            residual_values = np.maximum(residual_values, 0)
            ax_abs.bar(x_pos, residual_values, bottom=bottom, label='Other (Residual)',
                  color='#d3d3d3', edgecolor='white', linewidth=0.5, alpha=0.85)
            
            ax_abs.set_ylabel('Mean Consumption (PPS)' if country_idx == 0 else '', fontsize=9, fontweight='bold')
            ax_abs.set_title(f'{country_name} - {urban_label}', fontsize=10, fontweight='bold')
            ax_abs.set_ylim(0, max_y_absolute)
            ax_abs.grid(True, alpha=0.3, axis='y')
            ax_abs.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000):.0f}k' if x >= 1000 else f'{int(x)}'))
            ax_abs.set_xticklabels([])
            
            # PERCENTAGE VALUES ROW
            ax_pct = plt.subplot(2, 6, country_idx * 3 + urban_idx + 7)
            
            bottom_pct = np.zeros(len(urban_data))
            total_consumption = urban_data['total_consumption'].fillna(0).values
            total_consumption = np.where(total_consumption > 0, total_consumption, 1)
            
            for h_idx, component in enumerate(housing_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    percentages = (values / total_consumption) * 100
                    ax_pct.bar(x_pos, percentages, bottom=bottom_pct, label=f'{component}',
                          color=housing_color, hatch=housing_hatches[h_idx % len(housing_hatches)],
                          edgecolor='white', linewidth=0.5, alpha=0.85)
                    bottom_pct += percentages
            
            for t_idx, component in enumerate(transport_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    percentages = (values / total_consumption) * 100
                    ax_pct.bar(x_pos, percentages, bottom=bottom_pct, label=f'{component}',
                          color=transport_color, hatch=transport_hatches[t_idx % len(transport_hatches)],
                          edgecolor='white', linewidth=0.5, alpha=0.85)
                    bottom_pct += percentages
            
            for o_idx, component in enumerate(other_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    percentages = (values / total_consumption) * 100
                    ax_pct.bar(x_pos, percentages, bottom=bottom_pct, label=component,
                          color=other_colors[o_idx % len(other_colors)], edgecolor='white', linewidth=0.5, alpha=0.85)
                    bottom_pct += percentages
            
            residual_values = urban_data['residual'].fillna(0).values
            residual_values = np.maximum(residual_values, 0)
            residual_percentages = (residual_values / total_consumption) * 100
            ax_pct.bar(x_pos, residual_percentages, bottom=bottom_pct, label='Other (Residual)',
                  color='#d3d3d3', edgecolor='white', linewidth=0.5, alpha=0.85)
            
            ax_pct.set_ylabel('% of Total Consumption' if country_idx == 0 else '', fontsize=9, fontweight='bold')
            ax_pct.set_ylim(0, 100)
            ax_pct.set_xticks(x_pos)
            ax_pct.set_xticklabels(urban_data['decile'].values, fontsize=8)
            ax_pct.set_xlabel('Decile', fontsize=8)
            ax_pct.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Housing + Transport Consumption: France vs Belgium (2020)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    handles, labels = axes_abs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=6, fontsize=9, framealpha=0.95)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])
    
    output = os.path.join(graphs_dir, 'FR_BE_housing_transport_parallel.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("HBS PARALLEL COMPARISON - FRANCE vs BELGIUM 2020")
    print("="*80)
    
    dirs = setup_directories()
    pps_df = load_pps_data(dirs)
    
    # Load France
    print("\n--- PROCESSING FRANCE ---")
    df_fr = load_country_2020('FR')
    if df_fr.empty:
        print("ERROR: Could not load France data")
        return
    
    df_fr = calculate_consumption_in_pps(df_fr, pps_df)
    if 'EUR_HE00_pps' not in df_fr.columns:
        df_fr['EUR_HE00_pps'] = df_fr['EUR_HE00']
    
    df_fr = assign_simple_deciles(df_fr)
    all_components_fr = calculate_all_components_by_decile_urbanization(df_fr)
    
    # Load Belgium
    print("\n--- PROCESSING BELGIUM ---")
    df_be = load_country_2020('BE')
    if df_be.empty:
        print("ERROR: Could not load Belgium data")
        return
    
    df_be = calculate_consumption_in_pps(df_be, pps_df)
    if 'EUR_HE00_pps' not in df_be.columns:
        df_be['EUR_HE00_pps'] = df_be['EUR_HE00']
    
    df_be = assign_simple_deciles(df_be)
    all_components_be = calculate_all_components_by_decile_urbanization(df_be)
    
    # Create parallel visualization
    print("\n--- CREATING VISUALIZATION ---")
    if not all_components_fr.empty and not all_components_be.empty:
        plot_housing_transport_parallel_fr_be(all_components_fr, all_components_be, dirs)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
