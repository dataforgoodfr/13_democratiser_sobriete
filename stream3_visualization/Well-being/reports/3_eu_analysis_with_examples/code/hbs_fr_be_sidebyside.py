"""
HBS Analysis - France vs Belgium Side-by-Side Comparison
Creates side-by-side housing + transport visualizations for FR and BE.
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


def load_france_2020():
    """Load HBS data for France 2020 only."""
    print("\n=== LOADING FRANCE 2020 DATA ===")
    
    external_hbs_base = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/HBS"
    folder_2020 = os.path.join(external_hbs_base, "HBS2020/HBS2020")
    
    if not os.path.exists(folder_2020):
        print(f"ERROR: Directory not found: {folder_2020}")
        return pd.DataFrame()
    
    hh_files = glob.glob(os.path.join(folder_2020, "HBS_HH_*.xlsx"))
    fr_hh_files = [f for f in hh_files if 'FR' in os.path.basename(f)]
    
    if not fr_hh_files:
        print("ERROR: No France household files found")
        return pd.DataFrame()
    
    filepath = fr_hh_files[0]
    
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


def calculate_all_components_by_decile_urbanization(df):
    """Calculate ALL components (housing, transport, food, health, education) AND residual."""
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
            
            denominator = valid['HA10'].sum()
            
            consumption_ae = valid[consumption_col] / valid['HB061']
            total_weighted = (consumption_ae * valid['HA10']).sum()
            total_value = total_weighted / denominator if denominator > 0 else 0
            
            component_values = {}
            component_sum = 0
            
            for code, component_name in components.items():
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
                
                comp_ae = comp_raw / valid['HB061']
                comp_weighted = (comp_ae * valid['HA10']).sum()
                comp_value = comp_weighted / denominator if denominator > 0 else 0
                
                component_values[component_name] = comp_value
                component_sum += comp_value
            
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


def plot_housing_transport_sidebyside_absolute(result_df_fr, result_df_be, dirs):
    """Create side-by-side absolute value charts for FR and BE."""
    print("\n=== CREATING SIDE-BY-SIDE ABSOLUTE HOUSING + TRANSPORT VISUALIZATION ===")
    
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
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Housing + Transport Consumption by Income Decile and Urbanization: France vs Belgium (2020)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    for country_idx, (country_name, result_df) in enumerate([('France (FR)', result_df_fr), ('Belgium (BE)', result_df_be)]):
        for urban_idx, urban_label in enumerate(urbanization_order):
            ax = axes[urban_idx, country_idx]
            
            urban_data = result_df[result_df['urbanization'] == urban_label].copy()
            urban_data['decile'] = pd.Categorical(urban_data['decile'], categories=decile_order, ordered=True)
            urban_data = urban_data.sort_values('decile')
            
            if urban_data.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_visible(False)
                continue
            
            x_pos = np.arange(len(urban_data))
            bottom = np.zeros(len(urban_data))
            
            # Plot housing components
            for h_idx, component in enumerate(housing_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    ax.bar(x_pos, values, bottom=bottom, label=f'{component}',
                          color=housing_color, hatch=housing_hatches[h_idx % len(housing_hatches)],
                          edgecolor='white', linewidth=0.8, alpha=0.85)
                    # Add value labels
                    for i, val in enumerate(values):
                        if val > 50:  # Only show label if segment is large enough
                            ax.text(i, bottom[i] + val/2, f'{int(val)}', ha='center', va='center', fontsize=6, color='black')
                    bottom += values
            
            # Plot transport components
            for t_idx, component in enumerate(transport_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    ax.bar(x_pos, values, bottom=bottom, label=f'{component}',
                          color=transport_color, hatch=transport_hatches[t_idx % len(transport_hatches)],
                          edgecolor='white', linewidth=0.8, alpha=0.85)
                    # Add value labels
                    for i, val in enumerate(values):
                        if val > 50:  # Only show label if segment is large enough
                            ax.text(i, bottom[i] + val/2, f'{int(val)}', ha='center', va='center', fontsize=6, color='black')
                    bottom += values
            
            # Plot other components
            for o_idx, component in enumerate(other_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    ax.bar(x_pos, values, bottom=bottom, label=component,
                          color=other_colors[o_idx % len(other_colors)], edgecolor='white', linewidth=0.8, alpha=0.85)
                    # Add value labels
                    for i, val in enumerate(values):
                        if val > 50:  # Only show label if segment is large enough
                            ax.text(i, bottom[i] + val/2, f'{int(val)}', ha='center', va='center', fontsize=6, color='black')
                    bottom += values
            
            # Plot residual
            residual_values = urban_data['residual'].fillna(0).values
            residual_values = np.maximum(residual_values, 0)
            ax.bar(x_pos, residual_values, bottom=bottom, label='Other (Residual)',
                  color='#d3d3d3', edgecolor='white', linewidth=0.8, alpha=0.85)
            # Add value labels for residual
            for i, val in enumerate(residual_values):
                if val > 50:  # Only show label if segment is large enough
                    ax.text(i, bottom[i] + val/2, f'{int(val)}', ha='center', va='center', fontsize=6, color='black')
            
            ax.set_ylabel('Mean Consumption (PPS)', fontsize=10, fontweight='bold')
            ax.set_title(f'{country_name} - {urban_label}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(urban_data['decile'].values)
            ax.set_xlabel('Income Decile', fontsize=10, fontweight='bold')
    
    # Get handles and labels from the first axis for the legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', fontsize=8, framealpha=0.95, bbox_to_anchor=(1, 0.5), ncol=1)
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'FR_BE_housing_transport_absolute_sidebyside.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_housing_transport_sidebyside_percentage(result_df_fr, result_df_be, dirs):
    """Create side-by-side percentage charts for FR and BE."""
    print("\n=== CREATING SIDE-BY-SIDE PERCENTAGE HOUSING + TRANSPORT VISUALIZATION ===")
    
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
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Housing + Transport (% of Total) by Income Decile and Urbanization: France vs Belgium (2020)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    for country_idx, (country_name, result_df) in enumerate([('France (FR)', result_df_fr), ('Belgium (BE)', result_df_be)]):
        for urban_idx, urban_label in enumerate(urbanization_order):
            ax = axes[urban_idx, country_idx]
            
            urban_data = result_df[result_df['urbanization'] == urban_label].copy()
            urban_data['decile'] = pd.Categorical(urban_data['decile'], categories=decile_order, ordered=True)
            urban_data = urban_data.sort_values('decile')
            
            if urban_data.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_visible(False)
                continue
            
            x_pos = np.arange(len(urban_data))
            bottom = np.zeros(len(urban_data))
            
            total_consumption = urban_data['total_consumption'].fillna(0).values
            total_consumption = np.where(total_consumption > 0, total_consumption, 1)
            
            # Plot housing components
            for h_idx, component in enumerate(housing_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    percentages = (values / total_consumption) * 100
                    ax.bar(x_pos, percentages, bottom=bottom, label=f'{component}',
                          color=housing_color, hatch=housing_hatches[h_idx % len(housing_hatches)],
                          edgecolor='white', linewidth=0.8, alpha=0.85)
                    # Add percentage labels
                    for i, pct in enumerate(percentages):
                        if pct > 2:  # Only show label if segment is large enough
                            ax.text(i, bottom[i] + pct/2, f'{pct:.1f}%', ha='center', va='center', fontsize=7)
                    bottom += percentages
            
            # Plot transport components
            for t_idx, component in enumerate(transport_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    percentages = (values / total_consumption) * 100
                    ax.bar(x_pos, percentages, bottom=bottom, label=f'{component}',
                          color=transport_color, hatch=transport_hatches[t_idx % len(transport_hatches)],
                          edgecolor='white', linewidth=0.8, alpha=0.85)
                    # Add percentage labels
                    for i, pct in enumerate(percentages):
                        if pct > 2:  # Only show label if segment is large enough
                            ax.text(i, bottom[i] + pct/2, f'{pct:.1f}%', ha='center', va='center', fontsize=7)
                    bottom += percentages
            
            # Plot other components
            for o_idx, component in enumerate(other_components):
                if component in urban_data.columns:
                    values = urban_data[component].fillna(0).values
                    percentages = (values / total_consumption) * 100
                    ax.bar(x_pos, percentages, bottom=bottom, label=component,
                          color=other_colors[o_idx % len(other_colors)], edgecolor='white', linewidth=0.8, alpha=0.85)
                    # Add percentage labels
                    for i, pct in enumerate(percentages):
                        if pct > 2:  # Only show label if segment is large enough
                            ax.text(i, bottom[i] + pct/2, f'{pct:.1f}%', ha='center', va='center', fontsize=7)
                    bottom += percentages
            
            # Plot residual
            residual_values = urban_data['residual'].fillna(0).values
            residual_values = np.maximum(residual_values, 0)
            residual_percentages = (residual_values / total_consumption) * 100
            ax.bar(x_pos, residual_percentages, bottom=bottom, label='Other (Residual)',
                  color='#d3d3d3', edgecolor='white', linewidth=0.8, alpha=0.85)
            # Add percentage labels for residual
            for i, pct in enumerate(residual_percentages):
                if pct > 2:  # Only show label if segment is large enough
                    ax.text(i, bottom[i] + pct/2, f'{pct:.1f}%', ha='center', va='center', fontsize=7)
            
            ax.set_ylabel('% of Total Consumption', fontsize=10, fontweight='bold')
            ax.set_title(f'{country_name} - {urban_label}', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(urban_data['decile'].values)
            ax.set_xlabel('Income Decile', fontsize=10, fontweight='bold')
    
    # Get handles and labels from the first axis for the legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', fontsize=8, framealpha=0.95, bbox_to_anchor=(1, 0.5), ncol=1)
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'FR_BE_housing_transport_percentage_sidebyside.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()



def plot_income_comparison_by_decile(df_fr, df_be, dirs):
    """Create diagnostic chart comparing equivalized income by decile for FR vs BE."""
    print("\n=== CREATING INCOME COMPARISON CHART (FR vs BE) ===")
    
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    decile_order = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    
    # Calculate mean equivalized income by decile
    data_to_plot = []
    
    for country_name, df in [('France', df_fr), ('Belgium', df_be)]:
        df_temp = df[df['income_decile'].notna()].copy()
        df_temp['EUR_HH099'] = pd.to_numeric(df_temp['EUR_HH099'], errors='coerce')
        df_temp['HB061'] = pd.to_numeric(df_temp['HB061'], errors='coerce')
        df_temp['HA10'] = pd.to_numeric(df_temp['HA10'], errors='coerce')
        
        valid = df_temp[(df_temp['EUR_HH099'].notna()) & (df_temp['HB061'].notna()) & 
                       (df_temp['HB061'] > 0) & (df_temp['HA10'].notna())].copy()
        
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
    
    ax1.bar(x - width/2, fr_data, width, label='France', color='#8dd3c7', alpha=0.8, edgecolor='black')
    ax1.bar(x + width/2, be_data, width, label='Belgium', color='#fb8072', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Equivalized Income (EUR)', fontsize=11, fontweight='bold')
    ax1.set_title('Mean Equivalized Income by Decile', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(decile_order)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000):.0f}k'))
    
    # Plot 2: Difference (BE - FR)
    diff = be_data - fr_data
    colors = ['green' if x > 0 else 'red' for x in diff]
    
    ax2.bar(decile_order, diff, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Income Decile', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Income Difference (EUR)', fontsize=11, fontweight='bold')
    ax2.set_title('Belgium Income - France Income (Green = BE higher)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000):.0f}k'))
    
    plt.tight_layout()
    
    output = os.path.join(graphs_dir, 'DIAGNOSTIC_income_comparison_FR_BE.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("HBS SIDE-BY-SIDE COMPARISON - FRANCE vs BELGIUM 2020")
    print("="*80)
    
    dirs = setup_directories()
    pps_df = load_pps_data(dirs)
    
    # Process France
    print("\n--- PROCESSING FRANCE ---")
    df_fr = load_france_2020()
    if df_fr.empty:
        print("ERROR: Could not load France data")
        return
    
    df_fr = calculate_consumption_in_pps(df_fr, pps_df)
    if 'EUR_HE00_pps' not in df_fr.columns:
        df_fr['EUR_HE00_pps'] = df_fr['EUR_HE00']
    
    df_fr = assign_simple_deciles(df_fr)
    all_components_fr = calculate_all_components_by_decile_urbanization(df_fr)
    
    if all_components_fr.empty:
        print("ERROR: Could not calculate components for France")
        return
    
    # Process Belgium
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
    
    if all_components_be.empty:
        print("ERROR: Could not calculate components for Belgium")
        return
    
    # Create visualizations
    print("\n--- CREATING VISUALIZATIONS ---")
    plot_housing_transport_sidebyside_absolute(all_components_fr, all_components_be, dirs)
    plot_housing_transport_sidebyside_percentage(all_components_fr, all_components_be, dirs)
    
    # Create income comparison diagnostic
    print("\n--- CREATING INCOME COMPARISON DIAGNOSTIC ---")
    plot_income_comparison_by_decile(df_fr, df_be, dirs)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
