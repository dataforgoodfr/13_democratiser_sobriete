"""
Create Visualizations for Sensitivity Analysis Results
=======================================================

This script generates comprehensive visualizations comparing neutrality years
and country rankings across different budget scenarios.

Generates:
- Scatter plots comparing neutrality years (Base vs Alternatives)
- Rank change visualizations
- Distribution plots
- Top movers/losers charts
- Summary heatmaps

For each combination of:
- Carbon Budget Allocation (Responsibility, Capability)
- Probability (50%, 67%)
- Emissions Scope (Territory, Consumption)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from pathlib import Path

# Try to import pycountry for ISO2 to ISO3 mapping
try:
    import pycountry
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False

# Configuration
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output', 'Sensitivity check'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output', 'Sensitivity check', 'Visuals'))

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color scheme
COLORS = {
    'Base_2025': '#3498db',      # Blue
    'Alternative_2018': '#e74c3c',  # Red
    'Alternative_2021': '#f39c12'   # Orange
}

def load_data():
    """Load the sensitivity analysis results."""
    print("Loading data...")
    
    full_results = pd.read_csv(os.path.join(INPUT_DIR, 'sensitivity_analysis_full_results.csv'))
    comparison = pd.read_csv(os.path.join(INPUT_DIR, 'sensitivity_neutrality_year_comparison.csv'))
    summary = pd.read_csv(os.path.join(INPUT_DIR, 'sensitivity_summary_statistics.csv'))
    
    print(f"Loaded full results: {full_results.shape}")
    print(f"Loaded comparison: {comparison.shape}")
    print(f"Loaded summary: {summary.shape}")
    
    return full_results, comparison, summary


def create_scatter_comparison(comparison_df, scope, distribution, warming, probability):
    """Create scatter plots comparing neutrality years across scenarios."""
    
    # Filter data
    df = comparison_df[
        (comparison_df['Emissions_scope'] == scope) &
        (comparison_df['Distribution_method'] == distribution) &
        (comparison_df['Warming_scenario'] == warming) &
        (comparison_df['Probability'] == probability)
    ].copy()
    
    if len(df) == 0:
        print(f"No data for {scope}, {distribution}, {warming}, {probability}")
        return
    
    # Remove countries with NaN values
    df = df.dropna(subset=['Base_2025', 'Alternative_2018', 'Alternative_2021'])
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Base 2025 vs Alternative 2018
    ax1 = axes[0]
    ax1.scatter(df['Base_2025'], df['Alternative_2018'], alpha=0.6, s=50, color=COLORS['Alternative_2018'])
    
    # Add diagonal line (y=x)
    min_val = min(df['Base_2025'].min(), df['Alternative_2018'].min())
    max_val = max(df['Base_2025'].max(), df['Alternative_2018'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Equal')
    
    ax1.set_xlabel('Neutrality Year - Base 2025', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Neutrality Year - Alternative 2018', fontsize=12, fontweight='bold')
    ax1.set_title('Base 2025 vs Alternative 2018', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Base 2025 vs Alternative 2021
    ax2 = axes[1]
    ax2.scatter(df['Base_2025'], df['Alternative_2021'], alpha=0.6, s=50, color=COLORS['Alternative_2021'])
    
    # Add diagonal line
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Equal')
    
    ax2.set_xlabel('Neutrality Year - Base 2025', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Neutrality Year - Alternative 2021', fontsize=12, fontweight='bold')
    ax2.set_title('Base 2025 vs Alternative 2021', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Neutrality Year Comparison\n{scope} | {distribution} | {warming} | {probability}',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    filename = f'scatter_comparison_{scope}_{distribution}_{warming}_{probability}.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {filename}")


def create_year_difference_chart(comparison_df, scope, distribution, warming, probability):
    """Create chart showing years lost/gained by replacing historical baselines with 2025.
    
    Framing: By replacing 2018/2021 values with 2025 ones, countries lose X years.
    Values shown as NEGATIVE = years LOST (red) - had more time with older baseline
    Values shown as POSITIVE = years GAINED (green) - have more time with 2025 baseline
    """
    
    # Filter data
    df = comparison_df[
        (comparison_df['Emissions_scope'] == scope) &
        (comparison_df['Distribution_method'] == distribution) &
        (comparison_df['Warming_scenario'] == warming) &
        (comparison_df['Probability'] == probability)
    ].copy()
    
    if len(df) == 0:
        return
    
    # Remove NaN values and "All" aggregates
    df = df.dropna(subset=['Base_2025', 'Alternative_2018', 'Alternative_2021'])
    df = df[df['Country'] != 'All']
    df = df[~df['Country'].str.contains('All', case=False, na=False)]
    
    # Calculate years change: INVERTED so negative = lost, positive = gained
    # Base_2025 - Alternative = negative means we lost time (2025 is earlier)
    df['Years_change_vs_2018'] = df['Base_2025'] - df['Alternative_2018']  # Negative = lost time
    df['Years_change_vs_2021'] = df['Base_2025'] - df['Alternative_2021']  # Negative = lost time
    
    # Filter out countries that overshot in both scenarios (1970)
    df = df[(df['Base_2025'] > 1970) | (df['Alternative_2018'] > 1970)]
    
    if len(df) == 0:
        print(f"No valid data for {scope}, {distribution}, {warming}, {probability}")
        return
    
    # Calculate average change across both baselines for sorting
    df['Avg_change'] = (df['Years_change_vs_2018'] + df['Years_change_vs_2021']) / 2
    
    # Sort by average change (most negative at top = most time lost, most positive at bottom = most time gained)
    df_sorted = df.sort_values('Avg_change', ascending=True)
    
    # Get top 10 losers (most negative average) and top 10 gainers (most positive average)
    top_losers = df_sorted.head(10)  # Most years LOST (most negative values)
    top_gainers = df_sorted.tail(10)  # Most years GAINED (most positive values)
    
    # Create figure with clear separation - increased size for readability
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.4})
    
    # ==================== TOP SECTION: Countries LOSING time ====================
    y_pos_top = np.arange(len(top_losers))
    width = 0.35
    
    # Color bars based on value: red for negative (lost), green for positive (gained)
    colors_2018_top = ['#d73027' if v < 0 else '#1a9850' for v in top_losers['Years_change_vs_2018']]
    colors_2021_top = ['#d73027' if v < 0 else '#1a9850' for v in top_losers['Years_change_vs_2021']]
    
    for i, (country, v2018, v2021) in enumerate(zip(top_losers['Country'], 
                                                      top_losers['Years_change_vs_2018'], 
                                                      top_losers['Years_change_vs_2021'])):
        ax1.barh(i + width/2, v2018, width, color=colors_2018_top[i], alpha=0.8, edgecolor='white', linewidth=0.5)
        ax1.barh(i - width/2, v2021, width, color=colors_2021_top[i], alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Add value labels for top section - position at end of each bar
    for i, (v2018, v2021) in enumerate(zip(top_losers['Years_change_vs_2018'], top_losers['Years_change_vs_2021'])):
        label_2018 = f'{int(v2018):+d}'
        label_2021 = f'{int(v2021):+d}'
        # Position labels outside the bar (to the left for negative, to the right for positive)
        offset = 0.8
        x_2018 = v2018 - offset if v2018 < 0 else v2018 + offset
        x_2021 = v2021 - offset if v2021 < 0 else v2021 + offset
        ha_2018 = 'right' if v2018 < 0 else 'left'
        ha_2021 = 'right' if v2021 < 0 else 'left'
        ax1.text(x_2018, i + width/2, label_2018,
                ha=ha_2018, va='center', fontsize=9, fontweight='bold')
        ax1.text(x_2021, i - width/2, label_2021,
                ha=ha_2021, va='center', fontsize=9, fontweight='bold', alpha=0.7)
    
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax1.set_yticks(y_pos_top)
    ax1.set_yticklabels(top_losers['Country'])
    ax1.set_xlabel('Years (negative = LOST, positive = GAINED)', fontsize=11, fontweight='bold')
    ax1.set_title(f'TOP 10: Countries LOSING the most time by using 2025 baseline\n'
                  f'{scope} | {distribution} | {warming} | {probability}',
                  fontsize=13, fontweight='bold', color='#d73027')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add padding to xlim for labels
    all_top_values = list(top_losers['Years_change_vs_2018']) + list(top_losers['Years_change_vs_2021'])
    top_min = min(all_top_values)
    top_max = max(all_top_values)
    ax1.set_xlim(top_min - 8, top_max + 8)
    ax1.axvspan(ax1.get_xlim()[0], 0, alpha=0.05, color='red')
    
    # Legend for top
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#d73027', alpha=0.8, label='2018 → 2025'),
                       Patch(facecolor='#d73027', alpha=0.5, label='2021 → 2025')]
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=9)
    
    # ==================== BOTTOM SECTION: Countries GAINING time ====================
    y_pos_bottom = np.arange(len(top_gainers))
    
    # Reverse order so most gained is at bottom
    top_gainers_rev = top_gainers.iloc[::-1]
    
    colors_2018_bottom = ['#d73027' if v < 0 else '#1a9850' for v in top_gainers_rev['Years_change_vs_2018']]
    colors_2021_bottom = ['#d73027' if v < 0 else '#1a9850' for v in top_gainers_rev['Years_change_vs_2021']]
    
    for i, (country, v2018, v2021) in enumerate(zip(top_gainers_rev['Country'], 
                                                      top_gainers_rev['Years_change_vs_2018'], 
                                                      top_gainers_rev['Years_change_vs_2021'])):
        ax2.barh(i + width/2, v2018, width, color=colors_2018_bottom[i], alpha=0.8, edgecolor='white', linewidth=0.5)
        ax2.barh(i - width/2, v2021, width, color=colors_2021_bottom[i], alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Add value labels for bottom section - position at end of each bar
    for i, (v2018, v2021) in enumerate(zip(top_gainers_rev['Years_change_vs_2018'], top_gainers_rev['Years_change_vs_2021'])):
        label_2018 = f'{int(v2018):+d}'
        label_2021 = f'{int(v2021):+d}'
        # Position labels outside the bar (to the right for positive, to the left for negative)
        offset = 0.8
        x_2018 = v2018 + offset if v2018 > 0 else v2018 - offset
        x_2021 = v2021 + offset if v2021 > 0 else v2021 - offset
        ha_2018 = 'left' if v2018 > 0 else 'right'
        ha_2021 = 'left' if v2021 > 0 else 'right'
        ax2.text(x_2018, i + width/2, label_2018,
                ha=ha_2018, va='center', fontsize=9, fontweight='bold')
        ax2.text(x_2021, i - width/2, label_2021,
                ha=ha_2021, va='center', fontsize=9, fontweight='bold', alpha=0.7)
    
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax2.set_yticks(y_pos_bottom)
    ax2.set_yticklabels(top_gainers_rev['Country'])
    ax2.set_xlabel('Years (negative = LOST, positive = GAINED)', fontsize=11, fontweight='bold')
    ax2.set_title(f'TOP 10: Countries GAINING the most time by using 2025 baseline',
                  fontsize=13, fontweight='bold', color='#1a9850')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add padding to xlim for labels
    all_bottom_values = list(top_gainers_rev['Years_change_vs_2018']) + list(top_gainers_rev['Years_change_vs_2021'])
    bottom_min = min(all_bottom_values)
    bottom_max = max(all_bottom_values)
    ax2.set_xlim(bottom_min - 8, bottom_max + 8)
    ax2.axvspan(0, ax2.get_xlim()[1], alpha=0.05, color='green')
    
    # Legend for bottom
    legend_elements2 = [Patch(facecolor='#1a9850', alpha=0.8, label='2018 → 2025'),
                        Patch(facecolor='#1a9850', alpha=0.5, label='2021 → 2025')]
    ax2.legend(handles=legend_elements2, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    filename = f'years_impact_{scope}_{distribution}_{warming}_{probability}.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {filename}")


def create_top_movers(comparison_df, scope, distribution, warming, probability):
    """Visualize countries with biggest neutrality year changes."""
    
    # Filter data
    df = comparison_df[
        (comparison_df['Emissions_scope'] == scope) &
        (comparison_df['Distribution_method'] == distribution) &
        (comparison_df['Warming_scenario'] == warming) &
        (comparison_df['Probability'] == probability)
    ].copy()
    
    if len(df) == 0:
        return
    
    # Remove NaN values and countries with 1970 (overshot)
    df = df.dropna(subset=['Diff_Alternative_2018', 'Diff_Alternative_2021'])
    df = df[df['Base_2025'] > 1970]  # Exclude countries that already overshot
    
    if len(df) == 0:
        print(f"No valid data (all countries overshot) for {scope}, {distribution}, {warming}, {probability}")
        return
    
    # Get top 15 biggest changes
    df['Max_Change'] = df[['Diff_Alternative_2018', 'Diff_Alternative_2021']].abs().max(axis=1)
    top_movers = df.nlargest(15, 'Max_Change')[['Country', 'Base_2025', 'Alternative_2018', 'Alternative_2021',
                                                   'Diff_Alternative_2018', 'Diff_Alternative_2021']].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(top_movers))
    width = 0.35
    
    # Plot bars showing the DIFFERENCE (in years) from Base 2025
    ax.bar(x_pos - width/2, top_movers['Diff_Alternative_2018'], width, 
           label='Difference: Alt 2018 - Base 2025', color=COLORS['Alternative_2018'], alpha=0.8)
    ax.bar(x_pos + width/2, top_movers['Diff_Alternative_2021'], width, 
           label='Difference: Alt 2021 - Base 2025', color=COLORS['Alternative_2021'], alpha=0.8)
    
    # Add value labels on bars
    for i, (diff_2018, diff_2021) in enumerate(zip(top_movers['Diff_Alternative_2018'], top_movers['Diff_Alternative_2021'])):
        # Label for 2018
        ax.text(i - width/2, diff_2018, f'+{int(diff_2018)}' if diff_2018 > 0 else f'{int(diff_2018)}',
                ha='center', va='bottom' if diff_2018 > 0 else 'top', fontsize=9, fontweight='bold')
        # Label for 2021
        ax.text(i + width/2, diff_2021, f'+{int(diff_2021)}' if diff_2021 > 0 else f'{int(diff_2021)}',
                ha='center', va='bottom' if diff_2021 > 0 else 'top', fontsize=9, fontweight='bold')
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Customize
    ax.set_ylabel('Years Added/Reduced to Neutrality Target', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 15 Countries with Biggest Changes in Neutrality Year\n{scope} | {distribution} | {warming} | {probability}\n(Positive = More Time, Negative = Less Time)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_movers['Country'], rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text box with base year context
    textstr = f'Base 2025 range: {int(top_movers["Base_2025"].min())} - {int(top_movers["Base_2025"].max())}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save
    filename = f'top_movers_{scope}_{distribution}_{warming}_{probability}.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {filename}")


def create_distribution_plot(comparison_df, scope, distribution, warming, probability):
    """Create distribution plots showing how neutrality years are distributed."""
    
    # Filter data
    df = comparison_df[
        (comparison_df['Emissions_scope'] == scope) &
        (comparison_df['Distribution_method'] == distribution) &
        (comparison_df['Warming_scenario'] == warming) &
        (comparison_df['Probability'] == probability)
    ].copy()
    
    if len(df) == 0:
        return
    
    # Remove NaN values
    df = df.dropna(subset=['Base_2025', 'Alternative_2018', 'Alternative_2021'])
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Histograms
    ax1 = axes[0]
    
    # Filter out 1970 (overshot countries) for better visualization
    df_valid = df[df['Base_2025'] > 1970]
    
    if len(df_valid) > 0:
        ax1.hist(df_valid['Base_2025'], bins=30, alpha=0.5, label='Base 2025', color=COLORS['Base_2025'])
        ax1.hist(df_valid['Alternative_2018'], bins=30, alpha=0.5, label='Alternative 2018', color=COLORS['Alternative_2018'])
        ax1.hist(df_valid['Alternative_2021'], bins=30, alpha=0.5, label='Alternative 2021', color=COLORS['Alternative_2021'])
        
        ax1.set_xlabel('Neutrality Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Countries', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Neutrality Years (Excluding Overshot Countries)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    else:
        ax1.text(0.5, 0.5, 'All countries have overshot their budgets', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
    
    # Plot 2: Box plots
    ax2 = axes[1]
    
    data_to_plot = [df['Base_2025'], df['Alternative_2018'], df['Alternative_2021']]
    labels = ['Base 2025', 'Alternative 2018', 'Alternative 2021']
    colors_list = [COLORS['Base_2025'], COLORS['Alternative_2018'], COLORS['Alternative_2021']]
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Neutrality Year', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution Statistics (Including Overshot Countries at 1970)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add count of overshot countries
    overshot_base = len(df[df['Base_2025'] == 1970])
    overshot_2018 = len(df[df['Alternative_2018'] == 1970])
    overshot_2021 = len(df[df['Alternative_2021'] == 1970])
    
    ax2.text(0.02, 0.98, f'Overshot countries:\nBase: {overshot_base} | 2018: {overshot_2018} | 2021: {overshot_2021}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle(f'Neutrality Year Distributions\n{scope} | {distribution} | {warming} | {probability}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    filename = f'distribution_{scope}_{distribution}_{warming}_{probability}.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {filename}")


def create_summary_heatmap(summary_df):
    """Create heatmap showing mean neutrality years across all scenarios."""
    
    # Pivot data for heatmap
    pivot_data = summary_df.pivot_table(
        index=['Emissions_scope', 'Distribution_method', 'Warming_scenario', 'Probability'],
        columns='Scenario_name',
        values='Mean_neutrality_year'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 16))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean Neutrality Year'})
    
    ax.set_title('Mean Neutrality Year Across All Scenarios', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scenario Parameters', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    filename = 'summary_heatmap_all_scenarios.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {filename}")


def create_iso2_to_iso3_mapping():
    """Create a mapping from ISO2 to ISO3 country codes from combined_data.csv."""
    try:
        # Load combined_data which has both ISO2 and ISO3 columns
        combined_data_path = os.path.join(INPUT_DIR, '..', 'combined_data.csv')
        combined_data = pd.read_csv(combined_data_path)
        
        if 'ISO2' in combined_data.columns and 'ISO3' in combined_data.columns:
            iso_mapping = combined_data[['ISO2', 'ISO3']].drop_duplicates()
            iso2_to_iso3 = dict(zip(iso_mapping['ISO2'], iso_mapping['ISO3']))
            print(f"Created ISO mapping with {len(iso2_to_iso3)} entries from combined_data.csv")
            return iso2_to_iso3
        else:
            print("Warning: ISO2 or ISO3 columns not found in combined_data.csv")
            return {}
    except Exception as e:
        print(f"Error loading ISO mapping from combined_data.csv: {e}")
        # Fallback to pycountry if available
        if HAS_PYCOUNTRY:
            mapping = {}
            for country in pycountry.countries:
                mapping[country.alpha_2] = country.alpha_3
            print(f"Using pycountry fallback with {len(mapping)} entries")
            return mapping
        else:
            print("Error: Could not create ISO mapping")
            return {}


def create_difference_maps(comparison_df, scope, distribution, warming, probability, iso2_to_iso3, global_color_range=None):
    """Create stacked world maps showing years lost by moving to 2025 baseline.
    
    Two maps stacked vertically with shared color scale:
    - Top: Years lost by replacing 2018 → 2025
    - Bottom: Years lost by replacing 2021 → 2025
    
    Args:
        global_color_range: tuple (min, max) for consistent color scale across all maps
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Filter data
    df = comparison_df[
        (comparison_df['Emissions_scope'] == scope) &
        (comparison_df['Distribution_method'] == distribution) &
        (comparison_df['Warming_scenario'] == warming) &
        (comparison_df['Probability'] == probability)
    ].copy()
    
    if len(df) == 0:
        print(f"No data for difference maps: {scope} | {distribution} | {warming} | {probability}")
        return
    
    # Remove NaN values and "All" aggregates
    df = df.dropna(subset=['Base_2025', 'Alternative_2018', 'Alternative_2021'])
    df = df[df['Country'] != 'All']
    df = df[~df['Country'].str.contains('All', case=False, na=False)]
    
    # Calculate years lost by switching to 2025 baseline
    # Positive = LOST time (had more time with older baseline)
    # Negative = GAINED time (have more time with 2025 baseline)
    # Years_lost = Alt - Base: if Alt (e.g. 2050) > Base (e.g. 2040), then lost 10 years by moving to 2025
    df['Years_lost_2018'] = df['Alternative_2018'] - df['Base_2025']
    df['Years_lost_2021'] = df['Alternative_2021'] - df['Base_2025']
    
    # Add ISO3 codes
    df['ISO3'] = df['ISO2'].map(iso2_to_iso3)
    
    # Use global color range if provided, otherwise calculate from this data
    if global_color_range:
        color_range_min, color_range_max = global_color_range
    else:
        all_diffs = pd.concat([df['Years_lost_2018'], df['Years_lost_2021']])
        color_range_min = all_diffs.min()
        color_range_max = all_diffs.max()
    
    # Diverging colorscale: Plotly maps zmin->0.0 and zmax->1.0 in colorscale
    # zmin is most negative (gained time) -> should be GREEN
    # zmax is most positive (lost time) -> should be RED
    # So: 0.0 = GREEN (min, negative, gained), 1.0 = RED (max, positive, lost)
    colorscale = [
        [0.0, '#1a9850'],    # Dark green (min value = most negative = most time gained)
        [0.25, '#91cf60'],   # Light green
        [0.45, '#d9ef8b'],   # Very light green
        [0.5, '#ffffbf'],    # White/yellow (neutral at midpoint)
        [0.55, '#fee08b'],   # Very light red
        [0.75, '#fc8d59'],   # Light red/orange
        [1.0, '#d73027']     # Dark red (max value = most positive = most time lost)
    ]
    
    # Create figure with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Years Lost by Replacing 2018 Baseline → 2025',
            f'Years Lost by Replacing 2021 Baseline → 2025'
        ),
        specs=[[{"type": "choropleth"}], [{"type": "choropleth"}]],
        vertical_spacing=0.05
    )
    
    # Map 1: 2018 → 2025
    fig.add_trace(
        go.Choropleth(
            locations=df['ISO3'],
            z=df['Years_lost_2018'],
            locationmode='ISO-3',
            colorscale=colorscale,
            zmin=color_range_min,
            zmax=color_range_max,
            marker_line_color='white',
            marker_line_width=0.5,
            hovertemplate="<b>%{text}</b><br>Years lost: %{z:.0f}<extra></extra>",
            text=df['Country'],
            colorbar=dict(
                title=dict(text="Years Lost", font=dict(size=12)),
                thickness=15,
                len=0.4,
                y=0.77,
                yanchor='middle'
            ),
            showscale=True
        ),
        row=1, col=1
    )
    
    # Map 2: 2021 → 2025
    fig.add_trace(
        go.Choropleth(
            locations=df['ISO3'],
            z=df['Years_lost_2021'],
            locationmode='ISO-3',
            colorscale=colorscale,
            zmin=color_range_min,
            zmax=color_range_max,
            marker_line_color='white',
            marker_line_width=0.5,
            hovertemplate="<b>%{text}</b><br>Years lost: %{z:.0f}<extra></extra>",
            text=df['Country'],
            colorbar=dict(
                title=dict(text="Years Lost", font=dict(size=12)),
                thickness=15,
                len=0.4,
                y=0.23,
                yanchor='middle'
            ),
            showscale=True
        ),
        row=2, col=1
    )
    
    # Update geos for both maps
    geo_settings = dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='white',
        projection_type='equirectangular',
        bgcolor='white',
        landcolor='lightgray',
        showlakes=False,
        center=dict(lat=20, lon=0),
        lataxis_range=[-55, 75],
        lonaxis_range=[-170, 180]
    )
    
    fig.update_geos(geo_settings, row=1, col=1)
    fig.update_geos(geo_settings, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        title=dict(
            text=f"Impact of Baseline Change on Neutrality Year<br>"
                 f"<sub>{scope} | {distribution} | {warming} | {probability}</sub>",
            font=dict(size=16, color="#2c3e50"),
            x=0.5
        ),
        margin={"r": 50, "t": 80, "l": 50, "b": 30},
        paper_bgcolor='#ffffff',
        font=dict(family="Arial, sans-serif", size=11, color="#2c3e50"),
    )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, color='#2c3e50')
    
    # Save
    filename = f'difference_maps_{scope}_{distribution}_{warming}_{probability}.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        fig.write_image(filepath, width=1200, height=900)
        print(f"Created: {filename}")
    except Exception as e:
        print(f"Warning: Could not save {filename} as PNG: {e}")


def create_world_map(full_results_df, scope, distribution, warming, probability, scenario_name, iso2_to_iso3):
    """Create a world map visualization showing neutrality years by country."""
    # This function is kept but not called in main() anymore
    pass
    
    # Create choropleth map
    fig = px.choropleth(
        df,
        locations="ISO3",
        locationmode='ISO-3',
        color="Neutrality_year_numeric",
        hover_name="Country",
        hover_data={
            "Neutrality_year": True,
            "Neutrality_year_numeric": False,
            "ISO3": False,
            "Starting_year_annual_emissions_MtCO2": ':.1f',
            "Country_budget_MtCO2": ':.1f'
        },
        color_continuous_scale=colorscale,
        range_color=[color_range_min, color_range_max],
        title=f"Zero Carbon Timeline: {scope} | {distribution} | {warming} | {probability}<br>{scenario_name}",
        labels={
            'Neutrality_year_numeric': 'Zero Carbon Year',
            'Starting_year_annual_emissions_MtCO2': 'Annual Emissions (Mt)',
            'Country_budget_MtCO2': 'Carbon Budget (Mt)'
        }
    )
    
    # Update traces for better styling
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.5,
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "Zero Carbon Year: %{customdata[0]}<br>" +
                      "Annual Emissions (Mt): %{customdata[1]:.1f}<br>" +
                      "Carbon Budget (Mt): %{customdata[2]:.1f}<extra></extra>",
        customdata=df[[
            'Neutrality_year', 'Starting_year_annual_emissions_MtCO2', 'Country_budget_MtCO2'
        ]].values
    )
    
    # Update geos for better map appearance
    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='white',
        projection_type='equirectangular',
        bgcolor='white',
        landcolor='lightgray',
        framecolor='white',
        showlakes=False,
        center=dict(lat=20, lon=0),
        lataxis_range=[-60, 80],
        lonaxis_range=[-180, 180]
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        width=1200,
        margin={"r": 100, "t": 80, "l": 100, "b": 50},
        geo=dict(
            projection_scale=1.0,
            showland=True,
            landcolor='lightgray',
            showocean=True,
            oceancolor='white',
            showlakes=False,
            showrivers=False,
            center=dict(lat=20, lon=0),
            lataxis_range=[-60, 80],
            lonaxis_range=[-180, 180]
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#2c3e50"
        ),
        title=dict(
            font=dict(size=16, color="#2c3e50", weight="bold"),
            x=0.5
        ),
        coloraxis_colorbar=dict(
            thickness=15,
            len=0.7,
            title=dict(
                text="Zero Carbon<br>Year",
                font=dict(size=12, color="#2c3e50"),
                side="right"
            )
        ),
        hoverlabel=dict(
            bgcolor="rgba(245, 245, 245, 0.9)",
            bordercolor="white",
            font=dict(color="black", size=11)
        )
    )
    
    # Save as PNG
    filename = f'world_map_{scope}_{distribution}_{warming}_{probability}_{scenario_name}.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        fig.write_image(filepath, width=1200, height=700)
        print(f"Created: {filename}")
    except Exception as e:
        print(f"Warning: Could not save {filename} as PNG. Make sure kaleido is installed: {e}")


def create_key_findings_visual(full_results_df, comparison_df):
    """Create a comprehensive visual highlighting key findings."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Filter to one scenario for analysis: Territory, Responsibility, 1.5°C, 50%
    scenario_filter = (
        (comparison_df['Emissions_scope'] == 'Territory') &
        (comparison_df['Distribution_method'] == 'Responsibility') &
        (comparison_df['Warming_scenario'] == '1.5°C') &
        (comparison_df['Probability'] == '50%')
    )
    df_scenario = comparison_df[scenario_filter].copy()
    df_scenario = df_scenario.dropna(subset=['Base_2025', 'Alternative_2018', 'Alternative_2021'])
    
    # 1. Overshot countries
    ax1 = fig.add_subplot(gs[0, 0])
    overshot_base = len(df_scenario[df_scenario['Base_2025'] == 1970])
    overshot_2018 = len(df_scenario[df_scenario['Alternative_2018'] == 1970])
    overshot_2021 = len(df_scenario[df_scenario['Alternative_2021'] == 1970])
    
    bars = ax1.bar(['Base 2025', 'Alt 2018', 'Alt 2021'], 
                   [overshot_base, overshot_2018, overshot_2021],
                   color=[COLORS['Base_2025'], COLORS['Alternative_2018'], COLORS['Alternative_2021']])
    ax1.set_ylabel('Number of Countries', fontweight='bold')
    ax1.set_title('Countries That Overshot Budget', fontweight='bold', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Mean neutrality year
    ax2 = fig.add_subplot(gs[0, 1])
    df_valid = df_scenario[df_scenario['Base_2025'] > 1970]
    if len(df_valid) > 0:
        mean_base = df_valid['Base_2025'].mean()
        mean_2018 = df_valid['Alternative_2018'].mean()
        mean_2021 = df_valid['Alternative_2021'].mean()
        
        bars = ax2.bar(['Base 2025', 'Alt 2018', 'Alt 2021'], 
                       [mean_base, mean_2018, mean_2021],
                       color=[COLORS['Base_2025'], COLORS['Alternative_2018'], COLORS['Alternative_2021']])
        ax2.set_ylabel('Mean Neutrality Year', fontweight='bold')
        ax2.set_title('Average Neutrality Year\n(Excl. Overshot)', fontweight='bold', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Countries reaching neutrality by decade
    ax3 = fig.add_subplot(gs[0, 2])
    decades = ['2020s', '2030s', '2040s', '2050s', '2060s', '2070s', '2080s', '2090s', '2100']
    
    def count_by_decade(series):
        counts = []
        for i, decade in enumerate([(2020, 2030), (2030, 2040), (2040, 2050), (2050, 2060), 
                                     (2060, 2070), (2070, 2080), (2080, 2090), (2090, 2100), (2100, 2110)]):
            count = len(series[(series > decade[0]) & (series <= decade[1])])
            counts.append(count)
        return counts
    
    if len(df_valid) > 0:
        x = np.arange(len(decades))
        width = 0.25
        
        ax3.bar(x - width, count_by_decade(df_valid['Base_2025']), width, label='Base 2025', 
                color=COLORS['Base_2025'], alpha=0.8)
        ax3.bar(x, count_by_decade(df_valid['Alternative_2018']), width, label='Alt 2018', 
                color=COLORS['Alternative_2018'], alpha=0.8)
        ax3.bar(x + width, count_by_decade(df_valid['Alternative_2021']), width, label='Alt 2021', 
                color=COLORS['Alternative_2021'], alpha=0.8)
        
        ax3.set_ylabel('Number of Countries', fontweight='bold')
        ax3.set_title('Countries by Neutrality Decade', fontweight='bold', fontsize=11)
        ax3.set_xticks(x)
        ax3.set_xticklabels(decades, rotation=45, ha='right', fontsize=9)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Top 10 countries with biggest increase in neutrality year
    ax4 = fig.add_subplot(gs[1, :])
    df_increase = df_scenario[df_scenario['Base_2025'] > 1970].copy()
    if len(df_increase) > 0:
        df_increase['Max_increase'] = df_increase[['Diff_Alternative_2018', 'Diff_Alternative_2021']].max(axis=1)
        top_increase = df_increase.nlargest(10, 'Max_increase')
        
        x_pos = np.arange(len(top_increase))
        width = 0.35
        
        ax4.barh(x_pos, top_increase['Diff_Alternative_2018'], width, 
                label='vs 2018', color=COLORS['Alternative_2018'], alpha=0.8)
        ax4.barh(x_pos + width, top_increase['Diff_Alternative_2021'], width, 
                label='vs 2021', color=COLORS['Alternative_2021'], alpha=0.8)
        
        ax4.set_yticks(x_pos + width/2)
        ax4.set_yticklabels(top_increase['Country'])
        ax4.set_xlabel('Years Added to Neutrality Target', fontweight='bold')
        ax4.set_title('Top 10 Countries: Biggest Increase in Neutrality Year\n(More time with larger budgets)', 
                      fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Sample countries comparison
    ax5 = fig.add_subplot(gs[2, :])
    sample_countries = ['China', 'India', 'Brazil', 'Indonesia', 'South Africa', 
                        'Mexico', 'Turkey', 'Argentina', 'Egypt', 'Nigeria']
    df_sample = df_scenario[df_scenario['Country'].isin(sample_countries)].copy()
    
    if len(df_sample) > 0:
        x_pos = np.arange(len(df_sample))
        width = 0.25
        
        ax5.bar(x_pos - width, df_sample['Base_2025'], width, label='Base 2025', 
                color=COLORS['Base_2025'], alpha=0.8)
        ax5.bar(x_pos, df_sample['Alternative_2018'], width, label='Alt 2018', 
                color=COLORS['Alternative_2018'], alpha=0.8)
        ax5.bar(x_pos + width, df_sample['Alternative_2021'], width, label='Alt 2021', 
                color=COLORS['Alternative_2021'], alpha=0.8)
        
        ax5.set_ylabel('Neutrality Year', fontweight='bold')
        ax5.set_title('Sample Countries: Neutrality Year Comparison', fontweight='bold', fontsize=12)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(df_sample['Country'], rotation=45, ha='right')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle('Key Findings: Sensitivity Analysis\nTerritory | Responsibility | 1.5°C | 50%',
                 fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    filename = 'key_findings_overview.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created: {filename}")


def main():
    """Main execution function."""
    print("="*80)
    print("CREATING SENSITIVITY ANALYSIS VISUALIZATIONS")
    print("="*80)
    
    # Load data
    full_results, comparison, summary = load_data()
    
    # Create ISO2 to ISO3 mapping
    iso2_to_iso3 = create_iso2_to_iso3_mapping()
    print(f"Created ISO2 to ISO3 mapping with {len(iso2_to_iso3)} entries")
    
    # Calculate GLOBAL color range for all difference maps (use actual min/max)
    comparison_clean = comparison.dropna(subset=['Base_2025', 'Alternative_2018', 'Alternative_2021'])
    comparison_clean = comparison_clean[comparison_clean['Country'] != 'All']
    comparison_clean = comparison_clean[~comparison_clean['Country'].str.contains('All', case=False, na=False)]
    all_diffs_2018 = comparison_clean['Alternative_2018'] - comparison_clean['Base_2025']
    all_diffs_2021 = comparison_clean['Alternative_2021'] - comparison_clean['Base_2025']
    all_diffs = pd.concat([all_diffs_2018, all_diffs_2021])
    global_min = all_diffs.min()
    global_max = all_diffs.max()
    global_color_range = (global_min, global_max)
    print(f"Global color range for maps: {global_color_range[0]:.0f} to {global_color_range[1]:.0f} years")
    
    # Define all scenario combinations
    scopes = ['Territory', 'Consumption']
    distributions = ['Responsibility', 'Capability']
    warmings = ['1.5°C', '2°C']
    probabilities = ['50%', '67%']
    
    total_combos = len(scopes) * len(distributions) * len(warmings) * len(probabilities)
    
    print(f"\nGenerating visualizations for {total_combos} scenario combinations...")
    print("="*80)
    
    counter = 0
    for scope in scopes:
        for distribution in distributions:
            for warming in warmings:
                for probability in probabilities:
                    counter += 1
                    print(f"\n[{counter}/{total_combos}] Processing: {scope} | {distribution} | {warming} | {probability}")
                    
                    # Create visualizations for this combination
                    # 1. Scatter comparison (kept as-is)
                    create_scatter_comparison(comparison, scope, distribution, warming, probability)
                    
                    # 2. Year difference chart (replaces rank_changes)
                    create_year_difference_chart(comparison, scope, distribution, warming, probability)
                    
                    # 3. Difference maps (stacked with shared legend and global color range)
                    create_difference_maps(comparison, scope, distribution, warming, probability, iso2_to_iso3, global_color_range)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    
    print(f"\nTotal scenario combinations processed: {counter}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nVisualization types created:")
    print("  - Scatter comparisons (Base vs Alternatives)")
    print("  - Year impact charts (years lost/gained)")
    print("  - Difference maps (stacked 2018 & 2021 comparisons)")


if __name__ == '__main__':
    main()
