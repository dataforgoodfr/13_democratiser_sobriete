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
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

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


def create_rank_comparison(comparison_df, scope, distribution, warming, probability):
    """Create visualizations showing rank changes across scenarios."""
    
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
    
    # Calculate ranks (lower neutrality year = better rank)
    df['Rank_Base_2025'] = df['Base_2025'].rank(method='min')
    df['Rank_Alternative_2018'] = df['Alternative_2018'].rank(method='min')
    df['Rank_Alternative_2021'] = df['Alternative_2021'].rank(method='min')
    
    # Calculate rank changes
    df['Rank_Change_2018'] = df['Rank_Alternative_2018'] - df['Rank_Base_2025']
    df['Rank_Change_2021'] = df['Rank_Alternative_2021'] - df['Rank_Base_2025']
    
    # Get top 20 biggest rank changes (in both directions)
    top_gainers = df.nlargest(10, 'Rank_Change_2018')[['Country', 'Rank_Change_2018', 'Rank_Change_2021']].copy()
    top_losers = df.nsmallest(10, 'Rank_Change_2018')[['Country', 'Rank_Change_2018', 'Rank_Change_2021']].copy()
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Top rank losers (rank went down = neutrality year pushed back)
    ax1 = axes[0]
    x_pos = np.arange(len(top_gainers))
    width = 0.35
    
    ax1.barh(x_pos, top_gainers['Rank_Change_2018'], width, label='vs 2018', color=COLORS['Alternative_2018'], alpha=0.8)
    ax1.barh(x_pos + width, top_gainers['Rank_Change_2021'], width, label='vs 2021', color=COLORS['Alternative_2021'], alpha=0.8)
    
    ax1.set_yticks(x_pos + width/2)
    ax1.set_yticklabels(top_gainers['Country'])
    ax1.set_xlabel('Rank Change (Positive = Worse Rank)', fontsize=11, fontweight='bold')
    ax1.set_title('Top 10 Countries: Biggest Rank Decrease\n(More time to reach neutrality in alternatives)', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Top rank gainers (rank went up = neutrality year pulled forward)
    ax2 = axes[1]
    x_pos = np.arange(len(top_losers))
    
    ax2.barh(x_pos, top_losers['Rank_Change_2018'], width, label='vs 2018', color=COLORS['Alternative_2018'], alpha=0.8)
    ax2.barh(x_pos + width, top_losers['Rank_Change_2021'], width, label='vs 2021', color=COLORS['Alternative_2021'], alpha=0.8)
    
    ax2.set_yticks(x_pos + width/2)
    ax2.set_yticklabels(top_losers['Country'])
    ax2.set_xlabel('Rank Change (Negative = Better Rank)', fontsize=11, fontweight='bold')
    ax2.set_title('Top 10 Countries: Biggest Rank Increase\n(Less time to reach neutrality in alternatives)', 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Overall title
    fig.suptitle(f'Rank Changes Across Scenarios\n{scope} | {distribution} | {warming} | {probability}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    filename = f'rank_changes_{scope}_{distribution}_{warming}_{probability}.png'
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
                    create_scatter_comparison(comparison, scope, distribution, warming, probability)
                    create_rank_comparison(comparison, scope, distribution, warming, probability)
                    create_top_movers(comparison, scope, distribution, warming, probability)
                    create_distribution_plot(comparison, scope, distribution, warming, probability)
    
    # Create summary visualizations
    print("\n" + "="*80)
    print("Creating summary visualizations...")
    print("="*80)
    
    create_summary_heatmap(summary)
    create_key_findings_visual(full_results, comparison)
    
    print("\n" + "="*80)
    print("VISUALIZATION CREATION COMPLETED!")
    print("="*80)
    print(f"\nTotal visualizations created: {counter * 4 + 2}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nVisualization types created:")
    print("  - Scatter comparisons (Base vs Alternatives)")
    print("  - Rank change charts")
    print("  - Top movers/biggest changes")
    print("  - Distribution plots")
    print("  - Summary heatmap")
    print("  - Key findings overview")


if __name__ == '__main__':
    main()
