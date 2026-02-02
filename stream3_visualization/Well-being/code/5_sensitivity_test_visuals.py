#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Visualization Script

This script creates visualizations from the sensitivity analysis results:
1. EU-27 All Deciles EWBI over time for all experiments
2. Comparison of baseline vs experiments
3. Uncertainty bands showing range of outcomes

Run this after 5_sensitivity_test.py has completed.

Author: Data Processing Pipeline
Date: 2024
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ===============================
# CONFIGURATION
# ===============================
OUTPUT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')))
SENSITIVITY_OUTPUT = OUTPUT_DIR / "5_sensibility_test"
SENSITIVITY_DATA = SENSITIVITY_OUTPUT / "data"
SENSITIVITY_GRAPH = SENSITIVITY_OUTPUT / "graph"

# Create directories if they don't exist
SENSITIVITY_GRAPH.mkdir(parents=True, exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.cm.tab10.colors
BASELINE_COLOR = '#2c3e50'
EXPERIMENT_ALPHA = 0.5


# ===============================
# DATA LOADING FUNCTIONS
# ===============================

def load_baseline_data():
    """Load baseline EWBI results."""
    baseline_path = SENSITIVITY_DATA / "baseline_ewbi.csv"
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        print(f"[OK] Loaded baseline data: {len(df):,} records")
        return df
    else:
        print(f"[ERROR] Baseline data not found at {baseline_path}")
        return None


def load_experiment_data():
    """Load all experiment EWBI results."""
    experiments = {}
    configs = {}
    
    # Find all experiment files
    experiment_files = sorted(SENSITIVITY_DATA.glob("experiment_*_ewbi.csv"))
    
    for exp_file in experiment_files:
        # Extract experiment number
        exp_name = exp_file.stem.replace("_ewbi", "")
        exp_num = exp_name.split("_")[1]
        
        # Load data
        df = pd.read_csv(exp_file)
        experiments[exp_name] = df
        
        # Load corresponding config
        config_path = SENSITIVITY_DATA / f"experiment_{exp_num}_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                configs[exp_name] = json.load(f)
    
    print(f"[OK] Loaded {len(experiments)} experiment datasets")
    return experiments, configs


def extract_eu27_timeseries(df, level=1):
    """
    Extract EU-27 All Deciles time series from EWBI data.
    
    Args:
        df: DataFrame with EWBI results
        level: Level to extract (1=EWBI, 2=EU Priorities)
    
    Returns:
        DataFrame with Year and Value columns
    """
    eu27_data = df[
        (df['Country'] == 'EU-27') &
        (df['Decile'] == 'All Deciles') &
        (df['Level'] == level)
    ].copy()
    
    if eu27_data.empty:
        return None
    
    return eu27_data[['Year', 'Value']].sort_values('Year')


def extract_eu27_by_priority(df):
    """
    Extract EU-27 All Deciles time series for each EU Priority.
    
    Args:
        df: DataFrame with EWBI results
    
    Returns:
        Dictionary mapping EU priority name to time series DataFrame
    """
    level2_data = df[
        (df['Country'] == 'EU-27') &
        (df['Decile'] == 'All Deciles') &
        (df['Level'] == 2)
    ].copy()
    
    if level2_data.empty:
        return {}
    
    priorities = {}
    for priority in level2_data['EU priority'].dropna().unique():
        priority_data = level2_data[level2_data['EU priority'] == priority]
        priorities[priority] = priority_data[['Year', 'Value']].sort_values('Year')
    
    return priorities


# ===============================
# VISUALIZATION FUNCTIONS
# ===============================

def plot_eu27_ewbi_all_experiments(baseline_df, experiments, configs, output_path):
    """
    Plot EU-27 All Deciles EWBI over time for baseline and all experiments.
    
    Args:
        baseline_df: Baseline EWBI DataFrame
        experiments: Dictionary of experiment DataFrames
        configs: Dictionary of experiment configurations
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract baseline EU-27 time series
    baseline_ts = extract_eu27_timeseries(baseline_df, level=1)
    
    if baseline_ts is not None:
        ax.plot(baseline_ts['Year'], baseline_ts['Value'], 
                color=BASELINE_COLOR, linewidth=3, label='Baseline',
                marker='o', markersize=6, zorder=10)
    
    # Plot each experiment
    for i, (exp_name, exp_df) in enumerate(experiments.items()):
        exp_ts = extract_eu27_timeseries(exp_df, level=1)
        if exp_ts is not None:
            color = COLORS[i % len(COLORS)]
            
            # Create label with key config differences
            config = configs.get(exp_name, {})
            label_parts = []
            if config.get('NORMALIZATION_METHOD') == 'zscore':
                label_parts.append('Z-score')
            if config.get('EU_PRIORITIES_AGGREGATION') == 'arithmetic':
                label_parts.append('Arith.Agg')
            if config.get('BREAK_THRESHOLD', 0.3) != 0.3:
                label_parts.append(f"BT={config.get('BREAK_THRESHOLD')}")
            if config.get('APPLY_MOVING_AVERAGE'):
                label_parts.append('MA')
            
            label = exp_name.replace('_', ' ').title()
            if label_parts:
                label += f" ({', '.join(label_parts)})"
            
            ax.plot(exp_ts['Year'], exp_ts['Value'],
                    color=color, linewidth=1.5, alpha=EXPERIMENT_ALPHA,
                    label=label, marker='s', markersize=4)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('EWBI Value', fontsize=12)
    ax.set_title('EU-27 EWBI Over Time: Baseline vs All Experiments\n(All Deciles Aggregation)', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] EU-27 EWBI time series: {output_path}")


def plot_eu27_ewbi_uncertainty_band(baseline_df, experiments, output_path):
    """
    Plot EU-27 EWBI with uncertainty band showing min/max across experiments.
    
    Args:
        baseline_df: Baseline EWBI DataFrame
        experiments: Dictionary of experiment DataFrames
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Collect all time series
    all_series = []
    years = None
    
    # Add baseline
    baseline_ts = extract_eu27_timeseries(baseline_df, level=1)
    if baseline_ts is not None:
        all_series.append(baseline_ts.set_index('Year')['Value'])
        years = baseline_ts['Year'].values
    
    # Add experiments
    for exp_name, exp_df in experiments.items():
        exp_ts = extract_eu27_timeseries(exp_df, level=1)
        if exp_ts is not None:
            all_series.append(exp_ts.set_index('Year')['Value'])
    
    if not all_series:
        print("[WARN] No EU-27 data found for uncertainty band plot")
        return
    
    # Combine into DataFrame
    combined = pd.concat(all_series, axis=1)
    combined.columns = range(len(all_series))
    
    # Calculate statistics
    mean_values = combined.mean(axis=1)
    min_values = combined.min(axis=1)
    max_values = combined.max(axis=1)
    std_values = combined.std(axis=1)
    
    years = combined.index.values
    
    # Plot uncertainty band (min-max range)
    ax.fill_between(years, min_values, max_values, 
                    color='lightblue', alpha=0.4, label='Range (Min-Max)')
    
    # Plot ±1 std band
    ax.fill_between(years, mean_values - std_values, mean_values + std_values,
                    color='steelblue', alpha=0.3, label='±1 Std Dev')
    
    # Plot mean
    ax.plot(years, mean_values, color='navy', linewidth=2, 
            label='Mean across experiments', linestyle='--')
    
    # Plot baseline
    if baseline_ts is not None:
        ax.plot(baseline_ts['Year'], baseline_ts['Value'],
                color=BASELINE_COLOR, linewidth=3, label='Baseline',
                marker='o', markersize=6, zorder=10)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('EWBI Value', fontsize=12)
    ax.set_title('EU-27 EWBI Over Time: Uncertainty Analysis\n(Showing range of outcomes across methodological variants)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] EU-27 uncertainty band: {output_path}")


def plot_eu27_by_priority(baseline_df, experiments, output_path):
    """
    Plot EU-27 time series for each EU Priority (Level 2).
    
    Args:
        baseline_df: Baseline EWBI DataFrame
        experiments: Dictionary of experiment DataFrames
        output_path: Path to save the plot
    """
    # Get all priorities from baseline
    baseline_priorities = extract_eu27_by_priority(baseline_df)
    
    if not baseline_priorities:
        print("[WARN] No EU Priority data found for baseline")
        return
    
    n_priorities = len(baseline_priorities)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (priority, baseline_ts) in enumerate(baseline_priorities.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Plot baseline
        ax.plot(baseline_ts['Year'], baseline_ts['Value'],
                color=BASELINE_COLOR, linewidth=2.5, label='Baseline',
                marker='o', markersize=5, zorder=10)
        
        # Collect experiment values for this priority
        exp_values = []
        for exp_name, exp_df in experiments.items():
            exp_priorities = extract_eu27_by_priority(exp_df)
            if priority in exp_priorities:
                exp_ts = exp_priorities[priority]
                ax.plot(exp_ts['Year'], exp_ts['Value'],
                        color='gray', linewidth=0.8, alpha=0.4)
                exp_values.append(exp_ts.set_index('Year')['Value'])
        
        # Add mean of experiments
        if exp_values:
            combined = pd.concat(exp_values, axis=1)
            mean_exp = combined.mean(axis=1)
            ax.plot(mean_exp.index, mean_exp.values,
                    color='red', linewidth=1.5, linestyle='--', 
                    label='Mean (experiments)')
        
        ax.set_title(priority, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_priorities, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('EU-27 EU Priorities Over Time: Baseline vs Experiments\n(All Deciles Aggregation)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] EU-27 by priority: {output_path}")


def plot_config_impact_heatmap(baseline_df, experiments, configs, output_path):
    """
    Create heatmap showing impact of each configuration parameter on EWBI.
    
    Args:
        baseline_df: Baseline EWBI DataFrame
        experiments: Dictionary of experiment DataFrames
        configs: Dictionary of experiment configurations
        output_path: Path to save the plot
    """
    # Get baseline EU-27 value for most recent year
    baseline_ts = extract_eu27_timeseries(baseline_df, level=1)
    if baseline_ts is None:
        return
    
    baseline_value = baseline_ts['Value'].iloc[-1]
    baseline_year = baseline_ts['Year'].iloc[-1]
    
    # Calculate deviation for each experiment
    deviations = []
    for exp_name, exp_df in experiments.items():
        exp_ts = extract_eu27_timeseries(exp_df, level=1)
        if exp_ts is not None:
            exp_value = exp_ts[exp_ts['Year'] == baseline_year]['Value'].values
            if len(exp_value) > 0:
                deviation = (exp_value[0] - baseline_value) / baseline_value * 100
                config = configs.get(exp_name, {})
                config['experiment'] = exp_name
                config['deviation_pct'] = deviation
                deviations.append(config)
    
    if not deviations:
        print("[WARN] No deviation data available for heatmap")
        return
    
    # Create DataFrame for analysis
    dev_df = pd.DataFrame(deviations)
    
    # Create figure with parameter impact analysis
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    params_to_plot = [
        ('NORMALIZATION_METHOD', 'Normalization Method'),
        ('EU_PRIORITIES_AGGREGATION', 'EU Priorities Aggregation'),
        ('EWBI_DECILE_AGGREGATION', 'EWBI Decile Aggregation'),
        ('BREAK_THRESHOLD', 'Break Threshold'),
        ('RESCALE_MIN', 'Rescale Minimum'),
        ('NORMALIZATION_APPROACH', 'Normalization Approach')
    ]
    
    for idx, (param, title) in enumerate(params_to_plot):
        ax = axes[idx]
        
        if param in dev_df.columns:
            # Group by parameter value and show deviation distribution
            grouped = dev_df.groupby(param)['deviation_pct'].agg(['mean', 'std', 'count'])
            
            x_labels = [str(x) for x in grouped.index]
            x_pos = range(len(x_labels))
            
            bars = ax.bar(x_pos, grouped['mean'], yerr=grouped['std'],
                         capsize=5, color='steelblue', alpha=0.7, edgecolor='navy')
            
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel('% Deviation from Baseline', fontsize=9)
            
            # Add count labels
            for i, (_, row) in enumerate(grouped.iterrows()):
                ax.annotate(f'n={int(row["count"])}', 
                           xy=(i, row['mean']), 
                           xytext=(0, 5),
                           textcoords='offset points',
                           ha='center', fontsize=8)
        else:
            ax.set_visible(False)
    
    fig.suptitle(f'Impact of Configuration Parameters on EU-27 EWBI ({baseline_year})\n'
                 f'(% Deviation from Baseline = {baseline_value:.4f})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Config impact heatmap: {output_path}")


def plot_experiment_comparison_table(baseline_df, experiments, configs, output_path):
    """
    Create a visual table comparing all experiments.
    
    Args:
        baseline_df: Baseline EWBI DataFrame
        experiments: Dictionary of experiment DataFrames
        configs: Dictionary of experiment configurations
        output_path: Path to save the plot
    """
    # Get baseline values
    baseline_ts = extract_eu27_timeseries(baseline_df, level=1)
    if baseline_ts is None:
        return
    
    latest_year = baseline_ts['Year'].max()
    baseline_latest = baseline_ts[baseline_ts['Year'] == latest_year]['Value'].values[0]
    
    # Build comparison data
    rows = []
    rows.append({
        'Experiment': 'BASELINE',
        'Break Threshold': 0.3,
        'Moving Avg': 'No',
        'Norm Method': 'percentile',
        'Rescale Min': 0.1,
        'Norm Approach': 'multi_year',
        'EU Prio Agg': 'geometric',
        'Decile Agg': 'geometric',
        'Cross-Dec Agg': 'geometric',
        f'EWBI ({latest_year})': f'{baseline_latest:.4f}',
        'Δ Baseline': '0.00%'
    })
    
    for exp_name, exp_df in experiments.items():
        config = configs.get(exp_name, {})
        exp_ts = extract_eu27_timeseries(exp_df, level=1)
        
        if exp_ts is not None:
            exp_latest = exp_ts[exp_ts['Year'] == latest_year]['Value'].values
            if len(exp_latest) > 0:
                deviation = (exp_latest[0] - baseline_latest) / baseline_latest * 100
                
                rows.append({
                    'Experiment': exp_name.replace('experiment_', 'Exp '),
                    'Break Threshold': config.get('BREAK_THRESHOLD', 'N/A'),
                    'Moving Avg': 'Yes' if config.get('APPLY_MOVING_AVERAGE') else 'No',
                    'Norm Method': config.get('NORMALIZATION_METHOD', 'N/A')[:4],
                    'Rescale Min': config.get('RESCALE_MIN', 'N/A'),
                    'Norm Approach': config.get('NORMALIZATION_APPROACH', 'N/A')[:5],
                    'EU Prio Agg': config.get('EU_PRIORITIES_AGGREGATION', 'N/A')[:4],
                    'Decile Agg': config.get('EWBI_DECILE_AGGREGATION', 'N/A')[:4],
                    'Cross-Dec Agg': config.get('EWBI_CROSS_DECILE_AGGREGATION', 'N/A')[:4],
                    f'EWBI ({latest_year})': f'{exp_latest[0]:.4f}',
                    'Δ Baseline': f'{deviation:+.2f}%'
                })
    
    # Create table figure
    comparison_df = pd.DataFrame(rows)
    
    fig, ax = plt.subplots(figsize=(20, len(rows) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=comparison_df.values,
        colLabels=comparison_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4CAF50'] * len(comparison_df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Color the header
    for i in range(len(comparison_df.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight baseline row
    for i in range(len(comparison_df.columns)):
        table[(1, i)].set_facecolor('#E8F5E9')
    
    plt.title('Sensitivity Analysis: Experiment Configuration Comparison', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Experiment comparison table: {output_path}")


# ===============================
# MAIN EXECUTION
# ===============================

def main():
    """Main execution function for visualizations."""
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS - VISUALIZATIONS")
    print("="*70)
    print(f"\nLoading data from: {SENSITIVITY_DATA}")
    
    # Load data
    baseline_df = load_baseline_data()
    if baseline_df is None:
        print("[ERROR] Cannot proceed without baseline data")
        return
    
    experiments, configs = load_experiment_data()
    if not experiments:
        print("[WARN] No experiment data found, only baseline will be plotted")
    
    # Create visualizations
    print("\n[GENERATING VISUALIZATIONS]")
    
    # 1. EU-27 EWBI time series with all experiments
    plot_eu27_ewbi_all_experiments(
        baseline_df, experiments, configs,
        SENSITIVITY_GRAPH / "eu27_ewbi_all_experiments.png"
    )
    
    # 2. EU-27 EWBI uncertainty band
    plot_eu27_ewbi_uncertainty_band(
        baseline_df, experiments,
        SENSITIVITY_GRAPH / "eu27_ewbi_uncertainty_band.png"
    )
    
    # 3. EU-27 by EU Priority
    plot_eu27_by_priority(
        baseline_df, experiments,
        SENSITIVITY_GRAPH / "eu27_by_priority.png"
    )
    
    # 4. Configuration impact heatmap
    plot_config_impact_heatmap(
        baseline_df, experiments, configs,
        SENSITIVITY_GRAPH / "config_impact_analysis.png"
    )
    
    # 5. Experiment comparison table
    plot_experiment_comparison_table(
        baseline_df, experiments, configs,
        SENSITIVITY_GRAPH / "experiment_comparison_table.png"
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nOutput saved to: {SENSITIVITY_GRAPH}")
    print("\nGenerated plots:")
    print("  • eu27_ewbi_all_experiments.png - Time series for all experiments")
    print("  • eu27_ewbi_uncertainty_band.png - Uncertainty analysis with bands")
    print("  • eu27_by_priority.png - EU Priorities breakdown")
    print("  • config_impact_analysis.png - Parameter impact analysis")
    print("  • experiment_comparison_table.png - Configuration comparison table")


if __name__ == '__main__':
    main()
