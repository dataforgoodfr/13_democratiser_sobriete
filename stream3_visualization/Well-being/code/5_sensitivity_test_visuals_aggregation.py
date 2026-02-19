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
SENSITIVITY_OUTPUT = OUTPUT_DIR / "5_sensitivity_test_data_aggregation"
SENSITIVITY_DATA = SENSITIVITY_OUTPUT / "data"
SENSITIVITY_GRAPH = SENSITIVITY_OUTPUT / "graph"

# Create directories if they don't exist
SENSITIVITY_GRAPH.mkdir(parents=True, exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.cm.tab10.colors
BASELINE_COLOR = '#2c3e50'
EXPERIMENT_ALPHA = 0.5

# Years for rank-based analyses
RANK_ANALYSIS_YEARS = [2004, 2014, 2023]

# Parameters metadata keys to ignore in sensitivity decomposition
CONFIG_META_KEYS = {'N_CHANGED', 'CHANGED_PARAMS', 'experiment', 'deviation_pct'}


# ===============================
# DATA LOADING FUNCTIONS
# ===============================

def load_baseline_data():
    """Load baseline EWBI results."""
    baseline_path = SENSITIVITY_DATA / "experiment_0_baseline_ewbi.csv"
    fallback_path = SENSITIVITY_DATA / "baseline_ewbi.csv"

    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        print(f"[OK] Loaded baseline data: {len(df):,} records")
        return df
    if fallback_path.exists():
        df = pd.read_csv(fallback_path)
        print(f"[OK] Loaded fallback baseline data: {len(df):,} records")
        return df

    print(f"[ERROR] Baseline data not found at {baseline_path} or {fallback_path}")
    return None


def load_experiment_data():
    """Load all experiment EWBI results."""
    experiments = {}
    configs = {}
    
    print(f"[DEBUG] Looking for experiment files in: {SENSITIVITY_DATA}")
    
    # Find all experiment files
    experiment_files = sorted(SENSITIVITY_DATA.glob("experiment_*_ewbi.csv"))
    
    print(f"[DEBUG] Found {len(experiment_files)} experiment files:")
    for exp_file in experiment_files:
        print(f"  - {exp_file.name}")
    
    for exp_file in experiment_files:
        # Extract experiment number
        exp_name = exp_file.stem.replace("_ewbi", "")
        exp_num = exp_name.split("_")[1]
        
        # Skip baseline experiment (handled separately)
        if exp_num == "0" or "baseline" in exp_name.lower():
            print(f"[DEBUG] Skipping baseline experiment: {exp_name}")
            continue
        
        # Load data
        df = pd.read_csv(exp_file)
        experiments[exp_name] = df
        
        # Check if EU-27 data exists in this experiment
        eu27_count = len(df[(df['Country'] == 'EU-27') & (df['Level'] == 1)])
        print(f"[DEBUG] {exp_name}: {len(df)} rows total, {eu27_count} EU-27 Level 1 rows")
        
        # Load corresponding config
        config_path = SENSITIVITY_DATA / f"experiment_{exp_num}_params.json"
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
    print(f"[DEBUG] extract_eu27_timeseries: Looking for Level={level}")
    print(f"[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] Columns: {df.columns.tolist()}")
    print(f"[DEBUG] Unique countries: {df['Country'].unique() if 'Country' in df.columns else 'No Country column'}")
    print(f"[DEBUG] Unique levels: {df['Level'].unique() if 'Level' in df.columns else 'No Level column'}")
    print(f"[DEBUG] Unique deciles: {df['Decile'].unique() if 'Decile' in df.columns else 'No Decile column'}")
    
    eu27_data = df[
        (df['Country'] == 'EU-27') &
        (df['Decile'] == 'All Deciles') &
        (df['Level'] == level)
    ].copy()

    print(f"[DEBUG] After basic filtering: {len(eu27_data)} rows")

    # Match app.py: use population-weighted arithmetic mean for EU-27 (levels 1 and 2)
    if not eu27_data.empty and 'Aggregation' in eu27_data.columns and level in (1, 2):
        print(f"[DEBUG] Unique aggregations: {eu27_data['Aggregation'].unique()}")
        eu27_data = eu27_data[
            eu27_data['Aggregation'] == 'Population-weighted arithmetic mean'
        ].copy()
        print(f"[DEBUG] After aggregation filtering: {len(eu27_data)} rows")
    
    if eu27_data.empty:
        print(f"[DEBUG] No EU-27 data found for level {level}")
        return None
    
    print(f"[DEBUG] Found EU-27 data: {len(eu27_data)} rows")
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

    # Match app.py: use population-weighted arithmetic mean for EU-27
    if not level2_data.empty and 'Aggregation' in level2_data.columns:
        level2_data = level2_data[
            level2_data['Aggregation'] == 'Population-weighted arithmetic mean'
        ].copy()
    
    if level2_data.empty:
        return {}
    
    priorities = {}
    for priority in level2_data['EU priority'].dropna().unique():
        priority_data = level2_data[level2_data['EU priority'] == priority]
        priorities[priority] = priority_data[['Year', 'Value']].sort_values('Year')
    
    return priorities


def extract_country_rankings(df, year):
    """Extract country rankings for EWBI (Level 1, All Deciles) for a given year."""
    ewbi_data = df[
        (df['Level'] == 1) &
        (df['Decile'] == 'All Deciles') &
        (df['Country'] != 'EU-27')
    ].copy()

    year_data = ewbi_data[ewbi_data['Year'] == year].copy()
    if year_data.empty:
        return None

    year_data['Rank'] = year_data['Value'].rank(ascending=False, method='min').astype(int)
    return year_data[['Country', 'Rank']].sort_values('Rank')


def compute_rank_differences(baseline_ranks, experiment_ranks):
    """Compute rank differences between baseline and experiment."""
    merged = baseline_ranks.merge(
        experiment_ranks,
        on='Country',
        suffixes=('_baseline', '_experiment')
    )
    merged['Rank_Diff'] = merged['Rank_baseline'] - merged['Rank_experiment']
    merged['Abs_Rank_Diff'] = merged['Rank_Diff'].abs()
    return merged


def load_rank_differences(baseline_df, experiments, year):
    """Compute rank differences for all experiments against baseline for a given year."""
    baseline_ranks = extract_country_rankings(baseline_df, year)
    if baseline_ranks is None:
        return []

    all_rank_diffs = []
    for exp_df in experiments.values():
        exp_ranks = extract_country_rankings(exp_df, year)
        if exp_ranks is None:
            continue
        all_rank_diffs.append(compute_rank_differences(baseline_ranks, exp_ranks))

    return all_rank_diffs


def get_parameter_names(configs):
    """Get parameter names from experiment configs, excluding metadata keys."""
    # For aggregation sensitivity analysis - only methodological variables
    METHODOLOGICAL_VARIABLES = {
        'NORMALIZATION_METHOD',
        'RESCALE_MIN',
        'NORMALIZATION_APPROACH',
        'PCA_SCOPE',
        'EU_PRIORITIES_APPROACH',
        'EU_PRIORITIES_AGGREGATION',
        'EWBI_DECILE_AGGREGATION',
        'EWBI_CROSS_DECILE_AGGREGATION'
    }
    
    param_names = set()
    for cfg in configs.values():
        for key in cfg.keys():
            if key not in CONFIG_META_KEYS and key in METHODOLOGICAL_VARIABLES:
                param_names.add(key)
    return sorted(param_names)


def compute_first_order_indices(y_values, param_values):
    """Compute weighted first-order variance contributions and indices."""
    y = np.asarray(y_values, dtype=float)
    if y.size == 0:
        return 0.0, 0.0

    var_y = float(np.var(y, ddof=0))
    if var_y == 0.0:
        return 0.0, 0.0

    param_values = np.asarray(param_values, dtype=object)
    unique_vals, counts = np.unique(param_values, return_counts=True)
    conditional_means = []
    weights = []
    for val, count in zip(unique_vals, counts):
        conditional_means.append(float(np.mean(y[param_values == val])))
        weights.append(count / y.size)

    if conditional_means:
        mean_mu = float(np.average(conditional_means, weights=weights))
        vi = float(np.average((np.asarray(conditional_means) - mean_mu) ** 2, weights=weights))
    else:
        vi = 0.0

    si = vi / var_y if var_y != 0.0 else 0.0
    return vi, si


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
    print(f"[DEBUG] Creating uncertainty band with {len(experiments)} experiments")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Collect all time series
    all_series = []
    years = None
    
    # Add baseline
    baseline_ts = extract_eu27_timeseries(baseline_df, level=1)
    if baseline_ts is not None:
        all_series.append(baseline_ts.set_index('Year')['Value'])
        years = baseline_ts['Year'].values
        print(f"[DEBUG] Baseline EU-27 timeseries: {len(baseline_ts)} years")
    else:
        print("[WARN] No baseline EU-27 data found")

    # Add experiments
    for exp_name, exp_df in experiments.items():
        exp_ts = extract_eu27_timeseries(exp_df, level=1)
        if exp_ts is not None:
            all_series.append(exp_ts.set_index('Year')['Value'])
            print(f"[DEBUG] {exp_name} EU-27 timeseries: {len(exp_ts)} years")
        else:
            print(f"[WARN] No EU-27 data found for {exp_name}")

    print(f"[DEBUG] Total series for uncertainty band: {len(all_series)}")
    
    if not all_series:
        print("[WARN] No EU-27 data found for uncertainty band plot")
        return

    if len(all_series) == 1:
        print("[WARN] Only baseline data available - no uncertainty band to show")
        # Still plot the baseline
        if baseline_ts is not None:
            ax.plot(baseline_ts['Year'], baseline_ts['Value'],
                    color=BASELINE_COLOR, linewidth=3, label='Baseline',
                    marker='o', markersize=6, zorder=10)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('EWBI Value', fontsize=12)
        ax.set_title('EU-27 EWBI Over Time: Baseline Only\n(No experiment data available for uncertainty analysis)', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] EU-27 baseline plot: {output_path}")
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


def plot_rank_shift_variance_decomposition(baseline_df, experiments, configs, output_path, year):
    """Stacked bar chart: variance decomposition of country rank shifts vs baseline."""
    # Use specified year
    baseline_ranks = extract_country_rankings(baseline_df, year)
    if baseline_ranks is None:
        print("[WARN] No baseline rankings available for variance decomposition")
        return

    baseline_ranks = baseline_ranks.set_index('Country')
    param_names = get_parameter_names(configs)
    if not param_names:
        print("[WARN] No parameters found in configs for variance decomposition")
        return

    country_indices = {}
    for country in baseline_ranks.index:
        y_values = []
        exp_param_values = {p: [] for p in param_names}

        for exp_name, exp_df in experiments.items():
            exp_ranks = extract_country_rankings(exp_df, year)
            if exp_ranks is None or country not in exp_ranks['Country'].values:
                continue

            exp_rank = int(exp_ranks[exp_ranks['Country'] == country]['Rank'].iloc[0])
            baseline_rank = int(baseline_ranks.loc[country, 'Rank'])
            rank_shift = baseline_rank - exp_rank
            y_values.append(rank_shift)

            cfg = configs.get(exp_name, {})
            for p in param_names:
                exp_param_values[p].append(cfg.get(p))

        if len(y_values) < 2:
            continue

        contributions = {}
        for p in param_names:
            vi, _ = compute_first_order_indices(y_values, exp_param_values[p])
            contributions[p] = vi

        country_indices[country] = contributions

    if not country_indices:
        print("[WARN] Not enough data for rank shift variance decomposition")
        return

    components = param_names
    country_sum_contrib = {
        c: float(np.sum([country_indices[c].get(p, 0.0) for p in components]))
        for c in country_indices
    }
    sorted_countries = sorted(
        country_sum_contrib.keys(),
        key=lambda c: country_sum_contrib[c],
        reverse=True
    )

    data = {comp: [country_indices[c].get(comp, 0.0) for c in sorted_countries] for comp in components}

    fig, ax = plt.subplots(figsize=(14, 10))
    left = np.zeros(len(sorted_countries))

    for comp in components:
        values = np.array(data[comp])
        ax.barh(sorted_countries, values, left=left, label=comp)
        left += values

    ax.set_xlabel('Variance Contribution (Rank Shift vs Baseline)', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    ax.set_title(f'Variance Decomposition of Country Rank Shifts vs Baseline ({int(year)})', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] Rank shift variance decomposition: {output_path}")


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


def create_summary_plot(all_rank_diffs, output_path):
    """Create summary plots showing rank-change sensitivity across all experiments.

    Writes two PNGs derived from `output_path`:
    - `*_mean_abs.png`: mean absolute rank change bar chart
    - `*_distribution.png`: per-country rank-change distribution (boxplot)
    """
    if not all_rank_diffs:
        print("[WARN] No rank differences available for summary plot")
        return None

    all_countries = set()
    for df in all_rank_diffs:
        all_countries.update(df['Country'].unique())

    summary_data = []
    for country in all_countries:
        changes = []
        for df in all_rank_diffs:
            if country in df['Country'].values:
                change = df[df['Country'] == country]['Rank_Diff'].values[0]
                changes.append(change)

        if changes:
            summary_data.append({
                'Country': country,
                'Mean_Rank_Diff': np.mean(changes),
                'Std_Rank_Diff': np.std(changes),
                'Mean_Abs_Rank_Diff': np.mean(np.abs(changes)),
                'Max_Abs_Rank_Diff': np.max(np.abs(changes))
            })

    summary_df = pd.DataFrame(summary_data).sort_values('Mean_Abs_Rank_Diff', ascending=False)

    output_path = Path(output_path)
    mean_abs_path = output_path.with_name(f"{output_path.stem}_mean_abs{output_path.suffix}")
    distribution_path = output_path.with_name(f"{output_path.stem}_distribution{output_path.suffix}")

    sorted_by_abs = summary_df.sort_values('Mean_Abs_Rank_Diff', ascending=True)
    max_abs = float(sorted_by_abs['Mean_Abs_Rank_Diff'].max()) if len(sorted_by_abs) else 0.0
    if max_abs <= 0:
        colors1 = ['lightgray'] * len(sorted_by_abs)
    else:
        colors1 = plt.cm.RdYlGn_r(sorted_by_abs['Mean_Abs_Rank_Diff'] / max_abs)

    # Plot 1: Mean absolute rank change
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 12))
    ax1.barh(sorted_by_abs['Country'], sorted_by_abs['Mean_Abs_Rank_Diff'], color=colors1)
    ax1.set_xlabel('Mean Absolute Rank Change', fontsize=12)
    ax1.set_ylabel('Country', fontsize=12)
    ax1.set_title('Mean Absolute Rank Change Across Experiments\n(Higher = More Sensitive to Methodology)', fontsize=14)
    plt.tight_layout()
    plt.savefig(mean_abs_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Plot 2: Distribution of rank changes per country
    country_changes = {country: [] for country in all_countries}
    for df in all_rank_diffs:
        for _, row in df.iterrows():
            country_changes[row['Country']].append(row['Rank_Diff'])

    sorted_countries = sorted_by_abs['Country'].tolist()
    box_data = [country_changes[c] for c in sorted_countries]

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 12))
    bp = ax2.boxplot(box_data, vert=False, patch_artist=True)
    positions = list(range(1, len(sorted_countries) + 1))
    ax2.set_yticks(positions)
    ax2.set_yticklabels(sorted_countries)
    ax2.axvline(x=0, color='red', linewidth=1, linestyle='--')
    ax2.set_xlabel('Rank Change Distribution', fontsize=12)
    ax2.set_ylabel('Country', fontsize=12)
    ax2.set_title('Distribution of Rank Changes per Country\n(Positive = Improved vs Baseline)', fontsize=14)

    for patch, median in zip(bp['boxes'], [np.median(d) if len(d) else 0 for d in box_data]):
        if median > 0:
            patch.set_facecolor('#90EE90')
        elif median < 0:
            patch.set_facecolor('#FFB6C1')
        else:
            patch.set_facecolor('#D3D3D3')

    plt.tight_layout()
    plt.savefig(distribution_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print(f"[SAVED] Summary plots: {mean_abs_path} and {distribution_path}")
    return summary_df


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
                    f'EWBI ({latest_year})': f'{exp_latest[0]:.4f}',
                    'Δ Baseline': f'{deviation:+.2f}%'
                })
    
    # Create table figure
    comparison_df = pd.DataFrame(rows)
    
    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.5 + 2))
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
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color the header
    for i in range(len(comparison_df.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight baseline row
    for i in range(len(comparison_df.columns)):
        table[(1, i)].set_facecolor('#E8F5E9')
    
    plt.title('Sensitivity Analysis: Aggregation Variables Comparison', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Experiment comparison table: {output_path}")



def plot_rank_shift_variance_heatmap(baseline_df, experiments, configs, output_path, year):
    """Heatmap of variance contributions per country and parameter."""
    # Use specified year
    baseline_ranks = extract_country_rankings(baseline_df, year)
    if baseline_ranks is None:
        print("[WARN] No baseline rankings available for heatmap")
        return

    baseline_ranks = baseline_ranks.set_index('Country')
    param_names = get_parameter_names(configs)
    if not param_names:
        print("[WARN] No parameters found in configs for heatmap")
        return

    # Build per-country contributions
    country_contribs = {}
    for country in baseline_ranks.index:
        y_values = []
        exp_param_values = {p: [] for p in param_names}

        for exp_name, exp_df in experiments.items():
            exp_ranks = extract_country_rankings(exp_df, year)
            if exp_ranks is None or country not in exp_ranks['Country'].values:
                continue

            exp_rank = int(exp_ranks[exp_ranks['Country'] == country]['Rank'].iloc[0])
            baseline_rank = int(baseline_ranks.loc[country, 'Rank'])
            rank_shift = baseline_rank - exp_rank
            y_values.append(rank_shift)

            cfg = configs.get(exp_name, {})
            for p in param_names:
                exp_param_values[p].append(cfg.get(p))

        if len(y_values) < 2:
            continue

        contribs = {}
        for p in param_names:
            vi, _ = compute_first_order_indices(y_values, exp_param_values[p])
            contribs[p] = vi

        country_contribs[country] = contribs

    if not country_contribs:
        print("[WARN] Not enough data for rank shift variance heatmap")
        return

    heatmap_df = pd.DataFrame.from_dict(country_contribs, orient='index')

    # Sort rows by total variance contributions (descending)
    heatmap_df['Total'] = heatmap_df.sum(axis=1)
    heatmap_df = heatmap_df.sort_values('Total', ascending=False)
    heatmap_df = heatmap_df.drop(columns=['Total'])

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(heatmap_df, ax=ax, cmap='viridis')

    ax.set_xlabel('Parameters', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    ax.set_title(f'Variance Contributions by Parameter and Country ({int(year)})', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] Rank shift variance heatmap: {output_path}")


def plot_eu27_variance_decomposition_over_time(baseline_df, experiments, configs, output_path):
    """
    Stacked area chart: variance decomposition (absolute variance contributions)
    over time for EU-27 EWBI difference vs baseline.
    """
    baseline_ts = extract_eu27_timeseries(baseline_df, level=1)
    if baseline_ts is None:
        print("[WARN] No baseline EU-27 time series available")
        return

    param_names = get_parameter_names(configs)
    if not param_names:
        print("[WARN] No parameters found in configs for EU-27 variance decomposition")
        return

    baseline_series = baseline_ts.set_index('Year')['Value']
    years = baseline_series.index.values

    # Prepare per-year variance decomposition (no non-additive component)
    components = param_names
    component_series = {comp: [] for comp in components}

    for year in years:
        y_values = []
        exp_param_values = {p: [] for p in param_names}

        for exp_name, exp_df in experiments.items():
            exp_ts = extract_eu27_timeseries(exp_df, level=1)
            if exp_ts is None or year not in exp_ts['Year'].values:
                continue

            exp_val = float(exp_ts[exp_ts['Year'] == year]['Value'].iloc[0])
            baseline_val = float(baseline_series.loc[year])
            y_values.append(exp_val - baseline_val)

            cfg = configs.get(exp_name, {})
            for p in param_names:
                exp_param_values[p].append(cfg.get(p))

        if len(y_values) < 2:
            for comp in components:
                component_series[comp].append(0.0)
            continue

        total_var = float(np.var(y_values, ddof=0))
        if total_var == 0.0:
            for comp in components:
                component_series[comp].append(0.0)
            continue

        sum_vi = 0.0
        for p in param_names:
            vi, _ = compute_first_order_indices(y_values, exp_param_values[p])
            component_series[p].append(vi)
            sum_vi += vi

    # Stacked area plot
    fig, ax = plt.subplots(figsize=(14, 8))
    stacked_values = [component_series[comp] for comp in components]
    ax.stackplot(years, stacked_values, labels=components)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Variance Contribution (EU-27 Δ vs Baseline)', fontsize=12)
    ax.set_title('EU-27 EWBI Variance Decomposition Over Time (Δ vs Baseline)', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] EU-27 variance decomposition over time: {output_path}")


def create_experiment_heatmap(all_rank_diffs, output_path):
    """Create a heatmap showing rank differences across all experiments and countries."""
    if not all_rank_diffs:
        print("[WARN] No rank differences available for heatmap")
        return

    all_countries = sorted(set.union(*[set(df['Country'].unique()) for df in all_rank_diffs]))

    matrix = np.zeros((len(all_countries), len(all_rank_diffs)))
    for exp_idx, df in enumerate(all_rank_diffs):
        for _, row in df.iterrows():
            country_idx = all_countries.index(row['Country'])
            matrix[country_idx, exp_idx] = row['Rank_Diff']

    fig, ax = plt.subplots(figsize=(14, 12))
    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(all_rank_diffs)))
    ax.set_xticklabels([f'Exp {i+1}' for i in range(len(all_rank_diffs))], rotation=45, ha='right')
    ax.set_yticks(range(len(all_countries)))
    ax.set_yticklabels(all_countries)

    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    ax.set_title('Rank Changes Across All Experiments\n(Green = Improved, Red = Worsened)', fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rank Change vs Baseline', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] Experiment heatmap: {output_path}")


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
        print(f"[DEBUG] Check that sensitivity analysis has been run and data exists in: {SENSITIVITY_DATA}")
        # List contents of data directory
        if SENSITIVITY_DATA.exists():
            files = list(SENSITIVITY_DATA.glob("*"))
            print(f"[DEBUG] Files in data directory: {[f.name for f in files]}")
        else:
            print(f"[ERROR] Data directory does not exist: {SENSITIVITY_DATA}")
    else:
        print(f"[OK] Found {len(experiments)} experiments to visualize")
    
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

    # 6. Variance decomposition of rank shifts by country (per year)
    for year in RANK_ANALYSIS_YEARS:
        plot_rank_shift_variance_decomposition(
            baseline_df, experiments, configs,
            SENSITIVITY_GRAPH / f"rank_shift_variance_decomposition_{year}.png",
            year
        )

        # 6b. Heatmap of rank shift variance contributions
        plot_rank_shift_variance_heatmap(
            baseline_df, experiments, configs,
            SENSITIVITY_GRAPH / f"rank_shift_variance_heatmap_{year}.png",
            year
        )

    # 7. EU-27 variance decomposition over time (Δ vs baseline)
    plot_eu27_variance_decomposition_over_time(
        baseline_df, experiments, configs,
        SENSITIVITY_GRAPH / "eu27_variance_decomposition_over_time.png"
    )

    # 8. Summary rank changes and experiment heatmap (per year)
    for year in RANK_ANALYSIS_YEARS:
        all_rank_diffs = load_rank_differences(baseline_df, experiments, year)
        create_summary_plot(
            all_rank_diffs,
            SENSITIVITY_GRAPH / f"summary_rank_changes_{year}.png"
        )
        create_experiment_heatmap(
            all_rank_diffs,
            SENSITIVITY_GRAPH / f"experiment_heatmap_{year}.png"
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
    print("  • rank_shift_variance_decomposition_<year>.png - Variance decomposition of country rank shifts")
    print("  • rank_shift_variance_heatmap_<year>.png - Heatmap of rank shift variance contributions")
    print("  • eu27_variance_decomposition_over_time.png - EU-27 variance decomposition over time")
    print("  • summary_rank_changes_<year>_mean_abs.png - Mean absolute rank change (bar chart)")
    print("  • summary_rank_changes_<year>_distribution.png - Rank change distribution (boxplot)")
    print("  • experiment_heatmap_<year>.png - Heatmap of rank changes across experiments")


if __name__ == '__main__':
    main()
