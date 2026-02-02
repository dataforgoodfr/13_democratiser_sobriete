#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Script - Monte Carlo Experiments

This script performs Monte Carlo sensitivity analysis on the EWBI computation pipeline:
1. Randomly samples parameter combinations from predefined options
2. Runs the full pipeline (stages 1, 2, 3, 4) for each experiment
3. Compares country rankings between experiments and baseline
4. Generates visualizations showing rank differences

Output:
- Data files for each experiment in output/5_sensibility_test/data/
- Graphs comparing rankings in output/5_sensibility_test/graph/

Author: Data Processing Pipeline
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import importlib.util
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ===============================
# CONFIGURATION
# ===============================
OUTPUT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')))
CODE_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
SENSITIVITY_OUTPUT = OUTPUT_DIR / "5_sensibility_test"
SENSITIVITY_DATA = SENSITIVITY_OUTPUT / "data"
SENSITIVITY_GRAPH = SENSITIVITY_OUTPUT / "graph"

# Create directories
SENSITIVITY_OUTPUT.mkdir(parents=True, exist_ok=True)
SENSITIVITY_DATA.mkdir(parents=True, exist_ok=True)
SENSITIVITY_GRAPH.mkdir(parents=True, exist_ok=True)

# Number of Monte Carlo experiments
N_EXPERIMENTS = 10

# Random seed for reproducibility
RANDOM_SEED = 42

# ===============================
# PARAMETER OPTIONS
# ===============================
PARAMETER_OPTIONS = {
    # Stage 1: Missing Data
    'BREAK_THRESHOLD': [0.2, 0.3, 0.4],
    'APPLY_MOVING_AVERAGE': [False, True],
    'MOVING_AVERAGE_WINDOW': [3, 5],
    
    # Stage 3: Normalization
    'NORMALIZATION_METHOD': ['percentile', 'zscore'],
    'RESCALE_MIN': [0.05, 0.1, 0.2],
    'NORMALIZATION_APPROACH': ['multi_year', 'per_year'],
    
    # Stage 4: Weighting and Aggregation
    'PCA_SCOPE': ['all_years', 'per_year'],
    'EU_PRIORITIES_AGGREGATION': ['geometric', 'arithmetic'],
    'EWBI_DECILE_AGGREGATION': ['geometric', 'arithmetic'],
    'EWBI_CROSS_DECILE_AGGREGATION': ['geometric', 'arithmetic']
}

# Baseline configuration (default values)
BASELINE_CONFIG = {
    'BREAK_THRESHOLD': 0.3,
    'APPLY_MOVING_AVERAGE': False,
    'MOVING_AVERAGE_WINDOW': 5,
    'NORMALIZATION_METHOD': 'percentile',
    'RESCALE_MIN': 0.1,
    'NORMALIZATION_APPROACH': 'multi_year',
    'PCA_SCOPE': 'all_years',
    'EU_PRIORITIES_AGGREGATION': 'geometric',
    'EWBI_DECILE_AGGREGATION': 'geometric',
    'EWBI_CROSS_DECILE_AGGREGATION': 'geometric'
}


# ===============================
# HELPER FUNCTIONS
# ===============================

def sample_configuration(rng):
    """
    Sample a random configuration by drawing uniformly from each parameter's options.
    
    Args:
        rng: numpy random generator
        
    Returns:
        Dictionary with sampled configuration
    """
    config = {}
    for param, options in PARAMETER_OPTIONS.items():
        idx = rng.integers(0, len(options))
        config[param] = options[idx]
    return config


def config_to_string(config):
    """Convert configuration to a readable string for logging."""
    lines = []
    for param, value in config.items():
        lines.append(f"  {param}: {value}")
    return "\n".join(lines)


def config_to_short_string(config):
    """Convert configuration to a short string for filenames."""
    parts = []
    # Abbreviate parameter names
    abbrev = {
        'BREAK_THRESHOLD': 'BT',
        'APPLY_MOVING_AVERAGE': 'MA',
        'MOVING_AVERAGE_WINDOW': 'MW',
        'NORMALIZATION_METHOD': 'NM',
        'RESCALE_MIN': 'RM',
        'NORMALIZATION_APPROACH': 'NA',
        'PCA_SCOPE': 'PS',
        'EU_PRIORITIES_AGGREGATION': 'EPA',
        'EWBI_DECILE_AGGREGATION': 'EDA',
        'EWBI_CROSS_DECILE_AGGREGATION': 'ECDA'
    }
    for param, value in config.items():
        short_param = abbrev.get(param, param[:3])
        if isinstance(value, bool):
            short_val = 'T' if value else 'F'
        elif isinstance(value, float):
            short_val = str(value).replace('.', '')
        else:
            short_val = str(value)[:3]
        parts.append(f"{short_param}{short_val}")
    return "_".join(parts)


def update_module_config(module_path, config_updates):
    """
    Update configuration variables in a module file.
    
    Args:
        module_path: Path to the Python module file
        config_updates: Dictionary of variable names and new values
    """
    with open(module_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for var_name, new_value in config_updates.items():
        # Handle different value types
        if isinstance(new_value, bool):
            new_value_str = str(new_value)
        elif isinstance(new_value, str):
            new_value_str = f"'{new_value}'"
        else:
            new_value_str = str(new_value)
        
        # Replace the variable assignment
        import re
        pattern = rf'^{var_name}\s*=\s*.+$'
        replacement = f'{var_name} = {new_value_str}'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    with open(module_path, 'w', encoding='utf-8') as f:
        f.write(content)


def run_pipeline_stage(stage_name, module_name):
    """
    Run a pipeline stage by importing and executing its main function.
    
    Args:
        stage_name: Human-readable stage name for logging
        module_name: Name of the module file (without .py)
    """
    print(f"\n{'='*50}")
    print(f"Running {stage_name}...")
    print('='*50)
    
    module_path = CODE_DIR / f"{module_name}.py"
    
    # Load module dynamically
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    
    # Suppress print statements during execution
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            spec.loader.exec_module(module)
            if hasattr(module, 'main'):
                module.main()
        except Exception as e:
            print(f"[ERROR] {stage_name} failed: {e}")
            raise
    
    print(f"[OK] {stage_name} completed")


def run_full_pipeline(config, experiment_id):
    """
    Run the full EWBI computation pipeline with a given configuration.
    
    Args:
        config: Dictionary with parameter configuration
        experiment_id: Identifier for this experiment
        
    Returns:
        DataFrame with final EWBI results
    """
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT {experiment_id}")
    print(f"{'#'*60}")
    print(f"\nConfiguration:")
    print(config_to_string(config))
    
    # Update Stage 1 configuration
    stage1_config = {
        'BREAK_THRESHOLD': config['BREAK_THRESHOLD'],
        'APPLY_MOVING_AVERAGE': config['APPLY_MOVING_AVERAGE'],
        'MOVING_AVERAGE_WINDOW': config['MOVING_AVERAGE_WINDOW']
    }
    update_module_config(CODE_DIR / '1_missing_data.py', stage1_config)
    
    # Update Stage 3 configuration
    stage3_config = {
        'NORMALIZATION_METHOD': config['NORMALIZATION_METHOD'],
        'RESCALE_MIN': config['RESCALE_MIN'],
        'NORMALIZATION_APPROACH': config['NORMALIZATION_APPROACH']
    }
    update_module_config(CODE_DIR / '3_normalisation_data.py', stage3_config)
    
    # Update Stage 4 configuration
    stage4_config = {
        'PCA_SCOPE': config['PCA_SCOPE'],
        'EU_PRIORITIES_AGGREGATION': config['EU_PRIORITIES_AGGREGATION'],
        'EWBI_DECILE_AGGREGATION': config['EWBI_DECILE_AGGREGATION'],
        'EWBI_CROSS_DECILE_AGGREGATION': config['EWBI_CROSS_DECILE_AGGREGATION'],
        'SKIP_EU27': False  # Compute EU-27 for visualization
    }
    update_module_config(CODE_DIR / '4_weighting_aggregation.py', stage4_config)
    
    # Run pipeline stages
    try:
        run_pipeline_stage("Stage 1: Missing Data Imputation", "1_missing_data")
        run_pipeline_stage("Stage 3: Normalization", "3_normalisation_data")
        run_pipeline_stage("Stage 4: Weighting & Aggregation", "4_weighting_aggregation")
    except Exception as e:
        print(f"[ERROR] Pipeline failed for experiment {experiment_id}: {e}")
        return None
    
    # Load and return results
    results_path = OUTPUT_DIR / "4_weighting_aggregation_output" / "ewbi_final_aggregated.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        return df
    else:
        print(f"[ERROR] Results not found at {results_path}")
        return None


def extract_country_rankings(df, year=None):
    """
    Extract country rankings from EWBI results.
    
    Args:
        df: DataFrame with EWBI results
        year: Specific year to extract (if None, uses most recent year)
        
    Returns:
        DataFrame with Country and Rank columns
    """
    # Filter for Level 1 (EWBI) and All Deciles
    ewbi_data = df[
        (df['Level'] == 1) & 
        (df['Decile'] == 'All Deciles') &
        (df['Country'] != 'EU-27')
    ].copy()
    
    if ewbi_data.empty:
        print("[WARN] No EWBI All Deciles data found")
        return None
    
    # Use most recent year if not specified
    if year is None:
        year = ewbi_data['Year'].max()
    
    # Filter for specific year
    year_data = ewbi_data[ewbi_data['Year'] == year].copy()
    
    if year_data.empty:
        print(f"[WARN] No data found for year {year}")
        return None
    
    # Rank countries (higher EWBI = better = rank 1)
    year_data['Rank'] = year_data['Value'].rank(ascending=False, method='min').astype(int)
    
    return year_data[['Country', 'Value', 'Rank']].sort_values('Rank')


def compute_rank_differences(baseline_ranks, experiment_ranks):
    """
    Compute rank differences between baseline and experiment.
    
    Args:
        baseline_ranks: DataFrame with baseline rankings
        experiment_ranks: DataFrame with experiment rankings
        
    Returns:
        DataFrame with rank differences per country
    """
    # Merge on Country
    merged = baseline_ranks.merge(
        experiment_ranks,
        on='Country',
        suffixes=('_baseline', '_experiment')
    )
    
    # Compute rank difference (positive = improved rank in experiment)
    merged['Rank_Diff'] = merged['Rank_baseline'] - merged['Rank_experiment']
    merged['Abs_Rank_Diff'] = merged['Rank_Diff'].abs()
    
    return merged


def create_rank_comparison_plot(rank_diffs, config, experiment_id, output_path):
    """
    Create a bar plot showing rank differences per country.
    
    Args:
        rank_diffs: DataFrame with rank differences
        config: Configuration dictionary
        experiment_id: Experiment identifier
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by rank difference
    sorted_data = rank_diffs.sort_values('Rank_Diff', ascending=True)
    
    # Color bars based on direction of change
    colors = ['#e74c3c' if x < 0 else '#27ae60' if x > 0 else '#95a5a6' 
              for x in sorted_data['Rank_Diff']]
    
    bars = ax.barh(sorted_data['Country'], sorted_data['Rank_Diff'], color=colors)
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Rank Difference (Positive = Improved Ranking)', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    ax.set_title(f'Experiment {experiment_id}: Country Rank Changes vs Baseline\n'
                 f'Mean Abs. Change: {rank_diffs["Abs_Rank_Diff"].mean():.2f}', fontsize=14)
    
    # Add configuration text box
    config_text = '\n'.join([f'{k}: {v}' for k, v in config.items()])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.02, 0.5, config_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Rank comparison plot: {output_path}")


def create_summary_plot(all_rank_diffs, output_path):
    """
    Create a summary plot showing mean rank changes across all experiments.
    
    Args:
        all_rank_diffs: List of DataFrames with rank differences from each experiment
        output_path: Path to save the plot
    """
    # Compute mean absolute rank change per country across all experiments
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
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Plot 1: Mean absolute rank change per country
    sorted_by_abs = summary_df.sort_values('Mean_Abs_Rank_Diff', ascending=True)
    colors1 = plt.cm.RdYlGn_r(sorted_by_abs['Mean_Abs_Rank_Diff'] / sorted_by_abs['Mean_Abs_Rank_Diff'].max())
    ax1.barh(sorted_by_abs['Country'], sorted_by_abs['Mean_Abs_Rank_Diff'], color=colors1)
    ax1.set_xlabel('Mean Absolute Rank Change', fontsize=12)
    ax1.set_ylabel('Country', fontsize=12)
    ax1.set_title('Mean Absolute Rank Change Across Experiments\n(Higher = More Sensitive to Methodology)', fontsize=14)
    
    # Plot 2: Distribution of rank changes (boxplot style)
    # Aggregate all changes per country
    country_changes = {country: [] for country in all_countries}
    for df in all_rank_diffs:
        for _, row in df.iterrows():
            country_changes[row['Country']].append(row['Rank_Diff'])
    
    # Convert to format for boxplot
    sorted_countries = sorted_by_abs['Country'].tolist()
    box_data = [country_changes[c] for c in sorted_countries]
    
    bp = ax2.boxplot(box_data, vert=False, patch_artist=True)
    ax2.set_yticklabels(sorted_countries)
    ax2.axvline(x=0, color='red', linewidth=1, linestyle='--')
    ax2.set_xlabel('Rank Change Distribution', fontsize=12)
    ax2.set_ylabel('Country', fontsize=12)
    ax2.set_title('Distribution of Rank Changes per Country\n(Positive = Improved vs Baseline)', fontsize=14)
    
    # Color boxplots
    for patch, median in zip(bp['boxes'], [np.median(d) for d in box_data]):
        if median > 0:
            patch.set_facecolor('#90EE90')
        elif median < 0:
            patch.set_facecolor('#FFB6C1')
        else:
            patch.set_facecolor('#D3D3D3')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] Summary plot: {output_path}")
    
    return summary_df


def create_experiment_heatmap(all_rank_diffs, all_configs, output_path):
    """
    Create a heatmap showing rank differences across all experiments and countries.
    
    Args:
        all_rank_diffs: List of DataFrames with rank differences
        all_configs: List of configuration dictionaries
        output_path: Path to save the plot
    """
    # Get all unique countries
    all_countries = sorted(set.union(*[set(df['Country'].unique()) for df in all_rank_diffs]))
    
    # Create matrix
    matrix = np.zeros((len(all_countries), len(all_rank_diffs)))
    
    for exp_idx, df in enumerate(all_rank_diffs):
        for _, row in df.iterrows():
            country_idx = all_countries.index(row['Country'])
            matrix[country_idx, exp_idx] = row['Rank_Diff']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use diverging colormap centered at 0
    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    
    # Set labels
    ax.set_xticks(range(len(all_rank_diffs)))
    ax.set_xticklabels([f'Exp {i+1}' for i in range(len(all_rank_diffs))], rotation=45, ha='right')
    ax.set_yticks(range(len(all_countries)))
    ax.set_yticklabels(all_countries)
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    ax.set_title('Rank Changes Across All Experiments\n(Green = Improved, Red = Worsened)', fontsize=14)
    
    # Add colorbar
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
    """Main execution function for sensitivity analysis."""
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS - MONTE CARLO EXPERIMENTS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of experiments: {N_EXPERIMENTS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Output directory: {SENSITIVITY_OUTPUT}")
    
    # Initialize random generator
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Store results
    all_configs = []
    all_results = []
    all_rank_diffs = []
    baseline_ranks = None
    
    # ===== RUN BASELINE =====
    print("\n" + "="*70)
    print("RUNNING BASELINE CONFIGURATION")
    print("="*70)
    
    baseline_result = run_full_pipeline(BASELINE_CONFIG, "BASELINE")
    
    if baseline_result is not None:
        # Save baseline results
        baseline_path = SENSITIVITY_DATA / "baseline_ewbi.csv"
        baseline_result.to_csv(baseline_path, index=False)
        print(f"[SAVED] Baseline results: {baseline_path}")
        
        # Extract baseline rankings
        baseline_ranks = extract_country_rankings(baseline_result)
        if baseline_ranks is not None:
            baseline_ranks_path = SENSITIVITY_DATA / "baseline_rankings.csv"
            baseline_ranks.to_csv(baseline_ranks_path, index=False)
            print(f"[SAVED] Baseline rankings: {baseline_ranks_path}")
            print(f"\nBaseline Rankings (Top 10):")
            print(baseline_ranks.head(10).to_string(index=False))
    else:
        print("[ERROR] Baseline run failed. Exiting.")
        return
    
    # ===== RUN EXPERIMENTS =====
    for exp_num in range(1, N_EXPERIMENTS + 1):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {exp_num} of {N_EXPERIMENTS}")
        print("="*70)
        
        # Sample random configuration
        config = sample_configuration(rng)
        all_configs.append(config)
        
        # Run pipeline
        result = run_full_pipeline(config, exp_num)
        
        if result is not None:
            all_results.append(result)
            
            # Save experiment results
            exp_data_path = SENSITIVITY_DATA / f"experiment_{exp_num}_ewbi.csv"
            result.to_csv(exp_data_path, index=False)
            
            # Save configuration
            config_path = SENSITIVITY_DATA / f"experiment_{exp_num}_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Extract rankings and compute differences
            exp_ranks = extract_country_rankings(result)
            
            if exp_ranks is not None and baseline_ranks is not None:
                rank_diffs = compute_rank_differences(baseline_ranks, exp_ranks)
                all_rank_diffs.append(rank_diffs)
                
                # Save rank differences
                rank_diff_path = SENSITIVITY_DATA / f"experiment_{exp_num}_rank_diffs.csv"
                rank_diffs.to_csv(rank_diff_path, index=False)
                
                # Create individual experiment plot
                plot_path = SENSITIVITY_GRAPH / f"experiment_{exp_num}_rank_comparison.png"
                create_rank_comparison_plot(rank_diffs, config, exp_num, plot_path)
                
                # Print summary
                print(f"\n[SUMMARY] Experiment {exp_num}:")
                print(f"  Mean absolute rank change: {rank_diffs['Abs_Rank_Diff'].mean():.2f}")
                print(f"  Max rank change: {rank_diffs['Abs_Rank_Diff'].max()}")
                print(f"  Countries with changed rank: {(rank_diffs['Rank_Diff'] != 0).sum()}")
        else:
            print(f"[WARN] Experiment {exp_num} failed, skipping")
    
    # ===== CREATE SUMMARY VISUALIZATIONS =====
    if all_rank_diffs:
        print("\n" + "="*70)
        print("CREATING SUMMARY VISUALIZATIONS")
        print("="*70)
        
        # Summary plot
        summary_path = SENSITIVITY_GRAPH / "summary_rank_changes.png"
        summary_df = create_summary_plot(all_rank_diffs, summary_path)
        
        # Save summary statistics
        summary_stats_path = SENSITIVITY_DATA / "summary_statistics.csv"
        summary_df.to_csv(summary_stats_path, index=False)
        print(f"[SAVED] Summary statistics: {summary_stats_path}")
        
        # Heatmap
        heatmap_path = SENSITIVITY_GRAPH / "experiment_heatmap.png"
        create_experiment_heatmap(all_rank_diffs, all_configs, heatmap_path)
        
        # Save all configurations
        all_configs_df = pd.DataFrame(all_configs)
        all_configs_df.index = [f"Experiment_{i+1}" for i in range(len(all_configs))]
        all_configs_path = SENSITIVITY_DATA / "all_experiment_configs.csv"
        all_configs_df.to_csv(all_configs_path)
        print(f"[SAVED] All configurations: {all_configs_path}")
        
        # Print final summary
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults:")
        print(f"  Experiments completed: {len(all_rank_diffs)}/{N_EXPERIMENTS}")
        print(f"  Data saved to: {SENSITIVITY_DATA}")
        print(f"  Graphs saved to: {SENSITIVITY_GRAPH}")
        
        # Overall statistics
        all_abs_changes = np.concatenate([df['Abs_Rank_Diff'].values for df in all_rank_diffs])
        print(f"\nOverall Rank Change Statistics:")
        print(f"  Mean absolute change: {np.mean(all_abs_changes):.2f}")
        print(f"  Median absolute change: {np.median(all_abs_changes):.2f}")
        print(f"  Max absolute change: {np.max(all_abs_changes)}")
        print(f"  Std of absolute change: {np.std(all_abs_changes):.2f}")
        
        # Most sensitive countries
        print(f"\nTop 5 Most Sensitive Countries:")
        print(summary_df.head(5)[['Country', 'Mean_Abs_Rank_Diff', 'Max_Abs_Rank_Diff']].to_string(index=False))
        
        # Least sensitive countries
        print(f"\nTop 5 Least Sensitive Countries:")
        print(summary_df.tail(5)[['Country', 'Mean_Abs_Rank_Diff', 'Max_Abs_Rank_Diff']].to_string(index=False))
    
    # Restore baseline configuration
    print("\n[RESTORE] Restoring baseline configuration in pipeline files...")
    update_module_config(CODE_DIR / '1_missing_data.py', {
        'BREAK_THRESHOLD': BASELINE_CONFIG['BREAK_THRESHOLD'],
        'APPLY_MOVING_AVERAGE': BASELINE_CONFIG['APPLY_MOVING_AVERAGE'],
        'MOVING_AVERAGE_WINDOW': BASELINE_CONFIG['MOVING_AVERAGE_WINDOW']
    })
    update_module_config(CODE_DIR / '3_normalisation_data.py', {
        'NORMALIZATION_METHOD': BASELINE_CONFIG['NORMALIZATION_METHOD'],
        'RESCALE_MIN': BASELINE_CONFIG['RESCALE_MIN'],
        'NORMALIZATION_APPROACH': BASELINE_CONFIG['NORMALIZATION_APPROACH']
    })
    update_module_config(CODE_DIR / '4_weighting_aggregation.py', {
        'PCA_SCOPE': BASELINE_CONFIG['PCA_SCOPE'],
        'EU_PRIORITIES_AGGREGATION': BASELINE_CONFIG['EU_PRIORITIES_AGGREGATION'],
        'EWBI_DECILE_AGGREGATION': BASELINE_CONFIG['EWBI_DECILE_AGGREGATION'],
        'EWBI_CROSS_DECILE_AGGREGATION': BASELINE_CONFIG['EWBI_CROSS_DECILE_AGGREGATION']
    })
    print("[OK] Baseline configuration restored")


if __name__ == '__main__':
    main()
