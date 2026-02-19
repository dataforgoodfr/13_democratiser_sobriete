#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Script - Monte Carlo Experiments (Data Treatment)

This script performs Monte Carlo sensitivity analysis on the EWBI computation pipeline
focusing on data treatment variables (data quality, smoothing):
1. Randomly samples parameter combinations from predefined options
2. Runs the full pipeline (stages 1, 2, 3, 4) for each experiment
3. Compares country rankings between experiments and baseline
4. Saves data for visualization (visualizations handled by separate script)

Output:
- Data files for each experiment in output/5_sensitivity_test_data_treatment/data/
- No graphs generated (handled by 5_sensitivity_test_visuals_data_treatment.py)

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
import matplotlib.pyplot as plt  # Keep import for visualization functions (even if not used in main)
# import seaborn as sns  # Not needed - visuals handled by separate script
from datetime import datetime
import importlib.util
from tqdm import tqdm

warnings.filterwarnings('ignore')


def _set_experiment_env(output_dir: Path, config: dict) -> None:
    """Apply per-experiment configuration via environment variables.

    Pipeline stages read these at import time (no file rewriting).
    """
    os.environ["EWBI_OUTPUT_DIR"] = str(output_dir)

    # Stage 1 (varied here)
    os.environ["EWBI_BREAK_THRESHOLD"] = str(config.get("BREAK_THRESHOLD", BASELINE_CONFIG["BREAK_THRESHOLD"]))
    os.environ["EWBI_APPLY_MOVING_AVERAGE"] = str(config.get("APPLY_MOVING_AVERAGE", BASELINE_CONFIG["APPLY_MOVING_AVERAGE"]))
    os.environ["EWBI_MOVING_AVERAGE_WINDOW"] = str(config.get("MOVING_AVERAGE_WINDOW", BASELINE_CONFIG["MOVING_AVERAGE_WINDOW"]))
    os.environ["EWBI_APPLY_MEAN_RESCALING"] = str(config.get("APPLY_MEAN_RESCALING", BASELINE_CONFIG["APPLY_MEAN_RESCALING"]))

    # Stage 3/4 (fixed baseline here)
    os.environ["EWBI_NORMALIZATION_METHOD"] = str(config.get("NORMALIZATION_METHOD", BASELINE_CONFIG["NORMALIZATION_METHOD"]))
    os.environ["EWBI_RESCALE_MIN"] = str(config.get("RESCALE_MIN", BASELINE_CONFIG["RESCALE_MIN"]))
    os.environ["EWBI_NORMALIZATION_APPROACH"] = str(config.get("NORMALIZATION_APPROACH", BASELINE_CONFIG["NORMALIZATION_APPROACH"]))

    os.environ["EWBI_PCA_SCOPE"] = str(config.get("PCA_SCOPE", BASELINE_CONFIG["PCA_SCOPE"]))
    os.environ["EWBI_EU_PRIORITIES_APPROACH"] = str(config.get("EU_PRIORITIES_APPROACH", BASELINE_CONFIG["EU_PRIORITIES_APPROACH"]))
    os.environ["EWBI_EU_PRIORITIES_AGGREGATION"] = str(config.get("EU_PRIORITIES_AGGREGATION", BASELINE_CONFIG["EU_PRIORITIES_AGGREGATION"]))
    os.environ["EWBI_EWBI_DECILE_AGGREGATION"] = str(config.get("EWBI_DECILE_AGGREGATION", BASELINE_CONFIG["EWBI_DECILE_AGGREGATION"]))
    os.environ["EWBI_EWBI_CROSS_DECILE_AGGREGATION"] = str(config.get("EWBI_CROSS_DECILE_AGGREGATION", BASELINE_CONFIG["EWBI_CROSS_DECILE_AGGREGATION"]))

    # Stage 4: Level 2 post-aggregation structural break adjustment (toggle only)
    os.environ["EWBI_LEVEL2_BREAK_ADJUSTMENT"] = str(
        config.get("LEVEL2_BREAK_ADJUSTMENT", BASELINE_CONFIG["LEVEL2_BREAK_ADJUSTMENT"])
    )

    # Ensure EU-27 exists for experiment outputs/plots.
    os.environ["EWBI_SKIP_EU27"] = "False"


def restore_all_configs():
    """Backward-compatible no-op.

    Previous versions rewrote stage modules and then restored them.
    The pipeline now uses environment variables only, so there's nothing to restore.
    """
    return

def copy_input_data_for_sensitivity():
    """
    Copy necessary input data from protected directories to sensitivity directories.
    This ensures we have the data we need without ever modifying the original files.
    """
    import shutil
    import os
    
    print("\n[SETUP] Copying input data for sensitivity analysis...")
    
    def _sync_tree(src: Path, dest: Path) -> None:
        """Incrementally copy missing files from src -> dest.

        This is important when dest exists but is incomplete (e.g., from an interrupted run).
        """
        src = Path(src)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        for root, dirs, files in os.walk(src):
            root_path = Path(root)
            rel = root_path.relative_to(src)
            target_root = dest / rel
            target_root.mkdir(parents=True, exist_ok=True)

            for dirname in dirs:
                (target_root / dirname).mkdir(parents=True, exist_ok=True)

            for filename in files:
                src_path = root_path / filename
                dest_path = target_root / filename

                try:
                    if dest_path.exists() and dest_path.stat().st_size == src_path.stat().st_size:
                        continue
                except OSError:
                    # If we can't stat, attempt copy.
                    pass

                shutil.copy2(src_path, dest_path)

    # Copy Stage 0 raw data (needed as input for Stage 1)
    src_raw_data = OUTPUT_DIR / "0_raw_data_EUROSTAT"
    dest_raw_data = SENSITIVITY_PIPELINE_OUTPUT / "0_raw_data_EUROSTAT"

    def _has_stage1_inputs(base_dir: Path) -> bool:
        required = [
            base_dir / "0_EU-SILC" / "3_final_merged_df" / "EU_SILC_household_final_summary.csv",
            base_dir / "0_EU-SILC" / "3_final_merged_df" / "EU_SILC_personal_final_summary.csv",
            base_dir / "0_LFS" / "LFS_household_final_summary.csv",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            print("[WARN] Sensitivity raw inputs look incomplete (missing Stage 1 inputs):")
            for p in missing[:6]:
                print(f"       - {p}")
            return False
        return True

    if src_raw_data.exists():
        if not dest_raw_data.exists():
            shutil.copytree(src_raw_data, dest_raw_data)
            print(f"[COPIED] Raw EUROSTAT data to sensitivity directory")
        else:
            _sync_tree(src_raw_data, dest_raw_data)
            print(f"[SYNC] Raw EUROSTAT data synced (filled missing files)")

        # Self-heal if an earlier interrupted run left an incomplete tree.
        if not _has_stage1_inputs(dest_raw_data):
            print("[REBUILD] Rebuilding sensitivity raw inputs from scratch...")
            shutil.rmtree(dest_raw_data, ignore_errors=True)
            shutil.copytree(src_raw_data, dest_raw_data)
            if _has_stage1_inputs(dest_raw_data):
                print("[OK] Sensitivity raw inputs rebuilt successfully")
            else:
                print("[ERROR] Sensitivity raw inputs still incomplete after rebuild")
    else:
        print(f"[WARN] Raw EUROSTAT data not found in {src_raw_data}")

    # Seed Stage 2 PCA results from the baseline pipeline.
    # Data-treatment experiments vary Stage 1 break adjustment, but should keep
    # PCA weights fixed; also avoids Windows path-too-long issues when generating
    # many visual outputs.
    stage2_src = OUTPUT_DIR / "2_multivariate_analysis_output" / "pca_results_full.json"
    stage2_dest_dir = SENSITIVITY_STAGE2_OUTPUT
    stage2_dest_dir.mkdir(parents=True, exist_ok=True)
    stage2_dest = stage2_dest_dir / "pca_results_full.json"
    if not stage2_dest.exists() and stage2_src.exists():
        shutil.copy2(stage2_src, stage2_dest)
        print(f"[COPIED] Stage 2 seed input: pca_results_full.json")
    elif stage2_dest.exists():
        print(f"[SKIP] Stage 2 seed already present: {stage2_dest.name}")
    else:
        print(f"[WARN] Stage 2 seed not found: {stage2_src} (Stage 4 may fall back to unweighted aggregation)")
    
    print("[OK] Input data setup complete")


def get_protected_states():
    """
    Get initial states of protected directories and files for monitoring.
    """
    # Create checksums or timestamps of protected files
    protected_states = {}
    
    for protected_dir in PROTECTED_DIRS:
        if protected_dir.exists():
            # Get modification time of directory
            protected_states[str(protected_dir)] = protected_dir.stat().st_mtime
    
    for protected_file in PROTECTED_FILES:
        if protected_file.exists():
            # Get modification time and size of file
            stat = protected_file.stat()
            protected_states[str(protected_file)] = (stat.st_mtime, stat.st_size)
    
    return protected_states


def check_protected_directories(initial_states):
    """
    Check that protected directories and files haven't been modified.
    """
    violations = []
    
    for protected_dir in PROTECTED_DIRS:
        if protected_dir.exists():
            current_mtime = protected_dir.stat().st_mtime
            if str(protected_dir) in initial_states:
                if current_mtime != initial_states[str(protected_dir)]:
                    violations.append(f"Directory modified: {protected_dir}")
    
    for protected_file in PROTECTED_FILES:
        if protected_file.exists():
            stat = protected_file.stat()
            current_state = (stat.st_mtime, stat.st_size)
            if str(protected_file) in initial_states:
                if current_state != initial_states[str(protected_file)]:
                    violations.append(f"File modified: {protected_file}")
    
    return violations

# ===============================
# CONFIGURATION
# ===============================
OUTPUT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')))
CODE_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
DATA_DIR = CODE_DIR.parent / 'data'  # Add data directory path
SENSITIVITY_OUTPUT = OUTPUT_DIR / "5_sensitivity_test_data_treatment"
SENSITIVITY_DATA = SENSITIVITY_OUTPUT / "data"
SENSITIVITY_GRAPH = SENSITIVITY_OUTPUT / "graph"

# PROTECTED DIRECTORIES - NEVER MODIFY THESE!
PROTECTED_DIRS = [
    OUTPUT_DIR / "0_raw_data_EUROSTAT",
    OUTPUT_DIR / "1_missing_data_output", 
    OUTPUT_DIR / "2_multivariate_analysis_output",
    OUTPUT_DIR / "3_normalisation_data_output",
    OUTPUT_DIR / "4_weighting_aggregation_output"
]
PROTECTED_FILES = [
    OUTPUT_DIR / "ewbi_master_aggregated.csv"
]

# SENSITIVITY-SPECIFIC OUTPUT DIRECTORIES
SENSITIVITY_PIPELINE_OUTPUT = SENSITIVITY_OUTPUT / "pipeline_output"
SENSITIVITY_STAGE1_OUTPUT = SENSITIVITY_PIPELINE_OUTPUT / "1_missing_data_output"
SENSITIVITY_STAGE2_OUTPUT = SENSITIVITY_PIPELINE_OUTPUT / "2_multivariate_analysis_output" 
SENSITIVITY_STAGE3_OUTPUT = SENSITIVITY_PIPELINE_OUTPUT / "3_normalisation_data_output"
SENSITIVITY_STAGE4_OUTPUT = SENSITIVITY_PIPELINE_OUTPUT / "4_weighting_aggregation_output"

# Create directories
SENSITIVITY_OUTPUT.mkdir(parents=True, exist_ok=True)
SENSITIVITY_DATA.mkdir(parents=True, exist_ok=True)
SENSITIVITY_PIPELINE_OUTPUT.mkdir(parents=True, exist_ok=True)
SENSITIVITY_STAGE1_OUTPUT.mkdir(parents=True, exist_ok=True)
SENSITIVITY_STAGE2_OUTPUT.mkdir(parents=True, exist_ok=True)
SENSITIVITY_STAGE3_OUTPUT.mkdir(parents=True, exist_ok=True)
SENSITIVITY_STAGE4_OUTPUT.mkdir(parents=True, exist_ok=True)
# SENSITIVITY_GRAPH.mkdir(parents=True, exist_ok=True)  # Not needed - visuals handled by separate script

# Number of Monte Carlo experiments
N_EXPERIMENTS = int(os.getenv("EWBI_SENSITIVITY_N_EXPERIMENTS", "20"))  # default: comprehensive sensitivity analysis

# Random seed for reproducibility
RANDOM_SEED = 42

# ===============================
# PARAMETER OPTIONS - DATA TREATMENT EXPERIMENT
# ===============================
PARAMETER_OPTIONS = {
    # Stage 1: Missing Data - Variables we don't control (data quality)
    'BREAK_THRESHOLD': [0.1, 0.2, 0.3],
    'APPLY_MOVING_AVERAGE': [False, True],
    'MOVING_AVERAGE_WINDOW': [3, 5],
    'APPLY_MEAN_RESCALING': [False, True],
    # Stage 4: Level 2 post-aggregation structural break adjustment (toggle only)
    'LEVEL2_BREAK_ADJUSTMENT': [True, False],
}

# Baseline configuration (default values)
BASELINE_CONFIG = {
    # Data treatment variables (will be varied in experiments)
    'BREAK_THRESHOLD': 0.2,
    'APPLY_MOVING_AVERAGE': True,
    'MOVING_AVERAGE_WINDOW': 5,
    'APPLY_MEAN_RESCALING': False,
    'LEVEL2_BREAK_ADJUSTMENT': True,
    # Methodological variables (fixed at baseline)
    'NORMALIZATION_METHOD': 'zscore',
    'RESCALE_MIN': 0.1,
    'NORMALIZATION_APPROACH': 'multi_year',
    'PCA_SCOPE': 'all_years',
    'EU_PRIORITIES_APPROACH': 'pca',
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
        'LEVEL2_BREAK_ADJUSTMENT': 'L2B',
        'NORMALIZATION_METHOD': 'NM',
        'RESCALE_MIN': 'RM',
        'NORMALIZATION_APPROACH': 'NA',
        'PCA_SCOPE': 'PS',
        'EU_PRIORITIES_APPROACH': 'EPP',
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


# Global backup storage for original configurations
    # NOTE: Older versions of this script rewrote stage files in-place to change
    # output directories and parameters. That approach caused pipeline clashes when
    # restoration didn't run (interruptions, crashes). We now configure stages via
    # environment variables only.


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

    # Ensure stage modules can import local helpers (e.g., pipeline_env.py).
    code_dir_str = str(CODE_DIR)
    if code_dir_str not in sys.path:
        sys.path.insert(0, code_dir_str)
    
    # Load module dynamically
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    
    # Capture stage stdout so we can surface it on failure.
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            spec.loader.exec_module(module)
            if hasattr(module, 'main'):
                module.main()
        except Exception as e:
            stage_log = f.getvalue()
            raise RuntimeError(
                f"{stage_name} failed with {type(e).__name__}: {e}\n\n--- Captured stage output ---\n{stage_log}"
            ) from e
    
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
    
    _set_experiment_env(SENSITIVITY_PIPELINE_OUTPUT, {
        **BASELINE_CONFIG,
        **config,
    })

    # Expected outputs (within sensitivity pipeline output)
    expected_stage1 = SENSITIVITY_STAGE1_OUTPUT / "raw_data_break_adjusted.csv"
    expected_stage2 = SENSITIVITY_STAGE2_OUTPUT / "pca_results_full.json"
    expected_stage3 = SENSITIVITY_STAGE3_OUTPUT / "level4_normalised_indicators.csv"
    expected_stage4 = SENSITIVITY_STAGE4_OUTPUT / "ewbi_final_aggregated.csv"
    
    # Run pipeline stages
    try:
        run_pipeline_stage("Stage 1: Missing Data Imputation", "1_missing_data")

        if not expected_stage1.exists():
            raise FileNotFoundError(
                f"Stage 1 completed but did not produce {expected_stage1}. "
                "Check that raw inputs exist under 0_raw_data_EUROSTAT in the sensitivity pipeline output."
            )

        # Stage 2 is seeded (fixed baseline) for treatment sensitivity.
        if not expected_stage2.exists():
            print(f"[WARN] Missing Stage 2 PCA seed ({expected_stage2.name}); Stage 4 may fall back to unweighted aggregation")

        run_pipeline_stage("Stage 3: Normalization", "3_normalisation_data")
        if not expected_stage3.exists():
            raise FileNotFoundError(
                f"Stage 3 completed but did not produce {expected_stage3}. "
                f"Expected output dir: {SENSITIVITY_STAGE3_OUTPUT}"
            )

        run_pipeline_stage("Stage 4: Weighting & Aggregation", "4_weighting_aggregation")
        if not expected_stage4.exists():
            raise FileNotFoundError(
                f"Stage 4 completed but did not produce {expected_stage4}. "
                f"Expected output dir: {SENSITIVITY_STAGE4_OUTPUT}"
            )
    except Exception as e:
        print(f"[ERROR] Pipeline failed for experiment {experiment_id}: {e}")
        return None
    
    # Load and return results
    # SAFE VERSION: Only read from sensitivity pipeline output, never touch main app files
    possible_paths = [
        SENSITIVITY_STAGE4_OUTPUT / "ewbi_final_aggregated.csv",  # Stage 4 output with EU-27
        SENSITIVITY_PIPELINE_OUTPUT / "ewbi_master_aggregated.csv",  # Stage 4 app-ready copy
    ]
    
    for results_path in possible_paths:
        if results_path.exists():
            df = pd.read_csv(results_path)
            print(f"[DEBUG] Using {results_path.name}: {len(df)} rows total")
            # Make a copy to avoid any reference issues
            return df.copy()
    
    print(f"[ERROR] No results files found in pipeline output directory")
    print(f"[INFO] This is normal if the pipeline failed or EU-27 generation was skipped")
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

def aggregate_arithmetic_mean(values, weights=None):
    """Compute arithmetic mean with optional weights."""
    if weights is None:
        return np.nanmean(values)
    return np.average(values, weights=weights)

def aggregate_geometric_mean(values, weights=None):
    """Compute geometric mean with optional weights."""
    if weights is None:
        return np.exp(np.nanmean(np.log(np.maximum(values, 1e-10))))
    log_values = np.log(np.maximum(values, 1e-10))
    return np.exp(np.average(log_values, weights=weights))

def compute_eu27_for_experiment(exp_data, population_data, exp_num):
    """Compute EU-27 aggregations for a single experiment."""
    print(f"\n[INFO] Computing EU-27 aggregations for experiment {exp_num}...")
    
    eu27_records = []
    
    # Group by year, level, decile, and indicators
    if 'Primary and raw data' in exp_data.columns and exp_data['Primary and raw data'].notna().any():
        group_cols = ['Year', 'Decile', 'Level', 'Primary and raw data']
    elif 'EU priority' in exp_data.columns and exp_data['EU priority'].notna().any():
        group_cols = ['Year', 'Decile', 'Level', 'EU priority']
    else:
        group_cols = ['Year', 'Decile', 'Level']
    
    for group_key, group_df in exp_data.groupby(group_cols):
        year = group_key[0]
        decile = group_key[1]
        level = group_key[2]
        
        # Merge with population data for weighting
        pop_for_merge = population_data.rename(columns={
            'country': 'Country',
            'year': 'Year', 
            'population': 'Population'
        })
        group_with_pop = group_df.merge(
            pop_for_merge[['Country', 'Year', 'Population']],
            on=['Country', 'Year'],
            how='left'
        )
        
        # Remove rows without population data
        group_with_pop = group_with_pop.dropna(subset=['Population'])
        
        # Require at least 15 countries for EU-27 aggregation
        n_countries = len(group_with_pop['Country'].unique())
        if n_countries < 15:
            continue
        
        if len(group_with_pop) == 0:
            continue
        
        # Compute population-weighted arithmetic mean
        weights = group_with_pop['Population'].values
        values = group_with_pop['Value'].values
        eu27_value = aggregate_arithmetic_mean(values, weights)
        
        if np.isnan(eu27_value):
            continue
        
        # Create EU-27 record
        eu27_record = {
            'Year': year,
            'Country': 'EU-27',
            'Decile': decile,
            'Level': level,
            'EU priority': group_key[3] if len(group_key) > 3 and len(group_cols) > 3 and 'EU priority' in group_cols else pd.NA,
            'Secondary': pd.NA,
            'Primary and raw data': group_key[3] if len(group_key) > 3 and len(group_cols) > 3 and 'Primary and raw data' in group_cols else pd.NA,
            'Type': 'Aggregation',
            'Aggregation': 'Population-weighted arithmetic mean',
            'Value': eu27_value,
            'datasource': pd.NA
        }
        
        # Get other columns from original data
        for col in group_df.columns:
            if col not in eu27_record and col not in ['Country', 'Value', 'Decile']:
                unique_vals = group_df[col].dropna().unique()
                if len(unique_vals) > 0:
                    eu27_record[col] = unique_vals[0]
        
        eu27_records.append(eu27_record)
    
    return pd.DataFrame(eu27_records)

def run_monte_carlo_experiments():
    """
    Generate and run Monte Carlo sensitivity experiments for data treatment variables.
    
    This function:
    1. Samples random data treatment parameter configurations
    2. Runs the full pipeline for each experiment  
    3. Saves results for later analysis
    """
    print("\n" + "="*70)
    print("MONTE CARLO SENSITIVITY EXPERIMENTS - DATA TREATMENT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of experiments: {N_EXPERIMENTS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Focus: Data treatment variables (break threshold, smoothing, rescaling)")
    print(f"  Output directory: {SENSITIVITY_OUTPUT}")
    
    # ===== PROTECTION VALIDATION =====
    print("\n" + "="*70)
    print("PROTECTION VALIDATION")
    print("="*70)
    
    # Get initial states of protected directories and files
    initial_protected_states = get_protected_states()
    print(f"[PROTECTED] Monitoring {len(PROTECTED_DIRS)} directories and {len(PROTECTED_FILES)} files")
    
    # Copy necessary input data to sensitivity directories
    copy_input_data_for_sensitivity()
    
    try:
        # Set random seed for reproducibility
        rng = np.random.default_rng(RANDOM_SEED)
        
        # ===== GENERATE BASELINE EXPERIMENT =====
        print(f"\n{'='*70}")
        print("BASELINE EXPERIMENT")  
        print("="*70)
        
        try:
            # Save baseline parameters as JSON
            baseline_param_path = SENSITIVITY_DATA / "experiment_0_baseline_params.json"
            try:
                # Convert numpy types to native Python types for JSON serialization
                json_baseline = {}
                for key, value in BASELINE_CONFIG.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        json_baseline[key] = value.item()
                    else:
                        json_baseline[key] = value
                
                with open(baseline_param_path, 'w') as f:
                    json.dump(json_baseline, f, indent=2)
                print(f"  Baseline parameters saved: {baseline_param_path}")
            except Exception as json_e:
                print(f"  Warning: Could not save baseline parameters as JSON: {json_e}")
            
            baseline_results = run_full_pipeline(BASELINE_CONFIG, "baseline")
            if baseline_results is not None:
                baseline_path = SENSITIVITY_DATA / "experiment_0_baseline_ewbi.csv"
                baseline_results.to_csv(baseline_path, index=False)
                print(f"✓ Baseline results saved: {baseline_path}")
            else:
                print("✗ Baseline experiment failed")
                return
        except Exception as e:
            print(f"✗ Baseline experiment failed: {e}")
            return
        
        # ===== GENERATE RANDOM EXPERIMENTS =====
        print(f"\n{'='*70}")
        print("DATA TREATMENT EXPERIMENTS") 
        print("="*70)
        print("Varying: break_threshold, moving_average, window_size, mean_rescaling")
        print("Fixed: normalization & aggregation methods (at baseline values)")
        
        successful_experiments = 0
        failed_experiments = 0
        
        for exp_num in range(1, N_EXPERIMENTS + 1):
            print(f"\n{'-'*50}")
            print(f"EXPERIMENT {exp_num}/{N_EXPERIMENTS}")
            print(f"{'-'*50}")
            
            try:
                # Sample random data treatment configuration
                config = sample_configuration(rng)
                
                # Add fixed baseline methodological parameters that don't vary
                full_config = {**BASELINE_CONFIG, **config}
                
                print(f"Data treatment configuration:")
                print(config_to_string(config))
                
                # Save experiment parameters as JSON
                param_path = SENSITIVITY_DATA / f"experiment_{exp_num}_params.json"
                try:
                    # Convert numpy types to native Python types for JSON serialization
                    json_config = {}
                    for key, value in full_config.items():
                        if hasattr(value, 'item'):  # numpy scalar
                            json_config[key] = value.item()
                        else:
                            json_config[key] = value
                    
                    with open(param_path, 'w') as f:
                        json.dump(json_config, f, indent=2)
                    print(f"  Parameters saved: {param_path}")
                except Exception as json_e:
                    print(f"  Warning: Could not save parameters as JSON: {json_e}")
                
                # Run pipeline with this configuration
                results = run_full_pipeline(full_config, exp_num)
                
                if results is not None and len(results) > 0:
                    # Save experiment results
                    exp_path = SENSITIVITY_DATA / f"experiment_{exp_num}_ewbi.csv"
                    results.to_csv(exp_path, index=False)
                    
                    print(f"✓ Experiment {exp_num} completed successfully")
                    print(f"  Results: {len(results):,} rows, {len(results['Country'].unique())} countries")
                    print(f"  Saved: {exp_path}")
                    
                    successful_experiments += 1
                else:
                    print(f"✗ Experiment {exp_num} failed: No results generated")
                    print(f"  Pipeline returned None or empty results")
                    failed_experiments += 1
                    
            except KeyError as ke:
                print(f"✗ Experiment {exp_num} failed: Configuration error - {ke}")
                print(f"  Missing configuration parameter or file not found")
                failed_experiments += 1
            except FileNotFoundError as fe:
                print(f"✗ Experiment {exp_num} failed: File not found - {fe}")
                print(f"  Check data files and pipeline stage outputs")
                failed_experiments += 1
            except Exception as e:
                print(f"✗ Experiment {exp_num} failed: {type(e).__name__}: {e}")
                print(f"  Full error details available in pipeline output above")
                failed_experiments += 1
                    
            # Progress update
            if exp_num % 5 == 0:
                print(f"\nProgress: {exp_num}/{N_EXPERIMENTS} experiments completed")
                print(f"Success rate: {successful_experiments}/{exp_num} ({successful_experiments/exp_num*100:.1f}%)")
        
        # ===== CLEANUP & SUMMARY =====
        print("\n" + "="*70)
        print("CLEANUP & RESTORATION")
        print("="*70)
        
        # CRITICAL: Restore all original configurations
        try:
            restore_all_configs()
            print("✓ Baseline configuration restored - app data is safe")
        except Exception as e:
            print(f"✗ Error restoring configs: {e}")
            print("⚠️  WARNING: Manual restoration may be needed")
        
        # Check that protected directories were never modified
        violations = check_protected_directories(initial_protected_states)
        if violations:
            print(f"\n🚨 CRITICAL ERROR: Protected directories were modified!")
            for violation in violations:
                print(f"  {violation}")
            print(f"\n⚠️  The app data may have been corrupted!")
            print("\n✗ SENSITIVITY TEST FAILED - Protected data was modified")
        else:
            print(f"\n✅ SUCCESS: All protected directories remained unchanged")
            print(f"  • {len(PROTECTED_DIRS)} protected directories verified")
            print(f"  • {len(PROTECTED_FILES)} protected files verified")
            print(f"  • App data integrity maintained")
        
        print("\n" + "="*70)
        print("DATA TREATMENT EXPERIMENTS COMPLETE")
        print("="*70)
        print(f"\nResults:")
        print(f"  Successful experiments: {successful_experiments}")
        print(f"  Failed experiments: {failed_experiments}") 
        print(f"  Total experiments: {successful_experiments + failed_experiments}")
        print(f"  Success rate: {successful_experiments/(successful_experiments + failed_experiments)*100:.1f}%")
        
        if successful_experiments > 0:
            print(f"\n✓ Data treatment experiment data generated successfully")
            print(f"  Data location: {SENSITIVITY_DATA}")
            print(f"  Files: experiment_0_baseline_ewbi.csv, experiment_*_ewbi.csv")
            print(f"\n✓ Baseline app configuration preserved")
            print(f"  Main app data: OUTPUT_DIR/ewbi_master_aggregated.csv (unchanged)")
            print(f"\nNext steps:")
            print(f"  1. Run EU-27 aggregation: python {__file__} --eu27")
            print(f"  2. Generate visualizations: python 5_sensitivity_test_visuals_data_treatment.py")
            print(f"  3. Compare with methodological sensitivity results")
        else:
            print(f"\n✗ No experiments completed successfully")
            print(f"  Check pipeline configuration and data availability")
            print(f"\n✓ Baseline configuration restored despite failures")

    finally:
        # CRITICAL: Always restore configurations and check protections
        print("\n" + "="*70)
        print("CLEANUP AND PROTECTION CHECK")  
        print("="*70)
        
        # Restore all original configurations
        restore_all_configs()
        
        # Check that protected directories were never modified
        violations = check_protected_directories(initial_protected_states)
        if violations:
            print(f"\n🚨 CRITICAL ERROR: Protected directories were modified!")
            for violation in violations:
                print(f"  {violation}")
            print(f"\n⚠️  The app data may have been corrupted!")
            raise Exception("Protected directories were modified during sensitivity analysis")
        else:
            print(f"\n✅ SUCCESS: All protected directories remained unchanged")
            print(f"  • {len(PROTECTED_DIRS)} protected directories verified")
            print(f"  • {len(PROTECTED_FILES)} protected files verified")
            print(f"  • App data integrity maintained")


def main_eu27():
    """Main execution function for EU-27 computation from existing experiments."""
    
    print("\n" + "="*70)
    print("EU-27 AGGREGATION FROM EXISTING SENSITIVITY EXPERIMENTS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Purpose: Add EU-27 aggregations to existing experiment data")
    print(f"  Output directory: {SENSITIVITY_OUTPUT}")
    
    # ===== LOAD POPULATION DATA =====
    print("\n" + "="*70)
    print("LOADING POPULATION DATA")
    print("="*70)
    
    pop_path = DATA_DIR / 'population_transformed.csv'
    if not pop_path.exists():
        print(f"[ERROR] Population data not found: {pop_path}")
        return
    
    population_data = pd.read_csv(pop_path)
    print(f"[OK] Loaded population data: {len(population_data):,} rows")
    print(f"     Years: {population_data['year'].min()}-{population_data['year'].max()}")
    print(f"     Countries: {len(population_data['country'].unique())}")
    
    # ===== PROCESS EXISTING EXPERIMENTS =====
    print("\n" + "="*70)
    print("PROCESSING EXISTING EXPERIMENTS")
    print("="*70)
    
    experiments_processed = 0
    experiments_with_eu27 = 0
    
    # Check all experiment files
    for exp_num in range(1, N_EXPERIMENTS + 1):
        exp_path = SENSITIVITY_DATA / f"experiment_{exp_num}_ewbi.csv"
        
        if not exp_path.exists():
            print(f"[SKIP] Experiment {exp_num}: Data file not found")
            continue
        
        # Load experiment data
        exp_data = pd.read_csv(exp_path)
        
        # Check if EU-27 data already exists
        existing_eu27 = exp_data[exp_data['Country'] == 'EU-27']
        if len(existing_eu27) > 0:
            print(f"[SKIP] Experiment {exp_num}: EU-27 data already exists ({len(existing_eu27):,} rows)")
            experiments_with_eu27 += 1
            continue
        
        print(f"\n[PROCESS] Experiment {exp_num}:")
        print(f"  Input data: {len(exp_data):,} rows, {len(exp_data['Country'].unique())} countries")
        
        # Compute EU-27 aggregations
        eu27_data = compute_eu27_for_experiment(exp_data, population_data, exp_num)
        
        if len(eu27_data) > 0:
            print(f"  Generated: {len(eu27_data):,} EU-27 aggregations")
            
            # Combine with original data
            exp_data_with_eu27 = pd.concat([exp_data, eu27_data], ignore_index=True)
            
            # Save updated experiment data
            exp_data_with_eu27.to_csv(exp_path, index=False)
            print(f"  Saved: {exp_path.name} ({len(exp_data_with_eu27):,} total rows)")
            
            # Show sample EU-27 EWBI values
            eu27_ewbi = eu27_data[
                (eu27_data['Level'] == 1) & 
                (eu27_data['Decile'] == 'All Deciles')
            ]
            if len(eu27_ewbi) > 0:
                years = sorted(eu27_ewbi['Year'].unique())
                latest_year = years[-1] if years else None
                if latest_year:
                    latest_ewbi = eu27_ewbi[eu27_ewbi['Year'] == latest_year]['Value'].iloc[0]
                    print(f"  EU-27 EWBI ({latest_year}): {latest_ewbi:.3f}")
            
            experiments_processed += 1
        else:
            print(f"  [WARN] No EU-27 data could be generated (insufficient country coverage)")
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("EU-27 AGGREGATION COMPLETE")
    print("="*70)
    
    print(f"\nResults:")
    print(f"  Experiments processed: {experiments_processed}")
    print(f"  Experiments with existing EU-27: {experiments_with_eu27}")
    print(f"  Total experiments: {experiments_processed + experiments_with_eu27}")
    
    if experiments_processed > 0:
        print(f"\n✓ Successfully added EU-27 aggregations to {experiments_processed} experiments")
        print(f"  Data location: {SENSITIVITY_DATA}")
        print(f"  Files updated: experiment_*_ewbi.csv")
    
    if experiments_with_eu27 > 0:
        print(f"\n• {experiments_with_eu27} experiments already had EU-27 data")
    
    print(f"\nNext steps:")
    print(f"  - EU-27 data is now available in all experiment files")
    print(f"  - Use visualization scripts to analyze EU-27 sensitivity")
    print(f"  - Compare EU-27 ranking stability across methodological variations")


def main():
    """Main execution function with options for different modes."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--eu27':
        # Run EU-27 aggregation mode
        main_eu27()
    elif len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Monte Carlo Data Treatment Sensitivity Analysis Script")
        print("=" * 60)
        print("Usage:")
        print("  python 5_sensitivity_test_data_treatment.py            # Generate experiments")
        print("  python 5_sensitivity_test_data_treatment.py --eu27     # Add EU-27 data")
        print("  python 5_sensitivity_test_data_treatment.py --help     # Show this help")
        print()
        print("Modes:")
        print("  1. Default mode: Generates Monte Carlo experiments")
        print("     - Samples random data treatment parameter combinations")
        print("     - Varies: break_threshold, moving_average, window_size, mean_rescaling")
        print("     - Keeps methodological variables fixed at baseline")
        print("     - Runs full EWBI pipeline for each experiment")
        print("     - Saves results to experiment_*_ewbi.csv files")
        print()
        print("  2. EU-27 mode: Adds EU-27 aggregations to existing experiments")
        print("     - Requires existing experiment files")
        print("     - Computes population-weighted EU-27 averages")
        print("     - Updates experiment files with EU-27 data")
        print()
        print("Data Treatment Variables:")
        print("  - BREAK_THRESHOLD: [0.1, 0.2, 0.3]")
        print("  - APPLY_MOVING_AVERAGE: [False, True]")
        print("  - MOVING_AVERAGE_WINDOW: [3, 5]")
        print("  - APPLY_MEAN_RESCALING: [False, True]")
    else:
        # Default mode: Generate experiments
        print("Monte Carlo Data Treatment Sensitivity Analysis")
        print("=" * 60)
        print("This will generate data treatment sensitivity experiments.")
        print("Use --eu27 flag to add EU-27 aggregations to existing experiments.")
        print("Use --help flag for more information.")
        print()
        
        # Check if experiments already exist
        existing_experiments = list(SENSITIVITY_DATA.glob("experiment_*_ewbi.csv"))
        if len(existing_experiments) > 0:
            print(f"Found {len(existing_experiments)} existing experiment files.")
            response = input("Do you want to regenerate all experiments? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Aborted. Use --eu27 flag to process existing experiments.")
                return
            else:
                print("Regenerating all data treatment experiments...")
        
        # Run experiment generation
        run_monte_carlo_experiments()


if __name__ == '__main__':
    main()
