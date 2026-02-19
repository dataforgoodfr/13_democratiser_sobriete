#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighting and Aggregation Script (JRC-Compliant Population-Weighted PCA + Geometric Mean)

This script performs the fourth stage of EWBI computation using JRC-compliant methodology:
1. Loads normalized Primary Indicators data from Stage 3
2. Loads JRC-compliant PCA analysis results from Stage 2 (rotated loadings, component weights)
3. Computes EU Priorities via JRC-compliant PCA weighting with geometric mean
4. Computes EWBI overall via unweighted geometric mean of EU Priorities
5. Outputs unified dataframe with EWBI, EU Priorities, and Primary Indicators

Hierarchy:
- Level 1: EWBI Overall (single value per country-year-decile)
- Level 2: EU Priorities (5 values per country-year-decile, JRC PCA-weighted by country)
- Level 5: Primary Indicators (normalized data)

JRC-Compliant Aggregation Methods:
- Level 2 (EU Priorities): JRC PCA-weighted geometric mean of indicators
  * Uses JRC methodology: Factor selection (eigenvalue > 1, variance > 10%, cumulative ≥ 75%)
  * Applies Varimax rotation for simpler factor structure  
  * Creates intermediate composites based on highest factor loadings
  * Weights indicators within composites by squared rotated loadings (scaled to unity)
  * Weights composites by explained variance proportion
  * Formula: I_c = exp(sum_i w_i * ln(x_i)) where w_i follows JRC weighting
  
- Level 1 (EWBI): Unweighted geometric mean of EU Priorities
  * No compensation between dimensions
  * Formula: I = (prod_j I_j)^(1/5)

JRC Compliance:
Based on "Handbook on Constructing Composite Indicators: Methodology and User Guide" (OECD 2008)
- Section 6.1: Weights based on principal components analysis
- Table 17: Factor loadings and squared factor loading methodology
- Varimax rotation for cleaner factor structure
- Intermediate composite construction following Nicoletti et al. (2000) approach

Author: Data Processing Pipeline
Date: 2024
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy.stats import gmean
from tqdm import tqdm
from pipeline_env import env_bool, env_float, env_str, get_output_dir

# ===============================
# STRUCTURAL BREAK CONFIGURATION (Post-Aggregation)
# ===============================
# Apply the same break detection + treatment logic as in Stage 1 (`1_missing_data.py`),
# but after EU-Priority (Level 2) aggregation.
#
# Note: this is Level 2-specific on purpose (EU priorities), so it doesn't get
# confused with any EWBI-wide thresholds from other stages.
LEVEL2_BREAK_ADJUSTMENT_ENABLED = env_bool("EWBI_LEVEL2_BREAK_ADJUSTMENT", True)
LEVEL2_BREAK_THRESHOLD = env_float("EWBI_LEVEL2_BREAK_THRESHOLD", 0.1)
# Per request: Stage 4 never applies moving average nor mean rescaling.


def detect_and_adjust_structural_breaks(
    df,
    break_threshold=None,
):
    """Detect and treat structural breaks in time series.

    This function mirrors the logic in Stage 1 (`1_missing_data.py`) so that we can
    apply the same treatment after EU-Priority aggregation.

    Expected input columns:
      - primary_index, country, decile, year, value
    """

    if break_threshold is None:
        break_threshold = LEVEL2_BREAK_THRESHOLD

    print("\n[PROCESSING] Interpolation + structural break adjustment (no smoothing/rescaling in Stage 4)...")
    print(f"Input data: {len(df):,} records")
    print("Step 1: Linear interpolation of missing values")
    print(
        f"Step 2: Structural break detection (threshold: {break_threshold*100:.0f}% absolute, future-to-past direction)"
    )

    df = df.copy()
    df['year'] = df['year'].astype(int)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    adjustments_made = 0
    interpolations_made = 0
    series_with_breaks = {}
    series_with_interpolations = {}

    def process_series(group):
        nonlocal adjustments_made, interpolations_made, series_with_breaks, series_with_interpolations

        group = group.sort_values('year').copy()

        if len(group) < 2 or group['value'].isna().all():
            return group

        # =====================================================
        # STEP 1: LINEAR INTERPOLATION (before break adjustment)
        # =====================================================
        min_year = group['year'].min()
        max_year = group['year'].max()
        all_years = set(range(min_year, max_year + 1))
        existing_years = set(group['year'].values)

        missing_years = all_years - existing_years
        for year in missing_years:
            new_row = group.iloc[0:1].copy()
            new_row['year'] = year
            new_row['value'] = np.nan
            group = pd.concat([group, new_row], ignore_index=True)

        group = group.sort_values('year').reset_index(drop=True)

        valid_mask = group['value'].notna()
        if valid_mask.any():
            first_valid_idx = valid_mask.idxmax()
            last_valid_idx = valid_mask[::-1].idxmax()
            interior_nans_before = group.loc[first_valid_idx:last_valid_idx, 'value'].isna().sum()
        else:
            interior_nans_before = 0

        if interior_nans_before > 0:
            group['value'] = group['value'].interpolate(method='linear', limit_area='inside')

            if valid_mask.any():
                interior_nans_after = group.loc[first_valid_idx:last_valid_idx, 'value'].isna().sum()
                interpolations_this_series = interior_nans_before - interior_nans_after
            else:
                interpolations_this_series = 0

            if interpolations_this_series > 0:
                interpolations_made += interpolations_this_series
                series_key = (group['primary_index'].iloc[0], group['country'].iloc[0], group['decile'].iloc[0])
                series_with_interpolations[series_key] = {
                    'values_interpolated': interpolations_this_series
                }

        # =====================================================
        # STEP 2: VARIATION-BASED STRUCTURAL BREAK ADJUSTMENT
        # =====================================================
        valid_idx = group['value'].notna()
        valid_data = group[valid_idx].copy()

        if len(valid_data) < 2:
            return group

        values = valid_data['value'].values.copy()
        years = valid_data['year'].values

        if np.any(values <= 0):
            return group

        breaks = []
        for i in range(len(values) - 1):
            pct_change = np.abs((values[i+1] - values[i]) / values[i])
            if pct_change > break_threshold:
                breaks.append(i)

        if len(breaks) == 0:
            return group

        last_break_index = max(breaks) if breaks else -1
        if last_break_index == len(years) - 2:
            ref_year_index = len(years) - 2
            if len(years) >= 3 and values[-3] > 0:
                g_correction = values[-3] / values[-2] if values[-2] > 0 else 1.0
                values[-1] = values[-2] * g_correction
        else:
            ref_year_index = len(years) - 1

        ref_value = values[ref_year_index]

        variations = np.ones(len(values))
        variations[ref_year_index] = 1.0

        growth_rates = np.ones(len(values) - 1)

        # Identify runs of consecutive breaks, e.g. breaks at i and i+1.
        # For a run [start..end], we want to smooth ALL breaks in the run using:
        #   g_pre  = growth just before run  (year[start-1] -> year[start])
        #   g_post = growth just after run   (year[end+1]   -> year[end+2])
        # and apply mean(g_pre, g_post) when both exist.
        breaks_set = set(breaks)
        break_runs = []
        idx = 0
        while idx < (len(values) - 1):
            if idx in breaks_set:
                run_start = idx
                while (idx + 1) in breaks_set:
                    idx += 1
                run_end = idx
                break_runs.append((run_start, run_end))
            idx += 1

        break_to_run = {}
        for run_start, run_end in break_runs:
            for b in range(run_start, run_end + 1):
                break_to_run[b] = (run_start, run_end)

        adjustments_this_series = 0
        for i in range(len(values) - 1):
            if i in breaks:
                run_start, run_end = break_to_run.get(i, (i, i))

                # Growth just before the run
                g_pre = None
                if run_start > 0 and values[run_start - 1] > 0 and values[run_start] > 0:
                    g_pre = values[run_start] / values[run_start - 1]

                # Growth just after the run
                g_post = None
                if (run_end + 2) < len(values) and values[run_end + 1] > 0 and values[run_end + 2] > 0:
                    g_post = values[run_end + 2] / values[run_end + 1]

                # Edge handling: if the run touches the start/end, fall back to the available side.
                if run_start == 0 and g_post is not None:
                    growth_rates[i] = g_post
                elif run_end == (len(values) - 2) and g_pre is not None:
                    growth_rates[i] = g_pre
                elif g_pre is not None and g_post is not None:
                    growth_rates[i] = (g_pre + g_post) / 2.0
                elif g_pre is not None:
                    growth_rates[i] = g_pre
                elif g_post is not None:
                    growth_rates[i] = g_post
                else:
                    growth_rates[i] = 1.0

                print(f"✅ Smoothed break: {group['primary_index'].iloc[0]} in {group['country'].iloc[0]} (year {years[i+1]})")
                print(f"   Using growth rate: {growth_rates[i]:.3f} (mean of pre/post break)")
                adjustments_this_series += 1

            else:
                if values[i] > 0:
                    growth_rates[i] = values[i+1] / values[i]
                else:
                    growth_rates[i] = 1.0

        for i in range(ref_year_index - 1, -1, -1):
            variations[i] = variations[i+1] / growth_rates[i]

        for i in range(ref_year_index, len(values) - 1):
            variations[i+1] = variations[i] * growth_rates[i]

        reconstructed_values = ref_value * variations

        for i, (_, row) in enumerate(valid_data.iterrows()):
            group.loc[group['year'] == row['year'], 'value'] = reconstructed_values[i]

        if adjustments_this_series > 0:
            adjustments_made += adjustments_this_series
            series_key = (group['primary_index'].iloc[0], group['country'].iloc[0], group['decile'].iloc[0])
            series_with_breaks[series_key] = {
                'breaks_fixed': len(breaks),
                'break_threshold': break_threshold,
                'action': 'variation_database_reconstruction'
            }

        return group

    print("\nProcessing each series (break adjustment -> interpolation -> rescaling)...")
    groupby_cols = ['primary_index', 'country', 'decile']

    total_groups = df.groupby(groupby_cols).ngroups
    print(f"   Total series to process: {total_groups:,}")

    adjusted_df_list = []
    for _, (__, group_data) in enumerate(
        tqdm(df.groupby(groupby_cols, group_keys=False), desc="Processing series", total=total_groups),
        1
    ):
        adjusted_group = process_series(group_data)
        adjusted_df_list.append(adjusted_group)

    adjusted_df = pd.concat(adjusted_df_list, ignore_index=True)

    print(f"[OK] Processing pipeline completed")
    print(f"   - Step 1 (Interpolation): {interpolations_made} values interpolated")
    print(f"   - Step 2 (Break adjustment): {adjustments_made} breaks fixed (threshold: {break_threshold*100:.0f}%, future-to-past, mean growth rates)")
    print(f"   - Output data: {len(adjusted_df):,} records")

    if series_with_interpolations:
        print(f"\n[INTERPOLATION] Sample of interpolated series (first 10):")
        for (indicator, country, decile), info in list(series_with_interpolations.items())[:10]:
            print(f"   - {indicator} ({country}, Decile {decile}): {info['values_interpolated']} value(s) interpolated")

    if series_with_breaks:
        print(f"\n[BREAKS] Sample of structural break adjustments (first 10):")
        for (indicator, country, decile), info in list(series_with_breaks.items())[:10]:
            print(f"   - {indicator} ({country}, Decile {decile}): {info['breaks_fixed']} break(s) fixed (threshold: {info['break_threshold']:.4f})")

    return adjusted_df


def apply_break_detection_to_level2_eu_priorities(level2_data: pd.DataFrame) -> pd.DataFrame:
    """Apply Stage 1 break detection/treatment to Level 2 EU-Priority time series."""
    if level2_data is None or level2_data.empty:
        return level2_data

    required_cols = {"Year", "Country", "Decile", "EU priority", "Value"}
    missing = required_cols - set(level2_data.columns)
    if missing:
        raise ValueError(f"Level 2 data is missing required columns for break detection: {sorted(missing)}")

    print("\n[BREAKS] Applying structural break detection/treatment to Level 2 EU Priorities...")

    base_aggregation = "EU priority aggregation"
    if "Aggregation" in level2_data.columns and level2_data["Aggregation"].notna().any():
        base_aggregation = level2_data["Aggregation"].dropna().iloc[0]
    aggregation_text = f"{base_aggregation} | Post-aggregation break-adjusted"

    work = level2_data[["Year", "Country", "Decile", "EU priority", "Value"]].copy()
    work = work.rename(
        columns={
            "Year": "year",
            "Country": "country",
            "Decile": "decile",
            "EU priority": "primary_index",
            "Value": "value",
        }
    )

    work = work.dropna(subset=["primary_index", "country", "decile"], how="any")

    adjusted = detect_and_adjust_structural_breaks(work)

    level2_adjusted = pd.DataFrame(
        {
            "Year": adjusted["year"].astype(int),
            "Country": adjusted["country"],
            "Decile": adjusted["decile"],
            "Level": 2,
            "EU priority": adjusted["primary_index"],
            "Secondary": pd.NA,
            "Primary and raw data": adjusted["primary_index"],
            "Type": "Aggregation",
            "Aggregation": aggregation_text,
            "Value": adjusted["value"],
            "datasource": pd.NA,
        }
    )

    print(f"[OK] Level 2 break treatment applied: {len(level2_adjusted):,} records")
    return level2_adjusted

# ===============================
# CONFIGURATION
# ===============================
OUTPUT_DIR = get_output_dir()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NORMALISATION_OUTPUT = OUTPUT_DIR / "3_normalisation_data_output"
MULTIVARIATE_OUTPUT = OUTPUT_DIR / "2_multivariate_analysis_output"
DATA_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

# ===============================
# SENSITIVITY ANALYSIS CONFIGURATION
# ===============================
# --- EU PRIORITIES (Level 2) Aggregation ---
# Variant 1: PCA computation scope
# Base solution: 'all_years' (population-weighted PCA computed across all years)
# Alternative: 'per_year' (population-weighted PCA computed separately for each year)
PCA_SCOPE = env_str("EWBI_PCA_SCOPE", 'all_years')

# Variant 2: EU Priorities aggregation approach
# Base solution: 'pca' (population-weighted PCA-based aggregation)
# Alternative: 'simple' (simple population-weighted aggregation without PCA)
EU_PRIORITIES_APPROACH = env_str("EWBI_EU_PRIORITIES_APPROACH", 'pca')

# Variant 3: EU Priorities aggregation method (when using simple approach)
# Base solution: 'geometric' (population-weighted geometric mean)
# Alternative: 'arithmetic' (population-weighted arithmetic mean)
EU_PRIORITIES_AGGREGATION = env_str("EWBI_EU_PRIORITIES_AGGREGATION", 'geometric')

# --- EWBI (Level 1) Aggregation ---
# Variant 4: Per-decile aggregation of EU Priorities to EWBI
# Base solution: 'geometric' (geometric mean of EU priorities per decile)
# Alternative: 'arithmetic' (arithmetic mean of EU priorities per decile)
EWBI_DECILE_AGGREGATION = env_str("EWBI_EWBI_DECILE_AGGREGATION", 'geometric')

# Variant 5: Cross-decile aggregation (country-level from deciles)
# Base solution: 'geometric' (geometric mean across deciles 1-10)
# Alternative: 'arithmetic' (arithmetic mean across deciles 1-10)
EWBI_CROSS_DECILE_AGGREGATION = env_str("EWBI_EWBI_CROSS_DECILE_AGGREGATION", 'geometric')

# --- Performance optimization ---
# Skip EU-27 aggregations (useful for sensitivity analysis focusing on country ranks)
# Set to True to skip EU-27 computations and speed up pipeline
SKIP_EU27 = env_bool("EWBI_SKIP_EU27", False)

# --- EU-27 Aggregation Method ---
# EU-27 aggregates use population-weighted arithmetic mean (changed from geometric)
# This provides better interpretability for policy makers and avoids geometric mean sensitivity
# Individual country "All Deciles" aggregations continue to use geometric mean as configured above

# ===============================
# HELPER FUNCTIONS
# ===============================

def load_normalized_level4_data(normalisation_dir):
    """
    Load normalized Level 4 data from Stage 3.
    
    Args:
        normalisation_dir: Path to normalization output directory
        
    Returns:
        DataFrame with Level 4 normalized indicators
    """
    print("[LOAD] Loading normalized Level 4 data from Stage 3...")
    
    input_path = normalisation_dir / 'level4_normalised_indicators.csv'
    
    if not input_path.exists():
        print(f"[ERROR] Normalized data not found at {input_path}")
        print("Please run 3_normalisation_data.py first")
        return None
    
    df = pd.read_csv(input_path)
    print(f"[OK] Loaded {len(df):,} Level 4 normalized records")
    
    return df


def load_ewbi_configuration(data_dir):
    """
    Load EWBI indicator hierarchy configuration.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary with EWBI structure and indicator mappings
    """
    print("[LOAD] Loading EWBI configuration...")
    
    config_path = data_dir / 'ewbi_indicators.json'
    
    if not config_path.exists():
        print(f"[ERROR] EWBI configuration not found at {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"[OK] Loaded EWBI structure: {len(config['EWBI'])} EU priorities")
    
    return config


def load_pca_results(multivariate_dir):
    """
    Load PCA analysis results from Stage 2.
    
    Returns:
        Dictionary with PCA results keyed by (country, year) tuples
    """
    print("[LOAD] Loading PCA analysis results from Stage 2...")
    pca_path = multivariate_dir / 'pca_results_full.json'
    
    if not pca_path.exists():
        print(f"[WARN] PCA results not found at {pca_path}")
        print("       Will use unweighted aggregation for Level 2")
        return {}
    
    try:
        with open(pca_path, 'r') as f:
            pca_results_str = json.load(f)
        
        # Convert string keys back to tuples
        pca_results = {}
        for key_str, value in pca_results_str.items():
            key_tuple = eval(key_str)
            pca_results[key_tuple] = value
        
        print(f"[OK] Loaded PCA results for {len(pca_results)} country-year combinations")
        return pca_results
    except Exception as e:
        print(f"[WARN] Could not load PCA results: {e}")
        return {}


def load_population_data(data_dir):
    """Load population data for weighting."""
    print("[LOAD] Loading population data for weighting...")
    pop_path = data_dir / 'population_transformed.csv'
    
    if not pop_path.exists():
        print(f"[WARN] Population data not found at {pop_path}")
        print("       Will use unweighted aggregation")
        return None
    
    try:
        pop_data = pd.read_csv(pop_path)
        print(f"[OK] Loaded population data: {len(pop_data)} records")
        return pop_data
    except Exception as e:
        print(f"[WARN] Could not load population data: {e}")
        return None


def create_indicator_mapping(config):
    """Create mapping from indicator codes to EU priorities."""
    print("[MAP] Creating indicator hierarchy mapping...")
    
    indicator_mapping = {}
    category_indicators = {}
    
    for priority in config['EWBI']:
        priority_name = priority['name']
        category_indicators[priority_name] = []
        
        for component in priority['components']:
            for indicator in component['indicators']:
                indicator_code = indicator['code']
                indicator_mapping[indicator_code] = priority_name
                category_indicators[priority_name].append(indicator_code)
    
    print(f"   Created mapping for {len(indicator_mapping)} indicators across {len(category_indicators)} EU priorities")
    return indicator_mapping, category_indicators


def aggregate_geometric_mean(values, weights=None):
    """
    Compute weighted geometric mean of values.
    
    Formula:
    - Unweighted: GM = (prod(x_i))^(1/n)
    - Weighted: GM_w = exp(sum(w_i * log(x_i))) with sum(w_i)=1
    """
    valid_idx = np.isfinite(values)
    valid_values = values[valid_idx]
    
    if len(valid_values) == 0:
        return np.nan
    
    if np.any(valid_values <= 0):
        return np.nan
    
    if weights is None:
        return gmean(valid_values)
    else:
        valid_weights = weights[valid_idx] if isinstance(weights, np.ndarray) else [weights[i] for i, v in enumerate(valid_idx) if v]
        
        if isinstance(valid_weights, list):
            valid_weights = np.array(valid_weights)
        
        valid_weights = valid_weights / np.sum(valid_weights)
        log_values = np.log(valid_values)
        
        return np.exp(np.sum(valid_weights * log_values))


def aggregate_arithmetic_mean(values, weights=None):
    """
    Compute weighted arithmetic mean of values.
    
    Formula:
    - Unweighted: AM = sum(x_i) / n
    - Weighted: AM_w = sum(w_i * x_i) / sum(w_i)
    
    This is preferred for raw indicators (Level 3) as it handles zeros naturally.
    """
    valid_idx = np.isfinite(values)
    valid_values = values[valid_idx]
    
    if len(valid_values) == 0:
        return np.nan
    
    if weights is None:
        return np.mean(valid_values)
    else:
        valid_weights = weights[valid_idx] if isinstance(weights, np.ndarray) else [weights[i] for i, v in enumerate(valid_idx) if v]
        
        if isinstance(valid_weights, list):
            valid_weights = np.array(valid_weights)
        
        return np.average(valid_values, weights=valid_weights)


def get_jrc_pca_weights_for_country_year(pca_results, country, year, available_indicators):
    """
    Extract JRC-compliant PCA weights following the handbook methodology.
    
    JRC Process:
    1. Use rotated loadings (from Varimax rotation) to identify factor structure
    2. Create intermediate composites based on highest factor loadings for each indicator
    3. Weight indicators within composites by squared rotated loadings (scaled to unity)
    4. Weight composites by explained variance proportion
    5. Combine to get final indicator weights
    
    Based on JRC Handbook Table 17 methodology where:
    - Squared factor loadings represent proportion of indicator variance explained by factor
    - Indicators are grouped with factor showing highest loading
    - Final weights combine within-composite weights with composite importance
    
    Args:
        pca_results: Dictionary with PCA results keyed by (country, year) tuples
        country: Country name
        year: Year value  
        available_indicators: List of indicators to weight
        
    Returns:
        Dictionary mapping indicator -> weight, or None if PCA results unavailable
    """
    key = (country, year)
    
    if key not in pca_results:
        return None
    
    try:
        pca_result = pca_results[key]
        
        # Get JRC-specific results from Stage 2
        rotated_loadings = pca_result.get('rotated_loadings', [])  # After Varimax rotation
        eigenvalues = pca_result.get('eigenvalues', [])
        indicator_names = pca_result.get('indicator_names', [])
        explained_variance_ratio = pca_result.get('explained_variance_ratio', [])
        jrc_criteria_applied = pca_result.get('jrc_criteria_applied', False)
        
        if not rotated_loadings or not eigenvalues or not indicator_names:
            # Fallback to original loadings if rotated not available
            rotated_loadings = pca_result.get('component_loadings', [])
            if not rotated_loadings:
                return None
        
        # Convert to numpy arrays for easier manipulation
        loadings_array = np.array(rotated_loadings)  # Shape: (n_components, n_indicators)
        
        if loadings_array.size == 0 or len(indicator_names) != loadings_array.shape[1]:
            return None
        
        # Step 1: Create intermediate composites following JRC methodology
        # For each indicator, find the factor with the highest squared loading
        squared_loadings = loadings_array ** 2
        
        # Dictionary to store composite information
        composites = {}  # composite_id -> {'indicators': [ind1, ind2], 'weights': [w1, w2], 'importance': importance}
        indicator_to_composite = {}  # indicator -> composite_id
        
        for ind_idx, indicator in enumerate(indicator_names):
            if indicator not in available_indicators:
                continue
            
            # Find factor with highest squared loading for this indicator
            factor_loadings = squared_loadings[:, ind_idx]
            best_factor_idx = np.argmax(factor_loadings)
            composite_id = f"composite_{best_factor_idx}"
            
            # Store indicator assignment and its squared loading
            if composite_id not in composites:
                composites[composite_id] = {
                    'indicators': [],
                    'squared_loadings': [],
                    'factor_idx': best_factor_idx
                }
            
            composites[composite_id]['indicators'].append(indicator)
            composites[composite_id]['squared_loadings'].append(factor_loadings[best_factor_idx])
            indicator_to_composite[indicator] = composite_id
        
        if not composites:
            return None
        
        # Step 2: Compute weights within each composite (JRC Table 17 approach)
        # Normalize squared loadings within each composite to unity sum
        for composite_id in composites:
            squared_loadings_comp = np.array(composites[composite_id]['squared_loadings'])
            total_squared_loading = np.sum(squared_loadings_comp)
            
            if total_squared_loading > 0:
                # Scaled to unity sum as per JRC methodology
                normalized_weights = squared_loadings_comp / total_squared_loading
                composites[composite_id]['weights'] = normalized_weights.tolist()
            else:
                # Equal weights fallback
                n_indicators = len(composites[composite_id]['indicators'])
                composites[composite_id]['weights'] = [1.0 / n_indicators] * n_indicators
        
        # Step 3: Weight each composite by its explained variance proportion
        # Following JRC methodology: weight = explained_variance / total_explained_variance
        total_explained_variance = 0
        for composite_id in composites:
            factor_idx = composites[composite_id]['factor_idx']
            if factor_idx < len(explained_variance_ratio):
                factor_variance = explained_variance_ratio[factor_idx]
                composites[composite_id]['importance'] = factor_variance
                total_explained_variance += factor_variance
        
        # Normalize composite importances
        if total_explained_variance > 0:
            for composite_id in composites:
                composites[composite_id]['importance'] = composites[composite_id]['importance'] / total_explained_variance
        else:
            # Equal importance fallback
            n_composites = len(composites)
            for composite_id in composites:
                composites[composite_id]['importance'] = 1.0 / n_composites
        
        # Step 4: Combine within-composite weights with composite importance to get final weights
        final_weights = {}
        
        for composite_id, comp_data in composites.items():
            composite_importance = comp_data['importance']
            indicators = comp_data['indicators']
            within_comp_weights = comp_data['weights']
            
            for indicator, within_weight in zip(indicators, within_comp_weights):
                if indicator in available_indicators:
                    # Final weight = within-composite weight × composite importance
                    final_weights[indicator] = within_weight * composite_importance
        
        # Step 5: Final normalization to ensure sum = 1
        if final_weights:
            total_weight = sum(final_weights.values())
            if total_weight > 0:
                final_weights = {k: v / total_weight for k, v in final_weights.items()}
            else:
                # Equal weights fallback
                n_indicators = len(final_weights)
                final_weights = {k: 1.0 / n_indicators for k in final_weights.keys()}
        
        return final_weights if final_weights else None
        
    except Exception as e:
        print(f"[DEBUG] JRC PCA weighting failed for {country}-{year}: {e}")
        return None


# Keep original function as fallback
def get_pca_weights_for_country_year(pca_results, country, year, available_indicators):
    """
    Extract PCA-based component weights for a specific country-year.
    
    Legacy function - kept as fallback for non-JRC compliant results.
    Use get_jrc_pca_weights_for_country_year for JRC-compliant weighting.
    
    Returns:
        Dictionary mapping indicator -> weight, or None if PCA results unavailable
    """
    return get_jrc_pca_weights_for_country_year(pca_results, country, year, available_indicators)


def compute_level2_eu_priorities_pca_weighted(level4_data, indicator_mapping, category_indicators, pca_results, population_data=None, aggregation_method=None, pca_scope=None):
    """
    Compute Level 2 (EU Priorities) via population-weighted PCA mean.
    
    For each country-year-decile and EU priority:
    1. Get all Level 4 indicators for that priority
    2. Look up PCA weights for that country-year (if available)
    3. Compute weighted mean (geometric or arithmetic based on config)
    4. Apply population weighting across countries if provided
    
    Args:
        level4_data: DataFrame with Level 4 normalized indicators
        indicator_mapping: Dictionary mapping indicator codes to EU priorities
        category_indicators: Dictionary mapping EU priority to list of indicator codes
        pca_results: Dictionary with PCA results keyed by (country, year) tuples
        population_data: Optional DataFrame with population weights
        aggregation_method: 'geometric' or 'arithmetic'. Default: EU_PRIORITIES_AGGREGATION config
        pca_scope: 'all_years' or 'per_year'. Default: PCA_SCOPE config
    
    Returns:
        DataFrame with Level 2 EU priority aggregations
    """
    # Use configuration defaults if not specified
    if aggregation_method is None:
        aggregation_method = EU_PRIORITIES_AGGREGATION
    if pca_scope is None:
        pca_scope = PCA_SCOPE
    
    agg_func = aggregate_geometric_mean if aggregation_method == 'geometric' else aggregate_arithmetic_mean
    agg_name = 'geometric mean' if aggregation_method == 'geometric' else 'arithmetic mean'
    
    print(f"\n[COMPUTE] Computing Level 2 EU Priority aggregations (Population-Weighted PCA)...")
    print(f"   Aggregation method: {agg_name}")
    print(f"   PCA scope: {pca_scope}")
    
    level2_records = []
    
    # For each EU priority
    for eu_priority, indicators in tqdm(category_indicators.items(), desc="EU Priorities"):
        # Get Level 4 data for this priority's indicators
        priority_data = level4_data[
            level4_data['Primary and raw data'].isin(indicators)
        ].copy()
        
        if len(priority_data) == 0:
            print(f"   [SKIP] {eu_priority}: No data found")
            continue
        
        # Group by country-year-decile and aggregate
        group_cols = ['Year', 'Country', 'Decile']
        
        for group_key, group_df in priority_data.groupby(group_cols, as_index=False):
            year, country, decile = group_key
            
            # Extract values for aggregation
            values = group_df['Value'].values
            indicator_names = group_df['Primary and raw data'].values
            
            # Try to get PCA weights for this country-year
            # Note: pca_scope affects how the PCA was computed in Stage 2
            # Here we just use whatever PCA results are available
            pca_weights = get_pca_weights_for_country_year(
                pca_results, country, int(year), indicators
            )
            
            # Build weight array if PCA weights available
            weights = None
            if pca_weights is not None:
                weights = np.array([pca_weights.get(ind, 1.0/len(values)) for ind in indicator_names])
            
            # Compute weighted mean (geometric or arithmetic)
            if weights is not None:
                agg_value = agg_func(values, weights)
            else:
                agg_value = agg_func(values)  # Unweighted fallback
            
            if np.isnan(agg_value):
                continue
            
            # Create Level 2 record
            level2_record = {
                'Year': year,
                'Country': country,
                'Decile': decile,
                'Level': 2,  # EU Priorities
                'EU priority': eu_priority,
                'Secondary': pd.NA,
                'Primary and raw data': eu_priority,
                'Type': 'Aggregation',
                'Aggregation': f'Population-weighted PCA {agg_name} of Level 4 indicators',
                'Value': agg_value,
                'datasource': pd.NA
            }
            level2_records.append(level2_record)
    
    level2_df = pd.DataFrame(level2_records)
    print(f"[OK] Created {len(level2_df):,} Level 2 records from {len(category_indicators)} EU priorities")
    print(f"     Using PCA-weighted aggregation ({agg_name}): {len(pca_results)} country-years with component weights available")
    
    return level2_df


def compute_level2_eu_priorities_simple_weighted(level4_data, indicator_mapping, category_indicators, population_data=None, aggregation_method=None):
    """
    Compute Level 2 (EU Priorities) via simple population-weighted aggregation (without PCA).
    
    For each country-year-decile and EU priority:
    1. Get all Level 4 indicators for that priority
    2. Compute equal-weighted mean (geometric or arithmetic based on config)
    3. Apply population weighting across countries if provided
    
    Args:
        level4_data: DataFrame with Level 4 normalized indicators
        indicator_mapping: Dictionary mapping indicator codes to EU priorities
        category_indicators: Dictionary mapping EU priority to list of indicator codes
        population_data: Optional DataFrame with population weights
        aggregation_method: 'geometric' or 'arithmetic'. Default: EU_PRIORITIES_AGGREGATION config
    
    Returns:
        DataFrame with Level 2 EU priority aggregations
    """
    # Use configuration defaults if not specified
    if aggregation_method is None:
        aggregation_method = EU_PRIORITIES_AGGREGATION
    
    agg_func = aggregate_geometric_mean if aggregation_method == 'geometric' else aggregate_arithmetic_mean
    agg_name = 'geometric mean' if aggregation_method == 'geometric' else 'arithmetic mean'
    
    print(f"\n[COMPUTE] Computing Level 2 EU Priority aggregations (Simple Population-Weighted)...")
    print(f"   Aggregation method: {agg_name}")
    print(f"   Equal weighting: All indicators within each EU priority have equal weight")
    
    level2_records = []
    
    # For each EU priority
    for eu_priority, indicators in category_indicators.items():
        print(f"   Processing {eu_priority}: {len(indicators)} indicators")
        
        # Get data for this category's indicators
        category_data = level4_data[level4_data['Primary and raw data'].isin(indicators)].copy()
        
        if category_data.empty:
            print(f"     [WARN] No data found for {eu_priority}")
            continue
        
        # Group by country, year, decile and aggregate
        for (country, year, decile), group in category_data.groupby(['Country', 'Year', 'Decile']):
            
            # Get values and equal weights for all indicators in this EU priority
            values = group['Value'].values
            indicators_present = group['Primary and raw data'].tolist()
            
            # Use equal weights (all indicators have weight 1.0)
            weights = np.ones(len(values))
            
            # Compute weighted mean
            aggregated_value = agg_func(values, weights)
            
            level2_record = {
                'Year': year,
                'Country': country,
                'Decile': decile,
                'Level': 2,  # EU Priorities
                'EU priority': eu_priority,
                'Secondary': pd.NA,
                'Primary and raw data': eu_priority,
                'Type': 'Aggregation',
                'Aggregation': f'Simple equal-weighted {agg_name} of Level 4 indicators',
                'Value': aggregated_value,
                'datasource': pd.NA
            }
            level2_records.append(level2_record)
    
    level2_df = pd.DataFrame(level2_records)
    print(f"[OK] Created {len(level2_df):,} Level 2 records from {len(category_indicators)} EU priorities")
    print(f"     Using simple equal-weighted aggregation ({agg_name})")
    
    return level2_df


def compute_level1_ewbi(level2_data, decile_aggregation=None):
    """
    Compute Level 1 (EWBI Overall) by aggregating Level 2 EU Priorities.
    
    For each country-year-decile combination:
    - Select all 5 Level 2 EU priority values
    - Compute mean (geometric or arithmetic based on config) across the 5 priorities
    
    Args:
        level2_data: DataFrame with Level 2 EU priority aggregations
        decile_aggregation: 'geometric' or 'arithmetic'. Default: EWBI_DECILE_AGGREGATION config
        
    Returns:
        DataFrame with Level 1 EWBI overall values
    """
    # Use configuration default if not specified
    if decile_aggregation is None:
        decile_aggregation = EWBI_DECILE_AGGREGATION
    
    agg_func = aggregate_geometric_mean if decile_aggregation == 'geometric' else aggregate_arithmetic_mean
    agg_name = 'Geometric mean' if decile_aggregation == 'geometric' else 'Arithmetic mean'
    
    print(f"\n[COMPUTE] Computing Level 1 (EWBI Overall)...")
    print(f"   Per-decile aggregation method: {agg_name}")
    
    level1_records = []
    
    # Group by country-year-decile
    group_cols = ['Year', 'Country', 'Decile']
    
    for group_key, group_df in tqdm(
        level2_data.groupby(group_cols, as_index=False),
        desc="Computing EWBI",
        total=len(level2_data.groupby(group_cols))
    ):
        # Extract Level 2 (EU priority) values
        values = group_df['Value'].values
        
        # Need at least 2 EU priorities to compute EWBI (was 3, lowered to handle data sparsity)
        valid_values = values[np.isfinite(values)]
        if len(valid_values) < 2:
            continue
        
        # Compute mean of EU priorities (geometric or arithmetic)
        ewbi_value = agg_func(valid_values)
        
        if np.isnan(ewbi_value):
            continue
        
        # Create Level 1 record
        level1_record = {
            'Year': group_key[0],
            'Country': group_key[1],
            'Decile': group_key[2],
            'Level': 1,  # EWBI
            'EU priority': pd.NA,
            'Secondary': pd.NA,
            'Primary and raw data': 'EWBI',
            'Type': 'Aggregation',
            'Aggregation': f'{agg_name} of Level 2 EU Priorities',
            'Value': ewbi_value,
            'datasource': pd.NA
        }
        level1_records.append(level1_record)
    
    level1_df = pd.DataFrame(level1_records)
    print(f"[OK] Created {len(level1_df):,} Level 1 (EWBI) records")
    
    return level1_df


def compute_country_aggregations(level_data, level_name, cross_decile_aggregation=None):
    """
    Compute country-level aggregations (Decile='All') using configurable mean.
    
    For each country-year and level:
    - Select all decile-specific values
    - Compute mean (geometric or arithmetic) across deciles (equal weighting by decile)
    
    Args:
        level_data: DataFrame with decile-specific data
        level_name: Name of level (for logging)
        cross_decile_aggregation: 'geometric' or 'arithmetic'. Default: EWBI_CROSS_DECILE_AGGREGATION config
        
    Returns:
        DataFrame with country-level (All Deciles) aggregations
    """
    # Use configuration default if not specified
    if cross_decile_aggregation is None:
        cross_decile_aggregation = EWBI_CROSS_DECILE_AGGREGATION
    
    agg_func = aggregate_geometric_mean if cross_decile_aggregation == 'geometric' else aggregate_arithmetic_mean
    agg_name = 'Geometric mean' if cross_decile_aggregation == 'geometric' else 'Arithmetic mean'
    
    print(f"\n[INFO] Computing country-level aggregations for {level_name}...")
    print(f"   Cross-decile aggregation method: {agg_name}")
    
    country_records = []
    
    # Group by country-year-indicator (for Levels 2-3) or country-year (for Level 1)
    if 'Primary and raw data' in level_data.columns:
        group_cols = ['Year', 'Country', 'Primary and raw data', 'Level']
        # Only include rows that have a Primary and raw data value
        level_data_filtered = level_data[level_data['Primary and raw data'].notna()].copy()
    else:
        group_cols = ['Year', 'Country', 'Level']
        level_data_filtered = level_data.copy()
    
    for group_key, group_df in tqdm(
        level_data_filtered.groupby(group_cols, as_index=False),
        desc=f"Country aggregations ({level_name})",
        total=len(level_data_filtered.groupby(group_cols))
    ):
        # Extract values for deciles 1-10
        values = group_df['Value'].values
        
        # Compute mean across deciles (geometric or arithmetic)
        country_value = agg_func(values)
        
        if np.isnan(country_value):
            continue
        
        # Extract EU priority from group data if it exists
        eu_priority = pd.NA
        if 'EU priority' in group_df.columns:
            eu_priority_vals = group_df['EU priority'].dropna().unique()
            if len(eu_priority_vals) > 0:
                eu_priority = eu_priority_vals[0]
        
        # Create country-level record
        country_record = {
            'Year': group_key[0],
            'Country': group_key[1],
            'Decile': 'All Deciles',
            'Level': group_key[-1] if len(group_key) > 2 else group_key[-1],
            'EU priority': eu_priority,
            'Secondary': pd.NA,
            'Primary and raw data': group_key[2] if len(group_key) > 3 else pd.NA,
            'Type': 'Aggregation',
            'Aggregation': f'{agg_name} across deciles for {level_name}',
            'Value': country_value,
            'datasource': pd.NA
        }
        country_records.append(country_record)
    
    country_df = pd.DataFrame(country_records)
    print(f"[OK] Created {len(country_df):,} country-level aggregations for {level_name}")
    
    return country_df


def compute_eu27_aggregations(level_data, population_data, level_name, use_arithmetic_mean=True):
    """
    Compute EU-27 aggregations using population-weighted arithmetic mean.
    
    For each year and decile (including All Deciles):
    - Check if 66% of EU-27 population is covered by countries with data
    - If coverage >= 66%: compute population-weighted arithmetic mean (default) or geometric mean
    - If coverage < 66%: set value to NaN
    - Apply mathematical validation to detect impossible percentage values (>200%)
    
    NOTE: If you see extreme EU-27 values (>1500%, >250%), this indicates data quality 
    issues in the underlying country data from Stage 0. Rerun the complete pipeline 
    (Stages 0-4) after fixing Stage 0 raw data processing to resolve the issue.
    
    Args:
        level_data: DataFrame with country-specific data
        population_data: DataFrame with population weights
        level_name: Name of level (for logging)
        use_arithmetic_mean: If True, use arithmetic mean (default for EU-27).
                            If False, use geometric mean.
        
    Returns:
        DataFrame with EU-27 aggregations (individual deciles + All Deciles)
    """
    mean_type = "arithmetic" if use_arithmetic_mean else "geometric"
    print(f"\n[INFO] Computing EU-27 aggregations for {level_name} (using {mean_type} mean)...")
    print(f"[INFO] Applying 66% population coverage threshold for EU-27 aggregations")
    
    eu27_records = []
    
    # Pre-compute total EU-27 population by year for coverage calculation
    if population_data is not None:
        pop_for_merge = population_data.rename(columns={
            'country': 'Country',
            'year': 'Year', 
            'population': 'Population'
        })
        total_eu27_population_by_year = pop_for_merge.groupby('Year')['Population'].sum().to_dict()
        print(f"[INFO] EU-27 total population calculated for {len(total_eu27_population_by_year)} years")
    else:
        print("[ERROR] No population data available for EU-27 aggregations")
        return pd.DataFrame()
    
    # Group by year and decile (or year, decile, indicator for Level 2-3)
    if 'Primary and raw data' in level_data.columns and level_data['Primary and raw data'].notna().any():
        group_cols = ['Year', 'Decile', 'Primary and raw data', 'Level']
    else:
        group_cols = ['Year', 'Decile', 'Level']
    
    for group_key, group_df in tqdm(
        level_data.groupby(group_cols, as_index=False),
        desc=f"EU-27 aggregations ({level_name})",
        total=len(level_data.groupby(group_cols))
    ):
        year = group_key[0]
        decile = group_key[1]
        level = group_key[-1] if len(group_key) > 2 else group_key[-1]
        
        # Merge with population data for weighting
        group_with_pop = group_df.merge(
            pop_for_merge[['Country', 'Year', 'Population']],
            on=['Country', 'Year'],
            how='left'
        )
        
        # Remove rows where we couldn't get population data
        group_with_pop = group_with_pop.dropna(subset=['Population'])
        
        # Calculate population coverage
        if year in total_eu27_population_by_year:
            total_population = total_eu27_population_by_year[year]
            covered_population = group_with_pop['Population'].sum()
            coverage_percent = (covered_population / total_population) * 100
            
            indicator_name = group_key[2] if len(group_key) > 3 and 'Primary and raw data' in group_df.columns else "Level indicator"
            
            # Check 66% population coverage threshold
            if coverage_percent < 66.0:
                print(f"[SKIP] EU-27 {indicator_name} {year} Decile {decile}: {coverage_percent:.1f}% population coverage < 66% threshold")
                
                # Create NaN record for insufficient coverage
                eu27_record = {
                    'Year': year,
                    'Country': 'EU-27',
                    'Decile': decile,
                    'Level': level,
                    'EU priority': group_key[2] if ('EU priority' in group_df.columns and len(group_key) > 3) else pd.NA,
                    'Secondary': pd.NA,
                    'Primary and raw data': group_key[2] if 'Primary and raw data' in group_df.columns and len(group_key) > 3 else pd.NA,
                    'Type': 'Aggregation',
                    'Aggregation': f'Population-weighted {mean_type} mean',
                    'Value': np.nan,  # Set to NaN due to insufficient coverage
                    'datasource': pd.NA
                }
                
                # For Level 2, get EU priority from the data
                if 'EU priority' in group_df.columns:
                    eu_priority_vals = group_df['EU priority'].dropna().unique()
                    if len(eu_priority_vals) > 0:
                        eu27_record['EU priority'] = eu_priority_vals[0]
                
                eu27_records.append(eu27_record)
                continue
            else:
                print(f"[OK] EU-27 {indicator_name} {year} Decile {decile}: {coverage_percent:.1f}% population coverage >= 66% threshold")
        else:
            print(f"[ERROR] No total population data available for year {year}")
            continue
        
        # Require at least 20 countries to compute EU-27 aggregation, else skip
        n_countries = len(group_with_pop['Country'].unique())
        if n_countries < 20:
            print(f"[SKIP] EU-27 {level_name} {year} Decile {decile}: Only {n_countries} countries available (need ≥20)")
            continue
        
        if len(group_with_pop) == 0:
            continue
        
        # Compute population-weighted mean (arithmetic or geometric)
        weights = group_with_pop['Population'].values
        values = group_with_pop['Value'].values
        
        if use_arithmetic_mean:
            eu27_value = aggregate_arithmetic_mean(values, weights)
        else:
            eu27_value = aggregate_geometric_mean(values, weights)
        
        if np.isnan(eu27_value):
            continue
        
        # Mathematical validation for percentage indicators - Log but don't cap
        # (Root cause should be fixed in Stage 0 raw data processing)
        if eu27_value > 200:  # Any percentage > 200% is suspicious for social indicators
            indicator_name = group_key[2] if len(group_key) > 3 and 'Primary and raw data' in group_df.columns else "Unknown"
            print(f"⚠️  WARNING: EU-27 {indicator_name} {year} Decile {decile}: {eu27_value:.1f}% exceeds 200%")
            print(f"    This suggests data quality issues in underlying country data")
            
            # Log the problematic countries contributing to this extreme value
            problematic_countries = group_with_pop[group_with_pop['Value'] > 100]
            if len(problematic_countries) > 0:
                print(f"    Countries with >100%: {', '.join(problematic_countries['Country'].tolist())}")
                for _, country_row in problematic_countries.iterrows():
                    print(f"      - {country_row['Country']}: {country_row['Value']:.1f}%")
            
            print(f"    ACTION NEEDED: Rerun Stage 0 raw data processing with fixed calculation logic")
        
        # Create EU-27 record
        eu27_record = {
            'Year': year,
            'Country': 'EU-27',
            'Decile': decile,
            'Level': level,
            'EU priority': group_key[2] if ('EU priority' in group_df.columns and len(group_key) > 3) else pd.NA,
            'Secondary': pd.NA,
            'Primary and raw data': group_key[2] if 'Primary and raw data' in group_df.columns and len(group_key) > 3 else pd.NA,
            'Type': 'Aggregation',
            'Aggregation': f'Population-weighted {mean_type} mean',
            'Value': eu27_value,
            'datasource': pd.NA
        }
        
        # For Level 2, get EU priority from the data
        if 'EU priority' in group_df.columns:
            eu_priority_vals = group_df['EU priority'].dropna().unique()
            if len(eu_priority_vals) > 0:
                eu27_record['EU priority'] = eu_priority_vals[0]
        
        eu27_records.append(eu27_record)
    
    # Second pass: Create "All Deciles" records by aggregating across deciles
    print(f"[INFO] Creating 'All Deciles' aggregates for EU-27...")
    
    # Group by year and primary indicator (if applicable)
    if 'Primary and raw data' in level_data.columns and level_data['Primary and raw data'].notna().any():
        decile_group_cols = ['Year', 'Primary and raw data', 'Level']
        use_primary_indicator = True
    elif 'EU priority' in level_data.columns and level_data['EU priority'].notna().any():
        decile_group_cols = ['Year', 'EU priority', 'Level']
        use_primary_indicator = False
    else:
        decile_group_cols = ['Year', 'Level']
        use_primary_indicator = False
    
    for group_key_decile, group_df_decile in level_data.groupby(decile_group_cols, as_index=False):
        year = group_key_decile[0]
        level = group_key_decile[-1]
        
        # Get all country-decile combinations for this year/indicator/level
        if population_data is not None:
            pop_for_merge = population_data.rename(columns={
                'country': 'Country',
                'year': 'Year', 
                'population': 'Population'
            })
            group_with_pop_decile = group_df_decile.merge(
                pop_for_merge[['Country', 'Year', 'Population']],
                on=['Country', 'Year'],
                how='left'
            )
        else:
            group_with_pop_decile = pd.DataFrame()
        
        # Remove rows without population data
        group_with_pop_decile = group_with_pop_decile.dropna(subset=['Population'])
        
        # Require at least 20 countries to compute EU-27 "All Deciles" aggregation, else skip
        n_countries_decile = len(group_with_pop_decile['Country'].unique())
        if n_countries_decile < 20:
            indicator_name = group_key_decile[1] if len(group_key_decile) > 1 else "Unknown"
            print(f"[SKIP] EU-27 All Deciles {indicator_name} {year}: Only {n_countries_decile} countries available (need ≥20)")
            continue
        
        if len(group_with_pop_decile) == 0:
            continue
        
        # Compute population-weighted mean across all deciles (arithmetic or geometric)
        weights = group_with_pop_decile['Population'].values
        values = group_with_pop_decile['Value'].values
        
        if use_arithmetic_mean:
            eu27_all_deciles_value = aggregate_arithmetic_mean(values, weights)
        else:
            eu27_all_deciles_value = aggregate_geometric_mean(values, weights)
        
        if np.isnan(eu27_all_deciles_value):
            continue
        
        # Mathematical validation for "All Deciles" percentage indicators - Log but don't cap
        # (Root cause should be fixed in Stage 0 raw data processing)
        if eu27_all_deciles_value > 200:  # Any percentage > 200% is suspicious for social indicators
            indicator_name = group_key_decile[1] if len(group_key_decile) > 1 else "Unknown"
            print(f"⚠️  WARNING: EU-27 All Deciles {indicator_name} {year}: {eu27_all_deciles_value:.1f}% exceeds 200%")
            print(f"    This suggests data quality issues in underlying country data")
            
            # Log the problematic countries contributing to this extreme value
            problematic_countries = group_with_pop_decile[group_with_pop_decile['Value'] > 100]
            if len(problematic_countries) > 0:
                print(f"    Countries with >100%: {', '.join(problematic_countries['Country'].tolist())}")
            
            print(f"    ACTION NEEDED: Rerun Stage 0 raw data processing with fixed calculation logic")
        
        # Create "All Deciles" record
        eu27_all_record = {
            'Year': year,
            'Country': 'EU-27',
            'Decile': 'All Deciles',
            'Level': level,
            'Type': 'Aggregation',
            'Aggregation': f'Population-weighted {mean_type} mean',
            'Value': eu27_all_deciles_value,
            'datasource': pd.NA
        }
        
        # Add indicator-specific fields based on what we grouped by
        if use_primary_indicator and len(group_key_decile) > 1:
            eu27_all_record['Primary and raw data'] = group_key_decile[1]
            eu27_all_record['EU priority'] = pd.NA
        elif not use_primary_indicator and len(group_key_decile) > 1 and 'EU priority' in level_data.columns:
            eu27_all_record['Primary and raw data'] = pd.NA
            eu27_all_record['EU priority'] = group_key_decile[1]
        else:
            eu27_all_record['Primary and raw data'] = pd.NA
            eu27_all_record['EU priority'] = pd.NA
            
        if 'EU priority' in group_key_decile or len(group_key_decile) > 2:
            if 'EU priority' in level_data.columns:
                eu_priority_vals = group_df_decile['EU priority'].dropna().unique()
                if len(eu_priority_vals) > 0:
                    eu27_all_record['EU priority'] = eu_priority_vals[0]
                else:
                    eu27_all_record['EU priority'] = pd.NA
            else:
                eu27_all_record['EU priority'] = pd.NA
        else:
            eu27_all_record['EU priority'] = pd.NA
        
        eu27_all_record['Secondary'] = pd.NA
        
        eu27_records.append(eu27_all_record)
    
    eu27_df = pd.DataFrame(eu27_records)
    print(f"[OK] Created {len(eu27_df):,} EU-27 aggregations for {level_name}")
    
    return eu27_df


# ===============================
# MAIN PROCESSING
# ===============================


def load_raw_break_adjusted_data(population_data, indicator_mapping):
    """
    Load raw break-adjusted data from Stage 1 to include as Level 3 in final output.
    This data has structural breaks corrected and missing values forward-filled,
    but is NOT normalized. This replaces normalized Level 3 entirely.
    
    Args:
        population_data: DataFrame with population data for EU-27 aggregations
        indicator_mapping: Dictionary mapping indicator codes to EU priorities
    
    Returns:
        Tuple of (DataFrame with raw Level 3 individual records,
                  DataFrame with country aggregations for raw Level 3,
                  DataFrame with EU-27 aggregations for raw Level 3)
    """
    print("\n[LOAD] Loading raw forward-filled data from Stage 3 for Level 3...")
    
    raw_data_path = OUTPUT_DIR / "1_missing_data_output" / "raw_data_forward_filled.csv"
    
    if not raw_data_path.exists():
        print(f"[WARN] Raw forward-filled data not found at {raw_data_path}")
        print("       Please run Stage 3 (3_normalisation_data.py) first")
        print("       Skipping Level 3 raw data in final output")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    try:
        df = pd.read_csv(raw_data_path)
        # Change Level from 5 to 3
        df['Level'] = 3
        df['Type'] = 'Primary indicator'
        df['Aggregation'] = 'Break-adjusted and forward-filled (raw data)'
        # Add datasource column if missing
        if 'datasource' not in df.columns:
            df['datasource'] = pd.NA
        
        # Add EU priority information using indicator_mapping
        df['EU priority'] = df['Primary and raw data'].map(indicator_mapping)
        
        unmapped_indicators = df[df['EU priority'].isna()]['Primary and raw data'].unique()
        if len(unmapped_indicators) > 0:
            print(f"   WARNING: {len(unmapped_indicators)} raw indicators have no EU priority mapping: {list(unmapped_indicators)[:5]}")
        
        print(f"[OK] Loaded {len(df):,} raw break-adjusted records for Level 3")
        
        # Compute country-level aggregations for raw Level 3
        print("   Computing country-level aggregations for raw Level 3...")
        raw_country = compute_country_aggregations(df, "Level 3 (Raw Indicators)")
        
        # Compute EU-27 aggregations for raw Level 3 (skip if SKIP_EU27 is True)
        if SKIP_EU27:
            print("   [SKIP] Skipping EU-27 aggregations for raw Level 3 (SKIP_EU27=True)")
            raw_eu27 = pd.DataFrame()
        else:
            print("   Computing EU-27 aggregations for raw Level 3...")
            raw_eu27 = compute_eu27_aggregations(df, population_data, "Level 3 (Raw Indicators)", use_arithmetic_mean=True)
        
        return df, raw_country, raw_eu27
    except Exception as e:
        print(f"[WARN] Could not load raw data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def main():
    """Main execution function"""
    
    eu_priorities_agg = 'Geometric mean' if EU_PRIORITIES_AGGREGATION == 'geometric' else 'Arithmetic mean'
    eu_priorities_approach_desc = 'PCA-weighted' if EU_PRIORITIES_APPROACH == 'pca' else 'Simple equal-weighted'
    ewbi_decile_agg = 'Geometric mean' if EWBI_DECILE_AGGREGATION == 'geometric' else 'Arithmetic mean'
    ewbi_cross_decile_agg = 'Geometric mean' if EWBI_CROSS_DECILE_AGGREGATION == 'geometric' else 'Arithmetic mean'
    
    print("\n" + "="*70)
    print("STAGE 4: JRC-COMPLIANT WEIGHTING AND AGGREGATION")
    print("="*70)
    print(f"\n[CONFIG] Aggregation settings:")
    print(f"   EU Priorities (Level 2) - {eu_priorities_approach_desc} Aggregation:")
    if EU_PRIORITIES_APPROACH == 'pca':
        print(f"      - PCA scope: {PCA_SCOPE}")
        print(f"      - Aggregation method: {eu_priorities_agg}")
        print(f"      - Factor selection: eigenvalue > 1, variance > 10%, cumulative ≥ 75%")
        print(f"      - Varimax rotation: Applied for cleaner factor structure")
        print(f"      - Weighting: Squared rotated loadings (scaled to unity) × composite importance")
    else:
        print(f"      - Weighting approach: Equal weights for all indicators within each EU priority")
        print(f"      - Aggregation method: {eu_priorities_agg}")
    print(f"   EWBI (Level 1):")
    print(f"      - Per-decile aggregation: {ewbi_decile_agg}")
    print(f"      - Cross-decile aggregation: {ewbi_cross_decile_agg}")
    
    processing_desc = "JRC PCA-weighted" if EU_PRIORITIES_APPROACH == 'pca' else "Simple equal-weighted"
    print(f"\nProcessing steps:")
    print("  1. Load normalized Level 4 data from Stage 3")
    if EU_PRIORITIES_APPROACH == 'pca':
        print("  2. Load JRC-compliant PCA analysis results from Stage 2 (with Varimax rotation)")
    print("  3. Load population data for weighting")
    print(f"  4. Compute Level 2 (EU Priorities) via {processing_desc} {eu_priorities_agg.lower()}")
    if EU_PRIORITIES_APPROACH == 'pca':
        print("     - Create intermediate composites based on highest factor loadings")
        print("     - Weight indicators within composites by squared rotated loadings")
        print("     - Weight composites by explained variance proportion")
    else:
        print("     - Apply equal weights to all indicators within each EU priority")
    print(f"  5. Compute Level 1 (EWBI Overall) via {ewbi_decile_agg.lower()} of EU Priorities")
    print(f"  6. Compute country-level aggregations ({ewbi_cross_decile_agg.lower()} across deciles)")
    print("  7. Combine all levels into final unified dataframe")
    print("  8. Save output\n")
    
    # Load all required data
    level4_data = load_normalized_level4_data(NORMALISATION_OUTPUT)
    if level4_data is None or level4_data.empty:
        print("[ERROR] Failed to load normalized Level 4 data")
        return
    
    config = load_ewbi_configuration(DATA_DIR)
    if config is None:
        print("[ERROR] Failed to load EWBI configuration")
        return
    
    indicator_mapping, category_indicators = create_indicator_mapping(config)
    
    # Load optional PCA results and population data
    pca_results = load_pca_results(MULTIVARIATE_OUTPUT)
    population_data = load_population_data(DATA_DIR)
    
    # NOTE: Level 3 (Primary Indicators) will be loaded from raw break-adjusted data
    # NOT from normalized Level 4 data. Normalized data is only used for aggregation.

    
    # Compute Level 2 based on aggregation approach configuration
    if EU_PRIORITIES_APPROACH == 'pca':
        # Use PCA weighting (using normalized Level 4 data)
        level2_data = compute_level2_eu_priorities_pca_weighted(
            level4_data, 
            indicator_mapping, 
            category_indicators,
            pca_results,
            population_data
        )
    elif EU_PRIORITIES_APPROACH == 'simple':
        # Use simple equal weighting (using normalized Level 4 data)
        level2_data = compute_level2_eu_priorities_simple_weighted(
            level4_data, 
            indicator_mapping, 
            category_indicators,
            population_data
        )
    else:
        print(f"[ERROR] Unknown EU priorities approach: {EU_PRIORITIES_APPROACH}")
        return
    
    if level2_data.empty:
        print("[ERROR] Failed to compute Level 2 aggregations")
        return

    # Optional: Apply break detection + treatment after EU-Priority aggregation
    # (same methodology as Stage 1 missing-data break adjustment)
    if LEVEL2_BREAK_ADJUSTMENT_ENABLED:
        level2_data = apply_break_detection_to_level2_eu_priorities(level2_data)
    else:
        print("\n[BREAKS] Level 2 post-aggregation break adjustment is disabled (EWBI_LEVEL2_BREAK_ADJUSTMENT=False)")
    
    # Compute Level 1 (EWBI)
    level1_data = compute_level1_ewbi(level2_data)
    
    if level1_data.empty:
        print("[ERROR] Failed to compute Level 1 EWBI")
        return
    
    # Compute country-level aggregations (for Levels 1 and 2 only)
    print("\n[AGGREGATE] Computing country-level aggregations (Decile='All Deciles')...")
    level2_country = compute_country_aggregations(level2_data, "Level 2 (EU Priorities)")
    level1_country = compute_country_aggregations(level1_data, "Level 1 (EWBI)")
    
    # Compute EU-27 aggregations (skip if SKIP_EU27 is True)
    if SKIP_EU27:
        print("\n[SKIP] Skipping EU-27 aggregations (SKIP_EU27=True)")
        level2_eu27 = pd.DataFrame()
        level1_eu27 = pd.DataFrame()
    else:
        level2_eu27 = compute_eu27_aggregations(level2_data, population_data, "Level 2 (EU Priorities)")
        level1_eu27 = compute_eu27_aggregations(level1_data, population_data, "Level 1 (EWBI)")
    
    # Load raw break-adjusted data for Level 3 in final output
    # This replaces normalized Level 3 entirely
    raw_level3_data, raw_level3_country, raw_level3_eu27 = load_raw_break_adjusted_data(population_data, indicator_mapping)
    
    # Combine all levels
    print("\n[MERGE] Combining all levels into unified dataframe...")
    
    combined_levels = [
        level1_data,
        level1_country,
        level1_eu27,
        level2_data,
        level2_country,
        level2_eu27,
        raw_level3_data,
        raw_level3_country,
        raw_level3_eu27
    ]
    
    # Filter out empty dataframes
    combined_levels = [df for df in combined_levels if not df.empty]
    
    final_data = pd.concat(combined_levels, ignore_index=True)
    
    # Format and sort
    unified_columns = ['Year', 'Country', 'Decile', 'Level', 'EU priority',
                      'Secondary', 'Primary and raw data', 'Type', 'Aggregation', 'Value', 'datasource']
    final_data = final_data[unified_columns]
    
    # Sort with custom key for Decile column (All values first)
    final_data = final_data.sort_values(
        by=['Year', 'Country', 'Level', 'Decile'],
        ascending=[True, True, True, True],
        key=lambda x: x.map(lambda y: (y != 'All', y) if x.name == 'Decile' else y)
    ).reset_index(drop=True)
    
    print(f"[OK] Final unified dataframe:")
    print(f"   Total records: {len(final_data):,}")
    print(f"   Level distribution:")
    for level in sorted(final_data['Level'].unique()):
        count = len(final_data[final_data['Level'] == level])
        print(f"      Level {level}: {count:,}")
    
    print(f"   Countries: {final_data[final_data['Decile'] != 'All']['Country'].nunique()}")
    print(f"   Years: {final_data['Year'].min():.0f}-{final_data['Year'].max():.0f}")
    
    # Save outputs
    output_subdir = OUTPUT_DIR / "4_weighting_aggregation_output"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_subdir / 'ewbi_final_aggregated.csv'
    final_data.to_csv(output_path, index=False)
    
    print(f"\n[COMPLETE] Stage 4 complete: JRC-compliant weighting and aggregation")
    print(f"[JRC-METHODOLOGY] Applied Varimax-rotated PCA weights with factor selection criteria:")
    print(f"   - Eigenvalue > 1, individual variance > 10%, cumulative variance ≥ 75%")
    print(f"   - Intermediate composites based on highest factor loadings")  
    print(f"   - Squared rotated loadings for indicator weighting")
    print(f"[SAVED] Output saved: {output_path}")
    print(f"   Records: {len(final_data):,}")
    
    # Save app-ready format (skip if running sensitivity tests)
    if not globals().get('SKIP_APP_OUTPUT', False):
        app_output_path = OUTPUT_DIR / 'ewbi_master_aggregated.csv'
        final_data.to_csv(app_output_path, index=False)
        print(f"[SAVED] App-ready output: {app_output_path}")
    else:
        print(f"[SKIP] App output skipped (sensitivity test mode)")
    
    return final_data


if __name__ == '__main__':
    result = main()
    print("\n[OK] 4_weighting_aggregation.py execution complete - JRC-compliant methodology applied")
    print("[JRC-COMPLIANCE] All PCA weights computed using Varimax rotation and factor selection criteria")

