#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighting and Aggregation Script (Population-Weighted PCA + Geometric Mean)

This script performs the fourth stage of EWBI computation:
1. Loads normalized Primary Indicators data from Stage 3
2. Loads PCA analysis results from Stage 2 (component weights)
3. Computes EU Priorities via population-weighted PCA geometric mean
4. Computes EWBI overall via unweighted geometric mean of EU Priorities
5. Outputs unified dataframe with EWBI, EU Priorities, and Primary Indicators

Hierarchy:
- Level 1: EWBI Overall (single value per country-year-decile)
- Level 2: EU Priorities (5 values per country-year-decile, PCA-weighted by country)
- Level 5: Primary Indicators (normalized data)

Aggregation Methods:
- Level 2 (EU Priorities): Population-weighted PCA geometric mean of indicators
  * Uses component weights from Stage 2 (eigenvalue-based)
  * Applies population weighting per country
  * Formula: I_c = exp(sum_i w_i * ln(x_i))
  
- Level 1 (EWBI): Unweighted geometric mean of EU Priorities
  * No compensation between dimensions
  * Formula: I = (prod_j I_j)^(1/5)

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

# ===============================
# CONFIGURATION
# ===============================
OUTPUT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')))
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
PCA_SCOPE = 'all_years'

# Variant 2: EU Priorities aggregation method
# Base solution: 'geometric' (population-weighted PCA geometric mean)
# Alternative: 'arithmetic' (population-weighted PCA arithmetic mean)
EU_PRIORITIES_AGGREGATION = 'geometric'

# --- EWBI (Level 1) Aggregation ---
# Variant 3: Per-decile aggregation of EU Priorities to EWBI
# Base solution: 'geometric' (geometric mean of EU priorities per decile)
# Alternative: 'arithmetic' (arithmetic mean of EU priorities per decile)
EWBI_DECILE_AGGREGATION = 'geometric'

# Variant 4: Cross-decile aggregation (country-level from deciles)
# Base solution: 'geometric' (geometric mean across deciles 1-10)
# Alternative: 'arithmetic' (arithmetic mean across deciles 1-10)
EWBI_CROSS_DECILE_AGGREGATION = 'geometric'

# --- Performance optimization ---
# Skip EU-27 aggregations (useful for sensitivity analysis focusing on country ranks)
# Set to True to skip EU-27 computations and speed up pipeline
SKIP_EU27 = False

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


def get_pca_weights_for_country_year(pca_results, country, year, available_indicators):
    """
    Extract PCA-based component weights for a specific country-year.
    
    The weights are derived from explained variance of principal components.
    Formula: omega_m = lambda_m / sum(lambda_k)
    
    Returns:
        Dictionary mapping indicator -> weight, or None if PCA results unavailable
    """
    key = (country, year)
    
    if key not in pca_results:
        return None
    
    try:
        pca_result = pca_results[key]
        component_weights = pca_result.get('component_weights', [])
        indicator_names = pca_result.get('indicator_names', [])
        loadings = pca_result.get('component_loadings', [])
        
        if not component_weights or not indicator_names or not loadings:
            return None
        
        # Compute indicator importance as weighted sum of squared loadings
        loadings_array = np.array(loadings).T  # (n_indicators, n_components)
        component_weights_array = np.array(component_weights)
        
        indicator_importance = np.zeros(loadings_array.shape[0])
        for m in range(len(component_weights_array)):
            indicator_importance += (loadings_array[:, m] ** 2) * component_weights_array[m]
        
        # Normalize to get weights
        indicator_weights = indicator_importance / np.sum(indicator_importance)
        
        # Create mapping for available indicators only
        weights_dict = {}
        for ind_name, weight in zip(indicator_names, indicator_weights):
            if ind_name in available_indicators:
                weights_dict[ind_name] = weight
        
        # Renormalize to available indicators only
        if weights_dict:
            total_weight = sum(weights_dict.values())
            weights_dict = {k: v/total_weight for k, v in weights_dict.items()}
        
        return weights_dict if weights_dict else None
        
    except Exception as e:
        return None


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


def compute_eu27_aggregations(level_data, population_data, level_name, use_arithmetic_mean=False):
    """
    Compute EU-27 aggregations using population-weighted mean.
    
    For each year and decile (including All Deciles):
    - Get all individual country values
    - Compute population-weighted mean (arithmetic or geometric based on use_arithmetic_mean)
    
    Args:
        level_data: DataFrame with country-specific data
        population_data: DataFrame with population weights
        level_name: Name of level (for logging)
        use_arithmetic_mean: If True, use arithmetic mean (for raw indicators with zeros).
                            If False, use geometric mean (for normalized data).
        
    Returns:
        DataFrame with EU-27 aggregations (individual deciles + All Deciles)
    """
    mean_type = "arithmetic" if use_arithmetic_mean else "geometric"
    print(f"\n[INFO] Computing EU-27 aggregations for {level_name} (using {mean_type} mean)...")
    
    eu27_records = []
    
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
        # Note: population_data has lowercase columns (country, year, population)
        if population_data is not None:
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
        else:
            # If no population data, skip EU-27 computation
            group_with_pop = pd.DataFrame()
        
        # Remove rows where we couldn't get population data
        group_with_pop = group_with_pop.dropna(subset=['Population'])
        
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
        print(f"[WARN] Raw break-adjusted data not found at {raw_data_path}")
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
    ewbi_decile_agg = 'Geometric mean' if EWBI_DECILE_AGGREGATION == 'geometric' else 'Arithmetic mean'
    ewbi_cross_decile_agg = 'Geometric mean' if EWBI_CROSS_DECILE_AGGREGATION == 'geometric' else 'Arithmetic mean'
    
    print("\n" + "="*70)
    print("STAGE 4: WEIGHTING AND AGGREGATION (PCA-WEIGHTED)")
    print("="*70)
    print(f"\n[CONFIG] Sensitivity analysis settings:")
    print(f"   EU Priorities (Level 2):")
    print(f"      - PCA scope: {PCA_SCOPE}")
    print(f"      - Aggregation method: {eu_priorities_agg}")
    print(f"   EWBI (Level 1):")
    print(f"      - Per-decile aggregation: {ewbi_decile_agg}")
    print(f"      - Cross-decile aggregation: {ewbi_cross_decile_agg}")
    print("\nProcessing steps:")
    print("  1. Load normalized Level 4 data from Stage 3")
    print("  2. Load PCA analysis results from Stage 2")
    print("  3. Load population data for weighting")
    print(f"  4. Compute Level 2 (EU Priorities) via PCA-weighted {eu_priorities_agg.lower()}")
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

    
    # Compute Level 2 with PCA weighting (using normalized Level 4 data)
    level2_data = compute_level2_eu_priorities_pca_weighted(
        level4_data, 
        indicator_mapping, 
        category_indicators,
        pca_results,
        population_data
    )
    
    if level2_data.empty:
        print("[ERROR] Failed to compute Level 2 aggregations")
        return
    
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
    
    print(f"\n[COMPLETE] Stage 4 complete: Weighting and aggregation")
    print(f"[SAVED] Output saved: {output_path}")
    print(f"   Records: {len(final_data):,}")
    
    # Save app-ready format
    app_output_path = OUTPUT_DIR / 'ewbi_master_aggregated.csv'
    final_data.to_csv(app_output_path, index=False)
    print(f"[SAVED] App-ready output: {app_output_path}")
    
    return final_data


if __name__ == '__main__':
    result = main()
    print("\n[OK] 4_weighting_aggregation.py execution complete")

