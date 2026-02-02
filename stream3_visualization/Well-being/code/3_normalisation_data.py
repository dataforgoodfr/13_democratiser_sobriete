#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalization Script (Winsorization + Percentile Scaling)

This script performs the third stage of EWBI computation:
1. Loads break-adjusted raw data from Stage 1
2. Applies forward fill to complete missing years
3. Applies Winsorization to remove extreme outliers (1st-99th percentiles)
4. Applies percentile scaling (empirical CDF transformation)
5. Rescales to [0.1, 1] range for geometric mean compatibility
6. Outputs normalized Level 4 data for aggregation

Data Hierarchy:
- Level 1: EWBI Overall
- Level 2: EU Priorities
- Level 3: Primary/Raw Indicators (break-adjusted, forward-filled)
- Level 4: Normalized Indicators (input to aggregation)

Methodology:
- Winsorization clips extreme values: x_win = clip(x, P1, P99)
- Percentile scaling transforms via empirical CDF: P(x) = rank(x) / n
- Rescaling inverts indicator and rescales: x' = 0.1 + 0.9 * (1 - P(x))
- Multi-year pooled normalization ensures temporal stability

Author: Data Processing Pipeline
Date: 2024
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from tqdm import tqdm

# ===============================
# CONFIGURATION
# ===============================
OUTPUT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# SENSITIVITY ANALYSIS CONFIGURATION
# ===============================
# Variant 1: Normalization method
# Base solution: 'percentile' (Winsorization + Percentile Scaling)
# Alternative: 'zscore' (Z-score standardization)
NORMALIZATION_METHOD = 'percentile'

# Variant 2: Rescaling range
# Base solution: [0.1, 1] to ensure positive values for geometric mean
# Alternatives: [0, 1], [0.01, 1], [0.2, 1]
RESCALE_MIN = 0.1
RESCALE_MAX = 1.0  # Maximum value after rescaling

# Variant 3: Normalization approach (temporal pooling)
# Base solution: 'multi_year' (pooled statistics across all years)
# Alternative: 'per_year' (normalize within each year separately)
NORMALIZATION_APPROACH = 'multi_year'

MISSING_DATA_OUTPUT = OUTPUT_DIR / "1_missing_data_output"

# ===============================
# NORMALIZATION FUNCTIONS
# ===============================

def winsorize_and_percentile_scale(values, lower_percentile=1, upper_percentile=99, 
                                     method=None, rescale_min=None, rescale_max=None):
    """
    Apply normalization to indicator values using configurable method and scale.
    
    Methods:
    1. 'percentile' (default): Winsorization + Percentile Scaling
       - Winsorize at 1st and 99th percentiles to remove extreme outliers
       - Apply empirical CDF (percentile scaling) to get values in [0,1]
       - Rescale to [rescale_min, rescale_max] with indicator inversion
    
    2. 'zscore': Z-score standardization
       - Standardize using z = (x - mean) / std
       - Apply sigmoid transformation for bounded output
       - Rescale to [rescale_min, rescale_max] with indicator inversion
    
    Formula (percentile method):
    - Winsorized: x_win = clip(x, P1, P99)
    - Percentile: P(x) = rank(x) / (n-1)
    - Rescaled: x' = rescale_min + (rescale_max - rescale_min) * (1 - P(x))
    
    Formula (zscore method):
    - Z-score: z = (x - mean) / std
    - Sigmoid: s = 1 / (1 + exp(-z))
    - Rescaled: x' = rescale_min + (rescale_max - rescale_min) * (1 - s)
    
    Args:
        values: Array of indicator values
        lower_percentile: Lower winsorization threshold (default: 1st percentile)
        upper_percentile: Upper winsorization threshold (default: 99th percentile)
        method: Normalization method ('percentile' or 'zscore'). Default: NORMALIZATION_METHOD config
        rescale_min: Minimum value for rescaling. Default: RESCALE_MIN config
        rescale_max: Maximum value for rescaling. Default: RESCALE_MAX config
    
    Returns:
        Transformed values in range [rescale_min, rescale_max] with indicators inverted
    """
    # Use configuration defaults if not specified
    if method is None:
        method = NORMALIZATION_METHOD
    if rescale_min is None:
        rescale_min = RESCALE_MIN
    if rescale_max is None:
        rescale_max = RESCALE_MAX
    
    if len(values) == 0 or np.all(np.isnan(values)):
        return values
    
    # Remove NaN values for processing
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    
    if len(finite_values) < 2:
        # Not enough data points for meaningful transformation
        return np.where(finite_mask, (rescale_min + rescale_max) / 2, np.nan)
    
    if method == 'zscore':
        # Z-score standardization method
        mean_val = np.mean(finite_values)
        std_val = np.std(finite_values)
        
        if std_val == 0 or std_val < 1e-10:
            # No variance - return middle value
            result = np.full_like(values, np.nan, dtype=float)
            result[finite_mask] = (rescale_min + rescale_max) / 2
            return result
        
        # Calculate z-scores
        z_scores = (finite_values - mean_val) / std_val
        
        # Apply sigmoid transformation to bound values in (0, 1)
        # sigmoid(z) = 1 / (1 + exp(-z))
        sigmoid_scores = 1 / (1 + np.exp(-z_scores))
        
        # Rescale to [rescale_min, rescale_max] with indicator inversion
        # Inversion: higher raw value -> lower final score (since these are deprivation indicators)
        scale_range = rescale_max - rescale_min
        final_scores = rescale_min + scale_range * (1 - sigmoid_scores)
        
    else:  # 'percentile' method (default)
        # Step 1: Winsorization at 1st and 99th percentiles
        lower_bound = np.percentile(finite_values, lower_percentile)
        upper_bound = np.percentile(finite_values, upper_percentile)
        
        winsorized_values = np.clip(finite_values, lower_bound, upper_bound)
        
        # Step 2: Apply empirical CDF (percentile scaling)
        # This gives us the empirical cumulative distribution (0 to 1)
        ranks = rankdata(winsorized_values, method='average')
        percentile_scores = (ranks - 1) / (len(ranks) - 1)  # Scale to [0, 1]
        
        # Step 3: Rescale to [rescale_min, rescale_max] with indicator inversion
        # Formula: x' = rescale_min + (rescale_max - rescale_min) * (1 - percentile(x))
        scale_range = rescale_max - rescale_min
        final_scores = rescale_min + scale_range * (1 - percentile_scores)
    
    # Reconstruct full array with NaN preservation
    result = np.full_like(values, np.nan, dtype=float)
    result[finite_mask] = final_scores
    
    return result


def load_imputed_data(missing_data_dir):
    """
    Load the break-adjusted and forward-filled data from Stage 1.
    
    Args:
        missing_data_dir: Path to missing data output directory
        
    Returns:
        DataFrame with raw data from Stage 1
    """
    print("[LOAD] Loading break-adjusted raw data from Stage 1...")
    
    input_path = missing_data_dir / 'raw_data_break_adjusted.csv'
    
    if not input_path.exists():
        print(f"[ERROR] Break-adjusted data not found at {input_path}")
        print("Please run 1_missing_data.py first")
        return None
    
    df = pd.read_csv(input_path)
    print(f"[OK] Loaded {len(df):,} records from Stage 1")
    
    # Convert to processing format
    df = df.rename(columns={
        'Year': 'year',
        'Country': 'country',
        'Decile': 'decile',
        'Quintile': 'quintile',
        'Primary and raw data': 'primary_index',
        'Value': 'value'
    }).copy()
    
    return df


def forward_fill_missing_data(df):
    """
    Forward fill missing data according to EU JRC methodology:
    - For each indicator/country/decile combination, fill missing years with forward fill
    - If no previous data exists, use backward fill
    - Ensure complete data coverage until the maximum year
    
    Args:
        df: DataFrame with structural breaks already adjusted
        
    Returns:
        DataFrame with forward-filled values
    """
    print("\n[FILL] Applying forward fill to complete missing data...")
    print(f"Input data: {len(df):,} records")
    
    # Convert year to int to avoid float comparison issues
    df = df.copy()
    df['Year'] = df['Year'].astype(int)
    
    # Get all unique years as a complete sequence
    all_years = sorted(df['Year'].unique())
    max_year = max(all_years)
    min_year = min(all_years)
    complete_year_range = list(range(min_year, max_year + 1))
    
    print(f"Year range: {min_year} to {max_year} ({len(complete_year_range)} years)")
    print(f"Creating complete timeline with forward fill...")
    
    # Use pandas groupby and apply for efficiency
    def fill_group(group):
        # Get the first year where this indicator has data
        first_data_year = group['Year'].min()
        
        # Create a complete year index for this group
        complete_index = pd.DataFrame({'Year': complete_year_range})
        
        # Merge with existing data
        merged = complete_index.merge(group, on='Year', how='left')
        
        # Forward fill ONLY from the first data point onward (JRC methodology)
        # Keep NaN values for years before first data point
        merged.loc[merged['Year'] >= first_data_year] = merged.loc[merged['Year'] >= first_data_year].ffill()
        
        # Remove rows that are still NaN AFTER the last available year
        # But preserve NaN rows BEFORE the first available year
        last_data_year = group['Year'].max()
        mask = (merged['Year'] < first_data_year) | (merged['Year'] <= last_data_year) | merged['Value'].notna()
        
        return merged[mask]
    
    # Group by the key columns and apply forward fill
    print("Applying forward fill by group...")
    groupby_cols = ['Primary and raw data', 'Country', 'Decile']
    completed_df = df.groupby(groupby_cols, group_keys=False).apply(fill_group).reset_index(drop=True)
    
    print(f"[OK] Forward fill completed: {len(completed_df):,} records")
    print(f"   Original records: {len(df):,}")
    print(f"   Added records: {len(completed_df) - len(df):,}")
    
    return completed_df


def normalize_raw_to_level4(df, normalization_approach=None, method=None, rescale_min=None, rescale_max=None):
    """
    Normalize raw data (Level 3) to create normalized indicators (Level 4).
    
    Creates a 1:1 relationship between raw and normalized indicator records.
    Each raw record gets a corresponding normalized record with normalized value.
    
    Args:
        df: DataFrame with raw data from Stage 1 (forward-filled)
        normalization_approach: 'multi_year' (pooled) or 'per_year' (per-year). Default: NORMALIZATION_APPROACH config
        method: Normalization method ('percentile' or 'zscore'). Default: NORMALIZATION_METHOD config
        rescale_min: Minimum rescaling value. Default: RESCALE_MIN config
        rescale_max: Maximum rescaling value. Default: RESCALE_MAX config
        
    Returns:
        DataFrame with normalized Level 4 data
    """
    # Use configuration defaults if not specified
    if normalization_approach is None:
        normalization_approach = NORMALIZATION_APPROACH
    if method is None:
        method = NORMALIZATION_METHOD
    if rescale_min is None:
        rescale_min = RESCALE_MIN
    if rescale_max is None:
        rescale_max = RESCALE_MAX
    
    method_name = 'Z-score standardization' if method == 'zscore' else 'Winsorization (1st-99th) + Percentile Scaling'
    
    print(f"\n[NORMALIZE] Creating Level 4 normalized data from raw indicators...")
    print(f"   Approach: {normalization_approach}")
    print(f"   Method: {method_name}")
    print(f"   Rescaling range: [{rescale_min}, {rescale_max}]")
    print(f"   All values will be strictly positive for geometric mean aggregation")
    
    normalized_records = []
    
    if normalization_approach == 'multi_year':
        # Method 1: Multi-year pooled normalization (most stable across years)
        print("   Using pooled statistics across all years for temporal stability...\n")
        
        # Group by indicator only (cross-decile, cross-year normalization)
        for indicator in tqdm(df['Primary and raw data'].unique(), desc="Normalizing indicators"):
            group_data = df[df['Primary and raw data'] == indicator].copy()
            
            # Extract values for normalization across ALL years and deciles
            values = group_data['Value'].values
            
            # Check if all values are NaN
            if np.all(np.isnan(values)):
                print(f"   [SKIP] {indicator}: No valid data")
                continue
            
            # Count valid (non-NaN) values
            valid_values = values[~np.isnan(values)]
            if len(valid_values) < 2:
                print(f"   [SKIP] {indicator}: Insufficient data ({len(valid_values)} values)")
                continue
            
            # Apply normalization with configurable method and scale
            normalized_values = winsorize_and_percentile_scale(
                values, method=method, rescale_min=rescale_min, rescale_max=rescale_max
            )
            
            # Create normalized records
            for i, (_, row) in enumerate(group_data.iterrows()):
                normalized_row = row.copy()
                normalized_row['Value'] = normalized_values[i]
                normalized_records.append(normalized_row)
    
    elif normalization_approach == 'per_year':
        # Method 2: Per-year normalization
        print("   Using per-year normalization approach...\n")
        
        # Group by year and indicator
        for (year, indicator), group_data in tqdm(
            df.groupby(['Year', 'Primary and raw data']),
            desc="Normalizing by year"
        ):
            values = group_data['Value'].values
            
            if np.all(np.isnan(values)):
                continue
            
            valid_values = values[~np.isnan(values)]
            if len(valid_values) < 2:
                continue
            
            # Apply normalization with configurable method and scale per year
            normalized_values = winsorize_and_percentile_scale(
                values, method=method, rescale_min=rescale_min, rescale_max=rescale_max
            )
            
            # Create normalized records
            for i, (_, row) in enumerate(group_data.iterrows()):
                normalized_row = row.copy()
                normalized_row['Value'] = normalized_values[i]
                normalized_records.append(normalized_row)
    
    # Convert to DataFrame
    if normalized_records:
        level4_data = pd.DataFrame(normalized_records)
        print(f"\n[OK] Created Level 4 normalized data: {len(level4_data):,} records")
        print(f"   1:1 relationship with raw data: {len(level4_data) == len(df)}")
        print(f"   Value range: [{level4_data['Value'].min():.4f}, {level4_data['Value'].max():.4f}]")
        print(f"   Expected range: [{rescale_min}, {rescale_max}]")
    else:
        print("[ERROR] No normalized data created")
        level4_data = pd.DataFrame()
    
    return level4_data


def convert_to_unified_structure(level4_data):
    """
    Convert normalized data to unified output structure with Level 4 metadata.
    
    Args:
        level4_data: DataFrame with normalized data in processing format
        
    Returns:
        DataFrame in unified structure
    """
    print("\n[CONVERT] Converting to unified Level 4 structure...")
    
    level4_unified = level4_data.copy()
    level4_unified = level4_unified.rename(columns={
        'year': 'Year',
        'country': 'Country',
        'decile': 'Decile',
        'primary_index': 'Primary and raw data',
        'value': 'Value'
    })
    
    # Drop Quintile column if it exists (not used in this dataset - only deciles)
    if 'Quintile' in level4_unified.columns:
        level4_unified = level4_unified.drop(columns=['Quintile'], errors='ignore')
    
    # Add Level 4 metadata (normalized indicators)
    level4_unified['Level'] = 4
    level4_unified['Type'] = 'Normalized indicator'
    level4_unified['Aggregation'] = 'Winsorization + Percentile Scaling'
    level4_unified['EU priority'] = pd.NA
    level4_unified['Secondary'] = pd.NA
    
    # Convert Decile to numeric
    level4_unified['Decile'] = pd.to_numeric(level4_unified['Decile'], errors='coerce')
    
    # Add datasource column
    level4_unified['datasource'] = pd.NA
    
    # Ensure column order (without Quintile)
    unified_columns = ['Year', 'Country', 'Decile', 'Level', 'EU priority',
                      'Secondary', 'Primary and raw data', 'Type', 'Aggregation', 'Value', 'datasource']
    level4_unified = level4_unified[unified_columns]
    
    print(f"[OK] Converted {len(level4_unified):,} records to unified Level 4 structure")
    
    return level4_unified


# ===============================
# MAIN PROCESSING
# ===============================

def main():
    """Main execution function"""
    
    method_name = 'Z-score standardization' if NORMALIZATION_METHOD == 'zscore' else 'Winsorization + Percentile Scaling'
    
    print("\n" + "="*70)
    print("STAGE 3: NORMALIZATION OF DATA")
    print("="*70)
    print(f"\n[CONFIG] Sensitivity analysis settings:")
    print(f"   - Normalization method: {method_name}")
    print(f"   - Rescaling range: [{RESCALE_MIN}, {RESCALE_MAX}]")
    print(f"   - Temporal approach: {NORMALIZATION_APPROACH}")
    print("\nProcessing steps:")
    print("  1. Load break-adjusted raw data (Level 3) from Stage 1")
    print("  2. Apply forward fill to complete missing data")
    if NORMALIZATION_METHOD == 'zscore':
        print("  3. Apply Z-score standardization")
        print("  4. Apply sigmoid transformation for bounded output")
    else:
        print("  3. Apply Winsorization (1st-99th percentiles)")
        print("  4. Apply Percentile Scaling (empirical CDF)")
    print(f"  5. Rescale to [{RESCALE_MIN}, {RESCALE_MAX}] range")
    print("  6. Create Level 4 normalized indicators from Level 3 raw data")
    print("  7. Save normalized Level 4 data\n")
    
    # Load imputed data
    df = load_imputed_data(MISSING_DATA_OUTPUT)
    if df is None:
        return
    
    # Apply forward fill (moved from Stage 1)
    print("\n[PREPROCESS] Applying forward fill to Stage 1 break-adjusted data...")
    df_columns_backup = df.columns.tolist()
    # Stage 1 outputs lowercase, convert to capital for processing
    df_renamed = df.rename(columns={
        'year': 'Year',
        'country': 'Country',
        'decile': 'Decile',
        'primary_index': 'Primary and raw data',
        'value': 'Value'
    }).copy()
    # forward_fill_missing_data expects capital letters
    df_filled = forward_fill_missing_data(df_renamed)
    df = df_filled  # Update df to use filled data with capital letters
    
    # Filter for non-"All" values
    print("\n[FILTER] Filtering for individual-level data...")
    print(f"Before filtering: {len(df):,} rows")
    
    raw_data = df[
        (df['Country'] != 'EU-27') &
        (df['Decile'] != 'All') &
        (df['Decile'].notna())
    ].copy()
    
    print(f"After filtering: {len(raw_data):,} rows")
    
    # Save the forward-filled raw data for Stage 4 Level 3 (raw indicator display)
    raw_forward_filled_path = MISSING_DATA_OUTPUT / 'raw_data_forward_filled.csv'
    raw_data.to_csv(raw_forward_filled_path, index=False)
    print(f"[SAVED] Forward-filled raw data saved: {raw_forward_filled_path}")
    
    # Convert data types
    raw_data['Year'] = raw_data['Year'].astype(int)
    raw_data['Value'] = pd.to_numeric(raw_data['Value'], errors='coerce')
    
    # Normalize raw data (Level 3) to Level 4 normalized
    level4_data = normalize_raw_to_level4(raw_data, normalization_approach=NORMALIZATION_APPROACH)
    
    if level4_data.empty:
        print("[ERROR] Normalization failed - no data produced")
        return
    
    # Convert to unified structure
    level4_unified = convert_to_unified_structure(level4_data)
    
    # Save output
    output_subdir = OUTPUT_DIR / "3_normalisation_data_output"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_subdir / 'level4_normalised_indicators.csv'
    level4_unified.to_csv(output_path, index=False)
    
    print(f"\n[COMPLETE] Stage 3 complete: Data normalization")
    print(f"[SAVED] Output saved: {output_path}")
    print(f"   Records: {len(level4_unified):,}")
    print(f"   Columns: {len(level4_unified.columns)}")
    
    return level4_unified


if __name__ == '__main__':
    result = main()
    print("\n[OK] 3_normalisation_data.py execution complete")
