#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Missing Data Imputation Script

This script handles the first stage of EWBI computation:
1. Merges raw indicator data from multiple sources (EU-SILC, LFS, etc.)
2. Applies linear interpolation to fill gaps between valid data points (missing years)
3. Detects and adjusts for structural breaks in time series data (>30% absolute change)
4. Applies average rescaling (mean-preservation) to maintain original series mean
5. Outputs processed data for subsequent forward fill in Stage 3

Methodology follows EU JRC guidelines for composite indicator construction.

Author: Data Processing Pipeline
Date: 2024
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from variable_mapping import should_filter_indicator
from pipeline_env import env_bool, env_float, env_int, get_output_dir

# ===============================
# CONFIGURATION
# ===============================
OUTPUT_DIR = get_output_dir()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# SENSITIVITY ANALYSIS CONFIGURATION
# ===============================
# Variant 1: Structural break detection threshold (absolute percentage change)
# Base solution: 0.20 (20%)
# Alternatives: 0.10 (10%), 0.20 (20%), 0.30 (30%)
BREAK_THRESHOLD = env_float("EWBI_BREAK_THRESHOLD", 0.2)

# Variant 2: Apply 5-year moving average after structural break adjustment
# Base solution: True (apply 5-year centered moving average)
# Alternative: False (no moving average smoothing)
APPLY_MOVING_AVERAGE = env_bool("EWBI_APPLY_MOVING_AVERAGE", True)
MOVING_AVERAGE_WINDOW = env_int("EWBI_MOVING_AVERAGE_WINDOW", 5)

# Variant 3: Apply mean rescaling (mean-preservation) after adjustments
# Base solution: False (no mean rescaling)
# Alternative: True (restore original series mean after break adjustments)
APPLY_MEAN_RESCALING = env_bool("EWBI_APPLY_MEAN_RESCALING", False)

# ===============================
# HELPER FUNCTIONS
# ===============================

def setup_directories() -> dict:
    """
    Set up directory structure for data loading and saving.
    
    Returns:
        Dictionary containing directory paths
    """
    INPUT_DATA_DIR = OUTPUT_DIR / "0_raw_data_EUROSTAT"
    
    dirs = {
        'output_base': OUTPUT_DIR,
        'input_data': INPUT_DATA_DIR,
        'silc_final': INPUT_DATA_DIR / "0_EU-SILC",
        'lfs_final': INPUT_DATA_DIR / "0_LFS",
        'final_output': INPUT_DATA_DIR / "1_final_df",
        'missing_data_output': OUTPUT_DIR / "1_missing_data_output"
    }
    
    for dir_path in dirs.values():
        if isinstance(dir_path, Path):
            dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def detect_and_adjust_structural_breaks(df, break_threshold=None, apply_moving_average=None, moving_average_window=None, apply_mean_rescaling=None):
    """
    Process time series data: interpolation, break adjustment, optional moving average, and rescaling.
    
    Processing Order:
    1. LINEAR INTERPOLATION: Fill gaps between valid data points
       - Add missing year rows within the data range
       - Interpolate missing values between existing data points (no extrapolation)
       - This ensures a complete series before break detection
    
    2. STRUCTURAL BREAK ADJUSTMENT: Detect and correct methodological discontinuities
       - REVERSED DIRECTION: Process breaks from future to past (last to first)
       - For each series (indicator/country/decile), calculate year-to-year percentage changes
       - Use configurable threshold (default: 30% absolute) to detect breaks
       - Handle breaks using mean of before and after-break growth rates:
         a) Break at START: Use post-break growth rate (t+1→t+2) to correct first value
         b) Break at MIDDLE: Use MEAN of pre-break and post-break growth rates for correction
         c) Break at END: Use pre-break growth rate to correct last value
         d) Break at BOTH start and end (only 2 points): Do nothing
    
    3. OPTIONAL MOVING AVERAGE: Apply centered moving average for smoothing
       - If enabled, applies a centered moving average (default: 5-year window)
       - Smooths time series after break adjustment, before rescaling
       - Handles edge cases at series boundaries
    
    4. OPTIONAL MEAN RESCALING (Mean-Preservation): Restore original series mean
       - If enabled, calculate scale factor to restore the original mean after adjustments
       - Apply proportional correction to all values in the series
    
    Example - Break at t=2012 with values [3.0 (2011), 4.0 (2012), 3.8 (2013)]:
      - Pre-break growth (2010→2011): e.g., 6.4%
      - Post-break growth (2012→2013): (3.8/4.0 - 1) = -5%
      - Mean growth rate: (6.4% + (-5%)) / 2 = 0.7%
      - Corrected 2012: 3.0 * 1.007 = 3.021
    
    Args:
        df: DataFrame with columns [primary_index, country, decile, year, value]
        break_threshold: Threshold for structural break detection (default: BREAK_THRESHOLD config)
        apply_moving_average: Whether to apply moving average smoothing (default: APPLY_MOVING_AVERAGE config)
        moving_average_window: Window size for moving average (default: MOVING_AVERAGE_WINDOW config)
        apply_mean_rescaling: Whether to apply mean rescaling (default: APPLY_MEAN_RESCALING config)
    
    Returns:
        DataFrame with adjusted values (preserves all original columns)
    """
    # Use configuration defaults if not specified
    if break_threshold is None:
        break_threshold = BREAK_THRESHOLD
    if apply_moving_average is None:
        apply_moving_average = APPLY_MOVING_AVERAGE
    if moving_average_window is None:
        moving_average_window = MOVING_AVERAGE_WINDOW
    if apply_mean_rescaling is None:
        apply_mean_rescaling = APPLY_MEAN_RESCALING
    print("\n[PROCESSING] Interpolation, structural break adjustment, and optional rescaling...")
    print(f"Input data: {len(df):,} records")
    print("Step 1: Linear interpolation of missing values")
    print(f"Step 2: Structural break detection (threshold: {break_threshold*100:.0f}% absolute, future-to-past direction)")
    if apply_moving_average:
        print(f"Step 3: {moving_average_window}-year moving average smoothing")
        if apply_mean_rescaling:
            print("Step 4: Mean rescaling (mean-preservation)")
        else:
            print("Step 4: No mean rescaling (preserve adjusted values)")
    else:
        if apply_mean_rescaling:
            print("Step 3: Mean rescaling (mean-preservation)")
        else:
            print("Step 3: No mean rescaling (preserve adjusted values)")
    
    df = df.copy()
    df['year'] = df['year'].astype(int)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    adjustments_made = 0
    interpolations_made = 0
    series_with_breaks = {}
    series_with_interpolations = {}
    
    def process_series(group):
        """Process individual series: interpolation -> break adjustment -> rescaling"""
        nonlocal adjustments_made, interpolations_made, series_with_breaks, series_with_interpolations
        
        group = group.sort_values('year').copy()
        
        # Skip if series has less than 2 years or all NaN
        if len(group) < 2 or group['value'].isna().all():
            return group
        
        # =====================================================
        # STEP 1: LINEAR INTERPOLATION (before break adjustment)
        # =====================================================
        # Get all years in the full range (including potential gaps)
        min_year = group['year'].min()
        max_year = group['year'].max()
        all_years = set(range(min_year, max_year + 1))
        existing_years = set(group['year'].values)
        
        # Add missing year rows if they don't exist
        missing_years = all_years - existing_years
        for year in missing_years:
            new_row = group.iloc[0:1].copy()
            new_row['year'] = year
            new_row['value'] = np.nan
            group = pd.concat([group, new_row], ignore_index=True)
        
        # Sort by year for proper interpolation
        group = group.sort_values('year').reset_index(drop=True)
        
        # Count NaN values BEFORE interpolation (these are the gaps to fill)
        # Only count interior NaN values (between first and last valid data points)
        valid_mask = group['value'].notna()
        if valid_mask.any():
            first_valid_idx = valid_mask.idxmax()
            last_valid_idx = valid_mask[::-1].idxmax()
            interior_nans_before = group.loc[first_valid_idx:last_valid_idx, 'value'].isna().sum()
        else:
            interior_nans_before = 0
        
        # Apply linear interpolation for interior NaN values only
        # (interpolate does not extrapolate by default, which is what we want)
        if interior_nans_before > 0:
            group['value'] = group['value'].interpolate(method='linear', limit_area='inside')
            
            # Count how many were actually filled
            valid_mask_after = group['value'].notna()
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
        # =====================================================
        # STEP 2: NEW VARIATION-BASED STRUCTURAL BREAK ADJUSTMENT
        # =====================================================
        # Re-extract valid data after interpolation
        valid_idx = group['value'].notna()
        valid_data = group[valid_idx].copy()
        
        if len(valid_data) < 2:
            return group
        
        values = valid_data['value'].values.copy()
        years = valid_data['year'].values
        
        # Skip if contains non-positive values (can't use percentage change)
        if np.any(values <= 0):
            return group
        
        # Store original values for mean rescaling
        original_values = values.copy()
        
        # Step 1: Detect breaks (30% variations)
        breaks = []
        for i in range(len(values) - 1):
            pct_change = np.abs((values[i+1] - values[i]) / values[i])
            if pct_change > break_threshold:
                breaks.append(i)  # Break between year[i] and year[i+1]
        
        if len(breaks) == 0:
            # No breaks detected - return original data
            return group
        
        # Step 2: Determine reference year
        # If break at the end, use year before the break as reference
        last_break_index = max(breaks) if breaks else -1
        if last_break_index == len(years) - 2:  # Break between last two years
            ref_year_index = len(years) - 2  # Year before the break
            # Step 2b: Correct the last variation using g(t_last-2, t_last-1)
            if len(years) >= 3 and values[-3] > 0:
                g_correction = values[-3] / values[-2] if values[-2] > 0 else 1.0
                values[-1] = values[-2] * g_correction
        else:
            ref_year_index = len(years) - 1  # Last year
        
        ref_year = years[ref_year_index]
        ref_value = values[ref_year_index]
        
        # Step 3: Build variation database (reference year = 1.0)
        variations = np.ones(len(values))  # Start with all = 1.0
        variations[ref_year_index] = 1.0  # Reference year
        
        # Calculate growth rates, handling breaks
        growth_rates = np.ones(len(values) - 1)  # Growth from year[i] to year[i+1]

        # Identify runs of consecutive breaks, e.g. breaks at i and i+1.
        # For a run [start..end], smooth all breaks in the run using:
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
                # Step 4: Handle break - use mean of pre and post break growth

                run_start, run_end = break_to_run.get(i, (i, i))

                # Growth just before the run
                g_pre = None
                if run_start > 0 and values[run_start - 1] > 0 and values[run_start] > 0:
                    g_pre = values[run_start] / values[run_start - 1]

                # Growth just after the run
                g_post = None
                if (run_end + 2) < len(values) and values[run_end + 1] > 0 and values[run_end + 2] > 0:
                    g_post = values[run_end + 2] / values[run_end + 1]
                
                # Step 5: Edge handling + mean of outside-run growth rates
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
                    growth_rates[i] = 1.0  # No growth if no data available
                    
                print(f"✅ Smoothed break: {group['primary_index'].iloc[0]} in {group['country'].iloc[0]} (year {years[i+1]})")
                print(f"   Using growth rate: {growth_rates[i]:.3f} (mean of pre/post break)")
                adjustments_this_series += 1
                    
            else:
                # Normal growth rate calculation
                if values[i] > 0:
                    growth_rates[i] = values[i+1] / values[i]
                else:
                    growth_rates[i] = 1.0
        
        # Step 3 continued: Build variation database working backwards from reference year
        for i in range(ref_year_index - 1, -1, -1):
            variations[i] = variations[i+1] / growth_rates[i]
            
        # Build variation database working forwards from reference year  
        for i in range(ref_year_index, len(values) - 1):
            variations[i+1] = variations[i] * growth_rates[i]
        
        # Step 6: Reconstruct series using reference_value * variation_ratio
        reconstructed_values = ref_value * variations
        
        # Update the group with reconstructed values
        # Merge back into original group structure
        for i, (_, row) in enumerate(valid_data.iterrows()):
            group.loc[group['year'] == row['year'], 'value'] = reconstructed_values[i]
        
        # Track breaks for reporting
        if adjustments_this_series > 0:
            adjustments_made += adjustments_this_series
            series_key = (group['primary_index'].iloc[0], group['country'].iloc[0], group['decile'].iloc[0])
            series_with_breaks[series_key] = {
                'breaks_fixed': len(breaks),
                'break_threshold': break_threshold,
                'action': 'variation_database_reconstruction'
            }
        
        # =====================================================
        # STEP 3 (OPTIONAL): MOVING AVERAGE SMOOTHING
        # =====================================================
        if apply_moving_average and len(group) >= moving_average_window:
            # Apply centered moving average to smooth the series
            # Using min_periods=1 to handle edge cases at series boundaries
            group['value'] = group['value'].rolling(
                window=moving_average_window, 
                center=True, 
                min_periods=1
            ).mean()
        
        # =====================================================
        # STEP 4: CONDITIONAL MEAN RESCALING (Mean-Preservation)
        # =====================================================
        if apply_mean_rescaling:
            # Get current values after break adjustment (and optional moving average)
            current_values = group['value'].dropna().values
            
            if len(current_values) > 0 and np.all(np.isfinite(current_values)):
                # Use original values from before break adjustment for mean calculation
                original_mean = np.mean(original_values) if 'original_values' in locals() else np.mean(current_values)
                new_mean = np.mean(current_values)
                if new_mean > 0 and original_mean > 0:
                    # Scale factor to restore original mean
                    scale_factor = original_mean / new_mean
                    
                    # Apply proportional correction to all values in the series
                    group['value'] = group['value'] * scale_factor
        
        return group
    
    # Apply processing pipeline by group
    print("\nProcessing each series (break adjustment -> interpolation -> rescaling)...")
    groupby_cols = ['primary_index', 'country', 'decile']
    
    # Count total number of groups for progress bar
    total_groups = df.groupby(groupby_cols).ngroups
    print(f"   Total series to process: {total_groups:,}")
    
    # Manual loop with progress tracking (more reliable than groupby().progress_apply())
    adjusted_df_list = []
    for idx, (group_key, group_data) in enumerate(tqdm(df.groupby(groupby_cols, group_keys=False), 
                                                         desc="Processing series", 
                                                         total=total_groups), 1):
        adjusted_group = process_series(group_data)
        adjusted_df_list.append(adjusted_group)
    
    adjusted_df = pd.concat(adjusted_df_list, ignore_index=True)
    
    print(f"[OK] Processing pipeline completed")
    print(f"   - Step 1 (Interpolation): {interpolations_made} values interpolated")
    print(f"   - Step 2 (Break adjustment): {adjustments_made} breaks fixed (threshold: {break_threshold*100:.0f}%, future-to-past, mean growth rates)")
    if apply_moving_average:
        print(f"   - Step 3 (Moving average): {moving_average_window}-year centered window applied")
        if apply_mean_rescaling:
            print(f"   - Step 4 (Mean rescaling): Applied to series with adjustments")
        else:
            print(f"   - Step 4 (Mean rescaling): Disabled - preserving adjusted values")
    else:
        if apply_mean_rescaling:
            print(f"   - Step 3 (Mean rescaling): Applied to series with adjustments")
        else:
            print(f"   - Step 3 (Mean rescaling): Disabled - preserving adjusted values")
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


# ===============================
# MAIN PROCESSING
# ===============================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("STAGE 1: MISSING DATA IMPUTATION")
    print("="*70)
    print(f"\n[CONFIG] Sensitivity analysis settings:")
    print(f"   - Break detection threshold: {BREAK_THRESHOLD*100:.0f}%")
    print(f"   - Break processing direction: Future-to-past (reversed)")
    print(f"   - Growth calculation: Mean of before/after-break rates")
    print(f"   - Moving average smoothing: {'Enabled (' + str(MOVING_AVERAGE_WINDOW) + '-year window)' if APPLY_MOVING_AVERAGE else 'Disabled'}")
    print(f"   - Mean rescaling: {'Enabled' if APPLY_MEAN_RESCALING else 'Disabled (baseline)'}")
    print("\nProcessing steps:")
    print("  1. Load raw indicator data from 0_raw_indicator_EU-SILC.py and 0_raw_indicator_LFS.py")
    print("  2. Merge data from EU-SILC and LFS sources")
    print("  3. Linear interpolation of missing values between data points")
    print(f"  4. Detect and adjust structural breaks ({BREAK_THRESHOLD*100:.0f}% threshold, future-to-past, mean growth)")
    if APPLY_MOVING_AVERAGE:
        print(f"  5. Apply {MOVING_AVERAGE_WINDOW}-year moving average smoothing")
        if APPLY_MEAN_RESCALING:
            print("  6. Mean rescaling (mean-preservation)")
            print("  7. Output processed data (forward fill applied in Stage 3)\n")
        else:
            print("  6. Output processed data (forward fill applied in Stage 3)\n")
    else:
        if APPLY_MEAN_RESCALING:
            print("  5. Mean rescaling (mean-preservation)")
            print("  6. Output processed data (forward fill applied in Stage 3)\n")
        else:
            print("  5. Output processed data (forward fill applied in Stage 3)\n")
    
    dirs = setup_directories()
    
    # Load processed indicator data from 0_raw_indicator_EU-SILC.py and 0_raw_indicator_LFS.py final outputs
    print("[LOAD] Loading processed indicator data from final summary files...")
    
    # Load EU-SILC final summary data (household + personal)
    silc_final_merged_dir = os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC', '3_final_merged_df')
    silc_household_file = os.path.join(silc_final_merged_dir, 'EU_SILC_household_final_summary.csv')
    silc_personal_file = os.path.join(silc_final_merged_dir, 'EU_SILC_personal_final_summary.csv')
    
    silc_data = pd.DataFrame()
    
    try:
        if os.path.exists(silc_household_file):
            silc_household = pd.read_csv(silc_household_file)
            silc_data = pd.concat([silc_data, silc_household], ignore_index=True)
            print(f"   [OK] EU-SILC Household: {len(silc_household):,} rows")
    except Exception as e:
        print(f"   [WARN] Could not load EU-SILC household data: {e}")
    
    try:
        if os.path.exists(silc_personal_file):
            silc_personal = pd.read_csv(silc_personal_file)
            silc_data = pd.concat([silc_data, silc_personal], ignore_index=True)
            print(f"   [OK] EU-SILC Personal: {len(silc_personal):,} rows")
    except Exception as e:
        print(f"   [WARN] Could not load EU-SILC personal data: {e}")
    
    if silc_data.empty:
        print(f"[WARN] No EU-SILC data loaded from {silc_final_merged_dir}")
    
    # Load LFS final summary data
    lfs_final_dir = os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_LFS')
    lfs_file = os.path.join(lfs_final_dir, 'LFS_household_final_summary.csv')
    
    lfs_data = pd.DataFrame()
    
    try:
        if os.path.exists(lfs_file):
            lfs_data = pd.read_csv(lfs_file)
            print(f"   [OK] LFS: {len(lfs_data):,} rows")
        else:
            print(f"   [WARN] LFS file not found at {lfs_file}")
    except Exception as e:
        print(f"   [WARN] Could not load LFS data: {e}")
    
    # Merge EU-SILC and LFS data
    print("\n[MERGE] Merging EU-SILC and LFS data...")
    
    if silc_data.empty and lfs_data.empty:
        print("[ERROR] No data loaded from either EU-SILC or LFS")
        return
    elif silc_data.empty:
        df = lfs_data.copy()
        print(f"   Using LFS data only: {len(df):,} rows")
    elif lfs_data.empty:
        df = silc_data.copy()
        print(f"   Using EU-SILC data only: {len(df):,} rows")
    else:
        df = pd.concat([silc_data, lfs_data], ignore_index=True)
        print(f"   [OK] Merged EU-SILC + LFS: {len(df):,} rows")
    
    print(f"   Columns: {list(df.columns)}")
    
    # Show what indicators and databases are present
    print(f"\n[DATA] Data Summary from loaded sources:")
    print(f"   Unique indicators: {df['primary_index'].nunique()}")
    print(f"   Unique databases: {df['database'].unique().tolist()}")
    
    # Count indicators by database
    if 'database' in df.columns:
        print(f"\n   Indicators by database:")
        for db in df['database'].unique():
            count = df[df['database'] == db]['primary_index'].nunique()
            print(f"      {db}: {count} indicators")
    
    # ===== FILTERING: Remove excluded indicators and NaN indicators =====
    print(f"\n[FILTER] Filtering excluded indicators and NaN values...")
    print(f"   Before filtering: {len(df):,} records")
    
    # Count what we're removing
    nans_before = df['primary_index'].isna().sum()
    excluded_count = 0
    if 'primary_index' in df.columns:
        excluded_mask = df['primary_index'].apply(should_filter_indicator)
        excluded_count = excluded_mask.sum() - nans_before
    
    # Apply filter: keep only valid indicators (not excluded, not NaN)
    df = df[~df['primary_index'].apply(should_filter_indicator)].copy()
    
    print(f"   After filtering: {len(df):,} records")
    print(f"   Removed {nans_before:,} NaN indicators")
    print(f"   Removed {excluded_count:,} excluded indicators (HH-SILC-1, AC-SILC-1, AN-SILC-1)")
    print(f"   Remaining indicators: {df['primary_index'].nunique()}")
    
    print(f"\n   Sample indicators (first 20):")
    sample_indicators = sorted(df['primary_index'].unique())[:20]
    for ind in sample_indicators:
        print(f"      - {ind}")
    
    # Rename columns to processing format
    df = df.rename(columns={
        'Year': 'year',
        'Country': 'country',
        'Decile': 'decile',
        'Quintile': 'quintile',
        'Primary and raw data': 'primary_index',
        'Value': 'value'
    }).copy()
    
    # Filter for raw data only - remove EU aggregates
    print("\n[FILTER] Filtering raw data for imputation...")
    print(f"Before filtering: {len(df):,} rows")
    
    raw_data_for_imputation = df[
        (df['country'] != 'EU-27') &
        (df['country'].notna()) &
        (df['decile'] != 'All') &
        (df['decile'].astype(str) != 'All') &
        (df['decile'].notna()) &
        (df['primary_index'].notna())
    ].copy()
    
    print(f"After filtering: {len(raw_data_for_imputation):,} rows")
    print(f"  Unique countries: {raw_data_for_imputation['country'].nunique()}")
    print(f"  Unique deciles: {sorted([str(d) for d in raw_data_for_imputation['decile'].unique()])}")
    print(f"  Unique indicators: {raw_data_for_imputation['primary_index'].nunique()}")
    
    # Convert data types
    raw_data_for_imputation['year'] = raw_data_for_imputation['year'].astype(int)
    raw_data_for_imputation['value'] = pd.to_numeric(raw_data_for_imputation['value'], errors='coerce')
    
    # Step 1: Apply structural break adjustment
    raw_data_adjusted = detect_and_adjust_structural_breaks(
        raw_data_for_imputation,
        break_threshold=BREAK_THRESHOLD,
        apply_moving_average=APPLY_MOVING_AVERAGE,
        moving_average_window=MOVING_AVERAGE_WINDOW,
        apply_mean_rescaling=APPLY_MEAN_RESCALING
    )
    
    # Output: Raw data with structural breaks (no forward fill - will be done in Stage 3)
    raw_data_final = raw_data_adjusted.rename(columns={
        'year': 'Year',
        'country': 'Country',
        'decile': 'Decile',
        'primary_index': 'Primary and raw data',
        'value': 'Value'
    }).copy()
    
    # Convert decile back to original format (handle float values from merge)
    raw_data_final['Decile'] = pd.to_numeric(raw_data_final['Decile'], errors='coerce').astype('Int64')
    
    # Drop Quintile column if it exists (not used in this dataset - only deciles)
    if 'quintile' in raw_data_final.columns:
        raw_data_final = raw_data_final.drop(columns=['quintile', 'Quintile'], errors='ignore')
    
    # Add metadata
    raw_data_final['Level'] = 3
    raw_data_final['EU priority'] = pd.NA
    raw_data_final['Secondary'] = pd.NA
    raw_data_final['Type'] = 'Statistical computation'
    raw_data_final['Aggregation'] = pd.NA
    
    # Build unified column order (without Quintile)
    unified_columns = ['Year', 'Country', 'Decile', 'Level', 'EU priority',
                      'Secondary', 'Primary and raw data', 'Type', 'Aggregation', 'Value']
    
    # Add datasource if present
    if 'datasource' in raw_data_final.columns:
        unified_columns.append('datasource')
    
    # Select only columns that exist in the dataframe
    existing_columns = [col for col in unified_columns if col in raw_data_final.columns]
    raw_data_final = raw_data_final[existing_columns]
    
    # ===== FINAL CLEANUP: Remove records with NaN in 'Primary and raw data' =====
    nans_before_save = raw_data_final['Primary and raw data'].isna().sum()
    if nans_before_save > 0:
        raw_data_final = raw_data_final[raw_data_final['Primary and raw data'].notna()]
        print(f"\n[CLEANUP] Removed {nans_before_save:,} records with NaN in 'Primary and raw data'")
        print(f"   Records after cleanup: {len(raw_data_final):,}")
    
    # ===== MATHEMATICAL VALIDATION: Check for impossible percentage values =====
    print(f"\n[VALIDATION] Final mathematical validation...")
    
    # Count extreme values for reporting only
    extreme_values = raw_data_final[raw_data_final['Value'] > 200]  # Values >200% are suspicious for social indicators
    if len(extreme_values) > 0:
        print(f"⚠️  WARNING: Found {len(extreme_values)} extreme values (>200%) - will be handled in later stages")
        print("   Top 5 extreme values:")
        top_extreme = extreme_values.nlargest(5, 'Value')[['Year', 'Country', 'Primary and raw data', 'Value']]
        for _, row in top_extreme.iterrows():
            print(f"      {row['Primary and raw data']} {row['Country']} {row['Year']}: {row['Value']:.1f}%")
        
        print(f"   ✅ Validation complete - extreme values will be handled downstream")
    else:
        print(f"   ✅ No extreme values detected - all values appear reasonable")
    
    # Save output
    output_path = dirs['missing_data_output'] / 'raw_data_break_adjusted.csv'
    raw_data_final.to_csv(output_path, index=False)
    
    print(f"\n[COMPLETE] Stage 1 complete: Break adjustment (future-to-past, mean growth), interpolation, and {'rescaling' if APPLY_MEAN_RESCALING else 'value preservation'}")
    print(f"[SAVED] Output saved: {output_path}")
    processing_desc = "break-adjusted (future-to-past), interpolated"
    if APPLY_MEAN_RESCALING:
        processing_desc += ", rescaled"
    else:
        processing_desc += ", values preserved"
    print(f"   Records: {len(raw_data_final):,} ({processing_desc}; forward fill in Stage 3)")
    print(f"   Columns: {len(raw_data_final.columns)}")
    
    return raw_data_final


if __name__ == '__main__':
    result = main()
    print("\n[OK] 1_missing_data.py execution complete")
