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

# ===============================
# CONFIGURATION
# ===============================
OUTPUT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# SENSITIVITY ANALYSIS CONFIGURATION
# ===============================
# Variant 1: Structural break detection threshold (absolute percentage change)
# Base solution: 0.30 (30%)
# Alternatives: 0.20 (20%), 0.40 (40%), 0.50 (50%)
BREAK_THRESHOLD = 0.3

# Variant 2: Apply 5-year moving average after structural break adjustment
# Base solution: False (no moving average)
# Alternative: True (apply 5-year centered moving average before average rescaling)
APPLY_MOVING_AVERAGE = False
MOVING_AVERAGE_WINDOW = 5

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


def detect_and_adjust_structural_breaks(df, break_threshold=None, apply_moving_average=None, moving_average_window=None):
    """
    Process time series data: interpolation, break adjustment, optional moving average, and rescaling.
    
    Processing Order:
    1. LINEAR INTERPOLATION: Fill gaps between valid data points
       - Add missing year rows within the data range
       - Interpolate missing values between existing data points (no extrapolation)
       - This ensures a complete series before break detection
    
    2. STRUCTURAL BREAK ADJUSTMENT: Detect and correct methodological discontinuities
       - For each series (indicator/country/decile), calculate year-to-year percentage changes
       - Use configurable threshold (default: 30% absolute) to detect breaks
       - Handle breaks based on their position in the series:
         a) Break at START: Use post-break growth rate (t+1→t+2) to correct first value
         b) Break at MIDDLE: Use pre-break growth rate to find corrected value, then rebase future values
         c) Break at END: Use pre-break growth rate to correct last value
         d) Break at BOTH start and end (only 2 points): Do nothing
    
    3. OPTIONAL MOVING AVERAGE: Apply centered moving average for smoothing
       - If enabled, applies a centered moving average (default: 5-year window)
       - Smooths time series after break adjustment, before rescaling
       - Handles edge cases at series boundaries
    
    4. AVERAGE RESCALING (Mean-Preservation): Restore original series mean
       - Calculate scale factor to restore the original mean after adjustments
       - Apply proportional correction to all values in the series
    
    Example - Break at t=2012 with values [3.0 (2011), 4.0 (2012), 3.8 (2013)]:
      - Pre-break growth (2011→2012 should be): Growth rate from 2010→2011
      - If 2010→2011 was 6.4%, then 2012 should be 3.0 * 1.064 = 3.192
      - For 2013+: Apply actual growth rates on corrected 2012
    
    Args:
        df: DataFrame with columns [primary_index, country, decile, year, value]
        break_threshold: Threshold for structural break detection (default: BREAK_THRESHOLD config)
        apply_moving_average: Whether to apply moving average smoothing (default: APPLY_MOVING_AVERAGE config)
        moving_average_window: Window size for moving average (default: MOVING_AVERAGE_WINDOW config)
    
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
    print("\n[PROCESSING] Interpolation, structural break adjustment, and rescaling...")
    print(f"Input data: {len(df):,} records")
    print("Step 1: Linear interpolation of missing values")
    print(f"Step 2: Structural break detection (threshold: {break_threshold*100:.0f}% absolute year-to-year change)")
    if apply_moving_average:
        print(f"Step 3: {moving_average_window}-year moving average smoothing")
        print("Step 4: Average rescaling (mean-preservation)")
    else:
        print("Step 3: Average rescaling (mean-preservation)")
    
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
        # STEP 2: STRUCTURAL BREAK ADJUSTMENT (after interpolation)
        # =====================================================
        # Re-extract valid data after interpolation
        valid_idx = group['value'].notna()
        valid_data = group[valid_idx].copy()
        
        if len(valid_data) < 3:
            return group
        
        values = valid_data['value'].values.copy()
        years = valid_data['year'].values
        
        # Skip if contains non-positive values (can't use percentage change)
        if np.any(values <= 0):
            return group
        
        # Use configurable absolute threshold (captured from outer scope)
        # break_threshold is passed from the outer function
        
        # Store original mean for later proportional correction
        original_mean = np.mean(values)
        
        # Detect and process ALL breaks using adaptive threshold
        # Loop through series looking for breaks and fixing them as we find them
        i = 1
        adjustments_this_series = 0
        max_iterations = len(years) * 10  # Safety limit to prevent infinite loops
        iterations = 0
        
        while i < len(years) and iterations < max_iterations:
            iterations += 1
            pct_change = np.abs((values[i] - values[i-1]) / values[i-1])
            
            if pct_change > break_threshold:  # Adaptive threshold - found a break
                break_year = years[i]
                is_first_break = (i == 1)
                is_last_break = (i == len(years) - 1)
                
                if is_first_break and is_last_break:
                    # Only 2 datapoints - skip
                    i += 1
                
                elif is_first_break:
                    # Break at START
                    if len(values) >= 3:
                        g_t1_t2 = values[2] / values[1]
                        value_at_start_corrected = values[1] / g_t1_t2
                        group.loc[group['year'] == years[0], 'value'] = value_at_start_corrected
                        
                        # Update values array for next iterations
                        values[0] = value_at_start_corrected
                        adjustments_this_series += 1
                    i += 1
                
                elif is_last_break:
                    # Break at END
                    if i >= 2:
                        g_pre_break = values[i - 1] / values[i - 2]
                        value_at_end_corrected = values[i - 1] * g_pre_break
                        group.loc[group['year'] == break_year, 'value'] = value_at_end_corrected
                        
                        # Update values array
                        values[i] = value_at_end_corrected
                        adjustments_this_series += 1
                    i += 1
                
                else:
                    # Break in MIDDLE
                    if i >= 2:
                        g_pre_break = values[i - 1] / values[i - 2]
                        value_at_break_corrected = values[i - 1] * g_pre_break
                        group.loc[group['year'] == break_year, 'value'] = value_at_break_corrected
                        
                        # Rebase future values using adjustment ratio
                        adjustment_ratio = value_at_break_corrected / values[i]
                        
                        for j in range(i + 1, len(values)):
                            year_j = years[j]
                            group.loc[group['year'] == year_j, 'value'] = values[j] * adjustment_ratio
                            values[j] = values[j] * adjustment_ratio
                        
                        # Update the break point value
                        values[i] = value_at_break_corrected
                        adjustments_this_series += 1
                    i += 1
            else:
                i += 1
        
        if adjustments_this_series > 0:
            adjustments_made += adjustments_this_series
            series_key = (group['primary_index'].iloc[0], group['country'].iloc[0], group['decile'].iloc[0])
            series_with_breaks[series_key] = {
                'breaks_fixed': adjustments_this_series,
                'break_threshold': break_threshold,
                'action': 'processed_all_breaks_in_series'
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
        # STEP 4: AVERAGE RESCALING (Mean-Preservation)
        # =====================================================
        # Get current values after break adjustment (and optional moving average)
        current_values = group['value'].dropna().values
        
        if len(current_values) > 0 and np.all(np.isfinite(current_values)):
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
    print(f"   - Step 2 (Break adjustment): {adjustments_made} breaks fixed (threshold: {break_threshold*100:.0f}%)")
    if apply_moving_average:
        print(f"   - Step 3 (Moving average): {moving_average_window}-year centered window applied")
        print(f"   - Step 4 (Rescaling): Applied to all series with adjustments")
    else:
        print(f"   - Step 3 (Rescaling): Applied to all series with adjustments")
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
    print(f"   - Moving average smoothing: {'Enabled (' + str(MOVING_AVERAGE_WINDOW) + '-year window)' if APPLY_MOVING_AVERAGE else 'Disabled'}")
    print("\nProcessing steps:")
    print("  1. Load raw indicator data from 0_raw_indicator_EU-SILC.py and 0_raw_indicator_LFS.py")
    print("  2. Merge data from EU-SILC and LFS sources")
    print("  3. Linear interpolation of missing values between data points")
    print(f"  4. Detect and adjust structural breaks ({BREAK_THRESHOLD*100:.0f}% year-to-year threshold)")
    if APPLY_MOVING_AVERAGE:
        print(f"  5. Apply {MOVING_AVERAGE_WINDOW}-year moving average smoothing")
        print("  6. Average rescaling (mean-preservation)")
        print("  7. Output processed data (forward fill applied in Stage 3)\n")
    else:
        print("  5. Average rescaling (mean-preservation)")
        print("  6. Output processed data (forward fill applied in Stage 3)\n")
    
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
    raw_data_adjusted = detect_and_adjust_structural_breaks(raw_data_for_imputation)
    
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
    
    # Save output
    output_path = dirs['missing_data_output'] / 'raw_data_break_adjusted.csv'
    raw_data_final.to_csv(output_path, index=False)
    
    print(f"\n[COMPLETE] Stage 1 complete: Break adjustment, interpolation, and rescaling")
    print(f"[SAVED] Output saved: {output_path}")
    print(f"   Records: {len(raw_data_final):,} (break-adjusted, interpolated, rescaled; forward fill in Stage 3)")
    print(f"   Columns: {len(raw_data_final.columns)}")
    
    return raw_data_final


if __name__ == '__main__':
    result = main()
    print("\n[OK] 1_missing_data.py execution complete")
