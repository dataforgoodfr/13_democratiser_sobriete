"""
Household Budget Survey (HBS) Data Processing Script

This script processes HBS data from multiple years and creates a final summary dataset
for use in the European Well-Being Index (EWBI).

Author: Data for Good - Well-being Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm


def setup_directories():
    """Set up directories for data processing using portable paths."""
    # Get the absolute path to the project output directory
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    
    # External data directory (modify this path according to your external data location)
    EXTERNAL_DATA_DIR = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI"
    
    dirs = {
        'external_data': EXTERNAL_DATA_DIR,
        'output_base': OUTPUT_DIR,
        'hbs_output': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_HBS'),
        'merged_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_HBS', '0_merged'),
        'decile_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_HBS', '1_income_decile'),
        'final_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_HBS')
    }
    
    # Create output directories if they don't exist
    for key, dir_path in dirs.items():
        if key not in ['external_data', 'output_base']:
            os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def stack_excels(folder, pattern):
    """
    Load and stack Excel files matching a pattern.
    
    Args:
        folder (str): Folder path to search for files
        pattern (str): File pattern to match
        
    Returns:
        pd.DataFrame: Concatenated dataframe from all matching files
    """
    files = glob.glob(os.path.join(folder, pattern))
    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f)
            df['source_file'] = os.path.basename(f)  # optional: track origin
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def combine_hbs_data(dirs):
    """
    Combine HBS data from multiple years and file types.
    
    Args:
        dirs (dict): Dictionary containing directory paths
        
    Returns:
        tuple: (household_all, household_members_all) dataframes
    """
    # Define paths for each year
    paths = {
        '2010': os.path.join(dirs['external_data'], r"0_data/HBS/HBS2010/HBS2010"),
        '2015': os.path.join(dirs['external_data'], r"0_data/HBS/HBS2015/HBS2015"),
        '2020': os.path.join(dirs['external_data'], r"0_data/HBS/HBS2020/HBS2020"),
    }

    # Define file patterns for each year and type
    patterns = {
        '2010': {'hh': "*_HBS_hh.xlsx", 'hm': "*_HBS_hm.xlsx"},
        '2015': {'hh': "*_MFR_hh.xlsx", 'hm': "*_MFR_hm.xlsx"},
        '2020': {'hh': "HBS_HH_*.xlsx",  'hm': "HBS_HM_*.xlsx"},
    }

    # Stack per year and type
    stacked = {'hh': [], 'hm': []}
    for year, folder in paths.items():
        for kind in ['hh', 'hm']:
            df = stack_excels(folder, patterns[year][kind])
            if not df.empty:
                df['year'] = year
                stacked[kind].append(df)

    # Final stacks
    hh_all = pd.concat(stacked['hh'], ignore_index=True) if stacked['hh'] else pd.DataFrame()
    hm_all = pd.concat(stacked['hm'], ignore_index=True) if stacked['hm'] else pd.DataFrame()

    # Save combined data
    hh_all.to_csv(os.path.join(dirs['merged_dir'], "HBS_combined_HH.csv"), index=False)
    hm_all.to_csv(os.path.join(dirs['merged_dir'], "HBS_combined_HM.csv"), index=False)

    print(f"Combined household data shape: {hh_all.shape}")
    print(f"Combined household members data shape: {hm_all.shape}")

    return hh_all, hm_all


def weighted_quantile(values, weights, quantiles):
    """
    Computes weighted quantiles. Values and weights must be 1D numpy arrays.
    
    Parameters:
        values (np.array): The data values.
        weights (np.array): The weights for each value.
        quantiles (np.array): The quantiles to compute (0 to 1).
        
    Returns:
        np.array: Weighted quantiles.
    """
    sorter = np.argsort(values)
    values_sorted = values[sorter]  
    weights_sorted = weights[sorter]

    cumsum_weights = np.cumsum(weights_sorted)
    total_weight = cumsum_weights[-1]
    normalized_weights = cumsum_weights / total_weight

    return np.interp(quantiles, normalized_weights, values_sorted)


def weighted_median(values, weights):
    """
    Calculate weighted median.
    
    Args:
        values: Array of values
        weights: Array of weights
        
    Returns:
        float: Weighted median value
    """
    values = pd.to_numeric(values, errors='coerce')
    weights = pd.to_numeric(weights, errors='coerce')
    
    mask = ~np.isnan(values) & ~np.isnan(weights)
    values = values[mask]
    weights = weights[mask]
    
    if len(values) == 0:
        return np.nan

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cumulative_weight = np.cumsum(weights)
    cutoff = cumulative_weight[-1] / 2.0

    return values[np.searchsorted(cumulative_weight, cutoff)]


def weighted_std(values, weights):
    """
    Calculate weighted standard deviation.
    
    Args:
        values: Array of values
        weights: Array of weights
        
    Returns:
        float: Weighted standard deviation
    """
    values = pd.to_numeric(values, errors='coerce')
    weights = pd.to_numeric(weights, errors='coerce')

    mask = ~np.isnan(values) & ~np.isnan(weights)
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0 or weights.sum() == 0:
        return np.nan

    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance)


def prepare_household_data(dirs):
    """
    Prepare household data with equivalized income and consumption indicators.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Prepared household dataset
    """
    print("Preparing household data...")
    
    cols_needed_household = [
        "HA04", "COUNTRY", "YEAR", "HA10", "EUR_HH095", "HB061", "HB075A",
        "EUR_HH099", "EUR_HH032", "EUR_HE01", "EUR_HE041", "EUR_HE042", "EUR_HE043", 
        "EUR_HE045", "EUR_HE06", "EUR_HE10", "EUR_HE09", "EUR_HJ08", "EUR_HJ90", "EUR_HE07"
    ]

    combined_hh_path = os.path.join(dirs['merged_dir'], "HBS_combined_HH.csv")
    
    # Load the data
    household_df = pd.read_csv(
        combined_hh_path,
        usecols=[col for col in cols_needed_household if col in pd.read_csv(
            combined_hh_path, nrows=1
        ).columns]
    )

    # Calculate equivalized disposable income
    household_df["equi_disp_inc"] = household_df["EUR_HH095"] / household_df["HB061"]

    # Merge housing costs (rent + imputed rent + maintenance)
    if all(col in household_df.columns for col in ["EUR_HE041", "EUR_HE042", "EUR_HE043"]):
        household_df["EUR_HE041"] = (
            household_df["EUR_HE041"].fillna(0) + 
            household_df["EUR_HE042"].fillna(0) + 
            household_df["EUR_HE043"].fillna(0)
        )

    # Save intermediate result
    household_df.to_csv(os.path.join(dirs['decile_dir'], "HBS_household_equi_inc.csv"), index=False)

    return household_df


def calculate_income_deciles(dirs):
    """
    Calculate income deciles for HBS data.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Dataset with income deciles
    """
    print("Calculating income deciles...")
    
    # Load the data
    household_df = pd.read_csv(os.path.join(dirs['decile_dir'], "HBS_household_equi_inc.csv"))
    
    # Convert weight column to numeric
    household_df["HA10"] = pd.to_numeric(household_df["HA10"], errors='coerce')

    # Define the deciles you want to compute
    deciles = np.arange(0.1, 1.0, 0.1)  # 0.1 to 0.9

    # Group the DataFrame by year and country
    def compute_deciles(group):
        values = group['equi_disp_inc'].to_numpy()
        weights = group['HA10'].to_numpy()
        
        # Remove NaN values
        mask = ~np.isnan(values) & ~np.isnan(weights)
        values = values[mask]
        weights = weights[mask]
        
        if len(values) == 0:
            return pd.Series([np.nan] * len(deciles), index=[f'decile_{int(d*10)}' for d in deciles])
        
        decile_values = weighted_quantile(values, weights, deciles)
        return pd.Series(decile_values, index=[f'decile_{int(d*10)}' for d in deciles])

    # Apply the function
    decile_df = household_df.groupby(['YEAR', 'COUNTRY']).apply(compute_deciles).reset_index()
    decile_df.rename(columns={'YEAR': 'Year', 'COUNTRY': 'Country'}, inplace=True)
    decile_df.to_csv(os.path.join(dirs['decile_dir'], "HBS_household_decile.csv"), index=False)

    # Load and merge with original data
    equival_inc_df = pd.read_csv(os.path.join(dirs['decile_dir'], "HBS_household_equi_inc.csv"))
    
    # Merge decile thresholds into the original dataframe
    equiv_with_deciles = equival_inc_df.merge(
        decile_df, left_on=['YEAR', 'COUNTRY'], right_on=['Year', 'Country'], how='left'
    )

    # Function to assign deciles safely, handling NaNs
    def assign_decile(row):
        income = row['equi_disp_inc']
        
        if pd.isna(income):
            return np.nan
        
        try:
            thresholds = [row[f'decile_{i}'] for i in range(1, 10)]
        except KeyError:
            return np.nan

        if any(pd.isna(thresholds)):
            return np.nan
        
        for i, threshold in enumerate(thresholds, start=1):
            if income <= threshold:
                return i
        return 10  # Income above all thresholds

    # Apply decile assignment
    tqdm.pandas(desc="Assigning income deciles")
    equiv_with_deciles['decile'] = equiv_with_deciles.progress_apply(assign_decile, axis=1)
    equiv_with_deciles.to_csv(os.path.join(dirs['decile_dir'], "HBS_household_data_with_decile.csv"), index=False)

    return equiv_with_deciles


def calculate_consumption_indicators(dirs):
    """
    Calculate consumption indicators with equivalization and national medians.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Dataset with consumption indicators
    """
    print("Calculating consumption indicators...")
    
    # Load data with deciles
    cols_needed = [
        'HA04', 'COUNTRY', 'YEAR', 'EUR_HH032', 'EUR_HH095', 'EUR_HH099',
        'HB061', "HB075A", 'HA10', 'equi_disp_inc', 'Year', 'Country', 'decile',
        "EUR_HE01", "EUR_HE041", "EUR_HE045", "EUR_HE06", "EUR_HE10", 
        "EUR_HE09", "EUR_HJ08", "EUR_HJ90", "EUR_HE07"
    ]

    equiv_with_deciles = pd.read_csv(
        os.path.join(dirs['decile_dir'], "HBS_household_data_with_decile.csv")
    )
    
    # Filter to only include needed columns that exist
    available_cols = [col for col in cols_needed if col in equiv_with_deciles.columns]
    equiv_with_deciles = equiv_with_deciles[available_cols]

    # List of columns to equivalize
    cols_to_equivalize = [
        "EUR_HE01", "EUR_HE041", "EUR_HE045", "EUR_HE06", "EUR_HE10", 
        "EUR_HE09", "EUR_HJ08", "EUR_HJ90", "EUR_HE07"
    ]

    # Convert the necessary columns to numeric and equivalize
    for col in cols_to_equivalize:
        if col in equiv_with_deciles.columns:
            equiv_with_deciles[col] = pd.to_numeric(equiv_with_deciles[col], errors='coerce')
            equiv_with_deciles[col + '_equiv_share'] = (
                equiv_with_deciles[col] / equiv_with_deciles['HB061'] / 
                equiv_with_deciles['equi_disp_inc'] * 100
            )

    # Compute medians and standard deviations
    def compute_medians(group):
        result = {}
        for col in equiv_with_deciles.columns:
            if not col.endswith('_equiv_share'):
                continue

            # Apply the parent-only filter to EUR_HE10_equiv_share (education expenses)
            if col == 'EUR_HE10_equiv_share' and 'HB075A' in group.columns:
                mask = group['HB075A'].isin([2, 5])
                values = group.loc[mask, col].to_numpy()
                weights = group.loc[mask, 'HA10'].to_numpy()
            else:
                values = group[col].to_numpy()
                weights = group['HA10'].to_numpy()

            result[col + '_national_median'] = weighted_median(values, weights)
            result[col + '_national_std'] = weighted_std(values, weights)
        return pd.Series(result)

    # Apply median calculation
    grouped = equiv_with_deciles.groupby(['COUNTRY', 'YEAR'])
    medians = grouped.apply(compute_medians).reset_index()
    equiv_with_deciles = equiv_with_deciles.merge(medians, on=['COUNTRY', 'YEAR'], how='left')

    # Create indicators based on median ± std
    equiv_share_cols = [col for col in equiv_with_deciles.columns if col.endswith('_equiv_share')]
    
    for col in equiv_share_cols:
        median_col = col + '_national_median'
        std_col = col + '_national_std'
        
        if median_col in equiv_with_deciles.columns and std_col in equiv_with_deciles.columns:
            # Above median + std indicator
            equiv_with_deciles[col + '_above_2M'] = (
                equiv_with_deciles[col] > equiv_with_deciles[median_col] - equiv_with_deciles[std_col]
            ).astype(int)
            
            # Below median - std indicator  
            equiv_with_deciles[col + '_below_M2'] = (
                equiv_with_deciles[col] <= equiv_with_deciles[median_col] + equiv_with_deciles[std_col]
            ).astype(int)

    # Save processed data
    equiv_with_deciles.to_csv(
        os.path.join(dirs['decile_dir'], "HBS_household_data_equiv_with_deciles.csv"), 
        index=False
    )

    return equiv_with_deciles


def process_final_indicators(dirs):
    """
    Process final HBS indicators and create summary dataset.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Final HBS indicators dataset
    """
    print("Processing final HBS indicators...")
    
    # Load processed data
    df = pd.read_csv(os.path.join(dirs['decile_dir'], "HBS_household_data_equiv_with_deciles.csv"))

    # Define the variable mapping for indicators
    variable_filters = {
        'EUR_HE01_equiv_share_above_2M': [1],  # AE-HBS-1
        'EUR_HE01_equiv_share_below_M2': [1],  # AE-HBS-2
        'EUR_HE041_equiv_share_above_2M': [1], # HH-HBS-1
        'EUR_HE041_equiv_share_below_M2': [1], # HH-HBS-2
        'EUR_HE045_equiv_share_above_2M': [1], # HE-HBS-1
        'EUR_HE045_equiv_share_below_M2': [1], # HE-HBS-2
        'EUR_HE06_equiv_share_above_2M': [1],  # AC-HBS-1
        'EUR_HE06_equiv_share_below_M2': [1],  # AC-HBS-2
        'EUR_HE10_equiv_share_above_2M': [1],  # IE-HBS-1
        'EUR_HE10_equiv_share_below_M2': [1],  # IE-HBS-2
        'EUR_HE09_equiv_share_above_2M': [1],  # IC-HBS-1
        'EUR_HE09_equiv_share_below_M2': [1],  # IC-HBS-2
        'EUR_HE07_equiv_share_above_2M': [1],  # TT-HBS-1
        'EUR_HE07_equiv_share_below_M2': [1],  # TT-HBS-2
        'EUR_HJ08_equiv_share_above_2M': [1],  # EC-HBS-1
        'EUR_HJ08_equiv_share_below_M2': [1],  # EC-HBS-2
        'EUR_HJ90_equiv_share_above_2M': [1],  # TS-HBS-1
        'EUR_HJ90_equiv_share_below_M2': [1]   # TS-HBS-2
    }

    # Precompute masks per row
    for var, condition in variable_filters.items():
        if var in df.columns:
            df[f"_valid_{var}"] = df[var].isin(condition) & df[var].notna()

    # Convert weight to numeric
    df["HA10"] = pd.to_numeric(df["HA10"], errors='coerce')

    # Validate decile coverage before processing
    country_decile_coverage = df.groupby('COUNTRY')['decile'].nunique()
    print(f"📋 Countries with complete decile coverage (10 deciles): {(country_decile_coverage == 10).sum()}")
    print(f"📋 Countries with partial decile coverage: {(country_decile_coverage < 10).sum()}")
    
    # Group by Year, Country, Decile and calculate indicators
    group_cols = ['COUNTRY', 'YEAR', "decile"]
    results = []

    for group_keys, group in df.groupby(group_cols):
        group_result = dict(zip(group_cols, group_keys))
        total_weight = group["HA10"].sum()

        for var in variable_filters.keys():
            if var in df.columns:
                mask = group[f"_valid_{var}"]
                weighted_sum = group.loc[mask, "HA10"].sum()
                share = (weighted_sum / total_weight * 100) if total_weight > 0 else np.nan
                group_result[f"{var}_share"] = share

        results.append(group_result)
    
    # Also calculate indicators for total population per country (decile = "All")
    print("📊 Computing total population indicators (decile = 'All')...")
    total_group_cols = ['COUNTRY', 'YEAR']
    
    for group_keys, group in df.groupby(total_group_cols):
        group_result = dict(zip(total_group_cols, group_keys))
        group_result['decile'] = "All"
        total_weight = group["HA10"].sum()

        for var in variable_filters.keys():
            if var in df.columns:
                mask = group[f"_valid_{var}"]
                weighted_sum = group.loc[mask, "HA10"].sum()
                share = (weighted_sum / total_weight * 100) if total_weight > 0 else np.nan
                group_result[f"{var}_share"] = share

        results.append(group_result)

    summary_df = pd.DataFrame(results)

    # Rename columns according to EWBI naming convention
    rename_dict = {
        "YEAR": "year", "COUNTRY": "country",
        'EUR_HE01_equiv_share_above_2M_share': "AE-HBS-1",
        'EUR_HE01_equiv_share_below_M2_share': "AE-HBS-2",
        'EUR_HE041_equiv_share_above_2M_share': "HH-HBS-1",
        'EUR_HE041_equiv_share_below_M2_share': "HH-HBS-2",
        'EUR_HE045_equiv_share_above_2M_share': "HE-HBS-1",
        'EUR_HE045_equiv_share_below_M2_share': "HE-HBS-2",
        'EUR_HE06_equiv_share_above_2M_share': "AC-HBS-1",
        'EUR_HE06_equiv_share_below_M2_share': "AC-HBS-2",
        'EUR_HE10_equiv_share_above_2M_share': "IE-HBS-1",
        'EUR_HE10_equiv_share_below_M2_share': "IE-HBS-2",
        'EUR_HE09_equiv_share_above_2M_share': "IC-HBS-1",
        'EUR_HE09_equiv_share_below_M2_share': "IC-HBS-2",
        'EUR_HE07_equiv_share_above_2M_share': "TT-HBS-1",
        'EUR_HE07_equiv_share_below_M2_share': "TT-HBS-2",
        'EUR_HJ08_equiv_share_above_2M_share': "EC-HBS-1",
        'EUR_HJ08_equiv_share_below_M2_share': "EC-HBS-2",
        'EUR_HJ90_equiv_share_above_2M_share': "TS-HBS-1",
        'EUR_HJ90_equiv_share_below_M2_share': "TS-HBS-2"
    }

    summary_df = summary_df.rename(columns=rename_dict)

    # Melt to long format
    columns_to_melt = [
        "AE-HBS-1", "AE-HBS-2", "HH-HBS-1", "HH-HBS-2", "HE-HBS-1", "HE-HBS-2",
        "AC-HBS-1", "AC-HBS-2", "IE-HBS-1", "IE-HBS-2", "IC-HBS-1", "IC-HBS-2",
        "TT-HBS-1", "TT-HBS-2", "EC-HBS-1", "EC-HBS-2", "TS-HBS-1", "TS-HBS-2"
    ]

    # Filter to only include columns that exist
    available_melt_cols = [col for col in columns_to_melt if col in summary_df.columns]

    df_melted = summary_df.melt(
        id_vars=["year", "country", "decile"],
        value_vars=available_melt_cols,
        var_name="primary_index",
        value_name="value"
    )

    df_melted["database"] = "HBS"

    # Remove rows with NaN values
    df_melted = df_melted[df_melted["value"].notna()]

    # Save final results
    df_melted.to_csv(os.path.join(dirs['final_dir'], "HBS_household_final_summary.csv"), index=False)

    return df_melted


def main():
    """Main function to execute the complete HBS data processing pipeline."""
    print("Starting HBS data processing...")
    
    # Setup directories
    dirs = setup_directories()
    print(f"External data directory: {dirs['external_data']}")
    print(f"Output directory: {dirs['output_base']}")
    
    # Combine raw HBS data from all years
    print("Combining HBS data from all years...")
    combine_hbs_data(dirs)
    
    # Prepare household data with equivalized income
    prepare_household_data(dirs)
    
    # Calculate income deciles
    calculate_income_deciles(dirs)
    
    # Calculate consumption indicators
    calculate_consumption_indicators(dirs)
    
    # Process final indicators
    final_df = process_final_indicators(dirs)
    
    print("HBS processing completed successfully!")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Countries: {final_df['country'].nunique()}")
    print(f"Years: {final_df['year'].nunique()}")
    print(f"Indicators: {final_df['primary_index'].nunique()}")
    print(f"Data saved to: {os.path.join(dirs['final_dir'], 'HBS_household_final_summary.csv')}")


if __name__ == "__main__":
    main()