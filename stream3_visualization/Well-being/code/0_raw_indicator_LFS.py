#!/usr/bin/env python3
"""
Labour Force Survey (LFS) Data Processing Script

This script processes Labour Force Survey (LFS) data from 1983-2023 for European countries.
It combines yearly data files, calculates employment indicators, and prepares final output
for the European Well-Being Index (EWBI).

The script handles the following indicators:
- RT-LFS-1: Multiple jobs (NUMJOB_2or3_pct)
- RT-LFS-2: Wish to work more (WISHMORE_2_pct)  
- RT-LFS-3: Overtime/extra hours (EXTRAHRS_gt0_pct)
- RU-LFS-1: Employment status (EMPSTAT_1_pct)

Author: Data Processing Pipeline
Date: 2024
"""

import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from pathlib import Path


def setup_directories() -> dict:
    """
    Set up directory structure for LFS data processing using portable paths.
    
    Returns:
        Dictionary containing directory paths
    """
    # Get the absolute path to the project output directory
    OUTPUT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')))
    
    # External data directory (modify this path according to your external data location)
    EXTERNAL_DATA_DIR = Path(r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI")
    
    dirs = {
        'external_data': EXTERNAL_DATA_DIR,
        'raw_data': EXTERNAL_DATA_DIR / "0_data" / "LFS" / "LFS_1983-2023_YEARLY_full_set-002" / "LFS_1983-2023_YEARLY_full_set",
        'output_base': OUTPUT_DIR,
        'output_dir': OUTPUT_DIR / "0_raw_data_EUROSTAT" / "0_LFS",
        'merged_dir': OUTPUT_DIR / "0_raw_data_EUROSTAT" / "0_LFS" / "0_merged",
        'final_dir': OUTPUT_DIR / "0_raw_data_EUROSTAT" / "0_LFS"
    }
    
    # Create output directories if they don't exist
    for dir_path in [dirs['output_dir'], dirs['merged_dir'], dirs['final_dir']]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def get_required_columns() -> list:
    """
    Define the required columns for LFS data processing.
    
    Returns:
        List of column names needed from raw data
    """
    return [
        "REFYEAR", 
        "COUNTRY",
        "COEFFHH",  # Household weight
        "HHNUM", 
        "COEFFY", 
        "HHCOMP",
        "HHPARTNR",  # Partner in household = 1 
        "DEGURBA", 
        "SEX", 
        "AGE", 
        "EMPSTAT", 
        "NUMJOB", 
        "SEEKWORK", 
        "WANTWORK", 
        "SEEKREAS", 
        "WANTREAS", 
        "WISHMORE", 
        "ILOSTAT", 
        "FTPT", 
        "TEMP", 
        "TEMPREAS", 
        "FTPTREAS", 
        "HWWISH", 
        "NEEDCARE", 
        "HATLEVEL", 
        "LEAVREAS", 
        "HWUSUAL", 
        "ABSILLINJ", 
        "EXTRAHRS", 
        "HWACTUAL", 
        "EDUCFED4", 
        "EDUCFED12", 
        "EDUCNFE12", 
        "GENHEALTH", 
        "GALI", 
        "INCDECIL",
        # Additional columns for new indicators
        "VARITIME",  # RT-LFS-4 - No freedom on working time choice
        "SHIFTWK",   # RT-LFS-5 - Shift work in main job
        "NIGHTWK",   # RT-LFS-6 - Night work in main job
        "SATWK",     # RT-LFS-7 - Saturday work in main job
        "SUNWK"      # RT-LFS-8 - Sunday work in main job
    ]


def combine_lfs_data(data_path: Path, cols_needed: list) -> pd.DataFrame:
    """
    Combine LFS data files from multiple countries and years.
    
    Args:
        data_path: Path to the directory containing country folders
        cols_needed: List of required columns
        
    Returns:
        Combined DataFrame with all LFS data
    """
    df_list = []
    
    print("🔄 Combining LFS data files...")
    
    # Get all country folders
    country_folders = [f for f in data_path.iterdir() if f.is_dir()]
    
    for folder_path in tqdm(country_folders, desc="Processing countries"):
        folder_name = folder_path.name
        
        # Match folders like "BE_YEAR", "FR_YEAR", etc.
        match_folder = re.match(r"([A-Z]{2})_YEAR", folder_name)
        if not match_folder:
            print(f"Skipping folder: {folder_name}")
            continue

        country_code = match_folder.group(1)

        # Process files in the folder
        csv_files = list(folder_path.glob("*.csv"))
        
        for file_path in csv_files:
            file_name = file_path.name
            match_file = re.match(rf"{country_code}(\d{{4}})_y\.csv", file_name)
            if not match_file:
                continue

            year = match_file.group(1)

            try:
                # First, check which columns are actually available in this file
                df_check = pd.read_csv(file_path, nrows=0)  # Read only header
                available_cols = set(df_check.columns)
                
                # Filter requested columns to only those available
                cols_to_use = [col for col in cols_needed if col in available_cols]
                missing_cols = [col for col in cols_needed if col not in available_cols]
                
                # Read with available columns only
                df = pd.read_csv(file_path, usecols=cols_to_use)
                
                # Add missing columns as NaN
                for missing_col in missing_cols:
                    df[missing_col] = np.nan
                
                # Ensure all requested columns are present in the correct order
                df = df.reindex(columns=cols_needed, fill_value=np.nan)
                
                df["country"] = country_code
                df["year"] = int(year)
                df_list.append(df)
                
                if missing_cols:
                    print(f"✅ Loaded {file_name} from {folder_name} with shape {df.shape} (missing: {missing_cols})")
                else:
                    print(f"✅ Loaded {file_name} from {folder_name} with shape {df.shape}")
                
            except Exception as e:
                print(f"❌ Failed to read {file_path}: {e}")

    # Concatenate all dataframes
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        print(f"📊 Combined DataFrame shape: {combined_df.shape}")
        return combined_df
    else:
        raise ValueError("No data was loaded successfully")


def weighted_percentage(df, condition_col, condition_func, weight_col='COEFFY', base_condition=None):
    """
    Calculate weighted percentage given a condition, excluding NaN values 
    and optionally applying a base filtering condition.
    
    Args:
        df: DataFrame to process
        condition_col: Column name to evaluate condition on
        condition_func: Function that returns boolean mask for numerator
        weight_col: Column name containing weights
        base_condition: Optional function to filter base population
        
    Returns:
        Weighted percentage as float (0-100), or NaN if no valid data
    """
    # Filter out rows where the condition column is NaN
    df_filtered = df[df[condition_col].notna()]

    # Apply base filtering condition (e.g., x != -1)
    if base_condition is not None:
        df_filtered = df_filtered[base_condition(df_filtered[condition_col])]

    if df_filtered.empty:
        return np.nan

    # Apply numerator-specific condition
    mask_valid = condition_func(df_filtered[condition_col])
    df_valid = df_filtered[mask_valid]

    weighted_sum = df_valid[weight_col].sum()
    total_weight = df_filtered[weight_col].sum()

    return (weighted_sum / total_weight * 100) if total_weight > 0 else np.nan


def calculate_lfs_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate LFS employment indicators by year, country, and income decile.
    
    Args:
        df: Combined LFS DataFrame
        
    Returns:
        DataFrame with calculated indicators
    """
    print("🧮 Calculating LFS indicators...")
    
    # Define columns needed for computation
    analysis_cols = [
        "year", 
        "country",
        "HHNUM", 
        "COEFFY",
        "INCDECIL",
        "NUMJOB",    # RT-LFS-1 - Number of jobs
        "WISHMORE",  # RT-LFS-2 - Wish to work more
        "EXTRAHRS",  # RT-LFS-3 - Overtime or extra hours
        "EMPSTAT",   # RU-LFS-1 - Being in employment
        # New indicators
        "VARITIME",  # RT-LFS-4 - No freedom on working time choice
        "SHIFTWK",   # RT-LFS-5 - Shift work in main job
        "NIGHTWK",   # RT-LFS-6 - Night work in main job
        "SATWK",     # RT-LFS-7 - Saturday work in main job
        "SUNWK",     # RT-LFS-8 - Sunday work in main job
        "NEEDCARE"   # EL-LFS-2 - No relevant care service
    ]
    
    # Filter and prepare data
    analysis_df = df[analysis_cols].copy()
    
    # Rename columns
    analysis_df = analysis_df.rename(columns={
        "INCDECIL": "decile"
    })
    
    # Filter out invalid income deciles
    analysis_df = analysis_df[
        analysis_df["decile"].notna() & 
        (analysis_df["decile"] != 99.0) 
    ]
    
    print(f"📊 Analysis dataset shape after filtering: {analysis_df.shape}")
    
    # Calculate indicators for each group
    results = []
    
    # Validate that we have data for all countries and deciles
    country_decile_coverage = analysis_df.groupby('country')['decile'].nunique()
    print(f"📋 Countries with complete decile coverage (10 deciles): {(country_decile_coverage == 1  # EWBI0).sum()}")
    print(f"📋 Countries with partial decile coverage: {(country_decile_coverage < 10).sum()}")
    
    groups = analysis_df.groupby(["year", "country", "decile"])
    
    for (year, country, decile), group in tqdm(groups, desc="Processing groups"):
        # RT-LFS-1: Number of jobs (multiple jobs)
        numjob_pct = weighted_percentage(
            group,
            "NUMJOB",
            lambda x: x.isin([2, 3, 4]),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-2: Wish to work more
        wishmore_pct = weighted_percentage(
            group,
            "WISHMORE",
            lambda x: (x == 2  # EU Priorities),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-3: Overtime or extra hours
        extrahrs_pct = weighted_percentage(
            group,
            "EXTRAHRS",
            lambda x: (x > 0),
            base_condition=lambda x: (x != 999)
        )
        
        # RU-LFS-1: Employment status (employed)
        empstat_pct = weighted_percentage(
            group,
            "EMPSTAT",
            lambda x: (x == 1  # EWBI),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-4: No freedom on working time choice
        varitime_pct = weighted_percentage(
            group,
            "VARITIME",
            lambda x: x.isin([3, 4]),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-5: Shift work in main job
        shiftwk_pct = weighted_percentage(
            group,
            "SHIFTWK",
            lambda x: (x == 1  # EWBI),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-6: Night work in main job
        nightwk_pct = weighted_percentage(
            group,
            "NIGHTWK",
            lambda x: x.isin([1, 2]),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-7: Saturday work in main job
        satwk_pct = weighted_percentage(
            group,
            "SATWK",
            lambda x: x.isin([1, 2]),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-8: Sunday work in main job
        sunwk_pct = weighted_percentage(
            group,
            "SUNWK",
            lambda x: x.isin([1, 2]),
            base_condition=lambda x: (x != 9)
        )
        
        # EL-LFS-2: No relevant care service
        needcare_pct = weighted_percentage(
            group,
            "NEEDCARE",
            lambda x: x.isin([1, 2]),
            base_condition=lambda x: (x != 9)
        )
        
        results.append({
            "year": year,
            "country": country,
            "decile": decile,
            "NUMJOB_2or3_pct": numjob_pct,
            "WISHMORE_2_pct": wishmore_pct,
            "EXTRAHRS_gt0_pct": extrahrs_pct,
            "EMPSTAT_1_pct": empstat_pct,
            "VARITIME_34_pct": varitime_pct,
            "SHIFTWK_1_pct": shiftwk_pct,
            "NIGHTWK_12_pct": nightwk_pct,
            "SATWK_12_pct": satwk_pct,
            "SUNWK_12_pct": sunwk_pct,
            "NEEDCARE_12_pct": needcare_pct
        })
    
    # Also calculate indicators for total population per country (decile = "All")
    print("📊 Computing total population indicators (decile = 'All')...")
    total_groups = analysis_df.groupby(["year", "country"])
    
    for (year, country), group in tqdm(total_groups, desc="Processing total population groups"):
        # RT-LFS-1: Number of jobs (multiple jobs)
        numjob_pct = weighted_percentage(
            group,
            "NUMJOB",
            lambda x: x.isin([2, 3, 4]),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-2: Wish to work more
        wishmore_pct = weighted_percentage(
            group,
            "WISHMORE",
            lambda x: (x == 2  # EU Priorities),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-3: Overtime or extra hours
        extrahrs_pct = weighted_percentage(
            group,
            "EXTRAHRS",
            lambda x: (x > 0),
            base_condition=lambda x: (x != 999)
        )
        
        # RU-LFS-1: Employment status (employed)
        empstat_pct = weighted_percentage(
            group,
            "EMPSTAT",
            lambda x: (x == 1  # EWBI),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-4: No freedom on working time choice
        varitime_pct = weighted_percentage(
            group,
            "VARITIME",
            lambda x: x.isin([3, 4]),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-5: Shift work in main job
        shiftwk_pct = weighted_percentage(
            group,
            "SHIFTWK",
            lambda x: (x == 1  # EWBI),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-6: Night work in main job
        nightwk_pct = weighted_percentage(
            group,
            "NIGHTWK",
            lambda x: x.isin([1, 2]),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-7: Saturday work in main job
        satwk_pct = weighted_percentage(
            group,
            "SATWK",
            lambda x: x.isin([1, 2]),
            base_condition=lambda x: (x != 9)
        )
        
        # RT-LFS-8: Sunday work in main job
        sunwk_pct = weighted_percentage(
            group,
            "SUNWK",
            lambda x: x.isin([1, 2]),
            base_condition=lambda x: (x != 9)
        )
        
        # EL-LFS-2: No relevant care service
        needcare_pct = weighted_percentage(
            group,
            "NEEDCARE",
            lambda x: x.isin([1, 2]),
            base_condition=lambda x: (x != 9)
        )
        
        results.append({
            "year": year,
            "country": country,
            "decile": "All",
            "NUMJOB_2or3_pct": numjob_pct,
            "WISHMORE_2_pct": wishmore_pct,
            "EXTRAHRS_gt0_pct": extrahrs_pct,
            "EMPSTAT_1_pct": empstat_pct,
            "VARITIME_34_pct": varitime_pct,
            "SHIFTWK_1_pct": shiftwk_pct,
            "NIGHTWK_12_pct": nightwk_pct,
            "SATWK_12_pct": satwk_pct,
            "SUNWK_12_pct": sunwk_pct,
            "NEEDCARE_12_pct": needcare_pct
        })

    return pd.DataFrame(results)


def prepare_final_output(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare final melted output format for LFS indicators.
    
    Args:
        results_df: DataFrame with calculated indicators
        
    Returns:
        Melted DataFrame in final format
    """
    print("📋 Preparing final output format...")
    
    # Rename columns to match indicator names
    rename_dict = {
        "NUMJOB_2or3_pct": "RT-LFS-1",
        "WISHMORE_2_pct": "RT-LFS-2",
        "EXTRAHRS_gt0_pct": "RT-LFS-3",
        "EMPSTAT_1_pct": "RU-LFS-1",
        "VARITIME_34_pct": "RT-LFS-4",
        "SHIFTWK_1_pct": "RT-LFS-5",
        "NIGHTWK_12_pct": "RT-LFS-6",
        "SATWK_12_pct": "RT-LFS-7",
        "SUNWK_12_pct": "RT-LFS-8",
        "NEEDCARE_12_pct": "EL-LFS-2"
    }
    
    results_df = results_df.rename(columns=rename_dict)
    
    # Define the columns to melt
    columns_to_melt = list(rename_dict.values())
    
    # Melt the DataFrame to long format
    df_melted = results_df.melt(
        id_vars=["year", "country", "decile"],
        value_vars=columns_to_melt,
        var_name="primary_index",
        value_name="value"
    )
    
    # Add database identifier
    df_melted["database"] = "LFS"
    
    return df_melted


def save_outputs(combined_df: pd.DataFrame, final_df: pd.DataFrame, dirs: dict) -> None:
    """
    Save intermediate and final outputs.
    
    Args:
        combined_df: Combined raw data
        final_df: Final processed indicators
        dirs: Directory paths dictionary
    """
    print("💾 Saving output files...")
    
    # Save combined raw data
    combined_output = dirs['merged_dir'] / "LFS_combined.csv"
    combined_df.to_csv(combined_output, index=False)
    print(f"✅ Saved combined data to: {combined_output}")
    
    # Save final indicators
    final_output = dirs['final_dir'] / "LFS_household_final_summary.csv"
    final_df.to_csv(final_output, index=False)
    print(f"✅ Saved final indicators to: {final_output}")


def process_lfs_indicators(dirs):
    """
    Process and calculate LFS indicators.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Final LFS indicators dataset
    """
    print("Processing LFS indicators...")
    
    # This function can be removed or renamed if not needed
    # as the main processing is handled by calculate_lfs_indicators
    
    return pd.DataFrame()  # Placeholder


def main():
    """
    Main execution function for LFS data processing.
    """
    print("🚀 Starting LFS Data Processing Pipeline")
    print("=" * 50)
    
    try:
        # Setup directories
        dirs = setup_directories()
        print(f"📂 External data directory: {dirs['external_data']}")
        print(f"📂 Output directory: {dirs['output_dir']}")
        
        # Get required columns
        cols_needed = get_required_columns()
        print(f"📋 Processing {len(cols_needed)} columns")
        
        # Step 1: Combine data from all countries and years
        combined_df = combine_lfs_data(dirs['raw_data'], cols_needed)
        
        # Step 2: Calculate indicators
        results_df = calculate_lfs_indicators(combined_df)
        
        # Step 3: Prepare final output format
        final_df = prepare_final_output(results_df)
        
        # Step 4: Save outputs
        save_outputs(combined_df, final_df, dirs)
        
        # Summary statistics
        print("\n📈 Processing Summary:")
        print(f"   • Total records processed: {len(combined_df):,}")
        print(f"   • Countries: {combined_df['country'].nunique()}")
        print(f"   • Years: {combined_df['year'].min()}-{combined_df['year'].max()}")
        print(f"   • Final indicators: {len(final_df):,} rows")
        print(f"   • Indicator types: {final_df['primary_index'].nunique()}")
        
        print("\n✅ LFS processing completed successfully!")
        
        return final_df
        
    except Exception as e:
        print(f"\n❌ Error during LFS processing: {str(e)}")
        raise


if __name__ == "__main__":
    # Execute main processing
    result = main()
    
    # Display sample of final results
    if result is not None:
        print("\n📊 Sample of final results:")
        print(result.head(10))
        print(f"\nFinal dataset shape: {result.shape}")
