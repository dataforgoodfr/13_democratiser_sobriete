"""
European Health Interview Survey (EHIS) Data Processing Script

This script processes EHIS data from multiple waves and creates a final summary dataset
for use in the European Well-Being Index (EWBI).

Author: Data for Good - Well-being Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import os


def setup_directories():
    """Set up directories for data processing using portable paths."""
    # Get the absolute path to the project output directory
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    
    # External data directory (modify this path according to your external data location)
    EXTERNAL_DATA_DIR = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI"
    
    dirs = {
        'external_data': EXTERNAL_DATA_DIR,
        'output_base': OUTPUT_DIR,
        'ehis_output': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EHIS'),
        'merged_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EHIS', '0_merged'),
        'quintile_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EHIS', '1_income_quintile'),
        'final_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EHIS')
    }
    
    # Create output directories if they don't exist
    for dir_path in [dirs['ehis_output'], dirs['merged_dir'], dirs['quintile_dir'], dirs['final_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def combine_ehis_waves(dirs):
    """
    Combine EHIS data from multiple waves into a single dataset.
    
    Args:
        dirs (dict): Dictionary containing directory paths
        
    Returns:
        pd.DataFrame: Combined EHIS dataset from all waves
    """
    # Set relative path from external data directory
    relative_path = r"0_data/EHIS/EHIS all waves"
    full_path = os.path.join(dirs['external_data'], relative_path)

    # List to collect dataframes
    dfs = []

    # Column mapping for Wave 1
    wave1_rename_map = {
        "YEAR": "REFYEAR",
        "PWGT": "WGT",
        "IP01": "COUNTRY",
        "HA1A": "HA01A",
        "FV1": "FV01",
        "HA1": "HA01",
        "PE6": "PE06",
        "SK1": "SK01",
        "AL1": "AL01"
    }

    # Loop through wave folders
    for i in range(1, 4):
        if i == 1:
            # For wave 1, files are in 'Data EHIS' subfolder
            wave_folder = os.path.join(full_path, f"EHIS wave {i}", "Data EHIS")
            file_path = os.path.join(wave_folder, f"EHIS{i}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, sep=',')  # Comma separator for wave 1
                df.rename(columns=wave1_rename_map, inplace=True)  # Rename columns
                df["wave"] = i  # Add a column to indicate the wave
                dfs.append(df)
            else:
                print(f"File not found: {file_path}")
        else:
            # For waves 2 and 3, files are directly in 'EHIS wave {i}' folder
            wave_folder = os.path.join(full_path, f"EHIS wave {i}")
            if os.path.exists(wave_folder):
                for file_name in os.listdir(wave_folder):
                    if file_name.endswith("_Anonymisation.csv"):
                        file_path = os.path.join(wave_folder, file_name)
                        df = pd.read_csv(file_path, sep=';')  # Semicolon separator for waves 2 and 3
                        df["wave"] = i  # Add a column to indicate the wave
                        df["country"] = file_name.split("_")[0]  # Extract country code
                        dfs.append(df)
            else:
                print(f"Folder not found: {wave_folder}")

    # Concatenate all dataframes
    if dfs:
        stacked_df = pd.concat(dfs, ignore_index=True)
        print(f"Stacked dataframe shape: {stacked_df.shape}")
        return stacked_df
    else:
        print("No dataframes were loaded.")
        return pd.DataFrame()


def process_income_quintiles(df):
    """
    Process income quintile information from EHIS data.
    
    Args:
        df (pd.DataFrame): Combined EHIS dataset
        
    Returns:
        pd.DataFrame: Dataset with processed quintile information
    """
    # Ensure IN04 is string and zero-padded
    df["IN04"] = df["IN04"].astype(str).str.zfill(2)

    # Clean HHINCOME column (if it's coded as string or object)
    df["HHINCOME"] = pd.to_numeric(df["HHINCOME"], errors='coerce')

    # Convert IN04 to numeric, invalid parsing results in NaN
    df["IN04_numeric"] = pd.to_numeric(df["IN04"], errors='coerce')

    # Create 'quintile' column using HHINCOME where available, otherwise based on IN04
    df["quintile"] = df["HHINCOME"]
    mask = df["quintile"].isna() & df["IN04_numeric"].notna()
    df.loc[mask, "quintile"] = ((df.loc[mask, "IN04_numeric"] + 1) // 2).astype(int)
    
    return df


def weighted_percentage(df, condition_col, condition_func, weight_col='WGT', base_condition=None):
    """
    Calculate weighted percentage given a condition, excluding NaN values 
    and optionally applying a base filtering condition to both numerator and denominator.
    
    Args:
        df (pd.DataFrame): Input dataframe
        condition_col (str): Column name to apply condition on
        condition_func (function): Function to apply to condition column
        weight_col (str): Weight column name
        base_condition (function): Base filtering condition
        
    Returns:
        float: Weighted percentage or NaN if no valid data
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

    return weighted_sum / total_weight if total_weight > 0 else np.nan


def calculate_ehis_indicators(df):
    """
    Calculate EHIS health indicators by year, country, and quintile.
    
    Args:
        df (pd.DataFrame): Processed EHIS dataset
        
    Returns:
        pd.DataFrame: Calculated indicators summary
    """
    # Define columns needed for household analysis
    cols_needed_household = [
        "PID", "HHID", "WGT", "REFYEAR", "COUNTRY", "REGION", "DEG_URB",
        "HATLEVEL", "HHTYPE", "quintile", "HS1", "HS2", "HS3", "AC1A", "AW1", "AW2",
        "HA1A", "HA1B", "MH1A", "MH1B", "UN1A", "UN1B", "UN2A", "UN2B", "UN2C", "UN2D",
        "PE6", "FV1", "SK1", "AL1", "SS1"
    ]
    
    # Filter to only include needed columns that exist
    available_cols = [col for col in cols_needed_household if col in df.columns]
    merged_df = df[available_cols].copy()

    # Filter out unwanted values in REFYEAR and quintile
    merged_df = merged_df[
        merged_df["REFYEAR"].notna() & 
        (merged_df["REFYEAR"] != -1) & 
        (merged_df["REFYEAR"] != 8008) &
        (merged_df["REFYEAR"] != 2118.0) &
        (merged_df["quintile"] != -1)
    ]

    # Rename columns for clarity
    merged_df = merged_df.rename(columns={"REFYEAR": "Year", "COUNTRY": "Country"})

    # Validate quintile coverage before processing
    country_quintile_coverage = merged_df.groupby('Country')['quintile'].nunique()
    print(f"📋 Countries with complete quintile coverage (5 quintiles): {(country_quintile_coverage == 5).sum()}")
    print(f"📋 Countries with partial quintile coverage: {(country_quintile_coverage < 5).sum()}")
    
    # Calculate indicators for each group
    results = []

    for (year, country, quintile), group in merged_df.groupby(["Year", "Country", "quintile"]):
        # Calculate various health indicators
        HA1a_pct = weighted_percentage(
            group, "HA1A", lambda x: x.isin([2, 3, 4]),
            base_condition=lambda x: (x != -1) & (x != -2)
        )

        AC1A_pct = weighted_percentage(
            group, "AC1A", lambda x: x.isin([1]),
            base_condition=lambda x: (x != -1)
        )

        AL1_pct = weighted_percentage(
            group, "AL1", lambda x: (x.isin([1])),
            base_condition=lambda x: (x != -1)
        )

        SK1_pct = weighted_percentage(
            group, "SK1", lambda x: (x.isin([1])),
            base_condition=lambda x: (x != -1)
        )

        UN2c_pct = weighted_percentage(
            group, "UN2C", lambda x: (x.isin([1])),
            base_condition=lambda x: (x != -1)
        )

        PE6_pct = weighted_percentage(
            group, "PE6", lambda x: (x.isin([0])),
            base_condition=lambda x: (x != -1)
        )

        MH1b_pct = weighted_percentage(
            group, "MH1B", lambda x: (x.isin([4])),
            base_condition=lambda x: (x != -1)
        )

        HA1b_pct = weighted_percentage(
            group, "HA1B", lambda x: (x.isin([2, 3, 4])),
            base_condition=lambda x: (x != -1) & (x != -2)
        )

        SS1_pct = weighted_percentage(
            group, "SS1", lambda x: (x.isin([1])),
            base_condition=lambda x: (x != -1)
        )

        MH1a_pct = weighted_percentage(
            group, "MH1A", lambda x: (x.isin([4])),
            base_condition=lambda x: (x != -1)
        )

        FV1_pct = weighted_percentage(
            group, "FV1", lambda x: (x.isin([4])),
            base_condition=lambda x: (x != -1)
        )

        results.append({
            "Year": year,
            "Country": country,
            "quintile": quintile,
            "HA1A": HA1a_pct,
            "FV1": FV1_pct,
            "MH1A": MH1a_pct,
            "SS1": SS1_pct,
            "HA1B": HA1b_pct,
            "MH1B": MH1b_pct,
            "PE6": PE6_pct,
            "UN2C": UN2c_pct,
            "SK1": SK1_pct,
            "AL1": AL1_pct,
            "AC1A": AC1A_pct
        })

    return pd.DataFrame(results)


def calculate_level5_statistics(df):
    """
    Calculate Level 5 statistics - raw percentages from microdata at different aggregation levels.
    
    Args:
        df (pd.DataFrame): Processed EHIS dataset with quintiles
        
    Returns:
        pd.DataFrame: Level 5 statistics (value_5A through value_5D)
    """
    print("Computing Level 5 statistics from EHIS microdata...")
    
    # Define columns needed for analysis
    cols_needed = [
        "PID", "HHID", "WGT", "REFYEAR", "COUNTRY", "quintile",
        "HA1A", "HA1B", "MH1A", "MH1B", "UN2C", "PE6", "FV1", "SK1", "AL1", "SS1", "AC1A"
    ]
    
    # Filter and clean data
    available_cols = [col for col in cols_needed if col in df.columns]
    merged_df = df[available_cols].copy()
    
    # Filter out unwanted values
    merged_df = merged_df[
        merged_df["REFYEAR"].notna() & 
        (merged_df["REFYEAR"] != -1) & 
        (merged_df["REFYEAR"] != 8008) &
        (merged_df["REFYEAR"] != 2118.0) &
        (merged_df["quintile"] != -1)
    ]
    
    # Rename columns
    merged_df = merged_df.rename(columns={"REFYEAR": "year", "COUNTRY": "country"})
    
    level5_results = []
    
    # Define indicator calculation functions (same as in main calculation)
    def calculate_indicator_percentages(group_data):
        """Calculate all indicator percentages for a given group"""
        return {
            "AN-EHIS-1": weighted_percentage(
                group_data, "HA1A", lambda x: x.isin([2, 3, 4]),
                base_condition=lambda x: (x != -1) & (x != -2)
            ),
            "AE-EHIS-1": weighted_percentage(
                group_data, "FV1", lambda x: x.isin([4]),
                base_condition=lambda x: (x != -1)
            ),
            "EL-EHIS-1": weighted_percentage(
                group_data, "MH1A", lambda x: x.isin([4]),
                base_condition=lambda x: (x != -1)
            ),
            "EC-EHIS-1": weighted_percentage(
                group_data, "SS1", lambda x: x.isin([1]),
                base_condition=lambda x: (x != -1)
            ),
            "ED-EHIS-1": weighted_percentage(
                group_data, "HA1B", lambda x: x.isin([2, 3, 4]),
                base_condition=lambda x: (x != -1) & (x != -2)
            ),
            "AH-EHIS-1": weighted_percentage(
                group_data, "MH1B", lambda x: x.isin([4]),
                base_condition=lambda x: (x != -1)
            ),
            "AH-EHIS-2": weighted_percentage(
                group_data, "PE6", lambda x: x.isin([0]),
                base_condition=lambda x: (x != -1)
            ),
            "AC-EHIS-1": weighted_percentage(
                group_data, "UN2C", lambda x: x.isin([1]),
                base_condition=lambda x: (x != -1)
            ),
            "AB-EHIS-1": weighted_percentage(
                group_data, "SK1", lambda x: x.isin([1]),
                base_condition=lambda x: (x != -1)
            ),
            "AB-EHIS-2": weighted_percentage(
                group_data, "AL1", lambda x: x.isin([1]),
                base_condition=lambda x: (x != -1)
            ),
            "AB-EHIS-3": weighted_percentage(
                group_data, "AC1A", lambda x: x.isin([1]),
                base_condition=lambda x: (x != -1)
            )
        }
    
    # Value_5A: Percentage per quintile, country, year
    print("Computing value_5A: Percentage per quintile, country, year...")
    for (year, country, quintile), group in merged_df.groupby(["year", "country", "quintile"]):
        indicator_pcts = calculate_indicator_percentages(group)
        
        for indicator_code, percentage in indicator_pcts.items():
            if not pd.isna(percentage):
                level5_results.append({
                    "primary_index": indicator_code,
                    "country": country,
                    "quintile": quintile,
                    "decile": np.nan,  # EHIS uses quintiles, not deciles
                    "year": year,
                    "value": percentage * 100,  # Convert to percentage
                    "database": "EHIS",
                    "level5_type": "value_5A"
                })
    
    # Value_5B: Percentage per quintile, year (all countries combined)
    print("Computing value_5B: Percentage per quintile, year (all countries)...")
    for (year, quintile), group in merged_df.groupby(["year", "quintile"]):
        indicator_pcts = calculate_indicator_percentages(group)
        
        for indicator_code, percentage in indicator_pcts.items():
            if not pd.isna(percentage):
                level5_results.append({
                    "primary_index": indicator_code,
                    "country": "All Countries",
                    "quintile": quintile,
                    "decile": np.nan,  # EHIS uses quintiles, not deciles
                    "year": year,
                    "value": percentage * 100,
                    "database": "EHIS",
                    "level5_type": "value_5B"
                })
    
    # Value_5C: Overall percentage per country, year (all quintiles combined)
    print("Computing value_5C: Overall percentage per country, year...")
    for (year, country), group in merged_df.groupby(["year", "country"]):
        indicator_pcts = calculate_indicator_percentages(group)
        
        for indicator_code, percentage in indicator_pcts.items():
            if not pd.isna(percentage):
                level5_results.append({
                    "primary_index": indicator_code,
                    "country": country,
                    "quintile": "All",  # EHIS uses quintile='All' for overall aggregation
                    "decile": np.nan,   # EHIS always has NaN deciles
                    "year": year,
                    "value": percentage * 100,
                    "database": "EHIS",
                    "level5_type": "value_5C"
                })
    
    # Value_5D: Overall percentage per year (all countries, all quintiles)
    print("Computing value_5D: Overall percentage per year (all countries)...")
    for year, group in merged_df.groupby(["year"]):
        indicator_pcts = calculate_indicator_percentages(group)
        
        for indicator_code, percentage in indicator_pcts.items():
            if not pd.isna(percentage):
                level5_results.append({
                    "primary_index": indicator_code,
                    "country": "All Countries",
                    "quintile": "All",  # EHIS uses quintile='All' for overall aggregation
                    "decile": np.nan,   # EHIS always has NaN deciles
                    "year": year,
                    "value": percentage * 100,
                    "database": "EHIS",
                    "level5_type": "value_5D"
                })
    
    level5_df = pd.DataFrame(level5_results)
    
    # EHIS data should maintain quintile structure only - no decile conversion needed
    # value_5A and value_5B use quintiles (1-5), value_5C and value_5D use decile='All' for aggregation
    
    print(f"Generated {len(level5_df)} Level 5 statistics")
    return level5_df


def format_final_output(df):
    """
    Format the final output dataset with proper column names and structure.
    
    Args:
        df (pd.DataFrame): Results dataframe
        
    Returns:
        pd.DataFrame: Formatted final dataset
    """
    # Rename columns according to EWBI naming convention
    rename_dict = {
        "Year": "year",
        "Country": "country",
        "HA1A": "AN-EHIS-1",
        "FV1": "AE-EHIS-1",
        "MH1A": "EL-EHIS-1",
        "SS1": "EC-EHIS-1",
        "HA1B": "ED-EHIS-1",
        "MH1B": "AH-EHIS-1",
        "PE6": "AH-EHIS-2",
        "UN2C": "AC-EHIS-1",
        "SK1": "AB-EHIS-1",
        "AL1": "AB-EHIS-2",
        "AC1A": "AB-EHIS-3"
    }

    df = df.rename(columns=rename_dict)

    # Define the columns to melt (convert from wide to long format)
    columns_to_melt = [
        "AN-EHIS-1", "AE-EHIS-1", "EL-EHIS-1", "EC-EHIS-1", "ED-EHIS-1",
        "AH-EHIS-1", "AH-EHIS-2", "AC-EHIS-1", "AB-EHIS-1", "AB-EHIS-2", "AB-EHIS-3"
    ]

    # Melt the DataFrame to long format
    df_melted = df.melt(
        id_vars=["year", "country", "quintile"],
        value_vars=columns_to_melt,
        var_name="primary_index",
        value_name="value"
    )

    # Add database identifier
    df_melted["database"] = "EHIS"

    # Remove rows with NaN values
    df_melted = df_melted[df_melted["value"].notna()]

    return df_melted


def main():
    """Main function to execute the complete EHIS data processing pipeline."""
    print("Starting EHIS data processing...")
    
    # Setup directories
    dirs = setup_directories()
    print(f"External data directory: {dirs['external_data']}")
    print(f"Output directory: {dirs['output_base']}")
    
    # Combine data from all EHIS waves
    print("Combining EHIS waves...")
    stacked_df = combine_ehis_waves(dirs)
    
    if stacked_df.empty:
        print("Error: No data was loaded. Exiting.")
        return
    
    # Process income quintiles
    print("Processing income quintiles...")
    stacked_df = process_income_quintiles(stacked_df)
    
    # Skip saving large combined dataset to avoid memory issues
    print("Skipping combined dataset save (too large)...")
    
    # Calculate EHIS indicators (Level 4 - normalized indicators)
    print("Calculating EHIS health indicators...")
    results_df = calculate_ehis_indicators(stacked_df)
    
    # Calculate Level 5 statistics (raw percentage statistics from microdata)
    print("Calculating Level 5 statistics...")
    level5_df = calculate_level5_statistics(stacked_df)
    
    # Format final output for Level 4 indicators
    print("Formatting Level 4 output...")
    final_df = format_final_output(results_df)
    
    # Save Level 4 results
    print("Saving Level 4 results...")
    level4_output_path = os.path.join(dirs['final_dir'], "EHIS_level4_indicators.csv")
    final_df.to_csv(level4_output_path, index=False)
    
    # Also save as expected filename for main integration
    household_summary_path = os.path.join(dirs['final_dir'], "EHIS_household_final_summary.csv")
    final_df.to_csv(household_summary_path, index=False)
    print(f"Also saved as: {household_summary_path}")
    
    # Also save as expected filename for final integration
    household_summary_path = os.path.join(dirs['final_dir'], "EHIS_household_final_summary.csv")
    final_df.to_csv(household_summary_path, index=False)
    
    # Save Level 5 statistics
    print("Saving Level 5 statistics...")
    level5_output_path = os.path.join(dirs['final_dir'], "EHIS_level5_statistics.csv")
    level5_df.to_csv(level5_output_path, index=False)
    
    print(f"EHIS processing completed successfully!")
    print(f"Level 4 dataset shape: {final_df.shape}")
    print(f"Level 5 statistics shape: {level5_df.shape}")
    print(f"Level 4 data saved to: {level4_output_path}")
    print(f"Level 5 statistics saved to: {level5_output_path}")
    
    # Display summary statistics
    print("\nLevel 4 Summary statistics:")
    print(f"Countries: {final_df['country'].nunique()}")
    print(f"Years: {final_df['year'].nunique()}")
    print(f"Indicators: {final_df['primary_index'].nunique()}")
    
    print("\nLevel 5 Summary statistics:")
    print(f"Countries: {level5_df['country'].nunique()}")
    print(f"Years: {level5_df['year'].nunique()}")
    print(f"Indicators: {level5_df['primary_index'].nunique()}")
    print(f"Level 5 types: {level5_df['level5_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()