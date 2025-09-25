"""
EU Statistics on Income and Living Conditions (EU-SILC) Data Processing Script

This script processes EU-SILC data from multiple files and creates final summary datasets
for use in the European Well-Being Index (EWBI).

Author: Data for Good - Well-being Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import os
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
        'silc_output': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC'),
        'merged_init_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC', '0_merged_init'),
        'merged_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC', '0_merged'),
        'decile_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC', '1_income_decile'),
        'overcrowd_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC', '2_overcrowding'),
        'final_merged_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC', '3_final_merged_df'),
        'final_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC')
    }
    
    # Create output directories if they don't exist
    for key, dir_path in dirs.items():
        if key not in ['external_data', 'output_base']:
            os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def combine_household_data(dirs):
    """
    Combine EU-SILC household data (H-files) from all countries and years.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Combined household dataset
    """
    cols_needed_household = [
        "HB010", "HB020", "HB030", "HY020", "HH030",
        "HS050", "HD080", "HS011", "HH050", "HS021",
        "HS060", "HS120", "HS040"
    ]

    base_path = os.path.join(dirs['external_data'], "0_data/EU-SILC/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set")
    dfs = []

    # Loop through countries and years
    for country in os.listdir(base_path):
        country_path = os.path.join(base_path, country)
        if not os.path.isdir(country_path):
            continue

        for year in os.listdir(country_path):
            year_path = os.path.join(country_path, year)
            if not os.path.isdir(year_path):
                continue

            year_suffix = year[-2:]
            file_name = f"UDB_c{country}{year_suffix}H.csv"
            file_path = os.path.join(year_path, file_name)

            if os.path.exists(file_path):
                try:
                    # Read only header first to see what columns exist
                    sample_df = pd.read_csv(file_path, sep=",", nrows=5)
                    available_cols = sample_df.columns.tolist()

                    # Determine which needed columns exist in this file
                    cols_present = [col for col in cols_needed_household if col in available_cols]

                    if not cols_present:
                        print(f"⚠️ No needed columns found in {file_name}, skipping.")
                        continue

                    # Read again using only available columns
                    df = pd.read_csv(file_path, sep=",", usecols=cols_present)

                    # Add missing cols as NaNs
                    for col in cols_needed_household:
                        if col not in df.columns:
                            df[col] = pd.NA

                    df["Country"] = country
                    df["Year"] = f"20{year}"
                    dfs.append(df)

                except Exception as e:
                    print(f"❌ Error reading {file_name}: {e}")

    # Merge all data
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(dirs['merged_dir'], "EU_SILC_combined_household_data.csv")
        final_df.to_csv(output_path, index=False)
        print(f"Household data combined: {final_df.shape}")
        return final_df
    else:
        print("❗No valid household dataframes were collected.")
        return pd.DataFrame()


def combine_household_register(dirs):
    """
    Combine EU-SILC household register data (D-files) from all countries and years.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Combined household register dataset
    """
    cols_needed_household = [
        "DB010",  # Year
        "DB020",  # Country
        "DB030",  # ID
        "DB090"   # Weight
    ]

    base_path = os.path.join(dirs['external_data'], "0_data/EU-SILC/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set")
    dfs = []

    # Loop through countries and years
    for country in os.listdir(base_path):
        country_path = os.path.join(base_path, country)
        if not os.path.isdir(country_path):
            continue

        for year in os.listdir(country_path):
            year_path = os.path.join(country_path, year)
            if not os.path.isdir(year_path):
                continue

            year_suffix = year[-2:]
            file_name = f"UDB_c{country}{year_suffix}D.csv"
            file_path = os.path.join(year_path, file_name)

            if os.path.exists(file_path):
                try:
                    sample_df = pd.read_csv(file_path, sep=",", nrows=5)
                    available_cols = sample_df.columns.tolist()
                    cols_present = [col for col in cols_needed_household if col in available_cols]

                    if not cols_present:
                        print(f"⚠️ No needed columns found in {file_name}, skipping.")
                        continue

                    df = pd.read_csv(file_path, sep=",", usecols=cols_present)

                    for col in cols_needed_household:
                        if col not in df.columns:
                            df[col] = pd.NA

                    df["Country"] = country
                    df["Year"] = f"20{year}"
                    dfs.append(df)

                except Exception as e:
                    print(f"❌ Error reading {file_name}: {e}")

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(os.path.join(dirs['merged_dir'], "EU_SILC_combined_household_register.csv"), index=False)
        print(f"Household register data combined: {final_df.shape}")
        return final_df
    else:
        print("❗No valid household register dataframes were collected.")
        return pd.DataFrame()


def combine_personal_register(dirs):
    """
    Combine EU-SILC personal register data (R-files) from all countries and years.
    
    Returns:
        pd.DataFrame: Combined personal register dataset
    """
    cols_needed_personal = [
        "RB010",  # Year
        "RB020",  # Country
        "RB030",  # ID
        "RB050",  # Weight
        "RB081",  # Age
        "RB082",  # Age of person interviewed (alternative age field)
        "RL010",  # IS-SILC-1 - Education at pre-school
        "RL020"   # IS-SILC-2 - Education at compulsory school
    ]

    base_path = os.path.join(dirs['external_data'], "0_data/EU-SILC/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set")
    dfs = []

    for country in os.listdir(base_path):
        country_path = os.path.join(base_path, country)
        if not os.path.isdir(country_path):
            continue

        for year in os.listdir(country_path):
            year_path = os.path.join(country_path, year)
            if not os.path.isdir(year_path):
                continue

            year_suffix = year[-2:]
            file_name = f"UDB_c{country}{year_suffix}R.csv"
            file_path = os.path.join(year_path, file_name)

            if os.path.exists(file_path):
                try:
                    sample_df = pd.read_csv(file_path, sep=",", nrows=5)
                    available_cols = sample_df.columns.tolist()
                    cols_present = [col for col in cols_needed_personal if col in available_cols]

                    if not cols_present:
                        print(f"⚠️ No needed columns found in {file_name}, skipping.")
                        continue

                    df = pd.read_csv(file_path, sep=",", usecols=cols_present)

                    for col in cols_needed_personal:
                        if col not in df.columns:
                            df[col] = pd.NA

                    df["Country"] = country
                    df["Year"] = f"20{year}"
                    dfs.append(df)

                except Exception as e:
                    print(f"❌ Error reading {file_name}: {e}")

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(os.path.join(dirs['merged_dir'], "EU_SILC_combined_personal_register.csv"), index=False)
        print(f"Personal register data combined: {final_df.shape}")
        return final_df
    else:
        print("❗No valid personal register dataframes were collected.")
        return pd.DataFrame()


def combine_personal_data(dirs):
    """
    Combine EU-SILC personal data (P-files) from all countries and years.
    
    Returns:
        pd.DataFrame: Combined personal dataset
    """
    cols_needed_personal = [
        "PB010", "PB020", "PB030", "PB140", "PB200",
        "PW010", "PW191", "PD060", "PD070", "PH010", "PH020", "PH030",
        "PL086", "PH040", "PH050", "PE041", "PL141", "PL145", "PL080"
    ]

    base_path = os.path.join(dirs['external_data'], "0_data/EU-SILC/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set")
    dfs = []

    for country in os.listdir(base_path):
        country_path = os.path.join(base_path, country)
        if not os.path.isdir(country_path):
            continue

        for year in os.listdir(country_path):
            year_path = os.path.join(country_path, year)
            if not os.path.isdir(year_path):
                continue

            year_suffix = year[-2:]
            file_name = f"UDB_c{country}{year_suffix}P.csv"
            file_path = os.path.join(year_path, file_name)

            if os.path.exists(file_path):
                try:
                    sample_df = pd.read_csv(file_path, sep=",", nrows=5)
                    available_cols = sample_df.columns.tolist()
                    cols_present = [col for col in cols_needed_personal if col in available_cols]

                    if not cols_present:
                        print(f"⚠️ No needed columns found in {file_name}, skipping.")
                        continue

                    df = pd.read_csv(file_path, sep=",", usecols=cols_present)

                    for col in cols_needed_personal:
                        if col not in df.columns:
                            df[col] = pd.NA

                    df["Country"] = country
                    df["Year"] = f"20{year}"
                    dfs.append(df)

                except Exception as e:
                    print(f"❌ Error reading {file_name}: {e}")

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(os.path.join(dirs['merged_dir'], "EU_SILC_combined_personal_data.csv"), index=False)
        print(f"Personal data combined: {final_df.shape}")
        return final_df
    else:
        print("❗No valid personal dataframes were collected.")
        return pd.DataFrame()


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


def calculate_income_deciles(dirs):
    """
    Calculate income deciles using equivalized disposable income.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Dataset with income deciles
    """
    print("Calculating income deciles...")
    
    # Load necessary data for equivalized income calculation
    cols_needed_household = ["HB010", "HB020", "HB030", "HY020"]
    cols_needed_personal = ["RB010", "RB020", "RB030", "RB081", "RB082"]

    household_df = pd.read_csv(
        os.path.join(dirs['merged_dir'], "EU_SILC_combined_household_data.csv"),
        usecols=cols_needed_household
    )

    personal_df = pd.read_csv(
        os.path.join(dirs['merged_dir'], "EU_SILC_combined_personal_register.csv"),
        usecols=cols_needed_personal
    )

    # Ensure both household and personal IDs are strings
    personal_df["RB030"] = personal_df["RB030"].astype(str)
    personal_df["RB040"] = personal_df["RB030"].str[:-2]
    household_df["HB030"] = household_df["HB030"].astype(str)

    # Merge personal and household data
    merged_df = personal_df.merge(
        household_df,
        left_on=["RB010", "RB020", "RB040"],
        right_on=["HB010", "HB020", "HB030"],
        how="left"
    )

    # Create unified age column using RB082 when RB081 is missing
    merged_df['age'] = merged_df['RB081'].fillna(merged_df['RB082'])

    # Define the OECD modified scale weights
    def oecd_weight(age):
        if pd.isna(age):
            return 0.5  # Default weight for missing age
        if age < 14:
            return 0.3
        elif age >= 14:
            return 0.5

    # Apply weights and sum by household
    merged_df["oecd_weight"] = merged_df["age"].apply(oecd_weight)

    # Set weight of first adult to 1.0
    merged_df.sort_values(by=["HB010", "HB020", "HB030", "age"], ascending=[True, True, True, False], inplace=True)

    # Create a flag for the first person in each household
    merged_df["person_rank"] = merged_df.groupby(["HB010", "HB020", "HB030"]).cumcount()
    merged_df["oecd_weight"] = merged_df.apply(
        lambda row: 1.0 if row["person_rank"] == 0 else row["oecd_weight"], axis=1
    )

    # Calculate equivalent size per household
    equiv_size_df = merged_df.groupby(["HB010", "HB020", "HB030"])["oecd_weight"].sum().reset_index()
    equiv_size_df.rename(columns={"oecd_weight": "equivalent_size"}, inplace=True)

    # Merge equivalent size back to household dataset
    household_df = household_df.merge(equiv_size_df, on=["HB010", "HB020", "HB030"], how="left")

    # Compute equivalised disposable income
    household_df["equi_disp_inc"] = household_df["HY020"] / household_df["equivalent_size"]

    # Save intermediate result
    household_df.to_csv(os.path.join(dirs['decile_dir'], "EU_SILC_household_equi_inc.csv"), index=False)

    # Load household register for weights
    household_register_df = pd.read_csv(
        os.path.join(dirs['merged_dir'], "EU_SILC_combined_household_register.csv")
    )[["DB010", "DB020", "DB030", "DB090"]]

    # Ensure consistent data types for merge columns
    household_df["HB030"] = household_df["HB030"].astype(str)
    household_register_df["DB030"] = household_register_df["DB030"].astype(str)
    household_df["HB020"] = household_df["HB020"].astype(str)
    household_register_df["DB020"] = household_register_df["DB020"].astype(str)
    household_df["HB010"] = household_df["HB010"].astype(int)
    household_register_df["DB010"] = household_register_df["DB010"].astype(int)

    equival_inc_df = household_df.merge(
        household_register_df,
        left_on=["HB010", "HB020", "HB030"],
        right_on=["DB010", "DB020", "DB030"],
        how="left"
    )

    # Define the deciles you want to compute
    deciles = np.arange(0.1, 1.0, 0.1)  # 0.1 to 0.9

    # Group the DataFrame by year and country
    def compute_deciles(group):
        values = group['equi_disp_inc'].to_numpy()
        weights = group['DB090'].to_numpy()
        decile_values = weighted_quantile(values, weights, deciles)
        
        return pd.Series(decile_values, index=[f'decile_{int(d*10)}' for d in deciles])

    # Apply the function
    decile_df = equival_inc_df.groupby(['HB010', 'HB020']).apply(compute_deciles).reset_index()
    decile_df.rename(columns={'HB010': 'year', 'HB020': 'country'}, inplace=True)
    decile_df.to_csv(os.path.join(dirs['decile_dir'], "EU_SILC_household_decile.csv"), index=False)

    # Create the decile category for each household
    equiv_with_deciles = equival_inc_df.merge(
        decile_df, left_on=['HB010', 'HB020'], right_on=['year', 'country'], how='left'
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
    equiv_with_deciles.to_csv(os.path.join(dirs['decile_dir'], "EU_SILC_household_data_with_decile.csv"), index=False)

    return equiv_with_deciles


def calculate_overcrowding(dirs):
    """
    Calculate overcrowding indicator based on EU-SILC data.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Dataset with overcrowding indicator
    """
    print("Calculating overcrowding indicators...")
    
    cols_needed_household = ["HB010", "HB020", "HB030", "HH030"]
    cols_needed_personal = ["PB030", "PB010", "PB020", "PB140", "PB200"]

    # Load the data
    print("Loading household data...")
    hh = pd.read_csv(
        os.path.join(dirs['merged_dir'], "EU_SILC_combined_household_data.csv"),
        usecols=cols_needed_household
    )

    print("Loading personal data...")
    pp = pd.read_csv(
        os.path.join(dirs['merged_dir'], "EU_SILC_combined_personal_data.csv"),
        usecols=cols_needed_personal
    )

    # Convert both merge keys to string
    pp['PB030'] = pp['PB030'].fillna(0).astype('int32').astype(str)
    hh['HB030'] = hh['HB030'].fillna(0).astype('int32').astype(str)

    # Extract household ID from personal ID (first part of the string)
    pp['household_id'] = pp['PB030'].str[:-2]

    # Perform the merge on year, country, and household ID
    data = pp.merge(hh, left_on=['PB010', 'PB020', 'household_id'],
                        right_on=['HB010', 'HB020', 'HB030'])

    # Calculate age
    data['age'] = data['PB010'] - data['PB140']

    # Categorize age for overcrowding rules
    def classify_person(row):
        age = row['age']
        if age >= 18:
            return 'adult'
        elif 12 <= age < 18:
            return 'teen'
        elif age < 12:
            return 'child'
        else:
            return 'unknown'

    print("Classifying age groups...")
    data['age_group'] = data.apply(classify_person, axis=1)

    # Group by household and calculate required rooms
    def required_rooms(group):
        # Count adults in consensual unions (PB200 = 1 or 2) who can share bedrooms
        adults_in_unions = group[(group['age_group'] == 'adult') & (group['PB200'].isin([1, 2]))]
        adults_single = group[(group['age_group'] == 'adult') & (~group['PB200'].isin([1, 2]) | group['PB200'].isna())]
        
        # For adults in unions, assume they can share bedrooms in pairs
        # This is a simplification - ideally we'd match actual couples
        n_couples = len(adults_in_unions) // 2
        n_single_adults = len(adults_single) + (len(adults_in_unions) % 2)  # Include unpaired adults
        
        teens = group[group['age_group'] == 'teen']
        n_teen_pairs = len(teens) // 2
        n_remaining_teens = len(teens) % 2
        children = group[group['age_group'] == 'child']
        n_child_pairs = len(children) // 2
        n_remaining_children = len(children) % 2
        
        required = (1 + n_couples + n_single_adults + n_teen_pairs + n_remaining_teens + 
                   n_child_pairs + n_remaining_children)
        return pd.Series({'required_rooms': required, 'HH030': group['HH030'].iloc[0]})

    # Get groups and show progress
    grouped = data.groupby(['HB010', 'HB020', 'HB030'])
    total_households = len(grouped)
    print(f"Processing {total_households:,} households for overcrowding calculation...")
    
    # Apply with progress bar
    tqdm.pandas(desc="Calculating required rooms")
    rooms_df = grouped.progress_apply(required_rooms).reset_index()

    # Calculate overcrowding indicator
    print("Calculating overcrowding status...")
    rooms_df['overcrowded'] = (rooms_df['HH030'] < rooms_df['required_rooms']).astype(int)

    print("Saving overcrowding results...")
    rooms_df.to_csv(os.path.join(dirs['overcrowd_dir'], "EU_SILC_household_data_with_overcrowding.csv"), index=False)
    print(f"Overcrowding calculation completed! Results saved for {len(rooms_df):,} households.")

    return rooms_df


def process_household_indicators(dirs):
    """
    Process and calculate household-level indicators from EU-SILC data.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Final household indicators dataset
    """
    print("Processing household indicators...")
    
    # Load and merge all necessary datasets
    cols_needed_household = [
        "HB010", "HB020", "HB030", "HS050", "HD080", "HS011", "HH050", 
        "HS021", "HS060", "HS120", "HS040"
    ]

    household_df = pd.read_csv(
        os.path.join(dirs['merged_dir'], "EU_SILC_combined_household_data.csv"),
        usecols=cols_needed_household
    )

    household_df_weight = pd.read_csv(
        os.path.join(dirs['merged_dir'], "EU_SILC_combined_household_register.csv")
    )[["DB010", "DB020", "DB030", "DB090"]]

    decile_df = pd.read_csv(
        os.path.join(dirs['decile_dir'], "EU_SILC_household_data_with_decile.csv")
    )
    
    overpop_df = pd.read_csv(
        os.path.join(dirs['overcrowd_dir'], "EU_SILC_household_data_with_overcrowding.csv")
    )

    # Convert merge keys to string
    household_df['HB030'] = household_df['HB030'].fillna(0).astype('int32').astype(str)
    decile_df['HB030'] = decile_df['HB030'].fillna(0).astype('int32').astype(str)
    overpop_df['HB030'] = overpop_df['HB030'].fillna(0).astype('int32').astype(str)
    household_df_weight['DB030'] = household_df_weight['DB030'].fillna(0).astype('int32').astype(str)

    # Merge all datasets
    merged_df = household_df.merge(
        household_df_weight, left_on=['HB010', 'HB020', 'HB030'], 
        right_on=['DB010', 'DB020', 'DB030'], how='left'
    )

    merged_df = merged_df.merge(
        decile_df[["HB010", "HB020", "HB030", "equi_disp_inc", "decile"]], 
        left_on=['HB010', 'HB020', 'HB030'], 
        right_on=['HB010', 'HB020', 'HB030'], how='left'
    )

    merged_df = merged_df.merge(
        overpop_df[["HB010", "HB020", "HB030", "overcrowded"]], 
        left_on=['HB010', 'HB020', 'HB030'], 
        right_on=['HB010', 'HB020', 'HB030'], how='left'
    )

    # Save merged dataset
    merged_df.to_csv(os.path.join(dirs['final_merged_dir'], "EU_SILC_household_final_merged.csv"), index=False)

    # Define variable filters for indicators
    df = merged_df
    
    variable_filters = {
        "HS050": [1],                          # AN-SILC-1
        "HD080": [2, 3],                       # HQ-SILC-2
        "HS011": lambda row: [1] if row["HB010"] < 2008 else [1, 2],  # HH-SILC-1
        "HH050": [2],                          # HE-SILC-1
        "HS021": lambda row: [1] if row["HB010"] < 2008 else [1, 2],  # HE-SILC-2
        "HS060": [2],                          # ES-SILC-1
        "HS120": [1, 2],                       # ES-SILC-2
        "HS040": [2],                          # TS-SILC-1
        "overcrowded": [1],                   # HQ-SILC-1
    }

    # Precompute masks per row
    for var, condition in variable_filters.items():
        if callable(condition):
            df[f"_valid_{var}"] = df.apply(
                lambda row: row[var] in condition(row) if pd.notnull(row[var]) else False, axis=1
            )
        else:
            df[f"_valid_{var}"] = df[var].isin(condition) & df[var].notna()

    # Validate decile coverage before processing  
    country_decile_coverage = df.groupby('HB020')['decile'].nunique()
    print(f"📋 Countries with complete decile coverage (10 deciles): {(country_decile_coverage == 10).sum()}")
    print(f"📋 Countries with partial decile coverage: {(country_decile_coverage < 10).sum()}")
    
    # Group by Year, Country, Decile and calculate indicators
    group_cols = ["HB010", "HB020", "decile"]
    results = []

    for group_keys, group in df.groupby(group_cols):
        group_result = dict(zip(group_cols, group_keys))
        total_weight = group["DB090"].sum()

        for var in variable_filters.keys():
            mask = group[f"_valid_{var}"]
            weighted_sum = group.loc[mask, "DB090"].sum()
            share = (weighted_sum / total_weight * 100) if total_weight > 0 else np.nan
            group_result[f"{var}_share"] = share

        results.append(group_result)
    
    # Also calculate indicators for total population per country (decile = "All")
    print("📊 Computing total population indicators (decile = 'All')...")
    total_group_cols = ["HB010", "HB020"]
    
    for group_keys, group in df.groupby(total_group_cols):
        group_result = dict(zip(total_group_cols, group_keys))
        group_result['decile'] = "All"
        total_weight = group["DB090"].sum()

        for var in variable_filters.keys():
            mask = group[f"_valid_{var}"]
            weighted_sum = group.loc[mask, "DB090"].sum()
            share = (weighted_sum / total_weight * 100) if total_weight > 0 else np.nan
            group_result[f"{var}_share"] = share

        results.append(group_result)

    summary_df = pd.DataFrame(results)

    # Rename columns according to EWBI naming convention
    rename_dict = {
        "HB010": "year", "HB020": "country", "HS050": "AN-SILC-1", "HD080": "HQ-SILC-2",
        "HS011": "HH-SILC-1", "HH050": "HE-SILC-1", "HS021": "HE-SILC-2",
        "HS060": "ES-SILC-1", "HS120": "ES-SILC-2", "HS040": "TS-SILC-1",
        "overcrowded": "HQ-SILC-1"
    }

    new_column_names = {}
    for col in summary_df.columns:
        if col.endswith("_share"):
            var = col.replace("_share", "")
            if var in rename_dict:
                new_column_names[col] = rename_dict[var]
        elif col in rename_dict:
            new_column_names[col] = rename_dict[col]

    summary_df2 = summary_df.rename(columns=new_column_names)

    # Melt to long format
    columns_to_melt = [
        "AN-SILC-1", "HQ-SILC-2", "HH-SILC-1", "HE-SILC-1", "HE-SILC-2",
        "ES-SILC-1", "ES-SILC-2", "TS-SILC-1", "HQ-SILC-1"
    ]

    df_melted = summary_df2.melt(
        id_vars=["year", "country", "decile"],
        value_vars=columns_to_melt,
        var_name="primary_index",
        value_name="value"
    )

    df_melted["database"] = "EU-SILC"

    # Save final household indicators
    df_melted.to_csv(os.path.join(dirs['final_merged_dir'], "EU_SILC_household_final_summary.csv"), index=False)

    return df_melted


def process_personal_indicators(dirs):
    """
    Process and calculate personal-level indicators from EU-SILC data.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Final personal indicators dataset
    """
    print("Processing personal indicators...")
    
    # Load datasets
    cols_needed_PD = [
        "PB010", "PB020", "PB030", "PW010", "PW191", "PD060", "PD070", 
        "PH010", "PH020", "PH030", "PL086", "PH040", "PH050", "PE041", 
        "PL141", "PL145", "PL080"
    ]

    cols_needed_PR = [
        "RB010", "RB020", "RB030", "RB050", "RB081", "RB082", "RL010", "RL020"
    ]

    pop_df = pd.read_csv(
        os.path.join(dirs['merged_dir'], "EU_SILC_combined_personal_data.csv"),
        usecols=cols_needed_PD
    )

    register_df = pd.read_csv(
        os.path.join(dirs['merged_dir'], "EU_SILC_combined_personal_register.csv"),
        usecols=cols_needed_PR
    )

    decile_df = pd.read_csv(
        os.path.join(dirs['decile_dir'], "EU_SILC_household_data_with_decile.csv")
    )[["HB010", "HB020", "HB030", "equi_disp_inc", "decile"]]

    # Convert merge keys to string
    pop_df['PB030'] = pop_df['PB030'].fillna(0).astype('int32').astype(str)
    decile_df['HB030'] = decile_df['HB030'].fillna(0).astype('int32').astype(str)
    register_df['RB030'] = register_df['RB030'].fillna(0).astype('int32').astype(str)

    # Extract household IDs
    pop_df["HB030"] = pop_df["PB030"].str[:-2]
    register_df["HB030"] = register_df["RB030"].str[:-2]

    # Merge datasets
    merged_df = pop_df.merge(
        decile_df, left_on=['PB010', 'PB020', 'HB030'], 
        right_on=['HB010', 'HB020', 'HB030'], how='left'
    )

    merged_df = merged_df.merge(
        register_df, left_on=['PB010', 'PB020', 'PB030'], 
        right_on=['RB010', 'RB020', 'RB030'], how='left'
    )

    # Create a unified age column:
    # 1. Try RB081 (age from register) if available
    # 2. Fall back to RB082 (age of person interviewed) if available  
    # 3. Calculate from PB140 (birth year) as final fallback
    merged_df['age'] = merged_df['RB081'].fillna(merged_df['RB082'])
    
    # For records still missing age, calculate from birth year (PB140)
    mask_missing_age = merged_df['age'].isna()
    if 'PB140' in merged_df.columns and mask_missing_age.sum() > 0:
        merged_df.loc[mask_missing_age, 'age'] = merged_df.loc[mask_missing_age, 'PB010'] - merged_df.loc[mask_missing_age, 'PB140']
    
    # Save merged dataset
    merged_df.to_csv(os.path.join(dirs['final_merged_dir'], "EU_SILC_personal_final_merged.csv"), index=False)

    # Define condition rules for personal indicators
    def make_conditions(df):
        return {
            "PW010": df["PW010"] < 4,                                     # EL-SILC-1
            "PW191": df["PW191"] < 4,                                     # EC-SILC-2
            "PD060": df["PD060"].isin([2, 3]),                            # IC-SILC-1
            "PD070": df["PD070"].isin([2, 3]),                            # IC-SILC-2
            "PH010": df["PH010"].isin([4, 5]),                            # AH-SILC-1
            "PH020": df["PH020"] == 1,                                    # AH-SILC-2
            "PH030": df["PH030"].isin([1, 2]),                            # AH-SILC-3
            "PL086": df["PL086"] < 7,                                     # AH-SILC-4
            "PH040": df["PH040"] == 2,                                    # AC-SILC-2
            "PH050": df["PH050"] == 1,                                    # AC-SILC-1
            "PE041": ((df["age"] > 15) & ((df["PE041"] == 0) | df["PE041"].isna())),  # IS-SILC-3
            "PL141": ((df["age"] > 17) & (
                        ((df["PB010"] < 2021) & (df["PL141"] == 2)) |
                        ((df["PB010"] >= 2021) & df["PL141"].isin([11, 12]))
                      )),                                                # RT-SILC-1
            "PL145": ((df["age"] > 17) & (df["PL145"] == 2)),            # RT-SILC-2
            "PL080": (df["PL080"] > 5),                                  # RU-SILC-1
            "RL010": ((df["RL010"] == 0) | df["RL010"].isna()),  # IS-SILC-1 (whole population)
            "RL020": ((df["RL020"] == 0) | df["RL020"].isna()),  # IS-SILC-2 (whole population)
        }

    # Apply conditions
    conditions = make_conditions(merged_df)
    for var, cond in conditions.items():
        merged_df[f"_valid_{var}"] = cond & merged_df[var].notna()

    # Group by Year, Country, Decile and calculate indicators
    group_cols = ["PB010", "PB020", "decile"]
    results = []

    for group_keys, group in merged_df.groupby(group_cols):
        group_result = dict(zip(group_cols, group_keys))
        total_weight = group["RB050"].sum()

        for var in conditions.keys():
            mask = group[f"_valid_{var}"]
            
            # FIXED: Check if this variable has any data for this group
            has_any_data = group[var].notna().any()
            
            if not has_any_data:
                # Variable not collected in this year/country/decile - use NaN
                group_result[f"{var}_share"] = np.nan
            else:
                # Variable was collected, calculate share normally
                valid_count = mask.sum()
                if valid_count == 0:
                    # All responses were invalid/negative - this is a true zero
                    group_result[f"{var}_share"] = 0.0
                else:
                    # Calculate weighted share of positive responses
                    weighted_sum = group.loc[mask, "RB050"].sum()
                    share = (weighted_sum / total_weight * 100) if total_weight > 0 else np.nan
                    group_result[f"{var}_share"] = share

        results.append(group_result)
    
    # Also calculate indicators for total population per country (decile = "All")
    print("📊 Computing total population indicators (decile = 'All')...")
    total_group_cols = ["PB010", "PB020"]
    
    for group_keys, group in merged_df.groupby(total_group_cols):
        group_result = dict(zip(total_group_cols, group_keys))
        group_result['decile'] = "All"
        total_weight = group["RB050"].sum()

        for var in conditions.keys():
            mask = group[f"_valid_{var}"]
            
            # FIXED: Check if this variable has any data for this group
            has_any_data = group[var].notna().any()
            
            if not has_any_data:
                # Variable not collected in this year/country - use NaN
                group_result[f"{var}_share"] = np.nan
            else:
                # Variable was collected, calculate share normally
                valid_count = mask.sum()
                if valid_count == 0:
                    # All responses were invalid/negative - this is a true zero
                    group_result[f"{var}_share"] = 0.0
                else:
                    # Calculate weighted share of positive responses
                    weighted_sum = group.loc[mask, "RB050"].sum()
                    share = (weighted_sum / total_weight * 100) if total_weight > 0 else np.nan
                    group_result[f"{var}_share"] = share

        results.append(group_result)

    summary_df2 = pd.DataFrame(results)

    # Rename columns according to EWBI naming convention
    rename_dict = {
        "PB010": "year", "PB020": "country", "RB050": "Weight",
        "PW010": "EL-SILC-1", "PW191": "EC-SILC-2", "PD060": "IC-SILC-1", "PD070": "IC-SILC-2",
        "PH010": "AH-SILC-1", "PH020": "AH-SILC-2", "PH030": "AH-SILC-3", "PL086": "AH-SILC-4",
        "PH040": "AC-SILC-2", "PH050": "AC-SILC-1", "PE041": "IS-SILC-3",
        "PL141": "RT-SILC-1", "PL145": "RT-SILC-2", "PL080": "RU-SILC-1",
        "RL010": "IS-SILC-1", "RL020": "IS-SILC-2"
    }

    new_column_names = {}
    for col in summary_df2.columns:
        if col.endswith("_share"):
            var = col.replace("_share", "")
            if var in rename_dict:
                new_column_names[col] = rename_dict[var]
        elif col in rename_dict:
            new_column_names[col] = rename_dict[col]

    summary_df2 = summary_df2.rename(columns=new_column_names)

    # Melt to long format
    columns_to_melt = [
        "EL-SILC-1", "EC-SILC-2", "IC-SILC-1", "IC-SILC-2", "AH-SILC-1", "AH-SILC-2",
        "AH-SILC-3", "AH-SILC-4", "AC-SILC-2", "AC-SILC-1",
        "IS-SILC-1", "IS-SILC-2", "IS-SILC-3",
        "RT-SILC-1", "RT-SILC-2", "RU-SILC-1"
    ]

    df_melted = summary_df2.melt(
        id_vars=["year", "country", "decile"],
        value_vars=columns_to_melt,
        var_name="primary_index",
        value_name="value"
    )

    df_melted["database"] = "EU-SILC"

    # Save final personal indicators
    df_melted.to_csv(os.path.join(dirs['final_merged_dir'], "EU_SILC_personal_final_summary.csv"), index=False)

    return df_melted


def main():
    """Main function to execute the complete EU-SILC data processing pipeline."""
    print("Starting EU-SILC data processing...")
    
    # Setup directories
    dirs = setup_directories()
    print(f"External data directory: {dirs['external_data']}")
    print(f"Output directory: {dirs['output_base']}")
    
    # Combine all raw data files
    print("Combining raw data files...")
    combine_household_data(dirs)
    combine_household_register(dirs)
    combine_personal_register(dirs)
    combine_personal_data(dirs)
    
    # Calculate income deciles
    calculate_income_deciles(dirs)
    
    # Calculate overcrowding indicators
    calculate_overcrowding(dirs)
    
    # Process household indicators
    household_df = process_household_indicators(dirs)
    
    # Process personal indicators
    personal_df = process_personal_indicators(dirs)
    
    print("EU-SILC processing completed successfully!")
    print(f"Household indicators shape: {household_df.shape}")
    print(f"Personal indicators shape: {personal_df.shape}")
    print(f"Data saved to: {dirs['final_dir']}")


if __name__ == "__main__":
    main()