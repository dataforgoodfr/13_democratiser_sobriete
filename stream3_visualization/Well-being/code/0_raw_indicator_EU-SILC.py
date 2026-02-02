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


def robust_read_csv(file_path, file_name, cols_needed, **kwargs):
    """
    Robust CSV reading with multiple encoding attempts and comprehensive error handling.
    
    Args:
        file_path (str): Path to the CSV file
        file_name (str): Name of the file for logging
        cols_needed (list): List of required columns
        **kwargs: Additional arguments for pd.read_csv
    
    Returns:
        pd.DataFrame or None: DataFrame if successful, None if failed
    """
    try:
        # Try reading with default encoding first
        sample_df = pd.read_csv(file_path, sep=",", nrows=5, **kwargs)
        available_cols = sample_df.columns.tolist()

        # Determine which needed columns exist in this file
        cols_present = [col for col in cols_needed if col in available_cols]

        if not cols_present:
            print(f"⚠️ No needed columns found in {file_name}, skipping.")
            return None

        # Read the full file using only available columns
        df = pd.read_csv(file_path, sep=",", usecols=cols_present, **kwargs)

        # Add missing columns as NaNs
        for col in cols_needed:
            if col not in df.columns:
                df[col] = pd.NA

        print(f"✅ Successfully processed {file_name} ({len(df):,} records)")
        return df

    except UnicodeDecodeError as e:
        print(f"⚠️ Encoding error in {file_name}, trying different encodings: {e}")
        # Try alternative encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                sample_df = pd.read_csv(file_path, sep=",", nrows=5, encoding=encoding, **kwargs)
                available_cols = sample_df.columns.tolist()
                cols_present = [col for col in cols_needed if col in available_cols]
                
                if not cols_present:
                    continue
                    
                df = pd.read_csv(file_path, sep=",", usecols=cols_present, encoding=encoding, **kwargs)
                
                for col in cols_needed:
                    if col not in df.columns:
                        df[col] = pd.NA
                
                print(f"✅ Successfully processed {file_name} with {encoding} encoding ({len(df):,} records)")
                return df
            except:
                continue
        
        print(f"❌ Could not read {file_name} with any encoding, skipping.")
        return None
        
    except PermissionError as e:
        print(f"❌ Permission error reading {file_name}: {e}")
        print("   File might be locked or in use. Skipping.")
        return None
        
    except Exception as e:
        print(f"❌ Error reading {file_name}: {e}")
        print(f"   File path: {file_path}")
        print("   Skipping this file and continuing with others.")
        return None


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
        "HS050", "HD080", "HS011", "HS021",
        "HS060", "HS120", "HS040",
        # New indicators for Energy and Housing (HQ-SILC-2 to HQ-SILC-8)
        "HC060", "HC070", "HS160", "HS170", "HH040", "HS180", "HC003"
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
                df = robust_read_csv(file_path, file_name, cols_needed_household)
                if df is not None:
                    df["Country"] = country
                    df["Year"] = f"20{year}"
                    dfs.append(df)
            else:
                print(f"⚠️ File not found: {file_name}")

    # Merge all data
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(dirs['merged_dir'], "EU_SILC_combined_household_data.csv")
        final_df.to_csv(output_path, index=False)
        print(f"✅ Household data combined: {final_df.shape} from {len(dfs)} files")
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
                df = robust_read_csv(file_path, file_name, cols_needed_household)
                if df is not None:
                    df["Country"] = country
                    df["Year"] = f"20{year}"
                    dfs.append(df)
            else:
                print(f"⚠️ File not found: {file_name}")

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
        "RL020",  # IS-SILC-2 - Education at compulsory school
        # New personal indicators
        "PE010",  # IS-SILC-4 - Participation in formal training (student/apprentice)
        "PE041",  # IS-SILC-5 - No secondary education
        "PH060",  # AC-SILC-3 - Unmet need for medical care  
        "PH040",  # AC-SILC-4 - Unmet need for dental examination and treatment
        "PD050"   # EC-SILC-3 - Get-together with friends or family
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
                df = robust_read_csv(file_path, file_name, cols_needed_personal)
                if df is not None:
                    df["Country"] = country
                    df["Year"] = f"20{year}"
                    dfs.append(df)
            else:
                print(f"⚠️ File not found: {file_name}")

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
        "PW191", "PD060", "PD070", "PH020", "PH030",
        "PL086", "PH040", "PH050", "PE041", "PL141", "PL145", "PL080",
        # New indicators for various EU priorities
        "PE010",    # IS-SILC-4 - Not participating in training
        "PH060",    # AC-SILC-4 - Unmet need for dental care
        "PD050"     # EC-SILC-3 - Get-together w/ friends/family
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
                df = robust_read_csv(file_path, file_name, cols_needed_personal)
                if df is not None:
                    df["Country"] = country
                    df["Year"] = f"20{year}"
                    dfs.append(df)
            else:
                print(f"⚠️ File not found: {file_name}")

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
    decile_df = equival_inc_df.groupby(['HB010', 'HB020']).apply(compute_deciles, include_groups=False).reset_index()
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

    # Convert both merge keys to string (use int64 to avoid overflow with large IDs)
    pp['PB030'] = pp['PB030'].fillna(0).astype('int64').astype(str)
    hh['HB030'] = hh['HB030'].fillna(0).astype('int64').astype(str)

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
    # Handle NaN values properly: if HH030 is NaN, overcrowded should be NaN
    overcrowded_condition = rooms_df['HH030'] < rooms_df['required_rooms']
    rooms_df['overcrowded'] = overcrowded_condition.where(rooms_df['HH030'].notna(), np.nan).astype('float')

    # Calculate person-level overcrowding indicator for accurate population shares
    print("Calculating person-level overcrowding indicator...")
    
    # Load personal register for weights
    cols_needed_register = ["RB010", "RB020", "RB030", "RB050"]
    pr = pd.read_csv(
        os.path.join(dirs['merged_dir'], "EU_SILC_combined_personal_register.csv"),
        usecols=cols_needed_register
    )
    
    # Extract household IDs (convert to string first in case they're numeric)
    pr['household_id'] = pr['RB030'].astype(str).str[:-2]
    
    # Calculate population weight (sum of personal weights) for each household
    household_population_weights = pr.groupby(['RB010', 'RB020', 'household_id'])['RB050'].sum().reset_index()
    household_population_weights.rename(columns={'RB050': 'population_weight'}, inplace=True)
    
    # Merge population weights with overcrowding data
    rooms_df = rooms_df.merge(household_population_weights,
                             left_on=['HB010', 'HB020', rooms_df['HB030'].astype(str).str[:-2]],
                             right_on=['RB010', 'RB020', 'household_id'], how='left')
    
    # Merge personal register with household overcrowding data for validation
    person_overcrowd = pr.merge(rooms_df, 
                               left_on=['RB010', 'RB020', 'household_id'],
                               right_on=['RB010', 'RB020', 'household_id'], 
                               how='left')
    
    # Calculate weighted person-level overcrowding for validation
    if not person_overcrowd.empty and 'RB050' in person_overcrowd.columns:
        valid_data = person_overcrowd.dropna(subset=['RB050', 'overcrowded'])
        if not valid_data.empty:
            total_weight = valid_data['RB050'].sum()
            overcrowded_weight = (valid_data['overcrowded'] * valid_data['RB050']).sum()
            person_overcrowd_pct = (overcrowded_weight / total_weight) * 100
            
            print(f"📊 Person-level overcrowding validation:")
            print(f"   Total persons: {len(valid_data):,}")
            print(f"   Persons in overcrowded households: {valid_data['overcrowded'].sum():,}")
            print(f"   Person-level overcrowding rate: {person_overcrowd_pct:.1f}%")

    print("Saving overcrowding results...")
    rooms_df.to_csv(os.path.join(dirs['overcrowd_dir'], "EU_SILC_household_data_with_overcrowding.csv"), index=False)
    print(f"Overcrowding calculation completed! Results saved for {len(rooms_df):,} households.")

    return rooms_df


def calculate_household_size(dirs):
    """
    Calculate household size and identify persons living alone for the "Persons living alone" indicator.
    Uses a PERSON-LEVEL approach with personal weights to get accurate population shares.
    
    This function calculates the share of POPULATION living in single-person households
    by counting individual persons in single-person households and weighting by personal weights.
    
    Args:
        dirs (dict): Dictionary containing directory paths
    
    Returns:
        pd.DataFrame: Dataset with household size and living alone indicator at household level
    """
    print("Calculating household size and living alone indicator...")
    print("Using PERSON-LEVEL approach with personal weights for accurate population shares...")
    
    # Load both household and personal data
    cols_needed_household = ["HB010", "HB020", "HB030"]
    cols_needed_personal = ["PB030", "PB010", "PB020"]
    cols_needed_register = ["RB010", "RB020", "RB030", "RB050"]  # Include personal weights

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
    
    print("Loading personal register for weights...")
    pr = pd.read_csv(
        os.path.join(dirs['merged_dir'], "EU_SILC_combined_personal_register.csv"),
        usecols=cols_needed_register
    )

    # Data type conversions
    pp['PB030'] = pp['PB030'].fillna(0).astype('int64').astype(str)
    pr['RB030'] = pr['RB030'].fillna(0).astype('int64').astype(str)
    hh['HB030'] = hh['HB030'].fillna(0).astype('int64').astype(str)

    # Extract household IDs (convert to string first in case they're numeric)
    pp['household_id'] = pp['PB030'].astype(str).str[:-2]
    pr['household_id'] = pr['RB030'].astype(str).str[:-2]
    
    # Merge personal data with register to get weights
    print("Merging personal data with register for weights...")
    personal_merged = pp.merge(pr, left_on=['PB010', 'PB020', 'PB030'], 
                               right_on=['RB010', 'RB020', 'RB030'], how='left')
    
    # Ensure household_id is preserved from the left side (pp)
    if 'household_id_x' in personal_merged.columns:
        personal_merged['household_id'] = personal_merged['household_id_x']
    elif 'household_id' not in personal_merged.columns:
        personal_merged['household_id'] = personal_merged['PB030'].astype(str).str[:-2]

    # Merge with household data
    data = personal_merged.merge(hh, left_on=['PB010', 'PB020', 'household_id'],
                        right_on=['HB010', 'HB020', 'HB030'])

    print(f"Merged data shape: {data.shape}")
    print(f"Years available: {sorted(data['HB010'].unique())}")
    print(f"Countries available: {sorted(data['HB020'].unique())}")
    
    # Count household sizes (persons per household)
    household_sizes = data.groupby(['HB010', 'HB020', 'HB030']).size().reset_index(name='household_size')
    
    # Merge household size back to person-level data
    data = data.merge(household_sizes, on=['HB010', 'HB020', 'HB030'])
    
    # Create living alone indicator at person level
    data['living_alone'] = (data['household_size'] == 1  # EWBI).astype(int)
    
    print(f"Total persons in data: {len(data):,}")
    print(f"Persons in single-person households: {data['living_alone'].sum():,}")
    
    # Calculate population-weighted shares by country/year for validation
    print("\n📊 Sample validation - France 2022 person-level calculation:")
    france_2022 = data[(data['HB020'] == 'FR') & (data['HB010'] == 2  # EU Priorities022) & (data['RB050'].notna())]
    if len(france_2022) > 0:
        total_weight = france_2022['RB050'].sum()
        living_alone_weight = (france_2022['living_alone'] * france_2022['RB050']).sum()
        france_share = (living_alone_weight / total_weight) * 100
        print(f"   France 2022: {france_share:.1f}% (using person weights)")
        print(f"   INSEE reference: 21.7%")
        print(f"   Difference: {abs(france_share - 21.7):.1f} percentage points")
    
    # Create household-level summary for consistency with other indicators
    # This aggregates to household level for the indicator processing pipeline
    household_summary = data.groupby(['HB010', 'HB020', 'HB030']).agg({
        'household_size': 'first',
        'living_alone': 'first',
        'RB050': 'sum'  # Sum personal weights = household population weight
    }).reset_index()
    
    # Rename for consistency
    household_summary.rename(columns={'RB050': 'population_weight'}, inplace=True)
    
    # Save the results
    household_summary.to_csv(os.path.join(dirs['final_merged_dir'], "EU_SILC_household_size.csv"), index=False)
    
    print(f"Household size calculation completed! Results saved for {len(household_summary):,} households.")
    print(f"Single-person households: {household_summary['living_alone'].sum():,} ({household_summary['living_alone'].mean()*100:.1f}%)")
    print(f"Years range: {household_summary['HB010'].min()}-{household_summary['HB010'].max()}")
    print(f"Countries: {len(household_summary['HB020'].unique())} unique countries")
    
    return household_summary


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
        "HB010", "HB020", "HB030", "HS050", "HD080", "HS011", 
        "HS021", "HS060", "HS120", "HS040",
        # New indicators
        "HC060", "HC070", "HS160", "HS170", "HH040", "HS180", "HC003"
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
    
    household_size_df = pd.read_csv(
        os.path.join(dirs['final_merged_dir'], "EU_SILC_household_size.csv")
    )

    # Convert merge keys to string (use int64 to avoid overflow with large IDs)
    household_df['HB030'] = household_df['HB030'].fillna(0).astype('int64').astype(str)
    decile_df['HB030'] = decile_df['HB030'].fillna(0).astype('int64').astype(str)
    overpop_df['HB030'] = overpop_df['HB030'].fillna(0).astype('int64').astype(str)
    # household_size_df['HB030'] is already string from calculate_household_size function
    household_size_df['HB030'] = household_size_df['HB030'].fillna('0').astype(str)
    household_df_weight['DB030'] = household_df_weight['DB030'].fillna(0).astype('int64').astype(str)

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
        overpop_df[["HB010", "HB020", "HB030", "overcrowded", "population_weight"]], 
        left_on=['HB010', 'HB020', 'HB030'], 
        right_on=['HB010', 'HB020', 'HB030'], how='left', suffixes=('', '_overcrowd')
    )
    
    merged_df = merged_df.merge(
        household_size_df[["HB010", "HB020", "HB030", "living_alone", "population_weight"]], 
        left_on=['HB010', 'HB020', 'HB030'], 
        right_on=['HB010', 'HB020', 'HB030'], how='left', suffixes=('', '_living_alone')
    )

    # Save merged dataset
    merged_df.to_csv(os.path.join(dirs['final_merged_dir'], "EU_SILC_household_final_merged.csv"), index=False)

    # Define variable filters for indicators (existing + new)
    df = merged_df
    
    variable_filters = {
        # Existing indicators
        "HS050": [1],                          # AN-SILC-1
        "HS011": lambda row: [1] if row["HB010"] < 2008 else [1, 2],  # HH-SILC-1
        "HS021": lambda row: [1] if row["HB010"] < 2008 else [1, 2],  # HE-SILC-2
        "HS060": [2],                          # ES-SILC-1
        "HS120": [1, 2],                       # ES-SILC-2
        "HS040": [2],                          # TS-SILC-1
        "overcrowded": [1],                    # HQ-SILC-1
        
        # New Energy and Housing indicators
        "HC060": [2],                          # HQ-SILC-2 - Cannot keep dwelling comfortably warm
        "HC070": [2],                          # HQ-SILC-3 - Cannot keep dwelling comfortably cool
        "HS160": [1],                          # HQ-SILC-4 - Dwelling too dark
        "HS170": [1],                          # HQ-SILC-5 - Noise from street
        "HH040": [1],                          # HQ-SILC-6 - Leaking roof / damp / rot
        "HS180": [1],                          # HQ-SILC-7 - Pollution or crime
        "HC003": [4, 99],                     # HQ-SILC-8 - No renovation measures
        "living_alone": [1],                   # EC-SILC-4 - Persons living alone
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
    print(f"📋 Countries with complete decile coverage (10 deciles): {(country_decile_coverage == 1  # EWBI0).sum()}")
    print(f"📋 Countries with partial decile coverage: {(country_decile_coverage < 10).sum()}")
    
    def calculate_overcrowding_percentage(group, min_population_weight_coverage=0.5):
        """
        Calculate overcrowding percentage with fallback to household weights when population weights are insufficient.
        
        Args:
            group: DataFrame group (by country/year/decile or country/year for total)
            min_population_weight_coverage: Minimum coverage threshold for using population weights
            
        Returns:
            float: Overcrowding percentage or NaN if overcrowding data is missing
        """
        # Check if overcrowding data is available at all for this group
        overcrowded_available = group['overcrowded'].notna().sum()
        total_households = len(group)
        
        # If no overcrowding data is available, return NaN (missing data)
        if overcrowded_available == 0:
            return np.nan
        
        mask = group[f"_valid_overcrowded"]
        overcrowded_households = group.loc[mask]
        
        # If we have overcrowding data but no households are overcrowded, return 0.0
        if len(overcrowded_households) == 0:
            return 0.0
        
        # Check population weight coverage (non-zero values)
        total_households = len(group)
        households_with_pop_weights = (group['population_weight'] > 0).sum()
        pop_weight_coverage = households_with_pop_weights / total_households if total_households > 0 else 0
        
        if pop_weight_coverage >= min_population_weight_coverage:
            # Use population weights (person-level calculation)
            population_weight_col = 'population_weight'
            overcrowded_population = overcrowded_households[population_weight_col].sum()
            total_population = group[population_weight_col].sum()
            percentage = (overcrowded_population / total_population * 100) if total_population > 0 else 0.0
        else:
            # Fall back to household weights (household-level calculation)
            overcrowded_weight = overcrowded_households['DB090'].sum()
            total_weight = group['DB090'].sum()
            percentage = (overcrowded_weight / total_weight * 100) if total_weight > 0 else 0.0
        
        return percentage
    
    # Group by Year, Country, Decile and calculate indicators
    group_cols = ["HB010", "HB020", "decile"]
    results = []

    for group_keys, group in df.groupby(group_cols):
        group_result = dict(zip(group_cols, group_keys))
        total_weight = group["DB090"].sum()

        for var in variable_filters.keys():
            # Check if all values are NaN for this variable in this group
            all_nan = group[var].isna().all()
            
            # Specific exclusion: Skip HQ-SILC-2 (HC060) for 2016 due to data quality issues
            year = group_keys[0]  # HB010 is the first element (year)
            if var == "HC060" and year == 2  # EU Priorities016:
                share = np.nan
            elif all_nan:
                # If all values are NaN, the indicator should be NaN
                share = np.nan
            else:
                # SPECIAL HANDLING FOR EC-SILC-4 (living_alone)
                # This should use person-level weights for accurate population representation
                if var == "living_alone":
                    # Use population_weight (sum of personal weights) instead of household weights
                    mask = group[f"_valid_{var}"]
                    single_person_households = group.loc[mask]
                    
                    if len(single_person_households) == 0:
                        share = 0.0
                    else:
                        # For single-person households, population_weight = personal weight of that person
                        # Use the correct population weight column (may have suffix from living alone merge)
                        if 'population_weight_living_alone' in group.columns:
                            population_weight_col = 'population_weight_living_alone'
                        elif 'population_weight' in group.columns:
                            population_weight_col = 'population_weight'
                        else:
                            print(f"⚠️ Warning: No population weight column found for living alone")
                            share = np.nan
                            continue
                            
                        living_alone_population = single_person_households[population_weight_col].sum()
                        # Total population is sum of all population weights
                        total_population = group[population_weight_col].sum()
                        share = (living_alone_population / total_population * 100) if total_population > 0 else np.nan
                
                elif var == "overcrowded":
                    # SPECIAL HANDLING FOR HQ-SILC-1 (overcrowded) - Fixed version with fallback
                    share = calculate_overcrowding_percentage(group)
                
                else:
                    # Normal calculation for other indicators using household weights
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
            # Check if all values are NaN for this variable in this group
            all_nan = group[var].isna().all()
            
            # Calculate data coverage for this variable in this group
            total_records = len(group)
            available_records = group[var].notna().sum()
            coverage_pct = (available_records / total_records * 100) if total_records > 0 else 0
            
            # Set minimum coverage threshold for reliable indicators (10%)
            min_coverage_threshold = 10.0
            
            # SPECIAL HANDLING FOR EC-SILC-4 (living_alone)
            # This should use person-level weights for accurate population representation
            if var == "living_alone":
                # Use population_weight (sum of personal weights) instead of household weights
                mask = group[f"_valid_{var}"]
                single_person_households = group.loc[mask]
                
                if len(single_person_households) == 0:
                    share = 0.0
                else:
                    # For single-person households, population_weight = personal weight of that person
                    # Use the correct population weight column (may have suffix)
                    if 'population_weight_living_alone' in group.columns:
                        population_weight_col = 'population_weight_living_alone'
                    elif 'population_weight' in group.columns:
                        population_weight_col = 'population_weight'
                    else:
                        print(f"⚠️ Warning: No population weight column found for living alone in total calculation")
                        share = np.nan
                        continue
                        
                    living_alone_population = single_person_households[population_weight_col].sum()
                    # Total population is sum of all population weights
                    total_population = group[population_weight_col].sum()
                    share = (living_alone_population / total_population * 100) if total_population > 0 else np.nan
                
            elif var == "overcrowded":
                # SPECIAL HANDLING FOR HQ-SILC-1 (overcrowded) - Fixed version with fallback
                share = calculate_overcrowding_percentage(group)
            else:
                # Normal calculation for other indicators
                if all_nan or coverage_pct < min_coverage_threshold:
                    # If all values are NaN or coverage is too low, the indicator should be NaN
                    share = np.nan
                else:
                    # Normal calculation when we have sufficient data coverage
                    mask = group[f"_valid_{var}"]
                    weighted_sum = group.loc[mask, "DB090"].sum()
                    share = (weighted_sum / total_weight * 100) if total_weight > 0 else np.nan
            
            group_result[f"{var}_share"] = share

        results.append(group_result)

    summary_df = pd.DataFrame(results)

    # Rename columns according to EWBI naming convention (existing + new)
    rename_dict = {
        "HB010": "year", "HB020": "country", 
        # Existing indicators
        "HS050": "AN-SILC-1",
        "HS011": "HH-SILC-1", "HS021": "HE-SILC-2",
        "HS060": "ES-SILC-1", "HS120": "ES-SILC-2", "HS040": "TS-SILC-1",
        "overcrowded": "HQ-SILC-1",
        # New Energy and Housing indicators as specified by user
        "HC060": "HQ-SILC-2",    # Keep dwelling comfortably warm
        "HC070": "HQ-SILC-3",    # Keep dwelling comfortably cool
        "HS160": "HQ-SILC-4",    # Dwelling too dark
        "HS170": "HQ-SILC-5",    # Noise from street
        "HH040": "HQ-SILC-6",    # Leaking roof / damp / rot
        "HS180": "HQ-SILC-7",    # Pollution or crime
        "HC003": "HQ-SILC-8",    # No renovation measures
        "living_alone": "EC-SILC-4"  # Persons living alone
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

    # Melt to long format (existing + new indicators)
    columns_to_melt = [
        # Existing indicators
        "AN-SILC-1", "HH-SILC-1", "HE-SILC-2",
        "ES-SILC-1", "ES-SILC-2", "TS-SILC-1", "HQ-SILC-1",
        # New Energy and Housing indicators
        "HQ-SILC-2", "HQ-SILC-3", "HQ-SILC-4", "HQ-SILC-5", 
        "HQ-SILC-6", "HQ-SILC-7", "HQ-SILC-8",
        # New Social indicator
        "EC-SILC-4"
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
        "PB010", "PB020", "PB030", "PW191", "PD060", "PD070", 
        "PH020", "PH030", "PL086", "PH040", "PH050", "PE041", 
        "PL141", "PL145", "PL080",
        # New indicators
        "PE010", "PH060", "PD050"
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

    # Convert merge keys to string (use int64 to avoid overflow with large IDs)
    pop_df['PB030'] = pop_df['PB030'].fillna(0).astype('int64').astype(str)
    decile_df['HB030'] = decile_df['HB030'].fillna(0).astype('int64').astype(str)
    register_df['RB030'] = register_df['RB030'].fillna(0).astype('int64').astype(str)

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

    # Define condition rules for personal indicators (existing + new)
    def make_conditions(df):
        return {
            # Existing indicators
            "PW191": df["PW191"] < 4,                                     # EC-SILC-2
            "PD060": df["PD060"].isin([2, 3]),                            # IC-SILC-1
            "PD070": df["PD070"].isin([2, 3]),                            # IC-SILC-2
            "PH020": df["PH020"] == 1  # EWBI,                                    # AH-SILC-2
            "PH030": df["PH030"].isin([1, 2]),                            # AH-SILC-3
            "PL086": df["PL086"] > 0,                                     # AH-SILC-4
            "PH050": df["PH050"] == 1  # EWBI,                                    # AC-SILC-1
            "PE041": ((df["age"] > 15) & ((df["PE041"] == 0) | df["PE041"].isna())),  # IS-SILC-3
            "PL141": ((df["age"] > 17) & (
                        ((df["PB010"] < 2021) & (df["PL141"] == 2  # EU Priorities)) |
                        ((df["PB010"] >= 2021) & df["PL141"].isin([11, 12]))
                      )),                                                # RT-SILC-1
            "PL145": ((df["age"] > 17) & (df["PL145"] == 2  # EU Priorities)),            # RT-SILC-2
            "PL080": (df["PL080"] > 5),                                  # RU-SILC-1
            "RL010": ((df["RL010"] == 0) | df["RL010"].isna()),         # IS-SILC-1 (whole population)
            "RL020": ((df["RL020"] == 0) | df["RL020"].isna()),         # IS-SILC-2 (whole population)
            
            # New indicators
            "PE010": df["PE010"] == 2  # EU Priorities,                                   # IS-SILC-4 - Not participating in training
            "PE041_new": ((df["PE041"].isin(['000', '100', 0, 100]))),  # IS-SILC-5 - No secondary education
            "PH060": df["PH060"] == 1  # EWBI,                                   # AC-SILC-3 - Unmet need for medical care
            "PH040": df["PH040"] == 1  # EWBI,                                   # AC-SILC-4 - Unmet need for dental care
            "PD050": df["PD050"].isin([2, 3]),                          # EC-SILC-3 - Get-together w/ friends/family
        }

    # Define mapping from indicator names to source column names
    def get_source_column_mapping():
        """Maps indicator names to their source column names."""
        return {
            # Most indicators use their own name as column
            "PW191": "PW191", 
            "PD060": "PD060",
            "PD070": "PD070",
            "PH020": "PH020",
            "PH030": "PH030",
            "PL086": "PL086",
            "PH040": "PH040",
            "PH050": "PH050",
            "PE041": "PE041",
            "PL141": "PL141",
            "PL145": "PL145",
            "PL080": "PL080",
            "RL010": "RL010",
            "RL020": "RL020",
            "PE010": "PE010",
            "PH060": "PH060", 
            "PH040": "PH040",
            "PD050": "PD050",
            
            # Special mappings for renamed indicators
            "PE041_new": "PE041",  # PE041_new indicator uses PE041 column
        }

    # Apply conditions
    conditions = make_conditions(merged_df)
    source_mapping = get_source_column_mapping()
    for var, cond in conditions.items():
        # Use the source column mapping instead of string replacement
        source_col = source_mapping.get(var, var)
        merged_df[f"_valid_{var}"] = cond & merged_df[source_col].notna()

    # Group by Year, Country, Decile and calculate indicators
    group_cols = ["PB010", "PB020", "decile"]
    results = []

    for group_keys, group in merged_df.groupby(group_cols):
        group_result = dict(zip(group_cols, group_keys))
        total_weight = group["RB050"].sum()

        for var in conditions.keys():
            source_col = source_mapping.get(var, var)  # Use mapping instead of string replacement
            mask = group[f"_valid_{var}"]
            
            # FIXED: Check if this variable has any data for this group
            has_any_data = group[source_col].notna().any()
            
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
            var_col = var.replace('_new', '')  # Handle the PE041_new case
            mask = group[f"_valid_{var}"]
            
            # FIXED: Check if this variable has any data for this group
            has_any_data = group[var_col].notna().any()
            
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

    # Rename columns according to EWBI naming convention (existing + new)
    rename_dict = {
        "PB010": "year", "PB020": "country", "RB050": "Weight",
        # Existing indicators
        "PW191": "EC-SILC-2", "PD060": "IC-SILC-1", "PD070": "IC-SILC-2",
        "PH020": "AH-SILC-2", "PH030": "AH-SILC-3", "PL086": "AH-SILC-4",
        "PH040": "AC-SILC-4", "PH050": "AC-SILC-1", "PE041": "IS-SILC-3",
        "PL141": "RT-SILC-1", "PL145": "RT-SILC-2", "PL080": "RU-SILC-1",
        "RL010": "IS-SILC-1", "RL020": "IS-SILC-2",
        # New indicators
        "PE010": "IS-SILC-4", "PE041_new": "IS-SILC-5", "PH060": "AC-SILC-3", 
        "PD050": "EC-SILC-3"
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

    # Melt to long format (existing + new indicators)
    columns_to_melt = [
        # Existing indicators
        "EC-SILC-2", "IC-SILC-1", "IC-SILC-2", "AH-SILC-2",
        "AH-SILC-3", "AH-SILC-4", "AC-SILC-1",
        "IS-SILC-1", "IS-SILC-2", "IS-SILC-3",
        "RT-SILC-1", "RT-SILC-2", "RU-SILC-1",
        # New indicators
        "IS-SILC-4", "IS-SILC-5", "AC-SILC-3", "AC-SILC-4", "EC-SILC-3"
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


def geometric_mean_robust(values):
    """
    Compute geometric mean while properly handling NaN values.
    NaN values are excluded from both numerator and denominator.
    """
    # Remove NaN values
    clean_values = values.dropna()
    
    if len(clean_values) == 0:
        return np.nan
    
    # Ensure all values are positive for geometric mean
    if (clean_values <= 0).any():
        return np.nan
    
    return np.exp(np.log(clean_values).mean())


def arithmetic_mean_robust(values):
    """
    Compute arithmetic mean while properly handling NaN values.
    NaN values are excluded from both numerator and denominator.
    """
    return values.mean()  # pandas mean() already excludes NaN values


# Update the aggregation functions to use robust mean calculations
def aggregate_to_country_level(df, indicator_col='indicator_value', groupby_cols=['country', 'year']):
    """
    Aggregate indicator values to country level using geometric mean.
    Properly handles NaN values by excluding them from computation.
    """
    def safe_geometric_mean(series):
        return geometric_mean_robust(series)
    
    return df.groupby(groupby_cols)[indicator_col].agg(safe_geometric_mean).reset_index()


def aggregate_hierarchical(df, level_col='level', value_col='value', groupby_cols=['country', 'year']):
    """
    Aggregate indicators hierarchically using arithmetic mean.
    Properly handles NaN values by excluding them from computation.
    """
    def safe_arithmetic_mean(series):
        return arithmetic_mean_robust(series)
    
    return df.groupby(groupby_cols + [level_col])[value_col].agg(safe_arithmetic_mean).reset_index()


def compute_level5_indicators(df):
    """
    Compute Level 5 indicators with proper NaN handling.
    When computing aggregations, NaN values are excluded from both numerator and denominator.
    """
    import numpy as np
    import pandas as pd
    
    # Ensure we're working with a copy
    df_work = df.copy()
    
    # Group by relevant columns for Level 5 computation
    groupby_cols = ['country', 'year', 'level5_indicator_name']  # Adjust based on actual column names
    
    # Apply geometric mean for individual-level to aggregate-level computation
    def robust_geometric_mean(series):
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return np.nan
        
        # Check for non-positive values
        if (clean_series <= 0).any():
            return np.nan
        
        # Compute geometric mean
        return np.exp(np.log(clean_series).mean())
    
    # Apply the robust aggregation
    level5_aggregated = df_work.groupby(groupby_cols)['indicator_value'].agg(robust_geometric_mean).reset_index()
    
    return level5_aggregated


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
    
    # Calculate household size and living alone indicator
    calculate_household_size(dirs)
    
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
