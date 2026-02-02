"""
HBS Data Loader Module

This module provides core data loading and processing functions for HBS analysis.
It handles:
- Directory setup
- PPS (Purchasing Power Parities) data loading from Eurostat
- HBS data loading from external directory
- PPS conversion of consumption values
- Income decile assignment
- Data aggregation by decile

Author: Data for Good - Well-being Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import os
import glob


def get_country_name(country_code):
    """
    Get the full country name from ISO country code.
    
    Args:
        country_code (str): ISO 2-letter country code (e.g., 'FR', 'DE')
        
    Returns:
        str: Full country name
    """
    country_names = {
        'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus', 'CZ': 'Czechia',
        'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia', 'ES': 'Spain', 'FI': 'Finland',
        'FR': 'France', 'GR': 'Greece', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland',
        'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta',
        'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'SE': 'Sweden',
        'SI': 'Slovenia', 'SK': 'Slovakia', 'CH': 'Switzerland'
    }
    return country_names.get(country_code, country_code)


def setup_directories():
    """
    Set up directories for data access.
    
    Returns:
        dict: Dictionary with paths to script, report, outputs, and external data directories
    """
    # Get the absolute path to the current script directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to script location
    dirs = {
        'script_dir': SCRIPT_DIR,
        'report_base': os.path.dirname(SCRIPT_DIR),
        'outputs': os.path.join(os.path.dirname(SCRIPT_DIR), 'outputs'),
        'external_data': os.path.join(os.path.dirname(SCRIPT_DIR), 'external_data'),
    }
    
    # Create output directories if they don't exist
    for key, dir_path in dirs.items():
        if key not in ['script_dir', 'report_base']:
            os.makedirs(dir_path, exist_ok=True)
    
    print(f"Directories set up:")
    print(f"  Script directory: {dirs['script_dir']}")
    print(f"  Output directory: {dirs['outputs']}")
    
    return dirs


def load_pps_data(dirs):
    """
    Load PPS (Purchasing Power Parities) data from the shared external datasets.
    
    PPS factors are used to convert nominal consumption values to comparable 
    purchasing power standards across countries.
    
    Args:
        dirs (dict): Directory paths from setup_directories()
        
    Returns:
        pd.DataFrame: PPS conversion factors by country and year.
                     Columns: ['country', year columns as integers, 'country_code']
    """
    print("\n=== LOADING PPS DATA ===")
    
    # Path to PPS file
    pps_path = os.path.join(
        dirs['report_base'], '..', 'shared', 'external_datasets', 
        'Purchasing power parities', 'prc_ppp_ind__custom_18896791_spreadsheet.xlsx'
    )
    
    if not os.path.exists(pps_path):
        print(f"WARNING PPS file not found: {pps_path}")
        print("Will proceed without PPS conversion (using nominal values)")
        return pd.DataFrame()
    
    try:
        # Load PPS data from Sheet 1
        print(f"Loading from: {pps_path}")
        pps_df = pd.read_excel(pps_path, sheet_name='Sheet 1', header=None)
        print(f"Raw data shape: {pps_df.shape}")
        
        # Parse the Eurostat structure
        # Row 8 contains TIME (years), data starts at row 10
        time_row = 8
        data_start = 10
        
        # Extract years from columns (every other column starting from column 1)
        year_columns = []
        for col in range(1, pps_df.shape[1], 2):
            val = pps_df.iloc[time_row, col]
            if pd.notna(val) and str(val).isdigit() and len(str(val)) == 4:
                year_columns.append((col, int(val)))
        
        print(f"Years found: {[y[1] for y in year_columns]}")
        
        # Extract countries with their PPS factors
        countries_data = []
        for i in range(data_start, len(pps_df)):
            country = pps_df.iloc[i, 0]
            if pd.notna(country) and str(country).strip():
                row_data = {'country': str(country).strip()}
                for col_idx, year in year_columns:
                    value = pps_df.iloc[i, col_idx]
                    if pd.notna(value) and str(value) != ':':
                        try:
                            row_data[year] = float(value)
                        except:
                            row_data[year] = np.nan
                    else:
                        row_data[year] = np.nan
                countries_data.append(row_data)
        
        # Create structured dataframe
        pps_structured = pd.DataFrame(countries_data)
        print(f"Parsed {len(pps_structured)} country entries")
        
        return pps_structured
        
    except Exception as e:
        print(f"ERROR loading PPS data: {e}")
        print("Will proceed without PPS conversion (using nominal values)")
        return pd.DataFrame()


def stack_excels(folder, pattern):
    """
    Load and stack Excel files matching a pattern from a folder.
    
    Args:
        folder (str): Folder path to search for files
        pattern (str): File glob pattern to match (e.g., "*.xlsx")
        
    Returns:
        pd.DataFrame: Concatenated dataframe from all matching files
    """
    files = glob.glob(os.path.join(folder, pattern))
    dfs = []
    
    if not files:
        print(f"  No files found matching pattern: {pattern}")
        return pd.DataFrame()
    
    for f in files:
        try:
            df = pd.read_excel(f)
            df['source_file'] = os.path.basename(f)
            dfs.append(df)
            print(f"  OK {os.path.basename(f)}: {df.shape}")
        except Exception as e:
            print(f"  ERROR reading {f}: {e}")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"  Combined: {combined.shape}")
        return combined
    else:
        return pd.DataFrame()


def load_hbs_data_from_external(dirs):
    """
    Load HBS (Household Budget Survey) data from external directory structure.
    
    Expects data to be organized in years (2010, 2015, 2020) with household (hh) 
    and household member (hm) files.
    
    Args:
        dirs (dict): Directory paths from setup_directories()
        
    Returns:
        tuple: (household_df, household_members_df) - Both as pandas DataFrames
    """
    print("\n=== LOADING HBS DATA ===")
    
    # External data directory path - UPDATE THIS IF NEEDED
    external_hbs_base = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/HBS"
    
    if not os.path.exists(external_hbs_base):
        print(f"❌ External HBS directory not found: {external_hbs_base}")
        print("Please update the path in load_hbs_data_from_external()")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Loading from: {external_hbs_base}")
    
    # Define paths for each year
    paths = {
        '2010': os.path.join(external_hbs_base, "HBS2010/HBS2010"),
        '2015': os.path.join(external_hbs_base, "HBS2015/HBS2015"),
        '2020': os.path.join(external_hbs_base, "HBS2020/HBS2020"),
    }

    # Define file patterns for each year
    patterns = {
        '2010': {'hh': "*_HBS_hh.xlsx", 'hm': "*_HBS_hm.xlsx"},
        '2015': {'hh': "*_MFR_hh.xlsx", 'hm': "*_MFR_hm.xlsx"},
        '2020': {'hh': "HBS_HH_*.xlsx",  'hm': "HBS_HM_*.xlsx"},
    }

    # Load data for each year
    household_dfs = []
    household_member_dfs = []
    
    for year, folder in paths.items():
        if not os.path.exists(folder):
            print(f"⚠ Directory not found for {year}: {folder}")
            continue
            
        print(f"\n{year} data:")
        
        # Load household files
        print(f"  Household files ({patterns[year]['hh']}):")
        hh_df = stack_excels(folder, patterns[year]['hh'])
        if not hh_df.empty:
            hh_df['year'] = year
            household_dfs.append(hh_df)
        
        # Load household member files
        print(f"  Household member files ({patterns[year]['hm']}):")
        hm_df = stack_excels(folder, patterns[year]['hm'])
        if not hm_df.empty:
            hm_df['year'] = year
            household_member_dfs.append(hm_df)
    
    # Combine all years
    if household_dfs:
        household_all = pd.concat(household_dfs, ignore_index=True)
        print(f"\nOK Household data combined: {household_all.shape}")
        if 'COUNTRY' in household_all.columns:
            countries = sorted(household_all['COUNTRY'].unique())
            print(f"  Countries: {countries}")
        print(f"  Years: {sorted(household_all['year'].unique())}")
    else:
        household_all = pd.DataFrame()
        print("WARNING No household data loaded")
    
    if household_member_dfs:
        household_members_all = pd.concat(household_member_dfs, ignore_index=True)
        print(f"OK Household members data combined: {household_members_all.shape}")
    else:
        household_members_all = pd.DataFrame()
        print("WARNING No household member data loaded")
    
    return household_all, household_members_all


def calculate_consumption_in_pps(household_df, pps_df):
    """
    Calculate consumption expenditure in PPS (Purchasing Power Standards).
    
    Converts nominal consumption values using PPS factors to make them comparable
    across countries. Formula: consumption_pps = consumption_nominal / pps_factor
    
    Args:
        household_df (pd.DataFrame): Household data with consumption columns (EUR_HExx)
        pps_df (pd.DataFrame): PPS conversion factors by country and year
        
    Returns:
        pd.DataFrame: Household data with PPS-adjusted columns added
    """
    print("\n=== CALCULATING CONSUMPTION IN PPS ===")
    
    if pps_df.empty:
        print("WARNING No PPS data available, returning original data")
        return household_df
    
    household_copy = household_df.copy()
    
    try:
        # Create mapping from PPS country names to ISO country codes
        country_mapping = {
            'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Cyprus': 'CY', 'Czechia': 'CZ',
            'Germany': 'DE', 'Denmark': 'DK', 'Estonia': 'EE', 'Spain': 'ES', 'Finland': 'FI',
            'France': 'FR', 'Greece': 'GR', 'Croatia': 'HR', 'Hungary': 'HU', 'Ireland': 'IE',
            'Italy': 'IT', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Latvia': 'LV', 'Malta': 'MT',
            'Netherlands': 'NL', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO', 'Sweden': 'SE',
            'Slovenia': 'SI', 'Slovakia': 'SK', 'Switzerland': 'CH',
            'Czech Republic': 'CZ', 'Slovak Republic': 'SK'
        }
        
        # Reverse mapping: PPS country name → ISO code
        pps_to_iso = {}
        for pps_country in pps_df['country'].tolist():
            for full_name, iso_code in country_mapping.items():
                if full_name.lower() in pps_country.lower() or pps_country.lower() in full_name.lower():
                    pps_to_iso[pps_country] = iso_code
                    break
        
        # Melt PPS data to long format
        year_cols = [col for col in pps_df.columns if isinstance(col, int)]
        pps_long = pps_df.melt(
            id_vars=['country'], 
            value_vars=year_cols,
            var_name='year', 
            value_name='pps_factor'
        )
        
        # Add ISO country codes
        pps_long['country_code'] = pps_long['country'].map(pps_to_iso)
        pps_long = pps_long.dropna(subset=['country_code', 'pps_factor'])
        
        # Ensure year is integer for merge
        pps_long['year'] = pd.to_numeric(pps_long['year'], errors='coerce').astype('Int64')
        pps_long = pps_long.dropna(subset=['year'])
        
        print(f"OK PPS data formatted: {len(pps_long)} country-year observations")
        print(f"  PPS data years: {sorted(pps_long['year'].unique())}")
        print(f"  PPS data countries: {sorted(pps_long['country_code'].unique())}")
        
        # Merge PPS factors with household data
        # Convert household year to integer for merge
        household_copy['year_int'] = pd.to_numeric(household_copy['year'], errors='coerce').astype('Int64')
        
        # Debug: check what we're trying to merge
        print(f"  Household countries: {sorted(household_copy['COUNTRY'].unique())}")
        print(f"  Household years: {sorted(household_copy['year_int'].unique())}")
        
        household_with_pps = household_copy.merge(
            pps_long[['country_code', 'year', 'pps_factor']],
            left_on=['COUNTRY', 'year_int'],
            right_on=['country_code', 'year'],
            how='left',
            suffixes=('', '_pps_year')
        )
        
        # Ensure year column is preserved correctly
        if 'year_pps_year' in household_with_pps.columns:
            household_with_pps['year'] = household_with_pps['year_pps_year']
            household_with_pps = household_with_pps.drop(columns=['year_pps_year'])
        
        # Apply PPS conversion to consumption columns
        consumption_columns = [col for col in household_copy.columns if col.startswith('EUR_HE')]
        
        # DEBUG: Check PPS factors for diagnosis
        pps_sample = household_with_pps[household_with_pps['pps_factor'].notna()][['COUNTRY', 'year', 'pps_factor']].drop_duplicates()
        print(f"\n  Sample PPS factors:")
        for _, row in pps_sample.head(5).iterrows():
            print(f"    {row['COUNTRY']} {int(row['year'])}: {row['pps_factor']:.4f} (1/pps_factor = {1/row['pps_factor']:.4f})")
        
        # Convert to PPS using vectorized operations (avoid DataFrame fragmentation)
        pps_converted_cols = {}
        for col in consumption_columns:
            if col in household_with_pps.columns and 'pps_factor' in household_with_pps.columns:
                # Convert to PPS: The Eurostat PPS factor represents purchasing power parity indices
                # where we need to divide the nominal values to convert to PPS-adjusted values
                # This accounts for price level differences between countries
                pps_converted_cols[f'{col}_pps'] = household_with_pps[col] / household_with_pps['pps_factor']
        
        # Add all converted columns at once using pd.concat (efficient)
        if pps_converted_cols:
            pps_converted_df = pd.DataFrame(pps_converted_cols, index=household_with_pps.index)
            household_with_pps = pd.concat([household_with_pps, pps_converted_df], axis=1)
        
        pps_conversions = len(pps_converted_cols)
        print(f"OK Applied PPS conversion to {pps_conversions} consumption columns")
        
        # Clean up temporary columns
        columns_to_drop = [col for col in ['year_int', 'country_code', 'year_y'] 
                          if col in household_with_pps.columns]
        household_with_pps = household_with_pps.drop(columns=columns_to_drop)
        
        # Report coverage
        pps_coverage = household_with_pps['pps_factor'].notna().sum() / len(household_with_pps) * 100
        print(f"OK PPS conversion coverage: {pps_coverage:.1f}% of households")
        
        return household_with_pps
        
    except Exception as e:
        print(f"ERROR in PPS conversion: {e}")
        print("Returning original data without PPS conversion")
        return household_df


def assign_income_deciles(household_df):
    """
    Assign household income deciles based on equivalent income.
    
    Uses weighted quantiles to assign each household to income decile D1-D10
    (D1 = poorest, D10 = richest). Weights reflect household representation 
    in the population.
    
    Args:
        household_df (pd.DataFrame): Household data with income column (EUR_HH099 or EUR_HH095)
        
    Returns:
        pd.DataFrame: Household data with 'income_decile' column added
    """
    print("\n=== ASSIGNING INCOME DECILES ===")
    
    household_copy = household_df.copy()
    
    try:
        # Ensure required columns are numeric
        if 'EUR_HH099' in household_copy.columns:
            income_col = 'EUR_HH099'
        elif 'EUR_HH095' in household_copy.columns:
            income_col = 'EUR_HH095'
        else:
            print("❌ No income column found (EUR_HH099 or EUR_HH095)")
            return household_copy
        
        household_copy[income_col] = pd.to_numeric(household_copy[income_col], errors='coerce')
        
        if 'HA10' not in household_copy.columns:
            print("WARNING Sample weight column (HA10) not found, using unweighted deciles")
            weight_col = None
        else:
            household_copy['HA10'] = pd.to_numeric(household_copy['HA10'], errors='coerce')
            weight_col = 'HA10'
        
        deciles = []
        
        # Process each country-year group
        for (country, year), group in household_copy.groupby(['COUNTRY', 'year']):
            # Filter valid records
            valid_mask = group[income_col].notna()
            
            if weight_col and weight_col in group.columns:
                valid_mask = valid_mask & group[weight_col].notna()
                weights = group.loc[valid_mask, weight_col]
            else:
                weights = None
            
            valid_group = group[valid_mask]
            
            if len(valid_group) < 10:
                continue
            
            # Calculate weighted quantiles
            if weights is not None:
                # Custom weighted quantile function
                income_values = valid_group[income_col].values
                weights_values = valid_group[weight_col].values
                
                # Sort by income
                sorted_idx = np.argsort(income_values)
                sorted_income = income_values[sorted_idx]
                sorted_weights = weights_values[sorted_idx]
                
                # Calculate cumulative weights
                cum_weights = np.cumsum(sorted_weights)
                cum_weights_norm = cum_weights / cum_weights[-1]
                
                # Find decile boundaries
                decile_boundaries = []
                for d in range(1, 10):
                    target = d / 10.0
                    idx = np.searchsorted(cum_weights_norm, target)
                    if idx < len(sorted_income):
                        decile_boundaries.append(sorted_income[idx])
                    else:
                        decile_boundaries.append(sorted_income[-1])
            else:
                # Unweighted quantiles
                decile_boundaries = np.quantile(valid_group[income_col], np.arange(0.1, 1.0, 0.1))
            
            # Assign deciles
            group_copy = valid_group.copy()
            group_copy['income_decile'] = pd.cut(
                group_copy[income_col],
                bins=[-np.inf] + list(decile_boundaries) + [np.inf],
                labels=['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
            )
            
            deciles.append(group_copy)
        
        if deciles:
            result = pd.concat(deciles, ignore_index=False)
            result = result.sort_index()
            
            # Fill non-assigned rows with NaN
            household_copy['income_decile'] = result['income_decile']
            
            print(f"OK Deciles assigned")
            print(f"  Distribution:")
            for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
                count = (household_copy['income_decile'] == decile).sum()
                if count > 0:
                    print(f"    {decile}: {count:,} households")
            
            return household_copy
        else:
            print("ERROR Could not assign deciles")
            return household_copy
            
    except Exception as e:
        print(f"ERROR in decile assignment: {e}")
        return household_copy


def get_consumption_by_decile(household_df, countries=None, consumption_col='EUR_HE00_pps'):
    """
    Calculate mean consumption per adult equivalent by income decile.
    
    Aggregates household-level data to decile-level statistics using sample weights.
    
    Args:
        household_df (pd.DataFrame): Household data with income_decile assigned
        countries (list): List of country codes to include (default: all)
        consumption_col (str): Consumption column to use (default: 'EUR_HE00_pps')
        
    Returns:
        pd.DataFrame: Mean consumption by country, year, and decile
    """
    print("\n=== CALCULATING CONSUMPTION BY DECILE ===")
    
    # Use fallback if PPS column doesn't exist
    if consumption_col not in household_df.columns:
        consumption_col = 'EUR_HE00' if 'EUR_HE00' in household_df.columns else 'EUR_HE00_pps'
        print(f"Using consumption column: {consumption_col}")
    
    results = []
    
    for (country, year, decile), group in household_df.groupby(['COUNTRY', 'year', 'income_decile']):
        if countries and country not in countries:
            continue
        
        if group.empty:
            continue
        
        # Ensure numeric columns
        group = group.copy()
        group['HA10'] = pd.to_numeric(group['HA10'], errors='coerce')
        group[consumption_col] = pd.to_numeric(group[consumption_col], errors='coerce')
        group['HB061'] = pd.to_numeric(group['HB061'], errors='coerce')
        
        # Filter valid records
        valid_mask = (
            group[consumption_col].notna() & 
            group['HA10'].notna() & 
            group['HB061'].notna() &
            (group[consumption_col] > 0) &
            (group['HB061'] > 0)
        )
        valid_group = group[valid_mask]
        
        if len(valid_group) > 0 and valid_group['HA10'].sum() > 0:
            # Calculate per adult equivalent consumption
            valid_group['consumption_per_ae'] = valid_group[consumption_col] / valid_group['HB061']
            
            # Calculate weighted mean
            weighted_mean = np.average(
                valid_group['consumption_per_ae'],
                weights=valid_group['HA10']
            )
            
            results.append({
                'country': country,
                'year': int(year),
                'decile': decile,
                'mean_consumption': weighted_mean,
                'household_count': len(valid_group),
                'total_weight': valid_group['HA10'].sum()
            })
    
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        print(f"OK Calculated consumption for {len(result_df)} country-year-decile combinations")
        countries_in_result = result_df['country'].unique()
        print(f"  Countries: {sorted(countries_in_result)}")
        print(f"  Years: {sorted(result_df['year'].unique())}")
    
    return result_df


def save_processed_data(household_df, consumption_by_decile, dirs):
    """
    Save processed data to CSV files.
    
    Args:
        household_df (pd.DataFrame): Processed household data
        consumption_by_decile (pd.DataFrame): Aggregated consumption by decile
        dirs (dict): Directory paths
    """
    print("\n=== SAVING PROCESSED DATA ===")
    
    # Create intermediate folder
    intermediate_dir = os.path.join(dirs['outputs'], 'intermediate')
    os.makedirs(intermediate_dir, exist_ok=True)
    
    if not household_df.empty:
        household_path = os.path.join(intermediate_dir, 'HBS_household_data_processed.csv')
        household_df.to_csv(household_path, index=False)
        print(f"OK Saved: {household_path}")
    
    if not consumption_by_decile.empty:
        consumption_path = os.path.join(intermediate_dir, 'consumption_by_decile_selected_countries.csv')
        consumption_by_decile.to_csv(consumption_path, index=False)
        print(f"OK Saved: {consumption_path}")
