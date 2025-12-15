"""
HBS Data Loader for Switzerland vs EU-27 Analysis

This script loads HBS data from external directory and creates consumption analysis visualizations.

Author: Data for Good - Well-being Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def setup_directories():
    """Set up directories for data access."""
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
    
    return dirs


def load_pps_data(dirs):
    """
    Load PPS (Purchasing Power Parities) data from the shared external datasets.
    
    Args:
        dirs (dict): Directory paths
        
    Returns:
        pd.DataFrame: PPS conversion factors by country and year
    """
    print("Loading PPS data...")
    
    # Path to PPS file
    pps_path = os.path.join(
        dirs['report_base'], '..', 'shared', 'external_datasets', 
        'Purchasing power parities', 'prc_ppp_ind__custom_18896791_spreadsheet.xlsx'
    )
    
    if not os.path.exists(pps_path):
        print(f"PPS file not found: {pps_path}")
        print("Will proceed without PPS conversion (using nominal values)")
        return pd.DataFrame()
    
    try:
        # Load PPS data from Sheet 1 (actual data)
        pps_df = pd.read_excel(pps_path, sheet_name='Sheet 1', header=None)
        print(f"PPS raw data loaded successfully. Shape: {pps_df.shape}")
        
        # Parse the known Eurostat structure (TIME row at 8, data starts at 10)
        time_row = 8
        data_start = 10
        
        # Validate structure
        if len(pps_df) <= data_start or 'TIME' not in str(pps_df.iloc[time_row, 0]):
            print("Warning: PPS file structure may have changed")
            return pd.DataFrame()
        
        # Extract years and countries as we learned from analysis
        year_columns = []
        for col in range(1, pps_df.shape[1], 2):  # Every other column (skip NaN columns)
            val = pps_df.iloc[time_row, col]
            if pd.notna(val) and str(val).isdigit() and len(str(val)) == 4:
                year_columns.append((col, str(val)))
        
        print(f"Years found in PPS data: {[y[1] for y in year_columns]}")
        
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
        
        # Create structured PPS dataframe
        pps_structured = pd.DataFrame(countries_data)
        
        # Validate we have key EU countries (Switzerland may not be available)
        swiss_found = any(any(swiss_name in str(country).lower() for swiss_name in ['switzerland', 'swiss', 'suisse']) 
                         for country in pps_structured['country'])
        print(f"Switzerland found in PPS data: {swiss_found}")
        if not swiss_found:
            print("Warning: Switzerland not found in PPS data. Analysis will focus on EU-27 countries only.")
        
        eu_countries_found = sum(1 for country in pps_structured['country'] 
                                if any(eu_name in str(country) for eu_name in 
                                ['Germany', 'France', 'Italy', 'Spain', 'Austria', 'Belgium']))
        print(f"EU countries found in PPS data: {eu_countries_found}")
        
        print(f"Parsed PPS data successfully. Final shape: {pps_structured.shape}")
        return pps_structured
        
    except Exception as e:
        print(f"Error loading PPS data: {e}")
        print("Will proceed without PPS conversion (using nominal values)")
        return pd.DataFrame()


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
            df['source_file'] = os.path.basename(f)
            dfs.append(df)
            print(f"Loaded: {os.path.basename(f)} - Shape: {df.shape}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Combined data shape: {combined.shape}")
        return combined
    else:
        print("No files found or loaded successfully")
        return pd.DataFrame()


def load_hbs_data_from_external(dirs):
    """
    Load HBS data from external directory structure.
    
    Args:
        dirs (dict): Directory paths
        
    Returns:
        tuple: (household_data, household_members_data)
    """
    print("Loading HBS data from external directory...")
    
    # External data directory path (you may need to modify this)
    external_hbs_base = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/HBS"
    
    if not os.path.exists(external_hbs_base):
        print(f"External HBS directory not found: {external_hbs_base}")
        print("Please update the path in the load_hbs_data_from_external function")
        return pd.DataFrame(), pd.DataFrame()
    
    # Define paths for each year
    paths = {
        '2010': os.path.join(external_hbs_base, "HBS2010/HBS2010"),
        '2015': os.path.join(external_hbs_base, "HBS2015/HBS2015"),
        '2020': os.path.join(external_hbs_base, "HBS2020/HBS2020"),
    }

    # Define file patterns for each year and type
    patterns = {
        '2010': {'hh': "*_HBS_hh.xlsx", 'hm': "*_HBS_hm.xlsx"},
        '2015': {'hh': "*_MFR_hh.xlsx", 'hm': "*_MFR_hm.xlsx"},
        '2020': {'hh': "HBS_HH_*.xlsx",  'hm': "HBS_HM_*.xlsx"},
    }

    # Load data for each year and type
    household_dfs = []
    household_member_dfs = []
    
    for year, folder in paths.items():
        if not os.path.exists(folder):
            print(f"Warning: Directory not found for {year}: {folder}")
            continue
            
        print(f"Loading {year} data from: {folder}")
        
        # Load household files
        print(f"  Loading household files with pattern: {patterns[year]['hh']}")
        hh_df = stack_excels(folder, patterns[year]['hh'])
        if not hh_df.empty:
            hh_df['year'] = year
            household_dfs.append(hh_df)
        
        # Load household member files
        print(f"  Loading household member files with pattern: {patterns[year]['hm']}")
        hm_df = stack_excels(folder, patterns[year]['hm'])
        if not hm_df.empty:
            hm_df['year'] = year
            household_member_dfs.append(hm_df)
    
    # Combine all years
    if household_dfs:
        household_all = pd.concat(household_dfs, ignore_index=True)
        print(f"\\nFinal household data shape: {household_all.shape}")
        print(f"Countries: {sorted(household_all['COUNTRY'].unique()) if 'COUNTRY' in household_all.columns else 'COUNTRY column not found'}")
        print(f"Years: {sorted(household_all['year'].unique())}")
    else:
        household_all = pd.DataFrame()
        print("No household data loaded")
    
    if household_member_dfs:
        household_members_all = pd.concat(household_member_dfs, ignore_index=True)
        print(f"Final household members data shape: {household_members_all.shape}")
    else:
        household_members_all = pd.DataFrame()
        print("No household member data loaded")
    
    return household_all, household_members_all


def calculate_consumption_in_pps(household_df, pps_df):
    """
    Calculate consumption expenditure in PPS (Purchasing Power Standards).
    
    Args:
        household_df (pd.DataFrame): Household data with consumption
        pps_df (pd.DataFrame): PPS conversion factors
        
    Returns:
        pd.DataFrame: Household data with PPS-adjusted consumption
    """
    print("Calculating consumption in PPS...")
    
    if pps_df.empty:
        print("No PPS data available, returning original data")
        return household_df
    
    # Parse PPS data from Eurostat format
    try:
        # Use the already loaded pps_df or reload if needed
        if pps_df.empty:
            print("PPS dataframe is empty, attempting to reload...")
            return household_df
            
        # Parse the raw PPS file structure that we know from analysis
        pps_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'shared', 'external_datasets', 
            'Purchasing power parities', 'prc_ppp_ind__custom_18896791_spreadsheet.xlsx'
        )
        
        df_raw = pd.read_excel(pps_path, sheet_name='Sheet 1', header=None)
        
        # Parse structure (row 8 has years, data starts at row 10)
        time_row = 8
        data_start = 10
        
        # Extract years (every other column starting from 1)
        year_columns = []
        for col in range(1, df_raw.shape[1], 2):
            val = df_raw.iloc[time_row, col]
            if pd.notna(val) and str(val).isdigit():
                year_columns.append((col, int(val)))
        
        # Extract country data
        countries_data = []
        for i in range(data_start, len(df_raw)):
            country = df_raw.iloc[i, 0]
            if pd.notna(country) and str(country).strip():
                row_data = {'country': str(country).strip()}
                for col_idx, year in year_columns:
                    value = df_raw.iloc[i, col_idx]
                    if pd.notna(value) and str(value) != ':':
                        try:
                            row_data[year] = float(value)
                        except:
                            row_data[year] = np.nan
                    else:
                        row_data[year] = np.nan
                countries_data.append(row_data)
        
        # Create PPS dataframe
        pps_clean = pd.DataFrame(countries_data)
        
        # Create comprehensive country mapping for HBS country codes to PPS country names
        # Based on the PPS file structure we analyzed
        country_mapping = {
            'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus', 'CZ': 'Czechia',
            'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia', 'ES': 'Spain', 'FI': 'Finland',
            'FR': 'France', 'GR': 'Greece', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland',
            'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta',
            'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'SE': 'Sweden',
            'SI': 'Slovenia', 'SK': 'Slovakia', 'CH': 'Switzerland'
        }
        
        # Alternative country name variations found in PPS data
        country_name_variations = {
            'Czech Republic': 'CZ',
            'Slovak Republic': 'SK', 
            'Euro area': None,  # Exclude aggregates
            'European Union': None,
            'Czechia': 'CZ',
            'Germany': 'DE',
            'Deutschland': 'DE'
        }
        
        # Reverse mapping: full names to country codes
        pps_to_iso = {}
        for pps_country in pps_clean['country'].tolist():
            # Direct match from variations
            if pps_country in country_name_variations:
                iso_code = country_name_variations[pps_country]
                if iso_code:
                    pps_to_iso[pps_country] = iso_code
                continue
                
            # Fuzzy match with standard mapping
            for iso, full_name in country_mapping.items():
                if (full_name.lower() in pps_country.lower() or 
                    pps_country.lower() in full_name.lower()):
                    pps_to_iso[pps_country] = iso
                    break
        
        # Melt PPS data to long format
        year_cols = [col for col in pps_clean.columns if isinstance(col, int)]
        pps_long = pps_clean.melt(
            id_vars=['country'], 
            value_vars=year_cols,
            var_name='year', 
            value_name='pps_factor'
        )
        
        # Add ISO country codes
        pps_long['country_code'] = pps_long['country'].map(pps_to_iso)
        pps_long = pps_long.dropna(subset=['country_code', 'pps_factor'])
        
        print(f"PPS data parsed successfully: {len(pps_long)} country-year observations")
        print(f"Countries with PPS data: {sorted(pps_long['country_code'].unique())})")
        print(f"Years available: {sorted(pps_long['year'].unique())})")
        
        # Show sample PPS factors for key countries
        sample_countries = ['CH', 'DE', 'FR', 'IT', 'ES']
        latest_pps_year = pps_long['year'].max()
        sample_pps = pps_long[
            (pps_long['country_code'].isin(sample_countries)) & 
            (pps_long['year'] == latest_pps_year)
        ]
        if not sample_pps.empty:
            print(f"\nSample PPS factors for {latest_pps_year}:")
            for _, row in sample_pps.iterrows():
                print(f"  {row['country_code']}: {row['pps_factor']:.3f}")
        
        # Convert household_df year column to integer
        household_df_copy = household_df.copy()
        household_df_copy['year_int'] = pd.to_numeric(household_df_copy['year'], errors='coerce').astype('Int64')
        
        # Merge PPS factors with household data
        household_with_pps = household_df_copy.merge(
            pps_long[['country_code', 'year', 'pps_factor']],
            left_on=['COUNTRY', 'year_int'],
            right_on=['country_code', 'year'],
            how='left'
        )
        
        # Calculate PPS-adjusted consumption
        consumption_columns = [col for col in household_df.columns if col.startswith('EUR_HE')]
        
        for col in consumption_columns:
            if col in household_with_pps.columns:
                # Convert to PPS: divide nominal euros by PPS factor
                household_with_pps[f'{col}_pps'] = household_with_pps[col] / household_with_pps['pps_factor']
        
        # Also convert equivalized consumption if it exists
        if 'consumption_per_adult_equiv' in household_with_pps.columns:
            household_with_pps['consumption_per_adult_equiv_pps'] = (
                household_with_pps['consumption_per_adult_equiv'] / household_with_pps['pps_factor']
            )
        
        # Drop temporary columns
        columns_to_drop = ['year_int', 'country_code', 'year_y']
        household_with_pps = household_with_pps.drop(columns=[col for col in columns_to_drop if col in household_with_pps.columns])
        
        # Report on PPS conversion success
        pps_coverage = household_with_pps['pps_factor'].notna().sum() / len(household_with_pps) * 100
        print(f"PPS conversion coverage: {pps_coverage:.1f}% of households")
        
        countries_with_pps = household_with_pps[household_with_pps['pps_factor'].notna()]['COUNTRY'].nunique()
        total_countries = household_with_pps['COUNTRY'].nunique()
        print(f"Countries with PPS data: {countries_with_pps}/{total_countries}")
        
        return household_with_pps
        
    except Exception as e:
        print(f"Error in PPS conversion: {e}")
        print("Returning original data without PPS conversion")
        return household_df


def analyze_mean_consumption_by_country(household_df):
    """
    Analyze mean consumption expenditure per adult equivalent by country.
    
    Args:
        household_df (pd.DataFrame): Household data
        
    Returns:
        pd.DataFrame: Mean consumption by country and year
    """
    print("Analyzing mean consumption by country...")
    
    # Ensure required columns are numeric
    numeric_cols = ['HA10', 'HB061', 'EUR_HE00']  # Weight, equivalent size, total consumption
    for col in numeric_cols:
        if col in household_df.columns:
            household_df[col] = pd.to_numeric(household_df[col], errors='coerce')
    
    # Calculate consumption per adult equivalent (nominal and PPS if available)
    if 'EUR_HE00' in household_df.columns and 'HB061' in household_df.columns:
        household_df['consumption_per_adult_equiv'] = household_df['EUR_HE00'] / household_df['HB061']
        
        # Also calculate PPS version if PPS conversion was done
        if 'EUR_HE00_pps' in household_df.columns:
            household_df['consumption_per_adult_equiv_pps'] = household_df['EUR_HE00_pps'] / household_df['HB061']
            print("Using PPS-adjusted consumption values")
        else:
            print("Using nominal consumption values (no PPS conversion available)")
    else:
        print("Warning: Cannot calculate per adult equivalent - missing EUR_HE00 or HB061")
        return pd.DataFrame()
    
    results = []
    
    # Determine which consumption variable to use (prefer PPS)
    consumption_var = 'consumption_per_adult_equiv_pps' if 'consumption_per_adult_equiv_pps' in household_df.columns else 'consumption_per_adult_equiv'
    
    # Calculate weighted means by country and year
    for (country, year), group in household_df.groupby(['COUNTRY', 'year']):
        if 'HA10' not in group.columns or group['HA10'].sum() == 0:
            continue
            
        # Remove invalid values
        valid_mask = (
            group[consumption_var].notna() & 
            group['HA10'].notna() & 
            (group[consumption_var] > 0)
        )
        valid_group = group[valid_mask]
        
        if len(valid_group) > 0 and valid_group['HA10'].sum() > 0:
            weighted_mean = np.average(
                valid_group[consumption_var], 
                weights=valid_group['HA10']
            )
            
            # Also calculate nominal version for comparison
            if consumption_var.endswith('_pps') and 'consumption_per_adult_equiv' in valid_group.columns:
                nominal_weighted_mean = np.average(
                    valid_group['consumption_per_adult_equiv'], 
                    weights=valid_group['HA10']
                )
            else:
                nominal_weighted_mean = weighted_mean
            
            # Get PPS factor for reference
            pps_factor = valid_group['pps_factor'].mean() if 'pps_factor' in valid_group.columns else np.nan
            
            results.append({
                'country': country,
                'year': int(year),
                'mean_consumption_per_adult_equiv': weighted_mean,
                'mean_consumption_per_adult_equiv_nominal': nominal_weighted_mean,
                'pps_factor': pps_factor,
                'is_pps_adjusted': consumption_var.endswith('_pps'),
                'household_count': len(valid_group),
                'total_weight': valid_group['HA10'].sum()
            })
    
    summary_df = pd.DataFrame(results)
    
    if not summary_df.empty:
        pps_adjusted_count = summary_df['is_pps_adjusted'].sum()
        total_count = len(summary_df)
        print(f"Analysis complete: {pps_adjusted_count}/{total_count} observations with PPS adjustment")
        
        # Show comparison of nominal vs PPS for a sample
        if 'mean_consumption_per_adult_equiv_nominal' in summary_df.columns:
            sample = summary_df[summary_df['is_pps_adjusted']].head(3)
            if not sample.empty:
                print(f"Sample comparison (nominal vs PPS):")
                for _, row in sample.iterrows():
                    nominal = row['mean_consumption_per_adult_equiv_nominal']
                    pps = row['mean_consumption_per_adult_equiv']
                    print(f"  {row['country']} {row['year']}: {nominal:.0f} EUR → {pps:.0f} PPS (factor: {row['pps_factor']:.2f})")
    
    return summary_df


def explore_data_structure(df, data_type="household"):
    """
    Explore and print basic information about the data structure.
    
    Args:
        df (pd.DataFrame): The data to explore
        data_type (str): Type of data ('household' or 'household_members')
    """
    if df.empty:
        print(f"No {data_type} data to explore")
        return
    
    print(f"=== {data_type.upper()} DATA EXPLORATION ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Show first few columns
    print(f"First 10 columns:")
    print(df.columns.tolist()[:10])
    
    # Key variables for our analysis
    key_variables = [
        'HA04',  # Household ID
        'COUNTRY', 'YEAR', 
        'HA10',  # Sample weight
        'EUR_HH095',  # Monetary net income
        'EUR_HH099',  # Net income
        'EUR_HE00',   # Total consumption
        'EUR_HE01',   # Food
        'EUR_HE04',   # Housing
        'EUR_HE07',   # Transport
        'HB05',       # Household size
        'HB061',      # Equivalent size OECD
    ]
    
    print(f"Key variables availability:")
    for var in key_variables:
        if var in df.columns:
            non_null = df[var].notna().sum()
            print(f"  ✓ {var}: {non_null:,} non-null values ({non_null/len(df)*100:.1f}%)")
        else:
            print(f"  ✗ {var}: Not available")
    
    # Show sample of data
    if not df.empty:
        print(f"Sample data (first 3 rows):")
        available_key_vars = [var for var in key_variables[:8] if var in df.columns]
        if available_key_vars:
            print(df[available_key_vars].head(3).to_string(index=False))


def save_loaded_data(household_df, household_members_df, dirs):
    """
    Save the loaded data to CSV files for further analysis.
    
    Args:
        household_df (pd.DataFrame): Household data
        household_members_df (pd.DataFrame): Household members data
        dirs (dict): Directory paths
    """
    print("Saving loaded data...")
    
    # Create intermediate folder for outputs
    intermediate_dir = os.path.join(dirs['outputs'], 'intermediate')
    os.makedirs(intermediate_dir, exist_ok=True)
    
    if not household_df.empty:
        household_path = os.path.join(intermediate_dir, 'HBS_household_data_loaded.csv')
        household_df.to_csv(household_path, index=False)
        print(f"✓ Saved household data: {household_path}")
        print(f"  Shape: {household_df.shape}")
    
    if not household_members_df.empty:
        members_path = os.path.join(intermediate_dir, 'HBS_household_members_data_loaded.csv')
        household_members_df.to_csv(members_path, index=False)
        print(f"✓ Saved household members data: {members_path}")
        print(f"  Shape: {household_members_df.shape}")


def create_consumption_visualization(consumption_summary, dirs):
    """
    Create visualization for mean consumption expenditure in PPS for EU-27 countries.
    
    Args:
        consumption_summary (pd.DataFrame): Mean consumption data by country
        dirs (dict): Directory paths
    """
    print("Creating consumption visualization...")
    
    if consumption_summary.empty:
        print("No consumption data available for visualization")
        return
    
    # Create graphs output directory
    graphs_dir = os.path.join(dirs['outputs'], 'graphs', 'HBS')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Define EU-27 countries
    eu27_countries = [
        'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 
        'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'
    ]
    
    # Check if Switzerland data is available in the dataset
    switzerland_available = 'CH' in consumption_summary['country'].unique()
    target_countries = eu27_countries + (['CH'] if switzerland_available else [])
    
    print(f"Switzerland data available: {switzerland_available}")
    print(f"Analyzing {len(target_countries)} countries: {len(eu27_countries)} EU-27{' + Switzerland' if switzerland_available else ''}")
    
    # Filter to target countries and get latest year data
    latest_year = consumption_summary['year'].max()
    latest_data = consumption_summary[
        (consumption_summary['year'] == latest_year) & 
        (consumption_summary['country'].isin(target_countries))
    ].copy()
    
    if latest_data.empty:
        print(f"No data available for {latest_year}")
        return
    
    # Sort by consumption level for better visualization
    latest_data = latest_data.sort_values('mean_consumption_per_adult_equiv', ascending=True)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Color coding: Switzerland in red (if available), EU-27 in blue
    colors = []
    switzerland_available = 'CH' in latest_data['country'].values
    for country in latest_data['country']:
        if country == 'CH':
            colors.append('red')
        elif country in eu27_countries:
            colors.append('steelblue')
        else:
            colors.append('gray')  # For any non-EU27, non-CH countries
    
    # Create horizontal bar chart
    bars = plt.barh(
        range(len(latest_data)), 
        latest_data['mean_consumption_per_adult_equiv'],
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Customize the plot
    plt.yticks(range(len(latest_data)), latest_data['country'])
    
    # Determine if data is PPS-adjusted for proper labeling
    is_pps = latest_data['is_pps_adjusted'].iloc[0] if 'is_pps_adjusted' in latest_data.columns else False
    currency_label = 'PPS' if is_pps else 'EUR (nominal)'
    
    # Create dynamic title based on data availability
    switzerland_in_data = 'CH' in latest_data['country'].values
    if switzerland_in_data:
        title_countries = "EU-27 vs Switzerland"
    else:
        title_countries = "EU-27 Countries"
    
    plt.xlabel(f'Mean Consumption Expenditure per Adult Equivalent ({currency_label})', fontsize=12)
    title_suffix = f' (PPS-adjusted)' if is_pps else ' (nominal values)'
    plt.title(f'Mean Consumption Expenditure per Adult Equivalent - {title_countries} ({latest_year}){title_suffix}', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, latest_data['mean_consumption_per_adult_equiv'])):
        plt.text(
            bar.get_width() + max(latest_data['mean_consumption_per_adult_equiv']) * 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{value:.0f}',
            ha='left', va='center', fontsize=9
        )
    
    # Add dynamic legend based on data availability
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='EU-27 Countries')]
    if switzerland_in_data:
        legend_elements.append(Patch(facecolor='red', alpha=0.7, label='Switzerland'))
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Format x-axis with thousands separator
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot with dynamic filename
    is_pps_suffix = '_pps_adjusted' if is_pps else '_nominal'
    switzerland_suffix = '_with_ch' if switzerland_in_data else '_eu27_only'
    output_path = os.path.join(graphs_dir, f'mean_consumption_per_adult_equivalent{switzerland_suffix}{is_pps_suffix}_{latest_year}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved consumption visualization: {output_path}")
    
    # Also create a summary table
    summary_table_path = os.path.join(dirs['outputs'], 'intermediate', 'consumption_summary_by_country.csv')
    latest_data.to_csv(summary_table_path, index=False)
    print(f"✓ Saved consumption summary table: {summary_table_path}")
    
    # Print statistics (with or without Switzerland)
    eu_countries_in_data = latest_data[latest_data['country'].isin(eu27_countries)]
    switzerland_in_data = 'CH' in latest_data['country'].values
    
    if not eu_countries_in_data.empty:
        eu_mean = eu_countries_in_data['mean_consumption_per_adult_equiv'].mean()
        
        print(f"\nConsumption Statistics ({latest_year}):")
        if switzerland_in_data:
            ch_consumption = latest_data[latest_data['country'] == 'CH']['mean_consumption_per_adult_equiv'].iloc[0]
            print(f"- Switzerland: {ch_consumption:,.0f} {currency_label} per adult equivalent")
            print(f"- EU-27 average: {eu_mean:,.0f} {currency_label} per adult equivalent")
            print(f"- Switzerland vs EU-27 ratio: {ch_consumption/eu_mean:.2f}")
            
            # Rank Switzerland among all countries
            switzerland_rank = (latest_data['mean_consumption_per_adult_equiv'] > ch_consumption).sum() + 1
            print(f"- Switzerland rank: {switzerland_rank}/{len(latest_data)} (highest consumption = 1)")
        else:
            print(f"- EU-27 average: {eu_mean:,.0f} {currency_label} per adult equivalent")
            eu_max = eu_countries_in_data['mean_consumption_per_adult_equiv'].max()
            eu_min = eu_countries_in_data['mean_consumption_per_adult_equiv'].min()
            print(f"- EU-27 range: {eu_min:,.0f} - {eu_max:,.0f} {currency_label}")
            
        print(f"- Countries with data: {len(latest_data)} total ({len(eu_countries_in_data)} EU-27{' + 1 Switzerland' if switzerland_in_data else ''})")


def main():
    """
    Main function to load HBS data and create consumption analysis.
    """
    print("HBS Data Loader and Analysis")
    print("="*50)
    
    # Setup directories
    dirs = setup_directories()
    print(f"Output directory: {dirs['outputs']}")
    
    # Load PPS data for conversion
    pps_df = load_pps_data(dirs)
    
    # Load HBS data from external directory
    household_df, household_members_df = load_hbs_data_from_external(dirs)
    
    # Explore data structure
    explore_data_structure(household_df, "household")
    explore_data_structure(household_members_df, "household_members")
    
    # Save loaded data
    save_loaded_data(household_df, household_members_df, dirs)
    
    if not household_df.empty:
        # Calculate consumption in PPS (if PPS data available)
        household_df_pps = calculate_consumption_in_pps(household_df, pps_df)
        
        # Analyze mean consumption by country
        consumption_summary = analyze_mean_consumption_by_country(household_df_pps)
        
        if not consumption_summary.empty:
            # Create visualization for EU-27 consumption comparison
            create_consumption_visualization(consumption_summary, dirs)
            
            # Save detailed consumption analysis
            consumption_path = os.path.join(dirs['outputs'], 'intermediate', 'detailed_consumption_analysis.csv')
            consumption_summary.to_csv(consumption_path, index=False)
            print(f"✓ Saved detailed consumption analysis: {consumption_path}")
        else:
            print("No consumption data available for analysis")
    
    print("\n" + "="*50)
    print("Analysis completed!")
    
    if not household_df.empty:
        print(f"\nGenerated outputs:")
        print(f"1. Raw data files in: {os.path.join(dirs['outputs'], 'intermediate')}")
        print(f"2. Consumption visualization in: {os.path.join(dirs['outputs'], 'graphs', 'HBS')}")
        print(f"3. Analysis results in: {dirs['outputs']}")
        
        # Report on data coverage
        if 'consumption_summary' in locals() and not consumption_summary.empty:
            total_countries = consumption_summary['country'].nunique()
            total_years = consumption_summary['year'].nunique()
            pps_adjusted = consumption_summary['is_pps_adjusted'].sum() if 'is_pps_adjusted' in consumption_summary.columns else 0
            
            # Define EU-27 countries for coverage check
            eu27_countries = [
                'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 
                'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'
            ]
            
            print(f"\nData Coverage Summary:")
            print(f"- Countries analyzed: {total_countries}")
            print(f"- Years covered: {total_years}")
            print(f"- Observations with PPS adjustment: {pps_adjusted}/{len(consumption_summary)}")
            
            # Check for Switzerland and key EU countries
            countries_analyzed = consumption_summary['country'].unique()
            if 'CH' in countries_analyzed:
                print(f"- Switzerland: ✓ Included")
            else:
                print(f"- Switzerland: ✗ Missing")
                
            eu_count = sum(1 for c in countries_analyzed if c in eu27_countries)
            print(f"- EU-27 countries: {eu_count}/27 included")


if __name__ == "__main__":
    main()