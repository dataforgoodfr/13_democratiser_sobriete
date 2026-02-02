#!/usr/bin/env python3
"""
Labour Force Survey (LFS) Analysis - Construction and Real Estate Sectors

This script analyzes Labour Force Survey data to compare workers in:
- CONSTRUCTION sector (NACE 410-439)
- REAL ESTATE ACTIVITIES (NACE 681-683)
- Overall market

Focus: Luxembourg (first iteration)

Dimensions analyzed:
- Type of workers (sex, age, citizenship, migration, education)
- Employment status (self-employed, employee, full/part-time, permanency)
- Working situation (hours worked, second job)
- Health (illness absence, activity limitation)
- Income (gross monthly pay)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("Set2")

# ============================================================================
# CONFIGURATION
# ============================================================================

COUNTRY = 'FR'  # France
YEAR = 2023  # Latest year available

# NACE codes
# Using NACE2_1D (1-digit classification with letters)
# Section F = Construction, Section L = Real Estate Activities
CONSTRUCTION_NACE = 'F'  # Section F: Construction
REALESTATE_NACE = 'L'    # Section L: Real Estate Activities

CONSTRUCTION_LABEL = 'Construction'
REALESTATE_LABEL = 'Real Estate'
OVERALL_LABEL = 'Overall Market'

# Sector colors (matching oecd_analysis.py COMPONENT_COLORS)
COLORS = {
    CONSTRUCTION_LABEL: '#ffd558',
    REALESTATE_LABEL: '#fb8072',
    OVERALL_LABEL: '#b3de69'
}

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """Set up directory paths for LFS data."""
    EXTERNAL_DATA_DIR = Path(r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI")
    
    dirs = {
        'raw_data': EXTERNAL_DATA_DIR / "0_data" / "LFS" / "LFS_1983-2023_YEARLY_full_set-002" / "LFS_1983-2023_YEARLY_full_set",
        'output': Path(__file__).parent.parent / 'outputs' / 'graphs' / 'LFS'
    }
    
    dirs['output'].mkdir(parents=True, exist_ok=True)
    
    return dirs


def load_lfs_data(country_code, year, data_path):
    """Load LFS data for a specific country and year."""
    print(f"\n=== LOADING LFS DATA FOR {country_code} {year} ===")
    
    # Look for country folder: e.g., "LU_YEAR"
    country_folder = data_path / f"{country_code}_YEAR"
    
    if not country_folder.exists():
        print(f"ERROR: Country folder not found: {country_folder}")
        return pd.DataFrame()
    
    # Look for year file: e.g., "LU2023_y.csv"
    year_file = country_folder / f"{country_code}{year}_y.csv"
    
    if not year_file.exists():
        print(f"ERROR: Year file not found: {year_file}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(year_file)
        print(f"OK Loaded: {year_file.name}, Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:20]}")
        return df
    except Exception as e:
        print(f"ERROR loading {year_file}: {e}")
        return pd.DataFrame()


# ============================================================================
# DATA FILTERING AND PREPARATION
# ============================================================================

def filter_by_sector(df, nace_code):
    """Filter LFS data by NACE 1-digit code."""
    # Try NACE2_1D first (newer classification)
    if 'NACE2_1D' not in df.columns and 'NACE1_1D' not in df.columns:
        print("ERROR: No NACE column found (neither NACE2_1D nor NACE1_1D)")
        return pd.DataFrame()
    
    # Check which NACE column to use
    nace_column = None
    if 'NACE2_1D' in df.columns:
        df['NACE2_1D'] = df['NACE2_1D'].astype(str).str.strip()
        # Check if NACE2_1D has actual sector data (not just 9.0 and NaN)
        unique_vals = df['NACE2_1D'].unique()
        has_sector_data = any(val not in ['9.0', 'nan', '9', 'NaN'] for val in unique_vals)
        if has_sector_data:
            nace_column = 'NACE2_1D'
    
    # If NACE2_1D doesn't have sector data, try NACE1_1D (older classification)
    if nace_column is None and 'NACE1_1D' in df.columns:
        df['NACE1_1D'] = df['NACE1_1D'].astype(str).str.strip()
        nace_column = 'NACE1_1D'
        print(f"Using NACE1_1D (older classification) for sector filtering")
    
    if nace_column is None:
        print("ERROR: No valid NACE column with sector data found")
        return pd.DataFrame()
    
    # Filter by matching NACE code
    mask = df[nace_column] == nace_code
    filtered = df[mask].copy()
    
    print(f"Filtered {len(filtered)} records for NACE code: {nace_code} (using {nace_column})")
    return filtered


def filter_workers_only(df):
    """Filter to only include workers (exclude non-workers with NACE=9)."""
    # Try NACE2_1D first, then NACE1_1D
    nace_column = None
    if 'NACE2_1D' in df.columns:
        df['NACE2_1D'] = df['NACE2_1D'].astype(str).str.strip()
        unique_vals = df['NACE2_1D'].unique()
        has_sector_data = any(val not in ['9.0', 'nan', '9', 'NaN'] for val in unique_vals)
        if has_sector_data:
            nace_column = 'NACE2_1D'
    
    if nace_column is None and 'NACE1_1D' in df.columns:
        df['NACE1_1D'] = df['NACE1_1D'].astype(str).str.strip()
        nace_column = 'NACE1_1D'
    
    if nace_column is None:
        return df
    
    # Keep only actual workers (exclude NACE=9 which represents non-workers/not applicable)
    mask = (df[nace_column] != '9') & (df[nace_column] != '9.0') & (df[nace_column] != '') & (df[nace_column] != 'nan')
    return df[mask].copy()


def prepare_analysis_data(df):
    """Prepare data with all required columns and valid records."""
    print(f"\n=== PREPARING ANALYSIS DATA ===")
    
    # First, derive MIGSTAT if it doesn't exist
    if 'MIGSTAT' not in df.columns and 'COUNTRYB' in df.columns:
        df = derive_migstat(df)
    
    required_cols = {
        'SEX': 'Sex (1=Male, 2=Female)',
        'AGE': 'Age (years)',
        'CITIZENSHIP': 'Citizenship',
        'COUNTRYB': 'Country of birth',
        'MIGREAS': 'Migration reason',
        'MIGSTAT': 'Migration status',
        'HATLEVEL': 'Education level',
        'HATFIELD': 'Field of education',
        'EDUCFED12': 'Formal education/training last 12 months',
        'ISCO08_1D': 'Occupation (ISCO 1-digit)',
        'STAPRO': 'Employment status',
        'FTPT': 'Full-time/Part-time',
        'TEMP': 'Permanency',
        'TEMPAGCY': 'Temporary agency',
        'HWACTUAL': 'Hours worked (main job)',
        'HWACTU2J': 'Hours worked (second job)',
        'ABSILLINJ': 'Days absent (illness)',
        'GALI': 'Activity limitation',
        'INCDECIL': 'Income decile',
        'COEFFY': 'Household weight'
    }
    
    # Check available columns
    available_cols = [col for col in required_cols.keys() if col in df.columns]
    missing_cols = [col for col in required_cols.keys() if col not in df.columns]
    
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
    
    print(f"Available columns for analysis: {len(available_cols)}/{len(required_cols)}")
    
    return df[available_cols].copy() if available_cols else df.copy()


def derive_migstat(df):
    """Derive migration status (MIGSTAT) from country of birth data."""
    print("Deriving MIGSTAT from country of birth data...")
    
    df = df.copy()
    
    # Convert to string for comparison
    df['COUNTRYB'] = df['COUNTRYB'].astype(str).str.strip().str.upper()
    df['COBFATH'] = df['COBFATH'].astype(str).str.strip().str.upper()
    df['COBMOTH'] = df['COBMOTH'].astype(str).str.strip().str.upper()
    
    # Initialize MIGSTAT
    df['MIGSTAT'] = np.nan
    
    # Determine migration status based on country of birth and parents' country of birth
    # 0 = Native-born with both parents native-born
    # 1 = Native-born with one parent born abroad (second generation)
    # 2 = Native-born with both parents born abroad (second generation)
    # 3 = Foreign-born (first generation)
    # 4 = Unknown
    
    # First, identify natives vs foreign-born
    # "NAT" means native country
    native_birth = df['COUNTRYB'] == 'NAT'
    foreign_birth = ~native_birth & (df['COUNTRYB'] != 'NO ANSWER')
    unknown = df['COUNTRYB'] == 'NO ANSWER'
    
    # For foreign-born: MIGSTAT = 3 (first generation)
    df.loc[foreign_birth, 'MIGSTAT'] = 3
    
    # For native-born, check parents
    native_both_parents = (df['COBFATH'] == 'NAT') & (df['COBMOTH'] == 'NAT')
    native_one_parent = ((df['COBFATH'] == 'NAT') & (df['COBMOTH'] != 'NAT') & (df['COBMOTH'] != 'NO ANSWER')) | \
                        ((df['COBFATH'] != 'NAT') & (df['COBFATH'] != 'NO ANSWER') & (df['COBMOTH'] == 'NAT'))
    foreign_both_parents = (df['COBFATH'] != 'NAT') & (df['COBMOTH'] != 'NAT') & \
                           (df['COBFATH'] != 'NO ANSWER') & (df['COBMOTH'] != 'NO ANSWER')
    
    df.loc[native_birth & native_both_parents, 'MIGSTAT'] = 0
    df.loc[native_birth & native_one_parent, 'MIGSTAT'] = 1
    df.loc[native_birth & foreign_both_parents, 'MIGSTAT'] = 2
    df.loc[unknown, 'MIGSTAT'] = 4
    
    print(f"MIGSTAT derivation complete. Distribution:")
    print(df['MIGSTAT'].value_counts().sort_index())
    
    return df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_demographics(df_construction, df_realestate, df_overall):
    """Analyze demographic characteristics."""
    print("\n=== ANALYZING DEMOGRAPHICS ===")
    
    results = {}
    
    # SEX distribution
    if 'SEX' in df_construction.columns:
        print("\n--- SEX DISTRIBUTION ---")
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            sex_dist = df_sector['SEX'].value_counts(normalize=True) * 100
            print(f"{sector_name}: {sex_dist.to_dict()}")
            results[f"{sector_name}_sex"] = sex_dist
    
    # AGE distribution
    if 'AGE' in df_construction.columns:
        print("\n--- AGE DISTRIBUTION ---")
        age_bins = [0, 24, 34, 44, 54, 64, 120]
        age_labels = ['15-24', '25-34', '35-44', '45-54', '55-64', '65+']
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['age_group'] = pd.cut(df_sector['AGE'], bins=age_bins, labels=age_labels, right=False)
            age_dist = df_sector['age_group'].value_counts(normalize=True).sort_index() * 100
            print(f"{sector_name}: {age_dist.to_dict()}")
            results[f"{sector_name}_age"] = age_dist
    
    return results


def analyze_employment_status(df_construction, df_realestate, df_overall):
    """Analyze employment status."""
    print("\n=== ANALYZING EMPLOYMENT STATUS ===")
    
    results = {}
    
    # STAPRO (Status in employment)
    if 'STAPRO' in df_construction.columns:
        print("\n--- EMPLOYMENT STATUS (STAPRO) ---")
        stapro_labels = {
            1: 'Self-employed (with employees)',
            2: 'Self-employed (without employees)',
            3: 'Employee',
            4: 'Family worker'
        }
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['STAPRO_clean'] = pd.to_numeric(df_sector['STAPRO'], errors='coerce')
            stapro_dist = df_sector['STAPRO_clean'].value_counts(normalize=True) * 100
            print(f"{sector_name}: {stapro_dist.to_dict()}")
            results[f"{sector_name}_stapro"] = stapro_dist
    
    # FTPT (Full-time/Part-time)
    if 'FTPT' in df_construction.columns:
        print("\n--- FULL-TIME / PART-TIME (FTPT) ---")
        ftpt_labels = {1: 'Full-time', 2: 'Part-time'}
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['FTPT_clean'] = pd.to_numeric(df_sector['FTPT'], errors='coerce')
            ftpt_dist = df_sector['FTPT_clean'].value_counts(normalize=True) * 100
            print(f"{sector_name}: {ftpt_dist.to_dict()}")
            results[f"{sector_name}_ftpt"] = ftpt_dist
    
    return results


def analyze_hours_worked(df_construction, df_realestate, df_overall):
    """Analyze hours worked."""
    print("\n=== ANALYZING HOURS WORKED ===")
    
    results = {}
    
    if 'HWACTUAL' in df_construction.columns:
        print("\n--- ACTUAL HOURS WORKED (Main Job) ---")
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['HWACTUAL_clean'] = pd.to_numeric(df_sector['HWACTUAL'], errors='coerce')
            hwactual_valid = df_sector['HWACTUAL_clean'][df_sector['HWACTUAL_clean'] > 0]
            
            if len(hwactual_valid) > 0:
                print(f"{sector_name}: Mean={hwactual_valid.mean():.1f}h, Median={hwactual_valid.median():.1f}h")
                results[f"{sector_name}_hours"] = hwactual_valid.describe()
    
    return results


def analyze_income(df_construction, df_realestate, df_overall):
    """Analyze income deciles."""
    print("\n=== ANALYZING INCOME ===")
    
    results = {}
    
    if 'INCDECIL' in df_construction.columns:
        print("\n--- INCOME DECILE DISTRIBUTION ---")
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['INCDECIL_clean'] = pd.to_numeric(df_sector['INCDECIL'], errors='coerce')
            incdecil_dist = df_sector['INCDECIL_clean'].value_counts(normalize=True) * 100
            print(f"{sector_name}: {incdecil_dist.sort_index().to_dict()}")
            results[f"{sector_name}_income_decil"] = incdecil_dist
    
    return results


def analyze_sector_by_income_decile(df_overall, df_raw):
    """Analyze income decile distribution within each sector."""
    print("\n=== ANALYZING INCOME DECILE DISTRIBUTION BY SECTOR ===")
    
    results = {}
    
    if 'INCDECIL' not in df_raw.columns or 'NACE2_1D' not in df_raw.columns:
        print("WARNING: INCDECIL or NACE2_1D column not found")
        return results
    
    df_raw['INCDECIL_clean'] = pd.to_numeric(df_raw['INCDECIL'], errors='coerce')
    
    # Filter for valid income deciles (1-10, exclude 99=missing)
    df_analysis = df_raw[
        (df_raw['INCDECIL_clean'] >= 1) & 
        (df_raw['INCDECIL_clean'] <= 10)
    ].copy()
    
    # Apply weights if available
    weights_all = df_analysis['COEFFY'] if 'COEFFY' in df_analysis.columns else np.ones(len(df_analysis))
    
    print(f"\n--- INCOME DECILE DISTRIBUTION WITHIN SECTORS (Normalized) ---")
    print(f"Analyzing {len(df_analysis)} records with valid income deciles")
    
    # Calculate totals for each sector
    construction_mask_all = df_analysis['NACE2_1D'] == 'F'
    realestate_mask_all = df_analysis['NACE2_1D'] == 'L'
    
    construction_total_weight = weights_all[construction_mask_all].sum()
    realestate_total_weight = weights_all[realestate_mask_all].sum()
    
    decile_results = {}
    
    for decile in sorted(df_analysis['INCDECIL_clean'].unique()):
        if pd.isna(decile):
            continue
            
        decile_mask = df_analysis['INCDECIL_clean'] == decile
        
        # Construction workers in this decile
        construction_in_decile_mask = construction_mask_all & decile_mask
        construction_weight_in_decile = weights_all[construction_in_decile_mask].sum()
        construction_share = (construction_weight_in_decile / construction_total_weight) * 100 if construction_total_weight > 0 else 0
        
        # Real Estate workers in this decile
        realestate_in_decile_mask = realestate_mask_all & decile_mask
        realestate_weight_in_decile = weights_all[realestate_in_decile_mask].sum()
        realestate_share = (realestate_weight_in_decile / realestate_total_weight) * 100 if realestate_total_weight > 0 else 0
        
        decile_results[int(decile)] = {
            'construction': construction_share,
            'realestate': realestate_share,
            'construction_n': construction_in_decile_mask.sum(),
            'realestate_n': realestate_in_decile_mask.sum()
        }
        
        print(f"Decile {int(decile):2d}: Construction {construction_share:5.2f}% | Real Estate {realestate_share:5.2f}%")
    
    results['by_decile'] = decile_results
    return results


def analyze_construction_by_income_and_migration(df_construction, df_raw):
    """Analyze construction workers by income decile with migration status breakdown."""
    print("\n=== ANALYZING CONSTRUCTION WORKERS BY INCOME DECILE AND MIGRATION STATUS ===")
    
    results = {}
    
    if 'INCDECIL' not in df_raw.columns or 'NACE2_1D' not in df_raw.columns:
        print("WARNING: Required columns not found")
        return results
    
    # Derive MIGSTAT if not already present
    if 'MIGSTAT' not in df_raw.columns and 'COUNTRYB' in df_raw.columns:
        df_raw = derive_migstat(df_raw)
    
    df_raw['INCDECIL_clean'] = pd.to_numeric(df_raw['INCDECIL'], errors='coerce')
    df_raw['MIGSTAT_clean'] = pd.to_numeric(df_raw['MIGSTAT'], errors='coerce')
    
    # Filter for construction workers (NACE F) with valid income deciles
    df_construction_analysis = df_raw[
        (df_raw['NACE2_1D'] == 'F') &
        (df_raw['INCDECIL_clean'] >= 1) & 
        (df_raw['INCDECIL_clean'] <= 10)
    ].copy()
    
    if len(df_construction_analysis) == 0:
        print("WARNING: No construction workers with valid income deciles")
        return results
    
    weights_all = df_construction_analysis['COEFFY'] if 'COEFFY' in df_construction_analysis.columns else np.ones(len(df_construction_analysis))
    total_construction_weight = weights_all.sum()
    
    print(f"\nAnalyzing {len(df_construction_analysis)} construction records with valid income deciles")
    
    migration_labels = {
        0: 'Native',
        1: 'Second-gen (1 parent)',
        2: 'Second-gen (2 parents)',
        3: 'First-gen',
        4: 'Unknown'
    }
    
    decile_results = {}
    
    for decile in sorted(df_construction_analysis['INCDECIL_clean'].unique()):
        if pd.isna(decile):
            continue
        
        decile_mask = df_construction_analysis['INCDECIL_clean'] == decile
        decile_weight = weights_all[decile_mask].sum()
        decile_share = (decile_weight / total_construction_weight) * 100 if total_construction_weight > 0 else 0
        
        # Calculate migration status breakdown within this decile
        migration_breakdown = {}
        for migstat_code in sorted([0, 1, 2, 3, 4]):
            migstat_mask = (decile_mask) & (df_construction_analysis['MIGSTAT_clean'] == migstat_code)
            migstat_weight = weights_all[migstat_mask].sum()
            migstat_share_of_decile = (migstat_weight / decile_weight) * 100 if decile_weight > 0 else 0
            migration_breakdown[migstat_code] = migstat_share_of_decile
        
        decile_results[int(decile)] = {
            'total_share': decile_share,
            'migration_breakdown': migration_breakdown,
            'n_records': decile_mask.sum()
        }
        
        print(f"Decile {int(decile):2d}: {decile_share:5.2f}% of construction workers | Migration: {', '.join([f'{migration_labels[k]}: {v:.1f}%' for k, v in migration_breakdown.items() if v > 0])}")
    
    results['by_decile'] = decile_results
    return results


def analyze_migration_status(df_construction, df_realestate, df_overall):
    """Analyze migration status."""
    print("\n=== ANALYZING MIGRATION STATUS ===")
    
    results = {}
    
    if 'MIGSTAT' in df_construction.columns:
        print("\n--- MIGRATION STATUS (MIGSTAT) ---")
        migstat_labels = {
            0: 'Native-born both parents native',
            1: 'Native-born one parent abroad',
            2: 'Native-born both parents abroad',
            3: 'Foreign-born (1st generation)',
            4: 'Unknown',
            9: 'Not applicable'
        }
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['MIGSTAT_clean'] = pd.to_numeric(df_sector['MIGSTAT'], errors='coerce')
            migstat_dist = df_sector['MIGSTAT_clean'].value_counts(normalize=True) * 100
            print(f"{sector_name}: {migstat_dist.to_dict()}")
            results[f"{sector_name}_migstat"] = migstat_dist
    
    return results


def analyze_temp_agency_by_migration(df_construction, df_realestate, df_overall):
    """Analyze temporary employment agency contracts by migration status."""
    print("\n=== ANALYZING TEMPORARY AGENCY CONTRACTS BY MIGRATION STATUS ===")
    
    results = {}
    
    if 'TEMPAGCY' not in df_construction.columns or 'MIGSTAT' not in df_construction.columns:
        print("WARNING: TEMPAGCY or MIGSTAT column not found")
        return results
    
    # Define immigration backgrounds
    migration_groups = {
        'Native (both parents native)': [0],
        'Second-gen (one parent abroad)': [1],
        'Second-gen (both parents abroad)': [2],
        'First-gen (foreign-born)': [3]
    }
    
    print("\n--- TEMPORARY AGENCY SHARE BY MIGRATION STATUS ---")
    
    for sector_name, df_sector in [
        ('Construction', df_construction),
        ('Real Estate', df_realestate),
        ('Overall', df_overall)
    ]:
        print(f"\n{sector_name}:")
        df_sector['TEMPAGCY_clean'] = pd.to_numeric(df_sector['TEMPAGCY'], errors='coerce')
        df_sector['MIGSTAT_clean'] = pd.to_numeric(df_sector['MIGSTAT'], errors='coerce')
        
        sector_results = {}
        
        for group_name, migstat_codes in migration_groups.items():
            # Filter by migration status and valid TEMPAGCY codes
            mask = df_sector['MIGSTAT_clean'].isin(migstat_codes)
            df_migstat = df_sector[mask].copy()
            
            if len(df_migstat) == 0:
                sector_results[group_name] = np.nan
                print(f"  {group_name}: No data")
                continue
            
            # Filter to only valid TEMPAGCY codes (1=No, 2=Yes, excluding 9=not applicable and NaN)
            valid_mask = df_migstat['TEMPAGCY_clean'].isin([1, 2])
            df_valid = df_migstat[valid_mask].copy()
            
            if len(df_valid) > 0:
                # Calculate weighted share of "Yes" (code 2)
                weights = df_valid['COEFFY'] if 'COEFFY' in df_valid.columns else np.ones(len(df_valid))
                total_weight = weights.sum()
                yes_mask = df_valid['TEMPAGCY_clean'] == 2
                yes_weight = weights[yes_mask].sum()
                share_yes = (yes_weight / total_weight) * 100
                sector_results[group_name] = share_yes
                print(f"  {group_name}: {share_yes:.1f}% (n={len(df_valid)})")
            else:
                sector_results[group_name] = np.nan
                print(f"  {group_name}: No valid TEMPAGCY data")
        
        results[sector_name] = sector_results
    
    return results


def analyze_migration_breakdown_of_temp_workers(df_construction, df_realestate, df_overall):
    """Analyze migration status breakdown among temporary agency workers."""
    print("\n=== ANALYZING MIGRATION BREAKDOWN OF TEMPORARY AGENCY WORKERS ===")
    
    results = {}
    
    if 'TEMPAGCY' not in df_construction.columns or 'MIGSTAT' not in df_construction.columns:
        print("WARNING: TEMPAGCY or MIGSTAT column not found")
        return results
    
    print("\n--- MIGRATION STATUS AMONG TEMP AGENCY WORKERS (TEMPAGCY=2) ---")
    
    for sector_name, df_sector in [
        ('Construction', df_construction),
        ('Real Estate', df_realestate),
        ('Overall', df_overall)
    ]:
        print(f"\n{sector_name}:")
        df_sector['TEMPAGCY_clean'] = pd.to_numeric(df_sector['TEMPAGCY'], errors='coerce')
        df_sector['MIGSTAT_clean'] = pd.to_numeric(df_sector['MIGSTAT'], errors='coerce')
        
        # Filter for temporary agency workers (TEMPAGCY = 2)
        temp_workers = df_sector[df_sector['TEMPAGCY_clean'] == 2].copy()
        
        if len(temp_workers) == 0:
            print("  No temporary agency workers")
            results[sector_name] = {}
            continue
        
        weights = temp_workers['COEFFY'] if 'COEFFY' in temp_workers.columns else np.ones(len(temp_workers))
        total_weight = weights.sum()
        
        sector_results = {}
        
        # Count by migration status (only codes 0-3, exclude 4=Unknown and 9=Not applicable)
        for code in sorted([0, 1, 2, 3]):
            mask = temp_workers['MIGSTAT_clean'] == code
            count = mask.sum()
            weight = weights[mask].sum() if count > 0 else 0
            share = (weight / total_weight) * 100
            
            migration_labels = {
                0: 'Native (both parents native)',
                1: 'Second-gen (one parent abroad)',
                2: 'Second-gen (both parents abroad)',
                3: 'First-gen (foreign-born)'
            }
            
            sector_results[code] = share
            print(f"  {migration_labels[code]}: {share:.1f}% (n={count})")
        
        results[sector_name] = sector_results
    
    return results


def analyze_education_level(df_construction, df_realestate, df_overall):
    """Analyze educational attainment level."""
    print("\n=== ANALYZING EDUCATION LEVEL ===")
    
    results = {}
    
    if 'HATLEVEL' in df_construction.columns:
        print("\n--- EDUCATIONAL ATTAINMENT LEVEL (HATLEVEL) ---")
        hatlevel_labels = {
            1: 'Primary',
            2: 'Lower secondary',
            3: 'Upper secondary',
            4: 'Post-secondary',
            5: 'First tertiary',
            6: 'Second tertiary'
        }
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['HATLEVEL_clean'] = pd.to_numeric(df_sector['HATLEVEL'], errors='coerce')
            hatlevel_dist = df_sector['HATLEVEL_clean'].value_counts(normalize=True) * 100
            print(f"{sector_name}: {hatlevel_dist.to_dict()}")
            results[f"{sector_name}_hatlevel"] = hatlevel_dist
    
    return results


def map_isced_to_achievement_level(isced_code):
    """Map detailed ISCED codes to broader achievement levels."""
    if pd.isna(isced_code):
        return 'Unknown'
    
    isced_code = int(isced_code)
    
    # No formal education or below ISCED 1
    if isced_code in [0]:
        return 'No formal education'
    # ISCED 1 Primary education
    elif isced_code in [100]:
        return 'Primary'
    # ISCED 2 Lower secondary education
    elif isced_code in [200]:
        return 'Lower secondary'
    # ISCED 3 Upper secondary education (all variants)
    elif isced_code in [342, 343, 344, 349, 352, 353, 354, 359, 392, 393, 394, 399]:
        return 'Upper secondary'
    # ISCED 4 Post-secondary non-tertiary education
    elif isced_code in [440, 450, 490]:
        return 'Post-secondary'
    # ISCED 5 Short-cycle tertiary education
    elif isced_code in [540, 550, 590]:
        return 'Short-cycle tertiary'
    # ISCED 6 Bachelor's or equivalent level
    elif isced_code in [600]:
        return "Bachelor's"
    # ISCED 7 Master's or equivalent level
    elif isced_code in [700]:
        return "Master's"
    # ISCED 8 Doctoral or equivalent level
    elif isced_code in [800]:
        return 'Doctorate'
    else:
        return 'Unknown'


def analyze_education_achievement(df_construction, df_realestate, df_overall):
    """Analyze education achievement levels (grouped ISCED codes)."""
    print("\n=== ANALYZING EDUCATION ACHIEVEMENT LEVELS ===")
    
    results = {}
    
    if 'HATLEVEL' in df_construction.columns:
        print("\n--- EDUCATION ACHIEVEMENT LEVELS (Grouped ISCED) ---")
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['HATLEVEL_clean'] = pd.to_numeric(df_sector['HATLEVEL'], errors='coerce')
            df_sector['achievement_level'] = df_sector['HATLEVEL_clean'].apply(map_isced_to_achievement_level)
            achievement_dist = df_sector['achievement_level'].value_counts(normalize=True) * 100
            print(f"{sector_name}: {achievement_dist.to_dict()}")
            results[f"{sector_name}_achievement"] = achievement_dist
    
    return results


def analyze_permanency(df_construction, df_realestate, df_overall):
    """Analyze permanency of main job."""
    print("\n=== ANALYZING JOB PERMANENCY ===")
    
    results = {}
    
    if 'TEMP' in df_construction.columns:
        print("\n--- PERMANENCY OF MAIN JOB (TEMP) ---")
        temp_labels = {
            1: 'Permanent',
            2: 'Temporary'
        }
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['TEMP_clean'] = pd.to_numeric(df_sector['TEMP'], errors='coerce')
            temp_dist = df_sector['TEMP_clean'].value_counts(normalize=True) * 100
            print(f"{sector_name}: {temp_dist.to_dict()}")
            results[f"{sector_name}_temp"] = temp_dist
    
    return results


def analyze_temp_agency(df_construction, df_realestate, df_overall):
    """Analyze temporary employment agency contracts."""
    print("\n=== ANALYZING TEMPORARY AGENCY CONTRACTS ===")
    
    results = {}
    
    if 'TEMPAGCY' in df_construction.columns:
        print("\n--- TEMPORARY EMPLOYMENT AGENCY (TEMPAGCY) ---")
        tempagcy_labels = {
            1: 'No',
            2: 'Yes'
        }
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['TEMPAGCY_clean'] = pd.to_numeric(df_sector['TEMPAGCY'], errors='coerce')
            # Filter to only valid codes (1=No, 2=Yes, excluding 9=not applicable and NaN)
            valid_mask = df_sector['TEMPAGCY_clean'].isin([1, 2])
            df_valid = df_sector[valid_mask].copy()
            tempagcy_dist = weighted_value_counts(df_valid, 'TEMPAGCY_clean')
            print(f"{sector_name}: {tempagcy_dist.to_dict()}")
            results[f"{sector_name}_tempagcy"] = tempagcy_dist
    
    return results


def analyze_total_hours_worked(df_construction, df_realestate, df_overall):
    """Analyze total hours worked (main job + second job)."""
    print("\n=== ANALYZING TOTAL HOURS WORKED ===")
    
    results = {}
    
    if 'HWACTUAL' in df_construction.columns and 'HWACTU2J' in df_construction.columns:
        print("\n--- TOTAL HOURS WORKED (Main Job + Second Job) ---")
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['HWACTUAL_clean'] = pd.to_numeric(df_sector['HWACTUAL'], errors='coerce')
            df_sector['HWACTU2J_clean'] = pd.to_numeric(df_sector['HWACTU2J'], errors='coerce')
            df_sector['HWTOTAL'] = df_sector['HWACTUAL_clean'].fillna(0) + df_sector['HWACTU2J_clean'].fillna(0)
            
            hwtotal_valid = df_sector['HWTOTAL'][(df_sector['HWTOTAL'] > 0) & (df_sector['HWTOTAL'] < 150)]
            
            if len(hwtotal_valid) > 0:
                print(f"{sector_name}: Mean={hwtotal_valid.mean():.1f}h, Median={hwtotal_valid.median():.1f}h")
                results[f"{sector_name}_total_hours"] = hwtotal_valid.describe()
    
    return results


def analyze_health_limitation(df_construction, df_realestate, df_overall):
    """Analyze days of absence from main job due to illness."""
    print("\n=== ANALYZING ABSENCE DUE TO ILLNESS ===")
    
    results = {}
    
    if 'ABSILLINJ' in df_construction.columns:
        print("\n--- DAYS OF ABSENCE (ABSILLINJ) ---")
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['ABSILLINJ_clean'] = pd.to_numeric(df_sector['ABSILLINJ'], errors='coerce')
            
            # Filter valid data (0-7 days, excluding 9=not applicable)
            valid_absence = df_sector['ABSILLINJ_clean'][(df_sector['ABSILLINJ_clean'] >= 0) & (df_sector['ABSILLINJ_clean'] <= 7)]
            
            if len(valid_absence) > 0:
                mean_absence = valid_absence.mean()
                median_absence = valid_absence.median()
                print(f"{sector_name}: Mean={mean_absence:.2f} days, Median={median_absence:.1f} days, n={len(valid_absence)}")
                results[f"{sector_name}_absence"] = valid_absence.describe()
            else:
                print(f"{sector_name}: No valid absence data")
    
    return results


def analyze_absence_incidence(df_construction, df_realestate, df_overall):
    """Analyze share of persons absent due to illness in a week."""
    print("\n=== ANALYZING ABSENCE INCIDENCE (Share with any absence) ===")
    
    results = {}
    
    if 'ABSILLINJ' in df_construction.columns:
        print("\n--- SHARE OF PERSONS WITH ABSENCE DUE TO ILLNESS ---")
        
        for sector_name, df_sector in [
            ('Construction', df_construction),
            ('Real Estate', df_realestate),
            ('Overall', df_overall)
        ]:
            df_sector['ABSILLINJ_clean'] = pd.to_numeric(df_sector['ABSILLINJ'], errors='coerce')
            
            # Filter valid data (0-7 days, excluding 9=not applicable)
            valid_mask = (df_sector['ABSILLINJ_clean'] >= 0) & (df_sector['ABSILLINJ_clean'] <= 7)
            valid_data = df_sector[valid_mask].copy()
            
            if len(valid_data) > 0:
                # Calculate weighted statistics
                weights = valid_data['COEFFY'] if 'COEFFY' in valid_data.columns else np.ones(len(valid_data))
                total_weight = weights.sum()
                
                # Share with any absence (> 0 days)
                has_absence = valid_data['ABSILLINJ_clean'] > 0
                absence_weight = weights[has_absence].sum()
                share_with_absence = (absence_weight / total_weight) * 100
                
                # Average days for those with absence
                if has_absence.any():
                    avg_days_with_absence = valid_data.loc[has_absence, 'ABSILLINJ_clean'].mean()
                else:
                    avg_days_with_absence = 0
                
                print(f"{sector_name}: {share_with_absence:.1f}% had any absence (avg {avg_days_with_absence:.1f} days among those absent), n={len(valid_data)}")
                results[f"{sector_name}_absence_incidence"] = {
                    'share_with_absence': share_with_absence,
                    'avg_days_with_absence': avg_days_with_absence,
                    'total_count': len(valid_data)
                }
            else:
                print(f"{sector_name}: No valid absence data")
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def weighted_value_counts(df_sector, column):
    """Calculate weighted value counts."""
    weights_col = 'COEFFY' if 'COEFFY' in df_sector.columns else None
    if weights_col:
        grouped = df_sector.groupby(column)[weights_col].sum()
        return grouped / grouped.sum() * 100
    else:
        return df_sector[column].value_counts(normalize=True) * 100


def plot_sex_distribution(df_construction, df_realestate, df_overall, output_dir):
    """Plot sex distribution by sector with weighted data."""
    if 'SEX' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sectors_data = {
        CONSTRUCTION_LABEL: weighted_value_counts(df_construction, 'SEX'),
        REALESTATE_LABEL: weighted_value_counts(df_realestate, 'SEX'),
        OVERALL_LABEL: weighted_value_counts(df_overall, 'SEX')
    }
    
    x = np.arange(2)
    width = 0.25
    
    for idx, (sector, data) in enumerate(sectors_data.items()):
        values = [data.get(1, 0), data.get(2, 0)]  # Male, Female
        bars = ax.bar(x + idx*width, values, width, label=sector, color=COLORS.get(sector, '#8dd3c7'))
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Sex', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Sex Distribution by Sector - France 2023 (Weighted)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Male', 'Female'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max([max(data.values) for data in sectors_data.values()]) * 1.15)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_sex_distribution.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_age_distribution(df_construction, df_realestate, df_overall, output_dir):
    """Plot age distribution by sector."""
    if 'AGE' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    age_bins = [0, 24, 34, 44, 54, 64, 120]
    age_labels = ['15-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    for sector_name, df_sector, color in [
        (CONSTRUCTION_LABEL, df_construction, COLORS[CONSTRUCTION_LABEL]),
        (REALESTATE_LABEL, df_realestate, COLORS[REALESTATE_LABEL]),
        (OVERALL_LABEL, df_overall, COLORS[OVERALL_LABEL])
    ]:
        df_sector['age_group'] = pd.cut(df_sector['AGE'], bins=age_bins, labels=age_labels, right=False)
        age_dist = df_sector['age_group'].value_counts(normalize=True).sort_index() * 100
        ax.plot(age_labels, age_dist.values, marker='o', linewidth=2, label=sector_name, color=color)
    
    ax.set_xlabel('Age Group', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Age Distribution by Sector - Luxembourg 2023', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_age_distribution.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_employment_status(df_construction, df_realestate, df_overall, output_dir):
    """Plot employment status distribution."""
    if 'STAPRO' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stapro_labels = {1: 'Self-empl.\n(with emp.)', 2: 'Self-empl.\n(no emp.)', 3: 'Employee', 4: 'Family worker'}
    
    for sector_name, df_sector, color in [
        (CONSTRUCTION_LABEL, df_construction, COLORS[CONSTRUCTION_LABEL]),
        (REALESTATE_LABEL, df_realestate, COLORS[REALESTATE_LABEL]),
        (OVERALL_LABEL, df_overall, COLORS[OVERALL_LABEL])
    ]:
        df_sector['STAPRO_clean'] = pd.to_numeric(df_sector['STAPRO'], errors='coerce')
        stapro_dist = df_sector['STAPRO_clean'].value_counts(normalize=True) * 100
        
        values = [stapro_dist.get(i, 0) for i in [1, 2, 3, 4]]
        x_pos = np.arange(4) + (list([CONSTRUCTION_LABEL, REALESTATE_LABEL, OVERALL_LABEL]).index(sector_name)) * 0.25
        ax.bar(x_pos, values, width=0.25, label=sector_name, color=color)
    
    ax.set_xlabel('Employment Status', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Employment Status by Sector - Luxembourg 2023', fontsize=13, fontweight='bold')
    ax.set_xticks(np.arange(4) + 0.25)
    ax.set_xticklabels([stapro_labels[i] for i in [1, 2, 3, 4]])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_employment_status.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_fulltime_parttime(df_construction, df_realestate, df_overall, output_dir):
    """Plot full-time/part-time distribution with weighted data and labels."""
    if 'FTPT' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.25
    
    for idx, (sector_name, df_sector, color) in enumerate([
        (CONSTRUCTION_LABEL, df_construction, COLORS[CONSTRUCTION_LABEL]),
        (REALESTATE_LABEL, df_realestate, COLORS[REALESTATE_LABEL]),
        (OVERALL_LABEL, df_overall, COLORS[OVERALL_LABEL])
    ]):
        df_sector['FTPT_clean'] = pd.to_numeric(df_sector['FTPT'], errors='coerce')
        ftpt_dist = weighted_value_counts(df_sector, 'FTPT_clean')
        
        values = [ftpt_dist.get(1, 0), ftpt_dist.get(2, 0)]  # Full-time, Part-time
        bars = ax.bar(x + idx*width, values, width, label=sector_name, color=color)
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Employment Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Full-time vs Part-time by Sector - France 2023 (Weighted)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Full-time', 'Part-time'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_fulltime_parttime.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_hours_worked_distribution(df_construction, df_realestate, df_overall, output_dir):
    """Plot hours worked distribution (scatter dots with weighted median lines)."""
    if 'HWACTUAL' not in df_construction.columns:
        return
    
    def weighted_median(values, weights):
        """Calculate weighted median."""
        # Sort by values
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Calculate cumulative weights
        cumsum_weights = np.cumsum(sorted_weights)
        total_weight = cumsum_weights[-1]
        
        # Find index where cumulative weight exceeds 50%
        median_idx = np.searchsorted(cumsum_weights, total_weight * 0.5)
        return sorted_values[min(median_idx, len(sorted_values) - 1)]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x_positions = {'Construction': 1, 'Real Estate': 2, 'Overall': 3}
    medians = {}
    line_width = 0.35  # Width of each sector's horizontal line
    
    for sector_name, df_sector, x_pos, color in [
        (CONSTRUCTION_LABEL, df_construction, x_positions['Construction'], COLORS[CONSTRUCTION_LABEL]),
        (REALESTATE_LABEL, df_realestate, x_positions['Real Estate'], COLORS[REALESTATE_LABEL]),
        (OVERALL_LABEL, df_overall, x_positions['Overall'], COLORS[OVERALL_LABEL])
    ]:
        df_sector['HWACTUAL_clean'] = pd.to_numeric(df_sector['HWACTUAL'], errors='coerce')
        
        # Filter valid data
        valid_mask = (df_sector['HWACTUAL_clean'] > 0) & (df_sector['HWACTUAL_clean'] < 100)
        hwactual_valid = df_sector.loc[valid_mask, 'HWACTUAL_clean'].values
        weights_valid = df_sector.loc[valid_mask, 'COEFFY'].values if 'COEFFY' in df_sector.columns else np.ones_like(hwactual_valid)
        
        if len(hwactual_valid) > 0:
            # Add jitter to x-axis for better visibility
            jitter_x = np.random.normal(x_pos, 0.04, size=len(hwactual_valid))
            ax.scatter(jitter_x, hwactual_valid, alpha=0.4, s=30, color=color, label=f'{sector_name} (n={len(hwactual_valid)})')
            
            # Calculate weighted median and store it
            weighted_med = weighted_median(hwactual_valid, weights_valid)
            medians[sector_name] = (x_pos, weighted_med)
    
    # Plot horizontal lines for each sector (centered on sector)
    for sector_name, (x_pos, weighted_med) in medians.items():
        # Draw horizontal line centered on each sector
        line_start = x_pos - line_width
        line_end = x_pos + line_width
        ax.plot([line_start, line_end], [weighted_med, weighted_med], color='black', linestyle='--', linewidth=2.5, alpha=0.7, zorder=10)
        # Place text label above the line
        ax.text(x_pos, weighted_med + 3.5, f'Weighted Median: {int(weighted_med)}h', ha='center', fontsize=10, fontweight='bold', color='black')
    
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-2, 102)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Construction', 'Real Estate', 'Overall'], fontsize=11)
    ax.set_xlabel('Sector', fontsize=11, fontweight='bold')
    ax.set_ylabel('Hours Worked per Week', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Hours Worked by Sector - France 2023\n(Dashed lines show weighted median)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_hours_worked_distribution.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_migration_status(df_construction, df_realestate, df_overall, output_dir):
    """Plot migration status distribution with all categories."""
    if 'MIGSTAT' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    migstat_labels = {
        0: 'Native\n(both parents native)',
        1: 'Second-gen\n(one parent abroad)',
        2: 'Second-gen\n(both parents abroad)',
        3: 'First-gen\n(foreign-born)',
        4: 'Unknown',
        9: 'Not applicable'
    }
    
    # Get all MIGSTAT values across all sectors
    all_migstat = pd.concat([
        pd.to_numeric(df_construction['MIGSTAT'], errors='coerce'),
        pd.to_numeric(df_realestate['MIGSTAT'], errors='coerce'),
        pd.to_numeric(df_overall['MIGSTAT'], errors='coerce')
    ])
    migstat_codes = sorted([int(x) for x in all_migstat.dropna().unique() if x in migstat_labels.keys()])
    
    sectors_data = {}
    for sector_name, df_sector in [
        (CONSTRUCTION_LABEL, df_construction),
        (REALESTATE_LABEL, df_realestate),
        (OVERALL_LABEL, df_overall)
    ]:
        df_sector_copy = df_sector.copy()
        df_sector_copy['MIGSTAT_clean'] = pd.to_numeric(df_sector_copy['MIGSTAT'], errors='coerce')
        sector_dist = weighted_value_counts(df_sector_copy, 'MIGSTAT_clean')
        sectors_data[sector_name] = sector_dist
    
    x = np.arange(len(migstat_codes))
    width = 0.25
    
    for idx, (sector_name, data) in enumerate(sectors_data.items()):
        values = [data.get(code, 0) for code in migstat_codes]
        bars = ax.bar(x + idx*width, values, width, label=sector_name, color=COLORS.get(sector_name, '#8dd3c7'))
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:  # Only show labels for values > 0.5%
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Migration Status', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Migration Status Distribution by Sector - France 2023 (Weighted)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([migstat_labels.get(code, str(code)) for code in migstat_codes], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_migration_status.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_temp_agency_by_migration(df_construction, df_realestate, df_overall, output_dir):
    """Plot temporary agency contracts by migration status."""
    if 'TEMPAGCY' not in df_construction.columns or 'MIGSTAT' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Define immigration backgrounds
    migration_groups = {
        'Native\n(both parents native)': [0],
        'Second-gen\n(one parent abroad)': [1],
        'Second-gen\n(both parents abroad)': [2],
        'First-gen\n(foreign-born)': [3]
    }
    
    x = np.arange(len(migration_groups))
    width = 0.25
    
    sectors_list = [
        (CONSTRUCTION_LABEL, df_construction, COLORS[CONSTRUCTION_LABEL]),
        (REALESTATE_LABEL, df_realestate, COLORS[REALESTATE_LABEL]),
        (OVERALL_LABEL, df_overall, COLORS[OVERALL_LABEL])
    ]
    
    for idx, (sector_name, df_sector, color) in enumerate(sectors_list):
        df_sector['TEMPAGCY_clean'] = pd.to_numeric(df_sector['TEMPAGCY'], errors='coerce')
        df_sector['MIGSTAT_clean'] = pd.to_numeric(df_sector['MIGSTAT'], errors='coerce')
        
        values = []
        for group_name, migstat_codes in migration_groups.items():
            # Filter by migration status
            mask = df_sector['MIGSTAT_clean'].isin(migstat_codes)
            df_migstat = df_sector[mask].copy()
            
            if len(df_migstat) == 0:
                values.append(0)
                continue
            
            # Filter to only valid TEMPAGCY codes (1=No, 2=Yes, excluding 9=not applicable and NaN)
            valid_mask = df_migstat['TEMPAGCY_clean'].isin([1, 2])
            df_valid = df_migstat[valid_mask].copy()
            
            if len(df_valid) > 0:
                # Calculate weighted share of "Yes" (code 2)
                weights = df_valid['COEFFY'] if 'COEFFY' in df_valid.columns else np.ones(len(df_valid))
                total_weight = weights.sum()
                yes_mask = df_valid['TEMPAGCY_clean'] == 2
                yes_weight = weights[yes_mask].sum()
                share_yes = (yes_weight / total_weight) * 100
                values.append(share_yes)
            else:
                values.append(0)
        
        bars = ax.bar(x + idx*width, values, width, label=sector_name, color=color)
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Migration Background', fontsize=11, fontweight='bold')
    ax.set_ylabel('Share with Temporary Agency Contract (%)', fontsize=11, fontweight='bold')
    ax.set_title('Share of Workers with Temporary Agency Contracts by Migration Status - France 2023 (Weighted)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(migration_groups.keys(), fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max([10, 15]))  # Set reasonable y-axis limit
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_temp_agency_by_migration.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_migration_breakdown_of_temp_workers(df_construction, df_realestate, df_overall, output_dir):
    """Plot migration status breakdown among temporary agency workers."""
    if 'TEMPAGCY' not in df_construction.columns or 'MIGSTAT' not in df_construction.columns:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    migration_labels = {
        0: 'Native\n(both parents native)',
        1: 'Second-gen\n(one parent abroad)',
        2: 'Second-gen\n(both parents abroad)',
        3: 'First-gen\n(foreign-born)'
    }
    
    sectors_list = [
        (CONSTRUCTION_LABEL, df_construction, COLORS[CONSTRUCTION_LABEL]),
        (REALESTATE_LABEL, df_realestate, COLORS[REALESTATE_LABEL]),
        (OVERALL_LABEL, df_overall, COLORS[OVERALL_LABEL])
    ]
    
    for ax_idx, (sector_name, df_sector, color) in enumerate(sectors_list):
        ax = axes[ax_idx]
        
        df_sector['TEMPAGCY_clean'] = pd.to_numeric(df_sector['TEMPAGCY'], errors='coerce')
        df_sector['MIGSTAT_clean'] = pd.to_numeric(df_sector['MIGSTAT'], errors='coerce')
        
        # Filter for temporary agency workers (TEMPAGCY = 2)
        temp_workers = df_sector[df_sector['TEMPAGCY_clean'] == 2].copy()
        
        if len(temp_workers) == 0:
            ax.text(0.5, 0.5, 'No temporary\nagency workers', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f'{sector_name}', fontsize=12, fontweight='bold')
            continue
        
        weights = temp_workers['COEFFY'] if 'COEFFY' in temp_workers.columns else np.ones(len(temp_workers))
        total_weight = weights.sum()
        
        values = []
        labels = []
        
        for code in sorted([0, 1, 2, 3]):
            mask = temp_workers['MIGSTAT_clean'] == code
            weight = weights[mask].sum() if mask.sum() > 0 else 0
            share = (weight / total_weight) * 100
            values.append(share)
            labels.append(migration_labels[code])
        
        # Create bar chart
        bars = ax.bar(labels, values, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Share (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'{sector_name}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    fig.suptitle('Migration Status Breakdown Among Temporary Agency Workers - France 2023 (Weighted)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_migration_breakdown_of_temp_workers.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_education_level(df_construction, df_realestate, df_overall, output_dir):
    """Plot education level distribution (top 8 categories)."""
    if 'HATLEVEL' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    sectors_list = [
        (CONSTRUCTION_LABEL, df_construction, COLORS[CONSTRUCTION_LABEL]),
        (REALESTATE_LABEL, df_realestate, COLORS[REALESTATE_LABEL]),
        (OVERALL_LABEL, df_overall, COLORS[OVERALL_LABEL])
    ]
    
    # Get top 8 education codes across all sectors
    all_education = pd.concat([df_construction['HATLEVEL'], df_realestate['HATLEVEL'], df_overall['HATLEVEL']])
    top_codes = all_education.value_counts().head(8).index.tolist()
    top_codes = sorted(top_codes)
    
    x = np.arange(len(top_codes))
    width = 0.25
    
    for idx, (sector_name, df_sector, color) in enumerate(sectors_list):
        df_sector['HATLEVEL_clean'] = pd.to_numeric(df_sector['HATLEVEL'], errors='coerce')
        sector_dist = df_sector['HATLEVEL_clean'].value_counts(normalize=True) * 100
        values = [sector_dist.get(code, 0) for code in top_codes]
        ax.bar(x + idx*width, values, width, label=sector_name, color=color)
    
    ax.set_xlabel('Education Level (ISCED Code)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Educational Attainment by Sector - France 2023 (Top 8 Categories)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    labels = [str(int(code)) for code in top_codes]
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_education_level.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_education_achievement(df_construction, df_realestate, df_overall, output_dir):
    """Plot education achievement levels (grouped ISCED codes) with weighted data and labels."""
    if 'HATLEVEL' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    sectors_list = [
        (CONSTRUCTION_LABEL, df_construction, COLORS[CONSTRUCTION_LABEL]),
        (REALESTATE_LABEL, df_realestate, COLORS[REALESTATE_LABEL]),
        (OVERALL_LABEL, df_overall, COLORS[OVERALL_LABEL])
    ]
    
    # Get all achievement levels
    all_achievement = []
    for _, df_sector, _ in sectors_list:
        df_sector['HATLEVEL_clean'] = pd.to_numeric(df_sector['HATLEVEL'], errors='coerce')
        df_sector['achievement_level'] = df_sector['HATLEVEL_clean'].apply(map_isced_to_achievement_level)
        all_achievement.append(df_sector['achievement_level'])
    
    all_achievement_combined = pd.concat(all_achievement)
    achievement_order = ['No formal education', 'Primary', 'Lower secondary', 'Upper secondary', 
                        'Post-secondary', 'Short-cycle tertiary', "Bachelor's", "Master's", 'Doctorate']
    # Filter to only those that exist in the data
    achievement_order = [a for a in achievement_order if a in all_achievement_combined.unique()]
    
    x = np.arange(len(achievement_order))
    width = 0.25
    
    for idx, (sector_name, df_sector, color) in enumerate(sectors_list):
        df_sector['HATLEVEL_clean'] = pd.to_numeric(df_sector['HATLEVEL'], errors='coerce')
        df_sector['achievement_level'] = df_sector['HATLEVEL_clean'].apply(map_isced_to_achievement_level)
        sector_dist = weighted_value_counts(df_sector, 'achievement_level')
        values = [sector_dist.get(level, 0) for level in achievement_order]
        bars = ax.bar(x + idx*width, values, width, label=sector_name, color=color)
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 1:  # Only show labels for values > 1%
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Education Achievement Level', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Education Achievement by Sector - France 2023 (Weighted)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(achievement_order, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_education_achievement.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_permanency(df_construction, df_realestate, df_overall, output_dir):
    """Plot job permanency distribution with weighted data and labels."""
    if 'TEMP' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dfs = {
        CONSTRUCTION_LABEL: (df_construction, COLORS[CONSTRUCTION_LABEL]),
        REALESTATE_LABEL: (df_realestate, COLORS[REALESTATE_LABEL]),
        OVERALL_LABEL: (df_overall, COLORS[OVERALL_LABEL])
    }
    
    x = np.arange(2)
    width = 0.25
    
    for idx, (sector, (df_sector, color)) in enumerate(dfs.items()):
        df_sector['TEMP_clean'] = pd.to_numeric(df_sector['TEMP'], errors='coerce')
        data = weighted_value_counts(df_sector, 'TEMP_clean')
        values = [data.get(1, 0), data.get(2, 0)]
        bars = ax.bar(x + idx*width, values, width, label=sector, color=color)
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Job Permanency', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Job Permanency by Sector - France 2023 (Weighted)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Permanent', 'Temporary'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_permanency.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_temp_agency(df_construction, df_realestate, df_overall, output_dir):
    """Plot temporary employment agency distribution with weighted data and labels."""
    if 'TEMPAGCY' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dfs = {
        CONSTRUCTION_LABEL: (df_construction, COLORS[CONSTRUCTION_LABEL]),
        REALESTATE_LABEL: (df_realestate, COLORS[REALESTATE_LABEL]),
        OVERALL_LABEL: (df_overall, COLORS[OVERALL_LABEL])
    }
    
    x = np.arange(2)
    width = 0.25
    
    for idx, (sector, (df_sector, color)) in enumerate(dfs.items()):
        df_sector['TEMPAGCY_clean'] = pd.to_numeric(df_sector['TEMPAGCY'], errors='coerce')
        # Filter to only valid codes (1=No, 2=Yes, excluding 9=not applicable and NaN)
        valid_mask = df_sector['TEMPAGCY_clean'].isin([1, 2])
        df_valid = df_sector[valid_mask].copy()
        data = weighted_value_counts(df_valid, 'TEMPAGCY_clean')
        values = [data.get(2, 0), data.get(1, 0)]  # Yes (code 2), No (code 1)
        bars = ax.bar(x + idx*width, values, width, label=sector, color=color)
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Temporary Agency Contract', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Temporary Agency Contracts by Sector - France 2023 (Weighted)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Yes', 'No'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_temp_agency.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_total_hours_worked(df_construction, df_realestate, df_overall, output_dir):
    """Plot total hours worked (main + second job) distribution."""
    if 'HWACTUAL' not in df_construction.columns or 'HWACTU2J' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for sector_name, df_sector, color in [
        (CONSTRUCTION_LABEL, df_construction, COLORS[CONSTRUCTION_LABEL]),
        (REALESTATE_LABEL, df_realestate, COLORS[REALESTATE_LABEL]),
        (OVERALL_LABEL, df_overall, COLORS[OVERALL_LABEL])
    ]:
        df_sector['HWACTUAL_clean'] = pd.to_numeric(df_sector['HWACTUAL'], errors='coerce')
        df_sector['HWACTU2J_clean'] = pd.to_numeric(df_sector['HWACTU2J'], errors='coerce')
        df_sector['HWTOTAL'] = df_sector['HWACTUAL_clean'].fillna(0) + df_sector['HWACTU2J_clean'].fillna(0)
        
        hwtotal_valid = df_sector['HWTOTAL'][(df_sector['HWTOTAL'] > 0) & (df_sector['HWTOTAL'] < 150)]
        
        if len(hwtotal_valid) > 0:
            ax.hist(hwtotal_valid, bins=30, alpha=0.5, label=sector_name, color=color)
    
    ax.set_xlabel('Total Hours Worked per Week (Main + Second Job)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Workers', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Total Hours Worked - Luxembourg 2023', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_total_hours_worked.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_health_limitation(df_construction, df_realestate, df_overall, output_dir):
    """Plot days of absence from main job due to illness (ABSILLINJ)."""
    if 'ABSILLINJ' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for sector_name, df_sector, color in [
        (CONSTRUCTION_LABEL, df_construction, COLORS[CONSTRUCTION_LABEL]),
        (REALESTATE_LABEL, df_realestate, COLORS[REALESTATE_LABEL]),
        (OVERALL_LABEL, df_overall, COLORS[OVERALL_LABEL])
    ]:
        df_sector['ABSILLINJ_clean'] = pd.to_numeric(df_sector['ABSILLINJ'], errors='coerce')
        # Filter valid data (0-7 days)
        valid_absence = df_sector['ABSILLINJ_clean'][(df_sector['ABSILLINJ_clean'] >= 0) & (df_sector['ABSILLINJ_clean'] <= 7)]
        
        if len(valid_absence) > 0:
            ax.hist(valid_absence, bins=15, alpha=0.5, label=f'{sector_name} (n={len(valid_absence)})', color=color, density=True)
    
    ax.set_xlabel('Days of Absence from Main Job (due to illness/injury)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Absence Days Due to Illness - France 2023', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_health_limitation.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_absence_incidence(df_construction, df_realestate, df_overall, output_dir):
    """Plot share of persons with any absence due to illness in a week with weighted data and labels."""
    if 'ABSILLINJ' not in df_construction.columns:
        return
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    sectors = [
        (CONSTRUCTION_LABEL, df_construction, COLORS[CONSTRUCTION_LABEL]),
        (REALESTATE_LABEL, df_realestate, COLORS[REALESTATE_LABEL]),
        (OVERALL_LABEL, df_overall, COLORS[OVERALL_LABEL])
    ]
    
    absence_shares = []
    sector_names = []
    
    for sector_name, df_sector, color in sectors:
        df_sector['ABSILLINJ_clean'] = pd.to_numeric(df_sector['ABSILLINJ'], errors='coerce')
        
        # Filter valid data (0-7 days, excluding 9=not applicable)
        valid_mask = (df_sector['ABSILLINJ_clean'] >= 0) & (df_sector['ABSILLINJ_clean'] <= 7)
        valid_data = df_sector[valid_mask].copy()
        
        if len(valid_data) > 0:
            # Calculate weighted statistics
            weights = valid_data['COEFFY'] if 'COEFFY' in valid_data.columns else np.ones(len(valid_data))
            total_weight = weights.sum()
            
            # Share with any absence (> 0 days)
            has_absence = valid_data['ABSILLINJ_clean'] > 0
            absence_weight = weights[has_absence].sum()
            share_with_absence = (absence_weight / total_weight) * 100
            
            absence_shares.append(share_with_absence)
            sector_names.append(sector_name)
    
    # Create bar chart
    x = np.arange(len(sector_names))
    bars = ax.bar(x, absence_shares, color=[COLORS.get(s, '#8dd3c7') for s in sector_names], width=0.6)
    
    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Share of Workers (%)', fontsize=11, fontweight='bold')
    ax.set_title('Share of Persons Absent Due to Illness in a Week - France 2023 (Weighted)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sector_names, fontsize=11)
    ax.set_ylim(0, max(absence_shares) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_absence_incidence.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_sector_by_income_decile(df_raw, output_dir):
    """Plot income decile distribution within each sector (normalized)."""
    if 'INCDECIL' not in df_raw.columns or 'NACE2_1D' not in df_raw.columns:
        return
    
    df_raw['INCDECIL_clean'] = pd.to_numeric(df_raw['INCDECIL'], errors='coerce')
    
    # Filter for valid income deciles (1-10, exclude 99=missing)
    df_analysis = df_raw[
        (df_raw['INCDECIL_clean'] >= 1) & 
        (df_raw['INCDECIL_clean'] <= 10)
    ].copy()
    
    if len(df_analysis) == 0:
        return
    
    # Apply weights if available
    weights_all = df_analysis['COEFFY'] if 'COEFFY' in df_analysis.columns else np.ones(len(df_analysis))
    
    # Calculate totals for each sector
    construction_mask_all = df_analysis['NACE2_1D'] == 'F'
    realestate_mask_all = df_analysis['NACE2_1D'] == 'L'
    
    construction_total_weight = weights_all[construction_mask_all].sum()
    realestate_total_weight = weights_all[realestate_mask_all].sum()
    
    # Calculate decile distribution
    decile_data = {}
    
    for decile in sorted(df_analysis['INCDECIL_clean'].unique()):
        if pd.isna(decile):
            continue
        
        decile_mask = df_analysis['INCDECIL_clean'] == decile
        
        # Construction workers in this decile (as % of all construction workers)
        construction_in_decile_mask = construction_mask_all & decile_mask
        construction_weight_in_decile = weights_all[construction_in_decile_mask].sum()
        construction_share = (construction_weight_in_decile / construction_total_weight) * 100 if construction_total_weight > 0 else 0
        
        # Real Estate workers in this decile (as % of all real estate workers)
        realestate_in_decile_mask = realestate_mask_all & decile_mask
        realestate_weight_in_decile = weights_all[realestate_in_decile_mask].sum()
        realestate_share = (realestate_weight_in_decile / realestate_total_weight) * 100 if realestate_total_weight > 0 else 0
        
        decile_data[int(decile)] = {
            'construction': construction_share,
            'realestate': realestate_share
        }
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 7))
    
    deciles = sorted(decile_data.keys())
    construction_values = [decile_data[d]['construction'] for d in deciles]
    realestate_values = [decile_data[d]['realestate'] for d in deciles]
    
    x = np.arange(len(deciles))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, construction_values, width, label=CONSTRUCTION_LABEL, 
                   color=COLORS[CONSTRUCTION_LABEL], edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, realestate_values, width, label=REALESTATE_LABEL,
                   color=COLORS[REALESTATE_LABEL], edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.3:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Income Decile (1=lowest, 10=highest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Sector Workers (%)', fontsize=12, fontweight='bold')
    ax.set_title('Income Distribution of Workers by Sector - France 2023 (Weighted)\n(% of Construction or Real Estate workers in each decile)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(construction_values), max(realestate_values)) * 1.15)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_sector_by_income_decile.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_construction_by_income_and_migration(df_construction, df_raw, output_dir):
    """Plot construction workers by income decile with migration status stacked bars."""
    if 'INCDECIL' not in df_raw.columns or 'NACE2_1D' not in df_raw.columns:
        return
    
    # Derive MIGSTAT if not already present
    if 'MIGSTAT' not in df_raw.columns and 'COUNTRYB' in df_raw.columns:
        df_raw = derive_migstat(df_raw)
    
    if 'MIGSTAT' not in df_raw.columns:
        return
    
    df_raw['INCDECIL_clean'] = pd.to_numeric(df_raw['INCDECIL'], errors='coerce')
    df_raw['MIGSTAT_clean'] = pd.to_numeric(df_raw['MIGSTAT'], errors='coerce')
    
    # Filter for construction workers (NACE F) with valid income deciles
    df_analysis = df_raw[
        (df_raw['NACE2_1D'] == 'F') &
        (df_raw['INCDECIL_clean'] >= 1) & 
        (df_raw['INCDECIL_clean'] <= 10)
    ].copy()
    
    if len(df_analysis) == 0:
        return
    
    weights_all = df_analysis['COEFFY'] if 'COEFFY' in df_analysis.columns else np.ones(len(df_analysis))
    total_construction_weight = weights_all.sum()
    
    # Migration status colors (matching oecd_analysis.py)
    migration_colors = {
        0: '#377eb8',  # Native
        1: '#984ea3',  # Second-gen 1 parent
        2: '#ff7f00',  # Second-gen 2 parents
        3: '#e41a1c',  # First-gen
        4: '#999999'   # Unknown
    }
    
    migration_labels_dict = {
        0: 'Native',
        1: 'Second-gen\n(1 parent)',
        2: 'Second-gen\n(2 parents)',
        3: 'First-gen',
        4: 'Unknown'
    }
    
    # Calculate decile and migration breakdown
    decile_data = {}
    
    for decile in sorted(df_analysis['INCDECIL_clean'].unique()):
        if pd.isna(decile):
            continue
        
        decile_mask = df_analysis['INCDECIL_clean'] == decile
        decile_weight = weights_all[decile_mask].sum()
        decile_share = (decile_weight / total_construction_weight) * 100 if total_construction_weight > 0 else 0
        
        # Calculate migration status breakdown
        migration_breakdown = {}
        for migstat_code in [0, 1, 2, 3, 4]:
            migstat_mask = decile_mask & (df_analysis['MIGSTAT_clean'] == migstat_code)
            migstat_weight = weights_all[migstat_mask].sum()
            migstat_share_of_total = (migstat_weight / total_construction_weight) * 100 if total_construction_weight > 0 else 0
            migration_breakdown[migstat_code] = migstat_share_of_total
        
        decile_data[int(decile)] = migration_breakdown
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    deciles = sorted(decile_data.keys())
    x = np.arange(len(deciles))
    
    # Prepare data for stacked bar
    migration_codes = [0, 1, 2, 3, 4]
    bottom = np.zeros(len(deciles))
    
    for migstat_code in migration_codes:
        values = [decile_data[d].get(migstat_code, 0) for d in deciles]
        
        # Only plot if there's data for this migration status
        if sum(values) > 0:
            ax.bar(x, values, bottom=bottom, label=migration_labels_dict[migstat_code],
                   color=migration_colors[migstat_code], edgecolor='white', linewidth=1.5)
            bottom += np.array(values)
    
    # Add total percentage labels on top of each bar
    for i, decile in enumerate(deciles):
        total = sum([decile_data[decile].get(code, 0) for code in migration_codes])
        ax.text(i, total + 0.3, f'{total:.1f}%', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Income Decile (1=lowest, 10=highest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Construction Workers (%)', fontsize=12, fontweight='bold')
    ax.set_title('Construction Workers by Income Decile and Migration Status - France 2023 (Weighted)\n(Stacked bar shows % of total construction workers in each decile, colors show migration breakdown)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(bottom) * 1.12)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_construction_by_income_and_migration.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_construction_by_income_and_sex(df_construction, df_raw, output_dir):
    """Plot construction workers by income decile and sex (stacked bar chart)."""
    print("\n=== ANALYZING CONSTRUCTION BY INCOME AND SEX ===")
    
    # Clean income deciles
    df_raw['INCDECIL_clean'] = pd.to_numeric(df_raw['INCDECIL'], errors='coerce')
    df_raw['SEX_clean'] = pd.to_numeric(df_raw['SEX'], errors='coerce')
    
    # Filter for construction workers (NACE F) with valid income deciles
    df_analysis = df_raw[
        (df_raw['NACE2_1D'] == 'F') &
        (df_raw['INCDECIL_clean'] >= 1) & 
        (df_raw['INCDECIL_clean'] <= 10) &
        (df_raw['SEX_clean'].isin([1, 2]))
    ].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid data for construction by income and sex")
        return
    
    weights_all = df_analysis['COEFFY'] if 'COEFFY' in df_analysis.columns else np.ones(len(df_analysis))
    total_construction_weight = weights_all.sum()
    
    # Sex colors (matching oecd_analysis.py - non-gendered colors)
    sex_colors = {
        1: '#fdb462',  # Male 
        2: '#b3de69'   # Female 
    }
    
    sex_labels_dict = {
        1: 'Male',
        2: 'Female'
    }
    
    # Calculate decile and sex breakdown
    decile_data = {}
    
    for decile in sorted(df_analysis['INCDECIL_clean'].unique()):
        if pd.isna(decile):
            continue
        
        decile_mask = df_analysis['INCDECIL_clean'] == decile
        
        # Calculate sex breakdown
        sex_breakdown = {}
        for sex_code in [1, 2]:
            sex_mask = decile_mask & (df_analysis['SEX_clean'] == sex_code)
            sex_weight = weights_all[sex_mask].sum()
            sex_share_of_total = (sex_weight / total_construction_weight) * 100 if total_construction_weight > 0 else 0
            sex_breakdown[sex_code] = sex_share_of_total
        
        decile_data[int(decile)] = sex_breakdown
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    deciles = sorted(decile_data.keys())
    x = np.arange(len(deciles))
    
    # Prepare data for stacked bar
    sex_codes = [1, 2]
    bottom = np.zeros(len(deciles))
    
    for sex_code in sex_codes:
        values = [decile_data[d].get(sex_code, 0) for d in deciles]
        
        # Only plot if there's data for this sex
        if sum(values) > 0:
            ax.bar(x, values, bottom=bottom, label=sex_labels_dict[sex_code],
                   color=sex_colors[sex_code], edgecolor='white', linewidth=1.5)
            bottom += np.array(values)
    
    # Add total percentage labels on top of each bar
    for i, decile in enumerate(deciles):
        total = sum([decile_data[decile].get(code, 0) for code in sex_codes])
        ax.text(i, total + 0.3, f'{total:.1f}%', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Income Decile (1=lowest, 10=highest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Construction Workers (%)', fontsize=12, fontweight='bold')
    ax.set_title('Construction Workers by Income Decile and Sex - France 2023 (Weighted)\n(Stacked bar shows % of total construction workers in each decile, colors show sex breakdown)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(bottom) * 1.12)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_construction_by_income_and_sex.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()
    
    # Also create percentage stacked version
    print("Creating percentage stacked version...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    deciles = sorted(decile_data.keys())
    x = np.arange(len(deciles))
    
    # Calculate percentages within each decile
    decile_percentages = {}
    for decile in deciles:
        total = sum([decile_data[decile].get(code, 0) for code in sex_codes])
        if total > 0:
            decile_percentages[decile] = {code: (decile_data[decile].get(code, 0) / total) * 100 for code in sex_codes}
        else:
            decile_percentages[decile] = {code: 0 for code in sex_codes}
    
    # Prepare data for percentage stacked bar
    bottom = np.zeros(len(deciles))
    
    for sex_code in sex_codes:
        values = [decile_percentages[d].get(sex_code, 0) for d in deciles]
        
        if sum(values) > 0:
            bars = ax.bar(x, values, bottom=bottom, label=sex_labels_dict[sex_code],
                   color=sex_colors[sex_code], edgecolor='white', linewidth=1.5)
            
            # Add percentage labels inside bars if space allows
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 5:  # Only show label if bar is tall enough
                    ax.text(bar.get_x() + bar.get_width()/2., bottom[i] + height/2,
                           f'{height:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
            
            bottom += np.array(values)
    
    ax.set_xlabel('Income Decile (1=lowest, 10=highest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Construction Workers Sex Distribution by Income Decile - France 2023 (Weighted)\n(100% stacked bar shows sex distribution within each decile)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_construction_by_income_and_sex_stacked_pct.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_construction_age_pyramid_comparison(country_code, data_path, output_dir):
    """Plot age pyramid for construction workers comparing years 2023, 2013, 2005."""
    print("\n=== CREATING CONSTRUCTION AGE PYRAMID COMPARISON ===")
    
    years = [2005, 2013, 2023]
    age_bins = [15, 25, 35, 45, 55, 65, 100]
    age_labels = ['15-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    year_data = {}
    
    for year in years:
        # Load data
        df = load_lfs_data(country_code, year, data_path)
        if df.empty:
            print(f"WARNING: Could not load data for {country_code} {year}")
            continue
        
        # Filter for construction workers
        df_construction = filter_by_sector(df, CONSTRUCTION_NACE)
        
        if 'AGE' not in df_construction.columns:
            print(f"WARNING: AGE column not available for {year}")
            continue
        
        # Create age groups
        df_construction['age_group'] = pd.cut(df_construction['AGE'], bins=age_bins, labels=age_labels, right=False)
        
        # Calculate weighted distribution
        age_dist = weighted_value_counts(df_construction, 'age_group')
        
        # Store data, ensuring all age groups are represented
        year_data[year] = {label: age_dist.get(label, 0) for label in age_labels}
        
        print(f"{year}: {year_data[year]}")
    
    if len(year_data) == 0:
        print("ERROR: No data available for age pyramid comparison")
        return
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(age_labels))
    width = 0.25
    
    year_colors = {
        2023: '#e41a1c',  # Red (matching oecd_analysis.py)
        2013: '#377eb8',  # Blue
        2005: '#4daf4a'   # Green
    }
    
    for idx, year in enumerate(sorted(year_data.keys())):
        if year in year_data:
            values = [year_data[year][label] for label in age_labels]
            offset = (idx - 1) * width  # Center the bars
            bars = ax.bar(x + offset, values, width, label=str(year), color=year_colors.get(year, '#95A5A6'))
            
            # Add percentage labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Construction Workers (%)', fontsize=12, fontweight='bold')
    ax.set_title('Age Distribution in Construction Sector - Comparison Across Years (Weighted)\nFrance: 2005, 2013, 2023',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(age_labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, title='Year', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limit
    all_values = [v for year_dict in year_data.values() for v in year_dict.values() if not np.isnan(v) and not np.isinf(v)]
    if len(all_values) > 0:
        ax.set_ylim(0, max(all_values) * 1.15)
    else:
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_construction_age_pyramid_comparison.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def derive_occupation_major_group(df):
    """Derive occupation major group from ISCO08_1D (1-digit level)."""
    if 'ISCO08_1D' not in df.columns:
        print("WARNING: ISCO08_1D column not found")
        return df
    
    df = df.copy()
    
    # Convert ISCO08_1D to integer, handle NaN and not applicable (99)
    df['ISCO08_1D_clean'] = pd.to_numeric(df['ISCO08_1D'], errors='coerce')
    
    # Map 10-level code to single digit (10 -> 1, 20 -> 2, etc.)
    # Exclude 99 (not applicable) and NaN
    df['OCCUPATION_MAJOR'] = df['ISCO08_1D_clean'] // 10
    df.loc[df['ISCO08_1D_clean'] == 99, 'OCCUPATION_MAJOR'] = np.nan
    
    # Map to occupation labels
    occupation_labels = {
        1: 'Managers',
        2: 'Professionals',
        3: 'Technicians',
        4: 'Clerical Support',
        5: 'Service & Sales',
        6: 'Skilled Agricultural',
        7: 'Craft & Trades',
        8: 'Plant & Machine Operators',
        9: 'Elementary Occupations'
    }
    
    df['OCCUPATION_LABEL'] = df['OCCUPATION_MAJOR'].map(occupation_labels)
    
    # Count valid occupations
    valid_count = df['OCCUPATION_LABEL'].notna().sum()
    print(f"Derived occupation for {valid_count} records")
    if valid_count > 0:
        print(f"Occupation distribution:")
        print(df['OCCUPATION_LABEL'].value_counts().to_dict())
    
    return df


def categorize_occupation(occupation_label):
    """Categorize occupation into 6 main categories."""
    if pd.isna(occupation_label):
        return 'Others'
    elif occupation_label == 'Craft & Trades':
        return 'Craft & Trades'
    elif occupation_label == 'Technicians':
        return 'Technicians'
    elif occupation_label == 'Professionals':
        return 'Professionals'
    elif occupation_label == 'Managers':
        return 'Managers'
    elif occupation_label == 'Clerical Support':
        return 'Clerical Support Workers'
    else:
        return 'Others'


def plot_construction_occupation_income_heatmap(df_raw, output_dir):
    """Plot heatmap of occupation by income decile for construction workers."""
    print("\n=== CREATING OCCUPATION x INCOME HEATMAP FOR CONSTRUCTION ===")
    
    # Filter for construction workers with valid income deciles
    df_raw['INCDECIL_clean'] = pd.to_numeric(df_raw['INCDECIL'], errors='coerce')
    
    # Determine NACE column to use
    nace_column = 'NACE2_1D'
    if nace_column in df_raw.columns:
        df_raw[nace_column] = df_raw[nace_column].astype(str).str.strip()
        unique_vals = df_raw[nace_column].unique()
        has_sector_data = any(val not in ['9.0', 'nan', '9', 'NaN'] for val in unique_vals)
        if not has_sector_data and 'NACE1_1D' in df_raw.columns:
            nace_column = 'NACE1_1D'
            df_raw[nace_column] = df_raw[nace_column].astype(str).str.strip()
    
    df_analysis = df_raw[
        (df_raw[nace_column] == 'F') &
        (df_raw['INCDECIL_clean'] >= 1) & 
        (df_raw['INCDECIL_clean'] <= 10)
    ].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid data for occupation-income heatmap")
        return
    
    # Derive occupation
    df_analysis = derive_occupation_major_group(df_analysis)
    
    # Filter for valid occupations and income
    df_analysis = df_analysis[df_analysis['OCCUPATION_LABEL'].notna()].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid occupation data")
        return
    
    # Categorize occupations into 5 categories
    df_analysis['OCCUPATION_CATEGORY'] = df_analysis['OCCUPATION_LABEL'].apply(categorize_occupation)
    
    # Get weights
    weights = df_analysis['COEFFY'] if 'COEFFY' in df_analysis.columns else np.ones(len(df_analysis))
    total_construction_weight = weights.sum()
    
    # Define category order
    occupation_categories = ['Craft & Trades', 'Clerical Support Workers', 'Technicians', 'Professionals', 'Managers', 'Others']
    
    # Calculate percentages of overall construction workers for each occupation-decile combination
    heatmap_data = []
    for occ_cat in occupation_categories:
        row_data = []
        occ_mask = df_analysis['OCCUPATION_CATEGORY'] == occ_cat
        
        for decile in range(1, 11):
            decile_mask = (df_analysis['INCDECIL_clean'] == decile) & occ_mask
            decile_weight = weights[decile_mask].sum()
            # Percentage of all construction workers in this occupation-decile combination
            pct = (decile_weight / total_construction_weight * 100) if total_construction_weight > 0 else 0
            row_data.append(pct)
        
        heatmap_data.append(row_data)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=occupation_categories, columns=[f'D{i}' for i in range(1, 11)])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': '% of all construction workers'},
                linewidths=0.5, ax=ax)
    
    ax.set_xlabel('Income Decile (1=lowest, 10=highest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Occupation Category', fontsize=12, fontweight='bold')
    ax.set_title('Construction Workers: Occupation by Income Decile - France 2023 (Weighted)\n(Each cell shows % of all construction workers in that occupation-income combination)',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_construction_occupation_income_heatmap.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_construction_occupation_by_sex(df_raw, output_dir):
    """Plot stacked bar chart of occupation by sex for construction workers."""
    print("\n=== CREATING OCCUPATION x SEX STACKED BAR CHART FOR CONSTRUCTION ===")
    
    # Determine NACE column to use
    nace_column = 'NACE2_1D'
    if nace_column in df_raw.columns:
        df_raw[nace_column] = df_raw[nace_column].astype(str).str.strip()
        unique_vals = df_raw[nace_column].unique()
        has_sector_data = any(val not in ['9.0', 'nan', '9', 'NaN'] for val in unique_vals)
        if not has_sector_data and 'NACE1_1D' in df_raw.columns:
            nace_column = 'NACE1_1D'
            df_raw[nace_column] = df_raw[nace_column].astype(str).str.strip()
    
    # Filter for construction workers
    df_analysis = df_raw[df_raw[nace_column] == 'F'].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid data for occupation-sex analysis")
        return
    
    # Derive occupation
    df_analysis = derive_occupation_major_group(df_analysis)
    
    # Categorize occupations into 5 categories
    df_analysis['OCCUPATION_CATEGORY'] = df_analysis['OCCUPATION_LABEL'].apply(categorize_occupation)
    
    # Clean sex variable
    df_analysis['SEX_clean'] = pd.to_numeric(df_analysis['SEX'], errors='coerce')
    
    # Filter for valid occupations and sex
    df_analysis = df_analysis[
        (df_analysis['OCCUPATION_CATEGORY'].notna()) &
        (df_analysis['SEX_clean'].isin([1, 2]))
    ].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid occupation-sex data")
        return
    
    # Get weights
    weights = df_analysis['COEFFY'] if 'COEFFY' in df_analysis.columns else np.ones(len(df_analysis))
    
    # Define category order
    occupation_categories = ['Craft & Trades', 'Clerical Support Workers', 'Technicians', 'Professionals', 'Managers', 'Others']
    
    # Calculate weighted counts by occupation category and sex
    occupation_data = {}
    for occ_cat in occupation_categories:
        occ_mask = df_analysis['OCCUPATION_CATEGORY'] == occ_cat
        male_weight = weights[occ_mask & (df_analysis['SEX_clean'] == 1)].sum()
        female_weight = weights[occ_mask & (df_analysis['SEX_clean'] == 2)].sum()
        occupation_data[occ_cat] = {'Male': male_weight, 'Female': female_weight}
    
    # Calculate total for percentage
    total_all = sum([occupation_data[occ]['Male'] + occupation_data[occ]['Female'] for occ in occupation_categories])
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    occupations = occupation_categories
    male_counts = [occupation_data[occ]['Male'] for occ in occupations]
    female_counts = [occupation_data[occ]['Female'] for occ in occupations]
    
    # Convert to percentages of total construction workers
    male_pcts = [(m / total_all) * 100 for m in male_counts]
    female_pcts = [(f / total_all) * 100 for f in female_counts]
    
    x = np.arange(len(occupations))
    width = 0.6
    
    # Create stacked bars with percentages (non-gendered colors)
    bars1 = ax.bar(x, male_pcts, width, label='Male', color='#b3de69')
    bars2 = ax.bar(x, female_pcts, width, bottom=male_pcts, label='Female', color='#fdb462')
    
    # Add percentage labels
    for i, occ in enumerate(occupations):
        total_pct = male_pcts[i] + female_pcts[i]
        male_count = male_counts[i]
        female_count = female_counts[i]
        total_count = male_count + female_count
        
        # Male percentage within this occupation
        if male_count > 0:
            male_pct_within = (male_count / total_count) * 100
            if male_pcts[i] > 3:  # Only show if bar is tall enough
                ax.text(i, male_pcts[i] / 2, f'{male_pct_within:.0f}%', 
                       ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Female percentage within this occupation
        if female_count > 0:
            female_pct_within = (female_count / total_count) * 100
            if female_pcts[i] > 3:  # Only show if bar is tall enough
                ax.text(i, male_pcts[i] + female_pcts[i] / 2, f'{female_pct_within:.0f}%',
                       ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Total percentage at top
        ax.text(i, total_pct + 0.5, f'{total_pct:.1f}%',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Occupation Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Construction Workers (%)', fontsize=12, fontweight='bold')
    ax.set_title('Construction Workers by Occupation and Sex - France 2023 (Weighted)\n(Bars show % of total construction workers; labels show sex distribution within occupation)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(occupations, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_construction_occupation_by_sex.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_construction_by_income_and_occupation(df_raw, output_dir):
    """Plot construction workers by income decile and occupation category (stacked bar chart)."""
    print("\n=== CREATING CONSTRUCTION BY INCOME AND OCCUPATION ===")
    
    # Clean income deciles
    df_raw['INCDECIL_clean'] = pd.to_numeric(df_raw['INCDECIL'], errors='coerce')
    
    # Determine NACE column to use
    nace_column = 'NACE2_1D'
    if nace_column in df_raw.columns:
        df_raw[nace_column] = df_raw[nace_column].astype(str).str.strip()
        unique_vals = df_raw[nace_column].unique()
        has_sector_data = any(val not in ['9.0', 'nan', '9', 'NaN'] for val in unique_vals)
        if not has_sector_data and 'NACE1_1D' in df_raw.columns:
            nace_column = 'NACE1_1D'
            df_raw[nace_column] = df_raw[nace_column].astype(str).str.strip()
    
    # Filter for construction workers with valid income deciles
    df_analysis = df_raw[
        (df_raw[nace_column] == 'F') &
        (df_raw['INCDECIL_clean'] >= 1) & 
        (df_raw['INCDECIL_clean'] <= 10)
    ].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid data for construction by income and occupation")
        return
    
    # Derive occupation
    df_analysis = derive_occupation_major_group(df_analysis)
    
    # Filter for valid occupations
    df_analysis = df_analysis[df_analysis['OCCUPATION_LABEL'].notna()].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid occupation data")
        return
    
    # Categorize occupations into 5 categories
    df_analysis['OCCUPATION_CATEGORY'] = df_analysis['OCCUPATION_LABEL'].apply(categorize_occupation)
    
    weights_all = df_analysis['COEFFY'] if 'COEFFY' in df_analysis.columns else np.ones(len(df_analysis))
    total_construction_weight = weights_all.sum()
    
    # Occupation category colors (matching oecd_analysis.py palette)
    occupation_colors = {
        'Craft & Trades': '#ffd558',
        'Clerical Support Workers': '#8dd3c7',
        'Technicians': '#fb8072',
        'Professionals': '#b3de69',
        'Managers': '#fdb462',
        'Others': '#bebada'
    }
    
    # Define category order
    occupation_categories = ['Craft & Trades', 'Clerical Support Workers', 'Technicians', 'Professionals', 'Managers', 'Others']
    
    # Calculate decile and occupation breakdown
    decile_data = {}
    
    for decile in sorted(df_analysis['INCDECIL_clean'].unique()):
        if pd.isna(decile):
            continue
        
        decile_mask = df_analysis['INCDECIL_clean'] == decile
        
        # Calculate occupation breakdown
        occ_breakdown = {}
        for occ_cat in occupation_categories:
            occ_mask = decile_mask & (df_analysis['OCCUPATION_CATEGORY'] == occ_cat)
            occ_weight = weights_all[occ_mask].sum()
            occ_share_of_total = (occ_weight / total_construction_weight) * 100 if total_construction_weight > 0 else 0
            occ_breakdown[occ_cat] = occ_share_of_total
        
        decile_data[int(decile)] = occ_breakdown
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    deciles = sorted(decile_data.keys())
    x = np.arange(len(deciles))
    
    # Prepare data for stacked bar
    bottom = np.zeros(len(deciles))
    
    for occ_cat in occupation_categories:
        values = [decile_data[d].get(occ_cat, 0) for d in deciles]
        
        # Only plot if there's data for this occupation
        if sum(values) > 0:
            ax.bar(x, values, bottom=bottom, label=occ_cat,
                   color=occupation_colors[occ_cat], edgecolor='white', linewidth=1.5)
            bottom += np.array(values)
    
    # Add total percentage labels on top of each bar
    for i, decile in enumerate(deciles):
        total = sum([decile_data[decile].get(cat, 0) for cat in occupation_categories])
        ax.text(i, total + 0.3, f'{total:.1f}%', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Income Decile (1=lowest, 10=highest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Construction Workers (%)', fontsize=12, fontweight='bold')
    ax.set_title('Construction Workers by Income Decile and Occupation - France 2023 (Weighted)\n(Stacked bar shows % of total construction workers in each decile, colors show occupation breakdown)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(bottom) * 1.12)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_construction_by_income_and_occupation.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()
    
    # Also create percentage stacked version
    print("Creating percentage stacked version...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    deciles = sorted(decile_data.keys())
    x = np.arange(len(deciles))
    
    # Calculate percentages within each decile
    decile_percentages = {}
    for decile in deciles:
        total = sum([decile_data[decile].get(cat, 0) for cat in occupation_categories])
        if total > 0:
            decile_percentages[decile] = {cat: (decile_data[decile].get(cat, 0) / total) * 100 for cat in occupation_categories}
        else:
            decile_percentages[decile] = {cat: 0 for cat in occupation_categories}
    
    # Prepare data for percentage stacked bar
    bottom = np.zeros(len(deciles))
    
    for occ_cat in occupation_categories:
        values = [decile_percentages[d].get(occ_cat, 0) for d in deciles]
        
        if sum(values) > 0:
            bars = ax.bar(x, values, bottom=bottom, label=occ_cat,
                   color=occupation_colors[occ_cat], edgecolor='white', linewidth=1.5)
            
            # Add percentage labels inside bars if space allows
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 5:  # Only show label if bar is tall enough
                    ax.text(bar.get_x() + bar.get_width()/2., bottom[i] + height/2,
                           f'{height:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
            
            bottom += np.array(values)
    
    ax.set_xlabel('Income Decile (1=lowest, 10=highest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage Within Decile (%)', fontsize=12, fontweight='bold')
    ax.set_title('Construction Workers: Occupation Distribution Within Each Income Decile - France 2023 (Weighted)\n(Each bar = 100%, showing occupation breakdown)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=11)
    ax.legend(title='Occupation Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_construction_by_income_and_occupation_stacked_pct.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_overall_by_income_and_occupation(df_raw, output_dir):
    """Plot overall workers by income decile and occupation category."""
    print("\n=== CREATING OVERALL WORKERS BY INCOME AND OCCUPATION ===")
    
    # Filter for overall workers with valid income deciles
    df_analysis = df_raw[
        (df_raw['ILOSTAT'] == 1) &  # Employed persons only
        (df_raw['INCDECIL_clean'] >= 1) & 
        (df_raw['INCDECIL_clean'] <= 10)
    ].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid data for overall by income and occupation")
        return
    
    # Derive occupation
    df_analysis = derive_occupation_major_group(df_analysis)
    
    # Filter for valid occupations
    df_analysis = df_analysis[df_analysis['OCCUPATION_LABEL'].notna()].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid occupation data")
        return
    
    weights_all = df_analysis['COEFFY'] if 'COEFFY' in df_analysis.columns else np.ones(len(df_analysis))
    total_weight = weights_all.sum()
    
    # Define all 10 occupation categories using the exact labels from derive_occupation_major_group
    # Order: First 5 as before, then expand "Others"
    occupation_categories = [
        'Craft & Trades',              # ISCO 7
        'Clerical Support',             # ISCO 4
        'Technicians',                  # ISCO 3
        'Professionals',                # ISCO 2
        'Managers',                     # ISCO 1
        'Service & Sales',              # ISCO 5
        'Skilled Agricultural',         # ISCO 6
        'Plant & Machine Operators',    # ISCO 8
        'Elementary Occupations'        # ISCO 9
        # Note: Armed Forces (ISCO 0) not included in derive_occupation_major_group mapping
    ]
    
    # Occupation category colors (9 distinct colors)
    occupation_colors = {
        'Craft & Trades': '#ffd558',
        'Clerical Support': '#8dd3c7',
        'Technicians': '#fb8072',
        'Professionals': '#b3de69',
        'Managers': '#fdb462',
        'Service & Sales': '#bebada',
        'Skilled Agricultural': '#80b1d3',
        'Plant & Machine Operators': '#fccde5',
        'Elementary Occupations': '#bc80bd'
    }
    
    # Calculate decile and occupation breakdown
    decile_data = {}
    
    for decile in sorted(df_analysis['INCDECIL_clean'].unique()):
        if pd.isna(decile):
            continue
        
        decile_mask = df_analysis['INCDECIL_clean'] == decile
        
        # Calculate occupation breakdown
        occ_breakdown = {}
        for occ_cat in occupation_categories:
            occ_mask = decile_mask & (df_analysis['OCCUPATION_LABEL'] == occ_cat)
            occ_weight = weights_all[occ_mask].sum()
            occ_share_of_total = (occ_weight / total_weight) * 100 if total_weight > 0 else 0
            occ_breakdown[occ_cat] = occ_share_of_total
        
        decile_data[int(decile)] = occ_breakdown
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 9))
    
    deciles = sorted(decile_data.keys())
    x = np.arange(len(deciles))
    
    # Prepare data for stacked bar
    bottom = np.zeros(len(deciles))
    
    for occ_cat in occupation_categories:
        values = [decile_data[d].get(occ_cat, 0) for d in deciles]
        
        if sum(values) > 0:
            ax.bar(x, values, bottom=bottom, label=occ_cat,
                   color=occupation_colors[occ_cat], edgecolor='white', linewidth=1.5)
            bottom += np.array(values)
    
    # Add total percentage labels on top of each bar
    for i, decile in enumerate(deciles):
        total = sum([decile_data[decile].get(cat, 0) for cat in occupation_categories])
        ax.text(i, total + 0.3, f'{total:.1f}%', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Income Decile (1=lowest, 10=highest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Overall Workers (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Workers by Income Decile and Occupation - France 2023 (Weighted)\n(Stacked bar shows % of total workers in each decile, all 9 ISCO occupation categories)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=11)
    ax.legend(title='Occupation', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(bottom) * 1.12)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_overall_by_income_and_occupation.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()
    
    # Also create percentage stacked version
    print("Creating percentage stacked version...")
    fig, ax = plt.subplots(figsize=(16, 9))
    
    deciles = sorted(decile_data.keys())
    x = np.arange(len(deciles))
    
    # Calculate percentages within each decile
    decile_percentages = {}
    for decile in deciles:
        total = sum([decile_data[decile].get(cat, 0) for cat in occupation_categories])
        if total > 0:
            decile_percentages[decile] = {cat: (decile_data[decile].get(cat, 0) / total) * 100 for cat in occupation_categories}
        else:
            decile_percentages[decile] = {cat: 0 for cat in occupation_categories}
    
    # Prepare data for percentage stacked bar
    bottom = np.zeros(len(deciles))
    
    for occ_cat in occupation_categories:
        values = [decile_percentages[d].get(occ_cat, 0) for d in deciles]
        
        if sum(values) > 0:
            bars = ax.bar(x, values, bottom=bottom, label=occ_cat,
                   color=occupation_colors[occ_cat], edgecolor='white', linewidth=1.5)
            
            # Add percentage labels inside bars if space allows
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 4:  # Only show label if bar is tall enough
                    ax.text(bar.get_x() + bar.get_width()/2., bottom[i] + height/2,
                           f'{height:.1f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            bottom += np.array(values)
    
    ax.set_xlabel('Income Decile (1=lowest, 10=highest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage Within Decile (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Workers: Occupation Distribution Within Each Income Decile - France 2023 (Weighted)\n(Each bar = 100%, showing all 9 ISCO occupation categories)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=11)
    ax.legend(title='Occupation', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_overall_by_income_and_occupation_stacked_pct.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_overall_by_income_and_occupation_detailed(df_raw, output_dir):
    """Plot overall workers by income decile and all 10 occupation categories."""
    print("\n=== CREATING OVERALL WORKERS BY INCOME AND OCCUPATION (DETAILED) ===")
    
    # Filter for overall workers with valid income deciles
    df_analysis = df_raw[
        (df_raw['ILOSTAT'] == 1) &  # Employed persons only
        (df_raw['INCDECIL_clean'] >= 1) & 
        (df_raw['INCDECIL_clean'] <= 10)
    ].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid data for overall by income and occupation")
        return
    
    # Derive occupation
    df_analysis = derive_occupation_major_group(df_analysis)
    
    # Filter for valid occupations
    df_analysis = df_analysis[df_analysis['OCCUPATION_LABEL'].notna()].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid occupation data")
        return
    
    weights_all = df_analysis['COEFFY'] if 'COEFFY' in df_analysis.columns else np.ones(len(df_analysis))
    total_weight = weights_all.sum()
    
    # Define all 10 occupation categories in the requested order
    # First 5 in the order we used before, then expanding "Others"
    occupation_categories = [
        'Craft and Related Trades Workers',           # ISCO 7
        'Clerical Support Workers',                    # ISCO 4
        'Technicians and Associate Professionals',     # ISCO 3
        'Professionals',                               # ISCO 2
        'Managers',                                    # ISCO 1
        'Service and Sales Workers',                   # ISCO 5
        'Skilled Agricultural, Forestry and Fishery Workers',  # ISCO 6
        'Plant and Machine Operators, and Assemblers', # ISCO 8
        'Elementary Occupations',                       # ISCO 9
        'Armed Forces Occupations'                      # ISCO 0
    ]
    
    # Occupation category colors (10 distinct colors)
    occupation_colors = {
        'Craft and Related Trades Workers': '#ffd558',
        'Clerical Support Workers': '#8dd3c7',
        'Technicians and Associate Professionals': '#fb8072',
        'Professionals': '#b3de69',
        'Managers': '#fdb462',
        'Service and Sales Workers': '#bebada',
        'Skilled Agricultural, Forestry and Fishery Workers': '#80b1d3',
        'Plant and Machine Operators, and Assemblers': '#fccde5',
        'Elementary Occupations': '#bc80bd',
        'Armed Forces Occupations': '#ccebc5'
    }
    
    # Calculate decile and occupation breakdown
    decile_data = {}
    
    for decile in sorted(df_analysis['INCDECIL_clean'].unique()):
        if pd.isna(decile):
            continue
        
        decile_mask = df_analysis['INCDECIL_clean'] == decile
        
        # Calculate occupation breakdown
        occ_breakdown = {}
        for occ_cat in occupation_categories:
            occ_mask = decile_mask & (df_analysis['OCCUPATION_LABEL'] == occ_cat)
            occ_weight = weights_all[occ_mask].sum()
            occ_share_of_total = (occ_weight / total_weight) * 100 if total_weight > 0 else 0
            occ_breakdown[occ_cat] = occ_share_of_total
        
        decile_data[int(decile)] = occ_breakdown
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 9))
    
    deciles = sorted(decile_data.keys())
    x = np.arange(len(deciles))
    
    # Prepare data for stacked bar
    bottom = np.zeros(len(deciles))
    
    for occ_cat in occupation_categories:
        values = [decile_data[d].get(occ_cat, 0) for d in deciles]
        
        if sum(values) > 0:
            ax.bar(x, values, bottom=bottom, label=occ_cat,
                   color=occupation_colors[occ_cat], edgecolor='white', linewidth=1.5)
            bottom += np.array(values)
    
    # Add total percentage labels on top of each bar
    for i, decile in enumerate(deciles):
        total = sum([decile_data[decile].get(cat, 0) for cat in occupation_categories])
        ax.text(i, total + 0.3, f'{total:.1f}%', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Income Decile (1=lowest, 10=highest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Overall Workers (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Workers by Income Decile and Occupation (Detailed) - France 2023 (Weighted)\n(Stacked bar shows % of total workers in each decile, all 10 ISCO occupation categories)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=11)
    ax.legend(title='Occupation', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(bottom) * 1.12)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_overall_by_income_and_occupation_detailed.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()
    
    # Also create percentage stacked version
    print("Creating detailed percentage stacked version...")
    fig, ax = plt.subplots(figsize=(16, 9))
    
    deciles = sorted(decile_data.keys())
    x = np.arange(len(deciles))
    
    # Calculate percentages within each decile
    decile_percentages = {}
    for decile in deciles:
        total = sum([decile_data[decile].get(cat, 0) for cat in occupation_categories])
        if total > 0:
            decile_percentages[decile] = {cat: (decile_data[decile].get(cat, 0) / total) * 100 for cat in occupation_categories}
        else:
            decile_percentages[decile] = {cat: 0 for cat in occupation_categories}
    
    # Prepare data for percentage stacked bar
    bottom = np.zeros(len(deciles))
    
    for occ_cat in occupation_categories:
        values = [decile_percentages[d].get(occ_cat, 0) for d in deciles]
        
        if sum(values) > 0:
            bars = ax.bar(x, values, bottom=bottom, label=occ_cat,
                   color=occupation_colors[occ_cat], edgecolor='white', linewidth=1.5)
            
            # Add percentage labels inside bars if space allows
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 4:  # Only show label if bar is tall enough
                    ax.text(bar.get_x() + bar.get_width()/2., bottom[i] + height/2,
                           f'{height:.1f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            bottom += np.array(values)
    
    ax.set_xlabel('Income Decile (1=lowest, 10=highest)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage Within Decile (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Workers: Occupation Distribution Within Each Income Decile (Detailed) - France 2023 (Weighted)\n(Each bar = 100%, showing all 10 ISCO occupation categories)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=11)
    ax.legend(title='Occupation', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_overall_by_income_and_occupation_detailed_pct.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_construction_occupation_age_heatmap(df_raw, output_dir):
    """Plot heatmap of occupation by age groups for construction workers."""
    print("\n=== CREATING OCCUPATION x AGE HEATMAP FOR CONSTRUCTION ===")
    
    # Determine NACE column to use
    nace_column = 'NACE2_1D'
    if nace_column in df_raw.columns:
        df_raw[nace_column] = df_raw[nace_column].astype(str).str.strip()
        unique_vals = df_raw[nace_column].unique()
        has_sector_data = any(val not in ['9.0', 'nan', '9', 'NaN'] for val in unique_vals)
        if not has_sector_data and 'NACE1_1D' in df_raw.columns:
            nace_column = 'NACE1_1D'
            df_raw[nace_column] = df_raw[nace_column].astype(str).str.strip()
    
    # Filter for construction workers
    df_analysis = df_raw[df_raw[nace_column] == 'F'].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid data for occupation-age analysis")
        return
    
    # Check if required columns exist
    if 'ISCO08_1D' not in df_analysis.columns:
        print("WARNING: ISCO08_1D not available, skipping occupation-age analysis")
        return
    
    if 'AGE' not in df_analysis.columns:
        print("WARNING: AGE column not available")
        return
    
    # Derive occupation
    df_analysis = derive_occupation_major_group(df_analysis)
    
    # Check if OCCUPATION_LABEL was created
    if 'OCCUPATION_LABEL' not in df_analysis.columns:
        print("WARNING: Could not derive occupation")
        return
    
    # Categorize occupations
    df_analysis['OCCUPATION_CATEGORY'] = df_analysis['OCCUPATION_LABEL'].apply(categorize_occupation)
    
    # Filter for valid occupations and age
    df_analysis = df_analysis[
        (df_analysis['OCCUPATION_CATEGORY'].notna()) &
        (df_analysis['AGE'].notna()) &
        (df_analysis['AGE'] >= 15) &
        (df_analysis['AGE'] <= 74)
    ].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid occupation-age data")
        return
    
    # Create age groups (same as age pyramid)
    age_bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    age_labels = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', 
                  '45-49', '50-54', '55-59', '60-64', '65-69', '70-74']
    
    df_analysis['age_group'] = pd.cut(df_analysis['AGE'], bins=age_bins, labels=age_labels, right=False)
    
    # Get weights
    weights = df_analysis['COEFFY'] if 'COEFFY' in df_analysis.columns else np.ones(len(df_analysis))
    
    # Define occupation category order
    occupation_categories = ['Craft & Trades', 'Clerical Support Workers', 'Technicians', 'Professionals', 'Managers', 'Others']
    
    # Calculate row-wise percentages (% of each occupation across age groups)
    heatmap_data = []
    for occ_cat in occupation_categories:
        row_data = []
        occ_mask = df_analysis['OCCUPATION_CATEGORY'] == occ_cat
        occ_total_weight = weights[occ_mask].sum()
        
        if occ_total_weight > 0:
            for age_label in age_labels:
                age_mask = (df_analysis['age_group'] == age_label) & occ_mask
                age_weight = weights[age_mask].sum()
                # Percentage of this occupation in this age group (row-wise percentage)
                pct = (age_weight / occ_total_weight * 100) if occ_total_weight > 0 else 0
                row_data.append(pct)
        else:
            row_data = [0] * len(age_labels)
        
        heatmap_data.append(row_data)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=occupation_categories, columns=age_labels)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': '% of occupation'},
                linewidths=0.5, ax=ax)
    
    ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Occupation Category', fontsize=12, fontweight='bold')
    ax.set_title('Construction Workers: Age Distribution by Occupation - France 2023 (Weighted)\n(Each row shows % of that occupation across age groups, rows sum to 100%)',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_construction_occupation_age_heatmap.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_construction_by_age_and_occupation(df_raw, output_dir):
    """
    Create stacked bar charts showing construction workers by age group with occupation breakdown.
    Similar to plot_construction_by_income_and_occupation but with age groups instead of income deciles.
    
    Creates two versions:
    1. Regular stacked bar chart (absolute weighted counts)
    2. Percentage stacked bar chart (each bar sums to 100%)
    
    Args:
        df_raw: Full LFS dataframe with all sectors
        output_dir: Directory to save the plot
    """
    print("\n=== CREATING AGE × OCCUPATION STACKED BAR CHARTS FOR CONSTRUCTION ===")
    
    # Determine NACE column to use
    nace_column = 'NACE2_1D'
    if nace_column in df_raw.columns:
        df_raw[nace_column] = df_raw[nace_column].astype(str).str.strip()
        unique_vals = df_raw[nace_column].unique()
        has_sector_data = any(val not in ['9.0', 'nan', '9', 'NaN'] for val in unique_vals)
        if not has_sector_data and 'NACE1_1D' in df_raw.columns:
            nace_column = 'NACE1_1D'
            df_raw[nace_column] = df_raw[nace_column].astype(str).str.strip()
    
    # Filter for construction workers
    df_analysis = df_raw[df_raw[nace_column] == 'F'].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid data for age-occupation analysis")
        return
    
    # Check if required columns exist
    if 'ISCO08_1D' not in df_analysis.columns:
        print("WARNING: ISCO08_1D not available, skipping age-occupation analysis")
        return
    
    if 'AGE' not in df_analysis.columns:
        print("WARNING: AGE column not available")
        return
    
    # Derive occupation
    df_analysis = derive_occupation_major_group(df_analysis)
    
    # Check if OCCUPATION_LABEL was created
    if 'OCCUPATION_LABEL' not in df_analysis.columns:
        print("WARNING: Could not derive occupation")
        return
    
    # Categorize occupations
    df_analysis['OCCUPATION_CATEGORY'] = df_analysis['OCCUPATION_LABEL'].apply(categorize_occupation)
    
    # Filter for valid occupations and age
    df_analysis = df_analysis[
        (df_analysis['OCCUPATION_CATEGORY'].notna()) &
        (df_analysis['AGE'].notna()) &
        (df_analysis['AGE'] >= 15) &
        (df_analysis['AGE'] <= 74)
    ].copy()
    
    if len(df_analysis) == 0:
        print("WARNING: No valid occupation-age data")
        return
    
    # Create age groups
    age_bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    age_labels = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', 
                  '45-49', '50-54', '55-59', '60-64', '65-69', '70-74']
    
    df_analysis['age_group'] = pd.cut(df_analysis['AGE'], bins=age_bins, labels=age_labels, right=False)
    
    # Get weights
    weights = df_analysis['COEFFY'] if 'COEFFY' in df_analysis.columns else np.ones(len(df_analysis))
    
    # Occupation categories
    occupation_categories = ['Craft & Trades', 'Clerical Support Workers', 'Technicians', 
                            'Professionals', 'Managers', 'Others']
    
    # Define colors for occupations (using palette colors)
    occupation_colors = {
        'Craft & Trades': '#e41a1c',
        'Clerical Support Workers': '#377eb8',
        'Technicians': '#4daf4a',
        'Professionals': '#984ea3',
        'Managers': '#ff7f00',
        'Others': '#999999'
    }
    
    # Calculate weighted counts by age group and occupation
    data_by_age = {occ: [] for occ in occupation_categories}
    
    for age_label in age_labels:
        age_mask = df_analysis['age_group'] == age_label
        for occ_cat in occupation_categories:
            occ_mask = df_analysis['OCCUPATION_CATEGORY'] == occ_cat
            combined_mask = age_mask & occ_mask
            weighted_count = weights[combined_mask].sum()
            data_by_age[occ_cat].append(weighted_count)
    
    # --- Plot 1: Regular stacked bar chart ---
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(age_labels))
    width = 0.8
    
    bottom = np.zeros(len(age_labels))
    
    for occ_cat in occupation_categories:
        values = data_by_age[occ_cat]
        ax.bar(x, values, width, label=occ_cat, bottom=bottom, 
               color=occupation_colors[occ_cat], edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Workers (weighted)', fontsize=12, fontweight='bold')
    ax.set_title('Construction Workers by Age Group and Occupation Type\nFrance 2023 (Weighted)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(age_labels, rotation=45, ha='right')
    ax.legend(title='Occupation Category', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    output = os.path.join(output_dir, 'LFS_construction_by_age_and_occupation.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"OK Saved: {output}")
    
    # --- Plot 2: Percentage stacked bar chart ---
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate percentages
    totals = bottom  # Already calculated from previous loop
    data_by_age_pct = {}
    for occ_cat in occupation_categories:
        data_by_age_pct[occ_cat] = [(data_by_age[occ_cat][i] / totals[i] * 100) if totals[i] > 0 else 0 
                                     for i in range(len(age_labels))]
    
    bottom = np.zeros(len(age_labels))
    
    for occ_cat in occupation_categories:
        values = data_by_age_pct[occ_cat]
        ax.bar(x, values, width, label=occ_cat, bottom=bottom, 
               color=occupation_colors[occ_cat], edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Construction Workers by Age Group and Occupation Type (Percentage)\nFrance 2023 (Weighted)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(age_labels, rotation=45, ha='right')
    ax.legend(title='Occupation Category', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    output = os.path.join(output_dir, 'LFS_construction_by_age_and_occupation_pct.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"OK Saved: {output}")


def plot_education_training_by_occupation(df_construction, df_realestate, df_overall, output_dir):
    """Plot participation in formal education/training by occupation category across sectors."""
    print("\n=== ANALYZING EDUCATION/TRAINING PARTICIPATION BY OCCUPATION ===")
    
    # Occupation categories
    occupation_categories = ['Craft & Trades', 'Clerical Support Workers', 'Technicians', 'Professionals', 'Managers', 'Others']
    
    # Prepare data for all three sectors
    sectors_data = []
    
    for sector_name, df_sector in [
        (CONSTRUCTION_LABEL, df_construction),
        (REALESTATE_LABEL, df_realestate),
        (OVERALL_LABEL, df_overall)
    ]:
        print(f"\n--- Processing {sector_name} ---")
        print(f"Initial records: {len(df_sector)}")
        
        # Check if EDUCFED12 exists
        if 'EDUCFED12' not in df_sector.columns:
            print(f"WARNING: EDUCFED12 column not found in {sector_name}")
            sectors_data.append((sector_name, {occ: 0 for occ in occupation_categories}))
            continue
        
        # Derive occupation
        df_sector_copy = df_sector.copy()
        
        # Check if ISCO08_1D column exists
        if 'ISCO08_1D' not in df_sector_copy.columns:
            print(f"WARNING: ISCO08_1D not available for {sector_name}, skipping occupation analysis")
            sectors_data.append((sector_name, {occ: 0 for occ in occupation_categories}))
            continue
        
        df_sector_copy = derive_occupation_major_group(df_sector_copy)
        
        # Check if OCCUPATION_LABEL was created
        if 'OCCUPATION_LABEL' not in df_sector_copy.columns:
            print(f"WARNING: Could not derive occupation for {sector_name}")
            sectors_data.append((sector_name, {occ: 0 for occ in occupation_categories}))
            continue
        
        # Categorize occupations
        df_sector_copy['OCCUPATION_CATEGORY'] = df_sector_copy['OCCUPATION_LABEL'].apply(categorize_occupation)
        
        # Clean EDUCFED12 variable
        df_sector_copy['EDUCFED12_clean'] = pd.to_numeric(df_sector_copy['EDUCFED12'], errors='coerce')
        
        # Check EDUCFED12 distribution
        print(f"EDUCFED12 value counts:")
        print(df_sector_copy['EDUCFED12_clean'].value_counts().sort_index())
        
        # Filter for valid data (1=Yes, 2=No, 9=Not applicable)
        df_valid = df_sector_copy[
            (df_sector_copy['OCCUPATION_CATEGORY'].notna()) &
            (df_sector_copy['EDUCFED12_clean'].isin([1, 2]))  # 1=Yes, 2=No
        ].copy()
        
        print(f"Valid records after filtering: {len(df_valid)}")
        
        if len(df_valid) == 0:
            print(f"WARNING: No valid EDUCFED12 data for {sector_name}")
            sectors_data.append((sector_name, {occ: 0 for occ in occupation_categories}))
            continue
        
        # Calculate participation rate by occupation
        weights = df_valid['COEFFY'] if 'COEFFY' in df_valid.columns else np.ones(len(df_valid))
        
        occupation_rates = {}
        for occ_cat in occupation_categories:
            occ_mask = df_valid['OCCUPATION_CATEGORY'] == occ_cat
            df_occ = df_valid[occ_mask]
            
            if len(df_occ) > 0:
                occ_weights = weights[occ_mask]
                total_weight = occ_weights.sum()
                yes_mask = df_occ['EDUCFED12_clean'] == 1  # 1 = Yes (participated)
                yes_weight = occ_weights[yes_mask].sum()
                participation_rate = (yes_weight / total_weight) * 100 if total_weight > 0 else 0
                occupation_rates[occ_cat] = participation_rate
                print(f"  {occ_cat}: {len(df_occ)} records, {participation_rate:.1f}% participation")
            else:
                occupation_rates[occ_cat] = 0
        
        sectors_data.append((sector_name, occupation_rates))
        
        print(f"\n{sector_name} education/training participation by occupation:")
        for occ, rate in occupation_rates.items():
            print(f"  {occ}: {rate:.1f}%")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(occupation_categories))
    width = 0.25
    
    for idx, (sector_name, rates) in enumerate(sectors_data):
        values = [rates[occ] for occ in occupation_categories]
        color = COLORS.get(sector_name, '#8dd3c7')
        
        bars = ax.bar(x + idx*width, values, width, label=sector_name, color=color)
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Occupation Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Participated in Formal Education/Training (%)', fontsize=12, fontweight='bold')
    ax.set_title('Participation in Formal Education and Training by Occupation (Last 12 Months) - France 2022 (Weighted)\n(% of workers in each occupation who participated)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(occupation_categories, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max([max([r for r in rates.values() if r > 0] + [10]) for _, rates in sectors_data]) * 1.15)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_education_training_by_occupation.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()


def plot_temp_agency_over_time(country_code, data_path, output_dir, start_year=2010, end_year=2023):
    """Plot share of workers with temporary employment agency contracts over time."""
    print("\n=== ANALYZING TEMPORARY AGENCY CONTRACTS OVER TIME ===")
    
    years = []
    construction_shares = []
    realestate_shares = []
    overall_shares = []
    
    for year in range(start_year, end_year + 1):
        # Load data
        df = load_lfs_data(country_code, year, data_path)
        if df.empty:
            print(f"WARNING: Could not load data for {country_code} {year}")
            continue
        
        # Filter by sector
        df_construction = filter_by_sector(df, CONSTRUCTION_NACE)
        df_realestate = filter_by_sector(df, REALESTATE_NACE)
        df_overall = filter_workers_only(df)
        
        # Check if TEMPAGCY column exists
        if 'TEMPAGCY' not in df_construction.columns:
            print(f"WARNING: TEMPAGCY not available for {country_code} {year}")
            continue
        
        # Calculate shares for each sector
        for df_sector, shares_list in [
            (df_construction, construction_shares),
            (df_realestate, realestate_shares),
            (df_overall, overall_shares)
        ]:
            df_sector['TEMPAGCY_clean'] = pd.to_numeric(df_sector['TEMPAGCY'], errors='coerce')
            # Filter to only valid codes (1=No, 2=Yes, excluding 9=not applicable and NaN)
            valid_mask = df_sector['TEMPAGCY_clean'].isin([1, 2])
            df_valid = df_sector[valid_mask].copy()
            
            if len(df_valid) > 0:
                # Calculate weighted share of "Yes" (code 2)
                weights = df_valid['COEFFY'] if 'COEFFY' in df_valid.columns else np.ones(len(df_valid))
                total_weight = weights.sum()
                yes_mask = df_valid['TEMPAGCY_clean'] == 2
                yes_weight = weights[yes_mask].sum()
                share_yes = (yes_weight / total_weight) * 100
                shares_list.append(share_yes)
            else:
                shares_list.append(np.nan)
        
        years.append(year)
        print(f"{year}: Construction={construction_shares[-1]:.1f}%, Real Estate={realestate_shares[-1]:.1f}%, Overall={overall_shares[-1]:.1f}%")
    
    # Create line plot
    if len(years) == 0:
        print("ERROR: No data available for time series analysis")
        return
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    ax.plot(years, construction_shares, marker='o', linewidth=2.5, markersize=8, 
            label=CONSTRUCTION_LABEL, color=COLORS[CONSTRUCTION_LABEL])
    ax.plot(years, realestate_shares, marker='s', linewidth=2.5, markersize=8, 
            label=REALESTATE_LABEL, color=COLORS[REALESTATE_LABEL])
    ax.plot(years, overall_shares, marker='^', linewidth=2.5, markersize=8, 
            label=OVERALL_LABEL, color=COLORS[OVERALL_LABEL])
    
    # Add value labels on points
    for year, construction, realestate, overall in zip(years, construction_shares, realestate_shares, overall_shares):
        ax.text(year, construction + 0.3, f'{construction:.1f}%', ha='center', fontsize=8, color=COLORS[CONSTRUCTION_LABEL])
        ax.text(year, realestate + 0.3, f'{realestate:.1f}%', ha='center', fontsize=8, color=COLORS[REALESTATE_LABEL])
        ax.text(year, overall + 0.3, f'{overall:.1f}%', ha='center', fontsize=8, color=COLORS[OVERALL_LABEL])
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Share with Temporary Agency Contract (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Share of Workers with Temporary Employment Agency Contracts Over Time - {country_code.upper()} ({start_year}-{end_year}) (Weighted)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(max(construction_shares), max(realestate_shares), max(overall_shares)) * 1.2)
    
    plt.tight_layout()
    output = os.path.join(output_dir, 'LFS_temp_agency_over_time.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"OK Saved: {output}")
    plt.close()




# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("LFS ANALYSIS - CONSTRUCTION AND REAL ESTATE SECTORS")
    print("="*80)
    
    dirs = setup_directories()
    
    # Load data
    df = load_lfs_data(COUNTRY, YEAR, dirs['raw_data'])
    if df.empty:
        print("ERROR: Could not load LFS data")
        return
    
    print(f"Total records: {len(df)}")
    
    # Filter by sector
    print(f"\n--- FILTERING BY SECTOR ---")
    df_construction = filter_by_sector(df, CONSTRUCTION_NACE)
    df_realestate = filter_by_sector(df, REALESTATE_NACE)
    df_overall = filter_workers_only(df)  # Overall market - workers only
    
    print(f"Construction: {len(df_construction)} records")
    print(f"Real Estate: {len(df_realestate)} records")
    print(f"Overall (workers only): {len(df_overall)} records")
    
    # Prepare data
    df_construction = prepare_analysis_data(df_construction)
    df_realestate = prepare_analysis_data(df_realestate)
    df_overall = prepare_analysis_data(df_overall)
    
    # Analyze demographics
    analyze_demographics(df_construction, df_realestate, df_overall)
    analyze_migration_status(df_construction, df_realestate, df_overall)
    analyze_education_level(df_construction, df_realestate, df_overall)
    analyze_education_achievement(df_construction, df_realestate, df_overall)
    
    # Analyze employment status
    analyze_employment_status(df_construction, df_realestate, df_overall)
    analyze_permanency(df_construction, df_realestate, df_overall)
    analyze_temp_agency(df_construction, df_realestate, df_overall)
    analyze_temp_agency_by_migration(df_construction, df_realestate, df_overall)
    analyze_migration_breakdown_of_temp_workers(df_construction, df_realestate, df_overall)
    
    # Analyze hours worked
    analyze_hours_worked(df_construction, df_realestate, df_overall)
    analyze_total_hours_worked(df_construction, df_realestate, df_overall)
    
    # Analyze health
    analyze_health_limitation(df_construction, df_realestate, df_overall)
    analyze_absence_incidence(df_construction, df_realestate, df_overall)
    
    # Analyze income
    analyze_income(df_construction, df_realestate, df_overall)
    analyze_sector_by_income_decile(df_overall, df)
    analyze_construction_by_income_and_migration(df_construction, df)
    
    # Create visualizations
    print(f"\n--- CREATING VISUALIZATIONS ---")
    plot_sex_distribution(df_construction, df_realestate, df_overall, dirs['output'])
    plot_age_distribution(df_construction, df_realestate, df_overall, dirs['output'])
    plot_migration_status(df_construction, df_realestate, df_overall, dirs['output'])
    plot_construction_by_income_and_sex(df_construction, df, dirs['output'])
    plot_construction_age_pyramid_comparison(COUNTRY, dirs['raw_data'], dirs['output'])
    plot_construction_by_income_and_occupation(df, dirs['output'])
    plot_overall_by_income_and_occupation(df, dirs['output'])
    plot_overall_by_income_and_occupation_detailed(df, dirs['output'])
    plot_construction_occupation_income_heatmap(df, dirs['output'])
    plot_construction_occupation_by_sex(df, dirs['output'])
    plot_construction_occupation_age_heatmap(df, dirs['output'])
    plot_construction_by_age_and_occupation(df, dirs['output'])
    
    # Load 2022 data for education/training analysis (EDUCFED12 not available in 2023)
    print(f"\n--- LOADING 2022 DATA FOR EDUCATION/TRAINING ANALYSIS ---")
    df_2022 = load_lfs_data(COUNTRY, 2022, dirs['raw_data'])
    if not df_2022.empty:
        df_construction_2022 = filter_by_sector(df_2022, CONSTRUCTION_NACE)
        df_realestate_2022 = filter_by_sector(df_2022, REALESTATE_NACE)
        df_overall_2022 = filter_workers_only(df_2022)
        df_construction_2022 = prepare_analysis_data(df_construction_2022)
        df_realestate_2022 = prepare_analysis_data(df_realestate_2022)
        df_overall_2022 = prepare_analysis_data(df_overall_2022)
        plot_education_training_by_occupation(df_construction_2022, df_realestate_2022, df_overall_2022, dirs['output'])
    
    plot_temp_agency_by_migration(df_construction, df_realestate, df_overall, dirs['output'])
    plot_migration_breakdown_of_temp_workers(df_construction, df_realestate, df_overall, dirs['output'])
    plot_education_level(df_construction, df_realestate, df_overall, dirs['output'])
    plot_education_achievement(df_construction, df_realestate, df_overall, dirs['output'])
    plot_employment_status(df_construction, df_realestate, df_overall, dirs['output'])
    plot_fulltime_parttime(df_construction, df_realestate, df_overall, dirs['output'])
    plot_permanency(df_construction, df_realestate, df_overall, dirs['output'])
    plot_temp_agency(df_construction, df_realestate, df_overall, dirs['output'])
    plot_sector_by_income_decile(df, dirs['output'])
    plot_construction_by_income_and_migration(df_construction, df, dirs['output'])
    plot_hours_worked_distribution(df_construction, df_realestate, df_overall, dirs['output'])
    plot_total_hours_worked(df_construction, df_realestate, df_overall, dirs['output'])
    plot_health_limitation(df_construction, df_realestate, df_overall, dirs['output'])
    plot_absence_incidence(df_construction, df_realestate, df_overall, dirs['output'])
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
