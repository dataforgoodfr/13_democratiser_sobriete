#!/usr/bin/env python3
"""
Final Dataset Integration Script

This script combines all processed European survey datasets (EU-SILC, HBS, LFS, EHIS)
and additional transport indicators to create the final European Well-Being Index (EWBI) dataset.

The script performs the following operations:
1. Loads processed datasets from each survey
2. Applies necessary filtering and cleaning
3. Integrates transport indicators (TT-SILC-1, TT-SILC-2)
4. Combines all datasets into a unified structure
5. Performs data validation and quality checks
6. Saves multiple output formats for different use cases

Author: Data Processing Pipeline
Date: 2024
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Filter out economic good indicators (only keep satisfiers)
# These indicators represent material goods rather than satisfiers and should be excluded
economic_indicators_to_remove = [
    'AN-SILC-1',
    'AE-HBS-1', 'AE-HBS-2',
    'HQ-SILC-2', 
    'HH-SILC-1', 'HH-HBS-1', 'HH-HBS-2', 'HH-HBS-3', 'HH-HBS-4',
    'HE-HBS-1', 'HE-HBS-2',
    'EC-HBS-1', 'EC-HBS-2',
    'ED-ICT-1', 'ED-EHIS-1',
    'AC-SILC-1', 'AC-SILC-2', 'AC-HBS-1', 'AC-HBS-2', 'AC-EHIS-1',
    'IE-HBS-1', 'IE-HBS-2',
    'IC-SILC-1', 'IC-SILC-2', 'IC-HBS-1', 'IC-HBS-2',
    'TT-SILC-1', 'TT-SILC-2', 'TT-HBS-1', 'TT-HBS-2',
    'TS-SILC-1', 'TS-HBS-1', 'TS-HBS-2'
]


def setup_directories() -> dict:
    """
    Set up directory structure for final data integration using portable paths.
    
    Returns:
        Dictionary containing directory paths
    """
    # Get the absolute path to the project output directory
    OUTPUT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')))
    
    # Input data directory - now pointing to the correct location within the project
    INPUT_DATA_DIR = OUTPUT_DIR / "0_raw_data_EUROSTAT"
    
    dirs = {
        'output_base': OUTPUT_DIR,
        'input_data': INPUT_DATA_DIR,
        # Use processed data from the project's 0_raw_data_EUROSTAT directory
        'silc_final': INPUT_DATA_DIR / "0_EU-SILC",
        'hbs_final': INPUT_DATA_DIR / "0_HBS", 
        'lfs_final': INPUT_DATA_DIR / "0_LFS",
        'ehis_final': INPUT_DATA_DIR / "0_EHIS",
        'transport_data': INPUT_DATA_DIR / "transport",  # Adjusted path for transport data
        'final_output': INPUT_DATA_DIR / "1_final_df"
    }
    
    # Create output directory if it doesn't exist
    dirs['final_output'].mkdir(parents=True, exist_ok=True)
    
    return dirs


def load_survey_datasets(dirs: dict) -> dict:
    """
    Load all processed survey datasets.
    
    Args:
        dirs: Directory paths dictionary
        
    Returns:
        Dictionary containing loaded DataFrames
    """
    print("📊 Loading processed survey datasets...")
    
    datasets = {}
    
    # Load EU-SILC datasets (both household and personal indicators)
    try:
        datasets['silc_household'] = pd.read_csv(
            dirs['silc_final'] / "3_final_merged_df" / "EU_SILC_household_final_summary.csv"
        )
        # EU-SILC uses deciles, add NaN quintile column
        datasets['silc_household']['quintile'] = pd.NA
        print(f"✅ EU-SILC Household: {len(datasets['silc_household']):,} rows")
        
        datasets['silc_personal'] = pd.read_csv(
            dirs['silc_final'] / "3_final_merged_df" / "EU_SILC_personal_final_summary.csv"
        )
        # EU-SILC uses deciles, add NaN quintile column
        datasets['silc_personal']['quintile'] = pd.NA
        print(f"✅ EU-SILC Personal: {len(datasets['silc_personal']):,} rows")
    except FileNotFoundError as e:
        print(f"❌ Error loading EU-SILC data: {e}")
        raise
    
    # Load HBS dataset
    try:
        datasets['hbs'] = pd.read_csv(
            dirs['hbs_final'] / "HBS_household_final_summary.csv"
        )
        # Filter out Italy as specified in the original notebook
        datasets['hbs'] = datasets['hbs'][datasets['hbs']['country'] != 'IT']
        # Filter out HE-HBS-1 and HE-HBS-2 indicators as specified
        datasets['hbs'] = datasets['hbs'][~datasets['hbs']['primary_index'].isin(['HE-HBS-1', 'HE-HBS-2'])]
        # HBS uses deciles, add NaN quintile column
        datasets['hbs']['quintile'] = pd.NA
        print(f"✅ HBS (excluding IT and HE-HBS indicators): {len(datasets['hbs']):,} rows")
    except FileNotFoundError as e:
        print(f"❌ Error loading HBS data: {e}")
        raise
    
    # Load LFS dataset
    try:
        datasets['lfs'] = pd.read_csv(
            dirs['lfs_final'] / "LFS_household_final_summary.csv"
        )
        # Filter out RU-LFS-1 indicator as specified
        datasets['lfs'] = datasets['lfs'][datasets['lfs']['primary_index'] != 'RU-LFS-1']
        # LFS uses deciles, add NaN quintile column
        datasets['lfs']['quintile'] = pd.NA
        print(f"✅ LFS (excluding RU-LFS-1): {len(datasets['lfs']):,} rows")
    except FileNotFoundError as e:
        print(f"❌ Error loading LFS data: {e}")
        raise
    
    # Load EHIS dataset
    try:
        datasets['ehis'] = pd.read_csv(
            dirs['ehis_final'] / "EHIS_level4_indicators.csv"
        )
        # EHIS uses 'quintile' (5 income groups) while others use 'decile' (10 income groups)
        # Keep EHIS as quintile data and add NaN decile column since others use decile
        datasets['ehis']['decile'] = pd.NA
        print(f"✅ EHIS Level 4: {len(datasets['ehis']):,} rows (quintile-based data)")
        
        # Load EHIS "All" population values from level5 statistics
        try:
            ehis_level5 = pd.read_csv(
                dirs['ehis_final'] / "EHIS_level5_statistics.csv"
            )
            # Filter for "All" population values (value_5C type)
            ehis_all = ehis_level5[
                (ehis_level5['level5_type'] == 'value_5C') & 
                (ehis_level5['decile'] == 'All')
            ].copy()
            
            if not ehis_all.empty:
                # Set quintile to "All" and decile to NaN for consistency
                ehis_all['quintile'] = 'All'
                ehis_all['decile'] = pd.NA
                # Select only the columns we need
                ehis_all = ehis_all[['year', 'country', 'quintile', 'primary_index', 'value', 'database']]
                # Add decile column for consistency
                ehis_all['decile'] = pd.NA
                
                # Combine with the quintile-based EHIS data  
                datasets['ehis'] = pd.concat([datasets['ehis'], ehis_all], axis=0, ignore_index=True)
                print(f"✅ EHIS All values: {len(ehis_all):,} rows added (total population data)")
            else:
                print("⚠️  No EHIS 'All' population values found in level5 statistics")
                
        except FileNotFoundError:
            print("⚠️  EHIS level5 statistics file not found - continuing without 'All' values")
        
        print(f"✅ EHIS Total: {len(datasets['ehis']):,} rows (quintile + All population data)")
    except FileNotFoundError as e:
        print(f"❌ Error loading EHIS data: {e}")
        raise
    
    return datasets


def process_transport_indicators(dirs: dict) -> pd.DataFrame:
    """
    Process transport accessibility and affordability indicators.
    
    Args:
        dirs: Directory paths dictionary
        
    Returns:
        Combined transport indicators DataFrame
    """
    print("🚌 Processing transport indicators...")
    
    # Check if transport data directory exists
    if not dirs['transport_data'].exists():
        print("⚠️  Warning: Transport data directory not found. Creating empty transport DataFrame.")
        # Return empty DataFrame with expected structure
        empty_df = pd.DataFrame(columns=[
            'year', 'country', 'quintile', 'primary_index', 'value', 'database'
        ])
        return empty_df
    
    transport_dfs = []
    
    # Process TT-SILC-1: Public Transport Affordability
    try:
        afford_pt = pd.read_csv(
            dirs['transport_data'] / "estat_ilc_mdes13b - Persons who cannot afford a regular use of public transport.csv",
            sep=";"
        )
        
        # Rename columns
        afford_pt = afford_pt.rename(columns={
            "geo": "country",
            "quantile": "quintile"
        })
        
        # Melt years
        year_cols = ['2014', '2015', '2016']
        afford_pt = afford_pt.melt(
            id_vars=['freq', 'unit', 'wstatus', 'quintile', 'country'],
            value_vars=year_cols,
            var_name="year",
            value_name="value"
        )
        
        # Add metadata and filter
        afford_pt["primary_index"] = "TT-SILC-1"
        afford_pt["database"] = "EU-SILC"
        afford_pt = afford_pt[afford_pt["wstatus"] == "POP"][
            ['year', 'country', 'quintile', 'primary_index', 'value', 'database']
        ]
        
        # Clean values (replace comma with dot, handle spaces)
        afford_pt["value"] = (
            afford_pt["value"]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        afford_pt["value"] = pd.to_numeric(afford_pt["value"], errors="coerce")
        
        transport_dfs.append(afford_pt)
        print(f"✅ TT-SILC-1 (Afford PT): {len(afford_pt):,} rows")
        
    except FileNotFoundError as e:
        print(f"⚠️  Warning: Could not load affordability data: {e}")
    
    # Process TT-SILC-2: Public Transport Accessibility
    try:
        access_pt = pd.read_csv(
            dirs['transport_data'] / "estat_ilc_hcmp06 - difficulty in accessing public transport.csv",
            sep=";"
        )
        
        # Rename columns
        access_pt = access_pt.rename(columns={
            "geo": "country",
            "quantile": "quintile"
        })
        
        # Melt years
        year_cols = ["2012", "2014"]
        access_pt = access_pt.melt(
            id_vars=['freq', 'unit', 'lev_diff', 'deg_urb', 'quintile', 'country'],
            value_vars=year_cols,
            var_name="year",
            value_name="value"
        )
        
        # Add metadata and filter
        access_pt["primary_index"] = "TT-SILC-2"
        access_pt["database"] = "EU-SILC"
        access_pt = access_pt[
            (access_pt["deg_urb"] == "TOTAL") & (access_pt["lev_diff"] == "VLOW")
        ][['year', 'country', 'quintile', 'primary_index', 'value', 'database']]
        
        transport_dfs.append(access_pt)
        print(f"✅ TT-SILC-2 (Access PT): {len(access_pt):,} rows")
        
    except FileNotFoundError as e:
        print(f"⚠️  Warning: Could not load accessibility data: {e}")
    
    # Combine transport indicators
    if transport_dfs:
        transport_combined = pd.concat(transport_dfs, axis=0, ignore_index=True)
        # Transport indicators use quintiles, add NaN decile column since others use decile
        transport_combined['decile'] = pd.NA
        return transport_combined
    else:
        print("⚠️  No transport data loaded")
        return pd.DataFrame()


def validate_decile_quintile_coverage(datasets: dict) -> None:
    """
    Validate that all datasets have proper decile/quintile coverage for all countries.
    
    Args:
        datasets: Dictionary of survey DataFrames
    """
    print("🔍 Validating decile/quintile coverage...")
    
    for dataset_name, df in datasets.items():
        if df is not None and not df.empty:
            # Check if dataset has decile column
            decile_col = 'decile' if 'decile' in df.columns else None
            
            if decile_col:
                # Check coverage by country
                coverage = df.groupby('country')[decile_col].nunique().sort_values()
                incomplete_countries = coverage[coverage < coverage.max()]
                
                if not incomplete_countries.empty:
                    print(f"⚠️  {dataset_name}: Countries with incomplete {decile_col} coverage:")
                    for country, count in incomplete_countries.items():
                        print(f"     {country}: {count} {decile_col}s (expected: {coverage.max()})")
                else:
                    print(f"✅ {dataset_name}: Complete {decile_col} coverage across all countries")
                    
                # Show decile/quintile range
                range_info = f"{df[decile_col].min()}-{df[decile_col].max()}"
                print(f"   {dataset_name} {decile_col} range: {range_info}")


def combine_all_datasets(datasets: dict, transport_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all survey datasets and transport indicators.
    
    Args:
        datasets: Dictionary of survey DataFrames
        transport_df: Transport indicators DataFrame
        
    Returns:
        Combined final DataFrame
    """
    print("🔗 Combining all datasets...")
    
    # Validate coverage before combining
    validate_decile_quintile_coverage(datasets)
    
    # Start with EU-SILC datasets (only household data available)
    if datasets['silc_personal'] is not None:
        final_df = pd.concat([
            datasets['silc_personal'], 
            datasets['silc_household']
        ], axis=0, ignore_index=True)
    else:
        final_df = datasets['silc_household'].copy()
        print("   Started with EU-SILC Household data only (Personal data not available)")
    
    # Add other survey datasets
    for dataset_name, df in [
        ('HBS', datasets['hbs']),
        ('LFS', datasets['lfs']),
        ('EHIS', datasets['ehis'])
    ]:
        if df is not None and not df.empty:
            final_df = pd.concat([final_df, df], axis=0, ignore_index=True)
            print(f"   Added {dataset_name}: {len(df):,} rows")
        else:
            print(f"   ⚠️ Skipped {dataset_name}: Empty or None dataset")
    
    # Add transport indicators if available
    if not transport_df.empty:
        final_df = pd.concat([final_df, transport_df], axis=0, ignore_index=True)
        print(f"   Added Transport: {len(transport_df):,} rows")
    else:
        print("   🚫 Transport indicators disabled (no data available)")
    
    # Rename database column to datasource as requested
    if 'database' in final_df.columns:
        final_df = final_df.rename(columns={'database': 'datasource'})
        print("✅ Renamed 'database' column to 'datasource'")
    
    # Ensure consistent column structure  
    required_columns = ['year', 'country', 'decile', 'quintile', 'primary_index', 'value', 'datasource']
    for col in required_columns:
        if col not in final_df.columns:
            print(f"⚠️ Warning: Missing column '{col}' in final dataset")
    
    # Verify quintile column is only populated for EHIS and transport indicators
    if 'quintile' in final_df.columns and 'datasource' in final_df.columns:
        non_ehis_transport_with_quintile = final_df[
            (~final_df['datasource'].isin(['EHIS', 'EU-SILC'])) & 
            (final_df['quintile'].notna())
        ]
        if len(non_ehis_transport_with_quintile) > 0:
            print(f"⚠️ Warning: Found {len(non_ehis_transport_with_quintile)} non-EHIS/transport records with quintile data")
        
        ehis_without_quintile = final_df[
            (final_df['datasource'] == 'EHIS') & 
            (final_df['quintile'].isna())
        ]
        if len(ehis_without_quintile) > 0:
            print(f"⚠️ Warning: Found {len(ehis_without_quintile)} EHIS records without quintile data")
    
    # Ensure decile column has consistent data types (handle mix of numbers and "All")
    if 'decile' in final_df.columns:
        final_df['decile'] = final_df['decile'].astype(str)
        unique_deciles = sorted(final_df['decile'].unique())
        print(f"   Decile values present: {unique_deciles}")
        
        # Validate that we have both individual deciles and "All" values
        has_individual_deciles = any(d.isdigit() for d in unique_deciles)
        has_total_population = "All" in unique_deciles
        print(f"   Has individual deciles: {has_individual_deciles}")
        print(f"   Has total population (All): {has_total_population}")
    
    # Remove rows with missing values
    initial_rows = len(final_df)
    final_df = final_df[final_df["value"].notna()]
    removed_rows = initial_rows - len(final_df)
    
    if removed_rows > 0:
        print(f"   Removed {removed_rows:,} rows with missing values")
    
    # Filter out economic indicators (material goods) - keep only satisfiers
    print(f"🔧 Filtering out {len(economic_indicators_to_remove)} economic indicators...")
    before_economic_filtering = len(final_df)
    final_df = final_df[~final_df['primary_index'].isin(economic_indicators_to_remove)]
    after_economic_filtering = len(final_df)
    economic_removed = before_economic_filtering - after_economic_filtering
    
    if economic_removed > 0:
        print(f"   Removed {economic_removed:,} rows with economic indicators")
        # Show which indicators were actually filtered out (those NOT in the final dataset)
        filtered_indicators = [ind for ind in economic_indicators_to_remove if ind not in final_df['primary_index'].unique()]
        print(f"   Economic indicators filtered: {filtered_indicators}")
    else:
        print("   No economic indicators found to remove")
    
    print(f"📊 Final combined dataset: {len(final_df):,} rows")
    
    return final_df


def perform_data_validation(df: pd.DataFrame) -> dict:
    """
    Perform data validation and generate summary statistics.
    
    Args:
        df: Final combined DataFrame
        
    Returns:
        Dictionary with validation results
    """
    print("🔍 Performing data validation...")
    
    validation = {}
    
    # Basic statistics
    validation['total_rows'] = len(df)
    validation['unique_countries'] = df['country'].nunique()
    validation['unique_years'] = df['year'].nunique() if 'year' in df.columns else 'N/A'
    validation['unique_indicators'] = df['primary_index'].nunique()
    validation['databases'] = df['datasource'].unique().tolist()
    
    # Year range
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        validation['year_range'] = f"{df['year'].min():.0f}-{df['year'].max():.0f}"
    
    # Countries
    validation['countries'] = sorted(df['country'].unique().tolist())
    
    # Indicators by database
    validation['indicators_by_db'] = df.groupby('datasource')['primary_index'].nunique().to_dict()
    
    # Missing value analysis
    validation['missing_values'] = df.isnull().sum().to_dict()
    
    # Value range analysis (convert to numeric first)
    numeric_values = pd.to_numeric(df['value'], errors='coerce')
    validation['value_stats'] = {
        'min': numeric_values.min(),
        'max': numeric_values.max(),
        'mean': numeric_values.mean(),
        'median': numeric_values.median()
    }
    
    # Zero value analysis (use numeric values)
    validation['zero_values'] = len(df[numeric_values == 0])
    validation['zero_value_pct'] = (validation['zero_values'] / len(df)) * 100
    
    return validation




def create_aggregated_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional aggregated data for "All countries" entries using median computation.
    This includes median across countries for each decile (1-10) and "All" separately.
    
    Args:
        df: Final combined DataFrame
        
    Returns:
        DataFrame with additional aggregated rows
    """
    print("📊 Creating aggregated data for 'All countries' using median computation...")
    
    aggregated_rows = []
    
    # Group by year, decile/quintile, and indicator to create country-level medians
    grouping_cols = ['year', 'primary_index']
    
    # Handle both decile and quintile columns
    if 'decile' in df.columns:
        grouping_cols.append('decile')
        income_group_col = 'decile'
    elif 'quintile' in df.columns:
        grouping_cols.append('quintile')
        income_group_col = 'quintile'
    else:
        print("⚠️ Warning: No decile or quintile column found for aggregation")
        return df
    
    # Create aggregations for each year, income group, and indicator
    for group_keys, group_data in df.groupby(grouping_cols):
        if len(group_keys) == 3:
            year, primary_index, income_group = group_keys
        else:
            continue
            
        # Only aggregate if we have multiple countries (at least 2)
        unique_countries = group_data['country'].nunique()
        if unique_countries >= 2:
            # Calculate median value across countries for this specific combination
            median_value = group_data['value'].median()
            
            # Determine datasource based on the most common one in the group
            datasource = group_data['datasource'].mode().iloc[0] if not group_data['datasource'].empty else 'Aggregated'
            
            # Create aggregated row
            agg_row = {
                'year': year,
                'country': 'All Countries',
                'primary_index': primary_index,
                'value': median_value,
                'datasource': f'{datasource}_Aggregated'
            }
            
            # Add the appropriate income group column
            if income_group_col == 'decile':
                agg_row['decile'] = income_group
                agg_row['quintile'] = pd.NA
            else:
                agg_row['quintile'] = income_group
                agg_row['decile'] = pd.NA
            
            aggregated_rows.append(agg_row)
    
    if aggregated_rows:
        agg_df = pd.DataFrame(aggregated_rows)
        combined_df = pd.concat([df, agg_df], ignore_index=True)
        
        # Count aggregations by income group type
        if income_group_col == 'decile':
            decile_counts = agg_df['decile'].value_counts().sort_index()
            print(f"   Added {len(aggregated_rows):,} aggregated rows using median across countries")
            print(f"   Median computed for deciles: {sorted(decile_counts.index.tolist())}")
            print(f"   Records per decile: {dict(decile_counts)}")
        else:
            quintile_counts = agg_df['quintile'].value_counts().sort_index()
            print(f"   Added {len(aggregated_rows):,} aggregated rows using median across countries")
            print(f"   Median computed for quintiles: {sorted(quintile_counts.index.tolist())}")
            print(f"   Records per quintile: {dict(quintile_counts)}")
        
        # Verify that we have "All" values aggregated
        if income_group_col == 'decile':
            all_values = agg_df[agg_df['decile'] == 'All']
            individual_deciles = agg_df[agg_df['decile'] != 'All']
            print(f"   'All' population aggregations: {len(all_values):,}")
            print(f"   Individual decile aggregations (1-10): {len(individual_deciles):,}")
        else:
            all_values = agg_df[agg_df['quintile'] == 'All']
            individual_quintiles = agg_df[agg_df['quintile'] != 'All']
            print(f"   'All' population aggregations: {len(all_values):,}")
            print(f"   Individual quintile aggregations (1-5): {len(individual_quintiles):,}")
        
        return combined_df
    else:
        print("   No aggregations created (insufficient country coverage)")
        return df






def create_unified_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the final DataFrame into the unified format for the app.
    
    Args:
        df: Final combined DataFrame
        
    Returns:
        Unified DataFrame with standardized structure
    """
    print("🔄 Creating unified dataframe structure...")
    
    # First create aggregated data
    df_with_aggregations = create_aggregated_data(df)
    
    # Create base unified dataframe
    unified_df = df_with_aggregations.copy()
    
    # Standardize column names and add required columns
    unified_df = unified_df.rename(columns={
        'year': 'Year',
        'country': 'Country', 
        'primary_index': 'Primary and raw data',
        'value': 'Value'
    })
    
    # Handle decile/quintile columns
    if 'decile' in unified_df.columns:
        unified_df = unified_df.rename(columns={'decile': 'Decile'})
    
    # Create Quintile column from Decile (proper 1-10 to 1-5 mapping)
    def decile_to_quintile(decile_val):
        """Convert decile values (1-10) to quintile equivalents (1-5)."""
        if decile_val == "All":
            return "All"
        try:
            decile_num = float(decile_val)
            # Map deciles 1-2 to quintile 1, 3-4 to quintile 2, etc.
            if decile_num in [1.0, 2.0]:
                return 1.0
            elif decile_num in [3.0, 4.0]:
                return 2.0
            elif decile_num in [5.0, 6.0]:
                return 3.0
            elif decile_num in [7.0, 8.0]:
                return 4.0
            elif decile_num in [9.0, 10.0]:
                return 5.0
            else:
                return decile_val  # Keep original if not in 1-10 range
        except (ValueError, TypeError):
            return decile_val
    
    # Create Quintile column - ONLY preserve existing quintile data, do NOT convert deciles to quintiles
    if 'quintile' in unified_df.columns:
        # Only preserve existing quintile data, keep NaN for datasets that should not have quintiles
        unified_df['Quintile'] = unified_df['quintile'].copy()
    else:
        # No existing quintile data, set all to NaN
        unified_df['Quintile'] = pd.NA
    
    # Add required metadata columns
    unified_df['Level'] = 5  # All current data is Level 5 (raw statistical data)
    unified_df['EU priority'] = pd.NA  # To be determined in next phases
    unified_df['Secondary'] = pd.NA  # To be determined in next phases
    unified_df['Type'] = 'Statistical computation'  # As specified
    
    # Set aggregation method based on data characteristics
    def determine_aggregation(row):
        """Determine aggregation method based on row characteristics."""
        if row['Decile'] == 'All' and row['Country'] != 'All Countries':
            return 'Median across countries'  # Country totals
        elif row['Country'] == 'All Countries':
            return 'Median across countries'  # Multi-country aggregations
        else:
            return 'Geometric mean inter-decile'  # Individual decile data
    
    unified_df['Aggregation'] = unified_df.apply(determine_aggregation, axis=1)
    
    # Reorder columns to match specification
    column_order = [
        'Year',
        'Country', 
        'Decile',
        'Quintile',
        'Level',
        'EU priority',
        'Secondary',
        'Primary and raw data',
        'Type',
        'Aggregation',
        'Value'
    ]
    
    # Add datasource column for reference (optional, can be removed if not needed)
    if 'datasource' in unified_df.columns:
        column_order.append('datasource')
    
    # Select and reorder columns
    available_cols = [col for col in column_order if col in unified_df.columns]
    unified_df = unified_df[available_cols]
    
    # Sort by logical order
    sort_columns = ['Year', 'Country', 'Primary and raw data']
    if 'Decile' in unified_df.columns:
        # Custom sort for Decile to handle "All" properly
        unified_df['_decile_sort'] = unified_df['Decile'].apply(
            lambda x: 999 if x == 'All' else (int(x) if str(x).isdigit() else 998)
        )
        sort_columns.extend(['_decile_sort'])
    
    unified_df = unified_df.sort_values(sort_columns)
    
    # Remove temporary sorting column
    if '_decile_sort' in unified_df.columns:
        unified_df = unified_df.drop('_decile_sort', axis=1)
    
    print(f"✅ Created unified dataframe: {len(unified_df):,} rows, {len(unified_df.columns)} columns")
    
    return unified_df


def save_outputs(df: pd.DataFrame, validation: dict, dirs: dict) -> None:
    """
    Save final outputs and create specialized datasets.
    
    Args:
        df: Final combined DataFrame
        validation: Validation results dictionary
        dirs: Directory paths dictionary
    """
    print("💾 Saving output files...")
    
    # Create unified dataframe structure
    unified_df = create_unified_dataframe(df)
    
    # Main output files
    outputs = {
        'df_clean.csv': df,  # Keep original format for compatibility
        'df_clean_with_transport.csv': df,  # Same as main for compatibility
        'unified_app_data.csv': unified_df  # New unified format for the app
    }
    
    for filename, data in outputs.items():
        filepath = dirs['final_output'] / filename
        data.to_csv(filepath, index=False)
        print(f"✅ Saved: {filepath}")
    
    # Display unified dataframe structure
    print("\n📊 Unified Dataframe Structure:")
    print(f"   Shape: {unified_df.shape}")
    print(f"   Columns: {list(unified_df.columns)}")
    
    # Show sample data
    print("\n📋 Sample of Unified Data:")
    print(unified_df.head(10).to_string(index=False, max_cols=6))
    
    # Show unique values for key columns
    print(f"\n🔍 Unique Values Summary:")
    print(f"   Years: {sorted(unified_df['Year'].unique())}")
    print(f"   Countries: {unified_df['Country'].nunique()} unique")
    print(f"   Deciles: {sorted(unified_df['Decile'].unique(), key=lambda x: (str(x) != 'All', str(x)))}")
    print(f"   Indicators: {unified_df['Primary and raw data'].nunique()} unique")
    print(f"   Aggregation methods: {list(unified_df['Aggregation'].unique())}")
    
    # Create Elma-specific dataset (as in original notebook)
    elma_indicators = ['HE-SILC-1', 'HH-HBS-3', 'HH-HBS-4']
    elma_df = unified_df[unified_df['Primary and raw data'].isin(elma_indicators)]
    
    if not elma_df.empty:
        elma_path = dirs['final_output'] / f"2025-05-12_df_elma_unified.csv"
        elma_df.to_csv(elma_path, index=False)
        print(f"✅ Saved Elma dataset (unified format): {elma_path} ({len(elma_df):,} rows)")
    
    # Save validation report
    validation_path = dirs['final_output'] / "data_validation_report.txt"
    with open(validation_path, 'w') as f:
        f.write("European Well-Being Index (EWBI) - Data Integration Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET SUMMARY\n")
        f.write("-" * 15 + "\n")
        f.write(f"Total Records: {validation['total_rows']:,}\n")
        f.write(f"Countries: {validation['unique_countries']}\n")
        f.write(f"Years: {validation['year_range']}\n")
        f.write(f"Indicators: {validation['unique_indicators']}\n")
        f.write(f"Databases: {', '.join(validation['databases'])}\n\n")
        
        f.write("UNIFIED DATAFRAME STRUCTURE\n")
        f.write("-" * 30 + "\n")
        f.write(f"Shape: {unified_df.shape}\n")
        f.write(f"Columns: {', '.join(unified_df.columns)}\n\n")
        
        f.write("COUNTRIES INCLUDED\n")
        f.write("-" * 17 + "\n")
        for country in validation['countries']:
            f.write(f"  • {country}\n")
        f.write("\n")
        
        f.write("INDICATORS BY DATABASE\n")
        f.write("-" * 22 + "\n")
        for db, count in validation['indicators_by_db'].items():
            f.write(f"  • {db}: {count} indicators\n")
        f.write("\n")
        
        f.write("DATA QUALITY\n")
        f.write("-" * 12 + "\n")
        f.write(f"Value Range: {validation['value_stats']['min']:.2f} - {validation['value_stats']['max']:.2f}\n")
        f.write(f"Mean Value: {validation['value_stats']['mean']:.2f}\n")
        f.write(f"Zero Values: {validation['zero_values']:,} ({validation['zero_value_pct']:.1f}%)\n")
    
    print(f"✅ Saved validation report: {validation_path}")


def analyze_data_coverage(df: pd.DataFrame) -> None:
    """
    Analyze and display data coverage by indicator and country.
    
    Args:
        df: Final combined DataFrame
    """
    print("\n📈 Data Coverage Analysis:")
    print("-" * 30)
    
    # Coverage by database
    print("\nRecords by Database:")
    db_counts = df['datasource'].value_counts()
    for db, count in db_counts.items():
        pct = (count / len(df)) * 100
        print(f"  • {db}: {count:,} ({pct:.1f}%)")
    
    # Coverage by indicator type (first part of indicator code)
    print("\nIndicators by Category:")
    df['indicator_category'] = df['primary_index'].str.split('-').str[0]
    cat_counts = df['indicator_category'].value_counts()
    for cat, count in cat_counts.items():
        indicators = df[df['indicator_category'] == cat]['primary_index'].nunique()
        print(f"  • {cat}: {indicators} indicators, {count:,} records")
    
    # Top countries by data availability
    print("\nTop Countries by Data Points:")
    country_counts = df['country'].value_counts().head(10)
    for country, count in country_counts.items():
        print(f"  • {country}: {count:,} data points")


def create_pivot_analysis(df: pd.DataFrame, dirs: dict) -> None:
    """
    Create pivot table analyses for key indicators (as done in original notebook).
    
    Args:
        df: Final combined DataFrame
        dirs: Directory paths dictionary
    """
    print("📋 Creating pivot table analyses...")
    
    # Key indicators for analysis
    key_indicators = ['EL-SILC-1', 'HE-SILC-1', 'TT-HBS-2', 'RT-LFS-2']
    
    pivot_results = {}
    
    for indicator in key_indicators:
        if indicator in df['primary_index'].values:
            # Create pivot for decile 1
            decile_col = 'decile' if 'decile' in df.columns else 'quintile'
            
            filtered_df = df[
                (df[decile_col] == 1) & 
                (df['primary_index'] == indicator)
            ]
            
            if not filtered_df.empty and 'year' in filtered_df.columns:
                pivot = filtered_df.pivot(
                    index='country', 
                    columns='year', 
                    values='value'
                ).sort_index().sort_index(axis=1)
                
                pivot_results[indicator] = pivot
                
                # Save pivot table
                pivot_path = dirs['final_output'] / f"pivot_{indicator.replace('-', '_')}.csv"
                pivot.to_csv(pivot_path)
                print(f"✅ Saved pivot for {indicator}: {pivot_path}")
    
    return pivot_results


def main():
    """
    Main execution function for final data integration.
    """
    print("🚀 Starting Final Data Integration Pipeline")
    print("=" * 50)
    
    try:
        # Setup directories
        dirs = setup_directories()
        print(f"📂 Output directory: {dirs['output_base']}")
        print(f"📂 Input data directory: {dirs['input_data']}")
        
        # Step 1: Load all survey datasets
        datasets = load_survey_datasets(dirs)
        
        # Step 2: Process transport indicators (DISABLED - no transport data available)
        # transport_df = process_transport_indicators(dirs)
        transport_df = pd.DataFrame()  # Empty DataFrame for now
        
        # Step 3: Combine all datasets
        final_df = combine_all_datasets(datasets, transport_df)
        
        # Step 4: Perform data validation
        validation = perform_data_validation(final_df)
        
        # Step 5: Analyze data coverage
        analyze_data_coverage(final_df)
        
        # Step 6: Create pivot analyses
        pivot_results = create_pivot_analysis(final_df, dirs)
        
        # Step 7: Save all outputs
        save_outputs(final_df, validation, dirs)
        
        # Final summary
        print(f"\n📊 Integration Summary:")
        print(f"   • Total records: {len(final_df):,}")
        print(f"   • Countries: {validation['unique_countries']}")
        print(f"   • Years: {validation['year_range']}")
        print(f"   • Indicators: {validation['unique_indicators']}")
        print(f"   • Databases: {len(validation['databases'])}")
        
        print("\n✅ Final data integration completed successfully!")
        
        return final_df, validation
        
    except Exception as e:
        print(f"\n❌ Error during final integration: {str(e)}")
        raise


if __name__ == "__main__":
    # Execute main processing
    result, validation_info = main()
    
    # Display sample of final results
    if result is not None:
        print("\n📊 Sample of final integrated dataset:")
        print(result.head(10))
        print(f"\nFinal dataset shape: {result.shape}")
        print("\nUnique indicators:")
        for indicator in sorted(result['primary_index'].unique()):
            count = len(result[result['primary_index'] == indicator])
            print(f"  • {indicator}: {count:,} records")