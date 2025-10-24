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
    # 'HQ-SILC-2',  # KEEP THIS NEW INDICATOR - housing quality satisfier
    'HH-SILC-1', 'HH-HBS-1', 'HH-HBS-2', 'HH-HBS-3', 'HH-HBS-4',
    'HE-HBS-1', 'HE-HBS-2',
    'EC-HBS-1', 'EC-HBS-2',
    'ED-ICT-1',
    'AC-SILC-1', 'AC-HBS-1', 'AC-HBS-2',
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
    
    
    # Skip EHIS dataset (EHIS indicators are excluded from EWBI)
    print("⏭️  Skipping EHIS data loading (EHIS indicators excluded from EWBI)")
    datasets['ehis'] = pd.DataFrame()  # Empty dataframe to avoid issues in later processing
    
    
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
            
            # Determine datasource based on the most common in the group
            datasource = group_data['datasource'].mode()[0] if not group_data['datasource'].mode().empty else 'Unknown'
            
            aggregated_rows.append({
                'year': year,
                'country': 'All Countries',
                income_group_col: income_group,
                'primary_index': primary_index,
                'value': median_value,
                'datasource': datasource
            })
    
    # Convert to DataFrame and combine with original data
    if aggregated_rows:
        aggregated_df = pd.DataFrame(aggregated_rows)
        
        # Add missing columns with NaN if they don't exist
        for col in df.columns:
            if col not in aggregated_df.columns:
                aggregated_df[col] = pd.NA
                
        # Reorder columns to match original DataFrame
        aggregated_df = aggregated_df[df.columns]
        
        # Combine with original data
        final_with_aggregated = pd.concat([df, aggregated_df], axis=0, ignore_index=True)
        
        print(f"✅ Added {len(aggregated_rows):,} aggregated rows for 'All Countries'")
        return final_with_aggregated
    else:
        print("⚠️ No aggregated data created")
        return df


def save_outputs(final_df: pd.DataFrame, validation: dict, dirs: dict) -> None:
    """
    Save final outputs and validation reports.
    
    Args:
        final_df: Final combined DataFrame
        validation: Validation results dictionary
        dirs: Directory paths dictionary
    """
    print("💾 Saving final outputs...")
    
    # Save main unified dataset
    final_output_file = dirs['final_output'] / "unified_app_data.csv"
    final_df.to_csv(final_output_file, index=False)
    print(f"✅ Saved unified dataset: {final_output_file}")
    
    # Save validation report
    validation_file = dirs['final_output'] / "data_validation_report.txt"
    with open(validation_file, 'w') as f:
        f.write("=== EWBI Data Validation Report ===\n\n")
        f.write(f"Dataset Summary:\n")
        f.write(f"- Total rows: {validation['total_rows']:,}\n")
        f.write(f"- Countries: {validation['unique_countries']}\n") 
        f.write(f"- Years: {validation['year_range']}\n")
        f.write(f"- Indicators: {validation['unique_indicators']}\n")
        f.write(f"- Databases: {', '.join(validation['databases'])}\n\n")
        
        f.write(f"Countries: {', '.join(validation['countries'])}\n\n")
        
        f.write(f"Indicators by Database:\n")
        for db, count in validation['indicators_by_db'].items():
            f.write(f"- {db}: {count} indicators\n")
        f.write("\n")
        
        f.write(f"Value Statistics:\n")
        f.write(f"- Min: {validation['value_stats']['min']:.2f}\n")
        f.write(f"- Max: {validation['value_stats']['max']:.2f}\n")
        f.write(f"- Mean: {validation['value_stats']['mean']:.2f}\n")
        f.write(f"- Median: {validation['value_stats']['median']:.2f}\n")
        f.write(f"- Zero values: {validation['zero_values']:,} ({validation['zero_value_pct']:.1f}%)\n")
    
    print(f"✅ Saved validation report: {validation_file}")


def main():
    """
    Main execution function for final data integration.
    """
    print("🚀 Starting Final Data Integration Pipeline")
    print("=" * 50)
    
    try:
        # Setup directories
        dirs = setup_directories()
        print(f"📂 Input data directory: {dirs['input_data']}")
        print(f"📂 Output directory: {dirs['final_output']}")
        
        # Step 1: Load all survey datasets
        datasets = load_survey_datasets(dirs)
        
        # Step 2: Process transport indicators
        transport_df = process_transport_indicators(dirs)
        
        # Step 3: Combine all datasets
        final_df = combine_all_datasets(datasets, transport_df)
        
        # Step 4: Create aggregated data for "All countries"
        final_df = create_aggregated_data(final_df)
        
        # Step 5: Perform data validation
        validation = perform_data_validation(final_df)
        
        # Step 6: Save outputs
        save_outputs(final_df, validation, dirs)
        
        # Summary statistics
        print("\n📈 Integration Summary:")
        print(f"   • Total records processed: {len(final_df):,}")
        print(f"   • Countries: {final_df['country'].nunique()}")
        print(f"   • Years: {final_df['year'].min()}-{final_df['year'].max()}")
        print(f"   • Indicators: {final_df['primary_index'].nunique()}")
        print(f"   • Databases: {', '.join(final_df['datasource'].unique())}")
        
        print("\n✅ Final data integration completed successfully!")
        
        return final_df
        
    except Exception as e:
        print(f"\n❌ Error during final data integration: {str(e)}")
        raise


if __name__ == "__main__":
    # Execute main processing
    result = main()
    
    # Display sample of final results
    if result is not None:
        print("\n📊 Sample of final results:")
        print(result.head(10))
        print(f"\nFinal dataset shape: {result.shape}")
        
        # Show new LFS indicators if present
        new_lfs_indicators = ['RT-LFS-4', 'RT-LFS-5', 'RT-LFS-6', 'RT-LFS-7', 'RT-LFS-8', 'EL-LFS-2']
        present_indicators = [ind for ind in new_lfs_indicators if ind in result['primary_index'].unique()]
        if present_indicators:
            print(f"\n✅ New LFS indicators present: {present_indicators}")
        else:
            print(f"\n⚠️ New LFS indicators not found in final dataset")