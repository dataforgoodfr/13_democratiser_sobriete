"""
EWBI Data Loader - Shared utility for loading European Well-Being Index data

This module provides standardized functions to load EWBI data from the Well-being pipeline
for use across all reports in the Well-being Reports directory.

Usage:
    from shared.code.ewbi_data_loader import load_ewbi_unified_data, get_country_data
    
    # Load the main EWBI dataset
    df = load_ewbi_unified_data()
    
    # Get data for specific country
    swiss_data = get_country_data(df, 'CH')
    eu_data = get_country_data(df, 'All Countries')
"""

import pandas as pd
import os
import sys
from pathlib import Path


def get_well_being_data_path():
    """Get the absolute path to the Well-being output directory"""
    current_file = Path(__file__).resolve()
    # Navigate from shared/code -> shared -> reports -> Well-being/output
    shared_code_dir = current_file.parent               # Up to shared/code
    shared_dir = shared_code_dir.parent                 # Up to shared
    reports_dir = shared_dir.parent                     # Up to reports
    well_being_dir = reports_dir.parent                 # Up to Well-being (reports is inside Well-being)
    well_being_output = well_being_dir / 'output'
    
    if not well_being_output.exists():
        raise FileNotFoundError(f"Well-being output directory not found: {well_being_output}")
    
    return well_being_output


def load_ewbi_unified_data():
    """
    Load the main EWBI unified dataset with all levels and PCA weighting
    
    Returns:
        pd.DataFrame: The unified EWBI dataset with all levels (1-5) and countries
    """
    data_path = get_well_being_data_path() / 'ewbi_master_aggregated.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"EWBI unified data file not found: {data_path}")
    
    print(f"Loading EWBI data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded EWBI data with shape: {df.shape}")
    
    return df


def get_country_data(df, country_code, level=None, decile='All'):
    """
    Extract data for a specific country from the EWBI dataset
    
    Args:
        df (pd.DataFrame): The EWBI unified dataset
        country_code (str): Country code ('CH', 'FR', 'All Countries', etc.)
        level (int, optional): Filter by specific level (1-5). If None, includes all levels
        decile (str): Filter by decile ('All', '1.0', '2.0', ..., '10.0')
    
    Returns:
        pd.DataFrame: Filtered data for the specified country
    """
    filtered_df = df[
        (df['Country'] == country_code) & 
        (df['Decile'] == decile)
    ].copy()
    
    if level is not None:
        filtered_df = filtered_df[filtered_df['Level'] == level]
    
    return filtered_df


def get_eu_priorities():
    """
    Get the list of EU priorities available in the EWBI system
    
    Returns:
        list: List of EU priority names
    """
    return [
        'Energy and Housing', 
        'Equality',
        'Health and Animal Welfare',
        'Intergenerational Fairness, Youth, Culture and Sport',
        'Social Rights and Skills, Quality Jobs and Preparedness'
    ]


def get_primary_indicators_for_priority(df, eu_priority):
    """
    Get all primary indicators (Level 5) for a specific EU priority
    
    Args:
        df (pd.DataFrame): The EWBI unified dataset
        eu_priority (str): EU priority name
    
    Returns:
        list: List of primary indicator codes for the given priority
    """
    indicators = df[
        (df['EU priority'] == eu_priority) & 
        (df['Primary and raw data'].notna()) &
        (df['Level'] == 5)
    ]['Primary and raw data'].unique()
    
    return sorted(indicators)


def get_housing_energy_indicators(df):
    """
    Get all primary indicators related to Housing and Energy
    
    Args:
        df (pd.DataFrame): The EWBI unified dataset
    
    Returns:
        list: List of housing and energy indicator codes
    """
    return get_primary_indicators_for_priority(df, 'Energy and Housing')


def load_country_comparison_data(df, country1_code, country2_code, level=5, decile='All'):
    """
    Load data for comparing two countries
    
    Args:
        df (pd.DataFrame): The EWBI unified dataset
        country1_code (str): First country code
        country2_code (str): Second country code  
        level (int): Level to filter by (1-5)
        decile (str): Decile to filter by
    
    Returns:
        tuple: (country1_data, country2_data) as DataFrames
    """
    country1_data = get_country_data(df, country1_code, level, decile)
    country2_data = get_country_data(df, country2_code, level, decile)
    
    return country1_data, country2_data


def get_time_series_data(df, country_code, indicator_code, include_deciles=False):
    """
    Get time series data for a specific country and indicator
    
    Args:
        df (pd.DataFrame): The EWBI unified dataset
        country_code (str): Country code
        indicator_code (str): Primary indicator code
        include_deciles (bool): Whether to include decile breakdown
    
    Returns:
        dict: Dictionary with 'overall' data and optionally 'deciles' data
    """
    result = {}
    
    # Get overall data (decile='All')
    overall_data = df[
        (df['Country'] == country_code) &
        (df['Primary and raw data'] == indicator_code) &
        (df['Decile'] == 'All')
    ].copy()
    
    if not overall_data.empty:
        result['overall'] = overall_data.sort_values('Year')
    
    # Get decile data if requested
    if include_deciles:
        decile_data = df[
            (df['Country'] == country_code) &
            (df['Primary and raw data'] == indicator_code) &
            (df['Decile'] != 'All') &
            (df['Decile'].notna())
        ].copy()
        
        if not decile_data.empty:
            result['deciles'] = decile_data.sort_values(['Year', 'Decile'])
    
    return result


def validate_ewbi_data(df):
    """
    Validate the EWBI dataset structure
    
    Args:
        df (pd.DataFrame): The EWBI dataset to validate
    
    Returns:
        dict: Validation results with counts and available data
    """
    required_columns = ['Country', 'Year', 'Level', 'Decile', 'Value', 'Primary and raw data', 'EU priority']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    validation_results = {
        'total_rows': len(df),
        'countries': sorted(df['Country'].unique()),
        'years': sorted(df['Year'].unique()),
        'levels': sorted(df['Level'].unique()),
        'deciles': sorted(df['Decile'].unique()),
        'eu_priorities': sorted(df['EU priority'].dropna().unique()),
        'primary_indicators_count': len(df['Primary and raw data'].dropna().unique())
    }
    
    return validation_results


if __name__ == "__main__":
    """Test the data loader functions"""
    try:
        # Load data
        df = load_ewbi_unified_data()
        
        # Validate data
        validation = validate_ewbi_data(df)
        print("\nData validation results:")
        for key, value in validation.items():
            print(f"  {key}: {value}")
        
        # Test country data extraction
        print(f"\nTesting data extraction...")
        swiss_data = get_country_data(df, 'CH', level=5)
        print(f"Switzerland Level 5 data: {len(swiss_data)} rows")
        
        eu_data = get_country_data(df, 'All Countries', level=5)
        print(f"EU-27 Level 5 data: {len(eu_data)} rows")
        
        # Test housing & energy indicators
        housing_indicators = get_housing_energy_indicators(df)
        print(f"\nHousing & Energy indicators: {len(housing_indicators)}")
        for indicator in housing_indicators[:5]:  # Show first 5
            print(f"  • {indicator}")
        if len(housing_indicators) > 5:
            print(f"  ... and {len(housing_indicators) - 5} more")
        
        print("\n✅ EWBI data loader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing data loader: {e}")