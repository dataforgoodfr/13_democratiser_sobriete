import pandas as pd
import numpy as np
from pathlib import Path

def analyze_data_quality():
    """Detailed analysis of the combined data quality."""
    
    # Load the combined data
    data_path = Path("../Output/combined_data.csv")
    df = pd.read_csv(data_path)
    
    print("="*80)
    print("DETAILED DATA QUALITY ANALYSIS")
    print("="*80)
    
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Countries: {df['ISO2'].nunique()}")
    print(f"Emissions scopes: {list(df['Emissions_scope'].unique())}")
    
    # Analyze by emissions scope
    print("\n" + "="*60)
    print("ANALYSIS BY EMISSIONS SCOPE")
    print("="*60)
    
    for scope in df['Emissions_scope'].unique():
        scope_data = df[df['Emissions_scope'] == scope]
        print(f"\n{scope} Emissions:")
        print(f"  Rows: {len(scope_data):,}")
        print(f"  Countries: {scope_data['ISO2'].nunique()}")
        print(f"  Years: {scope_data['Year'].min()} - {scope_data['Year'].max()}")
        print(f"  Latest year with data: {scope_data['Year'].max()}")
        
        # Check for missing cumulative emissions
        missing_cum = scope_data['Cumulative_CO2_emissions_Mt'].isna().sum()
        print(f"  Missing cumulative emissions: {missing_cum:,}")
        
        # Check for negative cumulative emissions
        negative_cum = (scope_data['Cumulative_CO2_emissions_Mt'] < 0).sum()
        print(f"  Negative cumulative emissions: {negative_cum:,}")
        
        # Check for zero cumulative emissions
        zero_cum = (scope_data['Cumulative_CO2_emissions_Mt'] == 0).sum()
        print(f"  Zero cumulative emissions: {zero_cum:,}")
    
    # Analyze specific issues
    print("\n" + "="*60)
    print("SPECIFIC DATA ISSUES")
    print("="*60)
    
    # Negative emissions
    negative_emissions = df[df['Cumulative_CO2_emissions_Mt'] < 0]
    if not negative_emissions.empty:
        print(f"\n⚠️  Negative cumulative emissions found:")
        print(negative_emissions[['ISO2', 'Country', 'Year', 'Emissions_scope', 'Cumulative_CO2_emissions_Mt']].to_string(index=False))
    
    # Missing emissions
    missing_emissions = df[df['Cumulative_CO2_emissions_Mt'].isna()]
    if not missing_emissions.empty:
        print(f"\n⚠️  Missing cumulative emissions:")
        print(f"Total missing: {len(missing_emissions):,}")
        
        # Group by country and scope
        missing_by_country = missing_emissions.groupby(['ISO2', 'Country', 'Emissions_scope']).size().reset_index(name='count')
        print("\nMissing data by country and scope (top 10):")
        print(missing_by_country.sort_values('count', ascending=False).head(10).to_string(index=False))
    
    # Zero emissions
    zero_emissions = df[df['Cumulative_CO2_emissions_Mt'] == 0]
    if not zero_emissions.empty:
        print(f"\nℹ️   Zero cumulative emissions:")
        print(f"Total zero: {len(zero_emissions):,}")
        
        # Group by country and scope
        zero_by_country = zero_emissions.groupby(['ISO2', 'Country', 'Emissions_scope']).size().reset_index(name='count')
        print("\nZero emissions by country and scope (top 10):")
        print(zero_by_country.sort_values('count', ascending=False).head(10).to_string(index=False))
    
    # Analyze US and World data specifically
    print("\n" + "="*60)
    print("US AND WORLD DATA ANALYSIS")
    print("="*60)
    
    us_data = df[df['ISO2'] == 'US']
    world_data = df[df['ISO2'] == 'WLD']
    
    print("\nUS Data:")
    for scope in us_data['Emissions_scope'].unique():
        scope_data = us_data[us_data['Emissions_scope'] == scope]
        latest_year = scope_data['Year'].max()
        latest_emissions = scope_data[scope_data['Year'] == latest_year]['Cumulative_CO2_emissions_Mt'].iloc[0]
        print(f"  {scope}: {latest_year} = {latest_emissions:,.0f} MtCO2")
    
    print("\nWorld Data:")
    for scope in world_data['Emissions_scope'].unique():
        scope_data = world_data[world_data['Emissions_scope'] == scope]
        latest_year = scope_data['Year'].max()
        latest_emissions = scope_data[scope_data['Year'] == latest_year]['Cumulative_CO2_emissions_Mt'].iloc[0]
        print(f"  {scope}: {latest_year} = {latest_emissions:,.0f} MtCO2")
    
    # Check data completeness by year
    print("\n" + "="*60)
    print("DATA COMPLETENESS BY YEAR")
    print("="*60)
    
    for scope in df['Emissions_scope'].unique():
        scope_data = df[df['Emissions_scope'] == scope]
        print(f"\n{scope} Emissions - Data completeness by year:")
        
        year_counts = scope_data.groupby('Year').size().reset_index(name='count')
        year_counts = year_counts.sort_values('Year')
        
        # Show years with missing data
        expected_countries = scope_data['ISO2'].nunique()
        incomplete_years = year_counts[year_counts['count'] < expected_countries]
        
        if not incomplete_years.empty:
            print("  Years with incomplete data:")
            for _, row in incomplete_years.iterrows():
                missing = expected_countries - row['count']
                print(f"    {int(row['Year'])}: {int(row['count'])} countries (missing {missing})")
        else:
            print("  All years have complete data")
    
    return df

if __name__ == "__main__":
    df = analyze_data_quality() 