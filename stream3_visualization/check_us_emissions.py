import pandas as pd
from pathlib import Path

def check_us_emissions():
    """Check raw US emissions data to verify territory vs consumption differences."""
    
    # Load the combined data
    data_path = Path("../Output/combined_data.csv")
    df = pd.read_csv(data_path)
    
    print("="*80)
    print("US EMISSIONS DATA VERIFICATION")
    print("="*80)
    
    # Filter for US data
    us_data = df[df['ISO2'] == 'US'].copy()
    
    print(f"Total US data rows: {len(us_data)}")
    print(f"Years available: {us_data['Year'].min()} - {us_data['Year'].max()}")
    print(f"Emissions scopes: {list(us_data['Emissions_scope'].unique())}")
    
    # Check data by scope
    for scope in us_data['Emissions_scope'].unique():
        scope_data = us_data[us_data['Emissions_scope'] == scope]
        print(f"\n{scope} Emissions:")
        print(f"  Rows: {len(scope_data)}")
        print(f"  Years: {scope_data['Year'].min()} - {scope_data['Year'].max()}")
        print(f"  Latest year: {scope_data['Year'].max()}")
    
    # Show sample years for comparison
    print("\n" + "="*80)
    print("SAMPLE YEARS COMPARISON")
    print("="*80)
    
    # Select key years to examine
    sample_years = [1970, 1980, 1990, 2000, 2010, 2020, 2022, 2023]
    
    for year in sample_years:
        year_data = us_data[us_data['Year'] == year]
        if not year_data.empty:
            print(f"\nYear {year}:")
            for _, row in year_data.iterrows():
                scope = row['Emissions_scope']
                annual = row['Annual_CO2_emissions_Mt']
                cumulative = row['Cumulative_CO2_emissions_Mt']
                population = row['Population']
                per_capita = row['Emissions_per_capita_ton']
                
                print(f"  {scope}:")
                print(f"    Annual: {annual:.2f} MtCO2")
                print(f"    Cumulative: {cumulative:.0f} MtCO2")
                print(f"    Population: {population:,.0f}")
                print(f"    Per capita: {per_capita:.2f} tCO2")
    
    # Calculate differences between scopes for key years
    print("\n" + "="*80)
    print("SCOPE DIFFERENCES ANALYSIS")
    print("="*80)
    
    for year in sample_years:
        year_data = us_data[us_data['Year'] == year]
        if len(year_data) == 2:  # Both scopes available
            territory = year_data[year_data['Emissions_scope'] == 'Territory'].iloc[0]
            consumption = year_data[year_data['Emissions_scope'] == 'Consumption'].iloc[0]
            
            annual_diff = consumption['Annual_CO2_emissions_Mt'] - territory['Annual_CO2_emissions_Mt']
            annual_diff_pct = (annual_diff / territory['Annual_CO2_emissions_Mt']) * 100 if territory['Annual_CO2_emissions_Mt'] != 0 else 0
            
            cumulative_diff = consumption['Cumulative_CO2_emissions_Mt'] - territory['Cumulative_CO2_emissions_Mt']
            cumulative_diff_pct = (cumulative_diff / territory['Cumulative_CO2_emissions_Mt']) * 100 if territory['Cumulative_CO2_emissions_Mt'] != 0 else 0
            
            print(f"\nYear {year}:")
            print(f"  Annual difference: {annual_diff:+.2f} MtCO2 ({annual_diff_pct:+.1f}%)")
            print(f"  Cumulative difference: {cumulative_diff:+.0f} MtCO2 ({cumulative_diff_pct:+.1f}%)")
    
    # Check for any anomalies
    print("\n" + "="*80)
    print("DATA QUALITY CHECKS")
    print("="*80)
    
    # Check for negative values
    negative_annual = us_data[us_data['Annual_CO2_emissions_Mt'] < 0]
    negative_cumulative = us_data[us_data['Cumulative_CO2_emissions_Mt'] < 0]
    
    if not negative_annual.empty:
        print(f"⚠️  Found {len(negative_annual)} rows with negative annual emissions")
        print(negative_annual[['Year', 'Emissions_scope', 'Annual_CO2_emissions_Mt']].to_string(index=False))
    
    if not negative_cumulative.empty:
        print(f"⚠️  Found {len(negative_cumulative)} rows with negative cumulative emissions")
        print(negative_cumulative[['Year', 'Emissions_scope', 'Cumulative_CO2_emissions_Mt']].to_string(index=False))
    
    # Check for missing values
    missing_annual = us_data[us_data['Annual_CO2_emissions_Mt'].isna()].sum()
    missing_cumulative = us_data[us_data['Cumulative_CO2_emissions_Mt'].isna()].sum()
    
    if missing_annual > 0:
        print(f"⚠️  Found {missing_annual} missing annual emissions values")
    
    if missing_cumulative > 0:
        print(f"⚠️  Found {missing_cumulative} missing cumulative emissions values")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("This analysis should help verify:")
    print("1. Whether territory and consumption emissions were similar in early years")
    print("2. When they started to diverge")
    print("3. The magnitude of differences in recent years")
    print("4. Data quality and consistency")
    
    return us_data

if __name__ == "__main__":
    us_data = check_us_emissions() 