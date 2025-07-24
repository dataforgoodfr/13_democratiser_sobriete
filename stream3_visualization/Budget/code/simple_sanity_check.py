#!/usr/bin/env python3
"""
Simple Sanity Check for Capacity Scenario
Just filters data from the existing output files.
"""

import pandas as pd

def main():
    print("=== SIMPLE CAPACITY SCENARIO SANITY CHECK ===\n")
    
    # Load the data files
    print("Loading data files...")
    combined_df = pd.read_csv('../Output/combined_data.csv')
    scenario_df = pd.read_csv('../Output/scenario_parameters.csv')
    
    # Countries to check
    countries = ['US', 'FR', 'DZ', 'WLD']
    country_names = {
        'US': 'United States of America',
        'FR': 'France', 
        'DZ': 'Algeria',
        'WLD': 'World'
    }
    
    print("=== DATA FROM COMBINED_DATA.CSV ===\n")
    
    for country_code in countries:
        print(f"--- {country_names[country_code]} ({country_code}) ---")
        
        # Get latest year with GDP data for this country
        country_data = combined_df[combined_df['ISO2'] == country_code]
        latest_year = country_data[country_data['GDP_PPP'] > 0]['Year'].max()
        
        print(f"Latest year with GDP data: {latest_year}")
        
        for scope in ['Territory', 'Consumption']:
            data = country_data[
                (country_data['Year'] == latest_year) & 
                (country_data['Emissions_scope'] == scope)
            ]
            
            if len(data) > 0:
                row = data.iloc[0]
                gdp_per_capita = row['GDP_PPP'] / row['Population'] if row['Population'] > 0 else 0
                
                print(f"  {scope}:")
                print(f"    GDP PPP: {row['GDP_PPP']:,.0f} $")
                print(f"    Population: {row['Population']:,.0f}")
                print(f"    GDP per capita: {gdp_per_capita:,.0f} $/person")
                print(f"    Annual emissions: {row['Annual_CO2_emissions_Mt']:,.2f} MtCO2")
                print(f"    Cumulative emissions: {row['Cumulative_CO2_emissions_Mt']:,.2f} MtCO2")
                print(f"    Share of capacity: {row['share_of_capacity']:.6f}")
                print(f"    Share of population: {row['share_of_population']:.6f}")
                print(f"    Share of GDP: {row['share_of_GDP_PPP']:.6f}")
        
        # Get cumulative population from 2050
        pop_2050 = country_data[
            (country_data['Year'] == 2050) & 
            (country_data['Emissions_scope'] == 'Territory')
        ]
        if len(pop_2050) > 0:
            print(f"  Cumulative population 1970-2050: {pop_2050.iloc[0]['Cumulative_population']:,.0f}")
    
    print("\n=== DATA FROM SCENARIO_PARAMETERS.CSV ===\n")
    
    # Filter scenario data for capacity scenario
    capacity_scenarios = scenario_df[scenario_df['Budget_distribution_scenario'] == 'Capacity']
    
    print(f"Capacity scenarios found: {len(capacity_scenarios)}")
    
    # Show sample of capacity scenario data
    if len(capacity_scenarios) > 0:
            print("\nSample capacity scenario data:")
            sample_cols = ['scenario_id', 'ISO2', 'Country', 'Emissions_scope', 'Country_carbon_budget']
            print(capacity_scenarios[sample_cols].head(10))
            
            # Show world aggregate capacity scenario
            world_capacity = capacity_scenarios[capacity_scenarios['ISO2'] == 'WLD']
            if len(world_capacity) > 0:
                print("\nWorld aggregate capacity scenarios:")
                print(world_capacity[['scenario_id', 'Emissions_scope', 'Country_carbon_budget']])

if __name__ == "__main__":
    main() 