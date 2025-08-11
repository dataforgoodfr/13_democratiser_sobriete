# Filter out rows where neutrality could not be calculated
original_rows = len(scenario_params)
scenario_params = scenario_params[scenario_params['Years_to_neutrality_from_latest_available'] != "N/A"].copy()
print(f"\nFiltered out {original_rows - len(scenario_params)} rows with 'N/A' neutrality years from scenario parameters.")

# Create a scenario_id for each unique combination
scenario_params['scenario_id'] = range(1, len(scenario_params) + 1)

# Add missing aggregate calculations for G20, EU, and IPCC regions
print("\n=== Adding Missing Aggregate Calculations ===")

# Define the aggregates we need to create
aggregates_to_create = [
    'G20', 'EU', 'Asia-Pacific Developed', 'Eastern Asia', 'Eurasia', 'Europe',
    'Latin America and Caribbean', 'Middle East', 'North America', 
    'South-East Asia and developing Pacific', 'Southern Asia'
]

# For each aggregate, create the missing scenarios
for aggregate_name in aggregates_to_create:
    print(f"\nProcessing {aggregate_name}...")
    
    # Get the existing aggregate data to use as template
    existing_aggregate = scenario_params[
        (scenario_params['Country'] == 'All') & 
        (scenario_params['Region'] == aggregate_name)
    ]
    
    if len(existing_aggregate) > 0:
        print(f"  Found existing data for {aggregate_name}")
        
        # Get all unique combinations of emissions_scope, warming_scenario, probability
        existing_combinations = existing_aggregate[['Emissions_scope', 'Warming_scenario', 'Probability_of_reach']].drop_duplicates()
        
        for _, combo in existing_combinations.iterrows():
            emissions_scope = combo['Emissions_scope']
            warming_scenario = combo['Warming_scenario']
            probability = combo['Probability_of_reach']
            
            # Check if Capacity scenario exists for this combination
            capacity_exists = scenario_params[
                (scenario_params['Country'] == 'All') &
                (scenario_params['Region'] == aggregate_name) &
                (scenario_params['Emissions_scope'] == emissions_scope) &
                (scenario_params['Warming_scenario'] == warming_scenario) &
                (scenario_params['Probability_of_reach'] == probability) &
                (scenario_params['Budget_distribution_scenario'] == 'Capacity')
            ]
            
            if len(capacity_exists) == 0:
                # Capacity scenario is missing, create it
                print(f"    Adding missing Capacity scenario for {emissions_scope} {warming_scenario} {probability}")
                
                # Get template row from existing data
                template_row = existing_aggregate[
                    (existing_aggregate['Emissions_scope'] == emissions_scope) &
                    (existing_aggregate['Warming_scenario'] == warming_scenario) &
                    (existing_aggregate['Probability_of_reach'] == probability) &
                    (existing_aggregate['Budget_distribution_scenario'] == 'Population')
                ].iloc[0]
                
                # Create Capacity scenario
                capacity_scenario = {
                    'ISO2': aggregate_name,
                    'Country': 'All',
                    'Region': aggregate_name,
                    'Emissions_scope': emissions_scope,
                    'Warming_scenario': warming_scenario,
                    'Probability_of_reach': probability,
                    'Budget_distribution_scenario': 'Capacity',
                    'Years_to_neutrality_from_latest_available': template_row['Years_to_neutrality_from_latest_available'],
                    'Years_to_neutrality_from_today': template_row['Years_to_neutrality_from_today'],
                    'Neutrality_year': template_row['Neutrality_year'],
                    'Latest_year': template_row['Latest_year'],
                    'Latest_population': template_row['Latest_population'],
                    'Latest_annual_CO2_emissions_Mt': template_row['Latest_annual_CO2_emissions_Mt'],
                    'Latest_cumulative_CO2_emissions_Mt': template_row['Latest_cumulative_CO2_emissions_Mt'],
                    'Latest_emissions_per_capita_t': template_row['Latest_emissions_per_capita_t'],
                    'Latest_cumulative_population': template_row['Latest_cumulative_population'],
                    'Share_of_cumulative_population_Latest_to_2050': template_row['Share_of_cumulative_population_Latest_to_2050'],
                    'Share_of_cumulative_population_1970_to_2050': template_row['Share_of_cumulative_population_1970_to_2050'],
                    'Share_of_cumulative_population_1970_to_latest': template_row['Share_of_cumulative_population_1970_to_latest'],
                    'share_of_capacity': template_row['share_of_capacity'],
                    'Global_Carbon_budget': template_row['Global_Carbon_budget'],
                    'Country_carbon_budget': template_row['Country_carbon_budget'],
                    'Country_budget_per_capita': template_row['Country_budget_per_capita'],
                    'Share_of_cumulative_emissions': template_row['Share_of_cumulative_emissions'],
                    'scenario_id': len(scenario_params) + 1
                }
                
                # Add to scenario_params
                scenario_params = pd.concat([scenario_params, pd.DataFrame([capacity_scenario])], ignore_index=True)
                print(f"      Added Capacity scenario with neutrality year {template_row['Neutrality_year']}")
    else:
        print(f"  No existing data found for {aggregate_name} - will need to create from scratch")

# Regenerate scenario_id values since we added new rows
scenario_params['scenario_id'] = range(1, len(scenario_params) + 1)

# Ensure NA, TR, and US are included
required_isos = ['NA', 'TR', 'US']
for iso in required_isos:
    if iso not in scenario_params['ISO2'].values:
        print(f"Warning: ISO2 code {iso} is missing in scenario parameters.") 