# Filter out rows where neutrality could not be calculated
original_rows = len(scenario_params)
scenario_params = scenario_params[scenario_params['Years_to_neutrality_from_latest_available'] != "N/A"].copy()
print(f"\nFiltered out {original_rows - len(scenario_params)} rows with 'N/A' neutrality years from scenario parameters.")

# Create a scenario_id for each unique combination
scenario_params['scenario_id'] = range(1, len(scenario_params) + 1)

# Fix aggregate capacity values by summing individual country capacities
print("\n=== Fixing Aggregate Capacity Values ===")

# Get the capacity data for individual countries (exclude aggregates)
individual_capacity = scenario_params[
    (~scenario_params['ISO2'].isin(['WLD', 'G20', 'EU'])) & 
    (scenario_params['Budget_distribution_scenario'] == 'Capacity')
].copy()

# Define which countries belong to each aggregate
aggregate_members = {
    'G20': ['AR', 'AU', 'BR', 'CA', 'CN', 'FR', 'DE', 'IN', 'ID', 'IT', 'JP', 'MX', 'RU', 'SA', 'ZA', 'KR', 'TR', 'GB', 'US'],
    'EU': ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE']
}

# Get IPCC regions from the data
ipcc_regions_list = scenario_params[scenario_params['Country'] == 'All']['ISO2'].unique()
ipcc_regions_list = [r for r in ipcc_regions_list if r not in ['WLD', 'G20', 'EU']]

# Calculate capacity for each aggregate
for aggregate_iso in ['G20', 'EU'] + list(ipcc_regions_list):
    print(f"  Processing {aggregate_iso}...")
    
    if aggregate_iso in aggregate_members:
        # For G20 and EU, use predefined member lists
        member_countries = aggregate_members[aggregate_iso]
    else:
        # For IPCC regions, get countries in that region
        member_countries = scenario_params[
            (scenario_params['Region'] == aggregate_iso) & 
            (scenario_params['Country'] != 'All') &
            (~scenario_params['ISO2'].isin(['WLD', 'G20', 'EU']))
        ]['ISO2'].unique().tolist()
    
    # Calculate aggregate capacity for each scope
    for scope in ['Territory', 'Consumption']:
        # Get capacity data for member countries
        member_capacity = individual_capacity[
            (individual_capacity['ISO2'].isin(member_countries)) &
            (individual_capacity['Emissions_scope'] == scope)
        ]
        
        if len(member_capacity) > 0:
            # Sum the capacity values of member countries
            aggregate_capacity = member_capacity['share_of_capacity'].sum()
            
            # Update the aggregate row in scenario_params
            mask = (scenario_params['ISO2'] == aggregate_iso) & (scenario_params['Emissions_scope'] == scope)
            if mask.any():
                scenario_params.loc[mask, 'share_of_capacity'] = aggregate_capacity
                print(f"    {scope}: {len(member_capacity)} countries, total capacity = {aggregate_capacity:.6f}")
            else:
                print(f"    {scope}: No data found for {aggregate_iso}")
        else:
            print(f"    {scope}: No member countries with capacity data")

print("=== End Aggregate Capacity Fix ===\n")

# Ensure NA, TR, and US are included
required_isos = ['NA', 'TR', 'US']
for iso in required_isos:
    if iso not in scenario_params['ISO2'].values:
        print(f"Warning: ISO2 code {iso} is missing in scenario parameters.") 