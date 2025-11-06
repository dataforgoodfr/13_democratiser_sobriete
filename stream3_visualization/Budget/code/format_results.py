import pandas as pd
import os

# Get the absolute path to the output directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))

print("Creating neutrality year table from scenario parameters...")
print("Loading scenario parameters data...")

# Load the scenario parameters data
scenario_parameters = pd.read_csv(os.path.join(DATA_DIR, 'scenario_parameters.csv'))

print("Processing map data combinations...")
print(f"Total rows in scenario data: {len(scenario_parameters)}")
print(f"Unique countries: {len(scenario_parameters['Country'].unique())}")

# Filter exactly like the app.py does:
# - Fixed: Warming_scenario = '1.5°C'
# - Fixed: Probability_of_reach = '50%' (most common scenario)
# - Exclude aggregates (WLD, EU, G20)
# - Only get individual countries (ISO_Type = 'Country')
base_filter = (
    (scenario_parameters['Warming_scenario'] == '1.5°C') &
    (scenario_parameters['Probability_of_reach'] == '50%') &
    (~scenario_parameters['ISO2'].isin(['WLD', 'EU', 'G20'])) &
    (scenario_parameters['ISO_Type'] == 'Country')  # Only individual countries
)

# Create the four combinations exactly as they appear in the app dropdowns
combinations = [
    ('Responsibility', 'Territory', 'Responsibility-Territorial'),
    ('Responsibility', 'Consumption', 'Responsibility-Consumption'), 
    ('Capability', 'Territory', 'Capability-Territorial'),
    ('Capability', 'Consumption', 'Capability-Consumption')
]

# Get all unique countries from the base filtered data
all_countries = scenario_parameters[base_filter][['Country', 'ISO2']].drop_duplicates().sort_values('Country')
results_df = all_countries[['Country']].copy()

print(f"Found {len(results_df)} individual countries")

# For each combination, extract the neutrality year
for budget_scenario, emissions_scope, col_name in combinations:
    print(f"Processing {col_name}...")
    
    # Filter exactly like app.py does for the map
    filtered_data = scenario_parameters[
        base_filter &
        (scenario_parameters['Budget_distribution_scenario'] == budget_scenario) &
        (scenario_parameters['Emissions_scope'] == emissions_scope)
    ].copy()
    
    if not filtered_data.empty:
        # Convert to numeric like in app.py
        filtered_data['Neutrality_year_numeric'] = pd.to_numeric(filtered_data['Neutrality_year'], errors='coerce')
        
        # Get neutrality year for each country
        neutrality_data = filtered_data[['Country', 'Neutrality_year_numeric']].copy()
        neutrality_data.rename(columns={'Neutrality_year_numeric': col_name}, inplace=True)
        
        # Merge with results
        results_df = results_df.merge(neutrality_data[['Country', col_name]], on='Country', how='left')
    else:
        # If no data for this combination, set to NaN
        results_df[col_name] = None

# Convert years to integers where possible (NaN will remain NaN)
for col in ['Responsibility-Territorial', 'Responsibility-Consumption', 
           'Capability-Territorial', 'Capability-Consumption']:
    if col in results_df.columns:
        results_df[col] = results_df[col].astype('Int64')  # nullable integer type

# Create a final formatted table
print("Neutrality Year Table: Values displayed on the Budget Dashboard Map")
print("="*70)
print()
print("This table shows the neutrality year (when carbon budget = 0) for each country")
print("across the 4 scenario combinations displayed in the app.py dashboard map")
print("(Fixed: 1.5°C warming scenario with 50% probability)")
print()

print("Key findings:")
print(f"• {len(results_df)} individual countries analyzed")

# Count countries with data for each scenario
for col in ['Responsibility-Territorial', 'Responsibility-Consumption', 
           'Capability-Territorial', 'Capability-Consumption']:
    if col in results_df.columns:
        count = len(results_df[results_df[col].notna()])
        print(f"• {count} countries have {col.replace('-', ' ')} neutrality data")
        
        # Show year range
        if count > 0:
            min_year = results_df[col].min()
            max_year = results_df[col].max()
            avg_year = round(results_df[col].mean(), 1)
            print(f"  - Year range: {min_year} to {max_year} (avg: {avg_year})")

print()

# Show first 20 rows as example
print("Sample of results (first 20 countries):")
print("-" * 70)
print(results_df.head(20).to_string(index=False))
print()
print("...")
print()

# Save the results to Output folder
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'map_neutrality_years_by_country.csv')
results_df.to_csv(output_file, index=False)
print(f"Full table saved to: {output_file}")

# Create a summary table
summary_data = []
for col in ['Responsibility-Territorial', 'Responsibility-Consumption', 
           'Capability-Territorial', 'Capability-Consumption']:
    if col in results_df.columns:
        data_col = results_df[col]
        summary_data.append({
            'Scenario': col.replace('-', ' '),
            'Countries with Data': len(data_col[data_col.notna()]),
            'Earliest Neutrality Year': data_col.min() if data_col.notna().any() else None,
            'Latest Neutrality Year': data_col.max() if data_col.notna().any() else None,
            'Average Neutrality Year': round(data_col.mean(), 1) if data_col.notna().any() else None
        })

coverage_summary = pd.DataFrame(summary_data)

print("Summary of Neutrality Years (Map Data):")
print("-" * 50)
print(coverage_summary.to_string(index=False))

# Save the summary
summary_output_file = os.path.join(output_dir, 'map_neutrality_summary.csv')
coverage_summary.to_csv(summary_output_file, index=False)
print(f"\nSummary saved to: {summary_output_file}")

print("\nThis table contains the exact neutrality years displayed on the dashboard map")
print("for each combination of Budget Distribution (Responsibility/Capability) ×")
print("Emissions Scope (Territorial/Consumption) scenarios.")