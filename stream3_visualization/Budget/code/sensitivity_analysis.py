"""
Sensitivity Analysis for Carbon Budget Scenarios
=================================================

This script tests different carbon budget assumptions to analyze their impact on
country neutrality years. It compares alternative budget scenarios with the base case
(2025 baseline) using different:
- Starting years: 2018 and 2021
- Temperature targets: 1.5°C and 2°C
- Probabilities: 50% and 67%
- Budget allocations: Responsibility and Capacity
- Emissions scopes: Territorial and Consumption

The analysis helps understand how sensitive neutrality year calculations are to
different budget assumptions and starting dates.
"""

import pandas as pd
import numpy as np
import os

# Configuration
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output', 'Sensitivity check'))

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define alternative budget scenarios (MtCO2)
ALTERNATIVE_BUDGETS = {
    2018: {
        '1.5°C': {'50%': 580000, '67%': 420000},
        '2°C': {'50%': 1500000, '67%': 1170000}
    },
    2021: {
        '1.5°C': {'50%': 500000, '67%': 400000},
        '2°C': {'50%': 1350000, '67%': 1150000}
    }
}

# Base case budget from 2025 (for comparison)
BASE_BUDGETS_2025 = {
    '1.5°C': {'50%': 247000, '67%': 60000},
    '2°C': {'50%': 1219000, '67%': 944000}
}


def load_data():
    """Load preprocessed data files."""
    print("Loading preprocessed data...")
    combined_df = pd.read_csv(os.path.join(DATA_DIR, 'combined_data.csv'))
    
    # Ensure "NA" is treated as a valid ISO2 code (for Namibia)
    combined_df['ISO2'] = combined_df['ISO2'].astype(str)
    combined_df.loc[combined_df['Country'] == 'Namibia', 'ISO2'] = 'NA'
    
    print(f"Loaded combined_data.csv: {combined_df.shape}")
    return combined_df


def get_country_data_for_year(combined_df, iso2, emissions_scope, target_year):
    """
    Get country emissions and cumulative data for a specific year.
    If target_year data is not available, use the closest available year.
    """
    country_data = combined_df[
        (combined_df['ISO2'] == iso2) &
        (combined_df['Emissions_scope'] == emissions_scope) &
        (combined_df['Year'] <= target_year) &
        (combined_df['Year'] != 2050)  # Exclude 2050 which has NaN values
    ].sort_values('Year')
    
    if len(country_data) == 0:
        return None
    
    # Get the latest available year <= target_year
    latest_data = country_data.iloc[-1]
    
    return {
        'year': int(latest_data['Year']),
        'annual_emissions': latest_data['Annual_CO2_emissions_Mt'],
        'cumulative_emissions': latest_data['Cumulative_CO2_emissions_Mt'],
        'cumulative_population': latest_data['Cumulative_population'],
        'population': latest_data['Population'],
        'gdp_ppp': latest_data.get('GDP_PPP', 0),
        'share_of_capacity': latest_data.get('share_of_capacity', 0)
    }


def calculate_budget_shares(combined_df, starting_year, emissions_scope):
    """
    Calculate population shares and capacity shares for budget allocation.
    Uses data from 1970 to starting_year for responsibility scenarios.
    """
    print(f"\nCalculating shares for {emissions_scope} scope, starting year {starting_year}...")
    
    # Get all countries with data for this scope
    scope_data = combined_df[
        (combined_df['Emissions_scope'] == emissions_scope) &
        (combined_df['Year'] >= 1970) &
        (combined_df['Year'] <= starting_year) &
        (combined_df['Year'] != 2050)
    ]
    
    countries = scope_data['ISO2'].unique()
    shares = {}
    
    for iso2 in countries:
        # Skip aggregates for share calculation
        if iso2 in ['WLD', 'EU', 'G20']:
            continue
        
        country_data = scope_data[scope_data['ISO2'] == iso2]
        
        if len(country_data) == 0:
            continue
        
        # Get data at starting_year (or closest available)
        year_data = country_data[country_data['Year'] <= starting_year].sort_values('Year').iloc[-1]
        
        shares[iso2] = {
            'cumulative_population': year_data['Cumulative_population'],
            'cumulative_emissions': year_data['Cumulative_CO2_emissions_Mt'],
            'latest_population': year_data['Population'],
            'share_of_capacity': year_data.get('share_of_capacity', 0)
        }
    
    # Calculate total cumulative population (1970 to starting_year)
    total_cum_pop = sum(s['cumulative_population'] for s in shares.values())
    
    # Calculate total cumulative emissions
    total_cum_emissions = sum(s['cumulative_emissions'] for s in shares.values())
    
    # Calculate population shares
    for iso2 in shares:
        shares[iso2]['population_share'] = shares[iso2]['cumulative_population'] / total_cum_pop if total_cum_pop > 0 else 0
        shares[iso2]['emissions_share'] = shares[iso2]['cumulative_emissions'] / total_cum_emissions if total_cum_emissions > 0 else 0
    
    # Calculate future population shares (starting_year to 2050)
    future_pop_data = combined_df[
        (combined_df['Emissions_scope'] == emissions_scope) &
        (combined_df['Year'] >= starting_year) &
        (combined_df['Year'] <= 2050)
    ]
    
    future_pop_totals = {}
    for iso2 in countries:
        if iso2 in ['WLD', 'EU', 'G20']:
            continue
        country_future = future_pop_data[future_pop_data['ISO2'] == iso2]
        if len(country_future) > 0:
            future_pop_totals[iso2] = country_future['Population'].sum()
    
    total_future_pop = sum(future_pop_totals.values())
    
    for iso2 in shares:
        if iso2 in future_pop_totals:
            shares[iso2]['future_population_share'] = future_pop_totals[iso2] / total_future_pop if total_future_pop > 0 else 0
        else:
            shares[iso2]['future_population_share'] = 0
    
    print(f"Calculated shares for {len(shares)} countries")
    return shares, total_cum_emissions


def calculate_neutrality_year(starting_year_data, country_budget):
    """
    Calculate the neutrality year based on starting year data and allocated budget.
    Uses linear decrease assumption: years_to_neutrality = 2 * budget / annual_emissions
    """
    annual_emissions = starting_year_data['annual_emissions']
    starting_year = starting_year_data['year']
    
    if pd.isna(annual_emissions) or annual_emissions <= 0:
        return None, None
    
    if country_budget < 0:
        # For negative budgets, find when they overshot historically
        # For simplicity in this analysis, mark as "overshot" (e.g., 1970)
        return 1970, 1970 - starting_year
    
    # Linear decrease assumption
    years_to_neutrality = int(round(2 * country_budget / annual_emissions))
    neutrality_year = starting_year + years_to_neutrality
    
    # Cap at 1970 (earliest) and 2100 (latest)
    if neutrality_year < 1970:
        neutrality_year = 1970
    elif neutrality_year > 2100:
        neutrality_year = 2100
    
    return neutrality_year, years_to_neutrality


def run_sensitivity_scenario(combined_df, starting_year, budget_config, distribution_method, 
                             emissions_scope, warming_scenario, probability):
    """
    Run a single sensitivity scenario and calculate neutrality years for all countries.
    
    Parameters:
    - combined_df: Full combined data
    - starting_year: Year to start the budget calculation from
    - budget_config: Dictionary with budget values
    - distribution_method: 'Responsibility' or 'Capability'
    - emissions_scope: 'Territory' or 'Consumption'
    - warming_scenario: '1.5°C' or '2°C'
    - probability: '50%' or '67%'
    """
    # Get global budget
    global_budget = budget_config[warming_scenario][probability]
    
    # Get budget shares
    shares, total_cum_emissions = calculate_budget_shares(combined_df, starting_year, emissions_scope)
    
    results = []
    
    # Total available budget (global + historical emissions)
    total_available = global_budget + total_cum_emissions
    
    # Calculate theoretical budgets for each country
    theoretical_budgets = {}
    positive_budgets_sum = 0
    
    for iso2, share_data in shares.items():
        starting_year_data = get_country_data_for_year(combined_df, iso2, emissions_scope, starting_year)
        
        if starting_year_data is None:
            continue
        
        # Calculate theoretical budget based on distribution method
        if distribution_method == 'Responsibility':
            population_share = share_data['population_share']
            country_cumulative = starting_year_data['cumulative_emissions']
            theoretical_budget = (total_available * population_share) - country_cumulative
        elif distribution_method == 'Capability':
            capacity_share = share_data['share_of_capacity']
            if capacity_share == 0 or pd.isna(capacity_share):
                continue  # Skip countries with no capacity data
            country_cumulative = starting_year_data['cumulative_emissions']
            theoretical_budget = (total_available * capacity_share) - country_cumulative
        else:
            continue
        
        theoretical_budgets[iso2] = {
            'theoretical_budget': theoretical_budget,
            'starting_year_data': starting_year_data,
            'share_data': share_data
        }
        
        # Sum positive budgets for normalization
        if theoretical_budget > 0:
            positive_budgets_sum += theoretical_budget
    
    # Normalize positive budgets to match global budget
    normalization_factor = global_budget / positive_budgets_sum if positive_budgets_sum > 0 else 1.0
    
    print(f"  Normalization factor: {normalization_factor:.6f}")
    print(f"  Positive budgets sum before normalization: {positive_budgets_sum:,.0f} MtCO2")
    print(f"  Global budget: {global_budget:,.0f} MtCO2")
    
    # Calculate final budgets and neutrality years
    for iso2, budget_info in theoretical_budgets.items():
        theoretical_budget = budget_info['theoretical_budget']
        starting_year_data = budget_info['starting_year_data']
        share_data = budget_info['share_data']
        
        # Normalize positive budgets
        if theoretical_budget > 0:
            final_budget = theoretical_budget * normalization_factor
        else:
            final_budget = theoretical_budget  # Keep negative budgets as-is
        
        # Calculate neutrality year
        neutrality_year, years_to_neutrality = calculate_neutrality_year(starting_year_data, final_budget)
        
        # Get country name
        country_name = combined_df[combined_df['ISO2'] == iso2]['Country'].iloc[0] if len(
            combined_df[combined_df['ISO2'] == iso2]) > 0 else iso2
        
        results.append({
            'ISO2': iso2,
            'Country': country_name,
            'Starting_year': starting_year,
            'Emissions_scope': emissions_scope,
            'Warming_scenario': warming_scenario,
            'Probability': probability,
            'Distribution_method': distribution_method,
            'Global_budget_MtCO2': global_budget,
            'Country_budget_MtCO2': final_budget,
            'Theoretical_budget_MtCO2': theoretical_budget,
            'Starting_year_annual_emissions_MtCO2': starting_year_data['annual_emissions'],
            'Starting_year_cumulative_emissions_MtCO2': starting_year_data['cumulative_emissions'],
            'Neutrality_year': neutrality_year,
            'Years_to_neutrality': years_to_neutrality,
            'Population_share': share_data['population_share'] if distribution_method == 'Responsibility' else None,
            'Capacity_share': share_data['share_of_capacity'] if distribution_method == 'Capability' else None
        })
    
    return pd.DataFrame(results)


def main():
    """Main execution function."""
    print("="*80)
    print("CARBON BUDGET SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Load data
    combined_df = load_data()
    
    all_results = []
    
    # Run base case (2025) for comparison
    print("\n" + "="*80)
    print("RUNNING BASE CASE (2025 Starting Year)")
    print("="*80)
    
    for emissions_scope in ['Territory', 'Consumption']:
        for distribution in ['Responsibility', 'Capability']:
            for warming in ['1.5°C', '2°C']:
                for prob in ['50%', '67%']:
                    print(f"\nProcessing: {emissions_scope} | {distribution} | {warming} | {prob}")
                    results = run_sensitivity_scenario(
                        combined_df, 2025, BASE_BUDGETS_2025,
                        distribution, emissions_scope, warming, prob
                    )
                    results['Scenario_name'] = 'Base_2025'
                    all_results.append(results)
    
    # Run alternative scenarios
    for starting_year, budget_config in ALTERNATIVE_BUDGETS.items():
        print("\n" + "="*80)
        print(f"RUNNING ALTERNATIVE SCENARIOS (Starting Year: {starting_year})")
        print("="*80)
        
        for emissions_scope in ['Territory', 'Consumption']:
            for distribution in ['Responsibility', 'Capability']:
                for warming in ['1.5°C', '2°C']:
                    for prob in ['50%', '67%']:
                        print(f"\nProcessing: {emissions_scope} | {distribution} | {warming} | {prob}")
                        results = run_sensitivity_scenario(
                            combined_df, starting_year, budget_config,
                            distribution, emissions_scope, warming, prob
                        )
                        results['Scenario_name'] = f'Alternative_{starting_year}'
                        all_results.append(results)
    
    # Combine all results
    print("\n" + "="*80)
    print("COMBINING RESULTS")
    print("="*80)
    
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    # Save full results
    output_file = os.path.join(OUTPUT_DIR, 'sensitivity_analysis_full_results.csv')
    all_results_df.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")
    print(f"Total rows: {len(all_results_df)}")
    
    # Create comparison summary: Base vs Alternatives
    print("\n" + "="*80)
    print("CREATING COMPARISON SUMMARIES")
    print("="*80)
    
    # Pivot to compare neutrality years across scenarios
    comparison_pivot = all_results_df.pivot_table(
        index=['ISO2', 'Country', 'Emissions_scope', 'Distribution_method', 'Warming_scenario', 'Probability'],
        columns='Scenario_name',
        values='Neutrality_year',
        aggfunc='first'
    ).reset_index()
    
    # Calculate differences from base case
    if 'Base_2025' in comparison_pivot.columns:
        for col in comparison_pivot.columns:
            if col.startswith('Alternative_'):
                comparison_pivot[f'Diff_{col}'] = comparison_pivot[col] - comparison_pivot['Base_2025']
    
    comparison_file = os.path.join(OUTPUT_DIR, 'sensitivity_neutrality_year_comparison.csv')
    comparison_pivot.to_csv(comparison_file, index=False)
    print(f"Neutrality year comparison saved to: {comparison_file}")
    
    # Create summary statistics
    summary_stats = []
    
    for scenario in all_results_df['Scenario_name'].unique():
        scenario_data = all_results_df[all_results_df['Scenario_name'] == scenario]
        
        for emissions_scope in scenario_data['Emissions_scope'].unique():
            for distribution in scenario_data['Distribution_method'].unique():
                for warming in scenario_data['Warming_scenario'].unique():
                    for prob in scenario_data['Probability'].unique():
                        subset = scenario_data[
                            (scenario_data['Emissions_scope'] == emissions_scope) &
                            (scenario_data['Distribution_method'] == distribution) &
                            (scenario_data['Warming_scenario'] == warming) &
                            (scenario_data['Probability'] == prob)
                        ]
                        
                        if len(subset) > 0:
                            # Filter out None values for statistics
                            neutrality_years = subset['Neutrality_year'].dropna()
                            
                            summary_stats.append({
                                'Scenario_name': scenario,
                                'Emissions_scope': emissions_scope,
                                'Distribution_method': distribution,
                                'Warming_scenario': warming,
                                'Probability': prob,
                                'Num_countries': len(subset),
                                'Mean_neutrality_year': neutrality_years.mean() if len(neutrality_years) > 0 else None,
                                'Median_neutrality_year': neutrality_years.median() if len(neutrality_years) > 0 else None,
                                'Min_neutrality_year': neutrality_years.min() if len(neutrality_years) > 0 else None,
                                'Max_neutrality_year': neutrality_years.max() if len(neutrality_years) > 0 else None,
                                'Countries_before_2030': len(neutrality_years[neutrality_years < 2030]),
                                'Countries_2030_2050': len(neutrality_years[(neutrality_years >= 2030) & (neutrality_years <= 2050)]),
                                'Countries_after_2050': len(neutrality_years[neutrality_years > 2050])
                            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_file = os.path.join(OUTPUT_DIR, 'sensitivity_summary_statistics.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")
    
    # Print sample results
    print("\n" + "="*80)
    print("SAMPLE RESULTS (Base 2025 vs Alternative 2018, Territory, Responsibility, 1.5°C, 50%)")
    print("="*80)
    
    sample_filter = (
        (all_results_df['Emissions_scope'] == 'Territory') &
        (all_results_df['Distribution_method'] == 'Responsibility') &
        (all_results_df['Warming_scenario'] == '1.5°C') &
        (all_results_df['Probability'] == '50%')
    )
    
    sample_countries = ['US', 'CN', 'DE', 'FR', 'IN', 'BR', 'NA']  # USA, China, Germany, France, India, Brazil, Namibia
    
    for country in sample_countries:
        country_data = all_results_df[
            sample_filter &
            (all_results_df['ISO2'] == country)
        ]
        
        if len(country_data) > 0:
            print(f"\n{country_data.iloc[0]['Country']} ({country}):")
            for _, row in country_data.iterrows():
                print(f"  {row['Scenario_name']}: Neutrality year = {row['Neutrality_year']}, "
                      f"Budget = {row['Country_budget_MtCO2']:,.0f} MtCO2")
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput files saved in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
