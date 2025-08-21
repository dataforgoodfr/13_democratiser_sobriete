#!/usr/bin/env python3
"""
Sanity check: Verify that sum of individual country theoretical positive budgets 
equals the Global_Carbon_budget for each scenario combination.
"""

import pandas as pd
import numpy as np

def main():
    # Load the scenario parameters
    df = pd.read_csv('../Output/scenario_parameters.csv')
    
    print("=== SANITY CHECK: Theoretical Budgets vs Global Carbon Budget ===\n")
    
    # Group by scenario combination (excluding aggregates)
    scenario_groups = df[df['Country'] != 'All'].groupby([
        'Emissions_scope', 'Warming_scenario', 'Probability_of_reach', 'Budget_distribution_scenario'
    ])
    
    all_checks_passed = True
    
    for (scope, warming, prob, budget_scenario), group in scenario_groups:
        print(f"Scenario: {scope} | {warming} | {prob}% | {budget_scenario}")
        
        # Get the global carbon budget for this scenario
        global_budget = group['Global_Carbon_budget'].iloc[0]
        
        # Calculate sum of ALL theoretical budgets (positive + negative, excluding aggregates)
        theoretical_total_sum = group[
            (group['Country'] != 'All')
        ]['Country_theoretical_budget'].sum()
        
        # Check if they match
        match = abs(theoretical_total_sum - global_budget) < 0.01  # Allow small floating point differences
        
        if match:
            print(f"  ✅ MATCH: Sum of ALL theoretical budgets = {theoretical_total_sum:,.0f} MtCO2")
            print(f"      Global Carbon Budget = {global_budget:,.0f} MtCO2")
        else:
            print(f"  ❌ MISMATCH: Sum of ALL theoretical budgets = {theoretical_total_sum:,.0f} MtCO2")
            print(f"      Global Carbon Budget = {global_budget:,.0f} MtCO2")
            print(f"      Difference = {theoretical_total_sum - global_budget:,.0f} MtCO2")
            all_checks_passed = False
        
        print()
    
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED! Theoretical budgets sum correctly to global budgets.")
    else:
        print("⚠️  SOME CHECKS FAILED! There are mismatches in theoretical budget sums.")
    
    return all_checks_passed

if __name__ == "__main__":
    main() 