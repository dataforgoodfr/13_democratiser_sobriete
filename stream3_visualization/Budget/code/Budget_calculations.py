# Define global carbon budgets (in MtCO2)
BUDGET_GLOBAL_lamboll_2C = {"33%": 1603000, "50%": 1219000, "67%": 944000}
BUDGET_GLOBAL_foster_2C = {"33%": 1450000, "50%": 1150000, "67%": 950000}
BUDGET_GLOBAL_lamboll_15C = {"33%": 480000, "50%": 247000, "67%": 60000}
BUDGET_GLOBAL_foster_15C = {"33%": 300000, "50%": 250000, "67%": 150000}

# Merge population data with emissions
df = latest_emissions_df

# Create an empty list to store reformatted rows
rows = []

# Iterate over all budget scenarios
for temp, budgets in [("2°C", [("lamboll", BUDGET_GLOBAL_lamboll_2C), ("foster", BUDGET_GLOBAL_foster_2C)]),
                      ("1.5°C", [("lamboll", BUDGET_GLOBAL_lamboll_15C), ("foster", BUDGET_GLOBAL_foster_15C)])]:
    for source, budget in budgets:
        for prob, BUDGET_GLOBAL in budget.items():
            POP_GLOBAL_2050 = df['population_2050'].sum()

            for method in ["equality", "responsibility"]:
                if method == "equality":
                    remaining_carbon_budget = (BUDGET_TOTAL / POP_GLOBAL_2050) * df["population_2050"]
                else:
                    total_emissions = df.groupby('scope')['CO2_emissions_latest'].sum().to_dict()
                    remaining_carbon_budget = (
                        (total_emissions[df['scope'].iloc[0]] + BUDGET_TOTAL) * (df["population_2050"] / POP_GLOBAL_2050)
                        - df["CO2_emissions_latest"]
                    )

                time_to_neutrality_lin = 2 * remaining_carbon_budget / df["CO2_emissions_latest"]
                time_to_neutrality_exp = -np.log(1 - 0.95) * remaining_carbon_budget / df["CO2_emissions_latest"]

                for curve, time_to_neutrality in [("linear", time_to_neutrality_lin), ("exponential", time_to_neutrality_exp)]:
                    for _, row in df.iterrows():
                        remaining_budget = remaining_carbon_budget.loc[row.name]
                        time_to_neutrality_value = time_to_neutrality.loc[row.name]

                        # Check if remaining_carbon_budget is negative and adjust time_to_neutrality
                        if remaining_budget < 0:
                            # Start summing emissions backwards from the latest year
                            cumulative_emissions = 0
                            country = row["country"]
                            latest_year = row["latest_year"]