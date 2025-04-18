import pandas as pd
import numpy as np

# Define the directory containing the data files
output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'

# Load the preprocessed data
latest_emissions_df = pd.read_csv(f"{output_directory}/Latest_emissions.csv")

# Define global carbon budgets (in GtCO2, converted to MtCO2)
BUDGET_GLOBAL_lamboll_2C = {"33%": 1603, "50%": 1219, "67%": 944}
BUDGET_GLOBAL_foster_2C = {"33%": 1450, "50%": 1150, "67%": 950}
BUDGET_GLOBAL_lamboll_15C = {"33%": 480, "50%": 247, "67%": 60}
BUDGET_GLOBAL_foster_15C = {"33%": 300, "50%": 250, "67%": 150}

# Merge population data with emissions
df = latest_emissions_df

# Create an empty list to store reformatted rows
rows = []

# Iterate over all budget scenarios
for temp, budgets in [("2°C", [("lamboll", BUDGET_GLOBAL_lamboll_2C), ("foster", BUDGET_GLOBAL_foster_2C)]),
                      ("1.5°C", [("lamboll", BUDGET_GLOBAL_lamboll_15C), ("foster", BUDGET_GLOBAL_foster_15C)])]:
    for source, budget in budgets:
        for prob, BUDGET_GLOBAL in budget.items():
            BUDGET_TOTAL = BUDGET_GLOBAL * 1000  # Convert to MtCO2
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

                            # Iterate through years from the latest year to 1990
                            for year in range(latest_year, 1989, -1):
                                # Access the emissions for the country and year from the historical emissions DataFrame
                                emissions_year = historical_emissions_df[
                                    (historical_emissions_df["country"] == country) &
                                    (historical_emissions_df["year"] == year) &
                                    (historical_emissions_df["scope"] == row["scope"])
                                ]["CO2_emissions"].values[0]
                                cumulative_emissions += emissions_year

                                # If cumulative emissions exceed the negative remaining budget, break
                                if cumulative_emissions >= abs(remaining_budget):
                                    time_to_neutrality_value = - (latest_year - year - 1)
                                    break

                        # Append the results to the rows list
                        rows.append({
                            "country": row["country"],
                            "CO2_emissions_latest": row["CO2_emissions_latest"],
                            "latest_year": row["latest_year"],
                            "population_2050": row["population_2050"],
                            "repartition_method": method,
                            "source_of_budget": source,
                            "probability_of_reach": prob,
                            "curve_type": curve,
                            "remaining_carbon_budget": remaining_budget,
                            "time_to_neutrality": time_to_neutrality_value,
                            "temperature": temp,
                            "scope": row["scope"]
                        })

# Convert list to DataFrame
df_final = pd.DataFrame(rows)

# Save the final cleaned dataset
df_final.to_csv(f"{output_directory}/carbon_budget_scenarios.csv", index=False)
