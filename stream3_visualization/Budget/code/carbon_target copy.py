#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

# Define the directory containing the data files
data_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Data'
output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'

# Load the datasets
emissions_file = f"{data_directory}/EDGAR- Total fossil_CO2.csv"
population_file = f"{data_directory}/UN population projections.csv"
consumption_file = f"{data_directory}/EORA_consumption_CO2_1990-2022.csv"

# Read the datasets
emissions = pd.read_csv(emissions_file, sep=';', encoding='latin1')
population = pd.read_csv(population_file, sep=';', encoding='latin1')
consumption = pd.read_csv(consumption_file, sep=';', encoding='latin1')

# Convert emission columns to numeric
emission_cols = [str(year) for year in range(1990, 2024)]
for col in emission_cols:
    emissions[col] = pd.to_numeric(emissions[col].astype(str).str.replace(',', '.'), errors='coerce')

# Convert population['2050'] to numeric safely
population['2050'] = pd.to_numeric(population['2050'].astype(str).str.replace(',', '.'), errors='coerce')

# Standardize column names
pop_col = "Region, subregion, country or area *"
emis_col = "Country"

# Create a mapping dictionary for manual correction based on the base file
correction_map_base = {
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Brunei Darussalam': 'Brunei',
    'China, Hong Kong SAR': 'Hong Kong',
    'China, Macao SAR': 'Macao',
    'China, Taiwan Province of China': 'Taiwan',
    "Côte d'Ivoire": 'Côte d\x92Ivoire',
    "Dem. People's Republic of Korea": 'North Korea',
    'Falkland Islands (Malvinas)': 'Falkland Islands',
    'Faroe Islands': 'Faroes',
    'France': 'France and Monaco',
    'Gambia': 'The Gambia',
    'Guam': 'Guam',
    'Guernsey': 'Guernsey',
    'Holy See': 'Italy, San Marino and the Holy See',
    'Iran (Islamic Republic of)': 'Iran',
    'Isle of Man': 'Isle of Man',
    'Israel': 'Israel and Palestine, State of',
    'Italy': 'Italy, San Marino and the Holy See',
    'Jersey': 'Jersey',
    'Kosovo (under UNSC res. 1244)': 'Kosovo',
    "Lao People's Democratic Republic": 'Laos',
    'Liechtenstein': 'Switzerland and Liechtenstein',
    'Marshall Islands': 'Marshall Islands',
    'Mayotte': 'Mayotte',
    'Micronesia (Fed. States of)': 'Micronesia',
    'Monaco': 'France and Monaco',
    'Montenegro': 'Serbia and Montenegro',
    'Montserrat': 'Montserrat',
    'Myanmar': 'Myanmar/Burma',
    'Nauru': 'Nauru',
    'Niue': 'Niue',
    'Northern Mariana Islands': 'Northern Mariana Islands',
    'Republic of Korea': 'South Korea',
    'Republic of Moldova': 'Moldova',
    'Russian Federation': 'Russia',
    'Saint Barthélemy': 'Saint Barthélemy',
    'Saint Helena': 'Saint Helena, Ascension and Tristan da Cunha',
    'Saint Martin (French part)': 'Saint Martin',
    'San Marino': 'Italy, San Marino and the Holy See',
    'Sao Tome and Principe': 'São Tomé and Príncipe',
    'Serbia': 'Serbia and Montenegro',
    'Sint Maarten (Dutch part)': 'Sint Maarten',
    'South Sudan': 'Sudan and South Sudan',
    'Spain': 'Spain and Andorra',
    'State of Palestine': 'Israel and Palestine, State of',
    'Sudan': 'Sudan and South Sudan',
    'Switzerland': 'Switzerland and Liechtenstein',
    'Syrian Arab Republic': 'Syria',
    'Tokelau': 'Tokelau',
    'Tuvalu': 'Tuvalu',
    'United Republic of Tanzania': 'Tanzania',
    'United States Virgin Islands': 'United States Virgin Islands',
    'United States of America': 'United States',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Wallis and Futuna Islands': 'Wallis and Futuna Islands'
}

# Apply corrections to population DataFrame
population["corrected_country"] = population[pop_col].replace(correction_map_base)

# Handle multiple rows mapping to the same country by summing population
population_final = population.groupby("corrected_country", as_index=False)["2050"].sum()

# Save the final cleaned dataset
population_final.to_csv(f"{output_directory}/pop_modified.csv", index=False)

# ## RCB
# ### 2°C

emissions["Total_Emissions_1990_2023"] = emissions.loc[:, "1990":"2023"].sum(axis=1)

# Define global carbon budgets (in GtCO2, converted to MtCO2)
BUDGET_GLOBAL_lamboll = {"33%": 1603, "50%": 1219, "67%": 944}
BUDGET_GLOBAL_foster = {"33%": 1450, "50%": 1150, "67%": 950}

# Merge population data with emissions
df = population_final.merge(emissions[["Country", "2023", "Total_Emissions_1990_2023"]],
                            left_on="corrected_country", right_on="Country")

# Create an empty list to store reformatted rows
rows = []

# List of year columns from the emissions dataframe (adjust based on your data)
year_columns = [str(year) for year in range(1970, 2024)]  # Assuming years are from 1970 to 2023

# Iterate over all budget scenarios
for source, budgets in [("lamboll", BUDGET_GLOBAL_lamboll), ("foster", BUDGET_GLOBAL_foster)]:
    for prob, BUDGET_GLOBAL in budgets.items():

        BUDGET_TOTAL = BUDGET_GLOBAL * 1000  # Convert to MtCO2

        for method in ["equality", "responsibility"]:
            if method == "equality":
                remaining_carbon_budget = (BUDGET_TOTAL / POP_GLOBAL_2050) * df["2050"]
            else:
                remaining_carbon_budget = (
                    (df["Total_Emissions_1990_2023"].sum() + BUDGET_TOTAL) * (df["2050"] / POP_GLOBAL_2050)
                    - df["Total_Emissions_1990_2023"]
                )

            time_to_neutrality_lin = 2 * remaining_carbon_budget / df["2023"]
            time_to_neutrality_exp = -np.log(1 - 0.95) * remaining_carbon_budget / df["2023"]

            for curve, time_to_neutrality in [("linear", time_to_neutrality_lin), ("exponential", time_to_neutrality_exp)]:

                for _, row in df.iterrows():
                    remaining_budget = remaining_carbon_budget.loc[row.name]
                    time_to_neutrality_value = time_to_neutrality.loc[row.name]

                    # Check if remaining_carbon_budget is negative and adjust time_to_neutrality
                    if remaining_budget < 0:
                        # Start summing emissions backwards from 2023
                        cumulative_emissions = 0
                        country = row["Country"]

                        # Iterate through years from 2023 to 1970
                        for year in range(2023, 1969, -1):
                            # Access the emissions for the country and year from the emissions DataFrame
                            emissions_year = emissions.loc[emissions["Country"] == country, str(year)].values[0]
                            cumulative_emissions += emissions_year

                            # If cumulative emissions exceed the negative remaining budget, break
                            if cumulative_emissions >= abs(remaining_budget):
                                time_to_neutrality_value = - (2023 - year-1)
                                break

                    # Append the results to the rows list
                    rows.append({
                        "country": row["Country"],
                        "emission_2023": row["2023"],
                        "population_2050": row["2050"],
                        "repartition_method": method,
                        "source_of_budget": source,
                        "probability_of_reach": prob,
                        "curve_type": curve,
                        "remaining_carbon_budget": remaining_budget,  # Store the computed value
                        "time_to_neutrality": time_to_neutrality_value  # Store the computed value
                    })

# Convert list to DataFrame
df_final = pd.DataFrame(rows)

# Save the final cleaned dataset
df_final.to_excel(f"{output_directory}/carbon_budget_2°C.xlsx", index=False)

# ### 1.5°C

# Define the global carbon budgets (in GtCO2, converted to MtCO2)
BUDGET_GLOBAL_lamboll = {"33%": 480, "50%": 247, "67%": 60}
BUDGET_GLOBAL_foster = {"33%": 300, "50%": 250, "67%": 150}

# Merge population data with emissions
df = population_final.merge(emissions[["Country", "2023", "Total_Emissions_1990_2023"]],
                            left_on="corrected_country", right_on="Country")

# Create an empty list to store reformatted rows
rows = []

# List of year columns from the emissions dataframe (adjust based on your data)
year_columns = [str(year) for year in range(1970, 2024)]  # Assuming years are from 1970 to 2023

# Iterate over all budget scenarios
for source, budgets in [("lamboll", BUDGET_GLOBAL_lamboll), ("foster", BUDGET_GLOBAL_foster)]:
    for prob, BUDGET_GLOBAL in budgets.items():

        BUDGET_TOTAL = BUDGET_GLOBAL * 1000  # Convert to MtCO2

        for method in ["equality", "responsibility"]:
            if method == "equality":
                remaining_carbon_budget = (BUDGET_TOTAL / POP_GLOBAL_2050) * df["2050"]
            else:
                remaining_carbon_budget = (
                    (df["Total_Emissions_1990_2023"].sum() + BUDGET_TOTAL) * (df["2050"] / POP_GLOBAL_2050)
                    - df["Total_Emissions_1990_2023"]
                )

            time_to_neutrality_lin = 2 * remaining_carbon_budget / df["2023"]
            time_to_neutrality_exp = -np.log(1 - 0.95) * remaining_carbon_budget / df["2023"]

            for curve, time_to_neutrality in [("linear", time_to_neutrality_lin), ("exponential", time_to_neutrality_exp)]:

                for _, row in df.iterrows():
                    remaining_budget = remaining_carbon_budget.loc[row.name]
                    time_to_neutrality_value = time_to_neutrality.loc[row.name]

                    # Check if remaining_carbon_budget is negative and adjust time_to_neutrality
                    if remaining_budget < 0:
                        # Start summing emissions backwards from 2023
                        cumulative_emissions = 0
                        country = row["Country"]

                        # Iterate through years from 2023 to 1970
                        for year in range(2023, 1969, -1):
                            # Access the emissions for the country and year from the emissions DataFrame
                            emissions_year = emissions.loc[emissions["Country"] == country, str(year)].values[0]
                            cumulative_emissions += emissions_year

                            # If cumulative emissions exceed the negative remaining budget, break
                            if cumulative_emissions >= abs(remaining_budget):
                                time_to_neutrality_value = - (2023 - year-1)
                                break

                    # Append the results to the rows list
                    rows.append({
                        "country": row["Country"],
                        "emission_2023": row["2023"],
                        "population_2050": row["2050"],
                        "repartition_method": method,
                        "source_of_budget": source,
                        "probability_of_reach": prob,
                        "curve_type": curve,
                        "remaining_carbon_budget": remaining_budget,  # Store the computed value
                        "time_to_neutrality": time_to_neutrality_value  # Store the computed value
                    })

# Convert list to DataFrame
df_final = pd.DataFrame(rows)

# Save the final cleaned dataset
df_final.to_excel(f"{output_directory}/carbon_budget_1.5°C.xlsx", index=False)

# ### EU

EU_list = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
           "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
           "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
           "Spain", "Sweden"]

EU_list_corrected = ["Italy, San Marino and the Holy See", "Czechia", "France and Monaco", "Spain and Andorra"] + EU_list

df_EU = df_final[df_final["country"].isin(EU_list_corrected)]
df_summed = df_EU.groupby(['repartition_method', 'source_of_budget', 'probability_of_reach', 'curve_type'], as_index=False).sum(numeric_only=True)

# Compute time_to_neutrality
for idx, row in df_summed.iterrows():
    remaining_budget = row["remaining_carbon_budget"]
    emission_2023 = row["emission_2023"]  # Total 2023 emissions for the EU

    # Compute time_to_neutrality normally if budget is positive
    if remaining_budget >= 0:
        df_summed.at[idx, "time_to_neutrality"] = 2 * remaining_budget / emission_2023  # Linear model
    else:
        # Handle negative remaining budget by summing past emissions until budget is neutralized
        cumulative_emissions = 0
        time_to_neutrality_value = np.nan  # Default in case no year meets the condition

        # Get total EU emissions per year from the emissions dataframe
        EU_emissions = emissions[emissions["Country"].isin(EU_list_corrected)].set_index("Country").loc[:, "1970":"2023"].sum()

        for year in range(2023, 1969, -1):  # Iterate backwards from 2023
            cumulative_emissions += EU_emissions[str(year)]

            if cumulative_emissions >= abs(remaining_budget):
                time_to_neutrality_value = year - 2023 + 1
                break

        df_summed.at[idx, "time_to_neutrality"] = time_to_neutrality_value

# Save the final cleaned dataset
df_summed.to_excel(f"{output_directory}/carbon_budget_2°C_EU.xlsx", index=False)
df_summed.to_excel(f"{output_directory}/carbon_budget_1.5°C_EU.xlsx", index=False)

# # Consumption-based (CBA) - 2022

# EDGAR- Total fossil_CO2.csv
emissions = pd.read_csv(consumption_file, sep=';', encoding='latin1')
# UN population projections.csv
population = pd.read_csv(population_file, sep=';', encoding='latin1')

# Convert emission columns to numeric and divide by 1,000
emission_cols = [str(year) for year in range(1990, 2023)]
for col in emission_cols:
    emissions[col] = pd.to_numeric(emissions[col].astype(str).str.replace(',', '.'), errors='coerce') / 1000

# Convert population['2050'] to numeric safely
population['2050'] = pd.to_numeric(population['2050'].astype(str).str.replace(',', '.'), errors='coerce')

# Step 2: Sum the numeric values
POP_GLOBAL_2050 = population['2050'].sum()

emissions = emissions.rename(columns={"ï»¿Country": "Country"})

# Standardize column names
pop_col = "Region, subregion, country or area *"
emis_col = "Country"

# Get unique country names
pop_countries = set(population[pop_col].unique())
emis_countries = set(emissions[emis_col].unique())

# Find mismatches
unmatched_emis = emis_countries - pop_countries
unmatched_pop = pop_countries - emis_countries

# Fuzzy matching function (handling NaN and converting to string)
def get_best_match(country, choices, threshold=80):
    if pd.isna(country):  # Ignore NaN values
        return None
    country = str(country)  # Ensure it's a string
    match, score = process.extractOne(country, [str(choice) for choice in choices])  # Convert choices to strings
    return match if score >= threshold else None

# Find closest matches
emis_matches = {country: get_best_match(country, pop_countries) for country in unmatched_emis}
pop_matches = {country: get_best_match(country, emis_countries) for country in unmatched_pop}

# Create a DataFrame for manual correction
emis_unmatched_df = pd.DataFrame(list(emis_matches.items()), columns=["Original", "Suggested"])
emis_unmatched_df["Manual_Correction"] = ""  # Column for manual edits

pop_unmatched_df = pd.DataFrame(list(pop_matches.items()), columns=["Original", "Suggested"])
pop_unmatched_df["Manual_Correction"] = ""  # Column for manual edits

# Save the unmatched DataFrames for manual correction
emis_unmatched_df.to_csv(f"{output_directory}/emis_unmatched_df.csv", index=False)
pop_unmatched_df.to_csv(f"{output_directory}/pop_unmatched_df.csv", index=False)

# ## Preparing data

# Extract relevant columns
pop_col = "Region, subregion, country or area *"
population = population[[pop_col, "2050"]]

# Drop NaN values in Manual_Correction
pop_matched = pop_matched.dropna(subset=["Manual_Correction"], how='all')

# Create a mapping dictionary
correction_map = {}
for _, row in pop_matched.iterrows():
    if row["Manual_Correction"] == "OK":
        correction_map[row["Original"]] = row["Suggested"]
    else:
        correction_map[row["Original"]] = row["Manual_Correction"]

# Apply corrections to population DataFrame
population["corrected_country"] = population[pop_col].replace(correction_map)

# Sum emissions from 1990 to 2022
df_pop = pd.read_csv(population_file, sep=';', encoding='latin1')

# Handle multiple rows mapping to the same country by summing population
population_final = population.groupby("corrected_country", as_index=False)["2050"].sum()

# Save the final cleaned dataset
population_final.to_csv(f"{output_directory}/pop_modified_eora.csv", index=False)

# ## RCB
# ### 2°C

emissions["Total_Emissions_1990_2022"] = emissions.loc[:, "1990":"2022"].sum(axis=1)

# Define global carbon budgets (in GtCO2, converted to MtCO2)
BUDGET_GLOBAL_lamboll = {"33%": 1603, "50%": 1219, "67%": 944}
BUDGET_GLOBAL_foster = {"33%": 1450, "50%": 1150, "67%": 950}

# Merge population data with emissions
df = population_final.merge(emissions[["Country", "2022", "Total_Emissions_1990_2022"]],
                            left_on="corrected_country", right_on="Country")

# Create an empty list to store reformatted rows
rows = []

# List of year columns from the emissions dataframe (adjust based on your data)
year_columns = [str(year) for year in range(1990, 2023)]  # Assuming years are from 1990 to 2022

# Iterate over all budget scenarios
for source, budgets in [("lamboll", BUDGET_GLOBAL_lamboll), ("foster", BUDGET_GLOBAL_foster)]:
    for prob, BUDGET_GLOBAL in budgets.items():

        BUDGET_TOTAL = BUDGET_GLOBAL * 1000  # Convert to MtCO2

        for method in ["equality", "responsibility"]:
            if method == "equality":
                remaining_carbon_budget = (BUDGET_TOTAL / POP_GLOBAL_2050) * df["2050"]
            else:
                remaining_carbon_budget = (
                    (df["Total_Emissions_1990_2022"].sum() + BUDGET_TOTAL) * (df["2050"] / POP_GLOBAL_2050)
                    - df["Total_Emissions_1990_2022"]
                )

            time_to_neutrality_lin = 2 * remaining_carbon_budget / df["2022"]
            time_to_neutrality_exp = -np.log(1 - 0.95) * remaining_carbon_budget / df["2022"]

            for curve, time_to_neutrality in [("linear", time_to_neutrality_lin), ("exponential", time_to_neutrality_exp)]:

                for _, row in df.iterrows():
                    remaining_budget = remaining_carbon_budget.loc[row.name]
                    time_to_neutrality_value = time_to_neutrality.loc[row.name]

                    # Check if remaining_carbon_budget is negative and adjust time_to_neutrality
                    if remaining_budget < 0:
                        # Start summing emissions backwards from 2022
                        cumulative_emissions = 0
                        country = row["Country"]

                        # Iterate through years from 2022 to 1970
                        for year in range(2022, 1969, -1):
                            # Access the emissions for the country and year from the emissions DataFrame
                            emissions_year = emissions.loc[emissions["Country"] == country, str(year)].values[0]
                            cumulative_emissions += emissions_year

                            # If cumulative emissions exceed the negative remaining budget, break
                            if cumulative_emissions >= abs(remaining_budget):
                                time_to_neutrality_value = - (2022 - year-1)
                                break

                    # Append the results to the rows list
                    rows.append({
                        "country": row["Country"],
                        "emission_2022": row["2022"],
                        "population_2050": row["2050"],
                        "repartition_method": method,
                        "source_of_budget": source,
                        "probability_of_reach": prob,
                        "curve_type": curve,
                        "remaining_carbon_budget": remaining_budget,  # Store the computed value
                        "time_to_neutrality": time_to_neutrality_value  # Store the computed value
                    })

# Convert list to DataFrame
df_final = pd.DataFrame(rows)

# Save the final cleaned dataset
df_final.to_excel(f"{output_directory}/carbon_budget_CBA_2°C.xlsx", index=False)

# ### 1.5°C

# Define the global carbon budgets (in GtCO2, converted to MtCO2)
BUDGET_GLOBAL_lamboll = {"33%": 480, "50%": 247, "67%": 60}
BUDGET_GLOBAL_foster = {"33%": 300, "50%": 250, "67%": 150}

# Merge population data with emissions
df = population_final.merge(emissions[["Country", "2022", "Total_Emissions_1990_2022"]],
                            left_on="corrected_country", right_on="Country")

# Create an empty list to store reformatted rows
rows = []

# List of year columns from the emissions dataframe (adjust based on your data)
year_columns = [str(year) for year in range(1990, 2023)]  # Assuming years are from 1990 to 2022

# Iterate over all budget scenarios
for source, budgets in [("lamboll", BUDGET_GLOBAL_lamboll), ("foster", BUDGET_GLOBAL_foster)]:
    for prob, BUDGET_GLOBAL in budgets.items():

        BUDGET_TOTAL = BUDGET_GLOBAL * 1000  # Convert to MtCO2

        for method in ["equality", "responsibility"]:
            if method == "equality":
                remaining_carbon_budget = (BUDGET_TOTAL / POP_GLOBAL_2050) * df["2050"]
            else:
                remaining_carbon_budget = (
                    (df["Total_Emissions_1990_2022"].sum() + BUDGET_TOTAL) * (df["2050"] / POP_GLOBAL_2050)
                    - df["Total_Emissions_1990_2022"]
                )

            time_to_neutrality_lin = 2 * remaining_carbon_budget / df["2022"]
            time_to_neutrality_exp = -np.log(1 - 0.95) * remaining_carbon_budget / df["2022"]

            for curve, time_to_neutrality in [("linear", time_to_neutrality_lin), ("exponential", time_to_neutrality_exp)]:

                for _, row in df.iterrows():
                    remaining_budget = remaining_carbon_budget.loc[row.name]
                    time_to_neutrality_value = time_to_neutrality.loc[row.name]

                    # Check if remaining_carbon_budget is negative and adjust time_to_neutrality
                    if remaining_budget < 0:
                        # Start summing emissions backwards from 2022
                        cumulative_emissions = 0
                        country = row["Country"]

                        # Iterate through years from 2022 to 1970
                        for year in range(2022, 1969, -1):
                            # Access the emissions for the country and year from the emissions DataFrame
                            emissions_year = emissions.loc[emissions["Country"] == country, str(year)].values[0]
                            cumulative_emissions += emissions_year

                            # If cumulative emissions exceed the negative remaining budget, break
                            if cumulative_emissions >= abs(remaining_budget):
                                time_to_neutrality_value = - (2022 - year-1)
                                break

                    # Append the results to the rows list
                    rows.append({
                        "country": row["Country"],
                        "emission_2022": row["2022"],
                        "population_2050": row["2050"],
                        "repartition_method": method,
                        "source_of_budget": source,
                        "probability_of_reach": prob,
                        "curve_type": curve,
                        "remaining_carbon_budget": remaining_budget,  # Store the computed value
                        "time_to_neutrality": time_to_neutrality_value  # Store the computed value
                    })

# Convert list to DataFrame
df_final = pd.DataFrame(rows)

# Save the final cleaned dataset
df_final.to_excel(f"{output_directory}/carbon_budget_CBA_1.5°C.xlsx", index=False)

# # Merging - Final database

# ## CO2

df_2C = pd.read_excel(f"{output_directory}/carbon_budget_2°C.xlsx")
df_15C = pd.read_excel(f"{output_directory}/carbon_budget_1.5°C.xlsx")

df_2C["temperature"] = "2°C"
df_15C["temperature"] = "1.5°C"
df_2C["scope"] = "territorial_CO2"
df_15C["scope"] = "territorial_CO2"

df = pd.concat([df_15C, df_2C], ignore_index=True)

# Save the final cleaned dataset
df.to_excel(f"{output_directory}/carbon_budget_CO2.xlsx", index=False)

# ## CBA

df_2C = pd.read_excel(f"{output_directory}/carbon_budget_CBA_2°C.xlsx")
df_15C = pd.read_excel(f"{output_directory}/carbon_budget_CBA_1.5°C.xlsx")

df_2C["temperature"] = "2°C"
df_15C["temperature"] = "1.5°C"
df_2C["scope"] = "consumption_based"
df_15C["scope"] = "consumption_based"

df = pd.concat([df_15C, df_2C], ignore_index=True)

# Save the final cleaned dataset
df.to_excel(f"{output_directory}/carbon_budget_CBA.xlsx", index=False)
