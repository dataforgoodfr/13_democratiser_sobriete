# Carbon Budget Visualization: Data Processing Methodology

## Overview

This document outlines the end-to-end data processing pipeline for the Carbon Budget Visualization project. The goal is to ingest raw data from various sources, clean and process it, and then generate a series of carbon budget scenarios based on different ethical principles of distribution.

The pipeline is executed by two main Python scripts:
1.  `Budget/code/ETL_preprocessing.py`: Handles data ingestion, cleaning, aggregation, and the calculation of key metrics.
2.  `Budget/code/ETL_scenarios.py`: Takes the processed data and generates the different future carbon budget scenarios.

---

## Step 1: Data Preprocessing & Metric Calculation (`ETL_preprocessing.py`)

This script serves as the foundation of the pipeline, consolidating multiple datasets into a clean, unified format and generating two key output files.

### A. Main Output: `combined_data.csv`

This file contains historical and projected data for countries and regional aggregates from 1990 to 2050.

#### Input Data Sources:
*   **Territorial Emissions:** `2025-04-22_CO2 Emissions_All Countries_ISO Code_1750-2023.xlsx`
*   **Consumption Emissions:** `2025-04-22_Consumption emissions MtCO2_ISO code.xlsx`
*   **Population:** `2025-04-21_Population per Country ISO code_1970-2050.xlsx`
*   **GDP (PPP):** `2025-04-21_GDP _PPP constant 2021 US$_per country ISO Code.xlsx`
*   **Country & Regional Mappings:** ISO codes, IPCC regions, and EU/G20 membership status are mapped from `28-04-2025_ISO_Codes_Mapping.xlsx` and `2024-04-21_IPCC Regional Breakdown_ISO Country Code.xlsx`.

#### Key Processing Steps:
1.  **Consolidation:** All data sources are loaded, cleaned, and merged into a single time-series dataframe.
2.  **Aggregation:** The script calculates aggregates for the World (WLD), EU, G20, and the 10 IPCC regions by summing the data from their constituent countries for each year. This is done separately for both "Territory" and "Consumption" emissions scopes to handle differences in data availability.
3.  **Share Calculations:** Several "share of total" metrics are calculated for each country and aggregate, per year. These include:
    *   `share_of_population`: Share of the world's annual population.
    *   `share_of_GDP_PPP`: Share of the world's annual GDP.
    *   `share_of_capacity`: This metric is used for the "Capacity" scenario. It is calculated to be **inversely proportional to GDP per capita** (`Population / GDP_per_capita`). This gives a larger share of the budget allocation to countries with lower economic capacity.

### B. Secondary Output: `planetary_boundary.csv`

This file provides a specific analysis based on a fixed planetary boundary.

*   **Methodology:** A total planetary budget of **830,000 MtCO2** is allocated to countries and regions based on their share of the world's cumulative population.
*   **Overshoot Calculation:** The script then analyzes the historical emissions data (from 1750) to determine the exact year in which each entity's cumulative emissions surpassed its allocated budget. The output file contains one row per country/aggregate, showing its budget and its overshoot year.

---

## Step 2: Scenario Generation (`ETL_scenarios.py`)

This script uses the `combined_data.csv` file to model future emissions pathways based on four distinct distribution principles.

#### Input Data Sources:
*   `combined_data.csv` (from Step 1)
*   `2025-04-21_Full file_Current carbon neutrality timeline per with Country ISO code.xlsx` (for the "Current Target" scenario)

#### Global Carbon Budgets:
The scenarios are based on the remaining global carbon budgets from the start of 2025, taken from the IPCC AR6 Synthesis Report. Budgets are available for both **1.5°C** and **2°C** warming targets at various probability levels (33%, 50%, 67%, 83%).

#### Distribution Scenarios:
1.  **Equality:** The remaining global budget is distributed to countries based on their **share of cumulative world population** from the latest available data year to 2050. This principle allocates budget based on human presence.
2.  **Responsibility:** A "total" budget is created by adding the *future* global budget to the *world's historical emissions* since 1990. This total amount is then allocated to each country based on its share of cumulative population from 1990 to **2050**. Finally, each country's own historical emissions are subtracted to determine its remaining budget. This method holds nations accountable for their past emissions.
3.  **Capacity:** The remaining global budget is distributed based on the `share_of_capacity` metric calculated in the preprocessing script. This allocates a larger portion of the remaining budget to countries with a lower GDP per capita, reflecting their reduced economic capacity to fund a rapid transition.
4.  **Current Target:** This scenario does not use the IPCC budgets. Instead, it calculates a linear pathway to neutrality based on each country's politically declared target year. If no target is declared, no pathway is calculated.

#### Final Outputs:
The script generates two files that are ready for visualization:
*   `scenario_parameters.csv`: Contains the detailed parameters for every unique scenario combination (country, warming target, probability, distribution principle), including the calculated neutrality year and country-specific budget.
*   `forecast_data.csv`: A long-format file containing the year-by-year forecasted emissions for every scenario, showing a linear decrease to zero.

Pour rendre l'environnement uv disponible depuis un jupyter notebook :

```
uv add --dev ipykernel
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project
```
