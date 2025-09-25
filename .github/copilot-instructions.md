# Copilot Instructions for 13_democratiser_sobriete

## Project Overview
This repository contains data pipelines and dashboards for the European Well-Being Index (EWBI) and related analyses. The main components are:
- **stream3_visualization/Well-being/**: EWBI dashboard, data aggregation, and indicator computation
- **stream3_visualization/Budget/**, **Decomposition/**: Additional dashboards and data processing
- **rag_system/**, **src/**: Data extraction, transformation, and domain logic

## Architecture & Data Flow
- **Raw data** (EU-SILC, Eurostat, etc.) is processed in `Well-being/code/0_raw_indicator_EU-SILC.py` and related scripts.
- **Aggregated outputs** are saved in `Well-being/output/` as CSVs for dashboard consumption.
- **Dashboards** (Dash/Plotly) read pre-aggregated data for fast, interactive analysis.
- **Hierarchical indicator structure**: Level 1 (EWBI) → Level 2 (EU Priorities) → Level 3 (Secondary Indicators) → Level 4 (Primary Indicators).

## Key Workflows
- **Data Preprocessing**: Run Jupyter notebooks and Python scripts in `Well-being/code/` to generate `primary_data_preprocessed.csv` and other outputs.
- **Aggregation**: Use `generate_outputs.py` to create master and time series files (`ewbi_master.csv`, `ewbi_time_series.csv`).
- **Dashboard Launch**: Run `ewbi_dashboard.py` for the main dashboard (`python ewbi_dashboard.py`).
- **Income Decile Calculation**: See `calculate_income_deciles` in `0_raw_indicator_EU-SILC.py` for custom weighted quantile logic.

## Conventions & Patterns
- **Data files**: All intermediate and final outputs are stored in `output/` subfolders, organized by indicator type and processing stage.
- **Indicator naming**: Follows EWBI convention (e.g., `IS-SILC-1`, `AN-SILC-1`). See `process_personal_indicators` and `process_household_indicators` for mapping.
- **Merging logic**: Household and personal data are joined using country/year/household/person IDs, with careful handling of missing columns.
- **Decile assignment**: Uses custom logic to assign income deciles per household, handling NaNs and edge cases.
- **Aggregation**: Geometric mean for decile-to-country, arithmetic mean for hierarchical aggregation.

## External Dependencies
- **Dash, pandas, plotly, tqdm**: Required for dashboard and data processing.
- **Poetry**: Recommended for dependency management (see main README for install instructions).

## Examples
- To add a new indicator, update the relevant processing function in `0_raw_indicator_EU-SILC.py` and ensure it is included in the aggregation pipeline.
- To debug data issues, inspect intermediate CSVs in `output/` and use the Jupyter notebook for stepwise analysis.

## References
- See `Well-being/code/README.md` and `Well-being/README.md` for workflow details and file structure.
- For conventions, review indicator mapping in `process_personal_indicators` and `process_household_indicators`.

---
If any section is unclear or missing, please provide feedback so this guide can be improved for future AI agents.