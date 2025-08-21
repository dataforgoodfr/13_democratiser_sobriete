# EWBI Dashboard - Active Files

This directory contains the active files for the EWBI dashboard system.

## Active Files (Keep)

### Core Dashboard
- **`ewbi_dashboard.py`** - The main Dash dashboard application

### Data Processing Pipeline
- **`generate_outputs.py`** - Generates the hierarchical data structure for the dashboard
- **`preprocessing_executed.ipynb`** - Jupyter notebook that preprocesses raw data

## Working Data Pipeline

1. **Preprocessing**: Run `preprocessing_executed.ipynb` to generate `primary_data_preprocessed.csv`
2. **Computation**: Run `generate_outputs.py` to generate:
   - `ewbi_master.csv` - Master dataframe with hierarchical structure
   - `ewbi_time_series.csv` - Time series dataframe
3. **Dashboard**: Run `ewbi_dashboard.py` to start the dashboard

## Archive

All other files have been moved to the `Archive/` folder to avoid confusion.

## Data Structure

The system works with a 4-level hierarchical structure:
- **Level 1**: EWBI (overall score)
- **Level 2**: EU Priorities
- **Level 3**: Secondary Indicators  
- **Level 4**: Primary Indicators

Economic indicators (marked in red) are automatically filtered out during processing. 