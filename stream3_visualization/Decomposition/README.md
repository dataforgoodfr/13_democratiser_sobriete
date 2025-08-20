# CO2 Decomposition Analysis Dashboard

This project provides an interactive dashboard for analyzing CO2 emissions decomposition scenarios across different regions, sectors, and decarbonization levers.

## Overview

The dashboard visualizes how different decarbonization levers contribute to CO2 emissions reductions over time:
- **Population**: Demographic changes
- **Sufficiency**: Consumption/production intensity changes  
- **Energy Efficiency**: Technological improvements
- **Supply Side Decarbonation**: Energy mix changes

## Files Structure

### Core Files
- `code/dashboard.py` - Main Dash application with interactive visualizations
- `code/data_preprocessing.py` - Processes raw Excel data into unified CSV format
- `code/create_sufficiency_scenarios_simple.py` - Generates World Sufficiency Lab scenarios
- `code/requirements.txt` - Python dependencies

### Data Files
- `Output/unified_decomposition_data.csv` - **Main dataset with ALL scenarios** (211 rows: 140 original + 70 WSL + 1 header)
- `Output/decomposition_summary.csv` - Summary statistics
- `Output/intermediary_decomposition_data.csv` - Detailed intermediate calculations (useful for analysis)

### Data Sources
- EU Commission scenarios (Fit-for-55, >85% decrease, >90% decrease, LIFE)
- Switzerland scenarios (Base, Zer0 A/B/C)
- World Sufficiency Lab scenarios (No Increase, With Sufficiency Measures)

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r code/requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   cd code
   python dashboard.py
   ```

3. **Access the dashboard:**
   Open your browser and go to `http://localhost:8051`

## Features

- **Waterfall Chart**: Shows CO2 emissions changes over time as percentages (2015 = 100%)
- **Bar Chart**: Displays lever contributions for selected scenarios
- **Interactive Controls**: Select zone, sector, and scenario combinations
- **Responsive Design**: Optimized for different screen sizes

## Data Generation Workflow

To regenerate the complete dataset with all scenarios, follow this **exact order**:

### Step 1: Process Raw Data
```bash
cd code
python data_preprocessing.py
```
**Outputs:**
- `Output/unified_decomposition_data.csv` - Contains ONLY original scenarios (EU + Switzerland)
- `Output/intermediary_decomposition_data.csv` - Detailed intermediate calculations
- `Output/decomposition_summary.csv` - Summary statistics

### Step 2: Generate World Sufficiency Lab Scenarios
```bash
python create_sufficiency_scenarios_simple.py
```
**Outputs:**
- `Output/world_sufficiency_lab_scenarios.csv` - Contains ONLY WSL scenarios

### Step 3: Combine All Scenarios (Manual Step)
```bash
# The WSL scenarios need to be manually combined with the original scenarios
# This ensures no duplicates and proper data integrity
# The final file should contain:
# - 140 original scenario rows (EU + Switzerland)
# - 70 World Sufficiency Lab scenario rows  
# - 1 header row
# Total: 211 rows
```

**Important Notes:**
- **Order matters**: Always run `data_preprocessing.py` first to establish the base data
- **No automatic combination**: The scripts don't automatically merge files to prevent data loss
- **Dashboard compatibility**: The dashboard reads from `unified_decomposition_data.csv` which should contain ALL scenarios
- **Data validation**: Check that the final file has exactly 211 rows (210 data + 1 header)

## Troubleshooting

### Common Issues
- **Missing WSL scenarios**: If you only see 140 rows, you need to run Step 2 and manually combine
- **Duplicate scenarios**: If you see more than 211 rows, there are duplicates that need to be removed
- **Dashboard not updating**: Restart the dashboard after data changes

### Data Verification
```bash
# Check the final dataset has correct number of rows
wc -l Output/unified_decomposition_data.csv
# Should show: 211

# Verify scenario distribution
grep -c "World Sufficiency Lab" Output/unified_decomposition_data.csv
# Should show: 70 (35 scenarios × 2 levers each)
```

## Scenarios

- **Original Scenarios**: Official EU and Switzerland decarbonization pathways
- **World Sufficiency Lab**: Explores sufficiency measures with 40% intensity decrease
- **Lever Analysis**: Breaks down emissions changes by population, sufficiency, efficiency, and supply-side factors 