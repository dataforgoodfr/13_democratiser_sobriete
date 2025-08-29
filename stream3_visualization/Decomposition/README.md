# CO2 Decomposition Analysis Dashboard

This project provides interactive dashboards for analyzing CO2 emissions decomposition scenarios across different regions, sectors, and decarbonization levers.

## Overview

The dashboard visualizes how different decarbonization levers contribute to CO2 emissions reductions over time:
- **Population**: Demographic changes
- **Sufficiency**: Consumption/production intensity changes  
- **Energy Efficiency**: Technological improvements
- **Supply Side Decarbonation**: Energy mix changes

## Current File Structure (August 2024)

### Core Application Files
- `code/app.py` - **Main EU/Switzerland decomposition dashboard**
- `code/app_world.py` - **World decomposition dashboard**
- `code/requirements.txt` - Python dependencies

### Data Processing Files
- `code/data_preprocessing.py` - **Main data processor** (EU + Switzerland + World scenarios integration)
- `code/world_data_preprocessing.py` - **World data processor** with corrected sign conventions
- `code/create_sufficiency_scenarios_simple.py` - **Integrated into main preprocessing** (no need to run separately)

### Data Files
- `data/2025-04-28_EC scenarios data_Decomposition_compiled.xlsx` - EU Commission scenarios
- `data/2025-08-13_CH scenarios data_Decomposition_Compiled.xlsx` - Switzerland scenarios  
- `data/2025-08-20_REMIND Shape_Data_Compiled.xlsx` - World data source

### Output Files
- `Output/unified_decomposition_data.csv` - **Main combined dataset** (300+ records: EU + Switzerland + World scenarios)
- `Output/world_unified_decomposition_data.csv` - World-specific dataset
- `Output/intermediary_decomposition_data.csv` - Debug/audit file (EU + Switzerland)
- `Output/world_intermediary_decomposition_data.csv` - Debug/audit file (World)

## Available Sectors

### EU Sectors
- Buildings - Residential
- Buildings - Services
- Transport - Passenger cars
- Transport - Rail
- Industry - Steel industry
- Industry - Non-ferrous metal industry
- Industry - Chemicals industry
- Industry - Non-Metallic Minerals industry
- Industry - Pulp, Paper & Print industry

### Switzerland Sectors
- Buildings - Residential
- Buildings - Services
- Passenger Land Transport
- Industry - Cement
- Cement industry
- Steel industry

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r code/requirements.txt
   ```

2. **Run the main dashboard (EU/Switzerland):**
   ```bash
   cd code
   python app.py
   ```

3. **Run the world dashboard:**
   ```bash
   cd code
   python app_world.py
   ```

4. **Access the dashboards:**
   - Main dashboard: `http://localhost:8050`
   - World dashboard: `http://localhost:8051`

## Data Generation Workflow

**Simplified workflow** - everything is now integrated into a single script:

### Single Command to Regenerate All Data:
```bash
cd Decomposition
python code/data_preprocessing.py
```

**What this does automatically:**
1. ✅ Processes EU and Switzerland data with new sectors
2. ✅ Generates World Sufficiency Lab scenarios
3. ✅ Combines all datasets into unified file
4. ✅ Applies consistent sign conventions across all zones

**Output:**
- `Output/unified_decomposition_data.csv` - Complete dataset (300+ records)
- All sectors and scenarios integrated automatically

## Recent Improvements (August 2024)

### ✅ Sign Convention Fix
- **World data** now uses same sign convention as EU/Switzerland
- **Negative percentages** = increases emissions
- **Positive percentages** = decreases emissions
- Consistent interpretation across all dashboards

### ✅ Sector Configuration Updates
- Added new EU sectors (Transport, Industry sub-sectors)
- Added new Switzerland sectors (Industry - Cement, Steel industry)
- Updated sector names to match actual data files

### ✅ Code Cleanup
- Removed obsolete scripts and intermediate files
- Streamlined data processing workflow
- Integrated World scenarios generation into main processor

### ✅ Path Fixes
- Fixed data directory detection for different run locations
- Fixed output directory paths for proper file saving

## Features

- **Waterfall Chart**: Shows CO2 emissions changes over time as percentages (2015 = 100%)
- **Bar Chart**: Displays lever contributions for selected scenarios
- **Interactive Controls**: Select zone, sector, and scenario combinations
- **Responsive Design**: Optimized for different screen sizes
- **Consistent Sign Convention**: Same interpretation across all zones

## Deployment to Clever Cloud

### European Decomposition Dashboard (`app.py`)
Deploy using:
```bash
git push clever-decomposition visualizations-combined:master
```

**Remote:** `clever-decomposition` → `git+ssh://git@push-n3-par-clevercloud-customers.services.clever-cloud.com/app_ac31ad44-d32f-4998-87c6-b9b699c29c63.git`

### World Decomposition Dashboard (`app_world.py`)
Deploy using:
```bash
git push clever-world-decomposition visualizations-combined:master
```

**Remote:** `clever-world-decomposition` → `git+ssh://git@push-n3-par-clevercloud-customers.services.clever-cloud.com/app_e1c3f118-5441-449a-99f3-fa4036bb2ad4.git`

### Available Clever Cloud Remotes
```bash
# List all available remotes
git remote -v

# Available remotes:
# - clever-budget → Budget dashboard
# - clever-decomposition → European decomposition dashboard  
# - clever-world-decomposition → World decomposition dashboard
# - origin → GitHub repository
```

### Deployment Workflow
1. **Make changes** to the dashboard code
2. **Commit changes** to the `visualizations-combined` branch
3. **Push to GitHub**: `git push origin visualizations-combined`
4. **Deploy to Clever Cloud**: Use the appropriate push command above
5. **Verify deployment** in Clever Cloud dashboard

## Troubleshooting

### Common Issues
- **Missing sectors**: Ensure data files contain the expected sector names
- **Sign convention confusion**: All zones now use consistent signs (negative = increases, positive = decreases)
- **Dashboard not updating**: Restart the dashboard after data changes

### Data Verification
```bash
# Check the final dataset has correct number of records
wc -l Output/unified_decomposition_data.csv
# Should show: 300+ (including header)

# Verify all zones are present
grep -c "EU" Output/unified_decomposition_data.csv
grep -c "Switzerland" Output/unified_decomposition_data.csv
grep -c "World" Output/unified_decomposition_data.csv
```

## Data Sources

- **EU Commission scenarios**: Fit-for-55, >85% decrease, >90% decrease, LIFE
- **Switzerland scenarios**: Base, Zer0 A/B/C
- **World Sufficiency Lab scenarios**: Consumption/Production per capita at 2015 levels
- **Lever Analysis**: Breaks down emissions changes by population, sufficiency, efficiency, and supply-side factors 