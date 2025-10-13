# EWBI Dashboard Runtime Files

This document lists the files that are **required** for running the EWBI Dashboard application.

## ✅ Required Files for App Runtime

### Core Application
- **`app_pca.py`** - Main dashboard application (Dash web app)
- **`variable_mapping.py`** - Display name mapping for indicators (imported by app_pca.py)

### Dependencies & Configuration  
- **`requirements.txt`** - Python package dependencies
- **`README.md`** - Documentation

### Assets & Styling
- **`assets/`** - Static files and CSS
  - **`assets/styles.css`** - Dashboard styling

### Data Files (in `../output/` directory)
- **`unified_all_levels_1_to_5_pca_weighted.csv`** - Main data file used by dashboard

### Deployment (optional for local development)
- **`deployment/`** - Deployment configuration files
  - **`deployment/Procfile`** - Process configuration
  - **`deployment/requirements.txt`** - Production dependencies
  - **`deployment/clevercloud.json`** - Cloud deployment config
  - **`deployment/deploy-clevercloud.sh`** - Deployment script
  - **`deployment/DEPLOYMENT.md`** - Deployment instructions

## ❌ NOT Required for App Runtime

These files are part of the **data processing pipeline** and are not needed to run the dashboard:

### Data Processing Scripts
- `0_raw_indicator_EHIS.py` - Raw EHIS data processing
- `0_raw_indicator_EU-SILC.py` - Raw EU-SILC data processing  
- `0_raw_indicator_HBS.py` - Raw HBS data processing
- `0_raw_indicator_LFS.py` - Raw LFS data processing
- `1_final_df.py` - Data finalization
- `2_preprocessing_executed_pca.py` - PCA preprocessing
- `3_generate_outputs_pca.py` - Output generation with PCA
- `4_PCA.py` - PCA analysis
- `apply_population_weighting.py` - Population weighting utilities
- `population_data_transform.py` - Population data transformation

### Utility & Fix Scripts
- `eu27_utils.py` - EU27 country utilities
- `fix_population_weights.py` - One-time population weight fix script

### Generated Files
- `__pycache__/` - Python bytecode cache

## 🚀 Quick Start

To run the dashboard, you only need:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard  
python app_pca.py
```

The app will load the pre-processed data from `../output/unified_all_levels_1_to_5_pca_weighted.csv` and start the dashboard on `http://localhost:8051`.