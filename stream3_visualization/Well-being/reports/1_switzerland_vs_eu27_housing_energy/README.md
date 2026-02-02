# Report 1: Switzerland vs EU-27 Housing & Energy Analysis

## 📊 Overview

This report provides a comprehensive comparison between Switzerland and the EU-27 for Housing and Energy indicators, combining EWBI data with external datasets for policy-relevant insights.

## 🎯 Objectives

1. **Comparative Analysis**: Direct comparison of Switzerland vs EU-27 performance
2. **Housing Focus**: Housing quality, affordability, and access indicators
3. **Energy Focus**: Energy efficiency, consumption, and access indicators  
4. **Policy Insights**: Data-driven recommendations for Swiss policy alignment
5. **Temporal Analysis**: Time series trends and convergence/divergence patterns

## 📁 Directory Structure

```
1_switzerland_vs_eu27_housing_energy/
├── code/
│   ├── swiss_vs_eu27_time_series.py      # Time series analysis (moved from Well-being)
│   ├── housing_analysis.py               # Housing-specific analysis
│   ├── energy_analysis.py                # Energy-specific analysis
│   ├── oecd_graphs_generator.py          # OECD data visualizations
│   ├── eurostat_analysis_swiss.py        # NEW: Eurostat housing/energy analysis for Switzerland
│   ├── integrated_analysis.py            # Combined housing + energy insights
│   └── report_generator.py               # Generate final report outputs
├── external_data/
│   └── [OECD data files]
├── outputs/
│   ├── intermediate/                     # Processed datasets
│   ├── tables/                          # Summary tables for report
│   ├── graphs/
│   │   ├── OECD/                        # OECD visualizations
│   │   └── EUROSTAT/                    # NEW: Eurostat visualizations (Switzerland vs EU27)
│   └── final/                           # Report-ready outputs
└── README.md                            # This file
```

## 📈 Data Sources

### EWBI Data (via shared utilities)
- **Level 5 Indicators**: Primary indicators for Housing and Energy priority
- **Level 2 Aggregation**: EU Priority level for Housing and Energy
- **Decile Analysis**: Income-based breakdown for inequality insights
- **Time Series**: Historical trends from available years

### External Data Sources
- **Housing Market Data**: Prices, affordability indices, rental markets
- **Housing Quality**: Overcrowding, basic facilities, housing conditions
- **Energy Consumption**: Household energy use, efficiency metrics
- **Energy Affordability**: Energy poverty, price indices
- **Policy Data**: Housing and energy policy frameworks

## 🔍 Key Analysis Components

### 1. Housing Analysis
- **EWBI Housing Indicators**: Quality, expense, access metrics
- **Market Analysis**: Price trends, affordability comparison
- **Policy Gaps**: Areas where Switzerland diverges from EU trends

### 2. Energy Analysis  
- **EWBI Energy Indicators**: Access, affordability, efficiency
- **Consumption Patterns**: Household energy use comparison
- **Efficiency Metrics**: Building efficiency, renewable adoption

### 3. Integrated Insights
- **Housing-Energy Nexus**: Energy efficiency in housing sector
- **Affordability Trade-offs**: Housing costs vs energy costs
- **Policy Synergies**: Integrated policy recommendations

### 4. Temporal Dynamics
- **Convergence Analysis**: Are Switzerland and EU converging?
- **Policy Impact**: Effect of major policy changes
- **Future Projections**: Trend extrapolation and scenarios

## 🚀 Running the Analysis

### Prerequisites
```bash
# Ensure EWBI pipeline has been run
cd ../Well-being/code
python 3_generate_outputs.py

# Install additional requirements if needed
pip install pandas plotly matplotlib seaborn
```

### Execution Steps

#### EUROSTAT Analysis (NEW - Adapted from EU Analysis)
The new `eurostat_analysis_swiss.py` script generates housing and energy visualizations specifically for Switzerland vs EU27 comparison:

```bash
# Generate Eurostat-based visualizations
python code/eurostat_analysis_swiss.py
```

**Output Location**: `outputs/graphs/EUROSTAT/`

**Generated Visualizations** (6 key themes):
1. **Average Number of Rooms Per Person** (`1_rooms_switzerland_vs_eu27_tenure.png`)
   - Compares rooms by tenure status (Owner, Tenant, Total)
   - Side-by-side bar chart: Switzerland vs EU27

2. **Real Estate Ownership** (`2_real_estate_switzerland_vs_eu27_quintiles.png`)
   - Persons owning real estate other than main residence
   - Breakdown by income quintile (Q1-Q5, Total)
   - Switzerland vs EU27 comparison

3. **Energy Efficiency Improvements** (`3_energy_efficiency_switzerland_vs_eu27_age.png`)
   - Dwellings with energy efficiency improvements (last 5 years)
   - Breakdown by age group (16+, 16-29, 25-34, 35-44, 45-64, 65+)
   - Switzerland vs EU27 comparison

4. **Business Enterprise R&D by NACE** (`4_berd_switzerland_vs_eu27_nace_*.png`)
   - Enterprise statistics by size class and NACE Rev. 2 activity
   - Data from 2021 onwards
   - Multiple files for different units (PPS per inhabitant, % of GDP)

5. **Under-occupied Dwellings** (`5_under_occupied_switzerland_vs_eu27_age.png`)
   - Share of people in under-occupied dwellings
   - Breakdown by age (<18, 18-64, 65+)
   - Switzerland vs EU27 comparison

6. **Tenure Status Distribution** (`6_tenure_status_switzerland_vs_eu27.png`)
   - Population distribution by tenure status and household type
   - Income group analysis (total population focus)
   - Switzerland vs EU27 comparison

**Data Source**: Uses datasets from `3_eu_analysis_with_examples/external_data/` (shared reference)

**Styling**: 
- Switzerland: Yellow (#ffd558)
- EU27: Blue (#80b1d3)
- Side-by-side bar charts for easy comparison
- Consistent with oecd_graphs_generator.py visual style

#### Other Analysis Scripts

1. **OECD Visualizations**:
   ```bash
   python code/oecd_graphs_generator.py
   ```

2. **Housing Analysis**:
   ```bash
   python code/housing_analysis.py
   ```

3. **Energy Analysis**:
   ```bash
   python code/energy_analysis.py
   ```

4. **Time Series Comparisons**:
   ```bash
   python code/swiss_vs_eu27_time_series.py
   ```

5. **Integrated Report**:
   ```bash
   python code/report_generator.py
   ```

## 📊 Expected Outputs

### Graphs (`outputs/graphs/`)
- **OECD folder**: OECD-based visualizations
- **EUROSTAT folder**: NEW! Eurostat housing/energy comparisons (6 visualizations):
  - Rooms comparison by tenure status
  - Real estate ownership by income
  - Energy efficiency by age group
  - Enterprise R&D by NACE sector
  - Under-occupied dwellings by age
  - Tenure status distribution
- Time series plots for each indicator
- Decile comparison charts
- Policy gap visualization
- Integrated housing-energy analysis

### Tables (`outputs/tables/`)
- Executive summary with key findings
- Policy recommendations
- Data appendices
- Interactive dashboard (if applicable)

## 🔧 Configuration

### Customizable Parameters
- Time period for analysis
- Specific indicators to include/exclude
- Decile analysis depth
- External data sources to integrate

### External Data Integration
- Place housing data in `external_data/housing/`
- Place energy data in `external_data/energy/`
- Update data loading scripts to include new sources

## 💡 Key Research Questions

1. **Performance Gap**: Where does Switzerland over/under-perform vs EU-27?
2. **Inequality Patterns**: Do income deciles show similar patterns in both regions?
3. **Policy Alignment**: Which Swiss policies align with EU best practices?
4. **Temporal Trends**: Are gaps widening or narrowing over time?
5. **Intervention Points**: Where would policy changes have greatest impact?

## 🔗 Dependencies

- **EWBI Pipeline**: `../Well-being/output/unified_all_levels_1_to_5_pca_weighted.csv`
- **Shared Utilities**: `../shared/code/ewbi_data_loader.py`
- **Visualization Tools**: `../shared/code/visualization_utils.py`
- **External Datasets**: Various sources in `external_data/`

## 📝 Notes

- This analysis focuses specifically on Housing and Energy priorities
- All EWBI levels (1-5) are available through the shared data loader
- External data integration allows for policy context and validation
- Time series analysis respects data availability for both regions
- Outputs are designed for policy briefings and academic publication

## 🤝 Contributing

When adding new analysis:
1. Follow the existing code structure
2. Use shared utilities for EWBI data access
3. Document data sources and assumptions
4. Update this README with new components

---

For technical support or questions about the EWBI data, refer to `../Well-being/README.md`.