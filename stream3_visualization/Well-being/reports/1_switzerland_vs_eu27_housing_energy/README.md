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
│   ├── integrated_analysis.py            # Combined housing + energy insights
│   └── report_generator.py               # Generate final report outputs
├── external_data/
│   ├── housing/                          # Switzerland and EU housing data
│   ├── energy/                           # Switzerland and EU energy data
│   └── policy/                           # Policy documents and frameworks
├── outputs/
│   ├── intermediate/                     # Processed datasets
│   ├── tables/                          # Summary tables for report
│   ├── graphs/                          # Visualizations
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

1. **Load and Prepare Data**:
   ```python
   # Uses shared utilities for EWBI data
   from shared.code.ewbi_data_loader import load_ewbi_unified_data, get_housing_energy_indicators
   from shared.code.visualization_utils import create_time_series_plot
   ```

2. **Run Housing Analysis**:
   ```bash
   python code/housing_analysis.py
   ```

3. **Run Energy Analysis**:
   ```bash
   python code/energy_analysis.py
   ```

4. **Generate Time Series Comparisons**:
   ```bash
   python code/swiss_vs_eu27_time_series.py
   ```

5. **Create Integrated Report**:
   ```bash
   python code/report_generator.py
   ```

## 📊 Expected Outputs

### Tables (`outputs/tables/`)
- `housing_indicators_comparison.csv`: Swiss vs EU housing metrics
- `energy_indicators_comparison.csv`: Swiss vs EU energy metrics  
- `temporal_trends_summary.csv`: Time series trend analysis
- `policy_gap_analysis.csv`: Areas for policy attention

### Graphs (`outputs/graphs/`)
- Time series plots for each indicator
- Decile comparison charts
- Policy gap visualization
- Integrated housing-energy analysis

### Final Report (`outputs/final/`)
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