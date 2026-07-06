# Report 1: Switzerland vs EU-27 Housing and Energy Analysis

## Overview

This report provides a comprehensive comparison between Switzerland and the EU-27 for Housing and Energy indicators, combining EWBI data with external datasets for policy-relevant insights.

## Objectives

1. **Comparative Analysis**: Direct comparison of Switzerland vs EU-27 performance
2. **Housing Focus**: Housing quality, affordability, and access indicators
3. **Energy Focus**: Energy efficiency, consumption, and access indicators  
4. **Policy Insights**: Data-driven recommendations for Swiss policy alignment
5. **Temporal Analysis**: Time series trends and convergence/divergence patterns

## Directory Structure

```
1_switzerland_vs_eu27_housing_energy/
├── code/
│   ├── eu-silc_swiss.py                     # EU-SILC data analysis for Switzerland
│   ├── eurostat_analysis_swiss.py           # Eurostat housing and energy analysis (CH vs EU-27)
│   ├── eurostat_federal_ghg_emissions.py    # Federal GHG emissions analysis
│   ├── eurostat_trade.py                    # Trade data analysis
│   ├── ewbi_treatment.py                    # EWBI data processing and treatment
│   ├── extract_excel_data.py                # Excel data extraction utilities
│   ├── fso_household_expense_analysis.py    # Swiss Federal Statistical Office household expense analysis
│   ├── hbs_analysis_switzerland_eu27.py     # Household Budget Survey comparative analysis
│   ├── housing_allowance_fix.py             # Housing allowance data corrections
│   ├── oecd_graphs_generator.py             # OECD data visualizations
│   ├── plot_functions.py                    # Shared plotting utilities
│   ├── swiss_energy_dependency.py           # Swiss energy dependency analysis
│   ├── 2_FINAL_trade.ipynb                  # Trade analysis notebook
│   └── 2_FINAL_trade_energy_materials.ipynb # Trade, energy and materials notebook
├── external_data/                           # External data sources (OECD, Eurostat, FSO)
└── outputs/
    ├── graphs/
    │   ├── OECD/                            # OECD-based visualizations
    │   └── EUROSTAT/                        # Eurostat housing and energy visualizations
    ├── intermediate/                        # Processed intermediate datasets
    ├── tables/                              # Summary tables
    └── final/                               # Report-ready outputs
```

## Data Sources

### EWBI Data

- Level 2: Housing and Energy EU Priority scores
- Level 3: Housing quality and Energy secondary indicator scores
- Level 4: Primary indicators for Housing and Energy
- Decile breakdown for inequality analysis

### External Data Sources
- **Housing Market Data**: Prices, affordability indices, rental markets
- **Housing Quality**: Overcrowding, basic facilities, housing conditions
- **Energy Consumption**: Household energy use, efficiency metrics
- **Energy Affordability**: Energy poverty, price indices
- **Policy Data**: Housing and energy policy frameworks

## Key Analysis Components

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

## Running the Analysis

### Prerequisites

Ensure the EWBI pipeline has been executed:
```bash
cd ../../code
python 4_weighting_aggregation.py
```

#### Eurostat Analysis (Switzerland vs EU-27)

```bash
python code/eurostat_analysis_swiss.py
```

Output location: `outputs/graphs/EUROSTAT/`

#### Other Analysis Scripts

1. OECD visualizations: `python code/oecd_graphs_generator.py`
2. EU-SILC analysis: `python code/eu-silc_swiss.py`
3. Household Budget Survey analysis: `python code/hbs_analysis_switzerland_eu27.py`
4. Swiss energy dependency: `python code/swiss_energy_dependency.py`

## Expected Outputs

### Graphs (`outputs/graphs/`)

- **OECD**: OECD-based visualizations
- **EUROSTAT**: Eurostat housing and energy comparisons (rooms, ownership, energy efficiency, under-occupation, tenure status)
- Time series plots for each indicator
- Decile comparison charts

### Tables (`outputs/tables/`)

- Summary tables with key findings
- Data appendices

## Configuration

### Customizable Parameters
- Time period for analysis
- Specific indicators to include/exclude
- Decile analysis depth
- External data sources to integrate

### External Data Integration
- Place housing data in `external_data/housing/`
- Place energy data in `external_data/energy/`
- Update data loading scripts to include new sources

## Key Research Questions

1. **Performance Gap**: Where does Switzerland over/under-perform vs EU-27?
2. **Inequality Patterns**: Do income deciles show similar patterns in both regions?
3. **Policy Alignment**: Which Swiss policies align with EU best practices?
4. **Temporal Trends**: Are gaps widening or narrowing over time?
5. **Intervention Points**: Where would policy changes have greatest impact?

## Dependencies

- **EWBI Pipeline**: Aggregated data in `../../output/ewbi_master_aggregated.csv`
- **Shared Utilities**: `../shared/code/ewbi_data_loader.py` and `../shared/code/visualization_utils.py`
- **External Datasets**: Various sources in `external_data/`

## Notes

- This analysis focuses on Housing and Energy priorities
- All EWBI levels (1–4) are available through the shared data loader
- External data integration provides policy context and validation
- Outputs are designed for policy briefings and academic publication

