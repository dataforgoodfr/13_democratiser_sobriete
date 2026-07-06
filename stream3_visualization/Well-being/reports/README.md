# Well-being Reports

This directory contains report-specific analyses using the European Well-Being Index (EWBI) pipeline data, combined with external data sources for comprehensive policy analysis.

## Directory Structure

```
reports/
├── 1_switzerland_vs_eu27_housing_energy/    # Switzerland vs EU-27 Housing and Energy comparison
├── 2_switzerland_comprehensive/             # Switzerland analysis across all EU priorities
├── 3_eu_analysis_with_examples/            # EU-wide analysis with country examples
├── 4_mobility_model/                       # Mobility-focused well-being analysis
├── 5_france/                               # France-specific well-being analysis
└── shared/                                  # Shared resources across all reports
    ├── code/
    │   ├── ewbi_data_loader.py             # Standardized EWBI data loading functions
    │   └── visualization_utils.py          # Common plotting and charting utilities
    └── external_datasets/                  # Shared external data sources
        ├── housing/
        ├── energy/
        ├── economics/
        └── demographics/
```

## Report Summaries

### 1. Switzerland vs EU-27 Housing and Energy

Comparative analysis between Switzerland and EU-27 for Housing and Energy indicators, combining EWBI data with OECD and Eurostat external datasets.

### 2. Switzerland Comprehensive

In-depth analysis of Switzerland across all EU priorities, with a focus on income decile decomposition and Swiss-specific external data sources.

### 3. EU Analysis with Examples

EU-wide analysis across all priorities with selected country examples to illustrate policy patterns, best practices, and areas for improvement.

### 4. Mobility Model

Analysis focused on mobility indicators and transport-related well-being dimensions.

### 5. France

France-specific analysis combining EWBI indicators with French national data sources.

## Shared Resources

### Code Utilities (`shared/code/`)
- `ewbi_data_loader.py`: Standardized EWBI data loading functions
- `visualization_utils.py`: Common plotting and charting utilities
- Additional utility modules as needed

### External Datasets (`shared/external_datasets/`)
- `housing/`: Housing market, affordability, and quality data
- `energy/`: Energy consumption, efficiency, and pricing data  
- `economics/`: Economic indicators, GDP, inflation data
- `demographics/`: Population, age structure, migration data

## Getting Started

1. **Data Dependencies**: Ensure the EWBI pipeline has been run and outputs are available in `../output/ewbi_master_aggregated.csv`

2. **Shared Utilities**: All reports can import from shared code:
   ```python
   from shared.code.ewbi_data_loader import load_ewbi_unified_data
   from shared.code.visualization_utils import create_time_series_plot
   ```

3. **External Data**: Place shared datasets in `shared/external_datasets/` organized by domain

4. **Report-Specific Analysis**: Each report has its own `code/`, `external_data/`, and `outputs/` directories

## Data Flow

1. **EWBI Pipeline** → Core well-being indicators and aggregations
2. **External Sources** → Additional policy-relevant data
3. **Report Analysis** → Combined analysis and visualization
4. **Outputs** → Report-ready tables, graphs, and summaries

## Integration with EWBI Pipeline

This reports structure maintains clean separation from the core EWBI pipeline while providing standardized access to its outputs. The shared utilities ensure consistent data loading and visualization across all reports.

## Notes

- Each report directory is self-contained for analysis purposes
- Shared resources promote consistency and reduce code duplication
- External datasets are organized by domain for easy reuse
- All reports can be run independently or as a suite

---

For specific report documentation, see the README.md file in each report directory.