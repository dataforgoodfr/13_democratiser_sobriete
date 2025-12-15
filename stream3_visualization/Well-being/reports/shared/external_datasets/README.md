# Shared External Datasets - README

This directory contains external datasets that are used across multiple reports in the Well-being Reports suite.

## 📁 Directory Structure

```
shared/external_datasets/
├── housing/          # Housing market and quality data
├── energy/           # Energy consumption and efficiency data
├── economics/        # Economic indicators and financial data  
├── demographics/     # Population and demographic data
└── README.md         # This file
```

## 🏠 Housing Data (`housing/`)

### Recommended Datasets
- **EU Housing Price Index**: Eurostat housing price trends
- **Housing Affordability Index**: OECD housing cost-to-income ratios
- **Swiss Housing Market Data**: Swiss Federal Statistical Office housing data
- **Overcrowding Statistics**: EU-SILC housing quality indicators
- **Social Housing Data**: National social housing provision statistics

### File Organization
- Name files descriptively: `eu_housing_prices_2015_2023.csv`
- Include metadata files: `housing_data_sources.md`
- Use standardized country codes (ISO 3166-1 alpha-2)

## ⚡ Energy Data (`energy/`)

### Recommended Datasets  
- **Household Energy Consumption**: Eurostat energy consumption by household
- **Energy Poverty Indicators**: EU Energy Poverty Observatory data
- **Swiss Energy Statistics**: Swiss Federal Office of Energy data
- **Energy Efficiency Indicators**: EU building efficiency measures
- **Renewable Energy Adoption**: National renewable energy statistics

### File Organization
- Separate consumption from efficiency data
- Include both absolute and per-capita measures
- Maintain temporal consistency across datasets

## 💰 Economics Data (`economics/`)

### Recommended Datasets
- **GDP and GNI**: OECD national accounts data
- **Income Distribution**: Gini coefficients and income shares
- **Swiss Economic Indicators**: State Secretariat for Economic Affairs data
- **EU Economic Convergence**: European Commission economic indicators
- **Inflation and Price Indices**: National consumer price index data

### File Organization
- Use purchasing power parity adjusted figures where appropriate
- Include confidence intervals for survey-based data
- Align temporal coverage with EWBI data availability

## 👥 Demographics Data (`demographics/`)

### Recommended Datasets
- **Population Statistics**: Eurostat demographic indicators
- **Age Structure Data**: Population pyramids and dependency ratios
- **Migration Statistics**: EU migration and mobility data
- **Swiss Demographics**: Swiss Federal Statistical Office population data
- **Household Composition**: EU-SILC household structure data

### File Organization
- Break down by age groups consistent with policy relevance
- Include both stock and flow measures for migration
- Align geographical units with EWBI country coverage

## 🔧 Data Standards

### File Naming Convention
```
[domain]_[geography]_[indicator]_[time_period].[extension]
```

Examples:
- `housing_eu27_price_index_2015_2023.csv`
- `energy_switzerland_consumption_2010_2022.xlsx`
- `economics_oecd_gini_coefficients_2020.csv`

### Required Metadata
Each dataset should include:
- **Source**: Organization and specific publication
- **Last Updated**: When data was downloaded/processed
- **Coverage**: Geographic and temporal scope
- **Methodology**: Brief description of measurement approach
- **Limitations**: Known data quality issues or gaps

### Column Standards
- **country**: ISO 3166-1 alpha-2 country codes (CH, DE, FR, etc.)
- **year**: 4-digit year format (2015, 2016, etc.)
- **indicator**: Descriptive indicator name
- **value**: Numeric value
- **unit**: Unit of measurement  
- **source**: Data source abbreviation

## 📊 Integration with Reports

### Automatic Loading
The shared data loader can automatically detect and load datasets:

```python
from shared.code.ewbi_data_loader import load_external_dataset

# Load housing data
housing_data = load_external_dataset('housing', 'eu_housing_prices_2015_2023.csv')

# Load energy data  
energy_data = load_external_dataset('energy', 'swiss_energy_consumption.csv')
```

### Cross-Report Usage
- **Report 1**: Housing and energy data for Switzerland vs EU-27 comparison
- **Report 2**: Swiss-specific data for national context
- **Report 3**: EU-wide data for cross-country examples

## 🔄 Update Procedures

### Regular Updates
- **Quarterly**: Update high-frequency indicators (prices, consumption)
- **Annual**: Update structural indicators (housing stock, demographics)
- **Ad-hoc**: Update following major data releases

### Version Control
- Keep previous versions in `archive/` subdirectories
- Document changes in `CHANGELOG.md` files
- Use semantic versioning for major methodology changes

## 💡 Best Practices

### Data Quality
- Validate external data against EWBI trends where possible
- Document known issues and limitations clearly
- Use official statistical sources where available

### Documentation
- Include data dictionaries for complex datasets  
- Document any transformations or calculations applied
- Provide contact information for data providers

### Harmonization
- Align temporal coverage with EWBI availability where possible
- Use consistent geographical units across datasets
- Standardize indicator definitions across countries

---

For questions about specific datasets or to request new data integration, contact the Well-being Reports team or refer to individual report documentation.