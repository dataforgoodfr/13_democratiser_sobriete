# European Well-Being Index (EWBI) - Computation Logic

This document provides a detailed technical explanation of how the European Well-Being Index (EWBI) is computed based on the implementation in the codebase.

## Overview

The EWBI is a composite indicator that measures well-being across European countries using a hierarchical aggregation structure. The computation involves multiple levels of aggregation from raw survey data to the final EWBI score.

## Hierarchical Structure

The EWBI follows a 4-level hierarchical structure (in the PCA version):

1. **Level 5 (Raw Data)**: Original survey responses from EU-SILC, HBS, LFS, and EHIS
2. **Level 4 (Normalized Data)**: Standardized indicators using winsorization and percentile scaling
3. **Level 2 (EU Priorities)**: Aggregated indicators grouped by EU policy priorities using PCA-based weights
4. **Level 1 (EWBI)**: Final composite index aggregating all EU priorities

*Note: Level 3 (Secondary Indicators) is skipped in the PCA version for simplification.*

## Data Sources and Preprocessing

### 1. Raw Data Collection (Level 5)

#### Data Sources:
- **EU-SILC (European Union Statistics on Income and Living Conditions)**: Household and personal indicators
- **HBS (Household Budget Survey)**: Consumption and expenditure data
- **LFS (Labor Force Survey)**: Employment and labor market indicators
- **EHIS (European Health Interview Survey)**: Health-related indicators

#### Key Processing Steps:

**EU-SILC Processing (`0_raw_indicator_EU-SILC.py`):**
- Combines household (H-files), personal (P-files), and register data (D-files, R-files)
- Calculates income deciles using equivalized disposable income with OECD modified scale:
  - First adult = 1.0 weight
  - Additional adults (≥14 years) = 0.5 weight
  - Children (<14 years) = 0.3 weight
- Computes overcrowding indicator based on dwelling room requirements
- Processes both household-level and personal-level indicators
- Applies survey weights for representative statistics

**Income Decile Calculation:**
```python
def weighted_quantile(values, weights, quantiles):
    """Computes weighted quantiles using household weights"""
    sorter = np.argsort(values)
    values_sorted = values[sorter]  
    weights_sorted = weights[sorter]
    cumsum_weights = np.cumsum(weights_sorted)
    total_weight = cumsum_weights[-1]
    normalized_weights = cumsum_weights / total_weight
    return np.interp(quantiles, normalized_weights, values_sorted)
```

**Indicator Computation Logic:**
- Uses binary filtering based on survey response codes
- Calculates weighted shares of population meeting specific conditions
- Example for household indicators:
```python
# For each indicator, define condition filters
variable_filters = {
    "HS050": [1],                          # AN-SILC-1: Financial distress
    "HS011": lambda row: [1] if row["HB010"] < 2008 else [1, 2],  # HH-SILC-1: Dwelling issues
    "overcrowded": [1],                    # HQ-SILC-1: Overcrowding
    # ... more indicators
}

# Calculate weighted shares
for var, condition in variable_filters.items():
    mask = group[f"_valid_{var}"]  # Boolean mask for condition
    weighted_sum = group.loc[mask, "DB090"].sum()  # Sum weights where condition is true
    share = (weighted_sum / total_weight * 100)  # Percentage meeting condition
```

### 2. Data Integration (`1_final_df.py`)

The integration process combines all survey datasets:

- **Harmonization**: Ensures consistent column structure across surveys
- **Filtering**: Removes economic goods indicators (keeps only satisfiers)
- **Aggregation**: Creates "All Countries" median values across EU member states
- **Validation**: Performs data quality checks and coverage analysis

### 3. Normalization Process (Level 4)

#### Winsorization and Percentile Scaling (`2_preprocessing_executed_pca.py`)

The normalization ensures all indicators are comparable and suitable for geometric mean aggregation:

**Step 1: Winsorization**
```python
def winsorize_data(data, limits=(0.05, 0.05)):
    """Apply winsorization to handle outliers"""
    return mstats.winsorize(data, limits=limits, nan_policy='omit')
```

**Step 2: Percentile Scaling**
```python
def percentile_scaling(values, target_min=0.1, target_max=1.0):
    """Scale values to (0.1, 1] range for geometric mean compatibility"""
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    
    # Linear scaling to target range
    scaled = target_min + (values - min_val) * (target_max - target_min) / (max_val - min_val)
    return scaled
```

**Normalization Approaches:**
- **Multi-year**: Uses pooled statistics across all years (reduces temporal variation)
- **Per-year**: Uses year-specific statistics (preserves temporal patterns)
- **Reference-year**: Uses a specific reference year's statistics

#### EHIS Quintile Conversion

EHIS data uses income quintiles instead of deciles. The conversion process:
```python
# Convert quintiles to deciles
# Quintile 1 -> Deciles 1&2, Quintile 2 -> Deciles 3&4, etc.
for quintile_val in [1, 2, 3, 4, 5]:
    decile_1 = (quintile_val * 2) - 1  # Q1->D1, Q2->D3, Q3->D5
    decile_2 = quintile_val * 2        # Q1->D2, Q2->D4, Q3->D6
```

## Aggregation Methodology

### 1. EU Priority Aggregation (Level 2)

#### PCA-Based Weighting (`3_generate_outputs_pca.py`)

The system uses Principal Component Analysis following JRC methodology to compute weights:

**Step 1: Correlation Analysis**
```python
# Check correlation structure
corr_matrix = data.corr()
mean_abs_corr = np.abs(corr_matrix.values).mean()

if mean_abs_corr < 0.1:
    # Use equal weights if correlations are too weak
    return equal_weights
```

**Step 2: Factor Extraction**
```python
# JRC criteria for factor selection:
# 1. Eigenvalues >= 1.0
# 2. Individual variance >= 10%
# 3. Cumulative variance >= 60%

eigenvalue_mask = eigenvalues >= 1.0
individual_variance_mask = explained_variance_ratio >= 0.10
factor_mask = eigenvalue_mask & individual_variance_mask
```

**Step 3: Varimax Rotation** (if multiple factors)
```python
if n_factors > 1:
    rotated_components = varimax_rotation(selected_components.T).T
```

**Step 4: Weight Calculation**
```python
# Calculate weights from squared loadings and factor importance
squared_loadings = rotated_components ** 2
factor_weights = selected_variance_ratios / selected_variance_ratios.sum()

# Final weight = (squared loading on dominant factor) * (factor weight)
for indicator in indicators:
    factor_idx = np.argmax(squared_loadings[:, indicator_idx])
    weight = squared_loadings[factor_idx, indicator_idx] * factor_weights[factor_idx]
```

#### Population-Weighted PCA

For country-level aggregations, the system uses population-weighted covariance matrices:

```python
def weighted_covariance_matrix(data, population_weights):
    """Compute population-weighted covariance matrix"""
    weights = population_weights / population_weights.sum()
    weighted_means = np.average(data, weights=weights, axis=0)
    centered_data = data - weighted_means
    
    # Weighted covariance with Bessel's correction
    bias_correction = weights.sum() / (weights.sum() - (weights**2).sum())
    cov_matrix = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            weighted_cov = np.sum(weights * centered_data[:, i] * centered_data[:, j])
            cov_matrix[i, j] = weighted_cov * bias_correction
    
    return cov_matrix
```

#### Weighted Geometric Mean Aggregation

EU priorities are calculated using weighted geometric means:

```python
def weighted_geometric_mean(values, weights):
    """Compute weighted geometric mean: Π (x_i ^ w_i)"""
    normalized_weights = weights / weights.sum()
    
    if (values <= 0).any():
        return np.nan  # Geometric mean requires positive values
    
    log_result = np.sum(normalized_weights * np.log(values))
    return np.exp(log_result)
```

### 2. EWBI Calculation (Level 1)

The final EWBI score is computed as the geometric mean of all EU priority indicators:

```python
def nan_aware_gmean(values):
    """Geometric mean that handles NaN values properly"""
    clean_values = values.dropna()
    
    if len(clean_values) == 0:
        return np.nan
    
    if (clean_values <= 0).any():
        return np.nan
    
    return gmean(clean_values)  # Using scipy.stats.gmean
```

### 3. Cross-Country Aggregations

#### Individual Countries to "All Countries"

Different aggregation methods are used depending on the level:

**Level 1 & 2 (Policy-relevant levels)**: Population-weighted averages
```python
def population_weighted_average(values, countries, year, population_data):
    """Compute population-weighted average for policy indicators"""
    weights_dict = get_population_weights(countries, year, population_data)
    weights = [weights_dict.get(country, 0.0) for country in countries]
    return weighted_arithmetic_mean(values, weights)
```

**Level 4 & 5 (Technical levels)**: Median across countries
```python
# For primary indicators, use median to reduce influence of outliers
aggregated_value = values.median()
```

#### Decile Aggregations

For each country, "All" decile values are computed as geometric means across individual deciles:

```python
def create_all_deciles(df, level_num):
    """Create 'All' decile aggregations using geometric mean"""
    for group_key, group in df.groupby(['Year', 'Country', 'EU priority']):
        values = group['Value']
        all_deciles_value = nan_aware_gmean(values)  # Geometric mean across deciles
```

## Technical Implementation Details

### Missing Value Handling

The system implements robust missing value handling:

1. **NaN Exclusion**: All aggregation functions exclude NaN values from both numerator and denominator
2. **Minimum Data Requirements**: Requires minimum number of valid observations for meaningful aggregation
3. **Graceful Degradation**: Falls back to simpler methods when advanced techniques fail

### Data Validation

Comprehensive validation ensures data quality:

```python
def perform_data_validation(df):
    """Validate final dataset"""
    validation = {
        'total_rows': len(df),
        'unique_countries': df['country'].nunique(),
        'year_range': f"{df['year'].min()}-{df['year'].max()}",
        'value_stats': {
            'min': df['value'].min(),
            'max': df['value'].max(),
            'mean': df['value'].mean()
        },
        'zero_values': len(df[df['value'] == 0]),
        'missing_values': df.isnull().sum().to_dict()
    }
    return validation
```

### Survey Weight Application

All calculations use appropriate survey weights:

- **Household indicators**: Household weights (DB090)
- **Personal indicators**: Personal weights (RB050)
- **Cross-country aggregations**: Population weights from Eurostat

## Output Structure

The final output includes multiple levels of aggregation:

1. **Individual country-decile combinations** for detailed analysis
2. **Country-level aggregations** ("All" deciles) for national comparisons
3. **EU-level aggregations** ("All Countries") for continental overview
4. **Time series data** for temporal analysis
5. **Master dataset** for latest year comprehensive view

### Column Structure

```python
columns = [
    'Year',           # Year of data
    'Country',        # ISO-2 country code or 'All Countries'
    'Decile',         # Income decile (1-10) or 'All'
    'Quintile',       # Income quintile (1-5) for EHIS, NaN otherwise
    'Level',          # Hierarchical level (1, 2, 4, 5)
    'EU priority',    # EU policy priority name
    'Secondary',      # Secondary indicator (not used in PCA version)
    'Primary and raw data',  # Primary indicator code
    'Type',           # 'Raw', 'Statistical computation', or 'Aggregation'
    'Aggregation',    # Method used for aggregation
    'Value',          # Computed value
    'datasource'      # Original data source
]
```

## Key Design Principles

1. **Non-compensability**: Uses geometric means to prevent high scores in one area from compensating for low scores in another
2. **Robustness**: Handles missing data gracefully and validates all inputs
3. **Transparency**: Maintains clear traceability from raw data to final indicators
4. **Scalability**: Modular design allows for easy addition of new indicators or countries
5. **Reproducibility**: Deterministic calculations with comprehensive logging

## Methodological Innovations

1. **Population-weighted PCA**: Accounts for country size in factor analysis
2. **Multi-year normalization**: Reduces temporal volatility while preserving trends
3. **Hierarchical validation**: Validates data quality at each aggregation level
4. **Flexible income grouping**: Handles both deciles (EU-SILC, HBS, LFS) and quintiles (EHIS)
5. **Robust outlier handling**: Uses winsorization instead of simple truncation

This computation framework ensures that the EWBI provides a reliable, comparable, and policy-relevant measure of well-being across European countries while maintaining methodological rigor and transparency.