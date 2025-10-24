# European Well-Being Index (EWBI) - Computation Logic

This document provides a detailed technical explanation of how the European Well-Being Index (EWBI) is computed based on the implementation in the codebase.

## Important Methodological Notes

### EHIS and HBS Data Exclusion in Current EWBI Implementation

**Critical Note for Scientific Reporting**: The current EWBI implementation uses only **EU-SILC and LFS data sources**. HBS (Household Budget Survey) and EHIS (European Health Interview Survey) data are no longer included in the calculations.

**Rationale for Exclusion**:
1. **Focus on Core Well-being Dimensions**: Concentrating on the most comprehensive and consistent data sources
2. **Methodological Consistency**: Maintaining uniform income stratification (deciles) across all indicators
3. **Data Quality and Coverage**: Ensuring temporal and geographical consistency

This is a significant methodological decision that should be clearly stated in any scientific publication using the EWBI.

## Executive Summary

The European Well-Being Index (EWBI) is a comprehensive, distributional measure of well-being that tracks quality of life across income groups within European countries. Unlike traditional indices that provide only country-level averages, the EWBI reveals how well-being varies across the income distribution, making it a powerful tool for understanding and addressing inequality. The index is constructed from detailed household and individual survey data (EU-SILC and Labor Force Survey) and produces well-being scores for each income decile (poorest 10%, second poorest 10%, etc.) as well as overall country scores.

The methodology follows a rigorous four-level aggregation process designed to ensure comparability and non-compensability. Starting from raw survey responses, indicators are first normalized using robust statistical techniques (winsorization and percentile scaling) to handle outliers while preserving relative rankings. These normalized indicators are then aggregated into policy-relevant dimensions (EU priorities such as health, education, economic security) using population-weighted Principal Component Analysis (PCA) that accounts for both statistical relationships between indicators and country population sizes. Finally, these dimensions are combined into the overall EWBI using geometric means, which prevents high performance in one area from masking poor performance in another.

A key innovation of the EWBI is its consistent maintenance of income stratification throughout the entire computation process. Every indicator, at every level of aggregation, is calculated separately for each income decile and for the overall population. This means researchers and policymakers can examine not just whether a country has high or low well-being overall, but specifically which income groups face the greatest challenges. For example, the EWBI can reveal that while a country's overall well-being appears adequate, the poorest decile experiences significant deprivation in housing quality or healthcare access.

The resulting dataset provides unprecedented granularity for policy analysis and academic research. Users can compare well-being across countries, track changes over time, analyze inequality within countries, and identify which specific dimensions of well-being (health, economic security, education, etc.) drive overall patterns. This makes the EWBI particularly valuable for evidence-based policymaking focused on reducing inequality, targeting interventions to specific income groups, and monitoring progress toward inclusive well-being goals across Europe.

## Overview

The EWBI is a composite indicator that measures well-being across European countries using a hierarchical aggregation structure. The computation involves multiple levels of aggregation from raw survey data to the final EWBI score.

## Hierarchical Structure

The EWBI follows a 4-level hierarchical structure (in the PCA version):

1. **Level 5 (Raw Data)**: Original survey responses from EU-SILC and LFS
2. **Level 4 (Normalized Data)**: Standardized indicators using winsorization and percentile scaling
3. **Level 2 (EU Priorities)**: Aggregated indicators grouped by EU policy priorities using PCA-based weights
4. **Level 1 (EWBI)**: Final composite index aggregating all EU priorities

*Note: Level 3 (Secondary Indicators) is skipped in the PCA version for simplification.*

## Data Sources and Preprocessing

### 1. Raw Data Collection (Level 5)

#### Data Sources:
- **EU-SILC (European Union Statistics on Income and Living Conditions)**: Household and personal indicators
- **LFS (Labor Force Survey)**: Employment and labor market indicators

*Note: HBS (Household Budget Survey) and EHIS (European Health Interview Survey) data are no longer used in the current EWBI implementation.*

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

#### Normalization Approaches

The system supports three normalization strategies to handle temporal variation:

**Multi-year Normalization (Default)**:
```python
NORMALIZATION_APPROACH = 'multi_year'
# Uses pooled statistics across all years for stability
# Reduces temporal variation while preserving trends
```

**Per-year Normalization**:
```python
NORMALIZATION_APPROACH = 'per_year'
# Uses year-specific statistics (original approach)
# Preserves temporal patterns but increases variation
```

**Reference-year Normalization**:
```python
NORMALIZATION_APPROACH = 'reference_year'
REFERENCE_YEAR = 2015
# Uses a specific reference year's statistics for all years
```

#### Income Stratification and Decile-Based Computation

All indicators in the current EWBI implementation use consistent income decile stratification based on equivalized disposable income from EU-SILC data. This ensures methodological consistency across all well-being dimensions.

**Key Principle**: The EWBI computation maintains **dual granularity** at every level:

## Decile-Based Computation Structure

### What is Computed Per Decile vs. Aggregated

The EWBI follows a systematic approach where **every indicator is computed both per individual decile (1-10) AND as country-level aggregations (decile = "All")**:

#### Level 5 (Raw Data): Per-Decile Calculation
```python
# Group by Year, Country, Decile and calculate indicators
group_cols = ["HB010", "HB020", "decile"]  # Year, Country, Decile

for group_keys, group in df.groupby(group_cols):
    # Calculate weighted share for this specific decile
    total_weight = group["DB090"].sum()
    mask = group[f"_valid_{var}"]
    weighted_sum = group.loc[mask, "DB090"].sum()
    share = (weighted_sum / total_weight * 100)
```

**Output**: 10 values per country per year per indicator (one for each income decile)

#### Level 5 (Raw Data): Country-Level Aggregation  
```python
# Also calculate indicators for total population per country (decile = "All")
total_group_cols = ["HB010", "HB020"]  # Year, Country only

for group_keys, group in df.groupby(total_group_cols):
    group_result['decile'] = "All"
    # Calculate across entire population, not stratified by income
```

**Output**: 1 value per country per year per indicator (population-wide average)

#### Level 4 (Normalized Data): Maintains Decile Structure
- **Per-decile normalization**: Each (country, year, decile, indicator) value is normalized using winsorization and percentile scaling
- **1:1 relationship**: Every Level 5 record has a corresponding normalized Level 4 record
- **Preserves stratification**: Decile 1-10 and "All" values are maintained separately

#### Level 2 (EU Priorities): PCA Aggregation Per Decile
```python
# Compute Level 2: PCA-weighted aggregation by (Year, Country, Decile, EU priority)
grouped = level4_filtered.groupby(['Year', 'Country', 'Decile', 'EU priority'])

for (year, country, decile, eu_priority), group in grouped:
    # Apply PCA weights within this specific decile
    level2_value = weighted_geometric_mean(values, pca_weights)
```

**Output**: EU Priority scores for each decile (1-10) AND "All" per country per year

#### Level 1 (EWBI): Final Aggregation Per Decile
```python
# Compute Level 1: Geometric mean of Level 2 indicators per (Year, Country, Decile)
for (year, country, decile), group in level2_df.groupby(['Year', 'Country', 'Decile']):
    ewbi_value = nan_aware_gmean(group['Value'])  # Across EU priorities
```

**Output**: Final EWBI scores for each decile (1-10) AND "All" per country per year

### Cross-Decile Aggregations ("All" Deciles)

For each level, **"All" decile aggregations are computed separately**:

```python
def create_all_deciles(df, level_num):
    """Create 'All' decile aggregations using geometric mean across deciles 1-10"""
    for group_key, group in df.groupby(['Year', 'Country', 'EU priority']):
        values = group['Value']  # Values from deciles 1-10
        all_deciles_value = nan_aware_gmean(values)  # Geometric mean across deciles
```

**Important Distinction**:
- **Decile = "All" (population-wide)**: Computed directly from raw survey data across entire population
- **"All" deciles aggregation**: Geometric mean of individual decile values (1-10)

These can differ because:
- Population-wide calculation uses all survey respondents
- Cross-decile aggregation uses geometric mean of 10 decile-specific values

### Summary: Complete Data Structure

The EWBI produces a comprehensive dataset with the following structure:

**For each combination of (Year, Country, Level, Indicator):**
- **10 individual decile values** (Decile = 1, 2, 3, ..., 10)
- **1 population-wide value** (Decile = "All")  
- **1 cross-decile aggregation** (computed from geometric mean of deciles 1-10)

**Example for Germany 2020, Level 1 (EWBI):**
- `DE, 2020, 1, EWBI, Decile=1`: EWBI score for poorest income decile
- `DE, 2020, 1, EWBI, Decile=2`: EWBI score for second poorest decile
- ...
- `DE, 2020, 1, EWBI, Decile=10`: EWBI score for richest income decile  
- `DE, 2020, 1, EWBI, Decile="All"`: EWBI score for entire German population

This dual structure enables both:
1. **Inequality analysis**: Comparing well-being across income deciles
2. **Country comparisons**: Using population-wide or aggregated measures

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

For country-level aggregations, the system uses population-weighted covariance matrices with sophisticated fallback strategies for missing population data:

```python
def get_population_weights(countries, year, population_data=None):
    """
    Get population weights with automatic fallback for missing data.
    
    Fallback strategy:
    1. Use exact year data if available
    2. For missing countries, use closest available year (prefer past years)
    3. If no historical data, use most recent available year
    4. Log all fallback operations for transparency
    """
    # Implementation handles missing population data gracefully
    # Ensures robust weighting even with incomplete time series
```

This approach ensures reliable population weighting for the PCA methodology even when exact year data is unavailable, maintaining methodological consistency across the time series.

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

#### Data Quality and Missing Value Handling

The system implements comprehensive data quality controls:

1. **Forward Fill Missing Data**: `forward_fill_missing_data()` function handles systematic gaps
2. **NaN Exclusion**: All aggregation functions exclude NaN values from both numerator and denominator  
3. **Minimum Data Requirements**: Requires minimum number of valid observations for meaningful aggregation
4. **Graceful Degradation**: Falls back to simpler methods when advanced techniques fail
5. **Coverage Thresholds**: Sets minimum coverage requirements (e.g., 10%) for reliable indicators

```python
def forward_fill_missing_data(df):
    """Forward fill missing data within country-indicator groups"""
    # Handles systematic data gaps while preserving data integrity
    
def nan_aware_gmean(values):
    """Geometric mean that handles NaN values properly"""
    clean_values = values.dropna()
    if len(clean_values) == 0:
        return np.nan
    return gmean(clean_values)
```

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