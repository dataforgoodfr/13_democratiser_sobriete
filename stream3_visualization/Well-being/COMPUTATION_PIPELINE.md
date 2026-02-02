# EWBI Computation Pipeline: From Raw Data to Dashboard

## Overview

This document describes the complete computational pipeline for the European Well-Being Index (EWBI), from raw data imputation through to dashboard visualization. The pipeline consists of four main stages:

1. **Stage 1**: Imputation of Missing Data (Structural Break Detection)
2. **Stage 3**: Normalization of Data (Forward Fill + Winsorization + Percentile Scaling)
3. **Stage 4**: Weighting and Aggregation (PCA-Weighted Geometric Mean)
4. **Dashboard**: Interactive visualization of multi-level indicators

---

## Data Hierarchy

The EWBI uses a hierarchical structure with 4 levels:

```
Level 1: EWBI Overall (single aggregate per country-year-decile)
         ↑
         Unweighted geometric mean of 6 EU Priorities
         ↑
Level 2: EU Priorities (6 values per country-year-decile)
         • Energy and Housing
         • Equality
         • Health and Animal Welfare
         • Intergenerational Fairness, Youth, Culture and Sport
         • Social Rights and Skills, Quality Jobs and Preparedness
         • Sustainable Transport and Tourism
         ↑
         Population-weighted PCA geometric mean of Level 4 indicators
         ↑
Level 4: Normalized Indicators (winsorized + percentile scaled to [0.1, 1])
         ↑
         Winsorization + Percentile Scaling + Rescaling
         ↑
Level 3: Raw Primary Indicators (break-adjusted, forward-filled, original scale)
```

---

## Stage 1: Imputation of Missing Data (`1_missing_data.py`)

### Overview
This stage handles structural breaks in time series data and prepares raw Level 3 (Primary Indicators) data for downstream processing.

### Data Input
- **EU-SILC Data**: Household and personal indicators from EU Statistics on Income and Living Conditions
- **LFS Data**: Labor Force Survey indicators
- **Source Location**: `0_raw_data_EUROSTAT/0_EU-SILC/3_final_merged_df/` and `0_raw_data_EUROSTAT/0_LFS/`

### Data Filtering
Before structural break detection, the pipeline applies several filters:

```
Raw Data → Filter out excluded indicators (HH-SILC-1, AC-SILC-1, AN-SILC-1)
         → Filter out NaN indicators
         → Filter for Country ≠ 'EU-27' (keep individual countries only)
         → Filter for Decile ≠ 'All' (keep individual deciles 1-10)
         → Keep only non-missing decile and country values
```

### Structural Break Detection and Adjustment

**Problem**: Time series may contain structural breaks due to survey methodology changes.

**Detection Algorithm**:

For each time series (defined by: indicator, country, decile combination):

1. Calculate year-to-year percentage changes:
$$\Delta_t = \left| \frac{v_t - v_{t-1}}{v_{t-1}} \right|$$

2. Use fixed threshold of **30% absolute change** to detect breaks:
$$\text{Break detected if } \Delta_t > 0.30$$

3. **Break Adjustment by Position**:

#### Case 1: Break at START (t = 1)
When break occurs in first year, use post-break growth rate (skipping the problematic year 1-2 transition) to correct:

$$v_0^{\text{corr}} = \frac{v_1}{g_{2 \to 3}}$$

where $g_{2 \to 3} = \frac{v_3}{v_2}$ is the growth rate from year 2 to 3 (the reliable post-break growth rate).

#### Case 2: Break in MIDDLE (1 < t < T)
When break occurs in middle of series:

**Step 1**: Correct the break point using pre-break growth rate:
$$v_t^{\text{corr}} = v_{t-1} \times g_{\text{pre}} = v_{t-1} \times \frac{v_{t-1}}{v_{t-2}}$$

**Step 2**: Compute the adjustment ratio between corrected and problematic value:
$$\text{adjustment\_ratio} = \frac{v_t^{\text{corr}}}{v_t}$$

**Step 3**: Rebase all future values using this constant ratio (preserves true growth rates forward):
$$v_j^{\text{adj}} = v_j \times \text{adjustment\_ratio}, \quad \forall j > t$$

#### Case 3: Break at END (t = T)
When break occurs in last year:

$$v_T^{\text{corr}} = v_{T-1} \times g_{\text{pre}} = v_{T-1} \times \frac{v_{T-1}}{v_{T-2}}$$

#### Case 4: Only 2 Data Points
If series has only 2 points with break at both positions, no adjustment applied.

**Mean Preservation**:
After break adjustments, apply proportional correction to preserve original mean:

$$\text{scale\_factor} = \frac{\text{original\_mean}}{\text{adjusted\_mean}}$$

$$v^{\text{final}} = v^{\text{adjusted}} \times \text{scale\_factor}$$

### Data Structure at Stage 1 Output

**DataFrame Format**:
```
Year | Country | Decile | Level | EU priority | Primary and raw data | Value
2010 |   AT    |   1    |   3   |    NA       |     IS-SILC-1       | 45.2
2010 |   AT    |   2    |   3   |    NA       |     IS-SILC-1       | 52.1
...
```

**Key Characteristics**:
- **Level**: 3 (Raw Primary Indicators)
- **Decile**: Individual deciles (1-10), not aggregated
- **Country**: Individual countries (AT, BE, etc.), not EU-27
- **Value**: Break-adjusted raw values (NOT normalized, NOT forward-filled)
- **Aggregation field**: NA (raw data, no aggregation applied yet)

### Output
- **File**: `1_missing_data_output/raw_data_break_adjusted.csv`
- **Records**: Individual country-decile-year-indicator combinations
- **Note**: Forward fill is applied in Stage 3, not Stage 1

---

## Stage 3: Normalization of Data (`3_normalisation_data.py`)

### Overview
This stage normalizes break-adjusted raw data to create comparable Level 4 indicators for aggregation. The goal is to rescale diverse indicators (with different units and scales) to a common range while preserving their relative relationships.

### Data Input
- **Source**: Break-adjusted raw data from Stage 1 (`1_missing_data_output/raw_data_break_adjusted.csv`)

### Pre-processing: Forward Fill

Before normalization, apply forward fill to complete missing years:

**Algorithm**:
For each series (indicator, country, decile):
1. Identify first year with available data: $t_{\min}$
2. For each missing year $t$ in range $[t_{\min}, t_{\max}]$:
   - If data available in year $t-1$, use: $v_t = v_{t-1}$
   - Only fill forward from first available data point
   - Preserve NaN values before $t_{\min}$

**Output**: Forward-filled raw data saved to `1_missing_data_output/raw_data_forward_filled.csv` for use by Stage 4.

### Normalization: Winsorization + Percentile Scaling + Rescaling

**Stage 3a: Winsorization**

Winsorization clips extreme outliers at the 1st and 99th percentiles:

$$\text{lower\_bound} = P_{1\%}(\text{values})$$
$$\text{upper\_bound} = P_{99\%}(\text{values})$$

$$v^{\text{winsorized}} = \text{clip}(v, \text{lower\_bound}, \text{upper\_bound})$$

**Stage 3b: Percentile Scaling (Empirical CDF)**

Apply empirical cumulative distribution function transformation:

$$\text{rank}(v) = \text{position of } v \text{ in sorted array}$$

$$P(v) = \frac{\text{rank}(v) - 1}{n - 1}$$

This transforms values to the range $[0, 1]$.

**Stage 3c: Rescaling to [0.1, 1] with Inversion**

Final rescaling to ensure:
- All values are strictly positive (required for geometric mean)
- Inverted scale so "higher is better" consistently
- Minimum floor of 0.1 (to prevent log(0) issues)

$$v^{\text{normalized}} = 0.1 + 0.9 \times (1 - P(v))$$

**Complete Formula**:
$$v^{\text{normalized}} = 0.1 + 0.9 \times \left(1 - \frac{\text{rank}(v_i^{\text{win}}) - 1}{n - 1}\right)$$

**Application Scope**:
- **Pooling method**: Multi-year pooled normalization (most common)
  - Compute percentiles across ALL years and ALL deciles simultaneously
  - Provides temporal stability and comparable scales across years

- **Data filtered**:
  ```
  Winsorized Data → Filter Country ≠ 'EU-27'
                 → Filter Decile ≠ 'All'
                 → Keep individual countries and deciles only
  ```

### 1:1 Mapping Between Raw and Normalized Data

Each raw data record from Stage 1 (after forward fill) produces exactly one normalized record:

$$n_{\text{normalized}} = n_{\text{raw}}$$

### Data Structure at Stage 3 Output

**DataFrame Format**:
```
Year | Country | Decile | Level | Primary and raw data | Value | Aggregation
2010 |   AT    |   1    |   4   |   IS-SILC-1         | 0.75  | Winsorization + Percentile Scaling
2010 |   AT    |   2    |   4   |   IS-SILC-1         | 0.62  | Winsorization + Percentile Scaling
...
```

**Key Characteristics**:
- **Level**: 4 (Normalized Indicators)
- **Decile**: Individual deciles (1-10)
- **Country**: Individual countries (no EU-27)
- **Value range**: $[0.1, 1]$ (strictly positive for geometric mean)
- **Aggregation**: "Winsorization + Percentile Scaling"

### Output
- **Forward-filled raw data**: `1_missing_data_output/raw_data_forward_filled.csv`
- **Normalized Level 4 data**: `3_normalisation_data_output/level4_normalised_indicators.csv`
- **Records**: 1:1 with forward-filled raw data (individual country-decile-year-indicator)

---

## Stage 4: Weighting and Aggregation (`4_weighting_aggregation.py`)

### Overview
This stage aggregates normalized Level 4 indicators through a hierarchical structure using **PCA-weighted geometric means** to create EU Priorities (Level 2) and overall EWBI (Level 1).

### Data Inputs
1. **Normalized Level 4 data**: From Stage 3 (`3_normalisation_data_output/level4_normalised_indicators.csv`)
2. **PCA analysis results**: From Stage 2 (component weights, eigenvalues) - optional
3. **Population data**: For country-level weighting
4. **Raw forward-filled data**: From Stage 3 (`1_missing_data_output/raw_data_forward_filled.csv`) - included as Level 3 in final output

### Level 2: EU Priorities (PCA-Weighted Geometric Mean)

**Aggregation Formula**:

For each country $c$, year $y$, decile $d$, and EU Priority $p$:

$$I_{c,y,d}^{(p)} = \exp\left(\sum_{i \in I_p} w_i^{(c,y)} \ln(x_{c,y,d,i})\right)$$

where:
- $I_p$ = set of normalized indicators belonging to EU Priority $p$
- $x_{c,y,d,i}$ = normalized value for indicator $i$ (from Level 4)
- $w_i^{(c,y)}$ = PCA-based weight for indicator $i$ in country $c$, year $y$
- $\sum_i w_i^{(c,y)} = 1$ (normalized weights)

**PCA Weights Calculation**:

The component weights are derived from eigenvalue-based explained variance:

$$w_i = \sum_m \lambda_m \times \text{loading}_{i,m}^2$$

where:
- $\lambda_m$ = eigenvalue of component $m$ (proportion of variance explained)
- $\text{loading}_{i,m}$ = loading of indicator $i$ on component $m$

**Special Cases**:
- If PCA weights unavailable: Fall back to unweighted geometric mean
- If fewer than 2 valid indicators: Skip aggregation

### Level 1: EWBI Overall (Unweighted Geometric Mean)

**Aggregation Formula**:

For each country $c$, year $y$, decile $d$:

$$\text{EWBI}_{c,y,d} = \left(\prod_{p=1}^{6} I_{c,y,d}^{(p)}\right)^{1/6}$$

This is the geometric mean of the 6 EU Priority values (unweighted):

$$\text{EWBI}_{c,y,d} = \exp\left(\frac{1}{6} \sum_{p=1}^{6} \ln(I_{c,y,d}^{(p)})\right)$$

**Rationale**: No compensation between dimensions; all EU Priorities equally important for overall well-being.

### Country-Level Aggregations (Decile='All Deciles')

After computing decile-specific values, aggregate across all 10 deciles to get country-level (across-decile) aggregations:

$$I_{c,y}^{\text{(country)}} = \exp\left(\frac{1}{10} \sum_{d=1}^{10} \ln(I_{c,y,d})\right)$$

**Output**: New records with `Decile='All Deciles'` added to dataset.

### EU-27 Aggregations

Aggregate individual countries to EU-27 using population weighting:

**For Levels 1 and 2 (Normalized data)**: Use **population-weighted geometric mean**:
$$I_{\text{EU-27},y,d}^{(p)} = \exp\left(\sum_{c} w_c \ln(I_{c,y,d}^{(p)})\right)$$

**For Level 3 (Raw indicators)**: Use **population-weighted arithmetic mean**:
$$I_{\text{EU-27},y,d}^{\text{(raw)}} = \sum_{c} w_c \times I_{c,y,d}^{\text{(raw)}}$$

where:
- $w_c = \frac{\text{population}_{c,y}}{\sum_c \text{population}_{c,y}}$ (population weight)
- Weights normalized to sum to 1

**Rationale for Arithmetic Mean at Level 3**: Raw indicators can have legitimate zero values (e.g., 0% overcrowded dwellings in some deciles). Geometric mean fails with zeros ($\ln(0) = -\infty$), so arithmetic mean is used for raw data aggregation.

**Two-Pass Approach**:

**Pass 1**: Create EU-27 aggregates for each decile $d \in [1,10]$
**Pass 2**: Create 'All Deciles' aggregates for EU-27 by averaging across deciles

### Level 3: Raw Primary Indicators (Loaded in Final Output)

Raw forward-filled data from Stage 3 is included as Level 3 in the final unified output:

**Data Characteristics**:
- **Values**: Break-adjusted and forward-filled but NOT normalized (original scale)
- **Aggregation**: "Break-adjusted and forward-filled (raw data)"
- **Type**: "Primary indicator"

### Final Unified Output Structure

**DataFrame Columns**:
```
Year | Country | Decile | Level | EU priority | Secondary | Primary and raw data | Type | Aggregation | Value
```

**Decile Values**:
- `1, 2, ..., 10`: Individual income deciles
- `'All Deciles'`: Aggregated across all deciles

**Country Values**:
- `AT, BE, BG, ...`: Individual countries (ISO-2 codes)
- `EU-27`: EU-27 aggregate (population-weighted)

**Level Values**:
- `1`: EWBI Overall
- `2`: EU Priorities
- `3`: Raw Primary Indicators

**Aggregation Field Values**:
- `'Population-weighted geometric mean'`: EU-27 aggregations for Levels 1-2
- `'Population-weighted arithmetic mean'`: EU-27 aggregations for Level 3 (raw indicators)
- `'Geometric mean of Level 2 EU Priorities'`: Level 1 EWBI values
- `'Population-weighted PCA geometric mean of Level 4 indicators'`: Level 2 EU Priorities
- `'Geometric mean across deciles for Level 3 (Raw Indicators)'`: Country aggregations for raw data
- `'Break-adjusted and forward-filled (raw data)'`: Raw Level 3 individual deciles

### Output
- **File**: `4_weighting_aggregation_output/ewbi_final_aggregated.csv`
- **App-ready copy**: `output/ewbi_master_aggregated.csv`

---

## Dashboard Visualization (`app.py`)

### Overview
The Dash/Plotly-based dashboard provides interactive exploration of the multi-level EWBI data.

### Data Input
- **Source**: `ewbi_master_aggregated.csv` (final unified output from Stage 4)

### Filtering for Dashboard Display

**Level Selection**:

- **Level 1 (EWBI Overall)**: Shows overall well-being index
- **Level 2 (EU Priority)**: Shows specific EU priority scores
- **Level 3 (Raw Indicator)**: Shows raw indicator values

### Charts

1. **European Map**: Choropleth map showing latest year values by country
2. **Time Series**: Line plot showing evolution over time for selected countries
3. **Decile Analysis**: Bar chart comparing values across income deciles
4. **Country Comparison**: Horizontal bar chart ranking countries

### EU-27 Display Logic

For EU-27 data in charts:
- **Levels 1-2**: Filter by `Aggregation == 'Population-weighted geometric mean'`
- **Level 3**: Filter by `Aggregation == 'Population-weighted arithmetic mean'`

---

## Summary of Formulas

### Core Aggregation Formulas

| Stage | Operation | Formula | Application |
|-------|-----------|---------|-------------|
| 1 | Structural Break Correction | $v^{\text{corr}} = v_{t-1} \times \frac{v_{t-1}}{v_{t-2}}$ | Break at middle of series |
| 1 | Mean Preservation | $v^{\text{final}} = v^{\text{adj}} \times \frac{\bar{v}_{\text{orig}}}{\bar{v}_{\text{adj}}}$ | After all break corrections |
| 3 | Winsorization | $v^{\text{win}} = \text{clip}(v, P_1, P_{99})$ | Clip to 1st-99th percentiles |
| 3 | Percentile Scaling | $P(v) = \frac{\text{rank}(v) - 1}{n - 1}$ | Empirical CDF |
| 3 | Rescaling | $v^{\text{norm}} = 0.1 + 0.9(1 - P(v))$ | Rescale to [0.1, 1] |
| 4 | PCA-Weighted GM | $I_p = \exp(\sum_i w_i \ln x_i)$ | EU Priority aggregation (Level 2) |
| 4 | Unweighted GM | $I = (\prod_i x_i)^{1/n}$ | EWBI aggregation (Level 1) |
| 4 | Pop-Weighted GM | $I = \exp(\sum_c w_c \ln x_c)$ | EU-27 aggregation (Levels 1-2) |
| 4 | Pop-Weighted AM | $I = \sum_c w_c \times x_c$ | EU-27 aggregation (Level 3 raw) |

### Key Parameters

| Parameter | Value | Context |
|-----------|-------|---------|
| Break detection threshold | 30% | Absolute year-to-year change |
| Winsorization lower | 1st percentile | Remove extreme lows |
| Winsorization upper | 99th percentile | Remove extreme highs |
| Rescaling range | [0.1, 1] | Positive values for log operations |
| GM minimum indicators | 2 | Minimum for EWBI computation |
| Pooling method | Multi-year | Normalization approach |

---

## File Structure Summary

### Input Files
- `0_raw_data_EUROSTAT/0_EU-SILC/`: EU-SILC raw data
- `0_raw_data_EUROSTAT/0_LFS/`: LFS raw data

### Intermediate Files
- `1_missing_data_output/raw_data_break_adjusted.csv`: Stage 1 output (break-adjusted)
- `1_missing_data_output/raw_data_forward_filled.csv`: Forward-filled raw data (created in Stage 3)
- `3_normalisation_data_output/level4_normalised_indicators.csv`: Stage 3 output (normalized Level 4)

### Final Output
- `4_weighting_aggregation_output/ewbi_final_aggregated.csv`: Full aggregated data
- `ewbi_master_aggregated.csv`: App-ready copy

---

## References

- **Methodology Source**: EU JRC (Joint Research Centre) guidelines for composite indicator construction
- **Data Sources**:
  - EU-SILC: European Union Statistics on Income and Living Conditions
  - LFS: Labour Force Survey
  - Population Data: Eurostat
- **Normalization**: Min-Max scaling via percentile transformation
- **Aggregation**: Population-weighted geometric mean (arithmetic mean for raw indicators with zeros)
