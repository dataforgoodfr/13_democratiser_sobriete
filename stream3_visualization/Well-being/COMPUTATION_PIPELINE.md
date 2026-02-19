# EWBI Computation Pipeline: From Raw Data to Dashboard

## Summary
The European Well-Being Index (EWBI) is a comprehensive, distributional measure of well-being that tracks quality of life across income groups within European countries. Unlike indices that only report national averages, the EWBI is computed separately for each income decile (poorest 10%, …, richest 10%) and also provides aggregate country and EU-27 values. It is built from harmonized microdata from EU-SILC (income, living conditions, housing, access to services, health and education-related items) and the Labour Force Survey (job quality and working-conditions indicators), covering multiple years and all EU Member States.

The EWBI computation follows a staged pipeline from raw survey variables to final composite indices. First, households are ranked within each country-year using equivalized disposable income (OECD modified equivalence scale) and assigned to income deciles using weighted quantiles. Primary indicators are then computed as weighted prevalence rates (percentages) for each country-year-decile. To ensure time-series consistency, the pipeline treats missingness and methodological discontinuities: it interpolates internal gaps (without extrapolating beyond observed endpoints) and detects/adjusts structural breaks using a threshold on year-to-year relative changes, optionally followed by smoothing. After that, indicators are forward-filled to complete the time series within the observed range.

Next, the pipeline produces a normalized indicator layer that is designed for aggregation with geometric means: values are winsorized to limit outlier influence, standardized (baseline: z-scores pooled over the full sample), and rescaled into a strictly positive interval (baseline: [0.1, 1.0]) with an explicit inversion so that higher values consistently represent worse outcomes (deprivation). An alternative normalization exists (percentile scaling), but the baseline method is z-score-based rescaling.

Aggregation is hierarchical and policy-facing. The EWBI groups 37 indicators into five EU Priorities (Energy and Housing; Equality; Health and Animal Welfare; Intergenerational Fairness, Youth, Culture and Sport; Social Rights and Skills, Quality Jobs and Preparedness). Within each priority, composites are constructed using PCA-based weights (principal components with explained-variance weighting, using rotated loadings), producing Level 2 priority scores per country-year-decile; Stage 4 can additionally apply an optional post-aggregation structural break adjustment on these Level 2 time series to reduce spurious jumps before computing the overall index. Finally, the Level 1 EWBI is computed as the unweighted geometric mean of the five priority scores, limiting compensation so that strong performance in one dimension cannot fully mask poor outcomes in another.

Importantly, the income stratification is maintained throughout: every transformation and aggregation step is performed at the country-year-decile level, and then (where needed) combined into “All Deciles” country values and EU-27 aggregates using explicit aggregation rules (baseline includes geometric aggregation across deciles and population-weighted arithmetic aggregation across countries for EU-27). The result is a dataset that supports both distributional and temporal analysis—enabling users to compare countries, track trends, quantify within-country inequality, and identify which policy dimensions and underlying indicators drive overall well-being patterns across the European income distribution.

## Overview

This document describes the complete computational pipeline for the European Well-Being Index (EWBI) from raw data collection to dashboard implementation. The pipeline consists of six main stages plus sensitivity analysis, following OECD (2008): "Handbook on Constructing Composite Indicators: Methodology and User Guide":

0. **Stage 0**: Data Collection and Indicator Selection
1. **Stage 1**: Missing Data Treatment (Interpolation and Structural Break Adjustment)
2. **Stage 2**: Multivariate Analysis (PCA Weighting Calculation)
3. **Stage 3**: Data Normalization (Winsorization and Percentile Scaling)
4. **Stage 4**: Weighting and Aggregation (Hierarchical Index Construction)
5. **Stage 5**: Sensitivity Analysis (Monte Carlo Testing)

---

## Stage 0: Data Collection and Indicator Selection (`0_raw_indicator_*.py`)

### Overview
This stage collects raw data from multiple sources and processes them into standardized indicator format. The stage extracts specific EWBI indicators from EU-SILC and LFS datasets, applies income decile calculations, and creates final merged datasets.

### Data Sources

**EU-SILC (EU Statistics on Income and Living Conditions)**
- **Household Data (H-files)**: Housing conditions, income, household composition
- **Personal Data (P-files)**: Individual characteristics, employment, education
- **Register Data (D/R-files)**: Sampling weights and demographic information
- **Time Coverage**: 2004-2023
- **Geographic Coverage**: All EU Member States

**LFS (Labour Force Survey)**
- **Individual Data**: Employment status, working conditions, job characteristics
- **Time Coverage**: 1983-2023
- **Geographic Coverage**: All EU Member States

### Complete EWBI Indicator Set

The EWBI uses 37 indicators grouped into 5 EU Priorities:

#### 1. Energy and Housing
- **HQ-SILC-1**: Overcrowded dwelling
- **HQ-SILC-2**: Cannot keep dwelling warm (HC060)
- **HQ-SILC-3**: Cannot keep dwelling cool (HC070)
- **HQ-SILC-4**: Dwelling too dark (HS160)
- **HQ-SILC-5**: Noise from street (HS170)
- **HQ-SILC-6**: Leaking roof/damp/rot (HH040)
- **HQ-SILC-7**: Pollution or crime (HS180)
- **HQ-SILC-8**: No renovation measures (HC003)
- **HE-SILC-2**: Behind on utility bills (HS021)

#### 2. Equality
- **ES-SILC-1**: Unable to handle unexpected costs (HS060)
- **ES-SILC-2**: Hard to make ends meet (HS120)
- **EC-SILC-2**: Low trust in others (PW191)
- **EC-SILC-3**: Cannot meet friends/family monthly (PD050)
- **EC-SILC-4**: Persons living alone

#### 3. Health and Animal Welfare
- **AH-SILC-2**: Poor self-rated health (PH020)
- **AH-SILC-3**: Limited by health problems (PH030)
- **AH-SILC-4**: Unable to work due to illness (PL086)
- **AC-SILC-3**: Unmet need for medical care (PH060)
- **AC-SILC-4**: Unmet need for dental care (PH040)

#### 4. Intergenerational Fairness, Youth, Culture and Sport
- **IS-SILC-3**: No formal education (PE041)
- **IS-SILC-4**: Not participating in training (PE010)
- **IS-SILC-5**: No secondary education (PE041)

#### 5. Social Rights and Skills, Quality Jobs and Preparedness
- **RT-SILC-1**: Adults on fixed-term contracts (PL141)
- **RT-SILC-2**: Adults working part-time (PL145)
- **RT-LFS-1**: Working multiple jobs (NUMJOB)
- **RT-LFS-2**: Wanting to work more hours (WISHMORE)
- **RT-LFS-3**: Doing overtime/extra hours (EXTRAHRS)
- **RT-LFS-4**: No freedom on working time (VARITIME)
- **RT-LFS-5**: Shift work (SHIFTWK)
- **RT-LFS-6**: Night work (NIGHTWK)
- **RT-LFS-7**: Saturday work (SATWK)
- **RT-LFS-8**: Sunday work (SUNWK)
- **RU-SILC-1**: Unemployed for over 6 months (PL080)

### Income Decile Calculation

**OECD Modified Equivalence Scale**:
For household $h$ with members aged $a_i$:

$$w_h = 1.0 + \sum_{i=2}^{n_h} w_i$$

where:
$$w_i = \begin{cases} 
0.5 & \text{if } a_i \geq 14 \\
0.3 & \text{if } a_i < 14
\end{cases}$$

**Equivalized Disposable Income**:
$$I_{h,eq} = \frac{HY020_h}{w_h}$$

**Income Deciles**: For each country $c$ and year $t$, households are ranked by $I_{h,eq}$ and assigned to deciles $d \in \{1,2,...,10\}$ using weighted quantiles:

$$D_{c,t,h} = \arg\min_d \left\{ \sum_{h': I_{h',eq} \leq I_{h,eq}} RB050_{h'} \geq \frac{d}{10} \sum_{h''} RB050_{h''} \right\}$$

### Data Processing Workflow

**Step 1: Data Extraction**
- Load EU-SILC files (H, P, D, R) by country and year
- Load LFS yearly files for all countries
- Extract required columns for each indicator

**Step 2: Data Merging**
- Link household and personal data using household ID (HB030/PB030)
- Merge with register data for sampling weights (DB090/RB050)
- Create unified person-household dataset

**Step 3: Indicator Calculation**

For binary indicators (most common):
$$I_{c,t,d,i} = \frac{\sum_{h \in (c,t,d)} \mathbf{1}_{X_{h,i} = \text{condition}} \times W_h}{\sum_{h \in (c,t,d)} W_h} \times 100$$

where:
- $c$ = country, $t$ = year, $d$ = decile, $i$ = indicator
- $X_{h,i}$ = raw variable value for household $h$
- $W_h$ = household weight or person weight as appropriate
- condition = specific value triggering the indicator

**Special Cases**:
- **Population weights** (sum of RB050): Used for person-based indicators (living alone, overcrowding)
- **Household weights** (DB090): Used for household-based indicators (housing quality)

### Data Structure Output

**DataFrame Format**:
```
Year | Country | Decile | Level | EU priority | Primary and raw data | Value | Type | Aggregation
2010 |    AT   |    1   |   3   |      -      |     HQ-SILC-1       |  12.3 | Primary | Raw percentage
2010 |    AT   |    2   |   3   |      -      |     HQ-SILC-1       |   8.7 | Primary | Raw percentage
...
```

**Key Characteristics**:
- **Level**: 3 (Primary/Raw Indicators)
- **Values**: Percentages (0-100) representing prevalence rates
- **Coverage**: Individual countries (AT, BE, ...) and deciles (1-10)
- **Missing**: NaN for unavailable country-year-decile combinations

### Output Files
- **EU-SILC**: `0_raw_data_EUROSTAT/0_EU-SILC/final_indicators.csv`
- **LFS**: `0_raw_data_EUROSTAT/0_LFS/final_indicators.csv`
- **Combined**: `0_raw_data_EUROSTAT/1_final_df/combined_raw_indicators.csv`

---

## Stage 1: Missing Data Treatment (`1_missing_data.py`)

### Overview
This stage handles temporal gaps and methodological discontinuities in the raw indicator time series. It applies linear interpolation to fill missing years and detects/adjusts structural breaks caused by survey methodology changes.

### Process Flow

**Input**: Raw indicator data from Stage 0 (break-unadjusted)
**Output**: Break-adjusted data with consistent time series

### Data Filtering
Before processing, several filters are applied:
$$D_{filtered} = \{(c,t,d,i,v) \in D_{raw} : c \neq \text{'EU-27'} \wedge d \neq \text{'All'} \wedge v \notin \{\text{NaN}, \text{excluded indicators}\}\}$$

Excluded indicators: HH-SILC-1, AC-SILC-1, AN-SILC-1 (data quality issues)

### Step 1: Linear Interpolation

For each time series $(i,c,d)$ with years $\{t_{min}, ..., t_{max}\}$:

**Complete Year Range**: Create complete timeline from $t_{min}$ to $t_{max}$
$$T_{complete} = \{t_{min}, t_{min}+1, ..., t_{max}\}$$

**Interior Interpolation**: For missing years $t_k$ where $t_{min} < t_k < t_{max}$:
$$v_{t_k} = v_{t_{k-1}} + \frac{v_{t_{k+1}} - v_{t_{k-1}}}{t_{k+1} - t_{k-1}} \times (t_k - t_{k-1})$$

**Boundary Preservation**: No extrapolation beyond first/last data points

### Step 2: Structural Break Detection

**Break Detection Threshold**: Default threshold $\tau = 0.2$ (20% absolute change)

For consecutive years $t$ and $t+1$:
$$\text{Break}_{t \to t+1} = \left|\frac{v_{t+1} - v_t}{v_t}\right| > \tau$$

### Step 3: Break Adjustment Algorithm

**Summary**: the implementation in `code/1_missing_data.py` does *not* only correct a single point and then propagate. Instead, it (i) detects break years, (ii) builds a full set of *year-to-year growth rates* where break intervals get “smoothed” growth rates computed from the non-break periods just outside the break region (including consecutive breaks), and (iii) reconstructs the whole series multiplicatively around a chosen reference year (a “variation database” approach).

**Process direction**: the reconstruction is anchored on a reference year chosen at the end of the series (typically the *last* year; if the last interval is itself a detected break, the reference is the *penultimate* year). Values are then reconstructed **future-to-past** (backward) from that anchor; a **small forward reconstruction** only occurs in the “break at the end” situation because the anchor is no longer the final year.

For a series $(i,c,d)$ with valid (positive) values $v_t$ at years $t_0 < \dots < t_n$:

1) **Detect breaks** on adjacent-year relative changes:
$$\text{Break}_{t_j \to t_{j+1}} = \left|\frac{v_{t_{j+1}} - v_{t_j}}{v_{t_j}}\right| > \tau$$

2) **Group consecutive breaks into runs** (e.g. breaks at $t_j\to t_{j+1}$ and $t_{j+1}\to t_{j+2}$ are treated as one break run).

3) **Compute growth rates** $g_j$ for each interval $t_j\to t_{j+1}$:
- If the interval is *not* a break: $g_j = \frac{v_{t_{j+1}}}{v_{t_j}}$.
- If the interval lies inside a break run $[j_{start}..j_{end}]$, the algorithm uses growth information *just outside the run*:
  - $g_{pre} = \frac{v_{t_{j_{start}}}}{v_{t_{j_{start}-1}}}$ (growth immediately before the run, if available)
  - $g_{post} = \frac{v_{t_{j_{end}+2}}}{v_{t_{j_{end}+1}}}$ (growth immediately after the run, if available)
  - Then sets the break-interval growth rate using edge-aware rules:
    - start-of-series run: use $g_{post}$
    - end-of-series run: use $g_{pre}$
    - otherwise: use the mean $g_j = \frac{g_{pre}+g_{post}}{2}$ when both exist
    - fallback: if only one exists, use the one available

#### Case 4 (the “extra” case): insufficient surrounding data
If neither $g_{pre}$ nor $g_{post}$ can be computed (e.g., very short series or a break run with no usable outside-run years), the implementation falls back to $g_j = 1.0$ (no growth) for that break interval.

4) **Choose a reference year** (typically the last year; if the last interval is a detected break, it uses the year before the last as reference), set $\text{var}(t_{ref})=1$, and reconstruct a multiplicative variation path:
$$\text{var}(t_{j}) = \frac{\text{var}(t_{j+1})}{g_j} \quad (\text{backward})$$
$$\text{var}(t_{j+1}) = \text{var}(t_{j}) \cdot g_j \quad (\text{forward})$$

5) **Reconstruct the adjusted series**:
$$v^{adj}_{t_j} = v_{t_{ref}} \cdot \text{var}(t_j)$$

This “variation database reconstruction” is what aligns the full time series with smoothed growth rates around break regions (including consecutive-break runs), rather than only adjusting a single year and scaling earlier values.

### Step 4: Optional Smoothing (Baseline: Enabled)

**5-Year Centered Moving Average**: If enabled:
$$v_t^{smooth} = \frac{1}{5} \sum_{k=-2}^{2} v_{t+k}^{adj}$$

**Edge Handling**: Use available values at series boundaries

### Step 5: Mean Preservation (Baseline: Disabled)

If enabled, apply proportional rescaling to preserve original series mean:
$$\text{scale} = \frac{\bar{v}_{original}}{\bar{v}_{adjusted}}$$
$$v_t^{final} = v_t^{adjusted} \times \text{scale}$$

### Configuration Parameters (Baseline vs. Variants)

| Parameter | Baseline | Alternative 1 | Alternative 2 | Alternative 3 |
|-----------|----------|---------------|---------------|---------------|
| Break Threshold | 0.2 (20%) | 0.1 (10%) | 0.3 (30%) | - |
| Moving Average | True | False | - | - |
| Window Size | 5 years | 3 years | - | - |
| Mean Rescaling | False | True | - | - |

### Data Structure Output

**DataFrame Format**: Same as input but with adjusted values
```
Year | Country | Decile | Primary and raw data | Value | Processing_stage
2010 |    AT   |    1   |     HQ-SILC-1       |  12.8 | break_adjusted
2010 |    AT   |    2   |     HQ-SILC-1       |   8.9 | break_adjusted
...
```

### Output Files
- **Main**: `1_missing_data_output/raw_data_break_adjusted.csv`
- **Interpolated only**: Used internally for forward fill in Stage 3

## Stage 2: Multivariate Analysis (`2_multivariate_analysis.py`)

### Overview
This stage performs Principal Component Analysis (PCA) on normalized indicators to determine optimal weighting schemes for EU Priority aggregation. It analyzes correlation structures and computes factor loadings used in Stage 4.

### Process Flow

**Input**: Break-adjusted data from Stage 1
**Output**: PCA weights and component analysis results

### Data Preparation

**Indicator Standardization**: Before PCA, each indicator is normalized per country-year:
$$z_{c,t,d,i} = \frac{x_{c,t,d,i} - \bar{x}_{c,t,i}}{\sigma_{c,t,i}}$$

where $\bar{x}_{c,t,i}$ and $\sigma_{c,t,i}$ are computed across all deciles for country $c$, year $t$, indicator $i$.

### PCA Analysis by EU Priority

For each EU Priority $p$ with indicators $\{i_1, i_2, ..., i_{n_p}\}$:

**Step 1: Correlation Matrix**
$$R_{i,j} = \text{corr}(z_i, z_j)$$

**Step 2: Eigenvalue Decomposition**
$$R = V \Lambda V^T$$

where $\Lambda = \text{diag}(\lambda_1, \lambda_2, ..., \lambda_{n_p})$ with $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_{n_p}$

**Step 3: Factor Selection Criteria**
- Kaiser criterion: $\lambda_k > 1.0$
- Variance threshold: Individual variance explained $> 10\%$
- Cumulative variance: $\sum_{k=1}^{m} \lambda_k / \sum_{k=1}^{n_p} \lambda_k \geq 0.75$

**Step 4: Varimax Rotation**
Apply orthogonal rotation to simplify factor structure:
$$F_{rotated} = F \times R_{varimax}$$

**Step 5: Component Weights**
Weights based on squared rotated factor loadings:
$$w_i = \frac{\sum_{k=1}^{m} \lambda_k \times f_{i,k}^2}{\sum_{j=1}^{n_p} \sum_{k=1}^{m} \lambda_k \times f_{j,k}^2}$$

where $f_{i,k}$ is the rotated loading of indicator $i$ on component $k$.

### Global PCA Analysis

**Cross-Priority Analysis**: PCA across ALL indicators simultaneously to understand global correlation structure:
- Identifies cross-priority relationships
- Validates EU Priority groupings
- Provides alternative weighting schemes

### Output Files
- **PCA Results**: `2_multivariate_analysis_output/pca_results_full.json`
- **Correlation matrices**: `2_multivariate_analysis_output/correlation_matrices.json`
- **Factor loadings**: `2_multivariate_analysis_output/factor_loadings.csv`
- **Visualizations**: Scree plots, biplots, loading plots

---

## Stage 3: Data Normalization (`3_normalisation_data.py`)

### Overview
This stage transforms break-adjusted raw data into normalized indicators suitable for aggregation. It applies forward fill, winsorization, and z-score standardization (baseline) or percentile scaling (alternative) to create Level 4 data with values in range [0.1, 1.0] for geometric mean computation.

### Process Flow

**Input**: Break-adjusted raw data from Stage 1
**Output**: Normalized Level 4 indicators ready for aggregation

### Step 1: Forward Fill Missing Data

**Purpose**: Complete time series by propagating last available value forward

For each series $(i,c,d)$ with first data year $t_{\min}$ and last year $t_{\max}$:

$$v_t = \begin{cases}
v_{t-1} & \text{if } v_{t-1} \neq \text{NaN} \text{ and } v_t = \text{NaN} \text{ and } t \geq t_{\min} \\
\text{NaN} & \text{if } t < t_{\min} \\
v_t & \text{otherwise}
\end{cases}$$

**Boundary Rule**: No backward fill or extrapolation beyond available data range

### Step 2: Normalization Methods

**Baseline Method: Z-Score Standardization**

**Step 2a: Winsorization**
Clip extreme values to reduce outlier impact:
$$v^{win}_{c,t,d,i} = \text{clip}(v_{c,t,d,i}, P_{1\%}(V_i), P_{99\%}(V_i))$$

where $V_i$ is the complete set of values for indicator $i$ across all countries, years, and deciles.

**Step 2b: Z-Score Standardization**
Apply cross-country standardization:
$$z_{c,t,d,i} = \frac{v^{win}_{c,t,d,i} - \bar{v}^{win}_i}{\sigma^{win}_i}$$

**Step 2c: Rescaling with Indicator Inversion**
Transform to [0.1, 1.0] range with deprivation scaling:
$$x_{c,t,d,i} = rescale_{min} + (rescale_{max} - rescale_{min}) \times \left(1 - \frac{z_{c,t,d,i} - z_{\min,i}}{z_{\max,i} - z_{\min,i}}\right)$$

**Alternative Method: Percentile Scaling**

**Empirical CDF Transformation**:
$$P_{c,t,d,i} = \frac{\text{rank}(v^{win}_{c,t,d,i}) - 1}{N_i - 1}$$

**Rescaled with Inversion**:
$$x_{c,t,d,i} = rescale_{min} + (rescale_{max} - rescale_{min}) \times (1 - P_{c,t,d,i})$$

### Baseline vs. Alternative Configurations

| Parameter | Baseline | Alternative 1 | Alternative 2 | Alternative 3 |
|-----------|----------|---------------|---------------|--------------|
| **Method** | zscore | percentile | zscore | percentile |
| **Rescale Range** | [0.1, 1.0] | [0.1, 1.0] | [0.2, 1.0] | [0.0, 1.0] |
| **Temporal Pooling** | multi_year | multi_year | per_year | multi_year |
| **Purpose** | Stable comparison | Rank-based | Conservative floor | Maximum range |

### Step 3: Data Quality Validation

**Range Verification**: All normalized values satisfy $x_{c,t,d,i} \in [rescale_{min}, rescale_{max}]$

**Geometric Mean Compatibility**: All values strictly positive ($x_{c,t,d,i} > 0$) to avoid $\log(0)$ errors

**1:1 Record Mapping**: Each raw data record produces exactly one normalized record:
$$|D_{\text{level4}}| = |D_{\text{raw}}|$$

### Data Structure Output

**DataFrame Format**:
```
Year | Country | Decile | Level | Primary and raw data | Value | Aggregation
2010 |    AT   |    1   |   4   |   HQ-SILC-1         | 0.847 | Z-score + Rescaling
2010 |    AT   |    2   |   4   |   HQ-SILC-1         | 0.623 | Z-score + Rescaling
...
```

**Key Properties**:
- **Level**: 4 (Normalized Indicators)
- **Value Range**: [rescale_min, rescale_max] with deprivation interpretation
- **Higher Values**: Indicate worse outcomes (deprivation indicators)
- **Temporal Stability**: Multi-year pooling ensures consistent scaling

### Output Files
- **Forward-filled raw**: `1_missing_data_output/raw_data_forward_filled.csv` (for Stage 4)
- **Normalized Level 4**: `3_normalisation_data_output/level4_normalised_indicators.csv`

---

## Stage 4: Weighting and Aggregation (`4_weighting_aggregation.py`)

### Overview
This stage constructs the hierarchical EWBI structure by aggregating normalized indicators through two levels: EU Priorities (Level 2) and EWBI Overall (Level 1). It uses Stage 2 PCA outputs to derive **indicator weights** within each EU Priority and then aggregates indicators (baseline: weighted geometric mean; alternative: weighted arithmetic mean). It also supports an optional post-aggregation structural break adjustment at Level 2, and computes EWBI as a geometric/arithmetic mean across the five priorities (baseline: unweighted geometric mean).

### Data Hierarchy

```
Level 1: EWBI Overall (single value per country-year-decile)
         ↑ Unweighted geometric mean
Level 2: EU Priorities (5 values per country-year-decile)
         • Energy and Housing
         • Equality  
         • Health and Animal Welfare
         • Intergenerational Fairness, Youth, Culture and Sport
         • Social Rights and Skills, Quality Jobs and Preparedness
         ↑ PCA-weighted geometric mean
Level 4: Normalized Indicators (from Stage 3)
Level 3: Raw Primary Indicators (forward-filled, included for reference)
```

### EU Priorities Aggregation (Level 4 → Level 2)

**Baseline Method (as implemented): PCA-derived indicator weights + weighted geometric mean**

For each EU Priority $p$ with indicators $\mathcal{I}_p$, Stage 4 uses the Stage 2 PCA outputs (rotated loadings + explained variance) to derive **indicator weights** $w_{c,t,i}^{(p)}$ for each country-year $(c,t)$ (JRC-style weighting; weights are normalized so that $\sum_{i\in\mathcal{I}_p} w_{c,t,i}^{(p)} = 1$).

The Level 2 EU Priority score is then computed as a **weighted geometric mean** across the Level 4 normalized indicators:
$$I_{c,t,d}^{(p)} = \exp\left(\sum_{i \in \mathcal{I}_p} w_{c,t,i}^{(p)} \ln\left(x_{c,t,d,i}\right)\right)$$

**Alternative Method (configurable): weighted arithmetic mean**
$$I_{c,t,d}^{(p)} = \sum_{i \in \mathcal{I}_p} w_{c,t,i}^{(p)} \times x_{c,t,d,i}$$

### Optional: Post-Aggregation Structural Break Adjustment (Level 2)

After Level 2 values $I_{c,t,d}^{(p)}$ are computed, Stage 4 can optionally apply a structural break adjustment on each Level 2 time series (per country $c$, decile $d$, and EU priority $p$) before constructing Level 1 EWBI.

**Purpose**: reduce large year-to-year jumps in Level 2 composites that would otherwise propagate into Level 1 EWBI.

**Break Detection**: a break at year $t \rightarrow t+1$ is detected when the absolute percent change exceeds a threshold:
$$\left|\frac{I_{t+1} - I_t}{I_t}\right| > \tau_{L2}$$

**Treatment (High Level)**:
- Detects break years and supports consecutive break runs.
- Reconstructs the affected segment by using smoothed (non-break) growth rates from outside the break region.
- Applies the reconstruction both forward and backward from a reference year to preserve continuity across the full time series.

**Configuration**:
- `EWBI_LEVEL2_BREAK_ADJUSTMENT` (default: `True`): enable/disable the Level 2 post-aggregation adjustment.
- `EWBI_LEVEL2_BREAK_THRESHOLD` (default: `0.1`): $\tau_{L2}$ used for Level 2 break detection.

**Note**: Stage 4 does not apply moving-average smoothing or mean rescaling. Those remain Stage 1 (missing data treatment) options.

### EWBI Overall Aggregation (Level 2 → Level 1)

**Baseline: Unweighted Geometric Mean**
$$\text{EWBI}_{c,t,d} = \left(\prod_{p=1}^{5} I_{c,t,d}^{(p)}\right)^{1/5}$$

Equivalent logarithmic form:
$$\text{EWBI}_{c,t,d} = \exp\left(\frac{1}{5} \sum_{p=1}^{5} \ln(I_{c,t,d}^{(p)})\right)$$

**Alternative: Arithmetic Mean**
$$\text{EWBI}_{c,t,d} = \frac{1}{5} \sum_{p=1}^{5} I_{c,t,d}^{(p)}$$

### Cross-Decile Aggregation (Individual Deciles → Country Total)

**Baseline: Geometric Mean Across Deciles**

For country-level aggregation with decile = "All Deciles":
$$I_{c,t}^{\text{country}} = \exp\left(\frac{1}{10} \sum_{d=1}^{10} \ln(I_{c,t,d})\right)$$

**Alternative: Arithmetic Mean Across Deciles**
$$I_{c,t}^{\text{country}} = \frac{1}{10} \sum_{d=1}^{10} I_{c,t,d}$$

### EU-27 Aggregation (Individual Countries → EU Average)

**Population-Weighted Arithmetic Mean** (for policy interpretability):

For normalized data (Levels 1-2):
$$I_{\text{EU-27},t,d}^{(p)} = \sum_{c \in \text{EU}} w_{c,t} \times I_{c,t,d}^{(p)}$$

For raw data (Level 3):
$$I_{\text{EU-27},t,d}^{\text{raw}} = \sum_{c \in \text{EU}} w_{c,t} \times I_{c,t,d}^{\text{raw}}$$

where population weights:
$$w_{c,t} = \frac{\text{Population}_{c,t}}{\sum_{c' \in \text{EU}} \text{Population}_{c',t}}$$

### Baseline vs. Alternative Configurations


| Component | Env var | Baseline | Other supported values |
|-----------|---------|----------|------------------------|
| **PCA Scope** | `EWBI_PCA_SCOPE` | `all_years` | `per_year` |
| **EU Priorities approach** | `EWBI_EU_PRIORITIES_APPROACH` | `pca` | `simple` (equal-weighted indicators within each priority) |
| **EU Priorities aggregation** | `EWBI_EU_PRIORITIES_AGGREGATION` | `geometric` | `arithmetic` |
| **EWBI (Level 1) aggregation across priorities (per decile)** | `EWBI_EWBI_DECILE_AGGREGATION` | `geometric` | `arithmetic` |
| **Cross-decile aggregation (country “All Deciles”)** | `EWBI_EWBI_CROSS_DECILE_AGGREGATION` | `geometric` | `arithmetic` |
| **Level 2 post-aggregation break adjustment** | `EWBI_LEVEL2_BREAK_ADJUSTMENT` | `True` | `False` |
| **Level 2 break threshold** | `EWBI_LEVEL2_BREAK_THRESHOLD` | `0.1` | any float (used as absolute relative-change threshold) |
| **Compute EU-27 aggregates** | `EWBI_SKIP_EU27` | `False` | `True` (skip EU-27 computation for speed) |

Notes:
- The **EU-27 aggregation mean type** is currently *hard-coded* in Stage 4 as population-weighted **arithmetic** mean for Levels 1–3 (`compute_eu27_aggregations(..., use_arithmetic_mean=True)`), i.e. it is not exposed as an env var in this script.
- **Component selection criteria** (eigenvalue/variance/cumulative thresholds) are applied in Stage 2 when generating the PCA results; Stage 4 consumes those results and does not currently expose those thresholds as runtime configuration.

### Data Structure Output

**Unified DataFrame Format**:
```
Year | Country | Decile | Level | EU priority | Secondary | Primary and raw data | Type | Aggregation | Value
2010 |    AT   |    1   |   1   |      -      |     -     |         -           |Composite|Geometric mean of EU Priorities|0.724
2010 |    AT   |    1   |   2   |Energy Housing|     -     |         -           |Priority |PCA-weighted geometric mean   |0.681
2010 |    AT   |    1   |   3   |      -      |     -     |   HQ-SILC-1         |Primary  |Forward-filled raw data        | 12.3
2010 |    AT   |    1   |   4   |Energy Housing|     -     |   HQ-SILC-1         |Normalized|Z-score + Rescaling           |0.847
...
```

**Level Definitions**:
- **Level 1**: EWBI Overall (single value per country-year-decile)
- **Level 2**: EU Priorities (5 values per country-year-decile)
- **Level 3**: Raw Primary Indicators (forward-filled, original scale)
- **Level 4**: Normalized Primary Indicators (rescaled for aggregation)

**Decile Values**:
- `1, 2, ..., 10`: Individual income deciles
- `'All Deciles'`: Cross-decile country aggregation

**Country Values**:
- `AT, BE, BG, ...`: Individual EU Member States
- `EU-27`: Population-weighted EU aggregate

### Output Files
- **Final aggregated**: `4_weighting_aggregation_output/ewbi_final_aggregated.csv`
- **App-ready copy**: `output/ewbi_master_aggregated.csv`
- **Metadata**: Aggregation methods and parameters used

---

## Stage 5: Sensitivity Analysis (`5_sensitivity_test_*.py`)

### Overview
This stage performs Monte Carlo sensitivity analysis to assess the robustness of EWBI rankings to methodological choices. It runs multiple experiments with different parameter configurations and analyzes the stability of country rankings.

### Sensitivity Testing Framework

**Two Experimental Designs**:

1. **Data Treatment Variables** (`5_sensitivity_test_data_treatment.py`)
   - Focus on data quality and preprocessing decisions
  - Tests robustness to structural break handling and smoothing (Stage 1) and the optional Level 2 post-aggregation break adjustment (Stage 4)

2. **Methodological Variables** (`5_sensitivity_test_data_aggregation.py`)
   - Focus on normalization and aggregation choices
   - Tests robustness to PCA weighting and geometric mean assumptions

### Monte Carlo Experiment Design

**Sample Size**: $N = 20$ experiments per analysis 
**Sampling**: Uniform random sampling from parameter space
**Baseline**: Fixed reference configuration for comparison

### Parameter Space: Data Treatment Variables

| Parameter | Baseline | Alternatives | Impact |
|-----------|----------|-------------|--------|
| **Break Threshold** | 0.2 (20%) | {0.1, 0.3} | Structural break sensitivity |
| **Moving Average** | True | {False} | Time series smoothing |
| **MA Window** | 5 years | {3 years} | Smoothing intensity |
| **Mean Rescaling** | False | {True} | Scale preservation |
| **Level 2 Break Adjustment** | True | {False} | Post-aggregation structural break correction at Level 2 |

**Random Configuration Sampling**:
$$\theta^{(k)} \sim \text{Uniform}(\Theta), \quad k = 1, 2, ..., 5$$

where $\Theta$ is the Cartesian product of all parameter alternatives.

### Parameter Space: Methodological Variables

| Parameter | Baseline | Alternatives | Impact |
|-----------|----------|-------------|--------|
| **Normalization Method** | Z-score | {Percentile} | Indicator scaling approach |
| **Rescale Range** | [0.1, 1.0] | {[0.0, 1.0], [0.2, 1.0]} | Geometric mean compatibility |
| **Temporal Pooling** | Multi-year | {Per-year} | Temporal stability |
| **PCA Scope** | All years | {Per-year} | Weight temporal variation |
| **EU Priority Method** | PCA-derived indicator weights | {Simple (equal weights within each priority)} | PCA-based weighting vs equal-weighted indicators |
| **Aggregation Type** | Geometric | {Arithmetic} | Compensation between indicators |
| **Decile Aggregation** | Geometric | {Arithmetic} | Cross-decile combination |

### Experiment Protocol

For each sampled configuration $\theta^{(k)}$:

**Step 1**: Update module configurations
```python
update_config(stage1, θ_data_treatment)
update_config(stage3, θ_normalization) 
update_config(stage4, θ_aggregation)
```

**Step 2**: Run full pipeline
```python
data_k = run_pipeline(stages=[1,2,3,4], config=θ^(k))
```

**Step 3**: Extract EWBI rankings
```python
ranking_k = extract_country_rankings(data_k, year=2019, level=1)
```

**Step 4**: Compare with baseline
```python
rank_diffs_k = compute_rank_differences(ranking_baseline, ranking_k)
mean_abs_change_k = rank_diffs_k["Abs_Rank_Diff"].mean()
```

### Robustness Metrics

The implemented sensitivity scripts quantify robustness using **rank differences** (not rank correlation).

**Rank Change (per country, per experiment)**:
$$\Delta R_{c,k} = R_{c,\text{baseline}} - R_{c,\theta^{(k)}}$$
$$|\Delta R_{c,k}| = \left|R_{c,\text{baseline}} - R_{c,\theta^{(k)}}\right|$$

**Experiment-level stability summary** (across countries $c$):
$$\text{MeanAbsChange}_k = \text{mean}_c\left(|\Delta R_{c,k}|\right)$$

**Country-level sensitivity summaries** (across experiments $k$):
- $\text{MeanAbs}_c = \text{mean}_k\left(|\Delta R_{c,k}|\right)$
- $\text{Std}_c = \text{std}_k\left(\Delta R_{c,k}\right)$
- $\text{MaxAbs}_c = \max_k\left(|\Delta R_{c,k}|\right)$

### Output Analysis

**Experiment Data**: Each experiment saves:
- Complete EWBI dataset with configuration ID
- Country rankings for latest year
- Parameter configuration used
- Rank differences vs baseline (computed from baseline + experiment rankings)

**Comparative Visualizations** (handled by separate visualization scripts):
- Rank-change distributions (and mean absolute rank changes)
- Country-specific rank stability
- Parameter sensitivity heatmaps
- Ranking trajectory plots

### Output Files
- **Experiment data**: `5_sensitivity_test_*/data/experiment_*.csv`
- **Summary statistics**: `5_sensitivity_test_*/summary_stats.json`
- **Configuration log**: `5_sensitivity_test_*/experiment_log.csv`
- **Baseline comparison**: `baseline_vs_experiments.csv`

---

## Summary of Mathematical Formulas

### Core Processing Formulas for baseline

| Stage | Operation | Formula | Application |
|-------|-----------|---------|-------------|
| **0** | Income Equivalization | $I_{h,eq} = \frac{HY020_h}{1.0 + \sum_{i=2}^{n_h} w_i}$ | OECD modified scale |
| **0** | Indicator Calculation | $I_{c,t,d,i} = \frac{\sum_{h \in (c,t,d)} \mathbf{1}_{X_{h,i} = \text{condition}} \times W_h}{\sum_{h \in (c,t,d)} W_h} \times 100$ | Binary prevalence rates |
| **1** | Linear Interpolation | $v_t = v_{t-1} + \frac{v_{t+1} - v_{t-1}}{t_{t+1} - t_{t-1}} \times (t - t_{t-1})$ | Fill missing years |
| **1** | Break Detection | $\text{Break} = \left|\frac{v_{t+1} - v_t}{v_t}\right| > \tau$ | Structural break threshold |
| **1** | Break Adjustment | Reconstruction using smoothed non-break growth rates (supports consecutive breaks) | Time-series discontinuity correction |
| **4** | Level 2 Break Detection (optional) | $\text{Break}_{L2} = \left|\frac{I_{t+1} - I_t}{I_t}\right| > \tau_{L2}$ | EU Priority time series (Level 2) |
| **4** | Level 2 Break Adjustment (optional) | Reconstruction using smoothed non-break growth rates (supports consecutive breaks) | Applied after Level 2 aggregation, before Level 1 |
| **2** | PCA Weights | $w_i = \frac{\sum_{k=1}^{m} \lambda_k \times f_{i,k}^2}{\sum_{j=1}^{n} \sum_{k=1}^{m} \lambda_k \times f_{j,k}^2}$ | Factor-based weighting |
| **3** | Z-Score Normalization | $z_{c,t,d,i} = \frac{v_{c,t,d,i} - \bar{v}_i}{\sigma_i}$ | Cross-country standardization |
| **3** | Percentile Scaling | $P_{c,t,d,i} = \frac{\text{rank}(v_{c,t,d,i}) - 1}{N_i - 1}$ | Empirical CDF |
| **3** | Indicator Inversion | $x_{c,t,d,i} = r_{min} + (r_{max} - r_{min}) \times (1 - P)$ | Deprivation scaling |
| **4** | EU Priority Composite | $I_{c,t,d}^{(p)} = \exp\left(\sum_{i \in \mathcal{I}_p} w_{c,t,i}^{(p)}\,\ln(x_{c,t,d,i})\right)$ | PCA-derived indicator weights + weighted geometric mean |
| **4** | EWBI Overall | $\text{EWBI}_{c,t,d} = \exp\left(\frac{1}{5} \sum_{p=1}^{5} \ln(I_{c,t,d}^{(p)})\right)$ | Unweighted geometric mean |
| **4** | Cross-Decile | $I_{c,t}^{country} = \exp\left(\frac{1}{10} \sum_{d=1}^{10} \ln(I_{c,t,d})\right)$ | Country-level aggregation |
| **4** | EU-27 Aggregation | $I_{EU27,t,d} = \sum_{c} w_{c,t} \times I_{c,t,d}$ | Population-weighted arithmetic |

### Key Parameters and Thresholds

| Parameter | Baseline Value | Alternative Values | Impact |
|-----------|----------------|-------------------|--------|
| **Break Threshold** ($\tau$) | 0.20 (20%) | {0.10, 0.30} | Break detection sensitivity |
| **Level 2 Break Threshold** ($\tau_{L2}$) | 0.10 (10%) | (toggle only in sensitivity) | Level 2 break detection sensitivity |
| **Level 2 Break Adjustment** | Enabled | {Disabled} | Applies/removes Level 2 post-aggregation break adjustment |
| **Rescale Range** | [0.1, 1.0] | {[0.0, 1.0], [0.2, 1.0]} | Geometric mean compatibility |
| **Winsorization** | [P1, P99] | Fixed | Outlier robustness |
| **PCA Components** | $\lambda > 1.0$ | Kaiser criterion | Dimensionality reduction |
| **EU Priorities** | 5 dimensions | Fixed by policy | EWBI structure |
| **Deciles** | 10 groups | Fixed by methodology | Income distribution |

## File Structure and Data Flow

### Processing Pipeline Files

**Stage 0: Data Collection**
- Input: Raw EU-SILC and LFS files from external sources
- Processing: `0_raw_indicator_EU-SILC.py`, `0_raw_indicator_LFS.py`
- Output: `0_raw_data_EUROSTAT/1_final_df/combined_raw_indicators.csv`

**Stage 1: Missing Data Treatment**
- Input: Combined raw indicators from Stage 0
- Processing: `1_missing_data.py`
- Output: `1_missing_data_output/raw_data_break_adjusted.csv`

**Stage 2: Multivariate Analysis**
- Input: Break-adjusted data from Stage 1
- Processing: `2_multivariate_analysis.py`
- Output: `2_multivariate_analysis_output/pca_results_full.json`

**Stage 3: Data Normalization**
- Input: Break-adjusted data from Stage 1
- Processing: `3_normalisation_data.py`
- Output: 
  - `1_missing_data_output/raw_data_forward_filled.csv` (for Stage 4)
  - `3_normalisation_data_output/level4_normalised_indicators.csv`

**Stage 4: Weighting and Aggregation**
- Input: Normalized data (Stage 3) + PCA weights (Stage 2) + Population data
- Processing: `4_weighting_aggregation.py`
- Output: 
  - `4_weighting_aggregation_output/ewbi_final_aggregated.csv`
  - `output/ewbi_master_aggregated.csv` (app-ready)

**Stage 5: Sensitivity Analysis**
- Input: Full pipeline (Stages 1-4) with parameter variations
- Processing: `5_sensitivity_test_data_treatment.py`, `5_sensitivity_test_data_aggregation.py`
- Output: `5_sensitivity_test_*/data/experiment_*.csv`

### Dashboard Application
- Input: `output/ewbi_master_aggregated.csv`
- Processing: `app.py`
- Output: Interactive Dash web application

### Configuration Files
- **Indicator mappings**: `variable_mapping.py`
- **Population data**: `data/population_transformed.csv`
- **EWBI structure**: `data/ewbi_indicators.json`

---

## Data Quality and Validation

### Coverage Requirements

**Temporal Coverage**: Complete time series from first available year to 2023
**Geographic Coverage**: All 27 EU Member States
**Income Coverage**: Complete decile distribution (deciles 1-10)

### Quality Checks

**Range Validation**:
- Raw indicators: $0 \leq v_{c,t,d,i} \leq 100$ (percentages)
- Normalized indicators: $r_{min} \leq x_{c,t,d,i} \leq r_{max}$
- Aggregated indices: $r_{min} \leq I_{c,t,d} \leq r_{max}$

**Consistency Checks**:
- Decile sum: $\sum_{d=1}^{10} n_{c,t,d} = n_{c,t}$ (population coverage)
- Weight sum: $\sum_{i \in \mathcal{I}_p} w_{i}^{(p)} = 1$ (normalized PCA weights)
- Aggregation hierarchy: Level 1 computable from Level 2 values

**Missing Data Patterns**:
- Document systematic missingness by country/year/indicator
- Validate forward fill assumptions (no structural changes)
- Check break adjustment effectiveness (reduced volatility)

### Robustness Assessment

**Sensitivity Metrics**:
- Rank-change stability: distribution of $\Delta R_{c,k}$ / $|\Delta R_{c,k}|$ across experiments
- Experiment-level stability: mean absolute rank change across countries per experiment
- Country sensitivity: $\text{MeanAbs}_c$, $\text{Std}_c$, and $\text{MaxAbs}_c$ over experiments
- Methodological sensitivity: identify most/least robust components (by induced rank changes)

**Validation Benchmarks**:
- Cross-validation with alternative composite indices
- Expert validation of indicator weights and aggregation methods
- Policy relevance assessment of final rankings

---

## References and Methodology Sources

**Official Guidelines**:
- OECD (2008): "Handbook on Constructing Composite Indicators: Methodology and User Guide"
- EU JRC (2019): "Guidelines for Developing and Reporting Composite Indicators"
- Eurostat (2023): "EU-SILC Methodological Guidelines and Description of Target Variables"

**Data Sources**:
- **EU-SILC**: European Union Statistics on Income and Living Conditions (Eurostat)
- **LFS**: Labour Force Survey (Eurostat)  
- **Population**: Demographic statistics (Eurostat)

**Technical Implementation**:
- **Programming**: Python 3.8+ with pandas, numpy, scikit-learn
- **Visualization**: Plotly Dash for interactive dashboard
- **Statistical Methods**: Principal Component Analysis with Varimax rotation
- **Aggregation**: Population-weighted geometric and arithmetic means

**Quality Assurance**:
- Monte Carlo sensitivity analysis (30 experiments per parameter set)
- Cross-validation with alternative methodological choices
- Robustness testing across different time periods and country subsets
