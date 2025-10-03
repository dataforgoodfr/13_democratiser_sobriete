# European Well-Being Index (EWBI) Dashboard - Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Data Architecture](#data-architecture)
3. [Level Structure & Computation Methods](#level-structure--computation-methods)
4. [Graph Types & Data Sources](#graph-types--data-sources)
5. [Detailed Chart Analysis](#detailed-chart-analysis)
6. [Data Processing Pipeline](#data-processing-pipeline)
7. [Aggregation Methods](#aggregation-methods)

## Overview

The European Well-Being Index (EWBI) Dashboard is a hierarchical analytical system that processes multiple European survey datasets to create composite well-being indicators. Based on the actual codebase analysis, the system implements a 5-level structure (Levels 1, 2, 3, 4, and 5) with geometric mean aggregation methods and income-based stratification.

## Data Architecture

### Core Data Sources (From Code Analysis)

#### 1. EU-SILC (European Union Statistics on Income and Living Conditions)
**File**: `0_raw_indicator_EU-SILC.py`
**Primary Functions**: `process_personal_indicators()` and `process_household_indicators()`

**Key Indicators Processed**:
- **Income-related**: Unable to handle unexpected costs, arrears on payments
- **Housing**: Too cold/warm housing, overcrowding, housing problems
- **Social participation**: Meeting with friends, participation in activities
- **Health**: Self-reported health status, unmet medical needs
- **Employment**: Work-life balance, job satisfaction

**Income Decile Calculation**:
```python
def calculate_income_deciles(df):
    # Custom weighted quantile calculation using HY020 (equivalised disposable income)
    # Assigns households to income deciles (1-10) using survey weights
    # Handles missing values and edge cases
    return df_with_deciles
```

**Data Structure**: 
- Country/Year/Household/Person level data
- Income deciles (1-10) plus "All" aggregate
- Temporal coverage: 2004-2023 (varies by country)

#### 2. EHIS (European Health Interview Survey)
**File**: `0_raw_indicator_EHIS.py`
**Frequency**: Every 5-6 years (2008, 2014, 2019)

**Key Indicators**:
- Physical activity levels
- Smoking and alcohol consumption
- Mental health indicators
- Healthcare access and utilization

**Income Stratification**: Uses income quintiles (1-5) instead of deciles

#### 3. HBS (Household Budget Survey) & LFS (Labour Force Survey)
**Processing**: Integrated within main EU-SILC pipeline
**Focus**: Expenditure patterns and employment statistics

### Unified Dataset Structure
**Output File**: `unified_all_levels_1_to_5.csv`
**Records**: ~291,623 observations
**Key Columns**:
- `Country`, `Year`, `Level`, `Decile`/`Quintile`
- `EU_priority`, `Secondary`, `Primary_indicator`
- `Value` (normalized 0-1 scale)
- `Aggregation` (method used for calculation)

## Level Structure & Computation Methods

### Level 5: Primary Indicators (Raw Statistics)
**Source**: Direct survey responses from EU-SILC, EHIS, HBS, LFS
**Processing**: Individual indicator calculations with normalization

**Example Indicators** (from `process_personal_indicators()`):
- `IS-SILC-1`: Unable to handle unexpected costs
- `IS-SILC-2`: Arrears on mortgage/rent payments
- `AN-SILC-1`: Meeting with friends/family
- `HE-SILC-1`: Self-reported health status

**Normalization**: Z-score normalization applied to convert raw percentages to 0-1 scale

### Level 4: Normalized Primary Indicators
**Source**: Normalization of Level 5 indicators per country/year/decile
**Processing**: Statistical normalization to ensure comparability across indicators

**Normalization Process**:
```python
def normalize_to_level4(df_level5):
    # Applies country/year/decile specific normalization
    # Ensures all indicators are on comparable 0-1 scale
    # Maintains decile stratification
    return df_level4
```

### Level 3: Secondary Indicators
**Source**: Aggregation of Level 4 indicators using geometric mean
**Configuration**: Defined in `ewbi_indicators.json`

**Aggregation Process** (from `3_generate_outputs.py`):
```python
def aggregate_to_secondary(df_level4):
    # Groups Level 4 indicators by secondary category
    # Applies geometric mean per country/year/decile
    # Creates secondary indicator values
    return df_level3
```

**Examples**:
- "Nutrition need" (aggregates food-related indicators)
- "Housing quality" (aggregates housing condition indicators)

### Level 2: EU Priorities
**Source**: Aggregation of Level 3 indicators using geometric mean
**Categories**: 7 EU policy priorities

**Aggregation Process**:
```python
def aggregate_to_eu_priorities(df_level3):
    # Groups secondary indicators by EU priority
    # Applies geometric mean per country/year/decile
    # Maintains decile stratification
    return df_level2
```

**EU Priority Categories**:
1. Agriculture and Food
2. Energy and Housing  
3. Equality
4. Health and Animal Welfare
5. Intergenerational Fairness, Youth, Culture and Sport
6. Social Rights and Skills, Quality Jobs and Preparedness
7. Sustainable Transport and Tourism

### Level 1: EWBI (Overall Well-being Index)
**Source**: Aggregation of Level 2 indicators using geometric mean
**Purpose**: Single composite well-being score

**Computation**:
```python
def calculate_ewbi(df_level2):
    # Takes all 7 EU Priority scores
    # Applies geometric mean per country/year/decile: (p1 × p2 × ... × p7)^(1/7)
    # Ensures poor performance in any area significantly impacts overall score
    return df_level1
```

## Graph Types & Data Sources

### 1. Choropleth Map (`create_choropleth_map()`)
**Data Filter**: 
- Latest available year per indicator
- Decile/Quintile = "All" 
- Individual countries only (excludes "All Countries")

**Implementation**:
```python
def create_choropleth_map(filtered_data, level):
    # Uses plotly.graph_objects for geographic visualization
    # Color scale: Green (better) to Red (worse)
    # Handles missing data with gray coloring
    return fig
```

### 2. Time Series Chart (`create_time_series()`)
**Data Filter**:
- All available years (2004-2023)
- Selected countries + "All Countries" median
- Decile/Quintile = "All"

**Implementation**:
```python
def create_time_series(filtered_data, countries):
    # Line plot showing temporal evolution
    # Includes trend analysis and year-over-year changes
    # Handles missing data gaps
    return fig
```

### 3. Decile/Quintile Analysis (`create_decile_chart()`)
**Data Filter**:
- Latest year
- Individual deciles/quintiles (excludes "All")
- Selected countries

**Implementation**:
```python
def create_decile_chart(filtered_data, level):
    # Bar chart showing income-based disparities
    # Reference line for "All" decile/quintile average
    # Different handling for EHIS (quintiles) vs other surveys (deciles)
    return fig
```

### 4. Radar Chart (`create_radar_chart()`)
**Scope**: Level 1 only
**Data**: All 7 EU Priorities for selected countries

**Implementation**:
```python
def create_radar_chart(ewbi_data, countries):
    # Multi-dimensional comparison across EU priorities
    # Overlays multiple countries for direct comparison
    # 0-1 scale with center at 0
    return fig
```

## Detailed Chart Analysis

### Level 1: EWBI Overall Analysis

#### Map Chart
- **Data**: `df[(Level==1) & (Year==latest) & (Decile=='All') & (Country!='All Countries')]`
- **Computation**: Geometric mean of 7 EU Priority scores per country
- **Values**: 0-1 scale where 1.0 represents perfect well-being across all dimensions
- **Interpretation**: Countries with higher scores (green) have better overall well-being

#### Time Series
- **Data**: `df[(Level==1) & (Decile=='All') & (Aggregation=='Median across countries')]`
- **Shows**: Long-term trends in European well-being (2004-2023)
- **Country Selection**: EU median ("All Countries") plus user-selected individual countries
- **Trend Analysis**: Reveals whether well-being is improving or declining over time

#### Decile Analysis
- **Data**: `df[(Level==1) & (Year==latest) & (Decile!='All') & (Aggregation=='Geometric mean level-1')]`
- **Income Inequality**: Shows how overall well-being varies by household income
- **Expected Pattern**: Generally higher deciles should show better well-being
- **Reference Line**: Overall country average ("All" deciles) for comparison

#### Radar Chart
- **Data**: `df[(Level==2) & (Year==latest) & (Decile=='All')]` for each EU Priority
- **Dimensions**: 7 EU Priorities forming the radar axes
- **Multi-Country**: Overlays multiple countries for direct comparison
- **Balance Analysis**: Reveals whether countries excel in all areas or have specific strengths/weaknesses

### Level 2: EU Priority Analysis

#### Map Chart
- **Data**: `df[(Level==2) & (Year==latest) & (Decile=='All') & (EU_priority==selected)]`
- **Computation**: Geometric mean of secondary indicators within the selected EU priority
- **Focus**: Geographic patterns for specific policy areas
- **Policy Insight**: Identifies which countries perform best in specific EU priority areas

#### Time Series
- **Data**: `df[(Level==2) & (EU_priority==selected) & (Decile=='All')]`
- **Policy Tracking**: Evolution of specific EU priority over time
- **Comparative**: Shows how different countries progress in the same policy area
- **Policy Evaluation**: Can reveal impact of policy interventions over time

#### Decile Analysis
- **Data**: `df[(Level==2) & (Year==latest) & (EU_priority==selected) & (Decile!='All')]`
- **Equity Analysis**: Whether policy benefits reach all income groups equally
- **Social Justice**: Reveals if EU priorities address inequality or exacerbate it
- **Target Identification**: Shows which income groups need more policy attention

#### Country Comparison
- **Data**: `df[(Level==2) & (Year==latest) & (EU_priority==selected) & (Decile=='All') & (Country!='All Countries')]`
- **Ranking**: All countries ordered by performance in selected EU priority
- **Best Practices**: Identifies top-performing countries for policy learning
- **Benchmarking**: Shows each country's position relative to others

### Level 3: Secondary Indicator Analysis

#### Map Chart
- **Data**: `df[(Level==3) & (Year==latest) & (Decile=='All') & (Secondary==selected)]`
- **Computation**: Geometric mean of Level 4 indicators within the secondary indicator
- **Specificity**: More focused analysis on particular well-being dimensions
- **Targeted Policy**: Useful for specific intervention planning

#### Time Series
- **Data**: `df[(Level==3) & (Secondary==selected) & (Decile=='All')]`
- **Trend Monitoring**: Detailed tracking of specific well-being dimensions
- **Early Warning**: Can detect emerging problems in specific areas
- **Intervention Assessment**: Evaluates effectiveness of targeted policies

#### Decile Analysis
- **Data**: `df[(Level==3) & (Year==latest) & (Secondary==selected) & (Decile!='All')]`
- **Detailed Inequality**: Income-based disparities in specific well-being areas
- **Targeted Support**: Identifies which income groups need support in specific dimensions
- **Root Cause Analysis**: Helps understand mechanisms behind inequality

#### Country Comparison
- **Data**: `df[(Level==3) & (Year==latest) & (Secondary==selected) & (Decile=='All') & (Country!='All Countries')]`
- **Specialized Ranking**: Countries ranked on specific well-being dimensions
- **Expertise Identification**: Shows which countries excel in particular areas
- **Learning Network**: Facilitates knowledge sharing on specific topics

### Level 4: Normalized Primary Indicators Analysis

#### Map Chart
- **Data**: `df[(Level==4) & (Year==latest) & (Decile=='All') & (Primary_indicator==selected)]`
- **Normalized Values**: Country/year/decile normalized statistical values (0-1 scale)
- **Survey-Specific**: Shows normalized survey response patterns geographically
- **Comparable**: Ensures fair comparison across different indicators

#### Time Series
- **Data**: `df[(Level==4) & (Primary_indicator==selected) & (Decile=='All')]`
- **Normalized Trends**: Normalized statistical evolution over time
- **Cross-Indicator**: Allows comparison between different types of indicators
- **Validation**: Validates higher-level trends with normalized survey data

#### Decile Analysis
- **Data**: `df[(Level==4) & (Year==latest) & (Primary_indicator==selected) & (Decile!='All')]`
- **Income Stratification**: How normalized indicators vary by household income
- **Standardized Inequality**: Normalized measurement of income-based disparities
- **Methodological**: Shows income effects after normalization

### Level 5: Raw Statistics Analysis

#### Map Chart
- **Data**: `df[(Level==5) & (Year==latest) & ((Decile=='All') | (Quintile=='All')) & (Primary_indicator==selected)]`
- **Raw Values**: Original survey percentages before normalization
- **Survey-Specific**: Shows actual survey response patterns geographically
- **Ground Truth**: Most granular view of actual conditions across Europe

#### Time Series
- **Data**: `df[(Level==5) & (Primary_indicator==selected) & ((Decile=='All') | (Quintile=='All'))]`
- **Statistical Trends**: Raw statistical evolution over time
- **Survey Continuity**: Shows how survey responses change over time
- **Validation**: Can validate higher-level trends with actual survey data

#### Decile/Quintile Analysis
- **Data Structure**:
  - **Non-EHIS**: `df[(Level==5) & (Primary_indicator==selected) & (Decile!='All')]`
  - **EHIS**: `df[(Level==5) & (Primary_indicator==selected) & (Quintile!='All')]`
- **Income Stratification**: How raw problems vary by household income
- **Survey-Level Inequality**: Direct measurement of income-based disparities
- **Methodological**: Shows how different surveys capture income effects

## Data Processing Pipeline

### Stage 1: Raw Data Processing (`0_raw_indicator_*.py`)

#### EU-SILC Processing Pipeline:
```python
def main_processing():
    # 1. Load raw EU-SILC data
    df = load_eusilc_data()
    
    # 2. Calculate income deciles per country/year
    df = calculate_income_deciles(df)
    
    # 3. Process personal-level indicators
    personal_indicators = process_personal_indicators(df)
    
    # 4. Process household-level indicators  
    household_indicators = process_household_indicators(df)
    
    # 5. Merge and aggregate by decile
    final_df = merge_and_aggregate(personal_indicators, household_indicators)
    
    # 6. Save processed indicators (Level 5)
    final_df.to_csv('output/primary_data_preprocessed.csv')
```

#### Income Decile Calculation:
```python
def calculate_income_deciles(df):
    # Uses equivalised disposable income (HY020)
    # Applies custom weighted quantile calculation
    # Handles edge cases and missing values
    # Assigns decile ranks 1-10 per country/year
    return df_with_deciles
```

### Stage 2: Data Integration (`1_final_df.py`)
```python
def integrate_datasets():
    # 1. Load all processed survey datasets (Level 5)
    eusilc_data = pd.read_csv('output/primary_data_preprocessed.csv')
    ehis_data = pd.read_csv('output/ehis_processed.csv')
    
    # 2. Harmonize country codes and time periods
    combined_data = harmonize_datasets([eusilc_data, ehis_data])
    
    # 3. Create "All Countries" aggregations using median
    combined_data = add_country_aggregations(combined_data)
    
    # 4. Standardize to Level 5 format
    level5_data = standardize_to_level5(combined_data)
    return level5_data
```

### Stage 3: Hierarchical Aggregation (`3_generate_outputs.py`)

#### Aggregation Workflow:
```python
def generate_hierarchical_data():
    # 1. Load Level 5 (raw primary indicators)
    level5 = load_primary_data()
    
    # 2. Normalize Level 5 → Level 4 (normalization)
    level4 = normalize_primary_indicators(level5)
    
    # 3. Aggregate Level 4 → Level 3 (geometric mean)
    level3 = aggregate_to_secondary_indicators(level4)
    
    # 4. Aggregate Level 3 → Level 2 (geometric mean)
    level2 = aggregate_to_eu_priorities(level3)
    
    # 5. Aggregate Level 2 → Level 1 (geometric mean)
    level1 = aggregate_to_ewbi(level2)
    
    # 6. Combine all levels
    unified_data = combine_all_levels([level1, level2, level3, level4, level5])
    
    # 7. Generate final outputs
    unified_data.to_csv('output/unified_all_levels_1_to_5.csv')
    return unified_data
```

### Stage 4: Dashboard Application (`4_app_new.py`)

#### Dash App Structure:
```python
def create_dashboard():
    # 1. Load unified dataset
    df = pd.read_csv('output/unified_all_levels_1_to_5.csv')
    
    # 2. Setup Dash app with Bootstrap theme
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # 3. Create interactive components
    app.layout = create_layout()
    
    # 4. Setup callbacks for chart updates
    @app.callback([Output('map-chart', 'figure'), ...], 
                  [Input('level-dropdown', 'value'), ...])
    def update_charts(level, country, year):
        # Dynamic chart generation based on user selections
        return update_all_charts(level, country, year)
```

## Aggregation Methods

### 1. Geometric Mean (Primary Aggregation Method)
**Formula**: `(∏ values[i])^(1/n)`
**Used For**:
- Level 4 → Level 3 aggregation
- Level 3 → Level 2 aggregation  
- Level 2 → Level 1 aggregation
- Within-decile aggregations for Levels 1-4

**Implementation**:
```python
def geometric_mean(values):
    # Handles missing values and zero values appropriately
    valid_values = values[~np.isnan(values) & (values > 0)]
    if len(valid_values) == 0:
        return np.nan
    return np.power(np.prod(valid_values), 1/len(valid_values))
```

### 2. Median Aggregation (Cross-Country)
**Used For**: "All Countries" aggregations in time series and cross-country comparisons
**Implementation**:
```python
def country_median(country_values):
    return np.median(country_values[~np.isnan(country_values)])
```

### 3. Z-Score Normalization (Level 5 → Level 4)
**Used For**: Converting raw survey percentages to normalized 0-1 scale
**Implementation**:
```python
def normalize_indicator(raw_values, country, year, decile):
    # Applies country/year/decile specific normalization
    # Ensures comparability across different indicators
    z_scores = (raw_values - reference_mean) / reference_std
    normalized = transform_to_01_scale(z_scores)
    return normalized
```

### Weighting System
**Configuration**: `ewbi_indicators.json`
**Structure**:
```json
{
  "EU_priorities": {
    "Agriculture and Food": {
      "weight": 1.0,
      "secondary_indicators": {
        "Nutrition need": {"weight": 0.5, "primary_indicators": {...}},
        "Nutrition expense": {"weight": 0.5, "primary_indicators": {...}}
      }
    }
  }
}
```

**Implementation**:
```python
def apply_geometric_aggregation(indicators, weights_config):
    # Loads indicator hierarchy from JSON configuration
    # Applies geometric mean for all level-to-level aggregations
    # Maintains country/year/decile stratification throughout
    return aggregated_indicators
```

---

*This documentation reflects the actual implementation found in the codebase, providing accurate technical details for understanding and maintaining the EWBI Dashboard system.*