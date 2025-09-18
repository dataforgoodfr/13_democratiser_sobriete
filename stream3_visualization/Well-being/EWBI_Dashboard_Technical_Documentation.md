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

The European Well-Being Index (EWBI) Dashboard is a multi-level analytical tool that visualizes well-being indicators across European countries. The system operates on a 5-level hierarchical structure, providing different granularities of analysis from overall well-being scores down to raw statistical data.

## Data Architecture

### Unified Data Structure
The dashboard operates on a single unified dataset: `unified_all_levels_1_to_5.csv` containing:
- **291,623 records** covering all levels, countries, years, and deciles/quintiles
- **Temporal coverage**: 2004-2023 (varies by indicator and data source)
- **Geographical coverage**: 33 European countries + "All Countries" aggregates
- **Income stratification**: Income deciles (1-10) plus "All" aggregate, or income quintiles (1-5) for EHIS data

### Data Sources Integration
The system integrates data from multiple European statistical surveys:

1. **EU-SILC (European Union Statistics on Income and Living Conditions)**
   - Primary source for income, social inclusion, and living conditions
   - Uses income deciles (1-10) for stratification
   - Annual data collection across EU countries

2. **EHIS (European Health Interview Survey)**
   - Health-related indicators and behaviors
   - Uses income quintiles (1-5) instead of deciles
   - Less frequent data collection (every 5-6 years)

3. **HBS (Household Budget Survey)**
   - Household expenditure patterns and consumption
   - Uses income deciles (1-10)
   - Periodic data collection

4. **LFS (Labour Force Survey)**
   - Employment and labor market indicators
   - Uses income deciles (1-10)
   - Quarterly/monthly data collection

### Hierarchical Structure Configuration
The system structure is defined in `ewbi_indicators.json`, containing:
- EU Priorities (Level 2) with their component weights
- Secondary Indicators (Level 3) within each priority
- Primary/Raw Indicators (Level 5) with their survey sources and weights

## Level Structure & Computation Methods

### Level 1: EWBI (Overall Well-being Score)
**Definition**: The comprehensive well-being index aggregating all EU priorities.

**Data Source**: Computed from Level 2 data using geometric mean aggregation.

**Computation Process**:
1. Takes all 7 EU Priority scores for each country/year/decile combination
2. Applies geometric mean: `EWBI = (Priority₁ × Priority₂ × ... × Priority₇)^(1/7)`
3. Results in values between 0-1 (higher = better well-being)

**Decile Aggregation**: 
- Individual deciles (1-10): Direct computation from Level 2 decile scores
- "All" deciles: Geometric mean across all individual decile scores

### Level 2: EU Priorities
**Definition**: Seven major policy areas defined by European Union priorities.

**Categories**:
1. Agriculture and Food
2. Energy and Housing
3. Equality
4. Health and Animal Welfare
5. Intergenerational Fairness, Youth, Culture and Sport
6. Social Rights and Skills, Quality Jobs and Preparedness
7. Sustainable Transport and Tourism

**Data Source**: Computed from Level 3 (Secondary Indicators) using arithmetic mean.

**Computation Process**:
1. Groups all secondary indicators within each EU priority
2. Applies weighted arithmetic mean based on component weights from `ewbi_indicators.json`
3. Each secondary indicator contributes proportionally to its assigned weight

**Decile Aggregation**:
- Individual deciles: Arithmetic mean of Level 3 decile scores within each priority
- "All" deciles: Arithmetic mean across decile scores 1-10

### Level 3: Secondary Indicators
**Definition**: 19 specific well-being dimensions representing focused areas within EU priorities.

**Examples**:
- Nutrition need, Nutrition expense (under Agriculture and Food)
- Housing quality, Housing affordability, Energy affordability (under Energy and Housing)
- Community, Digital Skills (under Equality)

**Data Source**: Computed from Level 5 (Raw Data) using arithmetic mean aggregation.

**Computation Process**:
1. Groups all primary/raw indicators within each secondary indicator
2. Applies weighted arithmetic mean based on indicator weights
3. Normalizes to 0-1 scale where higher values indicate better outcomes

**Decile Aggregation**:
- Individual deciles: Arithmetic mean of Level 5 decile scores within each secondary indicator
- "All" deciles: Geometric mean across decile scores 1-10

### Level 4: [Skipped in Current Implementation]
**Note**: Level 4 was originally designed for intermediate aggregation but is currently skipped in the dashboard implementation, jumping directly from Level 3 to Level 5.

### Level 5: Raw Statistics/Primary Indicators
**Definition**: Individual survey questions and statistical measurements from European surveys.

**Data Source**: Direct survey responses and statistical calculations from EU-SILC, EHIS, HBS, and LFS.

**Data Types**:
1. **Percentage indicators**: Share of population experiencing specific conditions
2. **Binary indicators**: Yes/No responses to specific questions
3. **Categorical indicators**: Ordinal responses converted to percentages
4. **Expenditure indicators**: Spending patterns relative to income

**Examples**:
- "Struggling to Prepare Meals" (EHIS survey, quintile-based)
- "Homes Too Cold in Winter" (EU-SILC survey, decile-based)
- "Unable to Handle Unexpected Costs" (EU-SILC survey, decile-based)

**Decile/Quintile Structure**:
- **EU-SILC, HBS, LFS indicators**: Use income deciles 1-10 plus "All" aggregate
- **EHIS indicators**: Use income quintiles 1-5 plus "All" aggregate

**Value Interpretation**:
- Raw statistical values (normalized format 0-1, processed from original percentages)
- **Lower values generally indicate better well-being** (fewer people struggling)
- Dashboard uses inverted color scales (green for low values, red for high values)

## Graph Types & Data Sources

### 1. European Map Chart
**Purpose**: Geographic visualization of well-being scores across European countries for the latest available year.

**Data Source per Level**:
- **Level 1**: Latest year EWBI scores, Decile="All", individual countries only
- **Level 2**: Latest year EU Priority scores, Decile="All", individual countries
- **Level 3**: Latest year Secondary Indicator scores, Decile="All", individual countries  
- **Level 5**: Latest year Raw Statistics, Decile/Quintile="All", individual countries

**Color Coding**:
- **Levels 1-3**: Green (high scores) to Red (low scores) - higher is better
- **Level 5**: Green (high values) to Red (low values) - higher normalized values (0-1) are better due to negative z-score normalization

**Geographic Mapping**: Uses ISO-3 country codes for choropleth mapping, focusing on European region.

### 2. Time Series Chart
**Purpose**: Shows evolution of indicators over time for selected countries.

**Data Source per Level**:
- **Level 1**: All years, "All Countries" + selected individual countries, Decile="All", Aggregation="Median across countries"
- **Level 2**: All years, EU Priority filter, "All Countries" + selected countries, Decile="All"
- **Level 3**: All years, Secondary Indicator filter, "All Countries" + selected countries, Decile="All"
- **Level 5**: All years, Primary Indicator filter, "All Countries" + selected countries, Decile/Quintile="All"

**Temporal Coverage**: 2004-2023 (varies by indicator availability)

### 3. Decile/Quintile Analysis Chart
**Purpose**: Income inequality analysis showing how indicators vary across income groups.

**Data Source per Level**:
- **Level 1-3**: Latest year, selected countries, Deciles 1-10 (excluding "All")
- **Level 5 (Non-EHIS)**: Latest year, selected countries, Deciles 1-10
- **Level 5 (EHIS)**: Latest year, selected countries, Quintiles 1-5

**Reference Lines**: Shows "All" decile/quintile value as horizontal dashed line for comparison.

**Income Interpretation**:
- **Decile 1**: Lowest income households
- **Decile 10**: Highest income households
- **Pattern Analysis**: Reveals whether well-being improves with income

### 4. Radar Chart / Country Comparison
**Purpose**: Multi-dimensional comparison across different aspects of well-being.

**Data Source per Level**:
- **Level 1**: Radar chart showing all 7 EU Priorities for selected countries (latest year)
- **Levels 2-5**: Horizontal bar chart comparing all countries for the specific indicator (latest year)

**Radar Chart Scaling**: 0-1 scale with higher values extending further from center.

## Detailed Chart Analysis

### Level 1: EWBI Overall Analysis

#### Map Chart
- **Data**: `unified_df[Level==1 & Year==latest & Decile=='All' & Country!='All Countries']`
- **Computation**: Geometric mean of 7 EU Priority scores per country
- **Values**: 0-1 scale where 1.0 represents perfect well-being across all dimensions
- **Interpretation**: Countries with higher scores (green) have better overall well-being

#### Time Series
- **Data**: `unified_df[Level==1 & Decile=='All' & Aggregation=='Median across countries']`
- **Shows**: Long-term trends in European well-being (2004-2023)
- **Country Selection**: EU median ("All Countries") plus user-selected individual countries
- **Trend Analysis**: Reveals whether well-being is improving or declining over time

#### Decile Analysis
- **Data**: `unified_df[Level==1 & Year==latest & Decile!='All' & Aggregation=='Geometric mean level-1']`
- **Income Inequality**: Shows how overall well-being varies by household income
- **Expected Pattern**: Generally higher deciles should show better well-being
- **Reference Line**: Overall country average ("All" deciles) for comparison

#### Radar Chart
- **Data**: `unified_df[Level==2 & Year==latest & Decile=='All']` for each EU Priority
- **Dimensions**: 7 EU Priorities forming the radar axes
- **Multi-Country**: Overlays multiple countries for direct comparison
- **Balance Analysis**: Reveals whether countries excel in all areas or have specific strengths/weaknesses

### Level 2: EU Priority Analysis

#### Map Chart
- **Data**: `unified_df[Level==2 & Year==latest & Decile=='All' & EU_priority==selected]`
- **Computation**: Arithmetic mean of secondary indicators within the selected EU priority
- **Focus**: Geographic patterns for specific policy areas
- **Policy Insight**: Identifies which countries perform best in specific EU priority areas

#### Time Series
- **Data**: `unified_df[Level==2 & EU_priority==selected & Decile=='All']`
- **Policy Tracking**: Evolution of specific EU priority over time
- **Comparative**: Shows how different countries progress in the same policy area
- **Policy Evaluation**: Can reveal impact of policy interventions over time

#### Decile Analysis
- **Data**: `unified_df[Level==2 & Year==latest & EU_priority==selected & Decile!='All']`
- **Equity Analysis**: Whether policy benefits reach all income groups equally
- **Social Justice**: Reveals if EU priorities address inequality or exacerbate it
- **Target Identification**: Shows which income groups need more policy attention

#### Country Comparison
- **Data**: `unified_df[Level==2 & Year==latest & EU_priority==selected & Decile=='All' & Country!='All Countries']`
- **Ranking**: All countries ordered by performance in selected EU priority
- **Best Practices**: Identifies top-performing countries for policy learning
- **Benchmarking**: Shows each country's position relative to others

### Level 3: Secondary Indicator Analysis

#### Map Chart
- **Data**: `unified_df[Level==3 & Year==latest & Decile=='All' & Secondary==selected]`
- **Computation**: Arithmetic mean of primary indicators within the secondary indicator
- **Specificity**: More focused analysis on particular well-being dimensions
- **Targeted Policy**: Useful for specific intervention planning

#### Time Series
- **Data**: `unified_df[Level==3 & Secondary==selected & Decile=='All']`
- **Trend Monitoring**: Detailed tracking of specific well-being dimensions
- **Early Warning**: Can detect emerging problems in specific areas
- **Intervention Assessment**: Evaluates effectiveness of targeted policies

#### Decile Analysis
- **Data**: `unified_df[Level==3 & Year==latest & Secondary==selected & Decile!='All']`
- **Detailed Inequality**: Income-based disparities in specific well-being areas
- **Targeted Support**: Identifies which income groups need support in specific dimensions
- **Root Cause Analysis**: Helps understand mechanisms behind inequality

#### Country Comparison
- **Data**: `unified_df[Level==3 & Year==latest & Secondary==selected & Decile=='All' & Country!='All Countries']`
- **Specialized Ranking**: Countries ranked on specific well-being dimensions
- **Expertise Identification**: Shows which countries excel in particular areas
- **Learning Network**: Facilitates knowledge sharing on specific topics

### Level 5: Raw Statistics Analysis

#### Map Chart
- **Data**: `unified_df[Level==5 & Year==latest & (Decile=='All' | Quintile=='All') & Primary_indicator==selected]`
- **Raw Values**: Normalized statistical values (0-1) processed from original European survey percentages
- **Survey-Specific**: Shows real survey response patterns geographically
- **Ground Truth**: Most granular view of actual conditions across Europe

#### Time Series
- **Data**: `unified_df[Level==5 & Primary_indicator==selected & (Decile=='All' | Quintile=='All')]`
- **Statistical Trends**: Raw statistical evolution over time
- **Survey Continuity**: Shows how survey responses change over time
- **Validation**: Can validate higher-level trends with actual survey data

#### Decile/Quintile Analysis
- **Data Structure**:
  - **Non-EHIS**: `unified_df[Level==5 & Primary_indicator==selected & Decile!='All']`
  - **EHIS**: `unified_df[Level==5 & Primary_indicator==selected & Quintile!='All']`
- **Income Stratification**: How raw problems vary by household income
- **Survey-Level Inequality**: Direct measurement of income-based disparities
- **Methodological**: Shows how different surveys capture income effects

#### Country Comparison
- **Data**: `unified_df[Level==5 & Year==latest & Primary_indicator==selected & (Decile=='All' | Quintile=='All') & Country!='All Countries']`
- **Raw Statistics Ranking**: Countries ordered by actual survey response rates
- **Problem Prevalence**: Shows where specific problems are most/least common
- **Survey Validation**: Cross-validates aggregated indices with survey reality

## Data Processing Pipeline

### Stage 1: Raw Data Collection (Scripts 0_*)
1. **EU-SILC Processing** (`0_raw_indicator_SILC.py`): Income and living conditions
2. **EHIS Processing** (`0_raw_indicator_EHIS.py`): Health interview surveys  
3. **HBS Processing** (`0_raw_indicator_HBS.py`): Household budget surveys
4. **LFS Processing** (`0_raw_indicator_LFS.py`): Labour force surveys

**Output**: Individual processed datasets per survey type with standardized formats.

### Stage 2: Data Integration (Script 1_*)
**File**: `1_final_df.py`
1. **Load** all processed survey datasets
2. **Harmonize** country codes, time periods, and variable names
3. **Standardize** to common structure with decile/quintile information
4. **Create** "All Countries" aggregations using median across countries
5. **Generate** unified Level 5 dataset

**Output**: Complete Level 5 raw statistics dataset.

### Stage 3: Hierarchical Aggregation (Script 3_*)
**File**: `3_generate_outputs.py`
1. **Level 5 → Level 3**: Arithmetic mean of primary indicators within secondary indicators
2. **Level 3 → Level 2**: Arithmetic mean of secondary indicators within EU priorities
3. **Level 2 → Level 1**: Geometric mean of EU priorities to form EWBI
4. **Decile Aggregation**: Create "All" decile summaries using geometric or arithmetic means
5. **Cross-Country Aggregation**: Generate "All Countries" statistics

**Output**: `unified_all_levels_1_to_5.csv` containing all hierarchical levels.

### Stage 4: Dashboard Integration (Script 4_*)
**File**: `4_app_new.py`
1. **Load** unified dataset into Dash application
2. **Create** filtering logic for hierarchical navigation
3. **Generate** dynamic visualizations based on user selections
4. **Handle** different aggregation types (deciles vs quintiles for EHIS)

## Aggregation Methods

### Geometric Mean (Used for Level Aggregation)
**Formula**: `(x₁ × x₂ × ... × xₙ)^(1/n)`
**Used For**: 
- Level 2 → Level 1 (EU Priorities to EWBI)
- Decile aggregation within levels (individual deciles to "All")

**Rationale**: Geometric mean ensures that poor performance in any dimension significantly impacts the overall score, preventing high scores in some areas from completely offsetting low scores in others.

### Arithmetic Mean (Used for Component Aggregation)  
**Formula**: `(x₁ + x₂ + ... + xₙ) / n`
**Used For**:
- Level 5 → Level 3 (Primary to Secondary indicators)
- Level 3 → Level 2 (Secondary to EU Priorities) 
- Cross-country aggregations

**Rationale**: Arithmetic mean provides straightforward averaging suitable for combining conceptually similar indicators within the same domain.

### Median Across Countries
**Used For**: "All Countries" aggregations in time series analysis
**Rationale**: More robust to outliers than arithmetic mean, providing representative European values.

### Weighting System
**Source**: Defined in `ewbi_indicators.json`
**Implementation**: 
- Component weights applied during arithmetic mean calculations
- Ensures important indicators have appropriate influence on aggregated scores
- Reflects expert judgment on relative importance of different well-being dimensions

---

*This documentation provides the complete technical foundation for understanding how the EWBI Dashboard processes, aggregates, and visualizes European well-being data across multiple analytical levels.*