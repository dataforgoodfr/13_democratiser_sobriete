# Master Dataframe Structure

## Overview
The master dataframe combines all levels of the EWBI (European Well-Being Index) indicators into a single comprehensive dataset, enabling hierarchical analysis from the top level (EWBI) down to individual primary indicators.

## Data Structure

### Dimensions
- **Countries**: 32 European countries
- **Deciles**: 10 income deciles (1 = lowest, 10 = highest)
- **Years**: 2004-2023 (latest year varies by indicator)
- **Total Rows**: 320 (32 countries × 10 deciles)

### Column Structure

#### 1. Metadata Columns
- `country`: Country code (AT, BE, BG, CH, CY, CZ, DE, DK, EE, EL, ES, EU, FI, FR, HR, HU, IE, IS, IT, LT, LU, LV, MK, MT, NL, NO, PL, PT, RO, RS, SE, SI, SK, UK)
- `decile`: Income decile (1-10)
- `ewbi_score`: Overall EWBI score (0-1, higher is better)
- `latest_year`: Most recent year with data
- `data_level`: Data completeness indicator

#### 2. EU Priority Level (6 indicators)
- `Agriculture and Food`
- `Energy and Housing`
- `Equality`
- `Health and Animal Welfare`
- `Intergenerational Fairness, Youth, Culture and Sport`
- `Social Rights and Skills, Quality Jobs and Preparedness`

#### 3. Secondary Indicator Level (18 indicators)
- `Agriculture_and_Food_Nutrition_expense`
- `Agriculture_and_Food_Nutrition_need`
- `Energy_and_Housing_Energy`
- `Energy_and_Housing_Housing_quality`
- `Equality_Community`
- `Equality_Life_satisfaction`
- `Equality_Security`
- `Health_and_Animal_Welfare_Accidents_and_addictive_behaviour`
- `Health_and_Animal_Welfare_Health_condition_and_impact`
- `Intergenerational_Fairness_Youth_Culture_and_Sport_Education`
- `Social_Rights_and_Skills_Quality_Jobs_and_Preparedness_Type_of_job_and_market_participation`
- `Social_Rights_and_Skills_Quality_Jobs_and_Preparedness_Unemployment`

#### 4. Primary Indicator Level (58 indicators)
- `primary_AB-EHIS-1`: Daily smoking
- `primary_AB-EHIS-2`: Daily alcohol consumption
- `primary_AB-EHIS-3`: Road traffic accidents
- `primary_AC-EHIS-1`: Cannot afford prescribed medicines
- `primary_AC-HBS-1`: Health expenditure above median
- `primary_AC-HBS-2`: Health expenditure below median
- `primary_AC-SILC-1`: Cannot afford medical examination
- `primary_AC-SILC-2`: Unmet medical need
- `primary_AE-EHIS-1`: Not eating fruit weekly
- `primary_AE-HBS-1`: Food expenditure above median
- `primary_AE-HBS-2`: Food expenditure below median
- `primary_AH-EHIS-1`: Feeling down/depressed
- `primary_AH-EHIS-2`: No physical activity
- `primary_AH-SILC-1`: Bad self-perceived health
- `primary_AH-SILC-2`: Chronic illness
- `primary_AH-SILC-3`: Health-related limitations
- `primary_AH-SILC-4`: Unable to work due to health
- `primary_AN-EHIS-1`: Difficulty preparing meals
- `primary_AN-SILC-1`: Cannot afford meat/fish every second day
- `primary_EC-EHIS-1`: No close people to count on
- `primary_EC-HBS-1`: Communication expenditure above median
- `primary_EC-HBS-2`: Communication expenditure below median
- `primary_EC-SILC-1`: Cannot meet friends/family monthly
- `primary_EC-SILC-2`: Do not trust others
- `primary_ED-EHIS-1`: Difficulty using telephone
- `primary_EL-EHIS-1`: Little interest/pleasure in activities
- `primary_EL-SILC-1`: Not satisfied with life
- `primary_ES-SILC-1`: Cannot face unexpected expenses
- `primary_ES-SILC-2`: Making ends meet with difficulty
- `primary_HE-HBS-1`: Housing/energy expenditure above median
- `primary_HE-HBS-2`: Housing/energy expenditure below median
- `primary_HE-SILC-1`: Cannot keep home warm
- `primary_HE-SILC-2`: Arrears on utility bills
- `primary_HH-HBS-1`: Rent expenditure above median
- `primary_HH-HBS-2`: Rent expenditure below median
- `primary_HH-SILC-1`: Arrears on mortgage/rent
- `primary_HQ-SILC-1`: Over-populated dwelling
- `primary_HQ-SILC-2`: Cannot replace worn furniture
- `primary_IC-HBS-1`: Recreation/culture expenditure above median
- `primary_IC-HBS-2`: Recreation/culture expenditure below median
- `primary_IC-SILC-1`: Cannot participate in leisure activities
- `primary_IC-SILC-2`: Cannot spend money on self weekly
- `primary_IE-HBS-1`: Education expenditure above median
- `primary_IE-HBS-2`: Education expenditure below median
- `primary_IS-SILC-3`: No formal education
- `primary_RT-LFS-1`: Multiple jobs
- `primary_RT-LFS-2`: Wish to work more hours
- `primary_RT-LFS-3`: Overtime/extra hours
- `primary_RT-SILC-1`: Fixed-term contract
- `primary_RT-SILC-2`: Part-time contract
- `primary_RU-SILC-1`: Unemployed for 6+ months
- `primary_TS-HBS-1`: Travel/holiday expenditure above median
- `primary_TS-HBS-2`: Travel/holiday expenditure below median
- `primary_TS-SILC-1`: Cannot afford annual holiday
- `primary_TT-HBS-1`: Transport expenditure above median
- `primary_TT-HBS-2`: Transport expenditure below median
- `primary_TT-SILC-1`: Cannot afford public transport
- `primary_TT-SILC-2`: Low access to public transport

## Hierarchical Relationships

### Level 1: EWBI
- **Single indicator**: Overall well-being score
- **Composition**: Geometric mean of all EU priorities

### Level 2: EU Priorities
- **6 categories**: Each represents a major policy area
- **Composition**: Weighted average of secondary indicators
- **Weights**: Vary by priority (e.g., Agriculture: 2/3 Nutrition need + 1/3 Nutrition expense)

### Level 3: Secondary Indicators
- **18 categories**: Each represents a specific well-being dimension
- **Composition**: Simple average of primary indicators
- **Mapping**: Each secondary indicator belongs to one EU priority

### Level 4: Primary Indicators
- **58 indicators**: Individual survey questions and expenditure measures
- **Composition**: Raw normalized values (0-1 scale)
- **Mapping**: Each primary indicator belongs to one secondary indicator

## Data Values

### Scale
- **Range**: 0-1 for all indicators
- **Interpretation**: 
  - 0 = Worst performance (highest deprivation/cost)
  - 1 = Best performance (lowest deprivation/cost)
- **Normalization**: Intra-decile and intra-indicator normalization

### Missing Values
- **Strategy**: Forward-fill then backward-fill
- **Rationale**: EU JRC methodology prefers continuity over ignoring indicators

## Usage for Visualization

### 1. Country Comparison
- **Purpose**: Compare countries across selected indicators
- **Data**: Latest year, all deciles combined
- **Levels**: Can show EWBI, EU priorities, secondary indicators, or primary indicators

### 2. Annual Evolution
- **Purpose**: Track changes over time for selected country
- **Data**: All years, all deciles combined
- **Levels**: Can show trend for any indicator level

### 3. Performance per Decile
- **Purpose**: Analyze inequality within countries
- **Data**: Latest year, by decile
- **Levels**: Can show distribution across any indicator level

### 4. Hierarchical Drill-Down
- **Level 1**: EWBI + 6 EU Priorities
- **Level 2**: 1 EU Priority + its Secondary Indicators
- **Level 3**: 1 Secondary Indicator + its Primary Indicators

## File Formats

### Cross-Sectional View
- **File**: `master_dataframe.csv`
- **Structure**: One row per country-decile combination
- **Use**: Country comparisons, decile analysis

### Time Series View
- **File**: `master_dataframe_time_series.csv`
- **Structure**: One row per country-decile-indicator-year combination
- **Use**: Trend analysis, time evolution

## Data Quality Notes

- **Coverage**: Not all indicators available for all countries
- **Years**: Data availability varies by indicator and country
- **Deciles**: Some indicators use quintiles (converted to deciles)
- **Weights**: Economic indicators are filtered out (only satisfiers included)

## Example Queries

### Get EWBI scores for all countries (latest year, all deciles)
```python
df = pd.read_csv('master_dataframe.csv')
ewbi_scores = df.groupby('country')['ewbi_score'].mean()
```

### Get secondary indicators for Energy and Housing priority
```python
energy_housing_cols = [col for col in df.columns if col.startswith('Energy_and_Housing_')]
energy_housing_data = df[['country', 'decile'] + energy_housing_cols]
```

### Get primary indicators for Nutrition need
```python
nutrition_primary_cols = [col for col in df.columns if 'Nutrition' in col and col.startswith('primary_')]
nutrition_data = df[['country', 'decile'] + nutrition_primary_cols]
``` 