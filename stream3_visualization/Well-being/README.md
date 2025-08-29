# European Well-Being Index (EWBI) Dashboard

## Overview

This folder contains a comprehensive dashboard for analyzing the European Well-Being Index (EWBI), a composite indicator measuring well-being across European countries. The dashboard provides multi-level analysis capabilities across 4 hierarchical levels of well-being indicators.

## 🏗️ Architecture

### Data Structure
The dashboard uses a **pre-calculated aggregated data structure** to ensure fast performance and avoid real-time calculations:

- `output/ewbi_master.csv`: Latest year for all levels and all deciles (including EU Average and countries)
- `output/ewbi_time_series.csv`: Historical time series (decile = All) for all years, levels, EU Average and countries
- `data/ewbi_indicators.json`: Configuration file defining the hierarchical structure

### Hierarchical Levels
1. **Level 1**: EWBI - Overall well-being score
2. **Level 2**: EU Priorities - 6 major policy areas
3. **Level 3**: Secondary Indicators - 19 specific well-being dimensions
4. **Level 4**: Primary Indicators - 27 individual survey questions and measures (satisfier indicators only)

## 🚀 Quick Start

### Prerequisites
```bash
pip install dash pandas plotly
```

### Running the Dashboard
```bash
cd Well-being/code
python ewbi_dashboard.py
```

The dashboard will be available at `http://localhost:8050`

### Data Generation
To regenerate the aggregated data files from raw inputs:

1) Preprocess primary indicator data (Level 4) from raw CSV
   - Open `Well-being/code/preprocessing_executed.ipynb`
   - Run all cells
   - Output: `Well-being/output/primary_data_preprocessed.csv`

2) Generate master and time series outputs using unified aggregation logic
```bash
cd Well-being/code
python generate_outputs.py
```
Outputs:
- `Well-being/output/ewbi_master.csv`
- `Well-being/output/ewbi_time_series.csv`

## 📊 Dashboard Features

### Interactive Controls
- **EU Priority Dropdown**: Select specific policy areas or view all
- **Secondary Indicator Dropdown**: Drill down to specific dimensions
- **Primary Indicator Dropdown**: Access individual survey measures
- **Country Filter**: Focus on specific countries or aggregates

### Visualization Components
Each level provides 4 complementary charts:
1. **European Map**: Choropleth visualization of scores across countries
2. **Decile Analysis**: Income-based breakdown of scores
3. **Country Comparison**: Cross-country ranking and comparison
4. **Time Series**: Historical evolution of indicators

### Data Aggregation Methods
- **Decile to Country**: Geometric mean (for primary indicators)
- **Level to Level**: Arithmetic mean (for hierarchical aggregation)
- **Country Aggregates**: EU Countries Average, All Countries Average

## 📁 File Structure

```
Well-being/
├── code/
│   ├── ewbi_dashboard.py               # Main dashboard application
│   ├── generate_outputs.py             # Unified data aggregation for master + time series
│   └── preprocessing_executed.ipynb    # Preprocessing notebook for primary indicators
├── data/
│   └── ewbi_indicators.json            # EWBI structure configuration
├── output/
│   ├── ewbi_master.csv                 # Latest year, all deciles
│   ├── ewbi_time_series.csv            # All years, decile = All
│   ├── MASTER_DATAFRAME_STRUCTURE.md   # Data structure documentation
│   └── Archive/                        # Archived intermediate/backup files
└── README.md                           # This file
```

## 🔧 Technical Details

### Data Sources
- **EU-SILC**: European Union Statistics on Income and Living Conditions
- **EU-EHIS**: European Health Interview Survey
- **HBS**: Household Budget Survey
- **LFS**: Labour Force Survey

### Aggregation Logic
1. **Primary Indicators (L4)**: Raw survey data normalized to 0–1 scale for each decile (1, 2, 3, ..., 10)
2. **Decile Aggregation (to "All")**: 
   - **L4 "All Deciles"**: Geometric mean across deciles (1-10) for each primary indicator
   - **L3 "All Deciles"**: Geometric mean of L3 individual decile scores for each secondary indicator
   - **L2 "All Deciles"**: Arithmetic mean of L2 individual decile scores for each EU priority
   - **L1 "All Deciles"**: Arithmetic mean of L1 individual decile scores for EWBI
3. **Level Roll-up (L4 → L3 → L2 → L1)**: 
   - **L4 → L3**: Arithmetic mean of primary indicators within each secondary indicator
   - **L3 → L2**: Arithmetic mean of secondary indicators within each EU priority
   - **L2 → L1**: Arithmetic mean of EU priorities to form EWBI
4. **EU Average**: Arithmetic mean across EU countries for each year/level/decile combination

### Complete Indicator Hierarchy

#### Level 1: EWBI (Overall Well-being)
- **EWBI**: Composite score across all EU Priorities

#### Level 2: EU Priorities (6 major policy areas)
1. **Agriculture and Food**
2. **Energy and Housing**
3. **Equality**
4. **Health and Animal Welfare**
5. **Intergenerational Fairness, Youth, Culture and Sport**
6. **Social Rights and Skills, Quality Jobs and Preparedness**
7. **Sustainable Transport and Tourism**

#### Level 3: Secondary Indicators (18 dimensions)
- **Agriculture and Food**: Nutrition need, Nutrition expense
- **Energy and Housing**: Housing quality, Energy, Housing expense
- **Equality**: Life satisfaction, Security, Community, Digital Skills
- **Health and Animal Welfare**: Health condition and impact, Health cost and medical care, Accidents and addictive behaviour
- **Intergenerational Fairness, Youth, Culture and Sport**: Education, Education expense, Leisure and culture
- **Social Rights and Skills, Quality Jobs and Preparedness**: Type of job and market participation, Unemployment
- **Sustainable Transport and Tourism**: Transport, Tourism

#### Level 4: Primary Indicators (58 individual measures)
- **AN-SILC-1**: Share of nth decile households in capacity to afford a meal with meat, chicken, fish or vegetarian equivalent every second day
- **AN-EHIS-1**: Share of nth quintile population facing a lot of difficulty in preparing meals
- **AE-HBS-1**: Percentage of households of the nth decile whose share of equivalised food and nonalcoholic beverages expenditure in equivalised disposable income is above twice the national median share
- **AE-HBS-2**: Percentage of households of the nth decile whose share of equivalised food and nonalcoholic beverages expenditure in equivalised disposable income is below half the national median share
- **AE-EHIS-1**: Share of nth quintile population not eating fruit, excluding juice less than once a week
- **HQ-SILC-1**: Share of nth decile households living in an over-populated dwelling
- **HQ-SILC-2**: Share of nth decile households incapable to replace worn-out furniture
- **HE-SILC-1**: Share of nth decile households unable to keep their homes warm in winter
- **HE-SILC-2**: Share of nth decile households having arrears on utility bills
- **HE-HBS-1**: Percentage of households of the nth decile whose share of equivalised housing, water, electricity, gas and other fuels expenditure in equivalised disposable income is above twice the national median share
- **HE-HBS-2**: Percentage of households of the nth decile whose share of equivalised housing, water, electricity, gas and other fuels expenditure in equivalised disposable income is below half the national median share
- **HH-SILC-1**: Share of nth decile households having arrears on mortgage or rental payments
- **HH-HBS-1**: Percentage of households of the nth decile whose share of equivalised rent related to the occupied dwelling expenditure in equivalised disposable income is above twice the national median share
- **HH-HBS-2**: Percentage of households of the nth decile whose share of equivalised rent related to the occupied dwelling expenditure in equivalised disposable income is below half the national median share
- **EL-SILC-1**: Share of nth decile households not satisfied with its life
- **EL-EHIS-1**: Share of nth quintile population that experience every day little interest or pleasure in doing things over the last 2 weeks
- **ES-SILC-1**: Share of nth decile households in capacity to face unexpected financial expenses
- **ES-SILC-2**: Share of nth decile households making ends meet with difficulty
- **EC-SILC-1**: Share of nth decile households incapable of getting-together with friends/family for a drink/meal at least once a month
- **EC-SILC-2**: Share of nth decile households not trusting others
- **EC-HBS-1**: Percentage of households of the nth decile whose share of equivalised communication expenditure in equivalised disposable income is above twice the national median share
- **EC-HBS-2**: Percentage of households of the nth decile whose share of equivalised communication expenditure in equivalised disposable income is below half the national median share
- **EC-EHIS-1**: Share of nth quintile population having no close people to count on in case of serious personal problems
- **ED-EHIS-1**: Share of nth quintile population having difficulty in using the telephone
- **AH-SILC-1**: Share of nth decile households having bad self-perceived general health
- **AH-SILC-2**: Share of nth decile households suffering from any chronic illness or condition
- **AH-SILC-3**: Share of nth decile households encountering limitation in activities because of health problems
- **AH-SILC-4**: Share of nth decile households unable to work for 6 months due to long-standing health problems
- **AH-EHIS-1**: Share of nth quintile population that are every day feeling down, depressed or hopeless over the last 2 weeks
- **AH-EHIS-2**: Share of nth quintile population that are never carrying out sports, fitness or recreational (leisure) physical activities
- **AC-SILC-1**: Share of nth decile households that could not afford medical examination or treatment
- **AC-SILC-2**: Share of nth decile households having unmet need for medical examination or treatment
- **AC-HBS-1**: Percentage of households of the nth decile whose share of equivalised health expenditure in equivalised disposable income is above twice the national median share
- **AC-HBS-2**: Percentage of households of the nth decile whose share of equivalised health expenditure in equivalised disposable income is below half the national median share
- **AC-EHIS-1**: Share of nth quintile population that could not afford prescribed medicines in the past 12 months
- **AB-EHIS-1**: Share of nth quintile population daily smoking
- **AB-EHIS-2**: Share of nth quintile population consuming alcoholic drink of any kind nearly every day in the past 12 months
- **AB-EHIS-3**: Share of nth quintile population that suffered a road traffic accident in the past 12 months
- **IS-SILC-3**: Share of nth decile households having no formal education
- **IE-HBS-1**: Percentage of households with parents of the nth decile whose share of equivalised education expenditure in equivalised disposable income is above twice the national median share
- **IE-HBS-2**: Percentage of households with parents of the nth decile whose share of equivalised education expenditure in equivalised disposable income is below half the national median share
- **IC-SILC-1**: Share of nth decile households incapable of regularly participating in a leisure activity
- **IC-SILC-2**: Share of nth decile households incapable of spending a small amount of money each week on yourself
- **IC-HBS-1**: Percentage of households of the nth decile whose share of equivalised recreation, and culture expenditure in equivalised disposable income is above twice the national median share
- **IC-HBS-2**: Percentage of households of the nth decile whose share of equivalised recreation, and culture expenditure in equivalised disposable income is below half the national median share
- **RT-SILC-1**: Share of nth decile households over 18 having a fixed-term contract
- **RT-SILC-2**: Share of nth decile households over 18 having a part-time contract
- **RT-LFS-1**: Share of nth decile households engage in two or more jobs
- **RT-LFS-2**: Share of nth decile households wishing to work more than the current number of usual hours
- **RT-LFS-3**: Share of nth decile households doing some overtime or extra hours worked in main job
- **RU-SILC-1**: Share of nth decile households over 18 years old unemployed for the last 6 months
- **TT-HBS-1**: Percentage of households of the nth decile whose share of equivalised transport expenditure in equivalised disposable income is above twice the national median share
- **TT-HBS-2**: Percentage of households of the nth decile whose share of equivalised transport expenditure in equivalised disposable income is below half the national median share
- **TT-SILC-1**: Share of nth quintile households incapable to afford public transport
- **TT-SILC-2**: Share of nth quintile households facing very low access to public transport
- **TS-SILC-1**: Share of nth decile households incapable to afford a one-week annual holiday away from home
- **TS-HBS-1**: Percentage of households of the nth decile whose share of equivalised travelling and holidays abroad expenditure in equivalised disposable income is above twice the national median share
- **TS-HBS-2**: Percentage of households of the nth decile whose share of equivalised travelling and holidays abroad expenditure in equivalised disposable income is below half the national median share

### Performance Optimizations
- Pre-calculated aggregates eliminate real-time computation
- Wide-format data structure for efficient column access
- Optimized chart rendering with Plotly
- Responsive design for various screen sizes

## 📈 Usage Examples

### Viewing Overall Well-being
1. Select "ALL" for EU Priority
2. Dashboard shows EWBI overview with all countries

### Analyzing Specific Policy Areas
1. Select an EU Priority (e.g., "Agriculture and Food")
2. View secondary indicators within that priority
3. Drill down to specific primary indicators

### Comparing Countries
1. Use country filter to select specific nations
2. Compare scores across different income deciles
3. Analyze historical trends over time

## 🛠️ Customization

### Adding New Indicators
1. Update `ewbi_indicators.json` with new structure
2. Regenerate data using `generate_outputs.py`
3. Dashboard automatically adapts to new structure

### Modifying Visualizations
- Chart colors and styling in individual chart functions
- Layout modifications in `app.layout`
- New chart types can be added to existing functions

## 💻 Local Development

### Running the Dashboard Locally
The dashboard can be run locally for development and testing:

```bash
cd Well-being/code
python app.py
```

### Local Access
- **Dashboard URL**: http://localhost:8050
- **Network Access**: http://0.0.0.0:8050 (for other devices on your network)
- **Development Mode**: Debug mode enabled for easier development

### Development Features
- **Hot Reload**: Code changes automatically refresh the dashboard
- **Debug Information**: Detailed error messages and logging
- **Local Data**: Uses local CSV files for development

### Start Script
Use the main start script to run all dashboards simultaneously:
```bash
./start_dashboards.sh
```
This will start:
- Budget Dashboard on port 8052
- Decomposition Dashboard on port 8051  
- Well-being Dashboard on port 8050

## 🔍 Troubleshooting

### Common Issues
- **Port conflicts**: Change port in dashboard script if 8052 is busy
- **Data not loading**: Ensure CSV files are in correct location
- **Charts not displaying**: Check browser console for JavaScript errors

### Performance Issues
- Large datasets may slow initial loading
- Consider filtering data for specific use cases
- Monitor memory usage with large country selections

## 📚 References

- **EWBI Methodology**: Based on EU JRC recommendations
- **Data Sources**: Eurostat and national statistical offices
- **Visualization**: Built with Dash and Plotly
- **Data Processing**: Pandas for data manipulation and aggregation

## 🚀 Deployment

This application is deployed on CleverCloud as a standalone service.

### Deployment Files
The dashboard is deployed directly from the main repository with the following key files:
- `code/app.py` - Main dashboard application
- `code/requirements.txt` - Python dependencies
- `data/ewbi_indicators.json` - Indicator definitions and structure
- `output/ewbi_master.csv` - Main data file
- `output/ewbi_time_series.csv` - Time series data

### Quick Deploy
After making changes to the code, deploy updates to Clever Cloud using Git:

```bash
# 1. Commit your changes
git add .
git commit -m "Description of your changes"

# 2. Push to GitHub (optional but recommended)
git push origin visualizations-combined

# 3. Push to Clever Cloud for automatic deployment
git push clever-well-being visualizations-combined:master
```

**Note:** The Clever Cloud remote is configured as `clever-well-being`. After pushing, Clever Cloud will automatically redeploy your application with the new changes.

### Recent Improvements (August 2024)
- **Directory Cleanup**: Removed 15+ outdated files and Archive directory
- **Indicator Names**: Updated primary indicators to use user-friendly descriptions
- **Dashboard Enhancement**: Modified interface to display descriptive names instead of codes
- **Start Script**: Updated to use correct filename (`app.py`)
- **Local Development**: Added local port information for easier development

### Indicator Naming Convention
Primary indicators now display in the format: `"Proposed Name (Code)"`
- **Before**: `"AN-EHIS-1"`
- **After**: `"Struggling to Prepare Meals (AN-EHIS-1)"`

This makes the dashboard much more user-friendly and accessible to non-technical users.

## 🤝 Contributing

When making changes:
1. Test dashboard functionality thoroughly
2. Update documentation for new features
3. Commit changes with descriptive messages
4. Ensure data integrity is maintained

## 📞 Support

For technical issues or questions about the dashboard:
1. Check this README first
2. Review the code comments
3. Check Git commit history for recent changes
4. Ensure all dependencies are properly installed 