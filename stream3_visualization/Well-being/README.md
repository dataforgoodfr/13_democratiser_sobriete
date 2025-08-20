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
3. **Level 3**: Secondary Indicators - 18 specific well-being dimensions
4. **Level 4**: Primary Indicators - 58 individual survey questions and measures

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
1. **Primary Indicators**: Raw survey data normalized to 0-1 scale
2. **Secondary Indicators**: Arithmetic mean of constituent primary indicators
3. **EU Priorities**: Arithmetic mean of constituent secondary indicators
4. **EWBI**: Arithmetic mean of all EU priority scores

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
2. Regenerate data using `ewbi_computation.py`
3. Dashboard automatically adapts to new structure

### Modifying Visualizations
- Chart colors and styling in individual chart functions
- Layout modifications in `app.layout`
- New chart types can be added to existing functions

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