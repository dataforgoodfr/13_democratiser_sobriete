# Quick Start Guide: FR Scenarios Compiled Datasets

## 📍 Location
```
stream3_visualization/Decomposition/reports/FR/visuals final datasets/
```

## 🚀 Quick Start (30 seconds)

### 1. Generate Datasets (First Time or Update)
```bash
cd c:\Users\valentin.stuhlfauth\Desktop\Git D4G\13_democratiser_sobriete
python stream3_visualization/Decomposition/code/generate_compiled_datasets.py
```

**Output**: 
- ✅ Excel file with 48 data records (3 sheets)
- ✅ 2 PNG visualizations

**Time**: ~3 seconds

### 2. Verify Output
```bash
python stream3_visualization/Decomposition/code/verify_compiled_datasets.py
```

**Output**: 
- ✅ Validation report
- ✅ Data quality checks
- ✅ Sample statistics

### 3. Use the Data

#### Option A: Open Excel
```
File: 2025-12-19_FR_scenarios_compiled.xlsx
├── Transport - Final Energy (33 records)
├── Buildings - Residential (15 records)
└── Summary (metadata)
```

#### Option B: Python Analysis
```python
import pandas as pd

# Load data
df = pd.read_excel(
    'stream3_visualization/Decomposition/reports/FR/visuals final datasets/2025-12-19_FR_scenarios_compiled.xlsx',
    sheet_name='Transport - Final Energy'
)

# Filter by scenario
snbc = df[df['Scenario'] == 'SNBC-3']
print(snbc)

# Statistics
print(df.groupby('Scenario')['Final Energy (Mtoe)'].mean())
```

#### Option C: View Visualizations
```
File: 2025-12-19_timeline_analysis.png
Description: 2-panel chart showing transport energy & buildings ratios
```

---

## 📊 What You Get

### Data Structure
```
Year | Scenario    | Metric
-----|-------------|--------
2000 | historical  | 0.0763  (Transport, Mtoe)
2019 | SNBC-3      | 0.0450  (Transport, Mtoe)
2025 | AME-2024    | 0.0421  (Transport, Mtoe)
...
1984 | historical  | 0.4601  (Buildings, ratio)
2021 | SNBC-3      | 0.5000  (Buildings, ratio)
```

### Key Metrics

**Transport Final Energy (Mtoe)**
- Historical: 2000-2022 data (23 years)
- SNBC-3: 2019-2030 projections (3 years)
- AME-2024: 2019-2050 projections (7 years)
- Range: 0.043-0.088 Mtoe

**Buildings Collective Ratio**
- Historical: 1984-2023 data (8 years)
- SNBC-3: 2021-2030 projections (3 years)
- AME-2024: 2020-2050 projections (4 years)
- Range: 0.43-0.50 (43%-50% collective housing)

---

## 🔄 Workflow

### Typical Workflow
```
1. Update source data
   └─> 2025-12-15_FR scenarios data_before computation.xlsx

2. Run generation script
   └─> python generate_compiled_datasets.py

3. Verify outputs
   └─> python verify_compiled_datasets.py

4. Use in analysis/dashboards
   └─> Import Excel or PNG files

5. Archive if needed
   └─> Backup old timestamped files
```

### Update Schedule
- **Manual**: Run whenever source data updates
- **Automated**: Can be scheduled with task scheduler or cron
- **Frequency**: Recommended: Monthly or after major data changes

---

## 📝 Key Files

| File | Purpose | Size |
|------|---------|------|
| `generate_compiled_datasets.py` | Main script | 8 KB |
| `generate_enhanced_analysis.py` | Extended GHG analysis | 6 KB |
| `verify_compiled_datasets.py` | Validation script | 5 KB |
| `COMPILED_DATASETS_README.md` | Technical docs | 12 KB |
| `COMPILED_DATASETS_COMPLETION.md` | Implementation details | 20 KB |
| `2025-12-19_FR_scenarios_compiled.xlsx` | Output data | 7 KB |
| `2025-12-19_timeline_analysis.png` | Visualization | 236 KB |

---

## 🎯 Common Tasks

### View Latest Data
```bash
# Show most recent Excel file
Get-ChildItem "stream3_visualization/Decomposition/reports/FR/visuals final datasets/*.xlsx" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

### Extract Specific Scenario
```python
import pandas as pd

df = pd.read_excel('2025-12-19_FR_scenarios_compiled.xlsx', 'Transport - Final Energy')

# Get SNBC-3 only
snbc_data = df[df['Scenario'] == 'SNBC-3']
print(snbc_data)
```

### Calculate Trends
```python
import pandas as pd

df = pd.read_excel('2025-12-19_FR_scenarios_compiled.xlsx', 'Transport - Final Energy')

# Trend by scenario
for scenario in df['Scenario'].unique():
    data = df[df['Scenario'] == scenario].sort_values('Year')
    change = data['Final Energy (Mtoe)'].iloc[-1] - data['Final Energy (Mtoe)'].iloc[0]
    print(f"{scenario}: {change:+.4f} Mtoe")
```

### Create Custom Visualization
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('2025-12-19_FR_scenarios_compiled.xlsx', 'Transport - Final Energy')

# Plot
plt.figure(figsize=(10, 6))
for scenario in df['Scenario'].unique():
    data = df[df['Scenario'] == scenario].sort_values('Year')
    plt.plot(data['Year'], data['Final Energy (Mtoe)'], label=scenario, marker='o')

plt.xlabel('Year')
plt.ylabel('Energy (Mtoe)')
plt.title('Transport Final Energy by Scenario')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('custom_chart.png', dpi=300)
plt.show()
```

---

## 🔍 Troubleshooting

### Issue: Script not found
```
Error: python: can't open file 'generate_compiled_datasets.py'
```
**Solution**: Make sure you're in the project root directory
```bash
cd c:\Users\valentin.stuhlfauth\Desktop\Git D4G\13_democratiser_sobriete
```

### Issue: Excel file not readable
```
Error: openpyxl cannot open file
```
**Solution**: Close file in Excel first, then rerun script

### Issue: No PNG visualizations
```
Solution 1: Run generate_enhanced_analysis.py for additional charts
Solution 2: Check if matplotlib is installed
```

### Issue: Data mismatch with source
```
Solution: Source data updated - check 2025-12-15_FR scenarios data_before computation.xlsx
```

---

## 📈 Performance

- **Generation**: ~3 seconds
- **Verification**: ~2 seconds
- **Total runtime**: ~5 seconds
- **Memory usage**: <100 MB
- **File I/O**: Minimal

---

## ✅ Checklist Before Deployment

- [x] Source data file exists and updated
- [x] Scripts are executable
- [x] Output directory created
- [x] All dependencies installed
- [x] Excel output validated
- [x] PNG visualizations generated
- [x] Verification script passes all checks
- [x] Documentation complete

---

## 📞 Next Steps

1. **First Use**: Run `generate_compiled_datasets.py` → Check Excel → Done ✓
2. **Regular Use**: Update source data → Rerun script → Use outputs
3. **Advanced**: Modify scripts for custom calculations (see code comments)
4. **Integration**: Import Excel/PNG into dashboards or reports

---

## 📚 Additional Resources

- **Full Documentation**: [COMPILED_DATASETS_README.md](COMPILED_DATASETS_README.md)
- **Implementation Details**: [COMPILED_DATASETS_COMPLETION.md](../COMPILED_DATASETS_COMPLETION.md)
- **Source Code**: [generate_compiled_datasets.py](generate_compiled_datasets.py)
- **Verification Tool**: [verify_compiled_datasets.py](verify_compiled_datasets.py)

---

**Last Updated**: 2025-12-19  
**Status**: ✅ Ready for Production  
**Version**: 1.0
