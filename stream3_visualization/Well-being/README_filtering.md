# EWBI Economic Indicator Filtering and Recalculation

This document explains how to filter out economic indicators from the EWBI analysis and recalculate the secondary indicators using simple averages instead of weighted geometric means.

## Overview

The changes requested are:
1. **Filter out economic indicators** marked as "ECONOMIC GOOD" in the `EWBI data cuts.xlsx` file
2. **Recalculate secondary indicators** using straight averages of remaining indicators instead of weighted geometric means

## Files Created

- `filter_economic_indicators.py` - Main script to automatically filter and recalculate
- `manual_economic_filter.py` - Helper script for manual identification and filtering
- `README_filtering.md` - This documentation

## Step-by-Step Process

### Step 1: Identify Economic Indicators

First, you need to identify which indicators are marked as "ECONOMIC GOOD" in the Excel file.

**Option A: Automatic Detection (Recommended)**
```bash
cd Well-being/code
python filter_economic_indicators.py
```

This script will attempt to read the Excel file and automatically identify economic indicators.

**Option B: Manual Identification**
```bash
cd Well-being/code
python manual_economic_filter.py
# Choose option 1 to list all indicators
```

This will create `all_indicators_list.csv` and `potential_economic_indicators.csv` in the output directory to help you identify economic indicators.

### Step 2: Filter and Recalculate

**Option A: Automatic (if Excel reading worked)**
The `filter_economic_indicators.py` script should have already completed the filtering and recalculation.

**Option B: Manual (if Excel reading failed)**
1. Edit `manual_economic_filter.py` and add the economic indicator codes to the `economic_indicators` list
2. Run the script again:
```bash
python manual_economic_filter.py
# Choose option 2 to filter with manual list
```

## What Gets Recalculated

### 1. Secondary Indicators
- **Before**: Weighted geometric mean of indicators within each component
- **After**: Simple arithmetic average of remaining indicators within each component

### 2. Priorities
- **Before**: Weighted geometric mean of components within each priority
- **After**: Simple arithmetic average of components within each priority

### 3. Final EWBI Scores
- **Before**: Weighted geometric mean of priorities
- **After**: Simple arithmetic average of priorities

## Output Files

All filtered outputs are saved with `_filtered` suffix:
- `primary_data_preprocessed_filtered.csv` - Primary data with economic indicators removed
- `secondary_indicators_filtered.csv` - Recalculated secondary indicators
- `eu_priorities_filtered.csv` - Recalculated priorities
- `ewbi_results_filtered.csv` - Final filtered EWBI scores
- `missing_indicators_filtered.csv` - List of missing indicators after filtering

## Example Economic Indicators

Based on the indicator descriptions, these are likely to be economic indicators:
- **AE-HBS-1, AE-HBS-2**: Food expenditure shares
- **HE-HBS-1, HE-HBS-2**: Housing/energy expenditure shares
- **HH-HBS-1, HH-HBS-2**: Housing expense shares
- **AC-HBS-1, AC-HBS-2**: Health cost shares
- **IE-HBS-1, IE-HBS-2**: Education expense shares
- **TT-HBS-1, TT-HBS-2**: Transport expense shares

## Verification

After running the scripts, verify that:
1. Economic indicators are removed from the filtered primary data
2. Secondary indicators are recalculated using simple averages
3. Final EWBI scores reflect the new methodology
4. All output files are generated with the expected structure

## Troubleshooting

### Excel File Reading Issues
If the automatic Excel reading fails:
1. Check that `EWBI data cuts.xlsx` exists in the data directory
2. Verify the sheet name is "Indicators to keep"
3. Use the manual identification approach instead

### Missing Data Issues
If some indicators are missing after filtering:
1. Check the `missing_indicators_filtered.csv` file
2. Verify that the economic indicators list is correct
3. Ensure all required indicators are available in the primary data

### Calculation Issues
If the recalculation fails:
1. Check that `primary_data_preprocessed.csv` exists
2. Verify the indicator configuration in `ewbi_indicators.json`
3. Check the console output for specific error messages

## Next Steps

After successful filtering and recalculation:
1. Update the dashboard to use the filtered results
2. Compare filtered vs. original results to understand the impact
3. Consider updating the methodology documentation
4. Validate that the new approach aligns with research objectives 