import pandas as pd
import numpy as np

# Load the final level 5 data
print("Loading level 5 data...")
df = pd.read_csv(r"C:\Users\valentin.stuhlfauth\OneDrive - univ-lyon2.fr\Bureau\git-Data4Good\13_democratiser_sobriete\stream3_visualization\Well-being\output\level4_level5_unified_data.csv")

# Filter for the specific indicators we're investigating
print("\nAnalyzing HE-SILC-2 (Behind on Utility Bills)...")
he_silc_2 = df[df['Primary and raw data'] == 'HE-SILC-2'].copy()

# Look at "All" decile data to avoid summing individual deciles
print("Years with all zeros for HE-SILC-2:")
he_all_data = he_silc_2[he_silc_2['Decile'] == 'All'].copy()
he_summary = he_all_data.groupby(['Year', 'Country'])['Value'].mean().reset_index()
zeros_by_year = he_summary[he_summary['Value'] == 0].groupby('Year').size()
print(zeros_by_year)

print("\nGermany HE-SILC-2 data by year (All decile only):")
germany_he = he_summary[he_summary['Country'] == 'DE'].sort_values('Year')
print(germany_he)

print("\nAnalyzing HQ-SILC-1 (Overcrowded dwelling)...")
hq_silc_1 = df[df['Primary and raw data'] == 'HQ-SILC-1'].copy()

print("\nGermany HQ-SILC-1 data by year (All decile only):")
hq_all_data = hq_silc_1[hq_silc_1['Decile'] == 'All'].copy()
hq_summary = hq_all_data.groupby(['Year', 'Country'])['Value'].mean().reset_index()
germany_hq = hq_summary[hq_summary['Country'] == 'DE'].sort_values('Year')
print(germany_hq)

print("\nDenmark HQ-SILC-1 data by year (All decile only):")
denmark_hq = hq_summary[hq_summary['Country'] == 'DK'].sort_values('Year')
print(denmark_hq)

print("\nSummary of data issues:")
print("=" * 50)
print("1. HE-SILC-2 (Behind on Utility Bills):")
print("   - All countries show 0 for 2004-2007")
print("   - Germany shows 0 from 2004-2010") 
print("   - This is due to different value coding before/after 2008")

print("\n2. HQ-SILC-1 (Overcrowded dwelling):")
print("   - Germany shows 0 for 2015-2019")
print("   - Denmark shows 0 for 2022-2023")
print("   - Likely due to missing data or different data collection")