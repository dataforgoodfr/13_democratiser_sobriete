import pandas as pd
import os

EXTERNAL_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'external_data')

df = pd.read_excel(os.path.join(EXTERNAL_DATA_DIR, 'oecd_dwellings_inhabitants.xlsx'))

# Look at the structure more carefully
print("Shape:", df.shape)
print("\nLooking for data table starting with 'Data for Figure'...")

# Find the data table (second table)
for idx, val in enumerate(df.iloc[:, 11]):
    if pd.notna(val) and 'Data for Figure' in str(val):
        print(f"\nData table starts at row {idx}")
        print(f"Column 11 value: {val}")
        break

# Extract the data table (columns 11-15, starting from appropriate row)
print("\nData section (columns 11-15):")
print("=" * 80)
for i in range(min(35, len(df))):
    row_data = [str(df.iloc[i, j]) if pd.notna(df.iloc[i, j]) else '' for j in range(11, 16)]
    print(f"Row {i}: {row_data}")
