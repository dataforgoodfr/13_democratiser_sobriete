import pandas as pd
import os

EXTERNAL_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'external_data')

print("Testing Excel file load...")
print(f"File path: {os.path.join(EXTERNAL_DATA_DIR, 'oecd_dwellings_inhabitants.xlsx')}")
print(f"File exists: {os.path.exists(os.path.join(EXTERNAL_DATA_DIR, 'oecd_dwellings_inhabitants.xlsx'))}")

try:
    df = pd.read_excel(os.path.join(EXTERNAL_DATA_DIR, 'oecd_dwellings_inhabitants.xlsx'))
    print(f"\nExcel loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    print(f"\nData types:")
    print(df.dtypes)
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
