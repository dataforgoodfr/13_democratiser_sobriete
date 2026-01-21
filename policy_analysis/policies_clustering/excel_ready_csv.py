import pandas as pd
import os

# --- Configuration ---
INPUT_FILE = "wsl_policies_clustered_and_named.csv"
OUTPUT_FILE = "wsl_policies_excel_ready.csv"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File not found: {INPUT_FILE}")
        return

    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 1. Remove the 'embedding' column if it exists
    if 'embedding' in df.columns:
        print("Removing 'embedding' column (causes Excel formatting issues)...")
        df = df.drop(columns=['embedding'])
    else:
        print("Note: 'embedding' column not found.")

    # 2. Optional: Clean up newlines in the policy text just in case
    # This replaces actual 'Enter' key breaks in text with a space so rows stay on one line
    if 'single_policy_item' in df.columns:
        df['single_policy_item'] = df['single_policy_item'].astype(str).str.replace('\n', ' ', regex=False)

    # 3. Save with 'utf-8-sig' (The 'sig' tells Excel how to read special characters)
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("✅ Done! You can now open 'wsl_policies_excel_ready.csv' in Excel safely.")

if __name__ == "__main__":
    main()