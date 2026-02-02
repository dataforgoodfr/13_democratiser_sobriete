"""
EU-SILC Tenure Analysis by Age Groups (Independent Adults Only)
================================================================
Analyzes housing ownership by age groups (18-30, 31-45, 46-60, 61+) 
for independent adults (people without parents in household).

Filter: RB220 = NaN AND RB230 = NaN (both father and mother NOT in household)

Data Source: EU-SILC Cross-sectional 2004-2023
Country: Luxembourg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Configuration
BASE_DATA_PATH = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/EU-SILC/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "graphs" / "EU-SILC"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRY = "LU"
START_YEAR = 2004
END_YEAR = 2023

# Age group definitions
AGE_GROUPS = {
    1: {"label": "Young Adults (18-30)", "range": (18, 30)},
    2: {"label": "Adults (31-45)", "range": (31, 45)},
    3: {"label": "Middle-aged (46-60)", "range": (46, 60)},
    4: {"label": "Seniors (61+)", "range": (61, 150)},
}

# Tenure categories based on HH020/HH021
# Pre-2010: HH020 codes (1=owner, 2=rented, 3=free)
# Post-2010: HH021 codes (1=owner, 2=rented, 3=free)
def categorize_tenure(tenure_value, year):
    """Categorize tenure as owner or renter
    
    HH021 (2010+): 1,2=owner | 3,4,5=renter
    HH020 (pre-2010): 1=owner | 2,3,4=renter
    """
    if pd.isna(tenure_value):
        return None
    tenure = int(tenure_value)
    
    if year >= 2010:
        # HH021: 1=owner without mortgage, 2=owner with mortgage
        if tenure in [1, 2]:
            return "owner"
        elif tenure in [3, 4, 5]:
            return "renter"
    else:
        # HH020: 1=owner
        if tenure == 1:
            return "owner"
        elif tenure in [2, 3, 4]:
            return "renter"
    
    return None


def get_tenure_column(year):
    """Determine which tenure column to use"""
    if year < 2010:
        return "HH020"
    else:
        return "HH021"


def load_personal_data_with_filtering(year, country=COUNTRY):
    """
    Load R-file (personal register) with age filtering for independent adults.
    
    Filter: RB220 = NaN AND RB230 = NaN (no father and no mother in household)
    This excludes infants/children living with parents.
    
    Age is computed from RB080 (year of birth) and interview date (RB100/RB110).
    Uses robust column loading with multiple fallback strategies.
    """
    r_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{country}{str(year)[-2:]}R.csv"
    
    if not os.path.exists(r_file_path):
        print(f"  R-file not found: {r_file_path}")
        return None
    
    try:
        # Try multiple column combinations for robust loading
        # RB080 (year of birth) available across all years
        # RB100/RB110 for interview year/month to compute age
        columns_to_try = [
            ['RB010', 'RB020', 'RB030', 'RB080', 'RB100', 'RB110', 'RB220', 'RB230'],
            ['RB010', 'RB020', 'RB030', 'RB080', 'RB100', 'RB220', 'RB230'],  # No interview month
            ['RB010', 'RB020', 'RB030', 'RB080', 'RB220', 'RB230'],  # No interview date
        ]
        
        df = None
        loaded_cols = None
        
        for cols in columns_to_try:
            try:
                df = pd.read_csv(r_file_path, usecols=cols, on_bad_lines='skip')
                loaded_cols = cols
                break
            except ValueError:
                # This column combination doesn't exist in this file
                continue
            except Exception:
                # Other errors - try next combination
                continue
        
        if df is None:
            print(f"  Could not load R-file with any column combination")
            return None
        
        # Filter for independent adults: no father AND no mother in household
        # NaN in RB220/RB230 means parent not in household
        df = df[df["RB220"].isna() & df["RB230"].isna()]
        
        if len(df) == 0:
            print(f"  No independent adults found (all have parents in household)")
            return None
        
        # Compute age from RB080 (year of birth) and interview date
        if 'RB080' not in loaded_cols:
            print(f"  RB080 (year of birth) not available in {year}")
            return None
        
        # Drop rows with missing birth year
        df = df.dropna(subset=["RB080"])
        
        # Convert to integers
        df["RB080"] = df["RB080"].astype(int)
        
        # Use RB100 (interview year) if available, otherwise use the survey year
        if 'RB100' in loaded_cols and df["RB100"].notna().sum() > 0:
            df["RB100"] = df["RB100"].astype(int)
            interview_year = df["RB100"]
        else:
            interview_year = year
        
        # Compute age
        df["age"] = interview_year - df["RB080"]
        
        # If we have interview month (RB110), adjust age more precisely
        if 'RB110' in loaded_cols and df["RB110"].notna().sum() > 0:
            # RB110 is interview month (1-12), RB081 is month of birth
            # We don't have birth month, so we'll use a simpler approach
            pass
        
        # Drop rows with missing or invalid age
        df = df.dropna(subset=["age"])
        df = df[df["age"] >= 0]
        
        # Filter for age >= 18
        df = df[df["age"] >= 18]
        
        if len(df) == 0:
            print(f"  No adults aged 18+ found")
            return None
        
        # Assign age groups
        df["age_group"] = df["age"].apply(assign_age_group)
        
        return df[["RB010", "RB020", "RB030", "age", "age_group"]]
    
    except Exception as e:
        print(f"  Error reading R-file for {year}: {e}")
        return None


def assign_age_group(age):
    """Assign person to age group based on age"""
    age = int(age)
    if 18 <= age <= 30:
        return 1
    elif 31 <= age <= 45:
        return 2
    elif 46 <= age <= 60:
        return 3
    elif age >= 61:
        return 4
    else:
        return None


def load_household_with_tenure(year, country=COUNTRY):
    """Load H-file (household characteristics) with tenure status"""
    hh_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{country}{str(year)[-2:]}H.csv"
    
    if not os.path.exists(hh_file_path):
        print(f"  H-file not found: {hh_file_path}")
        return None
    
    try:
        tenure_col = get_tenure_column(year)
        df = pd.read_csv(hh_file_path, usecols=["HB010", "HB020", "HB030", tenure_col])
        df = df.dropna(subset=[tenure_col])
        
        # Rename tenure column to standard name
        df["tenure"] = df[tenure_col].apply(lambda x: categorize_tenure(x, year))
        df = df[df["tenure"].notna()]
        
        return df[["HB010", "HB020", "HB030", "tenure"]]
    
    except Exception as e:
        print(f"  Error reading H-file for {year}: {e}")
        return None


def load_d_file_weights(year, country=COUNTRY):
    """Load D-file (household register) with weights"""
    d_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{country}{str(year)[-2:]}D.csv"
    
    if not os.path.exists(d_file_path):
        print(f"  D-file not found: {d_file_path}")
        return None
    
    try:
        df = pd.read_csv(d_file_path, usecols=["DB010", "DB020", "DB030", "DB090"])
        df = df.dropna(subset=["DB090"])
        return df
    except Exception as e:
        print(f"  Error reading D-file for {year}: {e}")
        return None


def calculate_ownership_by_age_group(year):
    """Calculate ownership rates by age group for a single year"""
    
    print(f"\nYear {year}:")
    
    # Load personal data (independent adults only)
    personal_df = load_personal_data_with_filtering(year)
    if personal_df is None or len(personal_df) == 0:
        return None
    
    # Load household tenure
    household_df = load_household_with_tenure(year)
    if household_df is None or len(household_df) == 0:
        return None
    
    # Load household weights
    weights_df = load_d_file_weights(year)
    if weights_df is None or len(weights_df) == 0:
        return None
    
    # Convert to string for merging
    personal_df["RB010"] = personal_df["RB010"].astype(str)
    personal_df["RB020"] = personal_df["RB020"].astype(str)
    personal_df["RB030"] = personal_df["RB030"].astype(str)
    
    household_df["HB010"] = household_df["HB010"].astype(str)
    household_df["HB020"] = household_df["HB020"].astype(str)
    household_df["HB030"] = household_df["HB030"].astype(str)
    
    weights_df["DB010"] = weights_df["DB010"].astype(str)
    weights_df["DB020"] = weights_df["DB020"].astype(str)
    weights_df["DB030"] = weights_df["DB030"].astype(str)
    
    # Extract household ID from personal ID (last 2 digits)
    personal_df["household_id"] = personal_df["RB030"].str[:-2]
    
    # Merge personal data with household data
    merged = personal_df.merge(
        household_df,
        left_on=["RB010", "RB020", "household_id"],
        right_on=["HB010", "HB020", "HB030"],
        how="left"
    )
    
    if len(merged) == 0:
        print(f"  No matching households found")
        return None
    
    # Merge with weights
    merged = merged.merge(
        weights_df,
        left_on=["RB010", "RB020", "household_id"],
        right_on=["DB010", "DB020", "DB030"],
        how="left"
    )
    
    # Drop rows without tenure or weight
    merged = merged.dropna(subset=["tenure", "DB090"])
    
    if len(merged) == 0:
        print(f"  No data after merging with weights")
        return None
    
    print(f"  Persons analyzed: {len(merged)}")
    
    # Calculate ownership by age group
    results = {}
    
    for age_group_id, age_info in AGE_GROUPS.items():
        group_data = merged[merged["age_group"] == age_group_id]
        
        if len(group_data) == 0:
            results[age_group_id] = None
            continue
        
        # Weighted calculations
        total_weight = group_data["DB090"].sum()
        owner_weight = group_data[group_data["tenure"] == "owner"]["DB090"].sum()
        
        ownership_pct = (owner_weight / total_weight * 100) if total_weight > 0 else None
        
        results[age_group_id] = {
            "label": age_info["label"],
            "count": len(group_data),
            "ownership_pct": ownership_pct,
        }
        
        print(f"  {age_info['label']:30} n={len(group_data):5}  ownership={ownership_pct:6.2f}%")
    
    return results


def create_summary_dataframe():
    """Create dataframe with results for all years"""
    all_results = {}
    
    for year in range(START_YEAR, END_YEAR + 1):
        result = calculate_ownership_by_age_group(year)
        
        if result is not None:
            all_results[year] = {
                f"age_group_{age_id}": result[age_id]["ownership_pct"]
                if result[age_id] is not None else None
                for age_id in AGE_GROUPS.keys()
            }
    
    if not all_results:
        print("\nNo results to summarize")
        return None
    
    df = pd.DataFrame(all_results).T
    df.index.name = "year"
    
    # Rename columns to include age group labels
    df.columns = [AGE_GROUPS[int(col.split("_")[2])]["label"] for col in df.columns]
    
    return df


def create_visualization(df):
    """Create multi-panel visualization of ownership by age group"""
    if df is None or df.empty:
        print("No data to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Housing Ownership by Age Group (Independent Adults Only)\nLuxembourg, EU-SILC", 
                 fontsize=16, fontweight="bold")
    
    axes = axes.flatten()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for idx, (age_group_id, age_info) in enumerate(AGE_GROUPS.items()):
        ax = axes[idx]
        col_name = age_info["label"]
        
        if col_name in df.columns:
            data = df[col_name].dropna()
            
            ax.plot(data.index, data.values, marker="o", linewidth=2, 
                   markersize=6, color=colors[idx])
            ax.fill_between(data.index, data.values, alpha=0.3, color=colors[idx])
            
            ax.set_title(col_name, fontsize=12, fontweight="bold")
            ax.set_ylabel("Ownership Rate (%)", fontsize=10)
            ax.set_xlabel("Year", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            # Add value labels
            for year, value in data.items():
                ax.text(year, value + 2, f"{value:.1f}%", ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "luxembourg_tenure_by_age_groups.png", dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved: {OUTPUT_DIR / 'luxembourg_tenure_by_age_groups.png'}")


def format_dataframe_for_output(df):
    """Format dataframe with percentage display"""
    if df is None or df.empty:
        return None
    
    df_formatted = df.copy()
    for col in df_formatted.columns:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    return df_formatted


def main():
    print("="*70)
    print("EU-SILC Tenure Analysis by Age Groups (Independent Adults Only)")
    print("="*70)
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"Country: {COUNTRY}")
    print(f"Filter: RB220 = -2 AND RB230 = -2 (no parents in household)")
    print("="*70)
    
    # Calculate results for all years
    df_results = create_summary_dataframe()
    
    if df_results is not None and not df_results.empty:
        print("\n" + "="*70)
        print("SUMMARY TABLE")
        print("="*70)
        print(format_dataframe_for_output(df_results).to_string())
        
        # Save to CSV
        csv_path = OUTPUT_DIR / "luxembourg_tenure_by_age_groups.csv"
        df_results.to_csv(csv_path)
        print(f"\nData saved: {csv_path}")
        
        # Create visualization
        create_visualization(df_results)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
