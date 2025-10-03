"""
Variable name mapping utility for EWBI indicators.
Maps current acronyms (like 'AN-EHIS-1') to proposed names (like 'Struggling to Prepare Meals').
"""

import pandas as pd
import os

# Variable name mapping from Excel file
VARIABLE_NAME_MAPPING = {
    "AN-EHIS-1": "Struggling to Prepare Meals",
    "AE-EHIS-1": "Rarely Eating Fruit",
    "HE-SILC-1": "Homes Too Cold in Winter",
    "HE-SILC-2 ": "Behind on Utility Bills",
    "HQ-SILC-1": "Overcrowded dwelling",
    "EL-SILC-1": "Low Life Satisfaction",
    "EL-EHIS-1": "No Joy in Daily Life",
    "ES-SILC-1": "Unable to Handle Unexpected Costs",
    "ES-SILC-2": "Hard to Make Ends Meet",
    "EC-SILC-1": "Can't Meet Friends or Family Monthly",
    "EC-SILC-2": "Low Trust in Others",
    "EC-EHIS-1": "No One to Rely On in a Crisis",
    "AH-SILC-1": "Poor Self-Rated Health",
    "AH-SILC-2": "Living with Chronic Illness",
    "AH-SILC-3": "Limited by Health Problems",
    "AH-SILC-4": "Unable to Work Due to Long-Term Illness",
    "AH-EHIS-1": "Feeling Depressed Every Day",
    "AH-EHIS-2": "No Physical Activity",
    "AB-EHIS-1": "Daily Smoking",
    "AB-EHIS-2": "Drinking Alcohol Nearly Every Day",
    "AB-EHIS-3 ": "Involved in a Road Accident",
    "IS-SILC-1": "Preschool-Age Children Not Enrolled",
    "IS-SILC-2": "School-Age Children Not Attending School",
    "IS-SILC-3": "No Formal Education",
    "RT-SILC-1 ": "Adults on Fixed-Term Contracts",
    "RT-SILC-2": "Adults Working Part-Time",
    "RT-LFS-1": "Working Multiple Jobs",
    "RT-LFS-2": "Wanting to Work More Hours",
    "RT-LFS-3": "Doing Overtime or Extra Hours",
    "RU-SILC-1": "Unemployed for Over 6 Months",
    "RU-LFS-1": "Currently Unemployed",
    "RR-AES-1": "Wanted to Learn, But Couldn't Participate"
}

def get_display_name(acronym):
    """
    Get the display name for a variable acronym.
    
    Args:
        acronym (str): The current acronym (e.g., 'AN-EHIS-1')
    
    Returns:
        str: The proposed display name or the original acronym if not found
    """
    if not acronym or pd.isna(acronym):
        return acronym
    
    # Handle potential extra spaces in the acronym
    cleaned_acronym = str(acronym).strip()
    
    # Try exact match first
    if cleaned_acronym in VARIABLE_NAME_MAPPING:
        return VARIABLE_NAME_MAPPING[cleaned_acronym]
    
    # Try with stripped spaces for keys that might have trailing spaces
    for key, value in VARIABLE_NAME_MAPPING.items():
        if key.strip() == cleaned_acronym:
            return value
    
    # If no mapping found, return the original acronym
    return acronym

def get_acronym_from_display_name(display_name):
    """
    Get the acronym for a display name (reverse mapping).
    
    Args:
        display_name (str): The display name
    
    Returns:
        str: The acronym or the original display name if not found
    """
    if not display_name or pd.isna(display_name):
        return display_name
    
    # Create reverse mapping
    reverse_mapping = {v: k.strip() for k, v in VARIABLE_NAME_MAPPING.items()}
    
    return reverse_mapping.get(display_name, display_name)

def load_and_verify_mapping():
    """
    Load the Excel file and verify our hardcoded mapping is up to date.
    This function can be used for validation purposes.
    """
    try:
        # Try to load the Excel file for verification
        excel_path = os.path.join(os.path.dirname(__file__), '..', 'data', '2025-08-29_EWBI-Renaming of Primary Indicators.xlsx')
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            excel_mapping = dict(zip(df['Current acronym'], df['Proposed name']))
            
            # Compare with our hardcoded mapping
            missing_in_hardcoded = set(excel_mapping.keys()) - set(k.strip() for k in VARIABLE_NAME_MAPPING.keys())
            missing_in_excel = set(k.strip() for k in VARIABLE_NAME_MAPPING.keys()) - set(excel_mapping.keys())
            
            if missing_in_hardcoded or missing_in_excel:
                print("Warning: Mapping differences detected!")
                if missing_in_hardcoded:
                    print(f"Missing in hardcoded mapping: {missing_in_hardcoded}")
                if missing_in_excel:
                    print(f"Missing in Excel file: {missing_in_excel}")
                return False
            else:
                print("Mapping verification successful!")
                return True
        else:
            print(f"Excel file not found at {excel_path}")
            return False
    except Exception as e:
        print(f"Error verifying mapping: {e}")
        return False