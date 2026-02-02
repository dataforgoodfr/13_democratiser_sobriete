"""
Variable name mapping utility for EWBI indicators.
Maps current acronyms (like 'IS-SILC-1') to proposed names (like 'Preschool-Age Children Not Enrolled').
Uses EU-SILC and LFS data sources only.
"""

import pandas as pd
import os

# Variable name mapping from Excel file
VARIABLE_NAME_MAPPING = {
    "HE-SILC-1": "Homes Too Cold in Winter",
    "HE-SILC-2 ": "Behind on Utility Bills",
    "HQ-SILC-1": "Overcrowded dwelling",
    "EL-SILC-1": "Low Life Satisfaction",
    "ES-SILC-1": "Unable to Handle Unexpected Costs",
    "ES-SILC-2": "Hard to Make Ends Meet",
    "EC-SILC-1": "Can't Meet Friends or Family Monthly",
    "EC-SILC-2": "Low Trust in Others",
    "AH-SILC-1": "Poor Self-Rated Health",
    "AH-SILC-2": "Living with Chronic Illness",
    "AH-SILC-3": "Limited by Health Problems",
    "AH-SILC-4": "Unable to Work Due to Long-Term Illness",
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
    "RR-AES-1": "Wanted to Learn, But Couldn't Participate",
    "HQ-SILC-2": "Cannot keep dwelling warm",
    "HQ-SILC-3": "Cannot keep dwelling cool", 
    "HQ-SILC-4": "Dwelling too dark",
    "HQ-SILC-5": "Noise from street",
    "HQ-SILC-6": "Leaking roof/damp/rot",
    "HQ-SILC-7": "Pollution or crime",
    "HQ-SILC-8": "No renovation measures",
    "IS-SILC-4": "Not participating in training",
    "IS-SILC-5": "No secondary education",
    "AC-SILC-3": "Unmet medical care need",
    "AC-SILC-4": "Unmet dental care need",
    "EC-SILC-3": "Cannot meet friends/family",
    "RT-LFS-4": "No freedom on working time",
    "RT-LFS-5": "Shift work",
    "RT-LFS-6": "Night work", 
    "RT-LFS-7": "Saturday work",
    "RT-LFS-8": "Sunday work",
    "EL-SILC-2": "No relevant care service",
    # Additional EU-SILC indicators
    "HQ-SILC-2": "Cannot keep dwelling comfortably warm",
    "HQ-SILC-3": "Cannot keep dwelling comfortably cool",
    "HQ-SILC-4": "Dwelling too dark",
    "HQ-SILC-5": "Noise from street",
    "HQ-SILC-6": "Leaking roof, damp or rot in dwelling",
    "HQ-SILC-7": "Pollution or crime",
    "HQ-SILC-8": "No renovation measures",
    "IS-SILC-4": "Not participating in formal training",
    "IS-SILC-5": "No secondary education",
    "AC-SILC-3": "Unmet need for medical care",
    "AC-SILC-4": "Unmet need for dental examination and treatment",
    "EC-SILC-3": "Cannot get together with friends or family",
    "EC-SILC-4": "Persons living alone"
}

# French translation mapping for indicator names
VARIABLE_NAME_MAPPING_FR = {
    "HE-SILC-1": "Maison trop froide en hiver",
    "HE-SILC-2 ": "Retard dans le paiement des charges",
    "HQ-SILC-1": "Logement surpeuplé",
    "EL-SILC-1": "Faible satisfaction de vie",
    "ES-SILC-1": "Incapacité à faire face à des dépenses imprévues",
    "ES-SILC-2": "Difficultés à joindre les deux bouts",
    "EC-SILC-1": "Incapable de rencontrer amis/famille mensuellement",
    "EC-SILC-2": "Faible confiance envers les autres",
    "AH-SILC-1": "État de santé auto-évalué médiocre",
    "AH-SILC-2": "Maladie chronique",
    "AH-SILC-3": "Limité par des problèmes de santé",
    "AH-SILC-4": "Incapable de travailler pour raison de santé",
    "IS-SILC-1": "Enfants d'âge maternel non scolarisés",
    "IS-SILC-2": "Enfants d'âge primaire non scolarisés",
    "IS-SILC-3": "Pas d'éducation formelle",
    "RT-SILC-1 ": "Adultes avec contrats à durée déterminée",
    "RT-SILC-2": "Adultes travaillant à temps partiel",
    "RT-LFS-1": "Travailleur avec plusieurs emplois",
    "RT-LFS-2": "Souhait de travailler plus d'heures",
    "RT-LFS-3": "Heures supplémentaires",
    "RU-SILC-1": "Chômage depuis plus de 6 mois",
    "RU-LFS-1": "Actuellement au chômage",
    "RR-AES-1": "Souhait d'apprendre mais incapacité à participer",
    "HQ-SILC-2": "Incapacité à maintenir le logement confortablement chaud",
    "HQ-SILC-3": "Incapacité à maintenir le logement confortablement frais",
    "HQ-SILC-4": "Logement trop sombre",
    "HQ-SILC-5": "Bruit de la rue",
    "HQ-SILC-6": "Toiture qui fuit/humidité/pourriture",
    "HQ-SILC-7": "Pollution ou criminalité",
    "HQ-SILC-8": "Aucune mesure de rénovation",
    "IS-SILC-4": "Non-participation à une formation",
    "IS-SILC-5": "Pas d'éducation secondaire",
    "AC-SILC-3": "Besoin non satisfait de soins médicaux",
    "AC-SILC-4": "Besoin non satisfait de soins dentaires",
    "EC-SILC-3": "Incapable de rencontrer amis/famille",
    "RT-LFS-4": "Pas de liberté sur les horaires de travail",
    "RT-LFS-5": "Travail par quarts",
    "RT-LFS-6": "Travail de nuit",
    "RT-LFS-7": "Travail le samedi",
    "RT-LFS-8": "Travail le dimanche",
    "EL-LFS-2": "Aucun service de garde adéquat",
    "EC-SILC-4": "Personnes vivant seules"
}

def get_display_name(acronym, language='en'):
    """
    Get the display name for a variable acronym.
    
    Args:
        acronym (str): The current acronym (e.g., 'IS-SILC-1')
        language (str): Language code ('en' for English, 'fr' for French). Default is 'en'.
    
    Returns:
        str: The display name in the requested language or the original acronym if not found
    """
    if not acronym or pd.isna(acronym):
        return acronym
    
    # Choose the mapping dictionary based on language
    mapping = VARIABLE_NAME_MAPPING if language == 'en' else VARIABLE_NAME_MAPPING_FR
    
    # Handle potential extra spaces in the acronym
    cleaned_acronym = str(acronym).strip()
    
    # Try exact match first
    if cleaned_acronym in mapping:
        return mapping[cleaned_acronym]
    
    # Try with stripped spaces for keys that might have trailing spaces
    for key, value in mapping.items():
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


# Indicators to EXCLUDE from plotting (should NOT appear in dashboard or be used in aggregations)
EXCLUDED_INDICATORS = {
    'HH-SILC-1',      # Should not be plotted
    'AC-SILC-1',      # No EU priority mapping
    'AN-SILC-1',      # No EU priority mapping
}

# List of valid indicators that SHOULD be included in analysis
VALID_INDICATORS = {
    'AC-SILC-3', 'AC-SILC-4', 'AH-SILC-2', 'AH-SILC-3', 'AH-SILC-4',
    'AN-SILC-1', 'EC-SILC-2', 'EC-SILC-3', 'EC-SILC-4', 'EL-LFS-2',
    'ES-SILC-1', 'ES-SILC-2', 'HE-SILC-2', 'HQ-SILC-1', 'HQ-SILC-2',
    'HQ-SILC-3', 'HQ-SILC-4', 'HQ-SILC-5', 'HQ-SILC-6', 'HQ-SILC-7',
    'HQ-SILC-8', 'IC-SILC-1', 'IC-SILC-2', 'IS-SILC-1', 'IS-SILC-2',
    'IS-SILC-3', 'IS-SILC-4', 'IS-SILC-5', 'RT-LFS-1', 'RT-LFS-2',
    'RT-LFS-3', 'RT-LFS-4', 'RT-LFS-5', 'RT-LFS-6', 'RT-LFS-7',
    'RT-LFS-8', 'RT-SILC-1', 'RT-SILC-2', 'RU-LFS-1', 'RU-SILC-1',
    'TS-SILC-1'
}

def should_filter_indicator(indicator):
    """
    Check if an indicator should be excluded from analysis.
    
    Args:
        indicator (str): The indicator code
    
    Returns:
        bool: True if indicator should be EXCLUDED, False if it should be included
    """
    if not indicator or pd.isna(indicator):
        return True  # Filter out NaN/None
    
    indicator_clean = str(indicator).strip()
    return indicator_clean in EXCLUDED_INDICATORS