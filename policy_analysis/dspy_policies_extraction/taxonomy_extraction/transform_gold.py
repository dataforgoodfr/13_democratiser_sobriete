import pandas as pd
import json

# 1. Load Excel or CSV
file_path = "taxonomy_extraction/2025-12-12_Prompt_engineering_taxonomy.xlsx"
df = pd.read_excel(file_path)

# 2. Prepare list to store JSON objects
json_data = []

def split_to_list(value):
    """Utility to safely split semicolon/comma separated cells into lists"""
    if pd.isna(value):
        return []
    return [v.strip() for v in str(value).replace(";", ",").split(",") if v.strip()]

for idx, row in df.iterrows():
    question = str(row.get("chunk", "")).strip()

    regional_group = row.get("Regional_group")
    if pd.isna(regional_group):
        regional_group = None

    json_data.append({
        "question": question,
        "regional_group": regional_group,
        "geographical_scopes": split_to_list(row.get("Geographical_scope")),
        "main_country_focus": split_to_list(row.get("Studied_country")),

        # Newly added dimensions
        "human_needs": split_to_list(row.get("Human_needs")),
        "natural_ressource": split_to_list(row.get("Natural_ressource")),
        "wellbeing": split_to_list(row.get("Wellbeing")),
        "justice_consideration": split_to_list(row.get("Justice_consideration")),
        "planetary_boundaries": split_to_list(row.get("Planetary_boundaries"))
    })

# 3. Save as JSONL
output_path = "transformed_data_all.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in json_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Transformed data saved to {output_path}")
