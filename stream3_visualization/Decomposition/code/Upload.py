import pandas as pd
import os
from sqlalchemy import create_engine


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://u4axloluqibskgvdikuy:g2rXgpHSbztokCbFxSyR@bk8htvifqendwt1wlzat-postgresql.services.clever-cloud.com:7327/bk8htvifqendwt1wlzat"
)

# Create database engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Define the output directory
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Output")

# List of (filename, table_name) pairs to upload
files_to_upload = [
    ("Decomposition_comparaison_sobriete_life_vs_s1.csv", "Decomposition_comparaison_sobriete_life_vs_s1"),
    ("Decomposition_comparaison_sobriete_vs_captage.csv", "Decomposition_comparaison_sobriete_vs_captage"),
    ("Decomposition_repartition_sobriete_vs_decarbonation.csv", "Decomposition_repartition_sobriete_vs_decarbonation"),
    ("Decomposition_par_levier_et_scenario_2040.csv", "Decomposition_par_levier_et_scenario_2040"),
    ("Decomposition_sobriete_2040_batiment_residentiel.csv", "Decomposition_sobriete_2040_batiment_residentiel"),
    ("Decomposition_sufficiency_indicators.csv", "Decomposition_sufficiency_indicators"),
]

for filename, table_name in files_to_upload:
    file_path = os.path.join(output_dir, filename)
    print(f"Uploading {filename} to table {table_name}...")
    df = pd.read_csv(file_path)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"{filename} uploaded successfully as {table_name}")

print("\nAll data uploaded successfully!")
