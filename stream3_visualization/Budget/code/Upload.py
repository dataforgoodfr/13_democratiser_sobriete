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

# Upload historical emissions data
print("Uploading historical emissions data...")
df = pd.read_csv(os.path.join(output_dir, "combined_data.csv"))
df.to_sql("Viz_Carbon_Budget_combined_data_historical", engine, if_exists="replace", index=False)
print("Historical emissions data uploaded successfully")

# Upload scenario parameters
print("\nUploading scenario parameters...")
df = pd.read_csv(os.path.join(output_dir, "scenario_parameters.csv"))
df.to_sql("Viz_Carbon_Budget_scenario_parameters", engine, if_exists="replace", index=False)
print("Scenario parameters uploaded successfully")

# Upload forecast data
print("\nUploading forecast data...")
df = pd.read_csv(os.path.join(output_dir, "forecast_data.csv"))
df.to_sql("Viz_Carbon_Budget_forecast_data", engine, if_exists="replace", index=False)
print("Forecast data uploaded successfully")

print("\nAll data uploaded successfully!")