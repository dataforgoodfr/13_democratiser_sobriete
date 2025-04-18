import pandas as pd
import os
from sqlmodel import create_engine, SQLModel


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://u4axloluqibskgvdikuy:g2rXgpHSbztokCbFxSyR@bk8htvifqendwt1wlzat-postgresql.services.clever-cloud.com:7327/bk8htvifqendwt1wlzat"
)

# Create database engine and tables
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SQLModel.metadata.create_all(engine)

df = pd.read_csv("emissions_trajectories.csv")
df.to_sql("CO2_emissions_trajectories", engine, if_exists="replace", index=False)

df = pd.read_csv("Historical_emissions.csv")
df.to_sql("CO2_historical_emissions", engine, if_exists="replace", index=False)