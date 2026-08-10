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

df = pd.read_csv("path_to_csv")
df.to_sql("table_name", engine, if_exists="replace", index=False)