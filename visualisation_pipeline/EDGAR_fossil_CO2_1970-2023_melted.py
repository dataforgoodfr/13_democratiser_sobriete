from sqlmodel import SQLModel, Field, create_engine, Session
import pandas as pd

DB_URL = "postgresql://u4axloluqibskgvdikuy:g2rXgpHSbztokCbFxSyR@bk8htvifqendwt1wlzat-postgresql.services.clever-cloud.com:7327/bk8htvifqendwt1wlzat"

engine = create_engine(DB_URL)
SQLModel.metadata.create_all(engine)

file = pd.read_csv("visualisation_pipeline/data/EDGAR_fossil_CO2_1970-2023.csv", delimiter=';', decimal=",")
file.drop(columns=["Substance", "EDGAR Country Code"], inplace=True)
file_t = pd.melt(file, id_vars=["Country"], var_name="Year", value_name="CO2 Emissions")
file_t.dropna(how='all', axis=1, inplace=True)
file_t.to_sql(name="EDGAR_fossil_CO2_1970-2023_melted", con=engine, if_exists="replace", index=False)