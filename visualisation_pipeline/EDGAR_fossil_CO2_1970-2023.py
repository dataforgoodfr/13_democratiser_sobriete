from sqlmodel import SQLModel, Field, create_engine, Session
import pandas as pd

DB_URL = "postgresql://u4axloluqibskgvdikuy:g2rXgpHSbztokCbFxSyR@bk8htvifqendwt1wlzat-postgresql.services.clever-cloud.com:7327/bk8htvifqendwt1wlzat"

engine = create_engine(DB_URL)
SQLModel.metadata.create_all(engine)

file = pd.read_csv("visualisation_pipeline/data/EDGAR_fossil_CO2_1970-2023.csv", delimiter=';')
file.drop(columns=["Substance", "EDGAR Country Code"], inplace=True)
file_t = file.set_index('Country')
file_t = file_t.transpose().reset_index().rename(columns={'index': 'year'})
file_t.columns.name = None
file_t.index.name = None
file_t.dropna(how='all', axis=1, inplace=True)
file_t.to_sql(name="EDGAR_fossil_CO2_1970-2023", con=engine, if_exists="replace", index=False)