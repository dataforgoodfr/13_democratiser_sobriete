"""
Small script to take a sample of the prescreening/phase 1 database and save it as a parquet file.
"""

import sqlite3
import pandas as pd
DB_PATH = "openalex_works.db"
PARQUET_OUTPUT_PATH = "works_sample.parquet"
SAMPLE_SIZE = 1_000_000  # number of records to sample

def main():
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT * FROM works where title is NOT NULL LIMIT {SAMPLE_SIZE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df.to_parquet(PARQUET_OUTPUT_PATH, index=False)
    print(f"Sample of {SAMPLE_SIZE} records saved to {PARQUET_OUTPUT_PATH}")

if __name__ == "__main__":
    main()