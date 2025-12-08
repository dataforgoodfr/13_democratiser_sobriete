"""
Takes works data from sqlite db and save it to mulitple parquet files with BATCH_SIZE documents each.
"""

import sqlite3
import os
import pandas as pd


DB_PATH = "openalex_works.db"
BATCH_SIZE = 1_000_000
OUTPUT_DIR = "outputs"


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM works where title is NOT NULL")
    total_rows = cursor.fetchone()[0]
    print(f"Total rows to export: {total_rows}")

    offset = 0
    rows_written = 0
    i = 0
    while offset < total_rows:
        # Fetch batch
        query = (
            f"SELECT * FROM works where title is NOT NULL LIMIT {BATCH_SIZE} OFFSET {offset}"
        )
        df = pd.read_sql_query(query, conn)

        filename = os.path.join(OUTPUT_DIR, f"chunk_{i}.parquet")
        df.to_parquet(filename)

        offset += BATCH_SIZE
        i += 1
        print(f"Progress: {min(offset, total_rows)}/{total_rows} rows processed")

    conn.close()
    print(f"Export complete! {rows_written} values written to {OUTPUT_DIR} in {i+1} files")


if __name__ == "__main__":
    main()
