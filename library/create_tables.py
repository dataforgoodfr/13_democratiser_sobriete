"""
Creates databse tables and fills them with data from parquet file.
"""

import pandas as pd
from library.database import create_tables, get_session
from library.database.models import OpenAlexWorks, ScrapingQueue
from sqlmodel import select


BUCKET_URL = "https://sufficiency-library.s3.fr-par.scw.cloud"
LIBRARY_KEY = "library_v1_2025-12-08.parquet"
BATCH_SIZE = 20_000


def populate_database(df: pd.DataFrame, batch_size: int = 10_000):
    # insert data into database by batches (millions of rows)
    with get_session() as session:
        for start in range(0, len(df), batch_size):
            end = start + batch_size
            batch_df = df.iloc[start:end]
            records = []
            for _, row in batch_df.iterrows():
                work = OpenAlexWorks(
                    openalex_id=row["id"],
                    doi=row.get("doi"),
                    title=row["title"],
                    abstract=row.get("abstract"),
                    language=row["language"],
                    publication_date=row["publication_date"],
                    publication_type=row["type"],
                    fwci=row.get("fwci"),
                    is_oa=row["is_oa"],
                    oa_status=row["oa_status"],
                    landing_page_url=row.get("landing_page_url"),
                    pdf_url=row.get("pdf_url"),
                )
                records.append(work)

                if row["is_oa"]:
                    queue_item = ScrapingQueue(
                        openalex_id=row["id"],
                        landing_page_url=row.get("landing_page_url"),
                        pdf_url=row.get("pdf_url"),
                    )
                    records.append(queue_item)

            session.add_all(records)
            session.commit()
            print(f"Inserted records {start} to {end}")


def get_existing_ids():
    """Get existing OpenAlex IDs in the database to avoid duplicates"""
    with get_session() as session:
        stmt = select(OpenAlexWorks.openalex_id)
        existing_ids = set(session.exec(stmt).all())
        return existing_ids


def main():
    create_tables()
    # try reading locally and if doesn't exist, read from S3
    try:
        df = pd.read_parquet(LIBRARY_KEY)
        print(f"Loaded {len(df)} records from local parquet file.")
    except FileNotFoundError:
        df = pd.read_parquet(f"{BUCKET_URL}/{LIBRARY_KEY}")
        df.to_parquet(LIBRARY_KEY)  # save locally for next time
        print(f"Loaded {len(df)} records from S3 parquet file.")

    # avoid reinserting existing records
    existing_ids = get_existing_ids()
    print(f"Found {len(existing_ids)} existing records in the database.")    
    initial_count = len(df)
    df = df[~df["id"].isin(existing_ids)]
    filtered_count = len(df)
    print(f"Filtered out {initial_count - filtered_count} existing records. {len(df)} new records to insert.")

    print("Populating database...")
    populate_database(df, batch_size=BATCH_SIZE)
    print("Done!")


if __name__ == "__main__":
    main()
