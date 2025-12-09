"""
Fetches unique IDs from OpenAlex database using pyalex, searching for keywords defined in KEYWORDS_CSV_PATH.
This is a first stage of a two-stage process:
1. Fetch work IDs and deduplicate them.
2. Fetch full data for each unique ID.
Stores everything in a local SQLite database.
"""

import sqlite3

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from library.connectors.openalex.openalex_connector import OpenAlexConnector

KEYWORDS_CSV_PATH = "sufficiency_keywords_regrouped_count.csv"
DB_PATH = "openalex_ids.db"

MAX_WORKS_PER_THEME = 2_500_000


class Theme(BaseModel):
    sector: str
    theme: str
    query: str


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
      
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS works (
            id TEXT PRIMARY KEY,
            raw_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_themes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sector TEXT,
            theme TEXT,
            query TEXT,
            count INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def is_theme_processed(theme: Theme) -> bool:
    """Check if a query has already been processed."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM processed_themes WHERE sector = ? AND theme = ?", (theme.sector, theme.theme))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def fetch_ids_for_theme(connector: OpenAlexConnector, theme: Theme):
    desc = f"{theme.sector} | {theme.theme}"
    query = theme.query
    if is_theme_processed(theme):
        print(f"Theme '{desc}' already processed, skipping...")
        return
    
    print(f"\nProcessing theme '{desc}'")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    buffer = []
    try:
        total_works = connector.count_works(query)
        print(f"  Total works found: {total_works}")
        if total_works > MAX_WORKS_PER_THEME:
            print(f'Too many articles found {total_works}, skipping.')
            conn.close()
            return

        id_iterator, total_count = connector.fetch_work_ids(query, per_page=200)
        for work_id in tqdm(id_iterator, total=total_count, desc=f"Fetching IDs for {desc}"):
            try:
                buffer.append((work_id,))
                count += 1
                if len(buffer) >= 1000:
                    cursor.executemany("INSERT OR IGNORE INTO works (id) VALUES (?)", buffer)
                    conn.commit()
                    buffer = []
            except Exception as e:
                print(f"Error while processing batch, skipping. '{desc}': {e}")
                continue
        
        # Insert remaining items in the buffer
        if buffer:
            cursor.executemany("INSERT OR IGNORE INTO works (id) VALUES (?)", buffer)
            conn.commit()
        
        # Mark query as processed
        cursor.execute(
            "INSERT OR REPLACE INTO processed_themes (sector, theme, query, count) VALUES (?, ?, ?, ?)",
            (theme.sector, theme.theme, theme.query, count)
        )
        conn.commit()
        
        print(f"  Collected {count} IDs for this query")

    except Exception as e:
        print(f"Error while processing query, skipping. '{desc}': {e}")
        
    finally:
        conn.close()


def count_unique_ids():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM works")
    total = cursor.fetchone()[0]
    conn.close()
    return total


def get_themes() -> list[Theme]:
    df = pd.read_csv(KEYWORDS_CSV_PATH)   
    themes = []
    for row in df.itertuples():
        t = Theme(sector=row.Sector, theme=row.Theme, query=row.Keywords)
        themes.append(t)
    return themes


def main():
    init_db()
    print('Database initialized.')
    connector = OpenAlexConnector(email="example@wsl.org")
    themes = get_themes()
    print(f"Total themes to process: {len(themes)}")
    for theme in themes:
        fetch_ids_for_theme(connector, theme)
    total_ids = count_unique_ids()
    print(f"\nTotal unique OpenAlex IDs collected: {total_ids}")


if __name__ == "__main__":
    main()
