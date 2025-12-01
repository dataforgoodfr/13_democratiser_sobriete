"""
Parallel version of get_openalex_ids.py.
Adapted with AI assistance.
"""

import sqlite3
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from library.connectors.openalex.openalex_connector import OpenAlexConnector

KEYWORDS_CSV_PATH = "sufficiency_keywords_regrouped_count.csv"
DB_PATH = "openalex_works_v2.db"
MAX_WORKERS = 2  # more than 2 workers may lead to rate limiting
MAX_WORKS_PER_THEME = 2_500_000

DB_LOCK = threading.Lock()
BAR_POSITION_QUEUE = queue.Queue()


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
    
    # Fixed table name to match usage
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
    # Use lock for reading just to be safe, though not strictly required for reads in WAL mode
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM processed_themes WHERE sector = ? AND theme = ?", 
            (theme.sector, theme.theme)
        )
        result = cursor.fetchone()
        conn.close()
    return result is not None


def fetch_ids_for_theme(connector: OpenAlexConnector, theme: Theme, pbar: tqdm):
    """
    The core logic. Accepts a pre-configured progress bar (pbar) to update.
    """
    desc = f"{theme.sector} | {theme.theme}"
    pbar.set_description(f"Processing: {desc}")
    pbar.reset()
    
    if is_theme_processed(theme):
        pbar.write(f"Already processed: {theme.sector} - {theme.theme}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    buffer = []
    
    try:
        total_works = connector.count_works(theme.query)
        
        if total_works > MAX_WORKS_PER_THEME:
            pbar.write(f"Too many ({total_works}) for {desc}. Skipping.")
            return

        id_iterator, total_count = connector.fetch_work_ids(theme.query, per_page=200)
        pbar.total = total_count
        pbar.refresh()

        for work_id in id_iterator:
            buffer.append((work_id,))
            count += 1
            pbar.update(1)
            
            # Batch Insert
            if len(buffer) >= 1000:
                with DB_LOCK:
                    cursor.executemany("INSERT OR IGNORE INTO works (id) VALUES (?)", buffer)
                    conn.commit()
                buffer = []

        # Insert remaining items
        if buffer:
            with DB_LOCK:
                cursor.executemany("INSERT OR IGNORE INTO works (id) VALUES (?)", buffer)
                conn.commit()
        
        # Mark as processed
        with DB_LOCK:
            cursor.execute(
                """
                INSERT OR REPLACE INTO processed_themes 
                (sector, theme, query, count) VALUES (?, ?, ?, ?)
                """,
                (theme.sector, theme.theme, theme.query, count)
            )
            conn.commit()
        
    except Exception as e:
        pbar.write(f"Error in {desc}: {e}")
        # Don't raise, just log, so other threads continue
    finally:
        conn.close()


def worker_wrapper(connector, theme):
    """
    AI-generated.
    Wraps the logic to handle the progress bar position.
    """
    # 1. Get a free slot ID (e.g., line 1, 2, 3 or 4)
    position = BAR_POSITION_QUEUE.get()
    
    try:
        # 2. Create the bar at that specific line
        # leave=True keeps the bar visible after completion (optional)
        with tqdm(position=position, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            fetch_ids_for_theme(connector, theme, pbar)
    finally:
        # 3. Give the slot back so the next task can use this line
        BAR_POSITION_QUEUE.put(position)


def fetch_ids_for_all_themes(connector: OpenAlexConnector, themes: list[Theme]):
    # Fill the queue with positions 1 to MAX_WORKERS
    # We reserve position 0 for the main "Total Progress" bar
    for i in range(1, MAX_WORKERS + 1):
        BAR_POSITION_QUEUE.put(i)

    print("\nStarting concurrent fetch...")
    
    # Main progress bar (position 0)
    with tqdm(total=len(themes), position=0, desc="Total Themes", bar_format="{l_bar}{bar}") as main_pbar:
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(worker_wrapper, connector, theme) 
                for theme in themes
            ]
            
            for _ in as_completed(futures):
                main_pbar.update(1)


def get_themes() -> list[Theme]:
    df = pd.read_csv(KEYWORDS_CSV_PATH)
    themes = []
    for row in df.itertuples():
        t = Theme(sector=row.Sector, theme=row.Theme, query=row.Keywords)
        themes.append(t)
    return themes


def count_unique_ids():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM works")
    total = cursor.fetchone()[0]
    conn.close()
    return total


def main():
    init_db()
    connector = OpenAlexConnector(email="example@wsl.org")
    themes = get_themes()
    print(f"Total themes to process: {len(themes)}")
    fetch_ids_for_all_themes(connector, themes)
    total_ids = count_unique_ids()
    print(f"\n\nDone! Total unique OpenAlex IDs collected: {total_ids}")


if __name__ == "__main__":
    main()