"""
Parallel version of get_openalex_ids.py.
Adapted with AI assistance.
"""

import sqlite3
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from library.connectors.openalex.openalex_connector import OpenAlexConnector, BASE_FILTERS, DESIRED_FIELDS

#IDS_PATH = "openalex_work_ids_tmp_251201-1046.csv"
DB_PATH = "openalex_works_v2.db"
MAX_WORKERS = 2  # more than 2 workers may lead to rate limiting
CHUNK_SIZE = 100

DB_LOCK = threading.Lock()
BAR_POSITION_QUEUE = queue.Queue()

desired_fields = [f if f != "abstract_inverted_index" else "abstract" for f in DESIRED_FIELDS]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

   # Add new columns if they don't exist
    for col in desired_fields:
        try:
            if col == 'has_fulltext':
                cursor.execute(f"ALTER TABLE works ADD COLUMN {col} INTEGER")
            else:
                cursor.execute(f"ALTER TABLE works ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise e

    conn.commit()
    conn.close()


def fetch_works_for_ids(connector: OpenAlexConnector, pool_nb: int, ids: list[str], pbar: tqdm):
    pbar.set_description(f"Processing pool: {pool_nb}")
    pbar.reset()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    buffer = []
    
    try:
        pbar.total = len(ids)
        pbar.refresh()

        for work in connector.get_works_from_ids(ids, filters=BASE_FILTERS):
            buffer.append((connector.get_entity_id_from_url(work['id']), *[work[field] for field in desired_fields if field != 'id']))
            count += 1
            pbar.update(1)
            
            # Batch Insert
            if len(buffer) >= 1000:
                with DB_LOCK:
                    cursor.executemany(f"""
                        INSERT OR REPLACE INTO works
                        ({', '.join(desired_fields)}) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, buffer)
                    conn.commit()
                buffer = []

        # Insert remaining items
        if buffer:
            with DB_LOCK:
                cursor.executemany(f"""
                        INSERT OR REPLACE INTO works
                        ({', '.join(desired_fields)}) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, buffer)
                conn.commit()
               
    except Exception as e:
        pbar.write(f"Error in pool {pool_nb}: {e}")
        # Don't raise, just log, so other threads continue
    finally:
        conn.close()


def worker_wrapper(connector: OpenAlexConnector, pool_nb: int, ids: list[str]):
    """
    Wraps the logic to handle the progress bar position.
    (AI-generated.) 
    """
    # 1. Get a free slot ID (e.g., line 1, 2, 3 or 4)
    position = BAR_POSITION_QUEUE.get()
    
    try:
        # 2. Create the bar at that specific line
        # leave=True keeps the bar visible after completion (optional)
        with tqdm(position=position, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            fetch_works_for_ids(connector, pool_nb, ids, pbar)
    finally:
        # 3. Give the slot back so the next task can use this line
        BAR_POSITION_QUEUE.put(position)


def fetch_works_for_all_ids(connector: OpenAlexConnector, ids: list[str]):
    pools = []
    pool_size = len(ids) // MAX_WORKERS
    for i in range(MAX_WORKERS):
        start_index = i * pool_size
        if i == MAX_WORKERS - 1:
            end_index = len(ids)
        else:
            end_index = (i + 1) * pool_size
        pools.append(ids[start_index:end_index])
    
    # position progress bars
    for i in range(0, MAX_WORKERS):
        BAR_POSITION_QUEUE.put(i)

    print("\nStarting concurrent fetch...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(worker_wrapper, connector, i, pool) 
            for i, pool in enumerate(pools)
        ]
        for _ in as_completed(futures):
            pass


def get_ids() -> list[str]:
    #df = pd.read_csv(IDS_PATH)
    #return df['id'].tolist()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM works where title is NULL")
    ids = cursor.fetchall()
    ids = [id_tuple[0] for id_tuple in ids]
    conn.close()
    print(f"Total ids to process: {len(ids)}")
    return ids


def count_fetched_works():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM works where title is NOT NULL")
    titles = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM works where abstract is NOT NULL")
    abstracts = cursor.fetchone()[0]
    conn.close()
    print(f"Fetched {titles} works with titles, {abstracts} with abstracts.")
    return titles, abstracts


def main():
    init_db()
    connector = OpenAlexConnector(email="example@wsl.org")
    all_ids = get_ids()
    fetch_works_for_all_ids(connector, all_ids)
    count_fetched_works()
    print('DONE!')


if __name__ == "__main__":
    main()