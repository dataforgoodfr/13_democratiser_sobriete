"""
Takes ids from sqlite db and save them to a txt file.
"""

import sqlite3


DB_PATH = "openalex_works_v2.db"
BATCH_SIZE = 1_000_000
OUTPUT_FILE = 'openalex_ids_0312.txt'


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM works")
    total_rows = cursor.fetchone()[0]
    print(f"Total rows to export: {total_rows}")
    
    # Open output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        offset = 0
        rows_written = 0
        
        while offset < total_rows:
            # Fetch batch
            cursor.execute(
                "SELECT id FROM works LIMIT ? OFFSET ?",
                (BATCH_SIZE, offset)
            )
            
            batch = cursor.fetchall()
            
            if not batch:
                break
            
            # Write batch to file
            for row in batch:
                value = row[0]
                # Handle None values
                if value is not None:
                    f.write(str(value) + '\n')
                    rows_written += 1
            
            offset += BATCH_SIZE
            print(f"Progress: {min(offset, total_rows)}/{total_rows} rows processed")
    
    conn.close()
    print(f"Export complete! {rows_written} values written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()