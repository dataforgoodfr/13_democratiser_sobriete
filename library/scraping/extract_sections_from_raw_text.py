"""
Takes parquet file with raw_text as input, cleans it and extract sections.
"""

import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import json

from library.scraping.extract_sections import extract_sections, patterns
from library.scraping.clean.cleaning_pipeline import cleaning_pipeline
from library.connectors.s3 import upload_to_s3


def process_text(text: str) -> dict[str, str]:
    cleaned_text = cleaning_pipeline(text)
    lines = cleaned_text.split("\n")
    sections = extract_sections(lines, patterns)
    sections["cleaned_text"] = cleaned_text
    return sections


def process_file(input_file: str, output_file: str, num_workers: int = None):
    df = pd.read_parquet(input_file)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_text, text): (idx, oa_id) for idx, (oa_id, text) in enumerate(zip(df.index, df["text"]))
        }

        with open(output_file, "w") as f:
            for future in tqdm(as_completed(futures), total=len(df), desc="Processing texts"):
                result = future.result()
                idx, oa_id = futures[future]
                result["openalex_id"] = oa_id
                f.write(json.dumps(result) + "\n")
                del futures[future]

    del df


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Process raw text from parquet file")

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting batch number (default: 1)",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=7,
        help="Ending batch number (default: 7)",
    )


    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )

    args = parser.parse_args()

    for i in range(args.start, args.end + 1):
        folder = f"outputs/batch_{i}"
        input_file = folder + "/raw_texts.parquet"
        output_file = folder + "/processed_texts.jsonl"
        print(f"Processing file: {input_file}")
        process_file(input_file, output_file, num_workers=args.workers)

    print(f"Processed data saved as {output_file}")
