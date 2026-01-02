"""
Takes parquet file with raw_text as input, cleans it and extract sections.
"""

import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

from library.scraping.extract_sections import extract_sections, patterns
from library.scraping.clean.cleaning_pipeline import cleaning_pipeline
from library.connectors.s3 import upload_to_s3


BASE = "https://sufficiency-library.s3.fr-par.scw.cloud"


def process_text(text: str) -> dict[str, str]:
    cleaned_text = cleaning_pipeline(text)
    lines = cleaned_text.split("\n")
    sections = extract_sections(lines, patterns)
    sections["cleaned_text"] = cleaned_text
    return sections


def process_file(input_file: str, num_workers: int = None):
    df = pd.read_parquet(input_file)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        sections_list = list(
            tqdm(
                executor.map(process_text, df["raw_text"]),
                total=len(df),
                desc="Processing texts",
            )
        )

    sections_df = pd.DataFrame(sections_list)
    result_df = pd.concat([df.drop(columns=["raw_text"]), sections_df], axis=1)
    return result_df


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Process raw text from parquet file")
    parser.add_argument(
        "--folder",
        help="Input folder on S3 (e.g., documents/batch_1). Should contain file raw_texts.parquet)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )

    args = parser.parse_args()

    input_file = BASE + "/" + args.folder + "/raw_texts.parquet"
    print(f"Processing file: {input_file}")

    result_df = process_file(input_file, num_workers=args.workers)
    result_df.to_parquet("/tmp/processed_texts.parquet")
    upload_to_s3("/tmp/processed_texts.parquet", args.folder + "/processed_texts.parquet")

    print(f"Processed data saved as {args.folder}/processed_texts.parquet on S3")
