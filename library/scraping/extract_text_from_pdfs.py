"""
Converts all pdf documents from a s3 folder to markdown files saved in another folder.
Additionally, saves the extracted text along with the document names (normally OpenAlex IDs) into a parquet file.
Avoids local storage as much as possible by using temporary files and S3.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os

import pandas as pd
from tqdm import tqdm

from library.database.database import create_tables
from library.database.text_extraction_queue_crud import (
    get_already_processed_ids,
    mark_paper_processed,
    mark_paper_failed,
)
from library.connectors.s3 import get_s3_client, upload_to_s3
from library.scraping.extract_pdf_content import get_markdown_pymupdf


S3_HOST = "https://sufficiency-library.s3.fr-par.scw.cloud"
S3_PREFIX = "documents"
S3_BASE_URL = f"{S3_HOST}/{S3_PREFIX}"

s3 = get_s3_client()


def process_pdf(s3_folder: str, document_id: str) -> dict:
    """Extract text from a single PDF, save it as md to S3 and return the record."""

    s3_prefix = s3_folder.replace(S3_HOST + "/", "")

    try:
        pdf_filename = f"{document_id}.pdf"
        md_filename = f"{document_id}.md"
        
        try:
            pdf_url = f"{s3_folder}/pdf/{document_id}.pdf"
            os.system(f"wget -q {pdf_url} -O {pdf_filename}")
            md_text = get_markdown_pymupdf(pdf_filename)
            
            with open(md_filename, 'w') as f:
                f.write(md_text)
            
            s3_md_key = f"{s3_prefix}/md/{document_id}.md"
            upload_to_s3(md_filename, s3_md_key, s3_client=s3)
        finally:
            if os.path.exists(pdf_filename):
                os.remove(pdf_filename)
            if os.path.exists(md_filename):
                os.remove(md_filename)

        mark_paper_processed(document_id, s3_folder)
        logging.info(f"Success {document_id}")
        return {"id": document_id, "text": md_text}

    except Exception as e:
        mark_paper_failed(document_id, s3_folder, str(e))
        logging.info(f"Failed {document_id}: {str(e)}")
        return {"id": document_id, "text": ""}


def main(s3_folder: str, num_workers: int = 1):
    create_tables()

    ids_url = f"{s3_folder}/ids.txt"
    ids_path = "ids.txt"
    os.system(f"wget -q {ids_url} -O {ids_path}")
    with open(ids_path, "r") as f:
        ids = [line.strip() for line in f.readlines()]

    print("Total documents to process:", len(ids))
    already_processed = get_already_processed_ids()
    ids = [doc_id for doc_id in ids if doc_id not in already_processed]
    print("Documents to process after filtering already processed:", len(ids))

    records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for document_id in ids:
            futures.append(executor.submit(process_pdf, s3_folder, document_id))

        for future in tqdm(futures, total=len(futures)):
            record = future.result()
            records.append(record)

    df = pd.DataFrame(records)
    parquet_filename = "extracted_texts.parquet"
    df.to_parquet(parquet_filename, index=False)
    upload_to_s3(parquet_filename, f"{s3_folder}/extracted_texts.parquet", s3_client=s3)
    if os.path.exists(parquet_filename):
        os.remove(parquet_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text from all PDFs in a S3 folder and save as markdown and parquet."
    )
    parser.add_argument(
        "--s3-folder", type=str, help="S3 sub-folder under documents/ containing the PDFs"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers for extracting text from PDFs (default: 10)",
    )

    args = parser.parse_args()
    main(s3_folder=f"{S3_BASE_URL}/{args.s3_folder}", num_workers=args.num_workers)
