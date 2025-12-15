"""
Converts all pdf documents from a s3 folder to markdown files saved in another folder.
Delete files from local storage after processing to avoid disk full errors.
Doesn't accumulate results in memory to avoid OOM errors.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
from tqdm import tqdm

from library.database.database import create_tables
from library.database.text_extraction_queue_crud import (
    get_already_processed_ids,
    mark_paper_processed,
    mark_paper_failed,
)
from library.connectors.s3 import upload_to_s3
from library.scraping.extract_pdf_content import get_markdown_pymupdf


S3_HOST = "https://sufficiency-library.s3.fr-par.scw.cloud"
S3_PREFIX = "documents"
S3_BASE_URL = f"{S3_HOST}/{S3_PREFIX}"


def process_pdf(s3_folder: str, document_id: str) -> None:
    """Extract text from a single PDF and save it as md to S3."""

    s3_prefix = s3_folder.replace(S3_HOST + "/", "")

    try:
        pdf_filename = f"{document_id}.pdf"
        md_filename = f"{document_id}.md"

        try:
            pdf_url = f"{s3_folder}/pdf/{document_id}.pdf"
            os.system(f"wget -q {pdf_url} -O {pdf_filename}")
            md_text = get_markdown_pymupdf(pdf_filename)

            with open(md_filename, "w") as f:
                f.write(md_text)

            s3_md_key = f"{s3_prefix}/md/{document_id}.md"
            upload_to_s3(md_filename, s3_md_key)
        finally:
            if os.path.exists(pdf_filename):
                os.remove(pdf_filename)
            if os.path.exists(md_filename):
                os.remove(md_filename)

        mark_paper_processed(document_id, s3_folder)
        return True, document_id

    except Exception as e:
        mark_paper_failed(document_id, s3_folder, str(e))
        return False, document_id


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

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        try:
            futures = {executor.submit(process_pdf, s3_folder, doc_id) for doc_id in ids}
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    success, message = future.result()
                    if success:
                        print(f"✓ Success {message}")
                    else:
                        print(f"✗ Failed {message}")
                except Exception as e:
                    print(f"✗ Exception: {e}")
                finally:
                    # Explicitly remove the future from the set to free memory
                    futures.discard(future)

        except KeyboardInterrupt:
            print("\nInterrupted! Cancelling remaining tasks...")
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text from all PDFs in a S3 folder and save as markdown."
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
