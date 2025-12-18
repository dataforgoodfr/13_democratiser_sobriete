"""
Converts all pdf documents from a s3 folder to markdown files saved in another folder.
Delete files from local storage after processing to avoid disk full errors.
Doesn't accumulate results in memory to avoid OOM errors.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import random
import shutil
import sys
from tqdm import tqdm

from library.database.database import create_tables
from library.database.text_extraction_queue_crud import (
    get_already_processed_ids,
    mark_paper_processed,
    mark_paper_failed,
)
from library.connectors.s3 import upload_to_s3, get_s3_client
from library.scraping.extract_pdf_content import (
    get_markdown_pymupdf,
    get_raw_text_pymupdf,
    save_text,
)


S3_HOST = "https://sufficiency-library.s3.fr-par.scw.cloud"  # TODO env variable
S3_PREFIX = "documents"
S3_BASE_URL = f"{S3_HOST}/{S3_PREFIX}"

TXT_TIMEOUT = 30  # seconds
MD_TIMEOUT = 600  # seconds

s3 = get_s3_client()


def process_pdf(
    s3_folder: str,
    document_id: str,
    markdown: bool = False,
    max_pages: int = 20,
    ocr: bool = False,
) -> None:
    """Extract text from a single PDF and save it as md to S3."""

    s3_prefix = s3_folder.replace(S3_HOST + "/", "")
    mode = "md" if markdown else "txt"

    try:
        os.makedirs("tmp", exist_ok=True)
        pdf_filename = f"tmp/{document_id}.pdf"
        output_filename = f"tmp/{document_id}.{mode}"

        pdf_url = f"{s3_folder}/pdf/{document_id}.pdf"
        os.system(f"wget -q {pdf_url} -O {pdf_filename}")
        if markdown:
            text, used_ocr = get_markdown_pymupdf(
                pdf_filename, max_pages_at_once=max_pages, ocr=ocr
            )
        else:
            text = get_raw_text_pymupdf(pdf_filename)

        save_text(text, output_filename)

        s3_key = f"{s3_prefix}/{mode}/{document_id}.{mode}"
        upload_to_s3(output_filename, s3_key, s3)

        mark_paper_processed(document_id, s3_folder, mode)

        msg = f"{document_id} in {s3_prefix}"
        if markdown:
            msg += f" (OCR used: {used_ocr})"
        return True, msg
    except Exception as e:
        mark_paper_failed(document_id, s3_folder, str(e), mode)
        print("Faulty", document_id)
        return False, f"{document_id} in {s3_prefix} - Error: {str(e)}"
    finally:
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)


def get_ids_for_folder(s3_folder: str) -> list[str]:
    ids_url = f"{s3_folder}/ids.txt"
    ids_path = "ids.txt"
    os.system(f"wget -q {ids_url} -O {ids_path}")
    with open(ids_path, "r") as f:
        ids = [line.strip() for line in f.readlines()]
    return ids


def clean_tmp():
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    os.makedirs("tmp", exist_ok=True)


def main(markdown: bool, num_workers: int, max_pages: int | None, limit: int | None, ocr: bool):
    clean_tmp()
    create_tables()
    already_processed = get_already_processed_ids(mode="md" if markdown else "txt")
    s3_folders = [f"{S3_BASE_URL}/batch_{i}" for i in range(1, 7)]

    all_tasks = []
    for s3_folder in s3_folders:
        ids = get_ids_for_folder(s3_folder)
        for doc_id in ids:
            if doc_id not in already_processed:
                all_tasks.append((s3_folder, doc_id))

    random.shuffle(all_tasks)  # avoids getting stuck on a corrupted PDF that generates errors

    if limit is not None:
        all_tasks = all_tasks[:limit]

    # Process in batches to ensure workers are refreshed and prevent a single segfault from killing the entire run.
    # this replaces max_tasks_per_child of ProcessPoolExecutor, which doesn't work properly
    batch_size = num_workers * 10

    with tqdm(total=len(all_tasks)) as pbar:
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i : i + batch_size]
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                try:
                    futures = {
                        executor.submit(
                            process_pdf, s3_folder, doc_id, markdown, max_pages, ocr
                        )
                        for s3_folder, doc_id in batch
                    }

                    for future in as_completed(futures):
                        try:
                            success, message = future.result(
                                timeout=MD_TIMEOUT if markdown else TXT_TIMEOUT
                            )
                            if success:
                                print(f"✓ Success {message}")
                            else:
                                print(f"✗ Failed {message}")
                        except Exception as e:
                            print(f"✗ Exception: {e}")
                        finally:
                            pbar.update(1)

                except KeyboardInterrupt:
                    print("\nInterrupted! Cancelling remaining tasks...")
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    sys.exit(0)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # avoids memory issues vs fork

    parser = argparse.ArgumentParser(
        description="Extract text from all PDFs in a S3 folder and save as markdown."
    )
    parser.add_argument(
        "--md",
        action="store_true",
        help="Convert to markdown instead of raw text (default: False)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers for extracting text from PDFs (default: 10)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to process at once in a single PDF (default: None). Only used in markdown mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of PDFs to process (default: all)",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Use OCR fallback when extracting markdown from PDFs (default: False)",
    )

    args = parser.parse_args()
    main(
        markdown=args.md,
        num_workers=args.num_workers,
        max_pages=args.max_pages,
        limit=args.limit,
        ocr=args.ocr,
    )
