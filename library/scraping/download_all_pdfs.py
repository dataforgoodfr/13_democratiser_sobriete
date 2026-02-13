"""
Script to download all PDFs from the database scraping queue.

1. Get all papers from the scraping queue
2. For each paper, attempt to download the PDF (in parallel)
3. Update the database with success/failure status
4. Get text with pymupdf and save alongside PDF
"""

import argparse
from datetime import datetime
import os
import logging
import concurrent.futures

from library.database.download_queue_crud import get_papers_to_scrape, mark_paper_failed, mark_paper_scraped
from library.database.models import ScrapingQueue
from library.scraping.download_pdf import download_pdf
from library.scraping.extract_pdf_content import get_markdown_pymupdf
from selenium import webdriver
from tqdm import tqdm


OUTPUT_DIR = "outputs"
LIMIT = None  # Max number of papers to process in one run
EXTRACT_MD = False


def process_paper(paper: ScrapingQueue, output_dir: str, webdriver: webdriver.Chrome | None) -> tuple[bool, str]:
    """
    Tries downloading PDF and updates database status.

    Returns:
        Tuple of (success: bool, message: str)
    """
    if not paper.pdf_url:
        return True, f"Skipped {paper.openalex_id} - no PDF URL"

    try:
        logging.info(f"Downloading PDF for {paper.openalex_id}...")
        download_path, used_selenium = download_pdf(
            url=paper.pdf_url,
            output_path=f"{output_dir}/pdf/{paper.openalex_id}.pdf",
            webdriver=webdriver
        )

        if download_path:
            mark_paper_scraped(paper.openalex_id, download_path, used_selenium)

            if EXTRACT_MD:
                md_text = get_markdown_pymupdf(download_path)
                with open(f"{output_dir}/md/{paper.openalex_id}.md", "w") as f:
                    f.write(md_text)

            return True, f"Successfully downloaded {paper.openalex_id}"
        else:
            mark_paper_failed(paper.openalex_id, error_message="Download failed", used_selenium=used_selenium)
            return False, f"Download failed for {paper.openalex_id}"

    except Exception as e:
        mark_paper_failed(paper.openalex_id, error_message=str(e), used_selenium=False)
        return False, f"Error downloading {paper.openalex_id}: {str(e)}"


def main(max_workers: int = 10, resume_from: datetime | None = None):
    """
    Download all PDFs in parallel.
    """
    papers_to_scrape = get_papers_to_scrape(limit=LIMIT, resume_from=resume_from)
    os.makedirs(f"{OUTPUT_DIR}/pdf", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/md", exist_ok=True)

    print(f"Found {len(papers_to_scrape)} papers to scrape.")

    if not papers_to_scrape:
        print("No papers to scrape.")
        return

    # set webdriver to None to not use selenium at all
    webdriver = None  # start_webdriver(PDF_DIR)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for paper in papers_to_scrape:
            future = executor.submit(process_paper, paper, OUTPUT_DIR, webdriver)
            futures.append(future)

        completed = 0
        successes = 0
        failures = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                success, message = future.result()
                if success:
                    successes += 1
                    print(f"✓ {message}")
                else:
                    failures += 1
                    print(f"✗ {message}")
            except Exception as e:
                failures += 1
                print(f"✗ Exception in worker: {e}")

            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{len(papers_to_scrape)} papers processed")

    print("\nDownload complete!")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Total processed: {completed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download all PDFs from the database scraping queue."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of parallel workers for downloading PDFs (default: 10)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume downloading papers scraped after this datetime (ISO format)"
    )
    args = parser.parse_args()

    resume_from = datetime.fromisoformat(args.resume_from) if args.resume_from else None
    main(max_workers=args.max_workers, resume_from=resume_from)
