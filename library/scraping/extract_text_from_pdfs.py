"""
Converts all pdf documents from a folder to markdown files saved in another folder.
Additionally, saves the extracted text along with the document names (normally OpenAlex IDs) into a parquet file.
"""

from concurrent.futures import ProcessPoolExecutor
import os
import sys
import pandas as pd
from tqdm import tqdm
from library.scraping.extract_pdf_content import get_markdown_pymupdf, save_markdown


def process_pdf(pdf_folder: str, filename: str) -> dict:
    """Process a single PDF file and return the record."""
    pdf_path = os.path.join(pdf_folder, filename)
    md_text = get_markdown_pymupdf(pdf_path)
    document_id = os.path.splitext(filename)[0]
    output_md_path = os.path.join(pdf_folder, "markdowns", f"{document_id}.md")
    save_markdown(md_text, output_md_path)
    return {"id": document_id, "extracted_text": md_text}


def main(pdf_folder: str, num_workers: int = 1):
    markdown_folder = os.path.join(pdf_folder, "markdowns")
    parquet_path = os.path.join(pdf_folder, "texts.parquet")
    os.makedirs(markdown_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for filename in pdf_files:
            futures.append(executor.submit(process_pdf, pdf_folder, filename))

        for future in tqdm(futures, total=len(futures)):
            record = future.result()
            records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_text_from_pdfs.py <pdf_folder> [num_workers]")
        print("  pdf_folder: Path to the folder containing PDF files")
        print("  num_workers: Number of parallel workers (default: 1)")
        sys.exit(1)

    pdf_folder = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    main(pdf_folder, num_workers)
