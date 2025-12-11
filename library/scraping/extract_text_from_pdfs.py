"""
Converts all pdf documents from a folder to markdown files saved in another folder.
Additionally, saves the extracted text along with the document names (normally OpenAlex IDs) into a parquet file.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from library.scraping.extract_pdf_content import get_markdown_pymupdf, save_markdown


def main(pdf_folder: str):
    markdown_folder = os.path.join(pdf_folder, "markdowns")
    parquet_path = os.path.join(pdf_folder, "texts.parquet")
    os.makedirs(markdown_folder, exist_ok=True)

    records = []
    for filename in tqdm(os.listdir(pdf_folder)):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            md_text = get_markdown_pymupdf(pdf_path)
            document_id = os.path.splitext(filename)[0]
            output_md_path = os.path.join(markdown_folder, f"{document_id}.md")
            save_markdown(md_text, output_md_path)
            records.append({"id": document_id, "extracted_text": md_text})

    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please provide the path to the folder containing PDF files."
    folder = sys.argv[1]
    main(folder)
