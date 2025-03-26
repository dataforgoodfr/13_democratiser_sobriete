from concurrent.futures import ThreadPoolExecutor
from pathlib import PosixPath

from rag_system.pipeline_scripts.fast_ingestion_pipeline import IndexingPipeline
import os

PDF_FOLDER = os.getenv("PDF_FOLDER", "./pipeline_scripts/pdf_test/")

if __name__ == "__main__":
    indexing_pipeline = IndexingPipeline(pdf_path=PDF_FOLDER)
    directory = PosixPath(str(PDF_FOLDER))
    os.chdir(str(directory))
    number_of_threads = 1

    with ThreadPoolExecutor(max_workers=number_of_threads) as executor:
        future = executor.map(indexing_pipeline.run, directory)
        result = future.result()