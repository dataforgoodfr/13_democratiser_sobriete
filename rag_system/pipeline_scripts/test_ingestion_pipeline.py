import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import PosixPath

from fast_ingestion_pipeline import IndexingPipeline

PDF_FOLDER = os.getenv("PDF_FOLDER", "./pipeline_scripts/pdf_test/")

if __name__ == "__main__":
    indexing_pipeline = IndexingPipeline(pdf_path=PDF_FOLDER)
    directory = PosixPath(str(PDF_FOLDER))
    # os.chdir(str(directory)) # Why ?
    number_of_threads = 1

    with ThreadPoolExecutor(max_workers=number_of_threads) as executor:
        results_iterator = executor.map(indexing_pipeline.run, directory.iterdir())
        result = list(results_iterator)
    print(result)
