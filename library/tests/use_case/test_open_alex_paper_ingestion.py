from icecream import ic
from library.usecase.open_alex_paper_ingestion import OpenAlexPaperIngestionUseCase
# from library.taxonomy.paper_taxonomy import OpenAlexPaper, PaperWithText

from library.scraping import extract_openalex as OpenAlexClient
from library.pdfextraction.llm import ollama_extraction as LlmClient
from library.pdfextraction.pdf import extract_pdf_content as PDFExtractor
# TODO : Add the JSON_STORAGE_FOLDER to the usecase, to store the extracted taxonomy
from library.infra import JSON_STORAGE_FOLDER, PDF_STORAGE_FOLDER # noqa: F401

my_instance = OpenAlexPaperIngestionUseCase(
    OpenAlexClient, LlmClient, PDFExtractor, JSON_STORAGE_FOLDER
)

my_instance.ingest_papers_with_query(
    query="construction", # requete complexe sur la construction
    limit=None,
    model="mistral-small:latest",
    prompt_type="basic"
    # , prompt_type="main_parts"
)

ic('Hell yeah!')
