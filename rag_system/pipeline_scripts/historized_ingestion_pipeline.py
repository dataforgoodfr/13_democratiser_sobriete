import os
import json
from argparse import ArgumentParser
import time
from pathlib import Path

import logfire

from kotaemon.base import Param, lazy
from pydantic import ValidationError

from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing
from kotaemon.llms import ChatOpenAI
from kotaemon.storages import QdrantVectorStore
from kotaemon.storages import LanceDBDocumentStore
from pipelineblocks.extraction.pdfextractionblock.pdf_to_markdown import \
    PdfExtractionToMarkdownBlock
from pipelineblocks.llm.ingestionblock.openai import OpenAIMetadatasLLMInference
from persist_taxonomy import get_open_alex_article, persist_article_metadata, reconcile_metadata
from taxonomy.paper_taxonomy import PaperTaxonomy


PDF_FOLDER = os.getenv("PDF_FOLDER", "./pipeline_scripts/pdf_test/")
config = {"api_key":"1"}
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.iffVHUO_-wh8xhUS9C3ydHN5zPdIUjd8tBz715mnvBQ"

# ---- Do not touch (temporary) ------------- #

ollama_host = "localhost"
qdrant_host = "116919ed-8e07-47f6-8f24-a22527d5d520.europe-west3-0.gcp.cloud.qdrant.io"

class HistorizedIndexingPipeline(VectorIndexing):
    pdf_extraction_block: PdfExtractionToMarkdownBlock = Param(
        lazy(PdfExtractionToMarkdownBlock).withx()
    )

    # At least, one taxonomy = one llm_inference_block
    # (Multiply the number of llm_inference_block when you need handle more than one taxonomy
    metadatas_llm_inference_block: OpenAIMetadatasLLMInference = Param(
        lazy(OpenAIMetadatasLLMInference).withx(
            llm=ChatOpenAI(
                base_url=f"http://{ollama_host}:11434/v1/",
                model="deepseek-r1:70b",
                api_key="ollama",
            ),
            taxonomy=PaperTaxonomy,
        )
    )

    # --- Final Kotaemon ingestion ----

    vector_store: QdrantVectorStore = Param(
        lazy(QdrantVectorStore).withx(
            url=f"https://{qdrant_host}:6333",
            api_key=api_key,
            collection_name="index_1",
        )
    )
    doc_store: LanceDBDocumentStore = Param(
        lazy(LanceDBDocumentStore).withx(
            path="./kotaemon-custom/kotaemon/ktem_app_data/user_data/docstore",
        ),
        ignore_ui=True,
    )
    embedding: OpenAIEmbeddings = Param(
        lazy(OpenAIEmbeddings).withx(
            base_url=f"http://{ollama_host}:11434/v1/",
            model="snowflake-arctic-embed2",
            api_key="ollama",
        ),
        ignore_ui=True,
    )

    pdf_path: str

    def run(self, pdf_path: str):
        """
        ETL pipeline for a single pdf file
        1. Extract text and taxonomy from pdf
        2. Transform taxonomy (flattening)
        3. Ingest text and taxonomy into the vector store

        Return nothing
        """
        tic = time.time()
        try:
            article_metadata = get_open_alex_article(pdf_path)
            print("found this one:")
            print(article_metadata)
            text_md = self.pdf_extraction_block.run(pdf_path, method="group_all")
        except Exception as e:
            print(e)
            logfire.error(e)
            return (False, str(pdf_path))

        try:
            llm_metadatas = self.metadatas_llm_inference_block.run(
                text_md, doc_type="entire_doc", inference_type="scientific", openalex_metadata=article_metadata
            )
            # Reconcile OpenAlex metadata with LLM output
            metadatas = reconcile_metadata(article_metadata, llm_metadatas)
        except ValidationError as e:
            print("Error happening during the text extraction")
            print(e)
            logfire.error(e)
            return (False, str(pdf_path))

        # Persist metadata to PostgreSQL
        try:
            print(f"Trying to persist article: {metadatas}")
            persist_article_metadata(metadatas)
        except Exception as e:
            print(e)
            print("Error happening during the metadata ingestion")
            print(e)
            logfire.error(e)
            return (False, str(pdf_path))

        metadatas_json = metadatas.model_dump()
        try:
            super().run(text=[text_md], metadatas=[metadatas_json])
        except Exception as e:
            print("Error happening during the vector ingestion")
            print(e)
            return (False, str(pdf_path))
        tac = time.time()

        print(f"Time taken: {tac - tic:.1f}")

        return (True, str(pdf_path))


def main():
    logfire.configure(token="pylf_v1_us_qTtmbDFpkfhFwzTfZyZrTJcl4C4lC7FhmZ65BgJ7dLDV")
    parser = ArgumentParser(description='Run pdf ingestion')
    parser.add_argument('--file-path', required=True, help='Path to the file')

    args = parser.parse_args()
    file_path = args.file_path
    folder_path = Path(file_path).parent
    logfire.notice("starting doc")
    indexing_pipeline = HistorizedIndexingPipeline(pdf_path=folder_path)
    print(f"Parsing document: {file_path}")

    indexing_pipeline.run(file_path)


if __name__ == "__main__":
    main()
