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
with open("secret.json") as f:
    config = json.load(f)

api_key = os.getenv("p", config["api_key"])

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
            print(text_md)
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
            logfire.error(e)
            return (False, str(pdf_path))

        # Persist metadata to PostgreSQL
        try:
            print(f"Trying to persist article: {metadatas}")
            persist_article_metadata(metadatas)
        except Exception as e:
            print("Error happening during the metadata ingestion")
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
    parser.add_argument('--file-path', help='Path to the file (single file mode)')
    parser.add_argument('--folder-path', help='Path to the folder (batch mode)')

    args = parser.parse_args()
    
    if args.file_path and args.folder_path:
        print("Error: Please specify either --file-path OR --folder-path, not both")
        return
    
    if not args.file_path and not args.folder_path:
        print("Error: Please specify either --file-path or --folder-path")
        return
    
    logfire.notice("starting doc processing")
    
    if args.file_path:
        # Single file mode (original behavior)
        file_path = args.file_path
        folder_path = Path(file_path).parent
        indexing_pipeline = HistorizedIndexingPipeline(pdf_path=folder_path)
        print(f"Parsing single document: {file_path}")
        indexing_pipeline.run(file_path)
        
    else:
        # Folder mode (new behavior)
        folder_path = Path(args.folder_path)
        if not folder_path.exists():
            print(f"Error: Folder {folder_path} does not exist")
            return
            
        # Get all PDF files in the folder
        pdf_files = list(folder_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return
            
        print(f"Found {len(pdf_files)} PDF files in {folder_path}")
        indexing_pipeline = HistorizedIndexingPipeline(pdf_path=folder_path)
        
        successful = 0
        failed = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                print(f"Processing [{i}/{len(pdf_files)}]: {pdf_file.name}")
                result = indexing_pipeline.run(str(pdf_file))
                if result[0]:  # success
                    successful += 1
                    print(f"✅ Successfully processed: {pdf_file.name}")
                else:
                    failed += 1
                    print(f"❌ Failed to process: {pdf_file.name}")
            except Exception as e:
                failed += 1
                print(f"❌ Error processing {pdf_file.name}: {e}")
                logfire.error(f"Error processing {pdf_file.name}: {e}")
                
        print("\n📊 Folder processing completed:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📋 Total: {len(pdf_files)}")


if __name__ == "__main__":
    main()
