import logging
import os
import time
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Tuple

from ktem.index.file.index import FileIndex
from ktem.index.file.pipelines import IndexPipeline
from llama_index.core.readers.file.base import default_file_metadata_func
from pipelineblocks.llm.ingestionblock.openai import OpenAIMetadatasLLMInference
from pydantic import BaseModel, ValidationError

# from persist_taxonomy import get_open_alex_article, persist_article_metadata, reconcile_metadata
from taxonomy.document import EntireDocument

from kotaemon.base import Document, Param, lazy
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing
from kotaemon.indices.splitters import TokenSplitter
from kotaemon.llms import ChatOpenAI
from kotaemon.loaders import PDFThumbnailReader

PDF_FOLDER = os.getenv("PDF_FOLDER", "./pipeline_scripts/pdf_test/")

# DOCSTORE_PATH = "/app/ktem_app_data/user_data/docstore" !!!
# Now, defined in the flowsettings.py (or in kotaemon for s3 path)

COLLECTION_ID = (
    1  # collection id is the integer at the end of the collection 'index_[ID] '
)
USER_ID = "25**********************"  # ! => Open the Kotaemon app ...
# go on a random page... see the logs to retrieve the USER ID !

METADATA_BASE = {
    "ingestion_method": "fast_script",
    "ingestion_origin_folder": PDF_FOLDER,
}  # This metadata base is added to all the chunks or docs ingested with this script

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256


# ---- Do not touch (temporary) ------------- #

OLLAMA_DEPLOYMENT = "docker"

ollama_host = "172.17.0.1" if OLLAMA_DEPLOYMENT == "docker" else "localhost"
# qdrant_host = "116919ed-8e07-47f6-8f24-a22527d5d520.europe-west3-0.gcp.cloud.qdrant.io"
#  ! to report in flowsetting.py !


# --- Additional logger ----

LOG_LEVEL = logging.INFO
# When you set the level, all messages from a higher level of severity are also
# logged. For example, when you set the log level to `INFO`, all `WARNING`,
# `ERROR` and `CRITICAL` messages are also logged, but `DEBUG` messages are not.
# Set a seed to enable reproducibility
SEED = 1
# Set a format to the logs.
LOG_FORMAT = "[%(levelname)s | " + " | %(asctime)s] - %(message)s"
# Name of the file to store the logs.
LOG_FILENAME = "script_execution.log"
# == Set up logging ============================================================
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    force=True,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILENAME, "a", "utf-8"), logging.StreamHandler()],
)


class RelevanceScore(BaseModel):
    relevance_score: float


class ExtractionError(Exception):
    pass


class IndexingPipelineShortCut(IndexPipeline):

    # --- PDF Extraction (optional... to replace Kotaemon Loader by default) ---

    """pdf_extraction_block : PdfExtractionToMarkdownBlock = Param(
        lazy(PdfExtractionToMarkdownBlock).withx(
        )
    )"""

    # --- LLM MODELS ---
    # At least, one taxonomy = one llm_inference_block
    # (Multiply the number of llm_inference_block when you need handle more than one taxonomy
    metadatas_llm_inference_block: OpenAIMetadatasLLMInference = Param(
        lazy(OpenAIMetadatasLLMInference).withx(
            llm=ChatOpenAI(
                base_url=f"http://{ollama_host}:11434/v1/",
                model="deepseek-r1:70b",
                api_key="ollama",
            ),
            taxonomy=EntireDocument,
        )
    )

    # --- EMBEDDINGS MODELS ---
    embedding: OpenAIEmbeddings = Param(
        lazy(OpenAIEmbeddings).withx(
            # base_url="http://172.17.0.1:11434/v1/",
            base_url=f"http://{ollama_host}:11434/v1/",
            model="snowflake-arctic-embed2",
            api_key="ollama",
        ),
        ignore_ui=True,
    )

    # --- Others params ---

    file_index_associated = FileIndex(
        app=None,
        id=COLLECTION_ID,
        name="sufficiency",
        config={
            "embedding": "default",
            "supported_file_types": ".png, .jpeg, .jpg, .tiff, .tif, .pdf, .xls, \
                  .xlsx, .doc, .docx, .pptx, .csv, .html, .mhtml, .txt, .md, .zip",
            "max_file_size": 1000,
            "max_number_of_files": 0,
            "private": True,
            "chunk_size": 0,
            "chunk_overlap": 0,
        },
    )

    file_index_associated.on_start()

    folder_path: str

    # ingestion_manager : IngestionManager = None

    Index: None | Any
    Source: None | Any
    collection_name: str
    FSPath: None | Path | str
    user_id: str
    loader: PDFThumbnailReader
    splitter: TokenSplitter
    vector_indexing: Callable[[IndexPipeline], VectorIndexing]

    def get_resources_set_up(self):

        self.VS = self.file_index_associated._vs
        self.DS = self.file_index_associated._docstore
        self.FSPath = self.file_index_associated._fs_path
        self.Index = self.file_index_associated._resources.get("Index")
        self.Source = self.file_index_associated._resources.get("Source")
        self.collection_name = f"index_{self.file_index_associated.id}"
        self.user_id = USER_ID
        self.loader = PDFThumbnailReader()
        self.splitter = TokenSplitter(
            chunk_size=CHUNK_SIZE or 1024,
            chunk_overlap=CHUNK_OVERLAP or 256,
            separator="\n\n",
            backup_separators=["\n", ".", "\u200B"],
        )
        self.vector_indexing = VectorIndexing(
            vector_store=self.VS, doc_store=self.DS, embedding=self.embedding
        )

    # --- CUSTOM PIPELINE LOGIC ----

    def concat__metadatas_layer(self, metadatas_base: dict, metadatas_root: dict):
        for key, value in metadatas_root.items():
            metadatas_base[key] = value
        return metadatas_base

    def enrich_metadatas_layer(
        self,
        metadatas_base: dict | None = None,
        doc_type: str = "unknown",
        inheritance_metadatas: dict | None = None,
        inheritance_fields_to_exclude: list | None = None,
        reapply_fields_to_root: list | None = None,
    ):
        """TODO Convert this function into method with a MetadatasManagement Object"""

        if metadatas_base is None:
            metadatas_base = {}
        metadatas_base["doc_type"] = doc_type

        if inheritance_metadatas is not None:

            applied_inheritance_metadatas = {}
            for key, value in inheritance_metadatas.items():
                if (
                    inheritance_fields_to_exclude is not None
                    and key in inheritance_fields_to_exclude
                ):
                    pass
                else:
                    applied_inheritance_metadatas[key] = value

            metadatas_base["extract_from"] = applied_inheritance_metadatas

            if reapply_fields_to_root is not None:

                for field in reapply_fields_to_root:

                    if field not in inheritance_metadatas.keys():
                        logging.warning(
                            f"Sorry, but the field {field} is not present in \
                            inheritance metadatas for reapplying :  {inheritance_metadatas.keys()}"
                        )
                    else:
                        metadatas_base[field] = inheritance_metadatas[field]

        return metadatas_base

    def custom_handle_docs(
        self, docs, file_id, file_path, article_metadata: dict | None = None
    ) -> int:
        s_time = time.time()

        # 1. Metadatas Extraction (and aggr.)

        try:

            text_docs = []
            non_text_docs = []
            thumbnail_docs = []
            other_vs_metadatas = []

            for doc in docs:
                doc_type = doc.metadata.get("type", "text")
                if doc_type == "text":
                    text_docs.append(doc)
                elif doc_type == "thumbnail":
                    thumbnail_docs.append(doc)
                else:
                    non_text_docs.append(doc)

            page_label_to_thumbnail = {
                doc.metadata["page_label"]: doc.doc_id for doc in thumbnail_docs
            }

            if self.splitter:
                all_chunks = self.splitter(text_docs)
            else:
                all_chunks = text_docs

            # add the thumbnails doc_id to the chunks
            for chunk in all_chunks:
                page_label = chunk.metadata.get("page_label", None)
                if page_label and page_label in page_label_to_thumbnail:
                    chunk.metadata["thumbnail_doc_id"] = page_label_to_thumbnail[
                        page_label
                    ]

            # ------------ CUSTOM LOGIC ---------------------
            # *** Example : let's make a llm inference on metadatas for entire doc ***

            text_md = "/n".join([doc.text for doc in text_docs])

            entire_doc = Document(text=text_md, id_=str(uuid.uuid4()))
            entire_doc.metadata = text_docs[0].metadata

            # MOCK
            # llm_metadatas = self.metadatas_llm_inference_block.run(text_md, doc_type='entire_doc')
            llm_metadatas = {"title_inference": "test", "author_inference": "test"}

            # Enrich metadatas with base functional metadatas
            metadatas_entire_doc = self.concat__metadatas_layer(
                metadatas_base=llm_metadatas, metadatas_root=METADATA_BASE
            )

            # Enrich metadatas with other functional metadatas
            metadatas_entire_doc = self.concat__metadatas_layer(
                metadatas_base={"doc_type": "entire_doc"},
                metadatas_root=metadatas_entire_doc,
            )

            # Enrich metadatas with article metadata if provided
            if article_metadata:
                metadatas_entire_doc = self.concat__metadatas_layer(
                    metadatas_base=article_metadata, metadatas_root=metadatas_entire_doc
                )

            # TODO later... inference chunks metadatas
            """text_vs_metadatas = self.inference_on_all_chunks(chunks=all_chunks,
                                                                metadata_vs_base=METADATA_BASE,
                                                                metadata_entire_doc = metadatas)"""
            # temporary unique metadatas for all chunks
            temporary_chunk_metadata = self.enrich_metadatas_layer(
                metadatas_base=METADATA_BASE,
                inheritance_metadatas=metadatas_entire_doc,
                inheritance_fields_to_exclude=[],  # here, we could exclude some fields
                reapply_fields_to_root=None,
            )

            # ------------ END CUSTOM LOGIC ---------------------

        except ValidationError:
            print("Error happening during the text extraction")
            # logfire.error(e)
            return 0

        # 2. Ingestion (Vectorstore, Docstore, internal sqldb)

        try:
            text_vs_metadatas = [
                temporary_chunk_metadata
                for _ in range(len(thumbnail_docs) + len(non_text_docs))
            ]  # temporary
            other_vs_metadatas = [
                METADATA_BASE for _ in range(len(thumbnail_docs) + len(non_text_docs))
            ]

            # All results to ingestion :

            to_index_chunks = all_chunks + non_text_docs + thumbnail_docs + [entire_doc]
            to_index_metadatas = (
                text_vs_metadatas + other_vs_metadatas + [metadatas_entire_doc]
            )

            # Add metadatas to chunks for doctstore (duplicate docstore + vectorstore)
            for chunk, metadatas in zip(to_index_chunks, to_index_metadatas):
                chunk.metadata.update(metadatas)

            logging.info(
                f"Got {len(thumbnail_docs)} text chunks - {len(thumbnail_docs)} \
                    page thumbnails - {len(non_text_docs)} other type chunks - 1 entire doc"
            )
            logging.info(f"And {len(to_index_metadatas)} metadatas list to index.")

            # /// DOC STORE Ingestion
            chunks = []
            n_chunks = 0
            chunk_size = self.chunk_batch_size * 4
            for start_idx in range(0, len(to_index_chunks), chunk_size):
                chunks = to_index_chunks[start_idx : start_idx + chunk_size]
                self.handle_chunks_docstore(chunks, file_id)
                n_chunks += len(chunks)

            # /// VECTOR STORE Ingestion
            def insert_chunks_to_vectorstore():
                chunks = []
                n_chunks = 0
                chunk_size = self.chunk_batch_size
                for start_idx in range(0, len(to_index_chunks), chunk_size):
                    chunks = to_index_chunks[start_idx : start_idx + chunk_size]
                    metadatas = to_index_metadatas[start_idx : start_idx + chunk_size]
                    self.handle_chunks_vectorstore(chunks, file_id, metadatas)
                    n_chunks += len(chunks)

            insert_chunks_to_vectorstore()
            """# run vector indexing in thread if specified
            if self.run_embedding_in_thread:
                print("Running embedding in thread")
                threading.Thread(
                    target=lambda: list(insert_chunks_to_vectorstore())
                ).start()
            else:
                yield from insert_chunks_to_vectorstore()"""

            print("indexing step took", time.time() - s_time)

        except Exception as e:
            print("Error happening during the vector ingestion")
            print(e)
            return 0

        # 3. Persist metadata to PostgreSQL
        try:
            print(f"Trying to persist article: {metadatas}")
            # persist_article_metadata(metadatas)
        except Exception:
            print("Error happening during the metadata ingestion")
            # logfire.error(e)
            return 0

        return n_chunks

    def run_one_file(
        self, file_path: str | Path, reindex: bool = False, **kwargs
    ) -> Tuple[bool, str]:

        tic = time.time()

        try:

            file_path = Path(file_path)  # TODO resolve path, file path...

            # check if the file is already indexed
            if isinstance(file_path, Path):
                file_path = file_path.resolve()

            file_id = self.get_id_if_exists(file_path)

            if isinstance(file_path, Path):
                if file_id is not None:

                    if not reindex:
                        raise ValueError(
                            f"File {file_path.name} already indexed. Please rerun with "
                            "reindex=True to force reindexing."
                        )
                    else:
                        # remove the existing records
                        self.delete_file(file_id)
                        file_id = self.store_file(file_path)

                else:
                    # add record to db
                    file_id = self.store_file(file_path)

            else:
                if file_id is not None:
                    raise ValueError(f"URL {file_path} already indexed.")
                else:
                    # add record to db
                    file_id = self.store_url(file_path)

            # extract the file
            if isinstance(file_path, Path):
                extra_info = default_file_metadata_func(str(file_path))
                file_path.name
            else:
                extra_info = {"file_name": file_path}

            extra_info["file_id"] = file_id
            extra_info["collection_name"] = self.collection_name

            docs = self.loader.load_data(file_path, extra_info=extra_info)
            logging.info("document extracted... ok.")

            # ---- MOCK ADDITIONAL METADATA EXTRACTED (BASE) -----------
            pdf_article_metadata = {"title": "test"}

            nb_chunks = self.handle_docs(docs, file_id, file_path, pdf_article_metadata)

            logging.info(
                f" Ingestion OK ! --- Nb chunks send to docstore & vector store : {nb_chunks}"
            )

            tac = time.time()

            print(f"Time taken: {tac - tic:.1f}")

            return (True, str(file_path))

        except Exception as e:
            print(e)
            # logfire.error(e)
            return (False, str(file_path))


def main():
    # logfire.configure(token="pylf_v1_us_qTtmbDFpkfhFwzTfZyZrTJcl4C4lC7FhmZ65BgJ7dLDV")
    parser = ArgumentParser(description="Run pdf ingestion")
    parser.add_argument("--file-path", required=True, help="Path to the file")
    parser.add_argument(
        "-fr",
        "--force_reindex",
        action="store_true",
        help="Force to reindex all the pdf files in the folder",
    )

    args = parser.parse_args()
    file_path = args.file_path
    folder_path = Path(file_path).parent
    # logfire.notice("starting doc")

    indexing_pipeline = IndexingPipelineShortCut(folder_path=folder_path)

    indexing_pipeline.get_resources_set_up()

    print(f"Parsing document: {file_path}")

    result = indexing_pipeline.run_one_file(
        file_path=file_path, reindex=args.force_reindex
    )

    print(result)


if __name__ == "__main__":
    main()
