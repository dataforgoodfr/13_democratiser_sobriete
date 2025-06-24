import time
import uuid

from pathlib import Path
from typing import Any, Callable
import threading

from decouple import config

from ktem.index.file.index import FileIndex
from ktem.index.file.pipelines import IndexPipeline
from llama_index.core.readers.file.base import default_file_metadata_func
from pipelineblocks.llm.ingestionblock.openai import OpenAIMetadatasLLMInference

from taxonomy.paper_taxonomy import PaperTaxonomy

from kotaemon.base import Document, Param, lazy
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing
from kotaemon.indices.splitters import TokenSplitter
from kotaemon.llms import ChatOpenAI
from kotaemon.loaders import PDFThumbnailReader, WebReader

from fast_ingestion.persist_taxonomy import reconcile_metadata, persist_article_metadata
from fast_ingestion.logging_config import configure_logging

logger = configure_logging()


CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256

DEFAULT_INGESTION_VERSION = "version_1"


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
                base_url=config("LLM_INFERENCE_URL", "http://localhost:11434/v1/"),
                model=config("LLM_INFERENCE_MODEL", "llama3.2:3b"), #TODO replace by "deepseek-r1:70b" ?
                api_key=config("LLM_INFERENCE_API_KEY", "ollama")
            ),
            taxonomy=PaperTaxonomy,
        )
    )

    # --- EMBEDDINGS MODELS ---
    embedding: OpenAIEmbeddings = Param(
        lazy(OpenAIEmbeddings).withx(
            # base_url="http://172.17.0.1:11434/v1/",
            base_url=config("EMBEDDING_MODEL_URL", "http://localhost:11434/v1/"),
            model=config("EMBEDDING_MODEL", "snowflake-arctic-embed2"),
            api_key=config("EMBEDDING_MODEL_API_KEY", "ollama")
        ),
        ignore_ui=True,
    )

    # --- Others params ---

    file_index_associated = FileIndex(
        app=None,
        id=config("COLLECTION_ID", 1),
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
        self.user_id = config("USER_ID", "123456")
        self.loader = WebReader()
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
                        logger.warning(
                            f"Sorry, but the field {field} is not present in \
                            inheritance metadatas for reapplying :  {inheritance_metadatas.keys()}"
                        )
                    else:
                        metadatas_base[field] = inheritance_metadatas[field]

        return metadatas_base

    def custom_handle_docs(
        self,
        docs,
        file_id,
        article_metadata: dict | None = None,
        ingestion_version: str = DEFAULT_INGESTION_VERSION
    ) -> int:
        
        s_time = time.time()

        # 1. Metadatas Extraction (and aggr.)

        logger.info("indeking pipeline started...")

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

            logger.info(f"file_id : {file_id}  LLM Inference...")

            try:
                llm_metadatas = self.metadatas_llm_inference_block.run(text_md,
                                                                   doc_type='entire_doc',
                                                                   inference_type = 'scientific_advanced',
                                                                   existing_metadata = article_metadata)
                logger.info(f"file_id : {file_id}  - LLM Inference Done.")

            except Exception as e:
                raise Exception(f"file_id : {file_id} - Error happening during LLM inference, error : {e}") from e


            # Reconcile metadatas
            reconciled_metadata = reconcile_metadata(article_metadata, llm_metadatas)

            # Persist metadata to PostgreSQL
            try:
                logger.info(f"file_id : {file_id} - Trying to persist article...")
                result_id = persist_article_metadata(reconciled_metadata)
                logger.info(f"file_id : {file_id} - Metadatas persistance ok with id : {result_id} (external db,  (table for metadatas statistics))")
            except Exception as e:
                raise Exception(f"file_id : {file_id} - Error happening during the metadata ingestion (table for metadatas statistics)") from e
                #logfire.error(e)
            
            reconciled_metadata = reconciled_metadata.model_dump() #convert to dict

            base_metadata = {"ingestion_version": ingestion_version,
                             "ingestion_method": "fast_script"}

            # Enrich metadatas with base functional metadatas
            metadatas_entire_doc = self.concat__metadatas_layer(
                metadatas_base=reconciled_metadata,
                metadatas_root=base_metadata
            )

            # Enrich metadatas with other functional metadatas
            metadatas_entire_doc = self.concat__metadatas_layer(
                metadatas_base={"doc_type": "entire_doc"},
                metadatas_root=metadatas_entire_doc,
            )

            # TODO later... inference chunks metadatas
            """text_vs_metadatas = self.inference_on_all_chunks(chunks=all_chunks,
                                                                metadata_vs_base=METADATA_BASE,
                                                                metadata_entire_doc = metadatas)"""
            # temporary unique metadatas for all chunks
            temporary_chunk_metadata = self.enrich_metadatas_layer(
                metadatas_base=base_metadata,
                inheritance_metadatas=metadatas_entire_doc,
                inheritance_fields_to_exclude=[],  # here, we could exclude some fields
                reapply_fields_to_root=None,
            )

            # ------------ END CUSTOM LOGIC ---------------------

        except Exception as e:
            raise Exception(f"file_id : {file_id} - Error happening during the text extraction / inference / metadata. error : {e}") from e
            # logfire.error(e)

        # 2. Ingestion (Vectorstore, Docstore, internal sqldb)

        try:
            text_vs_metadatas = [
                temporary_chunk_metadata
                for _ in range(len(all_chunks))
            ]  # temporary
            other_vs_metadatas = [
                metadatas_entire_doc for _ in range(len(thumbnail_docs) + len(non_text_docs))
            ]

            # All results to ingestion :

            to_index_chunks = all_chunks + non_text_docs + thumbnail_docs + [entire_doc]
            to_index_metadatas = text_vs_metadatas + other_vs_metadatas + [metadatas_entire_doc]

            # Add metadatas to chunks for doctstore (duplicate docstore + vectorstore)
            for chunk, metadatas in zip(to_index_chunks, to_index_metadatas, strict=False):
                chunk.metadata.update(metadatas)

            logger.info(
                f"file_id : {file_id} - Got {len(all_chunks)} text chunks - {len(thumbnail_docs)} \
                    page thumbnails - {len(non_text_docs)} other type chunks - 1 entire doc"
            )
            logger.info(f"And {len(to_index_metadatas)} metadatas list to index.")

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

            # TODO # Re-implemenent Multi-Threadng here
            #insert_chunks_to_vectorstore()
            # run vector indexing in thread if specified
            if self.run_embedding_in_thread:
                logger.info("Running embedding in thread")
                threading.Thread(
                    target=lambda: list(insert_chunks_to_vectorstore())
                ).start()
            else:
                insert_chunks_to_vectorstore()

            logger.info(f"file_id : {file_id} - indexing step took {time.time() - s_time:.2f}s")

        except Exception as e:
            raise Exception(f"file_id : {file_id} - Error happening during the vector ingestion - error : {e}") from e

        logger.info(f"indexing step took {time.time() - s_time:.2f}s")
        return n_chunks

    def run_one_file(
        self,
        file_path: str | Path,
        reindex: bool = False,
        article_metadatas: dict = None,
        ingestion_version: str = DEFAULT_INGESTION_VERSION,
        **kwargs
    ) -> str:

        tic = time.time()

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

        else: # if file_path is a URL
            if file_id is not None:

                if not reindex:
                    raise ValueError(f"URL {file_path} already indexed. Please rerun with "
                                        "reindex=True to force reindexing.")
                else:
                    # add record to db
                    self.delete_file(file_id)
                    file_id = self.store_url(file_path)

            else:
                # add record to db
                file_id = self.store_url(file_path)

        # extract the file
        if isinstance(file_path, Path):
            extra_info = default_file_metadata_func(str(file_path))
        else:
            extra_info = {"file_name": file_path}

        extra_info["file_id"] = file_id
        extra_info["collection_name"] = self.collection_name

        docs = self.loader.load_data(file_path, extra_info=extra_info)

        logger.info("document extracted... ok.")

        nb_chunks = self.custom_handle_docs(docs,
                                            file_id,
                                            article_metadatas,
                                            ingestion_version=ingestion_version)

        self.finish(file_id, file_path)

        tac = time.time()
        logger.info(f"INGESTION OK - file_id : {file_id} - nb_chunks : {nb_chunks} Time taken: {tac - tic:.1f}")

        return file_id                              
