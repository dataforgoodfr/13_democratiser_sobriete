import asyncio
import time
import os
from pathlib import Path
import pandas as pd
import numpy as np
import sys
from typing import Dict, Any

from kotaemon.base import Document
from kotaemon.storages import (
    ChromaVectorStore,
    InMemoryDocumentStore
)

from kotaemon.storages import BaseDocumentStore, BaseVectorStore
from kotaemon.indices.splitters import TokenSplitter
from kotaemon.indices import VectorIndexing
from kotaemon.embeddings import LCHuggingFaceEmbeddings, OpenAIEmbeddings
from kotaemon.rerankings import BaseReranking

from llama_index.core import SimpleDirectoryReader
from benchmark.utils.custom_file_reader import SimplePDFReader, PdfExtractionReader
from benchmark.utils.utils import (
    create_collection_name,
    read_args,
    load_config,
)
from benchmark.retrieval_evaluation.metrics import resolve_metrics

from benchmark.retrieval_evaluation.custom_evaluator import (
    CustomRetrieverEvaluator,
    postprocess_nodes,
    save_results,
    EmbeddingQAFinetuneDataset,
)

from benchmark.retrieval_evaluation.hf_rerankers import CrossEncoderReranking


async def load_corpus(
    folder_path, num_files_limit, pdf_reader, chunk_size, chunk_overlap, verbose=False
):
    if verbose:
        print(f"Loading PDFs from {folder_path}")

    reader = SimpleDirectoryReader(
        input_dir=folder_path,
        file_extractor={".pdf": pdf_reader},
        recursive=False,
        required_exts=[".pdf"],
        num_files_limit=num_files_limit,
        raise_on_error=False,
    )
    docs = reader.load_data(num_workers=4, show_progress=True)
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = TokenSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
        include_metadata=True,
        include_prev_next_rel=True,
    )
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)
    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


def initialize_vector_doc_store(
    vector_index: VectorIndexing,
    vector_store: BaseVectorStore,
    doc_store: BaseDocumentStore,
    documents: list[Document],
    collection_name: str,
    doc_store_folder: str,
    vector_store_folder,
    verbose: bool = False,
) -> None:

    # Ensure document store folder exists
    os.makedirs(doc_store_folder, exist_ok=True)
    os.makedirs(vector_store_folder, exist_ok=True)

    # Define a doc store filename
    doc_store_file = os.path.join(doc_store_folder, collection_name + ".json")

    # Load existing document store if available
    if os.path.isfile(doc_store_file):
        doc_store.load(path=doc_store_file)
        doc_count = doc_store.count()
        if verbose:
            print(f"{doc_count} documents have been loaded")
            if doc_count == 0:
                print("Warning: Document Store file exists but contains no documents!")
    else:
        print(f"Creating a new Document Store at {doc_store_file}")

    vector_count = vector_store.count()
    doc_count = doc_store.count()

    if vector_count == 0 and doc_count == 0:
        if verbose:
            print("Vector Store and Document Store are empty.")
        vector_index.run(documents)
        vector_index.doc_store.save(path=doc_store_file)
        if verbose:
            print("Indexing completed.")

    elif vector_count != doc_count:
        if verbose:
            print(
                f"Mismatch detected!\n"
                f"Vector Store contains {vector_count} documents.\n"
                f"Document Store contains {doc_count} documents.\n"
                "Re-indexing to maintain consistency..."
            )
        vector_index.vector_store.drop()  # Delete the collection
        vector_index.vector_store = ChromaVectorStore(
            path=vector_store_folder, collection_name=collection_name
        )
        vector_index.doc_store.drop()  # Ensures old data is removed

        vector_index.run(documents)
        vector_index.doc_store.save(path=doc_store_file)
        if verbose:
            print("Re-indexing completed. Stores are now synchronized.")

    else:
        if verbose:
            print("Vector Store and Document Store are already synchronized.")


def choose_embedding_model(embedding_spec: Dict[str, Any]):
    """
    Selects and returns an embedding model based on the provided specifications.

    Args:
        embedding_spec (dict): Dictionary containing provider details and available models.
        i (int): Index of the model to select from the list.

    Returns:
        An embedding model object compatible with LangChain.

    Raises:
        ValueError: If the provider is not supported or the index is out of range.
        KeyError: If essential keys are missing in `embedding_spec`.
    """
    provider = embedding_spec.get("provider")
    model_name = embedding_spec.get("model")

    if provider == "HuggingFace":
        return LCHuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    elif provider == "ollama":
        return OpenAIEmbeddings(
            api_key=embedding_spec.get("api_key"),
            base_url=embedding_spec.get("base_url", "http://localhost:11434/v1/"),
            model=model_name,
        )
    else:
        return None


def choose_reranker(reranker_spec):
    provider = reranker_spec.get("provider")
    model_name = reranker_spec.get("model")

    if provider == "HuggingFace":
        return CrossEncoderReranking(
            model_name=model_name, is_truncated=True, max_tokens=512, batch_size=6
        )
    else:
        return None


async def main():
    # Load config
    args = read_args()
    config = load_config(args.config_path)
    pdf_folder_path = str(config.get("source_folder"))  # Path to PDFs

    embedding_spec = config.get("embedding_spec")
    embedding_name = embedding_spec.get("model")  # Embedding model name
    reranker_spec = config.get("reranker_spec", None)
    db_path = config.get("vector_db_path", "./chromadb")  # Path to vector DB
    chunk_size = config.get("chunk_size", 512)  # Chunk size for splitting
    chunk_overlap = config.get("chunk_overlap", 20)  # Overlapping tokens
    num_files_limit = config.get("num_files_limit", -1)  # Max number of PDFs to process
    doc_store_folder = config.get("doc_db_path", "./docstore")
    verbose = config.get("verbose", False)
    # Get Retrieval Hyperparameters
    first_round_top_k_mult = config.get("first_round_top_k_mult", 10)
    top_k = config.get("top_k", 5)
    retrieval_mode = config.get("retrieval_mode", "vector")
    do_extend = config.get("do_extend", False)
    thumbnail_count = config.get("thumbnail_count", 0)

    # evaluation metrics
    evaluation_spec = config.get("evaluation_spec")
    metrics = evaluation_spec.get("metrics", ["hit_rate", "recall"])
    k_values = evaluation_spec.get("k_values", [1])

    if num_files_limit == -1:
        num_files_limit = int(
            sum(
                1
                for file in os.listdir(pdf_folder_path)
                if file.lower().endswith(".pdf")
            )
        )

    output_folder = config.get("output_folder", "./outputs/")
    output_filename = config.get("output_filename", "evaluation.csv")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(doc_store_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    query_file_path = config.get("query_file_path", None)  # Path to query file
    pdf_reader = PdfExtractionReader()  # or PDFThumbnailReader()

    # Load Corpus documents
    list_of_nodes = await load_corpus(
        pdf_folder_path,
        num_files_limit,
        pdf_reader,
        chunk_size,
        chunk_overlap,
        verbose=True,
    )
    list_of_documents = [Document(**node.to_dict()) for node in list_of_nodes]

    # Create a collection name
    collection_name = create_collection_name(
        embedding_model=embedding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Define vector store
    vector_store = ChromaVectorStore(path=db_path, collection_name=collection_name)
    # Define document store
    doc_store = (
        InMemoryDocumentStore()
    )  # ElasticSearch use BM25 (a TF IDF powerful extension) algo to retrieve documents / InMemoryDocumentStore does not participate to retrieval process

    # Choos Embedding and Reranker Model
    embed_model = choose_embedding_model(embedding_spec)
    reranker_model = choose_reranker(reranker_spec)

    vector_index = VectorIndexing(
        vector_store=vector_store, embedding=embed_model, doc_store=doc_store
    )

    initialize_vector_doc_store(
        vector_index,
        vector_store,
        doc_store,
        documents=list_of_documents,
        collection_name=collection_name,
        doc_store_folder=doc_store_folder,
        vector_store_folder=db_path,
        verbose=verbose,
    )

    retriever = vector_index.to_retrieval_pipeline(
        top_k=top_k,
        first_round_top_k_mult=first_round_top_k_mult,
        retrieval_mode=retrieval_mode,
        rerankers=[reranker_model] if reranker_model else None,
    )
    # retriever.run takes some parameters like
    # # ['do_extend', 'scope', "thumbnail_count"]

    metrics = resolve_metrics(metrics=metrics)

    print(f"Start evaluating the retriever for {embedding_name}")

    retriever_evaluator = CustomRetrieverEvaluator(
        metrics=metrics,
        retriever=retriever,
        node_postprocessor=postprocess_nodes,
        do_extend=do_extend,
        thumbnail_count=thumbnail_count,
        k_values=k_values,
    )

    qa_dataset = EmbeddingQAFinetuneDataset.from_json(path=query_file_path)

    eval_results = await retriever_evaluator.aevaluate_dataset(
        dataset=qa_dataset, workers=4, show_progress=True, k_values=k_values
    )

    reranker_name = reranker_spec.get("model")
    name = config.get("config_name")
    if not name:
        if reranker_model and reranker_name:
            name = f"{collection_name}_re_{reranker_name.replace('/', '-')}"
        else:
            name = collection_name

    save_results(name=name, eval_results=eval_results, output_file=output_path)

    print(f"Evaluation of {embedding_name} completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
