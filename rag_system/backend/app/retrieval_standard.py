import asyncio

from sentence_transformers import SentenceTransformer
import torch

from .config import settings
from .models import DocumentChunk
from .reranking import flashrank_rerank, llm_rerank, llm_rate_sufficiency
from .retrieval_shared import executor, qdrant_client


embedding_model = SentenceTransformer(
    settings.embedding_model,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
if embedding_model.device.type == "cuda":
    embedding_model = embedding_model.half()


async def embed_query(
    query: str,
    model: SentenceTransformer = embedding_model,
    dim: int = settings.embedding_dim,
):
    """Embed the query using the standard sentence-transformer retrieval model."""
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(executor, model.encode, query)
    return embedding[:dim]


async def retrieve_chunks(
    query: str,
    top_k: int = settings.k_vector_search,
    collection_name: str = settings.library_collection_name,
) -> list[DocumentChunk]:
    """Retrieve and rerank the top literature chunks for the standard chunk-first pipeline."""
    query_embedding = await embed_query(query)
    hits = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
    )
    chunks = [
        DocumentChunk(
            openalex_id=point.payload.get("openalex_id"),
            chunk_idx=point.payload.get("chunk_idx"),
            text=point.payload.get("text", ""),
            retrieved_rank=idx + 1,
        )
        for idx, point in enumerate(hits.points)
    ]

    if settings.rerank_method == "flashrank":
        reranked_chunks = await flashrank_rerank(query, chunks)
    elif settings.rerank_method == "llm":
        reranked_chunks = await llm_rerank(query, chunks)
    elif settings.rerank_method == "llm_sufficiency":
        reranked_chunks = await llm_rate_sufficiency(query, chunks)
    else:
        reranked_chunks = chunks

    for idx, chunk in enumerate(reranked_chunks):
        chunk.retrieved_rank = idx + 1
    return reranked_chunks