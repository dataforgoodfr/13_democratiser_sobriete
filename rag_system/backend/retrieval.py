import asyncio
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from qdrant_client import QdrantClient

from config import settings

qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)


embedding_model = SentenceTransformer(settings.embedding_model, device="cpu")
ranker = Ranker(max_length=settings.max_length_reranker)

executor = ThreadPoolExecutor(max_workers=2)


async def embed_query(
    query: str, model: SentenceTransformer = embedding_model, dim: int = settings.embedding_dim
):
    """
    Embed the query using the provided SentenceTransformer model.
    Make the embedding call async using a ThreadPoolExecutor.
    """
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(executor, model.encode, query)
    return embedding[:dim]


async def retrieve_documents(
    query: str,
    top_k: int = 20,
    reranker: Ranker = ranker,
    client: QdrantClient = qdrant_client,
    collection_name: str = settings.qdrant_collection_name,
) -> list[dict]:
    """Retrieve top-k documents from Qdrant based on the query embedding."""
    query_embedding = await embed_query(query)
    hits = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
    )
    results = rerank_documents(reranker, query, hits.points)
    return results


def rerank_documents(ranker: Ranker, query: str, documents: list[dict]) -> list[dict]:
    """Rerank documents based on the query using the provided Ranker."""
    passages = [{"id": d.id, "text": d.payload["abstract"]} for d in documents]
    rerank_request = RerankRequest(
        query=query,
        passages=passages,
    )
    results = ranker.rerank(rerank_request)
    return results[: settings.k_rerank]
