import asyncio

from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from qdrant_client import QdrantClient

from config import settings

qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

collection_name = "library-test"
embedding_dim = 128
embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
embedding_model = SentenceTransformer(embedding_model_name, device="cpu")
num_results = 5
ranker = Ranker(max_length=1024)


async def embed_query(
    query: str, model: SentenceTransformer = embedding_model, dim: int = embedding_dim
) -> list[float]:
    """Embed the query using the provided SentenceTransformer model."""
    embedding = model.encode(query)
    return embedding[:dim]


async def retrieve_documents(
    query: str,
    top_k: int = 20,
    reranker: Ranker = ranker,
    client: QdrantClient = qdrant_client,
    collection_name: str = collection_name,
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
    return results[:num_results]
