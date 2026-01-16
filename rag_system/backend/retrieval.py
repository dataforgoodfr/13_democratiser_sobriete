import asyncio
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from pyalex import Works

from config import settings
from models import DocumentChunk, Publication
from reranking import flashrank_rerank, llm_rerank


qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)
embedding_model = SentenceTransformer(settings.embedding_model, device="cpu").half()
executor = ThreadPoolExecutor(max_workers=2)


async def embed_query(
    query: str,
    model: SentenceTransformer = embedding_model,
    dim: int = settings.embedding_dim,
):
    """
    Embed the query using the provided SentenceTransformer model.
    Make the embedding call async using a ThreadPoolExecutor.
    """
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(executor, model.encode, query)
    return embedding[:dim]


async def retrieve_chunks(
    query: str,
    top_k: int = settings.k_vector_search,
    client: QdrantClient = qdrant_client,
    collection_name: str = settings.qdrant_collection_name,
) -> list[DocumentChunk]:
    """Retrieve top-k chunks from Qdrant based on the query embedding."""
    query_embedding = await embed_query(query)
    hits = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
    )
    chunks = [
        DocumentChunk(
            openalex_id=d.payload.get("openalex_id"),
            chunk_idx=d.payload.get("chunk_idx"),
            text=d.payload.get("text", ""),
            retrieved_rank=idx + 1,
        )
        for idx, d in enumerate(hits.points)
    ]
    if settings.rerank_method == "flashrank":
        reranked_chunks = await flashrank_rerank(query, chunks)
    elif settings.rerank_method == "llm":
        reranked_chunks = await llm_rerank(query, chunks)
    else:
        reranked_chunks = chunks
    # apply new rank
    for idx, chunk in enumerate(reranked_chunks):
        chunk.retrieved_rank = idx + 1
    return reranked_chunks


def get_publications_from_chunks(chunks: list[DocumentChunk]) -> list[Publication]:
    """Fetch publications from OpenAlex for the given chunks."""
    ids = [chunk.openalex_id for chunk in chunks]
    fields = [
        "id",
        "title",
        "doi",
        "abstract_inverted_index",
        "open_access",
        "authorships",
        "publication_year",
    ]
    works = Works().filter(openalex_id="|".join(ids)).select(fields).get()
    publications = []
    for work in works:
        authors = [
            f"{auth.get('author', {}).get('display_name')}"
            for auth in work.get("authorships", [])
        ]
        openalex_id = work["id"].split("/")[-1]
        publication = Publication(
            openalex_id=openalex_id,
            doi=work.get("doi"),
            title=work["title"],
            abstract=work["abstract"],
            authors=authors,
            publication_year=work["publication_year"],
            url=work.get("open_access", {}).get("oa_url"),
            retrieved_chunks=[chunk for chunk in chunks if chunk.openalex_id == openalex_id],
        )
        publications.append(publication)

    return publications
