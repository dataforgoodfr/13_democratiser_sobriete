import asyncio
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from qdrant_client import QdrantClient

from config import settings
from models import Document

qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)
embedding_model = SentenceTransformer(settings.embedding_model, device="cpu")

ranker = Ranker(max_length=settings.max_length_reranker)
reranking_client = AsyncOpenAI(
    base_url=settings.generation_api_url,
    api_key=settings.scw_secret_key,
)

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


async def retrieve_documents(
    query: str,
    top_k: int = settings.k_vector_search,
    reranker: Ranker = ranker,
    client: QdrantClient = qdrant_client,
    collection_name: str = settings.qdrant_collection_name,
) -> list[Document]:
    """Retrieve top-k documents from Qdrant based on the query embedding."""
    query_embedding = await embed_query(query)
    hits = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
    )
    documents = [
        Document(
            openalex_id=d.id,
            title=d.payload.get("title", ""),
            text=d.payload.get("abstract", ""),
        )
        for d in hits.points
    ]
    if settings.rerank_method == "flashrank":
        results = await flashrank_rerank(reranker, query, documents)
    elif settings.rerank_method == "llm":
        results = await llm_rerank(query, documents)
    return results


async def flashrank_rerank(
    ranker: Ranker,
    query: str,
    documents: list[Document],
    top_k: int = settings.k_rerank,
) -> list[Document]:
    """Rerank documents based on the query using the provided Ranker."""
    passages = [{"id": d.openalex_id, "text": d.text} for d in documents]
    rerank_request = RerankRequest(
        query=query,
        passages=passages,
    )
    reranked_results = ranker.rerank(rerank_request)

    # Map reranked results back to original documents
    document_map = {d.openalex_id: d for d in documents}
    reranked_documents = [document_map[result["id"]] for result in reranked_results[:top_k]]
    return reranked_documents


async def llm_rerank(
    query: str,
    documents: list[Document],
    top_k: int = settings.k_rerank,
    model_name: str = settings.llm_rerank_model,
) -> list[Document]:
    """Rerank documents using an LLM-based approach."""
    scores = await asyncio.gather(
        *[score_document(query, doc, model_name) for doc in documents]
    )

    # Sort by score descending and return top_k documents
    scored_docs = [(documents[i], scores[i]) for i in range(len(documents))]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]


async def score_document(query: str, doc: Document, model_name: str) -> tuple[int, float]:
    prompt = build_llm_rerank_prompt(query, doc)
    response = await reranking_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
    )
    logprobs_dict = response.choices[0].logprobs.content[0].top_logprobs
    yes_score = -100
    for token_info in logprobs_dict:
        if token_info.token.strip().lower() == "yes":
            yes_score = token_info.logprob
            break

    return yes_score


def build_llm_rerank_prompt(query: str, document: Document) -> str:
    prompt = f"""Query: {query}
Document: {document.text}
Is this document relevant to answering the query? Answer only 'Yes' or 'No'."""
    return prompt
