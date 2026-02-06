import asyncio

from flashrank import Ranker, RerankRequest

from .config import settings
from .dependencies import create_openai_client
from .models import DocumentChunk
from .prompts import SUFFICIENCY_RATING_PROMPT


ranker = Ranker(max_length=settings.max_length_reranker)
reranking_client = create_openai_client()


async def flashrank_rerank(
    query: str,
    documents: list[DocumentChunk],
    ranker: Ranker = ranker,
    top_k: int = settings.k_rerank,
) -> list[DocumentChunk]:
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
    documents: list[DocumentChunk],
    top_k: int = settings.k_rerank,
    model_name: str = settings.llm_rerank_model,
) -> list[DocumentChunk]:
    """Rerank documents using an LLM-based approach."""
    scores = await asyncio.gather(
        *[llm_score_document(query, doc, model_name) for doc in documents]
    )

    # Sort by score descending and return top_k documents
    scored_docs = [(documents[i], scores[i]) for i in range(len(documents))]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]


async def llm_score_document(
    query: str, doc: DocumentChunk, model_name: str
) -> tuple[int, float]:
    prompt = build_llm_score_prompt(query, doc)
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


def build_llm_score_prompt(query: str, document: DocumentChunk) -> str:
    prompt = f"""Query: {query}
Document: {document.text}
Is this document relevant to answering the query? Answer only 'Yes' or 'No'."""
    return prompt


async def llm_rate_sufficiency(
    query: str,
    documents: list[DocumentChunk],
    min_rating: int = settings.llm_filter_min_rating,
    max_results: int = settings.k_rerank,
    model_name: str = settings.llm_rerank_model,
) -> list[DocumentChunk]:
    """
    Rate document relevance to the query and to sufficiency using an LLM.
    Then returns at most max_results documents with relevance score above min_rating.
    """
    ratings = await asyncio.gather(
        *[llm_rate_document(query, doc, model_name) for doc in documents]
    )

    # Sort by score descending and return top_k documents
    scored_docs = [(documents[i], ratings[i]) for i in range(len(documents))]
    scored_docs = [t for t in scored_docs if t[1] >= min_rating]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:max_results]]


async def llm_rate_document(query: str, doc: DocumentChunk, model_name: str) -> tuple[int, int]:
    prompt = build_llm_sufficiency_rating_prompt(query, doc)
    response = await reranking_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0,
    )
    rate = response.choices[0].message.content.strip()

    try:
        rating = int(rate)
    except ValueError:
        rating = 0  # Default to 0 if parsing fails

    return rating


def build_llm_sufficiency_rating_prompt(query: str, document: DocumentChunk) -> str:
    prompt = f"""
{SUFFICIENCY_RATING_PROMPT}

Query: {query}
Document: {document.text}
"""
    return prompt.strip()
