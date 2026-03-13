import asyncio
import json
import random

from flashrank import Ranker, RerankRequest
from pydantic import ValidationError

from .config import settings
from .dependencies import create_openai_client
from .models import DocumentChunk, PolicyEvidenceChunk, PolicyRerankResponse, PolicySearchCandidate, PolicySearchResult
from .prompts import POLICY_RERANK_PROMPT, SUFFICIENCY_RATING_PROMPT

ranker = Ranker(model_name=settings.flashrank_model, max_length=settings.max_length_reranker)
reranking_client = create_openai_client()


async def rerank_evidence(
    query: str,
    chunks: list[PolicyEvidenceChunk],
    top_k: int = settings.k_rerank,
    max_input: int = 50,
) -> list[PolicyEvidenceChunk]:
    """Rerank evidence chunks against the user query using flashrank.

    If there are more than max_input chunks, only the first max_input are reranked
    (they arrive roughly ordered by sampling priority already).
    """
    if not chunks or ranker is None:
        return chunks[:top_k]

    pool = list(chunks)
    random.shuffle(pool)
    rerank_pool = pool[:max_input]

    # Use a composite ID since multiple chunks can share the same openalex_id
    chunk_map: dict[str, PolicyEvidenceChunk] = {}
    passages = []
    for chunk in rerank_pool:
        uid = f"{chunk.openalex_id}_{chunk.chunk_idx}"
        chunk_map[uid] = chunk
        passages.append({"id": uid, "text": chunk.text})

    reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    return [chunk_map[r["id"]] for r in reranked[:top_k] if r["id"] in chunk_map]


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


def _format_policy_impacts_for_prompt(policy: PolicySearchCandidate) -> str:
    lines = []
    for category, dimension_map in policy.impacts.items():
        if not isinstance(dimension_map, dict):
            continue
        lines.append(f"Category: {category}")
        for dimension, summaries in dimension_map.items():
            if not isinstance(summaries, list):
                summaries = [summaries]
            total_positive = 0
            total_neutral = 0
            total_negative = 0
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                total_positive += int(summary.get("positive", 0) or 0)
                total_neutral += int(summary.get("neutral", 0) or 0)
                total_negative += int(summary.get("negative", 0) or 0)
            lines.append(
                f"- {dimension}: positive={total_positive}, neutral={total_neutral}, negative={total_negative}"
            )
    return "\n".join(lines)


def build_policy_rerank_prompt(query: str, policy: PolicySearchCandidate) -> str:
    impacts = _format_policy_impacts_for_prompt(policy)
    return f"""
{POLICY_RERANK_PROMPT}

User query: {query}

Policy candidate: {policy.text}

Impacts summary:
{impacts or 'No impact summary available.'}
""".strip()


async def llm_rerank_policy(
    query: str,
    policy: PolicySearchCandidate,
    model_name: str = settings.llm_rerank_model,
) -> PolicyRerankResponse:
    response = await reranking_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": build_policy_rerank_prompt(query, policy)}],
        max_tokens=256,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": PolicyRerankResponse.__name__,
                "schema": PolicyRerankResponse.model_json_schema(),
            },
        },
    )

    content = response.choices[0].message.content
    try:
        return PolicyRerankResponse.model_validate_json(content)
    except ValidationError:
        parsed = json.loads(content)
        return PolicyRerankResponse.model_validate(parsed)


async def llm_rerank_policies(
    query: str,
    policies: list[PolicySearchCandidate],
    min_rating: int = settings.llm_filter_min_rating,
    max_results: int = settings.policy_max_retained,
    model_name: str = settings.llm_rerank_model,
) -> list[PolicySearchResult]:
    """Rerank policy candidates using policy descriptions plus structured impact summaries."""
    if not policies:
        return []

    ratings = await asyncio.gather(
        *[llm_rerank_policy(query, policy, model_name=model_name) for policy in policies],
        return_exceptions=True,
    )

    retained: list[PolicySearchResult] = []
    for policy, rating in zip(policies, ratings, strict=True):
        if isinstance(rating, Exception):
            continue
        if rating.relevance_score < min_rating:
            continue
        retained.append(
            PolicySearchResult(
                cluster_id=policy.cluster_id,
                policy_text=policy.text,
                count=policy.count,
                retrieved_rank=policy.retrieved_rank,
                retrieved_score=policy.retrieved_score,
                rerank_score=rating.relevance_score,
                rerank_reasoning=rating.reasoning,
                matched_impact_categories=rating.matched_impact_categories,
                matched_impact_dimensions=rating.matched_impact_dimensions,
                positive_count=policy.positive_count,
                neutral_count=policy.neutral_count,
                negative_count=policy.negative_count,
                impacts=policy.impacts,
            )
        )

    retained.sort(
        key=lambda policy: (
            policy.rerank_score,
            policy.negative_count,
            policy.positive_count + policy.neutral_count,
        ),
        reverse=True,
    )
    return retained[:max_results]
