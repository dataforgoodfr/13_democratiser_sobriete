import asyncio
import logging
from typing import Any

from qdrant_client import models as qdrant_models

from .config import settings
from .dependencies import create_openai_client
from .models import PolicyEvidenceChunk, PolicyReference, PolicySearchCandidate, PolicySearchResult
from .retrieval_shared import executor, qdrant_client

logger = logging.getLogger(__name__)


openai_embedding_client = create_openai_client()


async def embed_policy_query(
    query: str,
    model_name: str = settings.policy_embedding_model,
    dim: int = settings.policy_embedding_dim,
) -> list[float]:
    """Embed a policy-search query using the OpenAI-compatible embedding endpoint."""
    response = await openai_embedding_client.embeddings.create(
        model=model_name,
        input=query,
    )
    embedding = response.data[0].embedding
    return embedding[:dim] if dim else embedding


def summarize_policy_impacts(impacts: dict[str, Any]) -> tuple[list[str], list[str], int, int, int]:
    """Collapse nested policy impacts into categories, dimensions, and sentiment totals."""
    categories: list[str] = []
    dimensions: list[str] = []
    positive_count = 0
    neutral_count = 0
    negative_count = 0

    for category, dimension_map in impacts.items():
        categories.append(category)
        if not isinstance(dimension_map, dict):
            continue
        for dimension, summaries in dimension_map.items():
            dimensions.append(f"{category}: {dimension}")
            if not isinstance(summaries, list):
                summaries = [summaries]
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                positive_count += int(summary.get("positive", 0) or 0)
                neutral_count += int(summary.get("neutral", 0) or 0)
                negative_count += int(summary.get("negative", 0) or 0)

    return sorted(set(categories)), sorted(set(dimensions)), positive_count, neutral_count, negative_count


async def retrieve_policy_candidates(
    query: str,
    top_k: int = settings.policy_candidate_count,
    collection_name: str = settings.policy_collection_name,
) -> list[PolicySearchCandidate]:
    """Retrieve policy candidates from the policy collection for the policy-first pipeline."""
    query_embedding = await embed_policy_query(query)
    loop = asyncio.get_event_loop()
    hits = await loop.run_in_executor(
        executor,
        lambda: qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=["cluster_id", "text", "count", "impacts"],
            timeout=settings.qdrant_timeout,
        ),
    )

    candidates: list[PolicySearchCandidate] = []
    for idx, point in enumerate(hits.points):
        payload = point.payload or {}
        impacts = payload.get("impacts") or {}
        categories, dimensions, positive_count, neutral_count, negative_count = summarize_policy_impacts(impacts)
        candidates.append(
            PolicySearchCandidate(
                cluster_id=payload.get("cluster_id", f"policy-{idx + 1}"),
                text=payload.get("text", ""),
                count=int(payload.get("count", 0) or 0),
                impacts=impacts,
                retrieved_rank=idx + 1,
                retrieved_score=getattr(point, "score", None),
                impact_categories=categories,
                impact_dimensions=dimensions,
                positive_count=positive_count,
                neutral_count=neutral_count,
                negative_count=negative_count,
            )
        )
    return candidates


async def _fetch_library_chunk(
    reference: PolicyReference,
    collection_name: str = settings.library_collection_name,
) -> PolicyEvidenceChunk | None:
    loop = asyncio.get_event_loop()
    records, _ = await loop.run_in_executor(
        executor,
        lambda: qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="openalex_id",
                        match=qdrant_models.MatchValue(value=reference.openalex_id),
                    ),
                    qdrant_models.FieldCondition(
                        key="chunk_idx",
                        match=qdrant_models.MatchValue(value=reference.chunk_idx),
                    ),
                ]
            ),
            limit=1,
            with_payload=["openalex_id", "chunk_idx", "text"],
            with_vectors=False,
            timeout=settings.qdrant_timeout,
        ),
    )
    if not records:
        return None

    payload = records[0].payload or {}
    return PolicyEvidenceChunk(
        openalex_id=payload.get("openalex_id", reference.openalex_id),
        chunk_idx=int(payload.get("chunk_idx", reference.chunk_idx)),
        text=payload.get("text", ""),
        retrieved_rank=0,
        sentiment=reference.sentiment,
        impact_category=reference.impact_category,
        impact_dimension=reference.impact_dimension,
        policy_cluster_id=reference.policy_cluster_id,
        policy_label=reference.policy_label,
    )


async def retrieve_evidence_for_policies(
    query: str,
    retained_policies: list[PolicySearchResult],
    top_k: int = settings.k_rerank,
    search_cap: int = 100,
    collection_name: str = settings.library_collection_name,
) -> list[PolicyEvidenceChunk]:
    """Vector search in the library collection, scoped to papers cited by retained policies.

    Embeds the user query with the standard model, then queries the library filtered to
    the openalex_ids found in the policies' relevant impact dimensions.  Each returned
    chunk is cross-referenced against the reference maps to attach sentiment / dimension
    metadata: exact (openalex_id, chunk_idx) match first, paper-level fallback second.
    """
    from .policy_evidence import build_reference_maps
    from .retrieval_standard import embed_query

    exact_map, paper_map = build_reference_maps(retained_policies)
    if not paper_map:
        logger.warning("No policy references found; cannot retrieve evidence.")
        return []

    query_embedding = await embed_query(query)
    openalex_ids = list(paper_map.keys())
    logger.info(
        "Searching library for evidence in %d papers (top_k=%d, cap=%d)",
        len(openalex_ids), top_k, search_cap,
    )

    loop = asyncio.get_event_loop()
    hits = await loop.run_in_executor(
        executor,
        lambda: qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="openalex_id",
                        match=qdrant_models.MatchAny(any=openalex_ids),
                    )
                ]
            ),
            limit=search_cap,
            with_payload=["openalex_id", "chunk_idx", "text"],
            with_vectors=False,
            timeout=settings.qdrant_timeout,
        ),
    )

    evidence_chunks: list[PolicyEvidenceChunk] = []
    for rank, point in enumerate(hits.points, start=1):
        payload = point.payload or {}
        openalex_id = payload.get("openalex_id", "")
        chunk_idx = int(payload.get("chunk_idx", 0))
        text = payload.get("text", "")

        ref = exact_map.get((openalex_id, chunk_idx)) or paper_map.get(openalex_id)
        if ref is None:
            continue

        evidence_chunks.append(PolicyEvidenceChunk(
            openalex_id=openalex_id,
            chunk_idx=chunk_idx,
            text=text,
            retrieved_rank=rank,
            sentiment=ref.sentiment,
            impact_category=ref.impact_category,
            impact_dimension=ref.impact_dimension,
            policy_cluster_id=ref.policy_cluster_id,
            policy_label=ref.policy_label,
        ))
        if len(evidence_chunks) >= top_k:
            break

    logger.info("Retrieved %d evidence chunks from %d candidate papers", len(evidence_chunks), len(openalex_ids))
    return evidence_chunks


async def fetch_library_chunks_by_references(
    references: list[PolicyReference],
    collection_name: str = settings.library_collection_name,
) -> list[PolicyEvidenceChunk]:
    """Resolve sampled policy references into literature chunks from the library collection."""
    if not references:
        return []

    results = await asyncio.gather(
        *[_fetch_library_chunk(reference, collection_name=collection_name) for reference in references],
        return_exceptions=True,
    )

    evidence_chunks: list[PolicyEvidenceChunk] = []
    for result in results:
        if isinstance(result, PolicyEvidenceChunk):
            evidence_chunks.append(result)
        elif isinstance(result, Exception):
            logger.error("Failed to fetch library chunk: %s", result, exc_info=result)
    for idx, chunk in enumerate(evidence_chunks):
        chunk.retrieved_rank = idx + 1
    return evidence_chunks