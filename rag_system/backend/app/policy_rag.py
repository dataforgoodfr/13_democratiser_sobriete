"""
Optional second-stage RAG: identify policies cited in retrieved context,
then retrieve their impacts from a dedicated Qdrant policies collection.
Activated by POLICY_RAG_ENABLED=true in the environment.
"""

import asyncio
import logging

from .config import settings
from .generation import generate_response
from .models import ChatMessage, PolicyIdentificationResponse, PolicyImpact
from .retrieval import embed_query, qdrant_client

logger = logging.getLogger(__name__)

_IDENTIFY_POLICIES_PROMPT = """You are a policy analyst assistant.
Given the user query and context below (retrieved from research documents), identify all concrete policies that are explicitly named, described, or discussed that may be relevant to answer the user query.
Return only policies that are meaningfully present in the context — do not invent policies not mentioned.
For each policy, provide a short identifying name or description (1-2 sentences max)."""


async def identify_policies_in_context(
    query: str,
    context_str: str,
) -> list[str]:
    """
    Use the LLM to extract a list of policy names/descriptions from the retrieved context.
    Returns an empty list if no policies are found or on error.
    """
    prompt_messages = [
        ChatMessage(role="system", content=_IDENTIFY_POLICIES_PROMPT),
        ChatMessage(
            role="user",
            content=(
                f"User question: {query}\n\n"
                f"Retrieved context:\n{context_str}\n\n"
                "List all policies explicitly cited or described in this context."
            ),
        ),
    ]
    try:
        response = await generate_response(
            prompt_messages,
            response_format=PolicyIdentificationResponse,
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
        )
        logger.info(f"Identified {len(response.policies)} policies: {response.policies}")
        return response.policies
    except Exception as e:
        logger.error(f"Policy identification failed: {e}")
        return []


async def _retrieve_for_policy(policy_name: str) -> list[PolicyImpact]:
    """Embed a single policy name and query the policies Qdrant collection."""
    embedding = await embed_query(policy_name, dim=settings.policy_embedding_dim)
    loop = asyncio.get_event_loop()
    hits = await loop.run_in_executor(
        None,
        lambda: qdrant_client.query_points(
            collection_name=settings.policy_qdrant_collection_name,
            query=embedding,
            limit=settings.k_policy_search,
        ),
    )
    results = []
    for point in hits.points:
        payload = point.payload or {}
        cluster = payload.get("cluster")
        sufficiency_class = payload.get("sufficiency_classification")
        reasoning = payload.get("sufficiency_classification_reasoning")
        if cluster and sufficiency_class:
            results.append(
                PolicyImpact(
                    cluster=cluster,
                    sufficiency_class=sufficiency_class,
                    sufficiency_classification_reasoning=reasoning or "",
                )
            )
    return results


async def retrieve_policy_impacts(policy_names: list[str]) -> list[PolicyImpact]:
    """
    For each identified policy name, embed and query the policies collection.
    Deduplicates results by cluster name (first occurrence wins).
    """
    if not policy_names:
        return []

    # Retrieve for all policies concurrently
    all_results_nested = await asyncio.gather(
        *[_retrieve_for_policy(name) for name in policy_names],
        return_exceptions=True,
    )

    seen_clusters: set[str] = set()
    deduped: list[PolicyImpact] = []
    for result in all_results_nested:
        if isinstance(result, Exception):
            logger.error(f"Policy impact retrieval error: {result}")
            continue
        for impact in result:
            if impact.cluster not in seen_clusters:
                seen_clusters.add(impact.cluster)
                deduped.append(impact)

    logger.info(f"Retrieved {len(deduped)} unique policy impacts")
    return deduped


def build_policy_system_message(impacts: list[PolicyImpact]) -> str:
    """Format policy impacts as a system message to inject into the LLM context."""
    lines = [
        "The following policy information was retrieved from a specialised database. "
        "Use it to enrich your answer where relevant.\n"
    ]
    for impact in impacts:
        lines.append(f"Policy: {impact.cluster}")
        lines.append(f"Sufficiency assessment: {impact.sufficiency_classification_reasoning}")
        lines.append("")
    return "\n".join(lines)
