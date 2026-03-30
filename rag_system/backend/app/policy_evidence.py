"""
Reference parsing and balanced sampling for the policy-first pipeline.

Converts raw Qdrant policy payloads (nested impacts dict) into typed PolicyReference
objects, deduplicates them, and samples a balanced set across sentiments.
"""

from collections import defaultdict
import logging
import random

from .config import settings
from .models import PolicyReference, PolicySearchResult

logger = logging.getLogger(__name__)

_SENTIMENT_PRIORITY = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def _parse_reference(
    raw_ref: str,
    *,
    sentiment: str,
    impact_category: str,
    impact_dimension: str,
    policy_cluster_id: int | str,
    policy_label: str,
) -> PolicyReference | None:
    try:
        openalex_id, chunk_idx = raw_ref.rsplit("_", maxsplit=1)
        return PolicyReference(
            raw_ref=raw_ref,
            openalex_id=openalex_id,
            chunk_idx=int(chunk_idx),
            sentiment=sentiment,
            impact_category=impact_category,
            impact_dimension=impact_dimension,
            policy_cluster_id=policy_cluster_id,
            policy_label=policy_label,
        )
    except ValueError:
        logger.warning("Could not parse policy reference %s", raw_ref)
        return None


def _iter_policy_references(policy: PolicySearchResult):
    """Yield references only from impact dimensions the reranker flagged as relevant."""
    relevant_dims = set(policy.matched_impact_dimensions) if policy.matched_impact_dimensions else None

    for category, dimension_map in policy.impacts.items():
        if not isinstance(dimension_map, dict):
            continue
        for dimension, summaries in dimension_map.items():
            # Skip dimensions the reranker didn't select (when any were selected)
            if relevant_dims is not None and f"{category}: {dimension}" not in relevant_dims:
                continue
            if not isinstance(summaries, list):
                summaries = [summaries]
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                for sentiment in ("positive", "neutral", "negative"):
                    refs = summary.get(f"{sentiment}_refs") or []
                    for raw_ref in refs:
                        parsed = _parse_reference(
                            raw_ref,
                            sentiment=sentiment,
                            impact_category=category,
                            impact_dimension=dimension,
                            policy_cluster_id=policy.cluster_id,
                            policy_label=policy.policy_text,
                        )
                        if parsed:
                            yield parsed


def _deduplicate_references(references: list[PolicyReference]) -> list[PolicyReference]:
    """Keep the highest-priority sentiment per unique (openalex_id, chunk_idx) pair."""
    best_by_chunk: dict[tuple[str, int], PolicyReference] = {}
    for reference in references:
        key = (reference.openalex_id, reference.chunk_idx)
        current = best_by_chunk.get(key)
        if current is None or _SENTIMENT_PRIORITY[reference.sentiment] < _SENTIMENT_PRIORITY[current.sentiment]:
            best_by_chunk[key] = reference
    return list(best_by_chunk.values())


def build_reference_maps(
    retained_policies: list[PolicySearchResult],
) -> tuple[dict[tuple[str, int], PolicyReference], dict[str, PolicyReference]]:
    """Build two lookup maps from the relevant references across retained policies.

    Returns:
        exact_map: (openalex_id, chunk_idx) -> PolicyReference
        paper_map: openalex_id -> highest-priority PolicyReference for the paper (fallback)
    """
    all_refs = _deduplicate_references(
        [ref for policy in retained_policies for ref in _iter_policy_references(policy)]
    )
    exact_map: dict[tuple[str, int], PolicyReference] = {
        (ref.openalex_id, ref.chunk_idx): ref for ref in all_refs
    }
    paper_map: dict[str, PolicyReference] = {}
    for ref in all_refs:
        existing = paper_map.get(ref.openalex_id)
        if existing is None or _SENTIMENT_PRIORITY[ref.sentiment] < _SENTIMENT_PRIORITY[existing.sentiment]:
            paper_map[ref.openalex_id] = ref
    return exact_map, paper_map


def sample_policy_references(
    retained_policies: list[PolicySearchResult],
    refs_per_sentiment: int = settings.policy_refs_per_direction,
) -> list[PolicyReference]:
    """Sample a balanced number of positive, neutral, and negative refs across retained policies.

    As a side effect, writes sampled_*_refs counts back onto each policy object
    so the frontend can display them.
    """
    all_references = _deduplicate_references(
        [ref for policy in retained_policies for ref in _iter_policy_references(policy)]
    )
    if not all_references:
        return []

    bucketed: dict[str, list[PolicyReference]] = defaultdict(list)
    for reference in all_references:
        bucketed[reference.sentiment].append(reference)

    sampled: list[PolicyReference] = []
    for sentiment in ("positive", "neutral", "negative"):
        choices = bucketed.get(sentiment, [])
        sampled.extend(choices if len(choices) <= refs_per_sentiment else random.sample(choices, refs_per_sentiment))

    sampled_counts: dict[int | str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for reference in sampled:
        sampled_counts[reference.policy_cluster_id][reference.sentiment] += 1

    for policy in retained_policies:
        policy.sampled_positive_refs = sampled_counts[policy.cluster_id].get("positive", 0)
        policy.sampled_neutral_refs = sampled_counts[policy.cluster_id].get("neutral", 0)
        policy.sampled_negative_refs = sampled_counts[policy.cluster_id].get("negative", 0)

    return sampled
