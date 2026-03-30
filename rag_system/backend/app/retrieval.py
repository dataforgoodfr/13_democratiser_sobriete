"""Compatibility facade for retrieval helpers.

Prefer importing from `retrieval_standard`, `retrieval_policy`, or `retrieval_shared`
when working on a specific pipeline.
"""

from . import retrieval_policy as _policy
from . import retrieval_shared as _shared
from . import retrieval_standard as _standard

qdrant_client = _shared.qdrant_client
executor = _shared.executor


async def embed_query(*args, **kwargs):
    return await _standard.embed_query(*args, **kwargs)


async def retrieve_chunks(*args, **kwargs):
    return await _standard.retrieve_chunks(*args, **kwargs)


def get_publications_from_chunks(*args, **kwargs):
    return _shared.get_publications_from_chunks(*args, **kwargs)


async def embed_query_openai(*args, **kwargs):
    return await _policy.embed_policy_query(*args, **kwargs)


def summarize_policy_impacts(*args, **kwargs):
    return _policy.summarize_policy_impacts(*args, **kwargs)


async def retrieve_policy_candidates(*args, **kwargs):
    return await _policy.retrieve_policy_candidates(*args, **kwargs)


async def fetch_library_chunks_by_references(*args, **kwargs):
    return await _policy.fetch_library_chunks_by_references(*args, **kwargs)
