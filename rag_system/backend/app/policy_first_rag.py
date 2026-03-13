"""
Policy-first RAG pipeline.

Flow: analyze query → retrieve policy candidates → LLM rerank → sample evidence refs
      → fetch library chunks → build context → stream LLM answer.
"""

from datetime import UTC, datetime
import json
import logging
import time
from typing import Any

from .config import settings
from .database import ChatSession, ChatTurn, save_chat_turn
from .dependencies import escape_newlines
from .generation import generate_response, simulate_stream, stream_response
from .models import ChatMessage, PolicySearchResult, Publication, QueryRewriteResponse
from .prompts import BASE_SYSTEM_PROMPT, POLICY_QUERY_REWRITE_PROMPT, RAG_PROMPT
from .reranking import llm_rerank_policies
from .retrieval_policy import retrieve_evidence_for_policies, retrieve_policy_candidates
from .retrieval_shared import get_publications_from_chunks

logger = logging.getLogger(__name__)


async def rewrite_policy_query(messages: list[ChatMessage]) -> QueryRewriteResponse:
    """Decide whether to retrieve and rewrite the query for policy-first search."""
    system_message = ChatMessage(
        role="system",
        content=BASE_SYSTEM_PROMPT + " " + POLICY_QUERY_REWRITE_PROMPT,
    )
    return await generate_response(
        [system_message] + messages,
        response_format=QueryRewriteResponse,
        temperature=settings.query_rewrite_temperature,
        top_p=settings.query_rewrite_top_p,
        max_tokens=settings.query_rewrite_max_tokens,
        timeout=settings.query_rewrite_timeout,
    )


def build_policy_first_context(
    query: str,
    retained_policies: list[PolicySearchResult],
    evidence_documents: list[Publication] | list[Any],
) -> str:
    """Build the final context string for the policy-first generation step."""
    sections = ["<policy_context>", f"Original query: {query}", ""]

    for idx, policy in enumerate(retained_policies, start=1):
        sections.append(f"Policy {idx}: {policy.policy_text} (cluster {policy.cluster_id})")
        sections.append(f"Relevance score: {policy.rerank_score}/9")
        sections.append(f"Why relevant: {policy.rerank_reasoning}")
        sections.append(
            f"Impact counts: positive={policy.positive_count}, neutral={policy.neutral_count}, negative={policy.negative_count}"
        )
        sections.append(
            f"Sampled refs: positive={policy.sampled_positive_refs}, neutral={policy.sampled_neutral_refs}, negative={policy.sampled_negative_refs}"
        )
        if policy.matched_impact_categories:
            sections.append(f"Matched categories: {', '.join(policy.matched_impact_categories)}")
        if policy.matched_impact_dimensions:
            sections.append(f"Matched dimensions: {', '.join(policy.matched_impact_dimensions)}")
        sections.append("")

    sections += ["</policy_context>", "", "<evidence_context>"]

    for idx, document in enumerate(evidence_documents, start=1):
        if isinstance(document, Publication):
            sections.append(f"Document {idx} (OpenAlex ID: {document.openalex_id})")
            sections.append(f"Title: {document.title}")
            if document.abstract:
                sections.append(f"Abstract: {document.abstract}")
            for chunk in document.retrieved_chunks:
                if hasattr(chunk, "sentiment"):
                    sections.append(
                        f"Evidence chunk [{chunk.sentiment}] from {chunk.policy_label} / {chunk.impact_category} / {chunk.impact_dimension}: {chunk.text}"
                    )
                else:
                    sections.append(f"Evidence chunk: {chunk.text}")
            sections.append("")
        else:
            chunk = document
            sections.append(
                f"Document {idx} (OpenAlex ID: {chunk.openalex_id} - chunk {chunk.chunk_idx})"
                f" [{getattr(chunk, 'sentiment', 'unspecified')}]"
                f" from {getattr(chunk, 'policy_label', 'policy')}"
                f" / {getattr(chunk, 'impact_category', 'unknown category')}"
                f" / {getattr(chunk, 'impact_dimension', 'unknown dimension')}: {chunk.text}"
            )

    sections.append("</evidence_context>")
    return "\n".join(sections)


async def policy_first_rag_pipeline(
    messages: list[ChatMessage],
    chat_session: ChatSession | None = None,
    fetch_pubs: bool = settings.fetch_pubs,
):
    """Policy-first pipeline: retrieve policies, sample evidence refs, fetch literature, generate."""
    pipeline_start = time.time()
    user_query = [m for m in messages if m.role == "user"][-1].content
    chat_turn = ChatTurn(
        timestamp=datetime.now(UTC),
        user_query=user_query,
        session_id=chat_session.id if chat_session else None,
        turn_number=0,
    )

    try:
        # --- Query analysis ---
        yield "event: status\n\ndata: " + json.dumps({"step": "analyzing_query"}) + "\n\n"
        rewrite_start = time.time()
        rewrite_response = await rewrite_policy_query(messages)
        chat_turn.query_rewrite_time_ms = (time.time() - rewrite_start) * 1000
        chat_turn.should_retrieve = rewrite_response.should_retrieve

        if rewrite_response.should_retrieve is False:
            response_text = rewrite_response.rewritten_query_or_response
            yield "event: documents\n\ndata: " + json.dumps({"documents": []}) + "\n\n"
            yield "event: policies\n\ndata: " + json.dumps({"policies": []}) + "\n\n"
            async for chunk in simulate_stream(response_text):
                yield chunk
            chat_turn.response = response_text
            chat_turn.response_length = len(response_text)
            chat_turn.processing_time_ms = (time.time() - pipeline_start) * 1000
            chat_turn.status = "success"
            if settings.log_usage and chat_session:
                save_chat_turn(chat_session.id, chat_turn)
            return

        rewritten_query = rewrite_response.rewritten_query_or_response
        chat_turn.rewritten_query = rewritten_query

        # --- Policy retrieval & reranking ---
        yield "event: status\n\ndata: " + json.dumps({"step": "retrieving_policies"}) + "\n\n"
        retrieval_start = time.time()
        policy_candidates = await retrieve_policy_candidates(rewritten_query)
        retained_policies = await llm_rerank_policies(rewritten_query, policy_candidates)
        chat_turn.retrieval_time_ms = (time.time() - retrieval_start) * 1000
        chat_turn.retrieved_chunks_count = len(policy_candidates)
        chat_turn.retrieved_chunks = json.dumps([p.model_dump() for p in retained_policies])

        if not retained_policies:
            response_text = (
                "I could not find policy candidates with impact evidence that are relevant enough to answer that question. "
                "Try naming a sector, policy family, or impact dimension to narrow the search."
            )
            yield "event: documents\n\ndata: " + json.dumps({"documents": []}) + "\n\n"
            yield "event: policies\n\ndata: " + json.dumps({"policies": []}) + "\n\n"
            async for chunk in simulate_stream(response_text):
                yield chunk
            chat_turn.response = response_text
            chat_turn.response_length = len(response_text)
            chat_turn.processing_time_ms = (time.time() - pipeline_start) * 1000
            chat_turn.status = "nodocuments"
            if settings.log_usage and chat_session:
                save_chat_turn(chat_session.id, chat_turn)
            return

        yield "event: policies\n\ndata: " + json.dumps({"policies": [p.model_dump() for p in retained_policies]}) + "\n\n"

        # --- Evidence retrieval ---
        yield "event: status\n\ndata: " + json.dumps({"step": "retrieving_evidence"}) + "\n\n"
        evidence_chunks = await retrieve_evidence_for_policies(rewritten_query, retained_policies)
        logger.info("Evidence: %d chunks retrieved", len(evidence_chunks))
        documents = get_publications_from_chunks(evidence_chunks) if fetch_pubs else evidence_chunks

        chat_turn.retrieved_publications_count = len(documents)
        yield "event: documents\n\ndata: " + json.dumps({"documents": [d.model_dump() for d in documents]}) + "\n\n"

        # --- Generation ---
        context = build_policy_first_context(rewritten_query, retained_policies, documents)
        print(context)
        chat_turn.context_built = context
        chat_turn.context_length = len(context)

        augmented_messages = [
            ChatMessage(role="system", content=BASE_SYSTEM_PROMPT + "\n\n" + RAG_PROMPT),
            *messages,
            ChatMessage(
                role="system",
                content=(
                    "Use the following policy and evidence context to answer the user's question. "
                    "Make the pros and cons explicit when negative evidence exists.\n\n"
                    f"{context}\n\n"
                    "If the answer is not supported by the evidence, say so. Cite retrieved evidence using [Doc N] format."
                ),
            ),
        ]

        generation_start = time.time()
        response_text = ""
        async for chunk in stream_response(
            augmented_messages,
            max_tokens=settings.answer_max_tokens,
            temperature=settings.answer_temperature,
            top_p=settings.answer_top_p,
        ):
            response_text += chunk
            yield "data: " + escape_newlines(chunk) + "\n\n"
        chat_turn.generation_time_ms = (time.time() - generation_start) * 1000
        yield "data: [DONE]\n\n"

        chat_turn.response = response_text
        chat_turn.response_length = len(response_text)
        chat_turn.processing_time_ms = (time.time() - pipeline_start) * 1000
        chat_turn.status = "success"

    except Exception as e:
        chat_turn.status = "error"
        chat_turn.error_message = str(e)
        chat_turn.processing_time_ms = (time.time() - pipeline_start) * 1000
        raise
    finally:
        if settings.log_usage and chat_session:
            save_chat_turn(chat_session.id, chat_turn)
