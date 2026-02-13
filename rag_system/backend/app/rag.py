from datetime import datetime, UTC
import json
import time

from .config import settings
from .database import ChatSession, ChatTurn, save_chat_turn
from .dependencies import escape_newlines, get_logger
from .generation import (
    generate_response,
    simulate_stream,
    stream_response,
)
from .models import ChatMessage, DocumentChunk, Publication, QueryRewriteResponse
from .prompts import BASE_SYSTEM_PROMPT, RAG_PROMPT, QUERY_REWRITE_PROMPT
from .retrieval import get_publications_from_chunks, retrieve_chunks

logger = get_logger(__name__)


async def simple_rag_pipeline(
    messages: list[ChatMessage],
    chat_session: ChatSession | None = None,
    fetch_pubs: bool = settings.fetch_pubs,
):
    """
    A simple RAG pipeline: rewrite query, embed, retrieve, rerank, generate response.
    Logs results and response times to database.
    """

    # Initialize timing and logging variables
    pipeline_start = time.time()
    user_query = [m for m in messages if m.role == "user"][-1].content  # last user message
    chat_turn = ChatTurn(
        timestamp=datetime.now(UTC),
        user_query=user_query,
        session_id=chat_session.id if chat_session else None,
        turn_number=0,  # Will be set by save_chat_turn
    )

    try:
        # Query rewriting stage
        query_rewrite_start = time.time()
        query_rewrite_response = await rewrite_query(messages)
        chat_turn.query_rewrite_time_ms = (time.time() - query_rewrite_start) * 1000
        chat_turn.should_retrieve = query_rewrite_response.should_retrieve

        if query_rewrite_response.should_retrieve is False:
            # No retrieval needed - send empty documents then stream response
            chat_turn.rewritten_query = None
            response_text = query_rewrite_response.rewritten_query_or_response
            yield "event: documents\n\ndata: " + json.dumps({"documents": []}) + "\n\n"
            async for chunk in simulate_stream(response_text):
                yield chunk

            # Finalize log for non-retrieval case
            chat_turn.response = response_text
            chat_turn.response_length = len(response_text)
            chat_turn.processing_time_ms = (time.time() - pipeline_start) * 1000
            chat_turn.status = "success"
            if settings.log_usage and chat_session:
                save_chat_turn(chat_session.id, chat_turn)
            return

        rewritten_query = query_rewrite_response.rewritten_query_or_response
        chat_turn.rewritten_query = rewritten_query
        logger.info(f"Rewritten query: {rewritten_query}")

        # Retrieval stage
        retrieval_start = time.time()
        retrieved_chunks = await retrieve_chunks(rewritten_query)
        chat_turn.retrieval_time_ms = (time.time() - retrieval_start) * 1000
        chat_turn.retrieved_chunks_count = len(retrieved_chunks)
        chat_turn.retrieved_chunks = json.dumps([c.model_dump() for c in retrieved_chunks])
        logger.info(f"Retrieved {len(retrieved_chunks)} documents")

        if not retrieved_chunks:
            # No documents retrieved - stream a response indicating that
            response_text = """
            Our system didn't find documents relevant enough to both your query and the topic of sufficiency.
            We work on improving the retrieval engine. In the meantime feel free to rephrase your question or ask about a different aspect of sufficiency.
            """
            yield "event: documents\n\ndata: " + json.dumps({"documents": []}) + "\n\n"
            async for chunk in simulate_stream(response_text):
                yield chunk

            # Finalize log for no retrieval case
            chat_turn.response = response_text
            chat_turn.response_length = len(response_text)
            chat_turn.processing_time_ms = (time.time() - pipeline_start) * 1000
            chat_turn.status = "nodocuments"
            if settings.log_usage and chat_session:
                save_chat_turn(chat_session.id, chat_turn)
            return
        
        if fetch_pubs:
            retrieved_pubs = get_publications_from_chunks(retrieved_chunks)
            logger.info(f"Fetched {len(retrieved_pubs)} publications.")
            chat_turn.retrieved_publications_count = len(retrieved_pubs)
            documents = retrieved_pubs
            context = build_context_from_pubs(retrieved_pubs)
        else:
            documents = retrieved_chunks
            context = build_context_from_chunks(retrieved_chunks)

        chat_turn.context_built = context
        chat_turn.context_length = len(context)

        # Send documents first as special events
        docs_json = [doc.model_dump() for doc in documents]
        yield "event: documents\n\ndata: " + json.dumps({"documents": docs_json}) + "\n\n"

        system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + RAG_PROMPT
        context_instruction = (
            f"Use the following documents to answer the user's question:\n\n"
            f"{context}\n\n"
            "If the answer is not in the documents, say so. Remember to cite sources using [Doc N] format."
        )
        augmented_messages = [
            ChatMessage(role="system", content=system_prompt),
            *messages,
            ChatMessage(role="system", content=context_instruction),
        ]

        # Generation stage
        generation_start = time.time()
        stream = stream_response(
            augmented_messages,
            max_tokens=settings.answer_max_tokens,
            temperature=settings.answer_temperature,
            top_p=settings.answer_top_p,
        )
        response_text = ""
        async for chunk in stream:
            response_text += chunk
            yield "data: " + escape_newlines(chunk) + "\n\n"
        chat_turn.generation_time_ms = (time.time() - generation_start) * 1000

        yield "data: [DONE]\n\n"

        # Finalize successful log
        chat_turn.response = response_text if response_text else None
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


async def rewrite_query(messages: list[ChatMessage]) -> QueryRewriteResponse:
    """Rewrite the user query to be more effective for retrieval."""
    system_prompt = BASE_SYSTEM_PROMPT + " " + QUERY_REWRITE_PROMPT
    system_message = ChatMessage(role="system", content=system_prompt)
    messages = [system_message] + messages

    response = await generate_response(
        messages,
        response_format=QueryRewriteResponse,
        temperature=settings.query_rewrite_temperature,
        top_p=settings.query_rewrite_top_p,
        max_tokens=settings.query_rewrite_max_tokens,
    )
    return response


def build_context_from_pubs(publications: list[Publication]) -> str:
    """Build the context string from retrieved publications."""
    context_parts = []
    for i, pub in enumerate(publications):
        # add abstract + text from each chunk
        context_parts.append(f"Document {i+1}:\n\n")
        context_parts.append(f"Title: {pub.title}\n")
        context_parts.append(f"ABSTRACT\n{pub.abstract}\n")
        for i, chunk in enumerate(pub.retrieved_chunks):
            context_parts.append(f"CHUNK {i+1}\n{chunk.text}\n")
        context_parts.append("\n")

    context = "\n".join(context_parts)
    return f"<context>\n{context}\n</context>"


def build_context_from_chunks(documents: list[DocumentChunk]) -> str:
    """Build the context string from retrieved documents."""
    context_parts = []
    for i, doc in enumerate(documents):
        context_parts.append(f"Document {i+1}:\n{doc.text}\n")
    context = "\n".join(context_parts)
    return f"<context>\n{context}\n</context>"


async def generate_dummy_response():
    """Generate a dummy streaming response, simulating an LLM."""
    dummy_text = (
        "Sufficiency is a set of policy measures and daily practices "
        "which avoid the demand for energy, materials, land, water, and other natural resources,"
        "while delivering wellbeing for all within planetary boundaries."
    )

    async for chunk in simulate_stream(dummy_text):
        yield chunk
