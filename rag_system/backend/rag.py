import json

from config import settings
from dependencies import escape_paragraphs
from generation import (
    generate_response,
    simulate_stream,
    stream_response,
)
from models import ChatMessage, Document, QueryRewriteResponse
from prompts import BASE_SYSTEM_PROMPT, RAG_PROMPT, QUERY_REWRITE_PROMPT
from retrieval import retrieve_documents


async def simple_rag_pipeline(messages: list[ChatMessage]):
    """A simple RAG pipeline: rewrite query, embed, retrieve, rerank, generate response."""
    query_rewrite_response = await rewrite_query(messages)
    if query_rewrite_response.should_retrieve is False:
        # No retrieval needed - send empty documents then stream response
        yield "event: documents\ndata: " + json.dumps({"documents": []}) + "\n\n"
        async for chunk in simulate_stream(query_rewrite_response.rewritten_query_or_response):
            yield chunk
        return
    rewritten_query = query_rewrite_response.rewritten_query_or_response
    print(rewritten_query)
    retrieved_docs = await retrieve_documents(rewritten_query)
    print(f"Retrieved {len(retrieved_docs)} documents.")
    # Send documents first as a special event
    docs_json = [doc.model_dump() for doc in retrieved_docs]
    yield "event: documents\ndata: " + json.dumps({"documents": docs_json}) + "\n\n"
    
    context = build_context(retrieved_docs)
    system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + RAG_PROMPT
    context_instruction = (
        f"Use the following documents to answer the user's question:\n"
        f"{context}\n\n"
        "If the answer is not in the documents, say so. Remember to cite sources using [Doc N] format."
    )
    messages = [
        ChatMessage(role="system", content=system_prompt),
        *messages,
        ChatMessage(role="system", content=context_instruction),
    ]
    stream = stream_response(
        messages,
        max_tokens=settings.answer_max_tokens,
        temperature=settings.answer_temperature,
        top_p=settings.answer_top_p,
    )
    async for chunk in stream:
        yield "data: " + escape_paragraphs(chunk) + "\n\n"
    yield "data: [DONE]\n\n"


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


def build_context(documents: list[Document]) -> str:
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
