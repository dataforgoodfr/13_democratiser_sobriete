from generation import (
    generate_response,
    simulate_stream,
    stream_response,
)
from models import QueryRewriteResponse
from prompts import BASE_SYSTEM_PROMPT, RAG_PROMPT, QUERY_REWRITE_PROMPT
from retrieval import retrieve_documents


# TODO: handle accumulating messages
async def simple_rag_pipeline(
    user_query: str,
):
    """A simple RAG pipeline: rewrite query, embed, retrieve, rerank, generate response."""
    query_rewrite_response = await rewrite_query(user_query)
    if query_rewrite_response.should_retrieve is False:
        async for chunk in simulate_stream(query_rewrite_response.rewritten_query_or_response):
            yield chunk
        return
    rewritten_query = query_rewrite_response.rewritten_query_or_response
    retrieved_docs = await retrieve_documents(rewritten_query)
    context = build_context(retrieved_docs)
    system_prompt = BASE_SYSTEM_PROMPT + " " + RAG_PROMPT + f"\n{context}\n"
    async for chunk in stream_response(rewritten_query, system_prompt):
        yield "data: " + chunk + "\n\n"
    yield "data: [DONE]\n\n"


async def rewrite_query(query: str) -> QueryRewriteResponse:
    """Rewrite the user query to be more effective for retrieval."""
    system_prompt = BASE_SYSTEM_PROMPT + " " + QUERY_REWRITE_PROMPT
    response = await generate_response(query, system_prompt, QueryRewriteResponse)
    return response


def build_context(documents: list[dict]) -> str:
    """Build the context string from retrieved documents."""
    context_parts = []
    for i, doc in enumerate(documents):
        content = doc.get("text", "")
        context_parts.append(f"Document {i+1}:\n{content}\n")
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
