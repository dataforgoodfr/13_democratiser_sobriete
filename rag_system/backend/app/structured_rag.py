from .generation import generate_response
from .models import ChatMessage, StructuredOutputRequest, StructuredOutputResponse
from .prompts import BASE_SYSTEM_PROMPT, GENERIC_STRUCTURED_OUTPUT_PROMPT
from .rag import build_context_from_chunks, build_context_from_pubs
from .retrieval import get_publications_from_chunks, retrieve_chunks


async def structured_output_pipeline(request: StructuredOutputRequest) -> StructuredOutputResponse:
    """Generic non-streaming pipeline with optional retrieval and client-provided schema."""
    documents: list[dict] = []
    messages = [
        ChatMessage(role="system", content=BASE_SYSTEM_PROMPT),
        ChatMessage(role="system", content=GENERIC_STRUCTURED_OUTPUT_PROMPT),
        *request.messages,
    ]

    user_messages = [m.content for m in request.messages if m.role == "user"]
    retrieval_query = user_messages[-1] if user_messages else ""

    if not retrieval_query:
        return StructuredOutputResponse(output={}, documents=[])

    retrieved_chunks = await retrieve_chunks(retrieval_query)

    if request.fetch_pubs:
        retrieved_pubs = get_publications_from_chunks(retrieved_chunks)
        context = build_context_from_pubs(retrieved_pubs)
        documents = [pub.model_dump() for pub in retrieved_pubs]
    else:
        context = build_context_from_chunks(retrieved_chunks)
        documents = [chunk.model_dump() for chunk in retrieved_chunks]

    if documents:
        context_instruction = (
            "Use the following retrieved context to answer. "
            "If evidence is insufficient for a field in the schema, use the most cautious value allowed by the schema.\n\n"
            f"{context}"
        )
        messages.append(ChatMessage(role="system", content=context_instruction))

    output = await generate_response(
        messages=messages,
        response_format=request.output_schema,
        schema_name=request.schema_name,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        timeout=request.timeout,
    )

    return StructuredOutputResponse(output=output, documents=documents)
