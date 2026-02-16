from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    chat_id: str
    persona: str | None = None
    messages: list[ChatMessage]


class ChatChunk(BaseModel):
    """A chunk of the streaming response."""

    content: str
    done: bool = False


class QueryRewriteResponse(BaseModel):
    """Response from the query rewriting LLM."""

    should_retrieve: bool = Field(description="Whether document retrieval should be performed.")
    rewritten_query_or_response: str = Field(
        description="The rewritten query, or a response to the user if no retrieval is needed."
    )


class DocumentChunk(BaseModel):
    openalex_id: str
    chunk_idx: int
    text: str
    retrieved_rank: int


class Publication(BaseModel):
    openalex_id: str
    doi: str | None = None
    title: str
    abstract: str | None = None
    authors: list[str] | None = None
    publication_year: int | None = None
    url: str | None = None
    retrieved_chunks: list[DocumentChunk]


class FeedbackRequest(BaseModel):
    """Request body for feedback endpoint."""

    chat_id: str | None = None
    content: str


class StructuredOutputRequest(BaseModel):
    """Request body for generic non-streaming structured generation."""

    messages: list[ChatMessage]
    output_schema: dict[str, Any]
    schema_name: str = "ClientStructuredOutput"
    fetch_pubs: bool = True
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    max_tokens: int = Field(default=512, ge=1)
    timeout: int = Field(default=60, ge=1)


class StructuredOutputResponse(BaseModel):
    """Response body for generic structured generation endpoint."""

    output: dict[str, Any]
    documents: list[dict[str, Any]] = Field(default_factory=list)
