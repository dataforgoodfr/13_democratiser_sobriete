from typing import Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    messages: list[ChatMessage]


class ChatChunk(BaseModel):
    """A chunk of the streaming response."""

    content: str
    done: bool = False


class QueryRewriteResponse(BaseModel):
    """Response from the query rewriting LLM."""

    rewritten_query_or_response: str = Field(
        description="The rewritten query, or a response to the user if no retrieval is needed."
    )
    should_retrieve: bool = Field(description="Whether document retrieval should be performed.")


class DocumentChunk(BaseModel):
    openalex_id: str
    chunk_idx: int
    text: str
    retrieved_rank: int


class Publication(BaseModel):
    openalex_id: str
    doi: str | None = None
    title: str
    abstract: str
    authors: list[str] | None = None
    publication_year: int | None = None
    url: str | None = None
    retrieved_chunks: list[DocumentChunk]
