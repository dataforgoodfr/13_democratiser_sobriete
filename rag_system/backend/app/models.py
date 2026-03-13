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
    username: str | None = None
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


class PolicyEvidenceChunk(DocumentChunk):
    sentiment: Literal["positive", "neutral", "negative"]
    impact_category: str
    impact_dimension: str
    policy_cluster_id: int | str
    policy_label: str


class Publication(BaseModel):
    openalex_id: str
    doi: str | None = None
    title: str
    abstract: str | None = None
    authors: list[str] | None = None
    publication_year: int | None = None
    url: str | None = None
    retrieved_chunks: list[DocumentChunk | PolicyEvidenceChunk]


class PolicyIdentificationResponse(BaseModel):
    """Structured output for identifying policies cited in retrieved context."""

    policies: list[str] = Field(
        description="List of policy names or descriptions explicitly cited or described in the provided context."
    )


class PolicyImpact(BaseModel):
    """A policy with its impact information retrieved from the policies Qdrant collection."""

    cluster: str
    sufficiency_class: Literal['S', 'PS', 'NS']
    sufficiency_classification_reasoning: str


class PolicyRerankResponse(BaseModel):
    """Structured response returned by the policy reranker."""

    reasoning: str
    relevance_score: int = Field(ge=1, le=9)
    matched_impact_categories: list[str] = Field(default_factory=list)
    matched_impact_dimensions: list[str] = Field(default_factory=list)


class PolicySearchCandidate(BaseModel):
    """A raw policy candidate retrieved from the policies Qdrant collection."""

    cluster_id: int | str
    text: str
    count: int = 0
    impacts: dict[str, Any] = Field(default_factory=dict)
    retrieved_rank: int
    retrieved_score: float | None = None
    impact_categories: list[str] = Field(default_factory=list)
    impact_dimensions: list[str] = Field(default_factory=list)
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0


class PolicyReference(BaseModel):
    """A sampled literature reference extracted from policy impacts."""

    raw_ref: str
    openalex_id: str
    chunk_idx: int
    sentiment: Literal["positive", "neutral", "negative"]
    impact_category: str
    impact_dimension: str
    policy_cluster_id: int | str
    policy_label: str


class PolicySearchResult(BaseModel):
    """A policy retained after LLM reranking for final presentation and context building."""

    cluster_id: int | str
    policy_text: str
    count: int = 0
    retrieved_rank: int
    retrieved_score: float | None = None
    rerank_score: int
    rerank_reasoning: str
    matched_impact_categories: list[str] = Field(default_factory=list)
    matched_impact_dimensions: list[str] = Field(default_factory=list)
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    sampled_positive_refs: int = 0
    sampled_neutral_refs: int = 0
    sampled_negative_refs: int = 0
    impacts: dict[str, Any] = Field(default_factory=dict, exclude=True)


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
