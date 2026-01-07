from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    messages: list[ChatMessage]


class ChatChunk(BaseModel):
    """A chunk of the streaming response."""
    content: str
    done: bool = False
