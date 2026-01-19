from typing import Optional
from datetime import datetime
from sqlmodel import Field, Relationship, SQLModel, create_engine, Session

from dependencies import get_logger
from config import settings

logger = get_logger(__name__)

# Create database engine
engine = create_engine(
    settings.postgres_uri,
    echo=False,  # Set to True for SQL debugging
    connect_args={"check_same_thread": False} if "sqlite" in settings.postgres_uri else {},
)


def create_db_and_tables():
    """Create all database tables."""
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


class ChatSession(SQLModel, table=True):
    """Model for a chat session (conversation)."""

    __tablename__ = "chat_sessions"

    id: str = Field(primary_key=True)  # UUID provided by client
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Session metadata
    title: Optional[str] = None  # Can be auto-generated from first message

    # Client info (captured on session creation)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Session stats (updated as turns are added)
    turn_count: int = 0
    total_processing_time_ms: float = 0.0

    # Relationships
    turns: list["ChatTurn"] = Relationship(back_populates="session")


class ChatTurn(SQLModel, table=True):
    """Model for a single turn (user query + assistant response) in a chat session."""

    __tablename__ = "chat_turns"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="chat_sessions.id", index=True)
    turn_number: int = Field(index=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)

    # User query
    user_query: str

    # RAG Pipeline - Query Rewriting
    rewritten_query: Optional[str] = None
    should_retrieve: Optional[bool] = None

    # RAG Pipeline - Retrieval
    retrieved_chunks_count: int = 0
    retrieved_chunks: Optional[str] = None  # JSON
    retrieved_publications_count: int = 0

    # RAG Pipeline - Context Building
    context_built: Optional[str] = None
    context_length: int = 0

    # Response information
    response: Optional[str] = None
    response_length: int = 0

    # Status and error tracking
    status: str = "success"  # success, error, timeout
    error_message: Optional[str] = None
    pipeline_stage_failed: Optional[str] = None

    # Performance metrics
    processing_time_ms: float = 0.0
    query_rewrite_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0

    # Relationship
    session: Optional[ChatSession] = Relationship(back_populates="turns")


def get_or_create_session(
    session_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> ChatSession:
    """Get an existing session or create a new one."""
    with Session(engine) as db:
        if session_id:
            chat_session = db.get(ChatSession, session_id)
            if chat_session:
                return chat_session

        chat_session = ChatSession(
            id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
        logger.info(f"Created new chat session: {chat_session.id}")
        return chat_session


def save_chat_turn(session_id: str, turn: ChatTurn) -> ChatTurn:
    """Save a chat turn and update session stats."""
    try:
        with Session(engine) as db:
            # Get session and determine turn number
            chat_session = db.get(ChatSession, session_id)
            if not chat_session:
                raise ValueError(f"Session {session_id} not found")

            turn.session_id = session_id
            turn.turn_number = chat_session.turn_count + 1

            # Update session stats
            chat_session.turn_count += 1
            chat_session.total_processing_time_ms += turn.processing_time_ms
            chat_session.updated_at = datetime.utcnow()

            # Auto-generate title from first user query
            if chat_session.turn_count == 1 and not chat_session.title:
                chat_session.title = turn.user_query[:100]

            db.add(turn)
            db.add(chat_session)
            db.commit()
            db.refresh(turn)

            logger.info(
                f"Saved turn {turn.turn_number} for session {session_id} - "
                f"Status: {turn.status}, Time: {turn.processing_time_ms:.2f}ms"
            )
            return turn
    except Exception as e:
        logger.error(f"Error saving chat turn: {e}")


def get_session_history(session_id: str) -> list[ChatTurn]:
    """Get all turns for a session, ordered by turn number."""
    with Session(engine) as db:
        chat_session = db.get(ChatSession, session_id)
        if not chat_session:
            return []
        # Sort turns by turn_number
        return sorted(chat_session.turns, key=lambda t: t.turn_number)
