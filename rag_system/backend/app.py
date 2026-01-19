from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import settings
from database import create_db_and_tables, get_or_create_session, save_feedback
from dependencies import get_logger
from models import ChatRequest, FeedbackRequest
from rag import generate_dummy_response, simple_rag_pipeline


# TODO: move ml models init/clean here
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    create_db_and_tables()
    logger.info("Application startup complete")
    yield

logger = get_logger(__name__)
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat")
async def chat(chat_request: ChatRequest, http_request: Request):
    """Stream a chat response."""
    if settings.log_usage:
        chat_session = get_or_create_session(
            session_id=chat_request.chat_id,
            ip_address=http_request.client.host,
            user_agent=http_request.headers.get("user-agent"),
        )
    else:
        chat_session = None
    return StreamingResponse(
        # generate_dummy_response(),
        simple_rag_pipeline(chat_request.messages, chat_session),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/feedback")
async def submit_feedback(feedback_request: FeedbackRequest, http_request: Request):
    """Submit user feedback."""
    try:
        feedback = save_feedback(
            content=feedback_request.content,
            session_id=feedback_request.chat_id,
            ip_address=http_request.client.host,
            user_agent=http_request.headers.get("user-agent"),
        )
        return {"status": "success", "id": feedback.id}
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return {"status": "error", "message": str(e)}
