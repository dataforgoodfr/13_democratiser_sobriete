import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models import ChatRequest

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def generate_dummy_response():
    """Generate a dummy streaming response, simulating an LLM."""
    dummy_text = (
        "Sufficiency is a set of policy measures and daily practices "
        "which avoid the demand for energy, materials, land, water, and other natural resources,"
        "while delivering wellbeing for all within planetary boundaries."
    )

    words = dummy_text.split(" ")
    for i, word in enumerate(words):
        # Add space before word (except for first word)
        chunk = word if i == 0 else " " + word
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(0.1)  # Simulate LLM generation delay

    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Stream a chat response."""
    return StreamingResponse(
        generate_dummy_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
