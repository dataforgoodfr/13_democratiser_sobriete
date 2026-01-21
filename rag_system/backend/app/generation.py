import asyncio
from typing import Type

from pydantic import BaseModel, ValidationError

from .config import settings
from .dependencies import create_openai_client, escape_newlines, get_logger
from .models import ChatMessage


generation_client = create_openai_client()
logger = get_logger(__name__)

async def generate_response(
    messages: list[ChatMessage],
    max_tokens: int = 512,
    temperature: float = 0.15,
    top_p: float = 0.1,
    response_format: Type[BaseModel] | None = None,
    timeout: int = 60,
):
    try:
        kwargs = {
            "model": settings.generation_model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "timeout": timeout,
        }

        if response_format:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                },
            }

        response = await generation_client.chat.completions.create(**kwargs)

        if response_format:
            try:
                return response_format.model_validate_json(response.choices[0].message.content)
            except ValidationError:
                logger.warning("Validation error in query rewrite, retrying...")
                kwargs["temperature"] = 0  # Retry with deterministic output
                response = await generation_client.chat.completions.create(**kwargs)
                return response_format.model_validate_json(response.choices[0].message.content)

        return response.choices[0].message.content
    except asyncio.TimeoutError:
        return "\n\n[Generation timed out]\n\n"
    except ValidationError as e:
        logger.error(response.choices[0].message.content)
        raise e
        


async def stream_response(
    messages: list[ChatMessage],
    max_tokens: int = 512,
    temperature: float = 0.15,
    top_p: float = 0.1,
    timeout: int = 60,
):
    try:
        stream = await generation_client.chat.completions.create(
            model=settings.generation_model_name,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except asyncio.TimeoutError:
        yield "\n\n[Generation timed out]\n\n"


async def simulate_stream(text: str, delay: float = 0.1):
    words = text.split(" ")
    for i, word in enumerate(words):
        # Add space before word (except for first word)
        chunk = word if i == 0 else " " + word
        yield "data: " + escape_newlines(chunk) + "\n\n"
        await asyncio.sleep(delay)  # Simulate LLM generation delay

    yield "data: [DONE]\n\n"
