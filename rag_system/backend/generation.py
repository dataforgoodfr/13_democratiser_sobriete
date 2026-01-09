import os
import asyncio
from openai import AsyncOpenAI
from typing import Optional, Type
from pydantic import BaseModel

from config import settings

generation_client = AsyncOpenAI(
    base_url=settings.generation_api_url,
    api_key=settings.scw_secret_key,
)
model_name = settings.generation_model_name

# TODO: adjust temperature and top-p


async def generate_response(
    user_query: str,
    system_prompt: str,
    response_format: Optional[Type[BaseModel]] = None,
):
    kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 0.1,
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
        return response_format.model_validate_json(response.choices[0].message.content)

    return response.choices[0].message.content


async def stream_response(user_query: str, system_prompt: str):
    stream = await generation_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
        stream=True,
        max_tokens=512,
        temperature=0.1,
        top_p=0.1,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def simulate_stream(text: str):
    words = text.split(" ")
    for i, word in enumerate(words):
        # Add space before word (except for first word)
        chunk = word if i == 0 else " " + word
        yield "data: " + chunk + "\n\n"
        await asyncio.sleep(0.1)  # Simulate LLM generation delay

    yield "data: [DONE]\n\n"
