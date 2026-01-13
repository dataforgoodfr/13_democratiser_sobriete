from openai import AsyncOpenAI
from config import settings


def create_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.generation_api_url,
        api_key=settings.scw_secret_key,
    )


def escape_newlines(text: str) -> str:
    return text.replace("\n", "<|newline|>")
