from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # retrieval
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "library-test"
    qdrant_timeout: int = 10
    embedding_dim: int = 128
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    max_length_reranker: int = 1024
    k_vector_search: int = 20
    

    query_rewrite_temperature: float = 0.05
    query_rewrite_top_p: float = 0.1
    query_rewrite_max_tokens: int = 128
    query_rewrite_timeout: int = 15

    k_rerank: int = 5
    rerank_method : Literal["flashrank", "llm"] = "flashrank"
    llm_rerank_model: str = "mistral-small-3.2-24b-instruct-2506"

    # generation
    generation_api_url: str
    generation_model_name: str
    scw_access_key: str
    scw_secret_key: str

    answer_temperature: float = 0.01
    answer_top_p: float = 0.1
    answer_max_tokens: int = 512
    answer_timeout: int = 30

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
