from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    fetch_pubs: bool = True
    rag_pipeline: Literal["standard", "policy"] = "standard"

    # retrieval
    qdrant_url: str
    qdrant_api_key: str
    library_collection_name: str = "library-test"
    qdrant_timeout: int = 10
    embedding_dim: int = 128
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    max_length_reranker: int = 1024
    k_vector_search: int = 20

    query_rewrite_temperature: float = 0.05
    query_rewrite_top_p: float = 0.1
    query_rewrite_max_tokens: int = 128
    query_rewrite_timeout: int = 15

    k_rerank: int = 5  # number of documents to return after reranking (max if llm_sufficiency, exact otherwise)
    rerank_method: Literal["flashrank", "llm", "llm_sufficiency"] = "flashrank"
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2" # ms-marco-TinyBERT-L-2-v2
    llm_rerank_model: str = "mistral-small-3.2-24b-instruct-2506"
    llm_filter_min_rating: int = 5  # on a scale of 1-9, included

    # generation
    generation_api_url: str
    generation_model_name: str
    scw_access_key: str | None = None
    scw_secret_key: str

    answer_temperature: float = 0.01
    answer_top_p: float = 0.1
    answer_max_tokens: int = 2048
    answer_timeout: int = 30

    # policy RAG (optional second-stage retrieval)
    policy_rag_enabled: bool = False
    policy_collection_name: str = "policies-v1"
    k_policy_search: int = 3  # results per policy name
    policy_embedding_dim: int = 4096
    policy_embedding_model: str = "qwen3-embedding-8b"
    policy_candidate_count: int = 20
    policy_max_retained: int = 10
    policy_refs_per_direction: int = 2

    # database
    postgres_uri: str
    log_usage: bool = False  # log sessions to db, should only be true in production

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
