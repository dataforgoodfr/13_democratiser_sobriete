from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str
    qdrant_api_key: str
    generation_api_url: str
    generation_model_name: str
    scw_access_key: str
    scw_secret_key: str

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
