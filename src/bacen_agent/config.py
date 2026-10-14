from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenSearch
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_use_ssl: bool = False
    index_name: str = "bacen-documents"

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    hf_access_token: str
    
@lru_cache
def get_settings() -> Settings:
    return Settings()
