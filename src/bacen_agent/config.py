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
    
    #LLM
    llm_model: str = "claude-haiku-5-5"
    llm_effort: str = "medium"
    anthropic_key: str
    llm_input_price: float = 0.10    # US$ por milhão de tokens (Haiku 5.5)
    llm_output_price: float = 0.50
    
    #Cross-Enconder
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_candidates: int = 50
    retrieval_method: str = "rerank_rrf_c50"
    hybrid_bm25_weight: float = 0.25
    
    #Langfuse
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_base_url: str = "http://localhost:3000"
    
    max_rewrites: int = 1
    
@lru_cache
def get_settings() -> Settings:
    return Settings()
