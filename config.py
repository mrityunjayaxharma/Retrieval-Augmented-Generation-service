"""Configuration module for RAG service"""
import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    app_name: str = "RAG Service"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    max_pages: int = 50
    max_depth: int = 3
    crawl_delay_ms: int = 1000
    request_timeout: int = 30
    user_agent: str = "RAG-Service/1.0"

    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_model: str = "all-MiniLM-L6-v2"
    max_chunk_tokens: int = 512

    vector_db_path: str = "./data/vector_index"
    index_type: str = "IndexFlatL2"

    top_k: int = 5
    similarity_threshold: float = 0.7
    max_answer_length: int = 512

    generation_model: str = "microsoft/DialoGPT-medium"
    max_tokens: int = 150
    temperature: float = 0.7

    data_dir: str = "./data"
    logs_dir: str = "./logs"
    cache_dir: str = "./cache"

    allowed_domains: Optional[list] = None
    rate_limit: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
os.makedirs(settings.data_dir, exist_ok=True)
os.makedirs(settings.logs_dir, exist_ok=True)
os.makedirs(settings.cache_dir, exist_ok=True)
