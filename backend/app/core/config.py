import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Secure Document Intelligence"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"

    # Base Paths
    # Current file is backend/app/core/config.py -> root is 3 levels up from backend
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent.parent
    PROJECT_ROOT: Path = BACKEND_DIR.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    STORAGE_DIR: Path = DATA_DIR / "storage"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    SQLITE_DB_PATH: Path = STORAGE_DIR / "app.db"

    # Ingestion Constraints
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS: set[str] = {"pdf", "docx", "txt"}

    # Tokenizer-Aware Chunking Defaults
    DEFAULT_CHUNK_SIZE_TOKENS: int = 512
    DEFAULT_CHUNK_OVERLAP_TOKENS: int = 64
    TOKENIZER_ENCODING: str = "cl100k_base"

    # Retrieval & Vector Store Configuration
    CHROMA_PERSIST_DIR: Path = STORAGE_DIR / "chroma"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "ms-marco-TinyBERT-L-2-v2"
    DEFAULT_TOP_K: int = 5
    DEFAULT_RRF_K: int = 60
    BM25_CANDIDATES: int = 20
    SEMANTIC_CANDIDATES: int = 20

    # Local Ollama LLM Configuration
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_TIMEOUT_SECONDS: float = 180.0
    OLLAMA_TEMPERATURE: float = 0.0

    # Authentication & Security
    JWT_SECRET_KEY: str = "insecure_dev_secret_key_change_in_production_1234567890"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Notion Integration Configuration
    NOTION_API_KEY: Optional[str] = None
    NOTION_API_BASE_URL: str = "https://api.notion.com/v1"
    NOTION_API_VERSION: str = "2022-06-28"
    NOTION_TIMEOUT_SECONDS: float = 30.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure runtime directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

