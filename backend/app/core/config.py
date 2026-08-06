"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings for the backend."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: SecretStr = Field(..., min_length=1)
    gemini_model: str = "gemini-2.5-flash"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    sqlite_db_path: Path = Path("data/beauty_advisor.db")
    faiss_index_path: Path = Path("data/index/products.faiss")
    default_top_k: int = Field(default=5, ge=1, le=20)
    llm_timeout_seconds: int = Field(default=5, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""
    return Settings()
