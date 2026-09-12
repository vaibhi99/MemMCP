"""Environment-driven configuration for MemMCP.

Every setting has a safe default so the server runs fully offline with no
configuration at all. Override via environment variables (prefixed ``MEMMCP_``)
or a ``.env`` file. See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["hash", "sentence-transformers", "openai"]
VectorBackend = Literal["numpy", "chroma"]
ExtractionProvider = Literal["openai", "anthropic"]
PIIPolicy = Literal["block", "redact", "flag", "off"]


class Settings(BaseSettings):
    """Central configuration object, populated from env / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="MEMMCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Storage ----
    data_dir: Path = Field(default=Path("./.memmcp_data"))

    # ---- Embeddings ----
    embedding_provider: EmbeddingProvider = "hash"
    embedding_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"

    # ---- Vector store ----
    vector_backend: VectorBackend = "numpy"

    # ---- Fact extraction ----
    extraction_provider: ExtractionProvider = "openai"
    extraction_model: str = "gpt-4o-mini"
    extraction_temperature: float = 0.0
    extraction_max_tokens: int = 2048

    # ---- Retrieval ranking ----
    weight_relevance: float = 0.60
    weight_recency: float = 0.25
    weight_importance: float = 0.15
    recency_halflife_days: float = 30.0

    # ---- Distillation ----
    dedup_threshold: float = 0.92
    conflict_threshold: float = 0.60

    # ---- LLM Conflict Judge ----
    judge_enabled: bool = True
    judge_model: str = "gpt-4o-mini"

    # ---- Privacy ----
    pii_policy: PIIPolicy = "redact"

    # ---- API keys (read without the MEMMCP_ prefix) ----
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, value: Path) -> Path:
        return value.expanduser()

    @property
    def audit_log_path(self) -> Path:
        return self.data_dir / "audit.log.jsonl"

    @property
    def store_path(self) -> Path:
        return self.data_dir / "vectors"

    def ranking_weights(self) -> tuple[float, float, float]:
        """Return normalised (relevance, recency, importance) weights."""
        total = self.weight_relevance + self.weight_recency + self.weight_importance
        if total <= 0:
            return (1.0, 0.0, 0.0)
        return (
            self.weight_relevance / total,
            self.weight_recency / total,
            self.weight_importance / total,
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
