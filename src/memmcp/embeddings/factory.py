"""Factory that builds the Gemini embedder via LangChain."""

from __future__ import annotations

from ..config import Settings
from .base import Embedder
from .providers import GeminiEmbedder


def build_embedder(settings: Settings) -> Embedder:
    """Instantiate the Gemini embedder from ``settings``."""
    return GeminiEmbedder(
        model_name=settings.embedding_model,
        api_key=settings.google_api_key,
        dim=settings.embedding_dim,
    )
