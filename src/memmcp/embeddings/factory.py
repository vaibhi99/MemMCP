"""Factory that builds the OpenAI embedder."""

from __future__ import annotations

from ..config import Settings
from .base import Embedder
from .providers import OpenAIEmbedder


def build_embedder(settings: Settings) -> Embedder:
    """Instantiate the OpenAI embedder from ``settings``."""
    return OpenAIEmbedder(settings.embedding_model, settings.openai_api_key)
