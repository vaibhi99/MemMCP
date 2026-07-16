"""Factory that builds the configured embedder, with graceful fallback."""

from __future__ import annotations

import logging

from ..config import Settings
from .base import Embedder
from .hash_embedder import HashEmbedder

logger = logging.getLogger(__name__)


def build_embedder(settings: Settings) -> Embedder:
    """Instantiate the embedder named in ``settings``.

    If an optional provider is selected but its dependency is missing, we log a
    warning and fall back to the offline :class:`HashEmbedder` so the server
    still starts.
    """
    provider = settings.embedding_provider

    if provider == "hash":
        return HashEmbedder(dim=settings.embedding_dim)

    try:
        if provider == "sentence-transformers":
            from .providers import SentenceTransformerEmbedder

            return SentenceTransformerEmbedder(settings.embedding_model)
        if provider == "openai":
            from .providers import OpenAIEmbedder

            return OpenAIEmbedder(settings.embedding_model, settings.openai_api_key)
    except ImportError as exc:
        logger.warning("Embedding provider %r unavailable (%s); using hash.", provider, exc)
        return HashEmbedder(dim=settings.embedding_dim)

    logger.warning("Unknown embedding provider %r; using hash.", provider)
    return HashEmbedder(dim=settings.embedding_dim)
