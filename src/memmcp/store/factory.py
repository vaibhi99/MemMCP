"""Factory that builds the configured vector store, with graceful fallback."""

from __future__ import annotations

import logging

from ..config import Settings
from .base import VectorStore
from .numpy_store import NumpyVectorStore

logger = logging.getLogger(__name__)


def build_store(settings: Settings) -> VectorStore:
    """Instantiate the vector store named in ``settings``.

    Falls back to the offline :class:`NumpyVectorStore` if Chroma is selected
    but not installed.
    """
    backend = settings.vector_backend

    if backend == "chroma":
        try:
            from .chroma_store import ChromaVectorStore

            return ChromaVectorStore(settings.store_path)
        except ImportError as exc:
            logger.warning("Chroma backend unavailable (%s); using numpy.", exc)

    return NumpyVectorStore(settings.store_path / "memories.json")
