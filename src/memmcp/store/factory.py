"""Factory that builds the ChromaDB vector store."""

from __future__ import annotations

from ..config import Settings
from .base import VectorStore
from .chroma_store import ChromaVectorStore


def build_store(settings: Settings) -> VectorStore:
    """Instantiate the ChromaDB vector store from ``settings``."""
    return ChromaVectorStore(settings.store_path)
