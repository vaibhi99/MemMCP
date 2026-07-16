"""Pluggable vector stores.

The default :class:`NumpyVectorStore` persists to a JSON file and needs only
numpy — perfect for local use and tests. Install ``chromadb`` to switch to a
production vector database with the same interface.
"""

from .base import VectorStore
from .factory import build_store

__all__ = ["VectorStore", "build_store"]
