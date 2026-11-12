"""ChromaDB-backed vector store.

Install ``chromadb`` (included in core dependencies) to use.
"""

from .base import VectorStore
from .factory import build_store

__all__ = ["VectorStore", "build_store"]
