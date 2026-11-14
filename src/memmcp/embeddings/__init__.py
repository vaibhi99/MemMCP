"""Gemini embedding provider powered by LangChain.

Set ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) and optionally ``MEMMCP_EMBEDDING_MODEL``
(defaults to ``models/text-embedding-004``).
"""

from .base import Embedder, cosine_similarity
from .factory import build_embedder
from .providers import GeminiEmbedder

__all__ = ["Embedder", "GeminiEmbedder", "cosine_similarity", "build_embedder"]
