"""OpenAI-powered embedding provider.

Set ``OPENAI_API_KEY`` and optionally ``MEMMCP_EMBEDDING_MODEL``
(defaults to ``text-embedding-3-small``).
"""

from .base import Embedder, cosine_similarity
from .factory import build_embedder

__all__ = ["Embedder", "cosine_similarity", "build_embedder"]
