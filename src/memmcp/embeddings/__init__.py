"""Pluggable embedding providers.

The default :class:`HashEmbedder` is deterministic and dependency-free, so the
whole server works offline and tests are reproducible. Install extras to swap
in higher-quality embeddings without changing any calling code:

    MEMMCP_EMBEDDING_PROVIDER=sentence-transformers   # local model
    MEMMCP_EMBEDDING_PROVIDER=openai                  # hosted API
"""

from .base import Embedder, cosine_similarity
from .factory import build_embedder

__all__ = ["Embedder", "cosine_similarity", "build_embedder"]
