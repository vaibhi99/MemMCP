"""Embedding provider interface and vector-math helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Turns text into a fixed-dimension, L2-normalised vector."""

    #: Dimensionality of the vectors this embedder produces.
    dim: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single string."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings. Providers may override for efficiency."""
        return [self.embed(t) for t in texts]


def _to_array(vec) -> np.ndarray:
    return np.asarray(vec, dtype=np.float32)


def normalize(vec) -> list[float]:
    """L2-normalise a vector so dot product == cosine similarity."""
    arr = _to_array(vec)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


def cosine_similarity(a, b) -> float:
    """Cosine similarity in ``[-1, 1]`` (typically ``[0, 1]`` for text)."""
    va, vb = _to_array(a), _to_array(b)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
