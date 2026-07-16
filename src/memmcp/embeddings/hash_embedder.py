"""Deterministic, dependency-free embedder (the offline default).

This is a hashing embedder: it tokenises text and hashes tokens (plus
character trigrams for sub-word robustness) into a fixed-dimension bag-of-words
vector, then L2-normalises. It is **not** a semantic model — but it is fast,
fully offline, and deterministic, which makes it ideal for local development
and reproducible tests. Two texts that share vocabulary land near each other,
which is enough to exercise the entire retrieval/ranking/distillation pipeline.

Swap in ``sentence-transformers`` or ``openai`` for real semantic quality.
"""

from __future__ import annotations

import hashlib
import math
import re

from .base import Embedder, normalize

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to",
    "of", "in", "on", "for", "with", "as", "by", "at", "it", "this", "that",
    "i", "you", "we", "they", "he", "she", "my", "our", "your",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _char_ngrams(token: str, n: int = 3) -> list[str]:
    if len(token) <= n:
        return [token]
    padded = f"#{token}#"
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    """Hash a feature to a bucket index and a stable +/-1 sign."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    index = value % dim
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return index, sign


class HashEmbedder(Embedder):
    """Feature-hashing embedder with TF weighting and signed buckets."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _tokens(text)
        if not tokens:
            return vec

        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
            for gram in _char_ngrams(tok):
                counts[f"#{gram}"] = counts.get(f"#{gram}", 0) + 1

        for feature, count in counts.items():
            # Sub-linear TF weighting dampens very frequent tokens.
            weight = 1.0 + math.log(count)
            idx, sign = _bucket(feature, self.dim)
            vec[idx] += sign * weight

        return normalize(vec)
