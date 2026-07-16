"""Tests for the retrieval ranker: signal blending and token packing."""

import time

from memmcp.models import Memory
from memmcp.retrieval import Ranker, estimate_tokens


def _mem(content: str, importance: float, age_days: float) -> Memory:
    m = Memory(content=content, importance=importance)
    m.last_accessed_at = time.time() - age_days * 86_400.0
    return m


def test_relevance_dominates_with_default_weights():
    ranker = Ranker(weights=(0.6, 0.25, 0.15), recency_halflife_days=30)
    a = _mem("a", importance=0.1, age_days=0)
    b = _mem("b", importance=0.9, age_days=100)
    scored = ranker.score([(a, 0.9), (b, 0.2)])
    assert scored[0].memory is a  # high relevance beats high importance


def test_recency_breaks_ties():
    ranker = Ranker(weights=(0.5, 0.5, 0.0), recency_halflife_days=10)
    fresh = _mem("fresh", importance=0.5, age_days=0)
    stale = _mem("stale", importance=0.5, age_days=40)
    scored = ranker.score([(fresh, 0.5), (stale, 0.5)])
    assert scored[0].memory is fresh
    assert scored[0].recency > scored[1].recency


def test_pack_respects_top_k():
    ranker = Ranker(weights=(1.0, 0.0, 0.0), recency_halflife_days=30)
    mems = [(_mem(f"m{i}", 0.5, 0), 0.9 - i * 0.1) for i in range(5)]
    scored = ranker.score(mems)
    packed = ranker.pack(scored, token_budget=None, top_k=2)
    assert len(packed) == 2


def test_pack_respects_token_budget():
    ranker = Ranker(weights=(1.0, 0.0, 0.0), recency_halflife_days=30)
    long_text = "word " * 100  # ~125 tokens
    mems = [(_mem(long_text, 0.5, 0), 0.9), (_mem(long_text, 0.5, 0), 0.8)]
    scored = ranker.score(mems)
    packed = ranker.pack(scored, token_budget=50, top_k=10)
    # Budget only fits one long memory (but at least one is always returned).
    assert len(packed) == 1


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("a" * 40) >= 10
