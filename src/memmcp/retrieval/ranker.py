"""The ranker: combine signals, sort, and pack into a token budget."""

from __future__ import annotations

import math
import time

from ..models import Memory, ScoredMemory


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — good enough for budgeting."""
    return max(1, math.ceil(len(text) / 4))


def _recency_score(memory: Memory, halflife_days: float, now: float) -> float:
    """Exponential decay in ``(0, 1]`` based on time since last access.

    A memory used ``halflife_days`` ago scores 0.5; one used just now scores 1.
    """
    if halflife_days <= 0:
        return 1.0
    age_days = max(0.0, (now - memory.last_accessed_at) / 86_400.0)
    return math.pow(0.5, age_days / halflife_days)


class Ranker:
    """Blends relevance, recency and importance into a single score."""

    def __init__(
        self,
        weights: tuple[float, float, float],
        recency_halflife_days: float,
    ) -> None:
        self.w_relevance, self.w_recency, self.w_importance = weights
        self.recency_halflife_days = recency_halflife_days

    def score(
        self,
        candidates: list[tuple[Memory, float]],
        now: float | None = None,
    ) -> list[ScoredMemory]:
        """Score ``(memory, relevance)`` pairs and return them sorted desc."""
        now = now if now is not None else time.time()
        scored: list[ScoredMemory] = []
        for memory, relevance in candidates:
            # Clamp cosine into [0, 1]; negative similarity == irrelevant.
            relevance = max(0.0, min(1.0, relevance))
            recency = _recency_score(memory, self.recency_halflife_days, now)
            importance = max(0.0, min(1.0, memory.importance))
            final = (
                self.w_relevance * relevance
                + self.w_recency * recency
                + self.w_importance * importance
            )
            scored.append(
                ScoredMemory(
                    memory=memory,
                    score=final,
                    relevance=relevance,
                    recency=recency,
                    importance=importance,
                )
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    @staticmethod
    def pack(
        scored: list[ScoredMemory],
        token_budget: int | None,
        top_k: int | None = None,
    ) -> list[ScoredMemory]:
        """Greedily select top items subject to a token budget and/or top_k.

        Items are already sorted by score, so greedy selection keeps the most
        valuable memories. This is the "surface the 3 relevant facts, not 50
        pages" behaviour that keeps prompts small and on-point.
        """
        selected: list[ScoredMemory] = []
        used = 0
        for item in scored:
            if top_k is not None and len(selected) >= top_k:
                break
            if token_budget is not None:
                cost = estimate_tokens(item.memory.content)
                if used + cost > token_budget and selected:
                    # Budget exhausted; stop (but always allow at least one).
                    break
                used += cost
            selected.append(item)
        return selected
