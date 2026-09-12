"""Distillation: deduplicate and resolve conflicts before storing a fact.

Given a new candidate memory and the memories already in scope, the distiller
decides one of:

* **create** — genuinely new information; store it.
* **merge** — a near-duplicate already exists; reinforce that one instead of
  storing a redundant copy (keeps the set clean and avoids drift).
* **supersede** — a competing statement about the same slot exists (same
  conflict ``key``); mark the old belief stale and store the new one.

When a candidate falls into the similarity grey zone
(``conflict_threshold <= sim < dedup_threshold``), the distiller delegates
the contradiction-vs-compatibility decision to an optional
:class:`~memmcp.extraction.conflict_judge.ConflictJudge` (LLM-backed).
If no judge is configured, the system falls through to "create" (safe default).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..models import Memory

if TYPE_CHECKING:
    from .conflict_judge import ConflictJudge

logger = logging.getLogger(__name__)


@dataclass
class DistillDecision:
    action: str  # "create" | "merge" | "supersede"
    targets: list[Memory] = field(default_factory=list)
    similarity: float = 0.0
    reason: str = ""


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


class Distiller:
    """Pure decision logic; the manager applies the resulting mutations."""

    def __init__(
        self,
        dedup_threshold: float,
        conflict_threshold: float,
        judge: ConflictJudge | None = None,
    ) -> None:
        self.dedup_threshold = dedup_threshold
        self.conflict_threshold = conflict_threshold
        self.judge = judge

    def decide(
        self,
        new_memory: Memory,
        similar: list[tuple[Memory, float]],
        same_key: list[Memory],
    ) -> DistillDecision:
        """Classify ``new_memory`` against existing scope memories.

        Args:
            new_memory: The candidate (embedding populated).
            similar: ``(memory, cosine)`` neighbours from the vector store.
            same_key: Active memories sharing ``new_memory.key`` (may be empty).
        """
        new_norm = _norm(new_memory.content)

        # 1) Exact-text duplicate anywhere on the same key -> merge.
        for mem in same_key:
            if _norm(mem.content) == new_norm:
                return DistillDecision("merge", [mem], 1.0, "identical content")

        # 2) High semantic similarity -> duplicate -> merge into the closest.
        best_mem: Memory | None = None
        best_sim = 0.0
        for mem, sim in similar:
            if _norm(mem.content) == new_norm:
                return DistillDecision("merge", [mem], 1.0, "identical content")
            if sim >= self.dedup_threshold and sim > best_sim:
                best_mem, best_sim = mem, sim
                
        if best_mem is not None:
            return DistillDecision("merge", [best_mem], best_sim, "near-duplicate")

        # 3) Same conflict key, different content -> supersede stale beliefs.
        if new_memory.key:
            conflicts = [
                m
                for m in same_key
                if m.id != new_memory.id and _norm(m.content) != new_norm
            ]
            if conflicts:
                return DistillDecision(
                    "supersede",
                    conflicts,
                    self._max_sim(new_memory, conflicts, similar),
                    f"new value for slot '{new_memory.key}'",
                )

        # 4) Ambiguity zone: conflict_threshold <= sim < dedup_threshold.
        #    Delegate to the LLM judge if available.
        zone_candidates = [
            (mem, sim)
            for mem, sim in similar
            if self.conflict_threshold <= sim < self.dedup_threshold
        ]
        if zone_candidates and self.judge is not None:
            # Only evaluate the single closest candidate — keeps LLM calls
            # to at most 1 per distill operation.
            closest_mem, closest_sim = max(zone_candidates, key=lambda t: t[1])
            result = self.judge.judge(
                existing_fact=closest_mem.content,
                new_fact=new_memory.content,
            )
            logger.info(
                "LLM judge: %s (sim=%.3f) — %s",
                result.verdict, closest_sim, result.reason,
            )
            if result.verdict == "CONTRADICTION":
                return DistillDecision(
                    "supersede",
                    [closest_mem],
                    closest_sim,
                    f"llm_judge: {result.reason}",
                )
            # COMPATIBLE → fall through to create.

        return DistillDecision("create", [], best_sim, "novel information")

    @staticmethod
    def _max_sim(
        new_memory: Memory,
        conflicts: list[Memory],
        similar: list[tuple[Memory, float]],
    ) -> float:
        sim_map = {m.id: s for m, s in similar}
        return max((sim_map.get(m.id, 0.0) for m in conflicts), default=0.0)
