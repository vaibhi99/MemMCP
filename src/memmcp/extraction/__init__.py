"""Fact extraction & distillation — turning noisy chat into clean memory.

A raw transcript is full of dead ends, corrections and abandoned ideas. Storing
it verbatim re-teaches the model your *mistakes*, not your *conclusions*. This
package turns conversation into durable, self-contained facts and keeps the
memory set clean over time:

* :class:`FactExtractor` — conversation -> candidate facts (rule-based offline,
  LLM-based when configured).
* :class:`Distiller` — dedup near-identical facts and detect conflicts so
  updated beliefs ("moved to Go") supersede stale ones ("uses Postgres").
"""

from .distiller import Distiller, DistillDecision
from .fact_extractor import ExtractedFact, build_extractor

__all__ = ["Distiller", "DistillDecision", "ExtractedFact", "build_extractor"]
