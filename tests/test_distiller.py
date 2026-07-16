"""Tests for the distiller decision logic (create / merge / supersede)."""

from memmcp.extraction import Distiller
from memmcp.models import Memory


def _distiller() -> Distiller:
    return Distiller(dedup_threshold=0.92, conflict_threshold=0.60)


def test_novel_fact_is_created():
    d = _distiller()
    new = Memory(content="User uses Postgres.", key="user:database")
    decision = d.decide(new, similar=[], same_key=[])
    assert decision.action == "create"


def test_identical_content_merges():
    d = _distiller()
    existing = Memory(content="User uses Postgres.", key="user:database")
    new = Memory(content="user uses postgres.", key="user:database")
    decision = d.decide(new, similar=[], same_key=[existing])
    assert decision.action == "merge"
    assert decision.targets == [existing]


def test_near_duplicate_merges_by_similarity():
    d = _distiller()
    existing = Memory(content="User uses Postgres for storage.")
    new = Memory(content="User relies on Postgres for storage.")
    decision = d.decide(new, similar=[(existing, 0.95)], same_key=[])
    assert decision.action == "merge"


def test_same_key_different_value_supersedes():
    d = _distiller()
    old = Memory(content="User uses Postgres.", key="user:database")
    new = Memory(content="User uses MySQL.", key="user:database")
    decision = d.decide(new, similar=[(old, 0.4)], same_key=[old])
    assert decision.action == "supersede"
    assert old in decision.targets
