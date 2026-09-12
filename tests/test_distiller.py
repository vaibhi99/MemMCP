"""Tests for the distiller decision logic (create / merge / supersede).

Includes tests for the LLM conflict judge integration in the ambiguity zone.
"""

from unittest.mock import MagicMock

from memmcp.extraction import Distiller
from memmcp.extraction.conflict_judge import JudgeResult
from memmcp.models import Memory


def _distiller(judge=None) -> Distiller:
    return Distiller(dedup_threshold=0.92, conflict_threshold=0.60, judge=judge)


# ---- Original tests (no judge) ----

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


# ---- LLM judge integration tests ----

def _mock_judge(verdict: str, reason: str = "test") -> MagicMock:
    """Create a mock ConflictJudge that returns a fixed verdict."""
    judge = MagicMock()
    judge.judge.return_value = JudgeResult(verdict=verdict, reason=reason)
    return judge


def test_ambiguity_zone_calls_judge_contradiction():
    """When similarity is in [0.60, 0.92) and the judge says CONTRADICTION,
    the distiller should return 'supersede'."""
    judge = _mock_judge("CONTRADICTION", "deployment target changed")
    d = _distiller(judge=judge)

    existing = Memory(content="Team deploys manually via SSH.")
    new = Memory(content="Team uses automated GitHub Actions deployment.")

    decision = d.decide(new, similar=[(existing, 0.75)], same_key=[])

    assert decision.action == "supersede"
    assert existing in decision.targets
    assert "llm_judge" in decision.reason
    judge.judge.assert_called_once_with(
        existing_fact=existing.content,
        new_fact=new.content,
    )


def test_ambiguity_zone_calls_judge_compatible():
    """When similarity is in [0.60, 0.92) and the judge says COMPATIBLE,
    the distiller should return 'create'."""
    judge = _mock_judge("COMPATIBLE", "different languages are compatible")
    d = _distiller(judge=judge)

    existing = Memory(content="User writes Python for data processing.")
    new = Memory(content="User writes Go for microservices.")

    decision = d.decide(new, similar=[(existing, 0.78)], same_key=[])

    assert decision.action == "create"
    judge.judge.assert_called_once()


def test_no_judge_ambiguity_zone_defaults_to_create():
    """Without a judge, similarity in [0.60, 0.92) should default to 'create'
    (safe fallback — never discard information without confirmation)."""
    d = _distiller(judge=None)

    existing = Memory(content="Team deploys manually via SSH.")
    new = Memory(content="Team uses automated GitHub Actions deployment.")

    decision = d.decide(new, similar=[(existing, 0.75)], same_key=[])

    assert decision.action == "create"


def test_judge_not_called_below_conflict_threshold():
    """Similarity below 0.60 should go straight to 'create' without
    invoking the judge at all."""
    judge = _mock_judge("CONTRADICTION")
    d = _distiller(judge=judge)

    existing = Memory(content="User likes pizza.")
    new = Memory(content="User uses Postgres for primary database.")

    decision = d.decide(new, similar=[(existing, 0.35)], same_key=[])

    assert decision.action == "create"
    judge.judge.assert_not_called()


def test_judge_not_called_above_dedup_threshold():
    """Similarity >= 0.92 should merge without invoking the judge."""
    judge = _mock_judge("CONTRADICTION")
    d = _distiller(judge=judge)

    existing = Memory(content="User uses Postgres for storage.")
    new = Memory(content="User relies on Postgres for storage.")

    decision = d.decide(new, similar=[(existing, 0.95)], same_key=[])

    assert decision.action == "merge"
    judge.judge.assert_not_called()


def test_judge_picks_closest_candidate_in_zone():
    """When multiple candidates fall in the ambiguity zone, the judge
    should only be called once with the closest one."""
    judge = _mock_judge("CONTRADICTION", "superseded by closer match")
    d = _distiller(judge=judge)

    far = Memory(content="Team uses Jenkins for CI.")
    close = Memory(content="Team uses CircleCI for CI.")
    new = Memory(content="Team migrated CI to GitHub Actions.")

    decision = d.decide(new, similar=[(far, 0.65), (close, 0.82)], same_key=[])

    assert decision.action == "supersede"
    assert close in decision.targets
    # Judge should be called exactly once, with the closer candidate.
    judge.judge.assert_called_once_with(
        existing_fact=close.content,
        new_fact=new.content,
    )


def test_same_key_supersede_takes_priority_over_judge():
    """Key-based conflict resolution (rule 3) should fire before the
    ambiguity zone judge (rule 4), even if there are zone candidates."""
    judge = _mock_judge("COMPATIBLE")
    d = _distiller(judge=judge)

    old = Memory(content="User uses Postgres.", key="user:database")
    zone_mem = Memory(content="User likes SQL databases.")
    new = Memory(content="User uses MySQL.", key="user:database")

    decision = d.decide(
        new,
        similar=[(old, 0.4), (zone_mem, 0.75)],
        same_key=[old],
    )

    # Key-based supersede should win, judge should NOT be called.
    assert decision.action == "supersede"
    assert old in decision.targets
    judge.judge.assert_not_called()
