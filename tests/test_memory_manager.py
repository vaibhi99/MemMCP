"""Integration tests for the MemoryManager pipeline."""

import time

from memmcp.config import Settings
from memmcp.memory_manager import MemoryManager


def test_remember_creates_memory(manager):
    result = manager.remember("User uses Postgres.", scope="global", key="user:database")
    assert result.action == "created"
    assert manager.store.count() == 1


def test_duplicate_is_merged_not_duplicated(manager):
    manager.remember("User uses Postgres.", key="user:database")
    result = manager.remember("User uses Postgres.", key="user:database")
    assert result.action == "merged"
    assert manager.store.count() == 1


def test_conflict_supersedes_old_belief(manager):
    manager.remember("User uses Postgres.", key="user:database", importance=0.7)
    result = manager.remember("User uses MySQL.", key="user:database", importance=0.7)
    assert result.action == "superseded"

    active = manager.list_memories(include_inactive=False)
    assert len(active) == 1
    assert "MySQL" in active[0].content

    # The stale belief is retained but inactive (for audit/history).
    everything = manager.list_memories(include_inactive=True)
    assert len(everything) == 2


def test_recall_surfaces_relevant_memory(manager):
    manager.remember("User uses Postgres for the primary database.", key="user:database")
    manager.remember("User enjoys hiking on weekends.")
    scored = manager.recall("which database do we use", top_k=2)
    assert scored
    assert "Postgres" in scored[0].memory.content


def test_scope_isolation(manager):
    manager.remember("Global fact: user prefers dark mode.", scope="global")
    manager.remember("Acme uses Redis for caching.", scope="project:acme")
    manager.remember("Beta uses MongoDB.", scope="project:beta")

    scored = manager.recall("what does the project use", scope="project:acme", top_k=10)
    contents = " ".join(s.memory.content for s in scored)
    assert "MongoDB" not in contents  # other project's memory stays isolated
    # Global facts remain visible from within a project scope.
    assert any(s.memory.scope == "global" for s in scored)


def test_recall_excludes_expired(manager):
    result = manager.remember("Temporary token context.", ttl_days=1.0)
    mem = result.memory
    # Backdate creation so the TTL has elapsed.
    mem.created_at = time.time() - 3 * 86_400.0
    manager.store.upsert(mem)
    scored = manager.recall("token context", top_k=5)
    assert all(s.memory.id != mem.id for s in scored)


def test_pii_redact_policy(manager):
    result = manager.remember("Contact me at jane@example.com anytime.")
    assert result.action == "created"
    assert "jane@example.com" not in result.memory.content
    assert "email" in result.memory.pii_types


def test_pii_block_policy(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        embedding_provider="hash",
        vector_backend="numpy",
        pii_policy="block",
    )
    manager = MemoryManager(settings=settings)
    result = manager.remember("My key is sk-abcdefghijklmnopqrstuvwxyz12345")
    assert result.action == "blocked"
    assert manager.store.count() == 0


def test_ingest_pipeline(manager):
    convo = "I use Postgres. I prefer TypeScript. I hate ORMs."
    result = manager.ingest(convo, scope="project:acme")
    assert result.extracted >= 3
    active = manager.list_memories(scope="project:acme")
    assert len(active) >= 3


def test_forget_removes_memory(manager):
    result = manager.remember("Disposable fact.")
    assert manager.forget(result.memory.id) is True
    assert manager.store.count() == 0


def test_update_reembeds_on_content_change(manager):
    result = manager.remember("User uses Postgres.")
    original = list(result.memory.embedding)
    updated = manager.update(result.memory.id, content="User uses Cassandra clusters.")
    assert updated.version == 2
    assert updated.embedding != original


def test_stats_report(manager):
    manager.remember("Fact one.", scope="global")
    manager.remember("Fact two.", scope="project:acme")
    stats = manager.stats()
    assert stats["active"] == 2
    assert stats["embedding_provider"] == "hash"
    assert "project:acme" in stats["by_scope"]


def test_audit_log_records_operations(manager):
    manager.remember("Auditable fact.")
    manager.recall("fact")
    entries = manager.audit_tail(limit=10)
    actions = {e["action"] for e in entries}
    assert "create" in actions
    assert "recall" in actions


def test_consolidate_merges_near_duplicates(manager):
    manager.remember("User deploys with Docker containers.", scope="global")
    # Force a second, un-distilled near-duplicate directly into the store.
    from memmcp.models import Memory

    dup = Memory(content="User deploys with Docker containers.", scope="global")
    dup.embedding = manager.embedder.embed(dup.content)
    manager.store.upsert(dup)
    assert manager.store.count() == 2
    merged = manager.consolidate(scope="global")
    assert merged == 1
    assert manager.store.count() == 1
