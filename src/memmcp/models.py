"""Core data models for MemMCP.

A :class:`Memory` is one distilled, self-contained fact ("uses Postgres",
"prefers TypeScript") — not a raw chat transcript. Memories carry the metadata
needed for selective retrieval (importance), staleness handling (ttl, version,
superseded_by), scoping, and privacy (pii_types).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Memory:
    """A single distilled fact plus the metadata that powers ranking.

    Attributes:
        content: The distilled fact text.
        scope: Namespace this fact belongs to (see :mod:`memmcp.scoping`).
        importance: Caller / model estimate of long-term value, 0..1.
        key: Optional canonical subject-predicate key (e.g. ``"db.choice"``)
            used for conflict detection. Facts sharing a key are considered
            competing statements about the same thing.
        tags: Free-form labels for filtering.
        source: Where the fact came from (tool name, "manual", "ingest").
        ttl_days: Time-to-live in days; ``None`` means never expires.
        version: Incremented each time the fact is updated/superseded.
        superseded_by: Id of the memory that replaced this one (staleness).
        pii_types: PII categories detected in the content.
        embedding: Cached embedding vector (populated by the manager).
    """

    content: str
    scope: str = "global"
    importance: float = 0.5
    key: str | None = None
    tags: list[str] = field(default_factory=list)
    source: str = "manual"
    ttl_days: float | None = None

    # ---- lifecycle metadata ----
    id: str = field(default_factory=_new_id)
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    last_accessed_at: float = field(default_factory=_now)
    access_count: int = 0
    version: int = 1
    superseded_by: str | None = None
    pii_types: list[str] = field(default_factory=list)

    embedding: list[float] | None = None

    # -- derived helpers -------------------------------------------------
    @property
    def age_days(self) -> float:
        return max(0.0, (_now() - self.created_at) / 86_400.0)

    @property
    def is_active(self) -> bool:
        """A memory is active if it is neither superseded nor expired."""
        return self.superseded_by is None and not self.is_expired()

    def is_expired(self, at: float | None = None) -> bool:
        if self.ttl_days is None:
            return False
        at = at if at is not None else _now()
        return (at - self.created_at) > (self.ttl_days * 86_400.0)

    def touch(self) -> None:
        """Record an access (drives the recency signal & access_count)."""
        self.last_accessed_at = _now()
        self.access_count += 1

    def to_metadata(self) -> dict[str, Any]:
        """Flat, JSON-serialisable metadata for the vector store.

        The embedding is stored separately by the vector backend, so it is
        intentionally excluded here.
        """
        data = asdict(self)
        data.pop("embedding", None)
        # Vector-store metadata must be scalar; encode list fields as strings.
        data["tags"] = ",".join(self.tags)
        data["pii_types"] = ",".join(self.pii_types)
        return data

    @classmethod
    def from_metadata(cls, meta: dict[str, Any], embedding: list[float] | None = None) -> "Memory":
        meta = dict(meta)
        tags = meta.get("tags") or ""
        pii = meta.get("pii_types") or ""
        meta["tags"] = [t for t in tags.split(",") if t] if isinstance(tags, str) else list(tags)
        meta["pii_types"] = (
            [t for t in pii.split(",") if t] if isinstance(pii, str) else list(pii)
        )
        # Drop any unknown keys so the dataclass constructor stays happy.
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in meta.items() if k in allowed}
        mem = cls(**clean)
        mem.embedding = embedding
        return mem

    def summary(self) -> dict[str, Any]:
        """A compact, human/LLM-friendly view (no embedding, no PII replay)."""
        return {
            "id": self.id,
            "content": self.content,
            "scope": self.scope,
            "importance": round(self.importance, 3),
            "key": self.key,
            "tags": self.tags,
            "source": self.source,
            "age_days": round(self.age_days, 2),
            "version": self.version,
            "active": self.is_active,
            "pii_types": self.pii_types,
        }


@dataclass
class ScoredMemory:
    """A memory paired with the ranking breakdown that surfaced it."""

    memory: Memory
    score: float
    relevance: float
    recency: float
    importance: float

    def summary(self) -> dict[str, Any]:
        data = self.memory.summary()
        data["score"] = round(self.score, 4)
        data["signals"] = {
            "relevance": round(self.relevance, 4),
            "recency": round(self.recency, 4),
            "importance": round(self.importance, 4),
        }
        return data


@dataclass
class AddResult:
    """Outcome of a store/remember operation."""

    action: str  # "created" | "merged" | "superseded" | "blocked"
    memory: Memory | None
    superseded_ids: list[str] = field(default_factory=list)
    note: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "memory": self.memory.summary() if self.memory else None,
            "superseded_ids": self.superseded_ids,
            "note": self.note,
        }


@dataclass
class IngestResult:
    """Outcome of ingesting a raw conversation into distilled memories."""

    extracted: int
    stored: list[AddResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "extracted": self.extracted,
            "created": sum(1 for r in self.stored if r.action == "created"),
            "merged": sum(1 for r in self.stored if r.action == "merged"),
            "superseded": sum(1 for r in self.stored if r.action == "superseded"),
            "blocked": sum(1 for r in self.stored if r.action == "blocked"),
            "results": [r.summary() for r in self.stored],
        }
