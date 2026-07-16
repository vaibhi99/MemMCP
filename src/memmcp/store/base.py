"""Vector-store interface shared by every backend.

A store owns persistence of :class:`~memmcp.models.Memory` objects and their
embeddings, plus nearest-neighbour search filtered by scope. Ranking lives in
:mod:`memmcp.retrieval`, not here — the store only returns candidates by
vector similarity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Memory


class VectorStore(ABC):
    """Persistence + similarity search over memories."""

    @abstractmethod
    def upsert(self, memory: Memory) -> None:
        """Insert or replace a memory (embedding must be populated)."""

    @abstractmethod
    def get(self, memory_id: str) -> Memory | None:
        """Fetch a single memory by id."""

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Remove a memory; return True if it existed."""

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        scopes: list[str],
        top_k: int,
        include_inactive: bool = False,
    ) -> list[tuple[Memory, float]]:
        """Return up to ``top_k`` ``(memory, cosine_similarity)`` pairs.

        Only memories whose ``scope`` is in ``scopes`` are considered. Inactive
        (superseded / expired) memories are excluded unless ``include_inactive``.
        """

    @abstractmethod
    def all(self, scopes: list[str] | None = None, include_inactive: bool = True) -> list[Memory]:
        """Return every stored memory, optionally filtered by scope."""

    @abstractmethod
    def count(self) -> int:
        """Total number of stored memories."""
