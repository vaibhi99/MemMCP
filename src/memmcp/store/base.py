"""Vector-store interface shared by every backend.

A store owns persistence of :class:`~memmcp.models.Memory` objects and their
embeddings, plus nearest-neighbour search filtered by project name. Ranking
lives in :mod:`memmcp.retrieval`, not here — the store only returns candidates
by vector similarity.
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
        project_name: str | None,
        top_k: int,
        include_inactive: bool = False,
    ) -> list[tuple[Memory, float]]:
        """Return up to ``top_k`` ``(memory, cosine_similarity)`` pairs.

        When ``project_name`` is given, returns memories belonging to that
        project **plus** global memories (``project_name is None``).
        When ``project_name`` is ``None``, only global memories are returned.
        Inactive (superseded / expired) memories are excluded unless
        ``include_inactive``.
        """

    @abstractmethod
    def query_all_projects(
        self,
        embedding: list[float],
        top_k: int,
        include_inactive: bool = False,
    ) -> list[tuple[Memory, float]]:
        """Return up to ``top_k`` ``(memory, cosine_similarity)`` pairs
        across **all** projects (no project filter).

        Used for cross-project discovery when the caller doesn't know which
        project to search. Results include memories from every project and
        global scope.  Inactive memories are excluded unless
        ``include_inactive``.
        """

    @abstractmethod
    def all(self, project_name: str | None = None, include_inactive: bool = True) -> list[Memory]:
        """Return every stored memory, optionally filtered by project name.

        When ``project_name`` is given, returns memories belonging to that
        project **plus** global memories. When ``None``, returns everything.
        """

    @abstractmethod
    def count(self) -> int:
        """Total number of stored memories."""
