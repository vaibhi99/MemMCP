"""Offline vector store: in-memory index with JSON persistence.

Backed by a single numpy matrix for vectorised cosine search. Good to tens of
thousands of memories on a laptop; swap in Chroma for larger corpora. Writes
are persisted eagerly so state survives restarts (the MCP server is typically
launched fresh by each client).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from ..models import Memory
from .base import VectorStore


class NumpyVectorStore(VectorStore):
    """A simple, correct, persistent vector store using numpy."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._memories: dict[str, Memory] = {}
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for record in raw.get("memories", []):
            emb = record.pop("embedding", None)
            mem = Memory.from_metadata(record, embedding=emb)
            self._memories[mem.id] = mem

    def _flush(self) -> None:
        records = []
        for mem in self._memories.values():
            rec = mem.to_metadata()
            rec["embedding"] = mem.embedding
            records.append(rec)
        payload = {"version": 1, "memories": records}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self._path)

    # -- CRUD ------------------------------------------------------------
    def upsert(self, memory: Memory) -> None:
        if memory.embedding is None:
            raise ValueError("Memory.embedding must be set before upsert().")
        with self._lock:
            self._memories[memory.id] = memory
            self._flush()

    def get(self, memory_id: str) -> Memory | None:
        with self._lock:
            return self._memories.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            existed = self._memories.pop(memory_id, None) is not None
            if existed:
                self._flush()
            return existed

    # -- search ----------------------------------------------------------
    def query(
        self,
        embedding: list[float],
        scopes: list[str],
        top_k: int,
        include_inactive: bool = False,
    ) -> list[tuple[Memory, float]]:
        with self._lock:
            scope_set = set(scopes)
            candidates = [
                m
                for m in self._memories.values()
                if m.scope in scope_set and (include_inactive or m.is_active)
            ]
            if not candidates:
                return []

            matrix = np.asarray([m.embedding for m in candidates], dtype=np.float32)
            query_vec = np.asarray(embedding, dtype=np.float32)

            # Vectors are stored L2-normalised, but normalise defensively.
            matrix_norms = np.linalg.norm(matrix, axis=1)
            matrix_norms[matrix_norms == 0] = 1.0
            q_norm = np.linalg.norm(query_vec) or 1.0
            sims = (matrix @ query_vec) / (matrix_norms * q_norm)

            k = min(top_k, len(candidates))
            top_idx = np.argsort(-sims)[:k]
            return [(candidates[i], float(sims[i])) for i in top_idx]

    def all(self, scopes: list[str] | None = None, include_inactive: bool = True) -> list[Memory]:
        with self._lock:
            scope_set = set(scopes) if scopes is not None else None
            return [
                m
                for m in self._memories.values()
                if (scope_set is None or m.scope in scope_set)
                and (include_inactive or m.is_active)
            ]

    def count(self) -> int:
        with self._lock:
            return len(self._memories)
