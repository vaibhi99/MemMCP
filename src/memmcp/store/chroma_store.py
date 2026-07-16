"""Production vector-database backend using ChromaDB (optional).

Selected via ``MEMMCP_VECTOR_BACKEND=chroma`` after
``pip install memmcp[chroma]``. The full :class:`~memmcp.models.Memory` is
stored as a JSON payload in Chroma metadata, while ``scope`` is kept as a
filterable field so nearest-neighbour search can be constrained server-side.
Active/expiry filtering is applied in Python after retrieval because a memory
can expire purely by the passage of time (no write to flip a flag).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import Memory
from .base import VectorStore


class ChromaVectorStore(VectorStore):
    """VectorStore backed by a persistent Chroma collection."""

    def __init__(self, path: Path, collection_name: str = "memmcp") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "chromadb is not installed. Run `pip install memmcp[chroma]` "
                "or set MEMMCP_VECTOR_BACKEND=numpy."
            ) from exc

        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _encode(memory: Memory) -> dict[str, str]:
        record = memory.to_metadata()
        return {"scope": memory.scope, "payload": json.dumps(record)}

    @staticmethod
    def _decode(meta: dict, embedding: list[float] | None) -> Memory:
        record = json.loads(meta["payload"])
        return Memory.from_metadata(record, embedding=embedding)

    # -- CRUD ------------------------------------------------------------
    def upsert(self, memory: Memory) -> None:
        if memory.embedding is None:
            raise ValueError("Memory.embedding must be set before upsert().")
        self._collection.upsert(
            ids=[memory.id],
            embeddings=[memory.embedding],
            metadatas=[self._encode(memory)],
        )

    def get(self, memory_id: str) -> Memory | None:
        res = self._collection.get(ids=[memory_id], include=["metadatas", "embeddings"])
        if not res["ids"]:
            return None
        emb = res["embeddings"][0] if res.get("embeddings") is not None else None
        return self._decode(res["metadatas"][0], emb)

    def delete(self, memory_id: str) -> bool:
        if self.get(memory_id) is None:
            return False
        self._collection.delete(ids=[memory_id])
        return True

    # -- search ----------------------------------------------------------
    def query(
        self,
        embedding: list[float],
        scopes: list[str],
        top_k: int,
        include_inactive: bool = False,
    ) -> list[tuple[Memory, float]]:
        if self._collection.count() == 0:
            return []
        # Over-fetch so post-filtering for active memories still fills top_k.
        n = min(self._collection.count(), max(top_k * 3, top_k))
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where={"scope": {"$in": scopes}} if scopes else None,
            include=["metadatas", "embeddings", "distances"],
        )
        out: list[tuple[Memory, float]] = []
        metadatas = res["metadatas"][0]
        distances = res["distances"][0]
        embeddings = res["embeddings"][0]
        for meta, dist, emb in zip(metadatas, distances, embeddings):
            mem = self._decode(meta, list(emb) if emb is not None else None)
            if not include_inactive and not mem.is_active:
                continue
            similarity = 1.0 - float(dist)  # cosine distance -> similarity
            out.append((mem, similarity))
            if len(out) >= top_k:
                break
        return out

    def all(self, scopes: list[str] | None = None, include_inactive: bool = True) -> list[Memory]:
        where = {"scope": {"$in": scopes}} if scopes else None
        res = self._collection.get(where=where, include=["metadatas", "embeddings"])
        embeddings = res.get("embeddings") or [None] * len(res["ids"])
        out: list[Memory] = []
        for meta, emb in zip(res["metadatas"], embeddings):
            mem = self._decode(meta, list(emb) if emb is not None else None)
            if include_inactive or mem.is_active:
                out.append(mem)
        return out

    def count(self) -> int:
        return int(self._collection.count())
