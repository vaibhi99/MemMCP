"""MemoryManager — the brain that wires every subsystem together.

This is the pipeline the resume describes:

    ingest -> extract -> PII filter -> embed -> distill (dedup/conflict)
           -> store -> (later) retrieve -> rank -> token-budget -> audit

Everything above (embeddings, store, ranker, extractor, distiller, privacy)
is pluggable; this class orchestrates them and enforces scoping + governance.
It is transport-agnostic, so it is exercised directly by tests and wrapped by
the MCP server in :mod:`memmcp.server`.
"""

from __future__ import annotations

from typing import Any

from . import scoping
from .config import Settings, get_settings
from .embeddings import build_embedder
from .extraction import Distiller, build_extractor
from .models import AddResult, IngestResult, Memory, ScoredMemory
from .privacy import AuditLog, PIIScanner
from .retrieval import Ranker
from .store import build_store


class MemoryManager:
    """High-level API over the whole memory pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()

        self.embedder = build_embedder(self.settings)
        self.store = build_store(self.settings)
        self.ranker = Ranker(
            weights=self.settings.ranking_weights(),
            recency_halflife_days=self.settings.recency_halflife_days,
        )
        self.distiller = Distiller(
            dedup_threshold=self.settings.dedup_threshold,
            conflict_threshold=self.settings.conflict_threshold,
        )
        self.extractor = build_extractor(self.settings)
        self.pii = PIIScanner()
        self.audit = AuditLog(self.settings.audit_log_path)

    # ==================================================================
    # Write path
    # ==================================================================
    def remember(
        self,
        content: str,
        *,
        scope: str | None = None,
        importance: float = 0.5,
        key: str | None = None,
        tags: list[str] | None = None,
        source: str = "manual",
        ttl_days: float | None = None,
        actor: str = "unknown",
    ) -> AddResult:
        """Store a single fact, applying PII policy + distillation."""
        scope = scoping.normalize(scope)
        content = content.strip()
        if not content:
            return AddResult("blocked", None, note="empty content")

        # --- privacy gate ---
        pii_types: list[str] = []
        result = self.pii.scan(content)
        policy = self.settings.pii_policy
        if result.has_pii:
            if policy == "block":
                self.audit.record(
                    "store_blocked", actor=actor, scope=scope,
                    details={"pii": result.types},
                )
                return AddResult("blocked", None, note=f"PII detected: {result.types}")
            if policy == "redact":
                content = result.redacted
                pii_types = result.types
            elif policy == "flag":
                pii_types = result.types
            # "off": ignore

        memory = Memory(
            content=content,
            scope=scope,
            importance=max(0.0, min(1.0, importance)),
            key=key,
            tags=tags or [],
            source=source,
            ttl_days=ttl_days,
            pii_types=pii_types,
        )
        memory.embedding = self.embedder.embed(content)

        return self._distill_and_store(memory, actor=actor)

    def _distill_and_store(self, memory: Memory, actor: str) -> AddResult:
        readable = scoping.readable_scopes(memory.scope)
        similar = self.store.query(memory.embedding, readable, top_k=10)
        # Conflict resolution is confined to the writing scope so a project
        # write can never silently supersede a broader/global belief.
        same_key = [
            m
            for m in self.store.all(scopes=[memory.scope], include_inactive=False)
            if memory.key and m.key == memory.key
        ]

        decision = self.distiller.decide(memory, similar, same_key)

        if decision.action == "merge":
            target = decision.targets[0]
            target.importance = max(target.importance, memory.importance)
            target.tags = sorted(set(target.tags) | set(memory.tags))
            target.touch()
            self.store.upsert(target)
            self.audit.record(
                "merge", actor=actor, scope=memory.scope, memory_id=target.id,
                details={"reason": decision.reason, "similarity": round(decision.similarity, 3)},
            )
            return AddResult("merged", target, note=decision.reason)

        # Restrict supersede targets to the same scope.
        supersede_targets = [t for t in decision.targets if t.scope == memory.scope]
        if decision.action == "supersede" and supersede_targets:
            superseded_ids = []
            max_version = memory.version
            for target in supersede_targets:
                target.superseded_by = memory.id
                self.store.upsert(target)
                superseded_ids.append(target.id)
                max_version = max(max_version, target.version + 1)
            memory.version = max_version
            self.store.upsert(memory)
            self.audit.record(
                "supersede", actor=actor, scope=memory.scope, memory_id=memory.id,
                details={"reason": decision.reason, "superseded": superseded_ids},
            )
            return AddResult("superseded", memory, superseded_ids, decision.reason)

        # Default: create.
        self.store.upsert(memory)
        self.audit.record(
            "create", actor=actor, scope=memory.scope, memory_id=memory.id,
            details={"key": memory.key, "pii": memory.pii_types},
        )
        return AddResult("created", memory, note=decision.reason)

    def ingest(
        self,
        conversation: str | list[dict],
        *,
        scope: str | None = None,
        source: str = "ingest",
        actor: str = "unknown",
    ) -> IngestResult:
        """Extract facts from a raw conversation and store each distilled fact."""
        scope = scoping.normalize(scope)
        facts = self.extractor.extract(conversation)
        results: list[AddResult] = []
        for fact in facts:
            results.append(
                self.remember(
                    fact.content,
                    scope=scope,
                    importance=fact.importance,
                    key=fact.key,
                    tags=fact.tags,
                    source=source,
                    actor=actor,
                )
            )
        self.audit.record(
            "ingest", actor=actor, scope=scope,
            details={"extracted": len(facts), "stored": len(results)},
        )
        return IngestResult(extracted=len(facts), stored=results)

    # ==================================================================
    # Read path
    # ==================================================================
    def recall(
        self,
        query: str,
        *,
        scope: str | None = None,
        top_k: int = 5,
        token_budget: int | None = None,
        actor: str = "unknown",
    ) -> list[ScoredMemory]:
        """Return the most relevant memories for ``query`` within ``scope``.

        Blends relevance + recency + importance, then greedily packs the result
        into ``token_budget`` (surface the few facts that matter, not 50 pages).
        """
        scopes = scoping.readable_scopes(scope)
        query_vec = self.embedder.embed(query)
        # Over-fetch candidates so ranking has room to reorder.
        candidates = self.store.query(query_vec, scopes, top_k=max(top_k * 4, 20))
        scored = self.ranker.score(candidates)
        packed = self.ranker.pack(scored, token_budget=token_budget, top_k=top_k)

        # Record accesses so recency reflects real usage.
        for item in packed:
            item.memory.touch()
            self.store.upsert(item.memory)

        self.audit.record(
            "recall", actor=actor, scope=scoping.normalize(scope),
            details={
                "query_chars": len(query),
                "returned": [i.memory.id for i in packed],
            },
        )
        return packed

    # ==================================================================
    # Management
    # ==================================================================
    def get(self, memory_id: str) -> Memory | None:
        return self.store.get(memory_id)

    def forget(self, memory_id: str, *, actor: str = "unknown") -> bool:
        existed = self.store.delete(memory_id)
        self.audit.record("forget", actor=actor, memory_id=memory_id, details={"existed": existed})
        return existed

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        ttl_days: float | None = None,
        actor: str = "unknown",
    ) -> Memory | None:
        """Edit a memory in place, re-embedding if the content changed."""
        memory = self.store.get(memory_id)
        if memory is None:
            return None
        if content is not None and content.strip() and content.strip() != memory.content:
            scanned = self.pii.scan(content.strip())
            if scanned.has_pii and self.settings.pii_policy == "block":
                return memory  # refuse the edit; leave original untouched
            memory.content = scanned.redacted if self.settings.pii_policy == "redact" else content.strip()
            memory.pii_types = scanned.types if scanned.has_pii else []
            memory.embedding = self.embedder.embed(memory.content)
        if importance is not None:
            memory.importance = max(0.0, min(1.0, importance))
        if tags is not None:
            memory.tags = tags
        if ttl_days is not None:
            memory.ttl_days = ttl_days
        memory.version += 1
        memory.updated_at = memory.last_accessed_at
        self.store.upsert(memory)
        self.audit.record("update", actor=actor, scope=memory.scope, memory_id=memory.id)
        return memory

    def list_memories(
        self,
        *,
        scope: str | None = None,
        include_inactive: bool = False,
    ) -> list[Memory]:
        scopes = [scoping.normalize(scope)] if scope else None
        memories = self.store.all(scopes=scopes, include_inactive=include_inactive)
        return sorted(memories, key=lambda m: m.created_at, reverse=True)

    def consolidate(self, *, scope: str | None = None, actor: str = "system") -> int:
        """Merge near-duplicate active memories within a scope.

        A lightweight take on memory consolidation: for each active memory,
        fold any later near-duplicates into it (keeping the max importance).
        Returns the number of memories merged away.
        """
        scopes = [scoping.normalize(scope)] if scope else None
        active = [m for m in self.store.all(scopes=scopes, include_inactive=False)]
        merged = 0
        kept: list[Memory] = []
        for mem in sorted(active, key=lambda m: m.created_at):
            duplicate_of = None
            from .embeddings import cosine_similarity

            for keeper in kept:
                if keeper.scope != mem.scope or mem.embedding is None:
                    continue
                if cosine_similarity(keeper.embedding, mem.embedding) >= self.settings.dedup_threshold:
                    duplicate_of = keeper
                    break
            if duplicate_of is not None:
                duplicate_of.importance = max(duplicate_of.importance, mem.importance)
                duplicate_of.tags = sorted(set(duplicate_of.tags) | set(mem.tags))
                self.store.upsert(duplicate_of)
                self.store.delete(mem.id)
                merged += 1
            else:
                kept.append(mem)
        self.audit.record("consolidate", actor=actor, details={"merged": merged})
        return merged

    # ==================================================================
    # Structured project knowledge
    # ==================================================================
    def get_project_context(
        self,
        *,
        scope: str | None = None,
        categories: list[str] | None = None,
        detail_level: str = "full",
        actor: str = "unknown",
    ) -> dict[str, Any]:
        """Return a structured project snapshot grouped by entity category.

        Unlike ``recall`` (which returns a flat ranked list), this method
        organises memories into semantic groups (project info, modules,
        workflows, decisions, domain terms, status) so a client can inject
        a complete project model at session start.

        Args:
            scope: Namespace to search (sees ancestors + global).
            categories: Optional filter — only include these categories.
                Valid: identity, stack, project, module, workflow, decision,
                       domain, status.  ``None`` means all.
            detail_level: ``"full"`` (all content) or ``"summary"`` (first
                sentence only, for lightweight context injection).
            actor: Calling tool name (for audit).
        """
        from .models import CATEGORIES, _category_from_key

        scopes = scoping.readable_scopes(scope)
        active = [
            m for m in self.store.all(scopes=scopes, include_inactive=False)
        ]

        # Group by category.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for mem in sorted(active, key=lambda m: m.importance, reverse=True):
            cat = _category_from_key(mem.key)
            if cat is None:
                cat = "other"
            if categories and cat not in categories:
                continue

            content = mem.content
            if detail_level == "summary":
                # First sentence only.
                content = content.split(".")[0] + "." if "." in content else content

            entry: dict[str, Any] = {
                "content": content,
                "key": mem.key,
                "importance": round(mem.importance, 3),
            }
            grouped.setdefault(cat, []).append(entry)

        # Build the structured output.
        result: dict[str, Any] = {"scope": scoping.normalize(scope)}
        for cat_key in CATEGORIES:
            if categories and cat_key not in categories:
                continue
            items = grouped.get(cat_key, [])
            if items:
                result[cat_key] = items
        # Include uncategorised facts.
        if "other" in grouped and (not categories or "other" in categories):
            result["other"] = grouped["other"]

        self.audit.record(
            "project_context", actor=actor,
            scope=scoping.normalize(scope),
            details={"categories": list(result.keys()), "total_facts": sum(
                len(v) for v in result.values() if isinstance(v, list)
            )},
        )
        return result

    def ingest_codebase(
        self,
        summary: str,
        *,
        project_name: str | None = None,
        source: str = "codebase_scan",
        actor: str = "unknown",
    ) -> IngestResult:
        """Bootstrap project memory from a codebase description.

        Like ``ingest()`` but sets the scope to the project and uses a higher
        default importance since codebase descriptions are authoritative.

        Args:
            summary: High-level description of the codebase (what it is, its
                modules, architecture, etc.).
            project_name: Optional project name for scoping.
            source: Provenance label.
            actor: Calling tool name.
        """
        scope = scoping.make(project=project_name) if project_name else "global"
        return self.ingest(summary, scope=scope, source=source, actor=actor)

    def stats(self) -> dict[str, Any]:
        all_mems = self.store.all(include_inactive=True)
        active = [m for m in all_mems if m.is_active]
        by_scope: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for m in active:
            by_scope[m.scope] = by_scope.get(m.scope, 0) + 1
            cat = m.category or "other"
            by_category[cat] = by_category.get(cat, 0) + 1
        return {
            "total": len(all_mems),
            "active": len(active),
            "inactive": len(all_mems) - len(active),
            "with_pii": sum(1 for m in active if m.pii_types),
            "by_scope": by_scope,
            "by_category": by_category,
            "embedding_provider": self.settings.embedding_provider,
            "embedding_dim": self.embedder.dim,
            "vector_backend": self.settings.vector_backend,
            "extraction_provider": self.settings.extraction_provider,
            "pii_policy": self.settings.pii_policy,
        }

    def audit_tail(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.audit.tail(limit)
