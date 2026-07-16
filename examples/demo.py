"""End-to-end demo of MemMCP running fully offline (no API keys, no servers).

Run it::

    python examples/demo.py

It walks through the whole pipeline: ingesting a noisy conversation, storing
manual facts, deduplication, conflict/staleness handling, scoped + ranked
recall, PII redaction, and the audit log.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from memmcp.config import Settings
from memmcp.memory_manager import MemoryManager


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="memmcp_demo_"))
    settings = Settings(
        data_dir=tmp,
        embedding_provider="hash",
        vector_backend="numpy",
        extraction_provider="rule",
        pii_policy="redact",
    )
    mem = MemoryManager(settings=settings)

    banner("1. Ingest a noisy conversation -> distilled facts")
    conversation = """
    user: Hey, for the new service I use Postgres.
    assistant: Sounds good, Postgres is solid.
    user: Actually on reflection I prefer TypeScript for the whole stack.
    user: And honestly I hate ORMs.
    user: My name is Vaibhav by the way.
    """
    result = mem.ingest(conversation, scope="project:payments", actor="cursor")
    for r in result.stored:
        print(f"  [{r.action:9}] {r.memory.content if r.memory else '-'}")

    banner("2. A belief changes -> old fact is superseded, not duplicated")
    mem.remember("User uses MySQL.", scope="project:payments", key="user:database", importance=0.8)
    print("  Active facts in project:payments:")
    for m in mem.list_memories(scope="project:payments"):
        print(f"    - {m.content}  (key={m.key})")

    banner("3. Selective, ranked recall (surface the few facts that matter)")
    hits = mem.recall("what database and language should I use here?", scope="project:payments", top_k=3)
    for h in hits:
        s = h.summary()
        print(f"  score={s['score']:.3f}  {s['content']}")
        print(f"      signals={s['signals']}")

    banner("4. PII is redacted before storage")
    r = mem.remember("Ping me at vaibhav@example.com or 555-123-4567.")
    print(f"  stored content : {r.memory.content}")
    print(f"  detected PII   : {r.memory.pii_types}")

    banner("5. Scope isolation (another project cannot see these facts)")
    other = mem.recall("database", scope="project:analytics", top_k=5)
    print(f"  recall in project:analytics returned {len(other)} facts (isolated).")

    banner("6. Audit trail")
    for entry in mem.audit_tail(limit=6):
        print(f"  {entry['iso']}  {entry['action']:12} scope={entry['scope']}")

    banner("Stats")
    for k, v in mem.stats().items():
        print(f"  {k:22}: {v}")

    print(f"\nData written to: {tmp}")


if __name__ == "__main__":
    main()
