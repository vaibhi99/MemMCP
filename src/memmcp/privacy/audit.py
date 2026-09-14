"""Append-only audit log of every memory operation.

Each line is a self-contained JSON record (JSONL), so the log is easy to tail,
grep, ship to a SIEM, or replay. We log *what happened* (action, project,
actor, memory id, PII types) but never the raw sensitive content, so the audit
trail itself does not become a leak.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class AuditLog:
    """Thread-safe, append-only JSONL writer."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        action: str,
        *,
        actor: str = "unknown",
        project_name: str | None = None,
        memory_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append one audit event."""
        entry = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action,
            "actor": actor,
            "project_name": project_name,
            "memory_id": memory_id,
            "details": details or {},
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent ``limit`` audit entries (newest last)."""
        if not self._path.exists():
            return []
        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
