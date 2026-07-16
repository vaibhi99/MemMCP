"""Scoping — project / tool level namespaces for memories.

Cross-tool memory only works if it does not cross-contaminate. A fact learned
in one project should not leak into an unrelated one, and a private "tool"
scope should stay out of shared retrieval. Scopes are simple, hierarchical
strings so they map cleanly onto vector-store metadata filters.

Grammar::

    global                      shared across everything
    project:<name>              everything within a project
    project:<name>/tool:<tool>  a specific tool inside a project
    tool:<tool>                  a tool, across projects

Retrieval for a scope also matches its ancestors (a query inside
``project:acme/tool:cursor`` also sees ``project:acme`` and ``global`` facts),
so shared knowledge flows down while specific facts stay contained.
"""

from __future__ import annotations

import re

GLOBAL = "global"
_VALID = re.compile(r"^[a-zA-Z0-9_.\-]+$")


def normalize(scope: str | None) -> str:
    """Validate and canonicalise a scope string."""
    if not scope:
        return GLOBAL
    scope = scope.strip().lower()
    if scope == GLOBAL:
        return GLOBAL
    parts = scope.split("/")
    for part in parts:
        if ":" not in part:
            raise ValueError(
                f"Invalid scope segment {part!r}; expected '<kind>:<name>' "
                f"(e.g. 'project:acme' or 'tool:cursor')."
            )
        kind, _, name = part.partition(":")
        if kind not in {"project", "tool"} or not _VALID.match(name):
            raise ValueError(f"Invalid scope segment {part!r}.")
    return "/".join(parts)


def make(project: str | None = None, tool: str | None = None) -> str:
    """Build a scope from optional project + tool components."""
    segments = []
    if project:
        segments.append(f"project:{project.strip().lower()}")
    if tool:
        segments.append(f"tool:{tool.strip().lower()}")
    return normalize("/".join(segments)) if segments else GLOBAL


def ancestors(scope: str) -> list[str]:
    """Return ``scope`` plus every broader scope that it can read from.

    Example::

        ancestors("project:acme/tool:cursor")
        -> ["project:acme/tool:cursor", "project:acme", "global"]
    """
    scope = normalize(scope)
    result: list[str] = []
    if scope != GLOBAL:
        parts = scope.split("/")
        for i in range(len(parts), 0, -1):
            result.append("/".join(parts[:i]))
    result.append(GLOBAL)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    return [s for s in result if not (s in seen or seen.add(s))]


def readable_scopes(scope: str | None) -> list[str]:
    """Scopes visible to a reader at ``scope`` (self + ancestors)."""
    return ancestors(normalize(scope))
