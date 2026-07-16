"""MCP server exposing MemMCP to any Model Context Protocol client.

Built on the official ``mcp`` SDK's FastMCP. Clients (Claude Desktop, Cursor,
ChatGPT, custom IDEs) connect over stdio and call these tools to give the model
a persistent, shared, selective memory. The docstrings below are what the LLM
sees when deciding which tool to call, so they are written for the model.

Run it::

    memmcp                      # console script (see pyproject.toml)
    python -m memmcp.server     # module form
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .memory_manager import MemoryManager

mcp = FastMCP("memmcp")
_manager = MemoryManager()


# ==========================================================================
# Write tools
# ==========================================================================
@mcp.tool()
def remember(
    content: str,
    scope: str = "global",
    importance: float = 0.5,
    key: str | None = None,
    tags: list[str] | None = None,
    ttl_days: float | None = None,
    tool: str = "unknown",
) -> dict[str, Any]:
    """Store ONE durable fact about the user or project for future recall.

    Use this whenever you learn something worth remembering across sessions and
    tools (a preference, a decision, an identity detail, a constraint). Save a
    concise, self-contained fact — not a whole transcript.

    Args:
        content: The fact, e.g. "User prefers TypeScript and dislikes ORMs".
        scope: Namespace. "global", "project:<name>", "tool:<name>", or
            "project:<name>/tool:<name>". Facts are visible to that scope and
            its descendants.
        importance: 0..1 estimate of long-term value (identity/decisions high).
        key: Optional conflict slot (e.g. "user:database"). A new fact with the
            same key supersedes the old one, so beliefs stay current.
        tags: Optional labels for filtering.
        ttl_days: Optional expiry in days for facts that go stale on their own.
        tool: Name of the calling tool (recorded for audit).

    Returns a summary including the resulting action: created, merged (a
    duplicate already existed), superseded (an older conflicting fact was
    retired), or blocked (PII policy refused it).
    """
    result = _manager.remember(
        content, scope=scope, importance=importance, key=key,
        tags=tags, source=tool, ttl_days=ttl_days, actor=tool,
    )
    return result.summary()


@mcp.tool()
def ingest_conversation(
    conversation: str,
    scope: str = "global",
    tool: str = "unknown",
) -> dict[str, Any]:
    """Distill a raw conversation into clean facts and store them.

    Pass a transcript (or a summary of one). MemMCP extracts durable facts,
    drops noise/dead-ends, deduplicates against existing memory, and supersedes
    any contradicted beliefs. Prefer this over pasting whole chats around.

    Args:
        conversation: The raw text to distill.
        scope: Namespace to store the resulting facts in (see `remember`).
        tool: Calling tool name (for audit + provenance).

    Returns counts of extracted/created/merged/superseded facts.
    """
    return _manager.ingest(conversation, scope=scope, source=tool, actor=tool).summary()


# ==========================================================================
# Read tool
# ==========================================================================
@mcp.tool()
def recall(
    query: str,
    scope: str = "global",
    top_k: int = 5,
    token_budget: int | None = None,
    tool: str = "unknown",
) -> dict[str, Any]:
    """Retrieve only the most relevant remembered facts for the current task.

    Call this at the start of a task instead of asking the user to re-explain
    context. Returns the top facts ranked by relevance + recency + importance,
    trimmed to a token budget so you never overflow the context window.

    Args:
        query: What you need context about (the user's request/topic).
        scope: Namespace to search (sees this scope, its ancestors, and global).
        top_k: Maximum number of facts to return.
        token_budget: Optional cap on total tokens of returned facts.
        tool: Calling tool name (for audit).

    Returns a ready-to-inject ``context_block`` string plus the structured
    ``memories`` with their ranking signals.
    """
    scored = _manager.recall(
        query, scope=scope, top_k=top_k, token_budget=token_budget, actor=tool,
    )
    lines = [f"- {s.memory.content}" for s in scored]
    context_block = "Relevant remembered context:\n" + "\n".join(lines) if lines else ""
    return {
        "count": len(scored),
        "context_block": context_block,
        "memories": [s.summary() for s in scored],
    }


# ==========================================================================
# Management tools
# ==========================================================================
@mcp.tool()
def list_memories(
    scope: str | None = None,
    include_inactive: bool = False,
) -> dict[str, Any]:
    """List stored memories, optionally filtered to a scope.

    Set ``include_inactive`` to also show superseded/expired facts (useful for
    auditing how beliefs changed over time).
    """
    memories = _manager.list_memories(scope=scope, include_inactive=include_inactive)
    return {"count": len(memories), "memories": [m.summary() for m in memories]}


@mcp.tool()
def update_memory(
    memory_id: str,
    content: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
    ttl_days: float | None = None,
) -> dict[str, Any]:
    """Edit an existing memory by id (re-embeds if the content changes)."""
    memory = _manager.update(
        memory_id, content=content, importance=importance, tags=tags, ttl_days=ttl_days,
    )
    if memory is None:
        return {"ok": False, "error": f"No memory with id {memory_id!r}"}
    return {"ok": True, "memory": memory.summary()}


@mcp.tool()
def forget(memory_id: str, tool: str = "unknown") -> dict[str, Any]:
    """Permanently delete a memory by id."""
    existed = _manager.forget(memory_id, actor=tool)
    return {"ok": existed, "memory_id": memory_id}


@mcp.tool()
def consolidate(scope: str | None = None) -> dict[str, Any]:
    """Merge near-duplicate memories within a scope to keep memory clean."""
    merged = _manager.consolidate(scope=scope)
    return {"merged": merged}


# ==========================================================================
# Resources (read-only, browsable state)
# ==========================================================================
@mcp.resource("memmcp://stats")
def stats_resource() -> dict[str, Any]:
    """Current memory statistics and active configuration."""
    return _manager.stats()


@mcp.resource("memmcp://audit")
def audit_resource() -> list[dict[str, Any]]:
    """The most recent audit-log entries (what was stored/recalled and by whom)."""
    return _manager.audit_tail(limit=100)


def main() -> None:
    """Entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
