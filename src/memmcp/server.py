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

    ## When to use this tool
    Use `remember` when you identify **1 to 3 clear, distinct facts** during a
    conversation — a preference, a tech-stack choice, an architectural decision,
    an identity detail, a constraint, a workflow step, or a domain term.

    Call it in real time as the conversation progresses. You do NOT need to wait
    for the user to explicitly ask you to save something. If the user confirms
    a plan you proposed (e.g. "ok, let's go with that"), save the key decisions
    from your proposal immediately.

    ## When NOT to use this tool
    If you just generated (or the user pasted) a dense block of text containing
    **many facts** (architecture docs, meeting notes, large plans), use
    `ingest_conversation` instead — it extracts and deduplicates automatically.
    Rule of thumb: if you would need more than ~3 remember calls, prefer ingest.

    ## What to save (not just tech stack!)
    Save ALL categories of durable knowledge:
    - **Identity**: user name, role, company, team
    - **Stack**: language, framework, database, cache, cloud, editor, CI/CD
    - **Architecture**: project purpose, system design, deployment target
    - **Modules**: component names, responsibilities, boundaries, dependencies
    - **Workflows**: CI/CD pipelines, data flows, deployment steps
    - **Decisions**: architectural choices and the reasoning behind them
    - **Domain terms**: project-specific jargon and definitions
    - **Status**: current milestones, blockers, known tech debt

    ## How to write a good fact
    Write a concise, **third-person, present-tense, self-contained** sentence.
    Good:  "Project uses PostgreSQL as primary database."
    Bad:   "they said postgres" (not self-contained, ambiguous)

    Args:
        content: The fact, e.g. "Project uses PostgreSQL as primary database."
        scope: Namespace. "global", "project:<name>", "tool:<name>", or
            "project:<name>/tool:<name>". Facts are visible to that scope and
            its descendants.
        importance: 0..1 estimate of long-term value.
            - 0.85–0.95: identity, architecture, core decisions
            - 0.75–0.85: module descriptions, workflows
            - 0.70–0.80: stack choices
            - 0.65–0.80: domain terms, status
            - 0.50–0.65: preferences, dislikes
            - 0.40–0.50: ephemeral observations
        key: Optional conflict slot (e.g. "user:database"). A new fact with the
            same key supersedes the old one, so beliefs stay current.
            Common keys: user:name, user:language, user:framework, user:database,
            user:cloud, user:editor, user:current_project, project:architecture,
            project:deployment, project:purpose, project:status.
            For modules/workflows/decisions use dynamic keys like
            project:module:<name>, project:workflow:<name>, project:decision:<name>.
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
    """Distill a raw conversation or dense text into clean facts and store them.

    ## When to use this tool
    Use this when a large block of text needs to be turned into memory:
    - The user pastes a **meeting transcript, chat log, or design doc**.
    - You just generated a **dense response** (e.g. a full architecture plan,
      a multi-module breakdown, a deployment strategy) that contains many facts
      worth remembering. Pass your own response text into this tool.
    - The user asks you to "remember this" or "save this" about a big block.
    - Rule of thumb: if the text contains **more than ~3 distinct facts**, use
      this tool instead of calling `remember` many times.

    ## What happens internally
    MemMCP's backend uses a dedicated extraction LLM to read the full text
    (both user and assistant turns), extract only the durable conclusions,
    standardise them into clean facts with conflict keys, and deduplicate
    against existing memory. Dead ends, chit-chat, and corrected mistakes are
    discarded automatically.

    Args:
        conversation: The raw text to distill — can be a chat transcript, your
            own generated response, meeting notes, or any unstructured text.
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
# Structured project knowledge tools
# ==========================================================================
@mcp.tool()
def get_project_context(
    scope: str = "global",
    categories: list[str] | None = None,
    detail_level: str = "full",
    tool: str = "unknown",
) -> dict[str, Any]:
    """Get a structured snapshot of everything known about the project.

    Call this at the START of a session to load the full project model into
    your context — purpose, architecture, modules, workflows, decisions,
    domain terms, and status — grouped by category. Much richer than ``recall``
    which returns a flat ranked list for a specific query.

    Args:
        scope: Namespace to search (sees this scope, its ancestors, and global).
        categories: Optional filter — only include specific categories.
            Valid values: "identity", "stack", "project", "module", "workflow",
            "decision", "domain", "status". Pass null/omit for all categories.
        detail_level: "full" (complete content) or "summary" (first sentence
            only, for lightweight injection when context is tight).
        tool: Calling tool name (for audit).

    Returns a dict keyed by category, each containing a list of facts with
    their content, key, and importance. Only non-empty categories appear.
    """
    return _manager.get_project_context(
        scope=scope, categories=categories,
        detail_level=detail_level, actor=tool,
    )


@mcp.tool()
def ingest_codebase_summary(
    summary: str,
    project_name: str | None = None,
    scope: str | None = None,
    tool: str = "unknown",
) -> dict[str, Any]:
    """Bootstrap project memory from a high-level codebase description.

    Pass a description of the project — what it does, its architecture, the
    main modules and their responsibilities, key decisions, deployment setup,
    domain terminology. MemMCP will extract structured facts and store them.

    Ideal for onboarding: describe your project once, and every future session
    (across all connected tools) will already have the context.

    Args:
        summary: A description of the codebase/project. Can be multi-paragraph.
            Include as much detail as useful: what the project is, its modules,
            architecture, tech stack, deployment, key decisions, domain terms.
        project_name: Optional project name (used for scoping, e.g. "acme").
        scope: Explicit scope override. If omitted, auto-derived from
            project_name (e.g. "project:acme") or defaults to "global".
        tool: Calling tool name (for audit + provenance).

    Returns counts of extracted/created/merged/superseded facts.
    """
    if scope:
        from . import scoping
        result = _manager.ingest(summary, scope=scope, source="codebase_scan", actor=tool)
    else:
        result = _manager.ingest_codebase(
            summary, project_name=project_name, source="codebase_scan", actor=tool,
        )
    return result.summary()


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
