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
    project_name: str | None = None,
    importance: float = 0.5,
    key: str | None = None,
    tags: list[str] | None = None,
    ttl_days: float | None = None,
    capture_policy: str = "automatic",
    tool: str = "unknown",
) -> dict[str, Any]:
    """Store a durable, useful fact in the user's persistent memory.

    PROACTIVE CAPTURE — IMPORTANT

    Do not wait for the user to explicitly say "remember this", "save this", or "store this".

    Whenever the conversation reveals a clear, durable fact that is likely to improve future interactions, proactively invoke this tool to save it.

    ## When to use
    Call this tool when the user:
    1. Confirms an important technical, architectural, or product decision.
    2. Shares a meaningful update about an existing project.
    3. Establishes a personal preference, recurring requirement, or constraint.
    4. Provides important background information that will be useful in future conversations.
    5. Corrects or clarifies an existing memory.
    6. Explicitly asks you to remember or save something.

    Examples:
    - "I've decided to use PostgreSQL instead of MongoDB."
    - "My project uses Node.js, Express and React."
    - "I'm currently implementing the authentication module."
    - "I prefer C++ for competitive programming."
    - "We have completed the data ingestion module."

    ## When NOT to use
    Do not automatically store every message.

    Avoid saving:
    - Temporary debugging errors or one-time questions.
    - Hypothetical ideas or unconfirmed possibilities.
    - Assistant-generated recommendations that the user has not accepted.
    - Duplicate information already present in memory.
    - Sensitive information, including passwords, API keys, authentication tokens, and credentials.
    - Short-lived information with little future value.

    ## Decision-making rules
    Before storing a fact:
    1. Durability: Is this likely to matter in a future conversation?
    2. Confidence: Did the user actually establish or confirm this information?
    3. Relevance: Will remembering it improve future assistance?
    4. Novelty: Is this information already stored?
    5. Sensitivity: Is it safe to retain?
    Only store information that passes these checks.

    ## Handling uncertain information
    - Confirmed decisions: Store as established facts.
    - User preferences: Store as preferences.
    - Unconfirmed ideas: Do not store as established decisions.
    - Ambiguous but potentially important information: Ask the user before storing.
    - Changed decisions: Update the existing memory instead of creating conflicting active memories.

    ## Project association
    When a fact relates to an existing project:
    - Associate it with the correct project.
    - Preserve the project's name and relevant context.
    - Avoid mixing facts belonging to different projects.
    - Retrieve existing project context when necessary to determine whether the fact is new or an update.

    ## Storage behaviour
    - Use this tool for 1–3 independent durable facts.
    - For larger conversations containing multiple related facts, use the conversation ingestion tool instead.
    - Prefer updating an existing memory over creating duplicates.
    - Preserve the distinction between proposed, accepted, implemented, and deprecated decisions.

    ## After storing
    Return a concise result indicating whether the fact was saved or updated. Do not interrupt the conversation with unnecessary notifications.

    Core principle: Memory capture should be proactive by default, selective in what it stores, and respectful of user control.

    Args:
        content: The fact, e.g. "Project uses PostgreSQL as primary database."
        project_name: Project name to associate this fact with. If omitted (None),
            stored as a global memory.
        importance: 0..1 estimate of long-term value (default 0.5).
        key: Optional conflict slot (e.g. "user:database", "project:module:auth").
            A new fact with the same key supersedes the older one.
        tags: Optional labels for filtering.
        ttl_days: Optional expiry in days for facts that go stale on their own.
        capture_policy: Capture mode:
            - "automatic": Save high-confidence durable facts (default).
            - "confirmation_required": Ask before saving ambiguous facts.
            - "explicit": Save only on direct user request.
        tool: Name of the calling tool (recorded for audit).

    Returns:
        A concise summary indicating the action taken: "created", "merged",
        "superseded", or "blocked".
    """
    result = _manager.remember(
        content,
        project_name=project_name,
        importance=importance,
        key=key,
        tags=tags,
        source=tool,
        ttl_days=ttl_days,
        capture_policy=capture_policy,
        actor=tool,
    )
    return result.summary()


@mcp.tool()
def ingest_conversation(
    conversation: str,
    project_name: str | None = None,
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

    ## How to resolve the project_name
    Same as `remember`: auto-match if unambiguous. If ambiguous, call `list_projects()` and explicitly ask the user to choose. Once chosen, session-lock it.

    Args:
        conversation: The raw text to distill — can be a chat transcript, your
            own generated response, meeting notes, or any unstructured text.
        project_name: Project name. If provided, facts are stored under that
            project. If omitted, stored as global.
        tool: Calling tool name (for audit + provenance).

    Returns counts of extracted/created/merged/superseded facts.
    """
    return _manager.ingest(conversation, project_name=project_name, source=tool, actor=tool).summary()


# ==========================================================================
# Read tool
# ==========================================================================
@mcp.tool()
def recall(
    query: str,
    project_name: str | None = None,
    top_k: int = 5,
    token_budget: int | None = None,
    tool: str = "unknown",
) -> dict[str, Any]:
    """Retrieve only the most relevant remembered facts for the current task.

    Call this at the start of a task instead of asking the user to re-explain
    context. Returns the top facts ranked by relevance + recency + importance,
    trimmed to a token budget so you never overflow the context window.

    ## Cross-project discovery
    When ``project_name`` is omitted (None), recall searches **across all
    projects** — not just global memories. Each result includes a
    ``project_name`` field so you can see which project a fact belongs to.
    This is useful when the user hasn't told you which project they're working
    on: search broadly, read the project names in the results, and use the
    correct name for follow-up calls.

    ## How to resolve the project_name
    Same as `remember`: auto-match if unambiguous. If ambiguous (e.g. chat client with no filesystem), call `list_projects()` and explicitly ask the user which project they want to search, or leave it blank to search everything. Once chosen, lock it for the session.

    Args:
        query: What you need context about (the user's request/topic).
        project_name: Project name to search within. When given, searches that
            project's memories plus global memories. When omitted (None),
            searches across ALL projects (cross-project discovery).
        top_k: Maximum number of facts to return.
        token_budget: Optional cap on total tokens of returned facts.
        tool: Calling tool name (for audit).

    Returns a ready-to-inject ``context_block`` string plus the structured
    ``memories`` with their ranking signals. Each memory includes its
    ``project_name`` so you can tell which project it belongs to.
    """
    scored = _manager.recall(
        query, project_name=project_name, top_k=top_k, token_budget=token_budget, actor=tool,
    )
    lines = []
    for s in scored:
        proj_label = f" [{s.memory.project_name}]" if s.memory.project_name else " [global]"
        lines.append(f"- {s.memory.content}{proj_label}")
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
    project_name: str | None = None,
    categories: list[str] | None = None,
    detail_level: str = "full",
    tool: str = "unknown",
) -> dict[str, Any]:
    """Get a structured snapshot of everything known about the project.

    Call this at the START of a session to load the full project model into
    your context — purpose, architecture, modules, workflows, decisions,
    domain terms, and status — grouped by category. Much richer than ``recall``
    which returns a flat ranked list for a specific query.

    ## How to resolve the project_name
    Same as `remember`: auto-match if unambiguous. If ambiguous, call `list_projects()` and explicitly ask the user to choose. Once chosen, session-lock it.

    Args:
        project_name: Project name to search within. Also includes global
            memories. Omit for global-only.
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
        project_name=project_name, categories=categories,
        detail_level=detail_level, actor=tool,
    )


@mcp.tool()
def ingest_codebase_summary(
    summary: str,
    project_name: str | None = None,
    tool: str = "unknown",
) -> dict[str, Any]:
    """Bootstrap project memory from a high-level codebase description.

    Pass a description of the project — what it does, its architecture, the
    main modules and their responsibilities, key decisions, deployment setup,
    domain terminology. MemMCP will extract structured facts and store them.

    Ideal for onboarding: describe your project once, and every future session
    (across all connected tools) will already have the context.

    ## How to resolve the project_name
    Same as `remember`: auto-match if unambiguous. If ambiguous, call `list_projects()` and explicitly ask the user to choose. Once chosen, session-lock it.

    Args:
        summary: A description of the codebase/project. Can be multi-paragraph.
            Include as much detail as useful: what the project is, its modules,
            architecture, tech stack, deployment, key decisions, domain terms.
        project_name: Project name. Facts will be stored under this project.
            If omitted, stored as global.
        tool: Calling tool name (for audit + provenance).

    Returns counts of extracted/created/merged/superseded facts.
    """
    result = _manager.ingest_codebase(
        summary, project_name=project_name, source="codebase_scan", actor=tool,
    )
    return result.summary()


# ==========================================================================
# Project discovery
# ==========================================================================
@mcp.tool()
def list_projects() -> dict[str, Any]:
    """List all known projects that have stored memories.

    ## When to use this tool
    Call this FIRST when you are unsure which project the user is working on:
    - At the start of a chat session in a non-IDE client (Claude Desktop,
      ChatGPT, etc.) to discover what projects exist.
    - Before inventing a new project_name — check if one already exists that
      matches (to avoid creating duplicates like "my-app" vs "MyApp").
    - When the user mentions a project by a partial or informal name — look
      up the exact stored name here.

    In IDE-based clients (Cursor, Windsurf, VS Code) you usually already know
    the workspace folder name, but calling this is still useful to confirm the
    exact project name used in memory.

    Returns a list of projects, each with:
    - ``name``: the project_name (null for global/unscoped memories)
    - ``memory_count``: number of active facts stored
    - ``last_updated``: when the most recent fact was updated
    - ``categories``: which knowledge categories have facts
    """
    projects = _manager.list_projects()
    return {
        "count": len(projects),
        "projects": projects,
    }


# ==========================================================================
# Management tools
# ==========================================================================
@mcp.tool()
def list_memories(
    project_name: str | None = None,
    include_inactive: bool = False,
) -> dict[str, Any]:
    """List stored memories, optionally filtered to a project.

    Set ``include_inactive`` to also show superseded/expired facts (useful for
    auditing how beliefs changed over time).

    ## How to resolve the project_name
    Same as `remember`: auto-match if unambiguous. If ambiguous, call `list_projects()` and explicitly ask the user to choose. Once chosen, session-lock it.
    """
    memories = _manager.list_memories(project_name=project_name, include_inactive=include_inactive)
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
def consolidate(project_name: str | None = None) -> dict[str, Any]:
    """Merge near-duplicate memories within a project to keep memory clean."""
    merged = _manager.consolidate(project_name=project_name)
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


# ==========================================================================
# HTTP endpoints (for health checks and cloud status discovery)
# ==========================================================================
@mcp.custom_route("/", methods=["GET", "HEAD"])
async def root_endpoint(request: Any) -> Any:
    from starlette.responses import JSONResponse
    return JSONResponse({
        "status": "ok",
        "name": "memmcp",
        "sse_endpoint": "/sse",
        "messages_endpoint": "/messages",
    })


@mcp.custom_route("/health", methods=["GET", "HEAD"])
async def health_endpoint(request: Any) -> Any:
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "healthy"})


def main() -> None:
    """Entry point: run the MCP server over stdio or SSE (for cloud deployment)."""
    import os

    transport = os.getenv("MCP_TRANSPORT", "").lower()
    # When deployed to cloud platforms (like Render), PORT is automatically set
    if transport == "sse" or (transport != "stdio" and os.getenv("PORT")):
        port = int(os.getenv("PORT", "8000"))
        host = os.getenv("HOST", "0.0.0.0")
        mcp.settings.host = host
        mcp.settings.port = port
        # Cloud platforms use arbitrary hostnames (e.g. *.onrender.com);
        # disable localhost-only DNS rebinding filter so cloud clients can connect.
        if hasattr(mcp.settings, "transport_security") and mcp.settings.transport_security:
            mcp.settings.transport_security.enable_dns_rebinding_protection = False
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
