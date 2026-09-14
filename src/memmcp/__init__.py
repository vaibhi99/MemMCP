"""MemMCP — a cross-tool AI memory server built on the Model Context Protocol.

MemMCP gives LLM clients (Claude, Cursor, ChatGPT, IDEs) a persistent, shared,
*selective* memory layer so users stop re-explaining context across tools.

The package is organised into focused sub-systems:

    config          Environment-driven settings.
    models          Core data types (Memory, ScoredMemory, results).
    embeddings/     Pluggable embedding providers (hash, local, OpenAI).
    store/          Pluggable vector stores (numpy, Chroma).
    retrieval/      Relevance + recency + importance ranking.
    extraction/     Fact extraction and distillation (dedup, conflicts).
    privacy/        PII detection/redaction and audit logging.
    memory_manager  The orchestrator wiring everything together.
    server          The MCP server exposing tools to LLM clients.
"""

from .config import Settings, get_settings
from .models import Memory, ScoredMemory
from .memory_manager import MemoryManager

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "get_settings",
    "Memory",
    "ScoredMemory",
    "MemoryManager",
    "__version__",
]
