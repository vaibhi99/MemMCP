"""Shared pytest fixtures — everything runs offline with the hash embedder."""

from __future__ import annotations

import pytest

from memmcp.config import Settings
from memmcp.memory_manager import MemoryManager


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Isolated settings using a temp data dir and offline backends."""
    return Settings(
        data_dir=tmp_path / "data",
        embedding_provider="hash",
        embedding_dim=256,
        vector_backend="numpy",
        extraction_provider="rule",
        pii_policy="redact",
    )


@pytest.fixture
def manager(settings) -> MemoryManager:
    return MemoryManager(settings=settings)
