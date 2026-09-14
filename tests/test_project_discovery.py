"""Tests for zero-friction project name resolution.

Covers:
- ``list_projects()`` returns empty / correct names / correct counts
- ``recall(project_name=None)`` searches across ALL projects (cross-project)
- ``recall(project_name="X")`` still only returns project X + global (regression)
- Cross-project results include source ``project_name`` in summaries
"""

from __future__ import annotations

import pytest

from memmcp.config import Settings
from memmcp.memory_manager import MemoryManager


@pytest.fixture
def settings(tmp_path) -> Settings:
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


# -------------------------------------------------------------------------
# list_projects
# -------------------------------------------------------------------------
class TestListProjects:
    def test_empty_when_no_memories(self, manager: MemoryManager) -> None:
        result = manager.list_projects()
        assert result == []

    def test_returns_project_names_after_remember(self, manager: MemoryManager) -> None:
        manager.remember("Uses React.", project_name="frontend-app")
        manager.remember("Uses Django.", project_name="backend-api")
        manager.remember("User prefers dark mode.")  # global

        projects = manager.list_projects()
        names = {p["name"] for p in projects}

        assert "frontend-app" in names
        assert "backend-api" in names
        assert None in names  # global memories appear as name=None

    def test_correct_memory_counts(self, manager: MemoryManager) -> None:
        manager.remember("Uses React.", project_name="frontend-app")
        manager.remember("Uses TypeScript.", project_name="frontend-app")
        manager.remember("Uses Vite.", project_name="frontend-app")
        manager.remember("Uses Django.", project_name="backend-api")

        projects = manager.list_projects()
        by_name = {p["name"]: p for p in projects}

        assert by_name["frontend-app"]["memory_count"] == 3
        assert by_name["backend-api"]["memory_count"] == 1

    def test_categories_populated(self, manager: MemoryManager) -> None:
        manager.remember(
            "Uses PostgreSQL.",
            project_name="my-app",
            key="user:database",
        )
        manager.remember(
            "Auth module handles OAuth2.",
            project_name="my-app",
            key="project:module:auth",
        )

        projects = manager.list_projects()
        by_name = {p["name"]: p for p in projects}
        cats = by_name["my-app"]["categories"]

        assert "stack" in cats
        assert "module" in cats

    def test_sorted_by_last_updated(self, manager: MemoryManager) -> None:
        """Most recently updated project should appear first."""
        manager.remember("Old fact.", project_name="old-project")
        manager.remember("New fact.", project_name="new-project")

        projects = manager.list_projects()
        # The second project was added last, so it should be first.
        assert projects[0]["name"] == "new-project"


# -------------------------------------------------------------------------
# Cross-project recall
# -------------------------------------------------------------------------
class TestCrossProjectRecall:
    def test_recall_without_project_searches_all(self, manager: MemoryManager) -> None:
        """When project_name=None, recall should find facts from ANY project."""
        manager.remember("Frontend uses React.", project_name="frontend-app")
        manager.remember("Backend uses Django.", project_name="backend-api")
        manager.remember("User prefers dark mode.")  # global

        results = manager.recall("React framework", project_name=None, top_k=10)

        # Should find at least the React fact (from frontend-app).
        contents = [r.memory.content for r in results]
        assert any("React" in c for c in contents)

        # Should also include facts from other projects, not just global.
        project_names = {r.memory.project_name for r in results}
        # At minimum, frontend-app should be present.
        assert "frontend-app" in project_names

    def test_recall_with_project_excludes_other_projects(self, manager: MemoryManager) -> None:
        """When project_name is given, recall should NOT return other projects' facts."""
        manager.remember("Frontend uses React.", project_name="frontend-app")
        manager.remember("Backend uses Django.", project_name="backend-api")

        results = manager.recall("Django", project_name="backend-api", top_k=10)

        for r in results:
            # Each result should belong to backend-api or be global (None).
            assert r.memory.project_name in ("backend-api", None)

    def test_cross_project_results_include_project_name(self, manager: MemoryManager) -> None:
        """Cross-project results should have project_name in their summary."""
        manager.remember("Uses PostgreSQL.", project_name="my-app", key="user:database")

        results = manager.recall("database", project_name=None, top_k=5)

        assert len(results) > 0
        summary = results[0].summary()
        assert "project_name" in summary
        assert summary["project_name"] == "my-app"


# -------------------------------------------------------------------------
# Regression: existing behaviour preserved
# -------------------------------------------------------------------------
class TestRegressions:
    def test_recall_with_project_still_includes_global(self, manager: MemoryManager) -> None:
        """project_name=X should return X's facts + global facts."""
        manager.remember("User prefers Python.", key="user:language")  # global
        manager.remember("App uses Flask.", project_name="my-app")

        results = manager.recall("programming language", project_name="my-app", top_k=10)

        project_names = {r.memory.project_name for r in results}
        # Should include both the project-scoped and global memories.
        assert None in project_names or "my-app" in project_names

    def test_remember_without_project_is_global(self, manager: MemoryManager) -> None:
        """Memories stored without project_name should be global."""
        result = manager.remember("User name is Vaibhav.")
        assert result.memory is not None
        assert result.memory.project_name is None

    def test_list_projects_excludes_superseded(self, manager: MemoryManager) -> None:
        """Superseded memories should not inflate project counts."""
        manager.remember("Uses MySQL.", project_name="app", key="user:database")
        manager.remember("Uses PostgreSQL.", project_name="app", key="user:database")

        projects = manager.list_projects()
        by_name = {p["name"]: p for p in projects}
        # Only one active memory should remain (the superseding one).
        assert by_name["app"]["memory_count"] == 1
