"""Tests for the expanded project knowledge extraction and structured retrieval.

Covers:
- normalize_key with new dynamic prefixes (domain, debt)
- Memory.category property
- get_project_context structured grouping
- ingest_codebase bootstrapping
- Conflict resolution on project-level keys
"""

from memmcp.extraction.fact_extractor import normalize_key, _parse_llm_json
from memmcp.models import Memory, CATEGORIES, _category_from_key


# =========================================================================
# normalize_key — new dynamic prefixes
# =========================================================================
class TestNormalizeKeyDynamic:
    def test_module_key_passes_through(self):
        assert normalize_key("project:module:auth") == "project:module:auth"

    def test_workflow_key_passes_through(self):
        assert normalize_key("project:workflow:ci_cd") == "project:workflow:ci_cd"

    def test_decision_key_passes_through(self):
        assert normalize_key("project:decision:cloud_provider") == "project:decision:cloud_provider"

    def test_domain_key_passes_through(self):
        assert normalize_key("project:domain:payment_intent") == "project:domain:payment_intent"

    def test_debt_key_passes_through(self):
        assert normalize_key("project:debt:notification_polling") == "project:debt:notification_polling"

    def test_empty_dynamic_key_is_dropped(self):
        # "project:module:" with no name should not be accepted.
        assert normalize_key("project:module:") is None

    def test_new_static_keys(self):
        assert normalize_key("project:status") == "project:status"
        assert normalize_key("project:purpose") == "project:purpose"
        assert normalize_key("project:architecture") == "project:architecture"

    def test_new_aliases(self):
        assert normalize_key("project:goal") == "project:purpose"
        assert normalize_key("project:description") == "project:purpose"
        assert normalize_key("project:overview") == "project:purpose"
        assert normalize_key("project:deploy") == "project:deployment"
        assert normalize_key("project:arch") == "project:architecture"
        assert normalize_key("project:tech_debt") == "project:status"

    def test_case_and_spaces_normalised(self):
        assert normalize_key("Project:Module:Auth") == "project:module:auth"
        assert normalize_key("  project:domain:payment intent  ") == "project:domain:payment_intent"


# =========================================================================
# Memory.category property
# =========================================================================
class TestMemoryCategory:
    def test_identity_category(self):
        m = Memory(content="User's name is Alice.", key="user:name")
        assert m.category == "identity"

    def test_stack_category(self):
        m = Memory(content="User uses Postgres.", key="user:database")
        assert m.category == "stack"

    def test_project_category(self):
        m = Memory(content="Project uses microservices.", key="project:architecture")
        assert m.category == "project"

    def test_module_category(self):
        m = Memory(content="Module auth handles JWTs.", key="project:module:auth")
        assert m.category == "module"

    def test_workflow_category(self):
        m = Memory(content="CI pipeline runs tests.", key="project:workflow:ci")
        assert m.category == "workflow"

    def test_decision_category(self):
        m = Memory(content="Chose GCP for K8s.", key="project:decision:cloud")
        assert m.category == "decision"

    def test_domain_category(self):
        m = Memory(content="PaymentIntent is a pending charge.", key="project:domain:payment_intent")
        assert m.category == "domain"

    def test_status_category(self):
        m = Memory(content="Working on refunds.", key="project:status")
        assert m.category == "status"

    def test_debt_is_status_category(self):
        m = Memory(content="Polling instead of webhooks.", key="project:debt:notifications")
        assert m.category == "status"

    def test_no_key_means_no_category(self):
        m = Memory(content="Random fact.", key=None)
        assert m.category is None

    def test_unknown_key_means_no_category(self):
        m = Memory(content="Something odd.", key="unknown:thing")
        assert m.category is None

    def test_category_in_summary(self):
        m = Memory(content="Module auth handles JWTs.", key="project:module:auth")
        s = m.summary()
        assert s["category"] == "module"


# =========================================================================
# CATEGORIES constant
# =========================================================================
def test_all_expected_categories_present():
    expected = {"identity", "stack", "project", "module", "workflow", "decision", "domain", "status"}
    assert set(CATEGORIES.keys()) == expected


# =========================================================================
# _parse_llm_json — richer extraction
# =========================================================================
class TestParseLlmJsonRich:
    def test_parses_module_fact(self):
        raw = '[{"content": "Module auth handles JWTs.", "importance": 0.8, "key": "project:module:auth", "tags": ["module"]}]'
        facts = _parse_llm_json(raw)
        assert len(facts) == 1
        assert facts[0].key == "project:module:auth"
        assert facts[0].importance == 0.8

    def test_parses_decision_fact(self):
        raw = '[{"content": "Decision: cloud — chose GCP.", "importance": 0.9, "key": "project:decision:cloud", "tags": ["decision"]}]'
        facts = _parse_llm_json(raw)
        assert facts[0].key == "project:decision:cloud"

    def test_parses_domain_fact(self):
        raw = '[{"content": "Domain term PaymentIntent means pending charge.", "importance": 0.7, "key": "project:domain:payment_intent", "tags": ["domain"]}]'
        facts = _parse_llm_json(raw)
        assert facts[0].key == "project:domain:payment_intent"

    def test_parses_debt_fact(self):
        raw = '[{"content": "Tech debt: polling instead of webhooks.", "importance": 0.7, "key": "project:debt:notifications", "tags": ["status"]}]'
        facts = _parse_llm_json(raw)
        assert facts[0].key == "project:debt:notifications"

    def test_parses_mixed_categories(self):
        raw = """[
            {"content": "User uses Postgres.", "importance": 0.8, "key": "user:database", "tags": ["stack"]},
            {"content": "Module auth handles JWTs.", "importance": 0.8, "key": "project:module:auth", "tags": ["module"]},
            {"content": "Project deploys to GCP.", "importance": 0.85, "key": "project:deployment", "tags": ["project"]}
        ]"""
        facts = _parse_llm_json(raw)
        assert len(facts) == 3
        keys = {f.key for f in facts}
        assert keys == {"user:database", "project:module:auth", "project:deployment"}


# =========================================================================
# get_project_context — structured retrieval
# =========================================================================
class TestGetProjectContext:
    def test_groups_by_category(self, manager):
        manager.remember("User uses Postgres.", key="user:database", importance=0.8)
        manager.remember("Module auth handles JWTs.", key="project:module:auth", importance=0.8)
        manager.remember("Project purpose is a payments service.", key="project:purpose", importance=0.9)

        ctx = manager.get_project_context()
        assert "stack" in ctx
        assert "module" in ctx
        assert "project" in ctx
        # Stack should contain the Postgres fact.
        assert any("Postgres" in f["content"] for f in ctx["stack"])
        # Module should contain the auth fact.
        assert any("auth" in f["content"] for f in ctx["module"])

    def test_category_filter(self, manager):
        manager.remember("User uses Postgres.", key="user:database", importance=0.8)
        manager.remember("Module auth handles JWTs.", key="project:module:auth", importance=0.8)

        ctx = manager.get_project_context(categories=["module"])
        assert "module" in ctx
        assert "stack" not in ctx

    def test_summary_detail_level(self, manager):
        manager.remember(
            "Module auth handles JWT tokens. It also manages refresh tokens and session cookies.",
            key="project:module:auth",
            importance=0.8,
        )
        ctx = manager.get_project_context(detail_level="summary")
        assert "module" in ctx
        # Summary should only have the first sentence.
        content = ctx["module"][0]["content"]
        assert content.endswith(".")
        assert "refresh tokens" not in content

    def test_scope_isolation(self, manager):
        manager.remember("Acme module auth.", key="project:module:auth", scope="project:acme")
        manager.remember("Beta module billing.", key="project:module:billing", scope="project:beta")

        ctx = manager.get_project_context(scope="project:acme")
        module_contents = " ".join(f["content"] for f in ctx.get("module", []))
        assert "auth" in module_contents
        assert "billing" not in module_contents

    def test_uncategorised_facts_in_other(self, manager):
        manager.remember("Some random useful fact.", key=None, importance=0.5)
        ctx = manager.get_project_context()
        assert "other" in ctx

    def test_empty_when_no_memories(self, manager):
        ctx = manager.get_project_context()
        # Only "scope" key, no categories.
        assert ctx == {"scope": "global"}


# =========================================================================
# ingest_codebase — project bootstrapping
# =========================================================================
class TestIngestCodebase:
    def test_ingests_and_stores_facts(self, manager):
        description = (
            "I use Postgres for the database. "
            "We're building a payments API. "
            "I prefer TypeScript."
        )
        result = manager.ingest_codebase(description, project_name="payments")
        assert result.extracted >= 2
        active = manager.list_memories(scope="project:payments")
        assert len(active) >= 2

    def test_scopes_to_project(self, manager):
        result = manager.ingest_codebase("I use Redis.", project_name="myapp")
        for add_result in result.stored:
            if add_result.memory:
                assert add_result.memory.scope == "project:myapp"

    def test_defaults_to_global_without_project_name(self, manager):
        result = manager.ingest_codebase("I use Postgres.")
        for add_result in result.stored:
            if add_result.memory:
                assert add_result.memory.scope == "global"


# =========================================================================
# Conflict resolution on project-level keys
# =========================================================================
class TestProjectKeyConflicts:
    def test_architecture_supersedes(self, manager):
        manager.remember(
            "Project uses monolithic architecture.",
            key="project:architecture", importance=0.85,
        )
        result = manager.remember(
            "Project uses microservices architecture.",
            key="project:architecture", importance=0.85,
        )
        assert result.action == "superseded"
        active = manager.list_memories(include_inactive=False)
        assert len(active) == 1
        assert "microservices" in active[0].content

    def test_deployment_supersedes(self, manager):
        manager.remember("Project deploys to AWS.", key="project:deployment")
        result = manager.remember("Project deploys to GCP.", key="project:deployment")
        assert result.action == "superseded"

    def test_module_key_supersedes(self, manager):
        manager.remember("Module auth handles session cookies.", key="project:module:auth")
        result = manager.remember("Module auth handles JWTs.", key="project:module:auth")
        assert result.action == "superseded"
        active = manager.list_memories(include_inactive=False)
        assert any("JWTs" in m.content for m in active)


# =========================================================================
# stats includes category breakdown
# =========================================================================
def test_stats_includes_categories(manager):
    manager.remember("User uses Postgres.", key="user:database")
    manager.remember("Module auth handles JWTs.", key="project:module:auth")
    stats = manager.stats()
    assert "by_category" in stats
    assert stats["by_category"].get("stack", 0) >= 1
    assert stats["by_category"].get("module", 0) >= 1
