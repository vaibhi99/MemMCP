"""Fact extraction: conversation -> candidate distilled facts.

Two providers share one interface:

* :class:`RuleExtractor` (default, offline) — pattern-based extraction of
  first-person statements about identity, stack choices, preferences and
  dislikes, plus a small domain taxonomy that assigns a *conflict key* (e.g.
  "uses Postgres" and "uses MySQL" both map to ``user:database`` so the later
  one can supersede the earlier).
* LLM extractors (:func:`build_extractor` returns these when configured) —
  prompt an OpenAI/Anthropic model with a JSON-validated template.  These are
  the **recommended production path** — they extract architecture, modules,
  workflows, decisions, domain concepts, and project status in addition to
  tech stack facts.

All extractors return :class:`ExtractedFact` objects; storage, dedup and
conflict resolution happen later in the manager/distiller.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ..config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFact:
    """A candidate fact produced by an extractor (pre-storage)."""

    content: str
    importance: float = 0.5
    key: str | None = None
    tags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Domain taxonomy: maps a keyword to a canonical conflict key. Two facts that
# resolve to the same key are competing statements about the same slot, so a
# newer one supersedes the older (staleness handling).
# --------------------------------------------------------------------------
_TAXONOMY: dict[str, set[str]] = {
    "user:database": {
        "postgres", "postgresql", "mysql", "mongodb", "mongo", "sqlite",
        "cassandra", "dynamodb", "cosmosdb", "mariadb", "oracle",
    },
    "user:cache": {
        "redis", "memcached", "valkey",
    },
    "user:language": {
        "python", "go", "golang", "rust", "typescript", "javascript", "java",
        "c#", "csharp", "c++", "ruby", "php", "kotlin", "swift", "scala",
    },
    "user:framework": {
        "react", "vue", "angular", "svelte", "django", "flask", "fastapi",
        "express", "nextjs", "next.js", "spring", "rails", "laravel",
    },
    "user:cloud": {"aws", "azure", "gcp", "vercel", "netlify", "heroku", "cloudflare"},
    "user:editor": {"vscode", "cursor", "vim", "neovim", "emacs", "intellij", "sublime"},
    "user:os": {"windows", "macos", "mac", "linux", "ubuntu", "fedora", "wsl"},
}


def _category_for(text: str) -> str | None:
    tokens = set(re.findall(r"[a-z0-9#.+]+", text.lower()))
    for category, keywords in _TAXONOMY.items():
        if tokens & keywords:
            return category
    return None


class RuleExtractor:
    """Offline, deterministic pattern-based extractor."""

    # (regex, verb-normalisation, importance, optional-key-prefix)
    _USE_RE = re.compile(
        r"\b(i|we)\s+(?:am\s+|are\s+)?"
        r"(use|uses|used|using|prefer|prefers|chose|choose|chosen|"
        r"switched\s+to|moved\s+to|migrated\s+to|adopted|standardized\s+on)\s+"
        r"(.+)",
        re.IGNORECASE,
    )
    _DISLIKE_RE = re.compile(
        r"\b(i|we)\s+(hate|hates|dislike|dislikes|avoid|avoids|"
        r"don'?t\s+like|do\s+not\s+like|can'?t\s+stand)\s+(.+)",
        re.IGNORECASE,
    )
    _IDENTITY_RE = re.compile(
        r"\bmy\s+(name|email|company|team|role|title)\s+is\s+(.+)",
        re.IGNORECASE,
    )
    _WORKING_RE = re.compile(
        r"\b(i'?m|i\s+am|we'?re|we\s+are)\s+(working\s+on|building|developing|"
        r"designing|maintaining)\s+(.+)",
        re.IGNORECASE,
    )

    def extract(self, conversation: str | list[dict]) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        seen: set[str] = set()
        for sentence in self._sentences(conversation):
            for fact in self._match(sentence):
                norm = fact.content.strip().lower()
                if norm and norm not in seen:
                    seen.add(norm)
                    facts.append(fact)
        return facts

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _sentences(conversation: str | list[dict]) -> list[str]:
        """Flatten a conversation into user-authored sentences.

        Only user/human turns are mined so we never memorise the assistant's
        speculation as if it were fact.
        """
        text_parts: list[str] = []
        if isinstance(conversation, list):
            for turn in conversation:
                role = str(turn.get("role", "user")).lower()
                if role in {"user", "human"}:
                    text_parts.append(str(turn.get("content", "")))
        else:
            text_parts.append(str(conversation))
        blob = "\n".join(text_parts)
        # Split on sentence terminators and newlines.
        raw = re.split(r"(?<=[.!?])\s+|\n+", blob)
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _clean_object(obj: str) -> str:
        obj = obj.strip().rstrip(".!?,;")
        # Drop trailing subordinate clauses for a tighter fact.
        obj = re.split(r"\s+\b(because|since|so|but|and then)\b", obj)[0]
        return obj.strip()

    def _match(self, sentence: str) -> list[ExtractedFact]:
        out: list[ExtractedFact] = []

        m = self._IDENTITY_RE.search(sentence)
        if m:
            attr, value = m.group(1).lower(), self._clean_object(m.group(2))
            out.append(
                ExtractedFact(
                    content=f"User's {attr} is {value}.",
                    importance=0.9,
                    key=f"user:{attr}",
                    tags=["identity"],
                )
            )

        m = self._USE_RE.search(sentence)
        if m:
            subject = "Team" if m.group(1).lower() == "we" else "User"
            obj = self._clean_object(m.group(3))
            verb = re.sub(r"\s+", " ", m.group(2).lower())
            verb = {
                "use": "uses", "uses": "uses", "used": "uses", "using": "uses",
                "prefer": "prefers", "prefers": "prefers",
                "choose": "chose", "chose": "chose", "chosen": "chose",
                "adopted": "adopted", "standardized on": "standardized on",
                "switched to": "switched to", "moved to": "moved to",
                "migrated to": "migrated to",
            }.get(verb, verb)
            category = _category_for(obj)
            out.append(
                ExtractedFact(
                    content=f"{subject} {verb} {obj}.",
                    importance=0.7,
                    key=category,
                    tags=["stack"] + ([category.split(":")[1]] if category else []),
                )
            )

        m = self._DISLIKE_RE.search(sentence)
        if m:
            subject = "Team" if m.group(1).lower() == "we" else "User"
            obj = self._clean_object(m.group(3))
            category = _category_for(obj)
            key = f"dislike:{category}" if category else None
            out.append(
                ExtractedFact(
                    content=f"{subject} dislikes {obj}.",
                    importance=0.6,
                    key=key,
                    tags=["preference", "dislike"],
                )
            )

        m = self._WORKING_RE.search(sentence)
        if m:
            subject = "Team" if m.group(1).lower().startswith("we") else "User"
            obj = self._clean_object(m.group(3))
            out.append(
                ExtractedFact(
                    content=f"{subject} is working on {obj}.",
                    importance=0.65,
                    key="user:current_project",
                    tags=["project"],
                )
            )

        return out


# --------------------------------------------------------------------------
# LLM extractors (optional). They share the ExtractedFact contract.
# --------------------------------------------------------------------------
_LLM_PROMPT = """\
You are a **Project Knowledge Extractor**. Your job is to extract durable, \
self-contained FACTS about the user, their team, and their project from a \
conversation. Go beyond tech stack — capture architecture, modules, workflows, \
decisions, domain concepts, and project status.

Ignore chit-chat, dead ends, assistant speculation, and \
corrected mistakes — keep only final conclusions worth remembering later.

## Rules
1. Write each fact as a THIRD-PERSON, PRESENT-TENSE, SELF-CONTAINED sentence.
2. Use the EXACT verb patterns shown in the examples below — do NOT rephrase \
with synonyms (e.g. always "uses" not "likes", "utilizes", "works with").
3. Assign a canonical conflict key from the ALLOWED LIST when applicable, or null.
4. If the user corrected themselves, extract ONLY the final decision.
5. For facts that do NOT match any canonical pattern, use this format: \
"{{Subject}} {{verb}} {{detail}}." with key: null. \
Keep the same style: third-person, present-tense, concise, self-contained.
6. Extract ALL categories of knowledge — not just tech stack. Look for \
architecture, module descriptions, workflows, decisions, domain terms, \
and project status.

## Allowed Keys
### Identity
user:name, user:email, user:company, user:team, user:role

### Stack / Tooling
user:language, user:framework, user:database, user:cache,
user:cloud, user:editor, user:os, user:current_project,
user:testing, user:ci_cd, user:auth, user:styling

### Project-Level
project:purpose, project:architecture, project:deployment, project:status

### Dynamic Keys (replace <name> with the actual name, lowercased with underscores)
project:module:<name>    — for component/module descriptions
project:workflow:<name>  — for processes, pipelines, data flows
project:decision:<name>  — for architectural decisions (ADRs)
project:domain:<name>    — for project-specific terminology/concepts
project:debt:<name>      — for known technical debt items

## Canonical Fact Patterns (use these EXACT phrasings)

### Identity
  "User's name is {{value}}."                        → key: user:name
  "User's email is {{value}}."                       → key: user:email
  "User's role is {{value}}."                        → key: user:role
  "User works at {{value}}."                         → key: user:company

### Stack Choices (always use "uses")
  "User uses {{tool}} as primary database."          → key: user:database
  "User uses {{tool}} for caching."                  → key: user:cache
  "User uses {{language}} as primary language."      → key: user:language
  "User uses {{framework}} for frontend."            → key: user:framework
  "User uses {{framework}} for backend."             → key: user:framework
  "User uses {{provider}} for cloud hosting."        → key: user:cloud
  "User uses {{editor}} as code editor."             → key: user:editor
  "User uses {{tool}} for testing."                  → key: user:testing
  "User uses {{tool}} for CI/CD."                    → key: user:ci_cd

### Preferences & Dislikes
  "User prefers {{X}} over {{Y}}."                   → key: based on category
  "User dislikes {{thing}}."                         → key: null

### Project Info
  "User is working on {{description}}."              → key: user:current_project
  "Project purpose is {{description}}."              → key: project:purpose
  "Project uses {{pattern}} architecture."           → key: project:architecture
  "Project deploys to {{target}}."                   → key: project:deployment
  "Project status is {{description}}."               → key: project:status

### Modules (dynamic key — replace <name> with the module name)
  "Module {{name}} handles {{responsibility}}."      → key: project:module:<name>
  "Module {{name}} depends on {{other modules}}."    → key: project:module:<name>
  "Module {{name}} exposes {{API/interface}}."       → key: project:module:<name>

### Workflows (dynamic key — replace <name> with the workflow name)
  "Workflow {{name}} involves {{steps}}."            → key: project:workflow:<name>
  "Data flows from {{A}} to {{B}} via {{method}}."   → key: project:workflow:<name>

### Decisions (dynamic key — replace <name> with the decision topic)
  "Decision: {{topic}} — chose {{choice}} because {{reason}}."  → key: project:decision:<name>

### Domain Concepts (dynamic key — replace <name> with the term)
  "Domain term '{{term}}' means {{definition}}."     → key: project:domain:<name>

### Status & Tech Debt
  "Project status is {{description}}."               → key: project:status
  "Tech debt: {{description}}."                      → key: project:debt:<name>

## Importance Calibration
- Identity / Architecture / Core Decisions: 0.85 – 0.95
- Module descriptions / Workflows: 0.75 – 0.85
- Stack choices: 0.70 – 0.80
- Domain terms / Status: 0.65 – 0.80
- Preferences / Dislikes: 0.50 – 0.65
- Ephemeral observations: 0.40 – 0.50

## Example

Conversation:
---
user: I love using Postgres for my main DB and Redis for caching.
user: We're building a payments microservice in TypeScript. The auth module \
handles JWTs and the ledger module does double-entry bookkeeping.
user: For deployment, we use GitHub Actions to deploy to AWS EKS.
user: Actually, we decided to switch to GCP for deployment because of \
their better Kubernetes tooling.
user: We call the internal transfer format a "PaymentIntent" — it's our \
domain object that represents a pending charge.
user: Right now we're finishing the refund flow, and we have tech debt \
around the notification system — it's still using polling instead of webhooks.
---
JSON:
[
  {{"content": "User uses Postgres as primary database.", "importance": 0.8, \
"key": "user:database", "tags": ["stack", "database"]}},
  {{"content": "User uses Redis for caching.", "importance": 0.7, \
"key": "user:cache", "tags": ["stack", "cache"]}},
  {{"content": "User is working on a payments microservice.", "importance": 0.65, \
"key": "user:current_project", "tags": ["project"]}},
  {{"content": "User uses TypeScript as primary language.", "importance": 0.8, \
"key": "user:language", "tags": ["stack", "language"]}},
  {{"content": "Module auth handles JWT-based authentication.", "importance": 0.8, \
"key": "project:module:auth", "tags": ["module"]}},
  {{"content": "Module ledger handles double-entry bookkeeping.", "importance": 0.8, \
"key": "project:module:ledger", "tags": ["module"]}},
  {{"content": "Project deploys to GCP.", "importance": 0.85, \
"key": "project:deployment", "tags": ["project", "deployment"]}},
  {{"content": "Decision: cloud_provider — chose GCP because of better Kubernetes tooling.", \
"importance": 0.9, "key": "project:decision:cloud_provider", "tags": ["decision"]}},
  {{"content": "User uses GitHub Actions for CI/CD.", "importance": 0.75, \
"key": "user:ci_cd", "tags": ["stack", "ci_cd"]}},
  {{"content": "Domain term 'PaymentIntent' means a domain object representing a pending charge.", \
"importance": 0.7, "key": "project:domain:payment_intent", "tags": ["domain"]}},
  {{"content": "Project status is finishing the refund flow.", "importance": 0.65, \
"key": "project:status", "tags": ["status"]}},
  {{"content": "Tech debt: notification system uses polling instead of webhooks.", \
"importance": 0.7, "key": "project:debt:notification_polling", "tags": ["status", "debt"]}}
]

Now extract facts from this conversation:
---
{conversation}
---
JSON:"""


# --------------------------------------------------------------------------
# Key normalisation: clamp LLM-generated keys to the allowed vocabulary.
# --------------------------------------------------------------------------
_ALLOWED_KEYS: set[str] = {
    "user:name", "user:email", "user:company", "user:team", "user:role",
    "user:language", "user:framework", "user:database", "user:cache",
    "user:cloud", "user:editor", "user:os", "user:current_project",
    "user:testing", "user:ci_cd", "user:auth", "user:styling",
    "project:purpose", "project:architecture", "project:deployment",
    "project:status",
}

# Dynamic key prefixes — keys starting with these are valid without being
# in the static _ALLOWED_KEYS set.
_DYNAMIC_KEY_PREFIXES: tuple[str, ...] = (
    "project:module:",
    "project:workflow:",
    "project:decision:",
    "project:domain:",
    "project:debt:",
)

_KEY_ALIASES: dict[str, str] = {
    "user:db": "user:database",
    "user:primary_database": "user:database",
    "user:relational_db": "user:database",
    "user:caching": "user:cache",
    "user:redis": "user:cache",
    "user:memcached": "user:cache",
    "user:lang": "user:language",
    "user:primary_language": "user:language",
    "user:frontend_framework": "user:framework",
    "user:backend_framework": "user:framework",
    "user:ide": "user:editor",
    "user:code_editor": "user:editor",
    "user:project": "user:current_project",
    "user:hosting": "user:cloud",
    "user:cloud_provider": "user:cloud",
    "user:operating_system": "user:os",
    "user:test": "user:testing",
    "user:tests": "user:testing",
    "user:ci": "user:ci_cd",
    "user:cd": "user:ci_cd",
    "user:css": "user:styling",
    "user:authentication": "user:auth",
    "project:deploy": "project:deployment",
    "project:arch": "project:architecture",
    "project:goal": "project:purpose",
    "project:description": "project:purpose",
    "project:overview": "project:purpose",
    "project:tech_debt": "project:status",
    "project:current_status": "project:status",
    "project:state": "project:status",
}


def normalize_key(raw_key: str | None) -> str | None:
    """Clamp an LLM-generated key to the nearest allowed canonical slot.

    Returns the canonical key if one matches, or ``None`` to treat the fact
    as unkeyed (novel information that won't conflict-supersede anything).
    """
    if not raw_key:
        return None
    key = raw_key.strip().lower().replace(" ", "_")

    # Allow dynamic project keys (modules, workflows, decisions, domain, debt).
    for prefix in _DYNAMIC_KEY_PREFIXES:
        if key.startswith(prefix) and len(key) > len(prefix):
            return key

    # Direct match against the allowed set.
    if key in _ALLOWED_KEYS:
        return key
    # Known alias / common LLM variant.
    if key in _KEY_ALIASES:
        return _KEY_ALIASES[key]
    # Prefix match: e.g. "user:database_primary" -> "user:database".
    for allowed in sorted(_ALLOWED_KEYS, key=len, reverse=True):
        if key.startswith(allowed):
            return allowed
    # No match -> drop the key so the fact is treated as novel/unkeyed.
    logger.debug("Dropping unrecognised LLM key %r (no canonical match).", raw_key)
    return None


def _parse_llm_json(text: str) -> list[ExtractedFact]:
    """Best-effort parse of a model's JSON reply into ExtractedFacts."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    facts: list[ExtractedFact] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("content"):
            continue
        facts.append(
            ExtractedFact(
                content=str(item["content"]).strip(),
                importance=float(item.get("importance", 0.5)),
                key=normalize_key(item.get("key")),
                tags=[str(t) for t in item.get("tags", [])],
            )
        )
    return facts


class OpenAIExtractor:
    """Fact extraction via an OpenAI chat model."""

    def __init__(self, model: str, api_key: str | None, settings: Settings) -> None:
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._settings = settings

    def extract(self, conversation: str | list[dict]) -> list[ExtractedFact]:
        blob = _conversation_to_text(conversation)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": _LLM_PROMPT.format(conversation=blob)}],
            temperature=self._settings.extraction_temperature,
        )
        return _parse_llm_json(resp.choices[0].message.content or "")


class AnthropicExtractor:
    """Fact extraction via an Anthropic (Claude) model."""

    def __init__(self, model: str, api_key: str | None, settings: Settings) -> None:
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._settings = settings

    def extract(self, conversation: str | list[dict]) -> list[ExtractedFact]:
        blob = _conversation_to_text(conversation)
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._settings.extraction_max_tokens,
            messages=[{"role": "user", "content": _LLM_PROMPT.format(conversation=blob)}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return _parse_llm_json(text)


def _conversation_to_text(conversation: str | list[dict]) -> str:
    if isinstance(conversation, list):
        return "\n".join(
            f"{t.get('role', 'user')}: {t.get('content', '')}" for t in conversation
        )
    return str(conversation)


def build_extractor(settings: Settings):
    """Build the configured extractor, falling back to the rule extractor."""
    provider = settings.extraction_provider
    try:
        if provider == "openai":
            return OpenAIExtractor(settings.extraction_model, settings.openai_api_key, settings)
        if provider == "anthropic":
            return AnthropicExtractor(settings.extraction_model, settings.anthropic_api_key, settings)
    except ImportError as exc:
        logger.warning("Extraction provider %r unavailable (%s); using rules.", provider, exc)
    return RuleExtractor()
