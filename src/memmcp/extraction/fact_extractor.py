"""Fact extraction: conversation -> candidate distilled facts.

Two providers share one interface:

* :class:`RuleExtractor` (default, offline) — pattern-based extraction of
  first-person statements about identity, stack choices, preferences and
  dislikes, plus a small domain taxonomy that assigns a *conflict key* (e.g.
  "uses Postgres" and "uses MySQL" both map to ``user:database`` so the later
  one can supersede the earlier).
* LLM extractors (:func:`build_extractor` returns these when configured) —
  prompt an OpenAI/Anthropic model with a JSON-validated template.

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
        "redis", "cassandra", "dynamodb", "cosmosdb", "mariadb", "oracle",
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
_LLM_PROMPT = """You extract durable, self-contained FACTS about the user or \
their project from a conversation. Ignore chit-chat, dead ends, and corrected \
mistakes — keep only conclusions that would be useful to remember later.

Return STRICT JSON: a list of objects with keys:
  - "content": one concise fact sentence (third person, self-contained)
  - "importance": float 0..1 (how durable/valuable)
  - "key": short canonical slot like "user:database" or null
  - "tags": list of short strings

Conversation:
---
{conversation}
---
JSON:"""


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
                key=item.get("key") or None,
                tags=[str(t) for t in item.get("tags", [])],
            )
        )
    return facts


class OpenAIExtractor:
    """Fact extraction via an OpenAI chat model."""

    def __init__(self, model: str, api_key: str | None) -> None:
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def extract(self, conversation: str | list[dict]) -> list[ExtractedFact]:
        blob = _conversation_to_text(conversation)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": _LLM_PROMPT.format(conversation=blob)}],
            temperature=0,
        )
        return _parse_llm_json(resp.choices[0].message.content or "")


class AnthropicExtractor:
    """Fact extraction via an Anthropic (Claude) model."""

    def __init__(self, model: str, api_key: str | None) -> None:
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def extract(self, conversation: str | list[dict]) -> list[ExtractedFact]:
        blob = _conversation_to_text(conversation)
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
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
            return OpenAIExtractor(settings.extraction_model, settings.openai_api_key)
        if provider == "anthropic":
            return AnthropicExtractor(settings.extraction_model, settings.anthropic_api_key)
    except ImportError as exc:
        logger.warning("Extraction provider %r unavailable (%s); using rules.", provider, exc)
    return RuleExtractor()
