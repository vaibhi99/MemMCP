"""LLM-based conflict judge for the distillation ambiguity zone.

When two memories land in the similarity grey zone (between the dedup and
conflict thresholds), cosine distance alone cannot distinguish "same topic,
compatible facts" from "same topic, contradictory update".  This module
delegates that decision to a lightweight LLM call.

The judge answers one question: **does the new fact contradict and replace
the existing fact, or can both coexist?**

Two outcomes:
* ``CONTRADICTION`` → the distiller should **supersede** the old memory.
* ``COMPATIBLE``    → the distiller should **create** a new memory alongside.

Design choices:
* Small, structured prompt → minimal latency (~200-400 ms with gpt-4o-mini).
* JSON output with a one-sentence reason for audit trail.
* Graceful fallback: if the LLM call fails, default to "create" (safe side).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from ..config import Settings

logger = logging.getLogger(__name__)

JudgeVerdict = Literal["CONTRADICTION", "COMPATIBLE"]

_JUDGE_PROMPT = """\
You are an expert knowledge consistency judge for a memory system.
Compare an EXISTING memory against a NEW candidate memory and determine \
whether the new fact contradicts/replaces the existing one, or whether both \
can coexist as independent, compatible facts.

EXISTING MEMORY: "{existing_fact}"
NEW MEMORY: "{new_fact}"

Rules:
1. CONTRADICTION: The new fact invalidates, replaces, corrects, or directly \
conflicts with the existing fact. They cannot both be true simultaneously \
about the same subject.
2. COMPATIBLE: Both facts can be simultaneously true. They describe different \
aspects, different subjects, complementary information, or additive details.

Return ONLY valid JSON (no markdown fences):
{{"verdict": "CONTRADICTION" or "COMPATIBLE", "reason": "<one sentence>"}}
"""


@dataclass
class JudgeResult:
    """Outcome of a single LLM conflict judgement."""

    verdict: JudgeVerdict
    reason: str


def _parse_judge_response(text: str) -> JudgeResult:
    """Best-effort parse of the LLM's JSON reply."""
    # Strip markdown fences if the model wraps them anyway.
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON object in the text.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            logger.warning("Judge response was not valid JSON: %s", text[:200])
            return JudgeResult(verdict="COMPATIBLE", reason="parse failure — defaulting to compatible")

    verdict = str(data.get("verdict", "COMPATIBLE")).upper().strip()
    reason = str(data.get("reason", ""))

    if verdict not in ("CONTRADICTION", "COMPATIBLE"):
        logger.warning("Unexpected judge verdict %r; defaulting to COMPATIBLE.", verdict)
        verdict = "COMPATIBLE"

    return JudgeResult(verdict=verdict, reason=reason)  # type: ignore[arg-type]


class ConflictJudge:
    """LLM-backed judge that resolves ambiguous similarity-zone conflicts.

    Instantiated once by :class:`~memmcp.memory_manager.MemoryManager` and
    called from :meth:`Distiller.decide` when a candidate falls inside the
    ``[conflict_threshold, dedup_threshold)`` window.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._provider = settings.extraction_provider
        self._model = settings.judge_model
        self._client: object | None = None

    def _get_openai_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._settings.openai_api_key)
        return self._client

    def _get_anthropic_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        return self._client

    def judge(self, existing_fact: str, new_fact: str) -> JudgeResult:
        """Ask the LLM whether ``new_fact`` contradicts ``existing_fact``.

        Returns a :class:`JudgeResult` with verdict and reason.
        On any failure (network, auth, parsing), returns ``COMPATIBLE``
        so the system errs on the side of keeping both facts.
        """
        prompt = _JUDGE_PROMPT.format(
            existing_fact=existing_fact,
            new_fact=new_fact,
        )
        try:
            if self._provider == "openai":
                return self._call_openai(prompt)
            elif self._provider == "anthropic":
                return self._call_anthropic(prompt)
            else:
                logger.warning(
                    "Unknown judge provider %r; defaulting to COMPATIBLE.",
                    self._provider,
                )
                return JudgeResult(
                    verdict="COMPATIBLE",
                    reason=f"unsupported provider: {self._provider}",
                )
        except Exception:
            logger.exception("LLM judge call failed; defaulting to COMPATIBLE.")
            return JudgeResult(
                verdict="COMPATIBLE",
                reason="LLM call failed — defaulting to compatible",
            )

    def _call_openai(self, prompt: str) -> JudgeResult:
        client = self._get_openai_client()
        resp = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        text = resp.choices[0].message.content or ""
        return _parse_judge_response(text)

    def _call_anthropic(self, prompt: str) -> JudgeResult:
        client = self._get_anthropic_client()
        resp = client.messages.create(
            model=self._model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return _parse_judge_response(text)
