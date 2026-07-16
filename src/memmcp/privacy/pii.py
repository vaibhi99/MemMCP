"""Regex-based PII / secret detection and redaction.

Deliberately dependency-free and conservative: it catches the high-risk,
high-precision categories (emails, phones, cards, SSNs, IPs, cloud keys,
private keys, generic tokens). For heavier NLP-based PII (names, addresses)
plug in a model behind the same :class:`PIIScanner` interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PIIResult:
    """Outcome of scanning a piece of text."""

    types: list[str] = field(default_factory=list)
    redacted: str = ""
    has_pii: bool = False


# (type, compiled regex, placeholder). Order matters: match specific/long
# patterns (keys, cards) before generic ones (numbers) to avoid mis-labelling.
_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]+?-----END[^-]+-----"),
        "[REDACTED_PRIVATE_KEY]",
    ),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_API_KEY]"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
        "[REDACTED_JWT]",
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "[REDACTED_CARD]",
    ),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "phone",
        re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}(?!\w)"),
        "[REDACTED_PHONE]",
    ),
    (
        "ip_address",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
        "[REDACTED_IP]",
    ),
]


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum, to avoid flagging arbitrary long digit runs as cards."""
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    checksum = 0
    parity = len(nums) % 2
    for i, n in enumerate(nums):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        checksum += n
    return checksum % 10 == 0


class PIIScanner:
    """Scan text for PII and produce a redacted copy."""

    def scan(self, text: str) -> PIIResult:
        found: list[str] = []
        redacted = text
        for pii_type, pattern, placeholder in _PATTERNS:
            def _replace(match: re.Match[str]) -> str:
                value = match.group(0)
                if pii_type == "credit_card" and not _luhn_ok(value):
                    return value  # not a real card number; leave untouched
                if pii_type not in found:
                    found.append(pii_type)
                return placeholder

            redacted = pattern.sub(_replace, redacted)

        return PIIResult(types=found, redacted=redacted, has_pii=bool(found))
