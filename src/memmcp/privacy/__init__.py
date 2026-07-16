"""Privacy & governance for cross-tool memory.

Sharing memory across tools is exactly where secrets leak. Two safeguards:

* :mod:`pii` — detect and (per policy) block, redact, or flag PII/secrets
  before anything is stored or shared.
* :mod:`audit` — an append-only log of every read/write, so it is always
  auditable what context was stored and what was surfaced to which tool.
"""

from .audit import AuditLog
from .pii import PIIScanner, PIIResult

__all__ = ["AuditLog", "PIIScanner", "PIIResult"]
