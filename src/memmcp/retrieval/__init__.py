"""Retrieval ranking + token-budgeted context packing.

Vector search alone answers *"what is textually similar?"*. Good memory also
answers *"what is worth surfacing right now?"* — which blends three signals:

* **relevance** — cosine similarity to the query (from the vector store).
* **recency** — an exponential decay on time since the memory was last used,
  parameterised by a half-life. Freshly-used facts float up.
* **importance** — the caller/model's estimate of durable value, 0..1.

The final score is a weighted sum with configurable weights. Then, because
context windows overflow (the "lost in the middle" problem), we greedily pack
the highest-scoring memories into a token budget instead of dumping everything.
"""

from .ranker import Ranker, estimate_tokens

__all__ = ["Ranker", "estimate_tokens"]
