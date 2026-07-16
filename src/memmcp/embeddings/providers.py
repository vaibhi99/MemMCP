"""Optional, higher-quality embedding providers (lazy-imported).

These are only imported when selected, so the base install stays lightweight.
"""

from __future__ import annotations

from .base import Embedder, normalize


class SentenceTransformerEmbedder(Embedder):
    """Local semantic embeddings via ``sentence-transformers``.

    Install with ``pip install memmcp[local-embed]``.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run `pip install memmcp[local-embed]` or set "
                "MEMMCP_EMBEDDING_PROVIDER=hash."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]


class OpenAIEmbedder(Embedder):
    """Hosted embeddings via the OpenAI API.

    Install with ``pip install memmcp[openai]`` and set ``OPENAI_API_KEY``.
    """

    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "openai is not installed. Run `pip install memmcp[openai]` "
                "or set MEMMCP_EMBEDDING_PROVIDER=hash."
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self.dim = self._DIMS.get(model_name, 1536)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model_name, input=texts)
        return [normalize(item.embedding) for item in resp.data]
