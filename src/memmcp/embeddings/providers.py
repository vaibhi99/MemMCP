"""OpenAI embedding provider."""

from __future__ import annotations

from .base import Embedder, normalize


class OpenAIEmbedder(Embedder):
    """Hosted embeddings via the OpenAI API.

    Requires ``openai`` package and ``OPENAI_API_KEY`` to be set.
    """

    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self.dim = self._DIMS.get(model_name, 1536)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model_name, input=texts)
        return [normalize(item.embedding) for item in resp.data]
