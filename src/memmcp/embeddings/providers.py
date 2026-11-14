"""Gemini embedding provider using LangChain."""

from __future__ import annotations

from .base import Embedder, normalize


class GeminiEmbedder(Embedder):
    """Hosted embeddings via Google Generative AI using LangChain.

    Requires ``langchain-google-genai`` and ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
    """

    def __init__(
        self,
        model_name: str = "models/text-embedding-004",
        api_key: str | None = None,
        dim: int = 768,
    ) -> None:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is not installed. Run `pip install langchain-google-genai`."
            ) from exc

        self._model_name = model_name
        self.dim = dim
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )

    def embed(self, text: str) -> list[float]:
        vec = self._embeddings.embed_query(text)
        return normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._embeddings.embed_documents(texts)
        return [normalize(v) for v in vecs]
