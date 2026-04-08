"""Embedding providers for RepoMind."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when an embedding operation fails."""


@dataclass(slots=True)
class EmbeddingResult:
    """Container for embedding vectors."""

    vectors: np.ndarray


class Embedder:
    """Abstract embedding interface."""

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed multiple documents into vector space."""
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """SentenceTransformer-based local embedding provider."""

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        """Initialize embedder.

        Args:
            model_name: Hugging Face model id.
            batch_size: Number of texts to encode per embedding batch.
        """
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Install requirements first."
                ) from exc
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed a list of texts and return normalized float32 vectors."""
        if not texts:
            return EmbeddingResult(vectors=np.empty((0, 0), dtype=np.float32))

        model = self._get_model()
        batches: list[np.ndarray] = []
        for batch in self._iter_batches(texts, self._batch_size):
            vectors = model.encode(
                list(batch),
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=self._batch_size,
            )
            if not isinstance(vectors, np.ndarray):
                vectors = np.asarray(vectors)
            batches.append(vectors.astype(np.float32))

        merged = np.vstack(batches) if len(batches) > 1 else batches[0]
        return EmbeddingResult(vectors=merged)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string and return a normalized float32 vector."""
        if not text.strip():
            raise EmbeddingError("Query text cannot be empty.")
        model = self._get_model()
        vector = model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=1,
        )[0]
        return np.asarray(vector, dtype=np.float32)

    @staticmethod
    def _iter_batches(texts: Sequence[str], size: int) -> Iterable[Sequence[str]]:
        """Yield fixed-size batches from an input sequence."""
        for index in range(0, len(texts), size):
            yield texts[index : index + size]
