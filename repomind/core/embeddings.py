"""Embedding providers for RepoMind."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

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

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Install requirements first."
                ) from exc
            self._suppress_third_party_model_logs()
            # Model loading is expensive and should not spam normal CLI output.
            logger.debug("Loading embedding model: %s", self._model_name)
            try:
                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    "Unable to load embedding model. Check internet/model cache and try again."
                ) from exc
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed a list of texts and return normalized float32 vectors."""
        if not texts:
            return EmbeddingResult(vectors=np.empty((0, 0), dtype=np.float32))

        model = self._get_model()
        batches: list[np.ndarray] = []
        try:
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
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                "Embedding generation failed. Verify model availability and input files."
            ) from exc

        merged = np.vstack(batches) if len(batches) > 1 else batches[0]
        return EmbeddingResult(vectors=merged)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string and return a normalized float32 vector."""
        if not text.strip():
            raise EmbeddingError("Query text cannot be empty.")
        model = self._get_model()
        try:
            vector = model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=1,
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                "Query embedding failed. Verify embedding model setup."
            ) from exc
        return np.asarray(vector, dtype=np.float32)

    @staticmethod
    def _iter_batches(texts: Sequence[str], size: int) -> Iterable[Sequence[str]]:
        """Yield fixed-size batches from an input sequence."""
        for index in range(0, len(texts), size):
            yield texts[index : index + size]

    @staticmethod
    def _suppress_third_party_model_logs() -> None:
        """Reduce noisy model initialization output from upstream libraries."""
        try:
            from transformers import logging as transformers_logging

            transformers_logging.set_verbosity_error()
            transformers_logging.disable_progress_bar()
        except Exception:  # noqa: BLE001
            pass
