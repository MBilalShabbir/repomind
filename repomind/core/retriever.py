"""Retrieval service for querying indexed code chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from repomind.core.config import RepoMindConfig
from repomind.core.embeddings import Embedder
from repomind.core.indexer import ChunkMetadata

NO_INDEX_MESSAGE = "No RepoMind index found. Run 'repomind index' in this repo first."


@dataclass(slots=True)
class RetrievalResult:
    """Single retrieved chunk with score."""

    score: float
    metadata: ChunkMetadata


class CodeRetriever:
    """Loads FAISS index and metadata, then returns top-k relevant chunks."""

    def __init__(self, config: RepoMindConfig, embedder: Embedder) -> None:
        """Initialize retriever dependencies."""
        self._config = config
        self._embedder = embedder

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k chunks relevant to a user question."""
        faiss = _require_faiss()
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        index = self._load_index(self._config.index_path)
        metadata = self._load_metadata(self._config.metadata_path)

        query_vector = self._embedder.embed_query(question).reshape(1, -1)
        faiss.normalize_L2(query_vector)

        max_k = min(top_k, len(metadata))
        scores, idxs = index.search(query_vector.astype(np.float32), max_k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], idxs[0], strict=False):
            if idx < 0:
                continue
            meta = metadata[int(idx)]
            results.append(RetrievalResult(score=float(score), metadata=meta))
        return results

    @staticmethod
    def _load_index(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(NO_INDEX_MESSAGE)
        faiss = _require_faiss()
        return faiss.read_index(str(path))

    @staticmethod
    def _load_metadata(path: Path) -> list[ChunkMetadata]:
        if not path.exists():
            raise FileNotFoundError(NO_INDEX_MESSAGE)

        rows: list[ChunkMetadata] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                rows.append(ChunkMetadata(**payload))
        if not rows:
            raise RuntimeError("Metadata is empty. Rebuild index with 'repomind index'.")
        return rows


def _require_faiss() -> Any:
    """Import faiss lazily to support `repomind doctor` without hard dependency."""
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed. Install dependencies before running retrieval."
        ) from exc
    return faiss
