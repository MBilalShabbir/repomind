"""Environment diagnostics for RepoMind."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from repomind.core.config import RepoMindConfig


@dataclass(slots=True)
class DoctorReport:
    """Diagnostic report describing optional and required capabilities."""

    repomind_dir_exists: bool
    faiss_index_present: bool
    metadata_present: bool
    repo_indexed: bool
    local_embeddings_ok: bool
    openai_key_configured: bool
    anthropic_key_configured: bool


class DoctorService:
    """Build diagnostics for local RepoMind setup."""

    def inspect(self, config: RepoMindConfig) -> DoctorReport:
        """Inspect environment and dependency availability."""
        repomind_dir_exists = config.data_dir.exists()
        faiss_index_present = config.index_path.exists()
        metadata_present = config.metadata_path.exists()
        repo_indexed = repomind_dir_exists and faiss_index_present and metadata_present

        embedding_ok = self._has_module("sentence_transformers")

        return DoctorReport(
            repomind_dir_exists=repomind_dir_exists,
            faiss_index_present=faiss_index_present,
            metadata_present=metadata_present,
            repo_indexed=repo_indexed,
            local_embeddings_ok=embedding_ok,
            openai_key_configured=bool(config.openai_api_key),
            anthropic_key_configured=bool(config.anthropic_api_key),
        )

    @staticmethod
    def _has_module(module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None
