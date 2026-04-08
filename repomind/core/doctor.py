"""Environment diagnostics for RepoMind."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from repomind.core.config import RepoMindConfig


@dataclass(slots=True)
class DoctorReport:
    """Diagnostic report describing optional and required capabilities."""

    local_embeddings_ok: bool
    faiss_ok: bool
    openai_key_configured: bool
    anthropic_key_configured: bool
    openai_sdk_ok: bool
    anthropic_sdk_ok: bool


class DoctorService:
    """Build diagnostics for local RepoMind setup."""

    def inspect(self, config: RepoMindConfig) -> DoctorReport:
        """Inspect environment and dependency availability."""
        embedding_ok = self._has_module("sentence_transformers")
        faiss_ok = self._has_module("faiss")
        openai_sdk_ok = self._has_module("openai")
        anthropic_sdk_ok = self._has_module("anthropic")

        return DoctorReport(
            local_embeddings_ok=embedding_ok,
            faiss_ok=faiss_ok,
            openai_key_configured=bool(config.openai_api_key),
            anthropic_key_configured=bool(config.anthropic_api_key),
            openai_sdk_ok=openai_sdk_ok,
            anthropic_sdk_ok=anthropic_sdk_ok,
        )

    @staticmethod
    def _has_module(module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None
