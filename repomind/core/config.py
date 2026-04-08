"""Configuration management for RepoMind."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CONFIG_FILE = ".repomind/config.toml"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(slots=True)
class RepoMindConfig:
    """Runtime settings resolved from environment variables and config files."""

    project_root: Path
    data_dir: Path
    index_path: Path
    metadata_path: Path
    embedding_model: str
    openai_api_key: str | None
    anthropic_api_key: str | None
    llm_provider_preference: str | None


class ConfigLoader:
    """Loads RepoMind configuration from file and environment variables."""

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialize the loader.

        Args:
            project_root: Project root directory. Defaults to current working directory.
        """
        self._project_root = (project_root or Path.cwd()).resolve()

    def load(self) -> RepoMindConfig:
        """Load and merge configuration values.

        Environment variables override values from config file.

        Returns:
            Merged configuration object.
        """
        file_config = self._load_file_config(self._project_root / DEFAULT_CONFIG_FILE)

        # RepoMind storage is always repository-scoped to avoid cross-repo contamination.
        data_dir = (self._project_root / ".repomind").resolve()

        embedding_model = str(
            self._resolve_value(
                env_key="REPOMIND_EMBEDDING_MODEL",
                file_config=file_config,
                file_key="embedding_model",
                default=DEFAULT_EMBEDDING_MODEL,
            )
        )

        llm_preference = self._resolve_value(
            env_key="REPOMIND_LLM_PROVIDER",
            file_config=file_config,
            file_key="llm_provider",
            default=None,
        )

        config = RepoMindConfig(
            project_root=self._project_root,
            data_dir=data_dir,
            index_path=data_dir / "index.faiss",
            metadata_path=data_dir / "metadata.jsonl",
            embedding_model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            llm_provider_preference=str(llm_preference) if llm_preference else None,
        )
        return config

    @staticmethod
    def _resolve_value(
        *,
        env_key: str,
        file_config: dict[str, Any],
        file_key: str,
        default: Any,
    ) -> Any:
        """Resolve a setting from environment, file, then default."""
        env_value = os.getenv(env_key)
        if env_value is not None and env_value != "":
            return env_value
        if file_key in file_config:
            return file_config[file_key]
        return default

    @staticmethod
    def _load_file_config(path: Path) -> dict[str, Any]:
        """Read the TOML config file if present."""
        if not path.exists():
            return {}
        with path.open("rb") as fp:
            return tomllib.load(fp)
