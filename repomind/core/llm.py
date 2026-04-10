"""Optional LLM integrations for OpenAI and Anthropic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from repomind.core.config import RepoMindConfig
from repomind.core.indexer import ChunkMetadata

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMResponse:
    """Standardized LLM generation result."""

    provider: str
    text: str


class LLMClient:
    """Base interface for LLM providers."""

    provider: str

    def answer_question(self, question: str, contexts: Iterable[ChunkMetadata]) -> LLMResponse:
        """Generate an answer from context chunks."""
        raise NotImplementedError

    def explain_file(self, file_path: str, content: str) -> LLMResponse:
        """Generate a structured explanation for a file."""
        raise NotImplementedError

    def summarize_codebase(
        self,
        file_infos: list[tuple[str, list[str]]],
        folder_infos: list[tuple[str, int]],
        file_sample: list[str],
    ) -> LLMResponse:
        """Batch-summarize files, folders, and the whole project in one call.

        Args:
            file_infos: [(file_path, [key_symbol, ...]), ...] for important files.
            folder_infos: [(folder_path, file_count), ...] for key modules.
            file_sample: All indexed file paths for project-level context.

        Returns:
            LLMResponse whose text is in the parseable format produced by
            `_build_codebase_summary_prompt`.  Use `parse_codebase_summaries`
            to decode it into a dict.
        """
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI chat client wrapper."""

    provider = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini") -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI SDK is not installed.") from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def answer_question(self, question: str, contexts: Iterable[ChunkMetadata]) -> LLMResponse:
        """Generate a contextual answer using OpenAI."""
        prompt = _build_qa_prompt(question, contexts)
        response = self._client.responses.create(
            model=self._model,
            input=prompt,
            temperature=0.2,
        )
        text = response.output_text.strip()
        return LLMResponse(provider=self.provider, text=text)

    def explain_file(self, file_path: str, content: str) -> LLMResponse:
        """Generate file explanation using OpenAI."""
        prompt = _build_file_explain_prompt(file_path=file_path, content=content)
        response = self._client.responses.create(
            model=self._model,
            input=prompt,
            temperature=0.2,
        )
        return LLMResponse(provider=self.provider, text=response.output_text.strip())

    def summarize_codebase(
        self,
        file_infos: list[tuple[str, list[str]]],
        folder_infos: list[tuple[str, int]],
        file_sample: list[str],
    ) -> LLMResponse:
        """Batch-summarize files, folders, and project using OpenAI."""
        prompt = _build_codebase_summary_prompt(file_infos, folder_infos, file_sample)
        response = self._client.responses.create(
            model=self._model,
            input=prompt,
            temperature=0.2,
        )
        return LLMResponse(provider=self.provider, text=response.output_text.strip())


class AnthropicClient(LLMClient):
    """Anthropic messages API wrapper."""

    provider = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-latest") -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Anthropic SDK is not installed.") from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def answer_question(self, question: str, contexts: Iterable[ChunkMetadata]) -> LLMResponse:
        """Generate a contextual answer using Anthropic."""
        prompt = _build_qa_prompt(question, contexts)
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=800,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _anthropic_text(msg)
        return LLMResponse(provider=self.provider, text=text)

    def explain_file(self, file_path: str, content: str) -> LLMResponse:
        """Generate file explanation using Anthropic."""
        prompt = _build_file_explain_prompt(file_path=file_path, content=content)
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=800,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(provider=self.provider, text=_anthropic_text(msg))

    def summarize_codebase(
        self,
        file_infos: list[tuple[str, list[str]]],
        folder_infos: list[tuple[str, int]],
        file_sample: list[str],
    ) -> LLMResponse:
        """Batch-summarize files, folders, and project using Anthropic."""
        prompt = _build_codebase_summary_prompt(file_infos, folder_infos, file_sample)
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=600,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(provider=self.provider, text=_anthropic_text(msg))


class LLMRouter:
    """Selects the best available LLM provider based on config and environment."""

    def __init__(self, config: RepoMindConfig) -> None:
        """Initialize router with runtime configuration."""
        self._config = config

    def resolve(self) -> LLMClient | None:
        """Return an initialized LLM client if one is available."""
        # Automatic premium detection order:
        # 1) Anthropic when ANTHROPIC_API_KEY is configured
        # 2) OpenAI when OPENAI_API_KEY is configured
        return self._try_anthropic() or self._try_openai()

    def _try_openai(self) -> LLMClient | None:
        key = self._config.openai_api_key
        if not key:
            return None
        try:
            return OpenAIClient(api_key=key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI unavailable: %s", exc)
            return None

    def _try_anthropic(self) -> LLMClient | None:
        key = self._config.anthropic_api_key
        if not key:
            return None
        try:
            return AnthropicClient(api_key=key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Anthropic unavailable: %s", exc)
            return None


def _build_qa_prompt(question: str, contexts: Iterable[ChunkMetadata]) -> str:
    """Build a grounded QA prompt."""
    parts = [
        "You are RepoMind, assisting with codebase analysis.",
        "Use only the provided snippets when possible, and cite file paths.",
        f"Question: {question}",
        "Context snippets:",
    ]
    for idx, item in enumerate(contexts, start=1):
        parts.append(
            f"[{idx}] {item.file_path}:{item.start_line}-{item.end_line}\n{item.text[:1800]}"
        )
    parts.append("Provide a clear, implementation-focused answer.")
    return "\n\n".join(parts)


def _build_file_explain_prompt(file_path: str, content: str) -> str:
    """Build a prompt to explain file purpose and flow."""
    return (
        "You are RepoMind. Explain the given file in a structured way.\n\n"
        "Return sections: Purpose, Key Functions/Classes, Control Flow, Risks/Notes.\n\n"
        f"File: {file_path}\n\n"
        f"Content:\n{content[:18000]}"
    )


def _build_codebase_summary_prompt(
    file_infos: list[tuple[str, list[str]]],
    folder_infos: list[tuple[str, int]],
    file_sample: list[str],
) -> str:
    """Build a structured batch-summary prompt.

    The response must follow this exact line format so it can be parsed by
    `parse_codebase_summaries`:

        FILE <path>: <one-line description, ≤12 words>
        FOLDER <path>: <one-line description, ≤10 words>
        PROJECT: <1-2 sentence project summary>
    """
    file_lines = "\n".join(
        f"FILE {path} | symbols: {', '.join(syms) or 'none'}"
        for path, syms in file_infos
    )
    folder_lines = "\n".join(
        f"FOLDER {folder} | {count} file(s)"
        for folder, count in folder_infos
    )
    sample_text = "\n".join(file_sample[:60])
    return (
        "You are RepoMind. Summarize each codebase component in plain English.\n"
        "Respond ONLY using these exact line prefixes — one entry per line:\n"
        "  FILE <path>: <description up to 12 words>\n"
        "  FOLDER <path>: <description up to 10 words>\n"
        "  PROJECT: <1-2 sentence summary>\n\n"
        "Files to summarize:\n"
        f"{file_lines or '(none)'}\n\n"
        "Folders to summarize:\n"
        f"{folder_lines or '(none)'}\n\n"
        "All indexed files (context):\n"
        f"{sample_text}"
    )


def parse_codebase_summaries(text: str) -> dict[str, str]:
    """Decode the structured LLM response from `_build_codebase_summary_prompt`.

    Returns a dict with keys:
      ``"file:<path>"``   — per-file one-liners
      ``"folder:<path>"`` — per-folder one-liners
      ``"project"``       — overall project summary
    """
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        if line.startswith("FILE "):
            rest = line[5:]
            path, _, desc = rest.partition(":")
            if desc.strip():
                result[f"file:{path.strip()}"] = desc.strip()
        elif line.startswith("FOLDER "):
            rest = line[7:]
            path, _, desc = rest.partition(":")
            if desc.strip():
                result[f"folder:{path.strip()}"] = desc.strip()
        elif line.startswith("PROJECT:"):
            desc = line[8:].strip()
            if desc:
                result["project"] = desc
    return result


def _anthropic_text(message: object) -> str:
    """Extract text from Anthropic SDK response object."""
    parts: list[str] = []
    for block in getattr(message, "content", []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()
