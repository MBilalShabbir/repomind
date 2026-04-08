"""Summarization and prompt construction utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from repomind.core.indexer import ChunkMetadata
from repomind.core.retriever import RetrievalResult


@dataclass(slots=True)
class AskSummary:
    """Structured output for the ask command."""

    relevant_files: list[str]
    summary: str
    snippets: list[str]
    prompt: str
    llm_provider: str | None = None


class SummaryBuilder:
    """Builds user-facing summaries from retrieval outputs."""

    def build_for_question(
        self,
        question: str,
        results: Iterable[RetrievalResult],
        ai_answer: str | None = None,
        llm_provider: str | None = None,
    ) -> AskSummary:
        """Build summary for `ask` command output."""
        rows = list(results)
        if not rows:
            return AskSummary(
                relevant_files=[],
                summary="No relevant context found in the current index.",
                snippets=[],
                prompt=self._prompt_template(question, []),
                llm_provider=llm_provider,
            )

        files = [row.metadata.file_path for row in rows]
        ranked_files = [item for item, _ in Counter(files).most_common()]
        snippets = [self._format_snippet(row.metadata) for row in rows]

        fallback_summary = (
            "Top related files: "
            + ", ".join(ranked_files[:5])
            + ". Review snippets and refine the question for higher precision if needed."
        )

        return AskSummary(
            relevant_files=ranked_files,
            summary=ai_answer or fallback_summary,
            snippets=snippets,
            prompt=self._prompt_template(question, [r.metadata for r in rows]),
            llm_provider=llm_provider,
        )

    def explain_file_locally(self, file_path: Path, content: str) -> str:
        """Create a structured local explanation without external LLMs."""
        lines = content.splitlines()
        imports = [ln.strip() for ln in lines if ln.strip().startswith(("import ", "from "))][:10]
        defs = [ln.strip() for ln in lines if ln.strip().startswith(("def ", "class "))][:20]

        purpose = (
            f"This file appears to define behavior for `{file_path.name}` with "
            f"{len(lines)} lines of code."
        )
        key_functions = "\n".join(f"- {item}" for item in defs) or "- No obvious public functions/classes detected"
        flow = (
            "Execution likely follows top-level declarations and callable entry points. "
            "Inspect function/class definitions below for concrete behavior."
        )
        dependency_notes = "\n".join(f"- {item}" for item in imports) or "- No imports detected"

        return (
            "Purpose:\n"
            f"{purpose}\n\n"
            "Key Functions/Classes:\n"
            f"{key_functions}\n\n"
            "Flow:\n"
            f"{flow}\n\n"
            "Dependencies:\n"
            f"{dependency_notes}"
        )

    @staticmethod
    def _format_snippet(metadata: ChunkMetadata) -> str:
        excerpt = metadata.text.strip().replace("\n", "\n    ")[:1200]
        return (
            f"{metadata.file_path}:{metadata.start_line}-{metadata.end_line}\n"
            f"    {excerpt}"
        )

    @staticmethod
    def _prompt_template(question: str, contexts: list[ChunkMetadata]) -> str:
        context_block = "\n\n".join(
            f"File: {item.file_path}:{item.start_line}-{item.end_line}\n{item.text[:1200]}"
            for item in contexts
        )
        return (
            "You are assisting with a codebase question.\n"
            f"Question: {question}\n\n"
            "Use the following context snippets to produce a precise, actionable answer:\n\n"
            f"{context_block}\n\n"
            "Include file path references in your answer."
        )
