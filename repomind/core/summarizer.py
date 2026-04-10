"""Summarization and prompt construction utilities."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from repomind.core.indexer import ChunkMetadata
from repomind.core.retriever import RetrievalResult

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_DEF_CLASS_RE = re.compile(r"^(?:async\s+)?(?:def|class)\s+(\w+)", re.MULTILINE)
_IMPORT_RE = re.compile(r"^(?:from|import)\s+([\w.]+)", re.MULTILINE)
# First triple-quoted string that starts within the first 10 lines of a chunk.
_DOCSTRING_RE = re.compile(r'^\s*"""(.*?)"""', re.DOTALL)

# ---------------------------------------------------------------------------
# Keyword → purpose label map used by both FileSummarizer and FolderSummarizer
# ---------------------------------------------------------------------------

_PURPOSE_MAP: list[tuple[str, str]] = [
    ("auth", "authentication"),
    ("login", "user login"),
    ("token", "token handling"),
    ("jwt", "JWT operations"),
    ("permission", "access control"),
    ("session", "session management"),
    ("db", "database access"),
    ("model", "data models"),
    ("schema", "data schemas / validation"),
    ("migration", "database migrations"),
    ("config", "configuration"),
    ("setting", "application settings"),
    ("cli", "command-line interface"),
    ("route", "HTTP routing"),
    ("router", "request routing"),
    ("endpoint", "API endpoints"),
    ("middleware", "request middleware"),
    ("view", "views / templates"),
    ("handler", "event / request handling"),
    ("test", "tests"),
    ("util", "utilities"),
    ("helper", "helpers"),
    ("service", "business logic"),
    ("client", "external API client"),
    ("cache", "caching"),
    ("embed", "embedding generation"),
    ("index", "indexing"),
    ("retriev", "retrieval"),
    ("summar", "summarization"),
    ("llm", "LLM integration"),
    ("doctor", "diagnostics"),
    ("storage", "data storage"),
    ("faiss", "vector search"),
    ("overview", "project overview"),
    ("output", "output formatting"),
    ("log", "logging"),
    ("chunk", "text chunking"),
    ("scan", "file scanning"),
    ("pars", "parsing"),
    ("serial", "serialization"),
    ("worker", "background workers"),
    ("deploy", "deployment"),
    ("api", "API layer"),
    ("task", "task management"),
    ("queue", "job queue"),
    ("event", "event handling"),
    ("monitor", "monitoring"),
    ("metric", "metrics"),
]


# ---------------------------------------------------------------------------
# ask / explain shared dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AskSummary:
    """Structured output for the ask command."""

    relevant_files: list[str]
    summary: str
    snippets: list[str]
    prompt: str
    llm_provider: str | None = None


# ---------------------------------------------------------------------------
# New: file & folder summary dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FileSummary:
    """Heuristic or LLM-derived summary for a single source file."""

    file_path: str
    purpose: str
    key_symbols: list[str]


@dataclass(slots=True)
class FolderSummary:
    """Aggregated summary for a directory derived from its file summaries."""

    folder: str
    file_count: int
    purpose: str
    dominant_symbols: list[str]


# ---------------------------------------------------------------------------
# FileSummarizer
# ---------------------------------------------------------------------------


class FileSummarizer:
    """Derives per-file summaries from indexed chunks — no additional file I/O."""

    def summarize_heuristic(
        self,
        file_path: str,
        chunks: list[ChunkMetadata],
    ) -> FileSummary:
        """Build a `FileSummary` from chunk text using only fast heuristics."""
        if not chunks:
            return FileSummary(file_path=file_path, purpose="(empty)", key_symbols=[])

        full_text = "\n".join(c.text for c in chunks)

        # Public symbols only (skip private / dunder).
        all_syms = _DEF_CLASS_RE.findall(full_text)
        key_symbols = [s for s in dict.fromkeys(all_syms) if not s.startswith("_")][:8]

        # Purpose: docstring first, then name + symbol heuristics.
        purpose = (
            _docstring_first_line(chunks[0].text)
            or _heuristic_purpose(file_path, full_text)
        )
        return FileSummary(file_path=file_path, purpose=purpose, key_symbols=key_symbols)


# ---------------------------------------------------------------------------
# FolderSummarizer
# ---------------------------------------------------------------------------


class FolderSummarizer:
    """Derives folder-level summaries by aggregating FileSummary objects."""

    def summarize(
        self,
        folder: str,
        file_summaries: list[FileSummary],
    ) -> FolderSummary:
        """Aggregate file summaries into a single folder description."""
        all_symbols: list[str] = []
        concrete_purposes: list[str] = []

        for fs in file_summaries:
            all_symbols.extend(fs.key_symbols)
            if fs.purpose and not fs.purpose.startswith("module:"):
                concrete_purposes.append(fs.purpose)

        dominant_symbols = [s for s, _ in Counter(all_symbols).most_common(6)]

        # Try folder name first, then aggregate file purposes.
        folder_stem = Path(folder.rstrip("/")).name.lower()
        purpose = _heuristic_purpose_from_keywords(folder_stem)

        if not purpose and concrete_purposes:
            top = [p for p, _ in Counter(concrete_purposes).most_common(2)]
            purpose = " and ".join(top)

        if not purpose:
            purpose = f"{folder_stem} module"

        return FolderSummary(
            folder=folder,
            file_count=len(file_summaries),
            purpose=purpose,
            dominant_symbols=dominant_symbols,
        )


# ---------------------------------------------------------------------------
# Original SummaryBuilder (ask + explain)
# ---------------------------------------------------------------------------


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

        # Purpose: module docstring → name heuristic → line count fallback.
        docstring = _docstring_first_line(content)
        purpose_label = docstring or _heuristic_purpose(str(file_path), content)
        purpose = (
            f"{purpose_label} ({len(lines)} lines)"
            if purpose_label
            else f"`{file_path.name}` — {len(lines)} lines of code."
        )

        # Key symbols: public defs/classes, deduplicated, names only.
        all_syms = _DEF_CLASS_RE.findall(content)
        public = [s for s in dict.fromkeys(all_syms) if not s.startswith("_")][:12]
        key_functions = (
            "\n".join(f"- {s}" for s in public)
            or "- (no public functions or classes detected)"
        )

        # Dependencies: non-noise third-party imports, deduplicated by top-level name.
        imports_raw = _IMPORT_RE.findall(content)
        third_party = list(
            dict.fromkeys(
                m.split(".")[0]
                for m in imports_raw
                if m.split(".")[0] not in _NOISE_STDLIB
            )
        )[:8]
        dep_lines = (
            "\n".join(f"- {m}" for m in third_party)
            or "- (no notable external imports)"
        )

        # Flow: simple observation from symbol count + entry-point heuristic.
        entry = next(
            (s for s in public if s in {"main", "run", "start", "app", "cli", "execute"}),
            None,
        )
        flow = (
            f"Entry point: `{entry}()`. "
            if entry
            else ""
        )
        flow += (
            f"Defines {len(public)} public symbol(s). "
            "Trace calls through the exported functions/classes listed above."
        )

        return (
            f"Purpose:\n{purpose}\n\n"
            f"Key Functions / Classes:\n{key_functions}\n\n"
            f"Flow:\n{flow}\n\n"
            f"External Dependencies:\n{dep_lines}"
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Stdlib top-level names to suppress from dependency lists.
_NOISE_STDLIB: frozenset[str] = frozenset(
    {
        "os", "sys", "re", "json", "typing", "pathlib", "dataclasses",
        "collections", "abc", "logging", "hashlib", "io", "time", "math",
        "copy", "itertools", "functools", "contextlib", "enum", "warnings",
        "unittest", "string", "struct", "types", "inspect", "__future__",
        "traceback", "threading", "subprocess", "shutil", "tempfile",
        "datetime", "uuid", "random", "base64", "urllib", "http",
    }
)


def _docstring_first_line(text: str) -> str:
    """Extract the first meaningful line of a module/class docstring."""
    m = _DOCSTRING_RE.search(text)
    if not m:
        return ""
    first = m.group(1).strip().split("\n")[0].strip().rstrip(".")
    return first if first and len(first) < 120 else ""


def _heuristic_purpose(file_path: str, text: str) -> str:
    """Infer a short purpose label from filename + content keywords."""
    stem = Path(file_path).stem.lower()
    combined = stem + " " + text.lower()
    label = _heuristic_purpose_from_keywords(combined)
    return label or f"module: {Path(file_path).stem}"


def _heuristic_purpose_from_keywords(text: str) -> str:
    """Return the first matching purpose label from the keyword map."""
    for keyword, label in _PURPOSE_MAP:
        if keyword in text:
            return label
    return ""
