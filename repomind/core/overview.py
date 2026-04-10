"""Heuristic codebase overview from indexed metadata."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from repomind.core.indexer import ChunkMetadata

# File names that commonly serve as entry points or configuration anchors.
_IMPORTANT_NAMES: frozenset[str] = frozenset(
    {
        "main.py",
        "app.py",
        "server.py",
        "index.py",
        "run.py",
        "manage.py",
        "wsgi.py",
        "asgi.py",
        "__main__.py",
        "settings.py",
        "config.py",
        "configuration.py",
        "cli.py",
        "router.py",
        "routes.py",
        "urls.py",
        "setup.py",
        "pyproject.toml",
        "package.json",
        "Makefile",
        "Dockerfile",
    }
)

# Stdlib and very common names that add no domain signal.
_NOISE_IMPORTS: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "re",
        "json",
        "typing",
        "pathlib",
        "dataclasses",
        "collections",
        "abc",
        "logging",
        "hashlib",
        "io",
        "time",
        "math",
        "copy",
        "itertools",
        "functools",
        "contextlib",
        "enum",
        "warnings",
        "unittest",
        "string",
        "struct",
        "types",
        "inspect",
        "__future__",
    }
)

_IMPORT_RE = re.compile(r"^(?:from|import)\s+([\w.]+)", re.MULTILINE)


@dataclass(slots=True)
class OverviewResult:
    """Structured output from heuristic codebase analysis."""

    key_modules: list[str]
    important_files: list[str]
    heuristic_summary: str
    total_files: int
    total_chunks: int


class OverviewAnalyzer:
    """Derives key modules, important files, and a brief summary from FAISS metadata."""

    def __init__(self, metadata_path: Path) -> None:
        self._metadata_path = metadata_path

    def analyze(self) -> OverviewResult:
        rows = self._load_metadata()

        file_paths = [row.file_path for row in rows]
        unique_files = sorted(set(file_paths))
        total_files = len(unique_files)
        total_chunks = len(rows)

        key_modules = self._find_key_modules(unique_files)
        important_files = self._find_important_files(unique_files, file_paths)
        summary = self._build_summary(
            unique_files=unique_files,
            key_modules=key_modules,
            important_files=important_files,
            rows=rows,
        )

        return OverviewResult(
            key_modules=key_modules,
            important_files=important_files,
            heuristic_summary=summary,
            total_files=total_files,
            total_chunks=total_chunks,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_key_modules(unique_files: list[str]) -> list[str]:
        """Return directories ranked by file count, skipping trivial roots."""
        dir_counts: Counter[str] = Counter()
        for fp in unique_files:
            parent = str(Path(fp).parent)
            if parent in {".", ""}:
                continue
            # Credit every ancestor so a deep file also registers its top dir.
            parts = Path(fp).parts
            for depth in range(1, len(parts)):
                dir_counts["/".join(parts[:depth])] += 1

        if not dir_counts:
            return []

        # Keep only dirs with at least 2 files; prefer deeper paths over
        # parent paths that would dominate just because they own everything below.
        top_count = dir_counts.most_common(1)[0][1]
        threshold = max(2, top_count // 6)

        candidates = [d for d, c in dir_counts.most_common() if c >= threshold]

        # Remove a parent if every child of it is already listed.
        pruned: list[str] = []
        for candidate in candidates:
            # If any retained entry starts with this candidate + "/", skip it.
            dominated = any(
                kept.startswith(candidate + "/") for kept in pruned
            )
            if not dominated:
                pruned.append(candidate)
            if len(pruned) >= 8:
                break

        return [d + "/" for d in pruned]

    @staticmethod
    def _find_important_files(
        unique_files: list[str],
        all_file_paths: list[str],
    ) -> list[str]:
        """Return heuristically important files (entry points + dense chunks)."""
        by_name: list[str] = [
            fp for fp in unique_files if Path(fp).name in _IMPORTANT_NAMES
        ]

        chunk_counts: Counter[str] = Counter(all_file_paths)
        already = set(by_name)
        by_density = [
            fp
            for fp, _ in chunk_counts.most_common(10)
            if fp not in already
        ]

        combined = by_name + by_density
        return combined[:8]

    @staticmethod
    def _build_summary(
        unique_files: list[str],
        key_modules: list[str],
        important_files: list[str],
        rows: list[ChunkMetadata],
    ) -> str:
        total = len(unique_files)
        module_names = [m.rstrip("/") for m in key_modules[:5]]

        # Collect third-party library names from import lines.
        import_counts: Counter[str] = Counter()
        for row in rows:
            for match in _IMPORT_RE.finditer(row.text):
                top = match.group(1).split(".")[0]
                if top not in _NOISE_IMPORTS and not top.startswith("_"):
                    import_counts[top] += 1
        top_libs = [lib for lib, _ in import_counts.most_common(8)]

        parts: list[str] = [
            f"This project has {total} indexed source files"
            + (f" across {len(module_names)} top-level modules." if module_names else ".")
        ]
        if module_names:
            parts.append(f"Core areas: {', '.join(module_names)}.")
        if top_libs:
            parts.append(f"Key dependencies: {', '.join(top_libs[:5])}.")
        if important_files:
            entry_names = [Path(f).name for f in important_files[:3]]
            parts.append(f"Likely entry points: {', '.join(entry_names)}.")

        return " ".join(parts)

    def _load_metadata(self) -> list[ChunkMetadata]:
        rows: list[ChunkMetadata] = []
        with self._metadata_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(ChunkMetadata(**json.loads(stripped)))
        return rows
