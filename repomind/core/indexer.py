"""Codebase indexing service using FAISS."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from repomind.core.config import RepoMindConfig
from repomind.core.embeddings import Embedder
from repomind.core.faiss_store import require_faiss

logger = logging.getLogger(__name__)

IGNORED_DIRECTORIES = {".git", "node_modules", "venv", "__pycache__", ".repomind"}
IGNORED_DIRECTORIES |= {
    ".venv",
    "dist",
    "build",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
}
IGNORED_FILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
}

ProgressCallback = Callable[["IndexProgress"], None]


@dataclass(slots=True)
class ChunkMetadata:
    """Metadata for a text chunk stored in the vector index."""

    chunk_id: int
    file_path: str
    start_line: int
    end_line: int
    text: str


@dataclass(slots=True)
class FileFingerprint:
    """Stable fingerprint for file-change detection."""

    sha256: str
    size: int
    mtime_ns: int


@dataclass(slots=True)
class IndexProgress:
    """Progress signal emitted during indexing."""

    phase: str
    message: str
    processed_files: int = 0
    total_files: int = 0
    processed_chunks: int = 0


@dataclass(slots=True)
class IndexStats:
    """Summary of indexing output."""

    files_indexed: int
    chunks_indexed: int
    index_path: Path
    metadata_path: Path
    updated_files: int = 0
    removed_files: int = 0
    scanned_files: int = 0
    mode: str = "full"


class FileScanner:
    """Scans source files recursively while skipping known noisy directories."""

    def __init__(self, root: Path, max_file_size_bytes: int) -> None:
        """Initialize scanner.

        Args:
            root: Root directory to scan.
            max_file_size_bytes: Maximum file size to index.
        """
        self._root = root.resolve()
        self._max_file_size_bytes = max_file_size_bytes

    def list_files(self) -> list[Path]:
        """Return indexable files under root directory."""
        files: list[Path] = []
        for path in self._root.rglob("*"):
            if path.is_dir():
                continue
            if any(part in IGNORED_DIRECTORIES for part in path.parts):
                continue
            if any(part.endswith(".egg-info") for part in path.parts):
                continue
            if path.name in IGNORED_FILE_NAMES:
                continue
            if path.name.endswith((".min.js", ".min.css")):
                continue
            if path.name.startswith("."):
                continue
            if not self._is_text_file(path, self._max_file_size_bytes):
                continue
            files.append(path)
        return files

    @staticmethod
    def _is_text_file(path: Path, max_file_size_bytes: int) -> bool:
        try:
            if path.stat().st_size > max_file_size_bytes:
                return False
            with path.open("rb") as fp:
                sample = fp.read(2048)
            return b"\x00" not in sample
        except OSError:
            return False


class CodeChunker:
    """Splits file content into line-aware overlapping chunks."""

    def __init__(self, chunk_size: int = 1200, overlap: int = 200) -> None:
        """Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters.
            overlap: Overlap in characters between consecutive chunks.
        """
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def split(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into chunks preserving line ranges."""
        raw_lines = text.splitlines()
        if not raw_lines:
            return []

        chunks: list[tuple[str, int, int]] = []
        buffer: list[str] = []
        buffer_chars = 0
        start_line = 1

        for line in raw_lines:
            line_len = len(line) + 1
            if buffer and buffer_chars + line_len > self._chunk_size:
                chunk_text = "\n".join(buffer)
                end_line = start_line + len(buffer) - 1
                if chunk_text.strip():
                    chunks.append((chunk_text, start_line, end_line))

                overlap_lines = self._tail_lines_for_overlap(buffer)
                buffer = overlap_lines
                buffer_chars = self._estimated_chars(buffer)
                start_line = max(1, end_line - len(buffer) + 1)

            buffer.append(line)
            buffer_chars += line_len

        if buffer:
            chunk_text = "\n".join(buffer)
            end_line = start_line + len(buffer) - 1
            if chunk_text.strip():
                chunks.append((chunk_text, start_line, end_line))

        return chunks

    def _tail_lines_for_overlap(self, lines: list[str]) -> list[str]:
        """Return a suffix of lines that approximates overlap character budget."""
        selected: list[str] = []
        consumed = 0
        for line in reversed(lines):
            line_len = len(line) + 1
            if selected and consumed + line_len > self._overlap:
                break
            selected.append(line)
            consumed += line_len
            if consumed >= self._overlap:
                break
        selected.reverse()
        return selected

    @staticmethod
    def _estimated_chars(lines: list[str]) -> int:
        """Estimate character count for a line list."""
        if not lines:
            return 0
        return sum(len(line) + 1 for line in lines)


class CodeIndexer:
    """Coordinates scanning, chunking, embeddings, and FAISS persistence."""

    def __init__(self, config: RepoMindConfig, embedder: Embedder) -> None:
        """Initialize indexer dependencies."""
        self._config = config
        self._embedder = embedder
        self._chunker = CodeChunker()
        self._manifest_path = self._config.data_dir / "files.json"

    def index(
        self,
        root: Path,
        update: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexStats:
        """Build and persist FAISS index from the given root directory."""
        root = root.resolve()
        self._validate_root(root)
        if update and root != self._config.project_root:
            raise RuntimeError(
                "--update requires indexing from the current repository root ('.')."
            )
        self._config.data_dir.mkdir(parents=True, exist_ok=True)

        if update and self._has_existing_index():
            return self._incremental_index(root, progress_callback)

        mode = "full"
        if update:
            self._emit(
                progress_callback,
                IndexProgress(
                    phase="prepare",
                    message="No previous incremental state found. Running full indexing.",
                ),
            )
        stats = self._full_index(root, progress_callback)
        stats.mode = mode
        return stats

    def _full_index(
        self,
        root: Path,
        progress_callback: ProgressCallback | None,
    ) -> IndexStats:
        """Run a full index build from source files."""
        faiss = require_faiss("indexing")

        files = FileScanner(root, self._config.max_file_size_bytes).list_files()
        self._emit(
            progress_callback,
            IndexProgress(
                phase="scan",
                message=f"Scanning {len(files)} files...",
                total_files=len(files),
            ),
        )

        documents: list[str] = []
        metadata_rows: list[ChunkMetadata] = []
        manifest: dict[str, FileFingerprint] = {}

        for idx, path in enumerate(files, start=1):
            content = _read_file_content(path)
            if content is None:
                continue

            rel_path = str(path.relative_to(self._config.project_root))
            manifest[rel_path] = _fingerprint_file(path=path, content=content)

            for chunk_text, start_line, end_line in self._chunker.split(content):
                documents.append(chunk_text)
                metadata_rows.append(
                    ChunkMetadata(
                        chunk_id=len(metadata_rows),
                        file_path=rel_path,
                        start_line=start_line,
                        end_line=end_line,
                        text=chunk_text,
                    )
                )

            if idx == 1 or idx % 25 == 0 or idx == len(files):
                self._emit(
                    progress_callback,
                    IndexProgress(
                        phase="scan",
                        message="Chunking files",
                        processed_files=idx,
                        total_files=len(files),
                        processed_chunks=len(metadata_rows),
                    ),
                )

        if not metadata_rows:
            raise RuntimeError(
                "No indexable source files found. Add text/code files and run 'repomind index' again."
            )

        self._emit(
            progress_callback,
            IndexProgress(
                phase="embed",
                message=f"Embedding {len(metadata_rows)} chunks...",
                processed_files=len(files),
                total_files=len(files),
                processed_chunks=len(metadata_rows),
            ),
        )
        vectors = self._embedder.embed_documents(documents).vectors
        if vectors.ndim != 2:
            raise RuntimeError("Embedding provider returned invalid vector dimensions.")

        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._persist(index=index, metadata_rows=metadata_rows, manifest=manifest)
        self._emit(
            progress_callback,
            IndexProgress(
                phase="persist",
                message="Index artifacts saved.",
                processed_files=len(files),
                total_files=len(files),
                processed_chunks=len(metadata_rows),
            ),
        )

        files_indexed = len({row.file_path for row in metadata_rows})
        return IndexStats(
            files_indexed=files_indexed,
            chunks_indexed=len(metadata_rows),
            index_path=self._config.index_path,
            metadata_path=self._config.metadata_path,
            updated_files=files_indexed,
            removed_files=0,
            scanned_files=len(files),
            mode="full",
        )

    def _incremental_index(
        self,
        root: Path,
        progress_callback: ProgressCallback | None,
    ) -> IndexStats:
        """Update index by embedding only changed/new files and removing deleted files."""
        faiss = require_faiss("indexing")

        old_metadata_rows = self._load_metadata(self._config.metadata_path)
        old_manifest = self._load_manifest()
        old_index = faiss.read_index(str(self._config.index_path))

        if old_index.ntotal != len(old_metadata_rows):
            logger.warning("Index/metadata mismatch detected; falling back to full reindex.")
            return self._full_index(root, progress_callback)

        files = FileScanner(root, self._config.max_file_size_bytes).list_files()
        self._emit(
            progress_callback,
            IndexProgress(
                phase="scan",
                message=f"Scanning {len(files)} files for changes...",
                total_files=len(files),
            ),
        )

        current_manifest: dict[str, FileFingerprint] = {}
        changed_content_by_path: dict[str, str] = {}

        for idx, path in enumerate(files, start=1):
            content = _read_file_content(path)
            if content is None:
                continue

            rel_path = str(path.relative_to(self._config.project_root))
            fingerprint = _fingerprint_file(path=path, content=content)
            current_manifest[rel_path] = fingerprint

            previous = old_manifest.get(rel_path)
            if previous is None or previous.sha256 != fingerprint.sha256:
                changed_content_by_path[rel_path] = content

            if idx == 1 or idx % 25 == 0 or idx == len(files):
                self._emit(
                    progress_callback,
                    IndexProgress(
                        phase="scan",
                        message="Detecting changed files",
                        processed_files=idx,
                        total_files=len(files),
                    ),
                )

        current_paths = set(current_manifest)
        removed_paths = set(old_manifest) - current_paths
        changed_paths = set(changed_content_by_path)
        unchanged_paths = current_paths - changed_paths

        unchanged_rows, unchanged_indices = _select_unchanged_rows(
            old_metadata_rows,
            unchanged_paths,
        )

        unchanged_vectors = np.empty((0, 0), dtype=np.float32)
        if unchanged_indices:
            # Reuse vectors from previous index for unchanged files to avoid re-embedding.
            unchanged_vectors = np.vstack(
                [old_index.reconstruct(int(index)) for index in unchanged_indices]
            ).astype(np.float32)

        changed_rows: list[ChunkMetadata] = []
        changed_docs: list[str] = []
        for rel_path in sorted(changed_paths):
            for chunk_text, start_line, end_line in self._chunker.split(
                changed_content_by_path[rel_path]
            ):
                changed_docs.append(chunk_text)
                changed_rows.append(
                    ChunkMetadata(
                        chunk_id=0,
                        file_path=rel_path,
                        start_line=start_line,
                        end_line=end_line,
                        text=chunk_text,
                    )
                )

        self._emit(
            progress_callback,
            IndexProgress(
                phase="embed",
                message=(
                    f"Embedding {len(changed_rows)} changed chunks "
                    f"from {len(changed_paths)} files..."
                ),
                processed_files=len(files),
                total_files=len(files),
                processed_chunks=len(changed_rows),
            ),
        )

        changed_vectors = np.empty((0, 0), dtype=np.float32)
        if changed_docs:
            changed_vectors = self._embedder.embed_documents(changed_docs).vectors
            if changed_vectors.ndim != 2:
                raise RuntimeError("Embedding provider returned invalid vector dimensions.")

        all_rows = unchanged_rows + changed_rows
        if not all_rows:
            raise RuntimeError(
                "No indexable source files found. Add text/code files and run 'repomind index' again."
            )

        vectors = _combine_vectors(unchanged_vectors, changed_vectors)

        for idx, row in enumerate(all_rows):
            row.chunk_id = idx

        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._persist(index=index, metadata_rows=all_rows, manifest=current_manifest)
        self._emit(
            progress_callback,
            IndexProgress(
                phase="persist",
                message="Incremental index update saved.",
                processed_files=len(files),
                total_files=len(files),
                processed_chunks=len(all_rows),
            ),
        )

        files_indexed = len({row.file_path for row in all_rows})
        return IndexStats(
            files_indexed=files_indexed,
            chunks_indexed=len(all_rows),
            index_path=self._config.index_path,
            metadata_path=self._config.metadata_path,
            updated_files=len(changed_paths),
            removed_files=len(removed_paths),
            scanned_files=len(files),
            mode="update",
        )

    def _persist(
        self,
        *,
        index: Any,
        metadata_rows: list[ChunkMetadata],
        manifest: dict[str, FileFingerprint],
    ) -> None:
        """Persist FAISS index, metadata, and fingerprint manifest."""
        faiss = require_faiss("indexing")
        faiss.write_index(index, str(self._config.index_path))
        self._write_metadata(metadata_rows)
        self._write_manifest(manifest)

    def _write_metadata(self, metadata_rows: list[ChunkMetadata]) -> None:
        """Persist chunk metadata in JSONL format."""
        with self._config.metadata_path.open("w", encoding="utf-8") as fp:
            for row in metadata_rows:
                fp.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    def _load_metadata(self, path: Path) -> list[ChunkMetadata]:
        """Load chunk metadata from JSONL format."""
        rows: list[ChunkMetadata] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                rows.append(ChunkMetadata(**json.loads(line)))
        return rows

    def _write_manifest(self, manifest: dict[str, FileFingerprint]) -> None:
        """Write file fingerprints for incremental updates."""
        payload = {
            path: asdict(fingerprint)
            for path, fingerprint in sorted(manifest.items(), key=lambda item: item[0])
        }
        with self._manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    def _load_manifest(self) -> dict[str, FileFingerprint]:
        """Load fingerprint manifest for incremental update mode."""
        if not self._manifest_path.exists():
            return {}
        with self._manifest_path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        return {path: FileFingerprint(**value) for path, value in raw.items()}

    def _has_existing_index(self) -> bool:
        """Check whether required index artifacts already exist."""
        return (
            self._config.data_dir.exists()
            and self._config.index_path.exists()
            and self._config.metadata_path.exists()
            and self._manifest_path.exists()
        )

    def _validate_root(self, root: Path) -> None:
        """Ensure indexing target remains inside current repository context."""
        if not root.exists() or not root.is_dir():
            raise RuntimeError(f"Index path does not exist or is not a directory: {root}")
        if not root.is_relative_to(self._config.project_root):
            raise RuntimeError(
                "Index path must be inside the current working repository to preserve isolation."
            )

    @staticmethod
    def _emit(callback: ProgressCallback | None, progress: IndexProgress) -> None:
        """Emit a progress signal when callback is configured."""
        if callback is not None:
            callback(progress)


def _select_unchanged_rows(
    rows: list[ChunkMetadata],
    unchanged_paths: set[str],
) -> tuple[list[ChunkMetadata], list[int]]:
    """Select metadata rows and their vector indexes for unchanged files."""
    selected_rows: list[ChunkMetadata] = []
    selected_indices: list[int] = []
    for idx, row in enumerate(rows):
        if row.file_path in unchanged_paths:
            selected_rows.append(row)
            selected_indices.append(idx)
    return selected_rows, selected_indices


def _combine_vectors(unchanged: np.ndarray, changed: np.ndarray) -> np.ndarray:
    """Combine unchanged and changed vectors, validating dimensional consistency."""
    if unchanged.size == 0 and changed.size == 0:
        raise RuntimeError("Unable to build vectors from current repository state.")
    if unchanged.size == 0:
        return changed
    if changed.size == 0:
        return unchanged
    if unchanged.shape[1] != changed.shape[1]:
        raise RuntimeError(
            "Embedding dimension mismatch detected during incremental update. "
            "Run 'repomind index' for a full rebuild."
        )
    return np.vstack([unchanged, changed]).astype(np.float32)


def _read_file_content(path: Path) -> str | None:
    """Read text file content while skipping unreadable files."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning("Skipping unreadable file %s: %s", path, exc)
        return None


def _fingerprint_file(path: Path, content: str) -> FileFingerprint:
    """Create a file fingerprint for change detection."""
    stat = path.stat()
    digest = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
    return FileFingerprint(sha256=digest, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
