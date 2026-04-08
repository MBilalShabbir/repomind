"""Codebase indexing service using FAISS."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from repomind.core.config import RepoMindConfig
from repomind.core.embeddings import Embedder

logger = logging.getLogger(__name__)

IGNORED_DIRECTORIES = {".git", "node_modules", "venv", "__pycache__", ".repomind"}


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
class IndexStats:
    """Summary of indexing output."""

    files_indexed: int
    chunks_indexed: int
    index_path: Path
    metadata_path: Path
    updated_files: int = 0
    removed_files: int = 0


class FileScanner:
    """Scans source files recursively while skipping known noisy directories."""

    def __init__(self, root: Path) -> None:
        """Initialize scanner.

        Args:
            root: Root directory to scan.
        """
        self._root = root.resolve()

    def iter_files(self) -> Iterable[Path]:
        """Yield indexable files under root directory."""
        for path in self._root.rglob("*"):
            if path.is_dir():
                continue
            if any(part in IGNORED_DIRECTORIES for part in path.parts):
                continue
            if path.name.startswith("."):
                continue
            if not self._is_text_file(path):
                continue
            yield path

    @staticmethod
    def _is_text_file(path: Path) -> bool:
        try:
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
        """Split text into chunks preserving approximate line ranges."""
        if not text.strip():
            return []

        lines = text.splitlines(keepends=True)
        chunks: list[tuple[str, int, int]] = []
        buffer: list[str] = []
        start_line = 1
        current_len = 0

        for idx, line in enumerate(lines, start=1):
            buffer.append(line)
            current_len += len(line)
            if current_len >= self._chunk_size:
                chunk_text = "".join(buffer)
                chunks.append((chunk_text, start_line, idx))

                overlap_text = chunk_text[-self._overlap :]
                overlap_lines = overlap_text.splitlines(keepends=True)
                buffer = overlap_lines[:] if overlap_lines else []
                current_len = sum(len(x) for x in buffer)
                start_line = max(1, idx - len(buffer) + 1)

        if buffer:
            chunk_text = "".join(buffer)
            if chunk_text.strip():
                chunks.append((chunk_text, start_line, len(lines)))

        return chunks


class CodeIndexer:
    """Coordinates scanning, chunking, embeddings, and FAISS persistence."""

    def __init__(self, config: RepoMindConfig, embedder: Embedder) -> None:
        """Initialize indexer dependencies."""
        self._config = config
        self._embedder = embedder
        self._chunker = CodeChunker()
        self._manifest_path = self._config.data_dir / "files.json"

    def index(self, root: Path, update: bool = False) -> IndexStats:
        """Build and persist FAISS index from the given root directory."""
        root = root.resolve()
        self._validate_root(root)
        if update and root != self._config.project_root:
            raise RuntimeError(
                "--update requires indexing from the current repository root ('.')."
            )
        self._config.data_dir.mkdir(parents=True, exist_ok=True)

        if update and self._has_existing_index():
            return self._incremental_index(root)

        return self._full_index(root)

    def _full_index(self, root: Path) -> IndexStats:
        """Run a full index build from source files."""
        faiss = _require_faiss()
        scanner = FileScanner(root)

        documents: list[str] = []
        metadatas: list[ChunkMetadata] = []
        manifest: dict[str, FileFingerprint] = {}

        for path in scanner.iter_files():
            content = _read_file_content(path)
            if content is None:
                continue

            relative_path = str(path.relative_to(self._config.project_root))
            fingerprint = _fingerprint_file(path=path, content=content)
            manifest[relative_path] = fingerprint

            chunks = self._chunker.split(content)
            for chunk_text, start_line, end_line in chunks:
                documents.append(chunk_text)
                metadatas.append(
                    ChunkMetadata(
                        chunk_id=len(metadatas),
                        file_path=relative_path,
                        start_line=start_line,
                        end_line=end_line,
                        text=chunk_text,
                    )
                )

        if not metadatas:
            raise RuntimeError("No indexable files were found in the target directory.")

        vectors = self._embedder.embed_documents(documents).vectors
        if vectors.ndim != 2:
            raise RuntimeError("Embedding provider returned invalid vector dimensions.")

        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._persist(index=index, metadatas=metadatas, manifest=manifest)

        files_indexed = len({item.file_path for item in metadatas})
        return IndexStats(
            files_indexed=files_indexed,
            chunks_indexed=len(metadatas),
            index_path=self._config.index_path,
            metadata_path=self._config.metadata_path,
            updated_files=files_indexed,
            removed_files=0,
        )

    def _incremental_index(self, root: Path) -> IndexStats:
        """Update index by embedding only changed/new files and removing deleted files."""
        faiss = _require_faiss()
        old_metadata = self._load_metadata(self._config.metadata_path)
        old_manifest = self._load_manifest()

        old_index = faiss.read_index(str(self._config.index_path))
        if old_index.ntotal != len(old_metadata):
            logger.warning("Index/metadata mismatch detected; falling back to full reindex.")
            return self._full_index(root)

        scanner = FileScanner(root)
        current_manifest: dict[str, FileFingerprint] = {}
        changed_content: dict[str, str] = {}

        for path in scanner.iter_files():
            content = _read_file_content(path)
            if content is None:
                continue

            rel_path = str(path.relative_to(self._config.project_root))
            fingerprint = _fingerprint_file(path=path, content=content)
            current_manifest[rel_path] = fingerprint

            previous = old_manifest.get(rel_path)
            if previous is None or previous.sha256 != fingerprint.sha256:
                changed_content[rel_path] = content

        current_paths = set(current_manifest)
        removed_paths = set(old_manifest) - current_paths
        changed_paths = set(changed_content)
        unchanged_paths = current_paths - changed_paths

        unchanged_rows: list[ChunkMetadata] = []
        unchanged_indices: list[int] = []
        for idx, item in enumerate(old_metadata):
            if item.file_path in unchanged_paths:
                unchanged_rows.append(item)
                unchanged_indices.append(idx)

        unchanged_vectors = np.empty((0, 0), dtype=np.float32)
        if unchanged_indices:
            unchanged_vectors = np.vstack(
                [old_index.reconstruct(int(idx)) for idx in unchanged_indices]
            ).astype(np.float32)

        changed_rows: list[ChunkMetadata] = []
        changed_docs: list[str] = []
        for rel_path in sorted(changed_paths):
            chunks = self._chunker.split(changed_content[rel_path])
            for chunk_text, start_line, end_line in chunks:
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

        changed_vectors = np.empty((0, 0), dtype=np.float32)
        if changed_docs:
            changed_vectors = self._embedder.embed_documents(changed_docs).vectors
            if changed_vectors.ndim != 2:
                raise RuntimeError("Embedding provider returned invalid vector dimensions.")

        all_rows = unchanged_rows + changed_rows
        if not all_rows:
            raise RuntimeError("No indexable files were found in the target directory.")

        if unchanged_vectors.size == 0 and changed_vectors.size == 0:
            raise RuntimeError("Unable to build vectors from current repository state.")

        if unchanged_vectors.size == 0:
            vectors = changed_vectors
        elif changed_vectors.size == 0:
            vectors = unchanged_vectors
        else:
            if unchanged_vectors.shape[1] != changed_vectors.shape[1]:
                logger.warning(
                    "Embedding dimension changed; falling back to full reindex."
                )
                return self._full_index(root)
            vectors = np.vstack([unchanged_vectors, changed_vectors]).astype(np.float32)

        for idx, item in enumerate(all_rows):
            item.chunk_id = idx

        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._persist(index=index, metadatas=all_rows, manifest=current_manifest)

        files_indexed = len({item.file_path for item in all_rows})
        return IndexStats(
            files_indexed=files_indexed,
            chunks_indexed=len(all_rows),
            index_path=self._config.index_path,
            metadata_path=self._config.metadata_path,
            updated_files=len(changed_paths),
            removed_files=len(removed_paths),
        )

    def _persist(
        self,
        *,
        index: Any,
        metadatas: list[ChunkMetadata],
        manifest: dict[str, FileFingerprint],
    ) -> None:
        """Persist FAISS index, metadata, and fingerprint manifest."""
        faiss = _require_faiss()
        faiss.write_index(index, str(self._config.index_path))
        self._write_metadata(metadatas)
        self._write_manifest(manifest)

    def _write_metadata(self, metadatas: list[ChunkMetadata]) -> None:
        """Persist chunk metadata in JSONL format."""
        with self._config.metadata_path.open("w", encoding="utf-8") as fp:
            for item in metadatas:
                fp.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

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


def _require_faiss() -> Any:
    """Import faiss lazily to keep CLI diagnostics available without hard dependency."""
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed. Install dependencies before indexing."
        ) from exc
    return faiss
