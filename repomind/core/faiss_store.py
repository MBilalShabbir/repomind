"""FAISS utility helpers used across indexing and retrieval."""

from __future__ import annotations

from typing import Any


def require_faiss(action: str) -> Any:
    """Import `faiss` lazily with actionable errors.

    Args:
        action: Operation name used in error messages.

    Returns:
        Imported `faiss` module.
    """
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"faiss-cpu is not installed. Install dependencies before {action}."
        ) from exc
    return faiss
