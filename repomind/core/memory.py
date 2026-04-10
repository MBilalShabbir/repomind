"""Per-repository note memory stored in .repomind/memory.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class MemoryNote:
    """A single user-authored note attached to this repository."""

    id: int
    note: str
    created_at: str  # ISO 8601 UTC


class MemoryStore:
    """Read/write notes from a single JSON file inside .repomind/."""

    def __init__(self, memory_path: Path) -> None:
        self._path = memory_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, note: str) -> MemoryNote:
        """Append a new note and persist. Returns the saved entry."""
        notes = self._load()
        new_id = max((n.id for n in notes), default=0) + 1
        entry = MemoryNote(
            id=new_id,
            note=note.strip(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        notes.append(entry)
        self._save(notes)
        return entry

    def list(self) -> list[MemoryNote]:
        """Return all stored notes, oldest first."""
        return self._load()

    def forget(self, note_id: int) -> bool:
        """Remove the note with the given ID. Returns True if it existed."""
        notes = self._load()
        remaining = [n for n in notes if n.id != note_id]
        if len(remaining) == len(notes):
            return False
        self._save(remaining)
        return True

    def texts(self) -> list[str]:
        """Return bare note strings — used for prompt injection."""
        return [n.note for n in self._load()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[MemoryNote]:
        if not self._path.exists():
            return []
        try:
            with self._path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            return [MemoryNote(**item) for item in raw]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _save(self, notes: list[MemoryNote]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump([asdict(n) for n in notes], f, indent=2, ensure_ascii=False)
