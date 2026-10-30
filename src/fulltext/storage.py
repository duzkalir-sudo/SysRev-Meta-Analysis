"""Content-addressed, crash-safe storage for retrieved and uploaded full texts.

Files are stored as ``<root>/<sha256><suffix>``. Writes go to ``<root>/.partial/`` first
and are moved into place with :func:`os.replace` (atomic on POSIX and Windows) only after
the SHA-256 is known, so an interrupted download never leaves a half-written file that
looks complete. Identical bytes are stored once, which makes duplicate PDFs detectable.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.pdf_extractor.parser import HashMismatchError


@dataclass(frozen=True, slots=True)
class StoredFile:
    path: Path
    sha256: str
    size_bytes: int

    @property
    def filename(self) -> str:
        return self.path.name


class DocumentStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.partial_dir = self.root / ".partial"  # created lazily on first write

    def path_for(self, sha256: str, suffix: str) -> Path:
        return self.root / f"{sha256}{suffix}"

    def new_partial(self) -> Path:
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        return self.partial_dir / f"{uuid.uuid4().hex}.part"

    def commit_partial(self, partial: Path, sha256: str, suffix: str) -> StoredFile:
        final = self.path_for(sha256, suffix)
        size = partial.stat().st_size
        if final.exists():
            partial.unlink()  # identical content already stored
        else:
            os.replace(partial, final)
        return StoredFile(final, sha256, size)

    def put_bytes(self, data: bytes, suffix: str) -> StoredFile:
        sha = hashlib.sha256(data).hexdigest()
        partial = self.new_partial()
        partial.write_bytes(data)
        return self.commit_partial(partial, sha, suffix)

    def read_verified(self, filename: str, sha256: str) -> bytes:
        """Read a stored file and refuse it if its bytes no longer match ``sha256``."""
        path = self.root / Path(filename).name
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256:
            raise HashMismatchError(f"{path.name}: stored file hash {actual} != {sha256}")
        return data

    def cleanup_partials(self) -> int:
        """Remove leftovers of interrupted writes (safe to call at start-up)."""
        removed = 0
        if not self.partial_dir.exists():
            return 0
        for p in self.partial_dir.glob("*.part"):
            p.unlink(missing_ok=True)
            removed += 1
        return removed
