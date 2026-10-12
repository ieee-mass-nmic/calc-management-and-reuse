"""POSIX, single-writer immutable JSON store with checksums and atomic publication."""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from .graph import ArtifactSpec
from .identity import canonical_json, digest_bytes, loads


class CorruptArtifact(ValueError):
    pass


class CacheConflict(ValueError):
    pass


class StoreBusy(RuntimeError):
    pass


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._writing = False

    def path(self, key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Invalid artifact key")
        return self.root / key

    @contextmanager
    def writer(self):
        if self._writing:
            raise StoreBusy("Nested store writers are unsupported")
        with (self.root / ".writer.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StoreBusy(f"Another executor is writing {self.root}") from exc
            self._writing = True
            try:
                yield self
            finally:
                self._writing = False
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def load(self, spec: ArtifactSpec) -> object | None:
        directory = self.path(spec.key)
        if not directory.exists() and not directory.is_symlink():
            return None
        try:
            paths = [directory / "manifest.json", directory / "payload.json"]
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("Artifact must be a regular directory")
            if any(p.is_symlink() or not p.is_file() for p in paths):
                raise ValueError("Incomplete artifact or invalid file type")
            manifest = loads(paths[0].read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("Invalid manifest")
            if (
                manifest.get("format_version") != 1
                or manifest.get("state") != "complete"
                or manifest.get("key") != spec.key
                or manifest.get("spec") != spec.as_dict()
            ):
                raise ValueError("Artifact identity/manifest mismatch")
            data = paths[1].read_bytes()
            if len(data) != manifest.get("payload_bytes") or digest_bytes(data) != manifest.get(
                "payload_sha256"
            ):
                raise ValueError("Payload checksum mismatch")
            value = loads(data.decode("utf-8"))
            if value is None:
                raise ValueError("Null payloads are reserved for cache misses")
            return value
        except (ValueError, OSError, UnicodeError, TypeError, KeyError) as exc:
            raise CorruptArtifact(f"{spec.key}: {exc}") from exc

    def quarantine(self, spec: ArtifactSpec) -> Path:
        self._require_writer()
        target = self.root / f".corrupt-{spec.key}-{uuid.uuid4().hex}"
        os.rename(self.path(spec.key), target)
        return target

    def commit(self, spec: ArtifactSpec, payload: object) -> int:
        self._require_writer()
        if payload is None:
            raise ValueError("Null payloads are not supported")
        data = canonical_json(payload).encode("utf-8")
        existing = self.load(spec)
        if existing is not None:
            if canonical_json(existing).encode("utf-8") != data:
                raise CacheConflict("Same dependency identity produced a different payload")
            return 0
        manifest = {
            "format_version": 1,
            "state": "complete",
            "key": spec.key,
            "spec": spec.as_dict(),
            "payload_bytes": len(data),
            "payload_sha256": digest_bytes(data),
        }
        metadata = canonical_json(manifest).encode("utf-8")
        temporary = Path(tempfile.mkdtemp(prefix=".tmp-", dir=self.root))
        try:
            for name, contents in (("payload.json", data), ("manifest.json", metadata)):
                with (temporary / name).open("xb") as handle:
                    handle.write(contents)
                    handle.flush()
                    os.fsync(handle.fileno())
            self._sync_dir(temporary)
            os.rename(temporary, self.path(spec.key))
            self._sync_dir(self.root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return len(data) + len(metadata)

    def _require_writer(self):
        if not self._writing:
            raise StoreBusy("Mutation requires the store.writer() lock")

    @staticmethod
    def _sync_dir(path: Path):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
