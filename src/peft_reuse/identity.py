"""Strict canonical JSON and content identity; file names are never identities."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _validate(value: Any) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) in (list, tuple):
        for child in value:
            _validate(child)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for child in value.values():
            _validate(child)
        return
    raise ValueError(f"Not a finite JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    _validate(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def loads(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"Duplicate JSON field: {key}")
            out[key] = value
        return out

    value = json.loads(text, object_pairs_hook=pairs)
    _validate(value)
    return value


def read_json(path: Path) -> Any:
    return loads(path.read_text(encoding="utf-8"))
