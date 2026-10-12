"""Demand-driven execution: a verified cache hit prunes its entire ancestor branch."""

from __future__ import annotations

import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .backends.base import Backend
from .graph import Plan
from .identity import canonical_json
from .store import ArtifactStore, CorruptArtifact


@dataclass
class RunResult:
    outputs: dict
    metrics: dict
    events: list[dict]


def run(
    plan: Plan, backend: Backend, store: ArtifactStore | None = None, event_path: Path | None = None
) -> RunResult:
    """With no store, retain in-batch sharing but neither read nor write disk artifacts."""
    started = perf_counter()
    plan.topological()
    context = canonical_json(backend.context)
    if any(s.context_json != context for s in plan.nodes.values()):
        raise ValueError("Plan/backend execution context mismatch")
    memo, events = {}, []
    stream = event_path.open("x", encoding="utf-8") if event_path else None

    def emit(event):
        events.append(event)
        if stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

    def resolve(key):
        if key in memo:
            return memo[key]
        spec = plan.nodes[key]
        event = {
            "key": key,
            "operator": spec.operator,
            "status": "miss",
            "read_seconds": 0.0,
            "compute_seconds": 0.0,
            "write_seconds": 0.0,
            "written_bytes": 0,
        }
        value = None
        try:
            if store:
                read_start = perf_counter()
                try:
                    value = store.load(spec)
                except CorruptArtifact as exc:
                    event["corruption"] = str(exc)
                    event["quarantine"] = str(store.quarantine(spec))
                    event["status"] = "corrupt_recomputed"
                finally:
                    event["read_seconds"] = perf_counter() - read_start
            if value is not None:
                event["status"] = "hit"
            else:
                parents = [resolve(parent) for parent in spec.parents]
                compute_start = perf_counter()
                try:
                    value = backend.execute(spec, parents)
                finally:
                    event["compute_seconds"] = perf_counter() - compute_start
                if store:
                    write_start = perf_counter()
                    try:
                        event["written_bytes"] = store.commit(spec, value)
                    finally:
                        event["write_seconds"] = perf_counter() - write_start
            memo[key] = value
        except Exception as exc:
            event["status"] = "failed"
            event["error"] = f"{type(exc).__name__}: {exc}"
            emit(event)
            raise
        emit(event)
        return value

    try:
        with store.writer() if store else nullcontext():
            outputs = {request_id: resolve(key) for request_id, key in plan.roots.items()}
    finally:
        if stream:
            stream.close()
    metrics = {
        "wall_seconds": perf_counter() - started,
        "visited_nodes": len(events),
        "pruned_nodes": len(plan.nodes) - len(events),
        "statuses": dict(Counter(e["status"] for e in events)),
        "computed_by_operator": dict(
            Counter(e["operator"] for e in events if e["status"] != "hit")
        ),
        **{
            name: sum(e[name] for e in events)
            for name in ("read_seconds", "compute_seconds", "write_seconds", "written_bytes")
        },
    }
    return RunResult(outputs, metrics, events)
