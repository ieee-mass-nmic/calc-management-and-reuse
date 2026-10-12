"""Normalize effective settings once and retain immutable snapshots."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .identity import canonical_json, digest

EXECUTION = {"dtype": "float64", "batch_size": 4, "seed": 0}
SELECTION = {
    "lambda": 0.2,
    "clusters": 3,
    "cluster_seed": 0,
    "top_pct": 0.5,
    "alpha": 0.6,
    "min_cluster_size": 0,
}


def _fields(value: Any, allowed: set[str], where: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{where} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {where} fields: {sorted(unknown)}")
    return value


def _int(value: Any, minimum: int, name: str) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, lower: float, upper: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite in [{lower}, {upper}]")
    return float(value)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text: str

    @property
    def content_key(self) -> str:
        return digest({"text": self.text})


@dataclass(frozen=True)
class Request:
    request_id: str
    pool: tuple[Sample, ...]
    peft_json: str
    task_json: str
    model_json: str
    execution_json: str
    selection_json: str
    budget_json: str
    resolved_budget: int

    def config(self, field: str) -> dict:
        return json.loads(getattr(self, f"{field}_json"))

    @property
    def ordered_pool_key(self) -> str:
        return digest([[sample.sample_id, sample.content_key] for sample in self.pool])

    def summary(self) -> dict:
        return {
            "request_id": self.request_id,
            "ordered_pool_key": self.ordered_pool_key,
            "pool_size": len(self.pool),
            "peft": self.config("peft"),
            "task": self.config("task"),
            "model": self.config("model"),
            "execution": self.config("execution"),
            "selection": self.config("selection"),
            "raw_budget": self.config("budget"),
            "resolved_budget": self.resolved_budget,
        }


def normalize(manifest: dict) -> list[Request]:
    canonical_json(manifest)  # Also rejects NaN, Infinity and non-JSON inputs.
    _fields(manifest, {"schema_version", "pool", "defaults", "requests"}, "manifest")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    raw_pool = manifest.get("pool")
    if not isinstance(raw_pool, list) or not raw_pool:
        raise ValueError("pool must be a non-empty list")
    pool = []
    for row in raw_pool:
        _fields(row, {"id", "text"}, "sample")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Each sample needs a non-empty string id")
        if not isinstance(row.get("text"), str):
            raise ValueError("Each sample needs string text")
        pool.append(Sample(row["id"], row["text"]))
    if len({s.sample_id for s in pool}) != len(pool):
        raise ValueError("Duplicate sample IDs")
    sections = {"peft", "task", "model", "execution", "selection"}
    defaults = _fields(manifest.get("defaults", {}), sections, "defaults")
    rows = manifest.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("requests must be a non-empty list")
    result = []
    for row in rows:
        _fields(row, sections | {"id", "budget"}, "request")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Each request needs a non-empty string id")
        configs = {}
        for section in ("peft", "task", "model"):
            value = row.get(section, defaults.get(section))
            if not isinstance(value, dict) or not value:
                raise ValueError(f"{section} must be a non-empty object")
            configs[section] = canonical_json(value)
        execution = dict(EXECUTION)
        selection = dict(SELECTION)
        for name, target in (("execution", execution), ("selection", selection)):
            for source in (defaults, row):
                target.update(_fields(source.get(name, {}), set(target), name))
        if execution["dtype"] != "float64":
            raise ValueError("The synthetic backend only supports float64")
        _int(execution["batch_size"], 1, "batch_size")
        _int(execution["seed"], 0, "seed")
        _int(selection["clusters"], 1, "clusters")
        _int(selection["cluster_seed"], 0, "cluster_seed")
        _int(selection["min_cluster_size"], 0, "min_cluster_size")
        selection["lambda"] = _number(selection["lambda"], 0, float("inf"), "lambda")
        selection["alpha"] = _number(selection["alpha"], 0, 1, "alpha")
        selection["top_pct"] = _number(selection["top_pct"], 0, 1, "top_pct")
        if selection["top_pct"] == 0:
            raise ValueError("top_pct must be > 0")
        budget = _fields(row.get("budget"), {"count", "fraction"}, "budget")
        if len(budget) != 1:
            raise ValueError("Specify exactly one budget field: count or fraction")
        if "count" in budget:
            resolved = _int(budget["count"], 0, "budget.count")
            if resolved > len(pool):
                raise ValueError("budget.count exceeds pool size")
        else:
            fraction = _number(budget["fraction"], 0, 1, "budget.fraction")
            resolved = math.floor(len(pool) * fraction)
        result.append(
            Request(
                request_id=row["id"],
                pool=tuple(pool),
                peft_json=configs["peft"],
                task_json=configs["task"],
                model_json=configs["model"],
                execution_json=canonical_json(execution),
                selection_json=canonical_json(selection),
                budget_json=canonical_json(budget),
                resolved_budget=resolved,
            )
        )
    if len({r.request_id for r in result}) != len(result):
        raise ValueError("Duplicate request IDs")
    return result
