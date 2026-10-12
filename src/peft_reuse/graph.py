"""Immutable artifact identities and a deterministic shared DAG."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from .identity import canonical_json, digest


@dataclass(frozen=True)
class ArtifactSpec:
    operator: str
    version: str
    parents: tuple[str, ...]
    params_json: str
    context_json: str

    @classmethod
    def create(cls, operator: str, parents: list[str], params: dict, context: dict):
        return cls(operator, "1", tuple(parents), canonical_json(params), canonical_json(context))

    @property
    def params(self) -> dict:
        return json.loads(self.params_json)

    def as_dict(self) -> dict:
        return {
            "operator": self.operator,
            "version": self.version,
            "parents": list(self.parents),
            "params": self.params,
            "context": json.loads(self.context_json),
        }

    @property
    def key(self) -> str:
        return digest(self.as_dict())


@dataclass
class Plan:
    nodes: dict[str, ArtifactSpec] = field(default_factory=dict)
    roots: dict[str, str] = field(default_factory=dict)
    expanded_nodes: int = 0

    def add(self, spec: ArtifactSpec) -> str:
        key = spec.key
        if key in self.nodes and self.nodes[key] != spec:
            raise ValueError("Artifact identity collision")
        self.nodes[key] = spec
        self.expanded_nodes += 1
        return key

    def topological(self) -> list[str]:
        order, visiting, done = [], set(), set()

        def visit(key):
            if key in done:
                return
            if key in visiting:
                raise ValueError("Dependency cycle")
            if key not in self.nodes:
                raise ValueError(f"Missing dependency: {key}")
            spec = self.nodes[key]
            if spec.key != key:
                raise ValueError("Node key does not match artifact specification")
            visiting.add(key)
            for parent in spec.parents:
                visit(parent)
            visiting.remove(key)
            done.add(key)
            order.append(key)

        for root in self.roots.values():
            visit(root)
        if len(done) != len(self.nodes):
            raise ValueError("Unreachable nodes in execution plan")
        return order

    def summary(self) -> dict:
        self.topological()
        return {
            "requests": len(self.roots),
            "unique_nodes": len(self.nodes),
            "independent_nodes": self.expanded_nodes,
            "merged_nodes": self.expanded_nodes - len(self.nodes),
            "operators": dict(sorted(Counter(s.operator for s in self.nodes.values()).items())),
        }

    def as_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "roots": self.roots,
            "nodes": {k: self.nodes[k].as_dict() for k in self.topological()},
        }

    def mermaid(self) -> str:
        keys = self.topological()
        names = {k: f"n{i}" for i, k in enumerate(keys)}
        lines = ["flowchart LR"]
        for key in keys:
            spec = self.nodes[key]
            lines.append(f'  {names[key]}["{spec.operator} / {key[:8]}"]')
            for parent in spec.parents:
                lines.append(f"  {names[parent]} --> {names[key]}")
        return "\n".join(lines) + "\n"
