"""Small deterministic CPU demo. Features, scorer and clustering are NOT real PCU."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np

from .._reference.adaptive_quota import allocate_cluster_budgets, compute_cluster_values
from ..graph import ArtifactSpec
from ..identity import digest, digest_bytes


def _vector(value: object) -> np.ndarray:
    raw = bytes.fromhex(digest(value))
    return (
        np.array([int.from_bytes(raw[i : i + 4], "big") for i in range(0, 32, 4)], dtype=np.float64)
        / (2**32 - 1)
        * 2
        - 1
    )


class SyntheticBackend:
    """Operator methods are intentionally separate from storage and planning."""

    def __init__(self):
        source = Path(__file__).resolve().parents[1]
        code = {
            str(p.relative_to(source)): digest_bytes(p.read_bytes())
            for p in sorted(source.rglob("*.py"))
        }
        self._context = {
            "backend": "synthetic-v1",
            "code_digest": digest(code),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "system": platform.system(),
            "byteorder": sys.byteorder,
            "numerics": "numpy-float64-cpu-demo",
        }

    @property
    def context(self) -> dict:
        return dict(self._context)

    def execute(self, spec: ArtifactSpec, parents: list) -> object:
        if spec.context_json != ArtifactSpec.create("x", [], {}, self.context).context_json:
            raise ValueError("Plan/backend execution context mismatch")
        method = getattr(self, f"op_{spec.operator}", None)
        if method is None:
            raise ValueError(f"Unsupported operator: {spec.operator}")
        return method(spec.params, parents)

    def op_feature(self, params, parents):
        return _vector({"text": params["text"]}).tolist()

    def op_task(self, params, parents):
        return _vector({"task": params}).tolist()

    def op_peft(self, params, parents):
        return _vector({"peft": params}).tolist()

    def op_pool(self, params, parents):
        return {"ids": params["ids"], "features": parents}

    def op_cluster(self, params, parents):
        # Full-pool deterministic toy k-means, never incremental partial_fit.
        x = np.asarray(parents[0]["features"], dtype=np.float64)
        k = min(params["clusters"], len(x))
        rng = np.random.default_rng(params["seed"])
        centers = x[rng.choice(len(x), size=k, replace=False)].copy()
        for _ in range(5):
            labels = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
            for i in range(k):
                if np.any(labels == i):
                    centers[i] = x[labels == i].mean(axis=0)
        labels = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        return labels.tolist()

    def op_score_block(self, params, parents):
        x = np.asarray(parents[:-2], dtype=np.float64)
        task = np.asarray(parents[-2], dtype=np.float64)
        peft = np.asarray(parents[-1], dtype=np.float64)
        model = _vector({"model": params["model"], "seed": params["execution"]["seed"]})
        # Frozen batch shape/protocol participates in identity even for this small demo.
        mu = (x * (task + peft + model)).sum(axis=1) / 8
        sigma = 0.05 + np.abs((x * (task - peft)).sum(axis=1)) / 16
        return {"mu": mu.tolist(), "sigma": sigma.tolist()}

    def op_scores(self, params, parents):
        return {name: [v for block in parents for v in block[name]] for name in ("mu", "sigma")}

    def op_quality(self, params, parents):
        raw = parents[0]
        return (np.asarray(raw["mu"]) - params["lambda"] * np.asarray(raw["sigma"])).tolist()

    def op_statistics(self, params, parents):
        q, labels = np.asarray(parents[0]), np.asarray(parents[1])
        values = compute_cluster_values(q_scores=q, cluster_ids=labels, top_pct=params["top_pct"])
        # Use ordered records, not JSON object order, to preserve cluster traversal.
        return [
            {"cluster": k, "value": v, "size": int(np.sum(labels == k))} for k, v in values.items()
        ]

    def op_ranking(self, params, parents):
        q, labels = np.asarray(parents[0]), np.asarray(parents[1])
        result = []
        for k in np.unique(labels):
            members = np.flatnonzero(labels == k)
            ranked = members[np.argsort(-q[members])]
            result.append(
                {"cluster": int(k), "members": members.tolist(), "ranked": ranked.tolist()}
            )
        return result

    def op_quota(self, params, parents):
        rows = parents[0]
        budgets = allocate_cluster_budgets(
            cluster_values={r["cluster"]: r["value"] for r in rows},
            cluster_sizes={r["cluster"]: r["size"] for r in rows},
            total_budget=params["budget"],
            alpha=params["alpha"],
            min_cluster_size=params["min_cluster_size"],
        )
        return [{"cluster": k, "count": b} for k, b in budgets.items()]

    def op_select(self, params, parents):
        pool, labels, ranks, quotas, scores, quality, statistics = parents
        by_cluster = {r["cluster"]: r for r in ranks}
        selected = []
        for quota in quotas:
            count = quota["count"]
            rank = by_cluster[quota["cluster"]]
            # Whole-cluster output must retain original order, not score order.
            indices = rank["members"] if count >= len(rank["members"]) else rank["ranked"][:count]
            selected.extend(pool["ids"][i] for i in indices)
        return {
            "selected_ids": selected,
            "scored_ids": pool["ids"],
            "scores": scores,
            "quality": quality,
            "cluster_ids": labels,
            "statistics": statistics,
            "quotas": quotas,
        }
