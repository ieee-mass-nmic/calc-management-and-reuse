"""Independent per-request cold flow; no planner, artifact store or shared ranking."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from ._reference.adaptive_quota import pick_top_in_clusters
from .backends.synthetic import SyntheticBackend
from .identity import canonical_json
from .schema import Request


def cold_request(request: Request, backend: SyntheticBackend) -> dict:
    selection, execution = request.config("selection"), request.config("execution")
    features = [backend.op_feature({"text": s.text}, []) for s in request.pool]
    pool = {"ids": [s.sample_id for s in request.pool], "features": features}
    task = backend.op_task(request.config("task"), [])
    peft = backend.op_peft(request.config("peft"), [])
    labels = backend.op_cluster(
        {"clusters": selection["clusters"], "seed": selection["cluster_seed"]}, [pool]
    )
    batch_size = execution["batch_size"]
    blocks = [
        backend.op_score_block(
            {"model": request.config("model"), "execution": execution},
            features[i : i + batch_size] + [task, peft],
        )
        for i in range(0, len(features), batch_size)
    ]
    scores = backend.op_scores({}, blocks)
    quality = backend.op_quality({"lambda": selection["lambda"]}, [scores])
    statistics = backend.op_statistics({"top_pct": selection["top_pct"]}, [quality, labels])
    quotas = backend.op_quota(
        {
            "budget": request.resolved_budget,
            "alpha": selection["alpha"],
            "min_cluster_size": selection["min_cluster_size"],
        },
        [statistics],
    )
    selected = pick_top_in_clusters(
        sample_ids=pool["ids"],
        q_scores=np.asarray(quality),
        cluster_ids=np.asarray(labels),
        cluster_budgets={r["cluster"]: r["count"] for r in quotas},
    )
    return {
        "selected_ids": selected,
        "scored_ids": pool["ids"],
        "scores": scores,
        "quality": quality,
        "cluster_ids": labels,
        "statistics": statistics,
        "quotas": quotas,
    }


def verify(requests: list[Request], backend: SyntheticBackend, outputs: dict) -> dict:
    started = perf_counter()
    mismatches = []
    for request in requests:
        expected = cold_request(request, backend)
        actual = outputs.get(request.request_id)
        if canonical_json(expected) != canonical_json(actual):
            mismatches.append(request.request_id)
    return {
        "scope": "synthetic backend; ordered IDs and all reported intermediates",
        "requests": len(requests),
        "passed": not mismatches,
        "mismatches": mismatches,
        "reference_wall_seconds": perf_counter() - started,
    }
