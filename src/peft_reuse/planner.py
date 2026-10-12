"""Expand declared dependencies, then merge identical immutable artifacts."""

from .backends.base import Backend
from .graph import ArtifactSpec, Plan
from .schema import Request


def build_plan(requests: list[Request], backend: Backend) -> Plan:
    if not requests or len({r.request_id for r in requests}) != len(requests):
        raise ValueError("A plan needs requests with unique IDs")
    plan = Plan()
    context = backend.context

    def add(op, parents, params=None):
        return plan.add(ArtifactSpec.create(op, parents, params or {}, context))

    for request in requests:
        execution, selection = request.config("execution"), request.config("selection")
        features = [add("feature", [], {"text": s.text}) for s in request.pool]
        pool = add("pool", features, {"ids": [s.sample_id for s in request.pool]})
        task = add("task", [], request.config("task"))
        peft = add("peft", [], request.config("peft"))
        cluster = add(
            "cluster",
            [pool],
            {
                "clusters": selection["clusters"],
                "seed": selection["cluster_seed"],
            },
        )
        batch_size = execution["batch_size"]
        blocks = [
            add(
                "score_block",
                features[i : i + batch_size] + [task, peft],
                {
                    "model": request.config("model"),
                    "execution": execution,
                },
            )
            for i in range(0, len(features), batch_size)
        ]
        scores = add("scores", blocks)
        quality = add("quality", [scores], {"lambda": selection["lambda"]})
        statistics = add("statistics", [quality, cluster], {"top_pct": selection["top_pct"]})
        ranks = add("ranking", [quality, cluster])
        quota = add(
            "quota",
            [statistics],
            {
                "budget": request.resolved_budget,
                "alpha": selection["alpha"],
                "min_cluster_size": selection["min_cluster_size"],
            },
        )
        plan.roots[request.request_id] = add(
            "select",
            [pool, cluster, ranks, quota, scores, quality, statistics],
        )
    plan.topological()
    return plan
