import json
from copy import deepcopy

import numpy as np
import pytest

from peft_reuse._reference.adaptive_quota import (
    allocate_cluster_budgets,
    pick_top_in_clusters,
)
from peft_reuse.cli import main
from peft_reuse.executor import run
from peft_reuse.identity import canonical_json
from peft_reuse.planner import build_plan
from peft_reuse.reference import cold_request, verify
from peft_reuse.schema import normalize
from peft_reuse.store import ArtifactStore


def test_cold_warm_and_uncached_match_independent_reference(manifest, backend, tmp_path):
    requests = normalize(manifest)
    plan = build_plan(requests, backend)
    store = ArtifactStore(tmp_path)
    cold = run(plan, backend, store)
    warm = run(plan, backend, store)
    uncached = run(plan, backend)
    assert verify(requests, backend, cold.outputs)["passed"]
    assert cold.outputs == warm.outputs == uncached.outputs
    assert cold.metrics["computed_by_operator"]["cluster"] == 1
    assert cold.metrics["computed_by_operator"]["score_block"] == 16
    assert warm.metrics["computed_by_operator"] == {}
    assert warm.metrics["statuses"] == {"hit": 12}
    assert warm.metrics["pruned_nodes"] > 0
    for request in requests:
        selected = cold.outputs[request.request_id]["selected_ids"]
        assert len(selected) == len(set(selected)) == request.resolved_budget


def test_corrupt_result_is_quarantined_then_recomputed(manifest, backend, tmp_path):
    plan = build_plan(normalize(manifest), backend)
    store = ArtifactStore(tmp_path)
    expected = run(plan, backend, store)
    key = next(iter(plan.roots.values()))
    (store.path(key) / "payload.json").write_text("{}", encoding="utf-8")
    repaired = run(plan, backend, store)
    assert repaired.outputs == expected.outputs
    assert repaired.metrics["statuses"]["corrupt_recomputed"] == 1
    assert repaired.metrics["computed_by_operator"] == {"select": 1}
    assert list(tmp_path.glob(".corrupt-*"))


def test_resume_after_stage_failure_reuses_completed_ancestors(
    manifest, backend, tmp_path, monkeypatch
):
    manifest["requests"] = manifest["requests"][:1]
    requests = normalize(manifest)
    plan = build_plan(requests, backend)
    store = ArtifactStore(tmp_path / "cache")
    original = backend.execute

    def failing(spec, parents):
        if spec.operator == "cluster":
            raise RuntimeError("Injected cluster failure")
        return original(spec, parents)

    monkeypatch.setattr(backend, "execute", failing)
    with pytest.raises(RuntimeError, match="Injected"):
        run(plan, backend, store, tmp_path / "failure.jsonl")
    assert '"status": "failed"' in (tmp_path / "failure.jsonl").read_text()
    monkeypatch.setattr(backend, "execute", original)
    recovered = run(plan, backend, store)
    assert "feature" not in recovered.metrics["computed_by_operator"]
    assert verify(requests, backend, recovered.outputs)["passed"]


def test_pool_mutations_cached_outputs_match_cold(manifest, backend, tmp_path):
    manifest["requests"] = manifest["requests"][:1]
    store = ArtifactStore(tmp_path)
    variants = [deepcopy(manifest) for _ in range(5)]
    variants[1]["pool"][2]["text"] += " revised"
    variants[2]["pool"].reverse()
    variants[3]["pool"].append({"id": "new", "text": "New synthetic data"})
    del variants[4]["pool"][1]
    for variant in variants:
        requests = normalize(variant)
        actual = run(build_plan(requests, backend), backend, store)
        assert verify(requests, backend, actual.outputs)["passed"]


def test_budget_non_nesting_counterexample():
    values = dict(enumerate([1.5403466579, 2.8518864520, 0.4410372420, 2.8464618469]))
    sizes = dict.fromkeys(values, 100)
    small = allocate_cluster_budgets(
        cluster_values=values, cluster_sizes=sizes, total_budget=7, alpha=0.6
    )
    large = allocate_cluster_budgets(
        cluster_values=values, cluster_sizes=sizes, total_budget=8, alpha=0.6
    )
    assert list(small.values()) == [2, 2, 1, 2]
    assert list(large.values()) == [2, 3, 0, 3]


def test_whole_cluster_order_and_ties_match_frozen_reference(backend):
    ids, q, labels = ["a", "b", "c", "d"], [0.1, 0.9, 0.9, 0.5], [0, 0, 0, 0]
    ranks = backend.op_ranking({}, [q, labels])
    for budget in range(5):
        quota = [{"cluster": 0, "count": budget}]
        actual = backend.op_select({}, [{"ids": ids}, labels, ranks, quota, {}, q, []])
        expected = pick_top_in_clusters(
            sample_ids=ids,
            q_scores=np.asarray(q),
            cluster_ids=np.asarray(labels),
            cluster_budgets={0: budget},
        )
        assert actual["selected_ids"] == expected
    assert actual["selected_ids"] == ids


def test_reference_catches_output_order_error(manifest, backend):
    requests = normalize(manifest)[:1]
    output = cold_request(requests[0], backend)
    output["selected_ids"].reverse()
    assert not verify(requests, backend, {requests[0].request_id: output})["passed"]


def test_context_mismatch_cannot_hit_existing_cache(manifest, backend, tmp_path):
    plan = build_plan(normalize(manifest), backend)
    store = ArtifactStore(tmp_path)
    run(plan, backend, store)
    backend._context["numpy"] = "changed"
    with pytest.raises(ValueError, match="context mismatch"):
        run(plan, backend, store)


def test_cli_plan_verify_and_demo(manifest, tmp_path, capsys):
    source = tmp_path / "workload.json"
    source.write_text(canonical_json(manifest), encoding="utf-8")
    assert (
        main(
            [
                "plan",
                str(source),
                "--output",
                str(tmp_path / "plan.json"),
                "--mermaid",
                str(tmp_path / "plan.mmd"),
            ]
        )
        == 0
    )
    assert (tmp_path / "plan.mmd").read_text().startswith("flowchart LR")
    assert (
        main(
            [
                "verify",
                str(source),
                "--output",
                str(tmp_path / "run"),
                "--cache",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["verification"]["passed"]
    assert main(["demo", "--output", str(tmp_path / "demo")]) == 0
    assert json.loads((tmp_path / "demo/summary.json").read_text())["passed"]
    assert main(["run", str(source), "--output", str(tmp_path / "run")]) == 1
    assert "error:" in capsys.readouterr().err
