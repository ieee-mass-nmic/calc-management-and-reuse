from copy import deepcopy

import pytest

from peft_reuse.identity import canonical_json, loads
from peft_reuse.planner import build_plan
from peft_reuse.schema import normalize


def keys(plan, operator):
    return {k for k, spec in plan.nodes.items() if spec.operator == operator}


def test_budget_axis_merges_only_shared_stages(manifest, backend):
    plan = build_plan(normalize(manifest), backend)
    assert len(keys(plan, "cluster")) == 1
    assert len(keys(plan, "score_block")) == 4 * 4
    assert len(keys(plan, "ranking")) == 4
    assert len(keys(plan, "quota")) == 12
    assert len(keys(plan, "select")) == 12
    assert plan.summary()["merged_nodes"] > 0
    assert len(plan.topological()) == len(plan.nodes)


@pytest.mark.parametrize(
    ("section", "field", "value", "changed", "unchanged"),
    [
        (
            "selection",
            "lambda",
            0.7,
            {"quality", "statistics", "ranking", "quota", "select"},
            {"feature", "cluster", "score_block"},
        ),
        (
            "selection",
            "top_pct",
            0.75,
            {"statistics", "quota", "select"},
            {"feature", "cluster", "score_block", "quality", "ranking"},
        ),
        ("selection", "alpha", 0.9, {"quota", "select"}, {"statistics", "ranking", "score_block"}),
        ("execution", "batch_size", 8, {"score_block", "scores", "select"}, {"feature", "cluster"}),
        ("model", "revision", "demo-v2", {"score_block", "select"}, {"feature", "cluster"}),
        (
            "task",
            "sketch",
            ["Changed task content"],
            {"task", "score_block", "select"},
            {"feature", "cluster", "peft"},
        ),
        ("peft", "rank", 16, {"peft", "score_block", "select"}, {"feature", "cluster", "task"}),
    ],
)
def test_contract_invalidation(manifest, backend, section, field, value, changed, unchanged):
    manifest["requests"] = manifest["requests"][:1]
    before = build_plan(normalize(manifest), backend)
    row = manifest["requests"][0]
    row.setdefault(section, deepcopy(manifest["defaults"].get(section, {})))[field] = value
    after = build_plan(normalize(manifest), backend)
    for stage in changed:
        assert keys(before, stage).isdisjoint(keys(after, stage)), stage
    for stage in unchanged:
        assert keys(before, stage) == keys(after, stage), stage


def test_same_id_changed_content_invalidates_one_block_and_global_cluster(manifest, backend):
    manifest["requests"] = manifest["requests"][:1]
    before = build_plan(normalize(manifest), backend)
    manifest["pool"][2]["text"] += " Modified answer"
    after = build_plan(normalize(manifest), backend)
    assert len(keys(before, "feature") - keys(after, "feature")) == 1
    assert len(keys(before, "score_block") - keys(after, "score_block")) == 1
    assert keys(before, "cluster").isdisjoint(keys(after, "cluster"))
    assert before.roots != after.roots


def test_append_reuses_full_blocks_but_rebuilds_old_tail(manifest, backend):
    manifest["requests"] = manifest["requests"][:1]
    last = manifest["pool"].pop()
    before = build_plan(normalize(manifest), backend)
    manifest["pool"].append(last)
    after = build_plan(normalize(manifest), backend)
    assert len(keys(before, "score_block") & keys(after, "score_block")) == 3
    assert len(keys(before, "score_block") - keys(after, "score_block")) == 1
    assert keys(before, "cluster").isdisjoint(keys(after, "cluster"))


def test_reorder_preserves_content_features_but_not_ordered_pool(manifest, backend):
    before_request = normalize(manifest)[0]
    before = build_plan([before_request], backend)
    manifest["pool"].reverse()
    after_request = normalize(manifest)[0]
    after = build_plan([after_request], backend)
    assert before_request.ordered_pool_key != after_request.ordered_pool_key
    assert keys(before, "feature") == keys(after, "feature")
    assert keys(before, "pool").isdisjoint(keys(after, "pool"))
    assert keys(before, "cluster").isdisjoint(keys(after, "cluster"))


def test_normalization_freezes_inputs_and_resolves_budget(manifest, backend):
    manifest["requests"] = manifest["requests"][:1]
    row = manifest["requests"][0]
    row["budget"] = {"fraction": 1.0}
    frozen = normalize(manifest)[0]
    key = build_plan([frozen], backend).roots[frozen.request_id]
    row["peft"]["rank"] = 512
    manifest["pool"][0]["text"] = "Changed after normalization"
    assert frozen.resolved_budget == 16
    assert frozen.config("budget") == {"fraction": 1.0}
    assert build_plan([frozen], backend).roots[frozen.request_id] == key
    row["budget"] = {"count": 1}
    assert normalize(manifest)[0].resolved_budget == 1


@pytest.mark.parametrize(
    "budget",
    [
        {"count": True},
        {"count": 17},
        {"count": -1},
        {"fraction": 1.1},
        {"count": 1, "fraction": 0.2},
        {},
    ],
)
def test_reject_ambiguous_invalid_budget(manifest, budget):
    manifest["requests"][0]["budget"] = budget
    with pytest.raises(ValueError):
        normalize(manifest)


def test_strict_json_and_manifest_fields(manifest):
    with pytest.raises(ValueError, match="Duplicate"):
        loads('{"x": 1, "x": 2}')
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    manifest["defaults"]["selection"]["lamda"] = 0.2
    with pytest.raises(ValueError, match="Unknown"):
        normalize(manifest)


def test_missing_parent_rejected(manifest, backend):
    plan = build_plan(normalize(manifest), backend)
    del plan.nodes[next(iter(keys(plan, "feature")))]
    with pytest.raises(ValueError, match="Missing"):
        plan.topological()
