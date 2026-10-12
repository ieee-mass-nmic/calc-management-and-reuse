"""Three-stage labeling with independent held-out triples and cumulative RankNorm.

Phase 3 expands neighborhoods of *already observed* ranking disagreements.
It never reads unqueried high-fidelity truth to decide which examples to label.
"""
from __future__ import annotations

from dataclasses import replace
import heapq
import json

import numpy as np
import pandas as pd

from pcu_select.hi_fidelity import phase1_stratified, split_budget
from pcu_select.hi_fidelity.sampler import TripleSample
from pcu_select.scorer.inference import InferenceConfig, ScorerInference
from pcu_select.selection.cluster import ClusterConfig, cluster_samples


def triple_key(triple):
    return triple.sample_id, triple.peft_id, triple.task_id


def run_active_labeling(*, meta_pool, pefts, tasks, cache, layout, cfg, labeler):
    from pcu_select.pipeline.offline import _scorer_lookups, run_scorer_training

    sample_ids = cache.read_sample_id_index()
    if set(sample_ids) != {s.sample_id for s in meta_pool}:
        raise ValueError("high-fidelity candidate pool must match the feature cache")
    available = len(sample_ids) * len(pefts) * len(tasks)
    if cfg.q_hi_total <= 0 or cfg.q_hi_eval < 0 or cfg.q_hi_total + cfg.q_hi_eval > available:
        raise ValueError(f"label budget exceeds {available} distinct triples or is nonpositive")
    if not np.isclose(sum(cfg.phase_split), 1) or any(x < 0 for x in cfg.phase_split):
        raise ValueError("phase_split must be nonnegative and sum to one")
    budgets = split_budget(cfg.q_hi_total, cfg.phase_split)
    if budgets.phase1 < 2 * len(pefts) * len(tasks):
        raise ValueError("phase 1 needs at least two labels per PEFT/task; increase q_hi_total")
    features = cache.read_features()
    joint = np.stack([features[sid].e_x.joint for sid in sample_ids])
    clusters = cluster_samples(sample_ids=sample_ids, joint_embeddings=joint,
                               cfg=ClusterConfig(seed=cfg.global_seed))
    initial = phase1_stratified(sample_ids=sample_ids,
        sample_cluster=dict(zip(sample_ids, map(int, clusters.cluster_ids))), pefts=pefts, tasks=tasks,
        budget=budgets.phase1 + cfg.q_hi_eval, seed=cfg.global_seed)
    heldout, phase1 = initial[:cfg.q_hi_eval], initial[cfg.q_hi_eval:]
    excluded = {triple_key(t) for t in initial}
    lo = pd.read_parquet(layout.labels / "lo_fidelity.parquet")
    zx, zp, zt = _scorer_lookups(cache, layout)
    raw_parts = []
    phases = []
    layout.labels.mkdir(parents=True, exist_ok=True)

    def label_phase(triples, phase):
        raw = labeler._compute_raw_deltas(triples)
        raw["phase"] = phase
        raw_parts.append(raw)
        raw_all = pd.concat(raw_parts, ignore_index=True)
        raw_all.to_parquet(layout.labels / "hi_raw.parquet", index=False)
        labels = labeler._aggregate(labeler._rank_norm_within_bucket(raw_all))
        labeler.save_labels(labels, layout.labels / "hi_fidelity.parquet")
        phases.extend({**vars(t), "phase": phase} for t in triples)
        pd.DataFrame(phases).to_parquet(layout.labels / "queried_triples.parquet", index=False)

    label_phase(phase1, 1)
    for phase, count in ((2, budgets.phase2), (3, budgets.phase3)):
        if not count:
            continue
        ckpt = run_scorer_training(cache=cache, layout=layout, cfg=cfg,
                                  output_dir=layout.scorer / f"active_phase{phase - 1}")
        scorer = ScorerInference(ckpt, cfg=InferenceConfig(device=cfg.device, bf16=False))
        observed = pd.read_parquet(layout.labels / "hi_fidelity.parquet")
        candidates = []
        for (pid, tid), group in lo.groupby(["peft_id", "task_id"], sort=True):
            group = group[[ (sid, pid, tid) not in excluded for sid in group.sample_id ]]
            if group.empty:
                continue
            ids = group.sample_id.tolist()
            mu, sigma = scorer.score(np.stack([zx[s] for s in ids]), zp[pid][None], zt[tid][None])
            priority = sigma * (1 + 0.5 * np.maximum(group.u_lo.to_numpy(), 0))
            if phase == 3:
                truth = observed[(observed.peft_id == pid) & (observed.task_id == tid)]
                if len(truth) < 2:
                    raise ValueError("boundary phase requires observed labels in every condition")
                old_ids = truth.sample_id.tolist()
                pred, _ = scorer.score(np.stack([zx[s] for s in old_ids]), zp[pid][None], zt[tid][None])
                true_rank = truth.u_hi.rank(ascending=False).to_numpy()
                pred_rank = pd.Series(pred).rank(ascending=False).to_numpy()
                gap = np.abs(pred_rank - true_rank) / true_rank
                hard = np.argsort(-gap, kind="stable")[:min(16, len(gap))]
                refs = np.stack([features[old_ids[i]].e_x.joint for i in hard])
                refs /= np.maximum(np.linalg.norm(refs, axis=1, keepdims=True), 1e-8)
                embedding = np.stack([features[s].e_x.joint for s in ids])
                embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8)
                priority = ((embedding @ refs.T + 1) * (1 + gap[hard])[None]).max(axis=1)
            top = np.argsort(-priority, kind="stable")[:count]
            candidates.extend((float(priority[i]), ids[i], pid, tid) for i in top)
        chosen = heapq.nlargest(count, candidates)
        if len(chosen) != count:
            raise ValueError("insufficient unqueried triples for the requested phase budget")
        triples = [TripleSample(sid, pid, tid, phase) for _, sid, pid, tid in chosen]
        excluded.update(triple_key(t) for t in triples)
        label_phase(triples, phase)
        del scorer

    if heldout:
        raw_eval = labeler._compute_raw_deltas(heldout)
        raw_eval.to_parquet(layout.labels / "hi_eval_raw.parquet", index=False)
        labeler.save_labels(labeler._aggregate(labeler._rank_norm_within_bucket(raw_eval)),
                            layout.labels / "hi_eval.parquet")
    (layout.labels / "labeling_report.json").write_text(json.dumps({
        "requested_training_triples": cfg.q_hi_total, "training_triples": len(phases),
        "heldout_triples": len(heldout),
        "phase_counts": {str(p): sum(t["phase"] == p for t in phases) for p in (1, 2, 3)},
        "boundary_policy": "semantic neighborhoods of observed top-weighted ranking disagreements",
        "rank_normalization": "cumulative raw deltas within PEFT/task/anchor/horizon",
    }, indent=2))
    return layout.labels / "hi_fidelity.parquet"
