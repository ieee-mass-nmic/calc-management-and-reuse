"""Public, synthetic fixture: no original training data or model checkpoint."""


def make_manifest() -> dict:
    pool = [
        {"id": f"sample-{i:02}", "text": f"Synthetic instruction {i}: explain topic {i % 5}."}
        for i in range(16)
    ]
    requests = []
    for rank in (4, 8):
        for task in ("reasoning", "summarization"):
            for budget in (3, 6, 12):
                requests.append(
                    {
                        "id": f"lora-r{rank}-{task}-b{budget}",
                        "peft": {"family": "lora", "rank": rank, "targets": ["q_proj", "v_proj"]},
                        "task": {"name": task, "sketch": [f"Synthetic {task} example"]},
                        "budget": {"count": budget},
                    }
                )
    return {
        "schema_version": 1,
        "pool": pool,
        "defaults": {
            "model": {"name": "synthetic-scorer", "revision": "demo-v1"},
            "execution": {"dtype": "float64", "batch_size": 4, "seed": 0},
            "selection": {
                "lambda": 0.2,
                "clusters": 3,
                "cluster_seed": 0,
                "top_pct": 0.5,
                "alpha": 0.6,
                "min_cluster_size": 0,
            },
        },
        "requests": requests,
    }
