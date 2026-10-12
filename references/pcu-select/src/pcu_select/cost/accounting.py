"""GPU-hours and break-even accounting. See design doc §16."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class CostEvent:
    stage: str  # feat / lo / hi / scorer_train / apply_score / apply_select / target_train
    wall_time_sec: float
    gpu_hours: float
    peak_mem_mb: float = 0.0
    disk_written_mb: float = 0.0
    n_samples: int = 0
    peft_id: str | None = None
    task_id: str | None = None
    extra: dict = field(default_factory=dict)
    device: str = "unknown"
    status: str = "ok"
    cpu_time_sec: float = 0.0


class CostAccountant:
    def __init__(self, path: Path | str, *, device: str = "cpu"):
        self.path = Path(path)
        self.device = device
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: CostEvent) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    @contextmanager
    def stage(self, name: str, *, n_samples: int = 0, peft_id: str | None = None,
              task_id: str | None = None) -> Iterator[dict]:
        meta: dict = {}
        import torch

        cuda = self.device.startswith("cuda")
        if cuda and not torch.cuda.is_available():
            raise RuntimeError(f"cost stage {name}: CUDA requested but unavailable")
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        status = "ok"
        try:
            yield meta
        except BaseException as exc:
            status = "failed"
            meta.setdefault("extra", {})["error_type"] = type(exc).__name__
            raise
        finally:
            if cuda:
                torch.cuda.synchronize(self.device)
            t1 = time.perf_counter()
            wall = t1 - t0
            self.log(CostEvent(
                stage=name,
                wall_time_sec=wall,
                gpu_hours=wall / 3600.0 if cuda else 0.0,
                peak_mem_mb=(torch.cuda.max_memory_allocated(self.device) / 2**20
                             if cuda else 0.0),
                disk_written_mb=meta.get("disk_written_mb", 0.0),
                n_samples=n_samples,
                peft_id=peft_id,
                task_id=task_id,
                extra=meta.get("extra", {}),
                device=self.device, status=status, cpu_time_sec=time.process_time() - cpu0,
            ))

    def read_events(self) -> list[CostEvent]:
        if not self.path.exists():
            return []
        events: list[CostEvent] = []
        for line in self.path.read_text().splitlines():
            d = json.loads(line)
            events.append(CostEvent(**d))
        return events


def break_even_T(C_offline: float, C_apply: float, C_specific: float) -> float:
    """Return the number of target PEFTs T at which PCU-Select breaks even."""
    denom = C_specific - C_apply
    if denom <= 1e-9:
        return float("inf")
    return C_offline / denom


def total_cost(events: list[CostEvent]) -> float:
    return sum(e.gpu_hours for e in events)


def write_cost_report(workdir: Path | str, *, required_stages: set[str] | None = None) -> Path:
    """Report measured stage costs. Missing/failed work cannot masquerade as zero cost."""
    root = Path(workdir)
    events = CostAccountant(root / "cost" / "accounting.jsonl").read_events()
    if not events:
        raise ValueError("no measured cost events; run the pipeline before reporting costs")
    present = {e.stage for e in events if e.status == "ok"}
    missing = (required_stages or set()) - present
    if missing:
        raise ValueError(f"missing successful cost stages: {sorted(missing)}")
    stages: dict[str, dict] = {}
    for event in events:
        row = stages.setdefault(event.stage, {"calls": 0, "failed_calls": 0,
            "wall_time_sec": 0.0, "cpu_time_sec": 0.0, "gpu_hours": 0.0, "peak_mem_mb": 0.0})
        row["calls"] += 1
        row["failed_calls"] += event.status != "ok"
        for key in ("wall_time_sec", "cpu_time_sec", "gpu_hours"):
            row[key] += getattr(event, key)
        row["peak_mem_mb"] = max(row["peak_mem_mb"], event.peak_mem_mb)
    report = {
        "schema_version": 1,
        "gpu_hours_definition": "synchronized stage wall time on one allocated CUDA device; CPU/MPS = 0",
        "includes_failed_work": True,
        "devices": sorted({e.device for e in events}),
        "stages": stages,
        "total_wall_time_sec": sum(e.wall_time_sec for e in events),
        "total_gpu_hours": total_cost(events),
    }
    path = root / "cost" / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return path
