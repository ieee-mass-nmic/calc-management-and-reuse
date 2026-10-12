"""Inspectable plans, executable demo, and independent cold-reference verification."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from .backends import SyntheticBackend
from .demo import make_manifest
from .executor import run
from .identity import read_json
from .planner import build_plan
from .reference import verify
from .schema import normalize
from .store import ArtifactStore


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _unique_dir(prefix: str) -> Path:
    return Path(".runs") / f"{prefix}-{time.time_ns()}-{uuid.uuid4().hex[:6]}"


def execute_manifest(manifest: dict, output: Path, cache: Path | None, check: bool) -> dict:
    total_start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    requests = normalize(manifest)
    backend = SyntheticBackend()
    plan_start = time.perf_counter()
    plan = build_plan(requests, backend)
    planning_seconds = time.perf_counter() - plan_start
    write_json(output / "manifest.json", manifest)
    write_json(output / "plan.json", plan.as_dict())
    (output / "plan.mmd").write_text(plan.mermaid(), encoding="utf-8")
    result = run(plan, backend, ArtifactStore(cache) if cache else None, output / "events.jsonl")
    results = [
        {
            **r.summary(),
            "artifact_key": plan.roots[r.request_id],
            "result": result.outputs[r.request_id],
        }
        for r in requests
    ]
    write_json(output / "results.json", results)
    report = {
        "backend": backend.context,
        "research_measurement": False,
        "cache_policy": "reuse" if cache else "no-disk-cache",
        "cache_path": str(cache.resolve()) if cache else None,
        "plan": plan.summary(),
        "planning_seconds": planning_seconds,
        "execution": result.metrics,
        "verification": verify(requests, backend, result.outputs) if check else None,
        "output": str(output.resolve()),
    }
    report["total_wall_seconds_before_report_write"] = time.perf_counter() - total_start
    write_json(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PEFT-Reuse synthetic execution prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="Expand and merge a workload DAG")
    plan_parser.add_argument("manifest", type=Path)
    plan_parser.add_argument("--output", type=Path)
    plan_parser.add_argument("--mermaid", type=Path)
    for name in ("run", "verify"):
        cmd = sub.add_parser(
            name, help="Execute workload" if name == "run" else "Check cold equivalence"
        )
        cmd.add_argument("manifest", type=Path)
        cmd.add_argument("--output", type=Path)
        cmd.add_argument("--cache", type=Path, default=Path(".cache/artifacts"))
        cmd.add_argument(
            "--no-cache", action="store_true", help="Keep only in-batch memory sharing"
        )
        if name == "run":
            cmd.add_argument("--verify", action="store_true")
    demo = sub.add_parser("demo", help="Run 12 requests cold, warm, and without disk caching")
    demo.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            requests = normalize(read_json(args.manifest))
            plan = build_plan(requests, SyntheticBackend())
            if args.output:
                write_json(args.output, plan.as_dict())
            if args.mermaid:
                args.mermaid.parent.mkdir(parents=True, exist_ok=True)
                with args.mermaid.open("x", encoding="utf-8") as handle:
                    handle.write(plan.mermaid())
            print(json.dumps(plan.summary(), indent=2))
            return 0
        if args.command == "demo":
            output = args.output or _unique_dir("demo")
            output.mkdir(parents=True, exist_ok=False)
            manifest = make_manifest()
            write_json(output / "workload.json", manifest)
            reports = {}
            for name, cache in (
                ("cold", output / "cache"),
                ("warm", output / "cache"),
                ("shared_no_disk", None),
            ):
                reports[name] = execute_manifest(manifest, output / name, cache, True)
            summary = {
                "output": str(output.resolve()),
                "research_measurement": False,
                "note": "Synthetic CPU demo; timings are not PCU-Select speedup evidence.",
                "passed": all(r["verification"]["passed"] for r in reports.values()),
                "runs": {
                    name: {
                        "plan": r["plan"],
                        "execution": r["execution"],
                        "verification": r["verification"],
                    }
                    for name, r in reports.items()
                },
            }
            write_json(output / "summary.json", summary)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0 if summary["passed"] else 2
        report = execute_manifest(
            read_json(args.manifest),
            args.output or _unique_dir(args.command),
            None if args.no_cache else args.cache,
            args.command == "verify" or args.verify,
        )
        print(
            json.dumps(
                {
                    "output": report["output"],
                    "plan": report["plan"],
                    "execution": report["execution"],
                    "verification": report["verification"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2 if report["verification"] and not report["verification"]["passed"] else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
