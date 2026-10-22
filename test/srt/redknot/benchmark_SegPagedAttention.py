#!/usr/bin/env python3
"""Reproducible SegPaged component suite; no implicit model/QPS claims.

Default: CUDA smoke tests for operator, public API, and head-wise sharing.
CPU: --device cpu --components cache or head-sharing. Model runs require an explicit local
checkpoint and --components workflow. No holder or foreign process is killed
by this portable entry point; the caller must allocate an idle test GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    sources = list((REPO / "python/sglang/srt/mem_cache/head_kv").glob("*.py"))
    sources += list(HERE.glob("*segpaged*.py")) + [Path(__file__).resolve()]
    sources += list((REPO / "python/sglang/srt/layers/attention/redknot").glob("*.py"))
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(sources))}


def commands(args, output):
    jobs = []
    for component in args.components:
        result = output / (component + ".json")
        if component in ("operator", "public"):
            command = [sys.executable, str(HERE / "benchmark_segpaged_page_skip.py"),
                       "--preset", args.preset, "--device", args.device,
                       "--entry", "public" if component == "public" else "prevalidated"]
            if args.operator_lengths:
                command += ["--lengths", *args.operator_lengths]
            command += ["--warmup", str(args.warmup), "--repeats", str(args.repeats)]
        elif component == "cache":
            command = [sys.executable, str(HERE / "benchmark_segpaged_cache.py"),
                       "--preset", args.preset, "--device", args.device,
                       "--warmup", str(args.warmup), "--repeats", str(args.repeats),
                       "--requests", *map(str, args.requests)]
            if args.cache_lengths:
                command += ["--lengths", *args.cache_lengths]
            if args.cache_page_size:
                command += ["--page-size", str(args.cache_page_size)]
        elif component == "head-sharing":
            command = [sys.executable, str(HERE / "benchmark_segpaged_head_sharing.py"),
                       "--preset", args.preset, "--device", args.device,
                       "--local-head-ratios", *map(str, args.local_head_ratios),
                       "--budget-requests", str(args.budget_requests)]
            if args.head_sharing_lengths:
                command += ["--lengths", *args.head_sharing_lengths]
            elif args.preset == "full" and args.document_kv_gib is None:
                command += ["--lengths", "64K", "128K", "256K", "440K"]
            if args.document_kv_gib is not None:
                command += ["--document-kv-gib", str(args.document_kv_gib)]
            for option in ("heads", "head_dim", "page_size"):
                value = getattr(args, "head_sharing_" + option)
                if value is not None:
                    command += ["--" + option.replace("_", "-"), str(value)]
        else:
            command = [sys.executable, str(HERE / "benchmark_segpaged_workflow.py"),
                       "--model-path", str(args.model_path), "--recomputed",
                       "--context-tokens", str(args.model_context_tokens),
                       "--chunks", str(args.model_chunks),
                       "--datasets", *args.datasets, "--samples", str(args.samples),
                       "--warmup", str(args.model_warmup), "--repeats", str(args.model_repeats),
                       "--max-new-tokens", str(args.max_new_tokens)]
            if args.dataset_dir:
                command += ["--dataset-dir", str(args.dataset_dir)]
        jobs.append((component, command + ["--output", str(result)], result))
    return jobs


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--components", nargs="+", choices=("operator", "public", "cache", "head-sharing", "workflow"),
                        default=["operator", "public", "head-sharing"])
    parser.add_argument("--operator-lengths", nargs="+")
    parser.add_argument("--cache-lengths", nargs="+")
    parser.add_argument("--cache-page-size", type=int)
    parser.add_argument("--head-sharing-lengths", nargs="+")
    parser.add_argument("--local-head-ratios", nargs="+", type=float, default=[0., .5, .75])
    parser.add_argument("--budget-requests", type=int, default=4)
    parser.add_argument("--document-kv-gib", type=float)
    parser.add_argument("--head-sharing-heads", type=int)
    parser.add_argument("--head-sharing-head-dim", type=int)
    parser.add_argument("--head-sharing-page-size", type=int)
    parser.add_argument("--requests", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimqa", "multifieldqa_en"])
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--model-context-tokens", type=int, default=8192)
    parser.add_argument("--model-chunks", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--model-warmup", type=int, default=1)
    parser.add_argument("--model-repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot", action="store_true", help="also generate evidence-only PNG/SVG figures using matplotlib")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", args.device):
        parser.error("device must be cpu, cuda, or cuda:N")
    if len(set(args.components)) != len(args.components):
        parser.error("components must be unique")
    if args.warmup < 1 or args.repeats < 2 or min(args.requests) < 1:
        parser.error("warmup >= 1, repeats >= 2, and requests >= 1 are required")
    if args.cache_page_size is not None and args.cache_page_size < 1:
        parser.error("cache-page-size must be positive")
    if args.budget_requests < 1 or any(not 0 <= r <= 1 for r in args.local_head_ratios):
        parser.error("budget-requests >= 1 and local-head-ratios in [0, 1] required")
    if args.document_kv_gib is not None and not 0 < args.document_kv_gib < float("inf"):
        parser.error("document-kv-gib must be finite and positive")
    if args.document_kv_gib is not None and args.head_sharing_lengths:
        parser.error("choose document-kv-gib OR head-sharing-lengths")
    for option in ("heads", "head_dim", "page_size"):
        value = getattr(args, "head_sharing_" + option)
        if value is not None and value < 1:
            parser.error("head-sharing dimensions must be positive")
    if not args.device.startswith("cuda") and not set(args.components) <= {"cache", "head-sharing"}:
        parser.error("CPU can run cache/head-sharing correctness smoke only")
    if "workflow" in args.components:
        if args.model_path is None or not args.model_path.is_dir():
            parser.error("workflow requires an existing --model-path; nothing is downloaded implicitly")
        if args.device not in ("cuda", "cuda:0"):
            parser.error("workflow uses CUDA device 0; use CUDA_VISIBLE_DEVICES to select one physical GPU")
        if min(args.samples, args.model_warmup, args.model_repeats, args.model_context_tokens,
               args.model_chunks, args.max_new_tokens) < 1:
            parser.error("model shape and repetition arguments must be positive")
    return args


def main(argv=None):
    args = arguments(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or HERE / "results" / ("segpaged_suite_" + timestamp)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing evidence: {output}; choose a fresh output directory")
    manifest = dict(schema="segpaged-suite-v1", status="running", created_utc=timestamp,
                    python=sys.version, platform=platform.platform(),
                    cwd=str(REPO), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    source_sha256=source_hashes(), jobs=[],
                    scope_notes=[
                        "Seeded synthetic KV component tests are not model task-accuracy tests.",
                        "Simultaneously live cache requests are not concurrent serving QPS.",
                        "Stream-event time includes launch gaps; wall time also includes host/API work.",
                        "Active page bytes, fixed slab bytes, analytic work, and measured HBM traffic differ.",
                        "The optional real-model bridge includes online costs and may regress.",
                        "No automatic GPU process termination, model download, or repository mutation."])
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    try:
        for name, command, result_path in commands(args, output):
            row = dict(component=name, command=command, result=result_path.name,
                       log=name + ".log", status="running")
            manifest["jobs"].append(row)
            write_json(manifest_path, manifest)
            print(f"[suite] {name}: {' '.join(command)}", flush=True)
            start = time.perf_counter()
            with (output / row["log"]).open("w") as log:
                child = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in child.stdout:
                        log.write(line); log.flush()
                        print(line, end="", flush=True)
                    code = child.wait()
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait()
            row.update(returncode=code, wall_seconds=time.perf_counter() - start)
            if code != 0:
                row["status"] = "failed"
                raise RuntimeError(f"{name} failed with exit code {code}; see {row['log']}")
            data = json.loads(result_path.read_text())
            if data.get("status") != "complete":
                row["status"] = "failed"
                raise RuntimeError(f"{name} did not produce a complete report")
            row.update(status="complete", result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
            write_json(manifest_path, manifest)
        manifest["status"] = "complete"
    except BaseException as exc:
        for job in manifest["jobs"]:
            if job["status"] == "running":
                job["status"] = "failed"
        manifest["status"] = "failed"
        manifest["failure"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    from report_segpaged_benchmark import render
    render(output, plot=args.plot)
    print(f"[suite] complete: {output / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
