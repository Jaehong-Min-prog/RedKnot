#!/usr/bin/env python3
"""Paired CUDA experiment for SegPaged invisible-page/tile skipping.

By default this measures the *prevalidated attention operator*. ``--entry
public`` additionally includes public descriptor validation and its CPU/GPU
synchronization. Neither mode measures model TTFT, serving QPS, manager-side
descriptor construction, COW, or distributed transfer. Both arms use the same
tensors, masks, query positions, page order, and Split-KV configuration.
The reference arm retains full K/V traversal; the optimized arm sets
``skip_invisible=True``. Both use the current, finer Split-KV scheduling so the
timed reference is a tuned baseline. No different sparsity/quality policy is
applied. An additional untimed check compares the optimized output with the
legacy release's partition count to detect numerical changes from scheduling.

Examples, from any working directory:
  python benchmark_segpaged_page_skip.py --preset smoke --output smoke.json
  python benchmark_segpaged_page_skip.py --preset full --output full.json
  python benchmark_segpaged_page_skip.py --entry public --preset smoke --output public.json
  python benchmark_segpaged_page_skip.py --lengths 256K --queries 1 16 \
      --local-ratios 0 .5 .75 1 --warmup 5 --repeats 12 --output 256k.json

Every measured block alternates ABBA/BAAB order, contributing two samples per
arm. CUDA event timing covers the launch entry's device-stream interval; it
can include host launch gaps, and is not a profiler's sum of kernel durations.
Synchronized wall timing is reported separately. Compilation and warmups are
excluded. Inputs are deterministic random tensors, not a language dataset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

import torch


REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "python/sglang/srt/mem_cache/head_kv/triton_attention.py"


def load_operator(entry="prevalidated"):
    # The isolated package permits attention.py's relative Triton import without
    # loading head_kv.__init__ or the SGLang server and its optional dependencies.
    package_name = "_redknot_segpaged_skip_benchmark"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(SOURCE.parent)]
        sys.modules[package_name] = package
    source = SOURCE if entry == "prevalidated" else SOURCE.with_name("attention.py")
    name = f"{package_name}.{source.stem}"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if entry == "prevalidated":
        return module.triton_paged_attention

    def public_entry(query, keys, values, slots, lengths, positions, query_positions,
                     windows, sinks, q_per_kv, scale, causal, *, skip_invisible,
                     num_splits=None):
        return module.paged_attention(
            query, keys, values, slots, lengths, positions,
            query_positions=query_positions, num_q_per_kv=q_per_kv,
            scale=scale, causal=causal, windows=windows, sinks=sinks,
            backend="triton", skip_invisible=skip_invisible, num_splits=num_splits)

    return public_entry


def token_length(value):
    value = str(value).strip().lower()
    return int(value[:-1]) * 1024 if value.endswith("k") else int(value)


def command_output(argv):
    try:
        run = subprocess.run(argv, cwd=REPO, capture_output=True, text=True,
                             check=True, timeout=10)
        return run.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(device):
    props = torch.cuda.get_device_properties(device)
    packages = {}
    for name in ("torch", "triton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source_files = (SOURCE, SOURCE.with_name("attention.py"), Path(__file__).resolve())
    return dict(
        python=sys.version, platform=platform.platform(), packages=packages,
        torch_cuda=torch.version.cuda, device=str(device),
        device_name=props.name, device_total_memory_bytes=props.total_memory,
        compute_capability=list(torch.cuda.get_device_capability(device)),
        multiprocessors=props.multi_processor_count,
        git_head=command_output(["git", "rev-parse", "HEAD"]),
        source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source_files},
        nvidia_smi=command_output([
            "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,"
            "utilization.gpu,clocks.sm,clocks.mem,power.draw",
            "--format=csv,noheader,nounits"]),
    )


def percentile(values, fraction):
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(values):
    return dict(count=len(values), median=statistics.median(values),
                p95=percentile(values, .95), minimum=min(values), maximum=max(values))


def inputs(args, length, queries, device):
    pages = math.ceil(length / args.page_size)
    shape = (args.kv_heads * pages, args.page_size, args.head_dim)
    # Each length/query case has a fixed seed, independent of measurement order.
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + length * 17 + queries)
    dtype = getattr(torch, args.dtype)
    keys = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    values = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    qheads = args.kv_heads * args.q_per_kv
    query = torch.randn((qheads, queries, args.head_dim), device=device,
                        dtype=dtype, generator=generator)
    if args.physical_layout == "shuffled":
        slots = torch.randperm(args.kv_heads * pages, generator=generator,
                               device=device).reshape(args.kv_heads, pages)
    else:
        slots = torch.arange(args.kv_heads * pages, device=device).reshape(args.kv_heads, pages)
    lengths = torch.full((args.kv_heads, pages), args.page_size,
                         dtype=torch.int32, device=device)
    lengths[:, -1] = length - (pages - 1) * args.page_size
    positions = torch.arange(pages * args.page_size, device=device,
                             dtype=torch.int64).reshape(1, pages, args.page_size)
    positions = positions.expand(args.kv_heads, -1, -1).contiguous()
    # All repair queries are the last Nq positions; their causal/window policies
    # are therefore identical between arms but need not expose the same tokens.
    qpositions = torch.arange(length - queries, length, device=device,
                              dtype=torch.int64).expand(qheads, -1).contiguous()
    return query, keys, values, slots, lengths, positions, qpositions


def traffic_estimate(args, length, queries, local_heads):
    visible = 0
    for position in range(length - queries, length):
        causal_count = position + 1
        local_start = max(0, position - args.window + 1)
        sink_end = min(args.sink, causal_count)
        local_count = causal_count - local_start + min(sink_end, local_start)
        visible += args.q_per_kv * (
            local_heads * local_count + (args.kv_heads - local_heads) * causal_count)
    baseline = args.kv_heads * args.q_per_kv * queries * length
    element_bytes = torch.tensor([], dtype=getattr(torch, args.dtype)).element_size()
    return dict(
        scope="Analytic logical K/V row requests across query heads, not measured HBM bytes; "
              "excludes metadata, cache reuse, transaction granularity, and workspace.",
        baseline_logical_rows=baseline, policy_visible_logical_rows=visible,
        policy_invisible_fraction=1 - visible / baseline,
        baseline_logical_kv_bytes=baseline * 2 * args.head_dim * element_bytes,
        policy_visible_logical_kv_bytes=visible * 2 * args.head_dim * element_bytes,
    )


def compare(reference, optimized, args):
    torch.testing.assert_close(optimized, reference, atol=args.atol, rtol=args.rtol)
    a, b = reference.float().flatten(), optimized.float().flatten()
    delta = a - b
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0)
    return dict(max_absolute_error=delta.abs().max().item(),
                root_mean_square_error=delta.square().mean().sqrt().item(),
                cosine_similarity=cosine.item(),
                exact_element_fraction=(reference == optimized).float().mean().item(),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                reference="full-traversal attention with the same partition count; not an independent dense oracle",
                atol=args.atol, rtol=args.rtol)


def measure(call, start, end, device):
    # Public mode includes validation and its synchronization; neither mode
    # constructs manager descriptors or transfers new K/V inputs here.
    torch.cuda.synchronize(device)
    before = time.perf_counter_ns()
    start.record()
    output = call()
    end.record()
    end.synchronize()
    after = time.perf_counter_ns()
    return output, dict(cuda_stream_ms=start.elapsed_time(end),
                        synchronized_wall_ms=(after - before) / 1e6)


def run_case(args, operator, tensors, length, queries, ratio, device):
    local_heads = round(args.kv_heads * ratio)
    if not math.isclose(local_heads / args.kv_heads, ratio):
        raise ValueError(f"local ratio {ratio} cannot be represented by {args.kv_heads} KV heads")
    windows = torch.zeros(args.kv_heads, device=device, dtype=torch.int64)
    windows[:local_heads] = args.window
    sinks = torch.full((args.kv_heads,), args.sink, device=device, dtype=torch.int64)
    call_args = (*tensors, windows, sinks, args.q_per_kv, args.head_dim ** -.5, True)
    variants = dict(
        full_traversal=lambda: operator(*call_args, skip_invisible=False, num_splits=args.num_splits),
        skip_invisible=lambda: operator(*call_args, skip_invisible=True, num_splits=args.num_splits),
    )
    before_compile = time.perf_counter()
    first_reference = variants["full_traversal"]()
    first_optimized = variants["skip_invisible"]()
    torch.cuda.synchronize(device)
    preflight_seconds = time.perf_counter() - before_compile
    correctness = compare(first_reference, first_optimized, args)
    del first_reference, first_optimized
    for i in range(args.warmup):
        names = ("full_traversal", "skip_invisible") if i % 2 == 0 else ("skip_invisible", "full_traversal")
        for name in names:
            variants[name]()
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    raw = []
    for block in range(args.repeats):
        order = ("full_traversal", "skip_invisible", "skip_invisible", "full_traversal")
        if block % 2:
            order = ("skip_invisible", "full_traversal", "full_traversal", "skip_invisible")
        for ordinal, name in enumerate(order):
            value, timing = measure(variants[name], start, end, device)
            raw.append(dict(block=block, ordinal=ordinal, variant=name, **timing))
            del value
    final_correctness = compare(variants["full_traversal"](), variants["skip_invisible"](), args)
    pages = tensors[3].shape[1]
    row_count = args.kv_heads * args.q_per_kv * queries
    # Both entry modes have loaded this exact module by the first invocation.
    # Reuse the launcher's helper so reported geometry cannot silently drift.
    kernel_module = sys.modules["_redknot_segpaged_skip_benchmark.triton_attention"]
    num_splits = kernel_module.select_num_splits(pages, row_count, args.num_splits)
    legacy_splits = min(min(32, max(1, math.ceil(256 / row_count))),
                        max(1, math.ceil(pages / 8)))
    legacy_reference = operator(*call_args, skip_invisible=False, num_splits=legacy_splits)
    legacy_correctness = compare(legacy_reference, variants["skip_invisible"](), args)
    legacy_correctness["reference"] = "full traversal with legacy release partition heuristic; untimed"
    legacy_correctness["reference_num_splits"] = legacy_splits
    legacy_correctness["candidate_num_splits"] = num_splits
    del legacy_reference
    partial_records = row_count * num_splits if num_splits > 1 else 0
    result = dict(entry=args.entry, length_tokens=length, queries_per_head=queries,
                  physical_kv_heads=args.kv_heads, query_heads=args.kv_heads * args.q_per_kv,
                  local_kv_heads=local_heads, local_head_ratio=ratio,
                  window=args.window, sink=args.sink, head_dim=args.head_dim,
                  page_size=args.page_size, pages_per_kv_head=tensors[3].shape[1],
                  split_configuration=dict(num_splits=num_splits,
                                           pages_per_split=math.ceil(pages / num_splits),
                                           partial_records=partial_records,
                                           partial_tensor_bytes=partial_records * (args.head_dim + 2) * 4,
                                           default_partial_record_budget=32768,
                                           explicit_override_bypasses_record_budget=args.num_splits is not None,
                                           source=("same explicit override in both arms" if args.num_splits is not None
                                                   else "same shape-only finer partition heuristic in both arms")),
                  physical_layout=args.physical_layout, dtype=args.dtype,
                  warmup_calls_per_variant=args.warmup, abba_blocks=args.repeats,
                  preflight_compile_and_execution_seconds_excluded=preflight_seconds,
                  correctness_before=correctness, correctness_after=final_correctness,
                  legacy_partition_reference=legacy_correctness,
                  analytic_traffic=traffic_estimate(args, length, queries, local_heads),
                  raw_samples=raw, variants={})
    for name in variants:
        result["variants"][name] = {
            metric: summarize([r[metric] for r in raw if r["variant"] == name])
            for metric in ("cuda_stream_ms", "synchronized_wall_ms")}
    result["cuda_median_speedup"] = (
        result["variants"]["full_traversal"]["cuda_stream_ms"]["median"] /
        result["variants"]["skip_invisible"]["cuda_stream_ms"]["median"])
    result["wall_median_speedup"] = (
        result["variants"]["full_traversal"]["synchronized_wall_ms"]["median"] /
        result["variants"]["skip_invisible"]["synchronized_wall_ms"]["median"])
    block_speedups = []
    for block in range(args.repeats):
        rows = [r for r in raw if r["block"] == block]
        medians = {name: statistics.median(r["cuda_stream_ms"] for r in rows if r["variant"] == name)
                   for name in variants}
        block_speedups.append(medians["full_traversal"] / medians["skip_invisible"])
    result["paired_block_speedups"] = block_speedups
    result["paired_block_speedup_summary"] = summarize(block_speedups)
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=("smoke", "full"), default="full")
    parser.add_argument("--entry", choices=("prevalidated", "public"), default="prevalidated",
                        help="public includes descriptor validation and its CPU/GPU synchronization")
    parser.add_argument("--num-splits", type=int,
                        help="explicit Split-KV partition count in BOTH arms; default keeps launcher heuristic")
    parser.add_argument("--lengths", nargs="+", type=token_length)
    parser.add_argument("--queries", nargs="+", type=int)
    parser.add_argument("--local-ratios", nargs="+", type=float)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--q-per-kv", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--sink", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-layout", choices=("shuffled", "contiguous"), default="shuffled")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int, help="ABBA/BAAB blocks; two samples per variant per block")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--output", type=Path, default=Path("segpaged_page_skip_results.json"))
    args = parser.parse_args()
    smoke = args.preset == "smoke"
    if args.lengths is None:
        args.lengths = [4096] if smoke else [64*1024, 128*1024, 256*1024, 440*1024]
    if args.queries is None:
        args.queries = [1, 16]
    if args.local_ratios is None:
        args.local_ratios = [0., .75] if smoke else [0., .5, .75, 1.]
    if args.warmup is None:
        args.warmup = 2 if smoke else 3
    if args.repeats is None:
        args.repeats = 3 if smoke else 8
    if min(args.lengths + args.queries + [args.kv_heads, args.q_per_kv, args.head_dim, args.page_size, args.window]) < 1:
        parser.error("lengths, queries, head counts, dimensions, page size, and window must be positive")
    if min(args.lengths) < max(args.queries):
        parser.error("query count cannot exceed context length")
    if args.sink < 0 or args.warmup < 1 or args.repeats < 2:
        parser.error("sink must be nonnegative; warmup >= 1 and repeats >= 2")
    if args.num_splits is not None and not 1 <= args.num_splits <= 128:
        parser.error("num-splits must be in 1..128")
    if any(not 0 <= x <= 1 for x in args.local_ratios):
        parser.error("local ratios must lie between zero and one")
    return args


def main():
    args = arguments()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU; no CPU timing substitution is performed.")
    torch.cuda.set_device(device)
    operator = load_operator(args.entry)
    config = vars(args).copy()
    config["output"] = str(args.output.resolve())
    report = dict(
        schema_version=1, status="running", created_utc=datetime.now(timezone.utc).isoformat(),
        scope=("Single-GPU SegPaged prevalidated attention operator only. "
               if args.entry == "prevalidated" else
               "Single-GPU SegPaged public attention API including descriptor validation "
               "and its CPU/GPU synchronization. ") +
              "Neither model TTFT/QPS, full-model FLOPs, manager-side descriptor "
              "construction, nor measured HBM traffic is measured.",
        methodology="Same deterministic inputs and policy; tuned full traversal versus "
                    "invisible-page/tile skipping with identical current partition geometry; "
                    "alternating ABBA/BAAB blocks; CUDA-stream event and synchronized wall "
                    "samples; warmups excluded. Legacy partition numerical check is untimed.",
        config=config, environment=environment(device), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    checkpoint()
    try:
        with torch.inference_mode():
            for length in args.lengths:
                for queries in args.queries:
                    tensors = inputs(args, length, queries, device)
                    for ratio in args.local_ratios:
                        case = run_case(args, operator, tensors, length, queries, ratio, device)
                        report["cases"].append(case)
                        checkpoint()
                        old = case["variants"]["full_traversal"]["cuda_stream_ms"]["median"]
                        new = case["variants"]["skip_invisible"]["cuda_stream_ms"]["median"]
                        print(f"tokens={length} nq={queries} local={ratio:g}: "
                              f"full={old:.4f}ms skip={new:.4f}ms "
                              f"operator_speedup={case['cuda_median_speedup']:.3f}x "
                              f"max_abs={case['correctness_after']['max_absolute_error']:.3g}", flush=True)
                    del tensors
                    torch.cuda.synchronize(device)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = dict(type=type(exc).__name__, message=str(exc))
        checkpoint()
        raise
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    print(f"Saved {len(report['cases'])} measured operator cases to {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
