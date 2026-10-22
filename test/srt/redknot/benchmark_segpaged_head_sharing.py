#!/usr/bin/env python3
"""One-document, local-head-only sharing under an actually exhausted KV budget.

This is a seeded-tensor memory/admission experiment, not model inference or QPS.
Local KV heads share immutable source pages; global heads have different
per-request K/V. Both arms consume identical per-request payloads and use the
same page allocator, attention kernel and immutable partial-repair operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_segpaged_cache as common
from head_kv import CapacityError, HeadKVManager, HeadPagePool, ReuseProof, SegmentPatch, SegmentWrite
from head_kv.attention import paged_attention

CONTRACT = "synthetic-local-global-head-admission-v1"
CERTIFICATE = (
    "Synthetic unencoded K/V fixture: local heads are supplied reusable states; "
    "global heads are distinct seeded per-request states, not model-recomputed "
    "states. No real-model cross-context equivalence or quality certificate."
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)
    p.add_argument("--lengths", type=common.length_arg, nargs="+", default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--page-size", type=int, default=None)
    p.add_argument("--local-head-ratios", type=float, nargs="+", default=[0, 0.5, 0.75])
    p.add_argument("--prefix-tokens", type=int, default=8)
    p.add_argument("--query-tokens", type=int, default=1)
    p.add_argument("--cow-token-ratio", type=float, default=0.001,
                   help="contiguous initial document rows repaired on each local KV head")
    p.add_argument("--budget-requests", type=int, default=4,
                   help="default slab: this many independent full requests plus repair transaction reserve")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--capacity-pages", type=int)
    group.add_argument("--budget-gib", type=float)
    p.add_argument("--document-kv-gib", type=float,
                   help="derive document tokens from full-head K+V logical payload, rounded down to a full token")
    p.add_argument("--max-requests", type=int, default=64)
    p.add_argument("--fixed-requests", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--verify-requests", type=int, default=2)
    p.add_argument("--baseline-source-cache", choices=("none", "same-local"), default="none")
    p.add_argument("--reference-block-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("segpaged_head_sharing_results.json"))
    args = p.parse_args(argv)
    full = args.preset == "full"
    defaults = dict(heads=8 if full else 4, head_dim=64 if full else 16,
                    page_size=256 if full else 16, lengths=[65536] if full else [128],
                    dtype="float16" if args.device.startswith("cuda") else "float32")
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in ("heads", "head_dim", "page_size", "prefix_tokens", "query_tokens",
                 "budget_requests", "max_requests", "reference_block_size", "threads"):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    if args.verify_requests < 0 or any(n <= 0 for n in args.fixed_requests):
        p.error("verify-requests must be nonnegative; fixed-requests must be positive")
    if not 0 <= args.cow_token_ratio <= 1 or any(not 0 <= r <= 1 for r in args.local_head_ratios):
        p.error("ratios must be in [0, 1]")
    for name in ("budget_gib", "document_kv_gib"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error(f"{name} must be finite and positive")
    if args.capacity_pages is not None and args.capacity_pages <= 0:
        p.error("capacity-pages must be positive")
    if args.document_kv_gib is not None:
        bytes_per_token = 2 * args.heads * args.head_dim * torch.empty((), dtype=getattr(torch, args.dtype)).element_size()
        args.lengths = [int(args.document_kv_gib * 2 ** 30) // bytes_per_token]
    if any(n <= 0 for n in args.lengths):
        p.error("document lengths must be positive")
    return args


def make_fixtures(args, length, local_heads):
    generator = torch.Generator(device="cpu").manual_seed(args.seed + length)
    def rand(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device=args.device, dtype=getattr(torch, args.dtype))
    k = rand((local_heads, length, args.head_dim))
    v = rand((local_heads, length, args.head_dim))
    digest = hashlib.sha256()
    for value in (k, v):
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return dict(local_k=k, local_v=v, local_heads=local_heads, heads=args.heads,
                length=length, sha256=digest.hexdigest())


def request_payload(fixtures, args, index):
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 104729 * (index + 1) + fixtures["length"])
    def rand(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device=args.device, dtype=getattr(torch, args.dtype))
    global_heads = args.heads - fixtures["local_heads"]
    prefix_k = rand((args.heads, args.prefix_tokens, args.head_dim))
    prefix_v = rand((args.heads, args.prefix_tokens, args.head_dim))
    global_k = rand((global_heads, fixtures["length"], args.head_dim))
    global_v = rand((global_heads, fixtures["length"], args.head_dim))
    query = rand((args.heads, args.query_tokens, args.head_dim))
    rows = math.ceil(fixtures["length"] * args.cow_token_ratio)
    patches = tuple(SegmentPatch((0, h, "document"), tuple(range(rows)),
                                 fixtures["local_k"][h, :rows] + (index + 1) * 0.125,
                                 fixtures["local_v"][h, :rows] - (index + 1) * 0.25,
                                 f"synthetic-local-repair-{index}")
                    for h in range(fixtures["local_heads"]) if rows)
    return dict(prefix_k=prefix_k, prefix_v=prefix_v, global_k=global_k, global_v=global_v,
                query=query, patches=patches, repaired_rows=rows, index=index)


def build_local_cache(manager, fixtures, args):
    if fixtures["local_heads"] == 0:
        return None
    manager.create_request("publisher", context_id="offline-document", contract=CONTRACT)
    manager.update("publisher", writes=tuple(SegmentWrite(
        (0, h, "document"), fixtures["local_k"][h], fixtures["local_v"][h],
        tuple(range(fixtures["length"])), "offline-local-fixture")
        for h in range(fixtures["local_heads"])))
    version = manager.version("publisher")
    manager.cache("publisher", "local-document")
    manager.release_request("publisher")
    return version


def populate_request(manager, fixtures, args, backend, request_id, index, payload):
    if backend not in ("head_shared", "independent"):
        raise ValueError("unknown backend")
    context = f"request-context-{index}"
    manager.create_request(request_id, context_id=context, contract=CONTRACT)
    # Different offsets deliberately exercise non-prefix positional metadata.
    # The gap is logical only; synthetic unencoded K has no RoPE relocation.
    offset = args.prefix_tokens + index % 3
    positions = tuple(range(offset, offset + fixtures["length"]))
    writes = [SegmentWrite((0, h, "prefix"), payload["prefix_k"][h], payload["prefix_v"][h],
                           tuple(range(args.prefix_tokens)), f"prefix-{index}")
              for h in range(args.heads)]
    for h in range(args.heads):
        if h < fixtures["local_heads"]:
            if backend == "head_shared":
                continue
            k, v = fixtures["local_k"][h], fixtures["local_v"][h]
            provenance = "offline-local-fixture"
        else:
            k, v = payload["global_k"][h - fixtures["local_heads"]], payload["global_v"][h - fixtures["local_heads"]]
            provenance = f"request-global-{index}"
        writes.append(SegmentWrite((0, h, "document"), k, v, positions, provenance,
                                   reuse_kind="policy_approximate", validity_certificate=CERTIFICATE))
    manager.update(request_id, writes=tuple(writes))
    if backend == "head_shared":
        for h in range(fixtures["local_heads"]):
            manager.share_segment("local-document", request_id, (0, h, "document"), (0, h, "document"),
                                  source_is_cache=True, positions=positions,
                                  proof=ReuseProof("policy_approximate", "offline-local-fixture", context, CERTIFICATE))
    before_cow = common.stats_snapshot(manager)
    _, repair = common.timed(args.device, lambda: manager.update(request_id, patches=payload["patches"])
                             if payload["patches"] else None)
    after_cow = common.stats_snapshot(manager)
    return dict(request_id=request_id, index=index, document_position_start=offset,
                final_position=positions[-1], before_cow=before_cow, after_cow=after_cow,
                repair_wall_ms=repair["wall_ms"],
                repair_copied_bytes=after_cow["copied_bytes"] - before_cow["copied_bytes"],
                repair_written_bytes=after_cow["written_bytes"] - before_cow["written_bytes"])


def reference_attention(fixtures, args, payload):
    query = payload["query"].float()
    peak = torch.full(query.shape[:2], -torch.inf, device=query.device)
    denominator = torch.zeros_like(peak)
    numerator = torch.zeros_like(query)
    def consume(k, v):
        nonlocal peak, denominator, numerator
        scores = torch.matmul(query, k.float().transpose(-1, -2)) * args.head_dim ** -0.5
        next_peak = torch.maximum(peak, scores.amax(dim=-1))
        old_weight = torch.exp(peak - next_peak)
        weights = torch.exp(scores - next_peak[..., None])
        numerator = numerator * old_weight[..., None] + torch.matmul(weights, v.float())
        denominator = denominator * old_weight + weights.sum(dim=-1)
        peak = next_peak
    consume(payload["prefix_k"], payload["prefix_v"])
    for begin in range(0, fixtures["length"], args.reference_block_size):
        end = min(fixtures["length"], begin + args.reference_block_size)
        k = torch.cat((fixtures["local_k"][:, begin:end], payload["global_k"][:, begin:end]), dim=0)
        v = torch.cat((fixtures["local_v"][:, begin:end], payload["global_v"][:, begin:end]), dim=0)
        count = max(0, min(end, payload["repaired_rows"]) - begin)
        if count and fixtures["local_heads"]:
            k[:fixtures["local_heads"], :count] += (payload["index"] + 1) * 0.125
            v[:fixtures["local_heads"], :count] -= (payload["index"] + 1) * 0.25
        consume(k, v)
    return (numerator / denominator[..., None]).to(payload["query"].dtype)


def verify_request(manager, fixtures, args, plan, payload):
    with manager.bind(plan["request_id"]) as lease:
        descriptor = lease.descriptor(0, args.heads)
        qpos = torch.arange(plan["final_position"], plan["final_position"] + args.query_tokens,
                           device=args.device, dtype=torch.int64)
        actual = paged_attention(payload["query"], **descriptor, query_positions=qpos)
        expected = reference_attention(fixtures, args, payload)
        atol = 5e-3 if args.dtype == "bfloat16" else 8e-4 if args.dtype == "float16" else 2e-5
        if not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
            raise AssertionError("nonfinite attention output")
        error = float((actual.float() - expected.float()).abs().max())
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=atol)
        kv_error = 0.0
        for h in range(args.heads):
            got_k, got_v = lease.gather((0, h, "document"))
            if h < fixtures["local_heads"]:
                expected_k, expected_v = fixtures["local_k"][h].clone(), fixtures["local_v"][h].clone()
                n = payload["repaired_rows"]
                expected_k[:n] += (payload["index"] + 1) * 0.125
                expected_v[:n] -= (payload["index"] + 1) * 0.25
            else:
                expected_k, expected_v = payload["global_k"][h - fixtures["local_heads"]], payload["global_v"][h - fixtures["local_heads"]]
            for actual_kv, expected_kv in ((got_k, expected_k), (got_v, expected_v)):
                if not bool(torch.isfinite(actual_kv).all()):
                    raise AssertionError("nonfinite K/V payload")
                kv_error = max(kv_error, float((actual_kv - expected_kv).abs().max()))
        if kv_error:
            raise AssertionError(f"K/V mismatch {kv_error}")
    return dict(attention_max_abs_error=error, kv_max_abs_error=kv_error, atol=atol, rtol=atol)


def check_local_source(manager, fixtures, version):
    if version is None:
        return 0.0
    error = 0.0
    for h in range(fixtures["local_heads"]):
        segment = version.segments[(0, h, "document")]
        for page, ref in enumerate(segment.pages):
            begin = page * manager.pool.page_size
            extent = min(manager.pool.page_size, fixtures["length"] - begin)
            k, v = manager.pool.read_page(ref, extent)
            for got, source in ((k, fixtures["local_k"]), (v, fixtures["local_v"])):
                if not bool(torch.isfinite(got).all()):
                    raise AssertionError("nonfinite local source")
                error = max(error, float((got - source[h, begin:begin + extent]).abs().max()))
    if error:
        raise AssertionError("local source changed after COW")
    return error


def cleanup(manager):
    for name in list(manager._requests):
        manager.release_request(name)
    for key in list(manager._cache):
        manager.evict(key)
    common.synchronize(manager.pool.device)
    manager.pool.collect()


def run_case(args, length, local_ratio):
    local_heads = math.floor(args.heads * local_ratio)
    fixtures = make_fixtures(args, length, local_heads)
    element_size = fixtures["local_k"].element_size()
    document_bytes = 2 * args.heads * length * args.head_dim * element_size
    page_bytes = 2 * args.page_size * args.head_dim * element_size
    full_request_pages = args.heads * (math.ceil(length / args.page_size) + math.ceil(args.prefix_tokens / args.page_size))
    transaction_reserve = args.heads * math.ceil(math.ceil(length * args.cow_token_ratio) / args.page_size)
    capacity = args.capacity_pages or (int(args.budget_gib * 2 ** 30) // page_bytes if args.budget_gib is not None
                                       else args.budget_requests * full_request_pages + transaction_reserve)
    if capacity <= 0:
        raise ValueError("budget cannot hold a single page")
    arms = []
    # Each arm owns a single equal-budget slab, released before the next arm.
    # Admission/memory are exact counts, not a cross-arm wall-time comparison.
    for backend in ("independent", "head_shared"):
        pool = HeadPagePool(capacity, args.page_size, args.head_dim,
                            dtype=getattr(torch, args.dtype), device=args.device)
        manager = HeadKVManager(pool)
        steps, verified = [], []
        admitted = []
        source_version = None
        termination = "max_requests_guard"
        source_publish = dict(wall_ms=0.0, host_submit_ms=0.0)
        try:
            try:
                if backend == "head_shared" or args.baseline_source_cache == "same-local":
                    source_version, source_publish = common.timed(args.device, lambda: build_local_cache(manager, fixtures, args))
            except CapacityError as exc:
                termination = "source_cache_capacity_error"
                cleanup(manager)
                source_failure = str(exc)
            else:
                source_failure = None
            cached = common.stats_snapshot(manager)
            if source_failure is None:
                for index in range(args.max_requests):
                    name = f"request-{index}"
                    payload = request_payload(fixtures, args, index)
                    before = common.stats_snapshot(manager)
                    try:
                        plan, elapsed = common.timed(args.device, lambda: populate_request(
                            manager, fixtures, args, backend, name, index, payload))
                    except CapacityError as exc:
                        if name in manager._requests:
                            manager.release_request(name)
                        common.synchronize(args.device)
                        after = common.stats_snapshot(manager)
                        for key in ("allocated_bytes", "requests", "cached_roots", "pending_transactions",
                                    "pinned_pages", "reserved_pages", "retiring_pages"):
                            if after[key] != before[key]:
                                raise AssertionError(f"failed admission leaked {key}: {before[key]} -> {after[key]}")
                        termination = "capacity_error"
                        steps.append(dict(index=index, admitted=False, error=str(exc), before=before, after=after))
                        break
                    admitted.append(name)
                    current = common.stats_snapshot(manager)
                    version = manager.version(name)
                    local_alias = 0
                    if source_version is not None:
                        source_refs = {ref for s in source_version.segments.values() for ref in s.pages}
                        local_alias = sum(ref in source_refs for h in range(local_heads)
                                          for ref in version.segments[(0, h, "document")].pages)
                    global_refs = {ref for h in range(local_heads, args.heads)
                                   for ref in version.segments[(0, h, "document")].pages}
                    older_global = {ref for old in admitted[:-1] for h in range(local_heads, args.heads)
                                    for ref in manager.version(old).segments[(0, h, "document")].pages}
                    if global_refs & older_global:
                        raise AssertionError("global-head pages shared across requests")
                    source_refs = ({ref for segment in source_version.segments.values() for ref in segment.pages}
                                   if source_version is not None else set())
                    prefix_refs, all_global_refs, all_local_refs = set(), set(), set()
                    for live_name in admitted:
                        for (_, head, occurrence), segment in manager.version(live_name).segments.items():
                            target = (prefix_refs if occurrence == "prefix" else
                                      all_local_refs if head < local_heads else all_global_refs)
                            target.update(segment.pages)
                    owned_refs = source_refs | prefix_refs | all_global_refs | all_local_refs
                    accounting = dict(source_cache_bytes=len(source_refs) * page_bytes,
                                      private_prefix_bytes=len(prefix_refs) * page_bytes,
                                      private_global_bytes=len(all_global_refs) * page_bytes,
                                      private_local_bytes=len(all_local_refs - source_refs) * page_bytes,
                                      unique_owned_page_bytes=len(owned_refs) * page_bytes)
                    if accounting["unique_owned_page_bytes"] != current["allocated_bytes"]:
                        raise AssertionError("page-category accounting disagrees with actual active slab pages")
                    step = dict(index=index, admitted=True, resident_requests=len(admitted),
                                materialize_and_repair=elapsed, repair_wall_ms=plan["repair_wall_ms"],
                                repair_copied_bytes=plan["repair_copied_bytes"], repair_written_bytes=plan["repair_written_bytes"],
                                local_pages_still_alias_source=local_alias,
                                private_global_pages=len(global_refs), before=before, after=current,
                                active_page_accounting=accounting,
                                is_fixed_request_snapshot=len(admitted) in args.fixed_requests)
                    steps.append(step)
                    if index < args.verify_requests:
                        verified.append(dict(request_index=index, **verify_request(manager, fixtures, args, plan, payload)))
            source_error = check_local_source(manager, fixtures, source_version) if source_failure is None else None
            resident = common.stats_snapshot(manager)
            for name in admitted:
                manager.release_request(name)
            common.synchronize(args.device)
            released = common.stats_snapshot(manager)
            if released["allocated_bytes"] != cached["allocated_bytes"] or released["requests"]:
                raise AssertionError("request release did not return to source-cache footprint")
            cleanup(manager)
            final = common.stats_snapshot(manager)
            if final["allocated_bytes"] or final["pinned_pages"] or final["cached_roots"]:
                raise AssertionError("final pages/pins/cache not reclaimed")
            arms.append(dict(backend=backend, source_cache_policy="local_heads_only" if source_version else "none",
                             offline_publish=source_publish, source_cache=cached, source_publish_failure=source_failure,
                             admission_steps=steps, admitted_requests=len(admitted), termination=termination,
                             admission_is_censored=termination == "max_requests_guard",
                             resident_state=resident, after_request_release=released, final_state=final,
                             correctness=dict(verified_requests=verified, source_max_abs_error=source_error,
                                              model_quality_evaluated=False)))
        finally:
            cleanup(manager)
        del manager, pool
    independent, shared = arms
    matched = []
    for count in args.fixed_requests:
        snapshots = {}
        for arm in arms:
            snapshots[arm["backend"]] = next((step for step in arm["admission_steps"]
                                               if step["admitted"] and step["resident_requests"] == count), None)
        pair = dict(resident_requests=count, both_admitted=all(snapshots.values()))
        if pair["both_admitted"]:
            baseline_bytes = snapshots["independent"]["after"]["allocated_bytes"]
            shared_bytes = snapshots["head_shared"]["after"]["allocated_bytes"]
            pair.update(independent_active_bytes=baseline_bytes, head_shared_active_bytes=shared_bytes,
                        active_page_saving_fraction=1 - shared_bytes / baseline_bytes)
        matched.append(pair)
    return dict(document_tokens=length, document_full_head_logical_bytes=document_bytes,
                document_full_head_logical_gib=document_bytes / 2 ** 30,
                requested_document_kv_gib=args.document_kv_gib, fixture_sha256=fixtures["sha256"],
                config=dict(heads=args.heads, local_heads=local_heads, global_heads=args.heads - local_heads,
                            local_head_ratio_requested=local_ratio, local_head_ratio_actual=local_heads / args.heads,
                            head_dim=args.head_dim, page_size=args.page_size, capacity_pages=capacity,
                            reserved_bytes=capacity * page_bytes, full_request_pages=full_request_pages,
                            transaction_reserve_pages=transaction_reserve, cow_token_ratio=args.cow_token_ratio,
                            prefix_tokens=args.prefix_tokens,
                            local_document_logical_bytes=document_bytes * local_heads // args.heads,
                            global_document_logical_bytes_per_request=document_bytes * (args.heads - local_heads) // args.heads),
                arms=arms,
                summary=dict(independent_admitted=independent["admitted_requests"],
                             head_shared_admitted=shared["admitted_requests"],
                             observed_resident_request_ratio=(shared["admitted_requests"] / independent["admitted_requests"]
                                                              if independent["admitted_requests"] else None),
                             any_censored=any(arm["admission_is_censored"] for arm in arms),
                             fixed_request_comparisons=matched,
                             throughput_or_qps_measured=False))


def benchmark(args):
    torch.set_num_threads(args.threads)
    files = [Path(__file__).resolve(), Path(common.__file__).resolve(),
             *sorted((common.REPO / "python/sglang/srt/mem_cache/head_kv").glob("*.py"))]
    result = dict(schema_version=1, status="running", benchmark="local_head_shared_document_admission",
                  scope="real seeded tensor storage/admission; no model TTFT, FLOPs saving, or concurrent QPS",
                  contract=CERTIFICATE,
                  protocol=dict(source_cache="head_shared caches only local KV heads; baseline default no source cache",
                                private_state="global K/V and prefixes differ per request; identical seeds across arms",
                                budget="equal bounded slab; physically reserve/write each request until CapacityError",
                                guard="max_requests termination is censored, not a measured capacity limit",
                                admission="capacity includes transient immutable repair reservations, not only final live pages",
                                fixture="source tensors, per-request payload and verification temporaries excluded from bounded KV slab",
                                timing="per-admission component wall times include writes/metadata/COW, exclude seeded payload generation; not a model speed claim",
                                position="keys unencoded; different logical offsets do not prove real RoPE/context reuse"),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  environment=common.environment(args), cases=[],
                  source_sha256={str(f.relative_to(common.REPO)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    try:
        for length in args.lengths:
            for ratio in args.local_head_ratios:
                result["cases"].append(run_case(args, length, ratio))
                save()
                print(f"[head-sharing] tokens={length} local_ratio={ratio} {result['cases'][-1]['summary']}", flush=True)
    except BaseException as exc:
        result.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        save()
        raise
    result["status"] = "complete"
    save()
    return result


def main(argv=None):
    args = parse_args(argv)
    benchmark(args)
    print(f"[head-sharing] complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
