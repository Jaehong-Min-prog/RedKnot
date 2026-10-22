#!/usr/bin/env python3
"""Persistent non-prefix KV-page benchmark, not an LLM serving benchmark.

Seeded synthetic K/V are published once, retained after the publishing request
dies, and consumed by sequential warm batches. A batch has several *live*
requests; its attention calls execute sequentially on one stream. No number in
this report is model TTFT, concurrent serving QPS, or an accuracy certificate.

Both arms use the same bounded slab, immutable pages, transactional repair and
attention implementation. Only cache-hit attachment differs: shared references
versus independent physical copies. This deliberately does not handicap the
copy arm with a full-cache clone during a partial repair.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import statistics
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/sglang/srt/mem_cache"))
from head_kv import HeadKVManager, HeadPagePool, ReuseProof, SegmentPatch, SegmentWrite
from head_kv.attention import paged_attention

CONTRACT = "synthetic-unencoded-kv-fixture-v1"
CERTIFICATE = (
    "synthetic fixture only: K has no RoPE; copied and shared arms consume the "
    "same supplied K/V. This does not certify cross-context reuse in a model."
)


def length_arg(text):
    text = str(text).upper()
    return int(text[:-1]) * 1024 if text.endswith("K") else int(text)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)
    p.add_argument("--lengths", type=length_arg, nargs="+", default=None,
                   help="total document KV tokens, excluding the private prefix; accepts 64K")
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--page-size", type=int, default=None)
    p.add_argument("--documents", type=int, default=4)
    p.add_argument("--prefix-tokens", type=int, default=8)
    p.add_argument("--query-tokens", type=int, default=1)
    p.add_argument("--requests", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--shared-ratio", type=float, default=0.75,
                   help="fraction of document identities eligible for sharing; floor(D*r)")
    p.add_argument("--repair-head-ratio", type=float, default=0.25)
    p.add_argument("--repair-token-ratio", type=float, default=0.01,
                   help="contiguous rows repaired within document 0, per selected KV head")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--reference-block-size", type=int, default=4096)
    p.add_argument("--output", type=Path, default=Path("segpaged_cache_results.json"))
    args = p.parse_args(argv)
    full = args.preset == "full"
    defaults = dict(lengths=[4096, 16384] if full else [256], heads=4 if full else 2,
                    head_dim=64 if full else 32, page_size=256 if full else 64,
                    dtype="float16" if args.device.startswith("cuda") else "float32")
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in ("heads", "head_dim", "page_size", "documents", "prefix_tokens",
                 "query_tokens", "threads", "reference_block_size", "repeats"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.documents < 2:
        p.error("at least two documents are needed for non-prefix reorder")
    if any(n < args.documents for n in args.lengths) or any(n <= 0 for n in args.requests):
        p.error("lengths must cover all documents and request counts must be positive")
    if args.warmup < 0:
        p.error("--warmup must be nonnegative")
    for name in ("shared_ratio", "repair_head_ratio", "repair_token_ratio"):
        if not 0 <= getattr(args, name) <= 1:
            p.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def timed(device, fn):
    synchronize(device)
    start = time.perf_counter_ns()
    value = fn()
    submitted = time.perf_counter_ns()
    synchronize(device)
    end = time.perf_counter_ns()
    return value, {"wall_ms": (end - start) / 1e6,
                   "host_submit_ms": (submitted - start) / 1e6}


def make_fixtures(args, length, generator=None):
    generator = generator or torch.Generator(device="cpu").manual_seed(args.seed + length)
    dtype = getattr(torch, args.dtype)
    sizes = [length // args.documents + (i < length % args.documents)
             for i in range(args.documents)]
    def rand(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device=args.device, dtype=dtype)
    documents = [(rand((args.heads, n, args.head_dim)), rand((args.heads, n, args.head_dim)))
                 for n in sizes]
    prefixes = [(rand((args.heads, args.prefix_tokens, args.head_dim)),
                 rand((args.heads, args.prefix_tokens, args.head_dim)))
                for _ in range(max(args.requests))]
    queries = [rand((args.heads, args.query_tokens, args.head_dim))
               for _ in range(max(args.requests))]
    digest = hashlib.sha256()
    # Hash the actual (possibly rounded) tensors. The fixtures are not timed.
    for tensor in [t for pair in documents + prefixes for t in pair] + queries:
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return dict(documents=documents, prefixes=prefixes, queries=queries,
                sizes=sizes, sha256=digest.hexdigest(), document_tokens=length)


def build_cached_documents(manager, fixtures, args):
    """Publish each source once and drop the source request, not the cache root."""
    versions = {}
    for doc, (k, v) in enumerate(fixtures["documents"]):
        source = f"publisher-{doc}"
        manager.create_request(source, context_id=f"offline-document-{doc}", contract=CONTRACT)
        manager.update(source, writes=tuple(
            SegmentWrite((0, h, f"doc-{doc}"), k[h], v[h], tuple(range(len(k[h]))),
                         f"fixture-document-{doc}", position_basis="none")
            for h in range(args.heads)))
        versions[doc] = manager.version(source)
        manager.cache(source, f"cache-document-{doc}")
        manager.release_request(source)
    return versions


def request_plan(fixtures, args, request_id, request_index):
    # Deliberately rotated even for request zero. Each doc follows a different
    # private prefix; shared documents are never the request's prefix.
    shift = (request_index + 1) % args.documents
    order = list(range(shift, args.documents)) + list(range(shift))
    if request_index % 2:
        order.reverse()
    offset = args.prefix_tokens
    positions = {}
    for doc in order:
        positions[doc] = tuple(range(offset, offset + fixtures["sizes"][doc]))
        offset += fixtures["sizes"][doc]
    return dict(request_id=request_id, request_index=request_index, order=order,
                positions=positions, context_id=f"online-{request_id}",
                final_position=offset - 1)


def populate_request(manager, fixtures, args, backend, request_id, request_index):
    """Attach one non-prefix request; useful as a direct lifecycle test hook."""
    if backend not in ("shared_pages", "independent_pages"):
        raise ValueError("unknown backend")
    plan = request_plan(fixtures, args, request_id, request_index)
    manager.create_request(request_id, context_id=plan["context_id"], contract=CONTRACT)
    prefix_k, prefix_v = fixtures["prefixes"][request_index]
    writes = [SegmentWrite((0, h, "private-prefix"), prefix_k[h], prefix_v[h],
                           tuple(range(args.prefix_tokens)), f"private-prefix-{request_index}")
              for h in range(args.heads)]
    nshared = math.floor(args.documents * args.shared_ratio)
    for doc in plan["order"]:
        if backend == "shared_pages" and doc < nshared:
            continue
        k, v = fixtures["documents"][doc]
        writes.extend(SegmentWrite((0, h, f"doc-{doc}"), k[h], v[h], plan["positions"][doc],
                                   f"fixture-document-{doc}", position_basis="none",
                                   reuse_kind="policy_approximate", validity_certificate=CERTIFICATE)
                      for h in range(args.heads))
    manager.update(request_id, writes=tuple(writes))
    if backend == "shared_pages":
        for doc in plan["order"]:
            if doc >= nshared:
                continue
            proof = ReuseProof("policy_approximate", f"fixture-document-{doc}",
                               plan["context_id"], CERTIFICATE)
            for h in range(args.heads):
                key = (0, h, f"doc-{doc}")
                manager.share_segment(f"cache-document-{doc}", request_id, key, key,
                                      proof=proof, positions=plan["positions"][doc],
                                      source_is_cache=True)
    return plan


def repair_patches(fixtures, args):
    if args.repair_head_ratio == 0 or args.repair_token_ratio == 0:
        return (), 0, 0
    k, v = fixtures["documents"][0]
    nh = min(args.heads, max(1, math.ceil(args.heads * args.repair_head_ratio)))
    rows = min(len(k[0]), max(1, math.ceil(len(k[0]) * args.repair_token_ratio)))
    indices = tuple(range(rows))
    patches = tuple(SegmentPatch((0, h, "doc-0"), indices, k[h, :rows] + 0.125,
                                 v[h, :rows] - 0.25, "synthetic-repair-v1")
                    for h in range(nh))
    return patches, nh, rows


def reference_attention(fixtures, args, plan, repaired_heads, repaired_rows):
    """Independent FP32 blockwise softmax from fixtures, not manager pages.

    All queries are at/after the last key and global, hence all K/V are visible.
    Repair values are rounded to fixture dtype exactly as stored by the manager.
    Memory is O(H * query_tokens * block_size), not quadratic context length.
    """
    query = fixtures["queries"][plan["request_index"]].float()
    shape = query.shape[:2]
    peak = torch.full(shape, -torch.inf, device=query.device)
    denom = torch.zeros(shape, device=query.device)
    numerator = torch.zeros_like(query)
    segments = [(-1, fixtures["prefixes"][plan["request_index"]])]
    segments.extend((doc, fixtures["documents"][doc]) for doc in plan["order"])
    for doc, (full_k, full_v) in segments:
        for begin in range(0, full_k.shape[1], args.reference_block_size):
            end = min(full_k.shape[1], begin + args.reference_block_size)
            k, v = full_k[:, begin:end], full_v[:, begin:end]
            if doc == 0 and begin < repaired_rows:
                k, v = k.clone(), v.clone()
                count = min(end, repaired_rows) - begin
                k[:repaired_heads, :count] = k[:repaired_heads, :count] + 0.125
                v[:repaired_heads, :count] = v[:repaired_heads, :count] - 0.25
            scores = torch.matmul(query, k.float().transpose(-1, -2)) * args.head_dim ** -0.5
            new_peak = torch.maximum(peak, scores.amax(dim=-1))
            old_weight = torch.exp(peak - new_peak)
            weights = torch.exp(scores - new_peak[..., None])
            numerator = numerator * old_weight[..., None] + torch.matmul(weights, v.float())
            denom = denom * old_weight + weights.sum(dim=-1)
            peak = new_peak
    return (numerator / denom[..., None]).to(fixtures["queries"][0].dtype)


def stats_snapshot(manager):
    stats = manager.stats()
    stats["reserved_bytes"] = stats["reserved_slab_bytes"]
    stats["actual_slab_storage_bytes"] = (manager.pool.k.untyped_storage().nbytes()
                                           + manager.pool.v.untyped_storage().nbytes())
    assert stats["reserved_bytes"] == stats["actual_slab_storage_bytes"]
    return stats


def validate_cache(manager, fixtures, args, cached_versions):
    error = 0.0
    for doc, version in cached_versions.items():
        # Borrow cached refs through a temporary exact-context reader, then
        # release it. This verification is outside every measured interval.
        name = f"verify-document-{doc}"
        manager.create_request(name, context_id=version.context_id, contract=CONTRACT)
        try:
            for h in range(args.heads):
                key = (0, h, f"doc-{doc}")
                manager.share_segment(f"cache-document-{doc}", name, key, key,
                                      proof=ReuseProof("exact_context", f"fixture-document-{doc}",
                                                       version.context_id), source_is_cache=True)
            with manager.bind(name) as lease:
                for h in range(args.heads):
                    got = lease.gather((0, h, f"doc-{doc}"))
                    for value, expected in zip(got, fixtures["documents"][doc]):
                        if not bool(torch.isfinite(value).all()):
                            raise AssertionError("nonfinite cached source")
                        error = max(error, float((value.float() - expected[h].float()).abs().max()))
        finally:
            manager.release_request(name)
    if error != 0:
        raise AssertionError(f"cached source changed after COW: {error}")
    return error


def measure_batch(manager, fixtures, args, backend, count, repetition, phase,
                  cached_versions, expected_outputs):
    leases = []
    try:
        return _measure_batch(manager, fixtures, args, backend, count, repetition, phase,
                              cached_versions, expected_outputs, leases)
    finally:
        for lease in leases:
            lease.complete()
        for i in range(count):
            name = f"{phase}-{count}-{repetition}-{i}"
            try:
                manager.version(name)
            except KeyError:
                continue
            manager.release_request(name)
        synchronize(args.device)
        manager.pool.collect()


def _measure_batch(manager, fixtures, args, backend, count, repetition, phase,
                   cached_versions, expected_outputs, leases):
    names = [f"{phase}-{count}-{repetition}-{i}" for i in range(count)]
    before = stats_snapshot(manager)
    plans, attach = timed(args.device, lambda: [populate_request(
        manager, fixtures, args, backend, name, i) for i, name in enumerate(names)])
    after_attach = stats_snapshot(manager)
    cache_refs = {ref for version in cached_versions.values()
                  for segment in version.segments.values() for ref in segment.pages}
    request_refs = {ref for name in names for segment in manager.version(name).segments.values()
                    for ref in segment.pages}
    alias_counts = dict(shared_page_refs=len(cache_refs & request_refs),
                        independent_page_refs=len(request_refs - cache_refs))
    patches, nh, nr = repair_patches(fixtures, args)
    _, repair = timed(args.device, lambda: [manager.update(name, patches=patches)
                                           for name in names] if patches else None)
    after_repair = stats_snapshot(manager)
    def descriptors():
        result = []
        for name in names:
            lease = manager.bind(name)
            leases.append(lease)
            result.append(lease.descriptor(0, args.heads))
        return result
    descriptors_value, descriptor_time = timed(args.device, descriptors)
    qpositions = [torch.arange(plan["final_position"], plan["final_position"] + args.query_tokens,
                               device=args.device, dtype=torch.int64) for plan in plans]
    outputs, attention = timed(args.device, lambda: [paged_attention(
        fixtures["queries"][i], **descriptor, query_positions=qpositions[i], backend="auto")
        for i, descriptor in enumerate(descriptors_value)])
    errors = []
    atol = 5e-3 if args.dtype == "bfloat16" else 8e-4 if args.dtype == "float16" else 2e-5
    for i, (plan, output) in enumerate(zip(plans, outputs)):
        if i not in expected_outputs:
            expected_outputs[i] = reference_attention(fixtures, args, plan, nh, nr)
        expected = expected_outputs[i]
        if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(expected).all()):
            raise AssertionError("nonfinite attention output/reference")
        error = float((output.float() - expected.float()).abs().max())
        if not torch.allclose(output.float(), expected.float(), atol=atol, rtol=atol):
            raise AssertionError(f"attention mismatch {backend} request={i} error={error}")
        errors.append(error)
    # Inspect repaired rows using real pool pages, not the generated reference.
    kv_error = 0.0
    for lease in leases:
        for patch in patches:
            seg = lease.version.segments[patch.key]
            for row in range(len(patch.indices)):
                token = patch.indices[row]
                ref = seg.pages[token // args.page_size]
                k, v = manager.pool.read_page(ref, min(args.page_size,
                                            seg.length - token // args.page_size * args.page_size))
                offset = token % args.page_size
                if not bool(torch.isfinite(k[offset]).all() & torch.isfinite(v[offset]).all()):
                    raise AssertionError("nonfinite repaired K/V")
                kv_error = max(kv_error, float((k[offset] - patch.k[row]).abs().max()),
                               float((v[offset] - patch.v[row]).abs().max()))
    if kv_error != 0:
        raise AssertionError(f"repaired K/V mismatch: {kv_error}")
    for lease in leases:
        lease.complete()
    _, release = timed(args.device, lambda: [manager.release_request(name) for name in names])
    after_release = stats_snapshot(manager)
    if after_release["requests"] or after_release["allocated_bytes"] != before["allocated_bytes"]:
        raise AssertionError("request release did not return to persistent cache footprint")
    return dict(phase=phase, repetition=repetition, live_requests=count,
                request_document_orders=[plan["order"] for plan in plans],
                prefix_tokens=args.prefix_tokens, nonprefix_document_occurrences=count * args.documents,
                attach=attach, repair=repair, descriptor=descriptor_time,
                attention_batch=attention, release=release,
                attach_written_bytes=after_attach["written_bytes"] - before["written_bytes"],
                attach_copied_bytes=after_attach["copied_bytes"] - before["copied_bytes"],
                repair_copied_bytes=after_repair["copied_bytes"] - after_attach["copied_bytes"],
                repair_written_bytes=after_repair["written_bytes"] - after_attach["written_bytes"],
                repaired_heads=nh, repaired_rows_per_head=nr, before=before,
                after_attach=after_attach, after_repair=after_repair, after_release=after_release,
                alias_counts=alias_counts,
                correctness=dict(kv_max_abs_error=kv_error, attention_max_abs_error=max(errors),
                                 attention_error_by_request=errors, atol=atol, rtol=atol),
                attention_batch_serial_calls=count,
                rate_note="batch duration is sequential component execution; not concurrent model QPS")


def run_case(args, length, generator=None):
    fixtures = make_fixtures(args, length, generator)
    document_pages = sum(math.ceil(n / args.page_size) for n in fixtures["sizes"]) * args.heads
    prefix_pages = math.ceil(args.prefix_tokens / args.page_size) * args.heads
    patches, nh, nr = repair_patches(fixtures, args)
    repair_pages = nh * math.ceil(nr / args.page_size)
    # Identical capacity for both arms; includes cache roots, all independent
    # requests and one immutable repair transaction's temporary destination.
    capacity = document_pages + max(args.requests) * (document_pages + prefix_pages) + repair_pages + 8
    arms = []
    expected_outputs = {}
    states = {}
    try:
        for backend in ("shared_pages", "independent_pages"):
            pool = HeadPagePool(capacity, args.page_size, args.head_dim,
                                dtype=getattr(torch, args.dtype), device=args.device)
            manager = HeadKVManager(pool)
            # Register immediately, so a failed source publication is reclaimed.
            state = states[backend] = dict(manager=manager, raw=[])
            versions, offline = timed(args.device, lambda: build_cached_documents(manager, fixtures, args))
            cached = stats_snapshot(manager)
            assert cached["requests"] == 0 and cached["cached_roots"] == args.documents
            state.update(versions=versions, offline=offline, cached=cached)
        for count in args.requests:
            for rep in range(-args.warmup, args.repeats):
                phase = "warmup" if rep < 0 else "measured"
                order = list(states)
                if rep % 2:
                    order.reverse()
                for order_index, backend in enumerate(order):
                    state = states[backend]
                    batch = measure_batch(state["manager"], fixtures, args, backend, count, rep,
                                          phase, state["versions"], expected_outputs)
                    batch["execution_order"] = order
                    batch["execution_order_index"] = order_index
                    state["raw"].append(batch)
        for backend, state in states.items():
            manager, versions, raw = state["manager"], state["versions"], state["raw"]
            source_error = validate_cache(manager, fixtures, args, versions)
            for batch in raw:
                batch["correctness"]["source_unchanged"] = source_error == 0
            before_evict = stats_snapshot(manager)
            _, eviction = timed(args.device, lambda: [manager.evict(f"cache-document-{doc}")
                                                       for doc in range(args.documents)])
            final = stats_snapshot(manager)
            if final["allocated_bytes"] or final["pinned_pages"] or final["cached_roots"]:
                raise AssertionError("cache eviction did not reclaim all pages")
            arms.append(dict(backend=backend, offline_publish=state["offline"], cached_state=state["cached"],
                             raw_batches=raw, cache_source_max_abs_error=source_error,
                             before_evict=before_evict, cache_eviction=eviction, final_state=final))
    finally:
        # Exception cleanup is intentional: a correctness failure must not leave
        # live ownership/pins behind when this function is embedded in a suite.
        for state in states.values():
            manager = state["manager"]
            for name in list(manager._requests):
                manager.release_request(name)
            for key in list(manager._cache):
                manager.evict(key)
            synchronize(args.device)
            manager.pool.collect()
    summary = []
    for arm in arms:
        for count in args.requests:
            batches = [b for b in arm["raw_batches"] if b["phase"] == "measured"
                       and b["live_requests"] == count]
            first = batches[0]
            row = dict(backend=arm["backend"], live_requests=count,
                       active_bytes_after_attach=first["after_attach"]["allocated_bytes"],
                       active_bytes_after_repair=first["after_repair"]["allocated_bytes"],
                       reserved_bytes=first["after_attach"]["reserved_bytes"],
                       attach_written_bytes=first["attach_written_bytes"],
                       repair_copied_bytes=first["repair_copied_bytes"],
                       repair_written_bytes=first["repair_written_bytes"],
                       attention_max_abs_error=max(b["correctness"]["attention_max_abs_error"] for b in batches))
            for phase in ("attach", "repair", "descriptor", "attention_batch", "release"):
                row[f"{phase}_wall_ms_p50"] = statistics.median(b[phase]["wall_ms"] for b in batches)
            summary.append(row)
    return dict(context_document_tokens=length, actual_kv_tokens=length + args.prefix_tokens,
                fixture_sha256=fixtures["sha256"], document_lengths=fixtures["sizes"],
                config=dict(heads=args.heads, head_dim=args.head_dim, page_size=args.page_size,
                            query_tokens=args.query_tokens, capacity_pages=capacity,
                            document_pages=document_pages, shared_ratio_requested=args.shared_ratio,
                            shared_document_count=math.floor(args.documents * args.shared_ratio),
                            actual_shared_document_fraction=math.floor(args.documents * args.shared_ratio) / args.documents,
                            repaired_heads=nh, repaired_rows_per_head=nr),
                fixture_allocation=dict(persistent_slabs_simultaneously_resident=2,
                                        both_slabs_reserved_bytes=sum(a["cached_state"]["reserved_bytes"] for a in arms),
                                        note="per-arm active/reserved counts below exclude the other arm and source/reference fixtures"),
                arms=arms, summary=summary)


def environment(args):
    packages = {}
    for name in ("torch", "triton", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = dict(python=sys.version, platform=platform.platform(), packages=packages,
                  torch_cuda=torch.version.cuda, cpu_threads=torch.get_num_threads())
    if torch.device(args.device).type == "cuda":
        props = torch.cuda.get_device_properties(args.device)
        result["gpu"] = dict(name=props.name, total_memory=props.total_memory,
                             compute_capability=[props.major, props.minor])
    return result


def benchmark(args):
    torch.set_num_threads(args.threads)
    files = [Path(__file__), *sorted((REPO / "python/sglang/srt/mem_cache/head_kv").glob("*.py"))]
    result = dict(schema_version=1, status="running", benchmark="segpaged_persistent_nonprefix_cache",
                  scope="seeded tensor component benchmark; no model inference, TTFT, or serving QPS",
                  reuse_contract=CERTIFICATE,
                  protocol=dict(offline="one cache publication per arm; does not generate model KV",
                                warm="persistent cached pages survive every request release",
                                live_requests="simultaneously resident; attention calls run sequentially",
                                fairness="same tensors, same capacity, same immutable COW, same attention",
                                allocation="active bytes are occupied slab pages; reserved slab remains allocated",
                                cold="offline_publish is storage construction, not cold-start model inference",
                                timing="synchronized wall time, fixture/reference/verification excluded; public attention validation included",
                                ordering="both persistent pools resident; AB/BA alternating per repetition; small timing deltas remain exploratory"),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  environment=environment(args),
                  source_sha256={str(f.relative_to(REPO)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                  cases=[])
    def checkpoint():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    checkpoint()
    try:
        for length in args.lengths:
            result["cases"].append(run_case(args, length))
            checkpoint()
            print(f"[cache] document_tokens={length} passed; raw_repetitions_saved={args.repeats}", flush=True)
    except BaseException as exc:
        result.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        checkpoint()
        raise
    result["status"] = "complete"
    checkpoint()
    return result


def main(argv=None):
    args = parse_args(argv)
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"[cache] complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
