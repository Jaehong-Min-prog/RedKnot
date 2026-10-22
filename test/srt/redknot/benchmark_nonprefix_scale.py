#!/usr/bin/env python3
"""One-model-load, frozen-token non-prefix burst scaling experiment.

This is NOT production or SLA-qualified serving. Every measured burst starts
with registered offline documents and an empty request queue. Admission,
online recovery, query/decode, sampling and release are timed by run_queued.
The same unmodified HF reference, all-global paged control, private RedKnot
and shared RedKnot engines are reused; no model/head/MLP algorithm is changed.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

from benchmark_redknot_nonprefix import (
    HERE, ROOT, bootstrap_imports, digest, event, freeze_workload, profile_data,
    save, scores, summarize,
)

ARMS = ("recomputed_native", "recomputed_paged", "redknot_private", "redknot_shared")
OUTPUT_TOKENS = 32
GLOBAL_PREFIX_LAYERS = 60


class _ArmFinishedWithoutService(Exception):
    """Internal control flow; cleanup must still determine the returned fate."""


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def split_exact(tokens, chunks):
    """Balanced contiguous split, with no dropped/duplicated/re-encoded IDs."""
    chunks = _positive_int(chunks, "chunks")
    if chunks > len(tokens):
        raise ValueError("chunk count exceeds token count")
    width, remainder = divmod(len(tokens), chunks)
    result, cursor = [], 0
    for index in range(chunks):
        stop = cursor + width + (index < remainder)
        result.append(list(tokens[cursor:stop]))
        cursor = stop
    if cursor != len(tokens):
        raise AssertionError("exact chunk partition failed")
    return result


def prepare_cases(workload, *, cases=None, item_keys=None, chunks=(4, 8, 16),
                  concurrency=(4, 8, 16), requests=32, pool_gib=48):
    """Freeze the entire case selection before loading a model or seeing output."""
    items = {item["key"]: item for item in workload["items"]}
    if len(items) != len(workload["items"]):
        raise ValueError("duplicate frozen item keys")
    if workload.get("max_new_tokens") != OUTPUT_TOKENS:
        raise ValueError("scale protocol requires a frozen 32-token natural-EOS workload")
    if cases is None:
        selected = list(items) if item_keys is None else list(item_keys)
        if not selected or len(set(selected)) != len(selected) or any(key not in items for key in selected):
            raise ValueError("item selection must contain distinct frozen keys")
        cases = [dict(item_key=key, chunks=k, concurrency=c, requests=requests, pool_gib=pool_gib)
                 for key in selected for k in chunks for c in concurrency]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    result, seen = [], set()
    allowed = {"item_key", "chunks", "concurrency", "requests", "pool_gib"}
    for specification in cases:
        if not isinstance(specification, dict) or set(specification)-allowed:
            raise ValueError("invalid case fields")
        key = specification.get("item_key")
        if key not in items:
            raise ValueError("case item is not in frozen manifest")
        item = items[key]
        k = _positive_int(specification.get("chunks"), "chunks")
        c = _positive_int(specification.get("concurrency"), "concurrency")
        count = _positive_int(specification.get("requests", requests), "requests")
        budget = specification.get("pool_gib", pool_gib)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
            raise ValueError("pool_gib must be finite and positive")
        flat = [token for part in item["document_token_ids"] for token in part]
        pieces = split_exact(flat, k)
        prefixes = item["prefix_token_ids"][:count]
        if len(prefixes) != count or len({tuple(prefix) for prefix in prefixes}) != count:
            raise ValueError(f"{key}: need {count} distinct already-frozen private prefixes")
        if len(item["exact_prompt_sha256"]) < count:
            raise ValueError("not enough frozen exact prompt hashes")
        hashes = [digest(prefix+flat+item["query_token_ids"]) for prefix in prefixes]
        if hashes != item["exact_prompt_sha256"][:count]:
            raise ValueError("repartition changed a frozen exact prompt")
        identity = (key, k, c, count, budget)
        if identity in seen:
            raise ValueError("duplicate scale case")
        seen.add(identity)
        result.append(dict(case_id=f"case-{len(result):04d}", item_key=key,
            dataset=item["dataset"], context_tokens=len(flat), chunks=k, concurrency=c,
            requests=count, pool_gib=float(budget), chunk_lengths=list(map(len, pieces)),
            flat_document_sha256=digest(flat), chunk_tokens_sha256=digest(pieces),
            exact_prompt_sha256=hashes, prefix_tokens=list(map(len, prefixes)),
            query_tokens=len(item["query_token_ids"])))
    maximum = max(case["concurrency"] for case in result)
    if any(case["requests"] < 2*maximum for case in result):
        raise ValueError("every burst must contain at least twice the scan's maximum concurrency")
    return result


def capacity_plan(case, item, model_config, free_bytes, arm, headroom_gib=8):
    """Conservative preflight estimate, NOT measured maximum concurrency."""
    if arm not in ARMS or not math.isfinite(headroom_gib) or headroom_gib < 8:
        raise ValueError("unknown arm or unsafe headroom (<8 GiB)")
    def value(name):
        return model_config[name] if isinstance(model_config, dict) else getattr(model_config, name)
    layers, heads, dim = value("num_hidden_layers"), value("num_key_value_heads"), value("head_dim")
    kv_bytes_per_token = 2*2*layers*heads*dim  # K + V, explicitly BF16.
    maximum_tokens = max(case["prefix_tokens"]) + case["context_tokens"] + case["query_tokens"] + OUTPUT_TOKENS-1
    largest_request = maximum_tokens*kv_bytes_per_token
    resident = case["concurrency"]*largest_request
    headroom = int(headroom_gib*1024**3)
    pool_bytes = int(case["pool_gib"]*1024**3)
    required = (2*resident + largest_request if arm == "recomputed_native" else pool_bytes) + headroom
    return dict(allowed=required <= free_bytes, free_cuda_bytes=int(free_bytes),
        estimated_required_bytes=required, declared_headroom_bytes=headroom,
        max_reserved_sequence_tokens=maximum_tokens, kv_bytes_per_token=kv_bytes_per_token,
        nominal_resident_kv_bytes=resident, largest_reserved_request_kv_bytes=largest_request,
        native_batch_kv_factor=2, native_additional_request_factor=1,
        requested_concurrency=case["concurrency"], declared_pool_bytes=pool_bytes,
        basis=("2 x maximum nominal resident KV + largest reserved request KV + headroom: old and packed batch plus one unpack request"
               if arm == "recomputed_native" else "preallocated page slab + headroom; actual admission may block"),
        measurement=False, guarantees_no_oom=False)


def cuda_failure(error):
    """Also detect CUDA exceptions already captured in scheduler JSON records."""
    text = json.dumps(error, ensure_ascii=False).lower()
    return any(word in text for word in ("outofmemoryerror", "out of memory", "cuda error",
        "cudaerror", "cublas", "cudnn", "device-side assert", "illegal memory access"))


def planned_arm_order(case, repeat, arms):
    """Stable pre-output order, independent of local case IDs/GPU partitioning."""
    seed = digest(dict(item_key=case["item_key"], chunks=case["chunks"],
                       concurrency=case["concurrency"], repeat=repeat))
    offset = int(seed, 16) % len(arms)
    return list(arms[offset:])+list(arms[:offset])


def _exception(exc):
    return dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())


def _model_guard(model):
    return dict(forwards=[layer.self_attn.forward for layer in model.model.layers],
        config=copy.deepcopy(model.config.to_dict()),
        attention_implementation=model.config._attn_implementation,
        rotary=model.model.rotary_emb, inv_freq=model.model.rotary_emb.inv_freq.detach().clone(),
        amplitude=float(model.model.rotary_emb.attention_scaling))


def _guard_unchanged(model, guard, torch):
    if (getattr(model, "_redknot_nonprefix_owner", None) is not None
            or [layer.self_attn.forward for layer in model.model.layers] != guard["forwards"]
            or model.config.to_dict() != guard["config"]
            or model.config._attn_implementation != guard["attention_implementation"]
            or model.model.rotary_emb is not guard["rotary"]
            or float(model.model.rotary_emb.attention_scaling) != guard["amplitude"]
            or not torch.equal(model.model.rotary_emb.inv_freq, guard["inv_freq"])):
        raise RuntimeError("unsafe model/binding/RoPE state after arm cleanup; supervisor restart required")


def run_arm(model, tokenizer, profile, case, item, arm, repeat, *, torch, engine_types,
            queued_request, run_queued, eos, revision, warmup, persist, headroom_gib=8):
    """Return (complete evidence row, fatal flag); no retry or hidden fallback."""
    row = dict(arm=arm, item_key=item["key"], dataset=item["dataset"], repeat=repeat,
               status="starting", requested_concurrency=case["concurrency"], service=None)
    engine, fatal = None, False
    guard = None
    try:
        guard = _model_guard(model)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)
        plan = capacity_plan(case, item, model.config, torch.cuda.mem_get_info(0)[0], arm, headroom_gib)
        row["capacity_preflight"] = plan
        if not plan["allowed"]:
            row.update(status="preflight_blocked", reason="conservative estimated memory bound; not an observed OOM")
            raise _ArmFinishedWithoutService
        cfg = copy.deepcopy(profile)
        if arm == "recomputed_paged":
            cfg.head_class = [["global"]*cfg.num_kv_heads for _ in range(cfg.num_layers)]
            cfg.dense_prefix_layers = 0
            cfg._cached_tensors.clear()
        construction = time.perf_counter()
        if arm == "recomputed_native":
            engine = engine_types[0](model)
        else:
            page_bytes = 256*model.config.head_dim*2*2
            engine = engine_types[1](model, cfg, capacity_pages=int(case["pool_gib"]*1024**3)//page_bytes,
                page_size=256, backend="flash", sparse_ffn_schedule=None,
                model_revision=revision, share_local=arm != "redknot_private")
        engine.synchronize()
        row["engine_construction_seconds"] = time.perf_counter()-construction
        flat = [token for part in item["document_token_ids"] for token in part]
        chunks = split_exact(flat, case["chunks"])
        if digest(chunks) != case["chunk_tokens_sha256"]:
            raise AssertionError("frozen chunk partition changed")
        started, docs = time.perf_counter(), []
        for index, ids in enumerate(chunks):
            name = f"{case['case_id']}:document:{index}"
            engine.register_document(name, ids)
            docs.append(name)
        engine.synchronize()
        row["offline_document_seconds"] = time.perf_counter()-started
        def request(index, identity, output_tokens=OUTPUT_TOKENS):
            prefix, query = item["prefix_token_ids"][index], item["query_token_ids"]
            if digest(prefix+flat+query) != case["exact_prompt_sha256"][index]:
                raise AssertionError("request token hashes changed")
            return queued_request(identity, prefix, docs, query, output_tokens, eos)
        if warmup:
            row["warmup"] = run_queued(engine, [request(0, "warmup", 2)], max_active=1, ignore_eos=False)
            if row["warmup"]["status"] != "complete":
                row.update(status="warmup_failed", error=row["warmup"]["errors"])
                fatal = cuda_failure(row["warmup"])
                raise _ArmFinishedWithoutService
        torch.cuda.reset_peak_memory_stats(0)
        row["allocated_before_service"] = torch.cuda.memory_allocated(0)
        requests = [request(index, f"request:{index}") for index in range(case["requests"])]
        service = run_queued(engine, requests, max_active=case["concurrency"], ignore_eos=False)
        row["service"] = service
        for index, record in enumerate(service["records"]):
            record["frozen_prefix_index"] = index
            record["exact_prompt_sha256"] = case["exact_prompt_sha256"][index]
            record["text"] = tokenizer.decode(record["output_ids"], skip_special_tokens=True)
            record["accuracy"] = scores(record["text"], item["answers"]) if record["status"] == "completed" else None
        row.update(status=service["status"], peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            actual_max_active=service["peak_active_requests"], actual_max_forward_batch=service["max_forward_batch"],
            capacity_block_events=sum(record["capacity_blocks"] for record in service["records"]))
        fatal = cuda_failure(service.get("errors", [])) or any(cuda_failure(record.get("error", {})) for record in service["records"])
    except _ArmFinishedWithoutService:
        pass
    except (KeyboardInterrupt, SystemExit) as exc:
        row.update(status="interrupted", error=_exception(exc))
        fatal = True
    except Exception as exc:
        row.update(status="failed", error=_exception(exc))
        fatal = cuda_failure(row["error"])
    finally:
        # Save even an OOM result before attempting CUDA synchronization/close.
        row["fatal"] = fatal
        try:
            persist(row)
        except BaseException as exc:
            row["persistence_error"], fatal = _exception(exc), True
        cleanup_started = time.perf_counter()
        try:
            if engine is not None:
                engine.close()
                if getattr(engine, "requests", None):
                    raise RuntimeError("engine retained requests after close")
                pool = getattr(engine, "pool", None)
                if pool is not None:
                    stats = pool.stats()
                    row["pool_after_close"] = stats
                    if pool.poisoned or stats["pinned_pages"] or stats["free_pages"] != pool.capacity_pages:
                        raise RuntimeError("pool ownership/quarantine prevents safe reuse of model process")
                    del pool
            del engine
            engine = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(0)
            if guard is None:
                raise RuntimeError("initial model guard unavailable; supervisor restart required")
            _guard_unchanged(model, guard, torch)
            row["cleanup_status"] = "complete"
        except BaseException as exc:
            row["cleanup_status"], row["cleanup_error"], fatal = "failed", _exception(exc), True
        row["cleanup_seconds"] = time.perf_counter()-cleanup_started
        row["fatal"] = fatal
        persist(row)
    return row, fatal


def case_summary(rows, item, case, repeats):
    # Preflight failures remain visible as missing/failed cohorts, never zeros
    # or fabricated timings. Existing summary performs strict paired gates.
    usable = []
    for row in rows:
        if row.get("service") is None:
            continue
        checked = copy.deepcopy(row)
        if row.get("fatal") or row.get("cleanup_status") != "complete":
            checked["service"]["status"] = "failed"
            checked["service"].setdefault("errors", []).append(dict(stage="runner_cleanup",
                message="cleanup/model-state verification is incomplete or failed"))
        usable.append(checked)
    scored_item = dict(item, prefix_token_ids=item["prefix_token_ids"][:case["requests"]])
    summary = summarize(usable, [scored_item], expected_repeats=repeats)
    summary["arm_statuses"] = [{key: row.get(key) for key in ("arm", "repeat", "status", "fatal")}
                               for row in rows]
    # A named same-backend comparison remains available even if the native HF
    # control is preflight-blocked or intentionally absent. Never relabel its
    # measurements as native or compare unmatched successful subsets silently.
    def complete(row):
        service = row.get("service") or {}
        records = service.get("records", [])
        seconds = service.get("total_wall_seconds")
        return (not row.get("fatal", False) and row.get("cleanup_status") == "complete"
            and service.get("status") == "complete" and not service.get("errors")
            and len(records) == case["requests"] == service.get("completed_requests")
            and all(record.get("status") == "completed" for record in records)
            and not service.get("failed_requests", 0) and not service.get("cancelled_requests", 0)
            and isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0)
    groups = {}
    for row in rows:
        groups.setdefault((row["arm"], row["repeat"]), []).append(row)
    good = {identity: values[0] for identity, values in groups.items()
            if len(values) == 1 and complete(values[0])}
    comparisons = []
    for baseline in ("recomputed_native", "recomputed_paged"):
        for candidate in sorted({row["arm"] for row in rows}-{baseline}):
            paired = [repeat for repeat in range(repeats)
                      if (baseline, repeat) in good and (candidate, repeat) in good]
            base_wall = sum(good[baseline, repeat]["service"]["total_wall_seconds"] for repeat in paired)
            candidate_wall = sum(good[candidate, repeat]["service"]["total_wall_seconds"] for repeat in paired)
            count = len(paired)*case["requests"]
            comparisons.append(dict(baseline_arm=baseline, candidate_arm=candidate,
                paired_repeats=paired, paired_requests=count, all_expected_repeats_complete=len(paired) == repeats,
                baseline_qps=count/base_wall if base_wall else None,
                candidate_qps=count/candidate_wall if candidate_wall else None,
                qps_speedup=base_wall/candidate_wall if candidate_wall else None,
                scope="identical frozen input bursts; successful matched repeats only; requested/actual concurrency reported separately"))
    summary["paired_qps_comparisons"] = comparisons
    summary["not_measured"] += ["independent QA quality generalization from request replicas",
                                "exact native capacity when preflight blocked"]
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=HERE/"head_class/qwen3-32B_optimal_g15_lf_ret.json")
    parser.add_argument("--cases-json", type=Path, help="List of {item_key,chunks,concurrency,requests,pool_gib}")
    parser.add_argument("--items", nargs="+")
    parser.add_argument("--chunks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--pool-gib", type=float, default=48)
    parser.add_argument("--native-headroom-gib", type=float, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    _positive_int(args.repeats, "repeats")
    if not math.isfinite(args.native_headroom_gib) or args.native_headroom_gib < 8:
        raise ValueError("headroom must be at least 8 GiB")
    if len(args.arms) != len(set(args.arms)) or "recomputed_paged" not in args.arms:
        raise ValueError("distinct arms including the mandatory all-global paged control are required")
    if args.cases_json and args.items:
        raise ValueError("--cases-json and --items are mutually exclusive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    import torch
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    workload = freeze_workload(SimpleNamespace(manifest=args.manifest, output_tokens=OUTPUT_TOKENS), tokenizer)
    specifications = json.loads(args.cases_json.read_text()) if args.cases_json else None
    cases = prepare_cases(workload, cases=specifications, item_keys=args.items, chunks=args.chunks,
                          concurrency=args.concurrency, requests=args.requests, pool_gib=args.pool_gib)
    for case in cases:
        case["arm_order_by_repeat"] = [planned_arm_order(case, repeat, args.arms) for repeat in range(args.repeats)]
        case["arm_order_rule"] = "sha256(item_key,chunks,concurrency,repeat) modulo number of arms; fixed before model load"
    save(args.output/"frozen_workload.json", workload)
    save(args.output/"frozen_scale_plan.json", dict(schema="redknot-nonprefix-scale-plan-v1",
        workload_sha256=digest(workload), cases=cases, output_tokens=OUTPUT_TOKENS, natural_eos=True,
        global_prefix_layers=GLOBAL_PREFIX_LAYERS, ffn="full native MLP", arms=args.arms,
        repeats=args.repeats, selection="explicit frozen input selection before model/output; no winner selection"))
    if args.prepare_only:
        return 0
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("verified Transformers 4.57.1 required")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    bootstrap_imports()
    from sglang.srt.layers.attention.redknot.nonprefix_paged import NonPrefixPagedEngine
    from sglang.srt.layers.attention.redknot.nonprefix_scheduler import QueuedRequest, run_queued
    from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig
    from nonprefix_native_reference import NativeQueuedEngine
    from nonprefix_profile import load_effective_profile
    profile, diagnostics = load_effective_profile(args.profile, HeadClassConfig, GLOBAL_PREFIX_LAYERS)
    torch.manual_seed(20261016)
    started = time.perf_counter()
    model = transformers.Qwen3ForCausalLM.from_pretrained(str(args.model), local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": "cuda:0"}).eval()
    torch.cuda.synchronize(0)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(type(value) is not int for value in eos):
        raise ValueError("model must declare natural-EOS token IDs")
    revision = digest(model.config.to_dict())
    evidence = dict(schema="redknot-nonprefix-scale-v1", status="running", model_load_count=1,
        model_load_seconds=time.perf_counter()-started, model_config_sha256=revision,
        scope="non-production single-GPU hot prepared-document burst; serial timed online admission/recovery",
        native_reference="unchanged HF DynamicCache repack/unpack; same-backend all-global paged control mandatory",
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        profile=profile_data(profile), profile_diagnostics=diagnostics,
        workload_sha256=digest(workload), plan_sha256=digest(cases), cases=[],
        hardware=dict(gpu=torch.cuda.get_device_name(0), total_bytes=torch.cuda.get_device_properties(0).total_memory),
        source_hashes={str(path.relative_to(ROOT)):digest(path.read_bytes()) for path in (
            Path(__file__).resolve(), HERE/"benchmark_redknot_nonprefix.py", HERE/"nonprefix_native_reference.py", HERE/"nonprefix_profile.py",
            ROOT/"python/sglang/srt/layers/attention/redknot/nonprefix_paged.py",
            ROOT/"python/sglang/srt/layers/attention/redknot/nonprefix_scheduler.py",
            ROOT/"python/sglang/srt/mem_cache/head_kv/manager.py", ROOT/"python/sglang/srt/mem_cache/head_kv/pool.py")})
    items = {item["key"]:item for item in workload["items"]}
    save(args.output/"results.json", evidence)
    for case in cases:
        item = items[case["item_key"]]
        result = dict(case=case, status="running", results=[])
        evidence["cases"].append(result)
        def persist(row):
            identity = (row["arm"], row["repeat"])
            result["results"][:] = [old for old in result["results"] if (old["arm"], old["repeat"]) != identity]
            result["results"].append(copy.deepcopy(row))
            result["summary"] = case_summary(result["results"], item, case, args.repeats)
            save(args.output/(case["case_id"]+".json"), result)
            save(args.output/"results.json", evidence)
        for repeat in range(args.repeats):
            arms = case["arm_order_by_repeat"][repeat]
            for arm in arms:
                event("scale_arm_start", case=case["case_id"], arm=arm, repeat=repeat,
                      context=case["context_tokens"], chunks=case["chunks"], requested_concurrency=case["concurrency"])
                row, fatal = run_arm(model, tokenizer, profile, case, item, arm, repeat, torch=torch,
                    engine_types=(NativeQueuedEngine, NonPrefixPagedEngine), queued_request=QueuedRequest,
                    run_queued=run_queued, eos=eos, revision=revision, warmup=args.warmup,
                    persist=persist, headroom_gib=args.native_headroom_gib)
                persist(row)
                event("scale_arm_finished", case=case["case_id"], arm=arm, status=row["status"],
                    fatal=fatal, actual_max_active=row.get("actual_max_active"), actual_max_forward=row.get("actual_max_forward_batch"),
                    qps=(row.get("service") or {}).get("completed_qps"))
                if fatal:
                    result["status"], evidence["status"] = "fatal", "fatal_supervisor_restart_required"
                    save(args.output/(case["case_id"]+".json"), result)
                    save(args.output/"results.json", evidence)
                    return 2
        result["status"] = "complete" if all(row["status"] == "complete" for row in result["results"]) else "complete_with_blocked_or_failed_arms"
        save(args.output/(case["case_id"]+".json"), result)
        save(args.output/"results.json", evidence)
    evidence["status"] = "completed" if all(case["status"] == "complete" for case in evidence["cases"]) else "completed_with_blocked_or_failed_arms"
    save(args.output/"results.json", evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
