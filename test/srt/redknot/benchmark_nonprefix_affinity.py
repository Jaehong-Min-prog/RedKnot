#!/usr/bin/env python3
"""Single-node mixed-document scheduling A/B; NOT a distributed router.

All arms receive the same frozen, interleaved, already-arrived requests. The
only shared-vs-shared change is admission order. Canonical caches are resident
throughout; overlap is not a measurement of eviction, load savings or FLOPs.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import signal
import statistics
import time
from types import SimpleNamespace

from benchmark_redknot_nonprefix import (
    HERE, ROOT, bootstrap_imports, digest, event, freeze_workload, profile_data,
    save, scores,
)
from benchmark_nonprefix_scale import (
    GLOBAL_PREFIX_LAYERS, OUTPUT_TOKENS, _exception, _guard_unchanged,
    _model_guard, cuda_failure, split_exact,
)

ARMS = ("recomputed_paged_fcfs", "redknot_private_fcfs", "redknot_shared_fcfs",
        "redknot_private_affinity", "redknot_shared_affinity")


class _PreflightBlocked(Exception):
    pass


def prepare_plan(workload, keys, *, chunks=16, replicas=16, concurrency=8,
                 pool_gib=48., repeats=2, arms=ARMS, lookahead=32,
                 max_bypass=4, max_wait_seconds=120., recent_requests=8):
    """Predeclare all inputs, order and knobs; never select using model output."""
    items = {item["key"]: item for item in workload["items"]}
    if len(items) != len(workload["items"]):
        raise ValueError("duplicate source item keys")
    if len(keys) < 2 or len(set(keys)) != len(keys) or any(k not in items for k in keys):
        raise ValueError("at least two distinct frozen document groups required")
    if workload.get("max_new_tokens") != OUTPUT_TOKENS:
        raise ValueError("32-token natural-EOS source protocol required")
    for value in (chunks, replicas, concurrency, repeats, lookahead, recent_requests):
        if type(value) is not int or value < 1:
            raise ValueError("positive integer geometry required")
    if type(max_bypass) is not int or max_bypass < 0:
        raise ValueError("max_bypass must be nonnegative")
    for value in (pool_gib, max_wait_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("positive finite pool/wait bound required")
    if replicas*len(keys) < 2*concurrency:
        raise ValueError("need at least twice requested concurrency in the burst")
    if (len(arms) != len(set(arms)) or set(arms)-set(ARMS)
            or not {"redknot_shared_fcfs", "redknot_shared_affinity"} <= set(arms)):
        raise ValueError("distinct known arms with both shared scheduling policies required")
    groups = []
    for key in keys:
        item = items[key]
        flat = [t for part in item["document_token_ids"] for t in part]
        if digest(item["document_token_ids"]) != item["document_tokens_sha256"]:
            raise ValueError("source document hash mismatch")
        if item["context_tokens"] != len(flat):
            raise ValueError("source context length mismatch")
        parts = split_exact(flat, chunks)
        prefixes = item["prefix_token_ids"][:replicas]
        if len(prefixes) != replicas or len({tuple(p) for p in prefixes}) != replicas:
            raise ValueError("need distinct frozen private prefixes, no expansion")
        hashes = [digest(p+flat+item["query_token_ids"]) for p in prefixes]
        if hashes != item["exact_prompt_sha256"][:replicas]:
            raise ValueError("prompt hash mismatch")
        groups.append(dict(item_key=key, primary_qa_key=item.get("primary_qa_key", key),
            dataset=item.get("dataset"), variant=item.get("variant"),
            source_provenance=item.get("provenance"), augmentation=item.get("augmentation"),
            context_tokens=len(flat),
            chunk_lengths=list(map(len, parts)), chunks_sha256=digest(parts),
            prompt_sha256=hashes, query_tokens=len(item["query_token_ids"])))
    requests = [dict(request_id=f"request:{replica}:{group}", group_index=group,
        item_key=key, prefix_index=replica, arrival_seconds=0.,
        exact_prompt_sha256=groups[group]["prompt_sha256"][replica])
        for replica in range(replicas) for group, key in enumerate(keys)]
    # Forward then reverse is counterbalanced for the default two repeats.
    orders = [list(arms) if repeat % 2 == 0 else list(reversed(arms))
              for repeat in range(repeats)]
    return dict(schema="redknot-nonprefix-affinity-plan-v1", workload_sha256=digest(workload),
        groups=groups, requests=requests, concurrency=concurrency, pool_gib=pool_gib,
        chunks=chunks, repeats=repeats, arms=list(arms), arm_order_by_repeat=orders,
        output_tokens=OUTPUT_TOKENS, natural_eos=True,
        affinity_config=dict(lookahead=lookahead, max_bypass=max_bypass,
            max_wait_seconds=max_wait_seconds, recent_requests=recent_requests),
        global_prefix_layers=GLOBAL_PREFIX_LAYERS, ffn="full native MLP",
        scope="single-node hot resident canonical documents; serial timed recovery; no cache eviction or multi-node routing",
        independent_primary_qa_count=len({group["primary_qa_key"] for group in groups}),
        quality_scope="primary QA IDs only are independent; length variants and replicas are not; mechanism scan, not generalization certification",
        selection="explicit frozen keys; round-robin arrival order; no output-based selection")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _distribution(values):
    present = sorted(float(x) for x in values if _finite(x) and x >= 0)
    def quantile(q):
        if not present:
            return None
        pos = q*(len(present)-1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return present[lo] + (present[hi]-present[lo])*(pos-lo)
    return dict(count=len(present), missing_count=len(values)-len(present),
        mean=statistics.mean(present) if present else None, p50=quantile(.5), p95=quantile(.95),
        max=max(present) if present else None,
        method="linear empirical quantiles, not confidence intervals")


def _score_valid(score):
    return isinstance(score, dict) and all(_finite(score.get(m)) and 0 <= score[m] <= 1 for m in ("f1", "em"))


def _complete_row(row, plan):
    service = row.get("service") or {}
    records = service.get("records", [])
    expected = {x["request_id"]: x for x in plan["requests"]}
    seconds = service.get("total_wall_seconds")
    if not (row.get("status") == "complete" and row.get("cleanup_status") == "complete"
        and not row.get("fatal") and not row.get("cleanup_error") and not row.get("persistence_error")
        and service.get("status") == "complete" and not service.get("errors")
        and not service.get("failed_requests", 0) and not service.get("cancelled_requests", 0)
        and _finite(seconds) and seconds > 0
        and service.get("max_active") == plan["concurrency"] and service.get("ignore_eos") is False
        and len(records) == len(expected) == service.get("completed_requests")
        and {r.get("request_id") for r in records} == set(expected)):
        return False
    count = 0
    for record in records:
        spec = expected[record["request_id"]]
        ids = record.get("output_ids")
        if not (record.get("status") == "completed"
            and record.get("exact_prompt_sha256") == spec["exact_prompt_sha256"]
            and record.get("item_key") == spec["item_key"]
            and record.get("prefix_index") == spec["prefix_index"]
            and record.get("arrival_seconds") == spec["arrival_seconds"]
            and record.get("max_new_tokens") == plan["output_tokens"]
            and isinstance(ids, list) and 0 < len(ids) <= plan["output_tokens"]
            and all(type(token) is int and token >= 0 for token in ids)):
            return False
        count += len(ids)
    return count == service.get("completed_output_tokens")


def _quality_comparison(baseline, candidate):
    result = dict(baseline=baseline, candidate=candidate)
    for metric in ("f1", "em"):
        a, b = baseline.get(metric), candidate.get(metric)
        drop = 100*(a-b) if _finite(a) and _finite(b) else None
        result[metric+"_drop_pp"] = drop
        result[metric+"_relative_drop_percent"] = drop/a if drop is not None and a > 0 else None
    result["metric_2percent_gate"] = all(
        _finite(result[key]) and result[key] <= 2+1e-10
        for key in ("f1_drop_pp", "f1_relative_drop_percent", "em_drop_pp", "em_relative_drop_percent"))
    return result


def _quality_means(records_by_repeat, plan):
    """Prefixes within repeat, repeats within group; no best-prefix selection."""
    groups = {}
    for group in plan["groups"]:
        key = group["item_key"]
        expected = [spec for spec in plan["requests"] if spec["item_key"] == key]
        primary, robust = [], []
        for records in records_by_repeat:
            selected = [records.get(spec["request_id"]) for spec in expected]
            if not all(record is not None and _score_valid(record.get("accuracy")) for record in selected):
                continue
            zero = [record["accuracy"] for record in selected if record["prefix_index"] == 0]
            if len(zero) != 1:
                continue
            primary.append(zero[0])
            robust.append({m:statistics.mean(record["accuracy"][m] for record in selected) for m in ("f1", "em")})
        average = lambda values: {m:statistics.mean(score[m] for score in values) if values else None for m in ("f1", "em")}
        groups[key] = dict(primary_qa_key=group.get("primary_qa_key", key),
                          primary_replica0=average(primary), all_prefix=average(robust), scored_repeats=len(robust))
    overall = {}
    for scope in ("primary_replica0", "all_prefix"):
        by_primary = {}
        for group in groups.values():
            by_primary.setdefault(group["primary_qa_key"], []).append(group[scope])
        overall[scope] = {m:statistics.mean(statistics.mean(value[m] for value in values) for values in by_primary.values())
            if by_primary and all(_finite(value[m]) for values in by_primary.values() for value in values) else None
            for m in ("f1", "em")}
    return dict(groups=groups, **overall)


def summarize_rows(rows, plan):
    """Describe raw service, tail latency and group fairness without an SLO claim."""
    output = {}
    expected = {spec["request_id"]:spec for spec in plan["requests"]}
    order = list(expected)
    fields = ("queue_wait_seconds", "ttft_seconds", "service_ttft_seconds", "latency_seconds")
    for arm in plan["arms"]:
        selected = [row for row in rows if row["arm"] == arm]
        services = [row["service"] for row in selected if isinstance(row.get("service"), dict)]
        records = [record for service in services for record in service.get("records", [])]
        completed = [record for record in records if record.get("status") == "completed"]
        wall = sum(service["total_wall_seconds"] for service in services if _finite(service.get("total_wall_seconds")) and service["total_wall_seconds"] > 0)
        by_repeat = {}
        for row in selected:
            by_repeat.setdefault(row.get("repeat"), []).append(row)
        complete = (set(by_repeat) == set(range(plan["repeats"]))
                    and all(len(group) == 1 and _complete_row(group[0], plan) for group in by_repeat.values()))
        quality_rows = [{record["request_id"]:record for record in row["service"]["records"]}
                        for row in selected if _complete_row(row, plan)]
        groups = {}
        for group in plan["groups"]:
            subset = [r for r in records if expected.get(r.get("request_id"), {}).get("item_key") == group["item_key"]]
            done = [r for r in subset if r.get("status") == "completed"]
            groups[group["item_key"]] = dict(requests_observed=len(subset), completed=len(done),
                failed=sum(r.get("status") == "failed" for r in subset),
                cancelled=sum(r.get("status") == "cancelled" for r in subset),
                **{field:_distribution([r.get(field) for r in (subset if field == "queue_wait_seconds" else done)]) for field in fields})
        admission_order, bypass_counts = [], []
        for row in selected:
            current = {r["request_id"]:r for r in (row.get("service") or {}).get("records", [])}
            admitted = sorted((r for r in current.values() if _finite(r.get("admission_started_seconds"))),
                              key=lambda r:r["admission_started_seconds"])
            bypass = {}
            for index, identity in enumerate(order):
                started = current.get(identity, {}).get("admission_started_seconds")
                if _finite(started):
                    bypass[identity] = sum(_finite(current.get(later, {}).get("admission_started_seconds"))
                        and current[later]["admission_started_seconds"] < started for later in order[index+1:])
            bypass_counts.extend(bypass.values())
            admission_order.append(dict(repeat=row.get("repeat"), request_ids=[r["request_id"] for r in admitted],
                overtaken_by_later_requests=bypass))
        group_means = [g["queue_wait_seconds"]["mean"] for g in groups.values() if g["queue_wait_seconds"]["mean"] is not None]
        stages = {}
        for service in services:
            for name, seconds in service.get("stages_seconds", {}).items():
                if _finite(seconds): stages[name] = stages.get(name, 0)+seconds
        max_value = lambda field: max([s[field] for s in services if _finite(s.get(field))], default=None)
        output[arm] = dict(observed_rows=len(selected), expected_repeats=plan["repeats"], all_repeats_complete=complete,
            status_counts={status:sum(row.get("status") == status for row in selected) for status in {row.get("status") for row in selected}},
            completed_requests=len(completed), failed_requests=sum(r.get("status") == "failed" for r in records),
            cancelled_requests=sum(r.get("status") == "cancelled" for r in records),
            service_seconds=wall if wall else None, observed_qps=len(completed)/wall if wall else None,
            headline_qps=len(completed)/wall if complete and wall else None,
            generated_output_tokens=sum(len(r.get("output_ids", [])) for r in completed),
            actual_max_active=max_value("peak_active_requests"), actual_max_forward_batch=max_value("max_forward_batch"),
            capacity_blocks=sum(r.get("capacity_blocks", 0) for r in records), stages_seconds=stages,
            **{field:_distribution([r.get(field) for r in (records if field == "queue_wait_seconds" else completed)]) for field in fields},
            per_group=groups, admission_order=admission_order,
            max_observed_later_request_overtakes=max(bypass_counts, default=None),
            group_mean_queue_wait_spread_seconds=max(group_means)-min(group_means) if group_means else None,
            waits_above_affinity_promotion_threshold=sum(_finite(r.get("queue_wait_seconds"))
                and r["queue_wait_seconds"] > plan["affinity_config"]["max_wait_seconds"] for r in records),
            fairness_scope="descriptive per-group waits and actual admission inversions; promotion threshold is not an end-to-end latency SLA; intrinsic group lengths may differ",
            latency_scope="queue wait uses all observed admissions; TTFT and request completion latency use completed requests only, failures remain in counts",
            quality=_quality_means(quality_rows, plan), quality_certified=False)
    return output


def compare_rows(rows, plan):
    """Only all-complete paired repeats qualify for headline comparisons."""
    groups = {}
    expected = {x["request_id"]: x for x in plan["requests"]}
    for row in rows:
        groups.setdefault((row["arm"], row["repeat"]), []).append(row)
    good = {key: value[0] for key, value in groups.items() if len(value) == 1 and _complete_row(value[0], plan)}
    comparisons = []
    pairs = [("redknot_shared_fcfs", "redknot_shared_affinity"),
             ("redknot_private_fcfs", "redknot_private_affinity"),
             ("redknot_private_fcfs", "redknot_shared_fcfs"),
             ("redknot_private_affinity", "redknot_shared_affinity"),
             ("recomputed_paged_fcfs", "redknot_shared_affinity")]
    for baseline, candidate in pairs:
        if baseline not in plan["arms"] or candidate not in plan["arms"]:
            continue
        paired = [i for i in range(plan["repeats"]) if (baseline,i) in good and (candidate,i) in good]
        complete = (len(paired) == plan["repeats"] and all(
            {repeat for arm, repeat in groups if arm == name} == set(range(plan["repeats"]))
            for name in (baseline, candidate)))
        base_wall = sum(good[baseline,i]["service"]["total_wall_seconds"] for i in paired)
        new_wall = sum(good[candidate,i]["service"]["total_wall_seconds"] for i in paired)
        identities, base_records, candidate_records = [], [], []
        for i in paired:
            br = {r["request_id"]: r for r in good[baseline,i]["service"]["records"]}
            cr = {r["request_id"]: r for r in good[candidate,i]["service"]["records"]}
            base_records.append(br); candidate_records.append(cr)
            for key in expected:
                identities.append(br[key]["output_ids"] == cr[key]["output_ids"])
        bq, cq = _quality_means(base_records, plan), _quality_means(candidate_records, plan)
        primary = _quality_comparison(bq["primary_replica0"], cq["primary_replica0"])
        robust = _quality_comparison(bq["all_prefix"], cq["all_prefix"])
        per_group = {key:{scope:_quality_comparison(bq["groups"][key][scope], cq["groups"][key][scope])
                         for scope in ("primary_replica0", "all_prefix")} for key in bq["groups"]}
        observed_gate = (complete and primary["metric_2percent_gate"] and robust["metric_2percent_gate"]
            and all(value[scope]["metric_2percent_gate"] for value in per_group.values() for scope in ("primary_replica0", "all_prefix"))
            and all(group["scored_repeats"] == plan["repeats"] for quality in (bq, cq) for group in quality["groups"].values()))
        comparisons.append(dict(baseline=baseline, candidate=candidate, paired_repeats=paired,
            all_repeats_complete=complete,
            qps_ratio=base_wall/new_wall if complete and new_wall else None,
            provisional_paired_qps_ratio=base_wall/new_wall if new_wall else None,
            exact_output_fraction=statistics.mean(identities) if identities else None,
            mean_drop_pp={m:robust[m+"_drop_pp"] for m in ("f1", "em")},
            per_group_drop_pp={key:{m:value["all_prefix"][m+"_drop_pp"] for m in ("f1", "em")} for key,value in per_group.items()},
            primary_replica0=primary, all_prefix=robust, per_group_quality=per_group,
            observed_2percent_gate=bool(observed_gate),
            quality_scope="same sampled primary QA only; absolute pp AND relative percent, both primary0 and all-prefix, all groups; zero-baseline relative metrics are non-informative",
            quality_certified=False))
    return comparisons


def run_arm(model, tokenizer, profile, workload, plan, arm, repeat, *, torch,
            Engine, Request, run_queued, AffinityConfig, revision, eos, persist):
    row = dict(arm=arm, repeat=repeat, status="starting", service=None)
    engine, fatal, guard = None, False, None
    try:
        guard = _model_guard(model)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(0)
        if torch.cuda.mem_get_info(0)[0] < int((plan["pool_gib"]+8)*1024**3):
            row.update(status="preflight_blocked", reason="pool plus 8GiB workspace estimate unavailable; not measured OOM")
            raise _PreflightBlocked
        cfg = copy.deepcopy(profile)
        if arm.startswith("recomputed"):
            cfg.head_class = [["global"]*cfg.num_kv_heads for _ in range(cfg.num_layers)]
            cfg.dense_prefix_layers = 0
            cfg._cached_tensors.clear()
        began = time.perf_counter()
        engine = Engine(model, cfg, capacity_pages=int(plan["pool_gib"]*1024**3)//(256*model.config.head_dim*4),
            page_size=256, backend="flash", sparse_ffn_schedule=None, model_revision=revision,
            share_local=not arm.startswith("redknot_private"))
        engine.synchronize()
        row["engine_construction_seconds"] = time.perf_counter()-began
        items = {x["key"]:x for x in workload["items"]}
        docs = {}
        began = time.perf_counter()
        for group in plan["groups"]:
            item = items[group["item_key"]]
            parts = split_exact(sum(item["document_token_ids"], []), plan["chunks"])
            if digest(parts) != group["chunks_sha256"]:
                raise ValueError("chunk partition changed")
            names = [f"group:{group['item_key']}:chunk:{i}" for i in range(len(parts))]
            for name, ids in zip(names, parts):
                engine.register_document(name, ids)
            docs[group["item_key"]] = names
        engine.synchronize()
        row["offline_document_seconds"] = time.perf_counter()-began
        def make(spec, identity=None, tokens=OUTPUT_TOKENS):
            item = items[spec["item_key"]]
            if digest(item["prefix_token_ids"][spec["prefix_index"]]
                      + sum(item["document_token_ids"], []) + item["query_token_ids"]) != spec["exact_prompt_sha256"]:
                raise ValueError("frozen request input changed")
            return Request(identity or spec["request_id"], item["prefix_token_ids"][spec["prefix_index"]],
                docs[spec["item_key"]], item["query_token_ids"], tokens, eos,
                arrival_seconds=spec["arrival_seconds"])
        row["warmups"] = []
        for group_index in range(len(plan["groups"])):
            warmup = run_queued(engine, [make(plan["requests"][group_index], f"warmup:{group_index}", 2)], max_active=1)
            row["warmups"].append(warmup)
            if warmup["status"] != "complete":
                raise RuntimeError("warmup failed: "+json.dumps(warmup["errors"]))
        kwargs = {}
        if arm.endswith("_affinity"):
            kwargs = dict(admission_policy="share_affinity", affinity_config=AffinityConfig(**plan["affinity_config"]))
        torch.cuda.reset_peak_memory_stats(0)
        service = run_queued(engine, [make(spec) for spec in plan["requests"]],
            max_active=plan["concurrency"], **kwargs)
        # Preserve emitted IDs and timing even if later decoding/scoring fails.
        row.update(service=service, status=service["status"])
        fatal = (cuda_failure(service.get("errors", []))
                 or any(cuda_failure(record.get("error", {})) for record in service.get("records", [])))
        specs = {r["request_id"]:r for r in plan["requests"]}
        for record in service["records"]:
            spec = specs[record["request_id"]]
            record["item_key"] = spec["item_key"]
            record["prefix_index"] = spec["prefix_index"]
            record["exact_prompt_sha256"] = spec["exact_prompt_sha256"]
            record["frozen_input_reference"] = dict(manifest_item_key=spec["item_key"],
                private_prefix_index=spec["prefix_index"], document_field="document_token_ids (flatten in order)",
                query_field="query_token_ids", workload_sha256=plan["workload_sha256"],
                reconstruction="prefix_token_ids[private_prefix_index] + flattened document_token_ids + query_token_ids")
            record["text"] = tokenizer.decode(record["output_ids"], skip_special_tokens=True)
            record["accuracy"] = scores(record["text"], items[spec["item_key"]]["answers"]) if record["status"] == "completed" else None
        row.update(service=service, status=service["status"],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    except _PreflightBlocked:
        pass
    except (KeyboardInterrupt, SystemExit) as exc:
        row.update(status="interrupted", error=_exception(exc)); fatal = True
    except Exception as exc:
        row.update(status="failed", error=_exception(exc)); fatal = fatal or cuda_failure(row["error"])
    finally:
        row["fatal"] = fatal
        try:
            persist(row)
        except BaseException as exc:
            row["persistence_error"], fatal = _exception(exc), True
        cleanup_started = time.perf_counter()
        try:
            if engine is not None:
                engine.close()
                stats = engine.pool.stats()
                row["pool_after_close"] = stats
                if engine.requests or engine.pool.poisoned or stats["pinned_pages"] or stats["free_pages"] != engine.pool.capacity_pages:
                    raise RuntimeError("incomplete resource cleanup")
            del engine
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(0)
            if guard is None:
                raise RuntimeError("initial model guard unavailable; supervisor restart required")
            _guard_unchanged(model, guard, torch)
            row["cleanup_status"] = "complete"
        except BaseException as exc:
            row.update(cleanup_status="failed", cleanup_error=_exception(exc)); fatal = True
        row["fatal"] = fatal
        row["cleanup_seconds"] = time.perf_counter()-cleanup_started
        persist(row)
    return row, fatal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path, default=HERE/"head_class/qwen3-32B_optimal_g15_lf_ret.json")
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--replicas", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--pool-gib", type=float, default=48.)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--lookahead", type=int, default=32)
    p.add_argument("--max-bypass", type=int, default=4)
    p.add_argument("--max-wait-seconds", type=float, default=120.)
    p.add_argument("--recent-requests", type=int, default=8)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    import torch
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    workload = freeze_workload(SimpleNamespace(manifest=args.manifest, output_tokens=32), tokenizer)
    plan = prepare_plan(workload, args.items, chunks=args.chunks, replicas=args.replicas,
        concurrency=args.concurrency, pool_gib=args.pool_gib, repeats=args.repeats, arms=args.arms,
        lookahead=args.lookahead, max_bypass=args.max_bypass,
        max_wait_seconds=args.max_wait_seconds, recent_requests=args.recent_requests)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/"frozen_workload.json", workload)
    save(args.output/"frozen_affinity_plan.json", plan)
    if args.prepare_only:
        return 0
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("verified transformers 4.57.1 required")
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    bootstrap_imports()
    from sglang.srt.layers.attention.redknot.nonprefix_paged import NonPrefixPagedEngine
    from sglang.srt.layers.attention.redknot.nonprefix_scheduler import QueuedRequest, run_queued
    from sglang.srt.layers.attention.redknot.nonprefix_affinity import ShareAffinityConfig
    from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig
    from nonprefix_profile import load_effective_profile
    profile, diagnostics = load_effective_profile(args.profile, HeadClassConfig, GLOBAL_PREFIX_LAYERS)
    torch.manual_seed(20261016)
    started = time.perf_counter()
    model = transformers.Qwen3ForCausalLM.from_pretrained(str(args.model), local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"":"cuda:0"}).eval()
    torch.cuda.synchronize(0)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(type(x) is not int for x in eos):
        raise ValueError("valid natural EOS IDs required")
    revision = digest(model.config.to_dict())
    paths = [Path(__file__).resolve(), HERE/"benchmark_nonprefix_scale.py", HERE/"benchmark_redknot_nonprefix.py",
        HERE/"nonprefix_profile.py", args.profile,
        *[ROOT/"python/sglang/srt/layers/attention/redknot"/name for name in (
            "nonprefix_paged.py", "nonprefix_scheduler.py", "nonprefix_affinity.py")],
        *[ROOT/"python/sglang/srt/mem_cache/head_kv"/name for name in ("manager.py", "pool.py")]]
    result = dict(schema="redknot-nonprefix-affinity-v1", status="running", plan=plan,
        plan_sha256=digest(plan), workload_sha256=digest(workload), model_config_sha256=revision,
        config={key:str(value) if isinstance(value, Path) else value for key,value in vars(args).items()},
        model_load_seconds=time.perf_counter()-started, profile=profile_data(profile),
        profile_diagnostics=diagnostics, hardware=dict(gpu=torch.cuda.get_device_name(0),
            total_bytes=torch.cuda.get_device_properties(0).total_memory),
        source_hashes={str(path):digest(path.read_bytes()) for path in paths}, results=[])
    def persist(row):
        result["results"][:] = [r for r in result["results"] if (r["arm"],r["repeat"]) != (row["arm"],row["repeat"])]
        result["results"].append(copy.deepcopy(row))
        result["comparisons"] = compare_rows(result["results"], plan)
        result["arm_summary"] = summarize_rows(result["results"], plan)
        save(args.output/"results.json", result)
    for repeat, arms in enumerate(plan["arm_order_by_repeat"]):
        for arm in arms:
            event("affinity_arm_start", arm=arm, repeat=repeat)
            row, fatal = run_arm(model, tokenizer, profile, workload, plan, arm, repeat, torch=torch,
                Engine=NonPrefixPagedEngine, Request=QueuedRequest, run_queued=run_queued,
                AffinityConfig=ShareAffinityConfig, revision=revision, eos=eos, persist=persist)
            event("affinity_arm_finished", arm=arm, repeat=repeat, status=row["status"],
                  qps=(row.get("service") or {}).get("completed_qps"))
            if fatal:
                result["status"] = "fatal_supervisor_restart_required"
                save(args.output/"results.json", result)
                return 2
    result["status"] = "complete" if all(r["status"] == "complete" for r in result["results"]) else "incomplete"
    save(args.output/"results.json", result)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
