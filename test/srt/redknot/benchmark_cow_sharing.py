#!/usr/bin/env python3
"""Fixed single-corpus COW mechanism experiment, not a quality acceptance run.

Real frozen LongBench content; 32 synthetic repeated-arrival requests with
distinct private prefixes. Long variants contain labeled real distractors.
Only the selected case's corpus is registered. Never compare this workload's
absolute throughput to the earlier eight-QA experiment as a COW-only effect.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import benchmark_nonprefix_scale as scale
from benchmark_redknot_nonprefix import (
    HERE, ROOT, bootstrap_imports, digest, event, freeze_workload, profile_data, save,
)
from benchmark_selective_recovery import make_engine_factory, runtime_metadata

ARMS = ("recomputed_paged", "redknot_private", "redknot_shared")
ITEM_KEYS = ("hotpotqa:76:long", "musique:114:long")
GEOMETRIES = ((63, 16), (0, 4), (0, 16))
TOTAL_BYTES = 48 * 1024**3


def prepare_plan(workload, item_key, global_prefix_layers, concurrency, *, repeats=1,
                 publication_diagnostics=False, metadata_only_alias_sources=False):
    if item_key not in ITEM_KEYS or (global_prefix_layers, concurrency) not in GEOMETRIES:
        raise ValueError("only the six preregistered long-context cells are supported")
    if type(repeats) is not int or repeats not in (1, 2):
        raise ValueError("repeats must be one mechanism screen or two reversed rounds")
    if type(publication_diagnostics) is not bool:
        raise ValueError("publication_diagnostics must be boolean")
    if type(metadata_only_alias_sources) is not bool:
        raise ValueError("metadata_only_alias_sources must be boolean")
    if workload.get("schema") != "redknot-nonprefix-frozen-v1":
        raise ValueError("incompatible frozen workload")
    if workload.get("request_replicas") != 32:
        raise ValueError("exactly 32 frozen distinct private prefixes are required")
    items = {item["key"]: item for item in workload["items"]}
    if len(items) != len(workload["items"]) or item_key not in items:
        raise ValueError("missing or duplicate item")
    item = items[item_key]
    if (item.get("variant") != "long" or not item.get("augmentation")
            or not item.get("primary_qa_key")):
        raise ValueError("long augmentation and independent-primary provenance required")
    if digest(item["document_token_ids"]) != item["document_tokens_sha256"]:
        raise ValueError("frozen document hash mismatch")
    if len(item["prefix_token_ids"]) != 32 or len(item["exact_prompt_sha256"]) != 32:
        raise ValueError("no prefix truncation or expansion permitted")
    flat = sum(item["document_token_ids"], [])
    if item["context_tokens"] != len(flat) or not item.get("answers"):
        raise ValueError("invalid source context or answers")
    for prefix in item["prefix_token_ids"]:
        ids = prefix + flat + item["query_token_ids"]
        if (not prefix or not item["query_token_ids"] or len(ids) + 32 > 32768
                or any(type(t) is not int or t < 0 for t in ids)):
            raise ValueError("invalid frozen token IDs or context limit")
    case = scale.prepare_cases(workload, cases=[dict(item_key=item_key, chunks=16,
        concurrency=concurrency, requests=32, pool_gib=48)])[0]
    candidate = dict(name=f"g{global_prefix_layers}-hidden-r50-p8",
        global_prefix_layers=global_prefix_layers, row_fraction=.5, probe_layers=8,
        block_size=256, selection_mode="query_attention", recovery_mode="hidden_residual_int8")
    plan = dict(schema="redknot-cow-sharing-plan-v1", workload_sha256=digest(workload),
        case=case, candidate=candidate, arms=list(ARMS), repeats=repeats,
        arm_order_by_repeat=[list(ARMS) if r % 2 == 0 else list(reversed(ARMS))
                             for r in range(repeats)],
        total_kv_budget_bytes=TOTAL_BYTES, output_tokens=32, natural_eos=True,
        publication_diagnostics=publication_diagnostics,
        diagnostic_only=publication_diagnostics,
        scope="single-GPU hot service; only this case corpus registered in every arm; not production HTTP/TP",
        load_scope="synthetic repeated-arrival burst; 32 distinct frozen private prefixes; 1 independent primary QA",
        independent_primary_qa_count=1, primary_qa_key=item["primary_qa_key"],
        augmentation=item["augmentation"], source_provenance=item.get("provenance"),
        quality_policy="retain full outputs and unchanged F1/EM; no accuracy acceptance gate in this mechanism experiment",
        comparison_scope="private and COW use identical recovery/head/MLP algorithm; only share_local changes; recomputed is all-global full-row",
        memory_protocol="same 48GiB: actual hidden int8+scales outside pool plus page slab; all canonical local pages charged inside pool",
        selection="six fixed cells; no output-based case selection; blocked/failed cells retained",
        timing="includes online admission/recovery/publication/query/decode/sampling/release; offline preparation excluded and reported")
    # Absence is the original off schema, preserving its frozen plan hash.
    if metadata_only_alias_sources:
        plan["metadata_only_alias_sources"] = True
    return plan


def arm_factory(Engine, Config, *, arm, candidate, cache_bytes, page_bytes, receipt,
                publication_diagnostics=False, metadata_only_alias_sources=False):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if type(metadata_only_alias_sources) is not bool:
        raise ValueError("metadata_only_alias_sources must be boolean")
    selected = None if arm == ARMS[0] else candidate
    create = make_engine_factory(Engine, Config, candidate=selected,
        expected_cache_bytes=0 if selected is None else cache_bytes,
        total_bytes=TOTAL_BYTES, page_bytes=page_bytes, receipt=receipt)
    def checked(model, cfg, **kwargs):
        expected_share = arm != "redknot_private"
        if kwargs.get("share_local") is not expected_share:
            raise ValueError("runner share_local differs from frozen arm")
        expected_metadata = metadata_only_alias_sources and arm == "redknot_shared"
        if kwargs.pop("metadata_only_alias_sources", expected_metadata) is not expected_metadata:
            raise ValueError("runner metadata-only source mode differs from frozen arm")
        if expected_metadata:
            kwargs["metadata_only_alias_sources"] = True
        if publication_diagnostics:
            kwargs["publication_diagnostics"] = True
        return create(model, cfg, **kwargs)
    return checked


def overlap_receipt(service, *, source_bytes_by_document=None, share_local=False):
    """Count actual admitted owners, not requested concurrency or arrivals.

    Publication happens inside admission. The measured interval starts only at
    successful admitted and ends at released; it excludes intra-admission overlap.
    Duplicate bytes are a page-rounded source-payload bound, not measured QPS.
    """
    sources = source_bytes_by_document or {}
    records = {r["request_id"]: r for r in service.get("records", [])}
    if len(records) != len(service.get("records", [])):
        raise ValueError("duplicate request IDs in trace evidence")
    active, previous, peak, peak_duplicates, peak_bytes = set(), 0., 0, 0, 0
    admitted, released = set(), set()
    seconds_by_degree, any_overlap_seconds = {}, 0.
    for ev in service.get("trace", []):
        if ev.get("event") not in ("admitted", "released"):
            continue
        now = ev["seconds"]
        if not isinstance(now, (int, float)) or not math.isfinite(now) or now < previous:
            raise ValueError("invalid trace clock")
        counts = {}
        for rid in active:
            for doc in set(records[rid]["document_ids"]):
                counts[doc] = counts.get(doc, 0) + 1
        degree = max(counts.values(), default=0)
        seconds_by_degree[str(degree)] = seconds_by_degree.get(str(degree), 0.) + now - previous
        if degree >= 2:
            any_overlap_seconds += now - previous
        peak = max(peak, degree)
        peak_duplicates = max(peak_duplicates, sum(max(0, c-1) for c in counts.values()))
        peak_bytes = max(peak_bytes, sum(max(0, c-1)*sources.get(d, 0) for d, c in counts.items()))
        rid = ev["request_id"]
        if rid not in records:
            raise ValueError("unknown request in trace")
        if ev["event"] == "admitted":
            if rid in admitted:
                raise ValueError("duplicate active admission")
            active.add(rid)
            admitted.add(rid)
        else:
            if rid not in active:
                raise ValueError("release without admission")
            active.remove(rid)
            released.add(rid)
        previous = now
    wall = service.get("total_wall_seconds", 0.)
    if not math.isfinite(wall) or wall < previous:
        raise ValueError("service wall precedes trace")
    completed = {rid for rid,r in records.items() if r.get("status") == "completed"}
    if not completed <= admitted & released:
        raise ValueError("completed requests lack paired ownership trace")
    if not active:
        seconds_by_degree["0"] = seconds_by_degree.get("0", 0.) + wall - previous
    return dict(schema="redknot-cow-observed-overlap-v1", max_same_document_owners=peak,
        max_duplicate_document_owners=peak_duplicates, seconds_by_max_same_document_owners=seconds_by_degree,
        any_same_document_overlap_seconds=any_overlap_seconds,
        overlap_fraction_of_service=any_overlap_seconds/wall if wall else None,
        concurrent_duplicate_source_bytes_upper_bound=peak_bytes,
        share_local=share_local, all_traced_owners_released=not active,
        scope="admitted-to-released trace; byte bound counts only repeated simultaneous local canonical payload, not total memory savings or FLOPs")


def enrich_row(row, *, plan, budget, profile, cache_bytes, page_bytes):
    row["memory_accounting"] = copy.deepcopy(budget)
    row["effective_profile"] = copy.deepcopy(profile)
    row["candidate"] = None if row["arm"] == ARMS[0] else copy.deepcopy(plan["candidate"])
    row["diagnostic_only"] = plan["diagnostic_only"]
    expected_metadata = plan.get("metadata_only_alias_sources", False) and row["arm"] == "redknot_shared"
    if plan.get("metadata_only_alias_sources", False):
        row["metadata_only_alias_sources"] = expected_metadata
    service = row.get("service")
    row["audit_errors"] = []
    if not service:
        return
    expected_prompts = plan["case"]["exact_prompt_sha256"]
    for rec in service.get("records", []):
        index = rec.get("frozen_prefix_index")
        if (type(index) is not int or not 0 <= index < 32
                or rec.get("exact_prompt_sha256") != expected_prompts[index]):
            row["audit_errors"].append("request does not match frozen prefix/prompt")
        if "text" in rec and "output_ids" in rec:
            rec["output_text_sha256"] = digest(rec["text"].encode())
            rec["output_ids_sha256"] = digest(rec["output_ids"])
    if service.get("status") == "complete" and sorted(
            rec.get("frozen_prefix_index", -1) for rec in service.get("records", [])) != list(range(32)):
        row["audit_errors"].append("complete service does not contain every frozen prefix exactly once")
    stats = service.get("engine_stats_before", {})
    for stage in ("engine_stats_before", "engine_stats_after"):
        if (service.get(stage) or {}).get("metadata_only_alias_sources", False) is not expected_metadata:
            row["audit_errors"].append(f"{stage} metadata-only alias-source mode differs from frozen arm")
    expected = 0 if row["arm"] == ARMS[0] else cache_bytes
    for key in ("offline_hidden_cache_bytes", "offline_hidden_cache_limit_bytes", "offline_hidden_cache_peak_bytes"):
        if stats.get(key) != expected:
            row["audit_errors"].append(f"{key} does not equal expected hidden reservation")
    if any(stats.get(k) != 0 for k in ("offline_kv_cache_bytes", "offline_kv_cache_limit_bytes", "offline_kv_cache_peak_bytes")):
        row["audit_errors"].append("unexpected canonical full-KV cache")
    if (budget.get("accounted_reserved_bytes", TOTAL_BYTES+1) > TOTAL_BYTES
            or stats.get("pool", {}).get("reserved_slab_bytes") != budget.get("pool_reserved_bytes")):
        row["audit_errors"].append("actual slab violates frozen shared budget")
    local_heads = sum(p["head_type"] == "local" for layer in profile["policies"] for p in layer)
    names = [f"{plan['case']['case_id']}:document:{i}" for i in range(16)]
    sources = {name:math.ceil(n/256)*local_heads*page_bytes for name,n in zip(names,plan["case"]["chunk_lengths"])}
    try:
        row["sharing_observation"] = overlap_receipt(service, source_bytes_by_document=sources,
            share_local=row["arm"] == "redknot_shared")
    except (KeyError, TypeError, ValueError) as exc:
        row["audit_errors"].append("invalid overlap trace: "+str(exc))
    row["canonical_local_source_bytes_observed"] = stats.get("pool", {}).get("allocated_bytes")


def summarize(rows, plan):
    """No quality filtering. Technical integrity/cleanup failures fail closed."""
    result = dict(quality_gate_applied=False, statistical_significance_claimed=False,
        diagnostic_only=plan["diagnostic_only"], arm_statuses=[], arms={}, comparisons=[])
    for arm in ARMS:
        selected = [r for r in rows if r.get("arm") == arm]
        identities = [r.get("repeat") for r in selected]
        valid = sorted(identities) == list(range(plan["repeats"]))
        for row in selected:
            s = row.get("service") or {}
            records = s.get("records", [])
            ok = (row.get("status") == "complete" and row.get("cleanup_status") == "complete"
                and not row.get("fatal") and not row.get("audit_errors")
                and s.get("status") == "complete" and not s.get("errors")
                and s.get("completed_requests") == len(records) == 32
                and not s.get("failed_requests") and not s.get("cancelled_requests")
                and s.get("ignore_eos") is False
                and all(r.get("status") == "completed" and r.get("max_new_tokens") == 32 for r in records)
                and isinstance(s.get("total_wall_seconds"), (float,int))
                and math.isfinite(s["total_wall_seconds"]) and s["total_wall_seconds"] > 0)
            valid = valid and ok
            result["arm_statuses"].append(dict(arm=arm, repeat=row.get("repeat"),
                status=row.get("status"), cleanup_status=row.get("cleanup_status"), technically_valid=bool(ok),
                audit_errors=row.get("audit_errors", [])))
        seconds = sum((r.get("service") or {}).get("total_wall_seconds", 0.) for r in selected)
        count = sum((r.get("service") or {}).get("completed_requests", 0) for r in selected)
        emitted = sum((r.get("service") or {}).get("completed_output_tokens", 0) for r in selected)
        records = [rec for r in selected for rec in (r.get("service") or {}).get("records",[])]
        quality = {metric:sum(rec["accuracy"][metric] for rec in records)/len(records)
                   for metric in ("f1","em")} if records and all(rec.get("accuracy") for rec in records) else None
        qps = count/seconds if valid and seconds else None
        result["arms"][arm] = dict(all_expected_repeats_technically_valid=bool(valid),
            completed_requests=count, service_seconds=seconds, output_tokens=emitted,
            descriptive_all_prefix_quality=quality,
            stages_seconds={key:sum((r.get("service") or {}).get("stages_seconds",{}).get(key,0.) for r in selected)
                            for key in ("admission","query_forward","decode_forward","release","idle_wait")},
            peak_allocated_bytes=max((r.get("peak_allocated_bytes",0) for r in selected),default=0),
            peak_reserved_bytes=max((r.get("peak_reserved_bytes",0) for r in selected),default=0),
            observed_qps=qps, uninstrumented_qps=None if plan["diagnostic_only"] else qps,
            actual_max_active=max(((r.get("service") or {}).get("peak_active_requests",0) for r in selected), default=0),
            actual_max_forward_batch=max(((r.get("service") or {}).get("max_forward_batch",0) for r in selected),default=0),
            capacity_blocks=sum(rec.get("capacity_blocks",0) for r in selected for rec in (r.get("service") or {}).get("records",[])))
    for base, cand in ((ARMS[0],ARMS[1]),(ARMS[1],ARMS[2]),(ARMS[0],ARMS[2])):
        b,c=result["arms"][base]["uninstrumented_qps"],result["arms"][cand]["uninstrumented_qps"]
        br={(r["repeat"],rec.get("frozen_prefix_index")):rec for r in rows if r.get("arm")==base
            for rec in (r.get("service") or {}).get("records",[])}
        cr={(r["repeat"],rec.get("frozen_prefix_index")):rec for r in rows if r.get("arm")==cand
            for rec in (r.get("service") or {}).get("records",[])}
        pairs=set(br)&set(cr)
        exact=sum(br[k].get("output_ids")==cr[k].get("output_ids")
                  and br[k].get("text")==cr[k].get("text") for k in pairs)
        expected_pairs=32*plan["repeats"]
        result["comparisons"].append(dict(baseline=base,candidate=cand,qps_ratio=c/b if b and c else None,
            paired_outputs=len(pairs),token_identical=sum(br[k].get("output_ids")==cr[k].get("output_ids") for k in pairs),
            text_identical=sum(br[k].get("text")==cr[k].get("text") for k in pairs),
            exact_text_and_ids_identical=exact,
            storage_ablation_outputs_match=(exact==expected_pairs if len(pairs)==expected_pairs else None)
                if base=="redknot_private" and cand=="redknot_shared" else None,
            output_match_is_not_a_task_quality_certificate=True))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--profile",type=Path,default=HERE/"head_class/qwen3-32B_optimal_g15_lf_ret.json")
    parser.add_argument("--item-key",choices=ITEM_KEYS,required=True)
    parser.add_argument("--global-prefix-layers",type=int,choices=(0,63),required=True)
    parser.add_argument("--concurrency",type=int,choices=(4,16),required=True)
    parser.add_argument("--repeats",type=int,choices=(1,2),default=1)
    parser.add_argument("--publication-diagnostics",action="store_true")
    parser.add_argument("--metadata-only-alias-sources",action="store_true",
        help="opt-in metadata snapshots for the shared arm only; use a new source/run directory")
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args(argv)
    if not all(p.is_absolute() for p in (args.model,args.manifest,args.output,args.profile)):
        parser.error("model, manifest, output and profile paths must be absolute")
    workload=json.loads(args.manifest.read_text())
    plan=prepare_plan(workload,args.item_key,args.global_prefix_layers,args.concurrency,
        repeats=args.repeats,publication_diagnostics=args.publication_diagnostics,
        metadata_only_alias_sources=args.metadata_only_alias_sources)
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/"frozen_workload.json",workload)
    save(args.output/"frozen_cow_plan.json",plan)
    evidence=dict(schema="redknot-cow-sharing-v1",status="prepared",plan=plan,
        plan_sha256=digest(plan),workload_sha256=digest(workload),results=[],
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    save(args.output/"results.json",evidence)
    if args.prepare_only:
        return 0
    def interrupt(signum,frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    try:
        import torch
        import transformers
        if transformers.__version__ != "4.57.1" or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError("Transformers4.57.1 and exactly one visible GPU required")
        tokenizer=transformers.AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
        if digest(freeze_workload(SimpleNamespace(manifest=args.manifest,output_tokens=32),tokenizer))!=digest(workload):
            raise ValueError("frozen workload changed")
        bootstrap_imports()
        from sglang.srt.layers.attention.redknot.nonprefix_paged import NonPrefixPagedEngine
        from sglang.srt.layers.attention.redknot.nonprefix_scheduler import QueuedRequest,run_queued
        from sglang.srt.layers.attention.redknot.selective_recovery import SelectiveRecoveryConfig
        from sglang.srt.layers.attention.redknot.hidden_residual_recovery import estimate_hidden_residual_cache_bytes
        from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig
        from nonprefix_profile import load_effective_profile
        profile,diagnostics=load_effective_profile(args.profile,HeadClassConfig,args.global_prefix_layers)
        torch.manual_seed(20261019)
        began=time.perf_counter()
        model=transformers.Qwen3ForCausalLM.from_pretrained(str(args.model),local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",device_map={"":"cuda:0"}).eval()
        torch.cuda.synchronize(0)
        eos=model.generation_config.eos_token_id
        eos=eos if isinstance(eos,list) else [eos]
        if not eos or any(type(x) is not int for x in eos):
            raise ValueError("natural EOS token IDs required")
        sources=[Path(__file__).resolve(),HERE/"benchmark_nonprefix_scale.py",HERE/"benchmark_selective_recovery.py",
            HERE/"benchmark_nonprefix_affinity.py",HERE/"benchmark_redknot_nonprefix.py",HERE/"nonprefix_profile.py",args.profile,
            *[ROOT/"python/sglang/srt/layers/attention/redknot"/n for n in
              ("nonprefix_paged.py","nonprefix_scheduler.py","nonprefix_affinity.py","selective_recovery.py","hidden_residual_recovery.py")],
            *[ROOT/"python/sglang/srt/mem_cache/head_kv"/n for n in ("manager.py","pool.py")]]
        evidence.update(status="running",model_load_count=1,model_load_seconds=time.perf_counter()-began,
            model_config=model.config.to_dict(),model_config_sha256=digest(model.config.to_dict()),
            checkpoint_identity="local checkpoint path and config hash; not a cryptographic weight-file hash",
            profile_diagnostics=diagnostics,runtime=runtime_metadata(torch,transformers),
            source_hashes={str(p):digest(p.read_bytes()) for p in sources})
        case=plan["case"];item=next(i for i in workload["items"] if i["key"]==args.item_key)
        cache_bytes=estimate_hidden_residual_cache_bytes(model.config,case["chunk_lengths"],8,group_size=128)
        page_bytes=256*model.config.head_dim*4
        evidence["hidden_cache_estimator"]=dict(document_token_lengths=case["chunk_lengths"],probe_layers=8,
            group_size=128,expected_bytes=cache_bytes,registered_corpora=1)
        for repeat,arms in enumerate(plan["arm_order_by_repeat"]):
            for arm in arms:
                effective=copy.deepcopy(profile)
                if arm==ARMS[0]:
                    effective.head_class=[["global"]*effective.num_kv_heads for _ in range(effective.num_layers)]
                    effective.dense_prefix_layers=0;effective._cached_tensors.clear()
                metadata=profile_data(effective);budget={}
                factory=arm_factory(NonPrefixPagedEngine,SelectiveRecoveryConfig,arm=arm,candidate=plan["candidate"],
                    cache_bytes=cache_bytes,page_bytes=page_bytes,receipt=budget,
                    publication_diagnostics=args.publication_diagnostics,
                    metadata_only_alias_sources=args.metadata_only_alias_sources)
                def persist(row):
                    enrich_row(row,plan=plan,budget=budget,profile=metadata,cache_bytes=cache_bytes,page_bytes=page_bytes)
                    evidence["results"][:]=[r for r in evidence["results"] if (r["arm"],r["repeat"])!=(row["arm"],row["repeat"])]
                    evidence["results"].append(copy.deepcopy(row))
                    evidence["summary"]=summarize(evidence["results"],plan)
                    save(args.output/"results.json",evidence)
                event("cow_arm_start",arm=arm,repeat=repeat,case=case,memory_accounting=budget,
                    diagnostic_only=args.publication_diagnostics)
                row,fatal=scale.run_arm(model,tokenizer,effective,case,item,arm,repeat,torch=torch,
                    engine_types=(None,factory),queued_request=QueuedRequest,run_queued=run_queued,
                    eos=eos,revision=evidence["model_config_sha256"],warmup=True,persist=persist)
                event("cow_arm_end",arm=arm,repeat=repeat,status=row["status"],qps=(row.get("service") or {}).get("completed_qps"))
                if fatal:
                    evidence["status"]="fatal_supervisor_restart_required";save(args.output/"results.json",evidence);return 2
        valid=all(s["all_expected_repeats_technically_valid"] for s in evidence["summary"]["arms"].values())
        evidence["status"]="complete" if valid else "incomplete"
        save(args.output/"results.json",evidence)
        return 0 if valid else 1
    except BaseException as exc:
        evidence.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
            error=dict(type=type(exc).__name__,message=str(exc)))
        save(args.output/"results.json",evidence)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
