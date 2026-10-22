#!/usr/bin/env python3
"""CPU-only audit of all six COW screen cells and two separate diagnostics.

No quality gate or winner selection. Each ratio uses its own same-GPU arm.
Missing/failed cases remain visible; diagnostic timings never become QPS claims.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import tempfile

from benchmark_cow_sharing import ARMS, GEOMETRIES, ITEM_KEYS, TOTAL_BYTES, enrich_row, prepare_plan, summarize
from benchmark_nonprefix_affinity import _distribution
from benchmark_redknot_nonprefix import digest, scores
from benchmark_selective_recovery import memory_allocation

PUBLICATION_STAGES = ("head_pack", "head_write_and_release", "alias_publish", "boundary_partition", "final_bind")
SOURCE_FILES = (
    "test/srt/redknot/benchmark_cow_sharing.py", "test/srt/redknot/benchmark_nonprefix_scale.py",
    "test/srt/redknot/benchmark_selective_recovery.py", "test/srt/redknot/benchmark_nonprefix_affinity.py",
    "test/srt/redknot/benchmark_redknot_nonprefix.py", "test/srt/redknot/nonprefix_profile.py",
    "test/srt/redknot/head_class/qwen3-32B_optimal_g15_lf_ret.json",
    *["python/sglang/srt/layers/attention/redknot/"+n for n in
      ("nonprefix_paged.py", "nonprefix_scheduler.py", "nonprefix_affinity.py", "selective_recovery.py", "hidden_residual_recovery.py")],
    "python/sglang/srt/mem_cache/head_kv/manager.py", "python/sglang/srt/mem_cache/head_kv/pool.py")


def _reject(value):
    raise ValueError("nonfinite JSON: "+str(value))


def _float(value):
    number=float(value)
    return number if math.isfinite(number) else _reject(value)


def read_json(path):
    data=Path(path).read_bytes()
    return json.loads(data,parse_constant=_reject,parse_float=_float),digest(data)


def expected_jobs():
    jobs=[]
    for key in ITEM_KEYS:
        dataset=key.split(":")[0]
        for prefix,concurrency in GEOMETRIES:
            jobs.append(dict(name=f"screen_{dataset}_g{prefix}_c{concurrency}",item_key=key,
                global_prefix_layers=prefix,concurrency=concurrency,diagnostic_only=False))
    for key in ITEM_KEYS:
        jobs.append(dict(name=f"diagnostic_{key.split(':')[0]}_g0_c16",item_key=key,
            global_prefix_layers=0,concurrency=16,diagnostic_only=True))
    return jobs


def verify_sources(raw, source_root):
    hashes=raw.get("source_hashes") or {}
    anchors=[Path(p) for p in hashes if Path(p).name=="benchmark_cow_sharing.py"]
    if len(anchors)!=1:
        raise ValueError("missing/ambiguous frozen source anchor")
    remote=anchors[0].parents[3];root=Path(source_root).resolve();verified={}
    for path,sha in hashes.items():
        relative=Path(path).relative_to(remote)
        local=(root/relative).resolve()
        if not local.is_relative_to(root) or digest(local.read_bytes())!=sha:
            raise ValueError("source bytes mismatch: "+str(relative))
        verified[str(relative)]=sha
    if set(verified)!=set(SOURCE_FILES):
        raise ValueError("missing or unexpected execution source receipt")
    return verified


def audit(raw,workload,plan,expected,source_root):
    """Rebuild plan, cache allocation and output scores; do not trust summary."""
    rebuilt=prepare_plan(workload,expected["item_key"],expected["global_prefix_layers"],
        expected["concurrency"],repeats=plan["repeats"],publication_diagnostics=expected["diagnostic_only"],
        metadata_only_alias_sources=plan.get("metadata_only_alias_sources",False))
    if ("metadata_only_alias_sources" in expected
            and expected["metadata_only_alias_sources"] is not plan.get("metadata_only_alias_sources",False)):
        raise ValueError("metadata-only source mode differs from expected experiment")
    if (rebuilt!=plan or raw.get("plan")!=plan or raw.get("plan_sha256")!=digest(plan)
            or raw.get("workload_sha256")!=digest(workload)
            or raw.get("schema")!="redknot-cow-sharing-v1"):
        raise ValueError("frozen plan/workload identity mismatch")
    source=verify_sources(raw,source_root)
    cfg=raw.get("model_config") or {}
    if digest(cfg)!=raw.get("model_config_sha256"):
        raise ValueError("model configuration hash mismatch")
    for field in ("num_hidden_layers","num_key_value_heads","hidden_size","head_dim"):
        if type(cfg.get(field)) is not int or cfg[field]<=0:
            raise ValueError("invalid model geometry")
    lengths=plan["case"]["chunk_lengths"]
    hidden=sum(lengths)*(cfg["num_hidden_layers"]-8)*(cfg["hidden_size"]+math.ceil(cfg["hidden_size"]/128)*4)
    if hidden<=0 or raw.get("hidden_cache_estimator")!=dict(document_token_lengths=lengths,
            probe_layers=8,group_size=128,expected_bytes=hidden,registered_corpora=1):
        raise ValueError("incorrect single-corpus hidden-cache estimator")
    runtime=raw.get("runtime") or {}
    if runtime.get("visible_cuda_device_count")!=1:
        raise ValueError("not a single-visible-GPU run")
    expected_identities={(arm,r) for r in range(plan["repeats"]) for arm in ARMS}
    observed=[(r.get("arm"),r.get("repeat")) for r in raw.get("results",[])]
    if len(set(observed))!=len(observed) or not set(observed)<=expected_identities:
        raise ValueError("duplicate/unexpected arm or repeat")
    rows=copy.deepcopy(raw.get("results",[]));item=next(i for i in workload["items"] if i["key"]==expected["item_key"])
    candidate_profiles=set()
    for row in rows:
        arm=row["arm"];candidate=None if arm==ARMS[0] else plan["candidate"]
        if row.get("candidate")!=candidate or row.get("diagnostic_only")!=expected["diagnostic_only"]:
            raise ValueError("arm algorithm/diagnostic identity mismatch")
        expected_metadata=plan.get("metadata_only_alias_sources",False) and arm=="redknot_shared"
        if ("metadata_only_alias_sources" in plan
                and row.get("metadata_only_alias_sources") is not expected_metadata):
            raise ValueError("arm metadata-only source receipt differs from plan")
        if row.get("metadata_only_alias_sources",False) is not expected_metadata:
            raise ValueError("unexpected metadata-only arm mode")
        profile=row["effective_profile"];policies=profile["policies"]
        if (len(policies)!=cfg["num_hidden_layers"] or any(len(line)!=cfg["num_key_value_heads"] for line in policies)
                or digest(policies)!=profile["sha256"]):
            raise ValueError("policy topology/hash mismatch")
        local=sum(p["head_type"]=="local" for line in policies for p in line)
        if profile["effective_local_ratio"]!=local/(cfg["num_hidden_layers"]*cfg["num_key_value_heads"]):
            raise ValueError("incorrect actual local-head ratio")
        if arm==ARMS[0] and any(p["head_type"]!="global" for line in policies for p in line):
            raise ValueError("recomputed baseline is not all-global")
        if candidate:
            candidate_profiles.add(profile["sha256"])
            if local!=(4 if candidate["global_prefix_layers"]==63 else 435):
                raise ValueError("effective local count differs from predeclared profile")
        page_bytes=256*cfg["head_dim"]*4
        budget=memory_allocation(TOTAL_BYTES,hidden if candidate else 0,page_bytes)
        if row.get("memory_accounting")!=budget:
            raise ValueError("same-total-budget allocation mismatch")
        service=row.get("service")
        if not service:
            continue
        for stage in ("engine_stats_before","engine_stats_after"):
            stats=service.get(stage) or {}
            expected_config=dict(block_size=256,row_fraction=.5,probe_layers=8,
                max_offline_kv_bytes=hidden,selection_mode="query_attention",recovery_mode="hidden_residual_int8") if candidate else None
            if (stats.get("selective_recovery")!=expected_config
                    or stats.get("share_local")!=(arm!="redknot_private")
                    or stats.get("local_head_ratio")!=profile["effective_local_ratio"]
                    or stats.get("documents")!=16):
                raise ValueError("actual storage/algorithm/corpus differs from arm")
            if stats.get("metadata_only_alias_sources",False) is not expected_metadata:
                raise ValueError("actual metadata-only alias-source mode differs from arm")
        if service.get("admission_policy")!="fcfs":
            raise ValueError("unexpected scheduler")
        for rec in service.get("records",[]):
            if rec.get("status")!="completed":
                continue
            if scores(rec["text"],item["answers"])!=rec.get("accuracy"):
                raise ValueError("emitted-text score mismatch")
            if (digest(rec["text"].encode())!=rec.get("output_text_sha256")
                    or digest(rec["output_ids"])!=rec.get("output_ids_sha256")):
                raise ValueError("output hash mismatch")
            ad=rec.get("admission_stats") or {}
            if ad.get("stage_timing_synchronizes") is not True:
                raise ValueError("unsynchronized admission timing")
            diag=ad.get("publication_diagnostics")
            if expected["diagnostic_only"]:
                if (not diag or diag.get("schema")!="redknot-publication-stages-v1"
                        or diag.get("diagnostic_only") is not True
                        or set(diag.get("stage_seconds",{}))!=set(PUBLICATION_STAGES)):
                    raise ValueError("missing five-stage diagnostic receipt")
            elif diag is not None:
                raise ValueError("instrumented timing mislabeled as formal QPS")
        pool=row.get("pool_after_close") or {}
        if row.get("cleanup_status")=="complete" and (pool.get("free_pages")!=budget["pool_capacity_pages"]
                or any(pool.get(k)!=0 for k in ("reserved_pages","live_pages","retiring_pages","pinned_pages"))
                or pool.get("poisoned") is not False):
            raise ValueError("cleanup receipt retains page ownership")
        enrich_row(row,plan=plan,budget=budget,profile=profile,cache_bytes=hidden,page_bytes=page_bytes)
    if len(candidate_profiles)>1:
        raise ValueError("private/COW effective profiles differ")
    return rows,source


def _maximum(values):
    values=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
    return max(values) if values else None


def arm_details(rows,arm,summary,diagnostic):
    selected=[r for r in rows if r.get("arm")==arm]
    services=[r["service"] for r in selected if isinstance(r.get("service"),dict)]
    records=[rec for s in services for rec in s.get("records",[])]
    completed=[r for r in records if r.get("status")=="completed"]
    admissions=[r.get("admission_stats") or {} for r in records]
    details=copy.deepcopy(summary)
    details.update(observed_repeat_statuses=[{k:r.get(k) for k in
        ("repeat","status","fatal","cleanup_status","error","cleanup_error","audit_errors")} for r in selected],
        metadata_only_alias_sources_by_repeat=[dict(repeat=r.get("repeat"),
            requested=r.get("metadata_only_alias_sources",False),
            before=(r.get("service") or {}).get("engine_stats_before",{}).get("metadata_only_alias_sources",False),
            after=(r.get("service") or {}).get("engine_stats_after",{}).get("metadata_only_alias_sources",False)) for r in selected],
        latency_seconds={field:_distribution([r.get(field) for r in completed]) for field in
            ("queue_wait_seconds","ttft_seconds","service_ttft_seconds","latency_seconds")},
        actual_output_lengths=_distribution([len(r.get("output_ids",[])) for r in completed]),
        recovery_publication_seconds={key:sum(a.get(key,0.) for a in admissions)
            for key in ("prefix_prefill_s","online_recovery_s","kv_publish_and_partition_s")},
        preparation_seconds=[{k:r.get(k) for k in ("repeat","engine_construction_seconds","offline_document_seconds")} for r in selected],
        overlap_by_repeat=[dict(repeat=r.get("repeat"),observation=r.get("sharing_observation")) for r in selected],
        max_same_document_owners=_maximum((r.get("sharing_observation") or {}).get("max_same_document_owners") for r in selected),
        source_memory_by_repeat=[dict(repeat=r.get("repeat"),
            canonical_local_source_bytes_observed=r.get("canonical_local_source_bytes_observed"),
            concurrent_duplicate_source_bytes_upper_bound=(r.get("sharing_observation") or {}).get("concurrent_duplicate_source_bytes_upper_bound"),
            accounting=r.get("memory_accounting"),effective_local_ratio=(r.get("effective_profile") or {}).get("effective_local_ratio")) for r in selected],
        max_pool_high_water_pages=_maximum(p.get("high_water_pages") for r in selected for p in
            [r.get("pool_after_close") or {},(r.get("service") or {}).get("engine_stats_before",{}).get("pool",{}),
             (r.get("service") or {}).get("engine_stats_after",{}).get("pool",{})]),
        max_observed_live_pages=_maximum(a.get("pool_after_admit",{}).get("live_pages") for a in admissions),
        max_observed_pinned_pages=_maximum(a.get("pool_after_admit",{}).get("pinned_pages") for a in admissions),
        memory_scopes=dict(pool_high_water="since engine creation, including offline construction and warmup",
            cuda_allocator_peak="reset after warmup; whole-process service peak, includes weights/resident state",
            retained_budget="hidden payload plus preallocated slab, not entire process allocation"))
    if diagnostic:
        stages=[a["publication_diagnostics"] for a in admissions if a.get("publication_diagnostics")]
        details["publication_diagnostics"]=dict(diagnostic_only=True,requests_with_receipt=len(stages),
            stage_seconds={k:sum(s.get("stage_seconds",{}).get(k,0.) for s in stages) for k in PUBLICATION_STAGES},
            total_seconds=sum(s.get("total_seconds",0.) for s in stages),
            scope="instrumented, includes four extra syncs per admission; never an uninstrumented QPS comparison")
    return details


def summarize_job(directory,expected,source_root):
    directory=Path(directory)
    entry=dict(name=expected["name"],expected=expected,run_directory=str(directory.resolve()),status="missing",
        errors=[],integrity_verified=False,source_file_bytes_verified=False,arms={},comparisons=[],input_files_sha256={})
    if not directory.exists():
        return entry
    raw=None
    try:
        payload={}
        for name in ("results.json","frozen_workload.json","frozen_cow_plan.json"):
            payload[name],entry["input_files_sha256"][name]=read_json(directory/name)
        raw=payload["results.json"];plan=payload["frozen_cow_plan.json"]
        entry["raw_status"]=raw.get("status");entry["runtime"]=raw.get("runtime")
        entry["metadata_only_alias_sources"]=plan.get("metadata_only_alias_sources",False)
        entry["run_error"]=raw.get("error")
        rows,verified=audit(raw,payload["frozen_workload.json"],plan,expected,source_root)
        summary=summarize(rows,plan)
        entry.update(source_file_bytes_verified=True,verified_source_files=verified,
            arms={arm:arm_details(rows,arm,summary["arms"][arm],expected["diagnostic_only"]) for arm in ARMS},
            comparisons=summary["comparisons"],quality_gate_applied=False,
            plan_sha256=digest(plan),workload_sha256=digest(payload["frozen_workload.json"]))
        complete=raw.get("status")=="complete" and all(v["all_expected_repeats_technically_valid"] for v in summary["arms"].values())
        entry["integrity_verified"]=complete;entry["status"]="verified" if complete else "incomplete_or_failed"
        if not complete:
            entry["errors"].append("one or more expected arms/repeats is incomplete, technically invalid, or failed cleanup")
            for comp in entry["comparisons"]:comp["qps_ratio"]=None
        return entry
    except (OSError,ValueError,TypeError,KeyError,IndexError,AttributeError,ZeroDivisionError) as exc:
        entry["status"]="audit_failed";entry["errors"].append(f"{type(exc).__name__}: {exc}")
        if isinstance(raw,dict):
            entry["failure_diagnostics"]=[dict(arm=r.get("arm"),repeat=r.get("repeat"),status=r.get("status"),
                error=r.get("error"),cleanup_status=r.get("cleanup_status"),cleanup_error=r.get("cleanup_error"),
                service_status=(r.get("service") or {}).get("status"),service_errors=(r.get("service") or {}).get("errors"))
                for r in raw.get("results",[]) if isinstance(r,dict)]
        return entry


def summarize_matrix(directory,source_root):
    root=Path(directory).resolve();source_root=Path(source_root)
    if not source_root.is_absolute() or not source_root.is_dir():
        raise ValueError("explicit existing absolute frozen source-root required")
    expected=expected_jobs();names={j["name"] for j in expected}
    jobs=[summarize_job(root/j["name"],j,source_root) for j in expected]
    unexpected=sorted(str(p) for pattern in ("screen_*/results.json","diagnostic_*/results.json")
                      for p in root.glob(pattern) if p.parent.name not in names)
    return dict(schema="redknot-cow-sharing-matrix-summary-v1",matrix_root=str(root),source_root=str(source_root),
        independent_primary_qa_count=2,expected_formal_cells=6,expected_diagnostic_cells=2,
        screen_jobs=[j for j in jobs if not j["expected"]["diagnostic_only"]],
        diagnostic_jobs=[j for j in jobs if j["expected"]["diagnostic_only"]],
        unplanned_results_retained_as_paths=unexpected,
        all_six_formal_cells_verified=all(j["status"]=="verified" for j in jobs[:6]),
        quality_gate_applied=False,quality_certified=False,winner_selected=False,
        statistical_significance_claimed=False,
        scope="independent single-GPU hot experimental FCFS service, not production HTTP/TP8; never pool GPU jobs",
        caveats=["32 prefixes per case and repeated rounds do not increase independent QA N=2",
            "long inputs are complete real distractor+primary augmentation, not native independent long QA",
            "g0 versus g63 changes head/recovery policy; do not attribute their entire difference to COW",
            "COW mechanism comparison is shared/private within the same cell and same GPU",
            "private-copy materializes full local document KV, not an optimal compact-window private baseline",
            "all emitted text scores retained; quality explicitly not an acceptance gate",
            "ttft_seconds includes arrival queueing; service_ttft_seconds begins at admission",
            "overlap byte bounds are not measured total memory savings or compute savings",
            "diagnostic stages contain extra synchronization and are excluded from all QPS ratios"])


def write_new_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",prefix=".cow-summary-",dir=path.parent,delete=False) as f:
            temporary=Path(f.name);f.write(encoded);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)  # Atomic publication; fails if destination already exists.
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    result=summarize_matrix(args.root,args.source_root)
    write_new_json(args.output,result)
    for job in result["screen_jobs"]+result["diagnostic_jobs"]:
        print(job["name"],job["status"],[(c["baseline"],c["candidate"],c["qps_ratio"]) for c in job["comparisons"]])
    return 0 if result["all_six_formal_cells_verified"] else 1


if __name__=="__main__":
    raise SystemExit(main())
