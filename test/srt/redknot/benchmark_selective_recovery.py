#!/usr/bin/env python3
"""Fixed-cohort selective-recovery experiment against fresh full recomputation.

Single GPU, identical total KV budget, FCFS, natural EOS and full emitted-text
scoring. The retained offline recovery payload (full KV or quantized hidden
residuals) is charged against the pool budget, never hidden beside an equal
pool. Preparation is timed but
excluded from explicitly hot service QPS. Existing eight QA are a known
regression cohort, not unseen held-out evidence or a quality certificate.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from benchmark_nonprefix_affinity import prepare_plan, run_arm, summarize_rows
from benchmark_redknot_nonprefix import (
    HERE, ROOT, bootstrap_imports, digest, event, freeze_workload, profile_data, save,
)
from benchmark_nonprefix_scale import split_exact


BASELINE = "recomputed_paged_fcfs"
DEFAULT_CANDIDATES = [
    dict(name="g60-full", global_prefix_layers=60, row_fraction=1.0, probe_layers=2, block_size=256),
    dict(name="g60-r75", global_prefix_layers=60, row_fraction=.75, probe_layers=2, block_size=256),
    dict(name="g60-r50", global_prefix_layers=60, row_fraction=.5, probe_layers=2, block_size=256),
    dict(name="g48-r50", global_prefix_layers=48, row_fraction=.5, probe_layers=2, block_size=256),
]


def checked_candidates(candidates):
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 8:
        raise ValueError("one to eight predeclared candidates required")
    names = set()
    for item in candidates:
        required = set(DEFAULT_CANDIDATES[0])
        if (not isinstance(item, dict) or not required <= set(item)
                or set(item) - required - {"selection_mode", "recovery_mode"}):
            raise ValueError("candidate fields must match the fixed schema with optional selection_mode/recovery_mode")
        if item.get("selection_mode", "kv_drift") not in ("kv_drift", "query_attention"):
            raise ValueError("selection_mode must be kv_drift or query_attention")
        if item.get("recovery_mode", "offline_kv") not in ("offline_kv", "hidden_residual_int8"):
            raise ValueError("recovery_mode must be offline_kv or hidden_residual_int8")
        if item.get("recovery_mode") == "hidden_residual_int8" and item.get("selection_mode") != "query_attention":
            raise ValueError("hidden_residual_int8 requires explicit query_attention")
        name = item["name"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,63}", name)
                or name in names or name.startswith("recomputed")):
            raise ValueError("candidate names must be unique and not resemble recomputed")
        names.add(name)
        fraction = item["row_fraction"]
        if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError("row_fraction must be finite in (0, 1]")
        for field in ("global_prefix_layers", "probe_layers", "block_size"):
            if type(item[field]) is not int or item[field] < (0 if field == "global_prefix_layers" else 1):
                raise ValueError("layer/block counts must be valid integers")
    return copy.deepcopy(candidates)


def candidate_arm(candidate):
    return "redknot_" + candidate["name"]


def cohort_role(workload):
    if workload.get("selection_role") != "prospectively_frozen_holdout_for_selective_recovery":
        return "known_regression"
    excluded = workload.get("excluded_calibration_keys")
    if (not isinstance(excluded, list) or not excluded or any(not isinstance(key, str) for key in excluded)
            or type(workload.get("seed")) is not int
            or any(item.get("primary_qa_key", item["key"]) in excluded for item in workload["items"])):
        raise ValueError("prospective holdout declaration requires seed, exclusions, and disjoint QA IDs")
    return "prospectively_frozen_holdout"


def prepare_selective_plan(workload, keys, candidates, *, chunks=16, replicas=2,
                           concurrency=4, total_kv_gib=48., repeats=2):
    candidates = checked_candidates(candidates)
    # Reuse strict exact-token/chunk/prefix freezing, not its affinity policy.
    plan = prepare_plan(workload, keys, chunks=chunks, replicas=replicas,
                        concurrency=concurrency, pool_gib=total_kv_gib, repeats=repeats)
    arms = [BASELINE] + [candidate_arm(candidate) for candidate in candidates]
    plan.update(schema="redknot-selective-recovery-plan-v1", arms=arms,
        arm_order_by_repeat=[arms[:] if repeat % 2 == 0 else list(reversed(arms)) for repeat in range(repeats)],
        candidates=candidates, total_kv_budget_bytes=int(total_kv_gib * 1024**3),
        cohort_role=cohort_role(workload), global_prefix_layers=None,
        cohort_selection_provenance=dict(declared_selection_role=workload.get("selection_role"),
            seed=workload.get("seed"), excluded_calibration_keys=workload.get("excluded_calibration_keys", []),
            caveat="role is frozen selection provenance, not a statistical accuracy certificate"),
        head_policy_prefix_semantics="candidate global_prefix_layers protects heads as full-attention, not all document token rows; selective token-row execution is independently controlled by probe_layers and row_fraction",
        scheduling_policy="fcfs", ffn="native MLP on actually executed token rows",
        scope="single GPU hot full-request experimental service; not production HTTP/TP",
        memory_protocol="pool slab plus retained out-of-pool canonical full-KV storage must fit identical total budget; local canonical copies are already inside pool",
        selection="all explicit frozen item IDs and prefixes; candidates fixed before observing outputs; no outcome-based filtering",
        quality_gate="all-prefix aggregate F1 and EM: both <=2 percentage-point and <=2 percent relative drop vs recomputed; primary0 and per-item/dataset are reported, not extra hard gates",
        quality_scope="role follows frozen manifest provenance; independent QA count excludes repeated prefixes/repeats; no statistical or generalization certificate")
    if any(candidate.get("selection_mode") == "query_attention" for candidate in candidates):
        plan["query_selection_contract"] = dict(
            source="exact frozen workload query_token_ids; no answers supplied to recovery",
            query_state="transient causal probe against prefix/document KV; never published or appended to suffix",
            normal_query_forward="unchanged full frozen query tokens after admission",
            accounting="all additional query probe/selection work is included in online admission, TTFT and QPS; query QKV and FFN row vectors reported separately")
    if any(candidate.get("recovery_mode") == "hidden_residual_int8" for candidate in candidates):
        if keys != [item["key"] for item in workload["items"]] or any(
                len(item["prefix_token_ids"]) != replicas for item in workload["items"]):
            raise ValueError("hidden-residual experiment requires every frozen item and every prefix in manifest order")
        plan["memory_protocol"] = "pool slab plus retained out-of-pool offline recovery payload (full KV OR hidden int8 values and FP32 scales) must fit identical total budget; local canonical pages are already inside pool"
        plan["hidden_residual_contract"] = dict(
            offline_storage="group-int8 hidden residuals plus FP32 scales, not retained full canonical KV; local canonical pages remain charged inside slab",
            quantization=dict(dtype="int8", group_size=128, scale_dtype="float32"),
            cache_bytes_formula="sum(document_lengths) * (layers - probe_layers) * (hidden_size + ceil(hidden_size/128)*4)",
            execution="Q, attention and FFN document rows are selected after probe; KV projections execute every document row at every layer",
            accounting="independent Q/KV/attention/FFN row vectors including extra query probe; no unified QKV or total-FLOP saving claim",
            quality="same complete frozen cohort, all prefixes, full emitted-text scoring and unchanged absolute/relative F1+EM gates")
    return plan


def memory_allocation(total_bytes, cache_bytes, page_bytes):
    if any(type(v) is not int or v < 0 for v in (total_bytes, cache_bytes, page_bytes)) or page_bytes <= 0:
        raise ValueError("memory accounting requires nonnegative integer bytes and positive page size")
    pages = (total_bytes - cache_bytes) // page_bytes
    if pages <= 0:
        raise ValueError("canonical full-KV reservation exhausts the total KV budget")
    return dict(total_kv_budget_bytes=total_bytes, canonical_cache_reservation_bytes=cache_bytes,
                pool_capacity_pages=pages, pool_reserved_bytes=pages * page_bytes,
                accounted_reserved_bytes=pages * page_bytes + cache_bytes,
                rounding_unused_bytes=total_bytes - pages * page_bytes - cache_bytes,
                page_bytes=page_bytes)


def cache_active(candidate, num_layers):
    return candidate is not None and candidate["row_fraction"] < 1 and candidate["probe_layers"] < num_layers


def make_engine_factory(Engine, SelectiveConfig, *, candidate, expected_cache_bytes,
                        total_bytes, page_bytes, receipt):
    """Only adapt allocation/configuration; do not alter engine computation."""
    allocation = memory_allocation(total_bytes, expected_cache_bytes, page_bytes)
    receipt.update(allocation)

    def create(model, cfg, **kwargs):
        if kwargs["capacity_pages"] != total_bytes // page_bytes:
            raise RuntimeError("parent runner attempted a different total KV budget")
        kwargs["capacity_pages"] = allocation["pool_capacity_pages"]
        if candidate is not None and expected_cache_bytes > 0:
            selection = ({"selection_mode": "query_attention"}
                         if candidate.get("selection_mode") == "query_attention" else {})
            if candidate.get("recovery_mode") == "hidden_residual_int8":
                selection["recovery_mode"] = "hidden_residual_int8"
            kwargs["selective_recovery"] = SelectiveConfig(
                block_size=candidate["block_size"], row_fraction=candidate["row_fraction"],
                probe_layers=candidate["probe_layers"], max_offline_kv_bytes=expected_cache_bytes, **selection)
        return Engine(model, cfg, **kwargs)
    return create


def candidate_cache_bytes(candidate, model_config, lengths, dtype, *, estimate_kv, estimate_hidden=None):
    """Charge exactly one retained recovery representation, never both."""
    if not cache_active(candidate, model_config.num_hidden_layers):
        return 0
    if candidate.get("recovery_mode") == "hidden_residual_int8":
        if estimate_hidden is None:
            raise ValueError("hidden-residual estimator unavailable")
        return estimate_hidden(model_config, lengths, candidate["probe_layers"], group_size=128)
    return estimate_kv(model_config, lengths, dtype)


def enrich_row(row, *, candidate, profile_metadata, budget, role="known_regression"):
    row["candidate"] = copy.deepcopy(candidate)
    row["effective_profile"] = copy.deepcopy(profile_metadata)
    row["memory_accounting"] = copy.deepcopy(budget)
    row["cohort_role"] = role
    for record in (row.get("service") or {}).get("records", []):
        if isinstance(record.get("text"), str):
            record["output_text_sha256"] = digest(record["text"].encode())
        if isinstance(record.get("output_ids"), list):
            record["output_ids_sha256"] = digest(record["output_ids"])
    return row


def runtime_metadata(torch, transformers):
    packages = {}
    for name in ("flash-attn", "triton", "safetensors", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    driver_command = ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"]
    try:
        driver = dict(command=driver_command, output=subprocess.check_output(driver_command, text=True, timeout=15).strip())
    except (OSError, subprocess.SubprocessError) as exc:
        driver = dict(command=driver_command, error=dict(type=type(exc).__name__, message=str(exc)))
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        torch=torch.__version__, transformers=transformers.__version__, torch_cuda=torch.version.cuda,
        packages=packages, driver=driver, visible_physical_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"),
        visible_cuda_device_count=torch.cuda.device_count(), compute_capability=list(torch.cuda.get_device_capability(0)),
        mode=dict(model_eval=True, parameter_dtype="bfloat16", attention_backend="flash_attention_2",
                  generation="greedy argmax with natural EOS; no ignored EOS", service="single-GPU FCFS; hot resident canonical documents",
                  torch_float32_matmul_precision=torch.get_float32_matmul_precision(),
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--items", nargs="+", help="default: every frozen manifest item, in manifest order")
    parser.add_argument("--candidates-json", type=Path, help="explicit predeclared candidate list; default four candidates documented in source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=HERE / "head_class/qwen3-32B_optimal_g15_lf_ret.json")
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--total-kv-gib", type=float, default=48.)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.output.is_absolute() or not args.model.is_absolute():
        parser.error("model/output must be absolute paths")
    workload = json.loads(args.manifest.read_text())
    keys = args.items or [item["key"] for item in workload["items"]]
    candidates = json.loads(args.candidates_json.read_text()) if args.candidates_json else DEFAULT_CANDIDATES
    plan = prepare_selective_plan(workload, keys, candidates, chunks=args.chunks,
        replicas=args.replicas, concurrency=args.concurrency, total_kv_gib=args.total_kv_gib, repeats=args.repeats)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "frozen_workload.json", workload)
    save(args.output / "frozen_selective_plan.json", plan)
    if args.prepare_only:
        return 0
    import torch
    import transformers
    if transformers.__version__ != "4.57.1" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("verified Transformers 4.57.1 and exactly one visible CUDA device required")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    verified = freeze_workload(SimpleNamespace(manifest=args.manifest, output_tokens=32), tokenizer)
    if digest(verified) != digest(workload):
        raise RuntimeError("frozen workload changed during preparation")
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    bootstrap_imports()
    from sglang.srt.layers.attention.redknot.nonprefix_paged import NonPrefixPagedEngine
    from sglang.srt.layers.attention.redknot.nonprefix_scheduler import QueuedRequest, run_queued
    from sglang.srt.layers.attention.redknot.selective_recovery import SelectiveRecoveryConfig, estimate_selective_cache_bytes
    hidden_requested = bool(plan.get("hidden_residual_contract"))
    estimate_hidden = None
    if hidden_requested:
        from sglang.srt.layers.attention.redknot.hidden_residual_recovery import estimate_hidden_residual_cache_bytes
        estimate_hidden = estimate_hidden_residual_cache_bytes
    from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig
    from nonprefix_profile import load_effective_profile
    profiles, diagnostics = {}, {}
    for candidate in plan["candidates"]:
        arm = candidate_arm(candidate)
        profiles[arm], diagnostics[arm] = load_effective_profile(args.profile, HeadClassConfig, candidate["global_prefix_layers"])
    base = copy.deepcopy(next(iter(profiles.values())))
    base.head_class = [["global"] * base.num_kv_heads for _ in range(base.num_layers)]
    base.dense_prefix_layers = 0
    base._cached_tensors.clear()
    profiles[BASELINE] = base
    torch.manual_seed(20261019)
    began = time.perf_counter()
    model = transformers.Qwen3ForCausalLM.from_pretrained(str(args.model), local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": "cuda:0"}).eval()
    torch.cuda.synchronize(0)
    loaded = time.perf_counter() - began
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(type(x) is not int for x in eos):
        raise ValueError("valid natural EOS IDs required")
    revision = digest(model.config.to_dict())
    source_paths = [Path(__file__).resolve(), HERE / "benchmark_nonprefix_affinity.py",
        HERE / "benchmark_nonprefix_scale.py", HERE / "benchmark_redknot_nonprefix.py",
        HERE / "nonprefix_profile.py", args.profile,
        *[ROOT / "python/sglang/srt/layers/attention/redknot" / name for name in
          ("nonprefix_paged.py", "nonprefix_scheduler.py", "nonprefix_affinity.py", "selective_recovery.py")],
        *[ROOT / "python/sglang/srt/mem_cache/head_kv" / name for name in ("manager.py", "pool.py")]]
    if hidden_requested:
        source_paths.append(ROOT / "python/sglang/srt/layers/attention/redknot/hidden_residual_recovery.py")
    result = dict(schema="redknot-selective-recovery-v1", status="running", plan=plan,
        plan_sha256=digest(plan), workload_sha256=digest(workload), model_config_sha256=revision,
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        model_load_seconds=loaded, profile_diagnostics=diagnostics, runtime=runtime_metadata(torch, transformers),
        hardware=dict(gpu=torch.cuda.get_device_name(0), total_bytes=torch.cuda.get_device_properties(0).total_memory),
        source_hashes={str(path): digest(path.read_bytes()) for path in source_paths}, results=[])
    items = {item["key"]: item for item in workload["items"]}
    lengths = [len(part) for group in plan["groups"] for part in
               split_exact(sum(items[group["item_key"]]["document_token_ids"], []), plan["chunks"])]
    if hidden_requested:
        result["hidden_residual_cache_estimator"] = dict(model_config=model.config.to_dict(),
            document_token_lengths=lengths, group_size=128, value_bytes=1, scale_bytes=4,
            scope="each distinct registered document ID once; int8 H values without padding and ceil(H/128) FP32 scales per retained layer/token")
    page_bytes = 256 * model.config.head_dim * 4
    candidate_by_arm = {candidate_arm(item): item for item in plan["candidates"]}
    for repeat, arms in enumerate(plan["arm_order_by_repeat"]):
        for arm in arms:
            candidate = candidate_by_arm.get(arm)
            cache_bytes = candidate_cache_bytes(candidate, model.config, lengths, torch.bfloat16,
                estimate_kv=estimate_selective_cache_bytes, estimate_hidden=estimate_hidden)
            budget = {}
            factory = make_engine_factory(NonPrefixPagedEngine, SelectiveRecoveryConfig,
                candidate=candidate, expected_cache_bytes=cache_bytes,
                total_bytes=plan["total_kv_budget_bytes"], page_bytes=page_bytes, receipt=budget)
            metadata = profile_data(profiles[arm])
            def persist(row):
                enrich_row(row, candidate=candidate, profile_metadata=metadata, budget=budget, role=plan["cohort_role"])
                result["results"][:] = [old for old in result["results"] if
                                        (old["arm"], old["repeat"]) != (row["arm"], row["repeat"])]
                result["results"].append(copy.deepcopy(row))
                result["arm_summary"] = summarize_rows(result["results"], plan)
                save(args.output / "results.json", result)
            event("selective_arm_start", arm=arm, repeat=repeat, memory_accounting=budget)
            row, fatal = run_arm(model, tokenizer, profiles[arm], workload, plan, arm, repeat,
                torch=torch, Engine=factory, Request=QueuedRequest, run_queued=run_queued,
                AffinityConfig=None, revision=revision, eos=eos, persist=persist)
            event("selective_arm_finished", arm=arm, repeat=repeat, status=row["status"],
                  qps=(row.get("service") or {}).get("completed_qps"))
            if fatal:
                result["status"] = "fatal_supervisor_restart_required"
                save(args.output / "results.json", result)
                return 2
    result["status"] = "complete" if all(row["status"] == "complete" for row in result["results"]) else "incomplete"
    save(args.output / "results.json", result)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
