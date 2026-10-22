#!/usr/bin/env python3
"""Read-only report for one frozen non-prefix affinity experiment.

python report_nonprefix_affinity.py --input RUN_DIR [--output NEW_REPORT.json]

The shared-FCFS scheduling ablation is separate from the full-recomputed paged
comparison. No model/CUDA/network operation or mutation of raw evidence occurs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_analyzer():
    """Load stdlib-only analysis functions; main/model imports are not invoked."""
    path = HERE/"benchmark_nonprefix_affinity.py"
    original_path, original_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.path.insert(0, str(HERE))
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("_affinity_readonly_report_analysis", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode
    names = ("benchmark_nonprefix_affinity.py", "benchmark_nonprefix_scale.py", "benchmark_redknot_nonprefix.py")
    return module, {name:digest((HERE/name).read_bytes()) for name in names}


def read_json(path, errors):
    try:
        data = path.read_bytes()
        return json.loads(data), dict(path=str(path.resolve()), file_sha256=digest(data))
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return None, dict(path=str(path.resolve()), file_sha256=None)


def validate_protocol(raw, manifest, plan, analyzer):
    errors = []
    if raw.get("schema") != "redknot-nonprefix-affinity-v1":
        errors.append("unsupported/missing raw results schema")
    if manifest.get("schema") != "redknot-nonprefix-frozen-v1":
        errors.append("unsupported/missing frozen workload schema")
    if plan.get("schema") != "redknot-nonprefix-affinity-plan-v1":
        errors.append("unsupported/missing frozen affinity plan schema")
    if not manifest or raw.get("workload_sha256") != digest(manifest) or plan.get("workload_sha256") != digest(manifest):
        errors.append("workload hash mismatch/missing")
    if not plan or raw.get("plan_sha256") != digest(plan) or raw.get("plan") != plan:
        errors.append("plan hash/content mismatch/missing")
    try:
        keys = [group["item_key"] for group in plan["groups"]]
        requests_per_group = [sum(spec["item_key"] == key for spec in plan["requests"]) for key in keys]
        if not requests_per_group or len(set(requests_per_group)) != 1:
            raise ValueError("unequal/missing frozen replica groups")
        rebuilt = analyzer.prepare_plan(manifest, keys, chunks=plan["chunks"], replicas=requests_per_group[0],
            concurrency=plan["concurrency"], pool_gib=plan["pool_gib"], repeats=plan["repeats"],
            arms=plan["arms"], **plan["affinity_config"])
        for name in ("groups", "requests", "arm_order_by_repeat", "output_tokens", "natural_eos",
                     "global_prefix_layers", "ffn", "independent_primary_qa_count"):
            if rebuilt[name] != plan.get(name):
                errors.append(f"frozen {name} differs from manifest-derived protocol")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"cannot reconstruct frozen protocol: {exc}")
    return errors


def execution_errors(row, plan, raw, analyzer):
    """Check successful rows really used the declared scheduling/cache path."""
    if not analyzer._complete_row(row, plan):
        return []  # already ineligible; failure diagnostics are preserved verbatim
    errors = []
    arm, service = row["arm"], row["service"]
    policy = "share_affinity" if arm.endswith("_affinity") else "fcfs"
    if service.get("admission_policy") != policy:
        errors.append("recorded admission policy differs from arm")
    expected_local = 0. if arm.startswith("recomputed") else raw.get("profile", {}).get("effective_local_ratio")
    for stage in ("engine_stats_before", "engine_stats_after"):
        stats = service.get(stage) or {}
        if stats.get("share_local") is not (not arm.startswith("redknot_private")):
            errors.append(stage+": share_local differs from arm")
        if (not analyzer._finite(expected_local) or not analyzer._finite(stats.get("local_head_ratio"))
                or abs(stats["local_head_ratio"]-expected_local) > 1e-12):
            errors.append(stage+": effective head ratio mismatch/missing")
        if not arm.startswith("recomputed") and (not analyzer._finite(expected_local) or expected_local <= 0):
            errors.append("RedKnot arm has no verifiable local-head reuse")
        if stats.get("backend") != "flash":
            errors.append(stage+": unexpected/missing attention backend")
    return list(dict.fromkeys(errors))


def quality_errors(row, items, analyzer):
    errors = []
    for record in (row.get("service") or {}).get("records", []):
        if record.get("status") != "completed":
            continue
        item = items.get(record.get("item_key"), {})
        if not isinstance(record.get("text"), str) or not item.get("answers"):
            errors.append(f"{record.get('request_id')}: missing full text/ground truth for scoring check")
            continue
        expected = analyzer.scores(record["text"], item["answers"])
        actual = record.get("accuracy")
        if not analyzer._score_valid(actual) or any(abs(actual[m]-expected[m]) > 1e-12 for m in ("f1", "em")):
            errors.append(f"{record.get('request_id')}: stored F1/EM differs from full-output scoring")
    return errors


def gate_status(comparison, eligible, quality_verified, analyzer):
    if not eligible or not quality_verified:
        return "incomplete_or_unverified"
    metrics = [comparison["primary_replica0"], comparison["all_prefix"]]
    metrics += [value[scope] for value in comparison["per_group_quality"].values()
                for scope in ("primary_replica0", "all_prefix")]
    keys = ("f1_drop_pp", "em_drop_pp", "f1_relative_drop_percent", "em_relative_drop_percent")
    if any(analyzer._finite(metric.get(key)) and metric[key] > 2+1e-10 for metric in metrics for key in keys):
        return "failed_observed_2percent_gate"
    if any(metric.get(key) is None for metric in metrics for key in keys):
        return "non_informative_or_missing_metric"
    return "passed_observed_sample_only" if comparison["observed_2percent_gate"] else "incomplete_or_unverified"


def external_gpu_gate_evidence(paths):
    """Preserve supplied assertions without interpreting log text as a gate."""
    evidence, errors = [], []
    for path in paths:
        path = Path(path).resolve()
        try:
            data = path.read_bytes()
            evidence.append(dict(path=str(path), file_sha256=digest(data),
                                 full_text=data.decode("utf-8"), automatically_parsed=False))
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return dict(status="external_evidence_unverified" if evidence else "not_automatically_verified",
                automatically_verified=False, evidence=evidence, evidence_errors=errors,
                note="A completed model benchmark does not prove the independent GPU correctness gate passed. External assertion logs are preserved for manual review only; no test count, skip count, version or CUDA execution is inferred.",
                cow_scope="CPU forced-write COW tests and GPU immutable canonical-read/source-integrity/private-isolation tests are different evidence; this report does not infer a GPU forced-write COW test.")


def build_report(input_path, *, gpu_gate_evidence=()):
    source = Path(input_path).resolve()
    if source.is_dir():
        source = source/"results.json"
    directory = source.parent
    errors = []
    raw, raw_provenance = read_json(source, errors)
    manifest, manifest_provenance = read_json(directory/"frozen_workload.json", errors)
    companion_plan, plan_provenance = read_json(directory/"frozen_affinity_plan.json", errors)
    raw, manifest = raw or {}, manifest or {}
    plan = companion_plan or raw.get("plan") or {}
    analyzer, analysis_hashes = load_analyzer()
    errors.extend(validate_protocol(raw, manifest, plan, analyzer))
    if companion_plan is None:
        errors.append("companion frozen plan absent; embedded plan is diagnostic only")
    rows = raw.get("results", [])
    if any(row.get("arm") not in plan.get("arms", []) for row in rows):
        errors.append("unplanned arm rows present")
    items = {item["key"]:item for item in manifest.get("items", [])}
    complete_run = raw.get("status") == "complete"
    summary, comparisons = {}, []
    try:
        summary = analyzer.summarize_rows(rows, plan)
        comparisons = analyzer.compare_rows(rows, plan)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append(f"cannot summarize malformed evidence: {type(exc).__name__}: {exc}")
    paths, quality_checks = {}, {}
    for arm, values in summary.items():
        selected = [row for row in rows if row["arm"] == arm]
        paths[arm], quality_checks[arm] = [], []
        for row in selected:
            paths[arm] += [f"repeat {row.get('repeat')}: {error}" for error in execution_errors(row, plan, raw, analyzer)]
            quality_checks[arm] += [f"repeat {row.get('repeat')}: {error}" for error in quality_errors(row, items, analyzer)]
        values["execution_path_verified"] = bool(values["all_repeats_complete"] and not paths[arm] and not errors)
        values["execution_path_errors"] = paths[arm]
        values["full_output_scores_verified"] = bool(values["all_repeats_complete"] and not quality_checks[arm] and not errors)
        values["score_integrity_errors"] = quality_checks[arm]
        values["qualified_complete_arm"] = bool(complete_run and values["execution_path_verified"])
        if not values["qualified_complete_arm"]:
            values["headline_qps"] = None
        values["quality_certified"] = False
    for comparison in comparisons:
        a, b = summary[comparison["baseline"]], summary[comparison["candidate"]]
        eligible = bool(comparison["all_repeats_complete"] and a["qualified_complete_arm"] and b["qualified_complete_arm"])
        checked_quality = a["full_output_scores_verified"] and b["full_output_scores_verified"]
        comparison["qualified_complete_pair"] = eligible
        comparison["quality_gate_status"] = gate_status(comparison, eligible, checked_quality, analyzer)
        comparison["observed_2percent_gate"] = comparison["quality_gate_status"] == "passed_observed_sample_only"
        comparison["quality_certified"] = False
        if not eligible:
            comparison["qps_ratio"] = None
        comparison["diagnostic_ratio_scope"] = "provisional ratio may use a successful subset; never a final headline for partial/unverified runs"
        comparison["latency_changes"] = {}
        for field in ("queue_wait_seconds", "ttft_seconds", "service_ttft_seconds", "latency_seconds"):
            base, new = a[field].get("p95"), b[field].get("p95")
            comparison["latency_changes"][field] = dict(
                baseline_p95=base, candidate_p95=new,
                candidate_minus_baseline_p95_seconds=new-base if eligible and analyzer._finite(base) and analyzer._finite(new) else None,
                candidate_over_baseline_p95=new/base if eligible and analyzer._finite(base) and base > 0 and analyzer._finite(new) else None)
    main_pairs = {}
    for label, baseline in (("scheduling_only", "redknot_shared_fcfs"),
                            ("versus_full_recomputed_paged", "recomputed_paged_fcfs")):
        main_pairs[label] = next((value for value in comparisons if value["baseline"] == baseline
                                  and value["candidate"] == "redknot_shared_affinity"), None)
    selected_keys = [group["item_key"] for group in plan.get("groups", [])]
    selected_items = [items[key] for key in selected_keys if key in items]
    primary_keys = sorted({item.get("primary_qa_key", item["key"]) for item in selected_items})
    full_results = []
    for row in rows:
        service = row.get("service") or {}
        # Keep diagnostics and every original output; trace remains available in
        # the hash-addressed raw source, without duplicating it into this report.
        details = {key:copy.deepcopy(value) for key,value in row.items() if key not in ("service", "warmups")}
        details.update(full_outputs=copy.deepcopy(service.get("records", [])),
            service_status=service.get("status"), service_errors=service.get("errors"),
            admission_policy=service.get("admission_policy"), affinity=service.get("affinity"),
            engine_stats_before=service.get("engine_stats_before"), engine_stats_after=service.get("engine_stats_after"))
        full_results.append(details)
    return dict(schema="redknot-nonprefix-affinity-report-v1", generated_at_utc=datetime.now(timezone.utc).isoformat(),
        report_code_sha256=digest(Path(__file__).read_bytes()), current_analysis_code_sha256=analysis_hashes,
        provenance=dict(results=raw_provenance, frozen_workload=manifest_provenance, frozen_plan=plan_provenance,
                        original_source_hashes=raw.get("source_hashes"), model_config_sha256=raw.get("model_config_sha256"),
                        workload_sha256=raw.get("workload_sha256"), plan_sha256=raw.get("plan_sha256")),
        recorded_run_status=raw.get("status"), integrity_verified=not errors, integrity_errors=errors,
        all_run_complete=complete_run, independent_primary_qa_count=len(primary_keys), primary_qa_keys=primary_keys,
        quality_certified=False, hardware=raw.get("hardware"), profile=raw.get("profile"),
        gpu_correctness_gate=external_gpu_gate_evidence(gpu_gate_evidence),
        profile_diagnostics=raw.get("profile_diagnostics"), frozen_plan=plan,
        selected_full_frozen_inputs=selected_items, arm_summary=summary, comparisons=comparisons,
        main_pairs=main_pairs, full_results=full_results,
        scope="single-node experimental hot resident-document burst, NOT production/distributed routing or saturated/SLO serving; tiny QA mechanism scan, no generalization certification",
        timing_scope="includes queue/admission/online recovery/query/decode/sampling/request release and affinity selection; excludes model loading/offline cache construction/warmup/final engine teardown",
        quality_scope="full raw output English normalized F1 and EM; primary replica0 AND all-prefix, each group AND aggregate, both <=2 percentage points AND <=2 relative percent; zero baseline is non-informative",
        warnings=["Scheduling-only speedup is not speedup versus full recomputation; the two main pairs remain separate.",
                  "The reference is all-global paged, not unmodified native HF; no native QPS claim is available in this experiment.",
                  "Two primary QA, their length variants, request prefixes and repeats cannot establish a <=2% generalization guarantee.",
                  "All canonical sources are resident: affinity does not demonstrate eviction/loading savings or multi-node routing.",
                  "Sharing/lease metadata preserves the multihead alias path; this workload does not prove benefits from writes triggering COW splits.",
                  "Natural EOS output lengths can differ; report generated token counts alongside request QPS.",
                  "Queue-wait promotion bounds are not end-to-end latency guarantees; per-group tails and failures remain visible."])


def print_summary(report):
    def n(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "NA"
    print(f"STATUS={report['recorded_run_status']} VERIFIED={report['integrity_verified']} INDEPENDENT_QA={report['independent_primary_qa_count']} QUALITY_CERTIFIED=False")
    print(f"GPU_CORRECTNESS_GATE={report['gpu_correctness_gate']['status']} AUTOMATICALLY_VERIFIED=False")
    for arm, values in report["arm_summary"].items():
        print(f"{arm}: complete={values['qualified_complete_arm']} QPS={n(values['headline_qps'])} observed_diagnostic_QPS={n(values['observed_qps'])} actual_active/forward={values['actual_max_active']}/{values['actual_max_forward_batch']} completed={values['completed_requests']} output_tokens={values['generated_output_tokens']} TTFT_p95={n(values['ttft_seconds']['p95'])} queue_p95={n(values['queue_wait_seconds']['p95'])} completion_p95={n(values['latency_seconds']['p95'])}")
    for label, pair in report["main_pairs"].items():
        if pair is None:
            print(f"MAIN {label}: not measured")
        else:
            print(f"MAIN {label}: {pair['candidate']}/{pair['baseline']}={n(pair['qps_ratio'])} full_pair={pair['qualified_complete_pair']} quality={pair['quality_gate_status']} quality_certified=False")
            for scope in ("primary_replica0", "all_prefix"):
                value = pair[scope]
                print(f"  {scope}: F1_drop_pp={n(value['f1_drop_pp'])} F1_relative_drop_percent={n(value['f1_relative_drop_percent'])} EM_drop_pp={n(value['em_drop_pp'])} EM_relative_drop_percent={n(value['em_relative_drop_percent'])}")
    for error in report["integrity_errors"]:
        print("INTEGRITY_ERROR "+error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-gate-evidence", type=Path, action="append", default=[],
                        help="Optional external assertion log; preserved with hash, not automatically certified (repeatable)")
    args = parser.parse_args(argv)
    source = args.input/"results.json" if args.input.is_dir() else args.input
    output = args.output or source.parent/"report_nonprefix_affinity.json"
    protected = {source.resolve(), (source.parent/"frozen_workload.json").resolve(),
                 (source.parent/"frozen_affinity_plan.json").resolve()}
    protected.update(path.resolve() for path in args.gpu_gate_evidence)
    if output.resolve() in protected or output.exists():
        parser.error("refusing to overwrite an input or existing report")
    report = build_report(source, gpu_gate_evidence=args.gpu_gate_evidence)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(encoded)
    print_summary(report)
    print("REPORT_JSON "+str(output.resolve()))
    return 0 if report["integrity_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
