#!/usr/bin/env python3
"""Read-only evidence summaries for genuine-nonprefix experiment results.

Example:
  python report_nonprefix_results.py --inputs pilot_v4/results.json \
      pilot_v5/results.json --output nonprefix_evidence.json

Each run remains separate: prompts, profiles, and sample sets can differ.
Full unedited outputs are included in JSON; --print-outputs also prints them.
No model, tokenizer, CUDA, network, or GPU-holder operation is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _analysis_module():
    path = HERE / "benchmark_redknot_nonprefix.py"
    spec = importlib.util.spec_from_file_location("_nonprefix_evidence_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha256(path.read_bytes())


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values, *, eligible_count=None):
    """Linear-interpolated empirical quantiles, not confidence intervals."""
    values = sorted(float(value) for value in values if _finite(value) and value >= 0)
    def quantile(fraction):
        if not values:
            return None
        position = fraction * (len(values) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return dict(count=len(values), eligible_count=eligible_count,
                missing_count=None if eligible_count is None else eligible_count-len(values),
                mean=statistics.mean(values) if values else None,
                p50=quantile(0.5), p95=quantile(0.95),
                min=values[0] if values else None, max=values[-1] if values else None,
                quantile_method="linear empirical interpolation; not a confidence interval")


def _metric_sum(rows, field):
    values = [row.get(field) for row in rows]
    present = [value for value in values if _finite(value)]
    return dict(total=sum(present) if present else None, observed_count=len(present),
                missing_count=len(values)-len(present))


def _pool_evidence(row):
    service = row["service"]
    pool = (service.get("engine_stats_after") or {}).get("pool")
    if not isinstance(pool, dict):
        return None
    pages, capacity = pool.get("high_water_pages"), pool.get("capacity_pages")
    slab = pool.get("reserved_slab_bytes")
    page_bytes = int(slab / capacity) if _finite(slab) and _finite(capacity) and capacity > 0 and slab % capacity == 0 else None
    snapshots = [record.get("admission_stats", {}).get("pool_after_admit")
                 for record in service.get("records", []) if isinstance(record.get("admission_stats"), dict)]
    live = [snapshot["live_pages"] for snapshot in snapshots
            if isinstance(snapshot, dict) and _finite(snapshot.get("live_pages"))]
    return dict(item_key=row["item_key"], repeat=row.get("repeat", 0),
                occupied_page_high_water=pages,
                occupied_page_high_water_bytes=pages*page_bytes if _finite(pages) and page_bytes is not None else None,
                page_bytes=page_bytes,
                live_pages_max_observed_after_admission=max(live) if live else None,
                fixed_slab_capacity_pages=capacity, fixed_slab_reserved_bytes=slab,
                scope="engine lifetime including discarded warmup and earlier repeats; high-water is not reset per service",
                interpretation="occupied physical pages include canonical sources, private histories, and reserved suffix pages; not whole-model memory savings")


def _full_output(record):
    # No answer extraction, truncation, paraphrasing, or scoring cleanup.
    fields = ("request_id", "status", "finish_reason", "text", "output_ids", "accuracy",
              "ttft_seconds", "service_ttft_seconds", "latency_seconds", "error",
              "doc_offsets", "history_tokens", "prefix_tokens", "query_tokens")
    return {field: record[field] for field in fields if field in record}


def analyze_run(path, analyzer):
    path = Path(path).resolve()
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    if raw.get("schema") != "redknot-nonprefix-e2e-v1":
        raise ValueError(f"unsupported input schema: {raw.get('schema')!r}")
    rows = raw.get("results", [])
    config = raw.get("config", {})
    manifest_path = path.parent / "frozen_workload.json"
    warnings = []
    manifest = None
    manifest_bytes = None
    if manifest_path.exists():
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != "redknot-nonprefix-frozen-v1":
            raise ValueError("unsupported companion frozen workload schema")
        items = manifest.get("items", [])
        verified = bool(raw.get("workload_sha256")) and analyzer.digest(manifest) == raw["workload_sha256"]
        if not verified:
            warnings.append("Companion workload hash does not match the recorded workload digest; no quality acceptance.")
    else:
        # This permits useful failed-run diagnostics, but never infers that an
        # observed subset was the whole planned evaluation.
        observed = {}
        for row in rows:
            observed.setdefault(row["item_key"], dict(key=row["item_key"], dataset=row["dataset"]))
        items = list(observed.values())
        verified = False
        warnings.append("Companion frozen_workload.json is absent: expected sample coverage is unknown; no quality acceptance.")
    summary = analyzer.summarize(rows, items, expected_repeats=config.get("repeats", 1))
    run_complete = raw.get("status") == "completed"
    if not run_complete:
        warnings.append("The recorded run is not completed; partial measurements remain visible, but no final acceptance is reported.")
    observed_keys = {row["item_key"] for row in rows}
    if len(items) < 50:
        warnings.append("Small observed sample: prefix replicas and repeats are not independent QA examples; not a <=2% generalization certificate.")
    if manifest is not None:
        for manifest_field, config_field in (("instruction_at_query", "instruction_at_query"),
                                              ("varying_prefix_lengths", "vary_prefix_lengths")):
            if (manifest_field in manifest and config_field in config
                    and manifest[manifest_field] != config[config_field]):
                warnings.append(f"Frozen workload {manifest_field}={manifest[manifest_field]!r} overrides "
                                f"CLI config {config_field}={config[config_field]!r}; report the manifest protocol.")
    arm_metrics = {}
    for arm, aggregate in summary["arms"].items():
        arm_rows = [row for row in rows if row["arm"] == arm]
        completed = [record for row in arm_rows for record in row["service"].get("records", [])
                     if record.get("status") == "completed"]
        stages = defaultdict(float)
        for row in arm_rows:
            for stage, seconds in row["service"].get("stages_seconds", {}).items():
                if _finite(seconds):
                    stages[stage] += seconds
        admission = [record["admission_stats"] if isinstance(record.get("admission_stats"), dict) else {}
                     for row in arm_rows for record in row["service"].get("records", [])]
        pools = [value for row in arm_rows if (value := _pool_evidence(row)) is not None]
        high_water = [value["occupied_page_high_water"] for value in pools
                      if _finite(value["occupied_page_high_water"])]
        high_water_bytes = [value["occupied_page_high_water_bytes"] for value in pools
                            if _finite(value["occupied_page_high_water_bytes"])]
        compacted = sorted({stats["private_local_copy_compacted"] for stats in admission
                            if isinstance(stats.get("private_local_copy_compacted"), bool)})
        copy_scope = "not applicable"
        if arm == "redknot_private":
            copy_scope = ("full canonical local-document copies, not visible-window compacted"
                          if compacted == [False] else "visible-window compacted" if compacted == [True]
                          else "unknown or mixed private-copy policy")
            if compacted == [False]:
                warnings.append("Private-copy control is not compacted to the visible local window; sharing speedup is relative to that control, not an optimal private-cache baseline.")
        offline_by_item = {}
        for row in arm_rows:
            offline_by_item.setdefault(row["item_key"], row.get("offline_seconds"))
        quality = dict(
            primary_f1=aggregate.get("f1"), primary_em=aggregate.get("em"),
            primary_scope="predeclared replica zero; repeated scores averaged within unique QA",
            primary_gate=aggregate.get("primary_observed_2percent_gate", False),
            all_prefix_f1=aggregate.get("all_prefix_robustness", {}).get("f1"),
            all_prefix_em=aggregate.get("all_prefix_robustness", {}).get("em"),
            all_prefix_gate=aggregate.get("all_prefix_robustness", {}).get("observed_2percent_gate", False),
            observed_2percent_gate=bool(aggregate.get("observed_2percent_gate") and verified and run_complete),
            acceptance_scope="observed complete sample only; not statistical certification",
            independent_qa_scored=aggregate.get("unique_scored_examples"),
            independent_qa_expected=len(items) if manifest is not None else None,
            primary_datasets=aggregate.get("datasets"),
            all_prefix_details=aggregate.get("all_prefix_robustness"),
        )
        peaks = [row.get("peak_allocated_bytes") for row in arm_rows if _finite(row.get("peak_allocated_bytes"))]
        arm_metrics[arm] = dict(
            completed_requests=aggregate["requests"], failed_requests=aggregate["failed_requests"],
            cancelled_requests=aggregate["cancelled_requests"], completed_output_tokens=aggregate["output_tokens"],
            measured_service_seconds=aggregate["measured_wall_seconds"], measured_qps=aggregate["aggregate_qps"],
            measured_qps_scope="all this arm's measured service rows; do not compare unpaired partial aggregates",
            qps_ratio_vs_native=aggregate.get("qps_vs_native") if verified else None,
            ratio_scope="common successful item/repeat cohorts only, not the ratio of unpaired aggregate QPS",
            paired_cohorts=aggregate.get("paired_cohort_count", 0),
            expected_cohorts=aggregate.get("expected_cohorts") if manifest is not None else None,
            complete_paired_comparison=bool(aggregate.get("paired_comparison_complete") and verified and run_complete),
            paired_native_qps=aggregate.get("paired_native_qps") if verified else None,
            paired_candidate_qps=aggregate.get("paired_qps") if verified else None,
            quality=quality,
            queued_ttft_seconds=distribution([record.get("ttft_seconds") for record in completed], eligible_count=len(completed)),
            request_service_ttft_seconds=distribution([record.get("service_ttft_seconds") for record in completed], eligible_count=len(completed)),
            ttft_scope="completed requests only across all measured rows; queue TTFT starts at scheduled arrival; service TTFT starts at successful admission, not model-only time",
            stages_seconds=dict(stages),
            admission_breakdown={field: _metric_sum(admission, field) for field in (
                "prefix_prefill_s", "online_recovery_s", "kv_publish_and_partition_s")},
            admission_breakdown_request_count=sum(bool(stats) for stats in admission),
            admission_breakdown_eligible_requests=len(admission),
            pool_high_water_pages_max=max(high_water) if high_water else None,
            pool_high_water_bytes_max=max(high_water_bytes) if high_water_bytes else None,
            pool_cohorts=pools, cuda_peak_allocated_bytes_max=max(peaks) if peaks else None,
            private_copy_policy=copy_scope,
            offline_setup_seconds_by_unique_item=offline_by_item,
            offline_setup_included_in_qps=False,
        )
    per_item = []
    item_lookup = {item["key"]: item for item in items}
    ordered_keys = list(item_lookup) + sorted(observed_keys - set(item_lookup))
    for key in ordered_keys:
        item = item_lookup.get(key, {})
        item_rows = [row for row in rows if row["item_key"] == key]
        per_item.append(dict(
            item_key=key, dataset=item.get("dataset", item_rows[0]["dataset"] if item_rows else None),
            source_row=item.get("source_row"), source_id=item.get("source_id"),
            context_tokens=item.get("context_tokens"),
            prefix_token_lengths=[len(prefix) for prefix in item.get("prefix_token_ids", [])],
            question=item.get("question"), answers=item.get("answers", item_rows[0].get("answers") if item_rows else None),
            reference_arm="recomputed_native",
            cohorts=[dict(arm=row["arm"], repeat=row.get("repeat", 0),
                          service_status=row["service"].get("status"),
                          measured_qps=row["service"].get("completed_qps"),
                          measured_seconds=row["service"].get("total_wall_seconds"),
                          full_outputs=[_full_output(record) for record in row["service"].get("records", [])],
                          errors=row["service"].get("errors", [])) for row in item_rows],
        ))
    return dict(
        input=str(path), input_sha256=sha256(raw_bytes), recorded_status=raw.get("status"),
        schema=raw["schema"], config=config, profile=raw.get("profile"), hardware=raw.get("hardware"),
        profile_diagnostics=raw.get("profile_diagnostics"),
        effective_workload_protocol=raw.get("effective_workload_protocol"),
        execution_scope="single-GPU experimental Hugging Face Qwen3 engines with a Python queued scheduler; NOT production SGLang/TP8 serving",
        workload=dict(path=str(manifest_path) if manifest is not None else None,
                      file_sha256=sha256(manifest_bytes) if manifest_bytes is not None else None,
                      recorded_canonical_sha256=raw.get("workload_sha256"), hash_verified=verified,
                      independent_qa_expected=len(items) if manifest is not None else None,
                      independent_qa_observed=len(observed_keys),
                      request_replicas=manifest.get("request_replicas") if manifest is not None else None,
                      instruction_at_query=manifest.get("instruction_at_query") if manifest is not None else None,
                      varying_prefix_lengths=manifest.get("varying_prefix_lengths") if manifest is not None else None,
                      protocol_source="frozen manifest takes precedence over CLI config; missing manifest fields remain unknown, not inferred from CLI or raw effective metadata",
                      source_provenance=manifest.get("sources") if manifest is not None else None),
        original_code_hashes=raw.get("source_hashes"), model_config_sha256=raw.get("model_config_sha256"),
        warnings=list(dict.fromkeys(warnings)), error=raw.get("error"),
        arms=arm_metrics, per_item_full_outputs=per_item,
    )


def build_report(inputs):
    analyzer, analyzer_hash = _analysis_module()
    runs, errors = [], []
    for path in inputs:
        try:
            runs.append(analyze_run(path, analyzer))
        except Exception as exc:
            errors.append(dict(input=str(Path(path).resolve()), type=type(exc).__name__, message=str(exc)))
    return dict(schema="redknot-nonprefix-evidence-report-v1",
                generated_at_utc=datetime.now(timezone.utc).isoformat(),
                report_code_sha256=sha256(Path(__file__).read_bytes()),
                summary_analysis_code_sha256=analyzer_hash,
                aggregate_across_runs=False,
                interpretation="different pilots/configurations remain separate; no extrapolation, answer extraction, or production-performance claim",
                runs=runs, input_errors=errors)


def print_summary(report, *, print_outputs=False):
    def number(value, digits=4):
        return f"{value:.{digits}f}" if _finite(value) else "NA"
    for run in report["runs"]:
        print(f"RUN {run['input']} status={run['recorded_status']} independent_QA={run['workload']['independent_qa_observed']} manifest_verified={run['workload']['hash_verified']}")
        print(run["execution_scope"])
        for arm, values in run["arms"].items():
            quality, ttft = values["quality"], values["queued_ttft_seconds"]
            print(f"  {arm}: completed={values['completed_requests']} QPS={number(values['measured_qps'])} paired_native_ratio={number(values['qps_ratio_vs_native'])} paired={values['paired_cohorts']}/{values['expected_cohorts']} complete_pair={values['complete_paired_comparison']}")
            print(f"    primary F1/EM={number(quality['primary_f1'])}/{number(quality['primary_em'])} all-prefix F1/EM={number(quality['all_prefix_f1'])}/{number(quality['all_prefix_em'])} observed_gate={quality['observed_2percent_gate']}")
            print(f"    queued TTFT p50/p95={number(ttft['p50'])}/{number(ttft['p95'])}s service TTFT p50/p95={number(values['request_service_ttft_seconds']['p50'])}/{number(values['request_service_ttft_seconds']['p95'])}s occupied_page_high_water={values['pool_high_water_pages_max']} (not fixed slab bytes)")
            print("    stages_seconds=" + json.dumps(values["stages_seconds"], sort_keys=True))
        for warning in run["warnings"]:
            print("  WARNING: " + warning)
        if print_outputs:
            for item in run["per_item_full_outputs"]:
                print(f"ITEM {item['item_key']} reference={item['reference_arm']} answers={json.dumps(item['answers'], ensure_ascii=False)}")
                for cohort in item["cohorts"]:
                    for output in cohort["full_outputs"]:
                        print(f"  {cohort['arm']} repeat={cohort['repeat']} {output.get('request_id')} status={output.get('status')}")
                        print(output.get("text", "[no output text recorded]"))
    for error in report["input_errors"]:
        print("INPUT ERROR: " + json.dumps(error, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-outputs", action="store_true")
    args = parser.parse_args(argv)
    protected = {path.resolve() for path in args.inputs}
    protected.update(path.resolve().parent / "frozen_workload.json" for path in args.inputs)
    if args.output.resolve() in protected:
        parser.error("output must not overwrite an input or frozen workload")
    report = build_report(args.inputs)
    # Serialize before creating the artifact, so invalid evidence cannot leave
    # half a JSON report. Exclusive creation never overwrites an older report.
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(encoded)
    print_summary(report, print_outputs=args.print_outputs)
    print("REPORT_JSON " + str(args.output.resolve()))
    return 1 if report["input_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
