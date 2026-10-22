#!/usr/bin/env python3
"""Audit fixed-cohort hot-service QPS and full-output quality vs recomputed.

No failed/partial/memory-unaccounted run receives a headline speedup. Row-count
savings are execution counts, not measured total FLOPs. The known regression
cohort never becomes an unseen quality certificate through parameter tuning.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics

from benchmark_nonprefix_affinity import _complete_row, _quality_comparison, _quality_means, summarize_rows
from benchmark_redknot_nonprefix import digest, scores
from benchmark_selective_recovery import BASELINE, cache_active, candidate_arm, prepare_selective_plan


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def audit_run(raw, workload, plan, source_root=None):
    errors = []
    try:
        rebuilt = prepare_selective_plan(workload, [group["item_key"] for group in plan["groups"]],
            plan["candidates"], chunks=plan["chunks"], replicas=len(plan["requests"]) // len(plan["groups"]),
            concurrency=plan["concurrency"], total_kv_gib=plan["pool_gib"], repeats=plan["repeats"])
        if rebuilt != plan:
            errors.append("frozen plan cannot be reconstructed from the complete workload/configuration")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append("invalid frozen plan: " + str(exc))
    if (raw.get("schema") != "redknot-selective-recovery-v1" or raw.get("status") != "complete"
            or raw.get("plan") != plan or raw.get("plan_sha256") != digest(plan)
            or raw.get("workload_sha256") != digest(workload)):
        errors.append("run incomplete or frozen identity mismatch")
    hardware = raw.get("hardware") or {}
    if (not _sha(raw.get("model_config_sha256")) or not isinstance(hardware.get("gpu"), str)
            or type(hardware.get("total_bytes")) is not int or hardware["total_bytes"] <= 0):
        errors.append("missing model/hardware identity")
    source_hashes = raw.get("source_hashes") or {}
    required = {"benchmark_selective_recovery.py", "benchmark_nonprefix_affinity.py",
        "benchmark_nonprefix_scale.py", "benchmark_redknot_nonprefix.py", "nonprefix_profile.py",
        "nonprefix_paged.py", "nonprefix_scheduler.py", "nonprefix_affinity.py", "selective_recovery.py",
        "manager.py", "pool.py"}
    if plan.get("hidden_residual_contract"):
        required.add("hidden_residual_recovery.py")
    names = [Path(path).name for path in source_hashes]
    if (not required <= set(names) or len(names) != len(set(names))
            or any(not Path(path).is_absolute() or not _sha(value) for path, value in source_hashes.items())):
        errors.append("missing/ambiguous/invalid source provenance")
    verified = []
    if source_root is not None:
        anchors = [Path(path) for path in source_hashes if Path(path).name == "benchmark_selective_recovery.py"]
        if len(anchors) != 1:
            errors.append("missing source-root anchor")
        else:
            root = Path(source_root).resolve()
            for path, sha in source_hashes.items():
                try:
                    relative = Path(path).relative_to(anchors[0].parents[3])
                    local = (root / relative).resolve()
                    if not local.is_relative_to(root) or digest(local.read_bytes()) != sha:
                        raise ValueError("source hash/path mismatch")
                    verified.append(str(relative))
                except (OSError, ValueError, IndexError) as exc:
                    errors.append(f"source {path}: {exc}")
    expected = {(arm, repeat) for arm in plan.get("arms", []) for repeat in range(plan.get("repeats", 0))}
    observed = [(row.get("arm"), row.get("repeat")) for row in raw.get("results", [])]
    if len(observed) != len(expected) or set(observed) != expected:
        errors.append("missing, duplicate or unexpected arm/repeat")
    specs = {candidate_arm(item): item for item in plan.get("candidates", [])}
    answers = {item["key"]: item["answers"] for item in workload["items"]}
    query_ids = {item["key"]: item["query_token_ids"] for item in workload["items"]}
    contexts = {group["item_key"]: group["context_tokens"] for group in plan["groups"]}
    hidden_geometry = None
    if plan.get("hidden_residual_contract"):
        receipt = raw.get("hidden_residual_cache_estimator") or {}
        cfg = receipt.get("model_config") or {}
        lengths = [length for group in plan["groups"] for length in group["chunk_lengths"]]
        if (digest(cfg) != raw.get("model_config_sha256")
                or any(type(cfg.get(field)) is not int or cfg[field] <= 0 for field in ("hidden_size", "num_hidden_layers"))
                or receipt.get("document_token_lengths") != lengths
                or receipt.get("group_size") != 128 or receipt.get("value_bytes") != 1 or receipt.get("scale_bytes") != 4):
            errors.append("invalid hidden-residual estimator/model geometry receipt")
        else:
            hidden_geometry = dict(tokens=sum(lengths), layers=cfg["num_hidden_layers"], hidden=cfg["hidden_size"])
    for row in raw.get("results", []):
        arm = row.get("arm")
        prefix = f"{arm}/{row.get('repeat')}: "
        if not _complete_row(row, plan):
            errors.append(prefix + "incomplete execution/cleanup")
            continue
        candidate = None if arm == BASELINE else specs.get(arm)
        if row.get("candidate") != candidate:
            errors.append(prefix + "candidate configuration mismatch")
        if row.get("cohort_role") != plan.get("cohort_role"):
            errors.append(prefix + "cohort role mismatch")
        profile = row.get("effective_profile") or {}
        policies = profile.get("policies") or []
        try:
            if (not policies or any(not line for line in policies)
                    or len({len(line) for line in policies}) != 1
                    or profile.get("sha256") != digest(policies)):
                raise ValueError("invalid policy dimensions/hash")
            local = sum(policy["head_type"] == "local" for line in policies for policy in line)
            ratio = local / sum(map(len, policies))
            if profile.get("effective_local_ratio") != ratio:
                raise ValueError("effective head ratio mismatch")
            if arm == BASELINE and any(policy["head_type"] != "global" for line in policies for policy in line):
                raise ValueError("recomputed baseline is not all-global")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(prefix + str(exc))
            continue
        budget = row.get("memory_accounting") or {}
        fields = ("total_kv_budget_bytes", "canonical_cache_reservation_bytes", "pool_capacity_pages",
                  "pool_reserved_bytes", "accounted_reserved_bytes", "rounding_unused_bytes", "page_bytes")
        if (any(type(budget.get(key)) is not int or budget[key] < 0 for key in fields)
                or budget.get("page_bytes", 0) <= 0):
            errors.append(prefix + "missing memory accounting")
            continue
        if (budget["total_kv_budget_bytes"] != plan["total_kv_budget_bytes"]
                or budget["pool_reserved_bytes"] != budget["pool_capacity_pages"] * budget["page_bytes"]
                or budget["accounted_reserved_bytes"] != budget["pool_reserved_bytes"] + budget["canonical_cache_reservation_bytes"]
                or budget["accounted_reserved_bytes"] + budget["rounding_unused_bytes"] != budget["total_kv_budget_bytes"]
                or not 0 <= budget["rounding_unused_bytes"] < budget["page_bytes"]):
            errors.append(prefix + "unequal or invalid total KV budget")
        service = row["service"]
        if service.get("admission_policy") != "fcfs":
            errors.append(prefix + "scheduler policy changed")
        active = cache_active(candidate, len(policies))
        query_active = active and candidate.get("selection_mode") == "query_attention"
        hidden_active = active and candidate.get("recovery_mode") == "hidden_residual_int8"
        wanted_config = None if not active else dict(
            block_size=candidate["block_size"], row_fraction=candidate["row_fraction"],
            probe_layers=candidate["probe_layers"], max_offline_kv_bytes=budget["canonical_cache_reservation_bytes"])
        if query_active:
            wanted_config["selection_mode"] = "query_attention"
        if hidden_active:
            wanted_config["recovery_mode"] = "hidden_residual_int8"
            if hidden_geometry is not None:
                expected_cache = (hidden_geometry["tokens"] * (hidden_geometry["layers"]-candidate["probe_layers"])
                    * (hidden_geometry["hidden"]+math.ceil(hidden_geometry["hidden"]/128)*4))
                if len(policies) != hidden_geometry["layers"] or budget["canonical_cache_reservation_bytes"] != expected_cache:
                    errors.append(prefix + "hidden cache reservation differs from exact quantized payload estimator")
        for stage in ("engine_stats_before", "engine_stats_after"):
            stats = service.get(stage) or {}
            pool = stats.get("pool") or {}
            cache_key = "offline_hidden_cache" if hidden_active else "offline_kv_cache"
            actual = stats.get(cache_key + "_bytes")
            if (type(actual) is not int or actual != budget["canonical_cache_reservation_bytes"]
                    or pool.get("reserved_slab_bytes") != budget["pool_reserved_bytes"]
                    or pool.get("capacity_pages") != budget["pool_capacity_pages"]
                    or stats.get("local_head_ratio") != ratio or stats.get("backend") != "flash"
                    or stats.get("share_local") is not True):
                errors.append(prefix + f"{stage}: actual storage/policy does not match budget/profile")
            if (stats.get("selective_recovery") != wanted_config
                    or stats.get(cache_key + "_limit_bytes") != budget["canonical_cache_reservation_bytes"]):
                errors.append(prefix + "recovery configuration/cache limit changed")
            other_key = "offline_kv_cache" if hidden_active else "offline_hidden_cache"
            if any(type(stats.get(other_key+suffix, 0)) is not int or stats.get(other_key+suffix, 0) != 0
                   for suffix in ("_bytes", "_limit_bytes", "_peak_bytes")):
                errors.append(prefix + "unaccounted second offline recovery representation")
            if hidden_active:
                peak = stats.get("offline_hidden_cache_peak_bytes")
                if (type(peak) is not int or peak != budget["canonical_cache_reservation_bytes"]
                        or stats.get("offline_hidden_quantization") != dict(dtype="int8", group_size=128, scale_dtype="float32")):
                    errors.append(prefix + "hidden cache peak/quantization mismatch")
        if not active and budget["canonical_cache_reservation_bytes"] != 0:
            errors.append(prefix + "full-row fallback must not retain hidden canonical full KV")
        for record in service["records"]:
            text = record.get("text")
            if (not isinstance(text, str) or record.get("accuracy") != scores(text, answers[record["item_key"]])
                    or record.get("output_text_sha256") != digest(text.encode())
                    or record.get("output_ids_sha256") != digest(record["output_ids"])):
                errors.append(prefix + "full emitted output/hash/score mismatch")
            admission = record.get("admission_stats") or {}
            mode = ("experimental_hidden_residual_int8_v4" if hidden_active else
                    "experimental_query_selective_v3" if query_active else
                    "experimental_block_selective_v1" if active else "full_rows")
            if admission.get("recovery_mode") != mode:
                errors.append(prefix + "actual recovery mode differs from declared candidate")
            if query_active:
                ids = query_ids[record["item_key"]]
                p, layers = candidate["probe_layers"], len(policies)
                expected_query_rows = dict(
                    query_projected_rows_per_layer=[len(ids)] * p + [0] * (layers-p),
                    query_ffn_rows_per_layer=[len(ids)] * (p-1) + [0] * (layers-p+1))
                if hidden_active:
                    expected_query_rows["query_attention_rows_per_layer"] = [len(ids)] * (p-1) + [0] * (layers-p+1)
                if (admission.get("selection_mode") != "query_attention"
                        or type(admission.get("recovery_query_tokens")) is not int
                        or admission["recovery_query_tokens"] != len(ids)
                        or admission.get("recovery_query_sha256") != digest(ids)):
                    errors.append(prefix + "query selector identity/count differs from exact frozen query")
                for field, expected_counts in expected_query_rows.items():
                    counts = admission.get(field)
                    if (not isinstance(counts, list) or any(type(count) is not int for count in counts)
                            or counts != expected_counts):
                        errors.append(prefix + "missing or inconsistent actual " + field)
            else:
                unexpected_rows = False
                for field in ("query_projected_rows_per_layer", "query_ffn_rows_per_layer", "query_attention_rows_per_layer"):
                    if field in admission:
                        counts = admission[field]
                        unexpected_rows |= (not isinstance(counts, list) or len(counts) != len(policies)
                            or any(type(value) is not int or value != 0 for value in counts))
                if (admission.get("selection_mode", "kv_drift") != "kv_drift"
                        or admission.get("recovery_query_tokens", 0) != 0
                        or admission.get("recovery_query_sha256") is not None
                        or unexpected_rows):
                    errors.append(prefix + "unexpected query-aware work in baseline/drift/full-row path")
            if hidden_active and (admission.get("projected_rows_per_layer", "missing") is not None
                    or admission.get("row_execution_fraction", "missing") is not None):
                errors.append(prefix + "hidden residual must not claim unified QKV row execution")
            fields = (("q_rows_per_layer", "kv_rows_per_layer", "attention_rows_per_layer", "ffn_rows_per_layer")
                      if hidden_active else ("projected_rows_per_layer", "ffn_rows_per_layer"))
            for field in fields:
                counts = admission.get(field)
                if (not isinstance(counts, list) or len(counts) != len(policies)
                        or any(type(count) is not int or count < 0 for count in counts)):
                    errors.append(prefix + "missing actual " + field)
                elif not active and counts != [contexts[record["item_key"]]] * len(policies):
                    errors.append(prefix + "full-row path did not execute all document rows")
                elif active:
                    selected = admission.get("selected_token_rows")
                    full = contexts[record["item_key"]]
                    expected_rows = ([full] * len(policies) if hidden_active and field == "kv_rows_per_layer" else
                        [full] * candidate["probe_layers"] + [selected] * (len(policies)-candidate["probe_layers"]))
                    if (type(selected) is not int or not 0 < selected <= full
                            or counts != expected_rows):
                        errors.append(prefix + "actual selected/probe rows are inconsistent")
    return sorted(set(errors)), verified


def _dataset_scores(quality, plan):
    grouped = {}
    for group in plan["groups"]:
        grouped.setdefault(group["dataset"], []).append(quality["groups"][group["item_key"]])
    return {name: {scope: {metric: statistics.mean(item[scope][metric] for item in items)
                          for metric in ("f1", "em")} for scope in ("primary_replica0", "all_prefix")}
            for name, items in grouped.items()}


def execution_vectors(admission, layers, *, split_projections=False, query=False):
    """Normalize verified legacy counters only when a mixed/new plan needs it.

    Native and offline-KV paths project Q/K/V on identical rows. The hidden
    path does not: K/V stay full-row. Never alias its absent legacy counter.
    """
    zeros = [0] * layers
    if not split_projections:
        names = (("query_projected_rows_per_layer", "query_ffn_rows_per_layer") if query else
                 ("projected_rows_per_layer", "ffn_rows_per_layer"))
        return {name: admission.get(name, zeros) for name in names}
    if query:
        projected = admission.get("query_projected_rows_per_layer", zeros)
        ffn = admission.get("query_ffn_rows_per_layer", zeros)
        return dict(query_q_rows_per_layer=projected, query_kv_rows_per_layer=projected,
                    query_attention_rows_per_layer=admission.get("query_attention_rows_per_layer", ffn),
                    query_ffn_rows_per_layer=ffn)
    if admission.get("recovery_mode") == "experimental_hidden_residual_int8_v4":
        return {name: admission[name] for name in
                ("q_rows_per_layer", "kv_rows_per_layer", "attention_rows_per_layer", "ffn_rows_per_layer")}
    projected = admission["projected_rows_per_layer"]
    return dict(q_rows_per_layer=projected, kv_rows_per_layer=projected,
                attention_rows_per_layer=projected, ffn_rows_per_layer=admission["ffn_rows_per_layer"])


def build_report(raw, workload, plan, *, source_root=None):
    errors, verified = audit_run(raw, workload, plan, source_root)
    report = dict(schema="redknot-selective-recovery-report-v1", integrity_verified=not errors,
        errors=errors, cohort_role=plan.get("cohort_role"), quality_certified=False,
        runtime=raw.get("runtime"), cohort_selection_provenance=plan.get("cohort_selection_provenance"),
        source_file_bytes_verified=source_root is not None and not errors,
        verified_source_files=verified, independent_qa_count=plan.get("independent_primary_qa_count"),
        timing=plan.get("scope"), memory_protocol=plan.get("memory_protocol"), comparisons=[],
        interpretation="Compare each predeclared candidate with fresh same-backend recomputation. All-prefix aggregate is the quality gate; primary0/per-item/per-dataset are diagnostic. Row savings are not total-FLOP savings.")
    if errors:
        return report
    report["arm_summary"] = summarize_rows(raw["results"], plan)
    by_arm = {arm: sorted([row for row in raw["results"] if row["arm"] == arm], key=lambda row: row["repeat"])
              for arm in plan["arms"]}
    records = {arm: [{record["request_id"]: record for record in row["service"]["records"]} for row in rows]
               for arm, rows in by_arm.items()}
    qualities = {arm: _quality_means(values, plan) for arm, values in records.items()}
    base_summary = report["arm_summary"][BASELINE]
    split_projections = bool(plan.get("hidden_residual_contract"))
    document_fields = (("q_rows_per_layer", "kv_rows_per_layer", "attention_rows_per_layer", "ffn_rows_per_layer")
                       if split_projections else ("projected_rows_per_layer", "ffn_rows_per_layer"))
    for arm in plan["arms"]:
        rows = by_arm[arm]
        layers = len(rows[0]["effective_profile"]["policies"])
        counters = {}
        for field in document_fields:
            vectors = [execution_vectors(record["admission_stats"], layers, split_projections=split_projections)[field]
                       for row in rows for record in row["service"]["records"]]
            counters[field] = dict(total=sum(map(sum, vectors)), per_layer_total=[sum(values) for values in zip(*vectors)])
        report["arm_summary"][arm].update(executed_document_rows=counters,
            preparation_seconds=[dict(repeat=row["repeat"], engine=row.get("engine_construction_seconds"),
                                      offline_documents=row.get("offline_document_seconds"), excluded_from_hot_qps=True) for row in rows],
            memory_accounting_by_repeat=[row["memory_accounting"] for row in rows],
            actual_peak_allocated_bytes=max(row.get("peak_allocated_bytes", 0) for row in rows),
            actual_peak_reserved_bytes=max(row.get("peak_reserved_bytes", 0) for row in rows))
        if split_projections:
            report["arm_summary"][arm]["offline_recovery_storage_by_repeat"] = [dict(
                repeat=row["repeat"],
                full_kv_bytes=row["service"]["engine_stats_before"].get("offline_kv_cache_bytes", 0),
                hidden_int8_and_scale_bytes=row["service"]["engine_stats_before"].get("offline_hidden_cache_bytes", 0),
                total_retained_bytes=row["memory_accounting"]["canonical_cache_reservation_bytes"],
                pool_slab_bytes=row["memory_accounting"]["pool_reserved_bytes"],
                hidden_quantization=row["service"]["engine_stats_before"].get("offline_hidden_quantization")) for row in rows]
        if plan.get("query_selection_contract"):
            query_counters = {}
            query_fields = (("query_q_rows_per_layer", "query_kv_rows_per_layer", "query_attention_rows_per_layer", "query_ffn_rows_per_layer")
                            if split_projections else ("query_projected_rows_per_layer", "query_ffn_rows_per_layer"))
            for field in query_fields:
                vectors = [execution_vectors(record["admission_stats"], layers, split_projections=split_projections, query=True)[field]
                           for row in rows for record in row["service"]["records"]]
                query_counters[field] = dict(total=sum(map(sum, vectors)),
                    per_layer_total=[sum(values) for values in zip(*vectors)])
            report["arm_summary"][arm]["additional_query_probe_rows"] = query_counters
    for arm in plan["arms"][1:]:
        candidate_summary = report["arm_summary"][arm]
        bq, cq = qualities[BASELINE], qualities[arm]
        all_prefix = _quality_comparison(bq["all_prefix"], cq["all_prefix"])
        primary = _quality_comparison(bq["primary_replica0"], cq["primary_replica0"])
        pairs = [(records[BASELINE][repeat][identity], records[arm][repeat][identity])
                 for repeat in range(plan["repeats"]) for identity in records[BASELINE][repeat]]
        datasets = {key: {scope: _quality_comparison(values[scope], _dataset_scores(cq, plan)[key][scope])
                         for scope in ("primary_replica0", "all_prefix")}
                    for key, values in _dataset_scores(bq, plan).items()}
        savings = {}
        for field in document_fields:
            old = base_summary["executed_document_rows"][field]["total"]
            new = candidate_summary["executed_document_rows"][field]["total"]
            savings[field] = dict(recomputed_rows=old, candidate_rows=new, fraction_saved=1-new/old if old else None)
        ratio = candidate_summary["headline_qps"] / base_summary["headline_qps"]
        report["comparisons"].append(dict(baseline=BASELINE, candidate=arm,
            candidate_config=by_arm[arm][0]["candidate"], qps_ratio=ratio,
            recomputed_qps=base_summary["headline_qps"], candidate_qps=candidate_summary["headline_qps"],
            requests_compared=len(pairs), token_identical=sum(a["output_ids"] == b["output_ids"] for a, b in pairs),
            text_identical=sum(a["text"] == b["text"] for a, b in pairs),
            all_prefix=all_prefix, primary_replica0=primary,
            per_dataset=datasets,
            per_item={key: {scope: _quality_comparison(bq["groups"][key][scope], cq["groups"][key][scope])
                           for scope in ("primary_replica0", "all_prefix")} for key in bq["groups"]},
            observed_all_prefix_2percent_gate=all_prefix["metric_2percent_gate"],
            measured_qps_uplift_and_quality_gate=ratio > 1 and all_prefix["metric_2percent_gate"],
            statistical_significance_claimed=False, quality_certified=False, executed_row_savings=savings))
        if plan.get("query_selection_contract"):
            report["comparisons"][-1]["additional_query_probe_rows"] = candidate_summary["additional_query_probe_rows"]
            combined = {}
            query_pairs = ([(field, "query_"+field) for field in document_fields] if split_projections else [
                    ("projected_rows_per_layer", "query_projected_rows_per_layer"),
                    ("ffn_rows_per_layer", "query_ffn_rows_per_layer")])
            for document_field, query_field in query_pairs:
                old = (base_summary["executed_document_rows"][document_field]["total"]
                       + base_summary["additional_query_probe_rows"][query_field]["total"])
                new = (candidate_summary["executed_document_rows"][document_field]["total"]
                       + candidate_summary["additional_query_probe_rows"][query_field]["total"])
                combined[document_field] = dict(recomputed_rows=old, candidate_rows=new,
                    fraction_saved=1-new/old if old else None)
            report["comparisons"][-1]["document_plus_extra_query_recovery_row_savings"] = combined
            report["comparisons"][-1]["row_savings_scope"] = (
                "executed_row_savings counts document rows only; additional query probe rows shown separately and in combined recovery rows; counts exclude ordinary query/decode/prefix work and are not total FLOPs; measured QPS/TTFT already include all probe work")
            if split_projections:
                report["comparisons"][-1]["row_savings_scope"] += (
                    "; Q, KV projection, attention and FFN rows are separate operation classes; hidden-residual KV projections execute every document row, so combined KV work may exceed recomputation due to the query probe; no unified QKV saving is reported")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    paths = [args.input / name for name in ("results.json", "frozen_workload.json", "frozen_selective_plan.json")]
    report = build_report(*[json.loads(path.read_text()) for path in paths], source_root=args.source_root)
    report["input_files_sha256"] = {str(path.resolve()): digest(path.read_bytes()) for path in paths}
    with args.output.open("x") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["integrity_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
