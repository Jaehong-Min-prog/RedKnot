#!/usr/bin/env python3
"""Read-only evidence report for gpu*/results.json non-prefix scale runs.

Usage: python report_nonprefix_scale.py --input-dir SCALE_RUN [--output REPORT.json]
No model, CUDA, tokenizer, network, scheduler, or GPU-holder operation is used.
Cases and GPUs are never pooled into a throughput total. Original outputs are
preserved verbatim; a preflight skip is not an observed OOM or a speedup.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


ARMS = ("recomputed_native", "recomputed_paged", "redknot_private", "redknot_shared")
BASELINES = ("recomputed_native", "recomputed_paged", "redknot_private")
TERMINAL_CASES = {"complete", "complete_with_blocked_or_failed_arms"}


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    values = sorted(float(x) for x in values if finite(x) and x >= 0)
    def quantile(q):
        if not values:
            return None
        position = (len(values)-1)*q
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi]-values[lo])*(position-lo)
    return dict(count=len(values), p50=quantile(.5), p95=quantile(.95),
                mean=statistics.mean(values) if values else None,
                method="linear empirical quantiles; not confidence intervals")


def read_json(path, errors):
    try:
        data = path.read_bytes()
        return json.loads(data), dict(path=str(path.resolve()), file_sha256=digest(data))
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return None, dict(path=str(path.resolve()), file_sha256=None)


def mean(values):
    return statistics.mean(values) if values else None


def score_valid(value):
    return isinstance(value, dict) and all(finite(value.get(k)) and 0 <= value[k] <= 1 for k in ("f1", "em"))


def metric_drops(baseline, candidate):
    result = {}
    for metric in ("f1", "em"):
        a, b = baseline.get(metric), candidate.get(metric)
        drop = 100*(a-b) if finite(a) and finite(b) else None
        result[metric+"_drop_percentage_points"] = drop
        result[metric+"_relative_drop_percent"] = drop/a if drop is not None and a > 0 else None
    result["metric_2percent_gate"] = all(finite(value) and value <= 2+1e-10 for value in result.values())
    result["zero_baseline_policy"] = "relative drop is unknown, not passed, when baseline is zero"
    return result


def validate_case(case, item):
    """Reproduce only deterministic token partition/hash checks, never inference."""
    errors = []
    try:
        flat = [token for part in item["document_token_ids"] for token in part]
        k, n = case["chunks"], case["requests"]
        if type(k) is not int or not 0 < k <= len(flat) or type(n) is not int or n < 1:
            raise ValueError("invalid chunk/request counts")
        width, remainder = divmod(len(flat), k)
        lengths = [width+(index < remainder) for index in range(k)]
        pieces, cursor = [], 0
        for length in lengths:
            pieces.append(flat[cursor:cursor+length])
            cursor += length
        prefixes = item["prefix_token_ids"][:n]
        expected = [digest(prefix+flat+item["query_token_ids"]) for prefix in prefixes]
        checks = {
            "item key": case["item_key"] == item["key"],
            "dataset": case["dataset"] == item["dataset"],
            "context length": case["context_tokens"] == len(flat) == item["context_tokens"],
            "flat document hash": case["flat_document_sha256"] == digest(flat),
            "chunk hash": case["chunk_tokens_sha256"] == digest(pieces),
            "chunk lengths": case["chunk_lengths"] == lengths,
            "prefix count": len(prefixes) == n,
            "distinct prefixes": len({tuple(prefix) for prefix in prefixes}) == n,
            "prefix lengths": case["prefix_tokens"] == list(map(len, prefixes)),
            "query length": case["query_tokens"] == len(item["query_token_ids"]),
            "frozen prompt hashes": case["exact_prompt_sha256"] == expected == item["exact_prompt_sha256"][:n],
            "original document hash": item["document_tokens_sha256"] == digest(item["document_token_ids"]),
        }
        errors.extend(name+" mismatch" for name, good in checks.items() if not good)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid case/manifest: {exc}")
    return errors


def row_issues(row, case):
    issues = []
    service = row.get("service") or {}
    records = service.get("records", [])
    if row.get("status") != "complete" or service.get("status") != "complete":
        issues.append("row/service not complete")
    if row.get("fatal") or row.get("cleanup_status") != "complete" or row.get("cleanup_error") or row.get("persistence_error"):
        issues.append("fatal, incomplete cleanup, or persistence failure")
    if service.get("errors") or service.get("failed_requests", 0) or service.get("cancelled_requests", 0):
        issues.append("service errors/failures/cancellations")
    if row.get("item_key") != case["item_key"] or row.get("dataset") != case["dataset"]:
        issues.append("row input identity mismatch")
    if row.get("requested_concurrency") != case["concurrency"] or service.get("max_active") != case["concurrency"]:
        issues.append("requested concurrency mismatch")
    if not finite(service.get("total_wall_seconds")) or service["total_wall_seconds"] <= 0:
        issues.append("invalid service wall time")
    if len(records) != case["requests"] or service.get("completed_requests") != case["requests"]:
        issues.append("missing/incomplete request set")
    seen, token_count = set(), 0
    for record in records:
        index = record.get("frozen_prefix_index")
        if type(index) is not int or not 0 <= index < case["requests"] or index in seen:
            issues.append("duplicate/invalid frozen prefix index")
            continue
        seen.add(index)
        if record.get("status") != "completed":
            issues.append(f"prefix {index} not completed")
        if record.get("exact_prompt_sha256") != case["exact_prompt_sha256"][index]:
            issues.append(f"prefix {index} prompt hash mismatch")
        if record.get("max_new_tokens") != 32:
            issues.append(f"prefix {index} output budget mismatch")
        ids = record.get("output_ids")
        if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
            issues.append(f"prefix {index} missing/invalid generated IDs")
        else:
            token_count += len(ids)
    if service.get("completed_output_tokens") != token_count:
        issues.append("completed output token count mismatch")
    return list(dict.fromkeys(issues))


def summarize_arm(arm, rows, case, *, repeats, planned, integrity, case_terminal):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("repeat")].append(row)
    expected = set(range(repeats))
    diagnostics, primary, robust, stages = [], [], [], defaultdict(float)
    completed, failed, cancelled, output_tokens, wall = 0, 0, 0, 0, 0.0
    ttft, service_ttft, peaks, batches, blocks = [], [], [], [], []
    for row in rows:
        service = row.get("service") or {}
        records = service.get("records", [])
        valid_scores = [r for r in records if r.get("status") == "completed" and score_valid(r.get("accuracy"))]
        zero = [r for r in valid_scores if r.get("frozen_prefix_index") == 0]
        if len(zero) == 1:
            primary.append(zero[0]["accuracy"])
        if (len(valid_scores) == case["requests"]
                and {r.get("frozen_prefix_index") for r in valid_scores} == set(range(case["requests"]))):
            robust.append({key: mean([r["accuracy"][key] for r in valid_scores]) for key in ("f1", "em")})
        for record in records:
            status = record.get("status")
            completed += status == "completed"
            failed += status == "failed"
            cancelled += status == "cancelled"
            if status == "completed":
                output_tokens += len(record.get("output_ids", []))
                ttft.append(record.get("ttft_seconds"))
                service_ttft.append(record.get("service_ttft_seconds"))
            if finite(record.get("capacity_blocks")):
                blocks.append(record["capacity_blocks"])
        seconds = service.get("total_wall_seconds")
        if finite(seconds) and seconds > 0:
            wall += seconds
        for key, value in service.get("stages_seconds", {}).items():
            if finite(value):
                stages[key] += value
        if finite(service.get("peak_active_requests")):
            peaks.append(service["peak_active_requests"])
        if finite(service.get("max_forward_batch")):
            batches.append(service["max_forward_batch"])
        diagnostics.append(dict(repeat=row.get("repeat"), recorded_status=row.get("status"),
            issues=row_issues(row, case), reason=row.get("reason"), error=row.get("error"),
            fatal=row.get("fatal"), cleanup_status=row.get("cleanup_status"), cleanup_error=row.get("cleanup_error"),
            capacity_preflight=row.get("capacity_preflight"), service_errors=service.get("errors"),
            requested_requests=service.get("requested_requests"), reported_completed_requests=service.get("completed_requests"),
            reported_failed_requests=service.get("failed_requests"), reported_cancelled_requests=service.get("cancelled_requests"),
            reported_output_tokens=service.get("completed_output_tokens"),
            offline_document_seconds=row.get("offline_document_seconds"),
            engine_construction_seconds=row.get("engine_construction_seconds"),
            runner_cleanup_seconds=row.get("cleanup_seconds"),
            actual_max_active=service.get("peak_active_requests"), actual_max_forward=service.get("max_forward_batch"),
            engine_stats_after=service.get("engine_stats_after"),
            peak_allocated_bytes=row.get("peak_allocated_bytes"),
            # Deliberately preserve every raw record, including failed outputs.
            full_outputs=records))
    complete = bool(planned and integrity and case_terminal and set(grouped) == expected
                    and all(len(values) == 1 and not row_issues(values[0], case) for values in grouped.values()))
    statuses = [row.get("status") for row in rows]
    if not planned:
        status = "not_planned"
    elif not rows:
        status = "not_measured_missing"
    elif all(value == "preflight_blocked" for value in statuses):
        status = "not_measured_preflight_blocked"
    elif complete:
        status = "complete"
    else:
        status = "incomplete_failed_or_unverified"
    def quality(values):
        return dict(f1=mean([score["f1"] for score in values]), em=mean([score["em"] for score in values]),
                    scored_repeats=len(values), expected_repeats=repeats)
    return dict(arm=arm, status=status, complete_paired_eligible=complete,
        observed_rows=len(rows), expected_repeats=repeats, missing_repeats=sorted(expected-set(grouped)),
        duplicate_repeats=[key for key, values in grouped.items() if len(values) > 1],
        completed_requests=completed, failed_requests=failed, cancelled_requests=cancelled,
        observed_output_tokens=output_tokens, measured_service_seconds=wall if wall else None,
        measured_qps=completed/wall if wall else None,
        measured_qps_scope="completed requests / all observed service wall time for this case/arm; warmup excluded; partial rows are diagnostics only",
        actual_max_active=max(peaks) if peaks else None, actual_max_forward_batch=max(batches) if batches else None,
        capacity_block_events=sum(blocks) if blocks else (0 if rows and any(r.get("service") for r in rows) else None),
        queued_ttft_seconds=distribution(ttft), request_service_ttft_seconds=distribution(service_ttft),
        ttft_scope="completed requests only; queued TTFT starts at scheduled arrival; service TTFT starts at admission, not model-only",
        stages_seconds=dict(stages), primary_replica0=quality(primary), all_prefix=quality(robust),
        independent_primary_qa_count=1, quality_scope="one primary QA; variants/requests/repeats are not independent QA; raw recorded F1/EM, no answer extraction",
        measurements=diagnostics)


def comparisons(arms):
    result = []
    for baseline in BASELINES:
        for candidate in ARMS:
            if candidate == baseline:
                continue
            a, b = arms[baseline], arms[candidate]
            paired = a["complete_paired_eligible"] and b["complete_paired_eligible"]
            primary = metric_drops(a["primary_replica0"], b["primary_replica0"])
            robust = metric_drops(a["all_prefix"], b["all_prefix"])
            scored = all(values[metric]["scored_repeats"] == values["expected_repeats"]
                         for values in (a, b) for metric in ("primary_replica0", "all_prefix"))
            result.append(dict(baseline_arm=baseline, candidate_arm=candidate,
                complete_paired_comparison=paired,
                qps_speedup=b["measured_qps"]/a["measured_qps"] if paired else None,
                baseline_qps=a["measured_qps"] if paired else None,
                candidate_qps=b["measured_qps"] if paired else None,
                reason=None if paired else f"not fully paired: baseline={a['status']}, candidate={b['status']}",
                primary_replica0_drops=primary, all_prefix_drops=robust,
                observed_2percent_gate=bool(paired and scored and primary["metric_2percent_gate"] and robust["metric_2percent_gate"]),
                quality_acceptance_scope="this single QA and chunk configuration only; not generalization or production certification"))
    return result


def analyze_job(directory):
    errors = []
    raw, raw_source = read_json(directory/"results.json", errors)
    manifest, manifest_source = read_json(directory/"frozen_workload.json", errors)
    plan, plan_source = read_json(directory/"frozen_scale_plan.json", errors)
    raw, manifest, plan = raw or {}, manifest or {}, plan or {}
    if raw.get("schema") != "redknot-nonprefix-scale-v1":
        errors.append("unsupported/missing results schema")
    if manifest.get("schema") != "redknot-nonprefix-frozen-v1":
        errors.append("unsupported/missing workload schema")
    if plan.get("schema") != "redknot-nonprefix-scale-plan-v1":
        errors.append("unsupported/missing plan schema")
    workload_verified = bool(manifest) and raw.get("workload_sha256") == digest(manifest) == plan.get("workload_sha256")
    plan_verified = bool(plan) and raw.get("plan_sha256") == digest(plan.get("cases"))
    if not workload_verified:
        errors.append("workload digest mismatch/missing")
    if not plan_verified:
        errors.append("case-plan digest mismatch/missing")
    config = raw.get("config", {})
    if plan.get("arms") != config.get("arms") or plan.get("repeats") != config.get("repeats"):
        errors.append("plan arms/repeats mismatch with recorded config")
    if (plan.get("output_tokens") != 32 or manifest.get("max_new_tokens") != 32
            or plan.get("natural_eos") is not True or plan.get("global_prefix_layers") != 60
            or plan.get("ffn") != "full native MLP"):
        errors.append("frozen scale protocol metadata mismatch")
    repeats = plan.get("repeats", 1)
    if type(repeats) is not int or repeats < 1:
        errors.append("invalid expected repeat count")
        repeats = 1
    items = {item["key"]:item for item in manifest.get("items", [])}
    if len(items) != len(manifest.get("items", [])):
        errors.append("duplicate frozen item keys")
    planned = plan.get("cases", [])
    if len({case["case_id"] for case in planned}) != len(planned):
        errors.append("duplicate planned case IDs")
    observed = defaultdict(list)
    for entry in raw.get("cases", []):
        observed[entry.get("case", {}).get("case_id")].append(entry)
    orphaned = [entry for key, entries in observed.items() if key not in {case["case_id"] for case in planned} for entry in entries]
    if orphaned:
        errors.append("unplanned observed cases retained as orphaned evidence")
    cases = []
    for case in planned:
        entries = observed.get(case["case_id"], [])
        item = items.get(case["item_key"], {})
        issues = validate_case(case, item)
        if len(entries) != 1:
            issues.append(f"expected one observed case, found {len(entries)}")
        if any(entry.get("case") != case for entry in entries):
            issues.append("observed case differs from frozen plan")
        rows = [row for entry in entries for row in entry.get("results", [])]
        unknown = [row for row in rows if row.get("arm") not in ARMS]
        if unknown:
            issues.append("unknown arm rows present")
        status = entries[0].get("status") if len(entries) == 1 else "missing_or_duplicate"
        integrity = not errors and not issues
        arms = {arm:summarize_arm(arm, [row for row in rows if row.get("arm") == arm], case,
                    repeats=repeats, planned=arm in plan.get("arms", []), integrity=integrity,
                    case_terminal=status in TERMINAL_CASES) for arm in ARMS}
        augmented = item.get("augmentation") is not None
        cases.append(dict(case_key=f"{directory.name}/{case['case_id']}", case=case,
            recorded_case_status=status, evidence_verified=integrity, integrity_errors=issues,
            primary_qa_key=item.get("primary_qa_key", item.get("key")), independent_primary_qa_count=1,
            input_kind="augmented_rag" if augmented else "full_original_context" if item.get("complete_primary_context") is True else "unknown",
            input_scope="augmented RAG is not official LongBench length scoring; every chunk configuration has independently assessed outputs",
            source_provenance=item.get("provenance"), augmentation=item.get("augmentation"),
            question=item.get("question"), answers=item.get("answers"), arms=arms,
            paired_comparisons=comparisons(arms), unrecognized_arm_rows=unknown))
    return dict(job=directory.name, recorded_status=raw.get("status"), integrity_errors=errors,
        workload_hash_verified=workload_verified, case_plan_hash_verified=plan_verified,
        provenance=dict(results=raw_source, frozen_workload=manifest_source, frozen_scale_plan=plan_source,
                        recorded_workload_sha256=raw.get("workload_sha256"), recorded_plan_sha256=raw.get("plan_sha256"),
                        model_config_sha256=raw.get("model_config_sha256"), source_hashes=raw.get("source_hashes"),
                        dataset_sources=manifest.get("sources")),
        config=config, hardware=raw.get("hardware"), profile=raw.get("profile"),
        profile_diagnostics=raw.get("profile_diagnostics"), cases=cases, orphaned_cases=orphaned)


def build_report(input_dir):
    directory = Path(input_dir).resolve()
    jobs, errors = [], []
    paths = sorted(path for path in directory.glob("gpu*") if path.is_dir())
    if not paths:
        errors.append(dict(input=str(directory), error="no gpu* job directories found"))
    for path in paths:
        try:
            jobs.append(analyze_job(path))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(dict(input=str(path), error=f"{type(exc).__name__}: {exc}"))
    cases = [case for job in jobs for case in job["cases"]]
    primary = {case["primary_qa_key"] for case in cases if case["primary_qa_key"] is not None}
    return dict(schema="redknot-nonprefix-scale-report-v1", generated_at_utc=datetime.now(timezone.utc).isoformat(),
        input_directory=str(directory), report_code_sha256=digest(Path(__file__).read_bytes()),
        independent_primary_qa_count=len(primary), primary_qa_keys=sorted(primary), planned_cases=len(cases),
        throughput_aggregation_across_cases_or_gpus=False,
        timing_scope="single-GPU hot prepared-document burst; timed queue/admission/recovery/query/decode/sampling/request-release; offline construction, model load and final engine teardown excluded",
        scope="mechanism/load scan, not independent-QA generalization, production serving, saturated/SLO-qualified QPS or FLOP savings",
        comparison_scope="strict same-case fully successful repeats only; native, all-global paged, and same-algorithm private-copy baselines remain separately named",
        preflight_scope="preflight_blocked means not measured under a conservative estimate, not an observed OOM or a capacity win",
        warnings=["Natural EOS: 32 is the maximum output budget, not a fixed generated length.",
                  "Different chunk boundaries can change approximate offline local KV and must retain separate quality results.",
                  "Primary replica zero and all-prefix means are both reported; replicas and length variants do not increase independent QA N.",
                  "Private-copy control retains full canonical local-document copies, not a visible-window-compacted optimum.",
                  "Discovered local gpu* directories determine coverage; missing undownloaded directories cannot be inferred."],
        jobs=jobs, input_errors=errors)


def print_summary(report):
    def number(value):
        return f"{value:.4f}" if finite(value) else "NA"
    print(f"INDEPENDENT_PRIMARY_QA={report['independent_primary_qa_count']} CASES={report['planned_cases']} NO_CROSS_GPU_QPS_SUM")
    print(report["scope"])
    for job in report["jobs"]:
        print(f"JOB {job['job']} status={job['recorded_status']} manifest_verified={job['workload_hash_verified']} plan_verified={job['case_plan_hash_verified']}")
        for error in job["integrity_errors"]:
            print("  INTEGRITY_ERROR "+error)
        for case in job["cases"]:
            c = case["case"]
            print(f" CASE {case['case_key']} {c['item_key']} tokens={c['context_tokens']} chunks={c['chunks']} requested_B={c['concurrency']} requests={c['requests']} kind={case['input_kind']} verified={case['evidence_verified']}")
            for arm, a in case["arms"].items():
                p, q, t = a["primary_replica0"], a["all_prefix"], a["queued_ttft_seconds"]
                print(f"  {arm}: {a['status']} QPS={number(a['measured_qps'])} active/forward={a['actual_max_active']}/{a['actual_max_forward_batch']} blocks={a['capacity_block_events']} completed={a['completed_requests']} output_tokens={a['observed_output_tokens']} TTFT_p50/p95={number(t['p50'])}/{number(t['p95'])} primary_F1/EM={number(p['f1'])}/{number(p['em'])} allprefix_F1/EM={number(q['f1'])}/{number(q['em'])}")
            for comparison in case["paired_comparisons"]:
                if comparison["candidate_arm"] == "redknot_shared":
                    print(f"  shared/{comparison['baseline_arm']}={number(comparison['qps_speedup'])} complete_pair={comparison['complete_paired_comparison']} observed_gate={comparison['observed_2percent_gate']} reason={comparison['reason']}")
    for error in report["input_errors"]:
        print("INPUT_ERROR "+json.dumps(error, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.input_dir/"report_nonprefix_scale.json"
    if output.exists():
        parser.error("output already exists; evidence/report overwrite is refused")
    report = build_report(args.input_dir)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(encoded)
    print_summary(report)
    print("REPORT_JSON "+str(output.resolve()))
    return 1 if report["input_errors"] or any(job["integrity_errors"] or any(case["integrity_errors"] for case in job["cases"]) for job in report["jobs"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
