#!/usr/bin/env python3
"""Render read-only, qualified reports from real-model SegPaged QPS raw JSON.

Only derivative REPORT.md, REPORT.provenance.json and optionally a PNG are
written. Optional native-FA2 control inputs remain separate from the paged
storage ablation. Native-logit gate failures, skipped workloads and partial
runs remain visible. Metrics are recomputed from measured waves, never inferred
from memory capacity or a selected fastest repeat. Matplotlib is optional.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


SCHEMA = "segpaged-real-model-qps-storage-ablation-v1"
NATIVE_SCHEMA = "segpaged-native-fa2-model-reference-v1"


def paged_backend(raw, cohort=None):
    """Use recorded backend identity, including old gate-only raw records."""
    candidates = []
    if cohort is not None:
        candidates += [cohort.get("paged_backend"), cohort.get("independent_native_model_gate", {}).get("paged_backend")]
    candidates += [raw.get("paged_backend"), raw.get("arguments", {}).get("paged_backend")]
    candidates += [c.get("independent_native_model_gate", {}).get("paged_backend") for c in raw.get("cohorts", [])]
    candidates += [raw.get("failed_independent_model_gate", {}).get("paged_backend")]
    candidates += [g.get("paged_backend") for g in raw.get("failed_independent_model_gates", [])]
    for value in candidates:
        if value in ("triton", "direct_triton_head_paged"):
            return "triton"
        if value in ("flash", "flash_attention_paged", "flash_paged"):
            return "flash"
    return "unrecorded"


def backend_label(raw, cohort=None):
    value = paged_backend(raw, cohort)
    return {"triton": "Experimental Triton head-paged",
            "flash": "Experimental FlashAttention paged",
            "unrecorded": "Paged backend not recorded"}[value]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_number(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0))


def number(value, places=3):
    return f"{value:,.{places}f}" if finite_number(value) else "N/A"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |"]
                      + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def length_label(length):
    if finite_number(length) and length > 0:
        return f"{length / 1024:g}K ({int(length):,} tokens)"
    return "unknown length"


def qualification(cohort):
    # Old smoke files do not have numerical_validation_status/qualification.
    # The native gate itself takes precedence over every presentation label.
    gate = cohort.get("independent_native_model_gate", {})
    if gate.get("passed") is False:
        return "DIAGNOSTIC — native-logit gate FAILED"
    if gate.get("passed") is True:
        return "Numerical gate passed; task accuracy NOT evaluated"
    return "UNVERIFIED — independent native gate missing"


def summarize_cohort(cohort):
    """Recompute summaries and accounting; keep malformed samples visible."""
    grouped = defaultdict(list)
    warnings = []
    measured_count = 0
    measured_requests = 0
    measured_tokens = 0
    for index, row in enumerate(cohort.get("samples", [])):
        if row.get("warmup", False):
            warnings.append(f"samples[{index}] marked warmup; excluded from measured totals")
            continue
        measured_count += 1
        requests = row.get("completed_requests")
        tokens = row.get("completed_output_tokens")
        if finite_number(requests, positive=True):
            measured_requests += int(requests)
        if finite_number(tokens, positive=True):
            measured_tokens += int(tokens)
        ratio, batch = row.get("shared_ratio"), row.get("batch_size")
        elapsed = row.get("admission_inclusive_seconds")
        if not (finite_number(ratio) and 0 <= ratio <= 1
                and isinstance(batch, int) and not isinstance(batch, bool) and batch > 0
                and requests == batch and finite_number(elapsed, positive=True)):
            warnings.append(f"samples[{index}] malformed ratio/batch/completion/time; omitted from metrics")
            continue
        computed = requests / elapsed
        if finite_number(row.get("qps")) and not math.isclose(row["qps"], computed, rel_tol=1e-6, abs_tol=1e-9):
            warnings.append(f"samples[{index}] stored QPS disagrees with completions/time; using recomputed QPS")
        grouped[(float(ratio), batch)].append(row)
    groups = {}
    for key, rows in sorted(grouped.items()):
        qps = [r["completed_requests"] / r["admission_inclusive_seconds"] for r in rows]
        elapsed = sum(r["admission_inclusive_seconds"] for r in rows)
        count = sum(r["completed_requests"] for r in rows)
        resident = [r.get("resident_generation_seconds") for r in rows]
        valid_resident = all(finite_number(x, positive=True) for x in resident)
        summary = dict(ratio=key[0], batch=key[1], repeats=len(rows),
                       completed_requests=count, qps_median=statistics.median(qps),
                       aggregate_qps=count / elapsed, qps_min=min(qps), qps_max=max(qps),
                       aggregate_resident_qps=count / sum(resident) if valid_resident else None,
                       rows=rows, max_batch_search_censored=any(r.get("max_batch_search_censored", False) for r in rows),
                       budget_max_batch=rows[0].get("budget_max_batch"),
                       cache_stats=rows[0].get("cache_stats", {}))
        for name in ("admission_seconds", "admission_inclusive_ttft_seconds", "resident_ttft_seconds"):
            values = [r.get(name) for r in rows]
            summary[name + "_median"] = statistics.median(values) if all(finite_number(v) for v in values) else None
        token_lists = [r.get("generated_token_ids") for r in rows]
        summary["repeat_token_identity"] = (all(t == token_lists[0] for t in token_lists)
                                            if all(t is not None for t in token_lists) else None)
        groups[key] = summary
    invalid = []
    for summary in cohort.get("summary", []):
        if summary.get("same_batch_baseline_token_identity") is False:
            invalid.append(f"recorded summary ratio={summary.get('shared_ratio')}, B={summary.get('batch_size')}: same-batch token identity failed")
    for (ratio, batch), group in groups.items():
        if any(row.get("all_generated_logits_finite") is False for row in group["rows"]):
            invalid.append(f"ratio={ratio:g}, B={batch}: nonfinite generated logits")
        baseline = groups.get((0.0, batch))
        if baseline:
            reference = baseline["rows"][0].get("generated_token_ids")
            outputs = [r.get("generated_token_ids") for r in group["rows"]]
            group["same_batch_token_identity"] = (all(t == reference for t in outputs)
                                                  if reference is not None and all(t is not None for t in outputs) else None)
            if group["same_batch_token_identity"] is False:
                invalid.append(f"ratio={ratio:g}, B={batch}: generated tokens differ from same-B private baseline")
    return dict(groups=groups, warnings=warnings, invalid=invalid,
                measured_waves=measured_count, completed_requests=measured_requests,
                completed_output_tokens=measured_tokens,
                warmup_waves=len(cohort.get("warmup_records", [])))


def maximal_groups(audit):
    result = {}
    for (ratio, batch), group in audit["groups"].items():
        if ratio not in result or batch > result[ratio]["batch"]:
            result[ratio] = group
    return result


def ratio_value(numerator, denominator):
    return numerator / denominator if finite_number(numerator) and finite_number(denominator, positive=True) else None


def ratio_text(numerator, denominator):
    value = ratio_value(numerator, denominator)
    return f"{value:.3f}×" if value is not None else "N/A"


def cohort_rows(cohort, audit):
    maxima = maximal_groups(audit)
    baseline = maxima.get(0.0, {})
    same_b4_baseline = audit["groups"].get((0.0, 4), {})
    rows = []
    for ratio, group in sorted(maxima.items()):
        b4 = audit["groups"].get((ratio, 4), {})
        batch = str(group["batch"])
        if group["max_batch_search_censored"] and group["batch"] == group["budget_max_batch"]:
            batch += " (search cap)"
        elif group["budget_max_batch"] != group["batch"]:
            batch += " (partial)"
        rows.append([f"{ratio * 100:g}%", batch, group["repeats"], group["completed_requests"],
                     number(group["qps_median"]), number(group["aggregate_qps"]),
                     number(group["aggregate_resident_qps"]),
                     ratio_text(group["aggregate_qps"], baseline.get("aggregate_qps")),
                     ratio_text(b4.get("aggregate_qps"), same_b4_baseline.get("aggregate_qps")),
                     number(group["admission_seconds_median"]),
                     number(group["admission_inclusive_ttft_seconds_median"]),
                     qualification(cohort)])
    return rows


def planned_coverage(raw):
    args = raw.get("arguments", {})
    planned = list(itertools.product(args.get("datasets", []), args.get("lengths", []), args.get("output_tokens", [])))
    seen = {}
    for cohort in raw.get("cohorts", []):
        outputs = [cohort["output_tokens"]] if "output_tokens" in cohort else cohort.get("planned_output_tokens", [])
        for output in outputs:
            seen[(cohort.get("dataset"), cohort.get("document_tokens"), output)] = cohort.get("status", "unknown")
    return [(dataset, length, output, seen.get((dataset, length, output), "NOT REACHED / no cohort record"))
            for dataset, length, output in planned]


def gate_rows(gate):
    native, paged = gate.get("native_token_ids"), gate.get("paged_token_ids")
    if isinstance(native, list) and isinstance(paged, list) and len(native) == len(paged) and native:
        agreement = f"{sum(a == b for a, b in zip(native, paged))}/{len(native)} tokens"
    else:
        agreement = "N/A"
    return [["Original gate passed", gate.get("passed", "not recorded")],
            ["Original acceptance rule", gate.get("acceptance", "not recorded")],
            ["First-logits max absolute error", number(gate.get("first_logits_max_abs_error"), 6)],
            ["First-logits cosine", number(gate.get("first_logits_cosine"), 9)],
            ["Independent native/paged greedy agreement", agreement + " (not task accuracy)"],
            ["Source length unchanged", str(gate.get("source_cache_length_unchanged", "not recorded"))
             + "; this checks length, not a content checksum"]]


def summarize_native(cohort):
    # Reuse timing/count validation without pretending that native FA2 has a
    # shared-ratio parameter or a bounded paged allocation.
    converted = dict(cohort, samples=[dict(row, shared_ratio=0.0) for row in cohort.get("samples", [])], summary=[])
    audit = summarize_cohort(converted)
    if any(row.get("source_cache_length_unchanged") is False for row in cohort.get("samples", [])
           if not row.get("warmup", False)):
        audit["invalid"].append("native source-cache length changed")
    return audit


def native_input_match(native_raw, native_cohort, paged_raw, paged_cohort, batch):
    """Strict comparability gate; full input hashes differ with cohort size."""
    checks = {}
    for field in ("model_config_sha256", "torch_version", "transformers_version"):
        left, right = native_raw.get(field), paged_raw.get(field)
        checks[field] = left is not None and right is not None and left == right
    for field in ("source_row_sha256", "document_tokens", "question_tokens", "output_tokens"):
        left, right = native_cohort.get(field), paged_cohort.get(field)
        checks[field] = left is not None and right is not None and left == right
    native_model = native_raw.get("model_path", native_raw.get("arguments", {}).get("model_path"))
    paged_model = paged_raw.get("model_path", paged_raw.get("arguments", {}).get("model_path"))
    checks["model_path"] = native_model is not None and paged_model is not None and native_model == paged_model
    left, right = native_cohort.get("question_token_ids"), paged_cohort.get("question_token_ids")
    checks["first_native_batch_question_token_ids"] = (
        isinstance(left, list) and isinstance(right, list) and len(left) >= batch and len(right) >= batch
        and all(isinstance(row, list) and len(row) == native_cohort.get("question_tokens") for row in left[:batch])
        and left[:batch] == right[:batch])
    return all(checks.values()), checks


def render_native_reference(native_inputs, paged_inputs):
    totals = dict(native_measured_waves=0, native_completed_requests=0,
                  native_completed_output_tokens=0, native_warmup_waves=0)
    if not native_inputs:
        return [], totals
    lines = ["## Independent native FA2 reference — a different comparison", "",
             "These inputs use unmodified Transformers Qwen3 + FA2 and private DynamicCache copies. "
             "They are **not bounded by the experimental page-pool budget**. Admission and growing-cache "
             "allocation costs are included, but the allocator/layout differ. Same-B comparisons below "
             "are a native execution reference; the paged 75%-sharing budget-max row changes concurrency "
             "and is not an equal-budget FA2 comparison. Native FA2 here is not a production serving scheduler.", "",
             "Cross-run timing ratios are computed only when the model configuration, model path, Torch/Transformers "
             "versions, source-row hash, document/question/output lengths, and the native batch's question-token "
             "sequences match. Full input_sha256 may legitimately differ because the paged workload records "
             "more questions. Incomplete, invalid or unmatched comparisons are rejected, not extrapolated. "
             "A matched timing pair does not override a failed native numerical gate: those ratios are explicitly "
             "diagnostic timing comparisons, not correctness-qualified native speedups. Triton and FlashAttention "
             "paged backends are labeled separately and are never pooled.", ""]
    comparative_rows = []
    for index, item in enumerate(native_inputs, 1):
        raw = item["raw"]
        lines += [f"### Native reference {index}: {item['path'].name}", "",
                  f"Status: **{raw.get('status', 'unknown')}**. Raw input: `{item['path']}`  ",
                  f"SHA-256: `{item['sha256']}`", "",
                  "Configuration:", "", "```json", json.dumps(raw.get("arguments", {}), indent=2, ensure_ascii=False), "```", "",
                  table(["Environment", "Recorded value"], [
                      ["GPU driver name", raw.get("gpu_name", "not recorded")],
                      ["Torch / Transformers", f"{raw.get('torch_version', '?')} / {raw.get('transformers_version', '?')}"],
                      ["Model-config SHA-256", raw.get("model_config_sha256", "not recorded")]]), ""]
        coverage = planned_coverage(raw)
        if coverage:
            lines += [table(["Source dataset", "Document length", "Output tokens", "Recorded status"],
                            [[d, length_label(n), out, status] for d, n, out, status in coverage]), ""]
        if raw.get("error"):
            lines += ["Native run error:", "", "```text", str(raw["error"]), "```", ""]
        for cohort in raw.get("cohorts", []):
            audit = summarize_native(cohort)
            for key in ("measured_waves", "completed_requests", "completed_output_tokens", "warmup_waves"):
                totals["native_" + key] += audit[key]
            label = f"{cohort.get('dataset', '?')} / {length_label(cohort.get('document_tokens'))} / output {cohort.get('output_tokens', '?')}"
            lines += [f"#### Native cohort: {label}", "",
                      f"Status: **{cohort.get('status', 'unknown')}**. Measured {audit['measured_waves']} waves / "
                      f"{audit['completed_requests']} requests / {audit['completed_output_tokens']} output tokens; "
                      f"{audit['warmup_waves']} warmup waves excluded.", ""]
            for warning in audit["invalid"] + audit["warnings"]:
                lines += [f"- {warning}"]
            if audit["invalid"] or audit["warnings"]:
                lines.append("")
            if not audit["groups"]:
                lines += ["No measured native QPS; no native comparison is computed.", ""]
                continue
            for (_, batch), native in sorted(audit["groups"].items()):
                lines += [table(["Native B", "Repeats", "Main median QPS", "Main aggregate QPS", "Resident aggregate QPS", "Admission s", "Main TTFT s"], [[
                    batch, native["repeats"], number(native["qps_median"]), number(native["aggregate_qps"]),
                    number(native["aggregate_resident_qps"]), number(native["admission_seconds_median"]),
                    number(native["admission_inclusive_ttft_seconds_median"])]]), ""]
                candidates = [(paged_item, candidate) for paged_item in paged_inputs
                              for candidate in paged_item["raw"].get("cohorts", [])
                              if (candidate.get("dataset"), candidate.get("document_tokens"), candidate.get("output_tokens"))
                              == (cohort.get("dataset"), cohort.get("document_tokens"), cohort.get("output_tokens"))]
                if not candidates:
                    lines += ["**Input match: false.** No paged cohort with the same dataset/length/output; comparison rejected.", ""]
                for paged_item, candidate in candidates:
                    matched, checks = native_input_match(raw, cohort, paged_item["raw"], candidate, batch)
                    paged_audit = summarize_cohort(candidate)
                    comparable = (matched and cohort.get("status") == "complete" and candidate.get("status") == "complete"
                                  and not audit["invalid"] and not paged_audit["invalid"]
                                  and not audit["warnings"] and not paged_audit["warnings"])
                    native_gate_passed = candidate.get("independent_native_model_gate", {}).get("passed") is True
                    numerical_gate_qualified = comparable and native_gate_passed
                    failed = [key for key, passed in checks.items() if not passed]
                    lines += [f"Paged counterpart: `{paged_item['path'].name}` — **{backend_label(paged_item['raw'], candidate)}**. "
                              f"**Input match: {str(matched).lower()}.** "
                              f"Timing comparable: **{str(comparable).lower()}**. "
                              f"Native-numerical-gate-qualified comparison: **{str(numerical_gate_qualified).lower()}**. "
                              + ("Mismatched/missing fields: " + ", ".join(failed) + "." if failed else "All required identity fields matched."), ""]
                    if not comparable:
                        lines += ["Comparison rejected: unmatched identity, incomplete measurement, or invalid/warning-bearing samples. No speedup is computed.", ""]
                        continue
                    if not numerical_gate_qualified:
                        lines += ["**DIAGNOSTIC TIMING ONLY: the paged native-logit gate failed or is missing. "
                                  "The ratios below do not establish a correctness-qualified speedup over native FA2.**", ""]
                    maxima = maximal_groups(paged_audit)
                    selections = [("Paged private, same B", paged_audit["groups"].get((0.0, batch))),
                                  ("Paged 75% sharing, same B", paged_audit["groups"].get((.75, batch))),
                                  ("Paged 75% sharing, paged-budget max B", maxima.get(.75))]
                    rows = []
                    for strategy, group in selections:
                        if group is None:
                            rows.append([strategy, "not measured", "N/A", "N/A", "N/A", "N/A", "N/A", qualification(candidate)])
                            continue
                        speedup = ratio_value(group["aggregate_qps"], native["aggregate_qps"])
                        comparative_rows.append(dict(label=label, strategy=strategy, main_speedup=speedup,
                                                     backend=paged_backend(paged_item["raw"], candidate),
                                                     native_numerical_gate_qualified=numerical_gate_qualified))
                        rows.append([strategy, group["batch"], number(group["aggregate_qps"]), number(group["aggregate_resident_qps"]),
                                     ratio_text(group["aggregate_qps"], native["aggregate_qps"]),
                                     ratio_text(group["aggregate_resident_qps"], native["aggregate_resident_qps"]),
                                     number(group["admission_inclusive_ttft_seconds_median"]), qualification(candidate)])
                    lines += [table(["Paged strategy (backend named above)", "B", "Main aggregate QPS", "Resident aggregate QPS", "Main timing ratio / native", "Resident timing ratio / native", "Main TTFT s", "Paged qualification"], rows), "",
                              "Ratios below 1 mean the paged path is slower than this native FA2 reference. "
                              "Different-B rows show an observed throughput comparison only, not an equal-memory-budget result.", ""]
            peak = [r.get("gpu_peak_allocated_bytes") for g in audit["groups"].values() for r in g["rows"]]
            peak = [x for x in peak if finite_number(x)]
            source = cohort.get("common_gpu_source_cache_bytes")
            private = cohort.get("nominal_final_private_kv_bytes")
            lines += [table(["Native memory; no fixed pool is enforced", "GiB"], [
                ["Common GPU HF source", number(source / 2 ** 30 if finite_number(source) else None, 4)],
                ["Nominal final private KV", number(private / 2 ** 30 if finite_number(private) else None, 4)],
                ["Source + nominal final private KV", number((source + private) / 2 ** 30 if finite_number(source) and finite_number(private) else None, 4)],
                ["Largest observed CUDA allocated peak", number(max(peak) / 2 ** 30 if peak else None, 4)]]), "",
                f"Source-row SHA-256: `{cohort.get('source_row_sha256', 'not recorded')}`  ",
                f"Native input SHA-256: `{cohort.get('input_sha256', 'not recorded')}`", ""]
        lines += ["Native implementation hashes:", "", table(["Source", "SHA-256"], sorted(raw.get("source_hashes", {}).items())), ""]
    # Place the takeaway before the detailed native sections, so an intra-kernel
    # gain cannot conceal a slower native comparison further down the report.
    slower = sum(row["main_speedup"] < 1 for row in comparative_rows)
    if comparative_rows:
        diagnostic = sum(not row["native_numerical_gate_qualified"] for row in comparative_rows)
        verdict = (f"Of {len(comparative_rows)} input-matched paged/native timing rows, {slower} have paged main QPS below native FA2; "
                   f"{diagnostic} rows are diagnostic only because the paged native numerical gate did not pass. "
                   "An improvement over private copies in the paged ablation must not be presented as a native-serving speedup.")
    else:
        verdict = "No eligible paged/native timing comparison is available. Native-serving speedup is unestablished."
    count_text = (f"Native reference counts are separate: **{totals['native_measured_waves']} measured waves, "
                  f"{totals['native_completed_requests']} completed requests, {totals['native_completed_output_tokens']} "
                  f"output tokens**; {totals['native_warmup_waves']} warmup waves excluded.")
    lines[2:2] = [f"**{verdict}**", "", count_text, ""]
    return lines, totals


def render_report(inputs, plot_name=None, plot_note=None):
    native_inputs = [item for item in inputs if item["raw"].get("schema") == NATIVE_SCHEMA]
    inputs = [item for item in inputs if item["raw"].get("schema") == SCHEMA]
    lines = ["# SegPaged: real-model QPS storage-ablation report", "",
             "This report measures completed requests from a real pretrained Qwen3 model. "
             "It does **not** establish production-serving QPS, non-prefix RedKnot/MLA reuse, "
             "or task accuracy. Native-logit gate failures remain failures; diagnostic timings do not override them.", "",
             "## Measurement and comparison", "",
             "Within each run, the baseline and sharing arms use the **same selected experimental paged-attention decoder** "
             "(Triton head-paged or FlashAttention paged, explicitly labeled). Different backends/runs are never pooled. "
             "The baseline privately copies all document KV heads; sharing aliases a selected physical KV-head subset. "
             "It is not a fully recomputed/native-FA2 serving baseline. The 100% arm is a full-prefix-sharing "
             "storage control using that same experimental kernel, not the performance of an optimized native prefix cache.", "",
             "Main QPS = measured completed requests / synchronized admission-inclusive wall time. "
             "That interval includes fresh admission/copies, question prefill, fixed-token greedy generation and cleanup; "
             "document prefill is offline and excluded. Resident-generation QPS excludes admission and cleanup. "
             "Rows are synchronous batched request waves, not continuous batching or an arrival-rate/SLO study. "
             "Output EOS is ignored to keep work fixed; continuation after EOS is not a quality example.", "",
             "At each sharing ratio, the main table reports the largest **actually measured** batch within the "
             "same online page budget. A search-cap label means capacity above that batch is unknown. "
             "The same-B4 column separates equal-concurrency behavior from capacity-enabled throughput. "
             "Both speedup columns use aggregate QPS, not the fastest repeat. TTFT is measured per request wave; "
             "larger batches can increase latency even when QPS rises. No tail-latency guarantee is inferred.", "",
             "The immutable GPU HF source cache is retained and charged in every arm, in addition to the "
             "identical online pool. This is an explicitly conservative source+pool configuration, not a "
             "minimum-memory production design. Active pages, reserved pool bytes and total CUDA allocation "
             "are distinct. Head-sharing capacity is never substituted for measured QPS.", ""]
    totals = dict(measured_waves=0, completed_requests=0, completed_output_tokens=0, warmup_waves=0)
    statuses = []
    for item in inputs:
        raw = item["raw"]
        audits = [summarize_cohort(c) for c in raw.get("cohorts", [])]
        item["audits"] = audits
        for audit in audits:
            for key in totals:
                totals[key] += audit[key]
        statuses.append([item["path"].name, backend_label(raw), raw.get("status", "unknown"), len(audits),
                         sum(c.get("status") == "complete" for c in raw.get("cohorts", [])),
                         sum(c.get("status", "").startswith("skipped") for c in raw.get("cohorts", [])),
                         sum(a["completed_requests"] for a in audits)])
    lines += ["## Run coverage", "", table(["Raw file", "Paged backend", "Run status", "Cohort records", "Complete", "Skipped", "Measured completed requests"], statuses), "",
              f"Across the paged storage-ablation files: **{totals['measured_waves']} measured waves, {totals['completed_requests']} "
              f"completed requests, {totals['completed_output_tokens']} generated output tokens**. "
              f"The {totals['warmup_waves']} warmup waves are excluded. These totals count repeated experiments "
              "and storage arms; they are not unique dataset examples. Counts in invalid or partial cohorts "
              "remain visible and are not quality-qualified completions.", ""]
    if plot_name:
        lines += ["## Measured throughput overview", "", f"![Measured aggregate QPS]({plot_name})", "",
                  "Bars show actual aggregate admission-inclusive QPS at each arm's largest measured batch. "
                  "Asterisks mark diagnostic cohorts whose original native-logit gate failed. "
                  "Missing, skipped, incomplete and invalid cohorts are not drawn or extrapolated. "
                  "Backend names identify each measured group. Colors identify private copies (blue), 75% head sharing "
                  "(orange), and the 100% sharing storage control (gray). Native FA2 control results are not plotted as "
                  "equal-pool-budget bars; see the separate native timing section.", ""]
    if plot_note:
        lines += [f"Plot note: {plot_note}", ""]
    native_lines, native_totals = render_native_reference(native_inputs, inputs)
    lines.extend(native_lines)
    totals.update(native_totals)
    for run_index, item in enumerate(inputs, 1):
        if item["raw"].get("schema") != SCHEMA:
            continue
        raw, path = item["raw"], item["path"]
        lines += [f"## Run {run_index}: {path.name}", "", f"Raw input: `{path}`  ",
                  f"SHA-256: `{item['sha256']}`  ", f"Status: **{raw.get('status', 'unknown')}**", "",
                  table(["Environment", "Recorded value"], [
                      ["Model", raw.get("model_path", "not recorded")],
                      ["Paged execution backend", backend_label(raw)],
                      ["GPU driver name", raw.get("gpu_name", "not recorded")],
                      ["GPU total GiB", number(raw.get("gpu_total_memory_bytes", 0) / 2 ** 30)],
                      ["Torch / Transformers", f"{raw.get('torch_version', '?')} / {raw.get('transformers_version', '?')}"],
                      ["Model-config SHA-256", raw.get("model_config_sha256", "not recorded")],
                      ["Model dimensions", json.dumps(raw.get("model_dimensions", {}), sort_keys=True)]]), "",
                  "### Experiment configuration", "", "```json", json.dumps(raw.get("arguments", {}), indent=2, ensure_ascii=False), "```", ""]
        coverage = planned_coverage(raw)
        if coverage:
            lines += ["### Planned coverage, including omissions", "",
                      table(["Source dataset", "Document length", "Output tokens", "Recorded status"],
                            [[d, length_label(n), out, status] for d, n, out, status in coverage]), ""]
        if raw.get("error"):
            lines += ["### Run error", "", "```text", str(raw["error"]), "```", ""]
        failed_gates = list(raw.get("failed_independent_model_gates", []))
        if raw.get("failed_independent_model_gate"):
            failed_gates.append(raw["failed_independent_model_gate"])
        for gate_index, gate in enumerate(failed_gates, 1):
            lines += [f"### Recorded failed native gate {gate_index}: {gate.get('dataset', '?')} / {length_label(gate.get('length'))}", "",
                      table(["Gate field", "Raw evidence"], gate_rows(gate)), ""]
        for cohort_index, (cohort, audit) in enumerate(zip(raw.get("cohorts", []), item["audits"]), 1):
            label = f"{cohort.get('dataset', '?')} / {length_label(cohort.get('document_tokens'))} / output {cohort.get('output_tokens', cohort.get('planned_output_tokens', '?'))}"
            lines += [f"### Cohort {cohort_index}: {label}", "", f"Backend: **{backend_label(raw, cohort)}**. "
                      f"Status: **{cohort.get('status', 'unknown')}**. "
                      f"Qualification: **{qualification(cohort)}**.", "",
                      f"Source row {cohort.get('source_row', '?')}; {cohort.get('question_tokens', '?')} question tokens; "
                      f"context repetition={cohort.get('repeated_context', 'not recorded')}; "
                      f"original context={cohort.get('original_context_tokens', '?')} tokens. "
                      "Contexts/questions may be truncated or repeated for a fixed shape. These are source-derived "
                      "systems workloads, not a LongBench score or a generalization result.", "",
                      f"Measured: {audit['measured_waves']} waves / {audit['completed_requests']} completed requests / "
                      f"{audit['completed_output_tokens']} output tokens; warmup: {audit['warmup_waves']} waves excluded.", ""]
            if audit["invalid"]:
                lines += ["**INVALID correctness evidence — do not interpret these speedups as qualified model throughput.**", ""]
            for warning in audit["invalid"] + audit["warnings"]:
                lines += [f"- {warning}"]
            if audit["invalid"] or audit["warnings"]:
                lines.append("")
            rows = cohort_rows(cohort, audit)
            if rows:
                lines += [table(["Shared KV heads", "Largest measured B", "Repeats", "Completions", "Median QPS", "Aggregate QPS", "Resident QPS", "vs private budget-max", "Same-B4 speedup", "Admission s", "Main TTFT s", "Qualification"], rows), ""]
                same_rows = []
                for (ratio, batch), group in sorted(audit["groups"].items()):
                    base = audit["groups"].get((0.0, batch), {})
                    same_rows.append([f"{ratio * 100:g}%", batch, group["repeats"], number(group["aggregate_qps"]),
                                      ratio_text(group["aggregate_qps"], base.get("aggregate_qps")),
                                      group.get("same_batch_token_identity", "no same-B reference"),
                                      group["repeat_token_identity"]])
                lines += ["All measured configurations (no best-repeat selection):", "",
                          table(["Shared heads", "B", "Repeats", "Aggregate QPS", "Same-B speedup", "Same-B generated-token identity", "Repeat token identity"], same_rows), ""]
            else:
                lines += ["**No measured QPS available for this cohort. No timing is imputed.**", ""]
            source_bytes = cohort.get("common_gpu_source_cache_bytes")
            pool_bytes = cohort.get("online_pool_reserved_bytes")
            total_bytes = cohort.get("total_source_plus_pool_reserved_kv_bytes")
            peaks = [row.get("gpu_peak_allocated_bytes") for group in audit["groups"].values() for row in group["rows"]]
            peaks = [x for x in peaks if finite_number(x)]
            memory_rows = []
            for name, value in [("Common immutable GPU HF source", source_bytes),
                                ("Fixed online page-pool reservation", pool_bytes),
                                ("Source + reserved pool", total_bytes),
                                ("Largest observed CUDA allocated peak (includes model and other live tensors)", max(peaks) if peaks else None)]:
                memory_rows.append([name, number(value / 2 ** 30 if finite_number(value) else None, 4)])
            lines += [table(["Memory accounting", "GiB"], memory_rows), ""]
            if isinstance(cohort.get("independent_native_model_gate"), dict):
                lines += [table(["Independent numerical check", "Recorded evidence"], gate_rows(cohort["independent_native_model_gate"])), ""]
            lines += [f"Input-token SHA-256: `{cohort.get('input_sha256', 'not recorded')}`  ",
                      f"Source-file SHA-256: `{cohort.get('source_file_sha256', 'not recorded')}`  ",
                      f"Source-row SHA-256: `{cohort.get('source_row_sha256', 'not recorded')}`", ""]
        lines += ["### Recorded implementation hashes", "",
                  table(["Source path at experiment time", "SHA-256"], sorted(raw.get("source_hashes", {}).items())), ""]
    lines += ["## Interpretation limits", "",
              "The native FA2 numerical gate is a correctness probe, separate from both the paged private-copy "
              "baseline and any optional, explicitly labeled native timing reference. "
              "A short matching greedy continuation or passing logits tolerance does not establish task accuracy. "
              "This benchmark does not validate non-prefix RoPE relocation, approximate local/global head semantics, "
              "MLA compatibility, multi-GPU serving, continuous batching, production scheduler integration, "
              "or long-run QPS under latency constraints. No F1/EM or percentage accuracy claim is derived here.", "",
              "Source and report hashes are recorded in REPORT.provenance.json. The renderer does not modify "
              "raw results, correct failed gates after the fact, select favorable examples, or fill missing points.", ""]
    return "\n".join(lines), totals


def make_plot(inputs, path):
    inputs = [item for item in inputs if item["raw"].get("schema") == SCHEMA]
    eligible = []
    for run_index, item in enumerate(inputs, 1):
        for cohort in item["raw"].get("cohorts", []):
            audit = summarize_cohort(cohort)
            if cohort.get("status") != "complete" or audit["invalid"]:
                continue
            maxima = maximal_groups(audit)
            if 0.0 not in maxima:
                continue
            eligible.append((run_index, cohort, maxima, paged_backend(item["raw"], cohort)))
    if not eligible:
        return None, "No completed, storage-valid cohort has a private-copy reference; no plot generated."
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, "Matplotlib is not installed; all measured values remain in the tables."
    colors = [(0.0, "Private copies (same paged kernel)", "#2563EB"),
              (.75, "75% physical KV-head sharing", "#F59E0B"),
              (1.0, "100% prefix-sharing storage control", "#64748B")]
    width = max(9.5, min(26, len(eligible) * 1.55))
    fig, ax = plt.subplots(figsize=(width, 5.8))
    bar_width = .25
    for series, (ratio, label, color) in enumerate(colors):
        for index, (_, cohort, maxima, _) in enumerate(eligible):
            group = maxima.get(ratio)
            if group is None:
                continue
            x = index + (series - 1) * bar_width
            diagnostic = cohort.get("independent_native_model_gate", {}).get("passed") is not True
            ax.bar(x, group["aggregate_qps"], bar_width * .92, color=color,
                   hatch="//" if diagnostic else None, edgecolor="white", linewidth=.6,
                   label=label if index == 0 else None)
            ax.text(x, group["aggregate_qps"], f"{group['aggregate_qps']:.2f}\nB={group['batch']}",
                    ha="center", va="bottom", fontsize=8)
    labels = []
    for run, cohort, _, backend in eligible:
        star = "*" if cohort.get("independent_native_model_gate", {}).get("passed") is not True else ""
        prefix = f"R{run} " if len(inputs) > 1 else ""
        labels.append(f"{prefix}{cohort.get('dataset', '?')}{star}\n{backend} | {cohort.get('document_tokens', 0) / 1024:g}K / {cohort.get('output_tokens', '?')} out")
    ax.set_xticks(range(len(eligible)), labels, fontsize=8, rotation=15 if len(eligible) > 6 else 0)
    ax.set_ylabel("Completed requests / second (admission inclusive)")
    ax.set_title("Real Qwen3 shared-prefix storage ablation: measured batch-wave QPS", fontsize=12)
    plotted_max = max(group["aggregate_qps"] for _, _, maxima, _ in eligible
                      for ratio, group in maxima.items() if ratio in (0.0, .75, 1.0))
    ax.set_ylim(0, plotted_max * 1.35)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # Explicit handles keep missing series from silently changing the legend.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in colors],
              loc="upper left", fontsize=8, frameon=False)
    fig.text(.01, .01, "* Hatched bars: diagnostic only; native-logit gate did not pass. No task-accuracy or production-serving claim.\n"
             "Within each labeled backend group: same kernel and common GPU source + fixed online pool budget. Native FA2 is separate; skips are absent.", fontsize=8)
    fig.tight_layout(rect=(0, .085, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="also render a PNG if Matplotlib is installed")
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in args.inputs]
    if len(set(paths)) != len(paths):
        parser.error("duplicate raw input paths would double-count measured completions")
    output = args.output_dir.resolve()
    targets = [output / "REPORT.md", output / "REPORT.provenance.json", output / "qps_storage_ablation.png"]
    if set(paths) & set(targets):
        parser.error("a derivative output must not overwrite a raw input")
    inputs = []
    for path in paths:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        if raw.get("schema") not in (SCHEMA, NATIVE_SCHEMA) or not isinstance(raw.get("cohorts"), list):
            parser.error(f"not a recognized real-model QPS raw file: {path}")
        inputs.append(dict(path=path, sha256=hashlib.sha256(raw_bytes).hexdigest(), raw=raw))
    output.mkdir(parents=True, exist_ok=True)
    plot_name, plot_note = make_plot(inputs, targets[2]) if args.plot else (None, None)
    markdown, totals = render_report(inputs, plot_name=plot_name, plot_note=plot_note)
    targets[0].write_text(markdown)
    produced = [targets[0]] + ([targets[2]] if plot_name else [])
    provenance = dict(schema="segpaged-qps-report-provenance-v1",
                      generated_utc=datetime.now(timezone.utc).isoformat(),
                      renderer=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
                      inputs=[dict(path=str(item["path"]), sha256=item["sha256"],
                                   schema=item["raw"]["schema"], status=item["raw"].get("status"),
                                   recorded_experiment_source_hashes=item["raw"].get("source_hashes", {})) for item in inputs],
                      totals=totals,
                      outputs=[dict(path=str(path), sha256=digest(path)) for path in produced],
                      renderer_writes_raw_inputs=False,
                      raw_inputs_unchanged_during_render=all(digest(item["path"]) == item["sha256"] for item in inputs))
    targets[1].write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(report=str(targets[0]), provenance=str(targets[1]),
                          plot=str(targets[2]) if plot_name else None, **totals), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
