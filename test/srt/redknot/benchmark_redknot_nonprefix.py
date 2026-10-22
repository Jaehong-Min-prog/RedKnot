#!/usr/bin/env python3
"""Frozen genuine-nonprefix QA and admission-inclusive queued service experiment.

This is an experimental single-GPU Qwen3 binding, NOT a production SGLang/TP
server. Exact prompt IDs are shared across all arms. Offline document creation
is excluded from hot service time but is separately timed and memory-accounted.
Every request has a private prefix before the shared document(s), recovered
online inside the measured interval. No output-based case selection is allowed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import faulthandler
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import signal
import string
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def bootstrap_imports():
    # Standalone experiment: no production server side effects/dependencies.
    for name in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.attention",
                 "sglang.srt.layers.attention.redknot", "sglang.srt.mem_cache"):
        if name in sys.modules:
            raise RuntimeError("standalone runner requires a fresh Python process")
        module = types.ModuleType(name)
        module.__path__ = [str((ROOT / "python").joinpath(*name.split(".")))]
        sys.modules[name] = module


def digest(data):
    if not isinstance(data, bytes):
        data = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def event(kind, **values):
    print(json.dumps(dict(event=kind, **values), ensure_ascii=False), flush=True)


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def scores(text, answers):
    pred = normalize(text)
    result = []
    for answer in answers:
        ref = normalize(answer)
        common = sum((Counter(pred.split()) & Counter(ref.split())).values())
        f1 = 2 * common / (len(pred.split()) + len(ref.split())) if pred or ref else 1.
        result.append((f1, float(pred == ref)))
    return dict(f1=max(x[0] for x in result), em=max(x[1] for x in result))


def freeze_workload(args, tokenizer):
    if args.manifest:
        payload = json.loads(args.manifest.read_text())
        if payload["schema"] != "redknot-nonprefix-frozen-v1":
            raise ValueError("incompatible frozen manifest")
        if not payload.get("items") or payload.get("max_new_tokens") != args.output_tokens:
            raise ValueError("empty manifest or output-token protocol mismatch")
        keys = [item["key"] for item in payload["items"]]
        vocab_size = len(tokenizer)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate manifest example IDs")
        for item in payload["items"]:
            if digest(item["document_token_ids"]) != item["document_tokens_sha256"]:
                raise ValueError("frozen document token hash mismatch")
            prefixes = item["prefix_token_ids"]
            chunks, query = item["document_token_ids"], item["query_token_ids"]
            if (not chunks or any(not chunk for chunk in chunks) or not query or not item.get("answers")
                    or len(prefixes) != payload["request_replicas"]
                    or len(prefixes) != len(item["exact_prompt_sha256"])
                    or any(not prefix for prefix in prefixes)):
                raise ValueError("invalid frozen request structure")
            if len({tuple(prefix) for prefix in prefixes}) != len(prefixes):
                raise ValueError("non-prefix workload requires different private prefixes")
            flat = sum(chunks, [])
            if len(flat) != item["context_tokens"]:
                raise ValueError("frozen context length mismatch")
            for prefix, expected in zip(prefixes, item["exact_prompt_sha256"]):
                ids = prefix + flat + query
                if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
                    raise ValueError("invalid frozen token IDs")
                if digest(ids) != expected:
                    raise ValueError("frozen exact prompt hash mismatch")
                if len(ids)+args.output_tokens > 32768:
                    raise ValueError("frozen prompt exceeds experiment context limit")
        return payload
    selected, sources, exclusions = [], {}, []
    excluded_keys = set()
    if getattr(args, "exclude_manifest", None):
        excluded_keys = {x["key"] for x in json.loads(args.exclude_manifest.read_text())["items"]}
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    for dataset in args.datasets:
        path = args.data / (dataset + ".jsonl")
        sources[dataset] = dict(path=str(path), sha256=digest(path.read_bytes()))
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        indices = list(range(len(rows)))
        random.Random(args.seed + sum(dataset.encode())).shuffle(indices)
        count = 0
        for index in indices:
            if f"{dataset}:{index}" in excluded_keys:
                exclusions.append(dict(dataset=dataset, row=index, reason="calibration_item_excluded"))
                continue
            row = rows[index]
            document = encode(row["context"])
            if not args.min_context <= len(document) <= args.max_context:
                exclusions.append(dict(dataset=dataset, row=index, reason="context_length", tokens=len(document)))
                continue
            # Exact token concatenation is frozen; neither arm decode/re-encodes.
            prefixes = [encode("<|im_start|>user\nRead the supplied documents and answer the question. "
                               "Give only the short answer, without explanation.\n"
                               f"Request reference: {replica + 731}." +
                               (" Administrative metadata is unrelated to the question." * replica
                                if getattr(args, "vary_prefix_lengths", False) else "") +
                               "\n" * (replica + 1))
                        for replica in range(args.replicas)]
            if getattr(args, "vary_prefix_lengths", False) and args.replicas > 1 and len(set(map(len, prefixes))) != args.replicas:
                raise ValueError("tokenizer did not produce distinct private-prefix lengths")
            query_instruction = ("\nUsing the documents above, answer the question. "
                "Output only the answer itself, without labels, explanations, or restating the question.\n"
                if getattr(args, "instruction_at_query", False) else "\n")
            query = encode(query_instruction + "Question: " + row["input"] + "\nAnswer:<|im_end|>\n"
                           "<|im_start|>assistant\n<think>\n\n</think>\n\n")
            if max(map(len, prefixes)) + len(document) + len(query) + args.output_tokens > 32768:
                exclusions.append(dict(dataset=dataset, row=index, reason="total_context_limit"))
                continue
            chunks = [document[i:i+args.document_chunk] for i in range(0, len(document), args.document_chunk)]
            item = dict(key=f"{dataset}:{index}", dataset=dataset, source_row=index,
                        source_id=row.get("_id"), answers=row["answers"], question=row["input"],
                        document_token_ids=chunks, document_tokens_sha256=digest(chunks),
                        context_tokens=len(document), prefix_token_ids=prefixes, query_token_ids=query,
                        exact_prompt_sha256=[digest(p + document + query) for p in prefixes])
            selected.append(item)
            count += 1
            if count == args.items_per_dataset:
                break
        if count != args.items_per_dataset:
            raise ValueError(f"{dataset}: only {count} length-eligible original examples")
    return dict(schema="redknot-nonprefix-frozen-v1", seed=args.seed, sources=sources,
                selection="seeded source-row permutation; full contexts; length-only eligibility before outputs",
                exclusions=exclusions, items=selected, tokenizer=str(args.model),
                document_chunk=args.document_chunk, max_new_tokens=args.output_tokens,
                unique_examples=len(selected), request_replicas=args.replicas,
                quality_scope="replica-zero primary plus all-prefix robustness; replicas are not independent QA examples",
                instruction_at_query=bool(getattr(args, "instruction_at_query", False)),
                varying_prefix_lengths=bool(getattr(args, "vary_prefix_lengths", False)),
                excluded_calibration_keys=sorted(excluded_keys))


def profile_data(profile):
    policies = [[vars(profile.get_strategy(l, h)) for h in range(profile.num_kv_heads)]
                for l in range(profile.num_layers)]
    return dict(policies=policies, effective_local_ratio=sum(p["head_type"] == "local" for row in policies for p in row)
                / (profile.num_layers * profile.num_kv_heads), sha256=digest(policies))


def summarize(results, items, expected_repeats=1):
    """Summarize measured rows without hiding failures or unpaired cohorts.

    Replica zero remains the predeclared primary QA metric. An additional
    all-prefix robustness metric averages all replicas within a repeat, then
    repeats within a unique source example, then equally weights examples.
    Both primary and robustness gates must pass. Throughput includes every
    measured repeat, but comparisons use common successful item/repeat cohorts.
    Neither repetition nor replicas inflate the independent QA N.
    """
    import math

    if type(expected_repeats) is not int or expected_repeats < 1:
        raise ValueError("expected_repeats must be positive")
    item_map = {item["key"]: item for item in items}
    expected = {(key, repeat) for key in item_map for repeat in range(expected_repeats)}
    datasets_keys = {dataset: [key for key, item in item_map.items() if item["dataset"] == dataset]
                     for dataset in sorted({item["dataset"] for item in items})}
    byarm, internal = {}, {}

    def mean(scores_list, metric):
        return statistics.mean(value[metric] for value in scores_list) if scores_list else None

    def score_valid(value):
        return isinstance(value, dict) and all(
            isinstance(value.get(key), (int, float)) and not isinstance(value[key], bool)
            and math.isfinite(value[key]) and 0 <= value[key] <= 1 for key in ("f1", "em"))

    def metric_drops(base_scores, candidate_scores):
        values = {}
        for metric in ("f1", "em"):
            baseline, candidate = mean(base_scores, metric), mean(candidate_scores, metric)
            values["paired_native_" + metric] = baseline
            values["paired_" + metric] = candidate
            drop = 100 * (baseline - candidate) if baseline is not None and candidate is not None else None
            values[metric + "_drop_pp"] = drop
            values[metric + "_relative_drop_percent"] = drop / baseline if drop is not None and baseline > 0 else None
        # Zero-baseline metrics are non-informative for the relative criterion.
        values["metric_2percent_gate"] = all(
            values[key] is not None and values[key] <= 2 + 1e-10
            for key in ("f1_drop_pp", "em_drop_pp", "f1_relative_drop_percent", "em_relative_drop_percent"))
        return values

    for arm in sorted({row["arm"] for row in results}):
        arm_rows = [row for row in results if row["arm"] == arm]
        cohorts, quality, robustness_quality = {}, {}, {}
        failed = cancelled = completed_requests = output_tokens = 0
        robustness_prefix_outputs = 0
        total, incomplete = 0.0, 0
        for row in arm_rows:
            key = (row["item_key"], row.get("repeat", 0))
            service = row["service"]
            records = service.get("records", [])
            cohorts.setdefault(key, []).append(row)
            observed_completed = sum(record.get("status") == "completed" for record in records)
            completed_requests += observed_completed
            failed += max(service.get("failed_requests", 0), sum(record.get("status") == "failed" for record in records))
            cancelled += max(service.get("cancelled_requests", 0), sum(record.get("status") == "cancelled" for record in records))
            output_tokens += service.get("completed_output_tokens", 0)
            wall = service.get("total_wall_seconds", 0)
            valid_wall = isinstance(wall, (int, float)) and math.isfinite(wall) and wall > 0
            total += wall if valid_wall else 0
            replica_count = len(item_map.get(key[0], {}).get("prefix_token_ids", records))
            complete = (key in expected and service.get("status") == "complete" and not service.get("errors")
                        and records and len(records) == replica_count
                        and observed_completed == len(records) == service.get("completed_requests")
                        and not service.get("failed_requests", 0) and not service.get("cancelled_requests", 0)
                        and valid_wall)
            incomplete += not complete
            if key in expected and records and records[0].get("status") == "completed":
                score = records[0].get("accuracy")
                if score_valid(score):
                    quality.setdefault(key[0], []).append(score)
            # A failed, missing, or unscorable variant must not disappear from
            # the denominator. Only a complete prefix set receives a repeat
            # robustness score; coverage gates separately require every repeat.
            if (key in expected and records and len(records) == replica_count
                    and all(record.get("status") == "completed"
                            and score_valid(record.get("accuracy")) for record in records)):
                robustness_quality.setdefault(key[0], []).append({
                    metric: mean([record["accuracy"] for record in records], metric)
                    for metric in ("f1", "em")
                })
                robustness_prefix_outputs += len(records)
        per_item = {key: {metric: mean(values, metric) for metric in ("f1", "em")}
                    for key, values in quality.items()}
        robustness_per_item = {key: {metric: mean(values, metric) for metric in ("f1", "em")}
                               for key, values in robustness_quality.items()}
        valid_cohorts = {}
        for key, rows in cohorts.items():
            if key not in expected or len(rows) != 1:
                continue
            row = rows[0]
            service, records = row["service"], row["service"].get("records", [])
            count = len(item_map[key[0]].get("prefix_token_ids", records))
            wall = service.get("total_wall_seconds", 0)
            if (service.get("status") == "complete" and not service.get("errors") and records
                    and len(records) == count == service.get("completed_requests")
                    and all(record.get("status") == "completed" for record in records)
                    and not service.get("failed_requests", 0) and not service.get("cancelled_requests", 0)
                    and isinstance(wall, (int, float)) and math.isfinite(wall) and wall > 0):
                valid_cohorts[key] = row
        coverage_complete = (bool(expected) and len(item_map) == len(items)
                             and set(cohorts) == expected and set(valid_cohorts) == expected
                             and all(len(rows) == 1 for rows in cohorts.values()))
        dataset_values, robustness_datasets = {}, {}
        for dataset, keys in datasets_keys.items():
            available = [per_item[key] for key in keys if key in per_item]
            dataset_values[dataset] = dict(expected=len(keys), completed=len(available),
                                          f1=mean(available, "f1"), em=mean(available, "em"))
            available_robustness = [robustness_per_item[key] for key in keys if key in robustness_per_item]
            robustness_datasets[dataset] = dict(
                expected=len(keys), completed=len(available_robustness),
                f1=mean(available_robustness, "f1"), em=mean(available_robustness, "em"))
        byarm[arm] = dict(
            unique_scored_examples=len(per_item), expected_examples=len(items),
            expected_repeats=expected_repeats, expected_cohorts=len(expected),
            measured_cohorts=len(arm_rows), successful_cohorts=len(valid_cohorts),
            incomplete_cohorts=int(incomplete), duplicate_cohorts=sum(len(v) > 1 for v in cohorts.values()),
            all_expected_cohorts_complete=coverage_complete, datasets=dataset_values,
            f1=mean(list(per_item.values()), "f1"), em=mean(list(per_item.values()), "em"),
            requests=completed_requests, failed_requests=failed, cancelled_requests=cancelled,
            measured_wall_seconds=total, aggregate_qps=completed_requests / total if total else 0,
            output_tokens=output_tokens, observed_2percent_gate=False,
            primary_observed_2percent_gate=False, qps_vs_native=None,
            primary_quality_scope="predeclared replica zero; f1/em at this level are primary metrics",
            all_prefix_robustness=dict(
                unique_scored_examples=len(robustness_per_item), expected_examples=len(items),
                scored_prefix_outputs=robustness_prefix_outputs,
                independent_sample_count=len(robustness_per_item),
                f1=mean(list(robustness_per_item.values()), "f1"),
                em=mean(list(robustness_per_item.values()), "em"),
                datasets=robustness_datasets, observed_2percent_gate=False,
                aggregation="all prefixes within repeat, repeats within unique QA example, then equal-weight QA examples; no prefix is selected or excluded",
            ),
        )
        internal[arm] = dict(scores=per_item, quality_counts={k:len(v) for k,v in quality.items()},
                             cohorts=valid_cohorts, robustness_scores=robustness_per_item,
                             robustness_counts={k:len(v) for k,v in robustness_quality.items()})

    if "recomputed_native" in byarm:
        base, baseline = byarm["recomputed_native"], internal["recomputed_native"]
        for arm, values in byarm.items():
            current = internal[arm]
            common_items = sorted(set(baseline["scores"]) & set(current["scores"]))
            values.update(metric_drops([baseline["scores"][key] for key in common_items],
                                       [current["scores"][key] for key in common_items]))
            values["quality_matched_examples"] = len(common_items)
            common = sorted(set(baseline["cohorts"]) & set(current["cohorts"]))
            # The frozen workload fixes replica count; also reject malformed
            # external rows whose actual completed counts differ between arms.
            common = [key for key in common if baseline["cohorts"][key]["service"]["completed_requests"]
                      == current["cohorts"][key]["service"]["completed_requests"]]
            values["paired_cohort_count"] = len(common)
            values["paired_item_count"] = len({key[0] for key in common})
            values["paired_comparison_complete"] = (set(common) == expected and bool(expected)
                and values["all_expected_cohorts_complete"] and base["all_expected_cohorts_complete"])
            if common:
                native_wall = sum(baseline["cohorts"][key]["service"]["total_wall_seconds"] for key in common)
                candidate_wall = sum(current["cohorts"][key]["service"]["total_wall_seconds"] for key in common)
                count = sum(current["cohorts"][key]["service"]["completed_requests"] for key in common)
                values.update(paired_native_qps=count / native_wall, paired_qps=count / candidate_wall,
                              qps_vs_native=native_wall / candidate_wall)
            per_dataset_pass = []
            for dataset, keys in datasets_keys.items():
                matched = [key for key in keys if key in common_items]
                comparison = metric_drops([baseline["scores"][key] for key in matched],
                                          [current["scores"][key] for key in matched])
                comparison["quality_matched_examples"] = len(matched)
                comparison["observed_2percent_gate"] = (len(matched) == len(keys)
                    and all(current["quality_counts"][key] == expected_repeats
                            and baseline["quality_counts"][key] == expected_repeats for key in matched)
                    and all((key, repeat) in baseline["cohorts"] and (key, repeat) in current["cohorts"]
                            for key in keys for repeat in range(expected_repeats))
                    and comparison["metric_2percent_gate"])
                values["datasets"][dataset].update(comparison)
                per_dataset_pass.append(comparison["observed_2percent_gate"])
            values["primary_observed_2percent_gate"] = bool(
                values["paired_comparison_complete"] and len(common_items) == len(items)
                and values["metric_2percent_gate"] and all(per_dataset_pass))
            robustness = values["all_prefix_robustness"]
            robust_common = sorted(set(baseline["robustness_scores"]) & set(current["robustness_scores"]))
            robustness.update(metric_drops([baseline["robustness_scores"][key] for key in robust_common],
                                           [current["robustness_scores"][key] for key in robust_common]))
            robustness["quality_matched_examples"] = len(robust_common)
            robust_dataset_pass = []
            for dataset, keys in datasets_keys.items():
                matched = [key for key in keys if key in robust_common]
                comparison = metric_drops([baseline["robustness_scores"][key] for key in matched],
                                          [current["robustness_scores"][key] for key in matched])
                comparison["quality_matched_examples"] = len(matched)
                comparison["observed_2percent_gate"] = (len(matched) == len(keys)
                    and all(current["robustness_counts"][key] == expected_repeats
                            and baseline["robustness_counts"][key] == expected_repeats for key in matched)
                    and all((key, repeat) in baseline["cohorts"] and (key, repeat) in current["cohorts"]
                            for key in keys for repeat in range(expected_repeats))
                    and comparison["metric_2percent_gate"])
                robustness["datasets"][dataset].update(comparison)
                robust_dataset_pass.append(comparison["observed_2percent_gate"])
                primary_dataset = values["datasets"][dataset]
                primary_dataset["primary_observed_2percent_gate"] = primary_dataset["observed_2percent_gate"]
                primary_dataset["observed_2percent_gate"] = bool(
                    primary_dataset["primary_observed_2percent_gate"] and comparison["observed_2percent_gate"])
            robustness["observed_2percent_gate"] = bool(
                values["paired_comparison_complete"] and len(robust_common) == len(items)
                and robustness["metric_2percent_gate"] and all(robust_dataset_pass))
            values["observed_2percent_gate"] = bool(
                values["primary_observed_2percent_gate"] and robustness["observed_2percent_gate"])
    return dict(arms=byarm, scope="observed sample only, not a generalization guarantee; no statistical certification",
                quality_metric="LongBench-style normalized English token F1 and exact match vs ground truth",
                quality_aggregation="primary replica zero repeat means per unique QA; additional all-prefix robustness averages prefixes within repeat then repeats within unique QA; replicas are not independent samples",
                gate="primary AND all-prefix robustness must both pass absolute F1/EM drops <=2 percentage points and relative F1/EM drops <=2 percent, per dataset and overall; zero baseline is non-informative",
                timing="queue-inclusive completed requests / total service wall time; includes online recovery; speedup uses matched successful item/repeat cohorts only",
                not_measured=["production TP8 scheduler", "SLA-qualified saturated serving", "total FLOP reduction"])


def main():
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"experiment interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=HERE / "datasets/LongBench/data")
    parser.add_argument("--profile", type=Path, default=HERE / "head_class/qwen3-32B_optimal_g15_lf_ret.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--exclude-manifest", type=Path, help="Exclude calibration source IDs before held-out selection")
    parser.add_argument("--instruction-at-query", action="store_true", help="Repeat the same short-answer instruction in every arm's suffix")
    parser.add_argument("--vary-prefix-lengths", action="store_true", help="Use distinct private-prefix lengths to test canonical KV reuse at different offsets")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimqa", "multifieldqa_en", "musique"])
    parser.add_argument("--items-per-dataset", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--min-context", type=int, default=2048)
    parser.add_argument("--max-context", type=int, default=28000)
    parser.add_argument("--document-chunk", type=int, default=4096)
    parser.add_argument("--pool-gib", type=float, default=20)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--global-prefix-layers", type=int, default=None,
                        help="Override protected early layers; otherwise honor profile front_full_layers/dense_prefix_layers")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arms", nargs="+", choices=["recomputed_native", "recomputed_paged", "redknot_private", "redknot_shared"],
                        default=["recomputed_native", "recomputed_paged", "redknot_private", "redknot_shared"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.items_per_dataset, args.replicas, args.concurrency, args.repeats, args.output_tokens,
           args.document_chunk, args.pool_gib) <= 0:
        raise ValueError("counts, chunks and memory must be positive")
    # Atomic ownership: concurrent writers cannot share an evidence directory.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    workload = freeze_workload(args, tokenizer)
    save(args.output / "frozen_workload.json", workload)
    event("workload_frozen", examples=len(workload["items"]), lengths=[x["context_tokens"] for x in workload["items"]])
    if args.prepare_only:
        return 0
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("experimental binding requires verified Transformers 4.57.1")
    bootstrap_imports()
    from sglang.srt.layers.attention.redknot.nonprefix_paged import NonPrefixPagedEngine
    from sglang.srt.layers.attention.redknot.nonprefix_scheduler import QueuedRequest, run_queued
    from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig
    from nonprefix_native_reference import NativeQueuedEngine
    from nonprefix_profile import load_effective_profile
    torch.manual_seed(args.seed)
    profile, profile_diagnostics = load_effective_profile(args.profile, HeadClassConfig, args.global_prefix_layers)
    converted = profile_diagnostics["retrieval_heads_promoted_global"]
    event("effective_head_profile", **profile_diagnostics)
    model = Qwen3ForCausalLM.from_pretrained(str(args.model), local_files_only=True,
             torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map={"": "cuda:0"}).eval()
    # The same native full MLP is used in all arms to isolate KV/reuse effects.
    headbytes = 256 * model.config.head_dim * 2 * 2
    capacity = int(args.pool_gib * 1024**3) // headbytes
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    evidence = dict(schema="redknot-nonprefix-e2e-v1", status="running", results=[],
                    config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    profile=profile_data(profile), retrieval_heads_explicitly_promoted_global=converted,
                    profile_diagnostics=profile_diagnostics,
                    ffn="full native Qwen3 MLP in all arms; no sparse FFN", hardware=dict(
                        gpu=torch.cuda.get_device_name(), total_bytes=torch.cuda.get_device_properties(0).total_memory,
                        capability=torch.cuda.get_device_capability(), torch=torch.__version__, transformers=transformers.__version__),
                    model_config_sha256=digest((args.model / "config.json").read_bytes()),
                    rope=dict(type=model.model.rotary_emb.rope_type,
                              scaling=model.config.rope_scaling,
                              attention_scaling=float(model.model.rotary_emb.attention_scaling)),
                    workload_sha256=digest(workload), source_hashes={p.name:digest(p.read_bytes()) for p in (
                        Path(__file__), HERE / "nonprefix_native_reference.py", HERE / "nonprefix_profile.py",
                        ROOT / "python/sglang/srt/layers/attention/redknot/nonprefix_paged.py",
                        ROOT / "python/sglang/srt/layers/attention/redknot/nonprefix_scheduler.py",
                        ROOT / "python/sglang/srt/mem_cache/head_kv/manager.py",
                        ROOT / "python/sglang/srt/mem_cache/head_kv/pool.py")})
    evidence["effective_workload_protocol"] = dict(
        instruction_at_query=workload.get("instruction_at_query", False),
        varying_prefix_lengths=workload.get("varying_prefix_lengths", False),
        prefix_lengths_by_item={item["key"]: list(map(len, item["prefix_token_ids"])) for item in workload["items"]})
    save(args.output / "results.json", evidence)
    try:
        for number, item in enumerate(workload["items"]):
            # Rotate arm order before seeing outputs to reduce timing-order bias.
            arms = args.arms[number % len(args.arms):] + args.arms[:number % len(args.arms)]
            for arm in arms:
                event("arm_start", arm=arm, item=item["key"], context_tokens=item["context_tokens"])
                gc.collect()
                torch.cuda.empty_cache()
                cfg = copy.deepcopy(profile)
                if arm == "recomputed_paged":
                    cfg.head_class = [["global"] * cfg.num_kv_heads for _ in range(cfg.num_layers)]
                    cfg.dense_prefix_layers = 0
                engine = (NativeQueuedEngine(model) if arm == "recomputed_native" else NonPrefixPagedEngine(
                    model, cfg, capacity_pages=capacity, backend="flash", sparse_ffn_schedule=None,
                    model_revision=evidence["model_config_sha256"], share_local=arm != "redknot_private"))
                try:
                    engine.synchronize()
                    start = time.perf_counter()
                    docs = []
                    for index, ids in enumerate(item["document_token_ids"]):
                        key = f"{item['key']}:chunk:{index}"
                        engine.register_document(key, ids)
                        docs.append(key)
                    engine.synchronize()
                    offline_seconds = time.perf_counter()-start
                    def request(replica, request_id, tokens=args.output_tokens):
                        prefix = item["prefix_token_ids"][replica]
                        flat = prefix + sum(item["document_token_ids"], []) + item["query_token_ids"]
                        assert digest(flat) == item["exact_prompt_sha256"][replica]
                        return QueuedRequest(request_id, prefix, docs, item["query_token_ids"], tokens, eos)
                    if args.warmup:
                        warm = run_queued(engine, [request(0, "warmup", 2)], max_active=1)
                        if warm["status"] != "complete":
                            raise RuntimeError(f"warmup failed: {warm['errors']}")
                    for repeat in range(args.repeats):
                        torch.cuda.reset_peak_memory_stats()
                        allocated_before = torch.cuda.memory_allocated()
                        requests = [request(r, f"request:{r}") for r in range(len(item["prefix_token_ids"]))]
                        result = run_queued(engine, requests, max_active=args.concurrency)
                        for r in result["records"]:
                            r["text"] = tokenizer.decode(r["output_ids"], skip_special_tokens=True)
                            r["accuracy"] = scores(r["text"], item["answers"]) if r["status"] == "completed" else None
                        row = dict(arm=arm, item_key=item["key"], dataset=item["dataset"], repeat=repeat,
                                   answers=item["answers"], offline_seconds=offline_seconds,
                                   allocated_before_service=allocated_before, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), service=result)
                        evidence["results"].append(row)
                        evidence["summary"] = summarize(evidence["results"], workload["items"], expected_repeats=args.repeats)
                        save(args.output / "results.json", evidence)
                        event("arm_result", arm=arm, item=item["key"], repeat=repeat, qps=result["completed_qps"],
                              status=result["status"], texts=[r["text"] for r in result["records"]],
                              scores=[r["accuracy"] for r in result["records"]])
                        if result["status"] != "complete":
                            raise RuntimeError(f"queued service failed: {result['errors']}")
                finally:
                    engine.close()
                    del engine
        evidence["status"] = "completed"
    except BaseException as exc:
        evidence["status"] = "failed"
        evidence["error"] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        evidence["summary"] = summarize(evidence["results"], workload["items"], expected_repeats=args.repeats)
        save(args.output / "results.json", evidence)
        event("finished", status=evidence["status"], summary=evidence["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
