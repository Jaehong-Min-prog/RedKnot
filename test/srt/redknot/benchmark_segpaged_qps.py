#!/usr/bin/env python3
"""Real-model completed-generation throughput for a head-sharing storage ablation.

This is NOT a production SGLang server, non-prefix RedKnot quality validation,
or MLA experiment. A single, position-identical document has already been
prefilled. Independent physical KV copies and partially/fully shared copies
feed the SAME full-attention Qwen3 model. Every generated token consumes the
paged attention result. A common GPU source cache is kept in ALL arms and
charged separately, in addition to the identical online page-pool budget.

Main QPS includes fresh admission/copies, question prefill, fixed-token greedy
generation and cleanup. Resident-generation QPS is explicitly secondary.
Requests in a wave run as a true model batch, not serial generation calls.
EOS is ignored to fix work; continuation after EOS is not a quality example.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
import transformers
from transformers import AutoTokenizer, Qwen3ForCausalLM

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from segpaged_qwen3_throughput_adapter import BatchedQwen3PagedDecode, required_capacity_pages


def adapter_type(backend="triton"):
    if backend == "triton":
        return BatchedQwen3PagedDecode
    if backend == "flash":
        from segpaged_qwen3_flash_backend import BatchedQwen3FlashPagedDecode
        return BatchedQwen3FlashPagedDecode
    raise ValueError(f"unknown experimental paged backend: {backend}")


def sync():
    torch.cuda.synchronize()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def event(name, **kwargs):
    print(json.dumps(dict(event=name, **kwargs), ensure_ascii=False), flush=True)


def quantile(values, fraction):
    ordered = sorted(values)
    point = (len(ordered) - 1) * fraction
    low, high = math.floor(point), math.ceil(point)
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


def cache_bytes(cache):
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for k, v in cache)


def prepare_workload(tokenizer, path, row_index, length, max_batch, question_length):
    with path.open() as handle:
        for index, line in enumerate(handle):
            if index == row_index:
                row = json.loads(line)
                break
        else:
            raise ValueError(f"missing source row {row_index}: {path}")
    context = tokenizer.encode(str(row["context"]), add_special_tokens=False)
    header = tokenizer.encode("<|im_start|>user\nRead the document below.\nDocument:\n", add_special_tokens=False)
    if not context or len(header) >= length:
        raise ValueError("empty or too short document workload")
    remaining = length - len(header)
    repeats = math.ceil(remaining / len(context))
    document = header + (context * repeats)[:remaining]
    native_question = str(row.get("input", "Summarize the document."))
    templates = [
        "Answer the following question using the document: " + native_question,
        "Summarize the main subject of the document in three sentences.",
        "Name three important people or organizations in the document and explain their roles.",
        "Describe the key events in the document in chronological order.",
        "What evidence in the document is most relevant to this question? " + native_question,
        "Give a short overview of the document, including its most important facts.",
        "List three concrete facts supported by the document, without inventing information.",
        "Explain how the main entities mentioned in the document are related.",
    ]
    pad = tokenizer.encode("\n", add_special_tokens=False)
    if len(pad) != 1:
        raise ValueError("this workload requires a one-token newline")
    suffixes, questions = [], []
    for request in range(max_batch):
        question = f"\nRequest {request + 1}: {templates[request % len(templates)]}"
        tail = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        tail_ids = tokenizer.encode(tail, add_special_tokens=False)
        # Truncation is explicit and stable across arms; this is a systems
        # stress workload derived from source text, not a LongBench score.
        question_ids = question_ids[:question_length - len(tail_ids)]
        ids = question_ids + tail_ids
        ids = pad * (question_length - len(ids)) + ids
        if len(ids) != question_length:
            raise AssertionError("question shape mismatch")
        suffixes.append(ids)
        questions.append(tokenizer.decode(question_ids, skip_special_tokens=False))
    metadata = dict(dataset=path.stem, source_row=row_index, source_file_sha256=sha(path),
                    source_row_sha256=hashlib.sha256(line.encode()).hexdigest(),
                    original_context_tokens=len(context), repeated_context=repeats > 1,
                    repetition_count=repeats, truncation_or_repetition_for_exact_length=True,
                    document_tokens=len(document), question_tokens=question_length,
                    input_sha256=hashlib.sha256(json.dumps([document, suffixes]).encode()).hexdigest(),
                    question_texts=questions, question_token_ids=suffixes,
                    quality_scope="fixed-token systems workload, not task-accuracy evaluation")
    return torch.tensor([document], device="cuda"), torch.tensor(suffixes, device="cuda"), metadata


@torch.inference_mode()
def prefill(model, document, chunk_size):
    sync()
    start = time.perf_counter()
    cache = None
    for start_token in range(0, document.shape[1], chunk_size):
        out = model(input_ids=document[:, start_token:start_token + chunk_size],
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
    sync()
    elapsed = time.perf_counter() - start
    del out
    return cache, elapsed


def capacity(model, length, batch, ratio, suffix, page):
    return required_capacity_pages(model, length, batch, ratio, suffix, page)


@torch.inference_mode()
def independent_model_gate(model, source, suffixes, length, page_size, new_tokens=16, paged_backend="triton"):
    """Compare actual pretrained GPU generation with independent native FA2."""
    source_length_before = source.get_seq_length()
    native_cache = copy.deepcopy(source)
    native_ids = []
    finite_flags = []
    result = model(input_ids=suffixes[:1], past_key_values=native_cache,
                   use_cache=True, logits_to_keep=1)
    native_first = result.logits[:, -1].float().clone()
    finite_flags.append(torch.isfinite(result.logits).all())
    token = result.logits[:, -1].argmax(-1, keepdim=True)
    native_ids.append(token)
    for _ in range(new_tokens - 1):
        result = model(input_ids=token, past_key_values=native_cache, use_cache=True, logits_to_keep=1)
        finite_flags.append(torch.isfinite(result.logits).all())
        token = result.logits[:, -1].argmax(-1, keepdim=True)
        native_ids.append(token)
    native_tokens = torch.cat(native_ids, dim=1).cpu().tolist()[0]
    del native_cache, result
    suffix_length = suffixes.shape[1] + new_tokens - 1
    pages = capacity(model, length, 1, 0.0, suffix_length, page_size)
    adapter = adapter_type(paged_backend)(model, source, document_length=length, batch_size=1,
                                     shared_kv_head_ratio=0.0, capacity_pages=pages,
                                     max_suffix_tokens=suffix_length, page_size=page_size)
    with adapter:
        result = adapter.forward(suffixes[:1])
        paged_first = result.logits[:, -1].float().clone()
        finite_flags.append(torch.isfinite(result.logits).all())
        paged_ids = []
        token = result.logits[:, -1].argmax(-1, keepdim=True)
        paged_ids.append(token)
        for _ in range(new_tokens - 1):
            result = adapter.forward(token)
            finite_flags.append(torch.isfinite(result.logits).all())
            token = result.logits[:, -1].argmax(-1, keepdim=True)
            paged_ids.append(token)
        paged_tokens = torch.cat(paged_ids, dim=1).cpu().tolist()[0]
    del adapter
    sync()
    finite = bool(torch.isfinite(paged_first).all())
    all_finite = bool(torch.stack(finite_flags).all())
    maximum = float((native_first - paged_first).abs().max())
    cosine = float(torch.nn.functional.cosine_similarity(native_first.flatten(), paged_first.flatten(), dim=0))
    unchanged = source.get_seq_length() == source_length_before
    gate = dict(native_backend="flash_attention_2", paged_backend=paged_backend,
                pretrained_gpu_model=True, batch_size=1, first_logits_max_abs_error=maximum,
                first_logits_cosine=cosine, first_token_equal=native_tokens[0] == paged_tokens[0],
                all_first_logits_finite=finite, native_token_ids=native_tokens, paged_token_ids=paged_tokens,
                all_native_and_paged_gate_logits_finite=all_finite,
                greedy_16_token_agreement=sum(a == b for a, b in zip(native_tokens, paged_tokens)) / new_tokens,
                source_cache_length_unchanged=unchanged,
                acceptance="finite first logits, max_abs<=0.5, cosine>=0.999, equal first token; not a task-quality certificate")
    gate["passed"] = all_finite and finite and unchanged and maximum <= .5 and cosine >= .999 and gate["first_token_equal"]
    if not gate["passed"]:
        event("independent_gate_failed", **gate)
    return gate


def configurations(model, length, args, output_tokens):
    suffix = args.question_tokens + output_tokens - 1
    budget = capacity(model, length, args.budget_requests, 0.0, suffix, args.page_size)
    configs = []
    for ratio in args.shared_ratios:
        largest = 0
        for batch in range(1, args.max_batch + 1):
            if capacity(model, length, batch, ratio, suffix, args.page_size) <= budget:
                largest = batch
        if not largest:
            continue
        batches = sorted({value for value in args.same_concurrency if value <= largest} | {largest})
        for batch in batches:
            configs.append(dict(shared_ratio=ratio, batch_size=batch,
                                budget_max_batch=largest,
                                max_batch_search_censored=largest == args.max_batch,
                                required_pages=capacity(model, length, batch, ratio, suffix, args.page_size),
                                capacity_pages=budget, max_suffix_tokens=suffix))
    if not any(x["shared_ratio"] == 0 for x in configs):
        raise ValueError("a private-copy baseline is mandatory")
    return configs


@torch.inference_mode()
def run_wave(model, tokenizer, source, document_length, suffixes, config, output_tokens, page_size, paged_backend="triton"):
    batch = config["batch_size"]
    query = suffixes[:batch]
    generated = []
    finite_flags = []
    gc.collect()
    sync()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    adapter = adapter_type(paged_backend)(
        model, source, document_length=document_length, batch_size=batch,
        shared_kv_head_ratio=config["shared_ratio"], capacity_pages=config["capacity_pages"],
        page_size=page_size, max_suffix_tokens=config["max_suffix_tokens"])
    try:
        with adapter:
            sync()
            admitted = time.perf_counter()
            result = adapter.forward(query)
            finite_flags.append(torch.isfinite(result.logits).all())
            token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(token)
            sync()
            first_token = time.perf_counter()
            for _ in range(output_tokens - 1):
                result = adapter.forward(token)
                finite_flags.append(torch.isfinite(result.logits).all())
                token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
                generated.append(token)
            sync()
            finished_generation = time.perf_counter()
            stats = adapter.stats()
            if stats["pool"]["capacity_pages"] - stats["pool"]["free_pages"] != config["required_pages"]:
                raise AssertionError("actual allocated pages disagree with admission geometry")
            peak = torch.cuda.max_memory_allocated()
    finally:
        adapter.close()
    del adapter
    sync()
    finished = time.perf_counter()
    # CPU export, detokenization, JSON and correctness comparisons are outside
    # the timed service interval; they cannot affect model generation.
    token_ids = torch.cat(generated, dim=1).cpu().tolist()
    resident = finished_generation - admitted
    elapsed = finished - start
    text = [tokenizer.decode(ids, skip_special_tokens=False) for ids in token_ids]
    eos = tokenizer.eos_token_id
    return dict(**config, output_tokens_per_request=output_tokens, completed_requests=batch,
                completed_output_tokens=batch * output_tokens,
                admission_seconds=admitted - start, resident_ttft_seconds=first_token - admitted,
                admission_inclusive_ttft_seconds=first_token - start,
                resident_generation_seconds=resident, cleanup_seconds=finished - finished_generation,
                admission_inclusive_seconds=elapsed, qps=batch / elapsed,
                resident_generation_qps=batch / resident,
                output_tokens_per_second=batch * output_tokens / elapsed,
                gpu_allocated_before_wave_bytes=allocated_before, gpu_peak_allocated_bytes=peak,
                cache_stats=stats, generated_token_ids=token_ids, generated_text=text,
                eos_encountered_requests=sum(eos in ids for ids in token_ids),
                all_generated_logits_finite=bool(torch.stack(finite_flags).all()),
                generation_policy="greedy, fixed output count, EOS ignored",
                actual_generation_uses_paged_attention=True)


def summarize(samples):
    groups = {}
    for row in samples:
        key = (row["shared_ratio"], row["batch_size"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (ratio, batch), rows in sorted(groups.items()):
        summary = dict(shared_ratio=ratio, batch_size=batch, repeats=len(rows),
                       budget_max_batch=rows[0]["budget_max_batch"],
                       max_batch_search_censored=rows[0]["max_batch_search_censored"],
                       actual_page_pool_allocation_verified=True)
        summary["aggregate_qps"] = sum(row["completed_requests"] for row in rows) / sum(row["admission_inclusive_seconds"] for row in rows)
        summary["aggregate_resident_qps"] = sum(row["completed_requests"] for row in rows) / sum(row["resident_generation_seconds"] for row in rows)
        for name in ("qps", "resident_generation_qps", "admission_seconds", "resident_ttft_seconds",
                     "admission_inclusive_ttft_seconds", "admission_inclusive_seconds", "output_tokens_per_second"):
            values = [row[name] for row in rows]
            summary[name + "_median"] = statistics.median(values)
            summary[name + "_p95_sample_quantile"] = quantile(values, .95)
            summary[name + "_min"] = min(values)
            summary[name + "_max"] = max(values)
        summary["repeat_output_identity"] = all(row["generated_token_ids"] == rows[0]["generated_token_ids"] for row in rows)
        same_batch_baseline = groups.get((0.0, batch))
        if same_batch_baseline:
            reference = same_batch_baseline[0]["generated_token_ids"]
            summary["same_batch_baseline_token_identity"] = all(row["generated_token_ids"] == reference for row in rows)
            summary["same_concurrency_qps_speedup"] = summary["qps_median"] / statistics.median(x["qps"] for x in same_batch_baseline)
        baseline_max_batch = max(b for r, b in groups if r == 0)
        baseline_max = groups[(0.0, baseline_max_batch)]
        summary["versus_baseline_max_concurrency_qps_speedup"] = summary["qps_median"] / statistics.median(x["qps"] for x in baseline_max)
        summary["versus_baseline_max_resident_qps_speedup"] = summary["resident_generation_qps_median"] / statistics.median(x["resident_generation_qps"] for x in baseline_max)
        ref = baseline_max[0]["generated_token_ids"]
        common = min(batch, baseline_max_batch)
        pairs = [(a, b) for seq_a, seq_b in zip(rows[0]["generated_token_ids"][:common], ref[:common]) for a, b in zip(seq_a, seq_b)]
        summary["cross_batch_overlap_token_agreement"] = sum(a == b for a, b in pairs) / len(pairs)
        summary["cross_batch_agreement_is_not_task_accuracy"] = True
        summary["cache_stats"] = rows[0]["cache_stats"]
        summary["gpu_peak_allocated_bytes_max"] = max(x["gpu_peak_allocated_bytes"] for x in rows)
        summaries.append(summary)
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--datasets-dir", type=Path, default=HERE / "datasets/LongBench/data")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimqa", "multifieldqa_en"])
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 16384, 31744])
    parser.add_argument("--output-tokens", nargs="+", type=int, default=[32])
    parser.add_argument("--question-tokens", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=512)
    parser.add_argument("--paged-backend", choices=["triton", "flash"], default="triton",
                        help="experimental execution backend; flash requires CUDA and page-aligned documents")
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--budget-requests", type=int, default=4)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--same-concurrency", nargs="+", type=int, default=[4])
    parser.add_argument("--shared-ratios", nargs="+", type=float, default=[0, .5, .75, 1])
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--profile-after-logit-gate-failure", action="store_true",
                        help="diagnostic storage throughput only; keep failed native-logit gate visible; require all 16 native tokens identical")
    parser.add_argument("--continue-after-gate-failure", action="store_true",
                        help="record rejected cohorts without measuring their QPS and continue the predeclared matrix")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing raw QPS result")
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("this experimental adapter requires transformers==4.57.1")
    if min(args.lengths + args.output_tokens + args.same_concurrency + [args.question_tokens, args.page_size, args.max_batch,
                                               args.budget_requests, args.repeats, args.prefill_chunk_size, args.cpu_threads]) <= 0:
        raise ValueError("positive lengths, batch sizes and repeats required")
    if args.warmup_rounds < 1 or 0 not in args.shared_ratios or 1 not in args.shared_ratios:
        raise ValueError("require full warmup and private/full-prefix-sharing controls")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(20261015)
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = dict(schema="segpaged-real-model-qps-storage-ablation-v1", status="running",
                   started_unix=time.time(), model_path=str(args.model_path),
                   arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   paged_backend=args.paged_backend,
                   torch_version=torch.__version__, transformers_version=transformers.__version__,
                   gpu_name=torch.cuda.get_device_name(), gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                   source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), HERE / "segpaged_qwen3_throughput_adapter.py",
                       ROOT / "python/sglang/srt/mem_cache/head_kv/pool.py",
                       ROOT / "python/sglang/srt/mem_cache/head_kv/attention.py",
                       ROOT / "python/sglang/srt/mem_cache/head_kv/triton_attention.py"]},
                   model_config_sha256=sha(args.model_path / "config.json"), cohorts=[],
                   scope=["Real pretrained model, real batched greedy generation using paged-attention logits",
                          "Identical-position shared prefix, not non-prefix RedKnot or MLA semantic validation",
                          "Full attention in every arm; ratio describes physical storage, not head-local approximation",
                          "Experimental single-stream batched append adapter, not production HeadKVManager/SGLang scheduler",
                          "Main QPS includes admission, question, fixed output and cleanup; excludes offline document prefill",
                          "Synchronous request waves, not continuous-batching or network serving benchmark",
                          "GPU offline source cache charged separately; identical fixed online pool budget across arms",
                          "Context may be repeated/truncated; no LongBench accuracy/generalization claim",
                          "1.0-sharing is a full-prefix cache reference, not a unique RedKnot advantage"])
    save(args.output, payload)
    if args.paged_backend == "flash":
        payload["source_hashes"][str((HERE / "segpaged_qwen3_flash_backend.py").relative_to(ROOT))] = sha(HERE / "segpaged_qwen3_flash_backend.py")
    try:
        event("model_loading", model=str(args.model_path))
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        model = Qwen3ForCausalLM.from_pretrained(args.model_path, local_files_only=True,
                                                dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                                                device_map="cuda:0").eval()
        payload["model_dimensions"] = {name: getattr(model.config, name) for name in
                                       ["num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings"]}
        payload["model_parameter_bytes"] = sum(x.numel() * x.element_size() for x in model.parameters())
        for length in args.lengths:
            if length + args.question_tokens + max(args.output_tokens) > model.config.max_position_embeddings:
                raise ValueError("requested sequence exceeds declared model context length")
            for dataset in args.datasets:
                document, suffixes, metadata = prepare_workload(tokenizer, args.datasets_dir / f"{dataset}.jsonl",
                                                                args.row_index, length, args.max_batch, args.question_tokens)
                event("offline_prefill", dataset=dataset, length=length)
                source, offline_seconds = prefill(model, document, args.prefill_chunk_size)
                source_bytes = cache_bytes(source)
                gate = independent_model_gate(model, source, suffixes, length, args.page_size, paged_backend=args.paged_backend)
                if not gate["passed"]:
                    payload.setdefault("failed_independent_model_gates", []).append(dict(dataset=dataset, length=length, **gate))
                    profile_ok = (args.profile_after_logit_gate_failure and gate["all_first_logits_finite"]
                                  and gate["all_native_and_paged_gate_logits_finite"] and gate["first_logits_cosine"] >= .999
                                  and gate["source_cache_length_unchanged"] and gate["native_token_ids"] == gate["paged_token_ids"])
                    if not profile_ok:
                        if not args.continue_after_gate_failure:
                            raise RuntimeError("independent native model correctness gate failed")
                        payload["cohorts"].append(dict(**metadata, status="skipped_native_gate_failure",
                                                       independent_native_model_gate=gate,
                                                       planned_output_tokens=args.output_tokens,
                                                       offline_prefill_seconds=offline_seconds))
                        save(args.output, payload)
                        event("cohort_skipped_native_gate_failure", dataset=dataset, length=length)
                        del source, document, suffixes
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    event("diagnostic_throughput_only", reason="native logits gate remains failed; identical 16-token continuation permits storage ablation only")
                for output_tokens in args.output_tokens:
                    configs = configurations(model, length, args, output_tokens)
                    cohort = dict(**metadata, output_tokens=output_tokens, offline_prefill_seconds=offline_seconds,
                                  common_gpu_source_cache_bytes=source_bytes, configurations=configs,
                                  independent_native_model_gate=gate,
                                  paged_backend=args.paged_backend,
                                  numerical_validation_status="passed" if gate["passed"] else "diagnostic_only_native_logit_gate_failed",
                                  online_pool_reserved_bytes=configs[0]["capacity_pages"] * 2 * args.page_size * model.config.head_dim * 2,
                                  samples=[], warmup_records=[], status="running")
                    cohort["total_source_plus_pool_reserved_kv_bytes"] = source_bytes + cohort["online_pool_reserved_bytes"]
                    payload["cohorts"].append(cohort)
                    for round_index in range(args.warmup_rounds + args.repeats):
                        warmup = round_index < args.warmup_rounds
                        # Alternate complete forward/reverse order. Never discard
                        # a slow measured wave or select the best run.
                        order = configs if round_index % 2 == 0 else list(reversed(configs))
                        for config in order:
                            event("wave_start", dataset=dataset, length=length, output_tokens=output_tokens,
                                  warmup=warmup, round=round_index, ratio=config["shared_ratio"], batch=config["batch_size"])
                            record = run_wave(model, tokenizer, source, length, suffixes, config, output_tokens, args.page_size, args.paged_backend)
                            record.update(round=round_index, warmup=warmup)
                            (cohort["warmup_records"] if warmup else cohort["samples"]).append(record)
                            save(args.output, payload)
                            event("wave_complete", qps=record["qps"], resident_qps=record["resident_generation_qps"],
                                  ttft=record["admission_inclusive_ttft_seconds"], admission=record["admission_seconds"])
                    cohort["summary"] = summarize(cohort["samples"])
                    for summary in cohort["summary"]:
                        summary["qualification"] = cohort["numerical_validation_status"]
                    if not all(row["all_generated_logits_finite"] for row in cohort["samples"]):
                        raise RuntimeError("nonfinite model logits; throughput cohort is invalid")
                    if not all(row.get("same_batch_baseline_token_identity", True) for row in cohort["summary"]):
                        raise RuntimeError("same-batch storage sharing changed generated tokens; cohort invalid")
                    cohort["status"] = "complete"
                    save(args.output, payload)
                del source, document, suffixes
                gc.collect()
                torch.cuda.empty_cache()
        skipped = sum(row["status"] == "skipped_native_gate_failure" for row in payload["cohorts"])
        payload["skipped_cohorts"] = skipped
        payload["status"] = "complete_with_skipped_cohorts" if skipped else "complete"
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        payload["finished_unix"] = time.time()
        save(args.output, payload)
    return 0 if payload["status"] in ("complete", "complete_with_skipped_cohorts") else 1


if __name__ == "__main__":
    raise SystemExit(main())
