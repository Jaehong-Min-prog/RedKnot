#!/usr/bin/env python3
"""Native HF FA2 control for experimental SegPaged model QPS measurements.

The document is prefilled once, as in benchmark_segpaged_qps.py. Every timed
wave deep-copies that cache and repeats its batch dimension to private per-
request caches, processes different questions, and greedily generates a fixed
number of tokens. Admission, generation and cache cleanup are timed separately
and together. No custom attention implementation or adapter is installed.

This is a same-concurrency native Transformers reference, NOT a fixed-page-pool
budget comparison or production continuous-batching server. DynamicCache grows
by concatenation and its transient workspace/allocations differ from the paged
adapter. Source-cache bytes, nominal active KV, and actual allocator peaks are
reported separately. Offline source construction is excluded in both protocols.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
import transformers
from transformers import AutoTokenizer, Qwen3ForCausalLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmark_segpaged_qps import (  # noqa: E402
    cache_bytes, event, prefill, prepare_workload, quantile, save, sha, sync,
)


@torch.inference_mode()
def run_native_wave(model, tokenizer, source, question, output_tokens):
    """Run genuine batched HF generation; CPU export is outside timings."""
    batch = question.shape[0]
    source_length = source.get_seq_length()
    generated, finite_flags = [], []
    gc.collect()
    sync()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    started = time.perf_counter()
    private = copy.deepcopy(source)
    if batch > 1:
        private.batch_repeat_interleave(batch)
    sync()
    admitted = time.perf_counter()
    result = model(input_ids=question, past_key_values=private, use_cache=True, logits_to_keep=1)
    finite_flags.append(torch.isfinite(result.logits).all())
    token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated.append(token)
    sync()
    first_token = time.perf_counter()
    for _ in range(output_tokens - 1):
        result = model(input_ids=token, past_key_values=private, use_cache=True, logits_to_keep=1)
        finite_flags.append(torch.isfinite(result.logits).all())
        token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token)
    sync()
    finished_generation = time.perf_counter()
    actual_cache_bytes = cache_bytes(private)
    final_cache_length = private.get_seq_length()
    peak = torch.cuda.max_memory_allocated()
    del result, private
    sync()
    finished = time.perf_counter()
    ids = torch.cat(generated, dim=1).cpu().tolist()
    elapsed = finished - started
    resident = finished_generation - admitted
    expected_length = source_length + question.shape[1] + output_tokens - 1
    if final_cache_length != expected_length:
        raise AssertionError("native cache sequence length does not match generated work")
    eos = tokenizer.eos_token_id
    return dict(batch_size=batch, output_tokens_per_request=output_tokens,
                completed_requests=batch, completed_output_tokens=batch * output_tokens,
                admission_seconds=admitted - started,
                resident_ttft_seconds=first_token - admitted,
                admission_inclusive_ttft_seconds=first_token - started,
                resident_generation_seconds=resident,
                cleanup_seconds=finished - finished_generation,
                admission_inclusive_seconds=elapsed, qps=batch / elapsed,
                resident_generation_qps=batch / resident,
                output_tokens_per_second=batch * output_tokens / elapsed,
                gpu_allocated_before_wave_bytes=allocated_before,
                gpu_peak_allocated_bytes=peak,
                actual_final_private_kv_bytes=actual_cache_bytes,
                private_cache_sequence_tokens=final_cache_length,
                source_cache_length_unchanged=source.get_seq_length() == source_length,
                generated_token_ids=ids,
                generated_text=[tokenizer.decode(row, skip_special_tokens=False) for row in ids],
                eos_encountered_requests=sum(eos in row for row in ids),
                all_generated_logits_finite=bool(torch.stack(finite_flags).all()),
                generation_policy="greedy, fixed output count, EOS ignored",
                model_attention_backend="unmodified native flash_attention_2",
                cache_implementation="Transformers DynamicCache with private repeated-batch copies")


def summarize(samples):
    summary = dict(repeats=len(samples), batch_size=samples[0]["batch_size"],
                   aggregate_qps=sum(x["completed_requests"] for x in samples) /
                                 sum(x["admission_inclusive_seconds"] for x in samples),
                   aggregate_resident_qps=sum(x["completed_requests"] for x in samples) /
                                          sum(x["resident_generation_seconds"] for x in samples),
                   repeat_output_identity=all(x["generated_token_ids"] == samples[0]["generated_token_ids"]
                                              for x in samples),
                   gpu_peak_allocated_bytes_max=max(x["gpu_peak_allocated_bytes"] for x in samples))
    for name in ("qps", "resident_generation_qps", "admission_seconds", "resident_ttft_seconds",
                 "admission_inclusive_ttft_seconds", "admission_inclusive_seconds", "output_tokens_per_second"):
        values = [x[name] for x in samples]
        summary.update({name + "_median": statistics.median(values),
                        name + "_p95_sample_quantile": quantile(values, .95),
                        name + "_min": min(values), name + "_max": max(values)})
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--datasets-dir", type=Path, default=HERE / "datasets/LongBench/data")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimqa", "multifieldqa_en"])
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 16384, 31744])
    parser.add_argument("--output-tokens", nargs="+", type=int, default=[32])
    parser.add_argument("--question-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite native reference results")
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("native reference requires transformers==4.57.1")
    if min(args.lengths + args.output_tokens + [args.question_tokens, args.prefill_chunk_size,
                                               args.batch_size, args.warmup_rounds, args.repeats]) <= 0:
        raise ValueError("all workload dimensions, warmups and repeats must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(20261015)
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = dict(schema="segpaged-native-fa2-model-reference-v1", status="running",
                   started_unix=time.time(), arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   torch_version=torch.__version__, transformers_version=transformers.__version__,
                   gpu_name=torch.cuda.get_device_name(),
                   gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                   model_config_sha256=sha(args.model_path / "config.json"),
                   source_hashes={p.name: sha(p) for p in [Path(__file__), HERE / "benchmark_segpaged_qps.py"]},
                   cohorts=[],
                   scope=["Unmodified native Qwen3 FA2 with genuine batched model generation",
                          "Same input preparation and fixed-output protocol as experimental paged suite",
                          "Same-concurrency native control, NOT equal bounded-page-pool budget proof",
                          "DynamicCache growing concatenation and transient workspace included in allocator peak",
                          "Timing includes fresh private cache admission, question, generation and cleanup",
                          "Offline document prefill excluded; common source cache retained and counted separately",
                          "Synchronous request waves, not production continuous-batching or network-serving QPS",
                          "Repeated/truncated document systems workload, not LongBench task accuracy"])
    save(args.output, payload)
    try:
        event("native_reference_model_loading", model=str(args.model_path))
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
                document, questions, metadata = prepare_workload(tokenizer, args.datasets_dir / f"{dataset}.jsonl",
                                                                 args.row_index, length, args.batch_size, args.question_tokens)
                source, offline_seconds = prefill(model, document, args.prefill_chunk_size)
                source_bytes = cache_bytes(source)
                for output_tokens in args.output_tokens:
                    cohort = dict(**metadata, output_tokens=output_tokens, batch_size=args.batch_size,
                                  common_gpu_source_cache_bytes=source_bytes,
                                  offline_prefill_seconds=offline_seconds,
                                  nominal_final_private_kv_bytes=args.batch_size * (length + args.question_tokens + output_tokens - 1) *
                                                                model.config.num_hidden_layers * model.config.num_key_value_heads *
                                                                model.config.head_dim * 2 * next(model.parameters()).element_size(),
                                  fixed_pool_budget_enforced=False, samples=[], warmup_records=[], status="running")
                    payload["cohorts"].append(cohort)
                    for repeat in range(args.warmup_rounds + args.repeats):
                        warmup = repeat < args.warmup_rounds
                        event("native_wave_start", dataset=dataset, length=length, batch=args.batch_size,
                              output_tokens=output_tokens, warmup=warmup, round=repeat)
                        row = run_native_wave(model, tokenizer, source, questions, output_tokens)
                        row.update(round=repeat, warmup=warmup)
                        (cohort["warmup_records"] if warmup else cohort["samples"]).append(row)
                        save(args.output, payload)
                        event("native_wave_complete", qps=row["qps"], resident_qps=row["resident_generation_qps"],
                              ttft=row["admission_inclusive_ttft_seconds"], admission=row["admission_seconds"])
                    if not all(x["all_generated_logits_finite"] and x["source_cache_length_unchanged"] for x in cohort["samples"]):
                        raise RuntimeError("native reference finite/source-length check failed")
                    if not all(x["actual_final_private_kv_bytes"] == cohort["nominal_final_private_kv_bytes"] for x in cohort["samples"]):
                        raise RuntimeError("actual native KV bytes differ from nominal full-head cache count")
                    cohort["summary"] = summarize(cohort["samples"])
                    cohort["status"] = "complete"
                    save(args.output, payload)
                del source, document, questions
                gc.collect()
                torch.cuda.empty_cache()
        payload["status"] = "complete"
    except Exception as exc:
        payload.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        payload["finished_unix"] = time.time()
        save(args.output, payload)
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
