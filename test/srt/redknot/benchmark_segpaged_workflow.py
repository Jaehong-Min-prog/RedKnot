#!/usr/bin/env python3
"""Real Qwen3 RedKnot workflow A/B, not an attention-only speedup benchmark.

Both arms execute the existing offline segments -> online global-head recovery
-> local/global query -> original full-attention generation workflow. Only the
query attention storage/dispatch changes. The legacy query logits are diagnostic
and do not drive generated text; this distinction is recorded explicitly.
All manager setup/copies/teardown are online. Offline prefill is separately timed.
The workload is batch one, synchronous, in process: reciprocal request latency
is serial requests/s, NOT a concurrent serving-QPS measurement.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
import transformers

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def persist(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sync():
    torch.cuda.synchronize()


def assert_original_backend(model):
    implementations = {model.config._attn_implementation}
    implementations.update(layer.self_attn.config._attn_implementation for layer in model.model.layers)
    if implementations != {"flash_attention_2"}:
        raise AssertionError(f"Attention implementation leaked between arms: {implementations}")


def select_rows(path, tokenizer, count, context_tokens, chunks, start_index):
    """Predeclared first eligible source rows; no output-based selection."""
    selected = []
    for index, line in enumerate(path.read_text().splitlines()):
        if index < start_index:
            continue
        row = json.loads(line)
        context, question = str(row.get("context", "")), str(row.get("input", ""))
        if not context or not question:
            continue
        ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        if len(ids) < context_tokens:
            continue
        ids = ids[:context_tokens]
        docs = [tokenizer.decode(ids[i * context_tokens // chunks:(i + 1) * context_tokens // chunks],
                                 skip_special_tokens=False) for i in range(chunks)]
        selected.append(dict(source_index=index, question=question, answers=row.get("answers", []),
                             docs=docs, original_context_tokens=len(tokenizer(context, add_special_tokens=False)["input_ids"]),
                             truncated_context=True, source_row_sha256=hashlib.sha256(line.encode()).hexdigest()))
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"{path.name}: only {len(selected)} eligible rows of >= {context_tokens} tokens")
    return selected


@torch.inference_mode()
def recomputed(model, tokenizer, input_ids, max_new_tokens, chunk_size):
    """Full recomputation, exact same input IDs, without any reused prefix."""
    sync()
    started = time.perf_counter()
    past = None
    for offset in range(0, input_ids.shape[1], chunk_size):
        out = model(input_ids=input_ids[:, offset:offset + chunk_size], past_key_values=past,
                    use_cache=True, logits_to_keep=1)
        past = out.past_key_values
    token = out.logits[0, -1].argmax().view(1, 1)
    generated = [int(token.item())]
    sync()
    ttft = time.perf_counter() - started
    for _ in range(max_new_tokens - 1):
        if generated[-1] == tokenizer.eos_token_id:
            break
        out = model(input_ids=token, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        token = out.logits[0, -1].argmax().view(1, 1)
        generated.append(int(token.item()))
    sync()
    generation_seconds = time.perf_counter() - started
    text = tokenizer.decode(generated, skip_special_tokens=True)
    del out, past
    gc.collect()
    torch.cuda.empty_cache()
    sync()
    return dict(ttft_seconds=ttft, generation_seconds=generation_seconds,
                request_seconds=time.perf_counter() - started,
                generated_token_ids=generated, output=text)


def summary(records):
    result = {}
    for arm in sorted({r["arm"] for r in records}):
        values = [r for r in records if r["arm"] == arm and not r["warmup"]]
        if not values:
            continue
        entry = dict(samples=len(values))
        for key in ("ttft_seconds", "request_seconds"):
            numbers = [r[key] for r in values]
            entry[key] = dict(median=statistics.median(numbers), minimum=min(numbers), maximum=max(numbers), raw=numbers)
        entry["serial_requests_per_second"] = len(values) / sum(r["request_seconds"] for r in values)
        for key in values[0].get("phases", {}):
            numbers = [r["phases"].get(key) for r in values]
            if key.endswith("seconds") and all(isinstance(x, (float, int)) for x in numbers):
                entry.setdefault("phase_median_seconds", {})[key] = statistics.median(numbers)
        result[arm] = entry
    if "native" in result and "managed_paged" in result:
        result["managed_vs_native"] = dict(
            ttft_speedup=result["native"]["ttft_seconds"]["median"] / result["managed_paged"]["ttft_seconds"]["median"],
            serial_request_rate_ratio=result["managed_paged"]["serial_requests_per_second"] / result["native"]["serial_requests_per_second"],
        )
    return result


@torch.inference_mode()
def run(args, report):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot import (
        HeadClassConfig, SparseFFNSchedule, get_global_offline_cache,
        offline_prefill_segments, run_redknot_offlinekv,
    )
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("This workflow experiment pins transformers==4.57.1")
    if not torch.cuda.is_available():
        raise RuntimeError("Real GPU benchmark requested; no synthetic timing fallback")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    head_config = HeadClassConfig.from_json(args.head_config)
    head_config.merge_retrieval_to_global()
    ffn_config = json.loads(args.ffn_config.read_text())
    schedule = SparseFFNSchedule(**ffn_config)
    cases = []
    for dataset in args.datasets:
        path = args.dataset_dir / f"{dataset}.jsonl"
        for row in select_rows(path, tokenizer, args.samples, args.context_tokens, args.chunks, args.start_index):
            cases.append(dict(dataset=dataset, dataset_file_sha256=digest(path), **row))
    started = time.perf_counter()
    print(json.dumps(dict(event="model_loading", path=args.model_path)), flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="flash_attention_2").eval()
    sync()
    if model.config.num_hidden_layers != head_config.num_layers or model.config.num_key_value_heads != head_config.num_kv_heads:
        raise ValueError("Head configuration does not match actual model topology")
    report.update(model_load_seconds=time.perf_counter() - started, model_config=model.config.to_dict(),
                  model_config_sha256=digest(Path(args.model_path) / "config.json"),
                  head_summary=head_config.summary(), ffn_config=ffn_config,
                  head_config_sha256=digest(args.head_config), ffn_config_sha256=digest(args.ffn_config),
                  gpu=dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                           total_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
                  torch_version=torch.__version__, transformers_version=transformers.__version__)
    print(json.dumps(dict(event="model_ready", load_seconds=report["model_load_seconds"], heads=report["head_summary"])), flush=True)
    arms = ["native", "managed_paged"] + (["managed_no_skip"] if args.skip_control else [])
    for case_index, case in enumerate(cases):
        assert_original_backend(model)
        print(json.dumps(dict(event="offline_start", case=case_index, dataset=case["dataset"], source_index=case["source_index"])), flush=True)
        sync()
        started = time.perf_counter()
        segments = offline_prefill_segments(model, tokenizer, case["docs"], chunk_size=args.prefill_chunk,
                                            model_id=args.model_path)
        sync()
        offline_seconds = time.perf_counter() - started
        query = "\n\nAnswer the question based on the passages above. Give the shortest exact answer span with no explanation.\nQuestion: " + case["question"] + "\nAnswer:"
        query_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
        all_ids = torch.cat([s.token_ids.to(model.device) for s in segments] +
                            [torch.tensor(query_ids, device=model.device)]).unsqueeze(0)
        if all_ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
            raise ValueError("Actual retokenized prompt + output exceeds native context")
        record = {k: v for k, v in case.items() if k != "docs"}
        record.update(query_text=query, offline_seconds=offline_seconds, segment_lengths=[s.doc_len for s in segments],
                      prompt_tokens=int(all_ids.shape[1]), query_tokens=len(query_ids),
                      prompt_token_sha256=hashlib.sha256(json.dumps(all_ids[0].tolist()).encode()).hexdigest(),
                      runs=[])
        report["cases"].append(record)
        reference_logits = None
        reference_text = None
        # Warm every arm at the full requested shape and output length. Measured
        # cycles alternate arm order to reduce monotonic drift bias.
        for cycle in range(args.warmup + args.repeats):
            warmup = cycle < args.warmup
            order = arms if cycle % 2 == 0 else list(reversed(arms))
            if args.recomputed:
                order = (["recomputed"] + order) if cycle % 2 == 0 else (order + ["recomputed"])
            for arm in order:
                assert_original_backend(model)
                torch.cuda.reset_peak_memory_stats()
                print(json.dumps(dict(event="run_start", case=case_index, cycle=cycle, arm=arm, warmup=warmup)), flush=True)
                if arm == "recomputed":
                    value = recomputed(model, tokenizer, all_ids, args.max_new_tokens, args.prefill_chunk)
                else:
                    phases, sparse_stats = {}, []
                    sync()
                    started = time.perf_counter()
                    logits, text_out, _, ttft = run_redknot_offlinekv(model, tokenizer,
                        segments_offline=segments, query_text=query, head_cfg=head_config,
                        max_new_tokens=args.max_new_tokens, sparse_ffn_schedule=schedule,
                        sparse_ffn_stats=sparse_stats, use_compile=False,
                        query_backend="native" if arm == "native" else "managed_paged",
                        paged_skip_invisible=arm != "managed_no_skip", timing_stats=phases)
                    sync()
                    request_seconds = time.perf_counter() - started
                    value = dict(ttft_seconds=ttft, request_seconds=request_seconds, phases=phases,
                                 output=text_out, sparse_ffn_stats=sparse_stats)
                    cpu_logits = logits.detach().float().cpu()
                    del logits
                    if arm == "native" and reference_logits is None:
                        reference_logits, reference_text = cpu_logits, text_out
                    if reference_logits is not None:
                        error = (cpu_logits - reference_logits).abs()
                        value["diagnostic_query_logits"] = dict(max_abs_error=float(error.max()),
                            mean_abs_error=float(error.mean()), argmax_match=int(cpu_logits.argmax()) == int(reference_logits.argmax()),
                            cosine=float(torch.nn.functional.cosine_similarity(cpu_logits, reference_logits, dim=0)),
                            all_finite=bool(torch.isfinite(cpu_logits).all()))
                        value["output_matches_native"] = text_out == reference_text
                value.update(arm=arm, cycle=cycle, warmup=warmup, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
                assert_original_backend(model)
                value["backend_before_and_after"] = "flash_attention_2"
                record["runs"].append(value)
                record["summary"] = summary(record["runs"])
                persist(args.output, report)
                print(json.dumps(dict(event="run_done", case=case_index, arm=arm, ttft_seconds=value["ttft_seconds"],
                                      request_seconds=value["request_seconds"], output=value["output"])), flush=True)
        del segments, all_ids
        get_global_offline_cache().clear()
        gc.collect()
        torch.cuda.empty_cache()
    report["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", default=os.environ.get("REDKNOT_MODEL_PATH"),
        help="Local Qwen3 checkpoint directory (or set REDKNOT_MODEL_PATH)",
    )
    parser.add_argument("--head-config", type=Path, default=HERE / "head_class/qwen3-32B_optimal_g15_lf_ret.json")
    parser.add_argument("--ffn-config", type=Path, default=HERE / "sparse_ffn_params/qwen3-32B.json")
    parser.add_argument("--dataset-dir", type=Path, default=HERE / "datasets/LongBench/data")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimqa", "multifieldqa_en"])
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--recomputed", action="store_true")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.model_path or not args.model_path.strip():
        parser.error("provide --model-path or set REDKNOT_MODEL_PATH to a local Qwen3 checkpoint directory")
    if min(args.samples, args.context_tokens, args.chunks, args.max_new_tokens, args.prefill_chunk, args.warmup, args.repeats) <= 0:
        parser.error("sample/length/chunk/warmup/repeat counts must be positive")
    source_files = [Path(__file__)]
    source_files += list((ROOT / "python/sglang/srt/layers/attention/redknot").glob("*.py"))
    source_files += list((ROOT / "python/sglang/srt/mem_cache/head_kv").glob("*.py"))
    report = dict(schema="redknot-segpaged-workflow-ab-v1", status="running", cases=[],
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_files},
        scope="Real pretrained BF16 Qwen3, existing RedKnot HF RAG workflow, single GPU batch one; optional managed query-stage integration, not SGLang scheduler/MLA integration.",
        timing="Warm actual emitted-first-token latency including all online cache preparation and full-attention seed; offline document prefill and model load excluded and reported separately. Full request wall time includes teardown. No network queueing or concurrent-QPS claim.",
        quality_caveat="Legacy head-class query logits are diagnostic; generated text comes from the subsequent full-attention seed. Text equality alone cannot validate the new query backend. Truncated real contexts are system-test inputs, not a benchmark-quality score.")
    try:
        run(args, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    persist(args.output, report)
    print(json.dumps(dict(event="complete", status=report["status"], output=str(args.output))), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
