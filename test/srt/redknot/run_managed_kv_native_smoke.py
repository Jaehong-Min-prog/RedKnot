#!/usr/bin/env python3
"""Real single-GPU native Scheduler lifecycle smoke; NOT a QPS/quality benchmark.

Run under an authenticated GPU supervisor with exactly one allocated visible
GPU and no holder on that GPU. This script never touches holders or other
servers. The child stays in the supervisor's process group. All artifacts go
to a new directory; failures retain logs and a failed receipt. No test stubs or
replacement SGLang modules are used by this executable.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


def source_root():
    return Path(__file__).resolve().parents[3]


def select_loopback_port():
    """Choose an available HTTP port with room for native derived ports.

    ServerArgs validates HTTP+10000 as its default gRPC port even when gRPC is
    disabled. OS-assigned ephemeral ports can exceed 55535; do not use bind(0).
    The reservation is released immediately before Popen. The usual bind race
    remains fail-closed (unique API key plus server-exit/readiness validation).
    """
    for _ in range(64):
        candidate = 20000 + uuid.uuid4().int % 30000
        with socket.socket() as reservation:
            try:
                reservation.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError("could not reserve a safe localhost HTTP port after 64 attempts")


def configure_native_ports(env, http_port):
    if type(http_port) is not int or not 1 <= http_port <= 55535:
        raise ValueError("HTTP port must leave room for native gRPC +10000 offset")
    # Override ambient values as well: disabled gRPC still undergoes validation.
    env.update(SGLANG_ENABLE_GRPC="0", SGLANG_GRPC_PORT=str(http_port + 10000))


def validate_inputs(args):
    output = Path(args.output)
    model = Path(args.model_path)
    if not output.is_absolute() or output.exists():
        raise ValueError("--output must be an absolute, new directory")
    if not model.is_absolute() or not (model / "config.json").is_file():
        raise ValueError("--model-path must be an existing absolute local checkpoint")
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("quantization_config"):
        raise ValueError("only unquantized local Qwen3 is supported by this smoke")
    if not 1 <= args.startup_timeout <= 600 or not 1 <= args.execution_timeout <= 300:
        raise ValueError("startup must be <=600s and execution <=300s")
    if not 0.1 <= args.mem_fraction_static <= 0.9:
        raise ValueError("mem fraction must be in [0.1, 0.9]")
    if not (source_root() / "python/sglang/srt/mem_cache/head_kv/production.py").is_file():
        raise ValueError("run this executable inside the complete frozen source tree")
    return output, model


def build_command(model, port, key, fraction):
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", str(model),
            "--host", "127.0.0.1", "--port", str(port), "--api-key", key,
            "--attention-backend", "redknot", "--redknot-managed-kv",
            "--redknot-page-size", "64", "--disable-radix-cache",
            "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
            "--disable-overlap-schedule", "--disable-custom-all-reduce",
            "--page-size", "1", "--tp-size", "1", "--max-running-requests", "4",
            "--chunked-prefill-size", "512", "--max-prefill-tokens", "8192",
            "--max-total-tokens", "16384", "--context-length", "8192",
            "--mem-fraction-static", str(fraction), "--dtype", "bfloat16",
            "--sampling-backend", "pytorch", "--skip-server-warmup"]


def extract_stats(info):
    states = info.get("internal_states")
    if not isinstance(states, list) or len(states) != 1:
        raise RuntimeError("expected one native Scheduler internal state")
    stats = states[0].get("redknot_managed_kv_stats")
    if not isinstance(stats, dict) or stats.get("mode") != "experimental_single_gpu_live_only":
        raise RuntimeError("native managed KV lifecycle was not enabled")
    if stats.get("nonprefix_import") is not False or stats.get("dense_kv_slab") is not False:
        raise RuntimeError("unexpected managed serving mode")
    if stats.get("poisoned"):
        raise RuntimeError("native physical KV pool is poisoned")
    return stats


def validate_idle(stats, *, minimum_created=0):
    for key in ("live_serving_requests", "requests", "cached_roots", "pending_transactions",
                "live_pages", "reserved_pages", "retiring_pages", "pinned_pages"):
        if stats.get(key) != 0:
            raise RuntimeError(f"native idle lifecycle leak: {key}={stats.get(key)!r}")
    if stats.get("free_pages") != stats.get("capacity_pages"):
        raise RuntimeError("native idle pool has unreclaimed physical pages")
    created, released = stats.get("created_serving_requests"), stats.get("released_serving_requests")
    if not isinstance(created, int) or created < minimum_created or created != released:
        raise RuntimeError("native created/released lifecycle counts disagree")


def validate_completion(result, *, chunked=False):
    if not isinstance(result, dict) or not isinstance(result.get("text"), str) or not result["text"]:
        raise RuntimeError("native request returned no output text")
    meta = result.get("meta_info", {})
    if meta.get("completion_tokens", 0) < 1 or meta.get("finish_reason", {}).get("type") == "abort":
        raise RuntimeError("native request did not complete generation")
    if chunked and meta.get("prompt_tokens", 0) <= 512:
        raise RuntimeError("chunked smoke did not actually exceed the configured prefill chunk")


def http(base, key, path, *, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data=data,
                                    headers={"Authorization": "Bearer " + key,
                                             "Content-Type": "application/json"})
    # Never send loopback requests through ambient corporate HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        raw = response.read().decode()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"status": response.status, "body": raw}


def remaining(deadline, cap=10):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("native smoke execution deadline exceeded")
    return min(cap, value)


def stop_child(process):
    """Only this Popen and descendants captured from its actual process tree."""
    if process is None:
        return []
    import psutil
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    if process.poll() is None:
        process.terminate()
    for child in descendants:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(descendants, timeout=10)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    _, alive = psutil.wait_procs(alive, timeout=5)
    return [p.pid for p in alive]


def finalize_run(receipt, persist, process, executor, previous, output):
    """A second termination signal must not interrupt bounded child cleanup.

    Restore pre-run handlers even if cleanup or receipt persistence raises. A
    cleanup exception must never leave an earlier successful receipt in place.
    The outer supervisor still owns the final process-group cleanup deadline.
    """
    failures = []

    def failed(stage, exc):
        failures.append(exc)
        receipt["status"] = "failed"
        receipt.setdefault("cleanup_errors", []).append(
            dict(stage=stage, type=type(exc).__name__, message=str(exc)))

    try:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            survivors = stop_child(process)
            receipt["cleanup_surviving_child_pids"] = survivors
            receipt["cleanup_verified"] = not survivors
            if survivors:
                failed("server_process_tree", RuntimeError("own child processes survived cleanup"))
        except BaseException as exc:
            receipt["cleanup_verified"] = False
            receipt["cleanup_surviving_child_pids"] = None  # unknown, not an empty success
            failed("server_process_tree", exc)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except BaseException as exc:
                failed("http_executor", exc)
        receipt["finished_unix"] = time.time()
        persist()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json")}), flush=True)
    if failures:
        raise RuntimeError("native smoke cleanup failed; see receipt; outer supervisor must reclaim") from failures[0]


def run(args):
    output, model = validate_inputs(args)
    output.mkdir(parents=True, exist_ok=False)
    root = source_root()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", SGLANG_ENABLE_GRPC="0")
    env["PYTHONPATH"] = str(root / "python") + os.pathsep + env.get("PYTHONPATH", "")
    receipt = dict(schema="redknot-native-managed-smoke-v1", status="running",
                   scope="native_single_gpu_live_kv_lifecycle_only", nonprefix_qps_claim=False,
                   quality_threshold_claim=False, started_unix=time.time(), stages=[],
                   model_path=str(model), config_sha256=hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
                   source_root=str(root), allocated_cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"))
    process = None
    executor = None

    def persist():
        temporary = output / "receipt.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(output / "receipt.json")

    def interrupted(signum, frame):
        raise InterruptedError(f"native smoke interrupted by signal {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        persist()
        startup_deadline = time.monotonic() + args.startup_timeout
        probe_code = (
            "import json,pathlib,sys,torch,transformers,sglang; "
            "expected=pathlib.Path(sys.argv[1]).resolve(); "
            "assert pathlib.Path(sglang.__file__).resolve().is_relative_to(expected); "
            "assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'allocate exactly one GPU'; "
            "x=torch.empty(1,device='cuda'); torch.cuda.synchronize(); "
            "print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,"
            "transformers=transformers.__version__,sglang_file=sglang.__file__,"
            "gpu=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0))))"
        )
        with (output / "gpu_import_probe.log").open("wb") as log:
            probe = subprocess.run([sys.executable, "-c", probe_code, str(root / "python")],
                                   cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=remaining(startup_deadline, 120))
        receipt["stages"].append(dict(name="real_gpu_import", returncode=probe.returncode))
        persist()
        if probe.returncode:
            raise RuntimeError("real GPU/source import probe failed; see gpu_import_probe.log")
        if args.probe_only:
            with (output / "gpu_help.log").open("wb") as log:
                help_run = subprocess.run([sys.executable, "-m", "sglang.launch_server", "--help"],
                                          cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=remaining(startup_deadline, 120))
            receipt["stages"].append(dict(name="gpu_visible_help", returncode=help_run.returncode))
            if help_run.returncode:
                raise RuntimeError("GPU-visible native --help failed; see gpu_help.log")
            receipt["status"] = "probe_passed_not_model_tested"
            return 0

        port = select_loopback_port()
        configure_native_ports(env, port)
        receipt["ports"] = dict(http=port, grpc_validated_but_disabled=int(env["SGLANG_GRPC_PORT"]))
        key = "native-smoke-" + uuid.uuid4().hex
        command = build_command(model, port, key, args.mem_fraction_static)
        receipt["command"] = ["<ephemeral-local-key>" if x == key else x for x in command]
        base = f"http://127.0.0.1:{port}"
        with (output / "server.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt["server_pid"] = process.pid
        persist()
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"native server exited {process.returncode} before readiness")
            try:
                initial = extract_stats(http(base, key, "/server_info", timeout=remaining(startup_deadline, 2)))
                validate_idle(initial)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                remaining(startup_deadline)
                time.sleep(0.5)
        receipt["initial_stats"] = initial
        receipt["stages"].append(dict(name="native_server_ready", elapsed_s=args.startup_timeout - remaining(startup_deadline, args.startup_timeout)))
        persist()
        deadline = time.monotonic() + args.execution_timeout
        prompts = ["What is two plus two? Answer briefly.",
                   "Name two colors in a clear sentence.",
                   "Read these records and summarize their main topic.\n" +
                   "".join(f"Record {i}: Immutable local heads can share cached attention pages safely.\n" for i in range(180)) +
                   "\nGive a one-sentence summary."]
        payloads = [dict(rid="native-smoke-" + uuid.uuid4().hex, text=prompt,
                         sampling_params=dict(temperature=0, max_new_tokens=16, ignore_eos=True))
                    for prompt in prompts]
        (output / "requests.json").write_text(json.dumps(payloads, indent=2) + "\n")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        start = time.monotonic()
        futures = [executor.submit(http, base, key, "/generate", payload=payload,
                                   timeout=remaining(deadline, args.execution_timeout)) for payload in payloads]
        results = []
        for i, future in enumerate(futures):
            result = future.result(timeout=remaining(deadline, args.execution_timeout))
            results.append(result)
            (output / f"response_{i}.json").write_text(json.dumps(result, indent=2) + "\n")
            validate_completion(result, chunked=i == 2)
        receipt["requests_wall_s"] = time.monotonic() - start
        receipt["stages"].append(dict(name="three_concurrent_real_completions", count=3))
        http(base, key, "/flush_cache?timeout=10", payload={}, timeout=remaining(deadline, 15))
        completed = extract_stats(http(base, key, "/server_info", timeout=remaining(deadline)))
        validate_idle(completed, minimum_created=initial["created_serving_requests"] + 3)
        receipt["after_completions_stats"] = completed
        persist()

        abort_payload = dict(rid="native-abort-" + uuid.uuid4().hex,
                             text="Write a long sequence of consecutive numbers, starting from one.",
                             sampling_params=dict(temperature=0, max_new_tokens=1024, ignore_eos=True))
        (output / "abort_request.json").write_text(json.dumps(abort_payload, indent=2) + "\n")
        future = executor.submit(http, base, key, "/generate", payload=abort_payload,
                                 timeout=remaining(deadline, args.execution_timeout))
        while True:
            observed = extract_stats(http(base, key, "/server_info", timeout=remaining(deadline)))
            if observed["live_serving_requests"] > 0:
                receipt["active_before_abort_stats"] = observed
                break
            if future.done():
                raise RuntimeError("abort request finished before observing a live managed handle")
            time.sleep(0.02)
        http(base, key, "/abort_request", payload={"rid": abort_payload["rid"]}, timeout=remaining(deadline))
        abort_result = future.result(timeout=remaining(deadline, args.execution_timeout))
        (output / "abort_response.json").write_text(json.dumps(abort_result, indent=2) + "\n")
        if abort_result.get("meta_info", {}).get("finish_reason", {}).get("type") != "abort":
            raise RuntimeError("native abort endpoint did not actually abort the active request")
        http(base, key, "/flush_cache?timeout=10", payload={}, timeout=remaining(deadline, 15))
        final = extract_stats(http(base, key, "/server_info", timeout=remaining(deadline)))
        validate_idle(final, minimum_created=completed["created_serving_requests"] + 1)
        receipt["final_stats"] = final
        receipt["stages"].append(dict(name="active_abort_and_flush_reclamation", status="passed"))
        receipt["status"] = "passed_native_lifecycle_smoke"
        return 0
    except BaseException as exc:
        receipt.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        finalize_run(receipt, persist, process, executor, previous, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--execution-timeout", type=float, default=300)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
