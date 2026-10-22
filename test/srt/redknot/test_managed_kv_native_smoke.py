"""CPU-only receipt/command validation; these tests DO NOT run a native server."""
import json
from types import SimpleNamespace

import pytest
import run_managed_kv_native_smoke as smoke

from run_managed_kv_native_smoke import (
    build_command, extract_stats, validate_completion, validate_idle, validate_inputs,
)


def healthy():
    return dict(mode="experimental_single_gpu_live_only", nonprefix_import=False, dense_kv_slab=False,
                poisoned=False, live_serving_requests=0, requests=0, cached_roots=0,
                pending_transactions=0, live_pages=0, reserved_pages=0, retiring_pages=0,
                pinned_pages=0, free_pages=100, capacity_pages=100,
                created_serving_requests=4, released_serving_requests=4)


def test_native_flags_fail_closed_and_bind_loopback():
    cmd = build_command("/local/qwen", 12345, "secret", 0.8)
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--tp-size") + 1] == "1"
    assert cmd[cmd.index("--attention-backend") + 1] == "redknot"
    for flag in ("--redknot-managed-kv", "--disable-radix-cache", "--disable-cuda-graph",
                 "--disable-piecewise-cuda-graph", "--disable-overlap-schedule"):
        assert flag in cmd


@pytest.mark.parametrize("random_value,expected", [(0, 20000), (29999, 49999), (30000, 20000)])
def test_port_selection_is_loopback_and_leaves_derived_port_room(monkeypatch, random_value, expected):
    binds, closes = [], []

    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closes.append(True)
        def bind(self, address):
            binds.append(address)

    monkeypatch.setattr(smoke.socket, "socket", Socket)
    monkeypatch.setattr(smoke.uuid, "uuid4", lambda: SimpleNamespace(int=random_value))
    assert smoke.select_loopback_port() == expected
    assert binds == [("127.0.0.1", expected)] and closes == [True]
    env = {"SGLANG_ENABLE_GRPC": "1", "SGLANG_GRPC_PORT": "99999"}
    smoke.configure_native_ports(env, expected)
    assert env == {"SGLANG_ENABLE_GRPC": "0", "SGLANG_GRPC_PORT": str(expected + 10000)}
    assert 1 <= int(env["SGLANG_GRPC_PORT"]) <= 65535


@pytest.mark.parametrize("busy_count", [1, 64])
def test_busy_ports_retry_but_never_unbounded(monkeypatch, busy_count):
    attempts, closes = [], []

    class Socket:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closes.append(True)
        def bind(self, address):
            attempts.append(address)
            if len(attempts) <= busy_count:
                raise OSError("test address already in use")

    monkeypatch.setattr(smoke.socket, "socket", Socket)
    monkeypatch.setattr(smoke.uuid, "uuid4", lambda: SimpleNamespace(int=len(attempts)))
    if busy_count == 64:
        with pytest.raises(RuntimeError, match="64 attempts"):
            smoke.select_loopback_port()
    else:
        assert smoke.select_loopback_port() == 20001
    assert len(attempts) == len(closes) == min(busy_count + 1, 64)


@pytest.mark.parametrize("port", [61417, 55536, 0, -1, True, 20000.0])
def test_invalid_derived_grpc_ports_reject(port):
    with pytest.raises(ValueError):
        smoke.configure_native_ports({}, port)


def test_stats_require_actual_managed_scheduler():
    stats = healthy()
    assert extract_stats({"internal_states": [{"redknot_managed_kv_stats": stats}]}) is stats
    validate_idle(stats, minimum_created=4)
    with pytest.raises(RuntimeError):
        extract_stats({"internal_states": []})
    with pytest.raises(RuntimeError):
        extract_stats({"internal_states": [{}]})


@pytest.mark.parametrize("field", ["live_serving_requests", "requests", "cached_roots",
                                    "pending_transactions", "live_pages", "reserved_pages",
                                    "retiring_pages", "pinned_pages"])
def test_any_live_resource_fails(field):
    stats = healthy()
    stats[field] = 1
    with pytest.raises(RuntimeError):
        validate_idle(stats)


@pytest.mark.parametrize("field,value", [("free_pages", 99), ("released_serving_requests", 3),
                                         ("created_serving_requests", None)])
def test_missing_pages_or_count_mismatch_fails(field, value):
    stats = healthy()
    stats[field] = value
    with pytest.raises(RuntimeError):
        validate_idle(stats)


def test_completion_and_real_chunking_required():
    result = dict(text="Four.", meta_info=dict(completion_tokens=16, prompt_tokens=1024,
                                             finish_reason={"type": "length"}))
    validate_completion(result, chunked=True)
    result["meta_info"]["prompt_tokens"] = 50
    with pytest.raises(RuntimeError):
        validate_completion(result, chunked=True)
    result["meta_info"]["finish_reason"] = {"type": "abort"}
    with pytest.raises(RuntimeError):
        validate_completion(result)


def test_model_and_nonoverwriting_receipts(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    args = SimpleNamespace(model_path=str(model), output=str(tmp_path / "new"),
                           startup_timeout=600, execution_timeout=300, mem_fraction_static=0.8)
    assert validate_inputs(args)[1] == model
    args.execution_timeout = 301
    with pytest.raises(ValueError):
        validate_inputs(args)
    args.execution_timeout = 300
    (tmp_path / "new").mkdir()
    with pytest.raises(ValueError):
        validate_inputs(args)


@pytest.mark.parametrize("fault", [None, "stop_raises", "survivors", "executor_raises", "persist_raises"])
def test_cleanup_masks_signals_and_restores_handlers_even_on_failure(monkeypatch, tmp_path, fault):
    previous = {smoke.signal.SIGTERM: object(), smoke.signal.SIGINT: object()}
    active = dict(previous)
    saved = []

    def set_signal(sig, handler):
        active[sig] = handler

    def assert_masked():
        assert all(active[sig] == smoke.signal.SIG_IGN for sig in previous)

    def stop(process):
        assert_masked()
        if fault == "stop_raises":
            raise OSError("test cleanup failure")
        return [123] if fault == "survivors" else []

    class Executor:
        def shutdown(self, **kwargs):
            assert_masked()
            assert kwargs == {"wait": False, "cancel_futures": True}
            if fault == "executor_raises":
                raise OSError("test executor failure")

    receipt = {"status": "passed_native_lifecycle_smoke"}

    def persist():
        assert_masked()
        saved.append(json.loads(json.dumps(receipt)))
        if fault == "persist_raises":
            raise OSError("test receipt IO failure")

    monkeypatch.setattr(smoke.signal, "signal", set_signal)
    monkeypatch.setattr(smoke, "stop_child", stop)
    if fault is None:
        smoke.finalize_run(receipt, persist, object(), Executor(), previous, tmp_path)
    else:
        with pytest.raises((RuntimeError, OSError)):
            smoke.finalize_run(receipt, persist, object(), Executor(), previous, tmp_path)
    assert active == previous
    assert len(saved) == 1 and "finished_unix" in saved[0]
    if fault in ("stop_raises", "survivors", "executor_raises"):
        assert saved[0]["status"] == "failed" and saved[0]["cleanup_errors"]
    if fault == "stop_raises":
        assert saved[0]["cleanup_surviving_child_pids"] is None
        assert saved[0]["cleanup_verified"] is False
