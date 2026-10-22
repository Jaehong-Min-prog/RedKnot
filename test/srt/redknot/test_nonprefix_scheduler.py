"""CPU fake-engine tests of actual queue/forward ownership, not speed evidence."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch


_SOURCE = Path(__file__).resolve().parents[3] / "python"


@contextmanager
def _isolated_imports():
    saved = {name: module for name, module in sys.modules.items()
             if name == "sglang" or name.startswith("sglang.")}
    for name in saved:
        del sys.modules[name]
    try:
        for name in (
            "sglang", "sglang.srt", "sglang.srt.layers",
            "sglang.srt.layers.attention", "sglang.srt.layers.attention.redknot",
            "sglang.srt.mem_cache", "sglang.srt.mem_cache.head_kv",
        ):
            module = types.ModuleType(name)
            module.__path__ = [str(_SOURCE.joinpath(*name.split(".")))]
            sys.modules[name] = module
        yield importlib.import_module(
            "sglang.srt.layers.attention.redknot.nonprefix_scheduler"
        )
    finally:
        for name in list(sys.modules):
            if name == "sglang" or name.startswith("sglang."):
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(scope="module")
def module():
    with _isolated_imports() as scheduler:
        yield scheduler


class Clock:
    def __init__(self):
        self.value = 123.0
        self.sleeps = []

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)


@pytest.fixture
def clock(module, monkeypatch):
    result = Clock()
    # Replace this module's time reference, not the interpreter's shared time.
    monkeypatch.setattr(module, "time", types.SimpleNamespace(
        perf_counter=result, sleep=result.sleep,
    ))
    return result


class FakeEngine:
    def __init__(self, module, clock, *, capacity=20, outputs=None,
                 impossible=(), fail_admit=None, fail_forward=None,
                 invalid_logits=None, fail_release_once=None):
        self.device = torch.device("cpu")
        self.module, self.clock, self.capacity = module, clock, capacity
        self.outputs = outputs or {}
        self.impossible = set(impossible)
        self.fail_admit, self.fail_forward = fail_admit, fail_forward
        self.invalid_logits = invalid_logits
        self.fail_release_once = fail_release_once
        self.requests = {}
        self.admit_calls, self.forward_calls, self.releases = [], [], []
        self.generated = {}
        self.hook = None

    def admit(self, request_id, prefix_ids, document_ids, max_suffix_tokens):
        self.clock.advance(0.01)
        self.admit_calls.append((request_id, tuple(prefix_ids), tuple(document_ids),
                                 max_suffix_tokens))
        if request_id == self.fail_admit:
            raise RuntimeError("injected admission error (atomic)")
        if request_id in self.impossible or len(self.requests) >= self.capacity:
            raise self.module.CapacityError("request does not fit")
        assert request_id not in self.requests
        self.requests[request_id] = {"reserved": max_suffix_tokens, "consumed": []}
        self.generated[request_id] = 0
        return self.requests[request_id]

    def forward(self, request_ids, input_ids):
        assert input_ids.dtype == torch.long
        assert input_ids.ndim == 2 and input_ids.shape[0] == len(request_ids)
        assert len(set(request_ids)) == len(request_ids)
        assert all(request_id in self.requests for request_id in request_ids)
        self.clock.advance(0.1)
        self.forward_calls.append((list(request_ids), input_ids.tolist()))
        if self.hook:
            self.hook()
        if len(self.forward_calls) == self.fail_forward:
            raise RuntimeError("injected model error")
        logits = torch.full((len(request_ids), 1, 32), -10.0)
        for row, request_id in enumerate(request_ids):
            state = self.requests[request_id]
            state["consumed"].extend(input_ids[row].tolist())
            assert len(state["consumed"]) <= state["reserved"]
            index = self.generated[request_id]
            planned = self.outputs.get(request_id, (3, 4, 5, 6, 7, 8))
            token = planned[min(index, len(planned) - 1)]
            logits[row, 0, token] = 10.0
            self.generated[request_id] += 1
        if self.invalid_logits == "nan":
            logits[0, 0, 0] = float("nan")
        elif self.invalid_logits == "shape":
            logits = logits[:, 0]
        return logits

    def release(self, request_id):
        self.clock.advance(0.02)
        self.releases.append(request_id)
        if request_id == self.fail_release_once:
            self.fail_release_once = None
            raise RuntimeError("injected release error")
        del self.requests[request_id]

    def synchronize(self):
        pass

    def stats(self):
        return {"active_requests": len(self.requests), "forwards": len(self.forward_calls)}


def request(module, name, *, query=(9, 10), count=3, eos=(), arrival=0, cancel=None):
    return module.QueuedRequest(
        request_id=name, prefix_ids=(1,), document_ids=("shared_document",),
        query_ids=query, max_new_tokens=count, eos_token_ids=eos,
        arrival_seconds=arrival, cancel_after_tokens=cancel,
    )


def row(result, name):
    return next(item for item in result["records"] if item["request_id"] == name)


def test_true_batched_forward_dynamic_refill_and_wall_accounting(module, clock):
    engine = FakeEngine(module, clock)
    result = module.run_queued(engine, [
        request(module, "a", count=1), request(module, "b", count=3),
        request(module, "c", count=1),
    ], max_active=2)
    assert result["status"] == "complete"
    assert result["completed_requests"] == 3
    assert result["completed_output_tokens"] == 5
    assert result["max_forward_batch"] == result["peak_active_requests"] == 2
    # One forward really consumes two request IDs, not two serialized forwards.
    assert engine.forward_calls == [
        (["a", "b"], [[9, 10], [9, 10]]),
        (["b"], [[3]]), (["c"], [[9, 10]]), (["b"], [[4]]),
    ]
    events = result["trace"]
    admitted_c = next(i for i, event in enumerate(events)
                      if event["event"] == "admitted" and event["request_id"] == "c")
    second_forward = [i for i, event in enumerate(events) if event["event"] == "forward"][1]
    assert admitted_c < second_forward
    assert result["stages_seconds"] == pytest.approx({
        "admission": 0.03, "query_forward": 0.2, "decode_forward": 0.2,
        "release": 0.06, "idle_wait": 0.0,
    })
    assert result["total_wall_seconds"] == pytest.approx(0.49)
    assert result["completed_qps"] == pytest.approx(3 / 0.49)
    assert row(result, "a")["ttft_seconds"] == pytest.approx(0.12)
    assert row(result, "a")["completion_seconds"] == pytest.approx(0.14)
    assert row(result, "c")["queue_wait_seconds"] == pytest.approx(0.14)
    assert row(result, "c")["ttft_seconds"] > row(result, "c")["service_ttft_seconds"]
    assert engine.requests == {}
    assert result["engine_stats_after"]["active_requests"] == 0
    assert [call[3] for call in engine.admit_calls] == [2, 4, 2]
    json.dumps(result, allow_nan=False)


def test_variable_queries_unpadded_buckets_and_mixed_age_decode(module, clock):
    engine = FakeEngine(module, clock)
    result = module.run_queued(engine, [
        request(module, "a", query=(7, 8), count=4),
        request(module, "b", query=(7, 8, 9), count=3),
        request(module, "c", query=(5, 6), count=4),
    ], max_active=3)
    assert result["completed_requests"] == 3
    assert engine.forward_calls[0] == (["a", "c"], [[7, 8], [5, 6]])
    assert (["b"], [[7, 8, 9]]) in engine.forward_calls
    assert any(ids == ["a", "b", "c"] and values == [[4], [3], [4]]
               for ids, values in engine.forward_calls)
    assert all(item["input_tokens_per_request"] in (1, 2, 3)
               for item in result["trace"] if item["event"] == "forward")
    assert row(result, "a")["output_ids"] == [3, 4, 5, 6]
    assert row(result, "b")["output_ids"] == [3, 4, 5]
    assert not engine.requests


def test_opt_in_recovery_gets_exact_query_without_consuming_regular_query(module, clock):
    class QueryEngine(FakeEngine):
        requires_recovery_query = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recovery_queries = []

        def admit(self, request_id, prefix_ids, document_ids, max_suffix_tokens, *, query_ids):
            self.recovery_queries.append((request_id, query_ids))
            self.clock.advance(0.04)  # All additional probe work remains inside admission.
            return super().admit(request_id, prefix_ids, document_ids, max_suffix_tokens)

    engine = QueryEngine(module, clock)
    result = module.run_queued(engine, [request(module, "a", query=(7, 8), count=1),
                                      request(module, "b", query=(11, 12, 13), count=1)], max_active=2)
    assert result["status"] == "complete"
    assert engine.recovery_queries == [("a", (7, 8)), ("b", (11, 12, 13))]
    assert engine.forward_calls == [(["a"], [[7, 8]]), (["b"], [[11, 12, 13]])]
    assert [call[3] for call in engine.admit_calls] == [2, 3]
    assert result["stages_seconds"]["admission"] == pytest.approx(.1)
    assert row(result, "a")["service_ttft_seconds"] >= .1
    assert result["completed_output_tokens"] == 2 and not engine.requests


@pytest.mark.parametrize("capability", [False, None, 1, "true"])
def test_default_and_non_boolean_capability_do_not_change_old_admit_api(module, clock, capability):
    engine = FakeEngine(module, clock)
    engine.requires_recovery_query = capability
    result = module.run_queued(engine, [request(module, "a", count=1)], max_active=1)
    assert result["status"] == "complete"
    assert engine.forward_calls == [(["a"], [[9, 10]])]


def test_query_capability_does_not_silently_fallback_if_engine_api_is_missing(module, clock):
    engine = FakeEngine(module, clock)
    engine.requires_recovery_query = True  # Signature deliberately lacks query_ids.
    result = module.run_queued(engine, [request(module, "a", count=1)], max_active=1)
    assert result["status"] != "complete" and result["completed_requests"] == 0
    assert not engine.forward_calls and not engine.requests


def test_request_local_admission_provenance_is_snapshotted(module, clock):
    class MetadataEngine(FakeEngine):
        def admit(self, request_id, prefix_ids, document_ids, max_suffix_tokens):
            super().admit(request_id, prefix_ids, document_ids, max_suffix_tokens)
            self.last_metadata = types.SimpleNamespace(
                admission_stats={"prefix_seconds": 0.003, "recovery_seconds": 0.005,
                                 "published_rows": {"local": 11, "global": 17}},
                doc_offsets=(3, 14), history_tokens=21,
            )
            return self.last_metadata

        def release(self, request_id):
            self.last_metadata.admission_stats["published_rows"]["local"] = 999
            return super().release(request_id)

    engine = MetadataEngine(module, clock)
    result = module.run_queued(engine, [request(module, "a", count=1)], max_active=1)
    record = row(result, "a")
    assert record["doc_offsets"] == [3, 14]
    assert record["history_tokens"] == 21
    assert record["admission_stats"] == {"prefix_seconds": 0.003, "recovery_seconds": 0.005,
                                         "published_rows": {"local": 11, "global": 17}}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("ignore_eos,expected", [(False, [3, 2]), (True, [3, 2, 4, 5])])
def test_eos_is_counted_and_optional_fixed_output_mode(module, clock, ignore_eos, expected):
    engine = FakeEngine(module, clock, outputs={"a": [3, 2, 4, 5]})
    result = module.run_queued(engine, [request(module, "a", count=4, eos=(2,))],
                               max_active=1, ignore_eos=ignore_eos)
    assert row(result, "a")["output_ids"] == expected
    assert row(result, "a")["finish_reason"] == ("length" if ignore_eos else "eos")
    assert len(engine.forward_calls) == len(expected)
    assert not engine.requests


def test_cancellations_release_and_do_not_inflate_completed_qps(module, clock):
    engine = FakeEngine(module, clock)
    result = module.run_queued(engine, [
        request(module, "skip", cancel=0), request(module, "cancel", cancel=2, count=6),
        request(module, "ok", count=3), request(module, "refill", count=1),
    ], max_active=2)
    assert result["cancelled_requests"] == 2
    assert result["completed_requests"] == 2
    assert result["all_emitted_output_tokens"] == 6
    assert result["completed_output_tokens"] == 4
    assert row(result, "cancel")["output_ids"] == [3, 4]
    assert row(result, "skip")["admission_attempts"] == 0
    assert "skip" not in engine.releases
    assert "cancel" in engine.releases
    assert not engine.requests


def test_capacity_backpressure_waits_for_release_then_refills(module, clock):
    engine = FakeEngine(module, clock, capacity=2)
    result = module.run_queued(engine, [
        request(module, "a", count=4), request(module, "b", count=3),
        request(module, "c", count=2),
    ], max_active=3)
    assert result["completed_requests"] == 3
    assert row(result, "c")["capacity_blocks"] == 1
    assert row(result, "c")["admission_attempts"] == 2
    assert row(result, "c")["admission_attempt_seconds"] == pytest.approx(0.02)
    assert row(result, "c")["admission_seconds"] == pytest.approx(0.01)
    assert result["peak_active_requests"] == 2
    assert not engine.requests


def test_impossible_capacity_marked_not_skipped_and_other_requests_run(module, clock):
    engine = FakeEngine(module, clock, impossible=("too_large",))
    result = module.run_queued(engine, [request(module, "too_large"),
                                      request(module, "ok", count=1)], max_active=2)
    assert result["status"] == "complete_with_failed_requests"
    assert result["completed_requests"] == result["failed_requests"] == 1
    assert row(result, "too_large")["finish_reason"] == "impossible_capacity"
    assert row(result, "too_large")["output_ids"] == []
    assert engine.releases == ["ok"]
    assert not engine.requests


@pytest.mark.parametrize("where", ["admit", "forward"])
def test_engine_errors_abort_without_fallback_and_release_every_live_request(module, clock, where):
    engine = FakeEngine(module, clock, fail_admit="b" if where == "admit" else None,
                        fail_forward=2 if where == "forward" else None)
    result = module.run_queued(engine, [request(module, name) for name in ("a", "b", "c")],
                               max_active=2)
    assert result["status"] == "failed"
    assert result["failed_requests"] == 3
    assert result["completed_requests"] == 0
    assert result["completed_qps"] == 0
    assert result["errors"][0]["type"] == "RuntimeError"
    assert not engine.requests
    if where == "forward":
        assert set(engine.releases) == {"a", "b"}
        assert len(engine.forward_calls) == 2  # No model retry/native fallback.
    else:
        assert engine.releases == ["a"]
        assert not engine.forward_calls
    # Owner is reset even after failure, so a fresh cohort may run.
    engine.fail_admit = engine.fail_forward = None
    followup = module.run_queued(engine, [request(module, "new", count=1)], max_active=1)
    assert followup["completed_requests"] == 1


@pytest.mark.parametrize("invalid", ["nan", "shape"])
def test_invalid_logits_fail_closed_and_release(module, clock, invalid):
    engine = FakeEngine(module, clock, invalid_logits=invalid)
    result = module.run_queued(engine, [request(module, "a")], max_active=1)
    assert result["status"] == "failed"
    assert result["completed_requests"] == 0
    assert result["all_emitted_output_tokens"] == 0
    assert result["errors"][0]["type"] == "ValueError"
    assert engine.releases == ["a"]


def test_cleanup_failure_is_visible_and_other_requests_are_released(module, clock):
    engine = FakeEngine(module, clock, fail_release_once="a")
    result = module.run_queued(engine, [request(module, "a", count=1),
                                      request(module, "b", count=1)], max_active=2)
    assert result["status"] == "failed"
    assert result["completed_requests"] == 0
    assert result["errors"][0]["message"] == "injected release error"
    assert engine.releases == ["a", "a", "b"]
    assert not engine.requests


def test_future_arrivals_are_fcfs_and_wait_is_included(module, clock):
    engine = FakeEngine(module, clock)
    result = module.run_queued(engine, [
        request(module, "later", count=1, arrival=0.5),
        request(module, "first", count=1, arrival=0.0),
    ], max_active=2)
    assert [item[0] for item in engine.admit_calls] == ["first", "later"]
    assert [item["request_id"] for item in result["records"]] == ["later", "first"]
    assert result["total_wall_seconds"] == pytest.approx(0.63)
    assert row(result, "later")["ttft_seconds"] == pytest.approx(0.11)
    assert result["stages_seconds"]["idle_wait"] == pytest.approx(0.37)
    assert clock.sleeps and max(clock.sleeps) <= 0.01


def test_single_owner_rejects_nested_queue_without_corrupting_outer(module, clock):
    engine = FakeEngine(module, clock)
    def nested():
        with pytest.raises(RuntimeError, match="active queue owner"):
            module.run_queued(engine, [request(module, "nested")], max_active=1)
    engine.hook = nested
    result = module.run_queued(engine, [request(module, "a", count=1)], max_active=1)
    assert result["completed_requests"] == 1
    assert not engine.requests


def test_empty_run_is_valid(module, clock):
    engine = FakeEngine(module, clock)
    result = module.run_queued(engine, [], max_active=1)
    assert result["completed_qps"] == result["total_wall_seconds"] == 0
    assert result["status"] == "complete"
    assert result["records"] == result["trace"] == []


@pytest.mark.parametrize("kwargs", [
    {"query_ids": ()}, {"query_ids": (-1,)}, {"query_ids": (True,)},
    {"max_new_tokens": 0}, {"max_new_tokens": 1.5}, {"max_new_tokens": True},
    {"arrival_seconds": -1}, {"arrival_seconds": float("nan")},
    {"cancel_after_tokens": -1}, {"document_ids": ("",)}, {"request_id": ""},
])
def test_request_validation(module, kwargs):
    options = dict(request_id="a", prefix_ids=(1,), document_ids=("doc",),
                   query_ids=(2,), max_new_tokens=2, eos_token_ids=())
    options.update(kwargs)
    with pytest.raises(ValueError):
        module.QueuedRequest(**options)


def test_invalid_queue_configuration_does_not_admit(module, clock):
    engine = FakeEngine(module, clock)
    with pytest.raises(ValueError, match="unique"):
        module.run_queued(engine, [request(module, "a"), request(module, "a")], max_active=1)
    with pytest.raises(ValueError, match="positive"):
        module.run_queued(engine, [], max_active=0)
    assert engine.admit_calls == []
