"""CPU contract tests for optional publication wall-clock diagnostics.

These exercise real tiny-Qwen admission, page publication and decode. CPU
clock/sync-call tests do not claim to measure CUDA phase costs or QPS.
"""
from contextlib import contextmanager
import math
import types

import pytest
import torch

from test_nonprefix_paged import (
    runtime, _tiny_model, _head_config, _DOCS, _hash_source_pages,
)


_STAGES = (
    "head_pack", "head_write_and_release", "alias_publish",
    "boundary_partition", "final_bind",
)


def _make(runtime, *, flag="default", mixed=True, share_local=True, hidden=False):
    kwargs = {} if flag == "default" else {"publication_diagnostics": flag}
    if hidden:
        kwargs["selective_recovery"] = runtime.core.SelectiveRecoveryConfig(
            block_size=3, row_fraction=.5, probe_layers=1,
            selection_mode="query_attention", recovery_mode="hidden_residual_int8",
            max_offline_kv_bytes=1 << 20)
    engine = runtime.core.NonPrefixPagedEngine(
        # Final E occurrence has 7 rows: a 5-row window starts inside its
        # first page, deliberately exercising the boundary transaction.
        _tiny_model(), _head_config(runtime, mixed=mixed, window=5),
        capacity_pages=2048, page_size=4, backend="torch",
        model_revision="publication-diagnostic-contract-v1", prefix_chunk_size=256,
        share_local=share_local, **kwargs)
    for name, ids in _DOCS.items():
        engine.register_document(name, ids)
    return engine


@contextmanager
def _trace_admission(runtime, monkeypatch, engine):
    trace = dict(syncs=0, clocks=0, transactions=[], checkpoints=[])
    sync, clock = engine.synchronize, runtime.core.time.perf_counter
    update, share = engine.manager.begin_update, engine.manager.share_segments_many
    checkpoint = engine._publication_checkpoint
    def synchronize():
        trace["syncs"] += 1
        return sync()
    def perf_counter():
        trace["clocks"] += 1
        return clock()
    def begin_update(request_id, **kwargs):
        trace["transactions"].append(("write", tuple(w.key for w in kwargs.get("writes", ())),
                                      kwargs.get("batched_writes", False)))
        return update(request_id, **kwargs)
    def share_segments_many(target_id, sources, **kwargs):
        trace["transactions"].append(("alias", tuple(
            tuple(item[1] for item in batch[1]) for batch in sources)))
        return share(target_id, sources, **kwargs)
    def finish(stages, name, started):
        trace["checkpoints"].append(name)
        return checkpoint(stages, name, started)
    with monkeypatch.context() as patch:
        # Replace only this module's time binding, not Python's shared module.
        patch.setattr(runtime.core, "time", types.SimpleNamespace(perf_counter=perf_counter))
        patch.setattr(engine, "synchronize", synchronize)
        patch.setattr(engine, "_publication_checkpoint", finish)
        patch.setattr(engine.manager, "begin_update", begin_update)
        patch.setattr(engine.manager, "share_segments_many", share_segments_many)
        yield trace


def _snapshot(runtime, monkeypatch, *, flag, mixed=True, share_local=True, hidden=False):
    engine = _make(runtime, flag=flag, mixed=mixed, share_local=share_local, hidden=hidden)
    source_before = _hash_source_pages(engine)
    try:
        with _trace_admission(runtime, monkeypatch, engine) as trace:
            query = {"query_ids": (81, 82)} if engine.requires_recovery_query else {}
            state = engine.admit("R", [1, 2, 3], ["E", "D", "E"], max_suffix_tokens=8, **query)
        with engine.manager.bind(state.manager_request_id) as lease:
            history = {key: tuple(t.clone() for t in lease.gather(key))
                       for key in lease.version.segments}
            positions = {key: segment.positions for key, segment in lease.version.segments.items()}
        outputs = [engine.forward(["R"], torch.tensor([[81, 82]])).clone()]
        outputs.append(engine.forward(["R"], outputs[0][:, -1].argmax(-1)[:, None]).clone())
        assert _hash_source_pages(engine) == source_before
        result = dict(stats=state.admission_stats, trace=trace, history=history,
                      positions=positions, outputs=outputs, engine_stats=engine.stats())
        engine.release("R")
        assert engine.pool.stats()["pinned_pages"] == 0
        return result
    finally:
        engine.close()
        assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages


def _assert_same_content(left, right):
    assert left["positions"] == right["positions"]
    assert left["history"].keys() == right["history"].keys()
    for key in left["history"]:
        for a, b in zip(left["history"][key], right["history"][key]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(left["outputs"], right["outputs"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert left["trace"]["transactions"] == right["trace"]["transactions"]
    for field in ("required_pages", "alias_publications", "alias_source_roots", "alias_segment_count",
                  "pool_after_admit", "recovery_mode", "recovery_approximate"):
        assert left["stats"][field] == right["stats"][field]


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_diagnostic_flag_requires_real_bool_before_pool_allocation(runtime, monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid diagnostic flag allocated a page pool")
    monkeypatch.setattr(runtime.core, "HeadPagePool", forbidden)
    with pytest.raises(ValueError, match="publication_diagnostics must be boolean"):
        runtime.core.NonPrefixPagedEngine(
            _tiny_model(), _head_config(runtime), capacity_pages=2048,
            page_size=4, backend="torch", publication_diagnostics=value)


def test_default_and_explicit_off_are_output_and_sync_compatible(runtime, monkeypatch):
    default = _snapshot(runtime, monkeypatch, flag="default")
    disabled = _snapshot(runtime, monkeypatch, flag=False)
    _assert_same_content(default, disabled)
    assert default["trace"] == disabled["trace"]
    assert default["trace"]["checkpoints"] == []
    assert default["stats"].keys() == disabled["stats"].keys()
    assert "publication_diagnostics" not in default["stats"]
    assert default["engine_stats"].keys() == disabled["engine_stats"].keys()
    assert "publication_diagnostics" not in default["engine_stats"]


@pytest.mark.parametrize("mixed,share_local,hidden", [
    (True, True, False), (True, False, False), (False, True, False), (True, True, True),
])
def test_diagnostics_split_five_stages_without_changing_kv_cow_or_output(
        runtime, monkeypatch, mixed, share_local, hidden):
    disabled = _snapshot(runtime, monkeypatch, flag=False, mixed=mixed, share_local=share_local, hidden=hidden)
    enabled = _snapshot(runtime, monkeypatch, flag=True, mixed=mixed, share_local=share_local, hidden=hidden)
    _assert_same_content(disabled, enabled)
    assert enabled["stats"].keys()-disabled["stats"].keys() == {"publication_diagnostics"}
    assert disabled["stats"].keys()-enabled["stats"].keys() == set()
    assert enabled["trace"]["syncs"] == disabled["trace"]["syncs"]+4
    assert enabled["trace"]["clocks"] == disabled["trace"]["clocks"]+5
    assert enabled["trace"]["checkpoints"] == list(_STAGES[:-1])
    details = enabled["stats"]["publication_diagnostics"]
    assert details["schema"] == "redknot-publication-stages-v1"
    assert details["diagnostic_only"] is True
    assert details["device_synchronizes"] is False  # This is a CPU contract test.
    assert details["extra_synchronization_calls"] == 4
    assert details["clock"] == "host_perf_counter_with_completion_sync"
    assert list(details["stage_seconds"]) == list(_STAGES)
    assert all(math.isfinite(value) and value >= 0 for value in details["stage_seconds"].values())
    assert math.isclose(sum(details["stage_seconds"].values()), details["total_seconds"], abs_tol=1e-12)
    assert details["total_seconds"] <= enabled["stats"]["kv_publish_and_partition_s"]
    assert "not an uninstrumented QPS result" in details["performance_caveat"]
    if mixed:
        assert enabled["engine_stats"]["boundary_copy_bytes"] > 0
    if mixed and share_local:
        assert enabled["stats"]["alias_publications"] == 1
        assert enabled["stats"]["alias_segment_count"] == 3


@pytest.mark.parametrize("stage", _STAGES[:-1])
def test_diagnostic_checkpoint_failure_retains_admission_rollback(runtime, monkeypatch, stage):
    engine = _make(runtime, flag=True)
    source_before = _hash_source_pages(engine)
    before = engine.pool.stats()
    checkpoint = engine._publication_checkpoint
    def injected(stages, name, started):
        value = checkpoint(stages, name, started)
        if name == stage:
            raise RuntimeError("injected diagnostic boundary failure")
        return value
    monkeypatch.setattr(engine, "_publication_checkpoint", injected)
    try:
        with pytest.raises(RuntimeError, match="injected diagnostic boundary failure"):
            engine.admit("R", [1, 2, 3], ["E", "D", "E"], max_suffix_tokens=8)
        assert not engine.requests
        assert not engine.manager._pending
        assert _hash_source_pages(engine) == source_before
        after = engine.pool.stats()
        for field in ("free_pages", "live_pages", "reserved_pages", "retiring_pages", "pinned_pages"):
            assert after[field] == before[field]
        assert not after["poisoned"]
    finally:
        engine.close()
        assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
