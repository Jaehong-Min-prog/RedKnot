"""Immutable position sharing, transaction-local identity checks, no mask change."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache"))
from head_kv import HeadKVManager, HeadPagePool, SegmentWrite
import head_kv.manager as manager_module
from test_nonprefix_paged import _isolated_imports, _tiny_model


def _manager():
    manager = HeadKVManager(HeadPagePool(64, 4, 8))
    manager.create_request("r", context_id="request", contract="test", namespace="test")
    return manager


def _writes(positions, count=6, *, offset=0):
    return [SegmentWrite((0, h, "D"), torch.full((len(positions[h]), 8), float(offset+h)),
                         torch.full((len(positions[h]), 8), float(-offset-h)),
                         positions[h], "source") for h in range(count)]


def _count_position_sets(monkeypatch, positions):
    references = tuple(positions)
    calls = []
    native_set = set
    def observed(items=()):
        if any(items is pos for pos in references):
            calls.append(items)
        return native_set(items)
    monkeypatch.setattr(manager_module, "set", observed, raising=False)
    return calls


@pytest.mark.parametrize("batched", [False, True])
def test_shared_builtin_tuple_is_validated_once_but_each_payload_is_published(batched, monkeypatch):
    manager = _manager()
    positions = tuple(range(7))
    calls = _count_position_sets(monkeypatch, [positions])
    version = manager.update("r", writes=_writes([positions]*6), batched_writes=batched)
    assert len(calls) == 1
    assert all(segment.positions is positions for segment in version.segments.values())
    with manager.bind("r") as lease:
        for h in range(6):
            k, v = lease.gather((0, h, "D"))
            torch.testing.assert_close(k, torch.full((7, 8), float(h)))
            torch.testing.assert_close(v, torch.full((7, 8), float(-h)))
    manager.release_request("r")
    assert manager.pool.stats()["free_pages"] == 64


def test_equal_but_distinct_tuples_are_not_value_hashed_or_interned(monkeypatch):
    manager = _manager()
    positions = [tuple(range(7)) for _ in range(6)]
    assert len({id(pos) for pos in positions}) == 6
    calls = _count_position_sets(monkeypatch, positions)
    version = manager.update("r", writes=_writes(positions), batched_writes=True)
    assert len(calls) == 6
    assert all(version.segments[0, h, "D"].positions is pos for h, pos in enumerate(positions))
    manager.release_request("r")


@pytest.mark.parametrize("kind", ["list", "tuple_subclass", "int_subclass", "bool"])
def test_custom_or_mutable_position_types_keep_full_validation(kind, monkeypatch):
    class CustomTuple(tuple):
        pass
    class CustomInt(int):
        pass
    positions = {"list": list(range(7)), "tuple_subclass": CustomTuple(range(7)),
                 "int_subclass": tuple(CustomInt(i) for i in range(7)),
                 "bool": (False, True, 2, 3, 4, 5, 6)}[kind]
    manager = _manager()
    calls = _count_position_sets(monkeypatch, [positions])
    manager.update("r", writes=_writes([positions]*6), batched_writes=True)
    assert len(calls) == 6
    manager.release_request("r")


@pytest.mark.parametrize("fault", ["payload_length", "payload_shape", "provenance", "reuse_kind", "certificate"])
def test_cached_position_success_never_bypasses_later_write_validation(fault):
    manager = _manager()
    positions = tuple(range(7))
    writes = _writes([positions]*6)
    last = writes[-1]
    fields = dict(key=last.key, k=last.k, v=last.v, positions=last.positions,
                  provenance=last.provenance, reuse_kind=last.reuse_kind,
                  validity_certificate=last.validity_certificate)
    if fault == "payload_length":
        fields.update(k=last.k[:-1], v=last.v[:-1])
    elif fault == "payload_shape":
        fields.update(k=last.k[:, :-1], v=last.v[:, :-1])
    elif fault == "provenance":
        fields["provenance"] = ""
    elif fault == "reuse_kind":
        fields["reuse_kind"] = "invalid-kind"
    elif fault == "certificate":
        fields["reuse_kind"] = "policy_approximate"
    writes[-1] = SegmentWrite(**fields)
    old, before = manager.version("r"), manager.pool.stats()
    with pytest.raises(ValueError):
        manager.update("r", writes=writes, batched_writes=True)
    assert manager.version("r") is old
    assert manager.pool.stats() == before


@pytest.mark.parametrize("bad_positions", [(0, 1, 2, 3, 4, 5, -1), (0, 1, 2, 3, 4, 5, 5)])
def test_distinct_invalid_tuple_after_valid_tuple_remains_fail_closed(bad_positions):
    manager = _manager()
    writes = _writes([tuple(range(7))]*5+[bad_positions])
    old, before = manager.version("r"), manager.pool.stats()
    with pytest.raises(ValueError):
        manager.update("r", writes=writes, batched_writes=True)
    assert manager.version("r") is old
    assert manager.pool.stats() == before


def test_stateful_int_subclass_cannot_hide_second_validation_failure():
    class ChangingInt(int):
        comparisons = 0
        def __lt__(self, other):
            self.comparisons += 1
            return self.comparisons > 1
    value = ChangingInt(0)
    positions = (value, 1, 2)
    manager = _manager()
    old, before = manager.version("r"), manager.pool.stats()
    with pytest.raises(ValueError, match="logical position"):
        manager.update("r", writes=_writes([positions]*2, count=2), batched_writes=True)
    assert value.comparisons == 2
    assert manager.version("r") is old and manager.pool.stats() == before


def test_tuple_subclass_with_stateful_iteration_is_not_memoized():
    class ChangingTuple(tuple):
        iterations = 0
        def __iter__(self):
            self.iterations += 1
            return iter((0, 1, 2) if self.iterations <= 2 else (-1, 1, 2))
    positions = ChangingTuple((0, 1, 2))
    manager = _manager()
    old, before = manager.version("r"), manager.pool.stats()
    with pytest.raises(ValueError, match="logical position"):
        manager.update("r", writes=_writes([positions]*2, count=2), batched_writes=True)
    assert manager.version("r") is old and manager.pool.stats() == before


def test_validation_memo_is_transaction_local_and_leases_keep_old_payload(monkeypatch):
    manager = _manager()
    positions = tuple(range(7))
    calls = _count_position_sets(monkeypatch, [positions])
    old = manager.update("r", writes=_writes([positions]*6), batched_writes=True)
    lease = manager.bind("r")
    current = manager.update("r", writes=_writes([positions]*6, offset=20), batched_writes=True)
    assert len(calls) == 2 and current.epoch == old.epoch+1
    torch.testing.assert_close(lease.gather((0, 0, "D"))[0], torch.zeros((7, 8)))
    tx = manager.begin_update("r", writes=_writes([positions]*6, offset=40), batched_writes=True)
    tx.abort()
    assert len(calls) == 3 and manager.version("r") is current
    manager.release_request("r")
    assert manager.pool.stats()["free_pages"] == 64-12  # Only old reader's pages remain.
    lease.complete()
    assert manager.pool.stats()["free_pages"] == 64


def test_engine_shares_only_immutable_position_metadata_across_heads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with _isolated_imports() as runtime:
            model = _tiny_model()
            classes = [["global", "global"], ["local", "local"], ["local", "global"]]
            config = runtime.config.HeadClassConfig(classes,
                [[7 if head == "local" else -1 for head in row] for row in classes],
                3, 2, default_sink_size=0)
            engine = runtime.core.NonPrefixPagedEngine(model, config, capacity_pages=2048,
                page_size=4, backend="torch", prefix_chunk_size=256)
            try:
                engine.register_document("D", list(range(20, 31)))
                document = engine.documents["D"]
                source = engine.manager.version(document.source_request_id)
                source_positions = [segment.positions for segment in source.segments.values()]
                assert len(source_positions) == 3
                assert all(pos is source_positions[0] for pos in source_positions)
                assert source_positions[0] == tuple(range(11))
                state = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
                version = engine.manager.version(state.manager_request_id)
                globals_ = [s.positions for key, s in version.segments.items() if key[2] == "global"]
                prefixes = [s.positions for key, s in version.segments.items() if key[2] == "prefix"]
                assert len(globals_) == 3 and all(pos is globals_[0] for pos in globals_)
                assert len(prefixes) == 3 and all(pos is prefixes[0] for pos in prefixes)
                assert globals_[0] == tuple(range(14)) and prefixes[0] == (0, 1, 2)
                assert globals_[0] is not source_positions[0]
                logits = engine.forward(["A"], torch.tensor([[81, 82, 83]]))
                assert torch.isfinite(logits).all()
            finally:
                engine.close()
            assert engine.pool.stats()["free_pages"] == 2048
    finally:
        torch.set_num_threads(previous)
