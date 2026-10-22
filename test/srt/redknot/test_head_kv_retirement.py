"""Incremental retirement must preserve ownership, pin and event lifetimes.

These tests intentionally inspect work counters as well as storage state: a
correct collector that scans every live page is still a regression here.
"""
import random
import sys
import weakref
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache"))
from head_kv import CapacityError, HeadPagePool, StaleReference


class Gate:
    __hash__ = None

    def __init__(self, ready=False, error=None):
        self.ready = ready
        self.error = error
        self.calls = 0

    def query(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.ready


def pool(capacity=16):
    return HeadPagePool(capacity, 2, 2)


def publish(p, refs):
    for ref in refs:
        p.write_page(ref, torch.ones(1, 2), torch.zeros(1, 2))
    p.seal(refs)


def check_gauges(p):
    stats = p.stats()
    pages = list(p._pages.values())
    assert stats["reserved_pages"] == sum(not x.sealed and x.owners > 0 for x in pages)
    assert stats["live_pages"] == sum(x.sealed and x.owners > 0 for x in pages)
    assert stats["retiring_pages"] == sum(x.owners == 0 for x in pages)
    assert stats["pinned_pages"] == sum(x.pins > 0 or bool(x.events) for x in pages)
    assert stats["allocated_bytes"] == len(pages) * p.page_bytes
    assert stats["free_pages"] + len(pages) == p.capacity_pages
    assert len(set(p._free)) == len(p._free)
    assert not set(p._free).intersection(p._pages)
    expected = {x.ref.slot: x.ref for x in pages
                if x.events or (x.owners == 0 and x.pins == 0)}
    assert p._collection_candidates == expected
    return stats


class NoLivePageScan(dict):
    """Ordinary point lookups work; full-live-page sweeps must not occur."""
    def __iter__(self):
        raise AssertionError("whole-live-page iteration")

    def items(self):
        raise AssertionError("whole-live-page items scan")

    def values(self):
        raise AssertionError("whole-live-page values scan")


def test_many_live_pages_and_few_retirees_have_bounded_collection_work():
    p = pool(4096)
    refs = p.reserve(4096)
    gate = Gate()
    p._pages = NoLivePageScan(p._pages)
    before = p.collection_stats()
    p.release(refs[:3], gate)
    assert p.collection_stats()["candidate_page_visits"] - before["candidate_page_visits"] == 3
    assert gate.calls == 1
    before = p.collection_stats()
    for _ in range(20):
        assert p.collect() == 0
        p.release(())
        p.unpin(())
        assert p.stats()["reserved_pages"] == 4093
    after = p.collection_stats()
    assert after["candidate_page_visits"] - before["candidate_page_visits"] == 20 * 4 * 3
    assert after["event_queries"] - before["event_queries"] == 20 * 4
    gate.ready = True
    assert p.collect() == 3
    before = p.collection_stats()
    for _ in range(10):
        p.release(())
        p.unpin(())
        p.stats()
        p.collect()
    assert p.collection_stats()["candidate_page_visits"] == before["candidate_page_visits"]
    assert p.collection_stats()["event_queries"] == before["event_queries"]
    p.release(refs[3:])
    assert p.stats()["free_pages"] == 4096


def test_event_free_owned_and_pinned_orphans_are_not_swept():
    p = pool(512)
    refs = p.reserve(512)
    publish(p, refs)
    p.pin(refs)
    p.release(refs[:256])
    p._pages = NoLivePageScan(p._pages)
    before = p.collection_stats()
    for _ in range(10):
        assert p.collect() == 0
        stats = p.stats()
        assert stats["live_pages"] == 256
        assert stats["retiring_pages"] == 256
        assert stats["pinned_pages"] == 512
    assert p.collection_stats()["candidate_page_visits"] == before["candidate_page_visits"]
    p.unpin(refs[:256])
    assert p.stats()["free_pages"] == 256
    assert p.collection_stats()["candidate_page_visits"] - before["candidate_page_visits"] == 256
    p.unpin(refs[256:])
    p.release(refs[256:])
    assert p.stats()["free_pages"] == 512


@pytest.mark.parametrize("trigger", ["release", "unpin", "reserve", "stats"])
def test_empty_operations_still_progress_prior_events(trigger):
    p = pool(2)
    ref, = p.reserve(1)
    gate = Gate()
    p.release([ref], gate)
    assert p.collection_stats()["candidate_pages"] == 1
    gate.ready = True
    if trigger == "reserve":
        assert p.reserve(0) == ()
    elif trigger == "stats":
        p.stats()
    else:
        getattr(p, trigger)(())
    assert p.stats()["free_pages"] == 2
    assert p.collection_stats()["candidate_pages"] == 0


def test_shared_event_identity_covers_owned_orphan_pinned_pages_and_buffers_once():
    p = pool(6)
    refs = p.reserve(6)
    publish(p, refs)
    p.pin(refs[:2])
    p.retain(refs[2:4])
    gate = Gate()
    tensor = torch.ones(1)
    tensor_ref = weakref.ref(tensor)
    p.defer_buffers([tensor], gate)
    p.defer_buffers([tensor], gate)
    del tensor
    p.release(refs, gate)
    assert gate.calls == 1
    assert p.stats()["pinned_pages"] == 6
    assert gate.calls == 2
    before = p.collection_stats()
    gate.ready = True
    assert p.collect() == 2  # refs 4..5; refs 0..1 pinned, 2..3 owned.
    after = p.collection_stats()
    assert after["event_queries"] - before["event_queries"] == 1
    assert after["buffer_entries_visited"] - before["buffer_entries_visited"] == 2
    assert tensor_ref() is None
    assert after["candidate_pages"] == 0
    assert p.stats()["pinned_pages"] == 2
    p.unpin(refs[:2])
    p.release(refs[2:4])
    assert check_gauges(p)["free_pages"] == 6


@pytest.mark.parametrize("sealed", [False, True])
def test_retaining_a_retiring_page_restores_correct_gauges_and_lifetime(sealed):
    p = pool(1)
    ref, = p.reserve(1)
    if sealed:
        publish(p, [ref])
    gate = Gate()
    p.release([ref], gate)
    assert check_gauges(p)["retiring_pages"] == 1
    p.retain([ref])
    stats = check_gauges(p)
    assert stats["retiring_pages"] == 0
    assert stats["live_pages" if sealed else "reserved_pages"] == 1
    gate.ready = True
    assert p.collect() == 0
    assert p.collection_stats()["candidate_pages"] == 0
    assert check_gauges(p)["pinned_pages"] == 0
    p.release([ref])
    assert check_gauges(p)["free_pages"] == 1


def test_pin_and_retain_can_resurrect_an_orphan_before_delayed_completion():
    p = pool(1)
    ref, = p.reserve(1)
    gate = Gate()
    p.release([ref], gate)
    p.pin([ref])
    gate.ready = True
    assert p.collect() == 0
    assert p.collection_stats()["candidate_pages"] == 0
    p.retain([ref])
    p.unpin([ref])
    assert check_gauges(p)["reserved_pages"] == 1
    p.release([ref])
    assert check_gauges(p)["free_pages"] == 1


def test_multiple_events_require_all_dependencies_even_after_owner_resurrection():
    p = pool(1)
    ref, = p.reserve(1)
    p.pin([ref])
    first, second = Gate(), Gate()
    p.release([ref], first)
    p.unpin([ref], second)
    first.ready = True
    assert p.collect() == 0
    assert p._get(ref).events == [second]
    p.retain([ref])
    p.release([ref])
    assert check_gauges(p)["retiring_pages"] == 1
    second.ready = True
    assert p.collect() == 1
    assert check_gauges(p)["free_pages"] == 1


def test_live_page_event_completion_is_pruned_before_final_release():
    p = pool(1)
    ref, = p.reserve(1)
    p.retain([ref])
    gate = Gate()
    p.release([ref], gate)
    assert check_gauges(p)["pinned_pages"] == 1
    gate.ready = True
    assert p.collect() == 0
    calls = gate.calls
    p.release([ref])
    assert gate.calls == calls
    assert check_gauges(p)["free_pages"] == 1


@pytest.mark.parametrize("owned", [False, True])
def test_failed_page_event_quarantines_live_or_retired_page_without_requery(owned):
    p = pool(2)
    refs = p.reserve(2)
    if owned:
        p.retain(refs)
    gate = Gate(error=RuntimeError("event failed"))
    with pytest.raises(RuntimeError, match="event failed"):
        p.release(refs, gate)
    assert p.poisoned and gate.calls == 1
    before = p.collection_stats()
    assert p.collect() == 0
    assert check_gauges(p)["free_pages"] == 0
    assert p.collection_stats()["candidate_page_visits"] == before["candidate_page_visits"]
    assert gate.calls == 1
    with pytest.raises(RuntimeError, match="quarantined"):
        p.reserve(0)


def test_failed_buffer_event_retains_buffers_and_pages_without_partial_release():
    p = pool(2)
    refs = p.reserve(2)
    gate = Gate(error=RuntimeError("buffer event failed"))
    tensor = torch.ones(1)
    tensor_ref = weakref.ref(tensor)
    p.defer_buffers([tensor], gate)
    del tensor
    with pytest.raises(RuntimeError, match="buffer event failed"):
        p.release(refs, gate)
    assert gate.calls == 1
    assert tensor_ref() is not None
    assert check_gauges(p)["retiring_pages"] == 2
    assert p.collection_stats()["candidate_page_visits"] == 0
    assert p.collection_stats()["retained_buffer_entries"] == 1


@pytest.mark.parametrize("dependency", ["page", "buffer"])
@pytest.mark.parametrize("error", [MemoryError, KeyboardInterrupt, SystemExit])
def test_interrupted_completion_query_also_quarantines_before_reuse(dependency, error):
    p = pool(1)
    ref, = p.reserve(1)
    gate = Gate(error=error("interrupted event query"))
    if dependency == "buffer":
        p.defer_buffers([torch.ones(1)], gate)
    with pytest.raises(error, match="interrupted event query"):
        p.release([ref], gate)
    assert p.poisoned and gate.calls == 1
    assert p.collect() == 0
    assert p.collection_stats()["candidate_pages"] == 1
    assert check_gauges(p)["retiring_pages"] == 1
    with pytest.raises(RuntimeError, match="quarantined"):
        p.reserve(1)
    assert gate.calls == 1


def test_descriptor_only_event_progress_has_no_page_sweep():
    p = pool(1024)
    p.reserve(1024)
    p._pages = NoLivePageScan(p._pages)
    gate = Gate()
    tensor = torch.ones(1)
    tensor_ref = weakref.ref(tensor)
    p.defer_buffers([tensor], gate)
    del tensor
    before = p.collection_stats()
    assert p.collect() == 0
    assert tensor_ref() is not None
    gate.ready = True
    assert p.collect() == 0
    assert tensor_ref() is None
    after = p.collection_stats()
    assert after["candidate_page_visits"] == before["candidate_page_visits"]
    assert after["event_queries"] - before["event_queries"] == 2


def test_recycled_slot_rejects_old_ref_and_candidate_generation():
    p = pool(1)
    old, = p.reserve(1)
    p.release([old])
    fresh, = p.reserve(1)
    assert fresh.slot == old.slot and fresh.generation > old.generation
    assert p.collection_stats()["candidate_pages"] == 0
    before = p.stats()
    with pytest.raises(StaleReference):
        p.release([old])
    with pytest.raises(StaleReference):
        p.unpin([old])
    assert p.stats() == before
    # A delayed stale candidate cannot retire an unrelated new generation.
    p._collection_candidates[old.slot] = old
    assert p.collect() == 0
    assert p._get(fresh).owners == 1
    assert p.collection_stats()["candidate_pages"] == 0
    p.release([fresh])
    assert check_gauges(p)["free_pages"] == 1


@pytest.mark.parametrize("failure", [MemoryError, KeyboardInterrupt])
def test_partial_reserve_rollback_preserves_existing_retirement_index(monkeypatch, failure):
    import head_kv.pool as module
    p = pool(5)
    old, = p.reserve(1)
    gate = Gate()
    p.release([old], gate)
    original, seen = module._Page, []

    def fail_second(ref):
        seen.append(ref)
        if len(seen) == 2:
            raise failure("metadata allocation interrupted")
        return original(ref)

    monkeypatch.setattr(module, "_Page", fail_second)
    with pytest.raises(failure):
        p.reserve(3)
    assert check_gauges(p)["free_pages"] == 4
    assert p._collection_candidates == {old.slot: old}
    monkeypatch.setattr(module, "_Page", original)
    refs = p.reserve(4)
    for failed in seen:
        new = next(ref for ref in refs if ref.slot == failed.slot)
        assert new.generation > failed.generation
    p.release(refs)
    gate.ready = True
    assert p.collect() == 1
    assert check_gauges(p)["free_pages"] == 5


def test_seal_repeated_refs_and_partial_validation_preserve_gauges():
    p = pool(3)
    first, second, third = p.reserve(3)
    p.write_page(first, torch.ones(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="uninitialized"):
        p.seal([first, first, second])
    stats = check_gauges(p)
    assert stats["live_pages"] == 1 and stats["reserved_pages"] == 2
    publish(p, [second, third])
    p.seal([first, second, third])
    assert check_gauges(p)["live_pages"] == 3


@pytest.mark.parametrize("operation", ["release", "unpin"])
def test_partial_unbalanced_operation_does_not_lose_retirement_tracking(operation):
    p = pool(1)
    ref, = p.reserve(1)
    if operation == "unpin":
        p.pin([ref])
        p.release([ref])
    with pytest.raises(RuntimeError, match="unbalanced"):
        getattr(p, operation)([ref, ref])
    # Original partial-operation behavior is retained, including collection
    # being deferred when a later decrement fails.
    assert p.collection_stats()["candidate_pages"] == 1
    assert p.collect() == 1
    assert check_gauges(p)["free_pages"] == 1


def test_collection_diagnostics_are_read_only_and_lifecycle_stats_are_stable():
    p = pool(1)
    p.reserve(1)
    first = p.collection_stats()
    assert p.collection_stats() == first
    lifecycle = p.stats()
    assert p.stats() == lifecycle
    assert p.collection_stats()["collection_calls"] == first["collection_calls"] + 2


def test_randomized_lifecycle_counters_match_independent_full_scan():
    rng = random.Random(193)
    p = pool(24)
    gates = []
    for _ in range(750):
        refs = [page.ref for page in p._pages.values()]
        ref = rng.choice(refs) if refs else None
        op = rng.randrange(9)
        if op == 0:
            try:
                p.reserve(rng.randrange(1, 4))
            except CapacityError:
                pass
        elif op == 1 and ref is not None:
            p.retain([ref])
        elif op == 2 and ref is not None:
            p.pin([ref])
        elif op in (3, 4) and ref is not None:
            page = p._get(ref)
            if (page.owners if op == 3 else page.pins) > 0:
                gate = Gate() if rng.randrange(2) else None
                if gate is not None:
                    gates.append(gate)
                (p.release if op == 3 else p.unpin)([ref], gate)
        elif op == 5 and ref is not None:
            page = p._get(ref)
            if not page.sealed and page.owners == 1 and not page.pins:
                publish(p, [ref])
        elif op == 6 and gates:
            rng.choice(gates).ready = True
        elif op == 7:
            p.release(())
        else:
            p.collect()
        check_gauges(p)
    for gate in gates:
        gate.ready = True
    for page in list(p._pages.values()):
        if page.pins:
            p.unpin([page.ref] * page.pins)
        # Unpin may reclaim an orphan, in which case there is no owner to drop.
        if page.owners:
            p.release([page.ref] * page.owners)
    assert check_gauges(p)["free_pages"] == 24
