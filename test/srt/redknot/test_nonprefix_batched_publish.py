"""Opt-in segment publication: exact payloads, immutable roots and retirement."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache"))
from head_kv import HeadKVManager, HeadPagePool, SegmentWrite, SegmentPatch
from head_kv.pool import StaleReference
import head_kv.manager as manager_module


def _payload(rows, dim=8, *, dtype=torch.float32, device="cpu"):
    # Deliberately non-contiguous, as a head view of token-major model output.
    k = torch.arange(rows*dim*2, dtype=torch.float32, device=device).reshape(rows, dim*2)[:, ::2]
    v = torch.arange(rows*dim, dtype=torch.float32, device=device).reshape(dim, rows).T
    return k.to(dtype=dtype), -v.to(dtype=dtype)


@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("rows", [1, 4, 5, 11, 12])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_segment_write_matches_payload_extent_and_never_touches_padding(fragmented, rows, dtype):
    pool = HeadPagePool(8, 4, 8, dtype=dtype)
    pool.k.fill_(-123)
    pool.v.fill_(-321)
    allocated = pool.reserve(6)
    count = (rows+3)//4
    refs = tuple(allocated[i] for i in ([5, 1, 3] if fragmented else [2, 3, 4])[:count])
    k, v = _payload(rows, dtype=torch.float64)
    expected_k, expected_v = pool.k.clone(), pool.v.clone()
    for i, ref in enumerate(refs):
        n = min(4, rows-i*4)
        expected_k[ref.slot, :n] = k[i*4:i*4+n]
        expected_v[ref.slot, :n] = v[i*4:i*4+n]
    pool.write_segment_pages(refs, k, v)
    assert pool.written_bytes == 2*rows*8*pool.k.element_size()
    assert pool.copied_bytes == 0
    assert [pool._get(ref).extent for ref in refs] == [min(4, rows-i*4) for i in range(count)]
    assert all(not pool._get(ref).sealed for ref in refs)
    torch.testing.assert_close(pool.k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(pool.v, expected_v, rtol=0, atol=0)
    pool.seal(refs)
    actual_k = torch.cat([pool.read_page(ref)[0] for ref in refs])
    actual_v = torch.cat([pool.read_page(ref)[1] for ref in refs])
    torch.testing.assert_close(actual_k, k.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(actual_v, v.to(dtype), rtol=0, atol=0)
    pool.release(allocated)
    assert pool.stats()["free_pages"] == 8


@pytest.mark.parametrize("fault", ["sealed", "shared", "pinned", "stale", "duplicate",
    "too_few", "too_many", "shape", "empty_nonzero_refs", "sparse", "meta", "poisoned"])
def test_all_destination_and_payload_validation_precedes_any_write(fault):
    pool = HeadPagePool(8, 4, 8)
    pool.k.fill_(51)
    pool.v.fill_(52)
    refs = list(pool.reserve(3))
    k, v = _payload(11)
    if fault == "sealed":
        pool.write_page(refs[-1], *(_payload(1)))
        pool.seal([refs[-1]])
    elif fault == "shared":
        pool.retain([refs[-1]])
    elif fault == "pinned":
        pool.pin([refs[-1]])
    elif fault == "stale":
        pool.release([refs[-1]])
        pool.reserve(1)
    elif fault == "duplicate":
        refs[-1] = refs[0]
    elif fault == "too_few":
        refs.pop()
    elif fault == "too_many":
        refs.extend(pool.reserve(1))
    elif fault == "shape":
        v = v[:, :-1]
    elif fault == "empty_nonzero_refs":
        k, v = _payload(0)
    elif fault == "sparse":
        k = k.to_sparse()
    elif fault == "meta":
        k, v = k.to("meta"), v.to("meta")
    elif fault == "poisoned":
        pool.poisoned = True
    before_k, before_v, before_bytes = pool.k.clone(), pool.v.clone(), pool.written_bytes
    extents = {slot: page.extent for slot, page in pool._pages.items()}
    with pytest.raises((ValueError, RuntimeError, StaleReference)):
        pool.write_segment_pages(refs, k, v)
    torch.testing.assert_close(pool.k, before_k, rtol=0, atol=0)
    torch.testing.assert_close(pool.v, before_v, rtol=0, atol=0)
    assert pool.written_bytes == before_bytes
    assert {slot: page.extent for slot, page in pool._pages.items()} == extents


def test_empty_segment_is_a_true_noop():
    pool = HeadPagePool(8, 4, 8)
    before = pool.stats()
    pool.write_segment_pages([], *_payload(0))
    assert pool.stats() == before


@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("cross_kv", [False, True])
def test_explicit_borrowed_boundary_payload_is_staged_without_altering_source(fragmented, cross_kv):
    pool = HeadPagePool(8, 4, 8)
    refs = pool.reserve(6)
    source = refs[:2]
    pool.write_segment_pages(source, *_payload(7))
    pool.seal(source)
    pool.pin(source)
    k = pool.k.view(-1, 8)[1:7]  # Explicit six-row payload, not a history gather.
    v = pool.v.view(-1, 8)[1:7]
    if cross_kv:
        k, v = v, k
    expected_k, expected_v = k.clone(), v.clone()
    source_k, source_v = pool.k[:2].clone(), pool.v[:2].clone()
    dest = (refs[5], refs[3]) if fragmented else refs[2:4]
    before = pool.written_bytes
    pool.write_segment_pages(dest, k, v)
    pool.seal(dest)
    torch.testing.assert_close(torch.cat([pool.read_page(r)[0] for r in dest]), expected_k)
    torch.testing.assert_close(torch.cat([pool.read_page(r)[1] for r in dest]), expected_v)
    torch.testing.assert_close(pool.k[:2], source_k)
    torch.testing.assert_close(pool.v[:2], source_v)
    assert pool.written_bytes-before == 2*6*8*pool.k.element_size()
    pool.unpin(source)
    pool.release(refs)


def _manager():
    pool = HeadPagePool(64, 4, 8)
    manager = HeadKVManager(pool)
    manager.create_request("r", context_id="context", contract="test", namespace="test")
    return manager


def _writes(offset=0):
    return [SegmentWrite((0, h, "D"), *[tensor+offset for tensor in _payload(rows)],
                         tuple(range(rows)), f"source-{offset}")
            for h, rows in enumerate((11, 5, 0))]


@pytest.mark.parametrize("batched", [False, True])
def test_manager_full_write_optin_keeps_publication_and_patch_contract(batched, monkeypatch):
    manager = _manager()
    calls = dict(page=0, segment=0)
    for name, key in (("write_page", "page"), ("write_segment_pages", "segment")):
        native = getattr(manager.pool, name)
        def observed(*args, _native=native, _key=key, **kwargs):
            calls[_key] += 1
            return _native(*args, **kwargs)
        monkeypatch.setattr(manager.pool, name, observed)
    options = {"batched_writes": True} if batched else {}  # Exercise untouched default.
    first = manager.update("r", writes=_writes(), **options)
    assert calls == (dict(page=0, segment=3) if batched else dict(page=5, segment=0))
    old = manager.bind("r")
    expected = old.gather((0, 0, "D"))
    patch_k, patch_v = _payload(2)
    # Replacing one head and repairing a different head remain one transaction.
    tx = manager.begin_update("r", writes=[_writes(100)[0]], patches=[SegmentPatch(
        (0, 1, "D"), (0, 4), patch_k, patch_v, "repaired")], remove=[(0, 2, "D")], **options)
    assert manager.version("r") is first
    result = tx.commit(wait=True)
    assert result.epoch == first.epoch+1
    torch.testing.assert_close(old.gather((0, 0, "D"))[0], expected[0])
    with manager.bind("r") as current:
        torch.testing.assert_close(current.gather((0, 0, "D"))[0], expected[0]+100)
        k, v = current.gather((0, 1, "D"))
        torch.testing.assert_close(k[[0, 4]], patch_k)
        torch.testing.assert_close(v[[0, 4]], patch_v)
        assert (0, 2, "D") not in current.version.segments
        assert old.gather((0, 2, "D"))[0].shape == (0, 8)
    old.complete()
    manager.release_request("r")
    assert manager.pool.stats()["free_pages"] == 64


def test_later_segment_copy_failure_keeps_old_root_and_retires_reserved_pages(monkeypatch):
    manager = _manager()
    old = manager.update("r", writes=_writes())
    before = manager.pool.stats()
    snapshot = manager.bind("r")
    expected = snapshot.gather((0, 0, "D"))[0].clone()
    native = manager.pool.write_segment_pages
    calls = []
    class Gate:
        ready = False
        def query(self):
            return self.ready
        def synchronize(self):
            self.ready = True
    gate = Gate()
    def interrupted(*args):
        calls.append(tuple(args[0]))
        native(*args)
        if len(calls) == 2:
            raise RuntimeError("injected after batched device copy")
    monkeypatch.setattr(manager.pool, "write_segment_pages", interrupted)
    monkeypatch.setattr(manager_module, "completion_event", lambda device: gate)
    with pytest.raises(RuntimeError, match="injected"):
        manager.update("r", writes=_writes(100), batched_writes=True)
    assert manager.version("r") is old
    assert manager.stats()["pending_transactions"] == 0
    assert not manager.pool.poisoned
    torch.testing.assert_close(snapshot.gather((0, 0, "D"))[0], expected)
    assert manager.pool.stats()["free_pages"] == before["free_pages"]-5
    gate.ready = True
    manager.pool.collect()
    assert manager.pool.stats()["free_pages"] == before["free_pages"]
    snapshot.complete()
    manager.release_request("r")
    assert manager.pool.stats()["free_pages"] == 64


def test_invalid_late_write_prevalidation_leaves_no_reservation():
    manager = _manager()
    old = manager.version("r")
    before = manager.pool.stats()
    writes = _writes()
    last = writes[-1]
    writes[-1] = SegmentWrite(last.key, last.k, last.v, (1,), last.provenance)
    with pytest.raises(ValueError, match="positions"):
        manager.update("r", writes=writes, batched_writes=True)
    assert manager.version("r") is old
    assert manager.pool.stats() == before


def test_failure_between_k_and_v_copy_is_unpublished_and_event_retired(monkeypatch):
    manager = _manager()
    old = manager.update("r", writes=[_writes()[0]])
    lease = manager.bind("r")
    expected = tuple(x.clone() for x in lease.gather((0, 0, "D")))
    free_before = manager.pool.stats()["free_pages"]
    class Gate:
        ready = False
        def query(self):
            return self.ready
    gate = Gate()
    reservations = []
    reserve = manager.pool.reserve
    def observed_reserve(*args, **kwargs):
        result = reserve(*args, **kwargs)
        reservations.extend(result)
        return result
    native_copy = torch.Tensor.copy_
    copied_k = []
    def fail_v(tensor, source, *args, **kwargs):
        pointer = tensor.untyped_storage().data_ptr()
        if pointer == manager.pool.v.untyped_storage().data_ptr():
            raise RuntimeError("injected before V copy")
        if pointer == manager.pool.k.untyped_storage().data_ptr():
            copied_k.append(tensor.clone())
        return native_copy(tensor, source, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(manager.pool, "reserve", observed_reserve)
        patch.setattr(manager_module, "completion_event", lambda device: gate)
        patch.setattr(torch.Tensor, "copy_", fail_v)
        with pytest.raises(RuntimeError, match="before V copy"):
            manager.update("r", writes=[_writes(100)[0]], batched_writes=True)
    assert len(copied_k) == 1 and len(reservations) == 3
    assert manager.version("r") is old
    assert not manager._pending and not manager.pool.poisoned
    assert manager.pool.stats()["free_pages"] == free_before-3
    assert all(not manager.pool._get(ref).sealed and manager.pool._get(ref).extent == 0
               and manager.pool._get(ref).owners == 0 for ref in reservations)
    for actual, before in zip(lease.gather((0, 0, "D")), expected):
        torch.testing.assert_close(actual, before)
    gate.ready = True
    manager.pool.collect()
    assert manager.pool.stats()["free_pages"] == free_before
    lease.complete()
    manager.release_request("r")
    assert manager.pool.stats()["free_pages"] == 64


@pytest.mark.parametrize("failure", ["create", "query"])
def test_completion_failure_preserves_existing_quarantine_contract(monkeypatch, failure):
    manager = _manager()
    old = manager.update("r", writes=[_writes()[0]])
    class BrokenEvent:
        def query(self):
            raise RuntimeError("injected event query failure")
        def synchronize(self):
            pass
    def event(device):
        if failure == "create":
            raise RuntimeError("injected event allocation failure")
        return BrokenEvent()
    monkeypatch.setattr(manager_module, "completion_event", event)
    with pytest.raises(RuntimeError, match="injected"):
        manager.update("r", writes=[_writes(100)[0]], batched_writes=True)
    assert manager.pool.poisoned
    assert manager._requests["r"] is old
    with pytest.raises(RuntimeError, match="quarantined"):
        manager.version("r")
    assert all(manager.pool._get(ref).owners >= 1 for seg in old.segments.values() for ref in seg.pages)
    with pytest.raises(RuntimeError, match="quarantined"):
        manager.pool.reserve(1)


def test_batched_abort_and_stale_epoch_preserve_existing_root():
    manager = _manager()
    old = manager.update("r", writes=_writes())
    free_before = manager.pool.stats()["free_pages"]
    with pytest.raises(RuntimeError, match="stale table epoch"):
        manager.begin_update("r", writes=_writes(100), expected_epoch=old.epoch-1, batched_writes=True)
    assert manager.pool.stats()["free_pages"] == free_before
    tx = manager.begin_update("r", writes=_writes(100), expected_epoch=old.epoch, batched_writes=True)
    assert manager.version("r") is old
    tx.abort()
    assert manager.version("r") is old
    assert manager.pool.stats()["free_pages"] == free_before
    manager.release_request("r")
    assert manager.pool.stats()["free_pages"] == 64


@pytest.mark.parametrize("fragmented", [False, True])
def test_one_device_copy_per_kv_tensor_independent_of_page_count(fragmented):
    pool = HeadPagePool(64, 4, 8)
    reserved = pool.reserve(64)
    refs = reserved[::2] if fragmented else reserved[:32]
    k, v = _payload(127)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        pool.write_segment_pages(refs, k, v)
    counts = {item.key: item.count for item in trace.key_averages()}
    if fragmented:
        assert counts.get("aten::index_copy_", 0) == 2
    else:
        assert counts.get("aten::copy_", 0) == 2
        assert counts.get("aten::index_copy_", 0) == 0
    assert pool.copied_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_segment_payload_and_nonblocking_transaction_retirement(fragmented, dtype):
    pool = HeadPagePool(8, 256, 64, device="cuda:0", dtype=dtype)
    refs = pool.reserve(5)
    destination = (refs[4], refs[1], refs[3]) if fragmented else refs[:3]
    k, v = [x/4 for x in _payload(611, dim=64, device="cuda:0")]
    pool.write_segment_pages(destination, k, v)
    pool.seal(destination)
    torch.testing.assert_close(torch.cat([pool.read_page(r)[0] for r in destination]), k.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([pool.read_page(r)[1] for r in destination]), v.to(dtype), rtol=0, atol=0)
    assert pool.written_bytes == 2*611*64*pool.k.element_size()
    from head_kv.pool import completion_event
    event = completion_event(pool.device)
    pool.release(refs, event)
    event.synchronize()
    pool.collect()
    assert pool.stats()["free_pages"] == 8
