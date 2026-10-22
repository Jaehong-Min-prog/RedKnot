"""Opt-in metadata-only alias preparation; no CUDA/QPS performance claim.

Real page ownership tests and seeded tiny-Qwen tests cover the boundary:
metadata capture grants no tensor-read lease, atomic publication rechecks
exact source roots, and actual attention/private-copy consumers stay pinned.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache"))
from head_kv import HeadKVManager, HeadPagePool, ReuseProof, SegmentPatch, SegmentWrite

from test_nonprefix_paged import runtime, _tiny_model, _head_config, _DOCS, _hash_source_pages


def _manager():
    manager = HeadKVManager(HeadPagePool(64, 4, 8))
    manager.create_request("source", context_id="source", contract="test-weights")
    manager.create_request("target", context_id="target", contract="test-weights")
    values = torch.arange(56, dtype=torch.float32).reshape(7, 8)
    writes = [SegmentWrite((0, h, "doc"), values+h, -values-h,
                           tuple(range(7)), "canonical-"+str(h)) for h in range(2)]
    manager.update("source", writes=writes, batched_writes=True)
    return manager, writes


def _aliases(root, *, occurrence="first"):
    return [(key, (key[0], key[1], occurrence), ReuseProof(
        "policy_approximate", segment.provenance, "target", "metadata-contract-test"))
        for key, segment in root.segments.items()]


def _owners(manager):
    return {page.ref: (page.owners, page.pins) for page in manager.pool._pages.values()}


def _close_manager(manager):
    for key in tuple(manager._cache):
        manager.evict(key)
    for key in tuple(manager._requests):
        manager.release_request(key)
    assert manager.pool.stats()["free_pages"] == manager.pool.capacity_pages


@pytest.mark.parametrize("cached", [False, True])
def test_capture_is_immutable_metadata_only_with_zero_pool_or_event_work(monkeypatch, cached):
    manager, _ = _manager()
    source_id = "cached" if cached else "source"
    if cached:
        manager.cache("source", source_id)
    expected = (manager._cache if cached else manager._requests)[source_id]
    before, collection = _owners(manager), manager.pool.collection_stats()
    with monkeypatch.context() as patch:
        for name in ("pin", "unpin", "retain", "release", "reserve", "read_page", "collect", "defer_buffers"):
            def forbidden(*args, _name=name, **kwargs):
                pytest.fail("metadata capture called pool."+_name)
            patch.setattr(manager.pool, name, forbidden)
        snapshots = manager.metadata_snapshots([source_id, source_id], source_is_cache=cached)
        assert list(snapshots) == [source_id]
        assert snapshots[source_id] is expected
        with pytest.raises(TypeError):
            snapshots[source_id] = None
    assert _owners(manager) == before
    assert manager.pool.collection_stats() == collection
    _close_manager(manager)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("mutation", ["remove", "rewrite", "recreate"])
def test_removed_rewritten_or_recreated_source_rejects_stale_snapshot_without_leak(cached, mutation):
    manager, writes = _manager()
    source_id = "cached" if cached else "source"
    if cached:
        manager.cache("source", source_id)
    captured = manager.metadata_snapshots([source_id], source_is_cache=cached)[source_id]
    shares = _aliases(captured)
    if mutation == "remove":
        manager.evict(source_id) if cached else manager.release_request(source_id)
    elif mutation == "rewrite":
        manager.update("source", writes=[replace(writes[0], k=writes[0].k+1)])
        if cached:
            manager.cache("source", source_id)
    else:
        if cached:
            manager.evict(source_id)
        manager.release_request("source")
        manager.create_request("source", context_id="source", contract="test-weights")
        manager.update("source", writes=writes)
        if cached:
            manager.cache("source", source_id)
        current = manager.metadata_snapshots([source_id], source_is_cache=cached)[source_id]
        old_refs = {ref for segment in captured.segments.values() for ref in segment.pages}
        new_refs = {ref for segment in current.segments.values() for ref in segment.pages}
        assert {ref.slot for ref in old_refs} == {ref.slot for ref in new_refs}
        assert old_refs.isdisjoint(new_refs)  # Reused slots have different generations.
    old, owners = manager.version("target"), _owners(manager)
    before = manager.pool.stats()
    with pytest.raises((KeyError, RuntimeError)):
        manager.share_segments_many("target", [(source_id, shares, cached)],
            expected_sources={(source_id, cached): captured})
    assert manager.version("target") is old
    assert _owners(manager) == owners
    assert manager.pool.stats() == before
    assert not manager._pending
    _close_manager(manager)


def test_pending_source_writer_snapshot_is_committed_root_not_staged_bytes():
    manager, writes = _manager()
    tx = manager.begin_update("source", patches=[SegmentPatch(
        (0, 0, "doc"), (0,), torch.full((1, 8), 999.), torch.full((1, 8), -999.), "updated")])
    captured = manager.metadata_snapshots(["source"])["source"]
    assert captured is tx.old
    result = manager.share_segments_many("target", [("source", _aliases(captured))],
        expected_sources={("source", False): captured})
    assert result.segments[0, 0, "first"].pages == captured.segments[0, 0, "doc"].pages
    tx.commit(wait=True)
    manager.release_request("source")
    with manager.bind("target") as reader:
        actual = reader.gather((0, 0, "first"))
        assert torch.equal(actual[0], writes[0].k) and torch.equal(actual[1], writes[0].v)
    _close_manager(manager)


def test_pending_writer_commit_after_capture_rejects_its_previous_snapshot():
    manager, _ = _manager()
    tx = manager.begin_update("source", patches=[SegmentPatch(
        (0, 0, "doc"), (0,), torch.ones(1, 8), torch.zeros(1, 8), "new")])
    captured = manager.metadata_snapshots(["source"])["source"]
    tx.commit(wait=True)
    owners, old = _owners(manager), manager.version("target")
    with pytest.raises(RuntimeError, match="stale source"):
        manager.share_segments_many("target", [("source", _aliases(captured))],
            expected_sources={("source", False): captured})
    assert manager.version("target") is old and _owners(manager) == owners
    _close_manager(manager)


def test_capture_materializes_iterable_before_taking_any_root_snapshot():
    manager, writes = _manager()
    old = manager.version("source")
    def source_ids():
        yield "source"
        manager.update("source", writes=[replace(writes[0], k=writes[0].k+1)])
        yield "source"
    captured = manager.metadata_snapshots(source_ids())
    assert captured["source"] is manager.version("source") and captured["source"] is not old
    _close_manager(manager)


def test_repeated_occurrences_cow_and_old_gpu_consumer_lifetime_are_unchanged():
    manager, writes = _manager()
    captured = manager.metadata_snapshots(["source", "source"])["source"]
    shares = _aliases(captured, occurrence="first")+_aliases(captured, occurrence="again")
    manager.share_segments_many("target", [("source", shares)],
        expected_sources={("source", False): captured})
    for segment in captured.segments.values():
        assert all(manager.pool._get(ref).owners == 2 for ref in segment.pages)
    reader = manager.bind("target")
    old_k, old_v = reader.gather((0, 0, "first"))
    manager.update("target", patches=[SegmentPatch(
        (0, 0, "first"), (0,), torch.full((1, 8), 99.), torch.full((1, 8), -99.), "cow")])
    current = manager.version("target")
    assert current.segments[0, 0, "first"].pages[0] != current.segments[0, 0, "again"].pages[0]
    assert current.segments[0, 1, "first"].pages == current.segments[0, 1, "again"].pages
    after_k, after_v = reader.gather((0, 0, "first"))
    assert torch.equal(old_k, after_k) and torch.equal(old_v, after_v)
    assert torch.equal(old_k, writes[0].k) and torch.equal(old_v, writes[0].v)
    class Event:
        ready = False
        def query(self): return self.ready
    event = Event()
    reader.complete(event)
    manager.release_request("source")
    manager.release_request("target")
    assert manager.pool.stats()["retiring_pages"] > 0
    assert manager.pool.stats()["free_pages"] < manager.pool.capacity_pages
    event.ready = True
    manager.pool.collect()
    _close_manager(manager)


@pytest.mark.parametrize("failure", ["bad_later_proof", "retain_failure"])
def test_failed_publication_keeps_source_and_target_ownership_atomic(monkeypatch, failure):
    manager, _ = _manager()
    captured = manager.metadata_snapshots(["source"])["source"]
    shares = _aliases(captured)
    if failure == "bad_later_proof":
        shares[-1] = (*shares[-1][:2], replace(shares[-1][2], source_provenance="wrong"))
    owners, old = _owners(manager), manager.version("target")
    with monkeypatch.context() as patch:
        if failure == "retain_failure":
            def rejected(refs):
                raise RuntimeError("injected retain validation failure")
            patch.setattr(manager.pool, "retain", rejected)
        with pytest.raises((ValueError, RuntimeError)):
            manager.share_segments_many("target", [("source", shares)],
                expected_sources={("source", False): captured})
    assert manager.version("target") is old and _owners(manager) == owners
    assert not manager._pending and not manager.pool.poisoned
    _close_manager(manager)


@pytest.mark.parametrize("ids,cache", [([None], False), ([""], False), ([1], False), (["source"], 1)])
def test_metadata_capture_invalid_arguments_and_quarantine_fail_closed(ids, cache):
    manager, _ = _manager()
    before = _owners(manager)
    with pytest.raises(ValueError):
        manager.metadata_snapshots(ids, source_is_cache=cache)
    assert _owners(manager) == before
    manager.pool.poisoned = True
    with pytest.raises(RuntimeError, match="quarantined"):
        manager.metadata_snapshots(["source"])
    manager.pool.poisoned = False  # Test-only: no device failure actually occurred.
    _close_manager(manager)


def _engine(runtime, *, flag="default", diagnostic=False, share_local=True, high_sharing=False,
            device="cpu", dtype=torch.float32, backend="torch"):
    kwargs = {} if flag == "default" else {"metadata_only_alias_sources": flag}
    config = (_head_config(runtime, window=5) if not high_sharing else runtime.config.HeadClassConfig(
        [["local", "local"]]*3, [[5, 5]]*3, 3, 2, head_sink_size=[[0, 0]]*3, default_sink_size=0))
    engine = runtime.core.NonPrefixPagedEngine(
        _tiny_model(device=device, dtype=dtype), config, capacity_pages=2048,
        page_size=256 if backend == "flash" else 4, backend=backend, model_revision="metadata-alias-test-v1",
        publication_diagnostics=diagnostic, share_local=share_local, **kwargs)
    for name, ids in _DOCS.items():
        engine.register_document(name, ids)
    return engine


def _run_engine(runtime, monkeypatch, *, flag, diagnostic=False, share_local=True, high_sharing=False,
                device="cpu", dtype=torch.float32, backend="torch"):
    engine = _engine(runtime, flag=flag, diagnostic=diagnostic, share_local=share_local, high_sharing=high_sharing,
                     device=device, dtype=dtype, backend=backend)
    sources = {doc.source_request_id for doc in engine.documents.values()}
    before = _hash_source_pages(engine)
    trace = dict(source_binds=0, alias_reads=0, publications=0)
    bind, read, publish = engine.manager.bind, engine.pool.read_page, engine.manager.share_segments_many
    alias_active = False
    alias = engine._publish_document_history
    def observed_bind(rid):
        if rid in sources:
            trace["source_binds"] += 1
        return bind(rid)
    def observed_read(*args, **kwargs):
        if alias_active:
            trace["alias_reads"] += 1
        return read(*args, **kwargs)
    def observed_publish(target, batches, **kwargs):
        trace["publications"] += 1
        assert kwargs["expected_sources"] == {
            (batch[0], False): engine.manager.version(batch[0]) for batch in batches}
        return publish(target, batches, **kwargs)
    def observed_alias(*args):
        nonlocal alias_active
        alias_active = True
        try:
            return alias(*args)
        finally:
            alias_active = False
    try:
        with monkeypatch.context() as patch:
            patch.setattr(engine.manager, "bind", observed_bind)
            patch.setattr(engine.pool, "read_page", observed_read)
            patch.setattr(engine.manager, "share_segments_many", observed_publish)
            patch.setattr(engine, "_publish_document_history", observed_alias)
            states = [engine.admit(name, prefix, ["E", "D", "E"], max_suffix_tokens=8)
                      for name, prefix in (("A", [1, 2, 3]), ("B", [9, 8, 7, 6]))]
        assert all(state.history_lease is not None and not state.history_lease.closed for state in states)
        assert engine.history_lease_acquisitions == 2
        versions = [engine.manager.version(state.manager_request_id) for state in states]
        for state, version in zip(states, versions):
            assert set(state.history_lease.refs) == {
                ref for segment in version.segments.values() for ref in segment.pages}
            for occurrence, name in enumerate(("E", "D", "E")):
                document = engine.documents[name]
                source = engine.manager.version(document.source_request_id)
                for (layer, head), source_key in document.source_keys.items():
                    pages = version.segments[layer, head, "document:"+str(occurrence)].pages
                    if share_local:
                        assert pages == source.segments[source_key].pages
                    else:
                        assert set(pages).isdisjoint(source.segments[source_key].pages)
        for key, segment in versions[0].segments.items():
            if key[2] == "global":
                assert set(segment.pages).isdisjoint(versions[1].segments[key].pages)
        roots = []
        for state in states:
            with engine.manager.bind(state.manager_request_id) as lease:
                roots.append({key: tuple(value.clone() for value in lease.gather(key))
                              for key in lease.version.segments})
        output = engine.forward(["A", "B"], torch.tensor([[81, 82], [91, 92]], device=engine.device)).clone()
        logits, decode_steps = output, []
        for _ in range(3):
            logits = engine.forward(["A", "B"], logits[:, -1].argmax(-1)[:, None]).clone()
            decode_steps.append(logits)
        decode = torch.cat(decode_steps, dim=1)
        assert _hash_source_pages(engine) == before
        stats = [state.admission_stats for state in states]
        engine_stats = engine.stats()
        for state in states:
            engine.release(state.request_id)
        assert all(state.history_lease.closed for state in states)
        assert engine.history_lease_completions == 2
        # release() can defer retirement behind an actual CUDA consumer event.
        # A closed lease is not a promise that its event has completed yet.
        engine.synchronize()
        engine.pool.collect()
        assert engine.pool.stats()["pinned_pages"] == 0
        return dict(trace=trace, roots=roots, output=output, decode=decode, stats=stats, engine_stats=engine_stats)
    finally:
        engine.close()
        assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
        assert engine.pool.stats()["pinned_pages"] == 0


@pytest.mark.parametrize("diagnostic,share_local,high_sharing", [
    (False, True, False), (True, True, False), (False, False, False),
    (True, False, False), (False, True, True),
])
def test_opt_in_bitexact_with_independent_diagnostics_and_private_copy_leases(
        runtime, monkeypatch, diagnostic, share_local, high_sharing):
    old = _run_engine(runtime, monkeypatch, flag=False, diagnostic=diagnostic,
                      share_local=share_local, high_sharing=high_sharing)
    new = _run_engine(runtime, monkeypatch, flag=True, diagnostic=diagnostic,
                      share_local=share_local, high_sharing=high_sharing)
    assert torch.equal(old["output"], new["output"]) and torch.equal(old["decode"], new["decode"])
    for a, b in zip(old["roots"], new["roots"]):
        assert a.keys() == b.keys()
        for key in a:
            assert all(torch.equal(x, y) for x, y in zip(a[key], b[key]))
    assert new["engine_stats"]["metadata_only_alias_sources"] is True
    assert "metadata_only_alias_sources" not in old["engine_stats"]
    for a, b in zip(old["stats"], new["stats"]):
        assert b.keys()-a.keys() == {"alias_source_snapshot_mode"}
        assert a.keys()-b.keys() == set()
        for field in ("alias_publications", "alias_source_roots", "alias_segment_count", "required_pages",
                      "pool_after_admit", "recovery_mode"):
            assert a[field] == b[field]
        assert ("publication_diagnostics" in b) == diagnostic
    if share_local:
        assert old["trace"]["source_binds"] == 4  # Two unique sources per request.
        assert new["trace"]["source_binds"] == 0
        assert old["trace"]["alias_reads"] == new["trace"]["alias_reads"] == 0
        assert old["trace"]["publications"] == new["trace"]["publications"] == 2
        assert all(s["alias_source_snapshot_mode"] == "metadata_only" for s in new["stats"])
    else:
        assert old["trace"] == new["trace"]
        assert new["trace"]["source_binds"] == 6 and new["trace"]["alias_reads"] > 0
        assert all(s["alias_source_snapshot_mode"] == "pinned_read_lease_private_copy" for s in new["stats"])


def test_default_is_existing_pinned_source_path_without_new_fields(runtime, monkeypatch):
    default = _run_engine(runtime, monkeypatch, flag="default")
    explicit = _run_engine(runtime, monkeypatch, flag=False)
    assert default["trace"] == explicit["trace"]
    assert default["stats"][0].keys() == explicit["stats"][0].keys()
    assert "alias_source_snapshot_mode" not in default["stats"][0]
    assert "metadata_only_alias_sources" not in default["engine_stats"]
    assert torch.equal(default["output"], explicit["output"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA paged FlashAttention")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_metadata_only_alias_sources_bitexact_and_lifecycle(runtime, monkeypatch, record_property, dtype):
    pytest.importorskip("flash_attn")
    old = _run_engine(runtime, monkeypatch, flag=False, device="cuda", dtype=dtype, backend="flash")
    new = _run_engine(runtime, monkeypatch, flag=True, device="cuda", dtype=dtype, backend="flash")
    assert len(old["roots"]) == len(new["roots"]) == 2
    for a, b in zip(old["roots"], new["roots"]):
        assert a.keys() == b.keys()
        for key in a:
            for x, y in zip(a[key], b[key]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(old["output"], new["output"], rtol=0, atol=0)
    torch.testing.assert_close(old["decode"], new["decode"], rtol=0, atol=0)
    assert old["trace"]["source_binds"] == 4 and new["trace"]["source_binds"] == 0
    assert old["trace"]["alias_reads"] == new["trace"]["alias_reads"] == 0
    assert old["trace"]["publications"] == new["trace"]["publications"] == 2
    for a, b in zip(old["stats"], new["stats"]):
        assert a["pool_after_admit"] == b["pool_after_admit"]
        assert b["alias_source_snapshot_mode"] == "metadata_only"
        assert "publication_diagnostics" not in a and "publication_diagnostics" not in b
    # All lifetime/ownership/source hashes above are checked inside _run_engine,
    # including finally cleanup. These are regression checks, not task quality.
    record_property("metadata_only_alias_policy", json.dumps("metadata-only-alias-sources-v1"))
    record_property("metadata_only_alias_parity", json.dumps(dict(
        dtype=str(dtype), requests=2, decode_steps=3,
        history_kv_bit_exact=True, query_logits_bit_exact=True, decode_logits_bit_exact=True,
        source_immutable=True, local_aliases_unchanged=True, global_pages_private=True,
        final_history_leases_retained=True, cleanup_all_pages_free=True,
        source_read_leases_omitted=True, gpu_reads_during_alias=0)))


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_engine_alias_switch_requires_real_bool(runtime, value):
    with pytest.raises(ValueError, match="metadata_only_alias_sources must be boolean"):
        _engine(runtime, flag=value)


@pytest.mark.parametrize("failure", ["publication", "final_history_bind"])
def test_enabled_admission_failure_keeps_sources_and_releases_new_references(runtime, monkeypatch, failure):
    engine = _engine(runtime, flag=True)
    source_before, before = _hash_source_pages(engine), engine.pool.stats()
    try:
        if failure == "publication":
            def rejected(*args, **kwargs):
                assert kwargs["expected_sources"]
                raise RuntimeError("injected alias failure")
            monkeypatch.setattr(engine.manager, "share_segments_many", rejected)
        else:
            original, calls = engine.manager.bind, 0
            sources = {doc.source_request_id for doc in engine.documents.values()}
            def rejected(rid):
                nonlocal calls
                if rid not in sources:
                    calls += 1
                    if calls == 2:  # Partition snapshot first, request-lifetime lease second.
                        raise RuntimeError("injected final history lease failure")
                return original(rid)
            monkeypatch.setattr(engine.manager, "bind", rejected)
        with pytest.raises(RuntimeError, match="injected"):
            engine.admit("A", [1, 2, 3], ["E", "D", "E"], max_suffix_tokens=8)
        assert not engine.requests and not engine.manager._pending
        assert _hash_source_pages(engine) == source_before
        after = engine.pool.stats()
        for field in ("free_pages", "live_pages", "reserved_pages", "retiring_pages", "pinned_pages"):
            assert after[field] == before[field]
        assert not engine.pool.poisoned
    finally:
        engine.close()
        assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
