"""Multi-source alias atomicity, COW lifetime and real admission regression."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache"))
from head_kv import HeadKVManager, HeadPagePool, ReuseProof, SegmentPatch, SegmentWrite
import head_kv.manager as manager_module

from test_nonprefix_paged import runtime, _engine, _tiny_model, _hash_source_pages


def _manager(*, position_basis="none"):
    m = HeadKVManager(HeadPagePool(64, 4, 8))
    data = {}
    for i, name in enumerate(("A", "B", "target")):
        m.create_request(name, context_id=name, contract="weights-v1", namespace="tenant")
        for head in range(2):
            key = (0, head, "doc")
            k = torch.arange(48, dtype=torch.float32).reshape(6, 8) + 100*i + 10*head
            v = -k
            data[name, head] = (k, v)
            m.update(name, writes=[SegmentWrite(key, k, v, tuple(range(6)),
                name + str(head), position_basis)])
    return m, data


def _shares(m, source, occurrence, *, cache=False, heads=(0, 1)):
    root = (m._cache if cache else m._requests)[source]
    return [((0, h, "doc"), (0, h, occurrence), ReuseProof(
        "policy_approximate", root.segments[0, h, "doc"].provenance,
        "target", "test-adapter")) for h in heads]


def _ownership(m):
    return {page.ref: (page.owners, page.pins) for page in m.pool._pages.values()}


def _gather(m, request_id, key):
    with m.bind(request_id) as lease:
        return tuple(t.clone() for t in lease.gather(key))


def test_multisource_one_publication_two_ref_scans_and_no_allocation(monkeypatch):
    m, _ = _manager()
    old = m.version("target")
    calls = []
    original_refs = manager_module._refs
    def refs(segments):
        calls.append(segments)
        return original_refs(segments)
    monkeypatch.setattr(manager_module, "_refs", refs)
    monkeypatch.setattr(m.pool, "reserve", lambda *a, **k: pytest.fail("alias allocated pages"))
    before = m.pool.written_bytes, m.pool.copied_bytes
    result = m.share_segments_many("target", [
        ("A", _shares(m, "A", "first")),
        ("B", _shares(m, "B", "second")),
        ("A", _shares(m, "A", "third")),
    ], expected_epoch=old.epoch)
    assert result.epoch == old.epoch + 1
    assert len(calls) == 2
    assert (m.pool.written_bytes, m.pool.copied_bytes) == before
    for head in range(2):
        first = result.segments[0, head, "first"]
        assert first.pages == result.segments[0, head, "third"].pages
        assert first.pages == m.version("A").segments[0, head, "doc"].pages
        assert all(m.pool._get(page).owners == 2 for page in first.pages)


@pytest.mark.parametrize("defect", [
    "missing_source", "malformed_batch", "malformed_share", "missing_key", "duplicate",
    "namespace", "contract", "provenance", "target_context", "certificate", "kind",
    "cross_head", "negative_positions", "duplicate_positions", "short_positions",
    "stale_source", "stale_target", "nonbool_cache",
])
def test_bad_later_source_leaves_no_partial_publication(defect):
    m, _ = _manager()
    shares = _shares(m, "B", "second")
    batch = ("B", shares)
    kwargs = {}
    if defect == "missing_source":
        batch = ("absent", shares)
    elif defect == "malformed_batch":
        batch = ("B",)
    elif defect == "malformed_share":
        shares[1] = ((0, 1, "doc"),)
    elif defect == "missing_key":
        shares[1] = ((8, 1, "missing"), *shares[1][1:])
    elif defect == "duplicate":
        shares[1] = (shares[1][0], (0, 1, "first"), shares[1][2])
    elif defect in ("namespace", "contract"):
        m._requests["B"] = replace(m.version("B"), **{defect: "incompatible"})
    elif defect in ("provenance", "target_context", "certificate", "kind"):
        field = {"provenance": "source_provenance"}.get(defect, defect)
        shares[1] = (*shares[1][:2], replace(shares[1][2], **{field: "" if defect == "certificate" else "bad"}))
    elif defect == "cross_head":
        shares[1] = (shares[1][0], (0, 0, "second"), shares[1][2])
    elif defect.endswith("positions"):
        positions = {"negative_positions": (-1, 1, 2, 3, 4, 5),
                     "duplicate_positions": (0, 1, 2, 3, 4, 4),
                     "short_positions": (0, 1)}[defect]
        shares[1] = (*shares[1], positions)
    elif defect == "stale_source":
        kwargs["expected_sources"] = {("A", False): m.version("A"),
                                      ("B", False): replace(m.version("B"))}
    elif defect == "stale_target":
        kwargs["expected_epoch"] = m.version("target").epoch + 1
    elif defect == "nonbool_cache":
        batch = ("B", shares, 1)
    old, owners = m.version("target"), _ownership(m)
    written = m.pool.written_bytes
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        m.share_segments_many("target", [("A", _shares(m, "A", "first")), batch], **kwargs)
    assert m.version("target") is old
    assert _ownership(m) == owners
    assert m.pool.written_bytes == written
    assert not m._pending


def test_cached_and_live_sources_with_same_name_are_distinct():
    m, data = _manager()
    m.cache("B", "A")
    cached = m._cache["A"]
    result = m.share_segments_many("target", [
        ("A", _shares(m, "A", "live")),
        ("A", _shares(m, "A", "cached", cache=True), True),
    ], expected_sources={("A", False): m.version("A"), ("A", True): cached})
    m.release_request("A")
    m.release_request("B")
    m.evict("A")
    for head in range(2):
        torch.testing.assert_close(_gather(m, "target", (0, head, "live"))[0], data["A", head][0])
        torch.testing.assert_close(_gather(m, "target", (0, head, "cached"))[0], data["B", head][0])
    assert result.segments[0, 0, "live"].pages != result.segments[0, 0, "cached"].pages
    m.release_request("target")
    assert m.stats()["free_pages"] == 64


def test_self_source_uses_captured_root_and_rejects_chained_alias():
    m, data = _manager()
    original = m.version("target")
    result = m.share_segments_many("target", [
        ("A", _shares(m, "A", "doc")),
        ("target", _shares(m, "target", "original")),
    ])
    assert result.segments[0, 0, "original"].pages == original.segments[0, 0, "doc"].pages
    torch.testing.assert_close(_gather(m, "target", (0, 0, "original"))[0], data["target", 0][0])
    old, owners = m.version("target"), _ownership(m)
    with pytest.raises(KeyError):
        m.share_segments_many("target", [
            ("A", _shares(m, "A", "new")),
            ("target", [((0, 0, "new"), (0, 0, "chained"),
                         ReuseProof("policy_approximate", "A0", "target", "explicit"))]),
        ])
    assert m.version("target") is old
    assert _ownership(m) == owners


def test_pending_target_rejected_but_pending_source_reads_committed_snapshot():
    m, data = _manager()
    patch = SegmentPatch((0, 0, "doc"), (0,), torch.ones(1, 8), torch.zeros(1, 8), "staged")
    tx_target = m.begin_update("target", patches=[patch])
    old, owners = m.version("target"), _ownership(m)
    with pytest.raises(RuntimeError, match="pending"):
        m.share_segments_many("target", [("A", _shares(m, "A", "first"))])
    assert m.version("target") is old and _ownership(m) == owners
    tx_target.abort()
    original = m.version("A")
    tx_source = m.begin_update("A", patches=[patch])
    result = m.share_segments_many("target", [("A", _shares(m, "A", "first"))],
                                   expected_sources={("A", False): original})
    tx_source.commit(wait=True)
    assert result.segments[0, 0, "first"].pages == original.segments[0, 0, "doc"].pages
    torch.testing.assert_close(_gather(m, "target", (0, 0, "first"))[0], data["A", 0][0])


def test_canonical_rope_reposition_rejected_and_lineage_preserved():
    m, _ = _manager(position_basis="rope-canonical")
    m._requests["A"] = replace(m.version("A"), segments={
        key: replace(segment, reuse_kind="policy_approximate", validity_certificate="upstream")
        for key, segment in m.version("A").segments.items()})
    old = m.version("target")
    shares = _shares(m, "B", "second")
    shares[0] = (*shares[0], tuple(range(9, 15)))
    with pytest.raises(ValueError, match="position-encoded"):
        m.share_segments_many("target", [("A", _shares(m, "A", "first")), ("B", shares)])
    assert m.version("target") is old
    result = m.share_segments_many("target", [("A", _shares(m, "A", "first"))])
    alias = result.segments[0, 0, "first"]
    assert alias.positions == tuple(range(6))
    assert alias.position_basis == "rope-canonical"
    assert alias.reuse_kind == "policy_approximate"
    assert alias.validity_certificate.startswith("lineage-v1:")


def test_exact_alias_preserves_upstream_policy_and_positions():
    m, _ = _manager()
    source = m.version("A")
    m._requests["target"] = replace(m.version("target"), context_id="A")
    segment = source.segments[0, 0, "doc"]
    m._requests["A"] = replace(source, segments={**source.segments, (0, 0, "doc"):
        replace(segment, reuse_kind="policy_approximate", validity_certificate="prior-policy")})
    proof = ReuseProof("exact_context", segment.provenance, "A")
    result = m.share_segments_many("target", [("A", [((0, 0, "doc"), (0, 0, "alias"), proof)])])
    assert result.segments[0, 0, "alias"].validity_certificate == "prior-policy"
    assert result.segments[0, 0, "alias"].reuse_kind == "policy_approximate"
    with pytest.raises(ValueError, match="identical context"):
        m.share_segments_many("target", [("A", [((0, 0, "doc"), (0, 0, "moved"), proof, tuple(range(1, 7)))])])


def test_old_readers_source_eviction_and_partial_head_cow_remain_isolated():
    m, data = _manager()
    old_reader = m.bind("target")
    result = m.share_segments_many("target", [("A", _shares(m, "A", "doc")),
                                               ("B", _shares(m, "B", "second"))])
    alias_reader = m.bind("target")
    m.cache("A", "cache-A")
    m.release_request("A")
    m.evict("cache-A")
    patch = SegmentPatch((0, 0, "doc"), (0,), torch.ones(1, 8), torch.zeros(1, 8), "repair")
    m.update("target", patches=[patch])
    repaired = m.version("target")
    assert repaired.segments[0, 0, "doc"].pages[0] != result.segments[0, 0, "doc"].pages[0]
    assert repaired.segments[0, 0, "doc"].pages[1] == result.segments[0, 0, "doc"].pages[1]
    assert repaired.segments[0, 1, "doc"].pages == result.segments[0, 1, "doc"].pages
    torch.testing.assert_close(old_reader.gather((0, 0, "doc"))[0], data["target", 0][0])
    torch.testing.assert_close(alias_reader.gather((0, 0, "doc"))[0], data["A", 0][0])
    m.release_request("target")
    m.release_request("B")
    assert m.stats()["free_pages"] < 64
    assert m.stats()["pinned_pages"] > 0
    old_reader.complete()
    alias_reader.complete()
    assert m.stats()["free_pages"] == 64


def test_empty_batches_and_single_source_apis_are_preserved():
    m, _ = _manager()
    old = m.version("target")
    assert m.share_segments_many("target", []) is old
    assert m.share_segments_many("target", [("A", [])]) is old
    assert m.share_segments("A", "target", []) is old
    one = m.share_segments("A", "target", _shares(m, "A", "first"))
    source, target, proof = _shares(m, "B", "second")[0]
    two = m.share_segment("B", "target", source, target, proof=proof)
    assert one.epoch == old.epoch + 1 and two.epoch == old.epoch + 2


def test_legacy_hashable_cache_key_and_truthy_cache_flag_are_preserved():
    m, _ = _manager()
    key = ("document", "A", 7)
    m.cache("A", key)
    shares = _shares(m, "A", "cached")
    m.release_request("A")
    result = m.share_segments(key, "target", shares, source_is_cache=1)
    assert (0, 0, "cached") in result.segments


def test_leased_source_replacement_is_rejected_even_when_provenance_stays_equal():
    m, _ = _manager()
    with m.bind("B") as old_source:
        shares = _shares(m, "B", "second")
        m.update("B", patches=[SegmentPatch((0, 0, "doc"), (0,),
            torch.ones(1, 8), torch.zeros(1, 8), "B0")])
        old, owners = m.version("target"), _ownership(m)
        with pytest.raises(RuntimeError, match="stale source"):
            m.share_segments_many("target", [("A", _shares(m, "A", "first")), ("B", shares)],
                expected_sources={("A", False): m.version("A"), ("B", False): old_source.version})
        assert m.version("target") is old
        assert _ownership(m) == owners


def test_failure_while_consuming_later_source_generator_is_atomic():
    m, _ = _manager()
    old, owners = m.version("target"), _ownership(m)
    def batches():
        yield "A", _shares(m, "A", "first")
        raise RuntimeError("source iteration failed")
    with pytest.raises(RuntimeError, match="iteration failed"):
        m.share_segments_many("target", batches())
    assert m.version("target") is old and _ownership(m) == owners


@pytest.mark.parametrize("modified", ["A", "target"])
def test_reentrant_generator_cannot_publish_over_a_changed_source_or_target(modified):
    m, _ = _manager()
    observed = {}
    def batches():
        yield "A", _shares(m, "A", "first")
        m.update(modified, patches=[SegmentPatch((0, 0, "doc"), (0,),
            torch.ones(1, 8), torch.zeros(1, 8), "intervening")])
        observed["target"] = m.version("target")
        observed["owners"] = _ownership(m)
        yield "B", _shares(m, "B", "second")
    with pytest.raises(RuntimeError, match="stale"):
        m.share_segments_many("target", batches())
    assert m.version("target") is observed["target"]
    assert _ownership(m) == observed["owners"]
    assert (0, 0, "first") not in m.version("target").segments


def test_real_admission_publishes_multidoc_once_and_pins_unique_sources(runtime, monkeypatch):
    engine = _engine(runtime, _tiny_model(), window=128)
    original_share = engine.manager.share_segments_many
    original_bind = engine.manager.bind
    publications, source_binds = [], []
    def share(target_id, batches, **kwargs):
        batches = list(batches)
        publications.append((target_id, batches, kwargs))
        return original_share(target_id, batches, **kwargs)
    def bind(request_id):
        if any(request_id == document.source_request_id for document in engine.documents.values()):
            source_binds.append(request_id)
        return original_bind(request_id)
    before_hash = _hash_source_pages(engine)
    monkeypatch.setattr(engine.manager, "share_segments_many", share)
    monkeypatch.setattr(engine.manager, "bind", bind)
    try:
        state = engine.admit("A", [1, 2, 3], ["D", "E", "D"], max_suffix_tokens=8)
        assert len(publications) == 1
        assert len(source_binds) == 2 and len(set(source_binds)) == 2
        assert state.admission_stats["alias_publications"] == 1
        assert state.admission_stats["alias_source_roots"] == 2
        assert state.admission_stats["alias_segment_count"] == 3
        root = engine.manager.version(state.manager_request_id)
        assert root.segments[1, 0, "document:0"].pages == root.segments[1, 0, "document:2"].pages
        assert root.segments[1, 0, "document:0"].positions == tuple(range(11))
        assert _hash_source_pages(engine) == before_hash
        assert len(publications[0][2]["expected_sources"]) == 2
    finally:
        engine.close()
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages


def test_real_multidoc_batch_matches_sequential_alias_baseline(runtime, monkeypatch):
    model = _tiny_model()
    results = []
    for sequential in (True, False):
        engine = _engine(runtime, model, window=128)
        if sequential:
            original = engine.manager.share_segments_many
            def sequential_share(target_id, batches, **kwargs):
                result = engine.manager.version(target_id)
                for batch in batches:
                    result = original(target_id, [batch], **kwargs)
                return result
            monkeypatch.setattr(engine.manager, "share_segments_many", sequential_share)
        try:
            engine.admit("A", [1, 2, 3], ["D", "E", "D"], max_suffix_tokens=8)
            engine.admit("B", [9, 8], ["E", "D"], max_suffix_tokens=8)
            outputs = [engine.forward(["A", "B"], torch.tensor([[81, 82, 83], [91, 92, 93]])).clone(),
                       engine.forward(["B", "A"], torch.tensor([[94], [84]])).clone()]
            results.append(outputs)
        finally:
            engine.close()
    for before, after in zip(*results):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_real_admission_failed_later_doc_cleans_target_and_all_source_pins(runtime, monkeypatch):
    engine = _engine(runtime, _tiny_model(), window=128)
    before = engine.pool.stats()
    source_versions = {name: engine.manager.version(doc.source_request_id)
                       for name, doc in engine.documents.items()}
    original = engine.manager.share_segments_many
    def invalid_later_doc(target_id, batches, **kwargs):
        batches = [(source, list(shares)) for source, shares in batches]
        source, shares = batches[-1]
        shares[-1] = (*shares[-1][:2], replace(shares[-1][2], source_provenance="wrong"))
        return original(target_id, batches, **kwargs)
    monkeypatch.setattr(engine.manager, "share_segments_many", invalid_later_doc)
    try:
        with pytest.raises(ValueError, match="proof"):
            engine.admit("A", [1, 2, 3], ["D", "E", "D"], max_suffix_tokens=8)
        assert not engine.requests
        assert all(engine.manager.version(engine.documents[name].source_request_id) is version
                   for name, version in source_versions.items())
        after = engine.pool.stats()
        assert after["free_pages"] == before["free_pages"]
        assert after["pinned_pages"] == before["pinned_pages"] == 0
        assert not engine.manager._pending
    finally:
        engine.close()


def test_engine_exposes_collection_work_without_an_extra_collection_pass(runtime, monkeypatch):
    engine = _engine(runtime, _tiny_model())
    try:
        before = engine.pool.collection_stats()
        result = engine.stats()
        after = engine.pool.collection_stats()
        assert result["pool_collection_stats"] == after
        # Existing engine.pool.stats() collects once. Reading the new work
        # counters adds no further collection pass or lifetime mutation.
        assert after["collection_calls"] == before["collection_calls"] + 1
        assert result["multi_document_alias_publication"] is True
        monkeypatch.setattr(engine.pool, "collection_stats", None)
        assert "pool_collection_stats" not in engine.stats()
    finally:
        engine.close()
