"""Atomic certified multi-head sharing, independent of a model/GPU server."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache"))
from head_kv import HeadKVManager, HeadPagePool, ReuseProof, SegmentWrite
import head_kv.manager as manager_module


def make_manager(heads=3, *, position_basis="rope-canonical", upstream="exact_context"):
    pool = HeadPagePool(max(128, heads*4), 4, 8)
    manager = HeadKVManager(pool)
    manager.create_request("source", context_id="document", contract="weights-v1", namespace="test")
    writes = []
    for h in range(heads):
        value = torch.arange(48, dtype=torch.float32).reshape(6, 8) + 100*h
        writes.append(SegmentWrite((0, h, "D"), value, -value, tuple(range(6)), "doc-provenance",
            position_basis=position_basis, reuse_kind=upstream,
            validity_certificate="upstream-policy" if upstream != "exact_context" else ""))
    manager.update("source", writes=writes)
    return manager


def add_target(manager, name="target", *, context="request", contract="weights-v1", namespace="test"):
    manager.create_request(name, context_id=context, contract=contract, namespace=namespace)


def aliases(heads=3, *, context="request", kind="policy_approximate", occurrence="occurrence", positions=None):
    proof = ReuseProof(kind, "doc-provenance", context, "canonical-query-counterrotation-v1")
    return [((0, h, "D"), (0, h, occurrence), proof, positions) for h in range(heads)]


def owners(manager):
    return {slot: (page.ref, page.owners, page.pins, len(page.events), page.sealed)
            for slot, page in manager.pool._pages.items()}


@pytest.mark.parametrize("upstream", ["exact_context", "certified_transform", "policy_approximate"])
@pytest.mark.parametrize("kind", ["exact_context", "certified_transform", "policy_approximate"])
def test_bulk_and_scalar_have_identical_segments_lineage_and_ownership(upstream, kind):
    manager = make_manager(upstream=upstream)
    context = "document" if kind == "exact_context" else "request"
    add_target(manager, "scalar", context=context)
    add_target(manager, "bulk", context=context)
    shares = aliases(context=context, kind=kind)
    before = manager.pool.stats()
    for source_key, target_key, proof, positions in shares:
        manager.share_segment("source", "scalar", source_key, target_key,
                              proof=proof, positions=positions)
    result = manager.share_segments("source", "bulk", shares)
    assert dict(result.segments) == dict(manager.version("scalar").segments)
    assert result.epoch == 1
    assert manager.version("scalar").epoch == len(shares)
    assert result.generation == manager.version("bulk").generation
    assert all(page.owners == 3 for page in manager.pool._pages.values())
    after = manager.pool.stats()
    assert after["allocated_bytes"] == before["allocated_bytes"]
    assert after["written_bytes"] == before["written_bytes"]
    assert after["copied_bytes"] == before["copied_bytes"]
    with manager.bind("bulk") as lease:
        for h in range(3):
            expected = torch.arange(48, dtype=torch.float32).reshape(6, 8)+100*h
            actual = lease.gather((0, h, "occurrence"))
            torch.testing.assert_close(actual[0], expected)
            torch.testing.assert_close(actual[1], -expected)
    for name in ("bulk", "scalar", "source"):
        manager.release_request(name)
    assert manager.pool.stats()["free_pages"] == manager.pool.capacity_pages


@pytest.mark.parametrize("bad", ["provenance", "context", "kind", "certificate", "positions-length",
    "positions-repeat", "positions-negative", "encoded-move", "head", "key-shape", "missing-source", "duplicate"])
def test_invalid_last_alias_is_atomic_before_any_ownership_change(bad, monkeypatch):
    manager = make_manager()
    add_target(manager)
    valid = aliases()[0]
    source_key, target_key, proof, positions = aliases()[1]
    if bad == "provenance":
        proof = ReuseProof(proof.kind, "different", proof.target_context, proof.certificate)
    elif bad == "context":
        proof = ReuseProof(proof.kind, proof.source_provenance, "different", proof.certificate)
    elif bad == "kind":
        proof = ReuseProof("unknown", proof.source_provenance, proof.target_context, proof.certificate)
    elif bad == "certificate":
        proof = ReuseProof(proof.kind, proof.source_provenance, proof.target_context, "")
    elif bad == "positions-length":
        positions = (0, 1)
    elif bad == "positions-repeat":
        positions = (0, 1, 2, 3, 4, 4)
    elif bad == "positions-negative":
        positions = (-1, 0, 1, 2, 3, 4)
    elif bad == "encoded-move":
        positions = tuple(range(20, 26))
    elif bad == "head":
        target_key = (0, 2, "occurrence")
    elif bad == "key-shape":
        target_key = (0, 1, "occurrence", "bad")
    elif bad == "missing-source":
        source_key = (0, 1, "missing")
    elif bad == "duplicate":
        source_key, target_key, proof, positions = valid
    old, snapshot = manager.version("target"), owners(manager)
    retain_calls = []
    original = manager.pool.retain
    monkeypatch.setattr(manager.pool, "retain", lambda refs: (retain_calls.append(tuple(refs)), original(refs))[1])
    with pytest.raises((ValueError, KeyError)):
        manager.share_segments("source", "target", [valid, (source_key, target_key, proof, positions)])
    assert manager.version("target") is old
    assert owners(manager) == snapshot
    assert retain_calls == []


@pytest.mark.parametrize("mismatch", ["namespace", "contract"])
def test_contract_namespace_mismatch_fails_closed(mismatch):
    manager = make_manager()
    add_target(manager, **{mismatch: "different"})
    old, snapshot = manager.version("target"), owners(manager)
    with pytest.raises(ValueError, match="namespace/model contract"):
        manager.share_segments("source", "target", aliases())
    assert manager.version("target") is old
    assert owners(manager) == snapshot


def test_exact_proof_cannot_change_context():
    manager = make_manager()
    add_target(manager)
    with pytest.raises(ValueError, match="exact reuse requires"):
        manager.share_segments("source", "target", aliases(kind="exact_context"))
    assert not manager.version("target").segments


def test_duplicate_occurrences_and_repeated_publication_own_each_page_once():
    manager = make_manager(heads=1)
    add_target(manager)
    proof = aliases(1)[0][2]
    manager.share_segments("source", "target", [((0, 0, "D"), (0, 0, "A"), proof),
                                                 ((0, 0, "D"), (0, 0, "B"), proof)])
    assert all(page.owners == 2 for page in manager.pool._pages.values())
    snapshot = owners(manager)
    manager.share_segments("source", "target", [((0, 0, "D"), (0, 0, "C"), proof)])
    assert owners(manager) == snapshot
    assert len(manager.version("target").segments) == 3
    manager.release_request("source")
    manager.release_request("target")
    assert manager.pool.stats()["free_pages"] == manager.pool.capacity_pages


def test_old_read_lease_survives_bulk_replacement_and_root_release():
    manager = make_manager(heads=2)
    add_target(manager)
    old_k, old_v = torch.full((5, 8), 999.), torch.full((5, 8), -999.)
    manager.update("target", writes=[SegmentWrite((0, 0, "occurrence"), old_k, old_v,
                                                   tuple(range(5)), "old-target")])
    old = manager.bind("target")
    manager.share_segments("source", "target", aliases(2))
    torch.testing.assert_close(old.gather((0, 0, "occurrence"))[0], old_k)
    manager.release_request("source")
    manager.release_request("target")
    assert manager.pool.stats()["pinned_pages"] == 2
    torch.testing.assert_close(old.gather((0, 0, "occurrence"))[1], old_v)
    old.complete()
    assert manager.pool.stats()["free_pages"] == manager.pool.capacity_pages


def test_cached_source_survives_request_release_and_cache_eviction_with_live_lease():
    manager = make_manager(heads=1)
    manager.cache("source", "cache-key")
    manager.release_request("source")
    add_target(manager)
    manager.share_segments("cache-key", "target", aliases(1), source_is_cache=True)
    lease = manager.bind("target")
    manager.evict("cache-key")
    manager.release_request("target")
    assert lease.gather((0, 0, "occurrence"))[0].shape == (6, 8)
    assert manager.pool.stats()["pinned_pages"] == 2
    lease.complete()
    assert manager.pool.stats()["free_pages"] == manager.pool.capacity_pages


def test_same_root_resolves_all_aliases_against_original_snapshot():
    manager = make_manager(heads=1)
    manager.update("source", writes=[SegmentWrite((0, 0, "E"), torch.ones(6, 8), torch.ones(6, 8),
                                                   tuple(range(6)), "second-document")])
    old = manager.version("source")
    shares = [((0, 0, "D"), (0, 0, "E"), ReuseProof("exact_context", "doc-provenance", "document")),
              ((0, 0, "E"), (0, 0, "D"), ReuseProof("exact_context", "second-document", "document"))]
    snapshot = owners(manager)
    new = manager.share_segments("source", "source", shares)
    assert new.segments[(0, 0, "D")].pages == old.segments[(0, 0, "E")].pages
    assert new.segments[(0, 0, "E")].pages == old.segments[(0, 0, "D")].pages
    assert owners(manager) == snapshot
    assert new.epoch == old.epoch+1


def test_unencoded_positions_can_move_with_explicit_transform_certificate():
    manager = make_manager(position_basis="none")
    add_target(manager)
    manager.share_segments("source", "target", aliases(positions=tuple(range(30, 36)), kind="certified_transform"))
    for segment in manager.version("target").segments.values():
        assert segment.positions == tuple(range(30, 36))
        assert segment.reuse_kind == "certified_transform"


def test_empty_batch_noop_pending_writer_rejected_and_poisoned_pool_rejected():
    manager = make_manager()
    add_target(manager)
    old, snapshot = manager.version("target"), owners(manager)
    assert manager.share_segments("source", "target", []) is old
    assert owners(manager) == snapshot
    tx = manager.begin_update("target")
    with pytest.raises(RuntimeError, match="pending writer"):
        manager.share_segments("source", "target", aliases())
    tx.abort()
    manager.pool.poisoned = True
    with pytest.raises(RuntimeError, match="quarantined"):
        manager.share_segments("source", "target", aliases())


def test_retain_failure_preserves_root_and_reference_counts(monkeypatch):
    manager = make_manager()
    add_target(manager)
    old, snapshot = manager.version("target"), owners(manager)
    def fail(refs):
        raise RuntimeError("injected retain failure")
    monkeypatch.setattr(manager.pool, "retain", fail)
    with pytest.raises(RuntimeError, match="injected retain"):
        manager.share_segments("source", "target", aliases())
    assert manager.version("target") is old
    assert owners(manager) == snapshot


def test_publication_failure_rolls_back_added_owners(monkeypatch):
    manager = make_manager()
    add_target(manager)
    old, snapshot = manager.version("target"), owners(manager)
    class RejectPublish(dict):
        def __setitem__(self, key, value):
            if key == "target":
                raise MemoryError("injected root publication failure")
            return super().__setitem__(key, value)
    monkeypatch.setattr(manager, "_requests", RejectPublish(manager._requests))
    with pytest.raises(MemoryError, match="injected root publication"):
        manager.share_segments("source", "target", aliases())
    assert manager.version("target") is old
    assert owners(manager) == snapshot


def test_bulk_performs_linear_root_scans_and_one_publication(monkeypatch):
    heads = 64
    manager = make_manager(heads=heads)
    add_target(manager, "bulk")
    add_target(manager, "scalar")
    visits = []
    original = manager_module._refs
    def count(segments):
        visits.append(len(segments))
        return original(segments)
    monkeypatch.setattr(manager_module, "_refs", count)
    manager.share_segments("source", "bulk", aliases(heads))
    bulk_visits = sum(visits)
    visits.clear()
    for source_key, target_key, proof, _ in aliases(heads):
        manager.share_segment("source", "scalar", source_key, target_key, proof=proof)
    scalar_visits = sum(visits)
    assert bulk_visits == heads
    assert scalar_visits == heads*heads
    assert dict(manager.version("bulk").segments) == dict(manager.version("scalar").segments)
    assert manager.version("bulk").epoch == 1
    assert manager.version("scalar").epoch == heads
