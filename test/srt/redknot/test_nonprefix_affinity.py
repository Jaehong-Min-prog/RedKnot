"""Affinity may change service order, never canonical identity or COW authority.

CPU correctness tests only; synthetic timings are not performance evidence.
"""
from pathlib import Path
import importlib
import sys
import types

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache"))
from head_kv import HeadKVManager, HeadPagePool, ReuseProof, SegmentPatch, SegmentWrite
from test_nonprefix_scheduler import Clock, FakeEngine, _isolated_imports
from test_nonprefix_paged import _isolated_imports as _core_imports, _tiny_model, _head_config


def _manager():
    manager = HeadKVManager(HeadPagePool(32, 4, 8))
    contract = "model=weights-v1;heads=profile-v1;rope=default;ffn=full"
    manager.create_request("document", context_id="position-zero", namespace="tenant-A", contract=contract)
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    manager.update("document", writes=[SegmentWrite(
        (0, head, "canonical"), data + head*100, -data-head*100, tuple(range(8)),
        "document-contents", position_basis="rope-canonical") for head in range(2)])
    return manager, data, contract


def _alias(manager, name, contract, offset, *, namespace="tenant-A"):
    manager.create_request(name, context_id=f"private-prefix-{name}", namespace=namespace, contract=contract)
    proof = ReuseProof("policy_approximate", "document-contents", f"private-prefix-{name}",
                       f"canonical-query-counterrotation-v1:offset={offset}")
    return manager.share_segments("document", name, [
        ((0, head, "canonical"), (0, head, "nonprefix-document"), proof) for head in range(2)])


@pytest.mark.parametrize("mismatch", ["tenant", "weights", "heads", "rope", "ffn"])
def test_same_document_name_and_bytes_do_not_authorize_cross_contract_sharing(mismatch):
    manager, _, contract = _manager()
    namespace = "tenant-B" if mismatch == "tenant" else "tenant-A"
    changed = {"weights": ("weights-v1", "weights-v2"), "heads": ("profile-v1", "profile-v2"),
               "rope": ("rope=default", "rope=yarn"), "ffn": ("ffn=full", "ffn=sparse")}
    if mismatch in changed:
        contract = contract.replace(*changed[mismatch])
    before = manager.pool.stats()
    with pytest.raises(ValueError, match="namespace/model contract"):
        _alias(manager, "other", contract, 13, namespace=namespace)
    assert not manager.version("other").segments
    assert manager.pool.stats() == before


def test_two_nonprefix_offsets_share_canonical_pages_and_cow_only_touched_head_pages():
    manager, data, contract = _manager()
    source = manager.version("document")
    a = _alias(manager, "A", contract, 5)
    b = _alias(manager, "B", contract, 17)
    for head in range(2):
        canonical = source.segments[0, head, "canonical"]
        assert a.segments[0, head, "nonprefix-document"].pages == canonical.pages
        assert b.segments[0, head, "nonprefix-document"].pages == canonical.pages
        assert a.segments[0, head, "nonprefix-document"].positions == tuple(range(8))
    old_a = manager.bind("A")
    before = manager.pool.stats()
    replacement_k, replacement_v = torch.full((1, 8), 999.), torch.full((1, 8), -999.)
    manager.update("A", patches=[SegmentPatch((0, 1, "nonprefix-document"), (1,),
                        replacement_k, replacement_v, "A-private-repair")])
    now_a = manager.version("A")
    assert now_a.segments[0, 0, "nonprefix-document"].pages == a.segments[0, 0, "nonprefix-document"].pages
    assert now_a.segments[0, 1, "nonprefix-document"].pages[0] != a.segments[0, 1, "nonprefix-document"].pages[0]
    assert now_a.segments[0, 1, "nonprefix-document"].pages[1] == a.segments[0, 1, "nonprefix-document"].pages[1]
    assert manager.pool.stats()["live_pages"] == before["live_pages"] + 1
    with manager.bind("document") as document, manager.bind("B") as other, manager.bind("A") as repaired:
        for head in range(2):
            expected = data+head*100
            torch.testing.assert_close(document.gather((0, head, "canonical"))[0], expected)
            torch.testing.assert_close(other.gather((0, head, "nonprefix-document"))[0], expected)
            torch.testing.assert_close(old_a.gather((0, head, "nonprefix-document"))[0], expected)
        assert torch.equal(repaired.gather((0, 1, "nonprefix-document"))[0][1], replacement_k[0])
    for name in ("A", "B", "document"):
        manager.release_request(name)
    assert manager.pool.stats()["pinned_pages"] == 4
    old_a.complete()
    assert manager.pool.stats()["free_pages"] == 32


def test_affinity_cannot_relabel_encoded_source_as_an_already_rotated_private_page():
    manager, _, contract = _manager()
    manager.create_request("target", context_id="private-prefix", namespace="tenant-A", contract=contract)
    before = manager.pool.stats()
    proof = ReuseProof("certified_transform", "document-contents", "private-prefix", "some-affinity-hint")
    with pytest.raises(ValueError, match="position-encoded K requires materialized"):
        manager.share_segments("document", "target", [
            ((0, 0, "canonical"), (0, 0, "global-position"), proof, tuple(range(13, 21)))])
    assert manager.pool.stats() == before
    assert not manager.version("target").segments


@pytest.fixture
def runtime(monkeypatch):
    with _isolated_imports() as scheduler:
        affinity = importlib.import_module("sglang.srt.layers.attention.redknot.nonprefix_affinity")
        clock = Clock()
        monkeypatch.setattr(scheduler, "time", types.SimpleNamespace(perf_counter=clock, sleep=clock.sleep))
        yield types.SimpleNamespace(scheduler=scheduler, affinity=affinity, clock=clock)


class Index:
    def __init__(self, mapping):
        self.mapping, self.calls = mapping, []

    def request_pages(self, request):
        self.calls.append(request.request_id)
        return dict(self.mapping.get(request.request_id, {}))


def _request(runtime, name, *, count=1, arrival=0, query=(9, 10), cancel=None):
    return runtime.scheduler.QueuedRequest(name, (1,), ("D",), query, count, (),
                                           arrival_seconds=arrival, cancel_after_tokens=cancel)


def _run(runtime, requests, mapping, *, engine=None, max_active=2, **config):
    engine = engine or FakeEngine(runtime.scheduler, runtime.clock)
    result = runtime.scheduler.run_queued(engine, requests, max_active=max_active,
        admission_policy="share_affinity", affinity_index=Index(mapping),
        affinity_config=runtime.affinity.ShareAffinityConfig(**config))
    return engine, result


def _admitted(engine):
    return [call[0] for call in engine.admit_calls]


@pytest.mark.parametrize("mapping", [{}, {name: {"same-physical-page": 128} for name in ("a", "b", "c", "d")}])
def test_empty_or_identical_affinity_falls_back_to_stable_fcfs(runtime, mapping):
    requests = [_request(runtime, name) for name in ("a", "b", "c", "d")]
    engine, result = _run(runtime, requests, mapping)
    assert result["completed_requests"] == 4
    assert _admitted(engine) == ["a", "b", "c", "d"]
    assert not engine.requests


def test_fcfs_policy_never_calls_optional_affinity_hook(runtime):
    class ForbiddenIndex:
        def request_pages(self, request):
            raise AssertionError("FCFS must not compute affinity")
    engine = FakeEngine(runtime.scheduler, runtime.clock)
    result = runtime.scheduler.run_queued(engine, [_request(runtime, "a"), _request(runtime, "b")],
        max_active=2, admission_policy="fcfs", affinity_index=ForbiddenIndex())
    assert result["completed_requests"] == 2 and _admitted(engine) == ["a", "b"]


def test_affinity_reorders_admission_not_tokens_and_records_arrival_queue_time(runtime):
    requests = [_request(runtime, name, count=2) for name in ("anchor", "cold", "hot")]
    outputs = {"anchor": (3, 4), "cold": (5, 6), "hot": (7, 8)}
    engine = FakeEngine(runtime.scheduler, runtime.clock, outputs=outputs)
    engine, result = _run(runtime, requests,
        {"anchor": {"shared": 4096}, "cold": {"other": 4096}, "hot": {"shared": 4096}},
        engine=engine, max_wait_seconds=100, max_bypass=8)
    assert _admitted(engine) == ["anchor", "hot", "cold"]
    assert result["max_forward_batch"] == 2 and result["completed_requests"] == 3
    assert {row["request_id"]: row["output_ids"] for row in result["records"]} == {k: list(v) for k, v in outputs.items()}
    assert all(row["ttft_seconds"] >= row["service_ttft_seconds"] for row in result["records"])
    assert not engine.requests


def test_future_hot_request_does_not_delay_ready_cold_request(runtime):
    requests = [_request(runtime, "anchor"), _request(runtime, "cold"), _request(runtime, "hot", arrival=1)]
    engine, result = _run(runtime, requests,
        {"anchor": {"shared": 4096}, "cold": {"other": 4096}, "hot": {"shared": 4096}},
        max_wait_seconds=100)
    assert _admitted(engine) == ["anchor", "cold", "hot"]
    first_batch = engine.forward_calls[0][0]
    assert first_batch == ["anchor", "cold"]
    assert result["completed_requests"] == 3
    hot = next(row for row in result["records"] if row["request_id"] == "hot")
    assert hot["admission_started_seconds"] >= 1


def test_overlap_is_weighted_by_unique_physical_bytes_not_document_count(runtime):
    requests = [_request(runtime, name) for name in ("anchor", "many_small", "one_large")]
    mapping = {"anchor": {"large": 1024, "small1": 8, "small2": 8},
               "many_small": {"small1": 8, "small2": 8}, "one_large": {"large": 1024}}
    engine, result = _run(runtime, requests, mapping, max_wait_seconds=100)
    assert _admitted(engine) == ["anchor", "one_large", "many_small"]
    assert result["completed_requests"] == 3


@pytest.mark.parametrize("limit", [0, 1, 2, 3])
def test_hard_bypass_bound_prevents_starvation_behind_arbitrarily_large_affinity(runtime, limit):
    names = ["anchor", "cold"] + [f"hot{i}" for i in range(12)]
    requests = [_request(runtime, name) for name in names]
    mapping = {name: {"shared": 10**18} for name in names if name != "cold"}
    mapping["cold"] = {"unpopular": 1}
    engine, result = _run(runtime, requests, mapping, max_bypass=limit, max_wait_seconds=1000)
    assert result["completed_requests"] == len(names)
    assert _admitted(engine).index("cold") <= limit + 1
    assert len(set(_admitted(engine))) == len(names)
    assert not engine.requests


def test_hard_age_override_cannot_be_outweighed_by_shared_bytes(runtime):
    names = ["anchor", "cold"] + [f"hot{i}" for i in range(8)]
    requests = [_request(runtime, name) for name in names]
    mapping = {name: {"shared": 10**18} for name in names if name != "cold"}
    engine, result = _run(runtime, requests, mapping, max_bypass=100, max_wait_seconds=.05)
    assert _admitted(engine)[:3] == ["anchor", "hot0", "cold"]
    assert result["completed_requests"] == len(names)


def test_capacity_blocked_candidates_do_not_spin_before_a_release(runtime):
    names = ["anchor", "cold", "hot0", "hot1"]
    engine = FakeEngine(runtime.scheduler, runtime.clock, capacity=1)
    engine, result = _run(runtime, [_request(runtime, name, count=3) for name in names],
        {name: {"shared": 10} for name in names if name != "cold"}, engine=engine,
        max_active=3, max_wait_seconds=100, max_bypass=2)
    assert result["completed_requests"] == len(names)
    assert all(row["admission_attempts"] <= len(names) for row in result["records"])
    attempts, epoch = {}, 0
    for event in result["trace"]:
        if event["event"] == "released":
            epoch += 1
        elif event["event"] == "admission_started":
            pair = event["request_id"], epoch
            assert pair not in attempts, "same capacity block retried without any release"
            attempts[pair] = True
    assert not engine.requests


class VariableCapacityEngine(FakeEngine):
    def __init__(self, runtime, *, costs, pages=3):
        super().__init__(runtime.scheduler, runtime.clock, capacity=1000)
        self.costs, self.pages = costs, pages

    def admit(self, request_id, prefix_ids, document_ids, max_suffix_tokens):
        if sum(self.costs[name] for name in self.requests) + self.costs[request_id] > self.pages:
            self.clock.advance(.01)
            self.admit_calls.append((request_id, tuple(prefix_ids), tuple(document_ids), max_suffix_tokens))
            raise self.module.CapacityError("physical pages unavailable")
        return super().admit(request_id, prefix_ids, document_ids, max_suffix_tokens)


def test_protected_oldest_drains_active_requests_instead_of_capacity_starvation(runtime):
    names = ["anchor", "cold"] + [f"hot{i}" for i in range(8)]
    requests = [_request(runtime, name, count=5 if name == "anchor" else 1) for name in names]
    costs = {name: 1 for name in names}
    costs.update(anchor=2, cold=3)
    engine = VariableCapacityEngine(runtime, costs=costs)
    engine, result = _run(runtime, requests, {name: {"shared": 1000} for name in names if name != "cold"},
        engine=engine, max_active=3, max_wait_seconds=1000, max_bypass=2)
    assert result["completed_requests"] == len(names)
    successful = [event["request_id"] for event in result["trace"] if event["event"] == "admitted"]
    assert successful.index("cold") <= 4
    cold = next(row for row in result["records"] if row["request_id"] == "cold")
    assert cold["capacity_blocks"] >= 1 and cold["status"] == "completed"
    assert not engine.requests


def test_empty_active_impossible_capacity_fails_one_request_and_makes_progress(runtime):
    requests = [_request(runtime, name) for name in ("too_large", "small", "hot")]
    engine = VariableCapacityEngine(runtime, costs={"too_large": 4, "small": 1, "hot": 1})
    engine, result = _run(runtime, requests,
        {name: {"shared": 128} for name in ("too_large", "small", "hot")}, engine=engine)
    assert result["completed_requests"] == 2 and result["failed_requests"] == 1
    impossible = next(row for row in result["records"] if row["request_id"] == "too_large")
    assert impossible["finish_reason"] == "impossible_capacity"
    assert impossible["admission_attempts"] == 1
    assert not engine.requests


@pytest.mark.parametrize("kwargs", [
    {"lookahead": 0}, {"lookahead": -1}, {"lookahead": True}, {"lookahead": 1.5},
    {"max_bypass": -1}, {"max_bypass": True}, {"recent_requests": -1}, {"recent_requests": True},
    {"max_wait_seconds": -1}, {"max_wait_seconds": float("nan")}, {"max_wait_seconds": float("inf")},
])
def test_invalid_affinity_configuration_fails_closed(runtime, kwargs):
    with pytest.raises((TypeError, ValueError)):
        runtime.affinity.ShareAffinityConfig(**kwargs)


def _index_engine(*, window=8):
    manager, _, _ = _manager()
    local = types.SimpleNamespace(layer=0, kv_head=0, head_type="local", window=window, sink_size=2)
    global_ = types.SimpleNamespace(layer=0, kv_head=1, head_type="global", window=-1, sink_size=0)
    document = types.SimpleNamespace(source_request_id="document", source_keys={(0, 0): (0, 0, "canonical")}, doc_len=8)
    return types.SimpleNamespace(namespace="tenant-A", manager=manager, pool=manager.pool,
        documents={"D": document}, policies=[[local, global_]], model_revision="weights-v1", share_local=True)


def _footprint_request(*docs, prefix=3):
    return types.SimpleNamespace(document_ids=tuple(docs), prefix_ids=tuple(range(prefix)))


def test_index_counts_only_visible_canonical_aliases_not_private_boundary_or_global_pages(runtime):
    engine = _index_engine(window=5)
    index = runtime.affinity.CanonicalAffinityIndex.from_engine(engine)
    footprint = dict(index.request_pages(_footprint_request("D", prefix=3)))
    # Visible source rows 3..7: row3 is a private compact boundary copy;
    # only original source page1 (rows4..7) remains physically shared.
    expected = engine.manager.version("document").segments[0, 0, "canonical"].pages[1]
    assert len(footprint) == 1
    identity, size = next(iter(footprint.items()))
    assert identity[-3:] == (expected.slot, expected.generation, expected.content_id)
    assert size == engine.pool.page_bytes
    assert footprint == dict(index.request_pages(_footprint_request("D", prefix=17)))


def test_repeated_document_occurrences_count_each_physical_page_once(runtime):
    engine = _index_engine(window=100)
    index = runtime.affinity.CanonicalAffinityIndex.from_engine(engine)
    one = dict(index.request_pages(_footprint_request("D")))
    repeated = dict(index.request_pages(_footprint_request("D", "D", "D")))
    assert repeated == one and len(one) == 2
    assert sum(repeated.values()) == 2*engine.pool.page_bytes


@pytest.mark.parametrize("change", ["namespace", "model_revision", "head_profile", "manager", "pool"])
def test_index_is_engine_scoped_and_rejects_changed_cache_contract(runtime, change):
    engine = _index_engine()
    index = runtime.affinity.CanonicalAffinityIndex.from_engine(engine)
    if change == "head_profile":
        engine.policies[0][0].window = 9
    elif change in ("manager", "pool"):
        setattr(engine, change, object())
    else:
        setattr(engine, change, "different")
    with pytest.raises(ValueError, match="engine/namespace/model/head profile"):
        index.request_pages(_footprint_request("D"))


def test_same_metadata_and_slot_numbers_in_other_engine_never_equal_affinity_keys(runtime):
    first, second = _index_engine(), _index_engine()
    first_index = runtime.affinity.CanonicalAffinityIndex.from_engine(first)
    second_index = runtime.affinity.CanonicalAffinityIndex.from_engine(second)
    first_keys = set(first_index.request_pages(_footprint_request("D")))
    second_keys = set(second_index.request_pages(_footprint_request("D")))
    assert first_keys and second_keys and first_keys.isdisjoint(second_keys)
    with pytest.raises(ValueError, match="engine/namespace/model/head profile"):
        first_index.validate_engine(second)


def test_source_cow_invalidates_old_affinity_snapshot_but_not_old_reader(runtime):
    engine = _index_engine()
    index = runtime.affinity.CanonicalAffinityIndex.from_engine(engine)
    before_keys = set(index.request_pages(_footprint_request("D")))
    reader = engine.manager.bind("document")
    old_payload = reader.gather((0, 0, "canonical"))[0].clone()
    engine.manager.update("document", patches=[SegmentPatch((0, 0, "canonical"), (1,),
        torch.ones(1, 8)*999, torch.ones(1, 8)*-999, "changed-source")])
    with pytest.raises(ValueError, match="canonical source changed"):
        index.request_pages(_footprint_request("D"))
    new_index = runtime.affinity.CanonicalAffinityIndex.from_engine(engine)
    assert set(new_index.request_pages(_footprint_request("D"))) != before_keys
    torch.testing.assert_close(reader.gather((0, 0, "canonical"))[0], old_payload, rtol=0, atol=0)
    engine.manager.release_request("document")
    reader.complete()
    assert engine.pool.stats()["free_pages"] == 32


@pytest.mark.parametrize("bad_source", ["namespace", "basis", "positions", "global_policy"])
def test_index_does_not_accept_untrusted_or_noncanonical_source_as_shared(runtime, bad_source):
    from dataclasses import replace
    from types import MappingProxyType
    engine = _index_engine()
    root = engine.manager.version("document")
    segments = dict(root.segments)
    source = segments[0, 0, "canonical"]
    if bad_source == "namespace":
        root = replace(root, namespace="another-tenant")
    elif bad_source == "basis":
        segments[source.key] = replace(source, position_basis="rope-global")
    elif bad_source == "positions":
        segments[source.key] = replace(source, positions=tuple(range(8, 16)))
    else:
        engine.policies[0][0].head_type = "global"
    # Fault injection into the source registry, not authorization to alias it.
    engine.manager._requests["document"] = replace(root, segments=MappingProxyType(segments))
    with pytest.raises(ValueError):
        runtime.affinity.CanonicalAffinityIndex.from_engine(engine)


def test_recent_grouping_survives_both_groups_being_seen_and_is_bounded(runtime):
    requests = [_request(runtime, name) for name in ("old_A", "latest_B", "pending_A", "pending_B", "next_B")]
    mapping = {request.request_id: {"A" if request.request_id.endswith("A") else "B": 128}
               for request in requests}
    config = runtime.affinity.ShareAffinityConfig(max_bypass=2, max_wait_seconds=100, recent_requests=2)
    selector = runtime.affinity.ShareAffinitySelector(Index(mapping), requests, config)
    selector.on_released("old_A")
    selector.on_released("latest_B")
    first = selector.choose(requests[2:4], [], now=0, release_epoch=2)
    assert first.selected_offset == 1 and first.source_canonical_overlap_bytes == 128
    assert first.recent_overlap_bytes_by_recency == (128, 0)
    selector.on_admitted(first, [requests[2]])
    selector.on_released("pending_B")
    second = selector.choose([requests[2], requests[4]], [], now=0, release_epoch=3)
    assert second.selected_offset == 1
    selector.on_admitted(second, [requests[2]])
    protected = selector.choose([requests[2], requests[4]], [], now=0, release_epoch=4)
    assert protected.selected_offset == 0 and protected.protected and protected.reason == "max_bypass"
    assert len(selector.recent) == 2


def test_active_pairwise_tiebreak_does_not_relabel_duplicate_overlap_as_unique_bytes(runtime):
    names = ("active_A1", "active_A2", "active_B", "candidate_B", "candidate_A")
    requests = [_request(runtime, name) for name in names]
    mapping = {name: {"A" if "A" in name else "B": 128} for name in names}
    selector = runtime.affinity.ShareAffinitySelector(Index(mapping), requests,
        runtime.affinity.ShareAffinityConfig(max_wait_seconds=100))
    decision = selector.choose(requests[3:], names[:3], now=0, release_epoch=0)
    assert decision.selected_offset == 1
    assert decision.active_canonical_overlap_bytes == 128
    assert decision.source_canonical_overlap_bytes == 128
    assert decision.active_request_overlap_sum_bytes == 256


@pytest.mark.parametrize("fault", ["raises", "negative_size", "bool_size", "inconsistent_size"])
def test_index_failure_releases_scheduler_owner_without_admission_or_model_fallback(runtime, fault):
    class BrokenIndex:
        def request_pages(self, request):
            if fault == "raises":
                raise ValueError("injected affinity index failure")
            return {"page": (-1 if fault == "negative_size" else True if fault == "bool_size"
                              else 128 if request.request_id == "a" else 256)}
    engine = FakeEngine(runtime.scheduler, runtime.clock)
    requests = [_request(runtime, "a"), _request(runtime, "b")]
    result = runtime.scheduler.run_queued(engine, requests, max_active=2,
        admission_policy="share_affinity", affinity_index=BrokenIndex())
    assert result["status"] == "failed" and result["completed_requests"] == 0
    assert not engine.admit_calls and not engine.forward_calls and not engine.requests
    assert id(engine) not in runtime.scheduler._OWNERS
    fallback = runtime.scheduler.run_queued(engine, requests, max_active=2, admission_policy="fcfs")
    assert fallback["completed_requests"] == 2  # Explicit subsequent run, never silent recovery.


def test_affinity_preparation_time_remains_inside_service_wall_and_qps(runtime):
    class SlowIndex(Index):
        def request_pages(self, request):
            runtime.clock.advance(.03)
            return super().request_pages(request)
    engine = FakeEngine(runtime.scheduler, runtime.clock)
    result = runtime.scheduler.run_queued(engine, [_request(runtime, "a"), _request(runtime, "b")],
        max_active=2, admission_policy="share_affinity", affinity_index=SlowIndex({}))
    assert result["affinity"]["index_setup_seconds"] == pytest.approx(.06)
    assert result["affinity"]["index_setup_included_in_service_wall"] is True
    assert result["total_wall_seconds"] == pytest.approx(.06+.02+.1+.04)
    assert result["completed_qps"] == pytest.approx(2/(.06+.02+.1+.04))


@pytest.mark.parametrize("device,dtype,page_size,window,doc_length,atol", [
    pytest.param("cpu", torch.float32, 4, 7, 11, 1e-5, id="cpu-torch"),
    pytest.param("cuda", torch.bfloat16, 256, 768, 773, .05, id="cuda-fa2-bf16",
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and FlashAttention2")),
    pytest.param("cuda", torch.float16, 256, 768, 773, .003, id="cuda-fa2-fp16",
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and FlashAttention2")),
])
def test_real_tiny_qwen_affinity_keeps_output_identity_and_canonical_bytes_immutable(
        device, dtype, page_size, window, doc_length, atol):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with _core_imports() as modules:
            scheduler = importlib.import_module("sglang.srt.layers.attention.redknot.nonprefix_scheduler")
            affinity = importlib.import_module("sglang.srt.layers.attention.redknot.nonprefix_affinity")
            model = _tiny_model(device=device, dtype=dtype)
            config = _head_config(modules, window=window)
            results = []
            logits_by_policy = []
            for policy in ("fcfs", "share_affinity"):
                engine = modules.core.NonPrefixPagedEngine(model, config, capacity_pages=2048 if device == "cpu" else 128,
                    page_size=page_size, backend="torch" if device == "cpu" else "flash", prefix_chunk_size=256)
                try:
                    engine.register_document("A", [20+i % 151 for i in range(doc_length)])
                    engine.register_document("B", [30+(3*i) % 151 for i in range(doc_length)])
                    source = {}
                    for document in engine.documents.values():
                        for segment in engine.manager.version(document.source_request_id).segments.values():
                            for ref in segment.pages:
                                source[ref] = tuple(value.clone() for value in engine.pool.read_page(ref))
                    emitted_logits = {}
                    full_canonical_reads = set()
                    native_forward = engine.forward
                    def observed_forward(request_ids, input_ids):
                        private_seen = set()
                        for state in engine.requests.values():
                            private = set(state.tail_all_refs)
                            for (layer, head), partitions in state.partitions.items():
                                for partition in partitions:
                                    if not partition.canonical:
                                        private.update(partition.pages)
                                    elif partition.key[2].startswith("document:"):
                                        assert set(partition.pages) <= set(source)
                                        for page_index, ref in enumerate(partition.pages):
                                            if min(page_size, partition.end-page_index*page_size) == page_size:
                                                full_canonical_reads.add(ref)
                            assert not private.intersection(source), "global/suffix pages must never alias canonical local pages"
                            assert not private.intersection(private_seen), "global/suffix KV must remain request private"
                            private_seen.update(private)
                        logits = native_forward(request_ids, input_ids)
                        for index, request_id in enumerate(request_ids):
                            emitted_logits.setdefault(request_id, []).append(logits[index].detach().clone())
                        return logits
                    engine.forward = observed_forward
                    requests = [scheduler.QueuedRequest(name, prefix, (doc,), (81, 82), 2, ())
                                for name, prefix, doc in (("anchor", (1,), "A"),
                                                         ("cold", (2, 3), "B"),
                                                         ("hot", (4, 5, 6, 7), "A"))]
                    result = scheduler.run_queued(engine, requests, max_active=2, admission_policy=policy,
                        affinity_config=affinity.ShareAffinityConfig(max_wait_seconds=100, max_bypass=8))
                    assert result["status"] == "complete" and result["completed_requests"] == 3
                    assert not engine.requests
                    assert full_canonical_reads, "integration must actually read shared canonical full pages"
                    if device == "cuda":
                        assert len(full_canonical_reads) >= 4  # Two full 256-token pages per source document.
                    for ref, before in source.items():
                        after = engine.pool.read_page(ref)
                        torch.testing.assert_close(after[0], before[0], rtol=0, atol=0)
                        torch.testing.assert_close(after[1], before[1], rtol=0, atol=0)
                    results.append(result)
                    logits_by_policy.append(emitted_logits)
                finally:
                    engine.close()
                assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
                assert getattr(model, "_redknot_nonprefix_owner", None) is None
            assert {row["request_id"]: row["output_ids"] for row in results[0]["records"]} == {
                row["request_id"]: row["output_ids"] for row in results[1]["records"]}
            orders = [[event["request_id"] for event in result["trace"] if event["event"] == "admitted"]
                      for result in results]
            assert orders == [["anchor", "cold", "hot"], ["anchor", "hot", "cold"]]
            for request_id, logits in logits_by_policy[0].items():
                assert len(logits) == len(logits_by_policy[1][request_id]) == 2
                for first, second in zip(logits, logits_by_policy[1][request_id]):
                    torch.testing.assert_close(first, second, rtol=0, atol=atol)
    finally:
        torch.set_num_threads(previous)
