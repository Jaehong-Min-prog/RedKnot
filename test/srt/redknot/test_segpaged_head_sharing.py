"""CPU contract tests for local-head-only SegPaged storage/admission.

The fixtures do not establish real-model reuse accuracy. These regressions
verify physical ownership, distinct global state, fair bounded admission and
honest reporting of measured capacity rather than fabricated serving QPS.
"""

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch


@pytest.fixture(scope="module")
def bench():
    name = "_test_segpaged_head_sharing_benchmark"
    source = Path(__file__).with_name("benchmark_segpaged_head_sharing.py")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def _args(bench, *extra):
    return bench.parse_args([
        "--device", "cpu", "--dtype", "float32", "--lengths", "16",
        "--heads", "4", "--head-dim", "8", "--page-size", "4",
        "--prefix-tokens", "3", "--query-tokens", "2",
        "--cow-token-ratio", "0.125", "--local-head-ratios", "0", "0.5", "1",
        "--budget-requests", "4", "--max-requests", "64",
        "--verify-requests", "2", "--fixed-requests", "1", "2", "4",
        "--reference-block-size", "5", *extra,
    ])


def _setup(bench, args, local_heads=2, capacity=512):
    fixtures = bench.make_fixtures(args, args.lengths[0], local_heads)
    pool = bench.HeadPagePool(capacity, args.page_size, args.head_dim,
                              dtype=getattr(torch, args.dtype), device=args.device)
    manager = bench.HeadKVManager(pool)
    source = bench.build_local_cache(manager, fixtures, args)
    return manager, fixtures, source


def _reclaimed(state):
    for key in ("requests", "cached_roots", "allocated_bytes", "reserved_pages",
                "retiring_pages", "pinned_pages", "pending_transactions"):
        assert state[key] == 0, (key, state)
    assert state["free_pages"] == state["capacity_pages"]


@pytest.mark.parametrize("local_heads", [0, 2, 4])
def test_cache_contains_only_local_heads_after_publisher_release(bench, local_heads):
    args = _args(bench)
    manager, fixtures, source = _setup(bench, args, local_heads)
    try:
        state = manager.stats()
        assert state["requests"] == 0
        assert state["cached_roots"] == int(local_heads > 0)
        assert state["live_pages"] == local_heads * math.ceil(fixtures["length"] / args.page_size)
        if not local_heads:
            assert source is None
        else:
            assert set(source.segments) == {(0, h, "document") for h in range(local_heads)}
            assert bench.check_local_source(manager, fixtures, source) == 0
        assert fixtures["local_k"].shape == (local_heads, 16, args.head_dim)
    finally:
        bench.cleanup(manager)
    _reclaimed(manager.stats())


def test_local_head_refs_shared_but_global_payload_and_refs_are_request_specific(bench):
    args = _args(bench, "--cow-token-ratio", "0")
    manager, fixtures, source = _setup(bench, args)
    payloads = [bench.request_payload(fixtures, args, i) for i in range(2)]
    try:
        assert not torch.equal(payloads[0]["global_k"], payloads[1]["global_k"])
        assert not torch.equal(payloads[0]["global_v"], payloads[1]["global_v"])
        assert not torch.equal(payloads[0]["prefix_k"], payloads[1]["prefix_k"])
        plans = [bench.populate_request(manager, fixtures, args, "head_shared", f"r{i}", i, payloads[i])
                 for i in range(2)]
        roots = [manager.version(f"r{i}") for i in range(2)]
        assert plans[0]["document_position_start"] != plans[1]["document_position_start"]
        for i, root in enumerate(roots):
            with manager.bind(f"r{i}") as lease:
                for h in range(args.heads):
                    key = (0, h, "document")
                    segment = root.segments[key]
                    assert segment.positions == tuple(range(plans[i]["document_position_start"],
                                                            plans[i]["document_position_start"] + 16))
                    assert min(segment.positions) >= args.prefix_tokens
                    assert segment.position_basis == "none"
                    got_k, got_v = lease.gather(key)
                    if h < fixtures["local_heads"]:
                        assert segment.pages == source.segments[key].pages
                        expected = fixtures["local_k"][h], fixtures["local_v"][h]
                    else:
                        other_refs = roots[1 - i].segments[key].pages
                        assert not set(segment.pages).intersection(other_refs)
                        g = h - fixtures["local_heads"]
                        expected = payloads[i]["global_k"][g], payloads[i]["global_v"][g]
                    for got, wanted in zip((got_k, got_v), expected):
                        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
            result = bench.verify_request(manager, fixtures, args, plans[i], payloads[i])
            assert result["kv_max_abs_error"] == 0
        assert bench.check_local_source(manager, fixtures, source) == 0
    finally:
        bench.cleanup(manager)
    _reclaimed(manager.stats())


def test_independent_arm_copies_local_heads_and_uses_same_request_payload(bench):
    args = _args(bench, "--cow-token-ratio", "0")
    manager, fixtures, source = _setup(bench, args)
    payload = bench.request_payload(fixtures, args, 1)
    try:
        plans = {}
        for backend in ("head_shared", "independent"):
            plans[backend] = bench.populate_request(manager, fixtures, args, backend, backend, 1, payload)
        a, b = (manager.version(name) for name in ("head_shared", "independent"))
        for h in range(args.heads):
            key = (0, h, "document")
            assert not set(a.segments[key].pages).intersection(b.segments[key].pages)
            if h < fixtures["local_heads"]:
                assert not set(b.segments[key].pages).intersection(source.segments[key].pages)
        with manager.bind("head_shared") as left, manager.bind("independent") as right:
            for key in a.segments:
                for got, wanted in zip(left.gather(key), right.gather(key)):
                    torch.testing.assert_close(got, wanted, atol=0, rtol=0)
        for backend in plans:
            assert bench.verify_request(manager, fixtures, args, plans[backend], payload)["kv_max_abs_error"] == 0
    finally:
        bench.cleanup(manager)


def test_local_cow_preserves_source_and_unmodified_pages(bench):
    args = _args(bench)
    manager, fixtures, source = _setup(bench, args)
    payload = bench.request_payload(fixtures, args, 0)
    try:
        plan = bench.populate_request(manager, fixtures, args, "head_shared", "repaired", 0, payload)
        root = manager.version("repaired")
        touched = math.ceil(payload["repaired_rows"] / args.page_size)
        for h in range(fixtures["local_heads"]):
            key = (0, h, "document")
            old, new = source.segments[key], root.segments[key]
            for page, (src, dst) in enumerate(zip(old.pages, new.pages)):
                assert (src != dst) == (page < touched)
        assert plan["after_cow"]["live_pages"] - plan["before_cow"]["live_pages"] == (
            touched * fixtures["local_heads"]
        )
        assert plan["repair_written_bytes"] == 2 * fixtures["local_heads"] * payload["repaired_rows"] * args.head_dim * 4
        assert bench.check_local_source(manager, fixtures, source) == 0
        assert bench.verify_request(manager, fixtures, args, plan, payload)["kv_max_abs_error"] == 0
        manager.release_request("repaired")
        state = manager.stats()
        assert state["requests"] == 0 and state["cached_roots"] == 1
        assert state["live_pages"] == fixtures["local_heads"] * math.ceil(16 / args.page_size)
    finally:
        bench.cleanup(manager)


def test_evict_source_while_requests_live_then_release_all(bench):
    args = _args(bench, "--cow-token-ratio", "0")
    manager, fixtures, _ = _setup(bench, args)
    payload = bench.request_payload(fixtures, args, 0)
    try:
        plan = bench.populate_request(manager, fixtures, args, "head_shared", "reader", 0, payload)
        before = manager.stats()["allocated_bytes"]
        manager.evict("local-document")
        assert manager.stats()["allocated_bytes"] == before
        assert bench.verify_request(manager, fixtures, args, plan, payload)["kv_max_abs_error"] == 0
    finally:
        bench.cleanup(manager)
    _reclaimed(manager.stats())


@pytest.mark.parametrize("local_heads", [0, 2, 4])
@pytest.mark.parametrize("repair", [0, 1])
def test_reference_matches_independent_dense_softmax(bench, local_heads, repair):
    args = _args(bench, "--cow-token-ratio", str(repair))
    fixtures = bench.make_fixtures(args, 16, local_heads)
    payload = bench.request_payload(fixtures, args, 1)
    local_k, local_v = fixtures["local_k"].clone(), fixtures["local_v"].clone()
    for patch in payload["patches"]:
        h = patch.key[1]
        local_k[h, list(patch.indices)] = patch.k
        local_v[h, list(patch.indices)] = patch.v
    k = torch.cat((payload["prefix_k"], torch.cat((local_k, payload["global_k"]), 0)), 1).double()
    v = torch.cat((payload["prefix_v"], torch.cat((local_v, payload["global_v"]), 0)), 1).double()
    q = payload["query"].double()
    expected = (q @ k.transpose(-1, -2) * args.head_dim ** -0.5).softmax(-1) @ v
    actual = bench.reference_attention(fixtures, args, payload)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=3e-6)


@pytest.fixture(scope="module")
def report(bench, tmp_path_factory):
    output = tmp_path_factory.mktemp("head-sharing") / "report.json"
    return bench.benchmark(_args(bench, "--output", str(output)))


def test_actual_admission_exhausts_same_bounded_slab_and_reports_no_qps(report):
    assert report["status"] == "complete"
    json.dumps(report, allow_nan=False)
    for case in report["cases"]:
        assert case["summary"]["throughput_or_qps_measured"] is False
        assert not case["summary"]["any_censored"]
        assert case["document_full_head_logical_bytes"] == 2 * 4 * 16 * 8 * 4
        reserves = []
        for arm in case["arms"]:
            assert arm["termination"] == "capacity_error"
            assert not arm["admission_is_censored"]
            steps = arm["admission_steps"]
            accepted = [s for s in steps if s["admitted"]]
            assert len(accepted) == arm["admitted_requests"]
            assert len(steps) == arm["admitted_requests"] + 1
            rejected = steps[-1]
            assert not rejected["admitted"]
            for field in ("allocated_bytes", "requests", "cached_roots", "pending_transactions",
                          "pinned_pages", "reserved_pages", "free_pages"):
                assert rejected["before"][field] == rejected["after"][field]
            for index, step in enumerate(accepted):
                assert step["resident_requests"] == index + 1
                assert step["after"]["requests"] == index + 1
                assert step["after"]["allocated_bytes"] > step["before"]["allocated_bytes"]
                assert step["private_global_pages"] == case["config"]["global_heads"] * 4
                assert math.isfinite(step["materialize_and_repair"]["wall_ms"])
                assert step["materialize_and_repair"]["wall_ms"] >= 0
            reserves.append(arm["source_cache"]["reserved_bytes"])
            assert arm["source_cache"]["requests"] == 0
            assert arm["after_request_release"]["allocated_bytes"] == arm["source_cache"]["allocated_bytes"]
            assert arm["correctness"]["model_quality_evaluated"] is False
            for validation in arm["correctness"]["verified_requests"]:
                assert validation["kv_max_abs_error"] == 0
                assert validation["attention_max_abs_error"] < 2e-5
            _reclaimed(arm["final_state"])
            assert arm["final_state"]["reserved_bytes"] == arm["source_cache"]["reserved_bytes"] > 0
        assert reserves[0] == reserves[1] == case["config"]["reserved_bytes"]


def test_baseline_has_no_unused_source_cache_and_zero_local_is_negative_control(report):
    for case in report["cases"]:
        independent, shared = case["arms"]
        assert independent["source_cache_policy"] == "none"
        assert independent["source_cache"]["allocated_bytes"] == 0
        assert shared["source_cache"]["live_pages"] == case["config"]["local_heads"] * 4
        if case["config"]["local_heads"] == 0:
            assert independent["admitted_requests"] == shared["admitted_requests"]
            assert case["summary"]["observed_resident_request_ratio"] == 1
        else:
            assert shared["admitted_requests"] > independent["admitted_requests"]


def test_physical_page_categories_sum_to_actual_active_bytes(report):
    for case in report["cases"]:
        page_bytes = case["config"]["reserved_bytes"] // case["config"]["capacity_pages"]
        for arm in case["arms"]:
            for step in arm["admission_steps"]:
                if not step["admitted"]:
                    continue
                account = step["active_page_accounting"]
                categories = ("source_cache_bytes", "private_prefix_bytes",
                              "private_global_bytes", "private_local_bytes")
                assert sum(account[k] for k in categories) == account["unique_owned_page_bytes"]
                assert account["unique_owned_page_bytes"] == step["after"]["allocated_bytes"]
                assert account["source_cache_bytes"] == arm["source_cache"]["allocated_bytes"]
                assert account["private_global_bytes"] == (
                    step["resident_requests"] * case["config"]["global_heads"] * 4 * page_bytes
                )
                assert account["private_prefix_bytes"] == step["resident_requests"] * 4 * page_bytes


def test_matched_request_memory_comparisons_keep_negative_single_request_savings(report):
    for case in report["cases"]:
        for pair in case["summary"]["fixed_request_comparisons"]:
            if not pair["both_admitted"]:
                assert "active_page_saving_fraction" not in pair
                continue
            count = pair["resident_requests"]
            measured = {}
            for arm in case["arms"]:
                step = next(s for s in arm["admission_steps"] if s["admitted"] and s["resident_requests"] == count)
                measured[arm["backend"]] = step["after"]["allocated_bytes"]
            assert pair["independent_active_bytes"] == measured["independent"]
            assert pair["head_shared_active_bytes"] == measured["head_shared"]
            assert pair["active_page_saving_fraction"] == 1 - measured["head_shared"] / measured["independent"]
            # Retaining an immutable source plus a repaired private page may
            # cost more for one request. Negative results must not be clipped.
            if count == 1 and case["config"]["local_heads"]:
                assert pair["active_page_saving_fraction"] < 0


def test_request_guard_is_censored_not_a_capacity_measurement(bench):
    case = bench.run_case(_args(bench, "--max-requests", "1"), 16, 0.5)
    assert case["summary"]["any_censored"]
    for arm in case["arms"]:
        assert arm["admission_is_censored"]
        assert arm["termination"] == "max_requests_guard"
        assert arm["admitted_requests"] == 1
        assert all(s["admitted"] for s in arm["admission_steps"])
        _reclaimed(arm["final_state"])


def test_cow_capacity_failure_rolls_back_full_request(bench):
    # Initial shared root fits exactly: 8 cached + 8 global + 4 prefix pages.
    # The subsequent immutable local repair requires two extra pages and fails.
    case = bench.run_case(_args(bench, "--capacity-pages", "20"), 16, 0.5)
    for arm in case["arms"]:
        assert arm["admitted_requests"] == 0
        assert arm["termination"] == "capacity_error"
        reject = arm["admission_steps"][-1]
        assert reject["before"]["allocated_bytes"] == reject["after"]["allocated_bytes"]
        assert reject["before"]["requests"] == reject["after"]["requests"] == 0
        _reclaimed(arm["final_state"])


def test_source_cache_capacity_failure_is_explicit_and_fully_reclaimed(bench):
    case = bench.run_case(_args(bench, "--capacity-pages", "2"), 16, 0.5)
    shared = case["arms"][1]
    assert shared["termination"] == "source_cache_capacity_error"
    assert not shared["admission_is_censored"]
    assert shared["source_publish_failure"]
    assert shared["admitted_requests"] == 0
    _reclaimed(shared["final_state"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.0])
def test_local_source_corruption_is_not_ignored(bench, bad):
    args = _args(bench)
    manager, fixtures, source = _setup(bench, args)
    try:
        ref = source.segments[(0, 0, "document")].pages[0]
        manager.pool.k[ref.slot, 0, 0] += bad
        with pytest.raises(AssertionError):
            bench.check_local_source(manager, fixtures, source)
    finally:
        bench.cleanup(manager)


@pytest.mark.parametrize("fault", ["attention", "publication", "after_attach"])
def test_unexpected_failure_reclaims_every_manager(bench, monkeypatch, fault):
    captured = []
    manager_type = bench.HeadKVManager

    class RecordingManager(manager_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    monkeypatch.setattr(bench, "HeadKVManager", RecordingManager)
    if fault == "attention":
        def broken_attention(query, **kwargs):
            return torch.full_like(query, float("nan"))
        monkeypatch.setattr(bench, "paged_attention", broken_attention)
    elif fault == "publication":
        original = bench.build_local_cache
        def broken_publication(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected publication failure")
        monkeypatch.setattr(bench, "build_local_cache", broken_publication)
    else:
        original = bench.populate_request
        def broken_attachment(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected attachment failure")
        monkeypatch.setattr(bench, "populate_request", broken_attachment)
    with pytest.raises((AssertionError, RuntimeError)):
        bench.run_case(_args(bench), 16, 0.5)
    assert captured
    for manager in captured:
        _reclaimed(manager.stats())


def test_seeded_request_payload_is_equal_across_arms_but_differs_across_requests(bench):
    args = _args(bench)
    fixtures = bench.make_fixtures(args, 16, 2)
    a, b = (bench.request_payload(fixtures, args, 5) for _ in range(2))
    c = bench.request_payload(fixtures, args, 6)
    for key in ("global_k", "global_v", "prefix_k", "prefix_v", "query"):
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        assert not torch.equal(a[key], c[key])


@pytest.mark.parametrize("option,value", [
    ("--heads", "0"), ("--page-size", "0"), ("--lengths", "0"),
    ("--capacity-pages", "0"), ("--budget-gib", "nan"),
    ("--document-kv-gib", "inf"), ("--local-head-ratios", "1.1"),
    ("--cow-token-ratio", "-1"), ("--max-requests", "0"),
    ("--verify-requests", "-1"),
])
def test_invalid_admission_configuration_is_rejected(bench, option, value):
    with pytest.raises(SystemExit):
        _args(bench, option, value)
