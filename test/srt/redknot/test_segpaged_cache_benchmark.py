"""CPU regressions for the persistent-cache component benchmark.

These tests check the experiment's contracts and accounting, not model quality
or a speedup target. In particular, resident requests are not serving QPS,
and the fixture's position-free keys are not a proof of real-model reuse.
"""

import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import pytest
import torch


_FILE = Path(__file__).with_name("benchmark_segpaged_cache.py")


@pytest.fixture(scope="module")
def bench():
    name = "_test_segpaged_persistent_benchmark"
    spec = importlib.util.spec_from_file_location(name, _FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def _args(bench, *extra):
    return bench.parse_args([
        "--device", "cpu", "--dtype", "float32", "--lengths", "32",
        "--heads", "2", "--head-dim", "8", "--page-size", "4",
        "--documents", "4", "--prefix-tokens", "3", "--query-tokens", "2",
        "--requests", "1", "3", "--warmup", "1", "--repeats", "2",
        "--reference-block-size", "5", "--repair-head-ratio", "0.5",
        "--repair-token-ratio", "0.25", *extra,
    ])


def _cached(bench, args):
    fixtures = bench.make_fixtures(args, args.lengths[0])
    pool = bench.HeadPagePool(256, args.page_size, args.head_dim,
                              dtype=getattr(torch, args.dtype), device="cpu")
    manager = bench.HeadKVManager(pool)
    versions = bench.build_cached_documents(manager, fixtures, args)
    return manager, fixtures, versions


def _cleanup(manager):
    # Test-only cleanup also covers deliberate fault injection below.
    for name in tuple(manager._requests):
        manager.release_request(name)
    for name in tuple(manager._cache):
        manager.evict(name)
    manager.pool.collect()


def _assert_reclaimed(state):
    for name in ("allocated_bytes", "requests", "cached_roots", "pinned_pages",
                 "reserved_pages", "retiring_pages", "pending_transactions"):
        assert state[name] == 0, (name, state)
    assert state["free_pages"] == state["capacity_pages"]


def test_offline_pages_survive_publisher_and_repeated_online_release(bench):
    args = _args(bench, "--shared-ratio", "1")
    manager, fixtures, versions = _cached(bench, args)
    try:
        cached = bench.stats_snapshot(manager)
        assert cached["requests"] == 0
        assert cached["cached_roots"] == args.documents
        assert cached["allocated_bytes"] > 0
        assert all(v.request_id.startswith("publisher-") for v in versions.values())
        for turn in range(3):
            name = f"sequential-{turn}"
            bench.populate_request(manager, fixtures, args, "shared_pages", name, turn)
            manager.release_request(name)
            state = bench.stats_snapshot(manager)
            assert state["requests"] == 0
            assert state["cached_roots"] == args.documents
            assert state["allocated_bytes"] == cached["allocated_bytes"]
            assert state["written_bytes"] - cached["written_bytes"] == (
                (turn + 1) * 2 * args.heads * args.prefix_tokens * args.head_dim * 4
            )
        assert bench.validate_cache(manager, fixtures, args, versions) == 0.0
    finally:
        _cleanup(manager)
    _assert_reclaimed(manager.stats())


@pytest.mark.parametrize("shared_ratio", [0, 0.5, 1])
def test_nonprefix_document_order_and_real_page_identity(bench, shared_ratio):
    args = _args(bench, "--shared-ratio", str(shared_ratio))
    manager, fixtures, versions = _cached(bench, args)
    try:
        plans = [bench.populate_request(manager, fixtures, args, "shared_pages",
                                        f"request-{i}", i) for i in range(3)]
        assert plans[0]["order"] != list(range(args.documents))
        assert len({tuple(p["order"]) for p in plans}) > 1
        for plan in plans:
            version = manager.version(plan["request_id"])
            previous = args.prefix_tokens
            for doc in plan["order"]:
                wanted = tuple(range(previous, previous + fixtures["sizes"][doc]))
                assert plan["positions"][doc] == wanted
                assert min(wanted) >= args.prefix_tokens > 0
                for head in range(args.heads):
                    key = (0, head, f"doc-{doc}")
                    original = versions[doc].segments[key]
                    segment = version.segments[key]
                    assert segment.positions == wanted
                    assert segment.position_basis == "none"
                    assert segment.reuse_kind == "policy_approximate"
                    assert segment.validity_certificate
                    if doc < math.floor(args.documents * shared_ratio):
                        assert segment.pages == original.pages
                    else:
                        assert not set(segment.pages).intersection(original.pages)
                    with manager.bind(plan["request_id"]) as lease:
                        actual = lease.gather(key)
                    for value, source in zip(actual, fixtures["documents"][doc]):
                        torch.testing.assert_close(value, source[head], rtol=0, atol=0)
                previous += fixtures["sizes"][doc]
            assert previous == fixtures["document_tokens"] + args.prefix_tokens
    finally:
        _cleanup(manager)
    _assert_reclaimed(manager.stats())


def test_independent_arm_really_copies_every_document(bench):
    args = _args(bench, "--shared-ratio", "1")
    manager, fixtures, versions = _cached(bench, args)
    try:
        before = manager.stats()
        bench.populate_request(manager, fixtures, args, "independent_pages", "copy", 0)
        version = manager.version("copy")
        cached_refs = {ref for source in versions.values()
                       for seg in source.segments.values() for ref in seg.pages}
        copied_refs = {ref for seg in version.segments.values() for ref in seg.pages}
        assert not cached_refs.intersection(copied_refs)
        expected = 2 * args.heads * (fixtures["document_tokens"] + args.prefix_tokens) * args.head_dim * 4
        assert manager.stats()["written_bytes"] - before["written_bytes"] == expected
    finally:
        _cleanup(manager)


def test_cow_detaches_only_repaired_head_pages_and_preserves_source(bench):
    args = _args(bench, "--shared-ratio", "1")
    manager, fixtures, versions = _cached(bench, args)
    try:
        bench.populate_request(manager, fixtures, args, "shared_pages", "repair", 0)
        original = manager.version("repair")
        patches, heads, rows = bench.repair_patches(fixtures, args)
        before = manager.stats()
        manager.update("repair", patches=patches)
        updated = manager.version("repair")
        dirty = {p.key: {i // args.page_size for i in p.indices} for p in patches}
        for key, old in original.segments.items():
            new = updated.segments[key]
            assert old.positions == new.positions
            for page, (left, right) in enumerate(zip(old.pages, new.pages)):
                assert (left != right) == (page in dirty.get(key, set()))
        assert manager.stats()["allocated_bytes"] - before["allocated_bytes"] == (
            sum(map(len, dirty.values())) * manager.pool.page_bytes
        )
        assert manager.stats()["written_bytes"] - before["written_bytes"] == (
            2 * heads * rows * args.head_dim * 4
        )
        with manager.bind("repair") as lease:
            for patch in patches:
                k, v = lease.gather(patch.key)
                torch.testing.assert_close(k[list(patch.indices)], patch.k, rtol=0, atol=0)
                torch.testing.assert_close(v[list(patch.indices)], patch.v, rtol=0, atol=0)
        assert bench.validate_cache(manager, fixtures, args, versions) == 0.0
    finally:
        _cleanup(manager)


def test_evict_cache_does_not_free_pages_still_used_by_online_request(bench):
    args = _args(bench, "--shared-ratio", "1")
    manager, fixtures, _ = _cached(bench, args)
    try:
        bench.populate_request(manager, fixtures, args, "shared_pages", "reader", 0)
        before = manager.stats()["allocated_bytes"]
        for doc in range(args.documents):
            manager.evict(f"cache-document-{doc}")
        state = manager.stats()
        assert state["cached_roots"] == 0 and state["requests"] == 1
        assert state["allocated_bytes"] == before
        with manager.bind("reader") as lease:
            k, _ = lease.gather((0, 0, "doc-0"))
            torch.testing.assert_close(k, fixtures["documents"][0][0][0], atol=0, rtol=0)
    finally:
        _cleanup(manager)
    _assert_reclaimed(manager.stats())


def test_fixed_slab_reserved_bytes_do_not_follow_active_page_savings(bench):
    args = _args(bench, "--shared-ratio", "1")
    manager, fixtures, _ = _cached(bench, args)
    try:
        initial = bench.stats_snapshot(manager)
        bench.populate_request(manager, fixtures, args, "shared_pages", "warm", 0)
        active = bench.stats_snapshot(manager)
        assert active["allocated_bytes"] > initial["allocated_bytes"]
        assert active["reserved_bytes"] == initial["reserved_bytes"]
        assert active["reserved_bytes"] == active["actual_slab_storage_bytes"]
    finally:
        _cleanup(manager)
    final = bench.stats_snapshot(manager)
    assert final["allocated_bytes"] == 0
    assert final["reserved_bytes"] == initial["reserved_bytes"] > 0


@pytest.mark.parametrize("repair_fraction", [0, 0.25, 1])
def test_blockwise_reference_matches_independent_dense_softmax(bench, repair_fraction):
    args = _args(bench, "--repair-token-ratio", str(repair_fraction))
    fixtures = bench.make_fixtures(args, args.lengths[0])
    plan = bench.request_plan(fixtures, args, "reference", 1)
    patches, heads, rows = bench.repair_patches(fixtures, args)
    pairs = [fixtures["prefixes"][1]]
    for doc in plan["order"]:
        k, v = (x.clone() for x in fixtures["documents"][doc])
        if doc == 0:
            for patch in patches:
                h = patch.key[1]
                k[h, list(patch.indices)] = patch.k
                v[h, list(patch.indices)] = patch.v
        pairs.append((k, v))
    k = torch.cat([p[0] for p in pairs], dim=1).double()
    v = torch.cat([p[1] for p in pairs], dim=1).double()
    q = fixtures["queries"][1].double()
    expected = (q @ k.transpose(-1, -2) * args.head_dim ** -0.5).softmax(-1) @ v
    actual = bench.reference_attention(fixtures, args, plan, heads, rows)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)


@pytest.fixture(scope="module")
def report(bench, tmp_path_factory):
    output = tmp_path_factory.mktemp("segpaged-cache-report") / "results.json"
    return bench.benchmark(_args(bench, "--output", str(output)))


def test_report_preserves_raw_warmup_measurement_and_true_accounting(report):
    assert report["status"] == "complete"
    json.dumps(report, allow_nan=False)
    assert "not concurrent" in report["cases"][0]["arms"][0]["raw_batches"][0]["rate_note"]
    assert "sequential" in report["protocol"]["live_requests"]
    case = report["cases"][0]
    assert len(case["fixture_sha256"]) == 64
    assert case["actual_kv_tokens"] == 35
    assert len(case["arms"]) == 2
    assert len({arm["cached_state"]["reserved_bytes"] for arm in case["arms"]}) == 1
    for arm in case["arms"]:
        assert arm["cached_state"]["requests"] == 0
        assert arm["cached_state"]["cached_roots"] == 4
        assert len(arm["raw_batches"]) == 6
        for count in [1, 3]:
            rows = [r for r in arm["raw_batches"] if r["live_requests"] == count]
            assert [r["repetition"] for r in rows] == [-1, 0, 1]
            assert [r["phase"] for r in rows] == ["warmup", "measured", "measured"]
            summary = next(s for s in case["summary"] if
                           s["backend"] == arm["backend"] and s["live_requests"] == count)
            for phase in ("attach", "repair", "descriptor", "attention_batch", "release"):
                times = [r[phase]["wall_ms"] for r in rows if r["phase"] == "measured"]
                assert all(math.isfinite(t) and t >= 0 for t in times)
                assert summary[f"{phase}_wall_ms_p50"] == statistics.median(times)
        for row in arm["raw_batches"]:
            assert row["before"]["allocated_bytes"] == row["after_release"]["allocated_bytes"]
            assert row["after_release"]["requests"] == 0
            assert row["after_release"]["pinned_pages"] == 0
            assert row["attention_batch_serial_calls"] == row["live_requests"]
            assert len(row["correctness"]["attention_error_by_request"]) == row["live_requests"]
            assert row["correctness"]["source_unchanged"]
            assert row["correctness"]["kv_max_abs_error"] == 0
            assert row["correctness"]["attention_max_abs_error"] <= 2e-5
            assert not ({"qps", "ttft", "ttft_speedup"} & row.keys())
        _assert_reclaimed(arm["final_state"])
        assert arm["final_state"]["reserved_bytes"] > 0


def test_shared_arm_reduces_active_pages_and_copy_bytes_not_reserved_slab(report):
    case = report["cases"][0]
    for count in [1, 3]:
        rows = {r["backend"]: r for r in case["summary"] if r["live_requests"] == count}
        shared, copied = rows["shared_pages"], rows["independent_pages"]
        assert shared["active_bytes_after_attach"] < copied["active_bytes_after_attach"]
        assert shared["attach_written_bytes"] < copied["attach_written_bytes"]
        assert shared["reserved_bytes"] == copied["reserved_bytes"]
        # Both arms use identical immutable partial-page repair, not a weak
        # full-history clone baseline in one arm.
        assert shared["repair_written_bytes"] == copied["repair_written_bytes"]
        assert shared["repair_copied_bytes"] == copied["repair_copied_bytes"]


def test_zero_sharing_and_zero_repair_negative_control(bench):
    args = _args(bench, "--requests", "1", "--warmup", "0", "--repeats", "1",
                 "--shared-ratio", "0", "--repair-token-ratio", "0")
    case = bench.run_case(args, 32)
    shared, copied = case["summary"]
    for name in ("active_bytes_after_attach", "active_bytes_after_repair", "reserved_bytes",
                 "attach_written_bytes", "repair_written_bytes", "repair_copied_bytes"):
        assert shared[name] == copied[name]
    for arm in case["arms"]:
        row = arm["raw_batches"][0]
        assert row["alias_counts"]["shared_page_refs"] == 0
        assert row["repair_written_bytes"] == row["repair_copied_bytes"] == 0
        _assert_reclaimed(arm["final_state"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.0])
def test_source_corruption_cannot_be_reported_as_zero_error(bench, bad):
    args = _args(bench)
    manager, fixtures, versions = _cached(bench, args)
    ref = versions[0].segments[(0, 0, "doc-0")].pages[0]
    try:
        # Fault injection intentionally violates the read-only page contract.
        manager.pool.k[ref.slot, 0, 0] += bad
        with pytest.raises((AssertionError, ValueError)):
            bench.validate_cache(manager, fixtures, args, versions)
    finally:
        _cleanup(manager)


@pytest.mark.parametrize("failure", ["raise", "nan", "finite_mismatch"])
def test_failed_attention_reclaims_requests_leases_and_cached_roots(bench, monkeypatch, failure):
    captured = []
    manager_type = bench.HeadKVManager

    class RecordingManager(manager_type):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            captured.append(self)

    def broken_attention(query, **kwargs):
        if failure == "raise":
            raise RuntimeError("injected operator failure")
        if failure == "finite_mismatch":
            return torch.full_like(query, 1000)
        return torch.full_like(query, float("nan"))

    monkeypatch.setattr(bench, "HeadKVManager", RecordingManager)
    monkeypatch.setattr(bench, "paged_attention", broken_attention)
    with pytest.raises((RuntimeError, AssertionError)):
        bench.run_case(_args(bench), 32)
    assert captured
    for manager in captured:
        _assert_reclaimed(manager.stats())


@pytest.mark.parametrize("stage", ["publication", "attachment", "descriptor"])
def test_setup_failure_reclaims_partial_state(bench, monkeypatch, stage):
    captured = []
    manager_type = bench.HeadKVManager

    class RecordingManager(manager_type):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            captured.append(self)

    monkeypatch.setattr(bench, "HeadKVManager", RecordingManager)
    if stage == "publication":
        original = manager_type.update

        def broken_update(manager, name, **kw):
            if name == "publisher-1":
                raise RuntimeError("injected source publication failure")
            return original(manager, name, **kw)

        monkeypatch.setattr(RecordingManager, "update", broken_update)
    elif stage == "attachment":
        original = bench.populate_request

        def broken_populate(*a, **kw):
            original(*a, **kw)
            raise RuntimeError("injected failure after request attachment")

        monkeypatch.setattr(bench, "populate_request", broken_populate)
    else:
        lease_type = sys.modules[manager_type.__module__].ReadLease

        def broken_descriptor(*a, **kw):
            raise RuntimeError("injected pinned descriptor failure")

        monkeypatch.setattr(lease_type, "descriptor", broken_descriptor)
    with pytest.raises(RuntimeError, match="injected"):
        bench.run_case(_args(bench), 32)
    assert captured
    for manager in captured:
        _assert_reclaimed(manager.stats())


@pytest.mark.parametrize("option,value", [
    ("--documents", "1"), ("--page-size", "0"), ("--heads", "0"),
    ("--lengths", "3"), ("--requests", "0"), ("--repeats", "0"),
    ("--warmup", "-1"), ("--shared-ratio", "1.1"),
    ("--repair-token-ratio", "-0.1"), ("--repair-head-ratio", "nan"),
])
def test_invalid_benchmark_configuration_is_rejected(bench, option, value):
    with pytest.raises(SystemExit):
        _args(bench, option, value)


def test_seeded_fixtures_are_identical_and_input_sensitive(bench):
    args = _args(bench)
    a = bench.make_fixtures(args, 32)
    b = bench.make_fixtures(args, 32)
    assert a["sha256"] == b["sha256"]
    args.seed += 1
    c = bench.make_fixtures(args, 32)
    assert a["sha256"] != c["sha256"]
    assert bench.length_arg("64K") == 65536
