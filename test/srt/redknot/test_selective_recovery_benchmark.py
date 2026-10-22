"""CPU-only plan, fair-budget and full-output report tests; no speed evidence."""
import copy
from types import SimpleNamespace

import pytest

from benchmark_redknot_nonprefix import digest
from benchmark_selective_recovery import (
    BASELINE, DEFAULT_CANDIDATES, cache_active, candidate_arm, candidate_cache_bytes, checked_candidates, cohort_role,
    enrich_row, make_engine_factory, memory_allocation, prepare_selective_plan,
)
from report_selective_recovery import build_report
from test_nonprefix_affinity_benchmark import workload, row as old_row


def candidate(**changes):
    return dict(dict(name="small-r50", global_prefix_layers=1, row_fraction=.5,
                     probe_layers=2, block_size=16), **changes)


def test_defaults_are_fixed_four_candidates_and_recomputation_is_always_present():
    data = workload()
    before = copy.deepcopy(data)
    plan = prepare_selective_plan(data, ["qa:0", "qa:1"], DEFAULT_CANDIDATES,
                                  chunks=4, replicas=2, concurrency=2)
    assert data == before
    assert plan["arms"] == [BASELINE, "redknot_g60-full", "redknot_g60-r75", "redknot_g60-r50", "redknot_g48-r50"]
    assert plan["arm_order_by_repeat"][1] == list(reversed(plan["arms"]))
    assert plan["cohort_role"] == "known_regression"
    assert plan["natural_eos"] and plan["output_tokens"] == 32
    assert plan["total_kv_budget_bytes"] == 48 * 1024**3
    assert plan["scheduling_policy"] == "fcfs"


@pytest.mark.parametrize("fault", ["empty", "many", "duplicate", "missing", "extra", "nan", "zero_fraction",
                                   "big_fraction", "bool_fraction", "negative_prefix", "zero_probe", "zero_block", "bad_name"])
def test_invalid_candidate_configs_fail_before_model(fault):
    values = [candidate()]
    if fault == "empty": values = []
    elif fault == "many": values = [candidate(name=f"r{i}") for i in range(9)]
    elif fault == "duplicate": values *= 2
    elif fault == "missing": values[0].pop("probe_layers")
    elif fault == "extra": values[0]["ignore_eos"] = True
    elif fault == "nan": values[0]["row_fraction"] = float("nan")
    elif fault == "zero_fraction": values[0]["row_fraction"] = 0
    elif fault == "big_fraction": values[0]["row_fraction"] = 1.1
    elif fault == "bool_fraction": values[0]["row_fraction"] = True
    elif fault == "negative_prefix": values[0]["global_prefix_layers"] = -1
    elif fault == "zero_probe": values[0]["probe_layers"] = 0
    elif fault == "zero_block": values[0]["block_size"] = 0
    elif fault == "bad_name": values[0]["name"] = "recomputed_fake"
    with pytest.raises(ValueError):
        checked_candidates(values)


def test_memory_budget_subtracts_full_cache_before_pool_rounding():
    budget = memory_allocation(10000, 2501, 256)
    assert budget["pool_capacity_pages"] == 29
    assert budget["pool_reserved_bytes"] == 7424
    assert budget["accounted_reserved_bytes"] == 9925
    assert budget["rounding_unused_bytes"] == 75
    with pytest.raises(ValueError):
        memory_allocation(1000, 1000, 256)
    with pytest.raises(ValueError):
        memory_allocation(1000, -1, 256)


def test_factory_changes_only_config_and_pool_with_physical_memory_receipt():
    calls, receipt = [], {}
    factory = make_engine_factory(lambda *args, **kwargs: calls.append(kwargs),
        lambda **kwargs: SimpleNamespace(**kwargs), candidate=candidate(), expected_cache_bytes=2501,
        total_bytes=10000, page_bytes=256, receipt=receipt)
    factory("model", "profile", capacity_pages=39, backend="flash", share_local=True)
    assert calls[0]["capacity_pages"] == 29
    assert calls[0]["selective_recovery"].max_offline_kv_bytes == 2501
    assert calls[0]["backend"] == "flash" and calls[0]["share_local"]
    with pytest.raises(RuntimeError, match="different total"):
        factory("model", "profile", capacity_pages=40)


def test_fullrow_candidate_uses_exact_old_path_and_retains_no_fullcache():
    calls = []
    factory = make_engine_factory(lambda *args, **kwargs: calls.append(kwargs),
        lambda **kwargs: pytest.fail("inactive config must not be constructed"),
        candidate=candidate(row_fraction=1), expected_cache_bytes=0,
        total_bytes=10000, page_bytes=256, receipt={})
    factory("model", "profile", capacity_pages=39)
    assert "selective_recovery" not in calls[0]
    assert not cache_active(candidate(row_fraction=1), 4)
    assert not cache_active(candidate(probe_layers=4), 4)


def evidence():
    data = workload()
    spec = candidate()
    plan = prepare_selective_plan(data, ["qa:0", "qa:1"], [spec],
                                  chunks=4, replicas=2, concurrency=2, repeats=1)
    names = ("benchmark_selective_recovery.py", "benchmark_nonprefix_affinity.py",
        "benchmark_nonprefix_scale.py", "benchmark_redknot_nonprefix.py", "nonprefix_profile.py",
        "nonprefix_paged.py", "nonprefix_scheduler.py", "nonprefix_affinity.py", "selective_recovery.py",
        "manager.py", "pool.py")
    result = dict(schema="redknot-selective-recovery-v1", status="complete", plan=plan,
        plan_sha256=digest(plan), workload_sha256=digest(data), model_config_sha256=digest("model"),
        hardware=dict(gpu="mock", total_bytes=128*1024**3),
        source_hashes={"/frozen/test/srt/redknot/"+name: digest(name) for name in names}, results=[])
    for arm in plan["arms"]:
        selective = arm != BASELINE
        policies = [[dict(head_type="global"), dict(head_type="local" if selective else "global")]
                    for _ in range(4)]
        profile = dict(policies=policies, sha256=digest(policies), effective_local_ratio=.5 if selective else 0)
        budget = memory_allocation(plan["total_kv_budget_bytes"], 4096 if selective else 0, 256)
        one = old_row(plan, arm, 0, seconds=4 if selective else 8)
        one["service"]["admission_policy"] = "fcfs"
        recovery_config = dict(block_size=16, row_fraction=.5, probe_layers=2, max_offline_kv_bytes=4096) if selective else None
        for stage in ("engine_stats_before", "engine_stats_after"):
            one["service"][stage] = dict(backend="flash", share_local=True,
                local_head_ratio=.5 if selective else 0, offline_kv_cache_bytes=4096 if selective else 0,
                offline_kv_cache_limit_bytes=4096 if selective else 0, selective_recovery=recovery_config,
                pool=dict(reserved_slab_bytes=budget["pool_reserved_bytes"], capacity_pages=budget["pool_capacity_pages"]))
        for record in one["service"]["records"]:
            record["text"] = record["item_key"].split(":")[1]
            record["admission_stats"] = dict(projected_rows_per_layer=[40, 40, 20, 20] if selective else [40]*4,
                ffn_rows_per_layer=[40, 40, 20, 20] if selective else [40]*4, selected_token_rows=20 if selective else 40,
                recovery_mode="experimental_block_selective_v1" if selective else "full_rows")
        enrich_row(one, candidate=spec if selective else None, profile_metadata=profile, budget=budget)
        result["results"].append(one)
    return result, data, plan


def test_complete_same_budget_report_keeps_quality_and_execution_rows_separate():
    report = build_report(*evidence())
    assert report["integrity_verified"], report["errors"]
    pair = report["comparisons"][0]
    assert pair["qps_ratio"] == 2
    assert pair["observed_all_prefix_2percent_gate"]
    assert pair["text_identical"] == pair["token_identical"] == 4
    assert pair["executed_row_savings"]["ffn_rows_per_layer"]["fraction_saved"] == .25
    assert not pair["quality_certified"] and not pair["statistical_significance_claimed"]
    assert not report["source_file_bytes_verified"]


@pytest.mark.parametrize("fault", ["missing_row", "duplicate_row", "failed", "cleanup", "partial", "planhash", "workloadhash",
    "source", "baseline_sparse", "baseline_rows", "memory_extra", "memory_hidden", "memory_actual", "memory_missing",
    "config", "policy", "row_missing", "row_fake", "score", "hash", "mode"])
def test_invalid_evidence_has_no_headline_ratio(fault):
    raw, data, plan = evidence()
    base, row = raw["results"]
    service = row["service"]
    if fault == "missing_row": raw["results"].pop()
    elif fault == "duplicate_row": raw["results"].append(copy.deepcopy(row))
    elif fault == "failed": row["status"] = "failed"
    elif fault == "cleanup": row["cleanup_status"] = "failed"
    elif fault == "partial": raw["status"] = "running"
    elif fault == "planhash": raw["plan_sha256"] = "wrong"
    elif fault == "workloadhash": raw["workload_sha256"] = "wrong"
    elif fault == "source": raw["source_hashes"] = {}
    elif fault == "baseline_sparse": base["service"]["engine_stats_before"]["selective_recovery"] = {}
    elif fault == "baseline_rows": base["service"]["records"][0]["admission_stats"]["projected_rows_per_layer"][0] = 20
    elif fault == "memory_extra": row["memory_accounting"]["total_kv_budget_bytes"] += 1
    elif fault == "memory_hidden": service["engine_stats_before"]["offline_kv_cache_bytes"] += 256
    elif fault == "memory_actual": service["engine_stats_after"]["pool"]["reserved_slab_bytes"] += 256
    elif fault == "memory_missing": row.pop("memory_accounting")
    elif fault == "config": service["engine_stats_before"]["selective_recovery"]["row_fraction"] = .75
    elif fault == "policy": service["admission_policy"] = "share_affinity"
    elif fault == "row_missing": service["records"][0]["admission_stats"].pop("ffn_rows_per_layer")
    elif fault == "row_fake": service["records"][0]["admission_stats"]["ffn_rows_per_layer"] = [40,40,1,1]
    elif fault == "score": service["records"][0]["accuracy"]["f1"] = .5
    elif fault == "hash": service["records"][0]["output_text_sha256"] = "wrong"
    elif fault == "mode": service["records"][0]["admission_stats"]["recovery_mode"] = "full_rows"
    report = build_report(raw, data, plan)
    assert not report["integrity_verified"] and report["comparisons"] == []


def test_quality_failure_remains_visible_even_when_qps_improves():
    raw, data, plan = evidence()
    record = raw["results"][1]["service"]["records"][0]
    record.update(text="wrong", output_ids=[5,6], accuracy=dict(f1=0.,em=0.),
                  output_text_sha256=digest(b"wrong"), output_ids_sha256=digest([5,6]))
    report = build_report(raw, data, plan)
    assert report["integrity_verified"]
    pair = report["comparisons"][0]
    assert pair["qps_ratio"] == 2
    assert not pair["observed_all_prefix_2percent_gate"]
    assert not pair["measured_qps_uplift_and_quality_gate"]
    assert pair["text_identical"] == 3


def test_replicas_do_not_inflate_independent_qa_count():
    report = build_report(*evidence())
    assert report["independent_qa_count"] == 2
    assert report["comparisons"][0]["requests_compared"] == 4


def test_prospectively_frozen_holdout_keeps_role_and_exclusion_provenance():
    data = workload()
    data.update(selection_role="prospectively_frozen_holdout_for_selective_recovery",
                seed=2026101902, excluded_calibration_keys=["qa:old"])
    plan = prepare_selective_plan(data, ["qa:0", "qa:1"], [candidate()], replicas=2, concurrency=2)
    assert plan["cohort_role"] == "prospectively_frozen_holdout"
    assert plan["cohort_selection_provenance"]["seed"] == 2026101902
    assert "not all document token rows" in plan["head_policy_prefix_semantics"]


@pytest.mark.parametrize("fault", ["overlap", "missing_seed", "missing_exclusions"])
def test_invalid_holdout_claim_rejects(fault):
    data = workload()
    data.update(selection_role="prospectively_frozen_holdout_for_selective_recovery",
                seed=2026101902, excluded_calibration_keys=["qa:old"])
    if fault == "overlap": data["excluded_calibration_keys"].append("qa:0")
    elif fault == "missing_seed": data.pop("seed")
    elif fault == "missing_exclusions": data.pop("excluded_calibration_keys")
    with pytest.raises(ValueError, match="holdout declaration"):
        cohort_role(data)


def test_optional_selector_preserves_legacy_plan_and_configuration_shape():
    data = workload()
    legacy = prepare_selective_plan(data, ["qa:0", "qa:1"], [candidate()], replicas=2, concurrency=2)
    assert "selection_mode" not in legacy["candidates"][0]
    assert "query_selection_contract" not in legacy
    explicit = candidate(selection_mode="query_attention")
    plan = prepare_selective_plan(data, ["qa:0", "qa:1"], [explicit], replicas=2, concurrency=2)
    assert plan["candidates"] == [explicit]
    assert "no answers" in plan["query_selection_contract"]["source"]
    assert "unchanged" in plan["query_selection_contract"]["normal_query_forward"]
    assert checked_candidates([candidate(selection_mode="kv_drift")])[0]["selection_mode"] == "kv_drift"


@pytest.mark.parametrize("mode", [None, "unknown", True, [], {}])
def test_invalid_optional_selection_mode_rejected(mode):
    with pytest.raises(ValueError, match="selection_mode"):
        checked_candidates([candidate(selection_mode=mode)])


def test_factory_forwards_only_explicit_active_query_mode():
    calls = []
    factory = make_engine_factory(lambda *args, **kwargs: calls.append(kwargs),
        lambda **kwargs: SimpleNamespace(**kwargs), candidate=candidate(selection_mode="query_attention"),
        expected_cache_bytes=2501, total_bytes=10000, page_bytes=256, receipt={})
    factory("model", "profile", capacity_pages=39)
    assert calls[0]["selective_recovery"].selection_mode == "query_attention"
    old_calls = []
    old = make_engine_factory(lambda *args, **kwargs: old_calls.append(kwargs),
        lambda **kwargs: SimpleNamespace(**kwargs), candidate=candidate(),
        expected_cache_bytes=2501, total_bytes=10000, page_bytes=256, receipt={})
    old("model", "profile", capacity_pages=39)
    assert not hasattr(old_calls[0]["selective_recovery"], "selection_mode")


def query_evidence():
    raw, data, _ = evidence()
    spec = candidate(selection_mode="query_attention")
    plan = prepare_selective_plan(data, ["qa:0", "qa:1"], [spec],
                                  chunks=4, replicas=2, concurrency=2, repeats=1)
    raw.update(plan=plan, plan_sha256=digest(plan))
    items = {item["key"]: item for item in data["items"]}
    for row in raw["results"]:
        if row["arm"] == BASELINE:
            continue
        row["candidate"] = copy.deepcopy(spec)
        for stage in ("engine_stats_before", "engine_stats_after"):
            row["service"][stage]["selective_recovery"]["selection_mode"] = "query_attention"
        for record in row["service"]["records"]:
            ids = items[record["item_key"]]["query_token_ids"]
            record["admission_stats"].update(selection_mode="query_attention",
                recovery_mode="experimental_query_selective_v3", recovery_query_tokens=len(ids),
                recovery_query_sha256=digest(ids), query_projected_rows_per_layer=[len(ids)]*2+[0]*2,
                query_ffn_rows_per_layer=[len(ids)]+[0]*3)
    return raw, data, plan


def test_query_report_counts_additional_probe_work_separately_without_relaxing_quality():
    report = build_report(*query_evidence())
    assert report["integrity_verified"], report["errors"]
    pair = report["comparisons"][0]
    assert pair["observed_all_prefix_2percent_gate"]
    assert pair["additional_query_probe_rows"]["query_projected_rows_per_layer"]["total"] == 16
    assert pair["additional_query_probe_rows"]["query_ffn_rows_per_layer"]["total"] == 8
    assert pair["executed_row_savings"]["ffn_rows_per_layer"]["fraction_saved"] == .25
    assert pair["document_plus_extra_query_recovery_row_savings"]["ffn_rows_per_layer"]["fraction_saved"] == pytest.approx(.2375)
    assert pair["document_plus_extra_query_recovery_row_savings"]["projected_rows_per_layer"]["fraction_saved"] == pytest.approx(.225)
    assert "document rows only" in pair["row_savings_scope"]
    assert report["arm_summary"][BASELINE]["additional_query_probe_rows"]["query_projected_rows_per_layer"]["total"] == 0
    assert not pair["quality_certified"]


@pytest.mark.parametrize("fault", ["query_hash", "query_count", "query_rows_missing", "query_rows_fake",
    "query_rows_bool", "query_ffn_extra", "query_mode", "selection_mode", "config_mode", "baseline_hidden_query"])
def test_query_identity_or_execution_mismatch_never_receives_headline(fault):
    raw, data, plan = query_evidence()
    row = raw["results"][1]
    stats = row["service"]["records"][0]["admission_stats"]
    if fault == "query_hash": stats["recovery_query_sha256"] = digest([99])
    elif fault == "query_count": stats["recovery_query_tokens"] += 1
    elif fault == "query_rows_missing": stats.pop("query_projected_rows_per_layer")
    elif fault == "query_rows_fake": stats["query_projected_rows_per_layer"] = [0]*4
    elif fault == "query_rows_bool": stats["query_projected_rows_per_layer"][2] = False
    elif fault == "query_ffn_extra": stats["query_ffn_rows_per_layer"][1] = 2
    elif fault == "query_mode": stats["recovery_mode"] = "experimental_block_selective_v1"
    elif fault == "selection_mode": stats["selection_mode"] = "kv_drift"
    elif fault == "config_mode": row["service"]["engine_stats_before"]["selective_recovery"].pop("selection_mode")
    elif fault == "baseline_hidden_query":
        raw["results"][0]["service"]["records"][0]["admission_stats"]["query_projected_rows_per_layer"] = [2]*4
    report = build_report(raw, data, plan)
    assert not report["integrity_verified"] and report["comparisons"] == []


def test_query_selector_still_fails_original_all_prefix_quality_gate():
    raw, data, plan = query_evidence()
    record = raw["results"][1]["service"]["records"][0]
    record.update(text="wrong", output_ids=[5,6], accuracy=dict(f1=0.,em=0.),
                  output_text_sha256=digest(b"wrong"), output_ids_sha256=digest([5,6]))
    report = build_report(raw, data, plan)
    assert report["integrity_verified"]
    assert report["comparisons"][0]["qps_ratio"] == 2
    assert not report["comparisons"][0]["observed_all_prefix_2percent_gate"]
    assert not report["comparisons"][0]["measured_qps_uplift_and_quality_gate"]


def hidden_evidence():
    raw, data, _ = query_evidence()
    for item in data["items"]:
        item["prefix_token_ids"] = item["prefix_token_ids"][:2]
        item["exact_prompt_sha256"] = item["exact_prompt_sha256"][:2]
    spec = candidate(selection_mode="query_attention", recovery_mode="hidden_residual_int8")
    plan = prepare_selective_plan(data, ["qa:0", "qa:1"], [spec],
                                  chunks=4, replicas=2, concurrency=2, repeats=1)
    cfg = dict(hidden_size=128, num_hidden_layers=4)
    lengths = [n for group in plan["groups"] for n in group["chunk_lengths"]]
    cache_bytes = sum(lengths)*2*(128+4)
    raw.update(plan=plan, plan_sha256=digest(plan), workload_sha256=digest(data),
        model_config_sha256=digest(cfg), hidden_residual_cache_estimator=dict(
            model_config=cfg, document_token_lengths=lengths, group_size=128, value_bytes=1, scale_bytes=4))
    raw["source_hashes"]["/frozen/test/srt/redknot/hidden_residual_recovery.py"] = digest("hidden_residual_recovery.py")
    row = raw["results"][1]
    row["candidate"] = spec
    budget = memory_allocation(plan["total_kv_budget_bytes"], cache_bytes, 256)
    row["memory_accounting"] = budget
    for stage in ("engine_stats_before", "engine_stats_after"):
        stats = row["service"][stage]
        stats.update(offline_kv_cache_bytes=0, offline_kv_cache_limit_bytes=0, offline_kv_cache_peak_bytes=0,
            offline_hidden_cache_bytes=cache_bytes, offline_hidden_cache_limit_bytes=cache_bytes,
            offline_hidden_cache_peak_bytes=cache_bytes,
            offline_hidden_quantization=dict(dtype="int8", group_size=128, scale_dtype="float32"),
            pool=dict(reserved_slab_bytes=budget["pool_reserved_bytes"], capacity_pages=budget["pool_capacity_pages"]))
        stats["selective_recovery"].update(recovery_mode="hidden_residual_int8", max_offline_kv_bytes=cache_bytes)
    for record in row["service"]["records"]:
        stats = record["admission_stats"]
        stats.update(recovery_mode="experimental_hidden_residual_int8_v4", projected_rows_per_layer=None,
            row_execution_fraction=None, q_rows_per_layer=[40, 40, 20, 20], kv_rows_per_layer=[40]*4,
            attention_rows_per_layer=[40, 40, 20, 20], query_attention_rows_per_layer=[2, 0, 0, 0])
    return raw, data, plan


def test_hidden_mode_plan_is_explicit_and_requires_all_frozen_items_prefixes():
    raw, data, plan = hidden_evidence()
    assert plan["hidden_residual_contract"]["quantization"]["group_size"] == 128
    legacy = evidence()[2]
    assert "hidden_residual_contract" not in legacy
    with pytest.raises(ValueError, match="every frozen"):
        prepare_selective_plan(data, ["qa:1", "qa:0"], plan["candidates"], chunks=4, replicas=2, concurrency=2)
    data["items"][0]["prefix_token_ids"].append([99])
    with pytest.raises(ValueError, match="every frozen"):
        prepare_selective_plan(data, ["qa:0", "qa:1"], plan["candidates"], chunks=4, replicas=2, concurrency=2)


@pytest.mark.parametrize("mode", [None, "unknown", True, [], {}])
def test_invalid_optional_recovery_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="recovery_mode"):
        checked_candidates([candidate(recovery_mode=mode)])


def test_hidden_requires_query_and_default_offline_mode_keeps_old_config():
    with pytest.raises(ValueError, match="query_attention"):
        checked_candidates([candidate(recovery_mode="hidden_residual_int8")])
    calls = []
    for mode in (None, "offline_kv", "hidden_residual_int8"):
        spec = candidate(selection_mode="query_attention")
        if mode is not None: spec["recovery_mode"] = mode
        factory = make_engine_factory(lambda *args, **kwargs: calls.append(kwargs),
            lambda **kwargs: SimpleNamespace(**kwargs), candidate=spec,
            expected_cache_bytes=1000, total_bytes=10000, page_bytes=256, receipt={})
        factory("model", "profile", capacity_pages=39)
    assert not hasattr(calls[0]["selective_recovery"], "recovery_mode")
    assert not hasattr(calls[1]["selective_recovery"], "recovery_mode")
    assert calls[2]["selective_recovery"].recovery_mode == "hidden_residual_int8"


def test_cache_estimator_chooses_one_representation_and_zero_for_full_fallback():
    calls = []
    kv = lambda *args: calls.append(("kv", args)) or 4096
    hidden = lambda *args, **kwargs: calls.append(("hidden", args, kwargs)) or 8192
    cfg = SimpleNamespace(num_hidden_layers=4)
    spec = candidate(selection_mode="query_attention", recovery_mode="hidden_residual_int8")
    assert candidate_cache_bytes(spec, cfg, [40, 40], "mock_dtype", estimate_kv=kv, estimate_hidden=hidden) == 8192
    assert len(calls) == 1 and calls[0][0] == "hidden" and calls[0][2] == dict(group_size=128)
    assert candidate_cache_bytes(candidate(), cfg, [40, 40], "mock_dtype", estimate_kv=kv) == 4096
    assert candidate_cache_bytes(dict(spec, row_fraction=1), cfg, [40], "mock", estimate_kv=kv) == 0
    assert candidate_cache_bytes(None, cfg, [40], "mock", estimate_kv=kv) == 0
    with pytest.raises(ValueError, match="estimator"):
        candidate_cache_bytes(spec, cfg, [40], "mock", estimate_kv=kv)


def test_hidden_report_separates_full_kv_from_sparse_q_attention_ffn_including_query():
    report = build_report(*hidden_evidence())
    assert report["integrity_verified"], report["errors"]
    pair = report["comparisons"][0]
    assert pair["observed_all_prefix_2percent_gate"] and pair["qps_ratio"] == 2
    saved = pair["executed_row_savings"]
    assert set(saved) == {"q_rows_per_layer", "kv_rows_per_layer", "attention_rows_per_layer", "ffn_rows_per_layer"}
    assert saved["kv_rows_per_layer"]["fraction_saved"] == 0
    assert saved["q_rows_per_layer"]["fraction_saved"] == .25
    combined = pair["document_plus_extra_query_recovery_row_savings"]
    assert combined["kv_rows_per_layer"]["fraction_saved"] == pytest.approx(-.025)
    assert combined["q_rows_per_layer"]["fraction_saved"] == pytest.approx(.225)
    assert combined["attention_rows_per_layer"]["fraction_saved"] == pytest.approx(.2375)
    assert combined["ffn_rows_per_layer"]["fraction_saved"] == pytest.approx(.2375)
    assert "no unified QKV" in pair["row_savings_scope"]
    storage = report["arm_summary"][pair["candidate"]]["offline_recovery_storage_by_repeat"][0]
    assert storage["full_kv_bytes"] == 0 and storage["hidden_int8_and_scale_bytes"] == 21120
    assert storage["total_retained_bytes"] == 21120


@pytest.mark.parametrize("fault", ["missing_source", "model_hash", "geometry", "lengths", "scale", "group", "formula",
    "hidden_actual", "hidden_limit", "hidden_peak", "quantization", "both_caches", "old_cache_peak",
    "baseline_hidden_cache", "wrong_mode", "legacy_qkv", "legacy_ratio", "missing_q", "kv_sparse",
    "q_full", "attention_full", "query_attention_missing", "query_attention_wrong", "fake_config"])
def test_hidden_unaccounted_storage_or_misleading_rows_has_no_ratio(fault):
    raw, data, plan = hidden_evidence()
    row = raw["results"][1]
    before = row["service"]["engine_stats_before"]
    admission = row["service"]["records"][0]["admission_stats"]
    if fault == "missing_source": raw["source_hashes"].pop("/frozen/test/srt/redknot/hidden_residual_recovery.py")
    elif fault == "model_hash": raw["model_config_sha256"] = digest("other")
    elif fault == "geometry": raw["hidden_residual_cache_estimator"]["model_config"]["num_hidden_layers"] += 1
    elif fault == "lengths": raw["hidden_residual_cache_estimator"]["document_token_lengths"].pop()
    elif fault == "scale": raw["hidden_residual_cache_estimator"]["scale_bytes"] = 2
    elif fault == "group": raw["hidden_residual_cache_estimator"]["group_size"] = 64
    elif fault == "formula":
        cfg = raw["hidden_residual_cache_estimator"]["model_config"]
        cfg["hidden_size"] = 64
        raw["model_config_sha256"] = digest(cfg)
    elif fault == "hidden_actual": before["offline_hidden_cache_bytes"] += 1
    elif fault == "hidden_limit": before["offline_hidden_cache_limit_bytes"] += 1
    elif fault == "hidden_peak": before["offline_hidden_cache_peak_bytes"] += 1
    elif fault == "quantization": before["offline_hidden_quantization"]["scale_dtype"] = "bfloat16"
    elif fault == "both_caches": before["offline_kv_cache_bytes"] = 1
    elif fault == "old_cache_peak": before["offline_kv_cache_peak_bytes"] = 1
    elif fault == "baseline_hidden_cache": raw["results"][0]["service"]["engine_stats_before"]["offline_hidden_cache_bytes"] = 1
    elif fault == "wrong_mode": admission["recovery_mode"] = "experimental_query_selective_v3"
    elif fault == "legacy_qkv": admission["projected_rows_per_layer"] = [40, 40, 20, 20]
    elif fault == "legacy_ratio": admission["row_execution_fraction"] = .75
    elif fault == "missing_q": admission.pop("q_rows_per_layer")
    elif fault == "kv_sparse": admission["kv_rows_per_layer"] = [40, 40, 20, 20]
    elif fault == "q_full": admission["q_rows_per_layer"] = [40]*4
    elif fault == "attention_full": admission["attention_rows_per_layer"] = [40]*4
    elif fault == "query_attention_missing": admission.pop("query_attention_rows_per_layer")
    elif fault == "query_attention_wrong": admission["query_attention_rows_per_layer"] = [2, 2, 0, 0]
    elif fault == "fake_config": before["selective_recovery"].pop("recovery_mode")
    report = build_report(raw, data, plan)
    assert not report["integrity_verified"] and not report["comparisons"]


def test_hidden_mode_does_not_relax_existing_full_output_quality_gate():
    raw, data, plan = hidden_evidence()
    record = raw["results"][1]["service"]["records"][0]
    record.update(text="wrong", output_ids=[5,6], accuracy=dict(f1=0.,em=0.),
                  output_text_sha256=digest(b"wrong"), output_ids_sha256=digest([5,6]))
    report = build_report(raw, data, plan)
    assert report["integrity_verified"]
    assert not report["comparisons"][0]["observed_all_prefix_2percent_gate"]
    assert not report["comparisons"][0]["measured_qps_uplift_and_quality_gate"]
