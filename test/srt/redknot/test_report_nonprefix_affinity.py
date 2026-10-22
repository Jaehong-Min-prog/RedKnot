"""Synthetic reporter evidence tests, not inference/performance measurements."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def reporter():
    path = Path(__file__).with_name("report_nonprefix_affinity.py")
    spec = importlib.util.spec_from_file_location("affinity_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.write_text(json.dumps(value))


def fixture_run(reporter, root, *, repeats=2, one_primary=False):
    analyzer, _ = reporter.load_analyzer()
    items = []
    for group in range(2):
        document, prefixes, query = [10+group, 12, 13, 14], [[20], [21, 22]], [30, group]
        item = dict(key=f"qa:{group}", primary_qa_key="one-primary" if one_primary else f"qa:{group}",
            dataset="qa", answers=["yes"], question="Original question?", context_tokens=4,
            document_token_ids=[document], document_tokens_sha256=reporter.digest([document]),
            prefix_token_ids=prefixes, query_token_ids=query,
            exact_prompt_sha256=[reporter.digest(p+document+query) for p in prefixes])
        items.append(item)
    manifest = dict(schema="redknot-nonprefix-frozen-v1", items=items, max_new_tokens=32)
    plan = analyzer.prepare_plan(manifest, [x["key"] for x in items], chunks=2, replicas=2,
                                 concurrency=2, pool_gib=1., repeats=repeats)
    rows = []
    local_ratio = 15/512
    for repeat in range(repeats):
        for arm, seconds in zip(analyzer.ARMS, [16., 12., 8., 6., 4.]):
            records = [dict(request_id=s["request_id"], item_key=s["item_key"], prefix_index=s["prefix_index"],
                status="completed", output_ids=[1, 2], text="yes", accuracy=dict(f1=1., em=1.),
                exact_prompt_sha256=s["exact_prompt_sha256"], arrival_seconds=s["arrival_seconds"],
                max_new_tokens=32, queue_wait_seconds=float(index), ttft_seconds=index+1.,
                service_ttft_seconds=.5, latency_seconds=index+2., admission_started_seconds=float(index),
                capacity_blocks=0)
                for index, s in enumerate(plan["requests"])]
            stats = dict(backend="flash", share_local=not arm.startswith("redknot_private"),
                         local_head_ratio=0. if arm.startswith("recomputed") else local_ratio,
                         bulk_head_alias_publication=True, pool={"high_water_pages":10})
            service = dict(status="complete", errors=[], max_active=2, ignore_eos=False,
                total_wall_seconds=seconds, completed_requests=4, failed_requests=0, cancelled_requests=0,
                completed_output_tokens=8, peak_active_requests=2, max_forward_batch=2,
                admission_policy="share_affinity" if arm.endswith("_affinity") else "fcfs",
                engine_stats_before=stats, engine_stats_after=stats,
                stages_seconds={"admission":seconds/2, "decode_forward":seconds/2}, records=records)
            rows.append(dict(arm=arm, repeat=repeat, status="complete", cleanup_status="complete",
                fatal=False, service=service, warmups=[{"completed_requests":99,"total_wall_seconds":1000.}]))
    raw = dict(schema="redknot-nonprefix-affinity-v1", status="complete", plan=plan,
        workload_sha256=reporter.digest(manifest), plan_sha256=reporter.digest(plan),
        model_config_sha256="model-hash", source_hashes={"original.py":"original-code-hash"},
        profile={"effective_local_ratio":local_ratio}, results=rows)
    root.mkdir(parents=True, exist_ok=True)
    write(root/"results.json", raw)
    write(root/"frozen_workload.json", manifest)
    write(root/"frozen_affinity_plan.json", plan)
    return raw, manifest, plan


def test_verified_main_pairs_are_distinct_and_raw_is_unchanged(reporter, tmp_path):
    raw, manifest, plan = fixture_run(reporter, tmp_path)
    before = {p.name:p.read_bytes() for p in tmp_path.glob("*.json")}
    result = reporter.build_report(tmp_path)
    assert result["integrity_verified"]
    assert result["integrity_errors"] == []
    assert result["independent_primary_qa_count"] == 2
    assert result["quality_certified"] is False
    assert result["gpu_correctness_gate"]["automatically_verified"] is False
    assert result["gpu_correctness_gate"]["status"] == "not_automatically_verified"
    assert result["main_pairs"]["scheduling_only"]["qps_ratio"] == 2
    assert result["main_pairs"]["versus_full_recomputed_paged"]["qps_ratio"] == 4
    assert result["main_pairs"]["scheduling_only"]["quality_gate_status"] == "passed_observed_sample_only"
    assert result["main_pairs"]["scheduling_only"]["quality_certified"] is False
    shared = result["arm_summary"]["redknot_shared_affinity"]
    assert shared["headline_qps"] == 1
    assert shared["completed_requests"] == 8  # discarded warmup is never counted
    assert shared["generated_output_tokens"] == 16
    assert shared["ttft_seconds"]["p95"] == 4
    assert shared["queue_wait_seconds"]["p95"] == 3
    assert shared["actual_max_active"] == 2
    assert shared["full_output_scores_verified"]
    assert result["selected_full_frozen_inputs"] == manifest["items"]
    assert result["full_results"][0]["full_outputs"] == raw["results"][0]["service"]["records"]
    assert result["provenance"]["original_source_hashes"] == raw["source_hashes"]
    assert result["provenance"]["results"]["file_sha256"] == hashlib.sha256(before["results.json"]).hexdigest()
    assert {p.name:p.read_bytes() for p in tmp_path.glob("*.json")} == before
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["running", "missing_repeat", "failed_cleanup", "duplicate", "input_hash"])
def test_partial_or_failed_pair_never_has_final_ratio_or_quality_gate(reporter, tmp_path, fault):
    raw, _, _ = fixture_run(reporter, tmp_path)
    if fault == "running": raw["status"] = "running"
    if fault == "missing_repeat": raw["results"].pop()
    if fault == "failed_cleanup": raw["results"][-1]["cleanup_status"] = "failed"
    if fault == "duplicate": raw["results"].append(copy.deepcopy(raw["results"][-1]))
    if fault == "input_hash": raw["results"][-1]["service"]["records"][0]["exact_prompt_sha256"] = "bad"
    write(tmp_path/"results.json", raw)
    result = reporter.build_report(tmp_path)
    pair = result["main_pairs"]["scheduling_only"]
    assert pair["qps_ratio"] is None
    assert pair["quality_gate_status"] == "incomplete_or_unverified"
    assert pair["observed_2percent_gate"] is False
    assert pair["quality_certified"] is False


@pytest.mark.parametrize("fault", ["manifest_hash", "plan_hash", "request_order", "missing_plan"])
def test_frozen_integrity_is_checked_even_when_results_claim_complete(reporter, tmp_path, fault):
    raw, manifest, plan = fixture_run(reporter, tmp_path)
    if fault == "manifest_hash":
        manifest["items"][0]["question"] = "changed"
        write(tmp_path/"frozen_workload.json", manifest)
    if fault == "plan_hash":
        plan["concurrency"] = 99
        write(tmp_path/"frozen_affinity_plan.json", plan)
    if fault == "request_order":
        plan["requests"] = list(reversed(plan["requests"]))
        raw["plan"] = plan
        raw["plan_sha256"] = reporter.digest(plan)
        write(tmp_path/"frozen_affinity_plan.json", plan)
        write(tmp_path/"results.json", raw)
    if fault == "missing_plan": (tmp_path/"frozen_affinity_plan.json").unlink()
    result = reporter.build_report(tmp_path)
    assert result["integrity_verified"] is False
    assert result["main_pairs"]["scheduling_only"]["qps_ratio"] is None
    assert result["full_results"][0]["full_outputs"][0]["text"] == "yes"


@pytest.mark.parametrize("fault", ["scheduler", "sharing", "allglobal", "backend"])
def test_execution_path_metadata_must_match_arm(reporter, tmp_path, fault):
    raw, _, _ = fixture_run(reporter, tmp_path)
    service = raw["results"][-1]["service"]
    if fault == "scheduler": service["admission_policy"] = "fcfs"
    if fault == "sharing": service["engine_stats_after"]["share_local"] = False
    if fault == "allglobal": service["engine_stats_after"]["local_head_ratio"] = 0.
    if fault == "backend": service["engine_stats_after"]["backend"] = "unknown"
    write(tmp_path/"results.json", raw)
    report = reporter.build_report(tmp_path)
    assert report["arm_summary"]["redknot_shared_affinity"]["execution_path_errors"]
    assert report["main_pairs"]["scheduling_only"]["qps_ratio"] is None


def test_scores_checked_against_full_text_not_only_stored_accuracy(reporter, tmp_path):
    raw, _, _ = fixture_run(reporter, tmp_path)
    raw["results"][-1]["service"]["records"][0]["text"] = "No. Full output was changed."
    write(tmp_path/"results.json", raw)
    report = reporter.build_report(tmp_path)
    pair = report["main_pairs"]["scheduling_only"]
    assert pair["qps_ratio"] == 2  # execution still measured, quality is unverifiable
    assert pair["observed_2percent_gate"] is False
    assert pair["quality_gate_status"] == "incomplete_or_unverified"
    assert report["arm_summary"]["redknot_shared_affinity"]["score_integrity_errors"]


def test_zero_baseline_is_noninformative_not_passed(reporter, tmp_path):
    raw, _, _ = fixture_run(reporter, tmp_path)
    for row in raw["results"]:
        for record in row["service"]["records"]:
            record.update(text="no", accuracy={"f1":0., "em":0.})
    write(tmp_path/"results.json", raw)
    pair = reporter.build_report(tmp_path)["main_pairs"]["scheduling_only"]
    assert pair["quality_gate_status"] == "non_informative_or_missing_metric"
    assert pair["observed_2percent_gate"] is False
    assert pair["all_prefix"]["f1_relative_drop_percent"] is None


def test_absolute_pp_and_relative_percent_are_both_gated(reporter):
    analyzer, _ = reporter.load_analyzer()
    metric = analyzer._quality_comparison({"f1":.5, "em":1.}, {"f1":.49, "em":1.})
    assert metric["f1_drop_pp"] == pytest.approx(1)
    assert metric["f1_relative_drop_percent"] == pytest.approx(2)
    worse = analyzer._quality_comparison({"f1":.5, "em":1.}, {"f1":.485, "em":1.})
    comparison = dict(primary_replica0=worse, all_prefix=worse,
                      per_group_quality={}, observed_2percent_gate=False)
    assert worse["f1_drop_pp"] == pytest.approx(1.5)
    assert worse["f1_relative_drop_percent"] == pytest.approx(3.)
    assert reporter.gate_status(comparison, True, True, analyzer) == "failed_observed_2percent_gate"


def test_primary_variants_do_not_inflate_independent_sample_count(reporter, tmp_path):
    fixture_run(reporter, tmp_path, one_primary=True)
    report = reporter.build_report(tmp_path)
    assert report["independent_primary_qa_count"] == 1
    assert report["arm_summary"]["redknot_shared_affinity"]["completed_requests"] == 8


def test_cli_default_json_and_refuse_overwrite(reporter, tmp_path, capsys):
    fixture_run(reporter, tmp_path)
    assert reporter.main(["--input", str(tmp_path)]) == 0
    output = tmp_path/"report_nonprefix_affinity.json"
    assert json.loads(output.read_text())["quality_certified"] is False
    printed = capsys.readouterr().out
    assert "MAIN scheduling_only" in printed and "MAIN versus_full_recomputed_paged" in printed
    with pytest.raises(SystemExit): reporter.main(["--input", str(tmp_path)])
    with pytest.raises(SystemExit): reporter.main(["--input", str(tmp_path), "--output", str(tmp_path/"results.json")])


def test_missing_result_returns_diagnostic_report_not_fake_success(reporter, tmp_path):
    result = reporter.build_report(tmp_path)
    assert not result["integrity_verified"]
    assert result["integrity_errors"]
    assert result["arm_summary"] == {}
    assert result["quality_certified"] is False


def test_external_gpu_assertions_preserved_without_inferring_pass(reporter, tmp_path):
    fixture_run(reporter, tmp_path)
    log = tmp_path/"gpu_assertions.log"
    log.write_text("3 passed, 0 skipped; external run claims CUDA\n")
    original = log.read_bytes()
    gate = reporter.build_report(tmp_path, gpu_gate_evidence=[log])["gpu_correctness_gate"]
    assert gate["status"] == "external_evidence_unverified"
    assert gate["automatically_verified"] is False
    assert gate["evidence"][0]["full_text"] == original.decode()
    assert gate["evidence"][0]["file_sha256"] == hashlib.sha256(original).hexdigest()
    assert gate["evidence"][0]["automatically_parsed"] is False
    assert log.read_bytes() == original
    with pytest.raises(SystemExit):
        reporter.main(["--input", str(tmp_path), "--gpu-gate-evidence", str(log), "--output", str(log)])


def test_missing_external_gpu_gate_log_is_visible(reporter, tmp_path):
    fixture_run(reporter, tmp_path)
    gate = reporter.build_report(tmp_path, gpu_gate_evidence=[tmp_path/"missing.log"])["gpu_correctness_gate"]
    assert gate["status"] == "not_automatically_verified"
    assert gate["automatically_verified"] is False
    assert gate["evidence_errors"]
