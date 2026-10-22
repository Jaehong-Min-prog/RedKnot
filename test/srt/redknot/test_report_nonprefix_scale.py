"""Synthetic evidence-schema tests, not model or performance measurements."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def reporter():
    path = Path(__file__).with_name("report_nonprefix_scale.py")
    spec = importlib.util.spec_from_file_location("scale_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.write_text(json.dumps(value))


def fixture_job(reporter, root, *, job="gpu0", primary="qa:0", variant="medium", repeats=1):
    directory = root/job
    directory.mkdir(parents=True)
    flat, prefixes, query = [1, 2, 3, 4], [[10], [11, 12]], [20]
    key = primary+":"+variant
    item = dict(key=key, primary_qa_key=primary, dataset="qa", context_tokens=4,
                document_token_ids=[flat], document_tokens_sha256=reporter.digest([flat]),
                prefix_token_ids=prefixes, query_token_ids=query,
                exact_prompt_sha256=[reporter.digest(p+flat+query) for p in prefixes],
                complete_primary_context=True, augmentation="complete second real context" if variant == "long" else None,
                provenance=[{"source_id": "source-0", "token_ids_sha256": "source-hash"}],
                question="Original question?", answers=["yes"])
    manifest = dict(schema="redknot-nonprefix-frozen-v1", items=[item], max_new_tokens=32,
                    independent_primary_qa_count=1, sources={"qa": {"sha256": "dataset-hash"}})
    case = dict(case_id="case-0000", item_key=key, dataset="qa", context_tokens=4,
                chunks=2, concurrency=1, requests=2, pool_gib=1.,
                chunk_lengths=[2, 2], flat_document_sha256=reporter.digest(flat),
                chunk_tokens_sha256=reporter.digest([[1, 2], [3, 4]]),
                exact_prompt_sha256=item["exact_prompt_sha256"], prefix_tokens=[1, 2], query_tokens=1)
    plan = dict(schema="redknot-nonprefix-scale-plan-v1", workload_sha256=reporter.digest(manifest),
                cases=[case], output_tokens=32, natural_eos=True, global_prefix_layers=60,
                ffn="full native MLP", arms=list(reporter.ARMS), repeats=repeats)
    rows = []
    for repeat in range(repeats):
        for arm, seconds in zip(reporter.ARMS, [8., 6., 5., 4.]):
            records = [dict(request_id=f"request:{index}", frozen_prefix_index=index,
                            exact_prompt_sha256=case["exact_prompt_sha256"][index], max_new_tokens=32,
                            output_ids=[40, 2], text="Yes.\n完整原文", accuracy={"f1": 1., "em": 1.},
                            status="completed", finish_reason="eos", capacity_blocks=index,
                            ttft_seconds=1.+index, service_ttft_seconds=.5+index/2)
                       for index in range(2)]
            service = dict(status="complete", errors=[], max_active=1, completed_requests=2,
                requested_requests=2, failed_requests=0, cancelled_requests=0, completed_output_tokens=4,
                peak_active_requests=1, max_forward_batch=1, total_wall_seconds=seconds,
                stages_seconds={"admission": seconds/2, "decode_forward": seconds/2}, records=records)
            rows.append(dict(arm=arm, repeat=repeat, item_key=key, dataset="qa", status="complete",
                requested_concurrency=1, fatal=False, cleanup_status="complete", service=service,
                offline_document_seconds=123., warmup={"completed_requests": 99, "total_wall_seconds": 1000.}))
    raw = dict(schema="redknot-nonprefix-scale-v1", status="completed",
        config=dict(arms=list(reporter.ARMS), repeats=repeats),
        source_hashes={"benchmark.py": "original-source-hash"}, model_config_sha256="config-hash",
        workload_sha256=reporter.digest(manifest), plan_sha256=reporter.digest(plan["cases"]),
        cases=[dict(case=case, status="complete", results=rows)])
    save(directory/"results.json", raw)
    save(directory/"frozen_workload.json", manifest)
    save(directory/"frozen_scale_plan.json", plan)
    return directory, raw, manifest, plan


def comparison(case, baseline="recomputed_native", candidate="redknot_shared"):
    return next(c for c in case["paired_comparisons"] if c["baseline_arm"] == baseline and c["candidate_arm"] == candidate)


def test_full_verified_report_no_warmup_or_offline_in_qps(reporter, tmp_path):
    directory, raw, _, _ = fixture_job(reporter, tmp_path)
    before = {p.name:p.read_bytes() for p in directory.glob("*.json")}
    report = reporter.build_report(tmp_path)
    assert report["input_errors"] == []
    assert report["independent_primary_qa_count"] == 1
    job = report["jobs"][0]
    assert not job["integrity_errors"]
    assert job["workload_hash_verified"] and job["case_plan_hash_verified"]
    assert job["provenance"]["source_hashes"] == raw["source_hashes"]
    assert job["provenance"]["results"]["file_sha256"] == hashlib.sha256(before["results.json"]).hexdigest()
    case = job["cases"][0]
    shared = case["arms"]["redknot_shared"]
    assert shared["status"] == "complete"
    assert shared["measured_qps"] == .5
    assert shared["completed_requests"] == 2
    assert shared["observed_output_tokens"] == 4
    assert shared["actual_max_active"] == shared["actual_max_forward_batch"] == 1
    assert shared["capacity_block_events"] == 1
    assert shared["queued_ttft_seconds"]["p50"] == 1.5
    assert shared["queued_ttft_seconds"]["p95"] == pytest.approx(1.95)
    assert shared["primary_replica0"]["f1"] == shared["all_prefix"]["f1"] == 1
    assert comparison(case)["qps_speedup"] == 2
    assert comparison(case, "recomputed_paged")["qps_speedup"] == 1.5
    assert comparison(case, "redknot_private")["qps_speedup"] == 1.25
    assert comparison(case)["observed_2percent_gate"] is True
    assert shared["measurements"][0]["full_outputs"] == raw["cases"][0]["results"][-1]["service"]["records"]
    assert {p.name:p.read_bytes() for p in directory.glob("*.json")} == before
    json.dumps(report, allow_nan=False)


def test_gpu_local_case_id_collision_never_pools_qps_or_qa_variants(reporter, tmp_path):
    fixture_job(reporter, tmp_path, job="gpu0", variant="medium")
    fixture_job(reporter, tmp_path, job="gpu1", variant="long")
    report = reporter.build_report(tmp_path)
    assert report["independent_primary_qa_count"] == 1
    assert report["planned_cases"] == 2
    assert report["throughput_aggregation_across_cases_or_gpus"] is False
    cases = [job["cases"][0] for job in report["jobs"]]
    assert [case["case_key"] for case in cases] == ["gpu0/case-0000", "gpu1/case-0000"]
    assert [case["input_kind"] for case in cases] == ["full_original_context", "augmented_rag"]
    assert all(case["arms"]["redknot_shared"]["measured_qps"] == .5 for case in cases)


def test_preflight_native_is_not_measured_not_oom_and_paged_ratios_remain(reporter, tmp_path):
    directory, raw, _, _ = fixture_job(reporter, tmp_path)
    native = raw["cases"][0]["results"][0]
    native.update(status="preflight_blocked", service=None,
                  reason="conservative bound; not an observed OOM", capacity_preflight={"measurement": False, "allowed": False})
    raw["cases"][0]["status"] = "complete_with_blocked_or_failed_arms"
    raw["status"] = "completed_with_blocked_or_failed_arms"
    save(directory/"results.json", raw)
    case = reporter.build_report(tmp_path)["jobs"][0]["cases"][0]
    assert case["arms"]["recomputed_native"]["status"] == "not_measured_preflight_blocked"
    assert case["arms"]["recomputed_native"]["measured_qps"] is None
    assert comparison(case)["qps_speedup"] is None
    assert comparison(case)["observed_2percent_gate"] is False
    assert comparison(case, "recomputed_paged")["qps_speedup"] == 1.5


@pytest.mark.parametrize("problem", ["cleanup", "record_hash", "duplicate_record", "missing_repeat", "duplicate_repeat", "running_case"])
def test_incomplete_or_bad_pair_never_gets_ratio(reporter, tmp_path, problem):
    directory, raw, _, _ = fixture_job(reporter, tmp_path, repeats=2)
    shared = raw["cases"][0]["results"][-1]
    if problem == "cleanup":
        shared["cleanup_status"] = "failed"
    elif problem == "record_hash":
        shared["service"]["records"][0]["exact_prompt_sha256"] = "wrong"
    elif problem == "duplicate_record":
        shared["service"]["records"][1]["frozen_prefix_index"] = 0
    elif problem == "missing_repeat":
        raw["cases"][0]["results"].pop()
    elif problem == "duplicate_repeat":
        raw["cases"][0]["results"].append(copy.deepcopy(shared))
    else:
        raw["cases"][0]["status"] = "running"
    save(directory/"results.json", raw)
    case = reporter.build_report(tmp_path)["jobs"][0]["cases"][0]
    assert comparison(case)["qps_speedup"] is None
    assert comparison(case)["observed_2percent_gate"] is False
    assert case["arms"]["redknot_shared"]["measured_qps"] is not None  # observed diagnostic, not acceptance


@pytest.mark.parametrize("problem", ["manifest", "plan", "plan_metadata", "case", "missing_manifest"])
def test_digest_or_metadata_mismatch_preserves_outputs_but_suppresses_ratios(reporter, tmp_path, problem):
    directory, raw, manifest, plan = fixture_job(reporter, tmp_path)
    if problem == "manifest":
        manifest["items"][0]["question"] = "changed"
        save(directory/"frozen_workload.json", manifest)
    elif problem == "plan":
        plan["cases"][0]["pool_gib"] = 99
        save(directory/"frozen_scale_plan.json", plan)
    elif problem == "plan_metadata":
        plan["repeats"] = 2  # plan_sha256 intentionally only covers cases
        save(directory/"frozen_scale_plan.json", plan)
    elif problem == "case":
        raw["cases"][0]["case"]["concurrency"] = 2
        save(directory/"results.json", raw)
    else:
        (directory/"frozen_workload.json").unlink()
    job = reporter.build_report(tmp_path)["jobs"][0]
    case = job["cases"][0]
    assert job["integrity_errors"] or case["integrity_errors"]
    assert comparison(case)["qps_speedup"] is None
    assert case["arms"]["redknot_shared"]["measurements"][0]["full_outputs"][0]["text"] == "Yes.\n完整原文"


def test_missing_planned_case_still_shown(reporter, tmp_path):
    directory, raw, _, _ = fixture_job(reporter, tmp_path)
    raw["cases"] = []
    raw["status"] = "fatal_supervisor_restart_required"
    save(directory/"results.json", raw)
    case = reporter.build_report(tmp_path)["jobs"][0]["cases"][0]
    assert case["recorded_case_status"] == "missing_or_duplicate"
    assert all(arm["status"] == "not_measured_missing" for arm in case["arms"].values())
    assert comparison(case)["qps_speedup"] is None


def test_primary_and_allprefix_not_best_replica_and_relative_gate(reporter, tmp_path):
    directory, raw, _, _ = fixture_job(reporter, tmp_path)
    for row in raw["cases"][0]["results"]:
        for record in row["service"]["records"]:
            record["accuracy"] = {"f1": .5, "em": .5}
    shared = raw["cases"][0]["results"][-1]
    shared["service"]["records"][0]["accuracy"]["f1"] = .48
    shared["service"]["records"][1]["accuracy"]["f1"] = .5
    save(directory/"results.json", raw)
    case = reporter.build_report(tmp_path)["jobs"][0]["cases"][0]
    quality = case["arms"]["redknot_shared"]
    assert quality["primary_replica0"]["f1"] == .48
    assert quality["all_prefix"]["f1"] == .49
    delta = comparison(case)["primary_replica0_drops"]
    assert delta["f1_drop_percentage_points"] == pytest.approx(2.)
    assert delta["f1_relative_drop_percent"] == pytest.approx(4.)
    assert comparison(case)["observed_2percent_gate"] is False


def test_cli_json_default_console_and_no_overwrite(reporter, tmp_path, capsys):
    directory, _, _, _ = fixture_job(reporter, tmp_path)
    assert reporter.main(["--input-dir", str(tmp_path)]) == 0
    output = tmp_path/"report_nonprefix_scale.json"
    assert json.loads(output.read_text())["planned_cases"] == 1
    assert "NO_CROSS_GPU_QPS_SUM" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        reporter.main(["--input-dir", str(tmp_path)])
    before = (directory/"results.json").read_bytes()
    with pytest.raises(SystemExit):
        reporter.main(["--input-dir", str(tmp_path), "--output", str(directory/"results.json")])
    assert (directory/"results.json").read_bytes() == before


def test_bad_job_does_not_discard_good_job(reporter, tmp_path):
    fixture_job(reporter, tmp_path)
    bad = tmp_path/"gpu1"
    bad.mkdir()
    (bad/"results.json").write_text("not JSON")
    report = reporter.build_report(tmp_path)
    assert len(report["jobs"]) == 2
    assert report["jobs"][1]["integrity_errors"]
    assert comparison(report["jobs"][0]["cases"][0])["qps_speedup"] == 2
