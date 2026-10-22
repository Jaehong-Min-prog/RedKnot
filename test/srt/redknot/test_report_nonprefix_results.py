"""Evidence-only reporter tests; fixtures are not performance measurements."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def reporter():
    spec = importlib.util.spec_from_file_location("_nonprefix_report_test", _HERE / "report_nonprefix_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _cohort(arm, key, seconds, *, pool=False, copy_compacted=None):
    records = []
    for index in range(4):
        record = dict(request_id=f"request:{index}", status="completed", finish_reason="eos",
                      text="No.", output_ids=[3, 2], accuracy={"f1": 1., "em": 1.},
                      ttft_seconds=index+1., service_ttft_seconds=0.5+0.1*index,
                      admission_stats=None)
        if pool:
            record["admission_stats"] = dict(prefix_prefill_s=0.01, online_recovery_s=0.2,
                kv_publish_and_partition_s=0.05, pool_after_admit=dict(live_pages=40+index),
                private_local_copy_compacted=copy_compacted)
        records.append(record)
    return dict(arm=arm, item_key=key, dataset=key.split(":")[0], repeat=0, answers=["no"],
                offline_seconds=1.25 if pool else 0.001,
                peak_allocated_bytes=9000000, peak_reserved_bytes=10000000,
                service=dict(status="complete", records=records, completed_requests=4,
                    failed_requests=0, cancelled_requests=0, completed_output_tokens=8,
                    total_wall_seconds=seconds, completed_qps=4/seconds, errors=[],
                    stages_seconds=dict(admission=seconds*0.6, query_forward=seconds*0.1,
                                        decode_forward=seconds*0.2, release=seconds*0.1),
                    engine_stats_after=dict(pool=dict(high_water_pages=60, capacity_pages=1000,
                        reserved_slab_bytes=1024000, live_pages=20)) if pool else {}))


def _evidence(tmp_path, name="pilot", *, keys=("qa:0",), status="completed"):
    directory = tmp_path / name
    directory.mkdir()
    workload = dict(schema="redknot-nonprefix-frozen-v1", request_replicas=4,
                    sources={"qa": {"sha256": "fixture-source-digest"}},
                    items=[dict(key=key, dataset=key.split(":")[0], source_row=index,
                                context_tokens=100, question="Original question?", answers=["no"],
                                prefix_token_ids=[[1, index] for index in range(4)])
                           for index, key in enumerate(keys)])
    raw = dict(schema="redknot-nonprefix-e2e-v1", status=status,
               config={"repeats": 1, "concurrency": 2},
               workload_sha256=_canonical_hash(workload),
               model_config_sha256="fixture-config-digest", source_hashes={"runner.py": "fixture-source"},
               results=[row for key in keys for row in (
                   _cohort("recomputed_native", key, 8),
                   _cohort("redknot_shared", key, 4, pool=True))])
    path = directory / "results.json"
    path.write_text(json.dumps(raw))
    (directory / "frozen_workload.json").write_text(json.dumps(workload))
    return path, raw, workload


def test_report_preserves_raw_and_distinguishes_qa_requests_pages_slab(reporter, tmp_path):
    path, raw, workload = _evidence(tmp_path)
    before = path.read_bytes()
    report = reporter.build_report([path])
    assert not report["input_errors"]
    assert report["aggregate_across_runs"] is False
    run = report["runs"][0]
    assert run["workload"]["hash_verified"] is True
    assert run["workload"]["independent_qa_expected"] == run["workload"]["independent_qa_observed"] == 1
    shared = run["arms"]["redknot_shared"]
    assert shared["completed_requests"] == 4
    assert shared["quality"]["independent_qa_scored"] == 1
    assert shared["quality"]["observed_2percent_gate"] is True
    assert shared["measured_qps"] == 1
    assert shared["qps_ratio_vs_native"] == 2
    assert shared["complete_paired_comparison"] is True
    assert shared["pool_high_water_pages_max"] == 60
    assert shared["pool_high_water_bytes_max"] == 61440
    assert shared["pool_cohorts"][0]["fixed_slab_reserved_bytes"] == 1024000
    assert "warmup" in shared["pool_cohorts"][0]["scope"]
    assert shared["queued_ttft_seconds"]["p50"] == 2.5
    assert shared["queued_ttft_seconds"]["p95"] == pytest.approx(3.85)
    assert shared["request_service_ttft_seconds"]["p50"] == pytest.approx(0.65)
    assert shared["admission_breakdown"]["online_recovery_s"]["total"] == pytest.approx(0.8)
    native = run["arms"]["recomputed_native"]
    assert native["pool_high_water_pages_max"] is None
    assert native["admission_breakdown"]["online_recovery_s"] == {"total": None, "observed_count": 0, "missing_count": 4}
    assert path.read_bytes() == before
    assert run["input_sha256"] == hashlib.sha256(before).hexdigest()
    json.dumps(report, allow_nan=False)


def test_full_output_is_retained_without_answer_extraction(reporter, tmp_path, capsys):
    path, raw, _ = _evidence(tmp_path)
    text = "No.\n\n完整输出仍然保留：**do not extract** <think>visible text</think>"
    raw["results"][1]["service"]["records"][2]["text"] = text
    path.write_text(json.dumps(raw))
    report = reporter.build_report([path])
    cohort = report["runs"][0]["per_item_full_outputs"][0]["cohorts"][1]
    assert cohort["full_outputs"][2]["text"] == text
    reporter.print_summary(report, print_outputs=True)
    output = capsys.readouterr().out
    assert text in output
    assert "reference=recomputed_native" in output


@pytest.mark.parametrize("status", ["running", "failed"])
def test_partial_or_failed_run_never_has_final_acceptance(reporter, tmp_path, status):
    path, _, _ = _evidence(tmp_path, status=status)
    values = reporter.build_report([path])["runs"][0]["arms"]["redknot_shared"]
    assert values["measured_qps"] == 1
    assert values["qps_ratio_vs_native"] == 2  # paired diagnostic remains visible
    assert values["complete_paired_comparison"] is False
    assert values["quality"]["observed_2percent_gate"] is False


def test_partial_comparison_uses_common_cohorts_not_unpaired_aggregate(reporter, tmp_path):
    path, raw, _ = _evidence(tmp_path, keys=("qa:0", "qa:1"))
    raw["status"] = "failed"
    raw["results"] = raw["results"][:3]
    raw["results"][2]["service"]["total_wall_seconds"] = 80
    path.write_text(json.dumps(raw))
    values = reporter.build_report([path])["runs"][0]["arms"]["redknot_shared"]
    assert values["paired_cohorts"] == 1
    assert values["expected_cohorts"] == 2
    assert values["qps_ratio_vs_native"] == 2  # not 11x from unequal averages
    assert values["complete_paired_comparison"] is False
    assert values["quality"]["observed_2percent_gate"] is False


@pytest.mark.parametrize("problem", ["missing", "mismatch"])
def test_unknown_workload_coverage_suppresses_acceptance_and_speedup(reporter, tmp_path, problem):
    path, _, workload = _evidence(tmp_path)
    manifest = path.parent / "frozen_workload.json"
    if problem == "missing":
        manifest.unlink()
    else:
        workload["items"][0]["question"] = "Different manifest"
        manifest.write_text(json.dumps(workload))
    run = reporter.build_report([path])["runs"][0]
    assert run["workload"]["hash_verified"] is False
    assert run["arms"]["redknot_shared"]["qps_ratio_vs_native"] is None
    assert run["arms"]["redknot_shared"]["quality"]["observed_2percent_gate"] is False
    assert run["warnings"]


def test_private_uncompacted_baseline_is_labeled(reporter, tmp_path):
    path, raw, _ = _evidence(tmp_path)
    raw["results"].append(_cohort("redknot_private", "qa:0", 6, pool=True, copy_compacted=False))
    path.write_text(json.dumps(raw))
    run = reporter.build_report([path])["runs"][0]
    assert "not visible-window compacted" in run["arms"]["redknot_private"]["private_copy_policy"]
    assert any("not an optimal private-cache baseline" in warning for warning in run["warnings"])


def test_pilots_are_not_pooled_and_bad_inputs_remain_visible(reporter, tmp_path):
    a, _, _ = _evidence(tmp_path, name="v4")
    b, _, _ = _evidence(tmp_path, name="v5")
    broken = tmp_path / "broken.json"
    broken.write_text("not JSON")
    report = reporter.build_report([a, broken, b])
    assert len(report["runs"]) == 2
    assert len(report["input_errors"]) == 1
    assert report["aggregate_across_runs"] is False
    assert all(run["workload"]["independent_qa_expected"] == 1 for run in report["runs"])


def test_cli_creates_json_only_and_never_overwrites_evidence(reporter, tmp_path, capsys):
    source, _, _ = _evidence(tmp_path)
    output = tmp_path / "report.json"
    before = source.read_bytes()
    assert reporter.main(["--inputs", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["schema"] == "redknot-nonprefix-evidence-report-v1"
    assert "REPORT_JSON" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        reporter.main(["--inputs", str(source), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        reporter.main(["--inputs", str(source), "--output", str(source)])
    assert error.value.code == 2
    assert source.read_bytes() == before


def test_missing_service_ttft_is_unknown_not_zero(reporter, tmp_path):
    path, raw, _ = _evidence(tmp_path)
    for record in raw["results"][1]["service"]["records"]:
        del record["service_ttft_seconds"]
    path.write_text(json.dumps(raw))
    metric = reporter.build_report([path])["runs"][0]["arms"]["redknot_shared"]["request_service_ttft_seconds"]
    assert metric["p50"] is metric["p95"] is None
    assert metric["count"] == 0
    assert metric["missing_count"] == 4


@pytest.mark.parametrize("manifest_value", [True, False])
def test_manifest_protocol_overrides_cli_and_raw_diagnostics_are_preserved(reporter, tmp_path, manifest_value):
    path, raw, workload = _evidence(tmp_path)
    workload.update(instruction_at_query=manifest_value, varying_prefix_lengths=manifest_value)
    raw["config"].update(instruction_at_query=not manifest_value, vary_prefix_lengths=not manifest_value)
    raw["effective_workload_protocol"] = dict(instruction_at_query=manifest_value,
        varying_prefix_lengths=manifest_value, prefix_lengths_by_item={"qa:0": [29, 37, 45, 53]})
    raw["profile_diagnostics"] = dict(effective_local_head_ratio=15/512,
        effective_counts={"local": 15, "global": 17, "dense": 480, "total": 512},
        declared_summary_matches_effective=False)
    raw["workload_sha256"] = _canonical_hash(workload)
    path.write_text(json.dumps(raw))
    manifest_path = path.parent / "frozen_workload.json"
    manifest_path.write_text(json.dumps(workload))
    before = path.read_bytes(), manifest_path.read_bytes()
    run = reporter.build_report([path])["runs"][0]
    assert run["workload"]["hash_verified"] is True
    assert run["workload"]["instruction_at_query"] is manifest_value
    assert run["workload"]["varying_prefix_lengths"] is manifest_value
    assert "takes precedence" in run["workload"]["protocol_source"]
    assert run["config"] == raw["config"]  # preserve invocation, do not rewrite provenance
    assert run["effective_workload_protocol"] == raw["effective_workload_protocol"]
    assert run["profile_diagnostics"] == raw["profile_diagnostics"]
    assert sum("overrides CLI config" in warning for warning in run["warnings"]) == 2
    assert (path.read_bytes(), manifest_path.read_bytes()) == before


@pytest.mark.parametrize("manifest_present", [True, False])
def test_missing_manifest_protocol_is_not_inferred(reporter, tmp_path, manifest_present):
    path, raw, _ = _evidence(tmp_path)
    raw["config"].update(instruction_at_query=True, vary_prefix_lengths=True)
    raw["effective_workload_protocol"] = dict(instruction_at_query=True, varying_prefix_lengths=True)
    path.write_text(json.dumps(raw))
    if not manifest_present:
        (path.parent / "frozen_workload.json").unlink()
    run = reporter.build_report([path])["runs"][0]
    assert run["workload"]["instruction_at_query"] is None
    assert run["workload"]["varying_prefix_lengths"] is None
    assert run["profile_diagnostics"] is None
    assert run["effective_workload_protocol"] == raw["effective_workload_protocol"]
    assert not any("overrides CLI config" in warning for warning in run["warnings"])
