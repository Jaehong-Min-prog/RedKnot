"""Frozen mixed-group protocol/paired reporting tests; not GPU speed evidence."""
import copy
import json
import sys
from types import SimpleNamespace

import pytest

import benchmark_nonprefix_affinity as bench


def workload():
    items = []
    for group in range(2):
        ids = list(range(40+group*40, 80+group*40))
        item = dict(key=f"qa:{group}", dataset="qa", answers=[str(group)],
            document_token_ids=[ids], context_tokens=len(ids),
            prefix_token_ids=[[i+1, 7] for i in range(16)], query_token_ids=[3,group+4])
        item["document_tokens_sha256"] = bench.digest([ids])
        item["exact_prompt_sha256"] = [bench.digest(p+ids+item["query_token_ids"])
                                       for p in item["prefix_token_ids"]]
        items.append(item)
    return dict(items=items, max_new_tokens=32)


def test_mixed_plan_is_interleaved_identical_tokens_and_counterbalanced():
    source = workload()
    old = copy.deepcopy(source)
    plan = bench.prepare_plan(source, ["qa:0", "qa:1"])
    assert source == old
    assert len(plan["requests"]) == 32
    assert [r["item_key"] for r in plan["requests"]] == ["qa:0", "qa:1"]*16
    assert plan["arm_order_by_repeat"] == [list(bench.ARMS), list(reversed(bench.ARMS))]
    for spec in plan["requests"]:
        assert spec["exact_prompt_sha256"] == source["items"][spec["group_index"]]["exact_prompt_sha256"][spec["prefix_index"]]
        assert spec["arrival_seconds"] == 0


@pytest.mark.parametrize("kind", ["onegroup", "duplicate", "prompt", "document", "missingprefix",
    "shortburst", "unknownarm", "missedcontrol", "nanpool", "badchunks", "badwait"])
def test_invalid_protocol_fails_before_model(kind):
    source, keys, kwargs = workload(), ["qa:0", "qa:1"], {}
    if kind == "onegroup": keys = ["qa:0"]
    if kind == "duplicate": keys = ["qa:0", "qa:0"]
    if kind == "prompt": source["items"][0]["exact_prompt_sha256"][0] = "bad"
    if kind == "document": source["items"][0]["document_tokens_sha256"] = "bad"
    if kind == "missingprefix": source["items"][0]["prefix_token_ids"].pop()
    if kind == "shortburst": kwargs["concurrency"] = 17
    if kind == "unknownarm": kwargs["arms"] = ["unknown", *bench.ARMS]
    if kind == "missedcontrol": kwargs["arms"] = ["redknot_shared_affinity"]
    if kind == "nanpool": kwargs["pool_gib"] = float("nan")
    if kind == "badchunks": kwargs["chunks"] = True
    if kind == "badwait": kwargs["max_wait_seconds"] = -1
    with pytest.raises(ValueError):
        bench.prepare_plan(source, keys, **kwargs)


def row(plan, arm, repeat, seconds=8):
    return dict(arm=arm, repeat=repeat, status="complete", cleanup_status="complete", fatal=False,
        service=dict(status="complete", errors=[], total_wall_seconds=seconds,
            max_active=plan["concurrency"], ignore_eos=False, failed_requests=0, cancelled_requests=0,
            completed_requests=len(plan["requests"]),
            completed_output_tokens=2*len(plan["requests"]),
            peak_active_requests=plan["concurrency"], max_forward_batch=plan["concurrency"],
            records=[dict(request_id=s["request_id"], status="completed", output_ids=[1,2],
                item_key=s["item_key"], prefix_index=s["prefix_index"], arrival_seconds=s["arrival_seconds"],
                max_new_tokens=plan["output_tokens"],
                exact_prompt_sha256=s["exact_prompt_sha256"], accuracy=dict(f1=1.,em=1.))
                for s in plan["requests"]]))


def test_summary_requires_all_repeats_and_exact_inputs_for_headline():
    plan = bench.prepare_plan(workload(), ["qa:0", "qa:1"])
    rows = [row(plan, arm, i, seconds) for i in range(2) for arm,seconds in (
        ("redknot_shared_fcfs",8), ("redknot_shared_affinity",4))]
    result = bench.compare_rows(rows, plan)[0]
    assert result["qps_ratio"] == 2 and result["exact_output_fraction"] == 1
    assert result["quality_certified"] is False
    assert result["mean_drop_pp"] == dict(f1=0,em=0)
    assert bench.compare_rows(rows[:-1], plan)[0]["qps_ratio"] is None
    rows[-1]["service"]["records"][0]["exact_prompt_sha256"] = "changed"
    assert bench.compare_rows(rows, plan)[0]["qps_ratio"] is None


@pytest.mark.parametrize("fault", ["duplicate", "failed", "cleanup", "fatal", "nan", "missing"])
def test_failed_or_duplicate_arm_never_produces_headline_speedup(fault):
    plan = bench.prepare_plan(workload(), ["qa:0", "qa:1"], repeats=1)
    rows = [row(plan, arm, 0) for arm in ("redknot_shared_fcfs", "redknot_shared_affinity")]
    if fault == "duplicate": rows.append(copy.deepcopy(rows[-1]))
    if fault == "failed": rows[-1]["status"] = "failed"
    if fault == "cleanup": rows[-1]["cleanup_status"] = "failed"
    if fault == "fatal": rows[-1]["fatal"] = True
    if fault == "nan": rows[-1]["service"]["total_wall_seconds"] = float("nan")
    if fault == "missing": rows[-1]["service"]["records"].pop()
    assert bench.compare_rows(rows, plan)[0]["qps_ratio"] is None


def test_quality_deltas_and_full_outputs_not_discarded_for_replica_groups():
    plan = bench.prepare_plan(workload(), ["qa:0", "qa:1"], repeats=1)
    rows = [row(plan, arm, 0) for arm in ("redknot_shared_fcfs", "redknot_shared_affinity")]
    rows[-1]["service"]["records"][0].update(output_ids=[9], accuracy=dict(f1=0.,em=0.))
    rows[-1]["service"]["completed_output_tokens"] -= 1
    result = bench.compare_rows(rows, plan)[0]
    assert result["exact_output_fraction"] == 31/32
    assert result["mean_drop_pp"] == dict(f1=100/32,em=100/32)
    assert result["per_group_drop_pp"]["qa:0"] == dict(f1=100/16,em=100/16)


@pytest.mark.parametrize("fault", ["declared_failed", "cancelled", "arrival", "prefix", "emptyids",
    "token_counter", "persistence", "cleanup_error", "boolean_wall", "extra_repeat", "output_budget"])
def test_strict_complete_gate_rejects_additional_protocol_faults(fault):
    plan = bench.prepare_plan(workload(), ["qa:0", "qa:1"], repeats=1)
    rows = [row(plan, arm, 0) for arm in ("redknot_shared_fcfs", "redknot_shared_affinity")]
    current, service = rows[-1], rows[-1]["service"]
    record = service["records"][0]
    if fault == "declared_failed": service["failed_requests"] = 1
    if fault == "cancelled": service["cancelled_requests"] = 1
    if fault == "arrival": record["arrival_seconds"] = 1
    if fault == "prefix": record["prefix_index"] = 99
    if fault == "emptyids": record["output_ids"] = []
    if fault == "token_counter": service["completed_output_tokens"] += 1
    if fault == "persistence": current["persistence_error"] = {"message":"failed write"}
    if fault == "cleanup_error": current["cleanup_error"] = {"message":"failed close"}
    if fault == "boolean_wall": service["total_wall_seconds"] = True
    if fault == "output_budget": record["max_new_tokens"] = 8
    if fault == "extra_repeat":
        extra = copy.deepcopy(current); extra["repeat"] = 1; rows.append(extra)
    result = bench.compare_rows(rows, plan)[0]
    assert result["qps_ratio"] is None
    assert result["observed_2percent_gate"] is False


def test_primary_and_allprefix_relative_drops_are_not_certification():
    plan = bench.prepare_plan(workload(), ["qa:0", "qa:1"], repeats=1)
    rows = [row(plan, arm, 0) for arm in ("redknot_shared_fcfs", "redknot_shared_affinity")]
    for values in rows:
        for record in values["service"]["records"]:
            record["accuracy"] = dict(f1=.5, em=.5)
    rows[-1]["service"]["records"][0]["accuracy"]["f1"] = .48
    result = bench.compare_rows(rows, plan)[0]
    group = result["per_group_quality"]["qa:0"]["primary_replica0"]
    assert group["f1_drop_pp"] == pytest.approx(2)
    assert group["f1_relative_drop_percent"] == pytest.approx(4)
    assert result["observed_2percent_gate"] is False
    assert result["quality_certified"] is False


def test_missing_scores_do_not_crash_or_certify_and_qa_variants_do_not_inflate_n():
    source = workload()
    for item in source["items"]: item["primary_qa_key"] = "same-primary"
    plan = bench.prepare_plan(source, ["qa:0", "qa:1"], repeats=1)
    assert plan["independent_primary_qa_count"] == 1
    rows = [row(plan, arm, 0) for arm in ("redknot_shared_fcfs", "redknot_shared_affinity")]
    rows[-1]["service"]["records"][0]["accuracy"] = None
    result = bench.compare_rows(rows, plan)[0]
    assert result["qps_ratio"] == 1  # complete performance data, unknown quality
    assert result["observed_2percent_gate"] is False
    assert result["quality_certified"] is False


def test_fairness_summary_has_queue_tail_latency_groups_and_actual_order():
    plan = bench.prepare_plan(workload(), ["qa:0", "qa:1"], repeats=1)
    current = row(plan, "redknot_shared_affinity", 0)
    records = current["service"]["records"]
    for index, record in enumerate(records):
        record.update(queue_wait_seconds=float(index), ttft_seconds=index+1.,
                      service_ttft_seconds=.5, latency_seconds=index+2.,
                      admission_started_seconds=float(index), capacity_blocks=0)
    records[0]["admission_started_seconds"] = 1.
    records[1]["admission_started_seconds"] = 0.
    result = bench.summarize_rows([current], plan)["redknot_shared_affinity"]
    assert result["headline_qps"] == 4
    assert result["queue_wait_seconds"]["p50"] == 15.5
    assert result["queue_wait_seconds"]["p95"] == pytest.approx(29.45)
    assert result["latency_seconds"]["p95"] == pytest.approx(31.45)
    assert result["per_group"]["qa:0"]["queue_wait_seconds"]["mean"] == 15
    assert result["per_group"]["qa:1"]["queue_wait_seconds"]["mean"] == 16
    assert result["group_mean_queue_wait_spread_seconds"] == 1
    assert result["admission_order"][0]["request_ids"][:2] == [records[1]["request_id"], records[0]["request_id"]]
    assert result["max_observed_later_request_overtakes"] == 1
    assert result["quality_certified"] is False
    current["cleanup_status"] = "failed"
    assert bench.summarize_rows([current], plan)["redknot_shared_affinity"]["headline_qps"] is None


def test_run_arm_preserves_service_when_text_decode_fails_and_cleans_up(monkeypatch):
    source = workload()
    plan = bench.prepare_plan(source, ["qa:0", "qa:1"], repeats=1)
    service = row(plan, "redknot_shared_fcfs", 0)["service"]
    snapshots, engines = [], []
    monkeypatch.setattr(bench, "_model_guard", lambda model: "guard")
    monkeypatch.setattr(bench, "_guard_unchanged", lambda *args: None)
    class Engine:
        def __init__(self, *args, **kwargs):
            engines.append(self); self.closed = False; self.requests = {}
            self.pool = SimpleNamespace(capacity_pages=10, poisoned=False,
                stats=lambda:dict(free_pages=10, pinned_pages=0))
        def register_document(self, *args): pass
        def synchronize(self): pass
        def close(self): self.closed = True
    class Tokenizer:
        def decode(self, *args, **kwargs): raise ValueError("decode failed")
    cuda = SimpleNamespace(empty_cache=lambda:None, synchronize=lambda *args:None,
        mem_get_info=lambda _: (1000*1024**3, 1000*1024**3), reset_peak_memory_stats=lambda _:None)
    request = lambda identity, *args, **kwargs: SimpleNamespace(request_id=identity)
    queued = lambda engine, requests, **kwargs: dict(status="complete", errors=[]) if requests[0].request_id.startswith("warmup") else copy.deepcopy(service)
    result, fatal = bench.run_arm(SimpleNamespace(config=SimpleNamespace(head_dim=128)), Tokenizer(), object(),
        source, plan, "redknot_shared_fcfs", 0, torch=SimpleNamespace(cuda=cuda), Engine=Engine,
        Request=request, run_queued=queued, AffinityConfig=None, revision="config", eos=[2],
        persist=lambda value:snapshots.append(copy.deepcopy(value)))
    assert result["status"] == "failed" and not fatal
    assert result["service"]["records"][0]["output_ids"] == [1,2]
    assert result["cleanup_status"] == "complete" and engines[0].closed
    assert result["service"]["records"][0]["frozen_input_reference"]["workload_sha256"] == plan["workload_sha256"]
    assert snapshots[-1]["cleanup_status"] == "complete"


def test_initial_guard_failure_is_recorded_and_fatal(monkeypatch):
    snapshots = []
    def fail(model): raise ValueError("guard failed")
    monkeypatch.setattr(bench, "_model_guard", fail)
    cuda = SimpleNamespace(empty_cache=lambda:None, synchronize=lambda *args:None)
    result, fatal = bench.run_arm(None, None, None, {}, {}, "redknot_shared_fcfs", 0,
        torch=SimpleNamespace(cuda=cuda), Engine=None, Request=None, run_queued=None,
        AffinityConfig=None, revision="", eos=[2], persist=lambda value:snapshots.append(copy.deepcopy(value)))
    assert fatal and result["status"] == "failed"
    assert result["cleanup_status"] == "failed"
    assert result["error"]["message"] == "guard failed"
    assert snapshots[-1]["fatal"] is True


def test_cli_affinity_knobs_are_frozen_before_execution(monkeypatch, tmp_path):
    destination = tmp_path/"prepared"
    fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs:object()))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(bench, "freeze_workload", lambda *args:workload())
    monkeypatch.setattr(sys, "argv", ["benchmark_nonprefix_affinity.py", "--model", str(tmp_path),
        "--manifest", str(tmp_path/"source.json"), "--items", "qa:0", "qa:1",
        "--output", str(destination), "--prepare-only", "--lookahead", "16", "--max-bypass", "2",
        "--max-wait-seconds", "25", "--recent-requests", "3"])
    assert bench.main() == 0
    plan = json.loads((destination/"frozen_affinity_plan.json").read_text())
    assert plan["affinity_config"] == dict(lookahead=16, max_bypass=2, max_wait_seconds=25., recent_requests=3)
