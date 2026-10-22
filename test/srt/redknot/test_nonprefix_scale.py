"""No-GPU provenance, capacity planning and fatal-cleanup gates for scaling."""
import copy
from types import SimpleNamespace

import pytest
import torch

import benchmark_nonprefix_scale as runner


def workload(prefixes=32):
    flat = list(range(50, 85))
    item = dict(key="qa:0", dataset="qa", answers=["answer"], context_tokens=len(flat),
        document_token_ids=[flat[:11], flat[11:]], prefix_token_ids=[[i+1, 41] for i in range(prefixes)],
        query_token_ids=[9, 10])
    item["document_tokens_sha256"] = runner.digest(item["document_token_ids"])
    item["exact_prompt_sha256"] = [runner.digest(p+flat+item["query_token_ids"]) for p in item["prefix_token_ids"]]
    return dict(schema="redknot-nonprefix-frozen-v1", max_new_tokens=32, items=[item], request_replicas=prefixes)


@pytest.mark.parametrize("chunks", [4, 8, 16])
def test_exact_repartition_preserves_all_tokens_and_original_prompts(chunks):
    source = workload()
    before = copy.deepcopy(source)
    case = runner.prepare_cases(source, chunks=[chunks], concurrency=[4], requests=32)[0]
    original = sum(source["items"][0]["document_token_ids"], [])
    split = runner.split_exact(original, chunks)
    assert sum(split, []) == original
    assert max(map(len, split))-min(map(len, split)) <= 1
    assert len(split) == chunks
    assert case["exact_prompt_sha256"] == source["items"][0]["exact_prompt_sha256"]
    assert source == before


@pytest.mark.parametrize("chunks", [0, -1, True, 1.0, 36])
def test_invalid_chunk_geometry_fails_before_model(chunks):
    with pytest.raises(ValueError):
        runner.split_exact(list(range(35)), chunks)


def test_full_matrix_is_frozen_and_uses_same_request_prefix_order():
    source = workload(40)
    cases = runner.prepare_cases(source)
    assert len(cases) == 9
    assert {(case["chunks"], case["concurrency"]) for case in cases} == {
        (k, c) for k in (4, 8, 16) for c in (4, 8, 16)}
    assert all(case["requests"] == 32 for case in cases)
    assert all(case["exact_prompt_sha256"] == source["items"][0]["exact_prompt_sha256"][:32] for case in cases)


@pytest.mark.parametrize("fault", ["insufficient_prefixes", "duplicate_prefix", "hash", "protocol",
    "request_burst", "unknown_case_field", "unknown_item", "duplicate_case", "nan_pool"])
def test_invalid_scan_input_is_rejected_without_silent_expansion(fault):
    source = workload()
    cases = [dict(item_key="qa:0", chunks=4, concurrency=16, requests=32, pool_gib=48)]
    item = source["items"][0]
    if fault == "insufficient_prefixes":
        item["prefix_token_ids"].pop()
    elif fault == "duplicate_prefix":
        item["prefix_token_ids"][-1] = item["prefix_token_ids"][0]
    elif fault == "hash":
        item["exact_prompt_sha256"][-1] = "bad"
    elif fault == "protocol":
        source["max_new_tokens"] = 64
    elif fault == "request_burst":
        cases[0]["requests"] = 31
    elif fault == "unknown_case_field":
        cases[0]["global_prefix_layers"] = 0
    elif fault == "unknown_item":
        cases[0]["item_key"] = "not-frozen"
    elif fault == "duplicate_case":
        cases.append(dict(cases[0]))
    elif fault == "nan_pool":
        cases[0]["pool_gib"] = float("nan")
    with pytest.raises(ValueError):
        runner.prepare_cases(source, cases=cases)


def test_every_case_has_twice_the_largest_scan_concurrency():
    with pytest.raises(ValueError, match="maximum concurrency"):
        runner.prepare_cases(workload(), cases=[
            dict(item_key="qa:0", chunks=4, concurrency=4, requests=8),
            dict(item_key="qa:0", chunks=8, concurrency=16, requests=32)])


def test_native_guard_uses_old_plus_packed_plus_one_largest_reserved_request():
    source = workload()
    case = runner.prepare_cases(source, chunks=[4], concurrency=[8])[0]
    config = dict(num_hidden_layers=64, num_key_value_heads=8, head_dim=128)
    tokens = max(case["prefix_tokens"])+35+2+31
    per_request = tokens*64*8*128*2*2
    required = (2*8+1)*per_request+8*1024**3
    plan = runner.capacity_plan(case, source["items"][0], config, required, "recomputed_native")
    assert plan["estimated_required_bytes"] == required
    assert plan["largest_reserved_request_kv_bytes"] == per_request
    assert plan["allowed"] and not plan["measurement"] and not plan["guarantees_no_oom"]
    assert not runner.capacity_plan(case, source["items"][0], config, required-1, "recomputed_native")["allowed"]
    paged = runner.capacity_plan(case, source["items"][0], config, 56*1024**3, "recomputed_paged")
    assert paged["estimated_required_bytes"] == 56*1024**3 and paged["allowed"]


@pytest.mark.parametrize("error", [{"type": "OutOfMemoryError"}, {"message": "CUDA out of memory"},
    [{"type": "RuntimeError", "message": "CUDA error: illegal memory access"}], {"message": "CUBLAS failed"}])
def test_scheduler_json_cuda_errors_are_fatal(error):
    assert runner.cuda_failure(error)
    assert not runner.cuda_failure({"type": "ValueError", "message": "bad input IDs"})


class Config:
    num_hidden_layers, num_key_value_heads, head_dim = 64, 8, 128
    _attn_implementation = "flash_attention_2"
    def to_dict(self):
        return dict(num_hidden_layers=64, num_key_value_heads=8, head_dim=128)


def fake_runtime():
    attention = SimpleNamespace(forward=lambda: None)
    model = SimpleNamespace(config=Config(), model=SimpleNamespace(
        layers=[SimpleNamespace(self_attn=attention)],
        rotary_emb=SimpleNamespace(inv_freq=torch.ones(4), attention_scaling=1.0)))
    cuda = SimpleNamespace(empty_cache=lambda: None, synchronize=lambda device: None,
        mem_get_info=lambda device: (10**12, 10**12), reset_peak_memory_stats=lambda device: None,
        memory_allocated=lambda device: 1, max_memory_allocated=lambda device: 2,
        max_memory_reserved=lambda device: 3)
    return model, SimpleNamespace(cuda=cuda, equal=torch.equal)


def service(requests, *, status="complete", errors=None):
    records = [dict(request_id=request.request_id, status="completed", capacity_blocks=0,
                    output_ids=[7], finish_reason="eos") for request in requests]
    return dict(status=status, records=records, errors=errors or [], peak_active_requests=2,
        max_forward_batch=2, completed_requests=len(records), failed_requests=0, cancelled_requests=0,
        completed_output_tokens=len(records), total_wall_seconds=2, completed_qps=len(records)/2)


def exercise(monkeypatch, behavior="success", *, warmup=False, cleanup_fail=False, blocked=False):
    model, fake_torch = fake_runtime()
    created, persisted, calls = [], [], []
    class Engine:
        def __init__(self, model, *args, **kwargs):
            self.requests = {}
            self.closed = False
            created.append(self)
        def synchronize(self):
            pass
        def register_document(self, identity, ids):
            calls.append((identity, list(ids)))
        def close(self):
            self.closed = True
            if cleanup_fail:
                raise RuntimeError("cleanup event failure")
    def queued(engine, requests, **kwargs):
        if behavior == "raised_oom":
            raise RuntimeError("CUDA out of memory")
        if behavior == "raised_cpu":
            raise ValueError("invalid data")
        if behavior == "captured_oom":
            return service(requests, status="failed", errors=[dict(type="OutOfMemoryError", message="allocation")])
        if behavior == "warmup_failed":
            return service(requests, status="failed", errors=[dict(type="ValueError", message="test")])
        if behavior == "mutated_binding":
            model.model.layers[0].self_attn.forward = lambda: "wrong"
        return service(requests)
    if blocked:
        fake_torch.cuda.mem_get_info = lambda device: (1, 10**12)
    source = workload()
    case = runner.prepare_cases(source, chunks=[4], concurrency=[4])[0]
    row, fatal = runner.run_arm(model, SimpleNamespace(decode=lambda ids, **kw: "answer"),
        SimpleNamespace(), case, source["items"][0], "recomputed_native", 0, torch=fake_torch,
        engine_types=(Engine, Engine), queued_request=lambda *a: SimpleNamespace(request_id=a[0]),
        run_queued=queued, eos=[7], revision="test", warmup=warmup,
        persist=lambda value: persisted.append(copy.deepcopy(value)))
    return row, fatal, created, persisted, calls


def test_success_records_full_outputs_actual_concurrency_and_verified_cleanup(monkeypatch):
    row, fatal, engines, persisted, docs = exercise(monkeypatch)
    assert not fatal and row["status"] == "complete" and row["cleanup_status"] == "complete"
    assert row["requested_concurrency"] == 4 and row["actual_max_active"] == row["actual_max_forward_batch"] == 2
    assert len(row["service"]["records"]) == 32
    assert all(record["text"] == "answer" for record in row["service"]["records"])
    assert engines[0].closed and len(persisted) == 2 and len(docs) == 4


@pytest.mark.parametrize("behavior", ["raised_oom", "captured_oom", "mutated_binding"])
def test_gpu_error_or_unrestored_binding_is_saved_and_fatal_after_cleanup(monkeypatch, behavior):
    row, fatal, engines, persisted, _ = exercise(monkeypatch, behavior)
    assert fatal and row["fatal"]
    assert engines[0].closed and persisted[0].get("cleanup_status") is None
    assert persisted[-1]["fatal"]


def test_independent_cpu_failure_can_continue_only_after_successful_cleanup(monkeypatch):
    row, fatal, engines, persisted, _ = exercise(monkeypatch, "raised_cpu")
    assert row["status"] == "failed" and not fatal and engines[0].closed
    assert row["cleanup_status"] == "complete"


def test_early_warmup_failure_cannot_hide_a_later_cleanup_failure(monkeypatch):
    row, fatal, engines, _, _ = exercise(monkeypatch, "warmup_failed", warmup=True, cleanup_fail=True)
    assert row["status"] == "warmup_failed" and row["cleanup_status"] == "failed"
    assert fatal and engines[0].closed


def test_preflight_block_is_not_oom_or_measured_capacity_and_does_not_construct_engine(monkeypatch):
    row, fatal, engines, persisted, docs = exercise(monkeypatch, blocked=True)
    assert row["status"] == "preflight_blocked" and not fatal
    assert not engines and not docs and row["service"] is None
    assert not row["capacity_preflight"]["measurement"]
    assert row["cleanup_status"] == "complete" and persisted[-1]["fatal"] is False


def test_atomic_save_preserves_earlier_file_if_replace_fails(tmp_path, monkeypatch):
    path = tmp_path/"case.json"
    runner.save(path, {"status": "first"})
    original = path.read_bytes()
    def broken(*args, **kwargs):
        raise OSError("injected atomic replace failure")
    monkeypatch.setattr(type(path), "replace", broken)
    with pytest.raises(OSError):
        runner.save(path, {"status": "second"})
    assert path.read_bytes() == original


def test_arm_rotation_does_not_depend_on_local_gpu_case_number():
    case = runner.prepare_cases(workload())[0]
    other_job = dict(case, case_id="case-0009")
    for repeat in (0, 1, 2):
        assert runner.planned_arm_order(case, repeat, runner.ARMS) == runner.planned_arm_order(other_job, repeat, runner.ARMS)
        assert set(runner.planned_arm_order(case, repeat, runner.ARMS)) == set(runner.ARMS)
    all_orders = {tuple(runner.planned_arm_order(case, repeat, runner.ARMS))
                  for case in runner.prepare_cases(workload()) for repeat in (0, 1)}
    assert len(all_orders) > 1


def test_same_backend_paired_speedup_is_named_when_native_is_absent():
    source = workload()
    case = runner.prepare_cases(source, chunks=[4], concurrency=[4])[0]
    rows = []
    for arm, seconds in (("recomputed_paged", 4), ("redknot_shared", 2)):
        records = [dict(status="completed", accuracy=dict(f1=1., em=1.), output_ids=[7]) for _ in range(32)]
        rows.append(dict(arm=arm, repeat=0, item_key="qa:0", dataset="qa", status="complete",
            fatal=False, cleanup_status="complete", service=dict(status="complete", errors=[], records=records,
                total_wall_seconds=seconds, completed_requests=32, failed_requests=0, cancelled_requests=0,
                completed_output_tokens=32)))
    summary = runner.case_summary(rows, source["items"][0], case, 1)
    same_backend = next(value for value in summary["paired_qps_comparisons"]
        if value["baseline_arm"] == "recomputed_paged" and value["candidate_arm"] == "redknot_shared")
    assert same_backend["qps_speedup"] == 2 and same_backend["all_expected_repeats_complete"]
    native = next(value for value in summary["paired_qps_comparisons"]
        if value["baseline_arm"] == "recomputed_native" and value["candidate_arm"] == "redknot_shared")
    assert native["qps_speedup"] is None
    rows[-1]["fatal"] = True
    after_failure = runner.case_summary(rows, source["items"][0], case, 1)
    same_backend = next(value for value in after_failure["paired_qps_comparisons"]
        if value["baseline_arm"] == "recomputed_paged" and value["candidate_arm"] == "redknot_shared")
    assert same_backend["qps_speedup"] is None and not same_backend["all_expected_repeats_complete"]
