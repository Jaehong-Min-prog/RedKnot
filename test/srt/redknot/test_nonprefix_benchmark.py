"""No-model tests for frozen prompt provenance and conservative sample gates."""
import copy
import importlib.util
import json
from pathlib import Path
import types

import pytest


_HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location(
        "_redknot_nonprefix_benchmark_test_target", _HERE / "benchmark_redknot_nonprefix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _items(*names, replicas=1):
    return [dict(key=name, dataset=name.split(":")[0], prefix_token_ids=[[1]] * replicas)
            for name in names]


def _row(arm, key, *, f1=1.0, em=1.0, repeat=0, seconds=1.0,
         statuses=("completed",), service_status="complete", errors=None):
    records = [dict(request_id=f"request:{index}", status=status,
                    accuracy=dict(f1=f1, em=em) if status == "completed" else None,
                    output_ids=[3] if status == "completed" else [])
               for index, status in enumerate(statuses)]
    return dict(arm=arm, item_key=key, dataset=key.split(":")[0], repeat=repeat,
                service=dict(records=records, status=service_status, errors=errors or [],
                             completed_requests=sum(s == "completed" for s in statuses),
                             failed_requests=sum(s == "failed" for s in statuses),
                             cancelled_requests=sum(s == "cancelled" for s in statuses),
                             completed_output_tokens=sum(s == "completed" for s in statuses),
                             total_wall_seconds=seconds))


def _summary(benchmark, baseline, candidate, *, items=None, repeats=1):
    if items is None:
        items = _items("qa:0")
    return benchmark.summarize(baseline + candidate, items, expected_repeats=repeats)


def test_root_and_import_do_not_load_model(benchmark):
    assert benchmark.ROOT == _HERE.parents[2]
    assert (benchmark.ROOT / "python/sglang/srt/layers/attention/redknot/nonprefix_paged.py").is_file()


def test_f1_and_em_are_ground_truth_metrics_not_output_agreement(benchmark):
    assert benchmark.scores("The Eiffel Tower!", ["Eiffel Tower", "Paris landmark"]) == {"f1": 1.0, "em": 1.0}
    partial = benchmark.scores("red green", ["red blue"])
    assert partial == {"f1": 0.5, "em": 0.0}
    assert benchmark.scores("", ["answer"]) == {"f1": 0.0, "em": 0.0}
    assert benchmark.scores("answer", [""]) == {"f1": 0.0, "em": 0.0}


@pytest.mark.parametrize("metric", ["f1", "em"])
def test_relative_two_percent_loss_cannot_pass_absolute_pp_gate(benchmark, metric):
    base_scores, candidate_scores = {"f1": 1.0, "em": 1.0}, {"f1": 1.0, "em": 1.0}
    base_scores[metric], candidate_scores[metric] = 0.5, 0.485
    result = _summary(benchmark, [_row("recomputed_native", "qa:0", **base_scores)],
                      [_row("redknot_shared", "qa:0", **candidate_scores)])
    values = result["arms"]["redknot_shared"]
    assert values[metric + "_drop_pp"] == pytest.approx(1.5)
    assert values[metric + "_relative_drop_percent"] == pytest.approx(3.0)
    assert values["observed_2percent_gate"] is False


def test_exact_two_percent_boundary_is_not_roundoff_failure(benchmark):
    result = _summary(benchmark, [_row("recomputed_native", "qa:0")],
                      [_row("redknot_shared", "qa:0", f1=0.98, em=0.98)])
    assert result["arms"]["redknot_shared"]["observed_2percent_gate"] is True


@pytest.mark.parametrize("metric", ["f1", "em"])
def test_zero_baseline_is_noninformative_not_accuracy_pass(benchmark, metric):
    scores = {"f1": 1.0, "em": 1.0}
    scores[metric] = 0.0
    result = _summary(benchmark, [_row("recomputed_native", "qa:0", **scores)],
                      [_row("redknot_shared", "qa:0", **scores)])
    values = result["arms"]["redknot_shared"]
    assert values[metric + "_relative_drop_percent"] is None
    assert values["observed_2percent_gate"] is False


def test_per_dataset_regression_cannot_be_hidden_by_other_dataset_improvement(benchmark):
    items = _items("weak:0", "better:0")
    result = _summary(benchmark, [
        _row("recomputed_native", "weak:0", f1=0.5),
        _row("recomputed_native", "better:0", f1=0.5),
    ], [
        _row("redknot_shared", "weak:0", f1=0.47),
        _row("redknot_shared", "better:0", f1=0.53),
    ], items=items)
    values = result["arms"]["redknot_shared"]
    assert values["f1_drop_pp"] == pytest.approx(0)
    assert values["datasets"]["weak"]["f1_relative_drop_percent"] == pytest.approx(6)
    assert values["datasets"]["weak"]["observed_2percent_gate"] is False
    assert values["observed_2percent_gate"] is False


@pytest.mark.parametrize("arm", ["recomputed_native", "redknot_shared"])
@pytest.mark.parametrize("failure", ["failed", "cancelled", "service_failed", "service_error"])
def test_failures_in_either_arm_invalidate_quality_gate(benchmark, arm, failure):
    rows = {name: _row(name, "qa:0", statuses=("completed", "completed"))
            for name in ("recomputed_native", "redknot_shared")}
    if failure in ("failed", "cancelled"):
        rows[arm] = _row(arm, "qa:0", statuses=("completed", failure))
    elif failure == "service_failed":
        rows[arm]["service"]["status"] = "failed"
    else:
        rows[arm]["service"]["errors"] = [{"message": "cleanup failed"}]
    result = _summary(benchmark, [rows["recomputed_native"]], [rows["redknot_shared"]],
                      items=_items("qa:0", replicas=2))
    values = result["arms"]["redknot_shared"]
    assert values["observed_2percent_gate"] is False
    assert values["paired_comparison_complete"] is False
    assert values["paired_cohort_count"] == 0
    assert values["qps_vs_native"] is None


def test_repeat_quality_averages_not_first_or_best_and_n_is_unique(benchmark):
    result = _summary(benchmark, [
        _row("recomputed_native", "qa:0", repeat=0),
        _row("recomputed_native", "qa:0", repeat=1),
    ], [
        _row("redknot_shared", "qa:0", f1=1.0, repeat=0),
        _row("redknot_shared", "qa:0", f1=0.5, repeat=1),
    ], repeats=2)
    values = result["arms"]["redknot_shared"]
    assert values["unique_scored_examples"] == 1
    assert values["requests"] == 2
    assert values["f1"] == pytest.approx(0.75)
    assert values["f1_drop_pp"] == pytest.approx(25)
    assert values["observed_2percent_gate"] is False


def test_repeat_mean_is_inside_equal_weight_unique_example_mean(benchmark):
    items = _items("qa:0", "qa:1")
    baseline = [_row("recomputed_native", key, repeat=rep) for key in ("qa:0", "qa:1") for rep in range(2)]
    candidate = [_row("redknot_shared", "qa:0", f1=0.0, repeat=0),
                 _row("redknot_shared", "qa:0", f1=1.0, repeat=1),
                 _row("redknot_shared", "qa:1", f1=1.0, repeat=0)]
    values = _summary(benchmark, baseline, candidate, items=items, repeats=2)["arms"]["redknot_shared"]
    assert values["f1"] == pytest.approx(0.75)  # mean([mean(0,1), mean(1)])
    assert values["all_expected_cohorts_complete"] is False
    assert values["observed_2percent_gate"] is False


def test_partial_qps_comparison_is_paired_not_unequal_workload_average(benchmark):
    result = _summary(benchmark, [
        _row("recomputed_native", "qa:0", seconds=1),
        _row("recomputed_native", "qa:1", seconds=9),
    ], [_row("redknot_shared", "qa:0", seconds=0.5)], items=_items("qa:0", "qa:1"))
    values = result["arms"]["redknot_shared"]
    assert values["paired_cohort_count"] == values["paired_item_count"] == 1
    assert values["qps_vs_native"] == pytest.approx(2)  # not 10x from unequal items
    assert values["paired_comparison_complete"] is False
    assert values["observed_2percent_gate"] is False


def test_missing_repeat_and_duplicate_repeat_do_not_fake_complete_coverage(benchmark):
    baseline = [_row("recomputed_native", "qa:0", repeat=rep) for rep in (0, 1)]
    for candidate in (
        [_row("redknot_shared", "qa:0", repeat=0)],
        [_row("redknot_shared", "qa:0", repeat=0), _row("redknot_shared", "qa:0", repeat=0)],
    ):
        values = _summary(benchmark, baseline, candidate, repeats=2)["arms"]["redknot_shared"]
        assert values["all_expected_cohorts_complete"] is False
        assert values["observed_2percent_gate"] is False


def test_replicas_do_not_inflate_unique_qa_samples(benchmark):
    statuses = ("completed",) * 4
    result = _summary(benchmark, [_row("recomputed_native", "qa:0", statuses=statuses)],
                      [_row("redknot_shared", "qa:0", statuses=statuses)],
                      items=_items("qa:0", replicas=4))
    values = result["arms"]["redknot_shared"]
    assert values["unique_scored_examples"] == 1
    assert values["requests"] == 4
    assert values["observed_2percent_gate"] is True
    assert "no statistical certification" in result["scope"]
    assert "not a generalization guarantee" in result["scope"]


def test_correct_primary_cannot_hide_wrong_nonprimary_prefix(benchmark):
    baseline = _row("recomputed_native", "qa:0", statuses=("completed",) * 4)
    candidate = _row("recomputed_paged", "qa:0", statuses=("completed",) * 4)
    candidate["service"]["records"][1]["accuracy"] = {"f1": 0.0, "em": 0.0}
    values = _summary(benchmark, [baseline], [candidate], items=_items("qa:0", replicas=4))["arms"]["recomputed_paged"]
    assert values["f1"] == values["em"] == 1.0
    assert values["primary_observed_2percent_gate"] is True
    robust = values["all_prefix_robustness"]
    assert robust["f1"] == robust["em"] == 0.75
    assert robust["f1_drop_pp"] == robust["em_drop_pp"] == 25.0
    assert robust["f1_relative_drop_percent"] == robust["em_relative_drop_percent"] == 25.0
    assert robust["observed_2percent_gate"] is False
    assert robust["unique_scored_examples"] == robust["independent_sample_count"] == 1
    assert robust["scored_prefix_outputs"] == 4
    assert values["datasets"]["qa"]["primary_observed_2percent_gate"] is True
    assert values["datasets"]["qa"]["observed_2percent_gate"] is False
    assert values["observed_2percent_gate"] is False


def test_all_prefix_scores_average_repeats_and_do_not_select_best(benchmark):
    baseline = [_row("recomputed_native", "qa:0", repeat=repeat, statuses=("completed",) * 4)
                for repeat in range(2)]
    candidate = [_row("redknot_shared", "qa:0", repeat=repeat, statuses=("completed",) * 4)
                 for repeat in range(2)]
    candidate[1]["service"]["records"][3]["accuracy"] = {"f1": 0.0, "em": 0.0}
    values = _summary(benchmark, baseline, candidate, items=_items("qa:0", replicas=4), repeats=2)["arms"]["redknot_shared"]
    robust = values["all_prefix_robustness"]
    assert robust["f1"] == robust["em"] == 0.875
    assert robust["scored_prefix_outputs"] == 8
    assert robust["independent_sample_count"] == 1
    assert values["primary_observed_2percent_gate"] is True
    assert values["observed_2percent_gate"] is False


def test_all_prefix_aggregation_keeps_equal_weight_unique_examples(benchmark):
    items = _items("qa:0", replicas=4) + _items("qa:1", replicas=2)
    baseline = [_row("recomputed_native", "qa:0", statuses=("completed",) * 4),
                _row("recomputed_native", "qa:1", statuses=("completed",) * 2)]
    candidate = [_row("redknot_shared", "qa:0", statuses=("completed",) * 4),
                 _row("redknot_shared", "qa:1", statuses=("completed",) * 2)]
    for row in candidate:
        row["service"]["records"][1]["accuracy"] = {"f1": 0.0, "em": 0.0}
    robust = _summary(benchmark, baseline, candidate, items=items)["arms"]["redknot_shared"]["all_prefix_robustness"]
    assert robust["f1"] == robust["em"] == 0.625  # mean([3/4, 1/2]), not 4/6
    assert robust["independent_sample_count"] == 2
    assert robust["scored_prefix_outputs"] == 6


def test_missing_nonprimary_score_is_not_dropped_from_robustness_denominator(benchmark):
    baseline = _row("recomputed_native", "qa:0", statuses=("completed",) * 4)
    candidate = _row("redknot_shared", "qa:0", statuses=("completed",) * 4)
    candidate["service"]["records"][2]["accuracy"] = None
    values = _summary(benchmark, [baseline], [candidate], items=_items("qa:0", replicas=4))["arms"]["redknot_shared"]
    assert values["primary_observed_2percent_gate"] is True
    assert values["all_prefix_robustness"]["f1"] is None
    assert values["all_prefix_robustness"]["unique_scored_examples"] == 0
    assert values["observed_2percent_gate"] is False


def test_nonprimary_dataset_regression_cannot_hide_in_pooled_robustness(benchmark):
    items = _items("weak:0", "better:0", replicas=2)
    baseline = [_row("recomputed_native", key, statuses=("completed",) * 2)
                for key in ("weak:0", "better:0")]
    candidate = [_row("redknot_shared", key, statuses=("completed",) * 2)
                 for key in ("weak:0", "better:0")]
    for row in baseline:
        row["service"]["records"][1]["accuracy"] = {"f1": 0.5, "em": 0.5}
    candidate[0]["service"]["records"][1]["accuracy"] = {"f1": 0.44, "em": 0.44}
    candidate[1]["service"]["records"][1]["accuracy"] = {"f1": 0.56, "em": 0.56}
    values = _summary(benchmark, baseline, candidate, items=items)["arms"]["redknot_shared"]
    robust = values["all_prefix_robustness"]
    assert values["primary_observed_2percent_gate"] is True
    assert robust["f1_drop_pp"] == pytest.approx(0)
    assert robust["metric_2percent_gate"] is True
    assert robust["datasets"]["weak"]["observed_2percent_gate"] is False
    assert robust["observed_2percent_gate"] is False
    assert values["observed_2percent_gate"] is False


def test_missing_records_or_nonfinite_scores_fail_closed(benchmark):
    baseline = [_row("recomputed_native", "qa:0")]
    for malformed in ("empty", "nan", "out_of_range"):
        candidate = _row("redknot_shared", "qa:0")
        if malformed == "empty":
            candidate["service"]["records"] = []
            candidate["service"]["completed_requests"] = 0
        else:
            candidate["service"]["records"][0]["accuracy"]["f1"] = float("nan") if malformed == "nan" else 2.0
        result = _summary(benchmark, baseline, [candidate])
        assert result["arms"]["redknot_shared"]["observed_2percent_gate"] is False
        json.dumps(result, allow_nan=False)


def test_no_native_control_means_no_speedup_or_quality_pass(benchmark):
    result = _summary(benchmark, [], [_row("redknot_shared", "qa:0")])
    assert result["arms"]["redknot_shared"]["qps_vs_native"] is None
    assert result["arms"]["redknot_shared"]["observed_2percent_gate"] is False


class _Tokenizer:
    def __len__(self):
        return 256

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) for char in text]


def _freeze_args(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for dataset in ("hotpotqa", "musique"):
        rows = [dict(_id=f"{dataset}-{index}", context="Original full source context. " * 2 + str(index),
                     input=f"Question {index}?", answers=[f"Answer {index}"])
                for index in range(6)]
        (data / (dataset + ".jsonl")).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return types.SimpleNamespace(
        manifest=None, datasets=["hotpotqa", "musique"], data=data, model=tmp_path / "model",
        seed=23, min_context=20, max_context=100, document_chunk=8,
        replicas=3, output_tokens=16, items_per_dataset=2,
    )


def test_freezing_is_deterministic_length_only_and_exact_prompt_tokens(benchmark, tmp_path):
    args, tokenizer = _freeze_args(tmp_path), _Tokenizer()
    first, second = benchmark.freeze_workload(args, tokenizer), benchmark.freeze_workload(args, tokenizer)
    assert first == second
    assert first["unique_examples"] == 4
    assert first["request_replicas"] == 3
    assert first["quality_scope"].startswith("replica-zero primary plus all-prefix robustness")
    for item in first["items"]:
        document = sum(item["document_token_ids"], [])
        assert len(document) == item["context_tokens"]
        assert args.min_context <= len(document) <= args.max_context
        assert benchmark.digest(item["document_token_ids"]) == item["document_tokens_sha256"]
        assert len(item["prefix_token_ids"]) == len(item["exact_prompt_sha256"]) == 3
        for index, prefix in enumerate(item["prefix_token_ids"]):
            assert benchmark.digest(prefix + document + item["query_token_ids"]) == item["exact_prompt_sha256"][index]
        assert item["prefix_token_ids"][0] != item["prefix_token_ids"][1]
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps(first))
    args.manifest = manifest
    assert benchmark.freeze_workload(args, tokenizer) == first


def test_query_instruction_preserves_source_selection_and_exact_arm_inputs(benchmark, tmp_path):
    args, tokenizer = _freeze_args(tmp_path), _Tokenizer()
    original = benchmark.freeze_workload(args, tokenizer)
    args.instruction_at_query = True
    updated = benchmark.freeze_workload(args, tokenizer)
    assert updated["instruction_at_query"]
    assert [x["key"] for x in original["items"]] == [x["key"] for x in updated["items"]]
    for old, new in zip(original["items"], updated["items"]):
        assert old["document_token_ids"] == new["document_token_ids"]
        assert old["prefix_token_ids"] == new["prefix_token_ids"]
        assert old["answers"] == new["answers"]
        assert "Output only the answer itself" in "".join(map(chr, new["query_token_ids"]))
        assert old["exact_prompt_sha256"] != new["exact_prompt_sha256"]
    args.exclude_manifest = tmp_path / "calibration.json"
    args.exclude_manifest.write_text(json.dumps(original))
    held_out = benchmark.freeze_workload(args, tokenizer)
    assert set(x["key"] for x in original["items"]).isdisjoint(x["key"] for x in held_out["items"])


def test_frozen_document_hash_corruption_is_rejected_before_model_load(benchmark, tmp_path):
    args, tokenizer = _freeze_args(tmp_path), _Tokenizer()
    payload = benchmark.freeze_workload(args, tokenizer)
    payload["items"][0]["document_token_ids"][0][0] += 1
    args.manifest = tmp_path / "corrupt.json"
    args.manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        benchmark.freeze_workload(args, tokenizer)


def test_variable_prefix_lengths_change_offsets_not_documents(benchmark, tmp_path):
    args, tokenizer = _freeze_args(tmp_path), _Tokenizer()
    original = benchmark.freeze_workload(args, tokenizer)
    args.vary_prefix_lengths = True
    varied = benchmark.freeze_workload(args, tokenizer)
    assert varied["varying_prefix_lengths"]
    for old, new in zip(original["items"], varied["items"]):
        assert old["document_token_ids"] == new["document_token_ids"]
        assert old["query_token_ids"] == new["query_token_ids"]
        assert old["answers"] == new["answers"]
        assert len(set(map(len, new["prefix_token_ids"]))) == args.replicas


@pytest.mark.parametrize("corruption", ["prefix", "query", "prefix_count", "hash_count", "empty_items", "output_tokens", "duplicate_key", "same_prefix", "empty_answers"])
def test_loaded_manifest_rejects_protocol_or_prompt_corruption(benchmark, tmp_path, corruption):
    args, tokenizer = _freeze_args(tmp_path), _Tokenizer()
    payload = benchmark.freeze_workload(args, tokenizer)
    item = payload["items"][0]
    if corruption == "prefix":
        item["prefix_token_ids"][0][0] += 1
    elif corruption == "query":
        item["query_token_ids"][0] += 1
    elif corruption == "prefix_count":
        item["prefix_token_ids"].pop()
    elif corruption == "hash_count":
        item["exact_prompt_sha256"].pop()
    elif corruption == "empty_items":
        payload["items"] = []
    elif corruption == "output_tokens":
        payload["max_new_tokens"] += 1
    elif corruption == "duplicate_key":
        payload["items"][1]["key"] = item["key"]
    elif corruption == "same_prefix":
        item["prefix_token_ids"][1] = item["prefix_token_ids"][0]
        item["exact_prompt_sha256"][1] = item["exact_prompt_sha256"][0]
    elif corruption == "empty_answers":
        item["answers"] = []
    args.manifest = tmp_path / "invalid-manifest.json"
    args.manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        benchmark.freeze_workload(args, tokenizer)


def test_atomic_save_does_not_leave_partial_json(benchmark, tmp_path):
    path = tmp_path / "nested" / "results.json"
    benchmark.save(path, {"status": "running", "rows": []})
    benchmark.save(path, {"status": "completed", "rows": [1, 2]})
    assert json.loads(path.read_text()) == {"status": "completed", "rows": [1, 2]}
    assert not path.with_suffix(".json.tmp").exists()
