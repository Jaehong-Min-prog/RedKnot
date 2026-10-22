"""CPU numerical and lifecycle tests for real-model paged throughput decoding.

The weights here are seeded, tiny, randomly initialized Qwen3 weights. These
tests validate the storage/forward contract, not pretrained model quality or
measured throughput. Every model output must actually consume the paged K/V.
"""

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


_HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def adapter_module():
    path = _HERE / "segpaged_qwen3_throughput_adapter.py"
    spec = importlib.util.spec_from_file_location("_tested_qps_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def benchmark_module():
    path = _HERE / "benchmark_segpaged_qps.py"
    spec = importlib.util.spec_from_file_location("_tested_qps_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model_and_document(*, backend="eager", length=11, device="cpu",
                        dtype=torch.float32, kv_heads=2):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.1":
        pytest.skip("adapter deliberately pins the Transformers 4.57.1 contract")
    config = transformers.Qwen3Config(
        vocab_size=127,
        hidden_size=32 * kv_heads,
        intermediate_size=64 * kv_heads,
        num_hidden_layers=2,
        num_attention_heads=2 * kv_heads,
        num_key_value_heads=kv_heads,
        head_dim=16,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = backend
    with torch.random.fork_rng():
        torch.manual_seed(73023)
        model = transformers.Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()
    document = torch.arange(1, length + 1, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        cache = model(document, use_cache=True).past_key_values
    return model, document, cache


def _make_adapter(module, model, cache, *, ratio=0.5, batch_size=2,
                  document_length=11, max_suffix_tokens=8, page_size=4,
                  capacity_pages=None):
    if capacity_pages is None:
        capacity_pages = module.required_capacity_pages(
            model, document_length, batch_size, ratio, max_suffix_tokens,
            page_size=page_size,
        )
    return module.BatchedQwen3PagedDecode(
        model,
        cache,
        document_length=document_length,
        batch_size=batch_size,
        shared_kv_head_ratio=ratio,
        capacity_pages=capacity_pages,
        page_size=page_size,
        max_suffix_tokens=max_suffix_tokens,
        device=next(model.parameters()).device,
    )


def _hf_answers(model, document_cache, chunks, *, batch_size=2):
    """Full HF dense-history oracle, independently populated for each batch row."""
    cache = copy.deepcopy(document_cache)
    cache.batch_repeat_interleave(batch_size)
    results = []
    with torch.no_grad():
        for tokens in chunks:
            output = model(tokens, past_key_values=cache, use_cache=True,
                           logits_to_keep=0)
            results.append(output.logits.clone())
            cache = output.past_key_values
    return results


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_batched_real_model_logits_match_hf(adapter_module, backend, ratio):
    model, _, cache = _model_and_document(backend=backend)
    chunks = [
        torch.tensor([[40, 41, 42], [77, 78, 79]]),
        torch.tensor([[43], [80]]),
        torch.tensor([[44, 45], [81, 82]]),
    ]
    expected = _hf_answers(model, cache, chunks)
    original_forwards = [layer.self_attn.forward for layer in model.model.layers]
    with _make_adapter(adapter_module, model, cache, ratio=ratio) as adapter:
        for tokens, oracle in zip(chunks, expected):
            with torch.no_grad():
                output = adapter.forward(tokens, logits_to_keep=0)
            assert output.past_key_values is None, "HF history must not shadow paged history"
            torch.testing.assert_close(output.logits, oracle, rtol=3e-5, atol=3e-6)
            # Distinct requests must not accidentally select one shared query.
            assert not torch.allclose(output.logits[0], output.logits[1])
    assert [layer.self_attn.forward for layer in model.model.layers] == original_forwards


@pytest.mark.parametrize("ratio, expected_pages", [(0.0, 40), (0.5, 34), (1.0, 28)])
def test_budget_geometry_and_actual_pool_reservation(adapter_module, ratio, expected_pages):
    model, _, cache = _model_and_document()
    required = adapter_module.required_capacity_pages(model, 11, 2, ratio, 8, page_size=4)
    assert required == expected_pages
    with _make_adapter(adapter_module, model, cache, ratio=ratio) as adapter:
        stats = adapter.stats()["pool"]
        assert stats["capacity_pages"] == expected_pages
        assert stats["free_pages"] == 0
        assert stats["allocated_bytes"] == expected_pages * 2 * 4 * 16 * 4
        assert stats["reserved_slab_bytes"] == stats["allocated_bytes"]


def test_shared_head_ratio_rounds_physical_kv_heads_down(adapter_module):
    model, _, _ = _model_and_document()
    pages = lambda ratio: adapter_module.required_capacity_pages(
        model, 11, 2, ratio, 8, page_size=4)
    assert pages(0.49) == pages(0.0)
    assert pages(0.99) == pages(0.5)


def test_too_small_pool_fails_instead_of_overcommitting(adapter_module):
    model, _, cache = _model_and_document()
    from head_kv import CapacityError
    required = adapter_module.required_capacity_pages(model, 11, 2, 0.5, 8, page_size=4)
    original = [layer.self_attn.forward for layer in model.model.layers]
    with pytest.raises(CapacityError):
        with _make_adapter(adapter_module, model, cache, capacity_pages=required - 1):
            pytest.fail("insufficient physical pages must not admit the batch")
    assert [layer.self_attn.forward for layer in model.model.layers] == original


def test_reset_drops_previous_request_suffix_and_preserves_document(adapter_module):
    model, _, cache = _model_and_document()
    first = [torch.tensor([[30, 31, 32], [60, 61, 62]]),
             torch.tensor([[33, 34], [63, 64]])]
    second = [torch.tensor([[90], [100]]), torch.tensor([[91, 92], [101, 102]])]
    expected = _hf_answers(model, cache, second)
    snapshots = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    with _make_adapter(adapter_module, model, cache) as adapter:
        with torch.no_grad():
            for tokens in first:
                adapter.forward(tokens)
            adapter.reset_suffix()
            for tokens, oracle in zip(second, expected):
                output = adapter.forward(tokens, logits_to_keep=0)
                torch.testing.assert_close(output.logits, oracle, rtol=3e-5, atol=3e-6)
    for layer, (keys, values) in zip(cache.layers, snapshots):
        torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, values, rtol=0, atol=0)


def test_source_cache_is_not_a_hidden_forward_dependency(adapter_module):
    model, _, cache = _model_and_document()
    tokens = torch.tensor([[31, 32], [71, 72]])
    expected, = _hf_answers(model, cache, [tokens])
    with _make_adapter(adapter_module, model, cache) as adapter:
        # After admission, only the bounded page pool may supply history.
        # Zeroing the original dense fixture must not alter generated logits.
        for layer in cache.layers:
            layer.keys.zero_()
            layer.values.zero_()
        with torch.no_grad():
            actual = adapter.forward(tokens, logits_to_keep=0)
        torch.testing.assert_close(actual.logits, expected, rtol=3e-5, atol=3e-6)


def test_paged_history_changes_reach_model_logits(adapter_module):
    model, _, cache = _model_and_document()
    tokens = torch.tensor([[31, 32], [71, 72]])
    with _make_adapter(adapter_module, model, cache) as adapter:
        with torch.no_grad():
            before = adapter.forward(tokens).logits.clone()
            adapter.reset_suffix()
            # Deliberate white-box corruption, only in an isolated test: a
            # no-op sidecar attention implementation would ignore this change.
            adapter.pool.v.zero_()
            after = adapter.forward(tokens).logits.clone()
        assert (after - before).abs().max().item() > 1e-4


def test_binding_restored_after_body_exception_and_close_is_idempotent(adapter_module):
    model, _, cache = _model_and_document()
    original = [layer.self_attn.forward for layer in model.model.layers]
    adapter = _make_adapter(adapter_module, model, cache)
    with pytest.raises(RuntimeError, match="intentional test exception"):
        with adapter:
            raise RuntimeError("intentional test exception")
    assert [layer.self_attn.forward for layer in model.model.layers] == original
    adapter.close()
    adapter.close()


def test_suffix_limit_rejected_and_reset_reusable(adapter_module):
    model, _, cache = _model_and_document()
    with _make_adapter(adapter_module, model, cache, max_suffix_tokens=3) as adapter:
        with torch.no_grad():
            adapter.forward(torch.tensor([[31, 32, 33], [71, 72, 73]]))
            with pytest.raises((ValueError, RuntimeError)):
                adapter.forward(torch.tensor([[34], [74]]))
            adapter.reset_suffix()
            actual = adapter.forward(torch.tensor([[41], [81]]))
        assert actual.logits.shape == (2, 1, model.config.vocab_size)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_ratios_fail_before_admission(adapter_module, ratio):
    model, _, _ = _model_and_document()
    with pytest.raises(ValueError):
        adapter_module.required_capacity_pages(model, 11, 2, ratio, 8, page_size=4)


def test_invalid_batch_does_not_advance_the_adapter(adapter_module):
    model, _, cache = _model_and_document()
    expected, = _hf_answers(model, cache, [torch.tensor([[31], [71]])])
    with _make_adapter(adapter_module, model, cache) as adapter:
        for bad in [torch.tensor([[31]]), torch.empty((2, 0), dtype=torch.long),
                    torch.ones((2, 1), dtype=torch.float32)]:
            with pytest.raises(ValueError):
                adapter.forward(bad)
            assert adapter.suffix_tokens == 0
        with torch.no_grad():
            actual = adapter.forward(torch.tensor([[31], [71]]))
        torch.testing.assert_close(actual.logits, expected, rtol=3e-5, atol=3e-6)


def test_close_releases_all_owned_and_pinned_pages(adapter_module):
    model, _, cache = _model_and_document()
    adapter = _make_adapter(adapter_module, model, cache)
    with adapter:
        assert adapter.pool.stats()["pinned_pages"] > 0
        adapter.forward(torch.tensor([[31], [71]]))
    stats = adapter.pool.stats()
    assert stats["free_pages"] == stats["capacity_pages"]
    assert stats["allocated_bytes"] == 0
    assert stats["pinned_pages"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ratio", [0.0, 0.75, 1.0])
def test_cuda_batched_real_model_paged_logits_match_hf(adapter_module, dtype, ratio):
    """Exercises actual Triton virtual-head batching, not the CPU fallback."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton is required for paged GPU attention")
    model, _, cache = _model_and_document(backend="sdpa", device="cuda", dtype=dtype,
                                        kv_heads=4)
    chunks = [
        torch.tensor([[40, 41, 42], [77, 78, 79]], device="cuda"),
        torch.tensor([[43], [80]], device="cuda"),
        torch.tensor([[44, 45], [81, 82]], device="cuda"),
    ]
    expected = _hf_answers(model, cache, chunks)
    tolerance = 3e-5 if dtype == torch.float32 else 2.5e-2
    with _make_adapter(adapter_module, model, cache, ratio=ratio) as adapter:
        for tokens, oracle in zip(chunks, expected):
            actual = adapter.forward(tokens, logits_to_keep=0)
            assert actual.past_key_values is None
            torch.testing.assert_close(actual.logits.float(), oracle.float(),
                                       rtol=tolerance, atol=tolerance)
        adapter.reset_suffix()
        actual = adapter.forward(chunks[0], logits_to_keep=0)
        torch.testing.assert_close(actual.logits.float(), expected[0].float(),
                                   rtol=tolerance, atol=tolerance)
        assert adapter.stats()["attention_calls"] == len(model.model.layers) * 4


def _benchmark_args(**updates):
    args = dict(question_tokens=3, budget_requests=4, page_size=4,
                shared_ratios=[0.0, 0.5, 0.75, 1.0], max_batch=16,
                same_concurrency=[2, 4], lengths=[11], output_tokens=[6],
                warmup_rounds=1, repeats=3, prefill_chunk_size=8, cpu_threads=1,
                row_index=0, datasets=["hotpotqa"])
    args.update(updates)
    return SimpleNamespace(**args)


def _measurement(ratio, batch, *, elapsed=2.0, resident=1.0, tokens=None):
    if tokens is None:
        tokens = [[11 + index, 31 + index] for index in range(batch)]
    return dict(shared_ratio=ratio, batch_size=batch, budget_max_batch=batch,
                max_batch_search_censored=False, completed_requests=batch,
                admission_inclusive_seconds=elapsed, resident_generation_seconds=resident,
                qps=batch / elapsed, resident_generation_qps=batch / resident,
                admission_seconds=elapsed - resident, resident_ttft_seconds=resident / 4,
                admission_inclusive_ttft_seconds=elapsed - resident + resident / 4,
                output_tokens_per_second=batch * 2 / elapsed,
                generated_token_ids=tokens, cache_stats={"test_fixture": True},
                gpu_peak_allocated_bytes=123456)


def test_summary_aggregate_rate_is_completed_requests_over_total_time(benchmark_module):
    samples = [_measurement(0.0, 2, elapsed=elapsed, resident=elapsed / 2)
               for elapsed in (1.0, 1.0, 8.0)]
    result, = benchmark_module.summarize(samples)
    assert result["aggregate_qps"] == pytest.approx(6 / 10)
    assert result["aggregate_resident_qps"] == pytest.approx(6 / 5)
    assert result["qps_median"] == 2.0
    assert result["resident_generation_qps_median"] == 4.0
    assert result["aggregate_qps"] != result["qps_median"]
    assert result["qps_p95_sample_quantile"] == 2.0
    assert result["repeats"] == 3


def test_summary_marks_same_batch_generated_token_mismatch(benchmark_module):
    baseline = _measurement(0.0, 2)
    modified = _measurement(0.5, 2, tokens=[[11, 31], [12, 99]])
    rows = benchmark_module.summarize([baseline, modified])
    shared = next(row for row in rows if row["shared_ratio"] == 0.5)
    assert shared["same_batch_baseline_token_identity"] is False
    assert shared["same_concurrency_qps_speedup"] == 1.0
    assert shared["cross_batch_overlap_token_agreement"] == 0.75
    assert shared["cross_batch_agreement_is_not_task_accuracy"] is True


def test_summary_repeat_identity_checks_every_repeat(benchmark_module):
    baseline = _measurement(0.0, 2)
    shared_a = _measurement(0.5, 2)
    shared_b = _measurement(0.5, 2, tokens=[[11, 31], [12, 99]])
    shared = next(row for row in benchmark_module.summarize([baseline, shared_a, shared_b])
                  if row["shared_ratio"] == 0.5)
    assert shared["repeat_output_identity"] is False
    assert shared["same_batch_baseline_token_identity"] is False


def test_summary_cross_batch_overlap_is_explicitly_not_task_quality(benchmark_module):
    baseline = _measurement(0.0, 2, tokens=[[1, 2], [3, 4]])
    shared = _measurement(0.75, 4, tokens=[[1, 9], [3, 4], [8, 9], [10, 11]])
    result = next(row for row in benchmark_module.summarize([baseline, shared])
                  if row["shared_ratio"] == 0.75)
    assert result["cross_batch_overlap_token_agreement"] == 0.75
    assert result["cross_batch_agreement_is_not_task_accuracy"] is True
    assert "same_batch_baseline_token_identity" not in result
    assert "same_concurrency_qps_speedup" not in result
    assert result["versus_baseline_max_concurrency_qps_speedup"] == 2.0


def test_configurations_hold_one_budget_across_all_sharing_ratios(benchmark_module):
    model, _, _ = _model_and_document(kv_heads=4)
    args = _benchmark_args()
    configs = benchmark_module.configurations(model, 11, args, 6)
    assert {row["capacity_pages"] for row in configs} == {160}
    assert {row["shared_ratio"] for row in configs} == {0.0, 0.5, 0.75, 1.0}
    expected_maxima = {0.0: 4, 0.5: 5, 0.75: 6, 1.0: 8}
    for row in configs:
        assert row["required_pages"] <= row["capacity_pages"]
        assert row["budget_max_batch"] == expected_maxima[row["shared_ratio"]]
        assert row["max_batch_search_censored"] is False
        assert row["max_suffix_tokens"] == args.question_tokens + 6 - 1
    for ratio, largest in expected_maxima.items():
        assert any(row["shared_ratio"] == ratio and row["batch_size"] == largest
                   for row in configs)


def test_configurations_expose_max_batch_search_censoring(benchmark_module):
    model, _, _ = _model_and_document(kv_heads=4)
    configs = benchmark_module.configurations(model, 11, _benchmark_args(max_batch=5), 6)
    for row in configs:
        expected_censored = row["shared_ratio"] != 0.0
        assert row["max_batch_search_censored"] is expected_censored
        assert row["budget_max_batch"] == (5 if expected_censored else 4)


@pytest.mark.parametrize("output_tokens, suffix_tokens, budget_pages", [(6, 8, 160), (7, 9, 192)])
def test_configurations_budget_includes_question_and_all_decode_input_rows(
    benchmark_module, output_tokens, suffix_tokens, budget_pages,
):
    model, _, _ = _model_and_document(kv_heads=4)
    configs = benchmark_module.configurations(model, 11, _benchmark_args(), output_tokens)
    assert all(row["max_suffix_tokens"] == suffix_tokens for row in configs)
    assert {row["capacity_pages"] for row in configs} == {budget_pages}
    # The final sampled token is emitted, not fed back through the model.
    assert suffix_tokens == 3 + output_tokens - 1


def test_configurations_require_private_copy_control(benchmark_module):
    model, _, _ = _model_and_document(kv_heads=4)
    with pytest.raises(ValueError, match="private-copy baseline is mandatory"):
        benchmark_module.configurations(model, 11,
                                        _benchmark_args(shared_ratios=[0.5, 0.75, 1]), 6)


@pytest.mark.parametrize("updates", [
    {"lengths": [0]}, {"output_tokens": [0]}, {"question_tokens": 0},
    {"page_size": 0}, {"max_batch": 0}, {"budget_requests": 0}, {"repeats": 0},
    {"warmup_rounds": 0}, {"shared_ratios": [0.5, 1.0]},
    {"shared_ratios": [0.0, 0.5]},
    {"same_concurrency": [0]}, {"prefill_chunk_size": 0}, {"cpu_threads": 0},
])
def test_main_invalid_measurement_gates_reject_before_gpu_access(
    benchmark_module, monkeypatch, tmp_path, updates,
):
    args = _benchmark_args(output=tmp_path / "result.json", **updates)
    monkeypatch.setattr(benchmark_module, "parse_args", lambda: args)

    def forbidden_gpu_access(*args, **kwargs):
        pytest.fail("invalid benchmark settings must fail before GPU access")

    monkeypatch.setattr(torch.cuda, "get_device_name", forbidden_gpu_access)
    with pytest.raises(ValueError):
        benchmark_module.main()
    assert not args.output.exists()


def test_main_refuses_to_overwrite_raw_results(benchmark_module, monkeypatch, tmp_path):
    path = tmp_path / "already-measured.json"
    content = '{"status":"complete","raw_evidence":true}\n'
    path.write_text(content)
    monkeypatch.setattr(benchmark_module, "parse_args", lambda: _benchmark_args(output=path))
    with pytest.raises(FileExistsError):
        benchmark_module.main()
    assert path.read_text() == content


@pytest.mark.parametrize("corruption", [None, "logit_shift", "nonfinite"])
def test_independent_native_logit_gate_is_fail_closed(
    benchmark_module, monkeypatch, corruption,
):
    model, _, source = _model_and_document(backend="sdpa")
    real_adapter = benchmark_module.BatchedQwen3PagedDecode

    class ModifiedAdapter(real_adapter):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            if corruption == "logit_shift":
                output.logits = output.logits + 0.75
            elif corruption == "nonfinite":
                output.logits = torch.full_like(output.logits, float("nan"))
            return output

    # The gate's native backend label reflects its production FA2 call site;
    # this CPU-only unit test validates the gate logic, not a GPU measurement.
    monkeypatch.setattr(benchmark_module, "sync", lambda: None)
    monkeypatch.setattr(benchmark_module, "BatchedQwen3PagedDecode", ModifiedAdapter)
    gate = benchmark_module.independent_model_gate(
        model, source, torch.tensor([[31, 32], [71, 72]]), 11, 4, new_tokens=3)
    assert gate["source_cache_length_unchanged"] is True
    assert gate["passed"] is (corruption is None)
    if corruption == "logit_shift":
        assert gate["first_token_equal"] is True
        assert gate["first_logits_max_abs_error"] > 0.5
    elif corruption == "nonfinite":
        assert gate["all_native_and_paged_gate_logits_finite"] is False
