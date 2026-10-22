"""CPU numerical checks of ragged native queues and the all-global control.

These seeded tiny random Qwen3 weights test implementation equivalence, not
task accuracy. The oracle recomputes the entire token sequence without a cache
for every comparison, independently of the reference engine's padded caches.
"""
from contextlib import contextmanager
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


_HERE = Path(__file__).resolve().parent
_SOURCE = _HERE.parents[2] / "python"


@contextmanager
def _imports():
    saved = {name: module for name, module in sys.modules.items()
             if name == "sglang" or name.startswith("sglang.")}
    for name in saved:
        del sys.modules[name]
    native_name = "_redknot_native_reference_test_target"
    previous_native = sys.modules.get(native_name)
    try:
        for name in ("sglang", "sglang.srt", "sglang.srt.layers",
                     "sglang.srt.layers.attention", "sglang.srt.layers.attention.redknot",
                     "sglang.srt.mem_cache"):
            module = types.ModuleType(name)
            module.__path__ = [str(_SOURCE.joinpath(*name.split(".")))]
            sys.modules[name] = module
        spec = importlib.util.spec_from_file_location(native_name, _HERE / "nonprefix_native_reference.py")
        native = importlib.util.module_from_spec(spec)
        sys.modules[native_name] = native
        spec.loader.exec_module(native)
        yield types.SimpleNamespace(
            native=native,
            core=importlib.import_module("sglang.srt.layers.attention.redknot.nonprefix_paged"),
            config=importlib.import_module("sglang.srt.layers.attention.redknot.head_config"),
            scheduler=importlib.import_module("sglang.srt.layers.attention.redknot.nonprefix_scheduler"),
        )
    finally:
        for name in list(sys.modules):
            if name == "sglang" or name.startswith("sglang."):
                del sys.modules[name]
        sys.modules.update(saved)
        if previous_native is None:
            sys.modules.pop(native_name, None)
        else:
            sys.modules[native_name] = previous_native


@pytest.fixture(scope="module")
def runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with _imports() as modules:
            yield modules
    finally:
        torch.set_num_threads(previous)


def _model():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.1":
        pytest.skip("the experimental binding requires Transformers 4.57.1")
    config = transformers.Qwen3Config(
        vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=512, attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(4901)
        return transformers.Qwen3ForCausalLM(config).eval()


_DOCUMENTS = {"D": [11, 12, 13, 14, 15, 16, 17, 18, 19], "E": [31, 32, 33, 34, 35]}


def _history(prefix, documents):
    return list(prefix) + [token for name in documents for token in _DOCUMENTS[name]]


@torch.inference_mode()
def _full_recompute_logits(model, tokens):
    return model(input_ids=torch.tensor([tokens]), use_cache=False, logits_to_keep=1).logits


@torch.inference_mode()
def _full_recompute_generate(model, tokens, count):
    history, output = list(tokens), []
    for _ in range(count):
        token = int(_full_recompute_logits(model, history)[0, 0].argmax())
        output.append(token)
        history.append(token)
    return output


def _make_engine(runtime, model, kind, prefix_chunk_size):
    if kind == "native":
        engine = runtime.native.NativeQueuedEngine(model, prefix_chunk_size=prefix_chunk_size)
    else:
        config = runtime.config.HeadClassConfig(
            [["global", "global"]] * 2, [[-1, -1]] * 2,
            2, 2, head_sink_size=[[0, 0]] * 2, default_sink_size=0,
        )
        engine = runtime.core.NonPrefixPagedEngine(
            model, config, capacity_pages=2048, page_size=4, backend="torch",
            sparse_ffn_schedule=None, prefix_chunk_size=prefix_chunk_size,
            share_local=kind == "allglobal_shared", model_revision="random-tiny-test-only",
        )
    for name, tokens in _DOCUMENTS.items():
        engine.register_document(name, tokens)
    return engine


@pytest.mark.parametrize("prefix_chunk_size", [1, 4, 128])
@pytest.mark.parametrize("kind", ["native", "allglobal_private", "allglobal_shared"])
def test_ragged_batches_and_refill_match_independent_no_cache_recompute(
    runtime, kind, prefix_chunk_size,
):
    model = _model()
    engine = _make_engine(runtime, model, kind, prefix_chunk_size)
    histories = {}
    try:
        for name, prefix, documents in (
            ("A", [1, 2, 3], ["D"]),
            ("B", [6, 5, 4, 3, 2, 1, 9, 8], ["E", "D"]),
        ):
            engine.admit(name, prefix, documents, max_suffix_tokens=16)
            histories[name] = _history(prefix, documents)

        def step(names, inputs):
            expected = []
            for name, ids in zip(names, inputs):
                histories[name].extend(ids)
                expected.append(_full_recompute_logits(model, histories[name]))
            actual = engine.forward(names, torch.tensor(inputs))
            torch.testing.assert_close(actual, torch.cat(expected), rtol=4e-5, atol=4e-6)
            if kind == "native":
                for name in names:
                    state = engine.requests[name]
                    for layer in state.cache.layers:
                        assert layer.keys.shape[0] == layer.values.shape[0] == 1
                        assert layer.keys.shape[2] == layer.values.shape[2] == len(histories[name])
                    assert state.history_tokens + state.suffix_tokens == len(histories[name])

        step(["A", "B"], [[41, 42], [51, 52]])
        step(["B"], [[53]])
        # Swapping rows and mixing decode ages must preserve request ownership.
        step(["B", "A"], [[54], [43]])
        engine.release("A")
        engine.admit("C", [7], ["D", "E", "D"], max_suffix_tokens=16)
        histories["C"] = _history([7], ["D", "E", "D"])
        step(["B", "C"], [[55, 56, 57], [61, 62, 63]])
        step(["C", "B"], [[64], [58]])
        assert engine.requests["B"].suffix_tokens == 8
        assert engine.requests["C"].suffix_tokens == 4
        if kind == "native":
            assert engine.stats()["repacked_kv_bytes"] > 0
        for name in ("B", "C"):
            engine.release(name)
        assert engine.requests == {}
    finally:
        engine.close()


@pytest.mark.parametrize("kind", ["native", "allglobal_private", "allglobal_shared"])
def test_scheduler_outputs_match_independent_full_recompute_with_mixed_query_lengths(runtime, kind):
    model = _model()
    engine = _make_engine(runtime, model, kind, prefix_chunk_size=4)
    Q = runtime.scheduler.QueuedRequest
    requests = [
        Q("A", [1, 2], ["D"], [41, 42], 4, []),
        Q("B", [8, 9, 7, 6], ["E", "D"], [51, 52, 53], 3, []),
        Q("C", [5], ["D", "E"], [61, 62], 2, []),
    ]
    expected = {
        request.request_id: _full_recompute_generate(
            model, _history(request.prefix_ids, request.document_ids) + list(request.query_ids),
            request.max_new_tokens,
        )
        for request in requests
    }
    try:
        result = runtime.scheduler.run_queued(engine, requests, max_active=2, ignore_eos=True)
        assert result["status"] == "complete", result["errors"]
        assert result["completed_requests"] == 3
        assert result["max_forward_batch"] == 2
        assert {record["request_id"]: record["output_ids"] for record in result["records"]} == expected
        assert all(record["ttft_seconds"] > 0 for record in result["records"])
        assert all(record["latency_seconds"] >= record["ttft_seconds"] for record in result["records"])
        assert engine.requests == {}
    finally:
        engine.close()


def test_native_document_registration_does_not_prefill_or_store_gpu_cache(runtime):
    model = _model()
    engine = _make_engine(runtime, model, "native", prefix_chunk_size=4)
    try:
        assert engine.requests == {}
        assert engine.forward_calls == 0
        assert all(ids.ndim == 1 and ids.device.type == "cpu" for ids in engine.documents.values())
        engine.register_document("D", _DOCUMENTS["D"])
        with pytest.raises(ValueError, match="collision"):
            engine.register_document("D", [99])
        assert engine.stats()["full_recomputation"] is True
        assert engine.stats()["offline_kv_reuse"] is False
    finally:
        engine.close()


def test_native_cache_padding_is_not_visible_attention(runtime):
    model = _model()
    engine = _make_engine(runtime, model, "native", prefix_chunk_size=128)
    try:
        # Very different lengths make an omitted left-padding mask observable.
        engine.admit("short", [1], ["E"], max_suffix_tokens=8)
        engine.admit("long", list(range(1, 61)), ["D", "E"], max_suffix_tokens=8)
        inputs = [[81, 82, 83], [84, 85, 86]]
        expected = torch.cat([
            _full_recompute_logits(model, _history([1], ["E"]) + inputs[0]),
            _full_recompute_logits(model, _history(list(range(1, 61)), ["D", "E"]) + inputs[1]),
        ])
        actual = engine.forward(["short", "long"], torch.tensor(inputs))
        torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)
    finally:
        engine.close()
