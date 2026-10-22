"""Safety and real-Qwen numerical tests for the optional FA2 paged backend.

Random tiny model weights validate implementation contracts only. These tests
are not pretrained-model accuracy evidence or performance measurements.
"""

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


_HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def backend_module():
    # The optional file must import even where FlashAttention is not installed.
    path = _HERE / "segpaged_qwen3_flash_backend.py"
    sys.path.insert(0, str(_HERE))
    spec = importlib.util.spec_from_file_location("_tested_flash_paged_backend", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)
        sys.path.remove(str(_HERE))


def _tiny_model(device="cpu", dtype=torch.float32):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.1":
        pytest.skip("the adapter requires Transformers 4.57.1")
    config = transformers.Qwen3Config(
        vocab_size=257, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4,
        head_dim=16, max_position_embeddings=1024, attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(617251)
        return transformers.Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()


def _hash_tensors(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().view(torch.uint8).cpu()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _document_hash(adapter):
    return _hash_tensors(
        tensor
        for ref in adapter._document_refs
        for tensor in (adapter.pool.k[ref.slot], adapter.pool.v[ref.slot])
    )


def test_flash_backend_cpu_rejected_before_flash_import(backend_module, monkeypatch):
    model = _tiny_model()
    original = [layer.self_attn.forward for layer in model.model.layers]
    # Setting this to None makes a late FlashAttention import fail if attempted;
    # the CPU safety check must happen first, with no fake CUDA environment.
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    with pytest.raises(ValueError, match="requires a CUDA model and cache"):
        backend_module.BatchedQwen3FlashPagedDecode(
            model, None, document_length=256, batch_size=2,
            shared_kv_head_ratio=0.75, capacity_pages=32, page_size=256,
            max_suffix_tokens=10,
        )
    assert [layer.self_attn.forward for layer in model.model.layers] == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ratio", [0.0, 0.75, 1.0])
def test_flash_paged_gqa_real_model_matches_native_and_preserves_source(
    backend_module, dtype, ratio,
):
    pytest.importorskip("flash_attn")
    from segpaged_qwen3_throughput_adapter import required_capacity_pages

    model = _tiny_model("cuda", dtype)
    document = (torch.arange(256, device="cuda") % 127 + 1).unsqueeze(0)
    chunks = [
        torch.tensor([[31, 32, 33, 34, 35, 36], [71, 72, 73, 74, 75, 76]], device="cuda"),
        *[torch.tensor([[37 + step], [77 + step]], device="cuda") for step in range(4)],
    ]
    with torch.no_grad():
        source = model(document, use_cache=True).past_key_values
        native_cache = copy.deepcopy(source)
        native_cache.batch_repeat_interleave(2)
        expected = []
        for tokens in chunks:
            output = model(tokens, past_key_values=native_cache, use_cache=True,
                           logits_to_keep=0)
            expected.append(output.logits.clone())
            native_cache = output.past_key_values
    source_hash = _hash_tensors(t for layer in source.layers for t in (layer.keys, layer.values))
    original = [layer.self_attn.forward for layer in model.model.layers]
    pages = required_capacity_pages(model, 256, 2, ratio, 10, page_size=256)
    adapter = backend_module.BatchedQwen3FlashPagedDecode(
        model, source, document_length=256, batch_size=2,
        shared_kv_head_ratio=ratio, capacity_pages=pages, page_size=256,
        max_suffix_tokens=10,
    )
    maximum_allowed_error = 0.003 if dtype == torch.float16 else 0.05
    native_calls = []
    flash_forward = adapter._flash_attention

    def observed_flash(query, keys, values, **kwargs):
        assert query.shape[0] == 2 * 4  # Batch x physical KV heads.
        assert query.shape[2] == 2      # GQA query heads per physical KV head.
        assert keys.shape[2] == values.shape[2] == 1
        assert kwargs["k"] is None and kwargs["v"] is None
        assert kwargs["causal"] is True
        native_calls.append(tuple(query.shape))
        return flash_forward(query, keys, values, **kwargs)

    adapter._flash_attention = observed_flash
    with adapter:
        document_hash_before = _document_hash(adapter)
        for tokens, oracle in zip(chunks, expected):
            actual = adapter.forward(tokens, logits_to_keep=0)
            assert actual.past_key_values is None, "no dense HF history fallback is allowed"
            error = (actual.logits.float() - oracle.float()).abs().max().item()
            assert error <= maximum_allowed_error, (dtype, ratio, error)
        assert _document_hash(adapter) == document_hash_before
        adapter.reset_suffix()
        assert adapter.suffix_tokens == 0
        assert torch.all(adapter._flash_cache_lengths == 256)
        reset_output = adapter.forward(chunks[0], logits_to_keep=0)
        error = (reset_output.logits.float() - expected[0].float()).abs().max().item()
        assert error <= maximum_allowed_error, (dtype, ratio, error)
        assert _document_hash(adapter) == document_hash_before
        stats = adapter.stats()
        assert stats["realized_shared_kv_head_ratio"] == ratio
        assert stats["attention_calls"] == len(model.model.layers) * (len(chunks) + 1)
        assert len(native_calls) == stats["attention_calls"]
        assert stats["native_flash_paged_cache_read_only"] is True
        assert stats["historical_kv_gather"] is False
        assert stats["pool"]["free_pages"] == 0
    assert [layer.self_attn.forward for layer in model.model.layers] == original
    assert _hash_tensors(t for layer in source.layers for t in (layer.keys, layer.values)) == source_hash
    assert source.get_seq_length() == 256
    pool_stats = adapter.pool.stats()
    assert pool_stats["free_pages"] == pages
    assert pool_stats["allocated_bytes"] == 0
    assert pool_stats["pinned_pages"] == 0
    adapter.close()  # Idempotent cleanup.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("document_length, page_size, message", [
    (255, 256, "page-aligned"), (256, 128, "positive multiple of 256"),
    (512, 0, "positive multiple of 256"),
])
def test_flash_backend_rejects_incompatible_layout_before_cache_access(
    backend_module, document_length, page_size, message,
):
    model = _tiny_model("cuda", torch.bfloat16)
    with pytest.raises(ValueError, match=message):
        backend_module.BatchedQwen3FlashPagedDecode(
            model, None, document_length=document_length, batch_size=2,
            shared_kv_head_ratio=0.75, capacity_pages=32, page_size=page_size,
            max_suffix_tokens=10,
        )
