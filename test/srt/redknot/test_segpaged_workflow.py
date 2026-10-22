"""Regressions for the managed query stage in the actual RedKnot RAG driver.

Synthetic tensors validate numerical contracts; the tiny mock model validates
driver wiring, cleanup and the *emitted* first-token timing boundary. These tests
are not model-quality or performance evidence. CUDA tests use the same adapter.
"""

from contextlib import contextmanager
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


_SOURCE = Path(__file__).resolve().parents[3] / "python"


@contextmanager
def _isolated_imports():
    saved = {
        name: module for name, module in sys.modules.items()
        if name == "sglang" or name.startswith("sglang.")
    }
    for name in saved:
        del sys.modules[name]
    try:
        for name in (
            "sglang", "sglang.srt", "sglang.srt.layers",
            "sglang.srt.layers.attention", "sglang.srt.layers.attention.redknot",
            "sglang.srt.mem_cache",
        ):
            module = types.ModuleType(name)
            module.__path__ = [str(_SOURCE.joinpath(*name.split(".")))]
            sys.modules[name] = module
        yield types.SimpleNamespace(
            helper=importlib.import_module(
                "sglang.srt.layers.attention.redknot.managed_query"
            ),
            driver=importlib.import_module(
                "sglang.srt.layers.attention.redknot.driver_batched"
            ),
            config=importlib.import_module(
                "sglang.srt.layers.attention.redknot.head_config"
            ),
            offline_cache=importlib.import_module(
                "sglang.srt.layers.attention.redknot.offline_cache"
            ),
            prefill=importlib.import_module(
                "sglang.srt.layers.attention.redknot.prefill"
            ),
        )
    finally:
        for name in list(sys.modules):
            if name == "sglang" or name.startswith("sglang."):
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(scope="module")
def runtime():
    with _isolated_imports() as modules:
        yield modules


def _case(*, device="cpu", dtype=torch.float32, local=(True, False), queries=4):
    generator = torch.Generator().manual_seed(815)
    heads, groups, length, dim = 2, 3, 17, 8

    def sample(shape):
        return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)

    # Retained offline local values deliberately differ from online global KV.
    # Actual retained positions have a hole, and must not be renumbered or
    # subjected to a second moving-window filter as query position increases.
    local_positions = (0, 1, 12, 13, 14, 15, 16)
    entry = {
        "k_global": sample((1, heads, length, dim)),
        "v_global": sample((1, heads, length, dim)),
        "k_local": sample((1, heads, len(local_positions), dim)),
        "v_local": sample((1, heads, len(local_positions), dim)),
        "is_local": torch.tensor(local, device=device),
        "local_positions": local_positions,
    }
    q = sample((1, heads * groups, queries, dim))
    k, v = (sample((1, heads, queries, dim)) for _ in range(2))
    return entry, q, k, v, groups


def _oracle(entry, q, k, v, groups):
    """Independent FP32 reference: fixed selected prefix + causal query suffix."""
    result = torch.empty_like(q, dtype=torch.float32)
    query_count, scale = q.shape[2], q.shape[-1] ** -0.5
    for query_head in range(q.shape[1]):
        kv_head = query_head // groups
        local = bool(entry["is_local"][kv_head])
        pk = entry["k_local" if local else "k_global"][0, kv_head].float()
        pv = entry["v_local" if local else "v_global"][0, kv_head].float()
        for row in range(query_count):
            keys = torch.cat((pk, k[0, kv_head, :row + 1].float()))
            values = torch.cat((pv, v[0, kv_head, :row + 1].float()))
            scores = q[0, query_head, row].float() @ keys.T * scale
            result[0, query_head, row] = scores.softmax(-1) @ values
    return result


@pytest.mark.parametrize("local", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("queries", [1, 4])
@pytest.mark.parametrize("skip_invisible", [False, True])
def test_managed_query_preserves_static_prefix_and_gqa(
    runtime, local, queries, skip_invisible
):
    entry, q, k, v, groups = _case(local=local, queries=queries)
    snapshots = {name: value.clone() for name, value in entry.items()
                 if isinstance(value, torch.Tensor)}
    cache = runtime.helper.ManagedHeadClassQueryCache(
        [entry], query_offset=17, query_length=queries, page_size=4,
        skip_invisible=skip_invisible,
    )
    try:
        cache.prefill_layer(0)
        actual = cache.attention(
            q, k, v, layer=0, num_q_per_kv=groups, sm_scale=q.shape[-1] ** -0.5,
        )
        native = runtime.driver._query_headclass_attention(
            q, torch.cat((entry["k_global"], k), 2),
            torch.cat((entry["v_global"], v), 2),
            torch.cat((entry["k_local"], k), 2),
            torch.cat((entry["v_local"], v), 2), entry["is_local"],
            num_q_per_kv=groups, sm_scale=q.shape[-1] ** -0.5,
        )
        torch.testing.assert_close(actual.float(), _oracle(entry, q, k, v, groups),
                                   rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual, native, rtol=2e-5, atol=2e-6)
        for name, expected in snapshots.items():
            torch.testing.assert_close(entry[name], expected, rtol=0, atol=0)
    finally:
        cache.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_managed_query_cuda_matches_native_and_reference(runtime, dtype):
    entry, q, k, v, groups = _case(device="cuda", dtype=dtype)
    outputs = []
    for enabled in (False, True):
        cache = runtime.helper.ManagedHeadClassQueryCache(
            [entry], query_offset=17, query_length=4, page_size=4,
            skip_invisible=enabled,
        )
        try:
            cache.prefill_layer(0)
            outputs.append(cache.attention(
                q, k, v, layer=0, num_q_per_kv=groups,
                sm_scale=q.shape[-1] ** -0.5,
            ))
        finally:
            cache.close()
    torch.cuda.synchronize()
    native = runtime.driver._query_headclass_attention(
        q, torch.cat((entry["k_global"], k), 2),
        torch.cat((entry["v_global"], v), 2),
        torch.cat((entry["k_local"], k), 2),
        torch.cat((entry["v_local"], v), 2), entry["is_local"],
        num_q_per_kv=groups, sm_scale=q.shape[-1] ** -0.5,
    )
    for actual in outputs:
        torch.testing.assert_close(actual.float(), _oracle(entry, q, k, v, groups),
                                   rtol=0.025, atol=0.025)
        torch.testing.assert_close(actual, native, rtol=0.025, atol=0.025)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


class _Clock:
    now = 100.0

    def __call__(self):
        return self.now


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim, self.num_key_value_groups, self.scaling = 4, 2, 0.5
        self.q_proj = torch.nn.Linear(16, 16, bias=False)
        self.k_proj = torch.nn.Linear(16, 8, bias=False)
        self.v_proj = torch.nn.Linear(16, 8, bias=False)
        self.o_proj = torch.nn.Identity()

    def forward(self, hidden_states, **kwargs):
        return hidden_states, None


class _TinyDriverModel(torch.nn.Module):
    """No learned weights/quality claim: exercise patched forward and timer."""

    def __init__(self, clock, *, fail_stage=None):
        super().__init__()
        self.attention = _Attention()
        self.model = types.SimpleNamespace(
            layers=[types.SimpleNamespace(self_attn=self.attention)]
        )
        self.config = types.SimpleNamespace(
            num_hidden_layers=1, model_type="test", _attn_implementation="eager"
        )
        self.device = torch.device("cpu")
        self.clock, self.fail_stage = clock, fail_stage
        self.calls = []

    def forward(self, input_ids, *, use_cache, past_key_values=None, **kwargs):
        stage = "seed" if use_cache else "headclass"
        self.calls.append(stage)
        self.clock.now += 5 if use_cache else 2
        hidden = torch.ones(1, input_ids.shape[1], 16)
        self.attention(
            hidden_states=hidden,
            position_embeddings=(torch.ones(1, input_ids.shape[1], 4),
                                 torch.zeros(1, input_ids.shape[1], 4)),
        )
        if self.fail_stage == stage:
            raise RuntimeError(f"injected {stage} failure")
        logits = torch.zeros(1, input_ids.shape[1], 5)
        logits[..., 2 if use_cache else 1] = 10
        return types.SimpleNamespace(logits=logits, past_key_values=past_key_values)


class _Tokenizer:
    eos_token_id = 4

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[0, 1]])}

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def _driver_case(runtime, monkeypatch, *, fail_stage=None):
    clock = _Clock()
    model = _TinyDriverModel(clock, fail_stage=fail_stage)
    config = runtime.config.HeadClassConfig(
        [["local", "global"]], [[3, -1]], 1, 2,
        head_sink_size=[[1, 0]],
    )
    segment = types.SimpleNamespace(
        doc_len=8, token_ids=torch.arange(8),
        kv=[(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4))],
    )

    def online(*args, **kwargs):
        clock.now += 3
        return [segment.kv]

    monkeypatch.setattr(runtime.driver, "online_forward_segments_flat", online)
    monkeypatch.setattr(runtime.driver, "_get_apply_rotary", lambda kind:
                        lambda q, k, cos, sin: (q, k))
    # Patch the driver's module-local time object, not Python's shared time.
    monkeypatch.setattr(runtime.driver, "time", types.SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return model, dict(
        tokenizer=_Tokenizer(), segments_offline=[segment], query_text="question",
        head_cfg=config, rope_helper=types.SimpleNamespace(), max_new_tokens=1,
    )


@pytest.mark.parametrize("backend", ["native", "managed_paged"])
def test_driver_ttft_ends_at_emitted_first_token(runtime, monkeypatch, backend):
    model, kwargs = _driver_case(runtime, monkeypatch)
    original = model.attention.forward
    stats = {}
    first_logits, text, query_length, ttft = runtime.driver.run_redknot_offlinekv(
        model, **kwargs, query_backend=backend, timing_stats=stats,
    )
    assert model.calls == ["headclass", "seed"]
    assert text == "2"
    # Preserve the established return value (head-class diagnostic logits), but
    # never confuse it with the seed forward that produces the emitted token.
    assert first_logits.argmax().item() == 1
    assert stats["emitted_first_token_id"] == 2
    assert query_length == 2
    assert ttft == pytest.approx(10.0)
    assert model.attention.forward == original


@pytest.mark.parametrize("backend", ["native", "managed_paged"])
@pytest.mark.parametrize("fail_stage", ["headclass", "seed"])
def test_driver_restores_model_forward_after_failure(runtime, monkeypatch, backend, fail_stage):
    model, kwargs = _driver_case(runtime, monkeypatch, fail_stage=fail_stage)
    original = model.attention.forward
    with pytest.raises(RuntimeError, match=f"injected {fail_stage} failure"):
        runtime.driver.run_redknot_offlinekv(model, **kwargs, query_backend=backend)
    assert model.attention.forward == original
    assert model.config._attn_implementation == "eager"


@pytest.mark.parametrize("fail_stage", ["headclass", "seed"])
def test_driver_failure_releases_managed_request(runtime, monkeypatch, fail_stage):
    model, kwargs = _driver_case(runtime, monkeypatch, fail_stage=fail_stage)
    managers = []
    implementation = runtime.helper.ManagedHeadClassQueryCache

    class _ObservedCache(implementation):
        def close(self):
            if self.manager is not None:
                managers.append(self.manager)
            super().close()

    monkeypatch.setattr(runtime.helper, "ManagedHeadClassQueryCache", _ObservedCache)
    with pytest.raises(RuntimeError, match=f"injected {fail_stage} failure"):
        runtime.driver.run_redknot_offlinekv(model, **kwargs, query_backend="managed_paged")
    assert len(managers) == 1
    stats = managers[0].stats()
    assert stats["requests"] == stats["pending_transactions"] == 0
    assert stats["free_pages"] == stats["capacity_pages"]


def test_driver_rejects_unknown_backend_before_online_work(runtime, monkeypatch):
    model, kwargs = _driver_case(runtime, monkeypatch)
    with pytest.raises(ValueError, match="query_backend"):
        runtime.driver.run_redknot_offlinekv(model, **kwargs, query_backend="silently_fallback")
    assert model.calls == []


@pytest.mark.parametrize("value", [None, 0, "true"])
def test_driver_rejects_invalid_paged_option_before_work(runtime, monkeypatch, value):
    model, kwargs = _driver_case(runtime, monkeypatch)
    with pytest.raises(TypeError, match="paged_skip_invisible"):
        runtime.driver.run_redknot_offlinekv(model, **kwargs, paged_skip_invisible=value)
    assert model.calls == []


def test_driver_rejects_empty_generation_before_work(runtime, monkeypatch):
    model, kwargs = _driver_case(runtime, monkeypatch)
    kwargs["max_new_tokens"] = 0
    with pytest.raises(ValueError, match="max_new_tokens"):
        runtime.driver.run_redknot_offlinekv(model, **kwargs)
    assert model.calls == []


def test_managed_query_keeps_actual_positions_and_releases_ownership(runtime):
    entry, q, k, v, groups = _case()
    cache = runtime.helper.ManagedHeadClassQueryCache(
        [entry], query_offset=17, query_length=4, page_size=4,
    )
    cache.prefill_layer(0)
    manager = cache.manager
    version = manager.version(cache.request_id)
    assert version.segments[(0, 0, "query-prefix")].positions == entry["local_positions"]
    assert version.segments[(0, 1, "query-prefix")].positions == tuple(range(17))
    cache.attention(q, k, v, layer=0, num_q_per_kv=groups, sm_scale=8 ** -0.5)
    assert cache.stats["manager_update_calls"] == 1
    assert cache.stats["manager_append_many_calls"] == 1
    assert cache.stats["managed_attention_calls"] == 1
    with pytest.raises(RuntimeError, match="one query forward"):
        cache.attention(q, k, v, layer=0, num_q_per_kv=groups, sm_scale=8 ** -0.5)
    cache.close()
    cache.close()
    stats = manager.stats()
    assert stats["requests"] == 0
    assert stats["pending_transactions"] == 0
    assert stats["live_pages"] == stats["reserved_pages"] == stats["pinned_pages"] == 0
    assert stats["free_pages"] == stats["capacity_pages"]


@pytest.mark.parametrize("name,value", [
    ("query_offset", 0), ("query_offset", True), ("query_length", 0),
    ("query_length", 1.5), ("page_size", 0), ("page_size", False),
])
def test_managed_query_rejects_invalid_geometry(runtime, name, value):
    entry, *_ = _case()
    options = dict(query_offset=17, query_length=4, page_size=4)
    options[name] = value
    with pytest.raises(ValueError, match=name):
        runtime.helper.ManagedHeadClassQueryCache([entry], **options)


@pytest.mark.parametrize("value", [None, 0, 1, "yes"])
def test_managed_query_skip_knob_is_explicit_bool(runtime, value):
    entry, *_ = _case()
    with pytest.raises(TypeError, match="skip_invisible"):
        runtime.helper.ManagedHeadClassQueryCache(
            [entry], query_offset=17, query_length=4, skip_invisible=value,
        )


@pytest.mark.parametrize("positions", [
    (0, 1, 12, 13, 14, 15, 15), (0, 1, 12, 13, 14, 15, 17),
    (-1, 1, 12, 13, 14, 15, 16), (0, 1),
])
def test_managed_query_rejects_invalid_retained_positions(runtime, positions):
    entry, *_ = _case()
    entry["local_positions"] = positions
    with pytest.raises(ValueError, match="local_positions"):
        runtime.helper.ManagedHeadClassQueryCache(
            [entry], query_offset=17, query_length=4,
        )


@pytest.mark.parametrize("mixed", [False, True])
def test_attention_impl_switch_snapshots_shared_configs_before_mutation(runtime, mixed):
    """HF attention modules often alias the exact model configuration object."""
    model_config = types.SimpleNamespace(_attn_implementation="flash_attention_2")
    eager_config = types.SimpleNamespace(_attn_implementation="eager")
    fa3_config = types.SimpleNamespace(_attn_implementation="flash_attention_3")
    configs = (
        [model_config, eager_config, model_config, fa3_config, eager_config]
        if mixed else [model_config] * 5
    )
    model = types.SimpleNamespace(
        config=model_config,
        model=types.SimpleNamespace(layers=[
            types.SimpleNamespace(self_attn=types.SimpleNamespace(config=config))
            for config in configs
        ]),
    )
    expected = [(config, config._attn_implementation) for config in configs]
    original = runtime.driver._switch_attn_impl(model, "sdpa")
    assert model.config._attn_implementation == "sdpa"
    assert all(config._attn_implementation == "sdpa" for config in configs)
    runtime.driver._restore_attn_impl(model, original)
    assert model.config._attn_implementation == "flash_attention_2"
    for config, implementation in expected:
        assert config._attn_implementation == implementation
    assert all(layer.self_attn.config is config
               for layer, config in zip(model.model.layers, configs))


def test_attention_impl_nested_switch_restores_each_shared_snapshot(runtime):
    config = types.SimpleNamespace(_attn_implementation="flash_attention_2")
    model = types.SimpleNamespace(
        config=config,
        model=types.SimpleNamespace(layers=[
            types.SimpleNamespace(self_attn=types.SimpleNamespace(config=config))
            for _ in range(3)
        ]),
    )
    outer = runtime.driver._switch_attn_impl(model, "sdpa")
    inner = runtime.driver._switch_attn_impl(model, "eager")
    assert config._attn_implementation == "eager"
    runtime.driver._restore_attn_impl(model, inner)
    assert config._attn_implementation == "sdpa"
    runtime.driver._restore_attn_impl(model, outer)
    assert config._attn_implementation == "flash_attention_2"


def test_attention_impl_restore_handles_parent_setter_propagation(runtime):
    children = [types.SimpleNamespace(_attn_implementation=name)
                for name in ("eager", "flash_attention_3")]

    class _CompositeConfig:
        def __init__(self):
            self._implementation = "flash_attention_2"

        @property
        def _attn_implementation(self):
            return self._implementation

        @_attn_implementation.setter
        def _attn_implementation(self, value):
            self._implementation = value
            for child in children:
                child._attn_implementation = value

    parent = _CompositeConfig()
    model = types.SimpleNamespace(
        config=parent,
        model=types.SimpleNamespace(layers=[
            types.SimpleNamespace(self_attn=types.SimpleNamespace(config=config))
            for config in children
        ] + [types.SimpleNamespace(self_attn=types.SimpleNamespace())]),
    )
    original = runtime.driver._switch_attn_impl(model, "sdpa")
    assert all(config._attn_implementation == "sdpa" for config in children)
    runtime.driver._restore_attn_impl(model, original)
    assert parent._attn_implementation == "flash_attention_2"
    assert [config._attn_implementation for config in children] == [
        "eager", "flash_attention_3",
    ]


@pytest.mark.parametrize("backend", ["native", "managed_paged"])
@pytest.mark.parametrize("fail_stage", [None, "seed"])
def test_driver_restores_aliased_attention_config_on_success_and_failure(
    runtime, monkeypatch, backend, fail_stage
):
    model, kwargs = _driver_case(runtime, monkeypatch, fail_stage=fail_stage)
    model.config._attn_implementation = "flash_attention_2"
    model.attention.config = model.config
    if fail_stage:
        with pytest.raises(RuntimeError, match="injected seed failure"):
            runtime.driver.run_redknot_offlinekv(model, **kwargs, query_backend=backend)
    else:
        runtime.driver.run_redknot_offlinekv(model, **kwargs, query_backend=backend)
    assert model.attention.config is model.config
    assert model.config._attn_implementation == "flash_attention_2"


class _CaptureTokenizer:
    bos_token_id = None

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.arange(7).unsqueeze(0)}


class _CapturePrefillModel(torch.nn.Module):
    """Deterministic hook/cache fixture; no checkpoint or quality evidence."""

    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(2):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Identity()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.o_proj = torch.nn.Identity()
            layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)
        self.model = types.SimpleNamespace(layers=self.layers)
        self.config = types.SimpleNamespace(num_hidden_layers=2, _name_or_path="fixture")
        self.device = torch.device("cpu")
        self.chunk_lengths = []

    def forward(self, input_ids, *, past_key_values, use_cache):
        self.chunk_lengths.append(input_ids.shape[1])
        for index, layer in enumerate(self.layers):
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4) + index
            projected = layer.self_attn.o_proj(hidden + 10)
            layer.mlp(projected + 20)
            key = hidden.reshape(1, input_ids.shape[1], 2, 2).transpose(1, 2)
            past_key_values.update(key, key + 0.5, index)
        return types.SimpleNamespace(past_key_values=past_key_values)


@pytest.mark.parametrize("capture_ffn,capture_attn", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_chunked_offline_capture_preserves_legacy_keys_and_accounts_storage(
    runtime, monkeypatch, capture_ffn, capture_attn,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = _CapturePrefillModel()
    store = runtime.offline_cache.OfflineKVCache()
    segment = runtime.prefill.offline_prefill_segment(
        model, _CaptureTokenizer(), "fixture", chunk_size=3,
        model_id="fixture-model", extra_key="existing-key", store=store,
        capture_ffn_delta=capture_ffn, capture_attn_output=capture_attn,
    )
    assert model.chunk_lengths == [3, 3, 1]
    assert segment.doc_len == 7
    extra = "existing-key"
    if capture_ffn or capture_attn:
        extra += f"|ffn_delta={int(capture_ffn)}|attn_output={int(capture_attn)}"
    expected_key = runtime.offline_cache.OfflineKVCache.compute_segment_id(
        "fixture-model", list(range(7)), extra=extra,
    )
    assert segment.segment_id == expected_key
    expected_bytes = sum(value.numel() * value.element_size()
                         for pair in segment.kv for value in pair)
    for name, enabled, addition in (
        ("ffn_delta", capture_ffn, 30), ("attn_output", capture_attn, 10),
    ):
        captured = getattr(segment, name)
        if not enabled:
            assert captured is None
            continue
        assert len(captured) == 2
        for index, value in enumerate(captured):
            expected = (torch.arange(7).float().view(1, 7, 1)
                        .expand(1, 7, 4) + index + addition)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            expected_bytes += value.numel() * value.element_size()
    assert segment.nbytes == expected_bytes
    assert store.stats() == dict(n_segments=1, device_bytes=0, host_bytes=expected_bytes)
    assert store.get(expected_key) is segment
    for layer in model.layers:
        assert not layer.mlp._forward_hooks
        assert not layer.self_attn.o_proj._forward_pre_hooks
    store.drop(expected_key)
    assert store.stats() == dict(n_segments=0, device_bytes=0, host_bytes=0)


def test_offline_segment_keeps_old_positional_metadata_argument(runtime):
    metadata = {"adapter": "existing-positional-call"}
    segment = runtime.offline_cache.OfflineSegment(
        "legacy", torch.tensor([1]), 1, [], torch.device("cpu"), 0, None, metadata,
    )
    assert segment.meta is metadata
    assert segment.ffn_delta is None
    assert segment.attn_output is None


def _flat_online_fixture(runtime, monkeypatch):
    model = _TinyDriverModel(_Clock())
    model.config.num_attention_heads = 4
    model.config.head_dim = 4
    config = runtime.config.HeadClassConfig(
        [["local", "global"]], [[3, -1]], 1, 2, head_sink_size=[[1, 0]],
    )
    segments = [types.SimpleNamespace(
        doc_len=length, token_ids=torch.arange(length),
        kv=[(torch.zeros(1, 2, length, 4), torch.ones(1, 2, length, 4))],
    ) for length in (3, 4, 2)]
    calls = []

    def custom(_base, **kwargs):
        calls.append(kwargs)
        length = kwargs["flat_ids"].shape[1]
        kwargs["captured_kv"][0] = (
            torch.zeros(1, 2, length, 4), torch.ones(1, 2, length, 4),
        )

    monkeypatch.setattr(runtime.driver, "_get_apply_rotary", lambda kind: None)
    monkeypatch.setattr(runtime.driver, "_run_flat_custom", custom)
    monkeypatch.setenv("REDKNOT_CUSTOM_FWD", "1")
    return model, segments, config, calls


def test_flat_online_accepts_existing_segments_without_optional_capture(runtime, monkeypatch):
    model, segments, config, calls = _flat_online_fixture(runtime, monkeypatch)
    result = runtime.driver.online_forward_segments_flat(
        model, segments_offline=segments, head_cfg=config, rope_helper=None,
        sparse_ffn_schedule=runtime.driver.SparseFFNSchedule(),
    )
    assert len(result) == 3
    assert [part[0][0].shape[2] for part in result] == [3, 4, 2]
    assert len(calls) == 1
    assert calls[0]["fallback_ffn_deltas"] is None
    assert calls[0]["fallback_attn_outputs"] is None


def test_flat_online_requested_local_output_reuse_requires_capture(runtime, monkeypatch):
    model, segments, config, calls = _flat_online_fixture(runtime, monkeypatch)
    with pytest.raises(RuntimeError, match="requires offline attention outputs"):
        runtime.driver.online_forward_segments_flat(
            model, segments_offline=segments, head_cfg=config, rope_helper=None,
            reuse_local_attn_output=True,
        )
    assert not calls


def test_flat_online_passes_captured_payloads_only_to_opted_in_paths(runtime, monkeypatch):
    model, segments, config, calls = _flat_online_fixture(runtime, monkeypatch)
    for index, segment in enumerate(segments[1:]):
        segment.ffn_delta = [torch.full((1, segment.doc_len, 16), index + 3.)]
        segment.attn_output = [torch.full((1, segment.doc_len, 16), index + 7.)]
    runtime.driver.online_forward_segments_flat(
        model, segments_offline=segments, head_cfg=config, rope_helper=None,
        sparse_ffn_schedule=runtime.driver.SparseFFNSchedule(),
        reuse_local_attn_output=True,
    )
    expected_ffn = torch.cat([segment.ffn_delta[0] for segment in segments[1:]], 1)
    expected_attn = torch.cat([segment.attn_output[0] for segment in segments[1:]], 1)
    torch.testing.assert_close(calls[0]["fallback_ffn_deltas"][0], expected_ffn)
    torch.testing.assert_close(
        calls[0]["fallback_attn_outputs"][0], expected_attn.view(1, 6, 4, 4).transpose(1, 2),
    )
    runtime.driver.online_forward_segments_flat(
        model, segments_offline=segments, head_cfg=config, rope_helper=None,
    )
    assert calls[1]["fallback_ffn_deltas"] is None
    assert calls[1]["fallback_attn_outputs"] is None


def test_flat_online_mixed_ffn_capture_fails_before_model_work(runtime, monkeypatch):
    model, segments, config, calls = _flat_online_fixture(runtime, monkeypatch)
    segments[1].ffn_delta = [torch.zeros(1, segments[1].doc_len, 16)]
    with pytest.raises(RuntimeError, match="cover every online document"):
        runtime.driver.online_forward_segments_flat(
            model, segments_offline=segments, head_cfg=config, rope_helper=None,
            sparse_ffn_schedule=runtime.driver.SparseFFNSchedule(),
        )
    assert not calls


@pytest.mark.parametrize("full_scope", [False, True])
def test_driver_managed_query_records_selected_or_full_actual_positions(
    runtime, monkeypatch, full_scope,
):
    model, kwargs = _driver_case(runtime, monkeypatch)
    positions = []
    implementation = runtime.helper.ManagedHeadClassQueryCache

    class _ObservedCache(implementation):
        def __init__(self, layer_kv, **options):
            positions.extend(kv["local_positions"] for kv in layer_kv)
            super().__init__(layer_kv, **options)

    monkeypatch.setattr(runtime.helper, "ManagedHeadClassQueryCache", _ObservedCache)
    stats = {}
    runtime.driver.run_redknot_offlinekv(
        model, **kwargs, query_backend="managed_paged",
        query_local_full_scope=full_scope, timing_stats=stats,
    )
    assert positions == ([tuple(range(8))] if full_scope else [(0, 5, 6, 7)])
    assert stats["query_local_full_scope"] is full_scope
    assert stats["reuse_local_attn_output"] is False
