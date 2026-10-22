"""Row execution, coordinate, lifetime and causal contracts; not task accuracy."""
import importlib
import math
from types import SimpleNamespace

import pytest
import torch

from test_nonprefix_paged import _isolated_imports, _tiny_model
from test_nonprefix_recovery_fastpath import _dense_flash_oracle, _fixture, _old_recovery


@pytest.fixture(scope="module")
def runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with _isolated_imports() as modules:
            modules.selective = importlib.import_module("sglang.srt.layers.attention.redknot.selective_recovery")
            yield modules
    finally:
        torch.set_num_threads(previous)


def _make(runtime, config=None, *, rope_type="default", layers=4, mixed=True,
          device="cpu", dtype=torch.float32, backend="torch", optimization=True):
    model = _tiny_model(rope_type=rope_type, num_layers=layers, device=device, dtype=dtype)
    heads = [["local" if mixed and l == layers-2 and h == 0 else "global" for h in range(2)]
             for l in range(layers)]
    policy = runtime.config.HeadClassConfig(heads,
        [[5 if t == "local" else -1 for t in row] for row in heads], layers, 2,
        default_sink_size=2)
    return runtime.core.NonPrefixPagedEngine(model, policy, capacity_pages=4096,
        page_size=256 if backend == "flash" else 4, backend=backend,
        prefix_chunk_size=256, selective_recovery=config,
        selective_execution_optimization=optimization)


def _register(engine):
    for name, ids in {"D": list(range(20, 31)), "E": list(range(50, 57))}.items():
        engine.register_document(name, ids)


@pytest.mark.parametrize("field,value", [
    ("block_size", 0), ("block_size", True), ("probe_layers", 0),
    ("row_fraction", 0), ("row_fraction", 1.01), ("row_fraction", float("nan")),
    ("row_fraction", True), ("max_offline_kv_bytes", -1),
])
def test_config_rejects_unbounded_or_invalid_values(runtime, field, value):
    with pytest.raises(ValueError):
        runtime.selective.SelectiveRecoveryConfig(**{field: value})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_retained_cache_estimator_includes_all_layers_heads_docs_and_dtype(runtime, dtype):
    model = _tiny_model(num_layers=4)
    actual = runtime.selective.estimate_selective_cache_bytes(model.config, [11, 7], dtype)
    assert actual == 2*4*2*16*18*torch.empty((), dtype=dtype).element_size()
    assert runtime.selective.estimate_selective_cache_bytes(model.config, [], dtype) == 0
    with pytest.raises(ValueError):
        runtime.selective.estimate_selective_cache_bytes(model.config, [0], dtype)


def test_selection_uses_valid_partial_block_mean_and_mandatory_tail(runtime):
    config = runtime.selective.SelectiveRecoveryConfig(block_size=4, row_fraction=.5)
    scores = torch.tensor([1.]*4 + [3.]*4 + [2.]*4 + [0.])
    selected, intervals = runtime.selective.select_token_blocks(scores, config)
    assert selected == (1, 3) and intervals == ((4, 8), (12, 13))
    # Ties are stable; adjacent selected blocks become one attention interval.
    selected, intervals = runtime.selective.select_token_blocks(torch.ones(13), config)
    assert selected == (0, 3) and intervals == ((0, 4), (12, 13))
    with pytest.raises(FloatingPointError):
        runtime.selective.select_token_blocks(torch.tensor([float("nan")]), config)


@pytest.mark.parametrize("mode", ["fraction1", "all_probe", "small_context"])
def test_full_rows_delegate_original_path_exactly(runtime, mode):
    Config = runtime.selective.SelectiveRecoveryConfig
    cfg = {"fraction1": Config(row_fraction=1, max_offline_kv_bytes=0),
           "all_probe": Config(probe_layers=4, max_offline_kv_bytes=0),
           "small_context": Config(block_size=256, row_fraction=.5)}[mode]
    reference = _make(runtime)
    candidate = _make(runtime, cfg)
    try:
        for engine in (reference, candidate):
            _register(engine)
        calls = []
        original = candidate._recover_online_full
        def observed(*args):
            calls.append(True)
            return original(*args)
        candidate._recover_online_full = observed
        outputs = []
        for engine in (reference, candidate):
            state = engine.admit("r", [1, 2, 3], ["D", "E"], max_suffix_tokens=8)
            outputs.append(engine.forward(["r"], torch.tensor([[81, 82, 83]])).clone())
            assert state.admission_stats["projected_rows_per_layer"] == [18]*4
            assert state.admission_stats["ffn_rows_per_layer"] == [18]*4
            assert state.admission_stats["recovery_approximate"] is False
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        assert calls == [True]
        if mode != "small_context":
            assert candidate.stats()["offline_kv_cache_bytes"] == 0
    finally:
        reference.close(); candidate.close()


def test_full_cache_budget_preflight_is_atomic_and_actual_storage_is_charged(runtime):
    tiny = runtime.selective.SelectiveRecoveryConfig(block_size=4, max_offline_kv_bytes=1)
    engine = _make(runtime, tiny)
    try:
        before = engine.pool.stats()
        with pytest.raises(runtime.core.CapacityError, match="offline KV budget"):
            engine.register_document("D", list(range(20, 31)))
        assert not engine.documents and engine.pool.stats() == before
        assert engine.manager.stats()["requests"] == 0
    finally:
        engine.close()
    config = runtime.selective.SelectiveRecoveryConfig(block_size=4)
    engine = _make(runtime, config)
    try:
        _register(engine)
        expected = runtime.selective.estimate_selective_cache_bytes(engine.model.config, [11, 7], engine.dtype)
        assert engine.stats()["offline_kv_cache_bytes"] == expected
        assert sum(d.recovery_nbytes for d in engine.documents.values()) == expected
        assert engine.pool.stats()["allocated_bytes"] > 0  # Local-page duplicate is separate.
    finally:
        engine.close()
    assert engine.stats()["offline_kv_cache_bytes"] == 0


def test_real_qkv_and_mlp_hooks_confirm_selected_rows_are_skipped(runtime):
    config = runtime.selective.SelectiveRecoveryConfig(block_size=4, row_fraction=.25, probe_layers=2)
    engine = _make(runtime, config, layers=6)
    try:
        _register(engine)
        prefix = engine._segment("prefix", [1, 2, 3], engine._prefill_ids([1, 2, 3]))
        docs = [engine._segment(d.doc_id, d.token_ids, d.recovery_kv) for d in engine.documents.values()]
        seen, hooks = {}, []
        for li, layer in enumerate(engine.layers):
            for name, module in (("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj),
                                 ("v", layer.self_attn.v_proj), ("mlp", layer.mlp)):
                seen[li, name] = []
                def observe(module, args, key=(li, name)):
                    seen[key].append(args[0].shape[1])
                hooks.append(module.register_forward_pre_hook(observe))
        try:
            recovered = engine._recover_online(prefix, docs)
        finally:
            for hook in hooks:
                hook.remove()
        stats = engine._last_recovery_stats
        selected = stats["selected_token_rows"]
        assert 0 < selected < 18
        expected = [18]*2+[selected]*4
        assert stats["projected_rows_per_layer"] == stats["ffn_rows_per_layer"] == expected
        assert math.isclose(stats["row_execution_fraction"], sum(expected)/(18*6))
        for li in range(6):
            for name in ("q", "k", "v", "mlp"):
                assert seen[li, name] == [expected[li]]
        assert [part[0][0].shape[2] for part in recovered] == [3, 11, 7]
    finally:
        engine.close()


@pytest.mark.parametrize("rope_type", ["default", "yarn"])
def test_block_attention_absolute_causality_matches_independent_flash_oracle(runtime, rope_type):
    cfg = runtime.selective.SelectiveRecoveryConfig(block_size=3, row_fraction=.5)
    engine = _make(runtime, cfg, rope_type=rope_type)
    try:
        _register(engine)
        prefix = engine._segment("prefix", [1, 2, 3, 4, 5], engine._prefill_ids([1, 2, 3, 4, 5]))
        docs = [engine._segment(engine.documents[name].doc_id, engine.documents[name].token_ids,
                                engine.documents[name].recovery_kv) for name in ("E", "D", "E")]
        expected = engine._recover_online(prefix, docs)
        selected = engine._last_recovery_stats["selected_block_ids"]
        engine.backend = "flash"
        engine._flash_dense = _dense_flash_oracle
        actual = engine._recover_online(prefix, docs)
        assert selected == engine._last_recovery_stats["selected_block_ids"]
        for part_a, part_b in zip(expected, actual):
            for layer_a, layer_b in zip(part_a, part_b):
                for tensor_a, tensor_b in zip(layer_a, layer_b):
                    torch.testing.assert_close(tensor_a, tensor_b, rtol=5e-5, atol=5e-6)
    finally:
        engine.backend = "torch"
        engine.close()


@pytest.mark.parametrize("rope_type", ["default", "yarn"])
def test_unit_offset_rotation_matches_native_absolute_rope(runtime, rope_type):
    engine = _make(runtime, rope_type=rope_type)
    try:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        value = torch.randn((1, 2, 7, 16))
        hidden = torch.zeros((1, 7, 64))
        rotary = engine.model.model.rotary_emb
        canonical_cos, canonical_sin = rotary(hidden, torch.arange(7)[None])
        global_cos, global_sin = rotary(hidden, torch.arange(11, 18)[None])
        _, canonical = apply_rotary_pos_emb(value, value, canonical_cos, canonical_sin)
        _, expected = apply_rotary_pos_emb(value, value, global_cos, global_sin)
        actual = runtime.selective.rotate_canonical_keys(canonical, 11, engine._inv_freq)
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
        torch.testing.assert_close(actual.square().sum(-1), canonical.square().sum(-1), rtol=5e-5, atol=5e-6)
    finally:
        engine.close()


def test_zero_prefix_repeated_documents_keep_sources_immutable_and_globals_private(runtime):
    cfg = runtime.selective.SelectiveRecoveryConfig(block_size=2, row_fraction=.5)
    engine = _make(runtime, cfg)
    try:
        _register(engine)
        snapshots = {name: [(k.clone(), v.clone()) for k, v in doc.recovery_kv]
                     for name, doc in engine.documents.items()}
        before = engine.stats()["offline_kv_cache_bytes"]
        a = engine.admit("A", [], ["D"], max_suffix_tokens=8)
        b = engine.admit("B", [1, 2, 3], ["E", "D", "D"], max_suffix_tokens=8)
        av, bv = engine.manager.version(a.manager_request_id), engine.manager.version(b.manager_request_id)
        a_global = {p for key, s in av.segments.items() if key[2] == "global" for p in s.pages}
        b_global = {p for key, s in bv.segments.items() if key[2] == "global" for p in s.pages}
        assert not a_global.intersection(b_global)
        for version in (av, bv):
            for key, segment in version.segments.items():
                if key[2] == "global":
                    assert segment.reuse_kind == "policy_approximate"
                    assert segment.validity_certificate.startswith("block-selective-v1:")
        source = engine.manager.version(engine.documents["D"].source_request_id)
        for (l, h), key in engine.documents["D"].source_keys.items():
            assert av.segments[l, h, "document:0"].pages == source.segments[key].pages
            assert bv.segments[l, h, "document:1"].pages == bv.segments[l, h, "document:2"].pages
        assert torch.isfinite(engine.forward(["A", "B"], torch.tensor([[81, 82], [91, 92]]))).all()
        engine.release("A"); engine.release("B")
        assert engine.stats()["offline_kv_cache_bytes"] == before
        for name, doc in engine.documents.items():
            for now, old in zip(doc.recovery_kv, snapshots[name]):
                for tensor, expected in zip(now, old):
                    torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    finally:
        engine.close()
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
    assert engine.pool.stats()["pinned_pages"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_selected_block_gate_with_real_flash_and_immutable_cow(runtime, dtype):
    """GPU qualification, distinct from pretrained quality and QPS gates."""
    pytest.importorskip("flash_attn")
    cfg = runtime.selective.SelectiveRecoveryConfig(block_size=4, row_fraction=.5)
    engine = _make(runtime, cfg, device="cuda", dtype=dtype, backend="flash", rope_type="yarn")
    try:
        _register(engine)
        snapshots = {name: [(k.clone(), v.clone()) for k, v in doc.recovery_kv]
                     for name, doc in engine.documents.items()}
        # Same arbitrary absolute query interval through both real FA2 and an
        # independent literal causal mask. A tail-alignment bug fails here.
        generator = torch.Generator(device="cuda").manual_seed(703)
        q = torch.randn((1, 4, 3, 16), generator=generator, device="cuda", dtype=dtype)
        k = torch.randn((1, 2, 11, 16), generator=generator, device="cuda", dtype=dtype)
        v = torch.randn(k.shape, generator=generator, device="cuda", dtype=dtype)
        pk = torch.randn((1, 2, 5, 16), generator=generator, device="cuda", dtype=dtype)
        pv = torch.randn(pk.shape, generator=generator, device="cuda", dtype=dtype)
        positions = torch.arange(13, 16, device="cuda")
        actual = runtime.selective._attention(engine, q, k, v, pk, pv, 2, positions)
        engine.backend = "torch"
        expected = runtime.selective._attention(engine, q, k, v, pk, pv, 2, positions)
        engine.backend = "flash"
        tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        a = engine.admit("A", [], ["D"], max_suffix_tokens=8)
        b = engine.admit("B", [1, 2, 3], ["E", "D", "D"], max_suffix_tokens=8)
        for state in (a, b):
            stats = state.admission_stats
            assert stats["recovery_mode"] == "experimental_block_selective_v1"
            assert stats["row_execution_fraction"] < 1
            assert stats["projected_rows_per_layer"] == stats["ffn_rows_per_layer"]
        for inputs in ([[81, 82], [91, 92]], [[83], [93]]):
            assert torch.isfinite(engine.forward(["A", "B"], torch.tensor(inputs, device="cuda"))).all()
        engine.release("A"); engine.release("B")
        engine.synchronize()
        for name, document in engine.documents.items():
            for current, frozen in zip(document.recovery_kv, snapshots[name]):
                for actual_tensor, expected_tensor in zip(current, frozen):
                    torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    finally:
        engine.backend = "flash"
        engine.close()
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
    assert engine.pool.stats()["pinned_pages"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_full_row_fallback_matches_original_bitwise(runtime, dtype):
    pytest.importorskip("flash_attn")
    outputs = []
    for config in (None, runtime.selective.SelectiveRecoveryConfig(
            row_fraction=1, max_offline_kv_bytes=0)):
        engine = _make(runtime, config, device="cuda", dtype=dtype, backend="flash")
        try:
            _register(engine)
            state = engine.admit("r", [1, 2, 3], ["E", "D"], max_suffix_tokens=8)
            assert state.admission_stats["recovery_mode"] == "full_rows"
            assert engine.stats()["offline_kv_cache_bytes"] == 0
            outputs.append(engine.forward(["r"], torch.tensor([[81, 82]], device="cuda")).clone())
        finally:
            engine.close()
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("profile", ["global", "local", "local_no_sink", "mixed", "different_windows"])
@pytest.mark.parametrize("prefix", [0, 3, 13])
def test_prepared_layout_matches_frozen_per_run_kernel_inputs_and_absolute_mask(runtime, profile, prefix):
    """Independent frozen implementation plus explicit absolute causal oracle.

    Compare actual kernel tensors/kwargs/order, not merely a close final
    value. Nonadjacent runs have different widths and a partial final block.
    """
    engine, (all_q, k, v, pk, pv) = _fixture(runtime, profile, prefix, 33)
    intervals = ((1, 4), (8, 11), (17, 22), (30, 33))
    row_ids = torch.cat([torch.arange(lo, hi) for lo, hi in intervals])
    q = all_q.index_select(2, row_ids)
    engine.backend = "flash"
    engine.layers = [SimpleNamespace(self_attn=SimpleNamespace(scaling=1/8))]
    frozen = [x.clone() for x in (q, k, v, pk, pv)]
    calls = []
    def observed(query, keys, values, **kwargs):
        assert query.is_contiguous() and keys.is_contiguous() and values.is_contiguous()
        calls.append((query.clone(), keys.clone(), values.clone(), kwargs))
        return _dense_flash_oracle(query, keys, values, **kwargs)
    engine._flash_dense = observed
    expected, cursor = [], 0
    for lo, hi in intervals:
        width = hi-lo
        expected.append(_old_recovery(engine, q[:, :, cursor:cursor+width],
            k[:, :, :hi], v[:, :, :hi], pk, pv, 0, 1/8))
        cursor += width
    previous, calls = calls, []
    actual = runtime.selective._prepared_block_attention(engine, q, k, v, pk, pv, 0, intervals)
    assert len(calls) == len(previous)
    for old, new in zip(previous, calls):
        assert old[3] == new[3]
        for a, b in zip(old[:3], new[:3]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(actual, torch.cat(expected, 2), rtol=0, atol=0)
    engine.backend = "torch"
    cursor = 0
    for lo, hi in intervals:
        width = hi-lo
        oracle = runtime.selective._attention(engine, q[:, :, cursor:cursor+width],
            k[:, :, :hi], v[:, :, :hi], pk, pv, 0, torch.arange(prefix+lo, prefix+hi))
        torch.testing.assert_close(actual[:, :, cursor:cursor+width], oracle, rtol=5e-5, atol=5e-6)
        cursor += width
    for old, current in zip(frozen, (q, k, v, pk, pv)):
        torch.testing.assert_close(old, current, rtol=0, atol=0)


@pytest.mark.parametrize("rope_type", ["default", "yarn"])
@pytest.mark.parametrize("prefix", [(), (1, 2, 3, 4, 5)])
def test_request_rotation_cache_is_per_occurrence_and_bitwise_legacy(runtime, rope_type, prefix):
    engine = _make(runtime, runtime.selective.SelectiveRecoveryConfig(block_size=3), rope_type=rope_type)
    try:
        _register(engine)
        docs = [engine._segment(name, engine.documents[name].token_ids, engine.documents[name].recovery_kv)
                for name in ("E", "D", "E")]
        factors = runtime.selective.prepare_request_rotations(docs, len(prefix), engine._inv_freq)
        assert len(factors) == 3
        if not prefix:
            assert factors[0] is None
        # Repeated E has a different request offset, not the first E's factors.
        assert factors[0] is not factors[2]
        for layer in range(4):
            expected = runtime.selective.canonical_layer(docs, layer, len(prefix), engine._inv_freq)
            actual = runtime.selective.canonical_layer(docs, layer, len(prefix), engine._inv_freq, factors)
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                assert a.data_ptr() != b.data_ptr()
        with pytest.raises(ValueError, match="occurrences"):
            runtime.selective.canonical_layer(docs, 0, len(prefix), engine._inv_freq, factors[:1])
    finally:
        engine.close()


@pytest.mark.parametrize("rope_type", ["default", "yarn"])
@pytest.mark.parametrize("flash", [False, True])
def test_prepared_recovery_same_selection_rows_history_and_accounting(runtime, monkeypatch, rope_type, flash):
    config = runtime.selective.SelectiveRecoveryConfig(block_size=3, row_fraction=.5, probe_layers=2)
    engines = [_make(runtime, config, rope_type=rope_type, layers=6, optimization=x) for x in (False, True)]
    results, stats = [], []
    factor_calls = []
    original_factors = runtime.selective._rotation_factors
    def observed_factors(offset, inv_freq):
        factor_calls.append(offset)
        return original_factors(offset, inv_freq)
    monkeypatch.setattr(runtime.selective, "_rotation_factors", observed_factors)
    try:
        for engine in engines:
            _register(engine)
            prefix = engine._segment("prefix", [1, 2, 3], engine._prefill_ids([1, 2, 3]))
            docs = [engine._segment(name, engine.documents[name].token_ids, engine.documents[name].recovery_kv)
                    for name in ("E", "D", "E")]
            if flash:
                engine.backend, engine._flash_dense = "flash", _dense_flash_oracle
            start = len(factor_calls)
            results.append(engine._recover_online(prefix, docs))
            stats.append(engine._last_recovery_stats)
            assert len(factor_calls)-start == stats[-1]["execution_counters"]["offset_rotation_factor_builds"]
            assert engine.stats()["selective_execution_optimization"] == engine.selective_execution_optimization
        for part_a, part_b in zip(*results):
            for layer_a, layer_b in zip(part_a, part_b):
                for a, b in zip(layer_a, layer_b):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key in ("selected_block_ids", "selected_intervals", "projected_rows_per_layer", "ffn_rows_per_layer"):
            assert stats[0][key] == stats[1][key]
        old, new = (s["execution_counters"] for s in stats)
        assert old["offset_rotation_factor_builds"] == 15 and new["offset_rotation_factor_builds"] == 3
        assert old["selected_rope_gathers"] == 8 and new["selected_rope_gathers"] == 2
        assert old["prepared_attention_layers"] == 0
        assert new["prepared_attention_layers"] == (4 if flash else 0)
        legacy_rows = 2*28*2 + 4*sum(3+hi for lo, hi in stats[0]["selected_intervals"])*2
        assert old["attention_kv_materialized_token_head_rows"] == legacy_rows
        assert new["attention_kv_materialized_token_head_rows"] == (6*28*2 if flash else legacy_rows)
        assert new["attention_kv_materialized_token_head_rows"] <= old["attention_kv_materialized_token_head_rows"]
    finally:
        for engine in engines:
            engine.backend = "torch"
            engine.close()


def test_execution_switch_validates_boolean_and_never_changes_full_row_dispatch(runtime, monkeypatch):
    with pytest.raises(ValueError, match="boolean"):
        _make(runtime, optimization=1)
    def forbidden(*args, **kwargs):
        raise AssertionError("full-row dispatch must not prepare selective execution")
    monkeypatch.setattr(runtime.selective, "prepare_request_rotations", forbidden)
    monkeypatch.setattr(runtime.selective, "_prepared_block_attention", forbidden)
    outputs = []
    for optimization in (False, True):
        engine = _make(runtime, runtime.selective.SelectiveRecoveryConfig(row_fraction=1, max_offline_kv_bytes=0),
                       optimization=optimization)
        try:
            _register(engine)
            state = engine.admit("r", [1, 2, 3], ["D", "E"], max_suffix_tokens=8)
            assert state.admission_stats["recovery_mode"] == "full_rows"
            assert engine.stats()["offline_kv_cache_bytes"] == 0
            outputs.append(engine.forward(["r"], torch.tensor([[81, 82]])).clone())
        finally:
            engine.close()
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_prepared_recovery_matches_legacy_bitwise(runtime, dtype):
    """v2 system gate: real kernel input/output parity, not a quality claim."""
    pytest.importorskip("flash_attn")
    cfg = runtime.selective.SelectiveRecoveryConfig(block_size=3, row_fraction=.5)
    outputs, histories, selections = [], [], []
    for optimization in (False, True):
        engine = _make(runtime, cfg, device="cuda", dtype=dtype, backend="flash",
                       rope_type="yarn", layers=6, optimization=optimization)
        try:
            _register(engine)
            snapshots = {name: [(k.clone(), v.clone()) for k, v in document.recovery_kv]
                         for name, document in engine.documents.items()}
            # Direct noncontiguous runs include mixed local/global heads and
            # partial boundary blocks; compare with the unchanged legacy calls.
            generator = torch.Generator(device="cuda").manual_seed(711)
            def tensor(heads, tokens):
                return torch.randn((1, heads, tokens, 16), generator=generator, device="cuda", dtype=dtype)
            q, k, v, pk, pv = tensor(4, 7), tensor(2, 19), tensor(2, 19), tensor(2, 5), tensor(2, 5)
            intervals = ((1, 3), (8, 10), (16, 19))
            pieces, cursor = [], 0
            for lo, hi in intervals:
                pieces.append(engine._recovery_flash_attention(q[:, :, cursor:cursor+hi-lo],
                    k[:, :, :hi], v[:, :, :hi], pk, pv, 4, engine.layers[4].self_attn.scaling))
                cursor += hi-lo
            prepared = runtime.selective._prepared_block_attention(engine, q, k, v, pk, pv, 4, intervals)
            torch.testing.assert_close(prepared, torch.cat(pieces, 2), rtol=0, atol=0)
            prefix = engine._segment("prefix", [1, 2, 3], engine._prefill_ids([1, 2, 3]))
            docs = [engine._segment(name, engine.documents[name].token_ids, engine.documents[name].recovery_kv)
                    for name in ("E", "D", "E")]
            recovered = engine._recover_online(prefix, docs)
            histories.append([[tuple(x.clone() for x in layer) for layer in part] for part in recovered])
            selections.append(engine._last_recovery_stats["selected_block_ids"])
            a = engine.admit("A", [], ["D"], max_suffix_tokens=8)
            b = engine.admit("B", [1, 2, 3], ["E", "D", "E"], max_suffix_tokens=8)
            current = []
            for inputs in ([[81, 82], [91, 92]], [[83], [93]]):
                current.append(engine.forward(["A", "B"], torch.tensor(inputs, device="cuda")).clone())
            outputs.append(current)
            for state in (a, b):
                counts = state.admission_stats["execution_counters"]
                assert counts["prepared_attention_layers"] == (4 if optimization else 0)
            engine.release("A"); engine.release("B")
            engine.synchronize()
            for name, document in engine.documents.items():
                for current, frozen in zip(document.recovery_kv, snapshots[name]):
                    for now, old in zip(current, frozen):
                        torch.testing.assert_close(now, old, rtol=0, atol=0)
        finally:
            engine.close()
        assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
        assert engine.pool.stats()["pinned_pages"] == 0
    assert selections[0] == selections[1]
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(*histories):
        for la, lb in zip(a, b):
            for x, y in zip(la, lb):
                torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_query_mode_is_explicit_and_default_config_schema_is_unchanged(runtime):
    cfg = runtime.selective.SelectiveRecoveryConfig()
    assert set(cfg.to_dict()) == {"block_size", "row_fraction", "probe_layers", "max_offline_kv_bytes"}
    assert runtime.selective.SelectiveRecoveryConfig(selection_mode="query_attention").to_dict()["selection_mode"] == "query_attention"
    with pytest.raises(ValueError, match="selection_mode"):
        runtime.selective.SelectiveRecoveryConfig(selection_mode="answer_attention")


@pytest.mark.parametrize("heads", [(0,), (0, 1)])
@pytest.mark.parametrize("query_tile,key_tile", [(1, 2), (3, 7), (16, 2048)])
def test_query_relevance_tiled_softmax_matches_independent_dense_formula(runtime, heads, query_tile, key_tile):
    generator = torch.Generator().manual_seed(741)
    q = torch.randn((1, 4, 7, 16), generator=generator)
    k = torch.randn((1, 2, 23, 16), generator=generator)
    expected = []
    for h in heads:
        scores = q[:, h*2:(h+1)*2].float() @ k[:, h:h+1].float().transpose(-1, -2) * .25
        expected.append(scores.softmax(-1).amax(2)[0])
    expected = torch.cat(expected).mean(0)
    actual, counts = runtime.selective.conditional_query_relevance(q, k, heads, 2, .25,
        query_tile=query_tile, key_tile=key_tile)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
    assert counts["query_relevance_score_evaluations"] == 2*len(heads)*2*7*23
    assert counts["query_relevance_max_score_tile_elements"] == 2*min(7, query_tile)*min(23, key_tile)
    assert "document-only" in counts["query_relevance_normalization"]


def test_query_relevance_changes_blocks_without_reading_answer_or_local_query_heads(runtime):
    cfg = runtime.selective.SelectiveRecoveryConfig(block_size=2, row_fraction=.5, selection_mode="query_attention")
    keys = torch.zeros((1, 2, 8, 4))
    keys[:, 0, :2, 0] = 1
    keys[:, 0, 2:4, 1] = 1
    first, second = torch.zeros((1, 4, 2, 4)), torch.zeros((1, 4, 2, 4))
    first[:, :2, :, 0] = 10
    second[:, :2, :, 1] = 10
    first[:, 2:] = 1000  # Excluded local-head query values cannot dominate.
    second[:, 2:] = -1000
    a, _ = runtime.selective.conditional_query_relevance(first, keys, (0,), 2, 1.)
    b, _ = runtime.selective.conditional_query_relevance(second, keys, (0,), 2, 1.)
    assert runtime.selective.select_token_blocks(a, cfg)[0] == (0, 3)
    assert runtime.selective.select_token_blocks(b, cfg)[0] == (1, 3)


def _monolithic_query_probe_reference(runtime, engine, prefix, docs, query_ids, probes):
    """Independent causal full-token probe: query cannot affect earlier docs."""
    document_ids = torch.cat([s.token_ids for s in docs]).tolist()
    ids = document_ids+list(query_ids)
    positions = torch.arange(prefix.doc_len, prefix.doc_len+len(ids), device=engine.device)[None]
    hidden = engine.model.model.embed_tokens(torch.tensor([ids], device=engine.device))
    embeddings = engine.model.model.rotary_emb(hidden, positions)
    doc_kv = []
    original_backend = engine.backend
    try:
        engine.backend = "torch"  # Literal absolute mask, not prepared FA2.
        for li in range(probes):
            layer = engine.layers[li]
            q, k, v = engine._project(layer.self_attn, layer.input_layernorm(hidden), embeddings)
            doc_kv.append((k[:, :, :len(document_ids)].clone(), v[:, :, :len(document_ids)].clone()))
            if li == probes-1:
                heads = [h for h, p in enumerate(engine.policies[li]) if p.head_type != "local"]
                heads = heads or list(range(engine.kv_heads))
                relevance = []
                for h in heads:
                    qq = q[:, h*engine.groups:(h+1)*engine.groups, len(document_ids):].float()
                    score = qq @ k[:, h:h+1, :len(document_ids)].float().transpose(-1, -2) * layer.self_attn.scaling
                    relevance.append(score.softmax(-1).amax(2)[0])
                return doc_kv, torch.cat(relevance).mean(0), q[:, :, len(document_ids):].clone()
            pk, pv = prefix.kv[li]
            output = runtime.selective._attention(engine, q, k, v, pk, pv, li, positions[0])
            hidden = hidden+layer.self_attn.o_proj(output.transpose(1, 2).reshape(*hidden.shape[:-1], -1))
            hidden = hidden+layer.mlp(layer.post_attention_layernorm(hidden))
    finally:
        engine.backend = original_backend


@pytest.mark.parametrize("rope_type", ["default", "yarn"])
def test_query_probe_matches_monolithic_causal_reference_and_actual_hooks(runtime, monkeypatch, rope_type):
    cfg = runtime.selective.SelectiveRecoveryConfig(block_size=3, row_fraction=.5, probe_layers=3,
                                                   selection_mode="query_attention")
    engine = _make(runtime, cfg, rope_type=rope_type)
    try:
        _register(engine)
        prefix = engine._segment("prefix", [1, 2, 3, 4, 5], engine._prefill_ids([1, 2, 3, 4, 5]))
        docs = [engine._segment(name, engine.documents[name].token_ids, engine.documents[name].recovery_kv)
                for name in ("E", "D", "E")]
        query = (81, 82, 83, 84, 85, 86, 87)
        reference, expected_scores, expected_query_q = _monolithic_query_probe_reference(runtime, engine, prefix, docs, query, 3)
        original, scored = runtime.selective.conditional_query_relevance, []
        def observed(*args, **kwargs):
            value = original(*args, **kwargs)
            scored.append((*value, args[0].clone()))
            return value
        monkeypatch.setattr(runtime.selective, "conditional_query_relevance", observed)
        seen, hooks = {}, []
        for li, layer in enumerate(engine.layers):
            for name, module in (("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj),
                                 ("v", layer.self_attn.v_proj), ("mlp", layer.mlp)):
                seen[li, name] = []
                def observe(module, args, key=(li, name)):
                    seen[key].append(args[0].shape[1])
                hooks.append(module.register_forward_pre_hook(observe))
        try:
            recovered = engine._recover_online(prefix, docs, query)
        finally:
            for hook in hooks:
                hook.remove()
        stats = engine._last_recovery_stats
        assert stats["recovery_mode"] == "experimental_query_selective_v3"
        assert stats["recovery_query_tokens"] == len(query)
        assert stats["recovery_query_sha256"] == runtime.selective.query_token_sha256(query)
        assert stats["query_projected_rows_per_layer"] == [7, 7, 7, 0]
        assert stats["query_ffn_rows_per_layer"] == [7, 7, 0, 0]
        assert scored[0][1]["query_relevance_scoring_kv_heads"] == [1]  # Layer 2's other head is local.
        torch.testing.assert_close(scored[0][0], expected_scores, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(scored[0][2], expected_query_q, rtol=2e-5, atol=2e-6)
        for li in range(3):
            for component in range(2):
                actual = torch.cat([part[li][component] for part in recovered[1:]], 2)
                torch.testing.assert_close(actual, reference[li][component], rtol=2e-5, atol=2e-6)
            for name in ("q", "k", "v"):
                assert seen[li, name] == [25, 7]
            assert seen[li, "mlp"] == ([7, 25] if li < 2 else [25])
        for name in ("q", "k", "v", "mlp"):
            assert seen[3, name] == [stats["selected_token_rows"]]
        # Changing the question may change selection, never earlier doc KV.
        other = engine._recover_online(prefix, docs, (91, 92, 93))
        for a, b in zip(recovered[1:], other[1:]):
            for li in range(3):
                for x, y in zip(a[li], b[li]):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
    finally:
        engine.close()


def test_query_admission_validation_provenance_and_chunked_forward_binding(runtime):
    cfg = runtime.selective.SelectiveRecoveryConfig(block_size=3, selection_mode="query_attention")
    engine = _make(runtime, cfg)
    try:
        _register(engine)
        assert engine.requires_recovery_query is True
        before = engine.pool.stats()
        for query, match in ((None, "actual query_ids"), ([], "invalid token"), ([True], "invalid token"),
                             (list(range(9)), "reserved suffix")):
            with pytest.raises(ValueError, match=match):
                engine.admit("bad", [1, 2, 3], ["D", "E"], max_suffix_tokens=8, query_ids=query)
            assert engine.pool.stats() == before and not engine.requests
        snapshots = {name: [(k.clone(), v.clone()) for k, v in doc.recovery_kv]
                     for name, doc in engine.documents.items()}
        states = [engine.admit(name, [1, 2, 3], ["E", "D", "E"], max_suffix_tokens=8, query_ids=query)
                  for name, query in (("A", [81, 82, 83]), ("B", [91, 92, 93]))]
        versions = [engine.manager.version(s.manager_request_id) for s in states]
        certificates = []
        for version in versions:
            certificates.append({s.validity_certificate for key, s in version.segments.items() if key[2] == "global"})
        assert len(certificates[0]) == len(certificates[1]) == 1
        assert certificates[0] != certificates[1]
        assert all(x.startswith("query-selective-v3:") for group in certificates for x in group)
        page_sets = [{p for key, s in version.segments.items() if key[2] == "global" for p in s.pages} for version in versions]
        assert not page_sets[0].intersection(page_sets[1])
        for (li, h), source in engine.documents["E"].source_keys.items():
            assert versions[0].segments[li, h, "document:0"].pages == versions[1].segments[li, h, "document:2"].pages
        with pytest.raises(ValueError, match="differ from recovery"):
            engine.forward(["A", "B"], torch.tensor([[99], [91]]))
        before_mismatch = engine.pool.stats()
        with pytest.raises(ValueError, match="differ from recovery"):
            engine.forward(["A", "B"], torch.tensor([[81], [99]]))
        assert engine.pool.stats() == before_mismatch  # Matching A was not partially written.
        assert [s.suffix_tokens for s in states] == [0, 0]
        for inputs in ([[81, 82], [91, 92]], [[83], [93]], [[84], [94]]):
            assert torch.isfinite(engine.forward(["A", "B"], torch.tensor(inputs))).all()
        engine.release("A"); engine.release("B")
        for name, document in engine.documents.items():
            for current, frozen in zip(document.recovery_kv, snapshots[name]):
                for now, old in zip(current, frozen):
                    torch.testing.assert_close(now, old, rtol=0, atol=0)
    finally:
        engine.close()
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages


@pytest.mark.parametrize("kind", ["fraction1", "all_blocks"])
def test_query_full_fallback_performs_no_query_work(runtime, monkeypatch, kind):
    cfg = runtime.selective.SelectiveRecoveryConfig(selection_mode="query_attention",
        row_fraction=1 if kind == "fraction1" else .5,
        block_size=256, max_offline_kv_bytes=0 if kind == "fraction1" else 8*1024**3)
    def forbidden(*args, **kwargs):
        raise AssertionError("full fallback must not run query scoring")
    monkeypatch.setattr(runtime.selective, "conditional_query_relevance", forbidden)
    outputs = []
    for config in (None, cfg):
        engine = _make(runtime, config)
        try:
            _register(engine)
            extra = {"query_ids": [81, 82]} if engine.requires_recovery_query else {}
            state = engine.admit("r", [1, 2, 3], ["D", "E"], max_suffix_tokens=8, **extra)
            assert state.admission_stats["recovery_mode"] == "full_rows"
            assert state.admission_stats.get("recovery_query_tokens", 0) == 0
            assert state.admission_stats.get("recovery_query_sha256") is None
            outputs.append(engine.forward(["r"], torch.tensor([[81, 82]])).clone())
        finally:
            engine.close()
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_query_selector_private_probe_and_flash_oracle(runtime, dtype):
    pytest.importorskip("flash_attn")
    cfg = runtime.selective.SelectiveRecoveryConfig(block_size=3, probe_layers=3,
                                                   selection_mode="query_attention")
    engine = _make(runtime, cfg, device="cuda", dtype=dtype, backend="flash", rope_type="yarn")
    try:
        _register(engine)
        snapshots = {name: [(k.clone(), v.clone()) for k, v in doc.recovery_kv]
                     for name, doc in engine.documents.items()}
        prefix = engine._segment("prefix", [1, 2, 3], engine._prefill_ids([1, 2, 3]))
        docs = [engine._segment(name, engine.documents[name].token_ids, engine.documents[name].recovery_kv)
                for name in ("E", "D", "E")]
        query = (81, 82, 83, 84)
        expected_probe, reference_scores, reference_query_q = _monolithic_query_probe_reference(runtime, engine, prefix, docs, query, 3)
        actual = engine._recover_online(prefix, docs, query)
        tolerance = 3e-2 if dtype == torch.bfloat16 else 5e-3
        for li in range(3):
            for component in range(2):
                current = torch.cat([part[li][component] for part in actual[1:]], 2)
                torch.testing.assert_close(current, expected_probe[li][component], rtol=tolerance, atol=tolerance)
        original_scorer, recorded = runtime.selective.conditional_query_relevance, []
        def capture(*args, **kwargs):
            value = original_scorer(*args, **kwargs)
            recorded.append((value[0].clone(), args[0].clone(), args[1].clone(), tuple(args[2])))
            return value
        runtime.selective.conditional_query_relevance = capture
        try:
            engine._recover_online(prefix, docs, query)
        finally:
            runtime.selective.conditional_query_relevance = original_scorer
        score_atol = 3e-4 if dtype == torch.bfloat16 else 3e-5
        torch.testing.assert_close(recorded[0][0], reference_scores, rtol=tolerance, atol=score_atol)
        torch.testing.assert_close(recorded[0][1], reference_query_q, rtol=tolerance, atol=tolerance)
        # Qualify two-pass tiling separately on exactly identical real CUDA
        # Q/K, without allowing earlier-layer BF16 differences to hide a
        # softmax-normalization error. Force both query and key tile splits.
        _, same_q, same_k, scoring_heads = recorded[0]
        dense_relevance = []
        scale = engine.layers[2].self_attn.scaling
        for h in scoring_heads:
            score = (same_q[:, h*engine.groups:(h+1)*engine.groups].float()
                     @ same_k[:, h:h+1].float().transpose(-1, -2)) * scale
            dense_relevance.append(score.softmax(-1).amax(2)[0])
        exact_tiled, tile_counts = runtime.selective.conditional_query_relevance(
            same_q, same_k, scoring_heads, engine.groups, scale, query_tile=2, key_tile=7)
        torch.testing.assert_close(exact_tiled, torch.cat(dense_relevance).mean(0), rtol=2e-6, atol=1e-7)
        assert tile_counts["query_relevance_max_score_tile_elements"] <= engine.groups*2*7
        other = engine._recover_online(prefix, docs, (91, 92, 93))
        for a, b in zip(actual[1:], other[1:]):
            for li in range(3):
                for x, y in zip(a[li], b[li]):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
        states = [engine.admit(name, [1, 2, 3], ["E", "D", "E"], max_suffix_tokens=8, query_ids=ids)
                  for name, ids in (("A", (81, 82, 83)), ("B", (91, 92, 93)))]
        for state in states:
            assert state.admission_stats["query_projected_rows_per_layer"] == [3, 3, 3, 0]
            assert state.admission_stats["query_ffn_rows_per_layer"] == [3, 3, 0, 0]
        for inputs in ([[81, 82], [91, 92]], [[83], [93]], [[84], [94]]):
            assert torch.isfinite(engine.forward(["A", "B"], torch.tensor(inputs, device="cuda"))).all()
        engine.release("A"); engine.release("B")
        engine.synchronize()
        for name, document in engine.documents.items():
            for current, frozen in zip(document.recovery_kv, snapshots[name]):
                for now, old in zip(current, frozen):
                    torch.testing.assert_close(now, old, rtol=0, atol=0)
    finally:
        engine.close()
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
    assert engine.pool.stats()["pinned_pages"] == 0
