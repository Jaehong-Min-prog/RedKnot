"""Hidden transport numerical/lifetime tests, not pretrained task evidence."""
import importlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

from test_nonprefix_paged import _isolated_imports, _tiny_model
from test_selective_recovery import _make, _register


@pytest.fixture(scope="module")
def runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with _isolated_imports() as modules:
            modules.selective = importlib.import_module("sglang.srt.layers.attention.redknot.selective_recovery")
            modules.hidden = importlib.import_module("sglang.srt.layers.attention.redknot.hidden_residual_recovery")
            yield modules
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def strict_cuda_fp32_reference(record_property):
    """The FP32 diagnostic must not silently inherit TF32 matmul settings."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        record_property("fp32_reference_tf32_enabled", "false")
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _config(runtime, **kwargs):
    settings = dict(
        block_size=3, row_fraction=.5, probe_layers=2,
        selection_mode="query_attention", recovery_mode="hidden_residual_int8")
    settings.update(kwargs)
    return runtime.selective.SelectiveRecoveryConfig(**settings)


def _context(engine, *, prefix_ids=(1, 2, 3), names=("E", "D", "E")):
    prefix = engine._segment("prefix", prefix_ids, engine._prefill_ids(prefix_ids))
    docs = [engine._segment(name, engine.documents[name].token_ids,
        engine.documents[name].recovery_kv, engine.documents[name].recovery_hidden) for name in names]
    return prefix, docs


def _legacy_dequantize(cache):
    """Frozen v4c numerical/layout reference, not the optimized method."""
    expanded = cache.scales.repeat_interleave(cache.group_size, dim=-1)[..., :cache.values.shape[-1]]
    return cache.values.float()*expanded


def _legacy_materialize_offline_hidden(docs, layer):
    return torch.cat([_legacy_dequantize(segment.meta["recovery_hidden"].layers[
        layer-segment.meta["recovery_hidden"].probe_layers]) for segment in docs], 0)[None]


def _legacy_project_split(attention, normalized, row_ids, full_embeddings, selected_embeddings):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    selected = normalized.index_select(1, row_ids)
    shape = (*normalized.shape[:-1], -1, attention.head_dim)
    qshape = (*selected.shape[:-1], -1, attention.head_dim)
    q = attention.q_norm(attention.q_proj(selected).view(qshape)).transpose(1, 2)
    k = attention.k_norm(attention.k_proj(normalized).view(shape)).transpose(1, 2)
    v = attention.v_proj(normalized).view(shape).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, q, *selected_embeddings)
    _, k = apply_rotary_pos_emb(k, k, *full_embeddings)
    return q, k, v


def _dequant_system_cases(runtime, device):
    aligned, fallback = [[1, 128], [3, 256], [3, 5120]], [[1, 64], [3, 129]]
    for tokens, width in aligned+fallback:
        values = ((torch.arange(tokens*width, device=device).reshape(tokens, width) % 255)-127).to(torch.int8)
        groups = math.ceil(width/128)
        scales = torch.linspace(0, .75, tokens*groups, device=device).reshape(tokens, groups)
        cache = runtime.hidden.QuantizedLayerInput(values, scales)
        frozen = values.clone(), scales.clone()
        # Noncontiguous output proves out= writes to the supplied occurrence,
        # not a hidden reshape copy. Odd columns are canaries.
        storage = torch.full((tokens+2, width*2), -777., device=device)
        output = storage[1:-1, ::2]
        result = cache.dequantize(out=output)
        assert result is output
        torch.testing.assert_close(result, _legacy_dequantize(cache), rtol=0, atol=0)
        assert (storage[0] == -777.).all() and (storage[-1] == -777.).all()
        assert (storage[:, 1::2] == -777.).all()
        assert torch.equal(values, frozen[0]) and torch.equal(scales, frozen[1])
    return aligned, fallback


def _single_rope_system_cases(runtime, device, dtype):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    for rope in ("default", "yarn"):
        model = _tiny_model(device=device, dtype=dtype, num_layers=1, rope_type=rope)
        hidden = torch.empty((1, 7, 64), device=device, dtype=dtype)
        embeddings = model.model.rotary_emb(hidden, torch.arange(5, 12, device=device)[None])
        value = torch.linspace(-2, 2, 7*4*16, device=device, dtype=dtype).reshape(1, 7, 4, 16).transpose(1, 2)
        assert not value.is_contiguous()
        frozen, factors = value.clone(), tuple(x.clone() for x in embeddings)
        expected, _ = apply_rotary_pos_emb(value, value, *embeddings)
        actual = runtime.hidden._apply_single_rotary(value, embeddings)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(value, frozen, rtol=0, atol=0)
        for original, snapshot in zip(embeddings, factors):
            torch.testing.assert_close(original, snapshot, rtol=0, atol=0)


@torch.inference_mode()
def _system_recovery_parity(runtime, monkeypatch, engine, prefix, docs):
    snapshots = {name: [(layer.values.clone(), layer.scales.clone()) for layer in document.recovery_hidden.layers]
                 for name, document in engine.documents.items()}
    budget = (engine.stats()["offline_hidden_cache_bytes"], engine.stats()["offline_hidden_cache_limit_bytes"])
    actual = _dense_history(engine._recover_online(prefix, docs, (81, 82, 83)))
    selection = engine._last_recovery_stats["selected_block_ids"]
    with monkeypatch.context() as patch:
        patch.setattr(runtime.hidden, "materialize_offline_hidden", _legacy_materialize_offline_hidden)
        patch.setattr(runtime.hidden, "_project_split", _legacy_project_split)
        reference = _dense_history(engine._recover_online(prefix, docs, (81, 82, 83)))
    selection_exact = selection == engine._last_recovery_stats["selected_block_ids"]
    full_history_exact = all(torch.equal(value, expected) for pair, target in zip(actual, reference)
                             for value, expected in zip(pair, target))
    try:
        fast = engine.admit("system-fast", [1, 2, 3], ["E", "D", "E"], max_suffix_tokens=8, query_ids=(81, 82, 83))
        with monkeypatch.context() as patch:
            patch.setattr(runtime.hidden, "materialize_offline_hidden", _legacy_materialize_offline_hidden)
            patch.setattr(runtime.hidden, "_project_split", _legacy_project_split)
            old = engine.admit("system-old", [1, 2, 3], ["E", "D", "E"], max_suffix_tokens=8, query_ids=(81, 82, 83))
        selection_exact &= fast.admission_stats["selected_block_ids"] == old.admission_stats["selected_block_ids"]
        query = torch.tensor([[81, 82, 83], [81, 82, 83]], dtype=torch.long, device=engine.device)
        logits = engine.forward(["system-fast", "system-old"], query)
        logits_exact = torch.equal(logits[0], logits[1])
        for _ in range(3):
            logits = engine.forward(["system-fast", "system-old"], logits[:, -1].argmax(-1)[:, None])
            logits_exact &= torch.equal(logits[0], logits[1])
    finally:
        for request_id in ("system-fast", "system-old"):
            if request_id in engine.requests:
                engine.release(request_id)
    cache_budget_exact = budget == (engine.stats()["offline_hidden_cache_bytes"], engine.stats()["offline_hidden_cache_limit_bytes"])
    source_immutable = all(torch.equal(value, expected)
        for name, document in engine.documents.items()
        for layer, snapshot in zip(document.recovery_hidden.layers, snapshots[name])
        for value, expected in zip((layer.values, layer.scales), snapshot))
    return dict(selection_exact=bool(selection_exact), full_history_exact=full_history_exact,
                logits_exact=bool(logits_exact), cache_budget_exact=cache_budget_exact, source_immutable=source_immutable)


def test_dequant_out_frozen_v4c_parity_and_private_slices(runtime):
    assert _dequant_system_cases(runtime, "cpu") == (
        [[1, 128], [3, 256], [3, 5120]], [[1, 64], [3, 129]])


@pytest.mark.parametrize("width", [128, 256, 129])
@pytest.mark.parametrize("layout", ["steps", "transposed"])
def test_dequant_strided_sources_and_outputs_are_bitexact(runtime, width, layout):
    tokens, groups = 3, math.ceil(width/128)
    if layout == "steps":
        values = torch.arange(tokens*width*4).remainder(255).sub(127).to(torch.int8).reshape(tokens*2, width*2)[::2, ::2]
        scales = torch.linspace(0, .75, tokens*groups*4).reshape(tokens*2, groups*2)[::2, ::2]
        output = torch.empty((tokens*2, width*2))[::2, ::2]
    else:
        values = torch.arange(tokens*width).remainder(255).sub(127).to(torch.int8).reshape(width, tokens).t()
        scales = torch.linspace(0, .75, tokens*groups).reshape(groups, tokens).t()
        output = torch.empty((width, tokens)).t()
    cache = runtime.hidden.QuantizedLayerInput(values, scales)
    snapshots = values.clone(), scales.clone()
    assert cache.dequantize(out=output) is output
    torch.testing.assert_close(output, _legacy_dequantize(cache), rtol=0, atol=0)
    assert torch.equal(values, snapshots[0]) and torch.equal(scales, snapshots[1])


@pytest.mark.parametrize("invalid", ["shape", "dtype", "expanded", "overlap", "values_alias", "scales_alias"])
def test_dequant_invalid_or_alias_output_fails_before_writes(runtime, invalid):
    values = torch.full((3, 128), 7, dtype=torch.int8)
    scales = torch.ones((3, 1))
    if invalid == "shape": output = torch.full((3, 127), -77.)
    elif invalid == "dtype": output = torch.full((3, 128), -77., dtype=torch.float64)
    elif invalid == "expanded": output = torch.full((1, 128), -77.).expand(3, -1)
    elif invalid == "overlap": output = torch.full((130,), -77.).as_strided((3, 128), (1, 1))
    elif invalid == "values_alias":
        storage = torch.full((3, 512), 7, dtype=torch.int8)
        values, output = storage[:, ::4], storage.view(torch.float32)
    else:
        storage = torch.full((3, 128), -77.)
        scales, output = storage[:, :1], storage
    snapshots = values.clone(), scales.clone(), output.clone()
    with pytest.raises(ValueError, match="output"):
        runtime.hidden.QuantizedLayerInput(values, scales).dequantize(out=output)
    for actual, expected in zip((values, scales, output), snapshots):
        assert torch.equal(actual, expected)


def test_occurrence_materialization_matches_frozen_cat_and_writes_single_output(runtime, monkeypatch):
    cache = runtime.hidden.quantize_layer_input(torch.arange(3*256).float().reshape(3, 256)-255)
    other = runtime.hidden.quantize_layer_input(torch.arange(2*256).float().reshape(2, 256)+23)
    def segment(layer):
        doc = runtime.hidden.HiddenResidualDocument((layer,), 2, layer.values.shape[0], 256)
        return SimpleNamespace(doc_len=doc.document_tokens, meta={"recovery_hidden": doc})
    docs = [segment(cache), segment(other), segment(cache)]
    snapshots = [(item.values.clone(), item.scales.clone()) for item in (cache, other)]
    original, outputs = runtime.hidden.QuantizedLayerInput.dequantize, []
    def observe(self, out=None):
        assert out is not None
        outputs.append(out)
        return original(self, out=out)
    with monkeypatch.context() as patch:
        patch.setattr(runtime.hidden.QuantizedLayerInput, "dequantize", observe)
        actual = runtime.hidden.materialize_offline_hidden(docs, 2)
    assert len(outputs) == 3
    assert all(out.untyped_storage().data_ptr() == actual.untyped_storage().data_ptr() for out in outputs)
    assert [out.storage_offset() for out in outputs] == [0, 3*256, 5*256]
    torch.testing.assert_close(actual, _legacy_materialize_offline_hidden(docs, 2), rtol=0, atol=0)
    repeated = actual[:, 5:].clone()
    actual[:, :3].zero_()
    assert torch.equal(actual[:, 5:], repeated)
    for item, frozen in zip((cache, other), snapshots):
        assert torch.equal(item.values, frozen[0]) and torch.equal(item.scales, frozen[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_single_rope_is_native_bitexact_on_strided_inputs(runtime, dtype):
    _single_rope_system_cases(runtime, "cpu", dtype)


@pytest.mark.parametrize("rope", ["default", "yarn"])
def test_system_fastpath_keeps_selection_history_logits_and_cache_bitexact(runtime, monkeypatch, rope):
    engine = _make(runtime, _config(runtime), layers=6, rope_type=rope)
    try:
        _register(engine)
        prefix, docs = _context(engine)
        flags = _system_recovery_parity(runtime, monkeypatch, engine, prefix, docs)
        assert all(flags.values()), flags
    finally:
        engine.close()
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
    assert engine.stats()["offline_hidden_cache_bytes"] == 0


def _history(parts):
    return [tuple(torch.cat([part[layer][i] for part in parts[1:]], dim=2)
                  for i in (0, 1)) for layer in range(len(parts[0]))]


def _dense_history(parts):
    return [tuple(pair) for pair in _history(parts)]


@torch.inference_mode()
def _raw_layer_inputs(engine, ids):
    chunks, hooks = [[] for _ in engine.layers], []
    try:
        for layer, module in enumerate(engine.layers):
            def observe(module, args, kwargs, layer=layer):
                chunks[layer].append((args[0] if args else kwargs["hidden_states"]).clone())
            hooks.append(module.register_forward_pre_hook(observe, with_kwargs=True))
        engine._prefill_ids(ids)
    finally:
        for handle in hooks:
            handle.remove()
    return [torch.cat(items, dim=1) for items in chunks]


def _literal_attention(engine, query, keys, values, query_positions, prefix_tokens, layer,
                       *, precision="fp32"):
    """Independent absolute mask, with explicit numerical reference choices.

    fp32 preserves the original gate oracle. probability_dtype isolates the
    softmax-probability cast only. hf_eager mirrors Transformers 4.57.1
    modeling_qwen3.py:145-152 (QK in model dtype, FP32 softmax, P cast, PV in
    model dtype). These alternatives are diagnostic, not threshold changes.
    """
    key_positions = torch.arange(keys.shape[2], device=query.device)
    output = torch.empty_like(query)
    for h, policy in enumerate(engine.policies[layer]):
        mask = key_positions[None] <= query_positions[:, None]
        if policy.head_type == "local":
            mask &= ((key_positions[None] >= query_positions[:, None]-policy.window)
                     | (key_positions[None] < min(prefix_tokens, policy.sink_size)))
        head_query = query[:, h*engine.groups:(h+1)*engine.groups]
        head_keys, head_values = keys[:, h:h+1], values[:, h:h+1]
        if precision == "hf_eager":
            score = head_query @ head_keys.transpose(-1, -2)
        else:
            score = head_query.float() @ head_keys.float().transpose(-1, -2)
        probabilities = (score*engine.layers[layer].self_attn.scaling).masked_fill(
            ~mask, -torch.inf).softmax(-1, dtype=torch.float32)
        if precision in ("hf_eager", "probability_dtype"):
            result = probabilities.to(query.dtype) @ head_values
        elif precision == "fp32":
            result = probabilities @ head_values.float()
        else:
            raise ValueError("unknown diagnostic attention precision")
        output[:, h*engine.groups:(h+1)*engine.groups] = result.to(query.dtype)
    return output


@torch.inference_mode()
def _literal_recovery(engine, prefix, docs, selected, *, offline=None,
                      precision="fp32", subtract_first=False):
    """Full-Q projection + explicit per-head absolute causal softmax oracle.

    This deliberately does not use split projection, prepared/run attention,
    or transport helpers. It computes extra work and is never a serving path.
    The selection is frozen from the actual query probe for isolation.
    """
    ids = torch.cat([segment.token_ids for segment in docs]).to(engine.device)[None]
    positions = torch.arange(prefix.doc_len, prefix.doc_len+ids.shape[1], device=engine.device)[None]
    hidden = engine.model.model.embed_tokens(ids)
    embeddings = engine.model.model.rotary_emb(hidden, positions)
    row_ids = torch.tensor(selected, dtype=torch.long, device=engine.device)
    result = []
    for li, layer in enumerate(engine.layers):
        probe = li < engine.selective_recovery.probe_layers
        q, k, v = engine._project(layer.self_attn, layer.input_layernorm(hidden), embeddings)
        rows = torch.arange(ids.shape[1], device=engine.device) if probe else row_ids
        query = q.index_select(2, rows)
        keys, values = torch.cat((prefix.kv[li][0], k), 2), torch.cat((prefix.kv[li][1], v), 2)
        output = _literal_attention(engine, query, keys, values, positions[0].index_select(0, rows),
                                    prefix.doc_len, li, precision=precision)
        active = hidden.index_select(1, rows)+layer.self_attn.o_proj(output.transpose(1, 2).reshape(1, len(rows), -1))
        active = active+layer.mlp(layer.post_attention_layernorm(active))
        result.append((k, v))
        if probe:
            hidden = active
        elif li < len(engine.layers)-1:
            if offline is None:
                current = torch.cat([s.meta["recovery_hidden"].layer(li) for s in docs], 0)[None]
                following = torch.cat([s.meta["recovery_hidden"].layer(li+1) for s in docs], 0)[None]
            else:
                current, following = offline[li], offline[li+1]
            updated = hidden.clone()
            # Scalar token loop is an intentionally independent oracle.
            for row in set(range(ids.shape[1]))-set(selected):
                updated[:, row] = ((hidden[:, row].float()+(following[:, row].float()-current[:, row].float()))
                    if subtract_first else
                    (hidden[:, row].float()+following[:, row].float()-current[:, row].float())).to(hidden.dtype)
            updated.index_copy_(1, row_ids, active)
            hidden = updated
    return result


def _tensor_errors(actual, expected):
    difference = actual.float()-expected.float()
    return {"max_abs": float(difference.abs().max()),
            "relative_l2": float(difference.norm()/expected.float().norm().clamp_min(1e-12)),
            "bit_exact": bool(torch.equal(actual, expected))}


def _history_errors(actual, expected, selected):
    tokens = actual[0][0].shape[2]
    rows = []
    for li, (pair, reference) in enumerate(zip(actual, expected)):
        metrics = {"layer0": li}
        for label, indices in (("all", list(range(tokens))), ("selected", selected),
                               ("omitted", sorted(set(range(tokens))-set(selected)))):
            if not indices:
                metrics[label] = None
                continue
            index = torch.tensor(indices, dtype=torch.long, device=pair[0].device)
            metrics[label] = {name: _tensor_errors(value.index_select(2, index), target.index_select(2, index))
                             for name, value, target in zip(("K", "V"), pair, reference)}
        rows.append(metrics)
    return rows


@torch.inference_mode()
def _capture_hidden_attention(engine, prefix, docs, query_ids):
    """Observe actual request h/attention without changing math or selection."""
    tokens, query_tokens = sum(s.doc_len for s in docs), len(query_ids)
    trace, handles = [dict() for _ in engine.layers], []
    try:
        for li, layer in enumerate(engine.layers):
            def hidden_input(module, args, li=li):
                if args[0].shape[1] == tokens:
                    trace[li]["hidden"] = args[0].clone()
            def attention_input(module, args, li=li):
                # This tiny diagnostic uses 25 document rows and >3 selected
                # rows; the 3-token auxiliary query stream is unambiguous.
                if args[0].shape[1] != query_tokens:
                    trace[li]["attention_flat"] = args[0].clone()
            handles.append(layer.input_layernorm.register_forward_pre_hook(hidden_input))
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(attention_input))
        result = _dense_history(engine._recover_online(prefix, docs, query_ids))
    finally:
        for handle in handles:
            handle.remove()
    assert all(set(entry) == {"hidden", "attention_flat"} for entry in trace)
    return result, trace


@torch.inference_mode()
def _same_input_diagnostics(runtime, engine, prefix, docs, selected, trace):
    """Separate projection shape, attention kernel and transport association."""
    rows = torch.tensor(selected, dtype=torch.long, device=engine.device)
    tokens = sum(s.doc_len for s in docs)
    positions = torch.arange(prefix.doc_len, prefix.doc_len+tokens, device=engine.device)[None]
    result = []
    for li, entry in enumerate(trace):
        layer, hidden = engine.layers[li], entry["hidden"]
        embeddings = engine.model.model.rotary_emb(hidden, positions)
        normalized = layer.input_layernorm(hidden)
        qfull, kfull, vfull = engine._project(layer.self_attn, normalized, embeddings)
        active = torch.arange(tokens, device=engine.device) if li < engine.selective_recovery.probe_layers else rows
        selected_embeddings = tuple(value.index_select(1, active) for value in embeddings)
        if li < engine.selective_recovery.probe_layers:
            q, k, v = qfull, kfull, vfull
        else:
            q, k, v = runtime.hidden._project_split(layer.self_attn, normalized, active, embeddings, selected_embeddings)
        metrics = {"layer0": li, "q_split_vs_full_gather": _tensor_errors(q, qfull.index_select(2, active)),
                   "k_split_vs_full": _tensor_errors(k, kfull), "v_split_vs_full": _tensor_errors(v, vfull)}
        live_attention = entry["attention_flat"].view(1, len(active), engine.query_heads, engine.head_dim).transpose(1, 2)
        keys, values = torch.cat((prefix.kv[li][0], k), 2), torch.cat((prefix.kv[li][1], v), 2)
        metrics["attention_value_scale"] = float(values.float().abs().max())
        metrics["attention_checked_query_rows"] = len(active)
        metrics["attention_value_scale_scope"] = "all prefix+document V at this layer; no query-probe V"
        for precision in ("fp32", "probability_dtype", "hf_eager"):
            reference = _literal_attention(engine, q, keys, values, positions[0].index_select(0, active),
                                            prefix.doc_len, li, precision=precision)
            metrics["same_input_attention_vs_"+precision] = _tensor_errors(live_attention, reference)
        if li >= engine.selective_recovery.probe_layers and li < len(engine.layers)-1:
            current = torch.cat([s.meta["recovery_hidden"].layer(li) for s in docs], 0)[None]
            following = torch.cat([s.meta["recovery_hidden"].layer(li+1) for s in docs], 0)[None]
            # Isolate the unselected arithmetic; selected rows are excluded.
            omitted = torch.tensor(sorted(set(range(tokens))-set(selected)), device=engine.device)
            subtract = (hidden.float()+(following-current)).to(hidden.dtype).index_select(1, omitted)
            left = (hidden.float()+following-current).to(hidden.dtype).index_select(1, omitted)
            metrics["transport_association_subtract_vs_left"] = _tensor_errors(subtract, left)
            expected = _independent_next_hidden(engine, docs, li, rows, trace)
            actual = trace[li+1]["hidden"]
            metrics["transport_actual_vs_independent"] = _tensor_errors(actual, expected)
            metrics["transport_selected_next_vs_independent"] = _tensor_errors(
                actual.index_select(1, rows), expected.index_select(1, rows))
            metrics["transport_omitted_next_vs_independent"] = _tensor_errors(
                actual.index_select(1, omitted), expected.index_select(1, omitted))
        result.append(metrics)
    return result


@torch.inference_mode()
def _independent_next_hidden(engine, docs, layer_index, rows, trace, *, delta_sign=1):
    """Literal per-row omitted update and independent selected O/MLP update.

    Uses the actual preceding FlashAttention output to isolate transport;
    attention itself has a separate same-input FP32-mask hard gate.
    Does not invoke the serving transport helper or copy the next state.
    """
    layer, hidden = engine.layers[layer_index], trace[layer_index]["hidden"]
    current = torch.cat([segment.meta["recovery_hidden"].layer(layer_index) for segment in docs], 0)[None]
    following = torch.cat([segment.meta["recovery_hidden"].layer(layer_index+1) for segment in docs], 0)[None]
    active = hidden.index_select(1, rows)+layer.self_attn.o_proj(trace[layer_index]["attention_flat"])
    active = active+layer.mlp(layer.post_attention_layernorm(active))
    expected = hidden.clone()
    selected = set(rows.tolist())
    for row in range(hidden.shape[1]):
        if row not in selected:
            expected[:, row] = (hidden[:, row].float()+delta_sign*(following[:, row]-current[:, row])).to(hidden.dtype)
    # Independent scalar assignment verifies both the selected-row values and
    # their destination rows, instead of reusing serving index_copy_.
    for source, destination in enumerate(rows.tolist()):
        expected[:, destination] = active[:, source]
    return expected


def _attention_contract(dtype, diagnostics):
    """Predeclared one-dtype-epsilon finite-tiny regression rule, not a proof.

    The relative norm check and V-scaled absolute check must BOTH pass. A
    zero/near-zero reference is not skipped. Nonfinite inputs/errors fail.
    This is an operator-level regression budget, not task-quality tolerance.
    """
    epsilon = torch.finfo(dtype).eps
    layers = []
    for row in diagnostics:
        error, scale = row["same_input_attention_vs_fp32"], row["attention_value_scale"]
        limit = epsilon*scale
        values = (scale, error["max_abs"], error["relative_l2"], limit)
        passed = (all(math.isfinite(value) and value >= 0 for value in values)
                  and error["relative_l2"] <= epsilon and error["max_abs"] <= limit)
        layers.append(dict(layer0=row["layer0"], value_scale=scale,
            max_abs=error["max_abs"], relative_l2=error["relative_l2"], max_abs_limit=limit,
            relative_l2_limit=epsilon, passed=passed, checked_query_rows=row["attention_checked_query_rows"]))
    return dict(dtype=str(dtype).removeprefix("torch."), epsilon=epsilon,
        rule="relative_l2<=dtype_eps;max_abs<=dtype_eps*max_abs(V)",
        value_scale_scope="all prefix+document V at this layer; no query-probe V", layers=layers)


def _is_exact(error):
    return error["bit_exact"] is True and error["max_abs"] == 0 and error["relative_l2"] == 0


def _layerwise_exact_status(diagnostics, probes, layers):
    qkv = all(_is_exact(row[key]) for row in diagnostics
              for key in ("q_split_vs_full_gather", "k_split_vs_full", "v_split_vs_full"))
    transport_rows = [row for row in diagnostics if probes <= row["layer0"] < layers-1]
    transport = (len(transport_rows) == layers-probes-1 and
        all(_is_exact(row[key]) for row in transport_rows for key in (
            "transport_actual_vs_independent", "transport_selected_next_vs_independent",
            "transport_omitted_next_vs_independent")))
    return qkv, transport


@torch.inference_mode()
def _semantic_mutation_checks(runtime, engine, prefix, docs, selected, trace):
    """Four real-tensor corruptions must fail the corresponding exact check.

    No mutation is applied to an engine/cache. These are deliberately wrong
    candidate outputs tested against the unchanged independent references.
    """
    li = engine.selective_recovery.probe_layers
    hidden, layer = trace[li]["hidden"], engine.layers[li]
    rows = torch.tensor(selected, dtype=torch.long, device=engine.device)
    positions = torch.arange(prefix.doc_len, prefix.doc_len+hidden.shape[1], device=engine.device)[None]
    embeddings = engine.model.model.rotary_emb(hidden, positions)
    selected_embeddings = tuple(value.index_select(1, rows) for value in embeddings)
    normalized = layer.input_layernorm(hidden)
    qfull, kfull, _ = engine._project(layer.self_attn, normalized, embeddings)
    wrong_rows = rows.roll(1)
    wrong_q, _, _ = runtime.hidden._project_split(layer.self_attn, normalized, wrong_rows, embeddings, selected_embeddings)
    offline = torch.cat([segment.meta["recovery_hidden"].layer(li) for segment in docs], 0)[None].to(hidden.dtype)
    _, wrong_k, _ = engine._project(layer.self_attn, layer.input_layernorm(offline), embeddings)
    double_rotated = runtime.selective.rotate_canonical_keys(kfull, 1, engine._inv_freq)
    inverse_delta = _independent_next_hidden(engine, docs, li, rows, trace, delta_sign=-1)
    candidates = {"delta_sign": (inverse_delta, trace[li+1]["hidden"]),
                  "discard_online_hidden": (wrong_k, kfull),
                  "wrong_row_index": (wrong_q, qfull.index_select(2, rows)),
                  "double_rope": (double_rotated, kfull)}
    detected = {}
    for name, (bad, correct) in candidates.items():
        try:
            torch.testing.assert_close(bad, correct, rtol=0, atol=0)
        except AssertionError:
            detected[name] = True
        else:
            detected[name] = False
    return detected


@pytest.mark.parametrize("width", [1, 64, 128, 129, 257])
@pytest.mark.parametrize("tokens", [1, 3])
def test_quantization_keeps_no_padding_storage_and_zero_groups(runtime, width, tokens):
    value = torch.linspace(-2, 3, tokens*width).reshape(tokens, width)
    value[0] = 0
    frozen = value.clone()
    cache = runtime.hidden.quantize_layer_input(value)
    assert cache.values.dtype == torch.int8 and cache.scales.dtype == torch.float32
    assert cache.values.untyped_storage().nbytes() == tokens*width
    assert cache.scales.untyped_storage().nbytes() == tokens*math.ceil(width/128)*4
    assert torch.equal(value, frozen)
    assert not cache.values[0].any() and torch.equal(cache.scales[0], torch.ones_like(cache.scales[0]))
    decoded = cache.dequantize()
    maximum_error = cache.scales.repeat_interleave(128, -1)[:, :width]/2+1e-6
    assert ((decoded-value).abs() <= maximum_error).all()


def test_config_schema_and_hidden_estimator(runtime):
    default = runtime.selective.SelectiveRecoveryConfig().to_dict()
    assert set(default) == {"block_size", "row_fraction", "probe_layers", "max_offline_kv_bytes"}
    assert _config(runtime).to_dict()["recovery_mode"] == "hidden_residual_int8"
    with pytest.raises(ValueError, match="requires query_attention"):
        runtime.selective.SelectiveRecoveryConfig(recovery_mode="hidden_residual_int8")
    with pytest.raises(ValueError, match="recovery_mode"):
        runtime.selective.SelectiveRecoveryConfig(recovery_mode="unknown")
    cfg = SimpleNamespace(num_hidden_layers=64, hidden_size=5120)
    assert runtime.hidden.estimate_hidden_residual_cache_bytes(cfg, [97350], 8) == 28784448000
    assert runtime.hidden.estimate_hidden_residual_cache_bytes(cfg, [], 8) == 0
    for lengths, probes, group in [([0], 8, 128), ([2], 0, 128), ([2], 65, 128), ([2], 8, 0)]:
        with pytest.raises(ValueError):
            runtime.hidden.estimate_hidden_residual_cache_bytes(cfg, lengths, probes, group)


@pytest.mark.parametrize("case", ["nan", "inf", "overflow_to_fp32", "int", "rank", "empty", "bad_group"])
def test_quantization_rejects_invalid_inputs_without_mutating_them(runtime, case):
    value = torch.ones((2, 64))
    group = 128
    if case == "nan": value[0, 0] = float("nan")
    elif case == "inf": value[0, 0] = float("inf")
    elif case == "overflow_to_fp32": value = torch.full((2, 64), 1e100, dtype=torch.float64)
    elif case == "int": value = value.int()
    elif case == "rank": value = value[None]
    elif case == "empty": value = value[:0]
    elif case == "bad_group": group = True
    frozen = value.clone()
    with pytest.raises(ValueError):
        runtime.hidden.quantize_layer_input(value, group)
    torch.testing.assert_close(value, frozen, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("rope", ["default", "yarn"])
def test_chunked_native_hooks_capture_inputs_and_exact_retained_bytes(runtime, rope):
    engine = _make(runtime, _config(runtime), layers=6, rope_type=rope)
    engine.prefix_chunk_size = 3
    try:
        raw = _raw_layer_inputs(engine, list(range(20, 31)))
        before = [len(layer._forward_pre_hooks) for layer in engine.layers]
        _register(engine)
        assert [len(layer._forward_pre_hooks) for layer in engine.layers] == before
        expected = runtime.hidden.estimate_hidden_residual_cache_bytes(engine.model.config, [11, 7], 2)
        assert engine.stats()["offline_hidden_cache_bytes"] == expected
        assert engine.stats()["offline_hidden_cache_peak_bytes"] == expected
        assert engine.stats()["offline_kv_cache_bytes"] == engine.stats()["offline_kv_cache_limit_bytes"] == 0
        assert sum(doc.recovery_hidden.unique_storage_bytes() for doc in engine.documents.values()) == expected
        assert all(doc.recovery_kv == () and doc.recovery_nbytes == 0 for doc in engine.documents.values())
        for li in range(2, 6):
            quantized = runtime.hidden.quantize_layer_input(raw[li][0])
            actual = engine.documents["D"].recovery_hidden.layers[li-2]
            torch.testing.assert_close(actual.values, quantized.values, rtol=0, atol=0)
            torch.testing.assert_close(actual.scales, quantized.scales, rtol=0, atol=0)
    finally:
        engine.close()
    assert engine.stats()["offline_hidden_cache_bytes"] == 0


@pytest.mark.parametrize("failure", ["budget", "prefill", "quantize", "publish"])
def test_registration_failure_removes_hooks_roots_and_retained_cache(runtime, monkeypatch, failure):
    cfg = _config(runtime, max_offline_kv_bytes=1 if failure == "budget" else 2**20)
    engine = _make(runtime, cfg, layers=6)
    before = engine.pool.stats()
    hooks_before = [dict(layer._forward_pre_hooks) for layer in engine.layers]
    def fail(*args, **kwargs):
        raise RuntimeError("injected capture failure")
    try:
        if failure == "prefill": monkeypatch.setattr(engine, "_prefill_ids", fail)
        elif failure == "quantize": monkeypatch.setattr(runtime.hidden, "quantize_layer_input", fail)
        elif failure == "publish": monkeypatch.setattr(engine.manager, "update", fail)
        with pytest.raises((RuntimeError, runtime.core.CapacityError)):
            engine.register_document("D", list(range(20, 31)))
        assert engine.pool.stats() == before and not engine.documents
        assert engine.manager.stats()["requests"] == 0
        assert engine.stats()["offline_hidden_cache_bytes"] == engine.stats()["offline_hidden_cache_peak_bytes"] == 0
        assert [dict(layer._forward_pre_hooks) for layer in engine.layers] == hooks_before
    finally:
        engine.close()


def test_partial_capture_failure_preserves_external_hooks_and_prior_documents(runtime, monkeypatch):
    engine = _make(runtime, _config(runtime), layers=6)
    engine.prefix_chunk_size = 3
    external_calls = []
    external = engine.layers[2].register_forward_pre_hook(lambda *args: external_calls.append(True))
    try:
        engine.register_document("D", list(range(20, 31)))
        frozen = engine.stats()["offline_hidden_cache_bytes"]
        pages, roots = engine.pool.stats(), engine.manager.stats()["requests"]
        counts = [len(layer._forward_pre_hooks) for layer in engine.layers]
        calls, quantize = [], runtime.hidden.quantize_layer_input
        def fail_later(*args, **kwargs):
            calls.append(True)
            if len(calls) == 3: raise RuntimeError("partial offline capture")
            return quantize(*args, **kwargs)
        monkeypatch.setattr(runtime.hidden, "quantize_layer_input", fail_later)
        with pytest.raises(RuntimeError, match="partial offline capture"):
            engine.register_document("E", list(range(50, 57)))
        assert len(calls) == 3 and external_calls
        assert [len(layer._forward_pre_hooks) for layer in engine.layers] == counts
        assert list(engine.documents) == ["D"]
        assert engine.stats()["offline_hidden_cache_bytes"] == frozen
        assert engine.pool.stats() == pages and engine.manager.stats()["requests"] == roots
    finally:
        external.remove(); engine.close()


@pytest.mark.parametrize("fallback", ["fraction1", "all_probe"])
def test_hidden_fullrow_configuration_keeps_original_path_no_cache(runtime, fallback):
    config = _config(runtime, **({"row_fraction": 1} if fallback == "fraction1" else {"probe_layers": 6}))
    config = runtime.selective.SelectiveRecoveryConfig(**{**config.to_dict(), "max_offline_kv_bytes": 0})
    reference, candidate = _make(runtime, layers=6), _make(runtime, config, layers=6)
    try:
        outputs = []
        for engine in (reference, candidate):
            _register(engine)
            state = engine.admit("r", [1, 2, 3], ["D", "E"], max_suffix_tokens=8)
            outputs.append(engine.forward(["r"], torch.tensor([[81, 82]])))
            assert state.admission_stats["recovery_mode"] == "full_rows"
            assert not engine.requires_recovery_query
            assert engine.stats()["offline_hidden_cache_bytes"] == engine.stats()["offline_hidden_cache_limit_bytes"] == 0
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
    finally:
        reference.close(); candidate.close()


@pytest.mark.parametrize("rope", ["default", "yarn"])
def test_first_sparse_full_kv_exact_and_split_rows_match_real_hooks(runtime, rope):
    engine = _make(runtime, _config(runtime), layers=6, rope_type=rope)
    try:
        _register(engine)
        prefix, docs = _context(engine)
        full = _dense_history(engine._recover_online_full(prefix, docs))
        seen, hooks = {}, []
        for li, layer in enumerate(engine.layers):
            for name, module in (("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj),
                                 ("v", layer.self_attn.v_proj), ("mlp", layer.mlp), ("o", layer.self_attn.o_proj)):
                seen[li, name] = []
                def observe(module, args, key=(li, name)):
                    seen[key].append(args[0].shape[1])
                hooks.append(module.register_forward_pre_hook(observe))
        try:
            actual = _dense_history(engine._recover_online(prefix, docs, (81, 82, 83)))
        finally:
            for handle in hooks: handle.remove()
        stats = engine._last_recovery_stats
        count, tokens = stats["selected_token_rows"], 25
        assert 0 < count < tokens
        rows = [tokens]*2+[count]*4
        assert stats["q_rows_per_layer"] == stats["attention_rows_per_layer"] == stats["ffn_rows_per_layer"] == rows
        assert stats["kv_rows_per_layer"] == [tokens]*6
        assert stats["projected_rows_per_layer"] is stats["row_execution_fraction"] is None
        for li in range(6):
            for name in ("q", "k", "v"):
                assert seen[li, name] == ([tokens, 3] if li < 2 else [count if name == "q" else tokens])
            for name in ("o", "mlp"):
                assert seen[li, name] == ([3, tokens] if li == 0 else [rows[li]])
        for li in range(3):  # Includes first sparse layer's complete K/V.
            for value, expected in zip(actual[li], full[li]):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
        selected = [row for lo, hi in stats["selected_intervals"] for row in range(lo, hi)]
        literal = _literal_recovery(engine, prefix, docs, selected)
        for pair, expected in zip(actual, literal):
            for value, target in zip(pair, expected):
                torch.testing.assert_close(value, target, rtol=2e-5, atol=3e-6)
    finally:
        engine.close()


@pytest.mark.parametrize("rope", ["default", "yarn"])
def test_fixed_selection_old_reset_float_bridge_int8_bridge_diagnostic(runtime, monkeypatch, rope):
    config = _config(runtime)
    bridge = _make(runtime, config, layers=6, mixed=False, rope_type=rope)
    old_config = runtime.selective.SelectiveRecoveryConfig(**{**config.to_dict(), "recovery_mode": "offline_kv"})
    old = _make(runtime, old_config, layers=6, mixed=False, rope_type=rope)
    try:
        for engine in (bridge, old): _register(engine)
        prefix, docs = _context(bridge)
        old_prefix, old_docs = _context(old)
        full = _dense_history(bridge._recover_online_full(prefix, docs))
        reset = _dense_history(old._recover_online(old_prefix, old_docs, (81, 82, 83)))
        quantized = _dense_history(bridge._recover_online(prefix, docs, (81, 82, 83)))
        stats = bridge._last_recovery_stats
        assert stats["selected_block_ids"] == old._last_recovery_stats["selected_block_ids"]
        selected = [i for lo, hi in stats["selected_intervals"] for i in range(lo, hi)]
        raw = {name: _raw_layer_inputs(bridge, document.token_ids) for name, document in bridge.documents.items()}
        def unquantized(segments, li):
            return torch.cat([raw[segment.segment_id][li] for segment in segments], 1)
        with monkeypatch.context() as patch:
            patch.setattr(runtime.hidden, "materialize_offline_hidden", unquantized)
            floating = _dense_history(bridge._recover_online(prefix, docs, (81, 82, 83)))
        offline = [unquantized(docs, li) for li in range(6)]
        literal_float = _literal_recovery(bridge, prefix, docs, selected, offline=offline)
        omitted = torch.tensor(sorted(set(range(25))-set(selected)))
        for component in (0, 1):
            assert (reset[2][component].index_select(2, omitted)-full[2][component].index_select(2, omitted)).norm() > .01
            torch.testing.assert_close(quantized[2][component], full[2][component], rtol=0, atol=0)
            torch.testing.assert_close(floating[2][component], full[2][component], rtol=0, atol=0)
        for actual_pair, reference_pair in zip(floating, literal_float):
            for value, expected in zip(actual_pair, reference_pair):
                torch.testing.assert_close(value, expected, rtol=2e-5, atol=3e-6)
        # No assertion that later quality improves: this is a causal repair,
        # not a proof that a fixed hidden delta models all nonlinear layers.
        assert any(not torch.equal(a, b) for pair_a, pair_b in zip(quantized[3:], floating[3:])
                   for a, b in zip(pair_a, pair_b))
    finally:
        bridge.close(); old.close()


@pytest.mark.parametrize("rope", ["default", "yarn"])
def test_forced_allrows_new_math_matches_full_and_queries_do_not_change_probe_history(runtime, monkeypatch, rope):
    engine = _make(runtime, _config(runtime), layers=6, rope_type=rope)
    try:
        _register(engine)
        prefix, docs = _context(engine)
        full = _dense_history(engine._recover_online_full(prefix, docs))
        first = _dense_history(engine._recover_online(prefix, docs, (81, 82, 83)))
        second = _dense_history(engine._recover_online(prefix, docs, (91, 92)))
        for li in range(3):
            for value, expected in zip(first[li], second[li]):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
        def forbid(*args, **kwargs): raise AssertionError("fallback must not run")
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_recover_online_full", forbid)
            patch.setattr(runtime.hidden, "select_token_blocks", lambda scores, config:
                (tuple(range(math.ceil(25/config.block_size))), ((0, 25),)))
            actual = _dense_history(engine._recover_online(prefix, docs, (81, 82, 83)))
        assert engine._last_recovery_stats["recovery_mode"] == "experimental_hidden_residual_int8_v4"
        assert engine._last_recovery_stats["q_rows_per_layer"] == [25]*6
        for pair, reference in zip(actual, full):
            for value, expected in zip(pair, reference):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
    finally:
        engine.close()


def test_transport_preserves_selected_rows_and_does_not_modify_any_source(runtime):
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    before = hidden.clone()
    current, following = hidden*2, hidden*3
    current_before, next_before = current.clone(), following.clone()
    selected, live = torch.tensor([1, 4]), torch.full((1, 2, 4), -99.)
    actual = runtime.hidden.transport_omitted_hidden(hidden, current, following, selected, live)
    expected = hidden*2
    expected[:, selected] = -99.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden, before, rtol=0, atol=0)
    torch.testing.assert_close(current, current_before, rtol=0, atol=0)
    torch.testing.assert_close(following, next_before, rtol=0, atol=0)
    assert actual.untyped_storage().data_ptr() != hidden.untyped_storage().data_ptr()


def test_diagnostic_eager_oracle_matches_installed_transformers_primary_code(runtime):
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
    engine = _make(runtime, _config(runtime), layers=6, mixed=False)
    try:
        query = torch.randn(1, 4, 3, 16)
        key, value = torch.randn(1, 2, 7, 16), torch.randn(1, 2, 7, 16)
        positions = torch.tensor([2, 4, 6])
        allowed = torch.arange(7)[None] <= positions[:, None]
        mask = torch.zeros((1, 1, 3, 7)).masked_fill(~allowed, -torch.inf)
        reference, _ = eager_attention_forward(engine.layers[0].self_attn, query, key, value,
                                               mask, engine.layers[0].self_attn.scaling)
        result = _literal_attention(engine, query, key, value, positions, 2, 0, precision="hf_eager")
        torch.testing.assert_close(result.transpose(1, 2), reference, rtol=0, atol=0)
    finally:
        engine.close()


def test_same_input_diagnostic_hooks_preserve_actual_selection_and_math(runtime):
    engine = _make(runtime, _config(runtime), layers=6, rope_type="yarn")
    try:
        _register(engine)
        prefix, docs = _context(engine)
        reference = _dense_history(engine._recover_online(prefix, docs, (81, 82, 83)))
        first_selection = engine._last_recovery_stats["selected_block_ids"]
        actual, trace = _capture_hidden_attention(engine, prefix, docs, (81, 82, 83))
        assert first_selection == engine._last_recovery_stats["selected_block_ids"]
        for pair, expected in zip(actual, reference):
            for value, target in zip(pair, expected):
                torch.testing.assert_close(value, target, rtol=0, atol=0)
        selected = [row for lo, hi in engine._last_recovery_stats["selected_intervals"] for row in range(lo, hi)]
        diagnostics = _same_input_diagnostics(runtime, engine, prefix, docs, selected, trace)
        assert len(diagnostics) == 6
        for row in diagnostics:
            assert row["k_split_vs_full"]["bit_exact"] and row["v_split_vs_full"]["bit_exact"]
            assert row["same_input_attention_vs_fp32"]["max_abs"] < 1e-6
        _, transport = _layerwise_exact_status(diagnostics, 2, 6)
        assert transport
        assert all(_semantic_mutation_checks(runtime, engine, prefix, docs, selected, trace).values())
        literal = _literal_recovery(engine, prefix, docs, selected, precision="probability_dtype", subtract_first=True)
        for pair, expected in zip(actual, literal):
            for value, target in zip(pair, expected):
                torch.testing.assert_close(value, target, rtol=2e-5, atol=3e-6)
    finally:
        engine.close()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["zero", "valid", "absolute", "relative", "zero_scale", "nan", "inf"])
def test_layerwise_attention_rule_has_fixed_dtype_scale_and_rejects_invalid_values(dtype, case):
    epsilon = torch.finfo(dtype).eps
    scale, absolute, relative = 2., epsilon, epsilon/2
    if case == "zero": scale, absolute, relative = 0., 0., 0.
    elif case == "absolute": absolute = 3*epsilon
    elif case == "relative": relative = 2*epsilon
    elif case == "zero_scale": scale, absolute, relative = 0., epsilon/100, 0.
    elif case == "nan": relative = float("nan")
    elif case == "inf": scale = float("inf")
    diagnostic = dict(layer0=0, attention_value_scale=scale, attention_checked_query_rows=25,
                      same_input_attention_vs_fp32={"max_abs": absolute, "relative_l2": relative})
    contract = _attention_contract(dtype, [diagnostic])
    assert contract["epsilon"] == epsilon
    assert contract["layers"][0]["max_abs_limit"] == epsilon*scale
    assert contract["layers"][0]["passed"] is (case in ("zero", "valid"))


@pytest.mark.parametrize("mutation", ["delta_sign", "discard_online_hidden", "wrong_row_index", "double_rope"])
def test_layerwise_exact_checks_detect_semantic_negative_examples(runtime, mutation):
    engine = _make(runtime, _config(runtime), layers=6, rope_type="yarn")
    try:
        _register(engine)
        prefix, docs = _context(engine)
        _, trace = _capture_hidden_attention(engine, prefix, docs, (81, 82, 83))
        selected = [row for lo, hi in engine._last_recovery_stats["selected_intervals"] for row in range(lo, hi)]
        detected = _semantic_mutation_checks(runtime, engine, prefix, docs, selected, trace)
        assert detected[mutation]
        # Exercise the actual-next-state contract too: an incorrect omitted
        # row in recorded output must not pass merely because two algebraic
        # formulations happen to agree on the inputs.
        omitted = next(row for row in range(25) if row not in selected)
        trace[3]["hidden"] = trace[3]["hidden"].clone()
        trace[3]["hidden"][0, omitted, 0] += 1
        diagnostics = _same_input_diagnostics(runtime, engine, prefix, docs, selected, trace)
        _, transport = _layerwise_exact_status(diagnostics, 2, 6)
        assert not transport
    finally:
        engine.close()


def _assert_cow_and_cache_immutable(engine):
    snapshots = {name: [(layer.values.clone(), layer.scales.clone()) for layer in doc.recovery_hidden.layers]
                 for name, doc in engine.documents.items()}
    query_a, query_b = (81, 82), (91, 92)
    a = engine.admit("A", [], ["D"], max_suffix_tokens=8, query_ids=query_a)
    b = engine.admit("B", [1, 2, 3], ["E", "D", "D"], max_suffix_tokens=8, query_ids=query_b)
    av, bv = engine.manager.version(a.manager_request_id), engine.manager.version(b.manager_request_id)
    ag = {page for key, segment in av.segments.items() if key[2] == "global" for page in segment.pages}
    bg = {page for key, segment in bv.segments.items() if key[2] == "global" for page in segment.pages}
    assert ag and bg and not ag.intersection(bg)
    source = engine.manager.version(engine.documents["D"].source_request_id)
    for (li, head), key in engine.documents["D"].source_keys.items():
        assert av.segments[li, head, "document:0"].pages == source.segments[key].pages
        assert bv.segments[li, head, "document:1"].pages == bv.segments[li, head, "document:2"].pages
    for version in (av, bv):
        for key, segment in version.segments.items():
            if key[2] == "global": assert segment.validity_certificate.startswith("hidden-residual-int8-v4:")
    before = [a.suffix_tokens, b.suffix_tokens]
    with pytest.raises(ValueError, match="query"):
        engine.forward(["A", "B"], torch.tensor([query_a, (91, 90)], device=engine.device))
    assert [a.suffix_tokens, b.suffix_tokens] == before
    assert torch.isfinite(engine.forward(["A", "B"], torch.tensor([query_a, query_b], device=engine.device))).all()
    engine.release("A"); engine.release("B")
    for name, doc in engine.documents.items():
        assert doc.recovery_kv == ()
        for layer, frozen in zip(doc.recovery_hidden.layers, snapshots[name]):
            torch.testing.assert_close(layer.values, frozen[0], rtol=0, atol=0)
            torch.testing.assert_close(layer.scales, frozen[1], rtol=0, atol=0)


def test_repeated_occurrences_cow_query_binding_and_hidden_source_immutability(runtime):
    engine = _make(runtime, _config(runtime), layers=6)
    try:
        _register(engine)
        _assert_cow_and_cache_immutable(engine)
    finally:
        engine.close()
    assert engine.pool.stats()["pinned_pages"] == 0
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_hidden_residual_recovery_matches_literal_oracle(
        runtime, monkeypatch, record_property, strict_cuda_fp32_reference, dtype):
    pytest.importorskip("flash_attn")
    engine = _make(runtime, _config(runtime), layers=6, rope_type="yarn",
                   device="cuda", dtype=dtype, backend="flash")
    numerical_checks = []
    hard_failures = []
    record_property("numerical_oracle_policy", json.dumps("hidden-residual-layerwise-v1"))
    try:
        _register(engine)
        expected_bytes = runtime.hidden.estimate_hidden_residual_cache_bytes(engine.model.config, [11, 7], 2)
        assert engine.stats()["offline_hidden_cache_bytes"] == expected_bytes
        assert sum(d.recovery_hidden.unique_storage_bytes() for d in engine.documents.values()) == expected_bytes
        assert engine.stats()["offline_kv_cache_bytes"] == 0
        assert all(d.recovery_kv == () for d in engine.documents.values())
        prefix, docs = _context(engine)
        full = _dense_history(engine._recover_online_full(prefix, docs))
        actual, trace = _capture_hidden_attention(engine, prefix, docs, (81, 82, 83))
        stats = engine._last_recovery_stats
        assert stats["execution_counters"]["prepared_attention_layers"] == 4
        assert stats["kv_rows_per_layer"] == [25]*6
        aligned_shapes, fallback_shapes = _dequant_system_cases(runtime, "cuda")
        _single_rope_system_cases(runtime, "cuda", dtype)
        system_parity = _system_recovery_parity(runtime, monkeypatch, engine, prefix, docs)
        system_parity.update(aligned_dequant_shapes=aligned_shapes, fallback_dequant_shapes=fallback_shapes,
                             dequant_bit_exact=True, rope_bit_exact=True)
        record_property("system_optimization_policy", json.dumps("hidden-dequant-out-single-rope-v1"))
        record_property("system_optimization_parity", json.dumps(system_parity))
        if not all(value for key, value in system_parity.items() if key.endswith("exact") or key == "source_immutable"):
            hard_failures.append("frozen-v4c system-optimization parity: "+str(system_parity))
        first_exact = all(torch.equal(value, expected) for li in range(3)
                          for value, expected in zip(actual[li], full[li]))
        record_property("first_sparse_full_kv_exact", json.dumps(first_exact))
        for li in range(3):
            for value, expected in zip(actual[li], full[li]):
                numerical_checks.append((f"first_sparse_full_layer{li}", value, expected, 0, 0))
        selected = [row for lo, hi in stats["selected_intervals"] for row in range(lo, hi)]
        literal = _literal_recovery(engine, prefix, docs, selected)
        error_rows = _history_errors(actual, literal, selected)
        record_property("literal_kv_error_by_layer_and_row_group", json.dumps(error_rows, allow_nan=False))
        tolerance = 3e-2 if dtype == torch.bfloat16 else 5e-3
        cumulative_failures = []
        for li, (pair, reference) in enumerate(zip(actual, literal)):
            for component, value, expected in zip(("K", "V"), pair, reference):
                try:
                    torch.testing.assert_close(value, expected, rtol=tolerance, atol=tolerance)
                except AssertionError as error:
                    cumulative_failures.append(f"layer{li}_{component}: {error}")
        record_property("original_cumulative_literal_threshold_passed", json.dumps(not cumulative_failures))
        record_property("original_cumulative_literal_threshold", json.dumps({
            "rtol": tolerance, "atol": tolerance, "role": "diagnostic_only_under_hidden-residual-layerwise-v1"}))
        record_property("original_cumulative_literal_threshold_failures", json.dumps(cumulative_failures))
        # v4c policy change: cumulative FP32-vs-Flash drift remains visible,
        # including the prior failed cutoff. The same cutoff also rejected
        # full-row Flash in v4b, so hard numerical gates now compare identical
        # per-layer inputs with explicit dtype/value-scale budgets below.
        variants, baseline_floor = {}, {}
        for precision in ("fp32", "probability_dtype", "hf_eager"):
            reference = _literal_recovery(engine, prefix, docs, list(range(25)), precision=precision)
            baseline_floor[precision] = _history_errors(full, reference, selected)
            if precision != "fp32":
                reference = _literal_recovery(engine, prefix, docs, selected, precision=precision)
                variants[precision] = _history_errors(actual, reference, selected)
        associated = _literal_recovery(engine, prefix, docs, selected, subtract_first=True)
        variants["fp32_transport_subtract_first"] = _history_errors(actual, associated, selected)
        variants["transport_association_only"] = _history_errors(associated, literal, selected)
        record_property("diagnostic_fullrow_flash_vs_literal_floor", json.dumps(baseline_floor, allow_nan=False))
        record_property("diagnostic_literal_numeric_variants", json.dumps(variants, allow_nan=False))
        same_input = _same_input_diagnostics(runtime, engine, prefix, docs, selected, trace)
        record_property("diagnostic_same_actual_input_stages", json.dumps(same_input, allow_nan=False))
        qkv_exact, transport_exact = _layerwise_exact_status(same_input, 2, 6)
        record_property("same_input_qkv_exact", json.dumps(qkv_exact))
        record_property("same_input_transport_exact", json.dumps(transport_exact))
        contract = _attention_contract(dtype, same_input)
        attention_passed = all(layer["passed"] for layer in contract["layers"])
        record_property("same_input_attention_contract", json.dumps(contract, allow_nan=False))
        record_property("same_input_attention_checks_passed", json.dumps(attention_passed))
        mutations = _semantic_mutation_checks(runtime, engine, prefix, docs, selected, trace)
        record_property("semantic_mutation_detections", json.dumps(mutations))
        record_property("semantic_mutation_checks_passed", json.dumps(all(mutations.values())))
        for label, passed in (("same-input QKV exact", qkv_exact), ("same-input actual transport exact", transport_exact),
                              ("same-input attention dtype-scaled limits", attention_passed),
                              ("semantic mutations detected", all(mutations.values()))):
            if not passed: hard_failures.append(label)
        def forbid_fallback(*args, **kwargs):
            raise AssertionError("GPU all-row gate must exercise new recovery math")
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_recover_online_full", forbid_fallback)
            patch.setattr(runtime.hidden, "select_token_blocks", lambda scores, cfg:
                (tuple(range(math.ceil(25/cfg.block_size))), ((0, 25),)))
            allrows = _dense_history(engine._recover_online(prefix, docs, (81, 82, 83)))
        assert engine._last_recovery_stats["q_rows_per_layer"] == [25]*6
        forced_exact = all(torch.equal(value, expected) for pair, reference in zip(allrows, full)
                           for value, expected in zip(pair, reference))
        record_property("forced_all_rows_full_kv_exact", json.dumps(forced_exact))
        for li, (pair, reference) in enumerate(zip(allrows, full)):
            for component, value, expected in zip(("K", "V"), pair, reference):
                numerical_checks.append((f"forced_allrow_layer{li}_{component}", value, expected, 0, 0))
        _assert_cow_and_cache_immutable(engine)
        hook_counts = [len(layer._forward_pre_hooks) for layer in engine.layers]
        with monkeypatch.context() as patch:
            def fail(*args, **kwargs): raise RuntimeError("injected CUDA registration failure")
            patch.setattr(runtime.hidden, "quantize_layer_input", fail)
            with pytest.raises(RuntimeError, match="injected CUDA"):
                engine.register_document("failed", [11, 12, 13])
        assert [len(layer._forward_pre_hooks) for layer in engine.layers] == hook_counts
        assert "failed" not in engine.documents
        assert engine.stats()["offline_hidden_cache_bytes"] == expected_bytes
    finally:
        engine.close()
    assert engine.pool.stats()["pinned_pages"] == 0
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
    assert engine.stats()["offline_hidden_cache_bytes"] == 0
    record_property("lifecycle_checks_completed", "true")
    failures = list(hard_failures)
    for label, value, expected, rtol, atol in numerical_checks:
        try:
            torch.testing.assert_close(value, expected, rtol=rtol, atol=atol)
        except AssertionError as error:
            failures.append(label+": "+str(error))
    if failures:
        pytest.fail("\n\n".join(failures))
