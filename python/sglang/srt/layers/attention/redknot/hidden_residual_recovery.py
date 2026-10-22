"""Experimental hidden-residual transport; no task-accuracy guarantee.

Keep the full online residual after the probe. Omitted rows subsequently add
the independent-document offline residual increment, instead of resetting to
offline KV. Every layer still projects all K/V rows; only Q/attention/O/MLP
skip selected-out rows. Offline hidden inputs are groupwise int8 with FP32
scales. These are immutable, separately budgeted retained payloads, NOT free
workspace. Head-local COW and publication belong to the unchanged engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from .selective_recovery import (
    _attention, _prepared_block_attention, conditional_query_relevance,
    query_token_sha256, select_token_blocks,
)


GROUP_SIZE = 128


def estimate_hidden_residual_cache_bytes(model_config, document_token_lengths,
                                        probe_layers, group_size=GROUP_SIZE):
    """Exact retained int8 values + FP32 scale payload, no padding values.

    Each unique registered document is counted once. Page-slab local KV,
    model weights, native offline-build KV and online transient peak are
    separate. Layers [P,L) store layer INPUT residuals, not normalized h.
    """
    lengths = tuple(document_token_lengths)
    layers, width = model_config.num_hidden_layers, model_config.hidden_size
    if (any(type(n) is not int or n <= 0 for n in lengths)
            or type(layers) is not int or layers <= 0
            or type(width) is not int or width <= 0
            or type(probe_layers) is not int or not 0 < probe_layers <= layers
            or type(group_size) is not int or group_size <= 0):
        raise ValueError("invalid hidden-cache dimensions, probe layers or lengths")
    return sum(lengths)*(layers-probe_layers)*(width+math.ceil(width/group_size)*4)


@dataclass(frozen=True)
class QuantizedLayerInput:
    values: torch.Tensor  # [N,H] int8; no padding retained
    scales: torch.Tensor  # [N,ceil(H/G)] FP32
    group_size: int = GROUP_SIZE

    def dequantize(self, out=None):
        """FP32 multiplication directly into a fresh/private output slice.

        Aligned widths broadcast one scale per group without expanding the
        scales or first materializing an FP32 copy of all int8 values. Input
        conversion by mul is exact for int8 and preserves the old FP32
        multiply. Unaligned widths retain the old expanded-scale layout.
        No cache tensor is written; out must be private. Step-strided and
        transposed outputs are supported when nonoverlap is provable from
        their strides; exotic interleaved layouts fail closed. No CUDA
        scalar reads occur.
        """
        if (self.values.ndim != 2 or self.scales.ndim != 2
                or type(self.group_size) is not int or self.group_size <= 0):
            raise ValueError("invalid grouped hidden cache layout")
        tokens, width = self.values.shape
        if (self.values.dtype != torch.int8 or self.scales.dtype != torch.float32
                or self.values.device != self.scales.device
                or self.scales.shape != (tokens, math.ceil(width/self.group_size))):
            raise ValueError("invalid int8 hidden values/FP32 scale layout")
        if out is None:
            out = torch.empty((tokens, width), dtype=torch.float32, device=self.values.device)
        if (out.shape != self.values.shape or out.dtype != torch.float32
                or out.device != self.values.device):
            raise ValueError("dequantization output must be matching nonoverlapping FP32 storage")
        extent = 1
        for stride, size in sorted((stride, size) for size, stride in zip(out.shape, out.stride()) if size > 1):
            if stride < extent:
                raise ValueError("dequantization output has overlapping or unsupported strides")
            extent += (size-1)*stride
        source_storages = {self.values.untyped_storage().data_ptr(), self.scales.untyped_storage().data_ptr()}
        if out.untyped_storage().data_ptr() in source_storages:
            raise ValueError("dequantization output must not alias immutable cache storage")
        if width % self.group_size == 0:
            # Split the last dimension only: this view also supports strided
            # rows/channels without copying the immutable source or output.
            shape = (tokens, width//self.group_size, self.group_size)
            torch.mul(self.values.view(shape), self.scales[..., None], out=out.view(shape))
        else:
            scales = self.scales.repeat_interleave(self.group_size, dim=-1)[..., :width]
            torch.mul(self.values, scales, out=out)
        return out


@dataclass(frozen=True)
class HiddenResidualDocument:
    layers: tuple
    probe_layers: int
    document_tokens: int
    hidden_size: int

    def quantized_layer(self, index):
        if not self.probe_layers <= index < self.probe_layers+len(self.layers):
            raise ValueError("offline hidden input layer outside retained range")
        return self.layers[index-self.probe_layers]

    def layer(self, index, out=None):
        return self.quantized_layer(index).dequantize(out=out)

    def unique_storage_bytes(self):
        storages = {}
        for layer in self.layers:
            for value in (layer.values, layer.scales):
                storage = value.untyped_storage()
                storages[str(value.device), storage.data_ptr()] = storage.nbytes()
        return sum(storages.values())


@torch.no_grad()
def quantize_layer_input(hidden, group_size=GROUP_SIZE):
    """Per-token/channel-group symmetric int8; FP32 scale and arithmetic.

    Zero groups use scale=1. Rounding is round-to-nearest-even. Validation
    synchronizes once per offline chunk/layer; no such quantization runs on
    the online request path. Inputs are never mutated or retained as views.
    """
    if (hidden.ndim != 2 or not hidden.is_floating_point()
            or min(hidden.shape) <= 0 or type(group_size) is not int or group_size <= 0):
        raise ValueError("quantization requires nonempty floating [N,H] and positive group size")
    working = hidden.float()
    if not bool(torch.isfinite(working).all()):
        raise ValueError("offline hidden input must be finite in FP32")
    width, groups = hidden.shape[-1], math.ceil(hidden.shape[-1]/group_size)
    grouped = F.pad(working, (0, groups*group_size-width)).reshape(hidden.shape[0], groups, group_size)
    maximum = grouped.abs().amax(-1)
    scales = torch.where(maximum == 0, torch.ones_like(maximum), maximum/127)
    # Subnormal FP32 groups may underflow maximum/127; use the smallest
    # normal scale in that case, preserving finiteness at negligible magnitude.
    scales = scales.clamp_min(torch.finfo(torch.float32).tiny)
    values = (grouped/scales[..., None]).round().clamp(-127, 127).to(torch.int8)
    return QuantizedLayerInput(values.reshape(hidden.shape[0], -1)[:, :width].clone(memory_format=torch.contiguous_format),
                               scales.contiguous(), group_size)


@torch.inference_mode()
def capture_offline_hidden(engine, ids, probe_layers):
    """Native position-zero prefill plus bounded chunk-wise pre-layer hooks.

    The returned native KV is transient and used only for local-page import;
    no full native KV is retained in HiddenResidualDocument. Hooks always
    detach, even on a prefill/capture error. Registration owns root rollback.
    """
    width, tokens = engine.model.config.hidden_size, len(ids)
    retained, cursors, hooks = [], [], []
    for _ in range(probe_layers, len(engine.layers)):
        retained.append(QuantizedLayerInput(
            torch.empty((tokens, width), dtype=torch.int8, device=engine.device),
            torch.empty((tokens, math.ceil(width/GROUP_SIZE)), dtype=torch.float32, device=engine.device)))
        cursors.append(0)
    try:
        for index in range(probe_layers, len(engine.layers)):
            slot = index-probe_layers
            def capture(module, args, kwargs, slot=slot):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != width:
                    raise ValueError("native offline hook requires B=1 layer-input residual")
                start, stop = cursors[slot], cursors[slot]+hidden.shape[1]
                if stop > tokens:
                    raise ValueError("native offline hook received too many token rows")
                quantized = quantize_layer_input(hidden[0])
                retained[slot].values[start:stop].copy_(quantized.values)
                retained[slot].scales[start:stop].copy_(quantized.scales)
                cursors[slot] = stop
            hooks.append(engine.layers[index].register_forward_pre_hook(capture, with_kwargs=True))
        kv = engine._prefill_ids(ids)
        if cursors != [tokens]*len(retained):
            raise ValueError("native offline hook did not capture every layer/token row")
        return kv, HiddenResidualDocument(tuple(retained), probe_layers, tokens, width)
    finally:
        for hook in hooks:
            hook.remove()


def materialize_offline_hidden(document_segments, layer):
    # One private occurrence-ordered allocation. Repeated document IDs share
    # immutable source cache, never request-dependent hidden or global KV.
    # Fill occurrence slices directly, avoiding per-document FP32 tensors
    # followed by another full-context concatenation/copy.
    cached = [segment.meta["recovery_hidden"].quantized_layer(layer) for segment in document_segments]
    if not cached:
        raise ValueError("hidden materialization requires document occurrences")
    width, device = cached[0].values.shape[1], cached[0].values.device
    if any(value.values.shape != (segment.doc_len, width) or value.values.device != device
           for segment, value in zip(document_segments, cached)):
        raise ValueError("hidden cache occurrence shapes/devices must agree")
    output = torch.empty((1, sum(segment.doc_len for segment in document_segments), width),
                         dtype=torch.float32, device=device)
    cursor = 0
    for segment, value in zip(document_segments, cached):
        stop = cursor+segment.doc_len
        value.dequantize(out=output[0, cursor:stop])
        cursor = stop
    return output


def transport_omitted_hidden(hidden, offline_current, offline_next, row_ids, selected_next):
    """FP32 increment, then cast; selected rows overwritten exactly afterward."""
    result = (hidden.float()+(offline_next-offline_current)).to(hidden.dtype)
    result.index_copy_(1, row_ids, selected_next)
    return result


def _apply_single_rotary(value, position_embeddings):
    """The exact native expression, without computing a discarded twin."""
    cos, sin = position_embeddings
    return (value*cos.unsqueeze(1))+(rotate_half(value)*sin.unsqueeze(1))


def _project_split(attention, normalized, row_ids, full_embeddings, selected_embeddings):
    selected = normalized.index_select(1, row_ids)
    shape = (*normalized.shape[:-1], -1, attention.head_dim)
    qshape = (*selected.shape[:-1], -1, attention.head_dim)
    q = attention.q_norm(attention.q_proj(selected).view(qshape)).transpose(1, 2)
    k = attention.k_norm(attention.k_proj(normalized).view(shape)).transpose(1, 2)
    v = attention.v_proj(normalized).view(shape).transpose(1, 2)
    # Q/K have different row counts. Each receives native absolute RoPE once;
    # this path never re-rotates an already positioned canonical K tensor.
    q = _apply_single_rotary(q, selected_embeddings)
    k = _apply_single_rotary(k, full_embeddings)
    return q, k, v


@torch.inference_mode()
def recover_hidden_residual(engine, prefix_segment, document_segments, config, query_ids):
    """Approximate full-hidden transport with actual split projection counts."""
    ids = torch.cat([s.token_ids for s in document_segments]).to(engine.device)[None]
    tokens, prefix_tokens = ids.shape[1], prefix_segment.doc_len
    positions = torch.arange(prefix_tokens, prefix_tokens+tokens, device=engine.device)[None]
    hidden = engine.model.model.embed_tokens(ids)
    embeddings = engine.model.model.rotary_emb(hidden, positions)
    query_tokens = len(query_ids)
    query_positions = torch.arange(prefix_tokens+tokens, prefix_tokens+tokens+query_tokens,
                                   device=engine.device)[None]
    query_hidden = engine.model.model.embed_tokens(torch.tensor([query_ids], dtype=torch.long, device=engine.device))
    query_embeddings = engine.model.model.rotary_emb(query_hidden, query_positions)
    layers = len(engine.layers)
    qrows, kvrows, arows, frows = [], [], [], []
    query_projected, query_attention, query_ffn = [0]*layers, [0]*layers, [0]*layers
    captured, offline_current = [], None
    selected = intervals = row_ids = selected_embeddings = None
    counters = dict(prepared_attention_layers=0, selected_rope_gathers=0,
                    offline_hidden_layer_materializations=0,
                    attention_kv_materialized_token_head_rows=0)
    for index, layer in enumerate(engine.layers):
        attn, is_probe = layer.self_attn, index < config.probe_layers
        normalized = layer.input_layernorm(hidden)
        if is_probe:
            q, k, v = engine._project(attn, normalized, embeddings)
            active_hidden = hidden
        else:
            q, k, v = _project_split(attn, normalized, row_ids, embeddings, selected_embeddings)
            active_hidden = hidden.index_select(1, row_ids)
        del normalized
        qrows.append(active_hidden.shape[1]); kvrows.append(tokens)
        arows.append(active_hidden.shape[1])
        captured.append((k, v))
        pk, pv = prefix_segment.kv[index]
        if is_probe:
            output = _attention(engine, q, k, v, pk, pv, index, positions[0])
            counters["attention_kv_materialized_token_head_rows"] += (prefix_tokens+tokens)*engine.kv_heads
            qq, qk, qv = engine._project(attn, layer.input_layernorm(query_hidden), query_embeddings)
            query_projected[index] = query_tokens
            if index < config.probe_layers-1:
                query_output = _attention(engine, qq, torch.cat((k, qk), 2), torch.cat((v, qv), 2),
                    pk, pv, index, query_positions[0])
                query_hidden = query_hidden+attn.o_proj(query_output.transpose(1, 2).reshape(*query_hidden.shape[:-1], -1))
                query_hidden = query_hidden+layer.mlp(layer.post_attention_layernorm(query_hidden))
                query_attention[index] = query_ffn[index] = query_tokens
                counters["attention_kv_materialized_token_head_rows"] += (prefix_tokens+tokens+query_tokens)*engine.kv_heads
                del query_output
            if index == config.probe_layers-1:
                heads = [h for h, policy in enumerate(engine.policies[index]) if policy.head_type != "local"]
                scores, diagnostics = conditional_query_relevance(qq, k, heads or list(range(engine.kv_heads)),
                                                                   engine.groups, attn.scaling)
                counters.update(diagnostics)
                selected, intervals = select_token_blocks(scores, config)
                row_ids = torch.cat([torch.arange(lo, hi, device=engine.device) for lo, hi in intervals])
                selected_embeddings = tuple(value.index_select(1, row_ids) for value in embeddings)
                counters["selected_rope_gathers"] = 2
                del query_hidden, query_embeddings, qq, qk, qv, scores
        elif engine.selective_execution_optimization and engine.backend == "flash":
            output = _prepared_block_attention(engine, q, k, v, pk, pv, index, intervals)
            counters["prepared_attention_layers"] += 1
            counters["attention_kv_materialized_token_head_rows"] += (prefix_tokens+tokens)*engine.kv_heads
        else:
            pieces, cursor = [], 0
            for lo, hi in intervals:
                width = hi-lo
                pieces.append(_attention(engine, q[:, :, cursor:cursor+width], k[:, :, :hi], v[:, :, :hi],
                                          pk, pv, index, positions[0, lo:hi]))
                counters["attention_kv_materialized_token_head_rows"] += (prefix_tokens+hi)*engine.kv_heads
                cursor += width
            output = torch.cat(pieces, dim=2)
        active_hidden = active_hidden+attn.o_proj(output.transpose(1, 2).reshape(*active_hidden.shape[:-1], -1))
        active_hidden = active_hidden+layer.mlp(layer.post_attention_layernorm(active_hidden))
        frows.append(active_hidden.shape[1])
        if is_probe:
            hidden = active_hidden  # Crucially retain ALL of h_P.
        elif index < layers-1:
            if offline_current is None:
                offline_current = materialize_offline_hidden(document_segments, index)
                counters["offline_hidden_layer_materializations"] += 1
            offline_next = materialize_offline_hidden(document_segments, index+1)
            counters["offline_hidden_layer_materializations"] += 1
            hidden = transport_omitted_hidden(hidden, offline_current, offline_next, row_ids, active_hidden)
            offline_current = offline_next
    result, cursor = [prefix_segment.kv], 0
    for segment in document_segments:
        stop = cursor+segment.doc_len
        result.append([(k[:, :, cursor:stop], v[:, :, cursor:stop]) for k, v in captured])
        cursor = stop
    return result, dict(
        recovery_mode="experimental_hidden_residual_int8_v4", recovery_approximate=True,
        selection_mode="query_attention", selected_token_rows=row_ids.numel(),
        selected_block_count=len(selected), selected_block_ids=list(selected),
        selected_intervals=[list(run) for run in intervals],
        q_rows_per_layer=qrows, kv_rows_per_layer=kvrows, attention_rows_per_layer=arows,
        ffn_rows_per_layer=frows, projected_rows_per_layer=None, row_execution_fraction=None,
        projection_accounting="Q/attention/O/FFN selected after probe; K/V and input norm full rows; no unified QKV saving",
        probe_score_layer=config.probe_layers-1,
        selector="block-mean conditional query-to-document relevance; mandatory final block",
        recovery_query_tokens=query_tokens, recovery_query_sha256=query_token_sha256(query_ids),
        query_projected_rows_per_layer=query_projected, query_attention_rows_per_layer=query_attention,
        query_ffn_rows_per_layer=query_ffn, full_row_fallback_reason=None,
        global_kv_storage="all rows projected from request-private live/transported hidden; no cross-request global alias",
        selective_execution_optimization=engine.selective_execution_optimization,
        recovery_selected_runs=len(intervals), execution_counters=counters,
        hidden_transport="unselected live residual plus consecutive offline int8-dequantized residual difference; FP32 arithmetic then model dtype",
        execution_workspace_scope="full request hidden; current/next FP32 dequantized offline hidden and correction temporary; captured full KV; one layer prepared KV; query probe; excludes retained cache; peak measured externally")
