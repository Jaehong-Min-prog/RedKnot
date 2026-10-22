"""Opt-in approximate block-selective recovery; no accuracy certificate.

Native position-zero document KV is immutable offline input. A small prefix
of layers executes the existing full-row recovery. Its last layer's K/V drift
or explicitly opt-in conditional query relevance ranks fixed token blocks.
Only those hidden rows continue through later QKV,
attention and MLP operations; other rows use offline KV, rotated into the
request coordinate system. Global history is always materialized privately.
This module does not publish aliases, change local-head COW, or skip request
prefix recovery. Row counts are execution diagnostics, not measured FLOPs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch


@dataclass(frozen=True)
class SelectiveRecoveryConfig:
    block_size: int = 256
    row_fraction: float = 0.5
    probe_layers: int = 2
    max_offline_kv_bytes: int = 8 * 1024**3
    selection_mode: str = "kv_drift"
    recovery_mode: str = "offline_kv"

    def __post_init__(self):
        for name in ("block_size", "probe_layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_offline_kv_bytes) is not int or self.max_offline_kv_bytes < 0:
            raise ValueError("max_offline_kv_bytes must be a nonnegative integer")
        if (isinstance(self.row_fraction, bool)
                or not isinstance(self.row_fraction, (int, float))
                or not math.isfinite(self.row_fraction)
                or not 0 < self.row_fraction <= 1):
            raise ValueError("row_fraction must be finite and in (0, 1]")
        if self.selection_mode not in ("kv_drift", "query_attention"):
            raise ValueError("selection_mode must be kv_drift or query_attention")
        if self.recovery_mode not in ("offline_kv", "hidden_residual_int8"):
            raise ValueError("recovery_mode must be offline_kv or hidden_residual_int8")
        if self.recovery_mode == "hidden_residual_int8" and self.selection_mode != "query_attention":
            raise ValueError("hidden_residual_int8 requires query_attention selection")

    def to_dict(self):
        result = asdict(self)
        # Preserve the exact frozen v1/v2 four-field default-policy schema.
        if self.selection_mode == "kv_drift":
            del result["selection_mode"]
        if self.recovery_mode == "offline_kv":
            del result["recovery_mode"]
        return result


def query_token_sha256(query_ids):
    return hashlib.sha256(json.dumps(list(query_ids), separators=(",", ":")).encode()).hexdigest()


def conditional_query_relevance(query, document_keys, head_indices, groups, scale,
                                *, query_tile=16, key_tile=2048):
    """FP32 doc-only softmax relevance, not native full-context attention mass.

    For each scoring query head: normalize across document keys only, take
    the maximum over query positions, then average heads. Two key-tiled
    passes compute the exact mathematical conditional normalization without
    an unbounded Q-by-N score matrix. Query/key tile sizes bound every score
    tensor; the accumulated head relevance is only O(GQA-group-size * N).
    """
    if (query.ndim != 4 or document_keys.ndim != 4 or query.shape[0] != 1
            or document_keys.shape[0] != 1 or not query.shape[2] or not document_keys.shape[2]
            or query.shape[1] != document_keys.shape[1]*groups
            or query.shape[-1] != document_keys.shape[-1]):
        raise ValueError("query relevance requires nonempty B=1 matching grouped Q/K")
    heads = tuple(head_indices)
    if (not heads or len(set(heads)) != len(heads)
            or any(type(h) is not int or not 0 <= h < document_keys.shape[1] for h in heads)
            or type(query_tile) is not int or query_tile <= 0
            or type(key_tile) is not int or key_tile <= 0):
        raise ValueError("invalid relevance heads or tile sizes")
    tokens, query_tokens = document_keys.shape[2], query.shape[2]
    relevance = torch.zeros(tokens, dtype=torch.float32, device=query.device)
    max_tile, evaluations = 0, 0
    for head in heads:
        maxima = torch.zeros((groups, tokens), dtype=torch.float32, device=query.device)
        for qlo in range(0, query_tokens, query_tile):
            qpart = query[:, head*groups:(head+1)*groups, qlo:qlo+query_tile].float()
            logz = torch.full((*qpart.shape[:-1], 1), -torch.inf, dtype=torch.float32, device=query.device)
            for klo in range(0, tokens, key_tile):
                keys = document_keys[:, head:head+1, klo:klo+key_tile].float()
                score = (qpart @ keys.transpose(-1, -2)) * scale
                logz = torch.logaddexp(logz, torch.logsumexp(score, dim=-1, keepdim=True))
                max_tile = max(max_tile, score.numel())
                evaluations += score.numel()
            for klo in range(0, tokens, key_tile):
                keys = document_keys[:, head:head+1, klo:klo+key_tile].float()
                score = (qpart @ keys.transpose(-1, -2)) * scale
                probability = torch.exp(score-logz)
                end = klo+keys.shape[2]
                maxima[:, klo:end] = torch.maximum(maxima[:, klo:end], probability.amax(2)[0])
                evaluations += score.numel()
        relevance += maxima.sum(0)
    return relevance/(len(heads)*groups), dict(
        query_relevance_score_evaluations=evaluations,
        query_relevance_max_score_tile_elements=max_tile,
        query_relevance_query_tile=query_tile, query_relevance_key_tile=key_tile,
        query_relevance_scoring_kv_heads=list(heads),
        query_relevance_normalization="conditional document-only softmax; max over query positions, mean global/dense query heads",
    )


def estimate_selective_cache_bytes(model_config, document_token_lengths, dtype):
    """Full canonical KV payload bytes, excluding the separately charged slab.

    Count each registered document ID once; repeated occurrences reuse it.
    This is a retained-payload estimate, not a CUDA transient/allocator peak.
    Engine registration additionally verifies actual unique storage bytes.
    """
    lengths = tuple(document_token_lengths)
    if any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError("document lengths must be positive integers")
    layers, heads = model_config.num_hidden_layers, model_config.num_key_value_heads
    dim = getattr(model_config, "head_dim", None)
    if dim is None:
        dim = model_config.hidden_size // model_config.num_attention_heads
    if any(type(n) is not int or n <= 0 for n in (layers, heads, dim)):
        raise ValueError("invalid model KV dimensions")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("canonical KV requires a floating-point dtype")
    return 2 * layers * heads * dim * sum(lengths) * torch.empty((), dtype=dtype).element_size()


def unique_storage_bytes(kv):
    storages = {}
    for pair in kv:
        for value in pair:
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def selected_block_budget(tokens, config):
    blocks = math.ceil(tokens / config.block_size)
    return blocks, max(1, math.ceil(blocks * config.row_fraction))


def select_token_blocks(token_scores, config):
    """Deterministic block-mean ranking with the final block always retained.

    Only one small vector of block scores crosses to the host. A tie uses the
    earlier block. Adjacent selected blocks coalesce into a single FA2 call.
    """
    if token_scores.ndim != 1 or not token_scores.numel():
        raise ValueError("token scores must be a nonempty vector")
    tokens = token_scores.numel()
    blocks, budget = selected_block_budget(tokens, config)
    scores = [token_scores[i:i+config.block_size].float().mean()
              for i in range(0, tokens, config.block_size)]
    values = torch.stack(scores).tolist()
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("non-finite recovery probe scores")
    picked = {blocks-1}
    ranked = sorted(range(blocks-1), key=lambda index: (-values[index], index))
    picked.update(ranked[:budget-1])
    selected = tuple(sorted(picked))
    intervals = []
    for block in selected:
        lo, hi = block*config.block_size, min(tokens, (block+1)*config.block_size)
        if intervals and intervals[-1][1] == lo:
            intervals[-1] = intervals[-1][0], hi
        else:
            intervals.append((lo, hi))
    return selected, tuple(intervals)


def _rotation_factors(offset, inv_freq):
    if not offset:
        return None
    phase = inv_freq.float() * float(offset)
    phase = torch.cat((phase, phase))
    return phase.cos(), phase.sin()


def _apply_rotation(keys, factors):
    if factors is None:
        return keys
    cos, sin = factors
    working = keys if keys.dtype == torch.float64 else keys.float()
    first, second = working.chunk(2, -1)
    return (working * cos
            + torch.cat((-second, first), -1) * sin).to(keys.dtype)


def prepare_request_rotations(segments, prefix_tokens, inv_freq):
    """Request-local unit rotations indexed by document occurrence, not ID.

    Keep exactly the v1 FP32 operations and rounding. No key/value storage or
    request-dependent descriptors enter the immutable offline document cache.
    """
    factors, offset = [], prefix_tokens
    for segment in segments:
        factors.append(_rotation_factors(offset, inv_freq))
        offset += segment.doc_len
    return tuple(factors)


def rotate_canonical_keys(keys, offset, inv_freq):
    """Unit R(offset), never a second YaRN attention-amplitude scaling.

    Native canonical keys already contain their model's RoPE amplitude.
    Zero offset is a view; callers materialize private concatenated history
    before any selected-row writes and never mutate this return in place.
    """
    # Keep the extra coordinate transform in FP32 for FP16/BF16 caches;
    # rounding trigonometric factors before multiplication compounds error.
    return _apply_rotation(keys, _rotation_factors(offset, inv_freq))


def canonical_layer(segments, layer, prefix_tokens, inv_freq, rotation_factors=None):
    if rotation_factors is not None and len(rotation_factors) != len(segments):
        raise ValueError("rotation descriptors must match document occurrences")
    offset, keys, values = prefix_tokens, [], []
    for occurrence, segment in enumerate(segments):
        k, v = segment.kv[layer]
        keys.append(rotate_canonical_keys(k, offset, inv_freq) if rotation_factors is None
                    else _apply_rotation(k, rotation_factors[occurrence]))
        values.append(v)
        offset += segment.doc_len
    # cat always allocates, including one segment. These private tensors may
    # be repaired; neither the canonical tensors nor another request can be.
    return torch.cat(keys, dim=2), torch.cat(values, dim=2)


def _attention(engine, q, k, v, pk, pv, layer, query_positions):
    attn = engine.layers[layer].self_attn
    if engine.backend == "flash":
        return engine._recovery_flash_attention(q, k, v, pk, pv, layer, attn.scaling)
    all_k, all_v = torch.cat((pk, k), 2), torch.cat((pv, v), 2)
    key_positions = torch.arange(all_k.shape[2], device=q.device)
    output = torch.empty_like(q)
    for h, policy in enumerate(engine.policies[layer]):
        mask = key_positions[None, :] <= query_positions[:, None]
        if policy.head_type == "local":
            mask &= ((key_positions[None, :] >= query_positions[:, None]-policy.window)
                     | (key_positions[None, :] < min(policy.sink_size, pk.shape[2])))
        query = q[:, h*engine.groups:(h+1)*engine.groups].float()
        score = query @ all_k[:, h:h+1].float().transpose(-1, -2) * attn.scaling
        output[:, h*engine.groups:(h+1)*engine.groups] = (
            score.masked_fill(~mask, -torch.inf).softmax(-1)
            @ all_v[:, h:h+1].float()).to(q.dtype)
    return output


def _prepared_block_attention(engine, q, k, v, pk, pv, layer, intervals):
    """Reuse one layer-private token-major KV layout for every selected run.

    This is the same FlashAttention invocation sequence, absolute causal mask,
    local-window/sink split and head-major LSE merge as the legacy per-run
    path. The only difference is preparing the group layouts once. It does
    not batch unrelated query positions into one causal call. B=1 is required
    so token-axis prefix/suffix slices remain contiguous for the same kernel.
    Unsupported backends/shapes are selected out *before* executing this
    helper; kernel/runtime failures are never caught and silently retried.

    Prepared tensors are layer-local transient workspace, not retained cache.
    K and V are both fresh allocations; none aliases offline document storage.
    """
    if engine.backend != "flash" or q.shape[0] != 1:
        raise ValueError("prepared attention requires FlashAttention and B=1")
    groups = {}
    for h, policy in enumerate(engine.policies[layer]):
        key = (policy.head_type == "local", policy.window, min(policy.sink_size, pk.shape[2]))
        if not key[0]:
            key = (False, -1, 0)
        groups.setdefault(key, []).append(h)
    full_group = len(groups) == 1
    prepared = []
    for (local, window, sink), kv_heads in groups.items():
        if full_group:
            query, prefix_k, prefix_v, keys, values = q, pk, pv, k, v
            query_heads = None
        else:
            query_heads = [h*engine.groups+j for h in kv_heads for j in range(engine.groups)]
            query, prefix_k, prefix_v = q[:, query_heads], pk[:, kv_heads], pv[:, kv_heads]
            keys, values = k[:, kv_heads], v[:, kv_heads]
        prepared.append((local, window, sink, query_heads,
            query.transpose(1, 2).contiguous(),
            torch.cat((prefix_k, keys), 2).transpose(1, 2).contiguous(),
            torch.cat((prefix_v, values), 2).transpose(1, 2).contiguous()))

    def attend(query, keys, values, *, causal, window=-1):
        out, lse, _ = engine._flash_dense(query, keys, values,
            softmax_scale=engine.layers[layer].self_attn.scaling,
            causal=causal, window_size=(window, 0) if window >= 0 else (-1, -1),
            return_attn_probs=True)
        return out.transpose(1, 2), lse

    pieces, cursor = [], 0
    for lo, hi in intervals:
        width, end = hi-lo, pk.shape[2]+hi
        output = None if full_group else torch.empty_like(q[:, :, cursor:cursor+width])
        # Interval first, group second: preserve the legacy kernel call order.
        for local, window, sink, query_heads, query, keys, values in prepared:
            run_query = query[:, cursor:cursor+width]
            if not local:
                result, _ = attend(run_query, keys[:, :end], values[:, :end], causal=True)
            else:
                recent_start = max(sink, end-(window+width))
                recent, recent_lse = attend(run_query, keys[:, recent_start:end],
                    values[:, recent_start:end], causal=True, window=window)
                if sink:
                    initial, initial_lse = attend(run_query, keys[:, :sink], values[:, :sink], causal=False)
                    z = torch.logaddexp(recent_lse.float(), initial_lse.float())
                    a = torch.exp(recent_lse.float()-z)[..., None]
                    b = torch.exp(initial_lse.float()-z)[..., None]
                    result = (recent.float()*a + initial.float()*b).to(q.dtype)
                else:
                    result = recent
            if full_group:
                output = result
            else:
                output[:, query_heads] = result
        pieces.append(output)
        cursor += width
    return torch.cat(pieces, dim=2)


@torch.inference_mode()
def recover_selective(engine, prefix_segment, document_segments, config, query_ids=None):
    """Return ordinary full-history KV plus auditable actual row counts.

    The same selected rows are carried from the post-probe residual through
    all later layers. New rows are never added without their hidden state.
    Selection is heuristic; it makes no claim of a <=2% task-quality bound.
    """
    ids = torch.cat([s.token_ids for s in document_segments]).to(engine.device)[None]
    tokens, prefix_tokens = ids.shape[1], prefix_segment.doc_len
    positions = torch.arange(prefix_tokens, prefix_tokens+tokens, device=engine.device)[None]
    hidden = engine.model.model.embed_tokens(ids)
    cos, sin = engine.model.model.rotary_emb(hidden, positions)
    query_mode = config.selection_mode == "query_attention"
    query_projected, query_ffn = [0]*len(engine.layers), [0]*len(engine.layers)
    query_tokens = len(query_ids) if query_mode else 0
    if query_mode:
        query_positions = torch.arange(prefix_tokens+tokens, prefix_tokens+tokens+query_tokens,
                                       device=engine.device)[None]
        query_hidden = engine.model.model.embed_tokens(torch.tensor([query_ids], dtype=torch.long, device=engine.device))
        query_embeddings = engine.model.model.rotary_emb(query_hidden, query_positions)
    optimized = getattr(engine, "selective_execution_optimization", False)
    rotation_factors = (prepare_request_rotations(document_segments, prefix_tokens, engine._inv_freq)
                        if optimized else None)
    nonzero_offsets = len(document_segments)-(1 if not prefix_tokens else 0)
    counters = dict(offset_rotation_factor_builds=nonzero_offsets if optimized else 0,
        selected_rope_gathers=0, prepared_attention_layers=0,
        attention_kv_materialized_token_head_rows=0,
        attention_kv_materialization_scope="logical prefix+document KV concatenation rows times KV heads; excludes group indexing and layout copies, not measured bandwidth")
    captured, projected_rows, ffn_rows = [], [], []
    selected, intervals, row_ids, selected_embeddings = None, None, None, None
    for layer_index, layer in enumerate(engine.layers):
        attn = layer.self_attn
        is_probe = layer_index < config.probe_layers
        if is_probe:
            emb = (cos, sin)
        elif optimized:
            emb = selected_embeddings
        else:
            emb = (cos.index_select(1, row_ids), sin.index_select(1, row_ids))
            counters["selected_rope_gathers"] += 2
        q, k, v = engine._project(attn, layer.input_layernorm(hidden), emb)
        projected_rows.append(hidden.shape[1])
        pk, pv = prefix_segment.kv[layer_index]
        if is_probe:
            captured.append((k, v))
            output = _attention(engine, q, k, v, pk, pv, layer_index, positions[0])
            counters["attention_kv_materialized_token_head_rows"] += (prefix_tokens+tokens)*engine.kv_heads
            if query_mode:
                # Query states only consume the already computed live causal
                # history. They never become document inputs or published KV.
                qq, qk, qv = engine._project(attn, layer.input_layernorm(query_hidden), query_embeddings)
                query_projected[layer_index] = query_tokens
                if layer_index < config.probe_layers-1:
                    query_output = _attention(engine, qq, torch.cat((k, qk), 2), torch.cat((v, qv), 2),
                        pk, pv, layer_index, query_positions[0])
                    query_hidden = query_hidden + attn.o_proj(
                        query_output.transpose(1, 2).reshape(*query_hidden.shape[:-1], -1))
                    query_hidden = query_hidden + layer.mlp(layer.post_attention_layernorm(query_hidden))
                    query_ffn[layer_index] = query_tokens
                    counters["attention_kv_materialized_token_head_rows"] += (prefix_tokens+tokens+query_tokens)*engine.kv_heads
                    del query_output
            if layer_index == config.probe_layers-1:
                # Score the last probe layer's *input* K/V, not its output
                # residual. Global/dense heads are the repair target.
                heads = [h for h, p in enumerate(engine.policies[layer_index]) if p.head_type != "local"]
                if not heads:
                    heads = list(range(engine.kv_heads))
                if query_mode:
                    scores, query_diagnostics = conditional_query_relevance(qq, k, heads, engine.groups, attn.scaling)
                    counters.update(query_diagnostics)
                else:
                    cached_k, cached_v = canonical_layer(document_segments, layer_index, prefix_tokens,
                        engine._inv_freq, rotation_factors)
                    if not optimized:
                        counters["offset_rotation_factor_builds"] += nonzero_offsets
                    live_k, live_v = k[:, heads].float(), v[:, heads].float()
                    error = ((live_k-cached_k[:, heads].float()).square().mean((0, 1, 3))
                             + (live_v-cached_v[:, heads].float()).square().mean((0, 1, 3)))
                    magnitude = live_k.square().mean((0, 1, 3)) + live_v.square().mean((0, 1, 3))
                    scores = error/(magnitude+1e-8)
                    del cached_k, cached_v, live_k, live_v, error, magnitude
                selected, intervals = select_token_blocks(scores, config)
                row_ids = torch.cat([torch.arange(lo, hi, device=engine.device) for lo, hi in intervals])
                if optimized:
                    selected_embeddings = (cos.index_select(1, row_ids), sin.index_select(1, row_ids))
                    counters["selected_rope_gathers"] += 2
                del scores
                if query_mode:
                    del query_hidden, query_embeddings, qq, qk, qv
        else:
            all_k, all_v = canonical_layer(document_segments, layer_index, prefix_tokens,
                engine._inv_freq, rotation_factors)
            if not optimized:
                counters["offset_rotation_factor_builds"] += nonzero_offsets
            # Repair all selected rows before any block consumes this layer.
            all_k.index_copy_(2, row_ids, k)
            all_v.index_copy_(2, row_ids, v)
            captured.append((all_k, all_v))
            if optimized and engine.backend == "flash" and q.shape[0] == 1:
                output = _prepared_block_attention(engine, q, all_k, all_v, pk, pv, layer_index, intervals)
                counters["prepared_attention_layers"] += 1
                counters["attention_kv_materialized_token_head_rows"] += (prefix_tokens+tokens)*engine.kv_heads
            else:
                pieces, cursor = [], 0
                for lo, hi in intervals:
                    width = hi-lo
                    # Causal FA2 is bottom-right aligned. Ending the key span at
                    # this block's absolute end aligns q=[P+lo,P+hi) exactly;
                    # using all context with packed/gathered queries is invalid.
                    pieces.append(_attention(engine, q[:, :, cursor:cursor+width],
                        all_k[:, :, :hi], all_v[:, :, :hi], pk, pv, layer_index,
                        positions[0, lo:hi]))
                    cursor += width
                    counters["attention_kv_materialized_token_head_rows"] += (prefix_tokens+hi)*engine.kv_heads
                output = torch.cat(pieces, dim=2)
        hidden = hidden + attn.o_proj(output.transpose(1, 2).reshape(*hidden.shape[:-1], -1))
        residual, x = hidden, layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(x)
        ffn_rows.append(x.shape[1])
        if layer_index == config.probe_layers-1:
            hidden = hidden.index_select(1, row_ids)
    result, cursor = [prefix_segment.kv], 0
    for segment in document_segments:
        stop = cursor+segment.doc_len
        result.append([(k[:, :, cursor:stop], v[:, :, cursor:stop]) for k, v in captured])
        cursor = stop
    selected_count = row_ids.numel()
    return result, dict(
        recovery_mode="experimental_query_selective_v3" if query_mode else "experimental_block_selective_v1",
        recovery_approximate=True,
        selected_token_rows=selected_count, selected_block_count=len(selected),
        selected_block_ids=list(selected), selected_intervals=[list(x) for x in intervals],
        projected_rows_per_layer=projected_rows, ffn_rows_per_layer=ffn_rows,
        row_execution_fraction=sum(projected_rows)/(tokens*len(engine.layers)),
        probe_score_layer=config.probe_layers-1,
        selector=("block-mean conditional query-to-document relevance; mandatory final block" if query_mode else
                  "block-mean relative global KV drift; mandatory final block"),
        selection_mode=config.selection_mode,
        recovery_query_tokens=query_tokens,
        recovery_query_sha256=query_token_sha256(query_ids) if query_mode else None,
        query_projected_rows_per_layer=query_projected,
        query_ffn_rows_per_layer=query_ffn,
        full_row_fallback_reason=None,
        global_kv_storage="private materialized history; no cross-request global alias",
        selective_execution_optimization=optimized,
        recovery_selected_runs=len(intervals), execution_counters=counters,
        execution_workspace_scope="request-local rotation/selected-RoPE, one layer's prepared private KV and optional query-probe hidden/KV plus bounded GQA_groups*16*2048 score tiles; no cross-request retention; CUDA transient peak measured externally",
    )
