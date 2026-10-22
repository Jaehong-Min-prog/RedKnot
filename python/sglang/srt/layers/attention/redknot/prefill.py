# Copyright 2024-2026 SGLang RedKnot Integration.
"""Offline prefill utilities for RedKnot segments.

This module produces per-segment KV blobs in the *same shape* that sglang's
attention backend later consumes::

    kv[layer_idx] = (K, V)   each [1, num_kv_heads, L, head_dim]

with rotary positions in ``[0, L)``. The result is wrapped in an
:class:`OfflineSegment` and registered with the global
:class:`OfflineKVCache` so future requests can splice it back in.

We deliberately implement this against the HuggingFace ``transformers``
runtime (which is what __REDKNOT_V02__ used). Running offline prefill through
sglang's full scheduler stack is possible too but requires a separate
"prefill-only" request mode; supporting that is left for a later patch.
"""

from __future__ import annotations

import gc
import logging
from typing import List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.attention.redknot.offline_cache import (
    OfflineKVCache,
    OfflineSegment,
    build_offline_segment,
    get_global_offline_cache,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def offline_prefill_segment(
    model,
    tokenizer,
    segment_text: str,
    *,
    prepend_bos: bool = False,
    chunk_size: int = 4096,
    model_id: Optional[str] = None,
    store: Optional[OfflineKVCache] = None,
    extra_key: str = "",
    capture_ffn_delta: bool = False,
    capture_attn_output: bool = False,
) -> OfflineSegment:
    """Run a chunked prefill on ``segment_text`` and stash the resulting KV.

    Parameters
    ----------
    model:
        A loaded ``transformers.AutoModelForCausalLM`` instance.
    tokenizer:
        Matching tokenizer.
    segment_text:
        Text content of the segment.
    prepend_bos:
        Prepend BOS to the first segment (matches __REDKNOT_V02__ behaviour).
    chunk_size:
        Sub-batch length used during the offline prefill to bound peak
        hidden-state memory; KV is fully retained.
    model_id:
        String identifier used by the global cache key. Defaults to the
        model's ``_name_or_path``.
    store:
        Optional override for the destination cache (defaults to the
        process-wide singleton).

    Returns
    -------
    :class:`OfflineSegment` (already registered with ``store``).
    """
    from transformers import DynamicCache  # local import to avoid hard dep

    ids = tokenizer(segment_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ]
    if prepend_bos and tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], dtype=ids.dtype)
        ids = torch.cat([bos, ids], dim=1)
    doc_len = int(ids.shape[1])

    base_model = model.model if hasattr(model, "model") else model
    n_layers = model.config.num_hidden_layers
    ffn_chunks: Optional[List[List[torch.Tensor]]] = None
    attn_chunks: Optional[List[List[torch.Tensor]]] = None
    hook_handles = []
    if capture_ffn_delta:
        ffn_chunks = [[] for _ in range(n_layers)]

        def make_ffn_hook(layer_idx: int):
            def hook(_module, _args, output):
                if not isinstance(output, torch.Tensor) or output.ndim != 3:
                    raise RuntimeError("offline FFN hook expected [B,L,H] tensor")
                ffn_chunks[layer_idx].append(output.detach().clone())

            return hook

        for layer_idx in range(n_layers):
            hook_handles.append(
                base_model.layers[layer_idx].mlp.register_forward_hook(
                    make_ffn_hook(layer_idx)
                )
            )

    if capture_attn_output:
        attn_chunks = [[] for _ in range(n_layers)]

        def make_attn_hook(layer_idx: int):
            def hook(_module, args):
                if not args or not isinstance(args[0], torch.Tensor):
                    raise RuntimeError("offline attention hook did not receive a tensor")
                value = args[0]
                if value.ndim != 3:
                    raise RuntimeError("offline attention hook expected [B,L,E]")
                attn_chunks[layer_idx].append(value.detach().clone())

            return hook

        for layer_idx in range(n_layers):
            hook_handles.append(
                base_model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                    make_attn_hook(layer_idx)
                )
            )

    past = DynamicCache()
    try:
        for start in range(0, doc_len, chunk_size):
            end = min(start + chunk_size, doc_len)
            chunk = ids[:, start:end].to(model.device)
            out = model(input_ids=chunk, past_key_values=past, use_cache=True)
            past = out.past_key_values
            del out
    finally:
        for handle in hook_handles:
            handle.remove()

    kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for li in range(n_layers):
        k = past.layers[li].keys.detach().clone()
        v = past.layers[li].values.detach().clone()
        kv.append((k, v))
    del past
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mid = model_id or getattr(model.config, "_name_or_path", "unknown")
    ffn_delta = None
    if ffn_chunks is not None:
        ffn_delta = [torch.cat(chunks, dim=1) for chunks in ffn_chunks]
        if any(delta.shape[1] != doc_len for delta in ffn_delta):
            raise RuntimeError("offline FFN capture does not cover the segment")
    attn_output = None
    if attn_chunks is not None:
        attn_output = [torch.cat(chunks, dim=1) for chunks in attn_chunks]
        if any(value.shape[1] != doc_len for value in attn_output):
            raise RuntimeError("offline attention capture does not cover the segment")
    cache_extra = extra_key
    if capture_ffn_delta or capture_attn_output:
        cache_extra = (
            f"{extra_key}|ffn_delta={int(capture_ffn_delta)}"
            f"|attn_output={int(capture_attn_output)}"
        )
    sid = OfflineKVCache.compute_segment_id(
        mid, ids[0].tolist(), prepend_bos=prepend_bos, extra=cache_extra
    )
    seg = build_offline_segment(
        segment_id=sid,
        token_ids=ids[0],
        kv=kv,
        ffn_delta=ffn_delta,
        attn_output=attn_output,
    )
    target = store or get_global_offline_cache()
    target.put(seg)
    logger.info(
        "RedKnot offline prefill: model=%s, len=%d, layers=%d, id=%s",
        mid,
        doc_len,
        n_layers,
        sid[:12],
    )
    return seg


@torch.no_grad()
def offline_prefill_segments(
    model,
    tokenizer,
    segment_texts: Sequence[str],
    *,
    chunk_size: int = 4096,
    model_id: Optional[str] = None,
    store: Optional[OfflineKVCache] = None,
    capture_ffn_delta: bool = False,
    capture_attn_output: bool = False,
) -> List[OfflineSegment]:
    """Prefill a list of segments, prepending BOS only to the first one.

    Mirrors the original RedKnot convention (``segment[0]`` carries BOS
    because it is the absolute prefix of the concatenated context).
    """
    out: List[OfflineSegment] = []
    for i, text in enumerate(segment_texts):
        seg = offline_prefill_segment(
            model,
            tokenizer,
            text,
            prepend_bos=(i == 0),
            chunk_size=chunk_size,
            model_id=model_id,
            store=store,
            capture_ffn_delta=capture_ffn_delta,
            capture_attn_output=capture_attn_output,
        )
        out.append(seg)
    return out
