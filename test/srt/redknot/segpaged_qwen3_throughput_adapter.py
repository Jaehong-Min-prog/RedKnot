"""Experimental batched Qwen3 decoder consuming physical head-paged KV.

This is a *shared-prefix storage ablation*, not non-prefix RedKnot reuse, an
approximate local-head policy, or the production HeadKVManager workflow. All
heads use full causal attention. A selected subset of physical KV heads aliases
one immutable prefilled document; all other document pages are private copies.
Different requests then consume distinct questions and generate real logits.

The model's attention projections, Q/K norms, RoPE, output projection, MLP,
residuals, embeddings and LM head are unchanged. No historical KV gather occurs
in decoding. Batch is flattened into virtual heads for the existing paged
attention kernel. An HF document-cache fixture is consumed during construction
only; callers may retain it between arms but must report it outside the pool
budget, identically in both arms.

Private suffix pages use an explicitly isolated append-only arena specialization:
they are exclusively owned, never published in an immutable RequestVersion,
and only previously invisible rows are written on one ordered CUDA stream.
Their descriptors expose those rows only after the writes. They remain pinned
until close. reset_suffix synchronizes before invalidating/reusing rows. This
avoids per-token immutable-tail COW and is NOT a claim that the production
manager currently implements this optimization. Shared document pages are sealed
and are never modified. The adapter must not be used concurrently or recursively.

Requires the pinned Transformers 4.57.1 Qwen3 source contract.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import types

import torch
import transformers
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb


_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "python/sglang/srt/mem_cache"))
from head_kv.attention import paged_attention, validate_paged_attention_inputs  # noqa: E402
from head_kv.pool import CapacityError, HeadPagePool, completion_event  # noqa: E402


def _geometry(model, document_length, batch_size, shared_kv_head_ratio,
              max_suffix_tokens, page_size):
    for name, value in (("document_length", document_length), ("batch_size", batch_size),
                        ("max_suffix_tokens", max_suffix_tokens), ("page_size", page_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(shared_kv_head_ratio, bool) or not math.isfinite(shared_kv_head_ratio):
        raise ValueError("shared_kv_head_ratio must be finite in [0,1]")
    if not 0 <= shared_kv_head_ratio <= 1:
        raise ValueError("shared_kv_head_ratio must be in [0,1]")
    cfg = model.config
    heads = cfg.num_key_value_heads
    shared = int(math.floor(heads * shared_kv_head_ratio + 1e-9))
    doc_pages = math.ceil(document_length / page_size)
    suffix_pages = math.ceil(max_suffix_tokens / page_size)
    layers = cfg.num_hidden_layers
    shared_pages = layers * shared * doc_pages
    private_doc_pages = layers * batch_size * (heads - shared) * doc_pages
    private_suffix_pages = layers * batch_size * heads * suffix_pages
    return dict(kv_heads=heads, shared_kv_heads=shared, layers=layers,
                doc_pages_per_head=doc_pages, suffix_pages_per_head=suffix_pages,
                shared_document_pages=shared_pages,
                private_document_pages=private_doc_pages,
                private_suffix_pages=private_suffix_pages,
                required_pages=shared_pages + private_doc_pages + private_suffix_pages)


def required_capacity_pages(model, document_length, batch_size, shared_kv_head_ratio,
                            max_suffix_tokens, page_size=128):
    """Exact reserved page count, including all private suffix capacity."""
    return _geometry(model, document_length, batch_size, shared_kv_head_ratio,
                     max_suffix_tokens, page_size)["required_pages"]


def _cache_layer(cache, index):
    if hasattr(cache, "layers"):
        layer = cache.layers[index]
        return layer.keys, layer.values
    pair = cache[index]
    return pair[0], pair[1]


class BatchedQwen3PagedDecode:
    """One equal-length cohort with real batched model generation.

    Construction allocates/attaches the document and suffix capacity. Use it as
    ``with BatchedQwen3PagedDecode(...) as adapter`` before calling forward.
    Each forward consumes an equal number of suffix tokens for every request;
    callers implement greedy generation and fixed output-length termination.
    Different cohorts may have different capacities but concurrent adapters on
    one model are prohibited. Full source construction/teardown can be measured
    separately from resident warm request processing.
    """

    def __init__(self, model, document_cache, *, document_length, batch_size,
                 shared_kv_head_ratio, capacity_pages, page_size=128,
                 max_suffix_tokens, device=None):
        if transformers.__version__ != "4.57.1":
            raise RuntimeError(f"expected Transformers 4.57.1, got {transformers.__version__}")
        if model.config.model_type != "qwen3":
            raise ValueError("only Qwen3ForCausalLM is supported")
        if type(capacity_pages) is not int or capacity_pages <= 0:
            raise ValueError("capacity_pages must be a positive integer")
        self.geometry = _geometry(model, document_length, batch_size,
                                  shared_kv_head_ratio, max_suffix_tokens, page_size)
        if capacity_pages < self.geometry["required_pages"]:
            raise CapacityError(f"cohort needs {self.geometry['required_pages']} pages, budget {capacity_pages}")
        self.model = model
        self.document_length = document_length
        self.batch_size = batch_size
        self.max_suffix_tokens = max_suffix_tokens
        self.page_size = page_size
        self.shared_kv_head_ratio = float(shared_kv_head_ratio)
        parameter = next(model.parameters())
        self.device = parameter.device if device is None else torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device != parameter.device:
            raise ValueError("adapter and entire model must be on the same device")
        self.dtype = parameter.dtype
        self.attentions = [layer.self_attn for layer in model.model.layers]
        if not all(isinstance(a, Qwen3Attention) for a in self.attentions):
            raise TypeError("unexpected Qwen3 attention source contract")
        if any(a.sliding_window is not None for a in self.attentions):
            raise ValueError("this shared-prefix ablation requires full-attention Qwen3 layers")
        self.head_dim = self.attentions[0].head_dim
        self.kv_heads = self.geometry["kv_heads"]
        self.query_heads = model.config.num_attention_heads
        self.virtual_kv_heads = batch_size * self.kv_heads
        self.virtual_query_heads = batch_size * self.query_heads
        self.groups = self.query_heads // self.kv_heads
        self.suffix_tokens = 0
        self.forward_calls = 0
        self.attention_calls = 0
        self._original = []
        self._closed = False
        self._entered = False
        self._active = False
        self._failed = False
        self._stream = (torch.cuda.current_stream(self.device).cuda_stream
                        if self.device.type == "cuda" else None)
        self.pool = HeadPagePool(capacity_pages, page_size, self.head_dim,
                                 dtype=self.dtype, device=self.device)
        self._owned = ()
        self._pinned = False
        self._slots = []
        self._lengths = []
        self._append_indices = []
        self._suffix_refs = []
        self._document_refs = []
        self._validated = False
        try:
            self._construct(document_cache)
        except BaseException:
            self.close()
            raise

    def _check_stream(self):
        if self.device.type == "cuda" and torch.cuda.current_stream(self.device).cuda_stream != self._stream:
            raise RuntimeError("append arena is single-stream; changing CUDA streams is unsupported")

    @torch.inference_mode()
    def _construct(self, cache):
        geo = self.geometry
        pd, ps = geo["doc_pages_per_head"], geo["suffix_pages_per_head"]
        shared_heads = geo["shared_kv_heads"]
        self._owned = self.pool.reserve(geo["required_pages"])
        cursor = 0

        def take(count):
            nonlocal cursor
            refs = self._owned[cursor:cursor + count]
            cursor += count
            return refs

        # Positions are identical across virtual heads. Expanded views retain
        # one compact positional template rather than B*H copies of history.
        doc_positions = torch.arange(pd * self.page_size, device=self.device, dtype=torch.int64)
        suffix_positions = torch.arange(self.document_length,
                                        self.document_length + ps * self.page_size,
                                        device=self.device, dtype=torch.int64)
        positions = torch.cat((doc_positions, suffix_positions)).view(pd + ps, self.page_size)
        self._positions = positions.unsqueeze(0).expand(self.virtual_kv_heads, -1, -1)
        self._windows = torch.zeros(self.virtual_kv_heads, device=self.device, dtype=torch.int64)
        self._sinks = torch.zeros_like(self._windows)
        self._row_offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int64)
        self._suffix_page_starts = torch.arange(ps, device=self.device, dtype=torch.int64) * self.page_size

        for layer in range(geo["layers"]):
            source_k, source_v = _cache_layer(cache, layer)
            expected = (1, self.kv_heads, self.document_length, self.head_dim)
            if tuple(source_k.shape) != expected or source_v.shape != source_k.shape:
                raise ValueError(f"layer {layer}: HF document cache shape must be {expected}")
            if source_k.dtype != self.dtype or source_v.dtype != self.dtype:
                raise ValueError("document cache must match model dtype")
            if source_k.device != self.device or source_v.device != self.device:
                raise ValueError("document cache must match model device")
            table = [[-1] * (pd + ps) for _ in range(self.virtual_kv_heads)]
            lengths = [[self.page_size] * (pd + ps) for _ in range(self.virtual_kv_heads)]
            shared = {}
            for head in range(self.kv_heads):
                for request in range(1 if head < shared_heads else self.batch_size):
                    refs = take(pd)
                    self._document_refs.extend(refs)
                    for page, ref in enumerate(refs):
                        lo = page * self.page_size
                        hi = min(lo + self.page_size, self.document_length)
                        self.pool.write_page(ref, source_k[0, head, lo:hi], source_v[0, head, lo:hi])
                    self.pool.seal(refs)
                    if head < shared_heads:
                        shared[head] = refs
                    else:
                        row = request * self.kv_heads + head
                        table[row][:pd] = [ref.slot for ref in refs]
            for request in range(self.batch_size):
                for head in range(self.kv_heads):
                    row = request * self.kv_heads + head
                    if head < shared_heads:
                        table[row][:pd] = [ref.slot for ref in shared[head]]
                    suffix = take(ps)
                    self._suffix_refs.extend(suffix)
                    table[row][pd:] = [ref.slot for ref in suffix]
                    lengths[row][pd - 1] = self.document_length - (pd - 1) * self.page_size
                    lengths[row][pd:] = [0] * ps
            slots = torch.tensor(table, device=self.device, dtype=torch.int64)
            page_lengths = torch.tensor(lengths, device=self.device, dtype=torch.int64)
            self._slots.append(slots)
            self._lengths.append(page_lengths)
            rows = slots[:, pd:, None] * self.page_size + self._row_offsets
            self._append_indices.append(rows.reshape(self.virtual_kv_heads, -1)[:, :self.max_suffix_tokens])
        if cursor != len(self._owned):
            raise AssertionError("page accounting mismatch")
        # Lifetime pins apply to the immutable source and exclusive unpublished
        # suffix arena. No external manager/version is handed an unsealed tail.
        self.pool.pin(self._owned)
        self._pinned = True
        q = torch.zeros((self.virtual_query_heads, 1, self.head_dim),
                        device=self.device, dtype=self.dtype)
        qp = torch.tensor([self.document_length], device=self.device, dtype=torch.int64)
        for slots, lengths in zip(self._slots, self._lengths):
            validate_paged_attention_inputs(q, self.pool.k, self.pool.v, slots, lengths,
                                            self._positions, query_positions=qp,
                                            num_q_per_kv=self.groups,
                                            windows=self._windows, sinks=self._sinks)
        self._validated = True

    def __enter__(self):
        if self._closed or self._entered:
            raise RuntimeError("adapter cannot be closed or already entered")
        if getattr(self.model, "_segpaged_qps_adapter", None) is not None:
            raise RuntimeError("another experimental paged adapter already owns this model")
        self._check_stream()
        self.model._segpaged_qps_adapter = self
        for attention in self.attentions:
            self._original.append((attention, attention.forward))
            attention.forward = types.MethodType(self._attention_forward, attention)
        self._entered = True
        return self

    def __exit__(self, *exc):
        self.close()

    @torch.inference_mode()
    def forward(self, input_ids, *, logits_to_keep=1):
        if self._closed or not self._entered or self._failed:
            raise RuntimeError("forward requires an entered, healthy adapter")
        if self._active:
            raise RuntimeError("concurrent/recursive adapter use is unsupported")
        self._check_stream()
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, tokens]")
        if input_ids.device != self.device or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be int64 on the model device")
        if input_ids.shape[0] != self.batch_size or input_ids.shape[1] <= 0:
            raise ValueError("input_ids must be a nonempty equal-length cohort")
        count = input_ids.shape[1]
        after = self.suffix_tokens + count
        if after > self.max_suffix_tokens:
            raise CapacityError("private suffix arena capacity exceeded")
        start = self.document_length + self.suffix_tokens
        if start + count > self.model.config.max_position_embeddings:
            raise ValueError("configured Qwen3 position limit exceeded")
        self._active = True
        self._current_before = self.suffix_tokens
        self._current_after = after
        positions = torch.arange(start, start + count, device=self.device, dtype=torch.int64)
        self._query_positions = positions
        try:
            output = self.model(input_ids=input_ids, use_cache=False, past_key_values=None,
                                cache_position=positions, position_ids=positions.unsqueeze(0),
                                attention_mask={"full_attention": None, "sliding_attention": None},
                                logits_to_keep=logits_to_keep, return_dict=True)
            if output.past_key_values is not None:
                raise AssertionError("paged decode unexpectedly created an HF history cache")
            self.suffix_tokens = after
            self.forward_calls += 1
            return output
        except BaseException:
            # A partial layer traversal cannot be retried as if it were atomic.
            self._failed = True
            raise
        finally:
            self._active = False

    def _attention_forward(self, attention, hidden_states, position_embeddings,
                           attention_mask, past_key_values=None, cache_position=None, **kwargs):
        if not self._active or past_key_values is not None:
            raise RuntimeError("paged attention invoked outside its controlled forward")
        shape = hidden_states.shape[:-1]
        if shape[0] != self.batch_size:
            raise ValueError("unexpected model batch geometry")
        projected = (*shape, -1, attention.head_dim)
        q = attention.q_norm(attention.q_proj(hidden_states).view(projected)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(hidden_states).view(projected)).transpose(1, 2)
        v = attention.v_proj(hidden_states).view(projected).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        layer = attention.layer_idx
        before, after = self._current_before, self._current_after
        # Every destination is a unique suffix row outside the prior visible
        # extent. There are no historical gathers and no writes to shared pages.
        idx = self._append_indices[layer][:, before:after].reshape(-1)
        self.pool.k.view(-1, self.head_dim).index_copy_(0, idx, k.reshape(-1, self.head_dim))
        self.pool.v.view(-1, self.head_dim).index_copy_(0, idx, v.reshape(-1, self.head_dim))
        self.pool.written_bytes += 2 * idx.numel() * self.head_dim * self.pool.k.element_size()
        pd = self.geometry["doc_pages_per_head"]
        tail_lengths = (after - self._suffix_page_starts).clamp(min=0, max=self.page_size)
        self._lengths[layer][:, pd:].copy_(tail_lengths.unsqueeze(0))
        query = q.reshape(self.virtual_query_heads, shape[1], self.head_dim)
        if self.device.type == "cuda":
            # Bounds/layout are validated once, then only monotone bounded
            # lengths change under this adapter's single-stream ownership.
            from head_kv.triton_attention import triton_paged_attention
            qp = self._query_positions.unsqueeze(0).expand(self.virtual_query_heads, -1)
            out = triton_paged_attention(query, self.pool.k, self.pool.v,
                                         self._slots[layer], self._lengths[layer], self._positions,
                                         qp, self._windows, self._sinks,
                                         self.groups, attention.scaling, True)
        else:
            out = paged_attention(query, self.pool.k, self.pool.v,
                                  self._slots[layer], self._lengths[layer], self._positions,
                                  query_positions=self._query_positions, num_q_per_kv=self.groups,
                                  scale=attention.scaling, windows=self._windows,
                                  sinks=self._sinks, backend="torch")
        self.attention_calls += 1
        out = out.view(self.batch_size, self.query_heads, shape[1], self.head_dim)
        out = out.transpose(1, 2).reshape(*shape, -1).contiguous()
        return attention.o_proj(out), None

    @torch.inference_mode()
    def reset_suffix(self):
        if self._closed or self._active or self._failed:
            raise RuntimeError("reset requires a healthy idle adapter")
        self._check_stream()
        # Required lifetime barrier: the next append may overwrite old rows.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        pd = self.geometry["doc_pages_per_head"]
        for lengths in self._lengths:
            lengths[:, pd:].zero_()
        self.suffix_tokens = 0

    def stats(self):
        if self._closed:
            raise RuntimeError("stats requires a live adapter")
        # Expanded positions and sliced append indices share backing storage;
        # count each allocation once, including padding retained by a view.
        descriptors = (self._slots + self._lengths + self._append_indices +
                       [self._positions, self._windows, self._sinks,
                        self._row_offsets, self._suffix_page_starts])
        if hasattr(self, "_query_positions"):
            descriptors.append(self._query_positions)
        storage_sizes = {}
        for tensor in descriptors:
            storage = tensor.untyped_storage()
            storage_sizes[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        page_bytes = self.pool.page_bytes
        return dict(scope="experimental_batched_real_qwen3_shared_prefix_storage_ablation",
                    production_manager_integration=False,
                    shared_document_immutable=True,
                    suffix_policy="exclusive_single_stream_append_only_private_arena",
                    full_causal_attention_all_heads=True,
                    requested_shared_kv_head_ratio=self.shared_kv_head_ratio,
                    realized_shared_kv_head_ratio=self.geometry["shared_kv_heads"] / self.kv_heads,
                    batch_size=self.batch_size, document_length=self.document_length,
                    suffix_tokens=self.suffix_tokens, max_suffix_tokens=self.max_suffix_tokens,
                    page_size=self.page_size, geometry=dict(self.geometry),
                    minimum_required_pool_bytes=self.geometry["required_pages"] * page_bytes,
                    shared_document_physical_bytes=self.geometry["shared_document_pages"] * page_bytes,
                    private_document_physical_bytes=self.geometry["private_document_pages"] * page_bytes,
                    private_suffix_reserved_bytes=self.geometry["private_suffix_pages"] * page_bytes,
                    descriptor_unique_storage_bytes=sum(storage_sizes.values()),
                    descriptor_unique_storage_allocations=len(storage_sizes),
                    external_hf_source_fixture_in_pool_budget=False,
                    pool=self.pool.stats(), forward_calls=self.forward_calls,
                    attention_calls=self.attention_calls,
                    all_generated_logits_consume_paged_attention=True)

    def close(self):
        if self._closed:
            return
        if self._active:
            raise RuntimeError("cannot close during an active forward")
        for attention, original in self._original:
            attention.forward = original
        self._original.clear()
        if getattr(self.model, "_segpaged_qps_adapter", None) is self:
            del self.model._segpaged_qps_adapter
        self._entered = False
        if self._owned:
            # Closing from a different stream is safe only after a device-wide
            # barrier. Keeping teardown synchronous also prevents benchmark
            # fixture destruction racing the last model output.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            event = completion_event(self.device)
            if self._pinned:
                self.pool.unpin(self._owned, event)
                self._pinned = False
            self.pool.release(self._owned, event)
            event.synchronize()
            self.pool.collect()
            self._owned = ()
        self._closed = True
