"""Optional native FA2 paged-cache backend for the Qwen3 storage ablation.

This file does not change the default Triton adapter or production RedKnot.
It preserves the same immutable head-page sharing and exclusive append-only
suffix arena, but routes actual attention through flash_attn_with_kvcache.

Each physical KV head becomes one virtual batch item. Its GQA query-head group
remains the query-head dimension; the physical pool is viewed as a paged cache
with one KV head per page. Block tables may alias immutable shared pages across
virtual requests. k/v=None makes the FA2 invocation read-only; only the adapter's
existing private-suffix writes update the pool. No historical KV is gathered.

The simple native cache layout requires a page-aligned source document: there
must be no hidden physical padding gap between document and suffix tokens.
Nonaligned documents fail closed, rather than silently changing positions.
"""

from __future__ import annotations

import inspect

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from segpaged_qwen3_throughput_adapter import BatchedQwen3PagedDecode


class BatchedQwen3FlashPagedDecode(BatchedQwen3PagedDecode):
    """Opt-in FA2 head-paged backend; all inherited safety contracts apply."""

    def __init__(self, model, document_cache, *, document_length, batch_size,
                 shared_kv_head_ratio, capacity_pages, page_size=512,
                 max_suffix_tokens, device=None):
        parameter = next(model.parameters())
        target = parameter.device if device is None else torch.device(device)
        if target.type != "cuda" or parameter.device.type != "cuda":
            raise ValueError("FA2 head-paged adapter requires a CUDA model and cache")
        if parameter.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("FA2 head-paged adapter supports only FP16/BF16")
        if type(page_size) is not int or page_size <= 0 or page_size % 256:
            raise ValueError("FA2 paged cache requires page_size to be a positive multiple of 256")
        if type(document_length) is not int or document_length <= 0 or document_length % page_size:
            raise ValueError("FA2 head-paged source document must be page-aligned")
        from flash_attn import flash_attn_with_kvcache
        parameters = inspect.signature(flash_attn_with_kvcache).parameters
        if not {"block_table", "cache_seqlens", "k", "v"}.issubset(parameters):
            raise RuntimeError("installed FlashAttention lacks the required paged KV-cache interface")
        self._flash_attention = flash_attn_with_kvcache
        super().__init__(model, document_cache, document_length=document_length,
                         batch_size=batch_size, shared_kv_head_ratio=shared_kv_head_ratio,
                         capacity_pages=capacity_pages, page_size=page_size,
                         max_suffix_tokens=max_suffix_tokens, device=device)
        try:
            # Tables are static: the base adapter preallocates all private tail
            # capacity and only the visible sequence length advances. Native
            # FA2 requires int32 block tables and sequence lengths.
            if self.pool.capacity_pages >= 2**31:
                raise ValueError("FA2 block indices exceed int32 capacity")
            self._flash_block_tables = [slots.to(torch.int32) for slots in self._slots]
            self._flash_cache_lengths = torch.full((self.virtual_kv_heads,), document_length,
                                                   dtype=torch.int32, device=self.device)
            self._flash_k_cache = self.pool.k.unsqueeze(2)
            self._flash_v_cache = self.pool.v.unsqueeze(2)
        except BaseException:
            self.close()
            raise

    def _attention_forward(self, attention, hidden_states, position_embeddings,
                           attention_mask, past_key_values=None, cache_position=None, **kwargs):
        if not self._active or past_key_values is not None:
            raise RuntimeError("FA2 head-paged attention invoked outside its controlled forward")
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
        # Same append-only private-row write as the original adapter. Shared
        # document spans never appear in these destination indices.
        idx = self._append_indices[layer][:, before:after].reshape(-1)
        self.pool.k.view(-1, self.head_dim).index_copy_(0, idx, k.reshape(-1, self.head_dim))
        self.pool.v.view(-1, self.head_dim).index_copy_(0, idx, v.reshape(-1, self.head_dim))
        self.pool.written_bytes += 2 * idx.numel() * self.head_dim * self.pool.k.element_size()
        pd = self.geometry["doc_pages_per_head"]
        tail_lengths = (after - self._suffix_page_starts).clamp(min=0, max=self.page_size)
        self._lengths[layer][:, pd:].copy_(tail_lengths.unsqueeze(0))
        self._flash_cache_lengths.fill_(self.document_length + after)

        # [B,Hkv,G,T,D] -> [B*Hkv,T,G,D]. Grouping is preserved exactly:
        # virtual request b*Hkv+h owns one KV head and its G query heads.
        query = q.reshape(self.batch_size, self.kv_heads, self.groups, shape[1], self.head_dim)
        query = query.permute(0, 1, 3, 2, 4).reshape(self.virtual_kv_heads, shape[1],
                                                  self.groups, self.head_dim).contiguous()
        output = self._flash_attention(query, self._flash_k_cache, self._flash_v_cache,
                                       k=None, v=None,
                                       cache_seqlens=self._flash_cache_lengths,
                                       block_table=self._flash_block_tables[layer],
                                       softmax_scale=attention.scaling, causal=True)
        # Native FA2's causal mask is bottom-right aligned: a question chunk
        # of T rows sees the correct prefix + preceding question rows when
        # cache_seqlens is document_length + all appended question tokens.
        self.attention_calls += 1
        output = output.reshape(self.batch_size, self.kv_heads, shape[1], self.groups, self.head_dim)
        output = output.permute(0, 2, 1, 3, 4).reshape(*shape, -1).contiguous()
        return attention.o_proj(output), None

    @torch.inference_mode()
    def reset_suffix(self):
        super().reset_suffix()
        self._flash_cache_lengths.fill_(self.document_length)

    def stats(self):
        result = super().stats()
        tensors = self._flash_block_tables + [self._flash_cache_lengths]
        unique = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            unique[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        result.update(attention_backend="flash_attn_with_kvcache_head_paged_virtual_batch",
                      virtual_batch_size=self.virtual_kv_heads,
                      virtual_cache_kv_heads=1,
                      virtual_query_heads=self.groups,
                      native_flash_paged_cache_read_only=True,
                      historical_kv_gather=False,
                      page_aligned_source_required=True,
                      flash_descriptor_additional_storage_bytes=sum(unique.values()))
        result["descriptor_unique_storage_bytes"] += sum(unique.values())
        result["descriptor_unique_storage_allocations"] += len(unique)
        return result
