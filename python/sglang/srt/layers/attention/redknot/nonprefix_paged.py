"""Experimental genuine non-prefix RedKnot serving for HF Qwen3.

Immutable position-zero *local* KV heads are shared by document occurrence;
global heads are recovered online using RedKnot's flat recovery computation.
Every emitted token consumes the resulting head-paged attention. There is no
native full-cache generation bypass. This is a single-device experimental
engine, not a replacement for SGLang's distributed request scheduler.

Canonical page positions never change. An occurrence's logical offset belongs
to this adapter: R(-offset) is applied to queries, not to shared keys. The
policy_approximate certificate records provenance, NOT an accuracy guarantee.
Private suffix pages form an exclusively owned, pinned, append-only arena;
they are not published as mutable HeadKVManager snapshots. Only previously
invisible rows are written, on one CUDA stream, and no reset is supported.

FA2 cannot combine block_table and cache_leftpad. Visible intervals therefore
alias page-aligned runs and compact-copy at most one leading partial page per
interval. LSE merging restores the exact attention union without gathering
full historical KV. CPU torch is a slow independent numerical oracle.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import copy
import threading
import time
import types
import uuid
from dataclasses import dataclass, field
from contextlib import ExitStack

import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from sglang.srt.mem_cache.head_kv.manager import HeadKVManager, ReuseProof, SegmentWrite
from sglang.srt.mem_cache.head_kv.pool import CapacityError, HeadPagePool, ImmediateEvent, completion_event

from .offline_cache import OfflineSegment
from .selective_recovery import (
    SelectiveRecoveryConfig, estimate_selective_cache_bytes, recover_selective,
    selected_block_budget, unique_storage_bytes, query_token_sha256,
)
from .hidden_residual_recovery import (
    capture_offline_hidden, estimate_hidden_residual_cache_bytes,
    recover_hidden_residual, GROUP_SIZE,
)


@dataclass(frozen=True)
class HistoricalPartition:
    key: tuple
    pages: tuple
    offset: int
    start: int
    end: int
    canonical: bool
    source_start: int = 0


@dataclass
class DocumentState:
    doc_id: str
    source_request_id: str
    token_ids: tuple
    doc_len: int
    source_keys: dict
    digest: str
    metadata: dict = field(default_factory=dict)
    recovery_kv: tuple = field(default_factory=tuple)
    recovery_nbytes: int = 0
    recovery_hidden: object = None
    recovery_hidden_nbytes: int = 0


@dataclass
class RequestState:
    request_id: str
    manager_request_id: str
    total_context_tokens: int
    prefix_ids: tuple
    document_ids: tuple
    document_offsets: tuple
    max_suffix_tokens: int
    suffix_tokens: int = 0
    failed: bool = False
    partitions: dict = field(default_factory=dict)
    tail_refs: dict = field(default_factory=dict)
    tail_all_refs: tuple = field(default_factory=tuple)
    tail_pinned: bool = False
    tail_rows: dict = field(default_factory=dict)
    admission_stats: dict = field(default_factory=dict)
    history_lease: object = None
    history_completed: bool = False
    history_release_failed: bool = False
    history_completion_uncertain: bool = False
    recovery_query_ids: tuple = field(default_factory=tuple)

    @property
    def doc_offsets(self):
        return self.document_offsets

    @property
    def history_tokens(self):
        return self.total_context_tokens


class NonPrefixPagedEngine:
    def __init__(self, model, head_cfg, *, capacity_pages, page_size=256,
                 backend="flash", sparse_ffn_schedule=None, model_revision="",
                 prefix_chunk_size=2048, share_local=True, selective_recovery=None,
                 selective_execution_optimization=True, publication_diagnostics=False,
                 metadata_only_alias_sources=False):
        if model.config.model_type != "qwen3":
            raise ValueError("this experimental binding supports Qwen3 only")
        if model.training:
            raise ValueError("model.eval() is required")
        if backend not in ("flash", "torch"):
            raise ValueError("backend must be flash or torch")
        if type(prefix_chunk_size) is not int or prefix_chunk_size <= 0:
            raise ValueError("prefix_chunk_size must be positive")
        if type(page_size) is not int or page_size <= 0:
            raise ValueError("page_size must be positive")
        parameter = next(model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        if any(p.device != self.device for p in model.parameters()):
            raise ValueError("single-device model required; TP is unsupported here")
        # Freeze the profile used by both recovery and decode. A caller cannot
        # mutate a shared config after canonical caches have been imported.
        self.model, self.head_cfg, self.backend = model, copy.deepcopy(head_cfg), backend
        self.layers = model.model.layers
        self.num_layers = len(self.layers)
        if selective_recovery is not None and not isinstance(selective_recovery, SelectiveRecoveryConfig):
            raise ValueError("selective_recovery must be SelectiveRecoveryConfig or None")
        if type(selective_execution_optimization) is not bool:
            raise ValueError("selective_execution_optimization must be boolean")
        if type(publication_diagnostics) is not bool:
            raise ValueError("publication_diagnostics must be boolean")
        if type(metadata_only_alias_sources) is not bool:
            raise ValueError("metadata_only_alias_sources must be boolean")
        # Diagnostic runs add CUDA synchronization at publication boundaries.
        # Keep disabled for formal throughput timing; no extra clocks, syncs
        # or result fields are introduced on the default path.
        self.publication_diagnostics = publication_diagnostics
        # Independent system-only A/B switch: alias preparation reads CPU
        # metadata, not source KV. True KV consumers still require read leases.
        self.metadata_only_alias_sources = metadata_only_alias_sources
        # System-only ablation. Both variants use identical selection, head
        # policy and private global history; the full-row path is unaffected.
        self.selective_execution_optimization = selective_execution_optimization
        self.selective_recovery = selective_recovery
        self._selective_active = bool(selective_recovery is not None
            and selective_recovery.row_fraction < 1
            and selective_recovery.probe_layers < self.num_layers)
        self._hidden_residual_active = bool(self._selective_active
            and selective_recovery.recovery_mode == "hidden_residual_int8")
        if self._selective_active and selective_recovery.max_offline_kv_bytes <= 0:
            raise ValueError("active selective recovery requires a positive offline KV budget")
        if self._selective_active and sparse_ffn_schedule is not None:
            raise ValueError("selective recovery plus sparse FFN is not yet validated")
        self._offline_kv_cache_bytes = 0
        self._offline_kv_cache_peak_bytes = 0
        self._offline_hidden_cache_bytes = 0
        self._offline_hidden_cache_peak_bytes = 0
        self._last_recovery_stats = {}
        self.kv_heads = model.config.num_key_value_heads
        self.query_heads = model.config.num_attention_heads
        self.groups = self.query_heads // self.kv_heads
        self.head_dim = self.layers[0].self_attn.head_dim
        self.page_size = page_size
        if (head_cfg.num_layers, head_cfg.num_kv_heads) != (self.num_layers, self.kv_heads):
            raise ValueError("head profile topology does not match this model")
        self.policies = [[self.head_cfg.get_strategy(l, h) for h in range(self.kv_heads)]
                         for l in range(self.num_layers)]
        for row in self.policies:
            for p in row:
                if p.head_type not in ("local", "global", "dense"):
                    raise ValueError("retrieval heads need an explicit upstream policy; not silently converted")
                if p.head_type == "local" and (p.window <= 0 or p.sink_size < 0):
                    raise ValueError("local heads require positive windows and nonnegative sinks")
        rotary = model.model.rotary_emb
        scaling = getattr(model.config, "rope_scaling", None)
        rope_type = (scaling or {}).get("rope_type", (scaling or {}).get("type", "default"))
        if rope_type not in ("default", "yarn") or getattr(rotary, "rope_type", "default") != rope_type:
            raise ValueError("only fixed default/YaRN RoPE is supported; dynamic and other RoPE types fail closed")
        self._rotary = rotary
        self._rope_type = rope_type
        self._rope_config = self._rope_config_state()
        self._rope_attention_scaling = float(rotary.attention_scaling)
        if not math.isfinite(self._rope_attention_scaling) or self._rope_attention_scaling <= 0:
            raise ValueError("RoPE attention scaling must be finite and positive")
        self._inv_freq = rotary.inv_freq.detach().float().clone()
        if not bool(torch.isfinite(self._inv_freq).all()):
            raise ValueError("RoPE frequencies must be finite")
        if 2 * self._inv_freq.numel() != self.head_dim:
            raise ValueError("partial rotary dimensions are unsupported")
        self._flash = self._flash_dense = None
        if backend == "flash":
            if self.device.type != "cuda" or self.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("flash backend requires CUDA FP16/BF16")
            if page_size % 256:
                raise ValueError("FlashAttention page_size must be a multiple of 256")
            from flash_attn import flash_attn_with_kvcache, flash_attn_func
            if not {"block_table", "return_softmax_lse"}.issubset(inspect.signature(flash_attn_with_kvcache).parameters):
                raise RuntimeError("installed FlashAttention lacks paged LSE support")
            self._flash = flash_attn_with_kvcache
            self._flash_dense = flash_attn_func
        self.pool = HeadPagePool(capacity_pages, page_size, self.head_dim,
                                 dtype=self.dtype, device=self.device)
        self.manager = HeadKVManager(self.pool)
        self.documents, self.requests = {}, {}
        self.sparse_ffn_schedule = sparse_ffn_schedule
        self.model_revision = str(model_revision)
        self.prefix_chunk_size = prefix_chunk_size
        self.share_local = bool(share_local)
        self.namespace = "nonprefix-qwen3:" + uuid.uuid4().hex
        self._thread = threading.get_ident()
        self._stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        self._closed, self._busy = False, False
        self.attention_calls = 0
        self.max_active_requests = 0
        self.admitted_requests = 0
        self.boundary_copy_bytes = 0
        self._plan_cache = {}
        self.history_lease_acquisitions = 0
        self.history_lease_completions = 0

    @property
    def requires_recovery_query(self):
        """Only the explicit approximate query selector needs admission IDs."""
        return self._selective_active and self.selective_recovery.selection_mode == "query_attention"

    def _rope_config_state(self):
        fields = ("rope_scaling", "rope_theta", "max_position_embeddings",
                  "original_max_position_embeddings", "head_dim", "partial_rotary_factor")
        return tuple({name: copy.deepcopy(getattr(config, name, None)) for name in fields}
                     for config in (self.model.config, self.model.model.rotary_emb.config))

    def _assert_rope_unchanged(self):
        rotary = self.model.model.rotary_emb
        if (rotary is not self._rotary or getattr(rotary, "rope_type", "default") != self._rope_type
                or self._rope_config_state() != self._rope_config):
            raise RuntimeError("RoPE configuration changed after engine creation")
        if float(rotary.attention_scaling) != self._rope_attention_scaling:
            raise RuntimeError("RoPE attention scaling changed after engine creation")
        if not torch.equal(rotary.inv_freq, self._inv_freq.to(rotary.inv_freq)):
            raise RuntimeError("RoPE frequencies changed after engine creation")

    def _check(self, *, for_cleanup=False):
        if self._closed:
            raise RuntimeError("engine is closed")
        if threading.get_ident() != self._thread or self._busy:
            raise RuntimeError("engine is single-threaded and non-reentrant")
        if self._stream is not None and torch.cuda.current_stream(self.device) != self._stream:
            raise RuntimeError("engine operations must remain on their original CUDA stream")
        if getattr(self.model, "_redknot_nonprefix_owner", None) is not None:
            raise RuntimeError("model already has an active RedKnot binding")
        if not for_cleanup:
            if self.pool.poisoned:
                raise RuntimeError("pool quarantined after an asynchronous lifetime failure")
            self._assert_rope_unchanged()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _ids(self, value, *, nonempty=True):
        if torch.is_tensor(value):
            if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
                raise ValueError("token IDs must be a one-dimensional integer sequence")
            value = value.tolist()
        result = tuple(value)
        if (nonempty and not result) or any(type(x) is not int or not 0 <= x < self.model.config.vocab_size for x in result):
            raise ValueError("invalid token IDs")
        return result

    def _root(self, identity, context):
        self.manager.create_request(identity, context_id=context, contract="qwen3-canonical-head-v1", namespace=self.namespace)

    @torch.inference_mode()
    def _prefill_ids(self, ids):
        if not ids:
            return [(self.pool.k.new_empty((1, self.kv_heads, 0, self.head_dim)),
                     self.pool.v.new_empty((1, self.kv_heads, 0, self.head_dim))) for _ in self.layers]
        cache = DynamicCache(config=self.model.config)
        for start in range(0, len(ids), self.prefix_chunk_size):
            part = torch.tensor([ids[start:start+self.prefix_chunk_size]], device=self.device, dtype=torch.long)
            out = self.model(input_ids=part, past_key_values=cache, use_cache=True, logits_to_keep=1)
            del out
        self._assert_rope_unchanged()
        return [(cache.layers[l].keys, cache.layers[l].values) for l in range(self.num_layers)]

    @torch.inference_mode()
    def register_document(self, doc_id, token_ids):
        self._check()
        if not isinstance(doc_id, str) or not doc_id or doc_id in self.documents:
            raise ValueError("document ID must be new and nonempty")
        ids = self._ids(token_ids)
        if len(ids) > self.model.config.max_position_embeddings:
            raise ValueError("document exceeds model position limit")
        digest = hashlib.sha256((self.model_revision + repr(ids)).encode()).hexdigest()
        rid = self.namespace + ":document:" + doc_id
        local = [(l, h) for l, row in enumerate(self.policies) for h, p in enumerate(row) if p.head_type == "local"]
        need = len(local) * math.ceil(len(ids) / self.page_size)
        if need > self.pool.stats()["free_pages"]:
            raise CapacityError(f"document needs {need} pages")
        if self._selective_active:
            estimate = (estimate_hidden_residual_cache_bytes(self.model.config, [len(ids)],
                            self.selective_recovery.probe_layers) if self._hidden_residual_active else
                        estimate_selective_cache_bytes(self.model.config, [len(ids)], self.dtype))
            retained = self._offline_hidden_cache_bytes if self._hidden_residual_active else self._offline_kv_cache_bytes
            if estimate + retained > self.selective_recovery.max_offline_kv_bytes:
                raise CapacityError("canonical document recovery cache exceeds the explicit offline KV budget"
                                    + (" (hidden int8 payload)" if self._hidden_residual_active else ""))
        self._root(rid, digest)
        try:
            hidden_cache = None
            if self._hidden_residual_active:
                kv, hidden_cache = capture_offline_hidden(self, ids, self.selective_recovery.probe_layers)
            else:
                kv = self._prefill_ids(ids) if local or self._selective_active else None
            canonical = tuple(kv) if self._selective_active and not self._hidden_residual_active else ()
            canonical_bytes = unique_storage_bytes(canonical)
            hidden_bytes = hidden_cache.unique_storage_bytes() if hidden_cache is not None else 0
            if (self._selective_active and canonical_bytes + self._offline_kv_cache_bytes
                    + hidden_bytes + self._offline_hidden_cache_bytes
                    > self.selective_recovery.max_offline_kv_bytes):
                raise CapacityError("actual recovery cache storage exceeds the explicit offline KV budget")
            keys = {(l, h): (l, h, "canonical") for l, h in local}
            canonical_positions = tuple(range(len(ids))) if local else ()
            writes = [SegmentWrite(keys[l, h], kv[l][0][0, h], kv[l][1][0, h],
                                   canonical_positions, digest, position_basis="rope-canonical")
                      for l, h in local]
            if writes:
                self.manager.update(rid, writes=writes, batched_writes=True)
            del kv, writes
            meta = dict(doc_id=doc_id, tokens=len(ids), local_head_count=len(local),
                        source_pages=need, canonical_positions=True, digest=digest,
                        offline_recovery_kv_bytes=canonical_bytes,
                        offline_recovery_hidden_bytes=hidden_bytes)
            self.documents[doc_id] = DocumentState(doc_id, rid, ids, len(ids), keys, digest, meta,
                                                  canonical, canonical_bytes, hidden_cache, hidden_bytes)
            self._offline_kv_cache_bytes += canonical_bytes
            self._offline_kv_cache_peak_bytes = max(self._offline_kv_cache_peak_bytes, self._offline_kv_cache_bytes)
            self._offline_hidden_cache_bytes += hidden_bytes
            self._offline_hidden_cache_peak_bytes = max(self._offline_hidden_cache_peak_bytes, self._offline_hidden_cache_bytes)
            return meta
        except BaseException:
            self.manager.release_request(rid)
            raise

    def _segment(self, name, ids, kv, recovery_hidden=None):
        segment = OfflineSegment(name, torch.tensor(ids, dtype=torch.long), len(ids), kv, self.device, 0)
        if recovery_hidden is not None:
            segment.meta["recovery_hidden"] = recovery_hidden
        return segment

    @torch.inference_mode()
    def _recover_online(self, prefix_segment, document_segments, query_ids=None):
        tokens = sum(s.doc_len for s in document_segments)
        config = self.selective_recovery
        all_blocks = False
        if self._selective_active:
            blocks, chosen = selected_block_budget(tokens, config)
            all_blocks = blocks == chosen
        if self._selective_active and not all_blocks:
            if self.requires_recovery_query:
                if query_ids is None:
                    raise ValueError("query_attention recovery requires actual query_ids")
                query_ids = self._ids(query_ids)
                if prefix_segment.doc_len+tokens+len(query_ids) > self.model.config.max_position_embeddings:
                    raise ValueError("query probe exceeds model position limit")
            recover = recover_hidden_residual if self._hidden_residual_active else recover_selective
            result, self._last_recovery_stats = recover(self, prefix_segment, document_segments, config, query_ids)
            return result
        # Exact fallback uses the unchanged original math; no canonical KV
        # reads, approximated hidden states, or block-attention decomposition.
        self._last_recovery_stats = dict(
            recovery_mode="full_rows", recovery_approximate=False,
            selected_token_rows=tokens,
            selected_block_count=math.ceil(tokens/config.block_size) if config else None,
            projected_rows_per_layer=[tokens]*self.num_layers,
            ffn_rows_per_layer=[tokens]*self.num_layers, row_execution_fraction=1.0,
            full_row_fallback_reason=("all_blocks_selected" if all_blocks else
                "full_row_configuration" if config else None))
        return self._recover_online_full(prefix_segment, document_segments)

    @torch.inference_mode()
    def _recover_online_full(self, prefix_segment, document_segments):
        """Flat RedKnot recovery, with a disjoint sink/window attention union.

        This mirrors online_forward_segments_flat's projection, norm, RoPE,
        residual and sparse-FFN computation. Its attention implementation is
        isolated here: the legacy helper double-counts sinks in early windows
        and applies the first local head's window to all local heads. Neither
        behavior is a desired head policy. No legacy module is monkeypatched.
        """
        ids = torch.cat([s.token_ids for s in document_segments]).to(self.device)[None]
        prefix_len = prefix_segment.doc_len
        pos = torch.arange(prefix_len, prefix_len + ids.shape[1], device=self.device)[None]
        hidden = self.model.model.embed_tokens(ids)
        cos, sin = self.model.model.rotary_emb(hidden, pos)
        captured = []
        kp = torch.arange(prefix_len + ids.shape[1], device=self.device)
        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(hidden)
            q, k, v = self._project(attn, x, (cos, sin))
            captured.append((k, v))
            pk, pv = prefix_segment.kv[li]
            if self.backend == "flash":
                output = self._recovery_flash_attention(q, k, v, pk, pv, li, attn.scaling)
            else:
                all_k, all_v = torch.cat((pk, k), 2), torch.cat((pv, v), 2)
                output = torch.empty_like(q)
                for h, policy in enumerate(self.policies[li]):
                    mask = kp[None, :] <= pos[0, :, None]
                    if policy.head_type == "local":
                        mask &= ((kp[None, :] >= pos[0, :, None] - policy.window)
                                 | (kp[None, :] < min(policy.sink_size, prefix_len)))
                    query = q[:, h*self.groups:(h+1)*self.groups].float()
                    scores = query @ all_k[:, h:h+1].float().transpose(-1, -2) * attn.scaling
                    output[:, h*self.groups:(h+1)*self.groups] = (
                        scores.masked_fill(~mask, -torch.inf).softmax(-1) @ all_v[:, h:h+1].float()).to(q.dtype)
            hidden = hidden + attn.o_proj(output.transpose(1, 2).reshape(*hidden.shape[:-1], -1))
            residual, x = hidden, layer.post_attention_layernorm(hidden)
            if self.sparse_ffn_schedule is not None and not self.sparse_ffn_schedule.is_dense_layer(li):
                from .sparse_ffn import apply_sparse_ffn, token_importance_from_attn
                sparse = apply_sparse_ffn(x, layer.mlp, layer_idx=li,
                    schedule=self.sparse_ffn_schedule, importance=token_importance_from_attn(output))
                hidden = residual + (sparse - x)
            else:
                hidden = residual + layer.mlp(x)
        result, cursor = [prefix_segment.kv], 0
        for segment in document_segments:
            stop = cursor + segment.doc_len
            result.append([(k[:, :, cursor:stop], v[:, :, cursor:stop]) for k, v in captured])
            cursor = stop
        return result

    def _recovery_flash_attention(self, q, k, v, prefix_k, prefix_v, layer, scale):
        """Same causal per-head mask as the literal torch recovery oracle."""
        def attend(query, keys, values, *, causal, window=-1):
            out, lse, _ = self._flash_dense(query.transpose(1, 2).contiguous(),
                keys.transpose(1, 2).contiguous(), values.transpose(1, 2).contiguous(),
                softmax_scale=scale, causal=causal, window_size=(window, 0) if window >= 0 else (-1, -1),
                return_attn_probs=True)
            return out.transpose(1, 2), lse

        groups = {}
        for h, policy in enumerate(self.policies[layer]):
            key = (policy.head_type == "local", policy.window, min(policy.sink_size, prefix_k.shape[2]))
            if not key[0]:
                key = (False, -1, 0)
            groups.setdefault(key, []).append(h)
        # A homogeneous layer selects every KV/query head in its original
        # order. Advanced indexing by that identity list would still allocate
        # and copy five large tensors; the final indexed scatter is redundant
        # too. Keep the same attention calls/masks/LSE arithmetic, only omit
        # those identity selections. Mixed-policy layers retain the old path.
        full_group = len(groups) == 1
        output = None if full_group else torch.empty_like(q)
        for (local, window, sink), kv_heads in groups.items():
            if full_group:
                query = q
                keys = torch.cat((prefix_k, k), 2)
                values = torch.cat((prefix_v, v), 2)
            else:
                query_heads = [h*self.groups+j for h in kv_heads for j in range(self.groups)]
                query = q[:, query_heads]
                keys = torch.cat((prefix_k[:, kv_heads], k[:, kv_heads]), 2)
                values = torch.cat((prefix_v[:, kv_heads], v[:, kv_heads]), 2)
            if not local:
                result, _ = attend(query, keys, values, causal=True)
            else:
                # Remove the sink span before window attention. The remaining
                # contiguous suffix retains bottom-right causal alignment, so
                # FA2's window mask still refers to the correct absolute rows.
                recent_k, recent_v = keys[:, :, sink:], values[:, :, sink:]
                keep = min(recent_k.shape[2], window+q.shape[2])
                recent, recent_lse = attend(query, recent_k[:, :, -keep:], recent_v[:, :, -keep:],
                                           causal=True, window=window)
                if sink:
                    initial, initial_lse = attend(query, keys[:, :, :sink], values[:, :, :sink], causal=False)
                    z = torch.logaddexp(recent_lse.float(), initial_lse.float())
                    a = torch.exp(recent_lse.float()-z)[..., None]
                    b = torch.exp(initial_lse.float()-z)[..., None]
                    result = (recent.float()*a + initial.float()*b).to(q.dtype)
                else:
                    result = recent
            if full_group:
                return result
            output[:, query_heads] = result
        return output

    @staticmethod
    def _project(attention, hidden, position_embeddings):
        shape = (*hidden.shape[:-1], -1, attention.head_dim)
        q = attention.q_norm(attention.q_proj(hidden).view(shape)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1, 2)
        v = attention.v_proj(hidden).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        return q, k, v

    def _visible(self, total, prefix_len, policy):
        if policy.head_type != "local":
            return [(0, total)]
        sink = min(policy.sink_size, prefix_len, total)
        tail = max(sink, total - policy.window)
        return ([(0, sink)] if sink else []) + ([(tail, total)] if tail < total else [])

    def _publish_document_history(self, rid, docs, offsets, context):
        """Publish all canonical aliases once with checked source snapshots.

        By default, repeated occurrences share one pinned source snapshot. The
        manager checks those exact snapshots at commit and performs a single
        root transition. Canonical positions and occurrence-specific query
        counterrotation certificates are unchanged. Private-copy ablations
        remain a separate materialized path with bounded per-document work.

        The opt-in metadata-only path does not read source tensors, so it can
        omit temporary source pins. Precise expected_sources identity checks
        and atomic destination retention remain mandatory. It does not alter
        the request-lifetime history lease used by actual attention readers.
        """
        if self.share_local:
            with ExitStack() as stack:
                snapshots = (self.manager.metadata_snapshots(doc.source_request_id for doc in docs)
                             if self.metadata_only_alias_sources else {})
                aliases = {}
                for occurrence, doc in enumerate(docs):
                    source_id = doc.source_request_id
                    if source_id not in aliases:
                        if not self.metadata_only_alias_sources:
                            snapshots[source_id] = stack.enter_context(self.manager.bind(source_id)).version
                        aliases[source_id] = []
                    source_root = snapshots[source_id]
                    certificate = "canonical-query-counterrotation-v1:" + doc.digest + ":" + str(offsets[occurrence])
                    for (l, h), source_key in doc.source_keys.items():
                        source = source_root.segments[source_key]
                        aliases[source_id].append((source_key, (l, h, f"document:{occurrence}"), ReuseProof(
                            "policy_approximate", source.provenance, context, certificate)))
                batches = [(source_id, shares) for source_id, shares in aliases.items() if shares]
                if batches:
                    self.manager.share_segments_many(rid, batches, expected_sources={
                        (source_id, False): snapshots[source_id] for source_id, _ in batches})
                result = dict(alias_publications=int(bool(batches)),
                              alias_source_roots=len(batches),
                              alias_segment_count=sum(len(shares) for _, shares in batches))
                if self.metadata_only_alias_sources:
                    result["alias_source_snapshot_mode"] = "metadata_only"
                return result
        for occurrence, doc in enumerate(docs):
            with self.manager.bind(doc.source_request_id) as lease:
                private = []
                certificate = "canonical-query-counterrotation-v1:" + doc.digest + ":" + str(offsets[occurrence])
                for (l, h), source_key in doc.source_keys.items():
                    source = lease.version.segments[source_key]
                    k, v = lease.gather(source_key)  # Admission-only storage ablation.
                    private.append(SegmentWrite((l, h, f"document:{occurrence}"), k, v,
                        source.positions, source.provenance, position_basis=source.position_basis,
                        reuse_kind="policy_approximate", validity_certificate=certificate))
                if private:
                    self.manager.update(rid, writes=private, batched_writes=True)
        result = dict(alias_publications=0, alias_source_roots=0, alias_segment_count=0)
        if self.metadata_only_alias_sources:
            result["alias_source_snapshot_mode"] = "pinned_read_lease_private_copy"
        return result

    def _publication_checkpoint(self, stages, name, started):
        """Diagnostic-only host wall interval, including its completion sync.

        Called only when explicitly enabled. The preceding checkpoint (or
        recovery's existing completion sync) separates GPU work between the
        intervals. These times include Python work and synchronization cost,
        not just kernel elapsed time, and must not replace uninstrumented QPS.
        """
        self.synchronize()
        finished = time.perf_counter()
        stages[name] = finished-started
        return finished

    @torch.inference_mode()
    def admit(self, request_id, prefix_ids, document_ids, *, max_suffix_tokens, query_ids=None):
        self._check()
        if not isinstance(request_id, str) or not request_id or request_id in self.requests:
            raise ValueError("request ID must be new and nonempty")
        prefix = self._ids(prefix_ids, nonempty=False)
        doc_ids = tuple(document_ids)
        if not doc_ids or any(x not in self.documents for x in doc_ids):
            raise ValueError("all nonempty document occurrences must be registered")
        if type(max_suffix_tokens) is not int or max_suffix_tokens <= 0:
            raise ValueError("max_suffix_tokens must be positive")
        if self.requires_recovery_query:
            if query_ids is None:
                raise ValueError("query_attention admission requires actual query_ids")
            query_ids = self._ids(query_ids)
            if len(query_ids) > max_suffix_tokens:
                raise ValueError("query probe exceeds reserved suffix length")
        elif query_ids is not None:
            raise ValueError("query_ids are only used by active query_attention recovery")
        docs = [self.documents[x] for x in doc_ids]
        offsets, total = [], len(prefix)
        for doc in docs:
            offsets.append(total)
            total += doc.doc_len
        if total + max_suffix_tokens > self.model.config.max_position_embeddings:
            raise ValueError("request plus reserved suffix exceeds model position limit")
        # Conservative exact main allocation plus one possible boundary per
        # visible interval/occurrence. It never admits beyond the bounded slab.
        pages = self.num_layers*self.kv_heads*math.ceil(max_suffix_tokens/self.page_size)
        for row in self.policies:
            for p in row:
                if p.head_type != "local":
                    pages += math.ceil(total/self.page_size)
                else:
                    pages += math.ceil(len(prefix)/self.page_size)
                    if not self.share_local:
                        pages += sum(math.ceil(d.doc_len/self.page_size) for d in docs)
                    for lo, hi in self._visible(total, len(prefix), p):
                        for off, length in [(0, len(prefix))] + list(zip(offsets, [d.doc_len for d in docs])):
                            a, b = max(lo, off)-off, min(hi, off+length)-off
                            if a < b and a % self.page_size:
                                pages += 1
        if pages > self.pool.stats()["free_pages"]:
            raise CapacityError(f"request requires {pages} additional pages; free={self.pool.stats()['free_pages']}")
        rid = self.namespace + ":request:" + request_id
        context = hashlib.sha256(repr((prefix, doc_ids, tuple(offsets))).encode()).hexdigest()
        query_hash = query_token_sha256(query_ids) if self.requires_recovery_query else None
        if query_hash is not None:
            # A selection conditioned on a different question cannot present
            # the same context/provenance identity, even for identical docs.
            context = hashlib.sha256(repr((context, query_hash)).encode()).hexdigest()
        state = RequestState(request_id, rid, total, prefix, doc_ids, tuple(offsets), max_suffix_tokens)
        if self.requires_recovery_query:
            state.recovery_query_ids = query_ids
        started = time.perf_counter()
        self._root(rid, context)
        try:
            tail_count = math.ceil(max_suffix_tokens/self.page_size)
            # One arena owns every layer/head suffix. Reserving and pinning
            # once avoids a full pool/event scan for each individual head.
            # Keep the whole reservation discoverable even if constructing a
            # later head's row descriptor fails before admission publishes.
            state.tail_all_refs = self.pool.reserve(self.num_layers*self.kv_heads*tail_count)
            self.pool.pin(state.tail_all_refs)
            state.tail_pinned = True
            for l in range(self.num_layers):
                for h in range(self.kv_heads):
                    first = (l*self.kv_heads+h)*tail_count
                    refs = state.tail_all_refs[first:first+tail_count]
                    state.tail_refs[l, h] = refs
                    rows = [ref.slot*self.page_size + j for ref in refs for j in range(self.page_size)]
                    state.tail_rows[l, h] = torch.tensor(rows[:max_suffix_tokens], device=self.device, dtype=torch.long)
            self.synchronize()
            prefix_started = time.perf_counter()
            prefix_kv = self._prefill_ids(prefix)
            self.synchronize()
            prefix_prefill_s = time.perf_counter()-prefix_started
            prefix_segment = self._segment("private-prefix", prefix, prefix_kv)
            doc_segments = [self._segment(d.doc_id, d.token_ids, d.recovery_kv, d.recovery_hidden) for d in docs]
            recovery_started = time.perf_counter()
            recovered = (self._recover_online(prefix_segment, doc_segments, query_ids)
                         if self.requires_recovery_query else self._recover_online(prefix_segment, doc_segments))
            self._assert_rope_unchanged()
            self.synchronize()
            online_recovery_s = time.perf_counter()-recovery_started
            publish_started = time.perf_counter()
            if self.publication_diagnostics:
                publication_stages = {}
                publication_checkpoint = publish_started
            writes = []
            global_positions = tuple(range(total))
            prefix_positions = global_positions[:len(prefix)]
            approximate_global = self._last_recovery_stats["recovery_approximate"]
            recovery_certificate = ("block-selective-v1:" + hashlib.sha256(repr((
                context, self.selective_recovery.to_dict(),
                self._last_recovery_stats.get("selected_block_ids"))).encode()).hexdigest()
                if approximate_global else "")
            if approximate_global and query_hash is not None:
                tag = "hidden-residual-int8-v4:" if self._hidden_residual_active else "query-selective-v3:"
                recovery_certificate = tag + hashlib.sha256(
                    repr((recovery_certificate, query_hash)).encode()).hexdigest()
            for l, row in enumerate(self.policies):
                for h, policy in enumerate(row):
                    if policy.head_type != "local":
                        k = torch.cat([part[l][0][0, h] for part in recovered], 0)
                        v = torch.cat([part[l][1][0, h] for part in recovered], 0)
                        writes.append(SegmentWrite((l, h, "global"), k, v, global_positions, context,
                            position_basis="rope-global",
                            reuse_kind="policy_approximate" if approximate_global else "exact_context",
                            validity_certificate=recovery_certificate))
                    elif prefix:
                        writes.append(SegmentWrite((l, h, "prefix"), prefix_kv[l][0][0, h], prefix_kv[l][1][0, h],
                                                   prefix_positions, context, position_basis="rope-global"))
            if self.publication_diagnostics:
                publication_checkpoint = self._publication_checkpoint(
                    publication_stages, "head_pack", publication_checkpoint)
            if writes:
                self.manager.update(rid, writes=writes, batched_writes=True)
            del writes, recovered, prefix_kv, prefix_segment, doc_segments
            if self.publication_diagnostics:
                publication_checkpoint = self._publication_checkpoint(
                    publication_stages, "head_write_and_release", publication_checkpoint)
            publication_stats = self._publish_document_history(rid, docs, offsets, context)
            if self.publication_diagnostics:
                publication_checkpoint = self._publication_checkpoint(
                    publication_stages, "alias_publish", publication_checkpoint)
            self._build_partitions(state)
            if self.publication_diagnostics:
                publication_checkpoint = self._publication_checkpoint(
                    publication_stages, "boundary_partition", publication_checkpoint)
            # The historical root is now final: no later forward publishes or
            # mutates it. One request-lifetime lease pins this immutable page
            # set, avoiding a full ownership/event scan on every decode token.
            state.history_lease = self.manager.bind(rid)
            self.history_lease_acquisitions += 1
            self.synchronize()
            if self.publication_diagnostics:
                publication_finished = time.perf_counter()
                publication_stages["final_bind"] = publication_finished-publication_checkpoint
                publication_stats["publication_diagnostics"] = dict(
                    schema="redknot-publication-stages-v1", diagnostic_only=True,
                    clock="host_perf_counter_with_completion_sync",
                    device_synchronizes=self.device.type == "cuda",
                    extra_synchronization_calls=4,
                    stage_seconds=publication_stages,
                    total_seconds=publication_finished-publish_started,
                    scope="head_pack includes provenance; head_write_and_release includes transient KV release; alias_publish includes private-copy ablation when share_local=False; boundary_partition includes boundary publication; final_bind includes existing final sync",
                    performance_caveat="diagnostic synchronization changes overlap and wall time; not an uninstrumented QPS result")
            state.admission_stats = dict(host_wall_s=time.perf_counter()-started, required_pages=pages,
                prefix_prefill_s=prefix_prefill_s, online_recovery_s=online_recovery_s,
                kv_publish_and_partition_s=time.perf_counter()-publish_started,
                stage_timing_synchronizes=True, online_recovery_tokens=total-len(prefix),
                private_prefix_tokens=len(prefix), document_projected_rows_per_layer=total-len(prefix),
                online_projection_policy="all token rows/all QKV heads; retain global KV afterward",
                private_local_copy_compacted=False, pool_after_admit=self.pool.stats(), **publication_stats,
                **self._last_recovery_stats)
            if self._last_recovery_stats["recovery_approximate"]:
                state.admission_stats["online_projection_policy"] = (
                    "full probe layers then Q/attention/O/FFN selected, K/V and input norm all rows; hidden residual transport"
                    if self._hidden_residual_active else
                    "full probe layers then fixed selected document rows/all QKV heads; private repaired global KV")
                # Preserve the legacy full-row scalar only for full recovery.
                state.admission_stats["document_projected_rows_per_layer"] = None
            self.requests[request_id] = state
            self._plan_cache.clear()
            self.admitted_requests += 1
            self.max_active_requests = max(self.max_active_requests, len(self.requests))
            return state
        except BaseException as original_error:
            try:
                self.synchronize()
                self._complete_history(state, ImmediateEvent())
                if state.tail_pinned:
                    self.pool.unpin(state.tail_all_refs)
                    state.tail_pinned = False
                if state.tail_all_refs:
                    self.pool.release(state.tail_all_refs)
                    state.tail_all_refs = ()
                self.manager.release_request(rid)
                self.pool.collect()
            except BaseException as cleanup_error:
                # Retain a discoverable root/arena if lifetime completion is
                # uncertain. Recycling pages after an event failure is unsafe.
                state.failed = True
                self.requests[request_id] = state
                self.pool.poisoned = True
                original_error.add_note(f"Admission cleanup quarantined: {cleanup_error}")
            raise

    def _build_partitions(self, state):
        rid = state.manager_request_id
        boundary_writes, pending = [], {}
        with self.manager.bind(rid) as lease:
            for l, row in enumerate(self.policies):
                for h, policy in enumerate(row):
                    if policy.head_type != "local":
                        key = (l, h, "global")
                        s = lease.version.segments[key]
                        pending[l, h] = [HistoricalPartition(key, s.pages, 0, 0, s.length, False)]
                        continue
                    parts = []
                    sources = ([("prefix", 0, len(state.prefix_ids), False)] if state.prefix_ids else [])
                    sources += [(f"document:{i}", off, self.documents[d].doc_len, True)
                                for i, (d, off) in enumerate(zip(state.document_ids, state.document_offsets))]
                    for lo, hi in self._visible(state.total_context_tokens, len(state.prefix_ids), policy):
                        for name, offset, length, canonical in sources:
                            a, b = max(lo, offset)-offset, min(hi, offset+length)-offset
                            if a >= b:
                                continue
                            key = (l, h, name)
                            segment = lease.version.segments[key]
                            if a % self.page_size:
                                stop = min(b, (a//self.page_size+1)*self.page_size)
                                source = segment.pages[a//self.page_size]
                                extent = min(self.page_size, length-(a//self.page_size)*self.page_size)
                                pk, pv = self.pool.read_page(source, extent)
                                ix = a % self.page_size
                                bkey = (l, h, f"boundary:{len(boundary_writes)}")
                                boundary_writes.append(SegmentWrite(bkey, pk[ix:ix+stop-a], pv[ix:ix+stop-a],
                                    segment.positions[a:stop], segment.provenance, position_basis=segment.position_basis,
                                    reuse_kind=segment.reuse_kind, validity_certificate=segment.validity_certificate))
                                parts.append((bkey, offset if canonical else 0, stop-a, canonical, a))
                                self.boundary_copy_bytes += 2*(stop-a)*self.head_dim*self.pool.k.element_size()
                                a = stop
                            if a < b:
                                pages = segment.pages[a//self.page_size:math.ceil(b/self.page_size)]
                                parts.append(HistoricalPartition(key, pages, offset if canonical else 0, 0, b-a, canonical, a))
                    pending[l, h] = parts
        if boundary_writes:
            self.manager.update(rid, writes=boundary_writes, batched_writes=True)
        version = self.manager.version(rid)
        for lh, parts in pending.items():
            state.partitions[lh] = [p if isinstance(p, HistoricalPartition) else HistoricalPartition(
                p[0], version.segments[p[0]].pages, p[1], 0, p[2], p[3], p[4]) for p in parts]

    def _counterrotate(self, query, offset):
        if not offset:
            return query
        phase = -float(offset)*self._inv_freq
        phase = torch.cat((phase, phase))
        # YaRN's amplitude was already applied to both native Q and K. This
        # extra coordinate change is a UNIT rotation, never a second scaling.
        cos, sin = phase.cos().to(query.dtype), phase.sin().to(query.dtype)
        a, b = query.chunk(2, -1)
        return query*cos + torch.cat((-b, a), -1)*sin

    def _torch_attention(self, q, states, layer, scale):
        B, _, T, D = q.shape
        result = torch.empty_like(q)
        for b, state in enumerate(states):
            for h in range(self.kv_heads):
                query = q[b, h*self.groups:(h+1)*self.groups]
                maximum = torch.full((self.groups, T), -torch.inf, device=self.device)
                denominator = torch.zeros_like(maximum)
                numerator = torch.zeros((self.groups, T, D), dtype=torch.float32, device=self.device)
                chunks = [(p, False) for p in state.partitions[layer, h]]
                chunks += [(HistoricalPartition((), state.tail_refs[layer, h], 0, 0, state.suffix_tokens+T, False), True)]
                for part, suffix in chunks:
                    pq = self._counterrotate(query, part.offset).float()
                    for i, ref in enumerate(part.pages):
                        count = min(self.page_size, part.end-i*self.page_size)
                        if count <= 0:
                            break
                        if suffix:
                            key, value = self.pool.k[ref.slot, :count], self.pool.v[ref.slot, :count]
                        else:
                            key, value = self.pool.read_page(ref, count)
                        scores = pq @ key.float().T * scale
                        if suffix:
                            mask = torch.arange(i*self.page_size, i*self.page_size+count, device=self.device)[None, :] <= (
                                state.suffix_tokens + torch.arange(T, device=self.device))[:, None]
                            scores = scores.masked_fill(~mask, -torch.inf)
                        local_max = scores.amax(-1)
                        merged = torch.maximum(maximum, local_max)
                        old_weight = torch.exp(maximum-merged).nan_to_num(0.)
                        weights = torch.exp(scores-merged[..., None]).nan_to_num(0.)
                        numerator = numerator*old_weight[..., None] + weights @ value.float()
                        denominator = denominator*old_weight + weights.sum(-1)
                        maximum = merged
                result[b, h*self.groups:(h+1)*self.groups] = (numerator/denominator[..., None]).to(q.dtype)
        return result

    def _flash_plan(self, states, layer):
        """Build immutable tables once per ordered active cohort and layer."""
        cache_key = (tuple(s.request_id for s in states), layer)
        if cache_key in self._plan_cache:
            return self._plan_cache[cache_key]
        B = len(states)
        page_rows, lengths, owners, offsets = [], [], [], []
        for b, state in enumerate(states):
            for h in range(self.kv_heads):
                for p in state.partitions[layer, h]:
                    page_rows.append([r.slot for r in p.pages])
                    lengths.append(p.end)
                    owners.append(b*self.kv_heads+h)
                    offsets.append(p.offset)
        table = torch.zeros((len(page_rows), max(map(len, page_rows), default=1)), device=self.device, dtype=torch.int32)
        for i, row in enumerate(page_rows):
            table[i, :len(row)] = torch.tensor(row, device=self.device, dtype=torch.int32)
        tails = [state.tail_refs[layer, h] for state in states for h in range(self.kv_heads)]
        tail_table = torch.zeros((B*self.kv_heads, max(map(len, tails))), device=self.device, dtype=torch.int32)
        append_rows = torch.zeros((B*self.kv_heads, max(s.max_suffix_tokens for s in states)), device=self.device, dtype=torch.long)
        for i, refs in enumerate(tails):
            tail_table[i, :len(refs)] = torch.tensor([r.slot for r in refs], device=self.device, dtype=torch.int32)
            state, head = states[i//self.kv_heads], i % self.kv_heads
            append_rows[i, :state.max_suffix_tokens] = state.tail_rows[layer, head]
        phase = -torch.tensor(offsets, device=self.device, dtype=torch.float32)[:, None] * self._inv_freq[None]
        phase = torch.cat((phase, phase), -1)[:, None, None, :]
        plan = dict(history_table=table, history_lengths=torch.tensor(lengths, device=self.device, dtype=torch.int32),
                    history_owner=torch.tensor(owners, device=self.device, dtype=torch.long),
                    rotation_cos=phase.cos().to(self.dtype), rotation_sin=phase.sin().to(self.dtype),
                    tail_table=tail_table, append_rows=append_rows,
                    all_owner=torch.tensor(owners+list(range(B*self.kv_heads)), device=self.device, dtype=torch.long))
        # A bounded cache handles the scheduler's query/decode groups. Release
        # and admission invalidate it; no descriptor outlives request ownership.
        if len(self._plan_cache) >= 4*self.num_layers:
            self._plan_cache.clear()
        self._plan_cache[cache_key] = plan
        return plan

    def _flash_attention(self, q, states, layer, scale):
        B, _, T, D = q.shape
        plan = self._flash_plan(states, layer)
        query = q.reshape(B, self.kv_heads, self.groups, T, D).permute(0, 1, 3, 2, 4).reshape(B*self.kv_heads, T, self.groups, D).contiguous()
        outputs, lses = [], []
        if plan["history_owner"].numel():
            history_query = query.index_select(0, plan["history_owner"])
            a, b = history_query.chunk(2, -1)
            history_query = history_query*plan["rotation_cos"] + torch.cat((-b, a), -1)*plan["rotation_sin"]
            hist, lse = self._flash(history_query.contiguous(), self.pool.k.unsqueeze(2), self.pool.v.unsqueeze(2),
                k=None, v=None, block_table=plan["history_table"], cache_seqlens=plan["history_lengths"],
                softmax_scale=scale, causal=False, return_softmax_lse=True)
            outputs.append(hist.float())
            lses.append(lse.transpose(1, 2).float())
        suffix, lse = self._flash(query, self.pool.k.unsqueeze(2), self.pool.v.unsqueeze(2), k=None, v=None,
            block_table=plan["tail_table"], cache_seqlens=self._current_tail_lengths,
            softmax_scale=scale, causal=True, return_softmax_lse=True)
        outputs.append(suffix.float())
        lses.append(lse.transpose(1, 2).float())
        values, logz = torch.cat(outputs), torch.cat(lses)
        owner = plan["all_owner"]
        ix = owner[:, None, None].expand(-1, T, self.groups)
        maxz = torch.full((B*self.kv_heads, T, self.groups), -torch.inf, device=self.device)
        maxz.scatter_reduce_(0, ix, logz, reduce="amax", include_self=True)
        weight = torch.exp(logz-maxz[owner]).nan_to_num(0.)
        denom = torch.zeros_like(maxz).index_add_(0, owner, weight)
        numer = torch.zeros((*maxz.shape, D), device=self.device).index_add_(0, owner, values*weight[..., None])
        merged = (numer/denom[..., None]).to(q.dtype)
        return merged.reshape(B, self.kv_heads, T, self.groups, D).permute(0, 1, 3, 2, 4).reshape(B, self.query_heads, T, D)

    def _attention_forward(self, attention, hidden_states, position_embeddings,
                           attention_mask=None, past_key_values=None, cache_position=None, **kwargs):
        if not self._busy or past_key_values is not None:
            raise RuntimeError("paged attention invoked outside the controlled forward")
        q, k, v = self._project(attention, hidden_states, position_embeddings)
        layer = attention.layer_idx
        T = hidden_states.shape[1]
        if self.backend == "flash":
            rows = self._flash_plan(self._current_states, layer)["append_rows"].gather(1, self._current_tail_cols).reshape(-1)
        else:
            rows = torch.cat([state.tail_rows[layer, h][state.suffix_tokens:state.suffix_tokens+T]
                              for state in self._current_states for h in range(self.kv_heads)])
        self.pool.k.view(-1, self.head_dim).index_copy_(0, rows, k.reshape(-1, self.head_dim))
        self.pool.v.view(-1, self.head_dim).index_copy_(0, rows, v.reshape(-1, self.head_dim))
        self.pool.written_bytes += 2*rows.numel()*self.head_dim*self.pool.k.element_size()
        output = (self._flash_attention if self.backend == "flash" else self._torch_attention)(
            q, self._current_states, layer, attention.scaling)
        self.attention_calls += 1
        output = output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
        return attention.o_proj(output), None

    @torch.inference_mode()
    def forward(self, request_ids, input_ids):
        self._check()
        names = tuple(request_ids)
        if not names or len(set(names)) != len(names) or any(x not in self.requests for x in names):
            raise ValueError("request IDs must be unique active requests")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a BxT torch.long tensor")
        if input_ids.device != self.device or input_ids.shape[0] != len(names) or input_ids.shape[1] < 1:
            raise ValueError("input batch/device/length mismatch")
        if bool(((input_ids < 0) | (input_ids >= self.model.config.vocab_size)).any()):
            raise ValueError("input token ID outside vocabulary")
        states, T = [self.requests[x] for x in names], input_ids.shape[1]
        if any(s.failed or s.suffix_tokens+T > s.max_suffix_tokens for s in states):
            raise ValueError("failed request or suffix capacity exceeded")
        for b, state in enumerate(states):
            # The admission probe must describe the query actually executed.
            # Validate its initial span, including callers that chunk a query;
            # later generated tokens have no such constraint. Reject before
            # any suffix-page writes, so a mismatched call can be corrected.
            start = state.suffix_tokens
            count = min(T, max(0, len(state.recovery_query_ids)-start))
            if count and tuple(input_ids[b, :count].tolist()) != state.recovery_query_ids[start:start+count]:
                raise ValueError("initial forward tokens differ from recovery query_ids")
        for state in states:
            lease = state.history_lease
            if (lease is None or lease.closed or state.history_completed
                    or self.manager.version(state.manager_request_id) is not lease.version):
                state.failed = True
                raise RuntimeError("immutable request history lease/root changed after admission")
        if getattr(self.model, "_redknot_nonprefix_owner", None) is not None:
            raise RuntimeError("model already has an active RedKnot binding")
        before = torch.tensor([s.suffix_tokens for s in states for _ in range(self.kv_heads)], device=self.device, dtype=torch.long)
        self._current_tail_cols = before[:, None] + torch.arange(T, device=self.device)[None]
        self._current_tail_lengths = (before+T).to(torch.int32)
        originals = []
        self._busy = True
        self._current_states = states
        self.model._redknot_nonprefix_owner = self
        try:
            for layer in self.layers:
                attention = layer.self_attn
                originals.append((attention, attention.forward))
                def patched(module, *args, **kwargs):
                    return self._attention_forward(module, *args, **kwargs)
                attention.forward = types.MethodType(patched, attention)
            positions = torch.tensor([s.total_context_tokens+s.suffix_tokens for s in states], device=self.device)[:, None]
            positions = positions + torch.arange(T, device=self.device)[None]
            out = self.model(input_ids=input_ids, position_ids=positions, use_cache=False, logits_to_keep=1,
                             attention_mask={"full_attention": None, "sliding_attention": None})
            logits = out.logits
            self._assert_rope_unchanged()
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("non-finite paged model logits")
            for state in states:
                state.suffix_tokens += T
            return logits
        except BaseException:
            for state in states:
                state.failed = True
            raise
        finally:
            try:
                for attention, original in originals:
                    attention.forward = original
            finally:
                if getattr(self.model, "_redknot_nonprefix_owner", None) is self:
                    delattr(self.model, "_redknot_nonprefix_owner")
                self._current_states = None
                self._busy = False

    def _complete_history(self, state, event):
        lease = state.history_lease
        if lease is None or state.history_completed:
            return
        if state.history_completion_uncertain or lease.closed:
            # ReadLease.closed can precede an unpin failure. Without a
            # successful complete() return we cannot infer whether its pin was
            # removed; never guess or double-unpin shared physical pages.
            state.failed = True
            state.history_completion_uncertain = True
            self.pool.poisoned = True
            raise RuntimeError("historical lease completion uncertain; root remains quarantined")
        try:
            lease.complete(event)
        except BaseException:
            state.failed = True
            state.history_release_failed = True
            state.history_completion_uncertain = lease.closed
            self.pool.poisoned = True
            raise
        state.history_completed = True
        self.history_lease_completions += 1

    def release(self, request_id):
        self._check(for_cleanup=True)
        state = self.requests[request_id]
        if state.history_release_failed:
            self.synchronize()
        # If CUDA event creation fails, keep the entire request discoverable
        # with its ownership intact, so synchronized teardown can retry.
        try:
            event = completion_event(self.device)
        except BaseException:
            state.failed = True
            self.pool.poisoned = True
            raise
        self._complete_history(state, event)
        self._plan_cache.clear()
        if state.tail_pinned:
            self.pool.unpin(state.tail_all_refs, event)
            state.tail_pinned = False
        if state.tail_all_refs:
            self.pool.release(state.tail_all_refs, event)
            state.tail_all_refs = ()
        self.manager.release_request(state.manager_request_id)
        del self.requests[request_id]
        self.pool.collect()

    def stats(self):
        descriptor_storages = {}
        tensors = [v for plan in self._plan_cache.values() for v in plan.values()]
        tensors += [v for state in self.requests.values() for v in state.tail_rows.values()]
        for tensor in tensors:
            storage = tensor.untyped_storage()
            descriptor_storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        result = dict(backend=self.backend, experimental_nonprefix_redknot=True, share_local=self.share_local,
            selective_recovery=self.selective_recovery.to_dict() if self.selective_recovery else None,
            selective_execution_optimization=self.selective_execution_optimization,
            offline_kv_cache_bytes=self._offline_kv_cache_bytes,
            offline_kv_cache_limit_bytes=(self.selective_recovery.max_offline_kv_bytes
                if self.selective_recovery and self.selective_recovery.recovery_mode == "offline_kv" else 0),
            offline_kv_cache_peak_bytes=self._offline_kv_cache_peak_bytes,
            offline_kv_memory_scope="retained full canonical KV outside page slab; slab/local duplicates charged separately; transient peak excluded",
            offline_hidden_cache_bytes=self._offline_hidden_cache_bytes,
            offline_hidden_cache_limit_bytes=(self.selective_recovery.max_offline_kv_bytes if self._hidden_residual_active else 0),
            offline_hidden_cache_peak_bytes=self._offline_hidden_cache_peak_bytes,
            offline_hidden_quantization=({"dtype": "int8", "group_size": GROUP_SIZE, "scale_dtype": "float32"}
                                         if self._hidden_residual_active else None),
            offline_hidden_memory_scope="retained layer-input int8 values and FP32 scales outside page slab; local KV charged inside slab; native offline-build KV and online transient peak excluded",
            bulk_head_alias_publication=True,
            multi_document_alias_publication=True,
            historical_lease_policy="one immutable root lease per admitted request; completion at release",
            history_lease_acquisitions=self.history_lease_acquisitions,
            history_lease_completions=self.history_lease_completions,
            active_history_leases=sum(s.history_lease is not None and not s.history_lease.closed for s in self.requests.values()),
            pool=self.pool.stats(), active_requests=len(self.requests), documents=len(self.documents),
            max_active_requests=self.max_active_requests, admitted_requests=self.admitted_requests,
            local_head_ratio=sum(p.head_type == "local" for row in self.policies for p in row)/(self.num_layers*self.kv_heads),
            attention_calls=self.attention_calls, boundary_copy_bytes=self.boundary_copy_bytes,
            descriptor_unique_storage_bytes=sum(descriptor_storages.values()),
            cached_layer_descriptors=len(self._plan_cache),
            rope_type=self._rope_type, rope_scaling=copy.deepcopy(self._rope_config[0]["rope_scaling"]),
            rope_attention_scaling=self._rope_attention_scaling,
            offset_rotation_amplitude=1.0, rope_parameters_immutable=True,
            historical_kv_gather_during_decode=False, canonical_query_counterrotation=True,
            local_decode_policy="fixed context sink+recent-window union, plus causal query/generated suffix",
            tail_policy="exclusive pinned append-only; single CUDA stream; no reset",
            reuse_certificate="policy provenance only; no task-accuracy guarantee",
            recovery_policy=("opt-in hidden residual int8 transport with full K/V projection; heuristic, no task-quality certificate"
                if self._hidden_residual_active else
                "opt-in block-selective canonical KV repair; heuristic, no task-quality certificate"
                if self._selective_active else
                "flat recovery with disjoint per-head sink/window; no legacy decode bypass"))
        collection_stats = getattr(self.pool, "collection_stats", None)
        if callable(collection_stats):
            # This diagnostic accessor is read-only: no additional collection
            # passes or CUDA event queries are introduced for measurement.
            result["pool_collection_stats"] = collection_stats()
        if self.metadata_only_alias_sources:
            result["metadata_only_alias_sources"] = True
        return result

    def close(self):
        if self._closed:
            return
        self._check(for_cleanup=True)
        self.synchronize()
        for request_id in list(self.requests):
            self.release(request_id)
        for doc in self.documents.values():
            self.manager.release_request(doc.source_request_id)
        self.documents.clear()
        self._offline_kv_cache_bytes = 0
        self._offline_hidden_cache_bytes = 0
        self.pool.collect()
        self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()
