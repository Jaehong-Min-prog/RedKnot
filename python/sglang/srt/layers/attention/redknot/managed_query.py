"""Optional managed-paged bridge for the existing offline-KV query phase.

This is an integration experiment, not a scheduler replacement. Historical
local KV has *already* been selected by the driver's static sink/window rule.
Consequently all physically present prefix rows remain visible to every query
row, exactly as in ``_query_headclass_attention``; applying another moving
local window here would change the algorithm. Only the query suffix is causal.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from numbers import Integral

import torch

from sglang.srt.mem_cache.head_kv.manager import HeadKVManager, SegmentWrite
from sglang.srt.mem_cache.head_kv.pool import HeadPagePool
from sglang.srt.mem_cache.head_kv.segpaged import ManagedSegPagedKVCache


class ManagedHeadClassQueryCache:
    """One lazily allocated slab/request for every layer of one query forward.

    Prefix admission and suffix append are deliberately in the caller's online
    timer. The native RedKnot driver does not yet persist these manager pages
    offline, so hiding their conversion cost would misstate end-to-end TTFT.
    """

    def __init__(self, layer_kv, *, query_offset, query_length, page_size=64,
                 skip_invisible=True):
        for name, value in (("query_offset", query_offset),
                            ("query_length", query_length), ("page_size", page_size)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(skip_invisible, bool):
            raise TypeError("skip_invisible must be bool")
        if not layer_kv:
            raise ValueError("at least one layer is required")
        self.layer_kv = layer_kv
        self.query_offset, self.query_length = int(query_offset), int(query_length)
        self.page_size, self.skip_invisible = int(page_size), skip_invisible
        self.num_layers = len(layer_kv)
        first = layer_kv[0]["k_global"]
        if first.ndim != 4 or first.shape[0] != 1:
            raise ValueError("managed query backend requires batch-one MHA/GQA KV")
        _, self.num_kv_heads, _, self.head_dim = first.shape
        self.device, self.dtype = first.device, first.dtype
        self._local_heads = []
        capacity = 0
        for kv in layer_kv:
            for name in ("k_global", "v_global", "k_local", "v_local"):
                value = kv[name]
                if (value.ndim != 4 or value.shape[0] != 1
                        or value.shape[1] != self.num_kv_heads
                        or value.shape[3] != self.head_dim
                        or value.device != self.device or value.dtype != self.dtype):
                    raise ValueError("managed query backend requires one device/dtype and uniform KV topology")
            if kv["k_global"].shape != kv["v_global"].shape or kv["k_local"].shape != kv["v_local"].shape:
                raise ValueError("K/V shapes must match")
            if kv["k_global"].shape[2] != self.query_offset:
                raise ValueError("global prefix length must equal query_offset")
            if kv["is_local"].shape != (self.num_kv_heads,) or kv["is_local"].dtype != torch.bool:
                raise ValueError("is_local must be a bool vector over physical KV heads")
            local_heads = tuple(kv["is_local"].tolist())
            self._local_heads.append(local_heads)
            positions = tuple(kv["local_positions"])
            if (len(positions) != kv["k_local"].shape[2]
                    or len(set(positions)) != len(positions)
                    or any(isinstance(p, bool) or not isinstance(p, Integral)
                           or p < 0 or p >= self.query_offset for p in positions)):
                raise ValueError("local_positions must identify every selected prefix row")
            for local in local_heads:
                prefix_length = kv["k_local" if local else "k_global"].shape[2]
                # Prefix and query are separate occurrences: no historical
                # partial-tail COW is required for this single query forward.
                capacity += math.ceil(prefix_length / self.page_size)
                capacity += math.ceil(self.query_length / self.page_size)
        self.capacity_pages = max(1, capacity)
        self.request_id = "redknot-query-" + uuid.uuid4().hex
        self.manager = None
        self.cache = None
        self._prepared = set()
        self._consumed = set()
        self._closed = False
        self._policy_lineage = "caller-declared-policy:" + hashlib.sha256(json.dumps(
            {"local_heads": self._local_heads,
             "local_positions": [list(kv["local_positions"]) for kv in layer_kv]},
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        self.stats = dict(manager_update_calls=0, manager_append_many_calls=0,
                          managed_attention_calls=0, manager_prepare_wall_seconds=0.0,
                          manager_append_wall_seconds=0.0,
                          managed_attention_submit_wall_seconds=0.0,
                          query_page_capacity=self.capacity_pages)

    def _ensure_manager(self):
        if self._closed:
            raise RuntimeError("managed query cache is closed")
        if self.manager is None:
            pool = HeadPagePool(self.capacity_pages, self.page_size, self.head_dim,
                                dtype=self.dtype, device=self.device)
            self.manager = HeadKVManager(pool)
            self.manager.create_request(self.request_id, context_id=self.request_id,
                                        contract="redknot-static-headclass-query-v1")
            self.cache = ManagedSegPagedKVCache(self.manager, self.request_id,
                                               num_layers=self.num_layers,
                                               num_kv_heads=self.num_kv_heads)

    def prefill_layer(self, layer):
        if isinstance(layer, bool) or not isinstance(layer, Integral) or not 0 <= layer < self.num_layers:
            raise ValueError("layer outside query topology")
        if layer in self._prepared:
            raise RuntimeError("query prefix was already admitted for this layer")
        started = time.perf_counter()
        self._ensure_manager()
        kv = self.layer_kv[layer]
        global_positions = tuple(range(self.query_offset))
        local_positions = tuple(int(p) for p in kv["local_positions"])
        writes = []
        for head, local in enumerate(self._local_heads[layer]):
            suffix = "local" if local else "global"
            writes.append(SegmentWrite(
                (int(layer), head, "query-prefix"),
                kv[f"k_{suffix}"][0, head], kv[f"v_{suffix}"][0, head],
                local_positions if local else global_positions,
                provenance=f"{self.request_id}:layer-{layer}:{suffix}",
                position_basis="target-rope",
                reuse_kind="policy_approximate" if local else "exact_context",
                # This is a declared reuse-policy lineage, not a quality proof.
                validity_certificate=self._policy_lineage if local else "",
            ))
        self.manager.update(self.request_id, writes=writes)
        self._prepared.add(layer)
        self.stats["manager_update_calls"] += 1
        self.stats["manager_prepare_wall_seconds"] += time.perf_counter() - started

    def attention(self, query, k, v, *, layer, num_q_per_kv, sm_scale):
        if (query.ndim != 4 or query.shape[0] != 1
                or query.shape[1:] != (self.num_kv_heads * num_q_per_kv, self.query_length, self.head_dim)
                or k.shape != (1, self.num_kv_heads, self.query_length, self.head_dim)
                or k.shape != v.shape):
            raise ValueError("query/suffix tensors disagree with managed query topology")
        if layer in self._consumed:
            raise RuntimeError("one query forward per layer is supported")
        if layer not in self._prepared:
            self.prefill_layer(layer)
        positions = tuple(range(self.query_offset, self.query_offset + self.query_length))
        started = time.perf_counter()
        self.manager.append_many(self.request_id, appends=[
            SegmentWrite((int(layer), head, "query-suffix"), k[0, head], v[0, head],
                         positions, f"{self.request_id}:layer-{layer}:suffix",
                         position_basis="target-rope")
            for head in range(self.num_kv_heads)
        ])
        self.stats["manager_append_many_calls"] += 1
        self.stats["manager_append_wall_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        result = self.cache.attention(
            query[0], layer=layer,
            query_positions=torch.arange(self.query_offset, self.query_offset + self.query_length,
                                         device=self.device, dtype=torch.int64),
            num_q_per_kv=num_q_per_kv, sm_scale=sm_scale,
            use_fused=query.is_cuda, causal=True, skip_invisible=self.skip_invisible,
        )
        self._consumed.add(layer)
        self.stats["managed_attention_calls"] += 1
        self.stats["managed_attention_submit_wall_seconds"] += time.perf_counter() - started
        return result.unsqueeze(0)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.manager is not None:
            self.stats.update(query_pool_written_bytes=self.manager.pool.written_bytes,
                              query_pool_copied_bytes=self.manager.pool.copied_bytes,
                              query_pool_high_water_pages=self.manager.pool.high_water_pages)
            self.manager.release_request(self.request_id)
            self.cache = None
            self.manager = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
