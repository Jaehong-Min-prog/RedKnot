"""Opt-in, single-process SGLang request lifecycle for managed head pages.

This is the bounded *live* MHA/GQA path, not the non-prefix document importer.
The real Scheduler retains its token accounting/ChunkCache; those token slots
are logical admission credits only. Attention owns one HeadPagePool instead of
also allocating the ordinary dense K/V tensors. No request or pointer may cross
a process boundary: overlap, distributed execution, radix caching and graphs
are deliberately rejected until their lifecycle protocols are implemented.
"""
from __future__ import annotations

import functools
import os
import uuid
from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.head_kv.manager import HeadKVManager
from sglang.srt.mem_cache.head_kv.pool import HeadPagePool


def validate_managed_server_args(args):
    """Fail closed, with no implicit rewrites of unrelated serving options."""
    if not getattr(args, "redknot_managed_kv", False):
        return
    required = {
        "attention_backend": "redknot", "tp_size": 1, "pp_size": 1,
        "dp_size": 1, "attn_cp_size": 1, "ep_size": 1, "moe_dp_size": 1,
        "page_size": 1, "disable_radix_cache": True,
        "disable_cuda_graph": True, "disable_piecewise_cuda_graph": True,
        "disable_overlap_schedule": True, "disaggregation_mode": "null",
        "schedule_policy": "fcfs",
    }
    errors = [f"{name}={expected!r} required"
              for name, expected in required.items()
              if getattr(args, name, None) != expected]
    forbidden = (
        "enable_lora", "enable_dp_attention", "enable_two_batch_overlap",
        "enable_pdmux", "enable_hierarchical_cache", "enable_hisparse",
        "enable_lmcache", "enable_streaming_session", "enable_memory_saver",
        "enable_torch_compile", "is_embedding", "prefill_only_disable_kv_cache",
        "enable_multimodal", "enable_mixed_chunk", "enable_dynamic_chunking",
        "enable_mis", "enable_symm_mem", "enable_torch_symm_mem",
        "enable_elastic_expert_backup", "enable_eplb", "enable_quant_communications",
        "disaggregation_decode_enable_offload_kvcache",
        "enable_redknot_sparse_moe", "redknot_sparse_ffn_enable",
    )
    errors.extend(f"{name} unsupported" for name in forbidden if getattr(args, name, False))
    for name in ("speculative_algorithm", "quantization", "decode_attention_backend",
                 "prefill_attention_backend", "hicache_storage_backend", "elastic_ep_backend",
                 "rl_on_policy_target", "redknot_progressive_topk_schedule", "dllm_algorithm"):
        if getattr(args, name, None):
            errors.append(f"{name} unsupported")
    if getattr(args, "cpu_offload_gb", 0):
        errors.append("CPU weight offload unsupported")
    if getattr(args, "device", None) not in (None, "cuda"):
        errors.append("CUDA serving required")
    if getattr(args, "kv_cache_dtype", "auto") not in ("auto", "bfloat16", "float16"):
        errors.append("unquantized FP16/BF16 KV required")
    page_size = getattr(args, "redknot_page_size", None)
    if type(page_size) is not int or page_size <= 0:
        errors.append("positive redknot_page_size required")
    if errors:
        raise ValueError("--redknot-managed-kv: " + "; ".join(errors))


@dataclass(frozen=True)
class ManagedBudget:
    logical_tokens: int
    capacity_pages: int
    reserved_tail_tokens: int


def managed_budget(token_budget, max_requests, page_size, layers, kv_heads):
    """Reserve bounded request-tail fragmentation and one layer's COW scratch.

    Logical credits charge every request's whole sequence, including any shared
    bytes. This first integration does not over-admit based on estimated sharing.
    A full-layer page of extra scratch is conservatively reserved for all layers.
    All previous GPU work is retired at the eager step boundary (see bind_batch).
    """
    if any(type(x) is not int or x <= 0
           for x in (token_budget, max_requests, page_size, layers, kv_heads)):
        raise ValueError("positive integer managed pool dimensions required")
    physical_tokens = token_budget // page_size * page_size
    reserve = max_requests * (page_size - 1) + page_size
    logical = physical_tokens - reserve
    if logical <= 0:
        raise ValueError("KV budget cannot cover request tail/COW reserve; reduce max-running-requests")
    return ManagedBudget(logical, physical_tokens // page_size * layers * kv_heads, reserve)


def prepare_managed_budget(runner):
    """Called before any production KV slab/allocator is allocated."""
    args = runner.server_args
    validate_managed_server_args(args)
    config = runner.model_config
    if (getattr(config.hf_config, "model_type", None) != "qwen3"
            or getattr(runner, "is_draft_worker", False)
            or getattr(runner, "mambaish_config", None)
            or getattr(runner, "is_hybrid_swa", False)
            or getattr(runner, "use_mla_backend", False)
            or getattr(config, "is_encoder_decoder", False)
            or getattr(config, "is_multimodal", False)
            or getattr(config.hf_config, "quantization_config", None)
            or getattr(runner, "model_is_mrope", False)
            or not getattr(runner, "is_generation", True)
            or runner.start_layer != 0
            or runner.num_effective_layers != config.hf_config.num_hidden_layers
            or config.head_dim != config.v_head_dim
            or runner.kv_cache_dtype not in (torch.float16, torch.bfloat16)):
        raise ValueError("managed serving currently supports unquantized, full-layer Qwen3 MHA/GQA only")
    budget = managed_budget(int(runner.max_total_num_tokens), int(runner.max_running_requests),
                            args.redknot_page_size, int(runner.num_effective_layers),
                            int(config.get_num_kv_heads(1)))
    runner.redknot_managed_budget = budget
    runner.max_total_num_tokens = budget.logical_tokens


class ManagedTokenPoolFacade:
    """Scheduler metadata, never a second dense KV allocation or a dummy read.

    Logical token indexes are meaningful to the admission allocator/ChunkCache
    only. Every attempt to consume them as tensor addresses fails loudly.
    """
    def __init__(self, runner, manager):
        self.manager = manager
        self.size = runner.max_total_num_tokens
        self.page_size = 1
        self.device = runner.device
        self.dtype = self.store_dtype = runner.kv_cache_dtype
        self.layer_num = runner.num_effective_layers
        self.start_layer, self.end_layer = runner.start_layer, runner.end_layer
        self.head_num = runner.model_config.get_num_kv_heads(1)
        self.head_dim = self.v_head_dim = manager.pool.head_dim
        self.layer_transfer_counter = None
        self.mem_usage = sum(self.get_kv_size_bytes()) / (1 << 30)

    def get_kv_size_bytes(self):
        pool = self.manager.pool
        return pool.k.numel() * pool.k.element_size(), pool.v.numel() * pool.v.element_size()

    @staticmethod
    def _unsupported(*args, **kwargs):
        raise RuntimeError("managed head KV has no dense token-indexed buffer; use the managed attention backend")

    get_key_buffer = get_value_buffer = get_kv_buffer = set_kv_buffer = _unsupported
    move_kv_cache = get_cpu_copy = load_cpu_copy = get_contiguous_buf_infos = _unsupported
    register_layer_transfer_counter = _unsupported


def create_managed_token_pool(runner):
    """Install one physical head pool and a same-process request owner."""
    budget = runner.redknot_managed_budget
    pool = HeadPagePool(budget.capacity_pages, runner.server_args.redknot_page_size,
                        runner.model_config.head_dim, dtype=runner.kv_cache_dtype,
                        device=runner.device)
    manager = HeadKVManager(pool)
    runner.redknot_shared_kv_manager = manager
    runner.redknot_shared_kv_lifecycle = ManagedRequestLifecycle(manager)
    return ManagedTokenPoolFacade(runner, manager)


@dataclass(frozen=True)
class _Binding:
    owner: object
    handle: object


class ManagedRequestLifecycle:
    """Connect actual Req lifetimes to stable, generation-fenced page handles.

    New requests start empty. Chunk continuation and decode must retain the same
    Req object; release/retraction removes its binding, so resumed prefill gets
    a new generation. User-provided IDs/slot indexes are never cache identities.
    """
    def __init__(self, manager):
        from sglang.srt.layers.attention.redknot.shared_kv import SharedKVBackend

        self.backend = SharedKVBackend(manager)
        self.pid = os.getpid()
        self.boot = uuid.uuid4().hex
        self._requests = {}
        self.created_requests = 0
        self.released_requests = 0
        self.forward_batches = 0

    def _check_process(self):
        if os.getpid() != self.pid:
            raise RuntimeError("managed KV request owner cannot cross a process boundary")

    def bind_batch(self, batch, forward_batch):
        self._check_process()
        requests = list(batch.reqs)
        if len({id(req) for req in requests}) != len(requests):
            raise ValueError("duplicate Req object in managed batch")
        mode = forward_batch.forward_mode.name
        if mode not in ("EXTEND", "DECODE", "IDLE"):
            raise ValueError("managed lifecycle supports normal eager EXTEND/DECODE only")
        if len(requests) != forward_batch.batch_size:
            raise ValueError("request count differs from managed forward batch")
        prefixes = ([int(x) - 1 for x in forward_batch.seq_lens_cpu.tolist()]
                    if mode == "DECODE" else list(batch.prefix_lens or ()))
        if mode == "IDLE":
            if requests:
                raise ValueError("idle managed batch must be empty")
            forward_batch.redknot_shared_kv_handles = []
            return
        if len(prefixes) != len(requests):
            raise ValueError("managed request prefix metadata is missing")
        for req, prefix in zip(requests, prefixes):
            if (getattr(req, "redknot_offline_segments", None)
                    or getattr(req, "redknot_reuse_plan", None)
                    or getattr(req, "lora_id", None)
                    or getattr(req, "multimodal_inputs", None)):
                raise ValueError("managed live serving does not import non-prefix, adapter or multimodal state")
            binding = getattr(req, "_redknot_managed_binding", None)
            if binding is not None:
                if not isinstance(binding, _Binding) or binding.owner is not self:
                    raise RuntimeError("foreign managed request binding")
                if self._requests.get(binding.handle) is not req:
                    raise RuntimeError("stale managed request binding")
                self.backend._version(binding.handle)
            elif mode != "EXTEND" or prefix != 0:
                raise ValueError("managed request continuation has no live KV handle")

        # Eager, non-overlap protocol: finish readers/retirements before admitting
        # a new step. This conservative synchronization is intentional; removing
        # it requires an event-aware scheduler credit/physical reservation bridge.
        pool = self.backend.manager.pool
        if pool.device.type == "cuda":
            torch.cuda.current_stream(pool.device).synchronize()
        pool.collect()
        new_requests = []
        handles = []
        try:
            for req in requests:
                binding = getattr(req, "_redknot_managed_binding", None)
                if binding is None:
                    request_id = f"serve:{self.boot}:{uuid.uuid4().hex}"
                    handle = self.backend.create_request(
                        request_id, context_id=request_id,
                        contract=f"model-runner:{self.boot}:encoded-mha-v1",
                        namespace=f"isolated-request:{request_id}",
                    )
                    binding = _Binding(self, handle)
                    req._redknot_managed_binding = binding
                    self._requests[handle] = req
                    new_requests.append(req)
                    self.created_requests += 1
                handles.append(binding.handle)
        except BaseException:
            for req in new_requests:
                self.release(req)
            raise
        forward_batch.redknot_shared_kv_handles = handles
        self.forward_batches += 1

    def release(self, req):
        self._check_process()
        binding = getattr(req, "_redknot_managed_binding", None)
        if binding is None:
            return False
        if (not isinstance(binding, _Binding) or binding.owner is not self
                or self._requests.get(binding.handle) is not req):
            raise RuntimeError("foreign or stale managed request release")
        self.backend.release_request(binding.handle)
        del self._requests[binding.handle]
        del req._redknot_managed_binding
        self.released_requests += 1
        return True

    def stats(self):
        self._check_process()
        return dict(self.backend.manager.stats(),
                    live_serving_requests=len(self._requests),
                    created_serving_requests=self.created_requests,
                    released_serving_requests=self.released_requests,
                    forward_batches=self.forward_batches,
                    mode="experimental_single_gpu_live_only",
                    nonprefix_import=False, dense_kv_slab=False)

    def abort_forward(self, batch=None, forward_batch=None):
        """Partial model forwards are not retryable: discard every touched root."""
        self._check_process()
        requests = list(batch.reqs) if batch is not None else [
            self._requests[handle]
            for handle in (getattr(forward_batch, "redknot_shared_kv_handles", None) or ())
            if handle in self._requests
        ]
        for req in requests:
            binding = getattr(req, "_redknot_managed_binding", None)
            if binding is not None and binding.owner is self:
                self.release(req)

    def assert_idle(self):
        """Flush must not silently discard a leaked live request."""
        self._check_process()
        if self._requests or self.backend.manager.stats()["requests"]:
            raise RuntimeError("managed KV still has live requests at scheduler flush")
        pool = self.backend.manager.pool
        if pool.device.type == "cuda":
            torch.cuda.current_stream(pool.device).synchronize()
        pool.collect()


def release_managed_request(req):
    binding = getattr(req, "_redknot_managed_binding", None)
    if binding is not None:
        if not isinstance(binding, _Binding):
            raise RuntimeError("invalid managed request binding")
        binding.owner.release(req)


def managed_forward_guard(fn):
    """Wrap the real worker entry, including sampling and ForwardBatch creation."""
    @functools.wraps(fn)
    def wrapped(self, batch=None, forward_batch=None, *args, **kwargs):
        lifecycle = getattr(self.model_runner, "redknot_shared_kv_lifecycle", None)
        if lifecycle is None:
            return fn(self, batch, forward_batch, *args, **kwargs)
        try:
            return fn(self, batch, forward_batch, *args, **kwargs)
        except BaseException:
            # Never retry a partially appended layer/request. A cleanup failure
            # intentionally replaces the exception with its cause chained.
            lifecycle.abort_forward(batch, forward_batch)
            raise
    return wrapped


def reject_managed_weight_update(fn):
    """Live roots may not survive a weights/adapters identity change."""
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        if getattr(self.model_runner, "redknot_shared_kv_lifecycle", None) is not None:
            return False, "managed KV serving requires restart for weight updates"
        return fn(self, *args, **kwargs)
    return wrapped
