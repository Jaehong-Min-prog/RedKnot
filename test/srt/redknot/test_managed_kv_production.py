"""Bounded native-lifecycle tests; NOT a substitute for a launched server test.

Optional heavyweight SGLang imports are isolated exactly as in the existing
backend tests. The manager, lifecycle, attention, facade, and release function
are real source. ScheduleBatch/Req data and legacy allocator calls are fixtures.
The release function is AST-loaded unchanged to avoid optional server packages.
"""
import ast
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from test_shared_kv_backend import _isolated_backend_imports


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def api():
    with _isolated_backend_imports() as runtime:
        production = importlib.import_module("sglang.srt.mem_cache.head_kv.production")
        yield NS(p=production, runtime=runtime)


def args(**kwargs):
    values = dict(redknot_managed_kv=True, attention_backend="redknot",
                  tp_size=1, pp_size=1, dp_size=1, attn_cp_size=1,
                  ep_size=1, moe_dp_size=1, page_size=1,
                  disable_radix_cache=True, disable_cuda_graph=True,
                  disable_piecewise_cuda_graph=True, disable_overlap_schedule=True,
                  disaggregation_mode="null", schedule_policy="fcfs",
                  redknot_page_size=4, kv_cache_dtype="auto")
    return NS(**(values | kwargs))


def owner(api, pages=128):
    pool = api.runtime.head_kv.HeadPagePool(pages, 4, 8)
    return api.p.ManagedRequestLifecycle(api.runtime.head_kv.HeadKVManager(pool))


def req(**kw):
    return NS(rid="same-public-id", req_pool_idx=0, **kw)


def batches(requests, lengths, prefixes, mode="EXTEND"):
    forward = NS(forward_mode=NS(name=mode), batch_size=len(requests),
                 seq_lens_cpu=torch.tensor([x + y for x, y in zip(lengths, prefixes)]),
                 seq_lens=torch.tensor([x + y for x, y in zip(lengths, prefixes)]),
                 extend_seq_lens=torch.tensor(lengths),
                 extend_prefix_lens=torch.tensor(prefixes),
                 positions=torch.tensor([p for n, start in zip(lengths, prefixes)
                                         for p in range(start, start + n)]))
    return NS(reqs=requests, prefix_lens=prefixes), forward


def layer():
    return NS(tp_q_head_num=4, tp_k_head_num=2, tp_v_head_num=2,
              qk_head_dim=8, v_head_dim=8, layer_id=0, scaling=8 ** -0.5,
              sliding_window_size=-1, is_cross_attention=False, logit_cap=0,
              pos_encoding_mode="NONE", attn_type="decoder")


def append(lifecycle, forward):
    n = len(forward.positions)
    return lifecycle.backend.forward(
        torch.randn(n, 4, 8), torch.randn(n, 2, 8), torch.randn(n, 2, 8),
        layer(), forward, decode=forward.forward_mode.name == "DECODE", save_kv_cache=True,
    )


@pytest.mark.parametrize("changed", [
    {"tp_size": 2}, {"dp_size": 2}, {"pp_size": 2}, {"attn_cp_size": 2},
    {"page_size": 64}, {"disable_radix_cache": False},
    {"disable_cuda_graph": False}, {"disable_piecewise_cuda_graph": False},
    {"disable_overlap_schedule": False}, {"attention_backend": "redknot_mla"},
    {"enable_lora": True}, {"enable_streaming_session": True},
    {"speculative_algorithm": "EAGLE"}, {"quantization": "fp8"},
    {"kv_cache_dtype": "fp8_e4m3"}, {"disaggregation_mode": "decode"},
    {"enable_hierarchical_cache": True}, {"enable_memory_saver": True},
    {"schedule_policy": "lpm"}, {"redknot_page_size": 0},
    {"prefill_attention_backend": "flashinfer"}, {"device": "cpu"},
])
def test_option_guards_are_opt_in(api, changed):
    api.p.validate_managed_server_args(args())
    with pytest.raises(ValueError, match="redknot-managed-kv"):
        api.p.validate_managed_server_args(args(**changed))
    api.p.validate_managed_server_args(args(redknot_managed_kv=False, **changed))


def test_budget_reserves_tail_and_cow_without_double_allocation(api):
    budget = api.p.managed_budget(1024, 8, 64, 4, 2)
    assert budget.logical_tokens == 456
    assert budget.capacity_pages == 128
    assert budget.reserved_tail_tokens == 568
    with pytest.raises(ValueError):
        api.p.managed_budget(64, 8, 64, 4, 2)
    with pytest.raises(ValueError):
        api.p.managed_budget(True, 8, 64, 4, 2)


def test_model_runner_budget_and_physical_factory_share_one_owner(api):
    runner = NS(
        server_args=args(), max_total_num_tokens=128, max_running_requests=2,
        model_config=NS(hf_config=NS(model_type="qwen3", num_hidden_layers=1),
                        head_dim=8, v_head_dim=8, get_num_kv_heads=lambda tp: 2),
        start_layer=0, end_layer=1, num_effective_layers=1,
        kv_cache_dtype=torch.bfloat16, device="cpu",
        req_to_token_pool=object(),
    )
    # CPU allocation is deliberately a unit-test fixture, not CUDA launch proof.
    api.p.prepare_managed_budget(runner)
    assert runner.max_total_num_tokens == 118
    runner.token_to_kv_pool = api.p.create_managed_token_pool(runner)
    manager = runner.redknot_shared_kv_manager
    assert manager.pool.capacity_pages == 64
    assert manager.pool.k.shape == (64, 4, 8)
    assert not hasattr(runner.token_to_kv_pool, "k_buffer")
    backend = api.runtime.registry.create_redknot_backend(runner)
    assert backend.shared_kv is runner.redknot_shared_kv_lifecycle.backend
    assert backend.shared_kv.manager is manager
    assert runner.token_to_kv_pool.manager is manager


@pytest.mark.parametrize("model_type", ["deepseek_v4", "qwen3_5", "qwen3_moe"])
def test_unsupported_state_models_rejected_before_physical_allocation(api, model_type):
    runner = NS(server_args=args(), model_config=NS(hf_config=NS(model_type=model_type)),
                max_total_num_tokens=128, max_running_requests=2)
    with pytest.raises(ValueError, match="Qwen3 MHA/GQA only"):
        api.p.prepare_managed_budget(runner)
    assert not hasattr(runner, "redknot_shared_kv_manager")


def test_create_chunk_decode_finish_and_slot_id_reuse(api):
    lifecycle = owner(api)
    first, second = req(), req()
    sb, fb = batches([first, second], [3, 2], [0, 0])
    lifecycle.bind_batch(sb, fb)
    h1, h2 = fb.redknot_shared_kv_handles
    assert h1.request_id != h2.request_id
    assert append(lifecycle, fb).shape == (5, 32)
    sb, fb = batches([second, first], [2, 1], [2, 3])
    lifecycle.bind_batch(sb, fb)
    assert fb.redknot_shared_kv_handles == [h2, h1]
    append(lifecycle, fb)
    sb, fb = batches([first, second], [1, 1], [4, 4], "DECODE")
    lifecycle.bind_batch(sb, fb)
    append(lifecycle, fb)
    assert lifecycle.backend.manager.stats()["requests"] == 2
    with pytest.raises(RuntimeError, match="live requests"):
        lifecycle.assert_idle()
    assert lifecycle.release(first)
    assert not lifecycle.release(first)
    lifecycle.release(second)
    lifecycle.assert_idle()
    assert lifecycle.backend.manager.stats()["free_pages"] == 128
    assert lifecycle.stats()["created_serving_requests"] == 2
    assert lifecycle.stats()["released_serving_requests"] == 2
    assert lifecycle.stats()["forward_batches"] == 3
    # Retraction and resume (or a new request recycling the same public rid/slot)
    # never revives the old generation.
    sb, fb = batches([first], [1], [0])
    lifecycle.bind_batch(sb, fb)
    assert fb.redknot_shared_kv_handles[0] != h1
    with pytest.raises(RuntimeError, match="stale"):
        lifecycle.backend._version(h1)
    lifecycle.release(first)


def test_all_batch_preflight_precedes_creation_and_rejects_nonprefix(api):
    lifecycle = owner(api)
    sb, fb = batches([req(), req(redknot_reuse_plan={"restore": True})], [1, 1], [0, 0])
    with pytest.raises(ValueError, match="does not import"):
        lifecycle.bind_batch(sb, fb)
    assert lifecycle.backend.manager.stats()["requests"] == 0
    sb, fb = batches([req()], [1], [5], "DECODE")
    with pytest.raises(ValueError, match="no live KV"):
        lifecycle.bind_batch(sb, fb)
    first = req()
    sb, fb = batches([first, first], [1, 1], [0, 0])
    with pytest.raises(ValueError, match="duplicate Req"):
        lifecycle.bind_batch(sb, fb)


def test_partial_creation_rolls_back_and_process_owner_is_fenced(api, monkeypatch):
    lifecycle = owner(api)
    create = lifecycle.backend.create_request
    count = 0
    def broken(*a, **kw):
        nonlocal count
        count += 1
        if count == 2:
            raise MemoryError("injected creation failure")
        return create(*a, **kw)
    monkeypatch.setattr(lifecycle.backend, "create_request", broken)
    sb, fb = batches([req(), req()], [1, 1], [0, 0])
    with pytest.raises(MemoryError):
        lifecycle.bind_batch(sb, fb)
    assert not lifecycle._requests
    assert not any(hasattr(r, "_redknot_managed_binding") for r in sb.reqs)
    lifecycle.pid += 1
    with pytest.raises(RuntimeError, match="process boundary"):
        lifecycle.bind_batch(sb, fb)


def test_worker_guard_aborts_partially_written_batch_including_sampling(api):
    lifecycle = owner(api)
    class Worker:
        model_runner = NS(redknot_shared_kv_lifecycle=lifecycle)
        @api.p.managed_forward_guard
        def forward_batch_generation(self, batch=None, forward_batch=None):
            lifecycle.bind_batch(batch, forward_batch)
            append(lifecycle, forward_batch)
            raise RuntimeError("sampling failed after attention")
    sb, fb = batches([req(), req()], [2, 2], [0, 0])
    with pytest.raises(RuntimeError, match="sampling failed"):
        Worker().forward_batch_generation(sb, fb)
    lifecycle.assert_idle()
    assert lifecycle.backend.manager.stats()["free_pages"] == 128
    assert not lifecycle.backend._views


def test_default_worker_guard_noop_and_live_weight_updates_rejected(api):
    class Worker:
        model_runner = NS()
        @api.p.managed_forward_guard
        def forward(self, batch=None, forward_batch=None, tag=None):
            return batch, forward_batch, tag
        @api.p.reject_managed_weight_update
        def update(self):
            return "updated"
    w = Worker()
    assert w.forward(1, 2, tag=3) == (1, 2, 3)
    assert w.update() == "updated"
    w.model_runner.redknot_shared_kv_lifecycle = owner(api)
    assert w.update() == (False, "managed KV serving requires restart for weight updates")


def test_facade_reports_real_slabs_and_never_returns_dummy_kv(api):
    lifecycle = owner(api)
    r = NS(max_total_num_tokens=80, device="cpu", kv_cache_dtype=torch.float32,
           num_effective_layers=1, start_layer=0, end_layer=1,
           model_config=NS(get_num_kv_heads=lambda tp: 2))
    facade = api.p.ManagedTokenPoolFacade(r, lifecycle.backend.manager)
    assert facade.get_kv_size_bytes() == (128 * 4 * 8 * 4,) * 2
    assert facade.size == 80
    for name in ("get_key_buffer", "get_value_buffer", "get_kv_buffer", "set_kv_buffer",
                 "move_kv_cache", "get_cpu_copy", "load_cpu_copy", "get_contiguous_buf_infos"):
        with pytest.raises(RuntimeError, match="no dense"):
            getattr(facade, name)(0)
    assert not hasattr(facade, "k_buffer")


def _load_release_function(namespace):
    path = ROOT / "python/sglang/srt/mem_cache/common.py"
    parsed = ast.parse(path.read_text())
    func = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                and node.name == "release_kv_cache")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), func], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["release_kv_cache"]


@pytest.mark.parametrize("is_insert", [True, False])
def test_actual_release_function_drops_handle_before_recycling_logical_slot(api, is_insert):
    lifecycle = owner(api)
    r = req()
    sb, fb = batches([r], [3], [0])
    lifecycle.bind_batch(sb, fb)
    append(lifecycle, fb)
    r.pop_overallocated_kv_cache = lambda: (0, 0)
    seen = []
    def finish(request, **kw):
        assert not hasattr(request, "_redknot_managed_binding")
        seen.append(("cache_finished", kw["is_insert"]))
    def free(request):
        assert not hasattr(request, "_redknot_managed_binding")
        seen.append(("logical_free", request.req_pool_idx))
        request.req_pool_idx = None
    tree = NS(cache_finished_req=finish, req_to_token_pool=NS(free=free))
    release = _load_release_function(dict(
        get_global_server_args=lambda: NS(page_size=1, speculative_algorithm=None, strip_thinking_cache=False),
        HybridReqToTokenPool=type("UnusedHybridFixture", (), {}),
    ))
    release(r, tree, is_insert=is_insert)
    assert seen == [("cache_finished", is_insert), ("logical_free", 0)]
    lifecycle.assert_idle()


def test_real_production_hook_sites_are_present_not_an_alternative_scheduler(api):
    paths = {
        "python/sglang/srt/model_executor/forward_batch_info.py": "managed_lifecycle.bind_batch(batch, ret)",
        "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py": "create_managed_token_pool(self)",
        "python/sglang/srt/managers/tp_worker.py": "@managed_forward_guard",
        "python/sglang/srt/managers/scheduler.py": "managed_lifecycle.assert_idle()",
        "python/sglang/srt/mem_cache/common.py": "release_managed_request(req)",
    }
    for path, text in paths.items():
        source = (ROOT / path).read_text()
        ast.parse(source)
        assert text in source
