"""Identity head-selection optimization; no mask or attention approximation.

The scalar reference is the previously used advanced-indexing implementation.
CPU tests record its exact kernel inputs and compare outputs and LSE behavior;
optional CUDA tests compare the actual FlashAttention kernel bit for bit.
"""
from types import SimpleNamespace

import pytest
import torch

from test_nonprefix_paged import _isolated_imports


@pytest.fixture(scope="module")
def runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with _isolated_imports() as modules:
            yield modules
    finally:
        torch.set_num_threads(previous)


def _old_recovery(engine, q, k, v, prefix_k, prefix_v, layer, scale):
    """Frozen pre-optimization implementation, including indexed scatter."""
    def attend(query, keys, values, *, causal, window=-1):
        out, lse, _ = engine._flash_dense(
            query.transpose(1, 2).contiguous(), keys.transpose(1, 2).contiguous(),
            values.transpose(1, 2).contiguous(), softmax_scale=scale,
            causal=causal, window_size=(window, 0) if window >= 0 else (-1, -1),
            return_attn_probs=True)
        return out.transpose(1, 2), lse

    groups = {}
    for h, policy in enumerate(engine.policies[layer]):
        key = (policy.head_type == "local", policy.window,
               min(policy.sink_size, prefix_k.shape[2]))
        if not key[0]:
            key = (False, -1, 0)
        groups.setdefault(key, []).append(h)
    output = torch.empty_like(q)
    for (local, window, sink), kv_heads in groups.items():
        query_heads = [h*engine.groups+j for h in kv_heads for j in range(engine.groups)]
        query = q[:, query_heads]
        keys = torch.cat((prefix_k[:, kv_heads], k[:, kv_heads]), 2)
        values = torch.cat((prefix_v[:, kv_heads], v[:, kv_heads]), 2)
        if not local:
            result, _ = attend(query, keys, values, causal=True)
        else:
            recent_k, recent_v = keys[:, :, sink:], values[:, :, sink:]
            keep = min(recent_k.shape[2], window+q.shape[2])
            recent, recent_lse = attend(query, recent_k[:, :, -keep:],
                                       recent_v[:, :, -keep:], causal=True, window=window)
            if sink:
                initial, initial_lse = attend(query, keys[:, :, :sink],
                                             values[:, :, :sink], causal=False)
                z = torch.logaddexp(recent_lse.float(), initial_lse.float())
                a = torch.exp(recent_lse.float()-z)[..., None]
                b = torch.exp(initial_lse.float()-z)[..., None]
                result = (recent.float()*a + initial.float()*b).to(q.dtype)
            else:
                result = recent
        output[:, query_heads] = result
    return output


def _dense_flash_oracle(q, k, v, *, softmax_scale, causal, window_size,
                        return_attn_probs):
    assert return_attn_probs is True
    groups = q.shape[2] // k.shape[2]
    query = q.transpose(1, 2).float()
    keys = k.transpose(1, 2).repeat_interleave(groups, dim=1).float()
    values = v.transpose(1, 2).repeat_interleave(groups, dim=1).float()
    score = query @ keys.transpose(-1, -2) * softmax_scale
    kp = torch.arange(k.shape[1], device=q.device)[None, :]
    qp = torch.arange(q.shape[1], device=q.device)[:, None] + k.shape[1]-q.shape[1]
    mask = torch.ones((q.shape[1], k.shape[1]), device=q.device, dtype=torch.bool)
    if causal:
        mask &= kp <= qp
    if window_size[0] >= 0:
        mask &= kp >= qp-window_size[0]
    if window_size[1] >= 0:
        mask &= kp <= qp+window_size[1]
    score = score.masked_fill(~mask, -torch.inf)
    output = (score.softmax(-1).nan_to_num(0.) @ values).transpose(1, 2).to(q.dtype)
    return output, torch.logsumexp(score, -1), None


def _fixture(runtime, profile, prefix, tokens, *, dtype=torch.float32, device="cpu"):
    engine = object.__new__(runtime.core.NonPrefixPagedEngine)
    engine.kv_heads, engine.groups = 4, 2
    policies = {
        "global": [("global", -1, 0)]*4,
        "local": [("local", 7, 4)]*4,
        "local_no_sink": [("local", 7, 0)]*4,
        "mixed": [("local", 7, 4), ("global", -1, 0),
                  ("local", 7, 4), ("dense", -1, 0)],
        "different_windows": [("local", 7, 4), ("local", 3, 4),
                              ("local", 7, 0), ("local", 7, 4)],
    }[profile]
    engine.policies = [[SimpleNamespace(head_type=t, window=w, sink_size=s)
                        for t, w, s in policies]]
    generator = torch.Generator(device=device).manual_seed(4712)
    def tensor(heads, length):
        # Match model QKV views: token-contiguous storage, noncontiguous H/T.
        return torch.randn((1, length, heads, 64), generator=generator,
                           device=device, dtype=dtype).transpose(1, 2)
    return engine, (tensor(8, tokens), tensor(4, tokens), tensor(4, tokens),
                    tensor(4, prefix), tensor(4, prefix))


@pytest.mark.parametrize("profile", ["global", "local", "local_no_sink", "mixed", "different_windows"])
@pytest.mark.parametrize("prefix,tokens", [(0, 1), (3, 7), (13, 33)])
def test_identity_selection_preserves_exact_inputs_masks_and_outputs(runtime, profile, prefix, tokens):
    engine, args = _fixture(runtime, profile, prefix, tokens)
    before = tuple(x.clone() for x in args)
    calls = []
    def observed(q, k, v, **kwargs):
        calls.append((q.clone(), k.clone(), v.clone(), kwargs))
        return _dense_flash_oracle(q, k, v, **kwargs)
    engine._flash_dense = observed
    expected = _old_recovery(engine, *args, 0, 1/8)
    previous, calls = calls, []
    actual = engine._recovery_flash_attention(*args, 0, 1/8)
    assert len(previous) == len(calls)
    for old, new in zip(previous, calls):
        assert old[3] == new[3]
        for a, b in zip(old[:3], new[:3]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for original, current in zip(before, args):
        torch.testing.assert_close(original, current, rtol=0, atol=0)


@pytest.mark.parametrize("profile", ["global", "local", "local_no_sink", "mixed"])
def test_homogeneous_layer_eliminates_identity_index_and_scatter(runtime, profile):
    engine, args = _fixture(runtime, profile, 13, 33)
    engine._flash_dense = _dense_flash_oracle
    def counts(fn):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            fn(engine, *args, 0, 1/8)
        return {item.key: item.count for item in trace.key_averages()}
    old = counts(_old_recovery)
    new = counts(type(engine)._recovery_flash_attention)
    if profile == "mixed":
        assert new.get("aten::index", 0) == old["aten::index"] == 10
        assert new.get("aten::index_put_", 0) == old["aten::index_put_"] == 2
    else:
        assert old["aten::index"] == 5 and old["aten::index_put_"] == 1
        assert new.get("aten::index", 0) == 0
        assert new.get("aten::index_put_", 0) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("profile", ["global", "local", "mixed"])
def test_actual_flash_attention_bitwise_parity(runtime, dtype, profile):
    flash_attn = pytest.importorskip("flash_attn")
    engine, args = _fixture(runtime, profile, 13, 33, dtype=dtype, device="cuda")
    engine._flash_dense = flash_attn.flash_attn_func
    expected = _old_recovery(engine, *args, 0, 1/8)
    actual = engine._recovery_flash_attention(*args, 0, 1/8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
