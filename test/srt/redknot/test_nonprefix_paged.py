"""Real tiny-Qwen tests for genuine non-prefix local-head reuse.

Seeded random weights test numerical and ownership contracts, not pretrained
task accuracy. A literal full-softmax oracle independently checks paged/LSE
decode; all-global tests also check against the native model's full recompute.
"""

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


_SOURCE = Path(__file__).resolve().parents[3] / "python"


@contextmanager
def _isolated_imports():
    saved = {name: module for name, module in sys.modules.items()
             if name == "sglang" or name.startswith("sglang.")}
    for name in saved:
        del sys.modules[name]
    try:
        for name in ("sglang", "sglang.srt", "sglang.srt.layers",
                     "sglang.srt.layers.attention", "sglang.srt.layers.attention.redknot",
                     "sglang.srt.mem_cache"):
            module = types.ModuleType(name)
            module.__path__ = [str(_SOURCE.joinpath(*name.split(".")))]
            sys.modules[name] = module
        yield types.SimpleNamespace(
            core=importlib.import_module("sglang.srt.layers.attention.redknot.nonprefix_paged"),
            config=importlib.import_module("sglang.srt.layers.attention.redknot.head_config"),
        )
    finally:
        for name in list(sys.modules):
            if name == "sglang" or name.startswith("sglang."):
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(scope="module")
def runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with _isolated_imports() as modules:
            yield modules
    finally:
        torch.set_num_threads(previous)


def _tiny_model(device="cpu", dtype=torch.float32, *, rope_type="default", num_layers=3):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.1":
        pytest.skip("the experimental model binding requires Transformers 4.57.1")
    config = transformers.Qwen3Config(
        vocab_size=193, hidden_size=64, intermediate_size=128,
        num_hidden_layers=num_layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=2048, attention_dropout=0.0,
        rope_scaling=({"rope_type": "yarn", "factor": 4.0,
                       "original_max_position_embeddings": 512}
                      if rope_type == "yarn" else None),
    )
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(512701)
        return transformers.Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()


def _head_config(runtime, *, mixed=True, window=7, sink=0):
    # Local-head reuse is deliberately confined to one middle layer, keeping
    # the two other layers as full-global controls; this is not a tuned profile.
    classes = [["global", "global"],
               ["local", "global"] if mixed else ["global", "global"],
               ["global", "global"]]
    return runtime.config.HeadClassConfig(
        classes, [[window if head == "local" else -1 for head in layer] for layer in classes],
        3, 2, head_sink_size=[[sink if head == "local" else 0 for head in layer]
                            for layer in classes], default_sink_size=0,
    )


@pytest.fixture
def cpu_recovery(runtime):
    return runtime


_DOCS = {"D": list(range(20, 31)), "E": list(range(50, 57))}
_PREFIXES = {"A": [1, 2, 3], "B": [9, 8, 7, 6, 5]}


def _engine(runtime, model, *, mixed=True, share_local=True, backend="torch",
            capacity_pages=2048, page_size=4, window=7):
    engine = runtime.core.NonPrefixPagedEngine(
        model, _head_config(runtime, mixed=mixed, window=window), capacity_pages=capacity_pages,
        page_size=page_size, backend=backend, sparse_ffn_schedule=None,
        model_revision="test-seeded-random-qwen3-v1", prefix_chunk_size=256,
        share_local=share_local,
    )
    for document_id, ids in _DOCS.items():
        engine.register_document(document_id, ids)
    return engine


def _native_logits(model, prefix, documents, chunks):
    device = next(model.parameters()).device
    full = prefix + [token for document in documents for token in _DOCS[document]]
    with torch.no_grad():
        cache = model(torch.tensor([full], device=device), use_cache=True).past_key_values
        result = []
        for chunk in chunks:
            output = model(torch.tensor([chunk], device=device), use_cache=True,
                           past_key_values=cache, logits_to_keep=1)
            result.append(output.logits.clone())
            cache = output.past_key_values
    return result


@pytest.mark.parametrize("documents", [["D"], ["E", "D"], ["D", "E", "D"]])
def test_all_global_matches_full_recompute_with_actual_nonprefix_context(
    cpu_recovery, documents,
):
    runtime = cpu_recovery
    model = _tiny_model()
    questions = {"A": [[81, 82, 83], [84]], "B": [[91, 92, 93], [94]]}
    expected = {rid: _native_logits(model, prefix, documents, questions[rid])
                for rid, prefix in _PREFIXES.items()}
    original = [layer.self_attn.forward for layer in model.model.layers]
    engine = _engine(runtime, model, mixed=False)
    try:
        for rid, prefix in _PREFIXES.items():
            engine.admit(rid, prefix, documents, max_suffix_tokens=8)
        for step in range(2):
            ids = torch.tensor([questions[rid][step] for rid in ("A", "B")])
            actual = engine.forward(["A", "B"], ids)
            oracle = torch.cat([expected[rid][step] for rid in ("A", "B")])
            torch.testing.assert_close(actual, oracle, rtol=5e-5, atol=5e-6)
        assert [layer.self_attn.forward for layer in model.model.layers] == original
    finally:
        engine.close()
    assert [layer.self_attn.forward for layer in model.model.layers] == original


def test_shared_and_private_mixed_head_storage_produce_same_logits(cpu_recovery):
    runtime = cpu_recovery
    model = _tiny_model()
    results = []
    for shared in (False, True):
        engine = _engine(runtime, model, share_local=shared)
        try:
            engine.admit("A", _PREFIXES["A"], ["D", "E"], max_suffix_tokens=8)
            engine.admit("B", _PREFIXES["B"], ["E", "D"], max_suffix_tokens=8)
            outputs = [engine.forward(["A", "B"], torch.tensor([[81, 82, 83], [91, 92, 93]]))]
            outputs.append(engine.forward(["B", "A"], torch.tensor([[94], [84]])))
            results.append([out.clone() for out in outputs])
        finally:
            engine.close()
    for private, shared in zip(*results):
        torch.testing.assert_close(private, shared, rtol=0, atol=0)


def test_batched_mixed_head_requests_match_independent_serial_execution(cpu_recovery):
    runtime = cpu_recovery
    model = _tiny_model()
    documents = {"A": ["D", "E"], "B": ["E", "D", "D"]}
    questions = {"A": [81, 82, 83], "B": [91, 92, 93]}
    engine = _engine(runtime, model)
    try:
        for rid in ("A", "B"):
            engine.admit(rid, _PREFIXES[rid], documents[rid], max_suffix_tokens=8)
        batched = engine.forward(["A", "B"], torch.tensor([questions["A"], questions["B"]])).clone()
        engine.release("A")
        engine.release("B")
        serial = []
        for rid in ("A", "B"):
            engine.admit(rid, _PREFIXES[rid], documents[rid], max_suffix_tokens=8)
            serial.append(engine.forward([rid], torch.tensor([questions[rid]])).clone())
            engine.release(rid)
        torch.testing.assert_close(batched, torch.cat(serial), rtol=5e-5, atol=5e-6)
    finally:
        engine.close()


def _hash_source_pages(engine):
    """Hash used bytes, not uninitialized physical page padding."""
    digest = hashlib.sha256()
    for document in engine.documents.values():
        with engine.manager.bind(document.source_request_id) as lease:
            for key in document.source_keys.values():
                for tensor in lease.gather(key):
                    digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _literal_rotate(tensor, angles):
    # Independently spell out split-half rotation, without the engine helper.
    half = tensor.shape[-1] // 2
    rotated = torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)
    return tensor * angles.cos() + rotated * angles.sin()


@torch.no_grad()
def _independent_query_oracle(engine, request_ids, token_ids):
    """Materialize admitted history, then ordinary full softmax per physical head.

    This deliberately does not call engine attention, its partition LSE merger,
    its RoPE helpers, or its current-token append path. Historical keys get
    literal +offset rotation instead of counterrotating Q per partition.
    """
    model = engine.model
    device = next(model.parameters()).device
    inv_freq = model.model.rotary_emb.inv_freq.float()
    qlen = token_ids.shape[1]
    positions = torch.stack([
        torch.arange(engine.requests[rid].total_context_tokens,
                     engine.requests[rid].total_context_tokens + qlen, device=device)
        for rid in request_ids
    ])
    history = {}
    for rid in request_ids:
        request = engine.requests[rid]
        assert request.suffix_tokens == 0, "this oracle intentionally starts before the first query"
        with engine.manager.bind(request.manager_request_id) as lease:
            for (layer, head), partitions in request.partitions.items():
                pieces_k, pieces_v = [], []
                for partition in partitions:
                    pages = [engine.pool.read_page(ref, min(engine.pool.page_size,
                             partition.end - index * engine.pool.page_size))
                             for index, ref in enumerate(partition.pages)]
                    k = torch.cat([page[0] for page in pages]).clone()
                    v = torch.cat([page[1] for page in pages]).clone()
                    if partition.canonical:
                        angles = inv_freq * float(partition.offset)
                        k = _literal_rotate(k.float(), torch.cat((angles, angles))).to(k.dtype)
                    pieces_k.append(k)
                    pieces_v.append(v)
                history[rid, layer, head] = (torch.cat(pieces_k), torch.cat(pieces_v))
    original = [layer.self_attn.forward for layer in model.model.layers]

    def make_attention(layer_index):
        def attention(attention_self, hidden_states, position_embeddings, **kwargs):
            shape = (*hidden_states.shape[:-1], -1, attention_self.head_dim)
            q = attention_self.q_norm(attention_self.q_proj(hidden_states).view(shape)).transpose(1, 2)
            k = attention_self.k_norm(attention_self.k_proj(hidden_states).view(shape)).transpose(1, 2)
            v = attention_self.v_proj(hidden_states).view(shape).transpose(1, 2)
            cos, sin = position_embeddings
            half = q.shape[-1] // 2
            q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            k_rot = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
            q = q * cos[:, None] + q_rot * sin[:, None]
            k = k * cos[:, None] + k_rot * sin[:, None]
            out = torch.empty_like(q)
            for batch, rid in enumerate(request_ids):
                for qhead in range(q.shape[1]):
                    kvhead = qhead // attention_self.num_key_value_groups
                    prior_k, prior_v = history[rid, layer_index, kvhead]
                    all_k = torch.cat((prior_k, k[batch, kvhead]))
                    all_v = torch.cat((prior_v, v[batch, kvhead]))
                    scores = q[batch, qhead].float() @ all_k.float().T * attention_self.scaling
                    # Every admitted historical token precedes the query; only
                    # the current suffix needs a triangular future-token mask.
                    future = torch.arange(len(all_k), device=device)[None, :] > (
                        len(prior_k) + torch.arange(qlen, device=device)[:, None])
                    scores.masked_fill_(future, float("-inf"))
                    out[batch, qhead] = (scores.softmax(-1) @ all_v.float()).to(q.dtype)
            result = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
            return attention_self.o_proj(result), None
        return attention

    try:
        for index, layer in enumerate(model.model.layers):
            layer.self_attn.forward = types.MethodType(make_attention(index), layer.self_attn)
        return model(token_ids, position_ids=positions, use_cache=False, logits_to_keep=1).logits.clone()
    finally:
        for layer, forward in zip(model.model.layers, original):
            layer.self_attn.forward = forward


def test_mixed_heads_lse_and_counterrotation_match_independent_full_softmax(cpu_recovery):
    model = _tiny_model()
    engine = _engine(cpu_recovery, model, window=8)
    try:
        # Three occurrences, two offsets, and unaligned 8-token local windows
        # prevent the test degenerating into a shared-prefix attention test.
        engine.admit("A", _PREFIXES["A"], ["D", "E", "D"], max_suffix_tokens=8)
        engine.admit("B", _PREFIXES["B"], ["E", "D"], max_suffix_tokens=8)
        ids = torch.tensor([[81, 82, 83], [91, 92, 93]])
        expected = _independent_query_oracle(engine, ["A", "B"], ids)
        actual = engine.forward(["A", "B"], ids)
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
    finally:
        engine.close()


def test_source_is_immutable_across_actual_nonprefix_offsets_and_decode(cpu_recovery):
    model = _tiny_model()
    engine = _engine(cpu_recovery, model)
    try:
        before = _hash_source_pages(engine)
        engine.admit("A", _PREFIXES["A"], ["D", "E", "D"], max_suffix_tokens=8)
        engine.admit("B", _PREFIXES["B"], ["E", "D"], max_suffix_tokens=8)
        assert engine.requests["A"].document_offsets != engine.requests["B"].document_offsets
        assert _hash_source_pages(engine) == before
        engine.forward(["A", "B"], torch.tensor([[81, 82, 83], [91, 92, 93]]))
        engine.forward(["B", "A"], torch.tensor([[94, 95], [84, 85]]))
        assert _hash_source_pages(engine) == before
        engine.release("A")
        engine.release("B")
        assert _hash_source_pages(engine) == before
    finally:
        engine.close()
    stats = engine.pool.stats()
    assert stats["free_pages"] == stats["capacity_pages"]
    assert stats["pinned_pages"] == 0
    assert stats["reserved_pages"] == 0


def test_local_heads_alias_canonical_pages_but_global_heads_are_request_private(runtime):
    engine = _engine(runtime, _tiny_model(), window=8)
    try:
        # Equal prefix lengths isolate contextual recomputation from positional
        # changes when comparing global V; different content remains essential.
        a = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
        b = engine.admit("B", [6, 7, 8], ["D"], max_suffix_tokens=8)
        doc = engine.documents["D"]
        source = engine.manager.version(doc.source_request_id).segments[doc.source_keys[1, 0]]
        av = engine.manager.version(a.manager_request_id)
        bv = engine.manager.version(b.manager_request_id)
        assert av.segments[1, 0, "document:0"].pages == source.pages
        assert bv.segments[1, 0, "document:0"].pages == source.pages
        for state in (a, b):
            parts = state.partitions[1, 0]
            assert all(p.start == 0 and p.end > 0 and p.canonical for p in parts)
            assert sum(p.end for p in parts) == 8
            positions = [p.source_start + j for p in parts for j in range(p.end)]
            assert positions == list(range(3, 11)), "window has no omitted or duplicated boundary rows"
            assert len(positions) == len(set(positions))
            assert any(p.source_start == 3 and not set(p.pages) & set(source.pages) for p in parts)
            assert any(set(p.pages) & set(source.pages) for p in parts), "interior is physically shared"
        assert not set(av.segments[2, 1, "global"].pages) & set(bv.segments[2, 1, "global"].pages)
        with engine.manager.bind(a.manager_request_id) as al, engine.manager.bind(b.manager_request_id) as bl:
            ak, avalue = al.gather((2, 1, "global"))
            bk, bvalue = bl.gather((2, 1, "global"))
        assert not torch.allclose(avalue[3:], bvalue[3:], atol=1e-6, rtol=1e-6)
        assert not torch.allclose(ak[3:], bk[3:], atol=1e-6, rtol=1e-6)
    finally:
        engine.close()


def test_registered_local_keys_are_actual_position_zero_native_document_keys(runtime):
    model = _tiny_model()
    expected = {}
    with torch.no_grad():
        for name, tokens in _DOCS.items():
            cache = model(torch.tensor([tokens]), use_cache=True).past_key_values
            expected[name] = (cache.layers[1].keys[0, 0].clone(),
                              cache.layers[1].values[0, 0].clone())
    engine = _engine(runtime, model)
    try:
        for name, document in engine.documents.items():
            assert set(document.source_keys) == {(1, 0)}, "global heads must not masquerade as reusable local state"
            with engine.manager.bind(document.source_request_id) as lease:
                key = document.source_keys[1, 0]
                assert lease.version.segments[key].positions == tuple(range(len(_DOCS[name])))
                k, v = lease.gather(key)
            torch.testing.assert_close(k, expected[name][0], rtol=0, atol=0)
            torch.testing.assert_close(v, expected[name][1], rtol=0, atol=0)
    finally:
        engine.close()


def test_actual_shared_local_value_pages_change_emitted_logits(runtime):
    model = _tiny_model()
    outputs = []
    for mutate in (False, True):
        engine = _engine(runtime, model, window=8)
        try:
            state = engine.admit("A", _PREFIXES["A"], ["D"], max_suffix_tokens=8)
            if mutate:
                source = engine.documents["D"]
                refs = engine.manager.version(source.source_request_id).segments[source.source_keys[1, 0]].pages
                actual_shared = [ref for p in state.partitions[1, 0] for ref in p.pages if ref in refs]
                assert actual_shared, "mutation must target physical shared pages, not a copied side cache"
                with torch.no_grad():
                    for ref in actual_shared:
                        engine.pool.v[ref.slot].add_(5.0)
            outputs.append(engine.forward(["A"], torch.tensor([[81, 82, 83]])).clone())
        finally:
            engine.close()
    assert (outputs[0] - outputs[1]).abs().max().item() > 0.01


def test_admission_capacity_failure_is_atomic(runtime):
    engine = _engine(runtime, _tiny_model(), capacity_pages=6)
    try:
        before = engine.pool.stats()
        digest = _hash_source_pages(engine)
        with pytest.raises(runtime.core.CapacityError):
            engine.admit("A", _PREFIXES["A"], ["D", "E"], max_suffix_tokens=8)
        assert not engine.requests
        after = engine.pool.stats()
        assert after["free_pages"] == before["free_pages"]
        assert after["reserved_pages"] == before["reserved_pages"]
        assert after["pinned_pages"] == 0
        assert _hash_source_pages(engine) == digest
    finally:
        engine.close()


@pytest.mark.parametrize("num_layers", [1, 3, 7])
def test_suffix_pool_lifecycle_call_count_is_constant_across_head_count(runtime, monkeypatch, num_layers):
    model = _tiny_model(num_layers=num_layers)
    config = runtime.config.HeadClassConfig(
        [["global", "global"] for _ in range(num_layers)],
        [[-1, -1] for _ in range(num_layers)], num_layers, 2,
        default_sink_size=0,
    )
    engine = runtime.core.NonPrefixPagedEngine(
        model, config, capacity_pages=512, page_size=4, backend="torch",
    )
    try:
        engine.register_document("D", _DOCS["D"])
        calls = {name: [] for name in ("reserve", "pin", "unpin", "release")}
        for name in calls:
            original = getattr(engine.pool, name)

            def observed(value, *args, _name=name, _original=original, **kwargs):
                recorded = value if _name == "reserve" else tuple(value)
                calls[_name].append((recorded, args, kwargs))
                return _original(recorded, *args, **kwargs)

            monkeypatch.setattr(engine.pool, name, observed)
        state = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=7)
        expected_tail_count = num_layers * 2 * 2
        all_tail = tuple(ref for refs in state.tail_refs.values() for ref in refs)
        assert state.tail_all_refs == all_tail
        assert len(all_tail) == len(set(all_tail)) == expected_tail_count
        # One suffix reservation, one global-history transaction reservation.
        # The count of reserve calls stays 2 as physical head count grows.
        assert [entry[0] for entry in calls["reserve"]] == [expected_tail_count, num_layers * 2 * 4]
        tail_set = set(all_tail)
        tail_pin = [entry for entry in calls["pin"] if tail_set.intersection(entry[0])]
        assert len(tail_pin) == 1 and tail_pin[0][0] == all_tail
        assert state.tail_pinned
        assert all(engine.pool._get(ref).owners == 1 and engine.pool._get(ref).pins == 1 for ref in all_tail)
        for (layer, head), refs in state.tail_refs.items():
            expected_rows = [ref.slot * 4 + offset for ref in refs for offset in range(4)][:7]
            assert state.tail_rows[layer, head].tolist() == expected_rows

        for entries in calls.values():
            entries.clear()
        engine.release("A")
        # One request-lifetime historical lease and one suffix arena, not one
        # unpin per head. These two constant calls have disjoint page sets.
        assert len(calls["unpin"]) == 2
        assert calls["unpin"][0][0] == state.history_lease.refs
        assert calls["unpin"][1][0] == all_tail
        assert not set(calls["unpin"][0][0]).intersection(all_tail)
        # Release the suffix arena and the manager's historical root once each.
        assert len(calls["release"]) == 2
        assert calls["release"][0][0] == all_tail
        assert all(call[1][0] is calls["release"][0][1][0] for call in calls["unpin"])
        assert state.history_completed and state.history_lease.closed
        assert not state.tail_pinned and state.tail_all_refs == ()
        assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
    finally:
        engine.close()


def test_suffix_arena_is_fully_released_when_second_row_descriptor_fails(runtime, monkeypatch):
    model = _tiny_model()
    engine = _engine(runtime, model)
    before = engine.pool.stats()
    native_tensor = torch.tensor
    calls = {name: [] for name in ("reserve", "pin", "unpin", "release")}
    try:
        with monkeypatch.context() as patch:
            for name in calls:
                original = getattr(engine.pool, name)

                def observed(value, *args, _name=name, _original=original, **kwargs):
                    recorded = value if _name == "reserve" else tuple(value)
                    calls[_name].append(recorded)
                    return _original(recorded, *args, **kwargs)

                patch.setattr(engine.pool, name, observed)
            row_calls = 0

            def fail_second_row(*args, **kwargs):
                nonlocal row_calls
                row_calls += 1
                if row_calls == 2:
                    raise RuntimeError("injected second suffix descriptor failure")
                return native_tensor(*args, **kwargs)

            patch.setattr(torch, "tensor", fail_second_row)
            with pytest.raises(RuntimeError, match="second suffix descriptor"):
                engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=7)
        assert calls["reserve"] == [12]
        assert len(calls["pin"]) == len(calls["unpin"]) == 1
        assert len(calls["pin"][0]) == 12
        assert calls["pin"][0] == calls["unpin"][0] == calls["release"][0]
        assert not engine.requests
        after = engine.pool.stats()
        for field in ("free_pages", "reserved_pages", "live_pages", "pinned_pages"):
            assert after[field] == before[field]
    finally:
        engine.close()


def test_admission_recovery_exception_releases_private_reservations(runtime, monkeypatch):
    model = _tiny_model()
    original = [layer.self_attn.forward for layer in model.model.layers]
    engine = _engine(runtime, model)
    try:
        before = engine.pool.stats()

        def fail_recovery(*args, **kwargs):
            raise RuntimeError("injected online recovery failure")

        monkeypatch.setattr(engine, "_recover_online", fail_recovery)
        with pytest.raises(RuntimeError, match="injected online recovery"):
            engine.admit("A", _PREFIXES["A"], ["D", "E"], max_suffix_tokens=8)
        assert not engine.requests
        after = engine.pool.stats()
        for field in ("free_pages", "reserved_pages", "pinned_pages", "live_pages"):
            assert after[field] == before[field]
        assert [layer.self_attn.forward for layer in model.model.layers] == original
    finally:
        engine.close()


def test_forward_failure_restores_bindings_and_quarantines_partial_requests(runtime, monkeypatch):
    model = _tiny_model()
    engine = _engine(runtime, model)
    original = [layer.self_attn.forward for layer in model.model.layers]
    try:
        engine.admit("A", _PREFIXES["A"], ["D"], max_suffix_tokens=8)
        attention = engine._torch_attention

        def fail_second_layer(q, states, layer, scale):
            if layer == 1:
                raise RuntimeError("injected attention failure")
            return attention(q, states, layer, scale)

        monkeypatch.setattr(engine, "_torch_attention", fail_second_layer)
        with pytest.raises(RuntimeError, match="injected attention"):
            engine.forward(["A"], torch.tensor([[81, 82, 83]]))
        assert [layer.self_attn.forward for layer in model.model.layers] == original
        assert not hasattr(model, "_redknot_nonprefix_owner")
        assert engine.requests["A"].failed
        assert engine.requests["A"].suffix_tokens == 0
        with pytest.raises(ValueError, match="failed request"):
            engine.forward(["A"], torch.tensor([[81]]))
    finally:
        engine.close()
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
    assert engine.pool.stats()["pinned_pages"] == 0


def test_final_admission_lease_failure_preserves_existing_lifetime_lease(runtime, monkeypatch):
    model = _tiny_model()
    engine = _engine(runtime, model)
    original = [layer.self_attn.forward for layer in model.model.layers]
    try:
        active = engine.admit("A", _PREFIXES["A"], ["D"], max_suffix_tokens=8)
        before = engine.pool.stats()
        actual_bind = engine.manager.bind
        build = engine._build_partitions
        final_bind_request = [None]

        def mark_final_bind(state):
            build(state)
            final_bind_request[0] = state.manager_request_id

        def fail_final_bind(request_id):
            if request_id == final_bind_request[0]:
                raise RuntimeError("injected final admission lease acquisition failure")
            return actual_bind(request_id)

        with monkeypatch.context() as patch:
            patch.setattr(engine, "_build_partitions", mark_final_bind)
            patch.setattr(engine.manager, "bind", fail_final_bind)
            with pytest.raises(RuntimeError, match="final admission lease acquisition"):
                engine.admit("B", _PREFIXES["B"], ["E", "D"], max_suffix_tokens=8)
        assert set(engine.requests) == {"A"}
        assert not active.history_lease.closed
        after = engine.pool.stats()
        for key in ("free_pages", "reserved_pages", "live_pages", "pinned_pages"):
            assert after[key] == before[key]
        assert engine.history_lease_acquisitions == 1
        assert engine.history_lease_completions == 0
        assert [layer.self_attn.forward for layer in model.model.layers] == original
        assert not hasattr(model, "_redknot_nonprefix_owner")
        assert not engine._busy
    finally:
        engine.close()
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages


def test_release_completion_failure_preserves_other_lease_and_allows_synchronized_cleanup(runtime, monkeypatch):
    model = _tiny_model()
    engine = _engine(runtime, model)
    original = [layer.self_attn.forward for layer in model.model.layers]
    try:
        a = engine.admit("A", _PREFIXES["A"], ["D"], max_suffix_tokens=8)
        b = engine.admit("B", _PREFIXES["B"], ["E", "D"], max_suffix_tokens=8)
        pinned_before = engine.pool.stats()["pinned_pages"]
        engine.forward(["A", "B"], torch.tensor([[81, 82], [91, 92]]))
        assert not a.history_lease.closed and not b.history_lease.closed

        def fail_before_unpin(event=None):
            raise RuntimeError("injected pre-unpin completion failure")

        with monkeypatch.context() as patch:
            patch.setattr(a.history_lease, "complete", fail_before_unpin)
            with pytest.raises(RuntimeError, match="pre-unpin completion"):
                engine.release("A")
        assert a.history_release_failed and not a.history_completion_uncertain
        assert not a.history_lease.closed and not b.history_lease.closed
        assert a.tail_pinned and b.tail_pinned
        assert set(engine.requests) == {"A", "B"}
        assert engine.history_lease_completions == 0
        assert engine.pool.stats()["pinned_pages"] == pinned_before
        assert [layer.self_attn.forward for layer in model.model.layers] == original
        assert not hasattr(model, "_redknot_nonprefix_owner")
        assert not engine._busy
        engine.release("A")
        assert a.history_completed and not b.history_lease.closed
        engine.release("B")
        assert b.history_completed and engine.history_lease_completions == 2
    finally:
        engine.close()
    # A lifetime failure quarantines the allocator rather than recycling pages
    # whose device-completion provenance could be uncertain. Owners and actual
    # pins must nevertheless balance on synchronized teardown.
    assert engine.pool.stats()["poisoned"]
    assert all(page.owners == 0 and page.pins == 0 for page in engine.pool._pages.values())
    assert not engine.requests and not engine.documents


def test_existing_model_owner_is_not_overwritten_or_removed(runtime):
    model = _tiny_model()
    engine = _engine(runtime, model)
    owner = object()
    try:
        state = engine.admit("A", _PREFIXES["A"], ["D"], max_suffix_tokens=8)
        model._redknot_nonprefix_owner = owner
        with pytest.raises(RuntimeError, match="active RedKnot binding"):
            engine.forward(["A"], torch.tensor([[81]]))
        assert model._redknot_nonprefix_owner is owner
        assert state.suffix_tokens == 0 and not state.failed
        assert not engine._busy
    finally:
        if getattr(model, "_redknot_nonprefix_owner", None) is owner:
            delattr(model, "_redknot_nonprefix_owner")
        engine.close()


@pytest.mark.parametrize("prefix", [[], [1, 2, 3]])
def test_empty_sink_and_fully_masked_suffix_pages_stay_finite(runtime, prefix):
    # A 6-token chunk crosses page boundaries; first query positions mask a
    # complete later suffix page. Its all-minus-inf LSE must carry zero mass.
    engine = _engine(runtime, _tiny_model(), window=1)
    try:
        state = engine.admit("A", prefix, ["D"], max_suffix_tokens=12)
        assert all(p.end > 0 for values in state.partitions.values() for p in values)
        ids = torch.tensor([[81, 82, 83, 84, 85, 86]])
        expected = _independent_query_oracle(engine, ["A"], ids)
        actual = engine.forward(["A"], ids)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
    finally:
        engine.close()


def test_head_specific_windows_and_sink_union_have_no_duplicates(runtime):
    model = _tiny_model()
    config = runtime.config.HeadClassConfig(
        [["global", "global"], ["local", "local"], ["global", "global"]],
        [[-1, -1], [3, 8], [-1, -1]], 3, 2,
        head_sink_size=[[0, 0], [2, 3], [0, 0]], default_sink_size=0,
    )
    engine = runtime.core.NonPrefixPagedEngine(
        model, config, capacity_pages=256, page_size=4, backend="torch",
        model_revision="test-seeded-random-qwen3-v1",
    )
    try:
        engine.register_document("D", _DOCS["D"])
        state = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
        for head, expected in ((0, [0, 1, 11, 12, 13]), (1, [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13])):
            positions = [p.offset + p.source_start + j for p in state.partitions[1, head]
                         for j in range(p.end)]
            assert positions == expected
            assert len(positions) == len(set(positions))
        ids = torch.tensor([[81, 82, 83]])
        oracle = _independent_query_oracle(engine, ["A"], ids)
        torch.testing.assert_close(engine.forward(["A"], ids), oracle, rtol=5e-5, atol=5e-6)
    finally:
        engine.close()


def test_flash_backend_rejects_cpu_before_optional_import(runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    with pytest.raises(ValueError, match="requires CUDA FP16/BF16"):
        _engine(runtime, _tiny_model(), backend="flash", page_size=256)


def test_static_yarn_nonunit_scale_all_global_matches_native_past_original_context(runtime):
    model = _tiny_model(rope_type="yarn")
    assert model.model.rotary_emb.attention_scaling > 1
    original_scaling = dict(model.config.rope_scaling)
    prefixes = {"A": [1 + index % 17 for index in range(519)],
                "B": [91 + index % 21 for index in range(541)]}
    chunks = {"A": [[81, 82, 83], [84, 85]], "B": [[111, 112, 113], [114, 115]]}
    expected = {rid: _native_logits(model, prefix, ["D", "E", "D"], chunks[rid])
                for rid, prefix in prefixes.items()}
    engine = _engine(runtime, model, mixed=False)
    try:
        for rid, prefix in prefixes.items():
            state = engine.admit(rid, prefix, ["D", "E", "D"], max_suffix_tokens=8)
            assert state.total_context_tokens > original_scaling["original_max_position_embeddings"]
        for step in range(2):
            ids = torch.tensor([chunks[rid][step] for rid in ("A", "B")])
            actual = engine.forward(["A", "B"], ids)
            oracle = torch.cat([expected[rid][step] for rid in ("A", "B")])
            torch.testing.assert_close(actual, oracle, rtol=8e-5, atol=8e-6)
        assert model.config.rope_scaling == original_scaling
        assert model.model.rotary_emb.rope_type == "yarn"
    finally:
        engine.close()


def test_static_yarn_mixed_heads_relocate_with_unit_rotation_not_attention_scale(runtime):
    model = _tiny_model(rope_type="yarn")
    scaling = model.model.rotary_emb.attention_scaling
    assert scaling > 1
    engine = _engine(runtime, model, window=8)
    try:
        engine.admit("A", [1 + i % 17 for i in range(519)], ["D", "E", "D"], max_suffix_tokens=8)
        engine.admit("B", [91 + i % 21 for i in range(541)], ["E", "D"], max_suffix_tokens=8)
        ids = torch.tensor([[81, 82, 83], [111, 112, 113]])
        oracle = _independent_query_oracle(engine, ["A", "B"], ids)
        actual = engine.forward(["A", "B"], ids)
        torch.testing.assert_close(actual, oracle, rtol=8e-5, atol=8e-6)
        query = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16) / 31
        # Native YaRN Q/K already contain their attention-scale multiplier.
        # Relocation must only rotate, never apply that multiplier again.
        shifted = engine._counterrotate(query * scaling, 541)
        torch.testing.assert_close(shifted.norm(dim=-1), (query * scaling).norm(dim=-1),
                                   rtol=1e-6, atol=1e-6)
        assert model.config.rope_scaling["rope_type"] == "yarn"
    finally:
        engine.close()


@pytest.mark.parametrize("mutation", ["factor", "attention_scaling", "inv_freq", "rotary_object"])
def test_static_yarn_cache_contract_rejects_rope_mutation_but_allows_cleanup(runtime, mutation):
    model = _tiny_model(rope_type="yarn")
    engine = _engine(runtime, model)
    try:
        engine.admit("A", _PREFIXES["A"], ["D"], max_suffix_tokens=8)
        if mutation == "factor":
            model.config.rope_scaling["factor"] = 8.0
        elif mutation == "attention_scaling":
            model.model.rotary_emb.attention_scaling *= 1.1
        elif mutation == "inv_freq":
            model.model.rotary_emb.inv_freq.mul_(1.1)
        else:
            replacement = _tiny_model(rope_type="yarn").model.rotary_emb
            model.model.rotary_emb = replacement
        with pytest.raises(RuntimeError, match="RoPE .* changed"):
            engine.forward(["A"], torch.tensor([[81]]))
        assert engine.requests["A"].suffix_tokens == 0
        assert not engine._busy
    finally:
        engine.close()
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sink", [0, 4])
@pytest.mark.parametrize("rope_type", ["default", "yarn"])
def test_cuda_flash_unaligned_nonprefix_windows_match_torch_and_literal_oracle(runtime, dtype, sink, rope_type):
    pytest.importorskip("flash_attn")
    model = _tiny_model("cuda", dtype, rope_type=rope_type)
    if rope_type == "yarn":
        assert model.model.rotary_emb.attention_scaling > 1
    docs = {"D": [20 + index % 80 for index in range(520)],
            "E": [50 + index % 90 for index in range(300)]}
    chunks = [torch.tensor([[101, 102, 103], [131, 132, 133]], device="cuda"),
              torch.tensor([[104, 105], [134, 135]], device="cuda")]
    results = []
    tolerance = 0.003 if dtype == torch.float16 else 0.05
    original = [layer.self_attn.forward for layer in model.model.layers]
    for backend, shared in (("torch", True), ("flash", False), ("flash", True)):
        engine = runtime.core.NonPrefixPagedEngine(
            model, _head_config(runtime, window=263, sink=sink), capacity_pages=256,
            page_size=256, backend=backend, share_local=shared,
            model_revision="test-seeded-random-qwen3-v1", prefix_chunk_size=256,
        )
        try:
            for name, ids in docs.items():
                engine.register_document(name, ids)
            source_hash = _hash_source_pages(engine)
            engine.admit("A", [1, 2, 3], ["E", "D"], max_suffix_tokens=8)
            engine.admit("B", [9, 8, 7, 6, 5], ["D", "E"], max_suffix_tokens=8)
            # Both start offsets are unaligned, and both end inside a page.
            for rid, source_start in (("A", 257), ("B", 37)):
                parts = [p for p in engine.requests[rid].partitions[1, 0] if p.canonical]
                assert min(p.source_start for p in parts) == source_start
                assert sum(p.end for p in parts) == 263
                assert len(parts) >= 2
            oracle = _independent_query_oracle(engine, ["A", "B"], chunks[0])
            if backend == "flash":
                native = engine._flash
                calls = []

                def observed(query, keys, values, **kwargs):
                    assert "cache_leftpad" not in kwargs, "paged FA2 does not support left padding"
                    assert kwargs["k"] is None and kwargs["v"] is None
                    assert kwargs["return_softmax_lse"] is True
                    assert query.shape[2] == 2, "GQA groups remain paired to physical KV heads"
                    calls.append((query.shape[0], kwargs["causal"]))
                    return native(query, keys, values, **kwargs)

                engine._flash = observed
            outputs = [engine.forward(["A", "B"], chunk).clone() for chunk in chunks]
            assert (outputs[0].float() - oracle.float()).abs().max().item() <= tolerance
            assert _hash_source_pages(engine) == source_hash
            assert all(state.suffix_tokens == 5 for state in engine.requests.values())
            if backend == "flash":
                assert len(calls) == len(model.model.layers) * len(chunks) * 2
                assert any(not causal for _, causal in calls) and any(causal for _, causal in calls)
            results.append(outputs)
            assert [layer.self_attn.forward for layer in model.model.layers] == original
        finally:
            engine.close()
        assert engine.pool.stats()["pinned_pages"] == 0
        assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
    for candidate in results[1:]:
        for reference, actual in zip(results[0], candidate):
            assert (reference.float() - actual.float()).abs().max().item() <= tolerance


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rope_type", ["default", "yarn"])
def test_cuda_all_global_flash_matches_native_recompute(runtime, dtype, rope_type):
    pytest.importorskip("flash_attn")
    model = _tiny_model("cuda", dtype, rope_type=rope_type)
    chunks = {"A": [[81, 82, 83], [84, 85]], "B": [[91, 92, 93], [94, 95]]}
    expected = {rid: _native_logits(model, prefix, ["D", "E", "D"], chunks[rid])
                for rid, prefix in _PREFIXES.items()}
    engine = _engine(runtime, model, mixed=False, backend="flash", page_size=256)
    tolerance = 0.003 if dtype == torch.float16 else 0.05
    try:
        for rid, prefix in _PREFIXES.items():
            engine.admit(rid, prefix, ["D", "E", "D"], max_suffix_tokens=8)
        for step in range(2):
            ids = torch.tensor([chunks[rid][step] for rid in ("A", "B")], device="cuda")
            actual = engine.forward(["A", "B"], ids)
            oracle = torch.cat([expected[rid][step] for rid in ("A", "B")])
            assert (actual.float() - oracle.float()).abs().max().item() <= tolerance
    finally:
        engine.close()
