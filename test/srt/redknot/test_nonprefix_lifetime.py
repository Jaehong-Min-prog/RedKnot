"""Request-lifetime historical leases: no per-token ownership/event scans."""
import pytest
import torch

from test_nonprefix_paged import (_isolated_imports, _tiny_model, _engine,
                                 _PREFIXES, _native_logits)


@pytest.fixture(scope="module")
def runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with _isolated_imports() as modules:
            yield modules
    finally:
        torch.set_num_threads(previous)


def _pin_snapshot(pool):
    return {ref: page.pins for page in pool._pages.values() for ref in (page.ref,)}


@pytest.mark.parametrize("steps", [1, 8])
def test_decode_performs_zero_bind_pin_unpin_collect_or_completion_calls(runtime, monkeypatch, steps):
    engine = _engine(runtime, _tiny_model())
    try:
        state = engine.admit("A", _PREFIXES["A"], ["D", "E"], max_suffix_tokens=16)
        lease = state.history_lease
        assert lease.version is engine.manager.version(state.manager_request_id)
        assert not lease.closed and not state.history_completed
        pinned = _pin_snapshot(engine.pool)
        calls = dict(bind=0, pin=0, unpin=0, collect=0, completion=0)
        with monkeypatch.context() as patch:
            for name, obj in (("bind", engine.manager), ("pin", engine.pool),
                              ("unpin", engine.pool), ("collect", engine.pool)):
                native = getattr(obj, name)
                def observed(*args, _name=name, _native=native, **kwargs):
                    calls[_name] += 1
                    return _native(*args, **kwargs)
                patch.setattr(obj, name, observed)
            native_completion = runtime.core.completion_event
            def completion(device):
                calls["completion"] += 1
                return native_completion(device)
            patch.setattr(runtime.core, "completion_event", completion)
            for step in range(steps):
                logits = engine.forward(["A"], torch.tensor([[81+step]]))
                assert torch.isfinite(logits).all()
        assert calls == dict(bind=0, pin=0, unpin=0, collect=0, completion=0)
        assert _pin_snapshot(engine.pool) == pinned
        assert engine.history_lease_acquisitions == 1
        assert engine.history_lease_completions == 0
        assert state.suffix_tokens == steps
        assert not lease.closed
        engine.release("A")
        assert lease.closed and state.history_completed
        assert engine.history_lease_completions == 1
    finally:
        engine.close()
    assert engine.pool.stats()["pinned_pages"] == 0
    assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages


def test_lifetime_lease_preserves_batched_native_allglobal_logits(runtime):
    model = _tiny_model(rope_type="yarn")
    questions = {"A": [[81, 82, 83], [84], [85]], "B": [[91, 92, 93], [94], [95]]}
    expected = {rid: _native_logits(model, prefix, ["D", "E"], questions[rid])
                for rid, prefix in _PREFIXES.items()}
    engine = _engine(runtime, model, mixed=False)
    try:
        states = [engine.admit(rid, prefix, ["D", "E"], max_suffix_tokens=8)
                  for rid, prefix in _PREFIXES.items()]
        for step in range(3):
            result = engine.forward(["A", "B"], torch.tensor([questions[r][step] for r in ("A", "B")]))
            torch.testing.assert_close(result, torch.cat([expected[r][step] for r in ("A", "B")]),
                                       rtol=5e-5, atol=5e-6)
        assert all(not s.history_lease.closed for s in states)
    finally:
        engine.close()


def test_release_uses_one_shared_event_for_history_and_tail(runtime, monkeypatch):
    engine = _engine(runtime, _tiny_model())
    try:
        state = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
        engine.forward(["A"], torch.tensor([[81, 82]]))
        events, unpins = [], []
        native_event, native_unpin = runtime.core.completion_event, engine.pool.unpin
        with monkeypatch.context() as patch:
            def event(device):
                value = native_event(device)
                events.append(value)
                return value
            def unpin(refs, event=None):
                unpins.append((tuple(refs), event))
                return native_unpin(refs, event)
            patch.setattr(runtime.core, "completion_event", event)
            patch.setattr(engine.pool, "unpin", unpin)
            engine.release("A")
        assert len(events) == 1
        assert len(unpins) == 2  # Immutable root lease, then exclusive tail arena.
        assert unpins[0][0] == state.history_lease.refs
        assert all(event is events[0] for _, event in unpins)
        assert state.history_completed and state.history_lease.closed
    finally:
        engine.close()


def test_release_does_not_recycle_history_or_tail_before_final_event(runtime, monkeypatch):
    engine = _engine(runtime, _tiny_model(), mixed=False)
    class Gate:
        ready = False
        def query(self):
            return self.ready
        def synchronize(self):
            if not self.ready:
                raise RuntimeError("test completion gate not ready")
    gate = Gate()
    try:
        state = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
        engine.forward(["A"], torch.tensor([[81, 82]]))
        allocated = engine.pool.stats()["capacity_pages"]-engine.pool.stats()["free_pages"]
        with monkeypatch.context() as patch:
            patch.setattr(runtime.core, "completion_event", lambda device: gate)
            engine.release("A")
        assert state.history_completed and state.history_lease.closed
        assert not engine.requests
        assert engine.pool.stats()["retiring_pages"] == allocated
        with pytest.raises(runtime.core.CapacityError):
            engine.pool.reserve(engine.pool.capacity_pages)
        gate.ready = True
        engine.pool.collect()
        assert engine.pool.stats()["free_pages"] == engine.pool.capacity_pages
    finally:
        gate.ready = True
        engine.close()


def test_final_admission_bind_failure_is_atomic_and_preserves_other_request(runtime, monkeypatch):
    engine = _engine(runtime, _tiny_model())
    try:
        active = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
        before = engine.pool.stats()
        pins = _pin_snapshot(engine.pool)
        original_build = engine._build_partitions
        native_bind = engine.manager.bind
        final_request = [None]
        with monkeypatch.context() as patch:
            def build(state):
                original_build(state)
                final_request[0] = state.manager_request_id
            def bind(request_id):
                if request_id == final_request[0]:
                    raise RuntimeError("injected final admission lease failure")
                return native_bind(request_id)
            patch.setattr(engine, "_build_partitions", build)
            patch.setattr(engine.manager, "bind", bind)
            with pytest.raises(RuntimeError, match="final admission lease"):
                engine.admit("B", [9, 8, 7, 6], ["E", "D"], max_suffix_tokens=8)
        assert set(engine.requests) == {"A"}
        assert not active.history_lease.closed
        assert _pin_snapshot(engine.pool) == pins
        after = engine.pool.stats()
        for key in ("free_pages", "allocated_bytes", "reserved_pages", "pinned_pages"):
            assert after[key] == before[key]
        assert torch.isfinite(engine.forward(["A"], torch.tensor([[81]]))).all()
    finally:
        engine.close()


def test_failed_model_forward_keeps_history_pinned_until_release(runtime, monkeypatch):
    model = _tiny_model()
    engine = _engine(runtime, model)
    original = [layer.self_attn.forward for layer in model.model.layers]
    try:
        state = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
        before = _pin_snapshot(engine.pool)
        native = engine._torch_attention
        with monkeypatch.context() as patch:
            def fail(q, states, layer, scale):
                if layer == 1:
                    raise RuntimeError("injected middle-layer failure")
                return native(q, states, layer, scale)
            patch.setattr(engine, "_torch_attention", fail)
            with pytest.raises(RuntimeError, match="middle-layer"):
                engine.forward(["A"], torch.tensor([[81, 82]]))
        assert state.failed and not state.history_lease.closed
        assert state.suffix_tokens == 0
        assert _pin_snapshot(engine.pool) == before
        assert [layer.self_attn.forward for layer in model.model.layers] == original
        assert not engine._busy and not hasattr(model, "_redknot_nonprefix_owner")
        engine.release("A")
        assert state.history_completed
    finally:
        engine.close()


def test_root_mutation_is_rejected_without_executing_model(runtime):
    engine = _engine(runtime, _tiny_model())
    try:
        state = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
        old = state.history_lease.version
        # Even an empty explicit manager publication changes the immutable
        # root identity; the adapter cannot silently follow external edits.
        engine.manager.update(state.manager_request_id)
        assert engine.manager.version(state.manager_request_id) is not old
        with pytest.raises(RuntimeError, match="history lease/root changed"):
            engine.forward(["A"], torch.tensor([[81]]))
        assert state.failed and engine.attention_calls == 0
        assert not state.history_lease.closed
    finally:
        engine.close()


def test_release_event_failure_retains_root_and_can_be_retried_after_sync(runtime, monkeypatch):
    engine = _engine(runtime, _tiny_model())
    try:
        state = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
        before = _pin_snapshot(engine.pool)
        with monkeypatch.context() as patch:
            def fail(device):
                raise RuntimeError("injected event record failure")
            patch.setattr(runtime.core, "completion_event", fail)
            with pytest.raises(RuntimeError, match="event record"):
                engine.release("A")
        assert engine.requests["A"] is state
        assert state.manager_request_id in engine.manager._requests
        assert not state.history_lease.closed and not state.history_completed
        assert state.failed and engine.pool.poisoned
        assert _pin_snapshot(engine.pool) == before
        engine.close()
        assert not engine.requests
        assert all(p.owners == 0 and p.pins == 0 for p in engine.pool._pages.values())
    finally:
        engine.close()


def test_open_lease_completion_failure_can_retry_synchronized_teardown(runtime, monkeypatch):
    engine = _engine(runtime, _tiny_model())
    try:
        state = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
        before = _pin_snapshot(engine.pool)
        with monkeypatch.context() as patch:
            def fail(event=None):
                raise RuntimeError("injected pre-unpin completion failure")
            patch.setattr(state.history_lease, "complete", fail)
            with pytest.raises(RuntimeError, match="pre-unpin"):
                engine.release("A")
        assert state.history_release_failed and not state.history_completion_uncertain
        assert not state.history_lease.closed
        assert _pin_snapshot(engine.pool) == before
        assert engine.requests["A"] is state
        engine.close()
        assert state.history_completed
        assert all(p.owners == 0 and p.pins == 0 for p in engine.pool._pages.values())
    finally:
        engine.close()


def test_uncertain_closed_lease_completion_quarantines_root_without_double_unpin(runtime, monkeypatch):
    engine = _engine(runtime, _tiny_model())
    state = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
    native = state.history_lease.complete
    with monkeypatch.context() as patch:
        def fail(event=None):
            native(event)
            raise RuntimeError("injected post-unpin completion failure")
        patch.setattr(state.history_lease, "complete", fail)
        with pytest.raises(RuntimeError, match="post-unpin"):
            engine.release("A")
    assert state.history_lease.closed and not state.history_completed
    assert state.history_completion_uncertain and engine.pool.poisoned
    assert state.manager_request_id in engine.manager._requests
    assert state.tail_pinned and state.tail_all_refs
    pins = _pin_snapshot(engine.pool)
    with pytest.raises(RuntimeError, match="completion uncertain"):
        engine.close()
    assert _pin_snapshot(engine.pool) == pins
    assert engine.requests["A"] is state
    # Deliberately quarantined CPU-only fixture. Production must abandon the
    # failed engine/context rather than infer whether an uncertain pin moved.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lifetime_lease_rejects_foreign_cuda_stream(runtime):
    model = _tiny_model(device="cuda", dtype=torch.bfloat16)
    engine = _engine(runtime, model, backend="flash", page_size=256)
    try:
        state = engine.admit("A", [1, 2, 3], ["D"], max_suffix_tokens=8)
        with torch.cuda.stream(torch.cuda.Stream()):
            with pytest.raises(RuntimeError, match="original CUDA stream"):
                engine.forward(["A"], torch.tensor([[81]], device="cuda"))
        assert not state.history_lease.closed and state.suffix_tokens == 0
        result = engine.forward(["A"], torch.tensor([[81]], device="cuda"))
        assert torch.isfinite(result).all()
    finally:
        engine.close()
