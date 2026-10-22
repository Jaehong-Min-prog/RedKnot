"""A single-owner, genuinely batched queue for the non-prefix model experiment.

This is an experimental scheduler, not an integration with SGLang's production
scheduler. The engine supplies admission (including online prefix/document
recovery), batched model forwards, and release. The scheduler never substitutes
a native/full-context answer for an answer produced by that engine.

The measured interval begins with an empty request queue and a prepared engine;
all request arrivals, serial admission/recovery, query/decode forwards, sampling,
and request cleanup are included. Offline document construction, model loading,
and engine construction happen outside this function and must be reported by
the caller separately. Admission is deliberately serialized. Query forwards
are bucketed by exact query length without padding, while ready decodes form a
true batch, including requests admitted at different times.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass
import math
from numbers import Integral
import threading
import time
import traceback
from typing import Any, Sequence

import torch

from sglang.srt.mem_cache.head_kv.pool import CapacityError


_OWNERS: set[int] = set()
_OWNERS_LOCK = threading.Lock()


@dataclass(frozen=True)
class QueuedRequest:
    request_id: str
    prefix_ids: Sequence[int]
    document_ids: Sequence[str]
    query_ids: Sequence[int]
    max_new_tokens: int
    eos_token_ids: Sequence[int]
    arrival_seconds: float = 0.0
    cancel_after_tokens: int | None = None

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a nonempty string")
        for name in ("prefix_ids", "query_ids", "eos_token_ids"):
            values = tuple(getattr(self, name))
            if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 0
                   for v in values):
                raise ValueError(f"{name} must contain nonnegative integer token IDs")
            object.__setattr__(self, name, tuple(int(v) for v in values))
        documents = tuple(self.document_ids)
        if any(not isinstance(v, str) or not v for v in documents):
            raise ValueError("document_ids must contain nonempty document IDs")
        object.__setattr__(self, "document_ids", documents)
        if not self.query_ids:
            raise ValueError("query_ids must not be empty")
        if (not isinstance(self.max_new_tokens, Integral)
                or isinstance(self.max_new_tokens, bool) or self.max_new_tokens <= 0):
            raise ValueError("max_new_tokens must be a positive integer")
        object.__setattr__(self, "max_new_tokens", int(self.max_new_tokens))
        arrival = float(self.arrival_seconds)
        if not math.isfinite(arrival) or arrival < 0:
            raise ValueError("arrival_seconds must be finite and nonnegative")
        object.__setattr__(self, "arrival_seconds", arrival)
        if self.cancel_after_tokens is not None:
            if (not isinstance(self.cancel_after_tokens, Integral)
                    or isinstance(self.cancel_after_tokens, bool)
                    or self.cancel_after_tokens < 0):
                raise ValueError("cancel_after_tokens must be a nonnegative integer")
            object.__setattr__(self, "cancel_after_tokens", int(self.cancel_after_tokens))


@dataclass
class _Active:
    request: QueuedRequest
    queried: bool = False


def _synchronize(engine):
    sync = getattr(engine, "synchronize", None)
    if callable(sync):
        sync()
    else:
        device = torch.device(getattr(engine, "device", "cpu"))
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def _stats(engine):
    value = getattr(engine, "stats", None)
    return value() if callable(value) else value


def run_queued(engine, requests: Sequence[QueuedRequest], *, max_active: int,
               ignore_eos: bool = False, admission_policy: str = "fcfs",
               affinity_config=None, affinity_index=None) -> dict[str, Any]:
    """Execute bounded admissions and dynamic batches; return JSON-friendly evidence.

    FCFS is unchanged by default. ``share_affinity`` only reorders arrived
    candidates within a bounded lookahead using canonical source-page locality;
    it never modifies a request, grants sharing, or waits for a future match.
    Hard age/bypass bounds protect old requests, including draining active work
    when a protected old request needs capacity. This is single-node scheduling,
    not distributed routing or a claim of additional saved KV allocations.

    ``engine.admit(id, prefix_ids, document_ids, max_suffix_tokens)`` must be
    atomic on failure: all partial request allocations must already be freed
    before raising. A ``CapacityError`` waits for a release if other requests
    are active; with no active request it marks the request impossible to admit.
    Other engine errors abort the run, mark every unfinished request, and still
    attempt cleanup of every admitted request. No model retry/fallback occurs.

    Only an engine explicitly advertising ``requires_recovery_query is True``
    receives the exact request ``query_ids`` during admission. Such a probe
    must be transient: it must not publish query KV or advance suffix state.
    Normal query forwarding below remains unchanged and is not skipped.

    ``engine.forward(ids, tensor[B,T])`` must accept mixed request context
    lengths and return finite logits[B,1,V]. It must append every input token to
    the corresponding engine state. A query emits the first output token; each
    subsequent model input is the previous generated token. Suffix reservation
    is therefore query_length + max_new_tokens - 1, not the number of outputs.

    All timestamps are seconds from the service start. TTFT is first-token
    availability minus scheduled arrival, including queueing and recovery.
    Per-request batch-stage times overlap and must not be summed across rows.
    The headline QPS counts only successfully completed requests and divides by
    the entire measured wall time, including idle arrival gaps and cleanup.
    """
    if not isinstance(max_active, Integral) or isinstance(max_active, bool) or max_active <= 0:
        raise ValueError("max_active must be a positive integer")
    if admission_policy not in ("fcfs", "share_affinity"):
        raise ValueError("admission_policy must be fcfs or share_affinity")
    if admission_policy == "share_affinity":
        from .nonprefix_affinity import CanonicalAffinityIndex, ShareAffinityConfig, ShareAffinitySelector
        affinity_config = affinity_config if affinity_config is not None else ShareAffinityConfig()
        if not isinstance(affinity_config, ShareAffinityConfig):
            raise TypeError("affinity_config must be ShareAffinityConfig")
    requests = list(requests)
    if any(not isinstance(request, QueuedRequest) for request in requests):
        raise TypeError("requests must contain QueuedRequest objects")
    identifiers = [request.request_id for request in requests]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("request IDs must be unique within a run")
    with _OWNERS_LOCK:
        if id(engine) in _OWNERS:
            raise RuntimeError("this engine already has an active queue owner")
        _OWNERS.add(id(engine))

    records = {
        request.request_id: {
            "request_id": request.request_id, "status": "pending",
            "arrival_seconds": request.arrival_seconds,
            "prefix_tokens": len(request.prefix_ids),
            "document_ids": list(request.document_ids),
            "query_tokens": len(request.query_ids),
            "max_new_tokens": request.max_new_tokens,
            "reserved_suffix_tokens": len(request.query_ids) + request.max_new_tokens - 1,
            "cancel_after_tokens": request.cancel_after_tokens,
            "output_ids": [], "finish_reason": None,
            "admission_attempts": 0, "capacity_blocks": 0,
            "first_admission_attempt_seconds": None,
            "admission_started_seconds": None, "admission_finished_seconds": None,
            "model_started_seconds": None, "first_token_seconds": None,
            "generation_finished_seconds": None, "completion_seconds": None,
            "queue_wait_seconds": None, "ttft_seconds": None,
            "service_ttft_seconds": None, "latency_seconds": None,
            "admission_attempt_seconds": 0.0, "admission_seconds": 0.0,
            "admission_stats": None, "doc_offsets": None, "history_tokens": None,
            "query_forward_seconds": 0.0, "decode_forward_seconds": 0.0,
            "release_seconds": 0.0,
        }
        for request in requests
    }
    pending = sorted(enumerate(requests), key=lambda pair: (pair[1].arrival_seconds, pair[0]))
    cursor = 0
    active: OrderedDict[str, _Active] = OrderedDict()
    trace: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    stages = {name: 0.0 for name in (
        "admission", "query_forward", "decode_forward", "release", "idle_wait",
    )}
    release_epoch = 0
    blocked: tuple[str, int] | None = None
    prefer_query = True
    peak_active = 0
    max_forward_batch = 0
    engine_before = engine_after = None
    selector = None
    affinity_setup_seconds = 0.0
    service_start = time.perf_counter()
    clock = lambda: time.perf_counter() - service_start

    def event(kind, **kwargs):
        trace.append({"event": kind, "seconds": clock(), **kwargs})

    def fail(record, stage, exc, *, code=None):
        record.update(status="failed", finish_reason=code or stage,
                      error={"stage": stage, "type": type(exc).__name__, "message": str(exc)})
        event("request_failed", request_id=record["request_id"], **record["error"])

    def release(request_id):
        nonlocal release_epoch
        began = clock()
        try:
            engine.release(request_id)
            _synchronize(engine)
        finally:
            elapsed = clock() - began
            stages["release"] += elapsed
            records[request_id]["release_seconds"] += elapsed
        del active[request_id]
        release_epoch += 1
        if selector is not None:
            selector.on_released(request_id)
        event("released", request_id=request_id, elapsed_seconds=elapsed,
              active_count=len(active))

    def finish(request_id, status, reason, generated_at):
        record = records[request_id]
        record["generation_finished_seconds"] = generated_at
        # Completion is only successful after ownership/resources are released.
        release(request_id)
        completed_at = clock()
        record.update(status=status, finish_reason=reason,
                      completion_seconds=completed_at,
                      latency_seconds=completed_at - record["arrival_seconds"])
        event(status, request_id=request_id, output_tokens=len(record["output_ids"]))

    try:
        # Drain earlier CUDA work before opening the timed service interval.
        _synchronize(engine)
        engine_before = _stats(engine)
        service_start = time.perf_counter()
        if admission_policy == "share_affinity":
            began = clock()
            if affinity_index is None:
                affinity_index = CanonicalAffinityIndex.from_engine(engine)
            validate = getattr(affinity_index, "validate_engine", None)
            if callable(validate):
                validate(engine)
            selector = ShareAffinitySelector(affinity_index, requests, affinity_config)
            affinity_setup_seconds = clock() - began
            event("affinity_index_ready", elapsed_seconds=affinity_setup_seconds,
                  config=affinity_config.to_dict())
        while cursor < len(pending) or active:
            # Single-owner admission. Recovery is never hidden off-clock.
            while cursor < len(pending) and len(active) < max_active:
                selected, decision = cursor, None
                if selector is not None:
                    candidates = [pair[1] for pair in pending[cursor:cursor + affinity_config.lookahead]]
                    decision = selector.choose(candidates, active, now=clock(), release_epoch=release_epoch)
                    if decision is None:
                        break
                    selected += decision.selected_offset
                request = pending[selected][1]
                if request.arrival_seconds > clock():
                    break
                record = records[request.request_id]
                if selector is None and blocked == (request.request_id, release_epoch):
                    break
                if decision is not None:
                    event("affinity_selected", request_id=request.request_id,
                          release_epoch=release_epoch, active_count=len(active), **decision.to_dict())
                if request.cancel_after_tokens == 0:
                    completed_at = clock()
                    record.update(status="cancelled", finish_reason="cancel_after_tokens",
                                  completion_seconds=completed_at,
                                  latency_seconds=completed_at - request.arrival_seconds)
                    event("cancelled", request_id=request.request_id, output_tokens=0)
                    if selected != cursor:
                        pending.insert(cursor, pending.pop(selected))
                    cursor += 1
                    continue
                began = clock()
                record["admission_attempts"] += 1
                if record["first_admission_attempt_seconds"] is None:
                    record["first_admission_attempt_seconds"] = began
                event("admission_started", request_id=request.request_id,
                      active_count=len(active))
                capacity_error = None
                try:
                    recovery_kwargs = ({"query_ids": request.query_ids}
                                       if getattr(engine, "requires_recovery_query", False) is True else {})
                    admitted_state = engine.admit(
                        request.request_id, request.prefix_ids, request.document_ids,
                        max_suffix_tokens=record["reserved_suffix_tokens"], **recovery_kwargs)
                except CapacityError as exc:
                    capacity_error = exc
                    _synchronize(engine)
                else:
                    # Track ownership before synchronization: failed sync must
                    # still release an otherwise successfully admitted request.
                    active[request.request_id] = _Active(request)
                    _synchronize(engine)
                finally:
                    elapsed = clock() - began
                    stages["admission"] += elapsed
                    record["admission_attempt_seconds"] += elapsed
                if capacity_error is not None:
                    record["capacity_blocks"] += 1
                    if selector is not None:
                        selector.on_capacity_block(request.request_id, release_epoch)
                    event("admission_capacity_block", request_id=request.request_id,
                          elapsed_seconds=elapsed, active_count=len(active),
                          message=str(capacity_error))
                    if active:
                        if selector is not None:
                            continue
                        else:
                            blocked = (request.request_id, release_epoch)
                            break
                    fail(record, "admission", capacity_error, code="impossible_capacity")
                    record["completion_seconds"] = clock()
                    record["latency_seconds"] = clock() - request.arrival_seconds
                    if selected != cursor:
                        pending.insert(cursor, pending.pop(selected))
                    cursor += 1
                    continue
                if selector is not None:
                    selector.on_admitted(decision, [pair[1] for pair in pending[cursor:selected]
                                                   if pair[1].arrival_seconds <= began])
                    record["admission_affinity"] = decision.to_dict()
                record.update(status="active", admission_started_seconds=began,
                              admission_finished_seconds=clock(), admission_seconds=elapsed,
                              queue_wait_seconds=began - request.arrival_seconds)
                # Snapshot cheap request-local provenance, not a scan of global
                # GPU allocation state. Preserve it after release and mutation.
                record["admission_stats"] = copy.deepcopy(
                    getattr(admitted_state, "admission_stats", None))
                offsets = getattr(admitted_state, "doc_offsets",
                                  getattr(admitted_state, "document_offsets", None))
                record["doc_offsets"] = list(offsets) if offsets is not None else None
                record["history_tokens"] = getattr(admitted_state, "history_tokens",
                                                   getattr(admitted_state, "total_context_tokens", None))
                peak_active = max(peak_active, len(active))
                event("admitted", request_id=request.request_id, elapsed_seconds=elapsed,
                      active_count=len(active))
                if selected != cursor:
                    pending.insert(cursor, pending.pop(selected))
                cursor += 1
                blocked = None

            if not active:
                if cursor < len(pending):
                    delay = max(0.0, pending[cursor][1].arrival_seconds - clock())
                    if delay:
                        began = clock()
                        time.sleep(min(delay, 0.01))
                        stages["idle_wait"] += clock() - began
                continue

            query_ids = [key for key, state in active.items() if not state.queried]
            decode_ids = [key for key, state in active.items() if state.queried]
            if query_ids and (prefer_query or not decode_ids):
                width = len(active[query_ids[0]].request.query_ids)
                batch_ids = [key for key in query_ids
                             if len(active[key].request.query_ids) == width]
                inputs = [active[key].request.query_ids for key in batch_ids]
                stage = "query_forward"
                prefer_query = False
            else:
                batch_ids = decode_ids
                inputs = [[records[key]["output_ids"][-1]] for key in batch_ids]
                width, stage = 1, "decode_forward"
                prefer_query = True

            began = clock()
            for request_id in batch_ids:
                if records[request_id]["model_started_seconds"] is None:
                    records[request_id]["model_started_seconds"] = began
            batch_event = {
                "event": "forward", "seconds": began, "stage": stage,
                "request_ids": list(batch_ids), "batch_size": len(batch_ids),
                "input_tokens_per_request": width, "active_count": len(active),
                "status": "running",
            }
            trace.append(batch_event)
            max_forward_batch = max(max_forward_batch, len(batch_ids))
            try:
                input_ids = torch.tensor(inputs, dtype=torch.long, device=engine.device)
                with torch.inference_mode():
                    logits = engine.forward(batch_ids, input_ids)
                _synchronize(engine)
                if (not isinstance(logits, torch.Tensor) or logits.ndim != 3
                        or logits.shape[0] != len(batch_ids) or logits.shape[1] != 1
                        or logits.shape[2] < 1):
                    raise ValueError("engine.forward must return logits[B,1,V]")
                if not bool(torch.isfinite(logits).all().item()):
                    raise ValueError("engine.forward returned non-finite logits")
                next_tokens = logits[:, 0].argmax(dim=-1).cpu().tolist()
                _synchronize(engine)
            except Exception:
                batch_event["status"] = "failed"
                raise
            finally:
                elapsed = clock() - began
                stages[stage] += elapsed
                batch_event.update(elapsed_seconds=elapsed, finished_seconds=clock())
                for request_id in batch_ids:
                    records[request_id][stage + "_seconds"] += elapsed
            generated_at = clock()
            batch_event["status"] = "complete"
            # Drop the logits reference before serialized admissions can need
            # that memory; recorded IDs, not a native seed, drive the next step.
            del logits, input_ids
            finished = []
            for request_id, token in zip(batch_ids, next_tokens):
                state, record = active[request_id], records[request_id]
                state.queried = True
                record["output_ids"].append(int(token))
                if record["first_token_seconds"] is None:
                    record.update(first_token_seconds=generated_at,
                                  ttft_seconds=generated_at - state.request.arrival_seconds,
                                  service_ttft_seconds=generated_at - record["admission_started_seconds"])
                    event("first_token", request_id=request_id, token_id=int(token),
                          available_seconds=generated_at)
                count = len(record["output_ids"])
                if not ignore_eos and token in state.request.eos_token_ids:
                    finished.append((request_id, "completed", "eos"))
                elif count >= state.request.max_new_tokens:
                    finished.append((request_id, "completed", "length"))
                elif (state.request.cancel_after_tokens is not None
                      and count >= state.request.cancel_after_tokens):
                    finished.append((request_id, "cancelled", "cancel_after_tokens"))
            # Record all model outputs before a release error can abort this
            # batch; logits were available for every row at generated_at.
            for request_id, status, reason in finished:
                finish(request_id, status, reason, generated_at)
            # Released slots are refilled at the top of the very next loop,
            # before another query or decode model forward is issued.
    except Exception as exc:
        errors.append({"stage": "service", "type": type(exc).__name__,
                       "message": str(exc), "traceback": traceback.format_exc()})
        for record in records.values():
            if record["status"] in ("pending", "active"):
                fail(record, "service_aborted", exc)
    finally:
        # Never allow an engine error (or cancellation) to strand other owners.
        for request_id in list(active):
            try:
                release(request_id)
            except Exception as exc:
                errors.append({"stage": "cleanup", "request_id": request_id,
                               "type": type(exc).__name__, "message": str(exc)})
                fail(records[request_id], "cleanup", exc)
            record = records[request_id]
            record["completion_seconds"] = clock()
            record["latency_seconds"] = clock() - record["arrival_seconds"]
        try:
            _synchronize(engine)
        except Exception as exc:
            errors.append({"stage": "final_synchronize", "type": type(exc).__name__,
                           "message": str(exc)})
        total_seconds = clock()
        # Statistics collection is outside the service metric, unlike release.
        try:
            engine_after = _stats(engine)
        except Exception as exc:
            errors.append({"stage": "statistics", "type": type(exc).__name__,
                           "message": str(exc)})
        with _OWNERS_LOCK:
            _OWNERS.discard(id(engine))

    rows = [records[request.request_id] for request in requests]
    completed = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] == "failed"]
    cancelled = [row for row in rows if row["status"] == "cancelled"]
    completed_tokens = sum(len(row["output_ids"]) for row in completed)
    return {
        "schema": "redknot-nonprefix-queued-service-v1",
        "status": "failed" if errors else ("complete_with_failed_requests" if failed else "complete"),
        "timing_scope": "prepared engine; arrivals + serialized admission/online recovery + batched query/decode + sampling + release",
        "excluded_setup": "model loading, engine construction, offline document construction",
        "scheduler": ("single-owner FCFS admission, exact-length query buckets, alternating ready decode batches"
                      if admission_policy == "fcfs" else
                      "single-owner bounded share-affinity admission, exact-length query buckets, alternating ready decode batches"),
        "admission_policy": admission_policy,
        "affinity": ({**selector.stats(), "index_setup_seconds": affinity_setup_seconds,
                      "index_setup_included_in_service_wall": True} if selector is not None else None),
        "max_active": int(max_active), "ignore_eos": bool(ignore_eos),
        "requested_requests": len(rows), "completed_requests": len(completed),
        "failed_requests": len(failed), "cancelled_requests": len(cancelled),
        "completed_output_tokens": completed_tokens,
        "all_emitted_output_tokens": sum(len(row["output_ids"]) for row in rows),
        "total_wall_seconds": total_seconds,
        "completed_qps": len(completed) / total_seconds if total_seconds else 0.0,
        "completed_output_tokens_per_second": completed_tokens / total_seconds if total_seconds else 0.0,
        "peak_active_requests": peak_active, "max_forward_batch": max_forward_batch,
        "stages_seconds": stages,
        "scheduler_overhead_seconds": max(0.0, total_seconds - sum(stages.values())),
        "engine_stats_before": engine_before, "engine_stats_after": engine_after,
        "records": rows, "trace": trace, "errors": errors,
    }
