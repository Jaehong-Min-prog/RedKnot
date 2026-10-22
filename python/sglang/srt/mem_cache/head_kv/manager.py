"""Transactional per-head KV ownership, occurrence sharing, COW and read leases."""
from __future__ import annotations

import math
import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from .pool import HeadPagePool, PageRef, completion_event

SegmentKey = tuple[int, int, str]


@dataclass(frozen=True)
class Segment:
    key: SegmentKey
    pages: tuple[PageRef, ...]
    positions: tuple[int, ...]
    provenance: str
    position_basis: str = "none"
    reuse_kind: str = "exact_context"
    validity_certificate: str = ""

    @property
    def length(self):
        return len(self.positions)


@dataclass(frozen=True)
class RequestVersion:
    request_id: str
    generation: str
    epoch: int
    context_id: str
    contract: str
    namespace: str
    segments: Mapping[SegmentKey, Segment]


@dataclass(frozen=True)
class SegmentWrite:
    key: SegmentKey
    k: torch.Tensor
    v: torch.Tensor
    positions: tuple[int, ...]
    provenance: str
    position_basis: str = "none"
    reuse_kind: str = "exact_context"
    validity_certificate: str = ""


@dataclass(frozen=True)
class SegmentPatch:
    key: SegmentKey
    indices: tuple[int, ...]
    k: torch.Tensor
    v: torch.Tensor
    provenance: str


@dataclass(frozen=True)
class ReuseProof:
    kind: str
    source_provenance: str
    target_context: str
    certificate: str = ""


def _refs(segments):
    # One owner per root, including repeated occurrences in that root.
    return tuple({p for s in segments.values() for p in s.pages})


class ReadLease:
    """Pins a snapshot; complete() records the last consumer's CUDA event."""
    def __init__(self, pool, version):
        self.pool, self.version = pool, version
        self.refs = _refs(version.segments)
        pool.pin(self.refs)
        self.closed = False
        self._descriptors = []

    def complete(self, event=None):
        if self.closed:
            return
        if event is None:
            try:
                event = completion_event(self.pool.device)
            except BaseException:
                self.pool.poisoned = True
                raise
        self.closed = True
        try:
            self.pool.defer_buffers(self._descriptors, event)
            self.pool.unpin(self.refs, event)
        except BaseException:
            self.pool.poisoned = True
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.complete()

    def descriptor(self, layer: int, num_kv_heads: int):
        if self.closed:
            raise RuntimeError("closed read lease")
        if num_kv_heads <= 0:
            raise ValueError("positive head count required")
        groups = [[] for _ in range(num_kv_heads)]
        for (l, h, _), seg in self.version.segments.items():
            if l == layer:
                if not 0 <= h < num_kv_heads:
                    raise ValueError("KV head outside descriptor topology")
                groups[h].append(seg)
        counts = [sum(len(s.pages) for s in segs) for segs in groups]
        width = max(1, max(counts))
        slots = torch.full((num_kv_heads, width), -1, dtype=torch.int64)
        lengths = torch.zeros_like(slots)
        positions = torch.full((num_kv_heads, width, self.pool.page_size), -1, dtype=torch.int64)
        for h, segs in enumerate(groups):
            page_index = 0
            seen = set()
            for seg in segs:
                if seen.intersection(seg.positions):
                    raise ValueError("overlapping logical positions within one KV head")
                seen.update(seg.positions)
                for i, ref in enumerate(seg.pages):
                    pos = seg.positions[i*self.pool.page_size:(i+1)*self.pool.page_size]
                    self.pool.read_page(ref, len(pos))  # validates generation/extent
                    slots[h, page_index] = ref.slot
                    lengths[h, page_index] = len(pos)
                    positions[h, page_index, :len(pos)] = torch.tensor(pos)
                    page_index += 1
        result = dict(k_pool=self.pool.k, v_pool=self.pool.v,
                      page_slots=slots.to(self.pool.device),
                      page_lengths=lengths.to(self.pool.device),
                      key_positions=positions.to(self.pool.device))
        # Keep descriptor buffers alive alongside the consumer lease.
        self._descriptors.append(result)
        return result

    def gather(self, key):
        """Explicit reference/export helper, never used by paged attention."""
        if self.closed:
            raise RuntimeError("closed read lease")
        seg = self.version.segments[key]
        pieces = [self.pool.read_page(p, min(self.pool.page_size, seg.length-i*self.pool.page_size))
                  for i, p in enumerate(seg.pages)]
        if not pieces:
            empty = self.pool.k.new_empty((0, self.pool.head_dim))
            return empty, empty.clone()
        return torch.cat([p[0] for p in pieces]), torch.cat([p[1] for p in pieces])


class WriteTransaction:
    def __init__(self, manager, old, segments, reservations, source_lease, event):
        self.manager, self.old = manager, old
        self.segments, self.reservations = segments, reservations
        self.source_lease, self.event = source_lease, event
        self.status = "PREPARED"
        self.staging = False
        self.cancel_requested = False

    def commit(self, *, wait=False):
        with self.manager._lock:
            if self.status != "PREPARED":
                raise RuntimeError(f"transaction already {self.status}")
            if self.staging:
                raise RuntimeError("transaction import is still staging")
            try:
                if wait:
                    self.event.synchronize()
                if not self.event.query():
                    return None
            except BaseException:
                self.manager.pool.poisoned = True
                self.status = "QUARANTINED"
                raise
            current = self.manager._requests.get(self.old.request_id)
            if self.cancel_requested or current is not self.old:
                self.abort()
                raise RuntimeError("stale request generation/version")
            result = replace(self.old, epoch=self.old.epoch + 1,
                             segments=MappingProxyType(dict(self.segments)))
            try:
                self.manager.pool.retain(_refs(result.segments))
                self.manager._requests[self.old.request_id] = result
                self.manager.pool.release(_refs(self.old.segments))
                self.manager.pool.release(self.reservations, self.event)
                self.source_lease.complete(self.event)
                self.manager._pending.pop(self.old.request_id, None)
                self.status = "COMMITTED"
            except BaseException:
                # Publication is irreversible. Device failure quarantines the
                # whole pool rather than pretending rollback can undo readers.
                self.manager.pool.poisoned = True
                self.status = "QUARANTINED"
                raise
            return result

    def abort(self):
        with self.manager._lock:
            if self.status != "PREPARED":
                return
            if self.staging:
                self.cancel_requested = True
                return
            self.status = "ABORTING"
            try:
                self.manager.pool.release(self.reservations, self.event)
                self.source_lease.complete(self.event)
                if self.manager._pending.get(self.old.request_id) is self:
                    self.manager._pending.pop(self.old.request_id)
                self.status = "ABORTED"
            except BaseException:
                self.manager.pool.poisoned = True
                self.status = "QUARANTINED"
                raise


class HeadKVManager:
    """Bounded immutable-page manager; one open write transaction per request.

    A SegmentWrite/Patch must come from a model adapter with complete upstream
    inputs and invalidation closure. This manager does not infer model validity.
    Initial MHA/GQA backend has no quantized/MLA mutation support: callers must
    not represent shared latent state as independent query-head pages.
    """

    def __init__(self, pool: HeadPagePool):
        self.pool = pool
        self._requests: dict[str, RequestVersion] = {}
        self._cache: dict[str, RequestVersion] = {}
        self._pending: dict[str, WriteTransaction] = {}
        self._lock = threading.RLock()

    def create_request(self, request_id, *, context_id, contract="mha-v1", namespace="default"):
        with self._lock:
            if request_id in self._requests:
                raise ValueError("request already exists")
            if not all(isinstance(x, str) and x for x in (request_id, context_id, contract, namespace)):
                raise ValueError("nonempty request/context/contract/namespace required")
            ver = RequestVersion(request_id, uuid.uuid4().hex, 0, context_id, contract,
                                 namespace, MappingProxyType({}))
            self._requests[request_id] = ver
            return ver

    def version(self, request_id):
        with self._lock:
            if self.pool.poisoned:
                raise RuntimeError("pool quarantined")
            return self._requests[request_id]

    def metadata_snapshots(self, source_ids, *, source_is_cache=False):
        """Capture immutable committed roots without a KV consumer lease.

        Only CPU metadata may be inspected through these snapshots: they do
        not pin pages, authorize tensor reads, or preserve source residency.
        An alias publisher must pass every captured root as expected_sources
        to share_segments_many, which revalidates identities and retains the
        destination's references atomically under this manager's lock. Source
        removal/rewrite between capture and publication must fail, not reuse
        a stale physical slot. Actual KV consumers must continue to bind().

        Pending writes do not replace their committed root. Resolve all IDs
        in one lock scope, including duplicate source occurrences. Materialize
        and validate the iterable before locking so its callbacks cannot
        mutate a partly captured collection through the reentrant lock.
        """
        if type(source_is_cache) is not bool:
            raise ValueError("source_is_cache must be a bool")
        source_ids = tuple(source_ids)
        if any(type(source_id) is not str or not source_id for source_id in source_ids):
            raise ValueError("metadata source IDs must be nonempty built-in strings")
        with self._lock:
            if self.pool.poisoned:
                raise RuntimeError("pool quarantined")
            roots = self._cache if source_is_cache else self._requests
            return MappingProxyType({source_id: roots[source_id] for source_id in source_ids})

    def bind(self, request_id):
        with self._lock:
            if self.pool.poisoned:
                raise RuntimeError("pool quarantined")
            return ReadLease(self.pool, self._requests[request_id])

    def fork(self, source_id, target_id, *, context_id=None):
        with self._lock:
            if target_id in self._requests:
                raise ValueError("target request exists")
            old = self._requests[source_id]
            if context_id is not None and context_id != old.context_id:
                raise ValueError("fork preserves context; use certified segment reuse for a new context")
            new = replace(old, request_id=target_id, generation=uuid.uuid4().hex, epoch=0)
            self.pool.retain(_refs(old.segments))
            self._requests[target_id] = new
            return new

    def cache(self, request_id, cache_key):
        with self._lock:
            ver = self._requests[request_id]
            self.pool.retain(_refs(ver.segments))
            old = self._cache.get(cache_key)
            self._cache[cache_key] = ver
            if old:
                self.pool.release(_refs(old.segments))

    def evict(self, cache_key):
        with self._lock:
            ver = self._cache.pop(cache_key)
            self.pool.release(_refs(ver.segments))

    def release_request(self, request_id):
        with self._lock:
            ver = self._requests.pop(request_id)
            tx = self._pending.get(request_id)
            if tx:
                tx.abort()
            self.pool.release(_refs(ver.segments))

    def _validate_write(self, write):
        l, h, occurrence = write.key
        if not isinstance(l, int) or not isinstance(h, int) or l < 0 or h < 0 or not isinstance(occurrence, str):
            raise ValueError("invalid segment key")
        if not write.provenance:
            raise ValueError("provenance required")
        if write.k.ndim != 2 or write.k.shape != write.v.shape or write.k.shape[1] != self.pool.head_dim:
            raise ValueError("invalid K/V write shape")
        if write.k.device != write.v.device:
            raise ValueError("K/V device mismatch")

    def begin_update(self, request_id, *, writes=(), patches=(), remove=(), expected_epoch=None,
                     batched_writes=False):
        """Validate and reserve the entire write set before issuing any copies.

        ``batched_writes`` opts full-segment payloads into a two-copy slab
        writer. The default per-page path, patch semantics, atomic publication
        and completion-event retirement are unchanged.
        """
        with self._lock:
            if type(batched_writes) is not bool:
                raise ValueError("batched_writes must be a bool")
            old = self._requests[request_id]
            if request_id in self._pending:
                raise RuntimeError("request already has a pending writer")
            if expected_epoch is not None and expected_epoch != old.epoch:
                raise RuntimeError("stale table epoch")
            keys = [w.key for w in writes] + [p.key for p in patches] + list(remove)
            if len(keys) != len(set(keys)):
                raise ValueError("each segment may appear once in a transaction")
            total = 0
            patch_groups = []
            # Positions are often identical immutable tuples across hundreds
            # of heads. Memoize only successfully checked built-in tuples of
            # built-in ints, by identity (never an O(tokens) value hash). Keep
            # the object alive to prevent id reuse; mutable/custom sequences
            # and stateful int/tuple subclasses retain full validation.
            validated_positions = {}
            for w in writes:
                self._validate_write(w)
                positions = w.positions
                # Payload length remains a per-write check, even on cache hit.
                if len(positions) != len(w.k):
                    raise ValueError("positions must match payload, with no duplicates")
                if validated_positions.get(id(positions)) is not positions:
                    if len(set(positions)) != len(positions):
                        raise ValueError("positions must match payload, with no duplicates")
                    cacheable = type(positions) is tuple
                    for position in positions:
                        if not isinstance(position, int) or position < 0:
                            raise ValueError("invalid logical position")
                        if type(position) is not int:
                            cacheable = False
                    if cacheable:
                        validated_positions[id(positions)] = positions
                if w.reuse_kind not in ("exact_context", "certified_transform", "policy_approximate"):
                    raise ValueError("unknown reuse kind")
                if not isinstance(w.validity_certificate, str) or (w.reuse_kind != "exact_context" and not w.validity_certificate):
                    raise ValueError("non-exact writes require an adapter validity certificate")
                total += math.ceil(len(w.k) / self.pool.page_size)
            for p in patches:
                self._validate_write(p)
                s = old.segments[p.key]
                if len(p.indices) != len(p.k) or len(set(p.indices)) != len(p.indices):
                    raise ValueError("invalid patch indices")
                if any(not isinstance(i, int) or i < 0 or i >= s.length for i in p.indices):
                    raise ValueError("patch outside segment")
                groups = {}
                for row, index in enumerate(p.indices):
                    groups.setdefault(index // self.pool.page_size, []).append((row, index % self.pool.page_size))
                total += len(groups)
                patch_groups.append(groups)
            for key in remove:
                if key not in old.segments:
                    raise KeyError(key)
            reserved = self.pool.reserve(total)
            lease = None
            segments = dict(old.segments)
            cursor = iter(reserved)
            try:
                lease = ReadLease(self.pool, old)
                for w in writes:
                    if batched_writes:
                        pages = tuple(next(cursor) for _ in range(math.ceil(len(w.k) / self.pool.page_size)))
                        self.pool.write_segment_pages(pages, w.k, w.v)
                    else:
                        pages = []
                        for i in range(0, len(w.k), self.pool.page_size):
                            ref = next(cursor)
                            self.pool.write_page(ref, w.k[i:i+self.pool.page_size], w.v[i:i+self.pool.page_size])
                            pages.append(ref)
                    segments[w.key] = Segment(w.key, tuple(pages), tuple(w.positions), w.provenance,
                                              w.position_basis, w.reuse_kind, w.validity_certificate)
                batch_plans = []
                patch_payloads = [(p.k, p.v) for p in patches]
                for payload_i, (p, groups) in enumerate(zip(patches, patch_groups)):
                    s = old.segments[p.key]
                    pages = list(s.pages)
                    for page_i, rows in groups.items():
                        ref = next(cursor)
                        extent = min(self.pool.page_size, s.length - page_i*self.pool.page_size)
                        batch_plans.append((s.pages[page_i], ref, extent,
                                            tuple(i for _, i in rows), payload_i,
                                            tuple(r for r, _ in rows)))
                        pages[page_i] = ref
                    segments[p.key] = replace(s, pages=tuple(pages), provenance=p.provenance)
                if len(batch_plans) == 1:
                    source, dest, extent, indices, payload_i, rows = batch_plans[0]
                    k, v = patch_payloads[payload_i]
                    idx = torch.tensor(rows, device=k.device, dtype=torch.long)
                    self.pool.clone_patch(source, dest, extent, indices,
                                          k.index_select(0, idx), v.index_select(0, idx))
                elif batch_plans:
                    self.pool.clone_patch_batch(batch_plans, patch_payloads)
                for key in remove:
                    del segments[key]
                self.pool.seal(reserved)
                event = completion_event(self.pool.device)
            except BaseException:
                try:
                    event = completion_event(self.pool.device)
                    self.pool.release(reserved, event)
                    if lease is not None:
                        lease.complete(event)
                except BaseException:
                    self.pool.poisoned = True
                raise
            tx = WriteTransaction(self, old, segments, reserved, lease, event)
            self._pending[request_id] = tx
            return tx

    def update(self, request_id, **kwargs):
        return self.begin_update(request_id, **kwargs).commit(wait=True)

    def share_segment(self, source_id, target_id, source_key, target_key, *, proof,
                      positions=None, source_is_cache=False):
        return self.share_segments(source_id, target_id,
                                   [(source_key, target_key, proof, positions)],
                                   source_is_cache=source_is_cache)

    @staticmethod
    def _prepare_shared_segment(source, target, source_key, target_key, proof, positions):
        """Validate one alias without changing any root, ownership or pins."""
        s = source.segments[source_key]
        pos = s.positions if positions is None else tuple(positions)
        if proof.source_provenance != s.provenance or proof.target_context != target.context_id:
            raise ValueError("reuse proof does not bind source and target")
        if proof.kind == "exact_context":
            if source.context_id != target.context_id or pos != s.positions:
                raise ValueError("exact reuse requires identical context and positions")
        elif proof.kind in ("certified_transform", "policy_approximate"):
            if not isinstance(proof.certificate, str) or not proof.certificate:
                raise ValueError("explicit adapter certificate/policy required")
        else:
            raise ValueError("unknown reuse proof kind")
        # Published source positions were validated at their creation. The
        # canonical no-transform case can reuse that immutable tuple without
        # repeatedly constructing a token-length set for every KV head.
        if positions is not None:
            if len(pos) != s.length or len(set(pos)) != len(pos) or any(not isinstance(p, int) or p < 0 for p in pos):
                raise ValueError("invalid target positions")
        if pos != s.positions and s.position_basis != "none":
            raise ValueError("position-encoded K requires materialized adapter transform before reuse")
        if (not isinstance(target_key, tuple) or len(target_key) != 3
                or target_key[:2] != source_key[:2] or not isinstance(target_key[2], str)):
            raise ValueError("cannot alias a different layer/KV group")
        # Preserve approximate/transformed lineage, including when an exact
        # context alias is requested downstream of an approximate source.
        if proof.kind == "exact_context":
            kind, certificate = s.reuse_kind, s.validity_certificate
        elif s.reuse_kind == "exact_context":
            kind, certificate = proof.kind, proof.certificate
        else:
            kind = ("policy_approximate" if "policy_approximate" in (s.reuse_kind, proof.kind)
                    else "certified_transform")
            lineage = [s.reuse_kind, s.provenance, s.validity_certificate,
                       proof.kind, proof.certificate, target.context_id, pos]
            certificate = "lineage-v1:" + hashlib.sha256(
                json.dumps(lineage, separators=(",", ":")).encode()).hexdigest()
        return replace(s, key=target_key, positions=pos, reuse_kind=kind,
                       validity_certificate=certificate)

    def share_segments(self, source_id, target_id, shares, *, source_is_cache=False):
        """Publish many certified aliases atomically, with one root epoch.

        ``shares`` contains ``(source_key, target_key, proof)`` triples, or
        quadruples with an optional explicit positions sequence. All aliases
        resolve against one captured source root; chained references to an
        alias created earlier in this batch are not supported. Destination
        keys must be unique. Empty batches return the current root unchanged.

        Namespace, model contract, position encoding and provenance rules are
        identical to share_segment. Validation of the *entire* batch precedes
        page ownership changes. Only references added/removed by the new root
        change owners; repeated occurrences in a root still own a page once.
        Read leases pin prior roots independently of this atomic publication.
        """
        return self.share_segments_many(
            target_id, [(source_id, shares, bool(source_is_cache))])

    def share_segments_many(self, target_id, sources, *, expected_epoch=None,
                            expected_sources=None):
        """Publish aliases from multiple source roots in one atomic update.

        Each item of ``sources`` is ``(source_id, shares)`` or
        ``(source_id, shares, source_is_cache)``. ``shares`` has the same
        triples/quadruples accepted by :meth:`share_segments`. Source roots
        are captured once under the manager lock, including repeated sources
        and the target itself. Aliases never resolve against another alias
        created within this batch. Destinations must be unique across *all*
        sources; replacing an existing target segment is supported.

        ``expected_sources``, when supplied, maps every selected
        ``(source_id, source_is_cache)`` to the exact RequestVersion captured
        by the caller. This binds adapter proofs to pinned read snapshots;
        an intervening source update/eviction is rejected, not silently
        rebound. Pending source writes do not affect their committed roots.

        All source contracts, provenance, positions and proofs are validated
        before any ownership changes. No KV pages are allocated or copied.
        One old/new reference scan, one differential retain/release, and one
        target publication cover the entire batch. Prior readers retain
        their own pins and source eviction cannot race with publication.
        """
        with self._lock:
            if self.pool.poisoned:
                raise RuntimeError("pool quarantined")
            target = self._requests[target_id]
            if target_id in self._pending:
                raise RuntimeError("target has a pending writer")
            if expected_epoch is not None and expected_epoch != target.epoch:
                raise RuntimeError("stale table epoch")
            prepared, destinations = [], set()
            captured = {}
            for batch in sources:
                if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
                    raise ValueError("a source batch must be (source_id, shares[, source_is_cache])")
                source_id, shares = batch[:2]
                source_is_cache = batch[2] if len(batch) == 3 else False
                if type(source_is_cache) is not bool:
                    raise ValueError("source_is_cache must be a bool")
                identity = source_id, source_is_cache
                if identity not in captured:
                    source = (self._cache if source_is_cache else self._requests)[source_id]
                    if expected_sources is not None and expected_sources.get(identity) is not source:
                        raise RuntimeError("stale source generation/version")
                    if (source.contract, source.namespace) != (target.contract, target.namespace):
                        raise ValueError("namespace/model contract mismatch")
                    captured[identity] = source
                source = captured[identity]
                for item in shares:
                    if not isinstance(item, (tuple, list)) or len(item) not in (3, 4):
                        raise ValueError("a share must be (source_key, target_key, proof[, positions])")
                    source_key, target_key, proof = item[:3]
                    positions = item[3] if len(item) == 4 else None
                    segment = self._prepare_shared_segment(source, target, source_key, target_key, proof, positions)
                    if target_key in destinations:
                        raise ValueError("duplicate target segment in share batch")
                    destinations.add(target_key)
                    prepared.append(segment)
            # Iterables may execute caller code while being consumed. The
            # RLock excludes other threads, not re-entrant manager operations
            # from that code; never overwrite a root changed in that window.
            if self._requests.get(target_id) is not target or target_id in self._pending:
                raise RuntimeError("stale target generation/version")
            for (source_id, source_is_cache), source in captured.items():
                if (self._cache if source_is_cache else self._requests).get(source_id) is not source:
                    raise RuntimeError("stale source generation/version")
            if not prepared:
                return target
            segs = dict(target.segments)
            segs.update((s.key, s) for s in prepared)
            new = replace(target, epoch=target.epoch+1, segments=MappingProxyType(segs))
            old_refs, new_refs = set(_refs(target.segments)), set(_refs(segs))
            added, removed = tuple(new_refs-old_refs), tuple(old_refs-new_refs)
            # retain() validates every generation before incrementing any
            # owner. If it fails, the target root remains exactly unchanged.
            self.pool.retain(added)
            try:
                self._requests[target_id] = new
            except BaseException:
                self.pool.release(added)
                raise
            try:
                self.pool.release(removed)
            except BaseException:
                # Like WriteTransaction.commit, a published root must not be
                # rolled back after a lifetime failure. Quarantine instead.
                self.pool.poisoned = True
                raise
            return new

    def truncate_segment(self, request_id, key, length):
        with self._lock:
            ver = self._requests[request_id]
            if request_id in self._pending:
                raise RuntimeError("request has pending writer")
            s = ver.segments[key]
            if not 0 <= length <= s.length:
                raise ValueError("invalid truncate length")
            segs = dict(ver.segments)
            segs[key] = replace(s, pages=s.pages[:math.ceil(length/self.pool.page_size)], positions=s.positions[:length])
            self.pool.retain(_refs(segs))
            new = replace(ver, epoch=ver.epoch+1, segments=MappingProxyType(segs))
            self._requests[request_id] = new
            self.pool.release(_refs(ver.segments))
            return new

    def append(self, request_id, key, k, v, positions, *, provenance):
        """Append with page reuse; sealed partial tails are always detached."""
        with self._lock:
            # Preserve the scalar API's requirement that its segment exists.
            s = self._requests[request_id].segments[key]
            return self.append_many(request_id, appends=(SegmentWrite(
                key, k, v, tuple(positions), provenance, s.position_basis,
                s.reuse_kind, s.validity_certificate),))

    def append_many(self, request_id, *, appends: Sequence[SegmentWrite],
                    expected_epoch=None):
        """Append all supplied heads with one reservation and root publication.

        Missing segment keys create their initial pages. Existing segments keep
        their complete historical pages, validity lineage and position basis;
        only a partial tail is copied alongside the current input chunk. The
        caller must already have transformed K into the declared position basis.

        Validation and admission cover the complete write set before any page
        is written. This is one request transaction, not a multi-request/model
        step transaction. Empty appends to existing segments are no-ops.
        """
        with self._lock:
            old = self._requests[request_id]
            if request_id in self._pending:
                raise RuntimeError("request has pending writer")
            if expected_epoch is not None and expected_epoch != old.epoch:
                raise RuntimeError("stale table epoch")
            appends = tuple(appends)
            keys = [w.key for w in appends]
            if len(keys) != len(set(keys)):
                raise ValueError("each segment may appear once in an append transaction")
            plans, total = [], 0
            for w in appends:
                self._validate_write(w)
                positions = tuple(w.positions)
                if len(positions) != len(w.k):
                    raise ValueError("positions must match append payload")
                if any(not isinstance(p, int) or isinstance(p, bool) or p < 0
                       for p in positions):
                    raise ValueError("invalid append positions")
                if len(set(positions)) != len(positions):
                    raise ValueError("duplicate append positions")
                s = old.segments.get(w.key)
                if s is None:
                    if w.reuse_kind not in ("exact_context", "certified_transform", "policy_approximate"):
                        raise ValueError("unknown reuse kind")
                    if not isinstance(w.validity_certificate, str) or (
                            w.reuse_kind != "exact_context" and not w.validity_certificate):
                        raise ValueError("non-exact writes require an adapter validity certificate")
                    full, tail = 0, 0
                else:
                    if w.position_basis != s.position_basis:
                        raise ValueError("append position basis differs from existing segment")
                    if set(s.positions).intersection(positions):
                        raise ValueError("duplicate append positions")
                    if not len(w.k):
                        continue
                    positions = s.positions + positions
                    full, tail = divmod(s.length, self.pool.page_size)
                count = math.ceil((tail + len(w.k)) / self.pool.page_size)
                plans.append((w, s, positions, full, tail, count))
                total += count
            if not plans:
                return old

            # Allocate host bookkeeping before reserving physical pages.
            segments = dict(old.segments)
            reserved = self.pool.reserve(total)
            lease = None
            tx = None
            try:
                lease = ReadLease(self.pool, old)
                cursor = 0
                for w, s, positions, full, tail, count in plans:
                    pages = reserved[cursor:cursor + count]
                    cursor += count
                    if tail:
                        oldk, oldv = self.pool.read_page(s.pages[-1], tail)
                        wk = torch.cat((oldk, w.k.to(self.pool.device, self.pool.dtype)))
                        wv = torch.cat((oldv, w.v.to(self.pool.device, self.pool.dtype)))
                        self.pool.copied_bytes += 2 * tail * self.pool.head_dim * self.pool.k.element_size()
                    else:
                        wk, wv = w.k, w.v
                    for i, ref in enumerate(pages):
                        start, end = i * self.pool.page_size, (i + 1) * self.pool.page_size
                        self.pool.write_page(ref, wk[start:end], wv[start:end])
                    if s is None:
                        segments[w.key] = Segment(w.key, pages, positions, w.provenance,
                                                  w.position_basis, w.reuse_kind, w.validity_certificate)
                    else:
                        # Appending exact new rows must not erase approximate
                        # or transformed lineage already present in the history.
                        segments[w.key] = replace(s, pages=s.pages[:full] + pages,
                                                  positions=positions, provenance=w.provenance)
                self.pool.seal(reserved)
                event = completion_event(self.pool.device)
                tx = WriteTransaction(self, old, segments, reserved, lease, event)
                self._pending[request_id] = tx
            except BaseException:
                try:
                    event = completion_event(self.pool.device)
                    self.pool.release(reserved, event)
                    if lease is not None:
                        lease.complete(event)
                    if tx is not None and self._pending.get(request_id) is tx:
                        self._pending.pop(request_id)
                except BaseException:
                    self.pool.poisoned = True
                raise
            return tx.commit(wait=True)

    def stats(self):
        with self._lock:
            return dict(self.pool.stats(), requests=len(self._requests), cached_roots=len(self._cache),
                        pending_transactions=len(self._pending))
