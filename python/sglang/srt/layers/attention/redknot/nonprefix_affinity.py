"""Bounded, advisory single-node affinity for canonical per-head KV pages.

This module selects admission order only. It neither grants a reuse proof nor
changes page ownership, RoPE positions, the head policy, or a request's tokens.
Canonical sources are permanently resident in the current experimental engine;
overlap is a locality hint, NOT additional saved allocations or a cache hit rate.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Integral
from types import MappingProxyType


@dataclass(frozen=True)
class ShareAffinityConfig:
    """Fairness limits bound further bypass, not end-to-end queue latency.

    Aged requests can still wait for in-flight recovery, decoding, or capacity.
    """
    lookahead: int = 32
    max_bypass: int = 8
    max_wait_seconds: float = 1.0
    recent_requests: int = 8

    def __post_init__(self):
        for name, minimum in (("lookahead", 1), ("max_bypass", 0), ("recent_requests", 0)):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))
        wait = float(self.max_wait_seconds)
        if isinstance(self.max_wait_seconds, bool) or not math.isfinite(wait) or wait < 0:
            raise ValueError("max_wait_seconds must be finite and nonnegative")
        object.__setattr__(self, "max_wait_seconds", wait)

    def to_dict(self):
        return asdict(self)


def _profile(engine):
    # Host metadata only: never stringify, hash, or read a GPU tensor.
    policies = tuple(tuple((p.layer, p.kv_head, p.head_type, p.window, p.sink_size)
                           for p in row) for row in engine.policies)
    return (str(engine.model_revision), policies)


class CanonicalAffinityIndex:
    """Immutable index of exact source pages, scoped to one engine and manager.

Each identity includes the engine namespace, manager instance, effective head
profile, root contract, canonical segment provenance/position basis and the
physical page's slot/generation/content ID. Same text or doc_id is insufficient.
Logical placement offsets deliberately do not change an unrotated source page.
The index owns no pages; source roots must remain registered and unchanged.
"""
    def __init__(self, engine):
        required = ("namespace", "manager", "pool", "documents", "policies", "model_revision")
        if any(not hasattr(engine, name) for name in required):
            raise TypeError("share affinity requires a canonical engine index or an explicit affinity_index")
        self.engine, self.manager, self.pool = engine, engine.manager, engine.pool
        self.namespace = engine.namespace
        self.profile = _profile(engine)
        self.profile_digest = hashlib.sha256(repr(self.profile).encode()).hexdigest()
        self.page_bytes, self.page_size = int(self.pool.page_bytes), int(self.pool.page_size)
        self._documents, position_checks, all_pages = {}, {}, set()
        for doc_id, document in engine.documents.items():
            version = self.manager.version(document.source_request_id)
            if version.namespace != self.namespace:
                raise ValueError("canonical source namespace differs from engine")
            entries = {}
            for (layer, head), key in document.source_keys.items():
                policy = engine.policies[layer][head]
                segment = version.segments[key]
                if (policy.head_type != "local" or key[:2] != (layer, head)
                        or segment.position_basis != "rope-canonical"):
                    raise ValueError("affinity source must be a canonical local-head segment")
                positions = segment.positions
                checked = position_checks.get(id(positions))
                if checked is None:
                    if (type(positions) is not tuple or len(positions) != document.doc_len
                            or any(type(value) is not int or value != i
                                   for i, value in enumerate(positions))):
                        raise ValueError("canonical source positions must be exactly 0..document_length-1")
                    position_checks[id(positions)] = (positions, document.doc_len)
                elif checked[0] is not positions or checked[1] != document.doc_len:
                    raise ValueError("canonical source position extent mismatch")
                if len(segment.pages) != math.ceil(document.doc_len / self.page_size):
                    raise ValueError("canonical source page extent mismatch")
                identity = ("canonical-source-page-v1", self.namespace, id(self.manager),
                            self.profile_digest, version.contract, layer, head,
                            segment.provenance, segment.position_basis, segment.reuse_kind,
                            segment.validity_certificate, document.doc_len)
                pages = tuple(identity + (p.slot, p.generation, p.content_id) for p in segment.pages)
                entries[layer, head] = pages
                all_pages.update(pages)
            self._documents[doc_id] = (document, version, dict(document.source_keys), entries,
                                       document.doc_len, document.source_request_id)
        self._indexed_page_count = len(all_pages)

    @classmethod
    def from_engine(cls, engine):
        return cls(engine)

    def validate_engine(self, engine):
        if (engine is not self.engine or engine.manager is not self.manager
                or engine.pool is not self.pool or engine.namespace != self.namespace
                or _profile(engine) != self.profile):
            raise ValueError("affinity index engine/namespace/model/head profile changed")

    def request_pages(self, request):
        self.validate_engine(self.engine)
        documents = []
        total = len(request.prefix_ids)
        for doc_id in request.document_ids:
            if doc_id not in self._documents:
                raise ValueError("document was not registered when the affinity index was built")
            document, version, keys, entries, length, source_id = self._documents[doc_id]
            if (self.engine.documents.get(doc_id) is not document
                    or document.source_request_id != source_id or document.doc_len != length
                    or document.source_keys != keys or self.manager.version(source_id) is not version):
                raise ValueError("canonical source changed; rebuild the affinity index")
            documents.append((total, length, entries))
            total += length
        pages = {}
        # Match _build_partitions' fixed historical view, including skipping a
        # leading unaligned row range that the engine copies to a private page.
        # The tail arena and prefix/global heads never count as canonical reuse.
        for layer, row in enumerate(self.engine.policies):
            for head, policy in enumerate(row):
                if policy.head_type != "local":
                    continue
                sink = min(policy.sink_size, len(request.prefix_ids), total)
                tail = max(sink, total - policy.window)
                for offset, length, entries in documents:
                    start, end = max(tail, offset) - offset, min(total, offset + length) - offset
                    if start >= end:
                        continue
                    if start % self.page_size:
                        start = min(end, (start // self.page_size + 1) * self.page_size)
                    if start >= end:
                        continue
                    for page in entries[layer, head][start // self.page_size:math.ceil(end / self.page_size)]:
                        pages[page] = self.page_bytes
        return MappingProxyType(pages)

    def stats(self):
        return {
            "kind": "engine-scoped canonical local-head physical page index",
            "engine_namespace": self.namespace, "model_revision": self.profile[0],
            "head_profile_sha256": self.profile_digest,
            "engine_share_local": bool(getattr(self.engine, "share_local", False)),
            "registered_documents": len(self._documents),
            "indexed_source_pages": self._indexed_page_count,
            "indexed_source_page_bytes": self._indexed_page_count * self.page_bytes,
            "request_footprint": "visible canonical historical partition pages; private boundary/prefix/global/tail excluded",
            "overlap_semantics": "source physical page-byte locality, not saved allocation bytes; private arm is a source-locality control",
            "residency": "canonical document roots already resident; no LRU eviction or cross-node routing",
        }


@dataclass(frozen=True)
class AffinityDecision:
    selected_offset: int
    source_canonical_overlap_bytes: int
    active_canonical_overlap_bytes: int
    active_request_overlap_sum_bytes: int
    recent_overlap_bytes_by_recency: tuple[int, ...]
    reason: str
    protected: bool
    bypass_count: int
    wait_seconds: float

    def to_dict(self):
        return asdict(self)


class ShareAffinitySelector:
    """Pure host scheduling state; max bypass/age overrides any affinity score."""
    def __init__(self, index, requests, config):
        self.config, self.index = config, index
        self.footprints, all_sizes = {}, {}
        for request in requests:
            pages = dict(index.request_pages(request))
            for identity, size in pages.items():
                hash(identity)
                if not isinstance(size, Integral) or isinstance(size, bool) or size <= 0:
                    raise ValueError("affinity page sizes must be positive integer physical bytes")
                if identity in all_sizes and all_sizes[identity] != size:
                    raise ValueError("same affinity page identity has inconsistent physical byte sizes")
                all_sizes[identity] = int(size)
            self.footprints[request.request_id] = {key: int(size) for key, size in pages.items()}
        self.bypasses = {request.request_id: 0 for request in requests}
        self.blocked_epoch = {}
        self.recent = deque(maxlen=config.recent_requests)
        self.counters = dict(selections=0, reordered_admissions=0, age_forced_admissions=0,
                             bypass_forced_admissions=0, capacity_blocks=0,
                             protected_drain_observations=0, maximum_bypass_count=0,
                             source_canonical_overlap_bytes=0, active_canonical_overlap_bytes=0)

    def choose(self, pending, active_ids, *, now, release_epoch):
        arrived = [(i, request) for i, request in enumerate(pending[:self.config.lookahead])
                   if request.arrival_seconds <= now]
        if not arrived:
            return None
        active_pages = set()
        for request_id in active_ids:
            active_pages.update(self.footprints[request_id])
        reference = set(active_pages)
        for request_id in self.recent:
            reference.update(self.footprints[request_id])
        oldest = arrived[0][1]
        age = now - oldest.arrival_seconds
        forced = ("max_bypass" if self.bypasses[oldest.request_id] >= self.config.max_bypass
                  else "max_wait" if age >= self.config.max_wait_seconds else None)
        if forced:
            if self.blocked_epoch.get(oldest.request_id) == release_epoch:
                self.counters["protected_drain_observations"] += 1
                return None
            candidates = [arrived[0]]
        else:
            candidates = [(i, request) for i, request in arrived
                          if self.blocked_epoch.get(request.request_id) != release_epoch]
        if not candidates:
            return None
        active_sets = [self.footprints[request_id] for request_id in active_ids]
        recent_sets = [self.footprints[request_id] for request_id in reversed(self.recent)]
        def overlap(request, pages):
            return sum(size for key, size in self.footprints[request.request_id].items() if key in pages)
        def score(request):
            # Pairwise bytes are a locality tie-break, not unique allocations.
            # A latest-first vector avoids equal-weight recent unions erasing
            # affinity as soon as two document groups have both been observed.
            return (overlap(request, active_pages),
                    sum(overlap(request, pages) for pages in active_sets),
                    tuple(overlap(request, pages) for pages in recent_sets))
        selected, request = max(candidates, key=lambda pair: (score(pair[1]), -pair[0]))
        unique_overlap = overlap(request, reference)
        active_overlap, pairwise_overlap, recent_overlaps = score(request)
        self.counters["selections"] += 1
        return AffinityDecision(selected, unique_overlap, active_overlap,
                                pairwise_overlap, recent_overlaps,
                                forced or ("source_overlap" if unique_overlap else "fcfs_no_overlap"),
                                forced is not None, self.bypasses[request.request_id],
                                max(0., now - request.arrival_seconds))

    def on_capacity_block(self, request_id, release_epoch):
        self.blocked_epoch[request_id] = release_epoch
        self.counters["capacity_blocks"] += 1

    def on_admitted(self, decision, older_arrived):
        for request in older_arrived:
            self.bypasses[request.request_id] += 1
            self.counters["maximum_bypass_count"] = max(
                self.counters["maximum_bypass_count"], self.bypasses[request.request_id])
        self.counters["reordered_admissions"] += int(decision.selected_offset > 0)
        self.counters["age_forced_admissions"] += int(decision.reason == "max_wait")
        self.counters["bypass_forced_admissions"] += int(decision.reason == "max_bypass")
        for key in ("source_canonical_overlap_bytes", "active_canonical_overlap_bytes"):
            self.counters[key] += getattr(decision, key)

    def on_released(self, request_id):
        self.recent.append(request_id)

    def stats(self):
        describe = getattr(self.index, "stats", None)
        return {"config": self.config.to_dict(), **self.counters,
                "bypass_counts": dict(self.bypasses),
                "reference": "active requests plus bounded recent completed requests; unique physical pages",
                "selection_order": ["active unique page bytes", "active pairwise overlap bytes (not unique allocations)",
                                    "latest-first recent overlap vector", "FCFS"],
                "fairness": "max_wait_seconds triggers aging protection, not a queue-latency SLA; max_bypass bounds later successful admissions",
                "score_is_saved_allocation_bytes": False,
                "index": describe() if callable(describe) else {"kind": "caller-supplied advisory index"}}
