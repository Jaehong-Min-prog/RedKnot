# Head-paged KV management and SegPagedAttention

This package provides bounded page ownership, head-level copy-on-write (COW),
and direct paged attention for MHA/GQA models. Physical pages belong to KV heads
or GQA groups. Query heads that share a KV group do not receive duplicate pages.

Three integrations have different scopes:

| Integration | Entry point | Current scope |
| --- | --- | --- |
| Attention adapter | `SharedKVBackend`, `ManagedSegPagedKVCache` | Caller-managed handles and eager attention calls |
| Native request lifecycle | `--redknot-managed-kv` | Opt-in single-GPU Qwen3 live serving |
| Non-prefix experiment | `NonPrefixPagedEngine`, `run_queued` | Standalone Qwen3 document reuse and admission scheduling |

The non-prefix experimental scheduler and cross-process snapshot protocol are
separate from the native request lifecycle. Enabling the native flag does not
enable non-prefix import or distributed serving.

## Ownership, publication, and retirement

A request owns an immutable segment-table version. Each segment is addressed by
`(layer, kv_head, occurrence)` and records physical page references, logical token
positions, position basis, and adapter-supplied provenance.

- Forks and repeated segment occurrences can share sealed pages. Each request
  root owns a physical page once even when it contains several references.
- Writes and repairs validate and reserve the complete destination set before
  copying. Repairs detach only touched pages; a full-page rewrite copies no old
  rows. Append retains complete historical pages and detaches a partial tail.
- A write transaction publishes after its device completion event, provided the
  request generation and table epoch still match. A failed reservation leaves
  the committed root unchanged.
- Read leases pin a table version through the consumer's completion event.
  Owner references, reader pins, and outstanding device events are tracked
  separately. An event failure quarantines the pool.
- Retirement visits event-bearing pages and reclaimable orphans rather than
  scanning all live immutable pages. Event queries are deduplicated within a
  collection pass. `collection_stats()` reports this host-side work separately.

The pool uses fixed K/V slabs of shape
`[capacity_pages, page_size, head_dim]`. `allocated_bytes` reports capacity
occupied by active pages; `reserved_slab_bytes` reports the backing allocation.
Sharing can admit more requests into that slab without reducing the slab's
already allocated CUDA memory.

```python
import torch
from sglang.srt.mem_cache.head_kv import (
    HeadKVManager, HeadPagePool, SegmentPatch, SegmentWrite,
)

pool = HeadPagePool(128, 16, 128, dtype=torch.bfloat16, device="cuda")
manager = HeadKVManager(pool)
manager.create_request(
    "parent", context_id="context-hash",
    contract="weights-adapters-kv-layout-position-v1", namespace="tenant-a",
)
key = (0, 0, "document-0")
k = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
manager.update("parent", writes=[
    SegmentWrite(key, k, v, tuple(range(32)), "adapter-state-hash"),
])
manager.fork("parent", "child")
manager.update("child", patches=[
    SegmentPatch(key, (19,), k[:1], v[:1], "repaired-state-hash"),
])
manager.release_request("child")
manager.release_request("parent")
pool.collect()  # In-flight events can defer physical reclamation.
```

The example's contract, context, and provenance strings are placeholders. The
model adapter supplies their actual identities and validates any repair inputs.

## Bulk document aliases and writes

`share_segments_many(target_id, sources, expected_epoch=...,
expected_sources=...)` publishes aliases from several committed source roots
with one target-table transition. Each source entry is
`(source_id, shares[, source_is_cache])`; each share is
`(source_key, target_key, ReuseProof[, positions])`.

All source contracts, namespaces, positions, proofs, and destination keys are
validated before ownership changes. Exact root identities in
`expected_sources[(source_id, source_is_cache)]` prevent an intervening rewrite
or eviction from silently rebinding the alias. Destination publication retains
added pages and releases removed pages without copying KV payloads.

`metadata_snapshots(source_ids, source_is_cache=False)` captures immutable roots
for CPU metadata inspection. These snapshots do not pin pages or authorize GPU
reads. A metadata-only alias publisher must pass the captured roots back through
`expected_sources`; tensor consumers use `bind()` read leases.

`update(..., batched_writes=True)` batches full-segment payload writes. Contiguous
destination slots use slab slices; fragmented slots use indexed copies. Patch
semantics, immutable publication, and event retirement remain the same.
`append_many` appends one request's heads in one transaction.

Atomicity covers a manager transaction, not an entire model forward or batch.
After a later layer or request fails, the caller must discard affected partial
requests before retrying.

## Attention and adapter contracts

`ReadLease.descriptor(layer, num_kv_heads)` produces page slots, extents, and
logical positions. `paged_attention` reads the slabs directly with GQA mapping,
causal masking, per-head windows/sinks, and online softmax across pages.
`gather()` is an explicit reference/export helper.

The Triton backend accumulates QK, softmax, and PV in FP32. It skips invisible
payload blocks while preserving the selected attention policy. Split counts
bound long page walks and partial-output workspace. The
`skip_invisible=False` and `num_splits=...` options support controlled ablations.
Public descriptor validation can synchronize CUDA; this implementation targets
decode and small query batches rather than high-throughput full prefill.

The registered `redknot` and `segpaged` backends accept a caller-owned
`runner.redknot_shared_kv_manager` set before backend construction. Their shared
adapter uses generation-fenced request handles and receives one handle per
request in `ForwardBatch.redknot_shared_kv_handles`. Positions and prefix lengths
must describe each request's actual causal history. Every layer uses the same
request handle, and every completion, cancellation, or failure releases it.

That low-level adapter alone does not replace the scheduler's original KV
allocation. Integrations must coordinate memory allocation and request lifetime
explicitly. It rejects unsupported graph, speculative, quantized-KV,
cross-attention, multiaxis-position, and incompatible-head-dimension paths.

## Opt-in native request lifecycle

`production.py` connects managed handles to real SGLang `Req` objects and
`ForwardBatch` construction. Completion, abort, and retraction release request
roots before logical request/token slots are recycled. Flush verifies that no
managed requests remain. Weight updates require a serving restart.

This integration allocates one physical head-page pool and a logical token
accounting facade instead of an additional ordinary K/V slab. Its budget reserves
request-tail fragmentation and COW scratch space. Logical admission charges each
request's whole sequence; it does not assume sharing when admitting requests.

The supported configuration is unquantized, full-layer Qwen3 MHA/GQA on one CUDA
GPU with FP16/BF16 KV, `redknot` attention, FCFS, `page_size=1`, and an explicit
positive `redknot_page_size`. Radix caching, overlap scheduling, and both CUDA
graph modes must be disabled. TP/PP/DP, non-prefix import, LoRA, speculative
decoding, and hybrid/MLA models are rejected. The eager step boundary synchronizes
the current CUDA stream before collecting retired pages and admitting more work.

The native smoke runner constructs these options and checks launched-server
lifecycle behavior. Run it in a compatible SGLang CUDA environment:

```bash
python test/srt/redknot/run_managed_kv_native_smoke.py \
  --model-path /path/to/local/Qwen3 \
  --output /path/to/new-smoke-results
```

This is a validation command, not a throughput or task-quality claim.

## Non-prefix reuse and admission experiments

`layers/attention/redknot/nonprefix_paged.py` implements a separate single-GPU
Hugging Face Qwen3 engine. Documents are computed independently at position zero.
Canonical local-head KV pages are immutable and shared across occurrences;
context-dependent global-head history is recovered privately online.
Occurrence-specific query counterrotation aligns canonical RoPE coordinates
without rewriting the shared keys. Position alignment does not establish that
cross-context reuse is exact.

The engine preserves the head policy during attention, builds page-aligned
historical partitions, copies leading partial-page boundaries when needed by
FlashAttention, and merges partition outputs using their log-sum-exp statistics.
Each request retains a historical read lease and a private append arena for its
query and generated suffix. Optional selective recovery is approximate and has
separate configuration and quality checks.

`nonprefix_scheduler.py` implements a single-owner queued service loop with
bounded active requests, timed admission/recovery, query-length buckets, batched
decode, cancellation, and cleanup. `nonprefix_affinity.py` offers admission
ordering based on visible canonical page overlap, with lookahead, age, and bypass
limits. Canonical documents remain resident; overlap measures locality, not
saved allocations or an eviction hit rate. Recovery/admission is serialized.
These modules do not route requests between nodes or install affinity into the
native SGLang scheduler.

Use the benchmark entry points to prepare frozen requests and compare service
arms with the same tokens and declared memory budget:

```bash
python test/srt/redknot/benchmark_redknot_nonprefix.py --help
python test/srt/redknot/benchmark_nonprefix_affinity.py --help
python test/srt/redknot/benchmark_cow_sharing.py --help
```

## Cross-process snapshots

`distributed.py` provides an HTTP object owner and durable SQLite authority;
`transfer.py` exports and imports real page payloads. Exports pin snapshots and
publish content-addressed objects through durable PREPARED/COMMITTED decisions.
Imports validate compatibility, layout, extents, and payload digests, reserve all
destination pages, and publish only after copies complete. Retention grants and
active transfers prevent premature owner-side collection.

Migration decisions include epoch fencing and output-frontier tracking.
Readiness is an adapter attestation; an integration must coordinate all required
GPU ranks before acknowledging it. This is a bounded transfer protocol with one
durable owner, not a completed TP/CP/PP serving path, replicated authority, or
GPU-direct/RDMA implementation. See module API docstrings for protocol budgets
and listener authentication requirements.

## Validation and measurement

From the repository root, use the project's matching PyTorch and pytest
environment. The following suites cover ownership, retirement, aliases, attention,
and integration contracts; CUDA-marked cases need a compatible GPU stack:

```bash
python -m pytest -q \
  test/srt/redknot/test_head_kv_manager.py \
  test/srt/redknot/test_head_kv_retirement.py \
  test/srt/redknot/test_head_kv_attention.py \
  test/srt/redknot/test_metadata_alias_snapshots.py \
  test/srt/redknot/test_nonprefix_multidoc_publish.py \
  test/srt/redknot/test_nonprefix_scheduler.py \
  test/srt/redknot/test_nonprefix_affinity.py \
  test/srt/redknot/test_managed_kv_production.py
```

Report full recomputation, private-copy reuse, and shared-page reuse as distinct
arms. Count offline document creation separately from hot online service, and
include admission, recovery, publication, first-token computation, and release
where the metric requires them. Report active page occupancy separately from
reserved slab memory and total GPU memory.

In `run_redknot_offlinekv`, emitted TTFT ends after the Phase E full-attention
seed actually produces the first output token. Phase D head-class query logits
remain diagnostic and can differ from that seed; their readiness time is
recorded separately in `timing_stats`. The optional managed query bridge changes
only Phase D, so output-text agreement alone does not validate it.
`benchmark_segpaged_workflow.py` reports a serial request rate, not concurrent
serving QPS. Concurrent benchmarks report both arrival-inclusive TTFT and service
TTFT with their declared admission boundary.

Low-precision backends can round differently. The non-prefix FlashAttention
partition merge currently uses CUDA FP32 `index_add_` reductions, whose summation
order can vary; bit-identical repeated outputs are not guaranteed. Record task
quality, generated texts, and numerical comparisons rather than inferring one
from another. Repeated prefixes/arrivals around the same question are not
independent QA samples.

Sharing, COW, provenance, and validity certificates preserve storage and declared
lineage. They do not prove task accuracy or make arbitrary prompts equivalent.
The model adapter remains responsible for context compatibility, position
transforms, head classification, and recovery quality. MLA latents and recurrent
states require representation-specific adapters.
