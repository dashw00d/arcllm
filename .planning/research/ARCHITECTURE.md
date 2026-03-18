# Architecture Research

**Domain:** Multi-GPU event-based synchronization layer for llama.cpp SYCL backend
**Researched:** 2026-03-17
**Confidence:** HIGH (based on direct source code analysis of the existing fork)

---

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        ggml compute graph dispatch                        │
│  ggml_backend_sycl_graph_compute() → ggml_sycl_op_mul_mat()              │
├──────────────────────────────────────────────────────────────────────────┤
│                     Row-Split Dispatch (ggml-sycl.cpp)                    │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  use_parallel_row_split  (ROW_ALLOW_MMVQ=1 && devices > 1)        │   │
│  │                                                                   │   │
│  │  use_event_sync = (GGML_SYCL_ROW_EVENTS != 0)                     │   │
│  │                                                                   │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐    │   │
│  │  │  Phase 1     │   │  Phase 2     │   │  Phase 3           │    │   │
│  │  │  Submit all  │→  │  Wait all    │→  │  Merge non-main    │    │   │
│  │  │  devices     │   │  completion  │   │  → dst             │    │   │
│  │  └──────────────┘   └──────────────┘   └────────────────────┘    │   │
│  └───────────────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────────────┤
│                   ggml_backend_sycl_context (common.hpp)                  │
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │  In-order queues │  │  OOO queues       │  │  Worker threads  │       │
│  │  qptrs[dev][str] │  │  ooo_qptrs[dev]  │  │  row_workers[dev]│       │
│  │  (legacy path)   │  │  (event path)     │  │  (unused/reverted│       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐                              │
│  │  Device pools    │  │  src1 copy cache  │                              │
│  │  pools[dev]      │  │  row_src1_cache   │                              │
│  │  host_pools[dev] │  │  [dev]            │                              │
│  └──────────────────┘  └──────────────────┘                              │
├──────────────────────────────────────────────────────────────────────────┤
│                     Per-Call Scratch (dev_data[])                         │
│                                                                           │
│  ┌──────────────┐  ┌───────────────────┐  ┌──────────────────────┐      │
│  │ src0_dd      │  │ src1_ddf / ddq    │  │ host_staging_buf     │      │
│  │ (weight rows)│  │ (activation copy) │  │ (pinned, per device) │      │
│  └──────────────┘  └───────────────────┘  └──────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | File | Responsibility | Notes |
|-----------|------|----------------|-------|
| `ggml_backend_sycl_context` | `common.hpp:364` | Per-backend state: queues, pools, cache, workers | One instance per llama.cpp backend context |
| `ooo_qptrs[dev]` | `common.hpp:370` | One OOO queue per device, lazy-initialized via `ooo_stream(dev)` | Required for event-based sync; batched cmdlists deadlock these |
| `qptrs[dev][str]` | `common.hpp:369` | In-order queues (legacy path), used by all non-row-split ops | These are the default ggml compute queues |
| `row_src1_cache[dev]` | `common.hpp:502` | Per-device pointer cache: skip redundant src1 cross-device copies | Consecutive attn_q/k/v matmuls all read the same norm output |
| `dev_data[]` | `ggml-sycl.cpp:3703` | Per-call scratch: weight slices, activation buffers, staging bufs | Stack-allocated per mul_mat call; staging buf is pre-allocated |
| `host_staging_buf` | `ggml-sycl.cpp:3721` | Per-device pinned host buffer for GPU→host→GPU copy staging | Sized to max(src1_bytes, merge_bytes) at call init |
| `dev_events[]` | `ggml-sycl.cpp:4053` | Per-device completion event from Phase 1 barrier | `ext_oneapi_submit_barrier()` = cudaEventRecord analog |
| `dev2dev_memcpy_staged` | `ggml-sycl.cpp:656` | Semi-async GPU→host wait, host→GPU async (legacy path) | GPU→host still blocks host; OOO path replaces this |
| `dev2dev_memcpy_staged_sync` | `ggml-sycl.cpp:677` | Fully blocking copy (merge phase, legacy path) | Both waits; currently Phase 3 still uses this for cross-GPU dst |
| `row_workers[]` | `common.hpp:514` | Persistent worker threads, one per non-main device | Attempted, reverted: caused src queue contention |

---

## Architectural Patterns

### Pattern 1: Pre-Sync Host Wait + Event-Based Phase 1

**What:** One `stream->wait()` on the main device's in-order queue before the row-split loop. All work inside Phase 1 (copy, quantize, kernel) uses OOO queues with `depends_on()` chaining — zero additional host waits. This is the CUDA pattern mapped to SYCL.

**When to use:** Every mul_mat call on the event path. The pre-sync is mandatory because upstream ops (norm, rope, etc.) enqueue on the in-order queue and their output must be stable before src1 is read.

**Why one wait is acceptable:** There are ~448 matmuls per token but the pre-sync latency (~300µs) is amortized once per matmul boundary, not once per device. The old path paid ~9 waits per matmul.

**Example (existing code):**
```cpp
// Pre-sync: drain in-order queue once
ctx.stream()->wait();

// Phase 1: OOO event chain per non-main device
sycl::event e_g2h = src_stream->submit([&](sycl::handler& h) {
    if (has_prev_staging) h.depends_on(e_prev_staging_free);
    h.memcpy(staging, src1_ddf_i_source, copy_bytes);
});
sycl::event e_h2g = stream->submit([&](sycl::handler& h) {
    h.depends_on(e_g2h);
    h.memcpy(src1_ddf_i, staging, copy_bytes);
});
stream->ext_oneapi_submit_barrier({e_h2g});  // gate kernel on copy
// ... launch kernel ...
dev_events[i] = stream->ext_oneapi_submit_barrier();  // record completion
```

**Trade-offs:**
- Requires `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` — batched mode deadlocks because unflushed command lists never signal their events for cross-queue `depends_on()`
- OOO queue `ext_oneapi_submit_barrier()` requires the event target queue to be OOO (not in-order) — mixing is the root cause of the "cross-queue barrier event crash" failure mode

### Pattern 2: Fan-Out Copy + Fan-In Barrier Event Chain

**What:** For N non-main devices, Phase 1 submits an independent copy+quantize+kernel chain to each OOO queue. The chains share one pinned staging buffer per device (no sharing across devices). Each chain ends with a barrier event. Phase 2 then waits on all N barrier events.

**Topology (3-device example):**

```
Main device in-order queue (pre-sync)
         |
         | host wait (1x per matmul)
         v
GPU0 src1 data is stable
         |
         +──────────────────────────────+
         |                              |
         v                              v
GPU1 OOO queue                     GPU2 OOO queue
  e_g2h: src1 GPU0→host (staging1)   e_g2h: src1 GPU0→host (staging2)
  e_h2g: host→GPU1                   e_h2g: host→GPU2
  barrier(e_h2g)                     barrier(e_h2g)
  quantize_q8_1                      quantize_q8_1
  barrier()                          barrier()
  mmvq kernel (rows 0..R/2-1)        mmvq kernel (rows R/2..R-1)
  dev_events[1] = barrier()          dev_events[2] = barrier()
         |                              |
         +──────────────────────────────+
                        |
                Phase 2: dev_events[i].wait() for each i
                        |
                Phase 3: merge GPU1/GPU2 outputs → dst
```

**When to use:** Any N-way tensor split where each device's output is a disjoint row slice that must be gathered into a single destination buffer.

**Trade-offs:** Staging buffer must be per-device (not shared) because GPU→host and host→GPU copies on different device queues can be concurrent. Sharing one staging buffer across devices would corrupt it.

### Pattern 3: Staging Buffer Reuse via Event Chaining (Ring-Free)

**What:** The staging buffer for one `i0` column-batch iteration must not be overwritten until the `e_h2g` from the prior iteration completes. Track this as `e_prev_staging_free`. Each new `e_g2h` submission `depends_on(e_prev_staging_free)`.

**When to use:** Whenever a single staging buffer is reused across a loop (ne13*ne12 iterations). Eliminates the need for a ring buffer of staging slots at the cost of a dependency edge.

**Example (existing code):**
```cpp
sycl::event e_prev_staging_free;
bool has_prev_staging = false;

for (int64_t i0 = 0; ...) {
    sycl::event e_g2h = src_stream->submit([&](sycl::handler& h) {
        if (has_prev_staging) h.depends_on(e_prev_staging_free);
        h.memcpy(staging, src, bytes);
    });
    sycl::event e_h2g = stream->submit([&](sycl::handler& h) {
        h.depends_on(e_g2h);
        h.memcpy(dst, staging, bytes);
    });
    e_prev_staging_free = e_h2g;
    has_prev_staging = true;
}
```

**Trade-offs:** Simple and correct. Adds one dependency edge per iteration. Alternative (ring buffer with N staging slots) would allow more copy overlap but adds allocation complexity; not needed at ne13*ne12 typically == 1 for decode.

### Pattern 4: OOO Queue Flush for Event Pool Exhaustion Prevention

**What:** After Phase 2, call `ooo_stream(i)->wait()` on all active device OOO queues. This releases the Level Zero event handles that back the submitted barriers.

**Why necessary:** L0 has a finite event pool. Each `ext_oneapi_submit_barrier()` consumes an event handle. Without flush, handles accumulate across matmul calls and the pool exhausts after ~100 matmuls, causing a crash.

**Placement:** End of the row-split loop body (after Phase 3), before returning from `mul_mat`. This is one `wait()` per device per matmul — acceptable since it happens after all GPU work is complete.

**Trade-offs:** One host wait per device per matmul replaces 9 host waits per device per matmul — still a massive win. Alternative: event pool size query + conditional flush. Not implemented; simple periodic flush is sufficient.

---

## Data Flow

### Request Flow (per token, decode phase)

```
llama_decode()
    │
    ▼
ggml_backend_sycl_graph_compute()
    │ (for each node in graph, ~900 nodes/token for GLM-4.7)
    ▼
ggml_sycl_op_mul_mat()  [~448 of the 900 nodes are MUL_MAT]
    │
    ▼ (if split && devices > 1 && ROW_ALLOW_MMVQ)
use_parallel_row_split path
    │
    ├── Pre-sync: ctx.stream()->wait()          [1 host wait per matmul]
    │
    ├── Phase 1 (per non-main device):
    │   ├── Check src1 cache (skip copy if same ptr)
    │   ├── OOO submit: GPU0→host staging (e_g2h)
    │   ├── OOO submit: host→GPU_i (e_h2g, depends_on e_g2h)
    │   ├── OOO barrier (gate quantize on e_h2g)
    │   ├── OOO submit: quantize_q8_1 kernel
    │   ├── OOO barrier (gate matmul on quantize)
    │   ├── OOO submit: mmvq/fused-mmq kernel
    │   └── dev_events[i] = OOO barrier()      [completion token]
    │
    ├── Phase 2 (per active device):
    │   └── dev_events[i].wait()               [1 host wait per device]
    │
    ├── Phase 3 (per non-main device):
    │   └── dev2dev_memcpy_staged_sync(dst)    [still blocking — open issue]
    │       (merge device partial rows into final destination)
    │
    └── OOO flush: ooo_stream(i)->wait()       [event pool management]
```

### State Management

```
ggml_backend_sycl_context (lifetime: backend context)
    │
    ├── ooo_qptrs[dev]        initialized once, reused every matmul
    ├── qptrs[dev][str]       initialized once, used by all non-row-split ops
    ├── row_src1_cache[dev]   updated each matmul, invalidated on new graph
    └── pools[dev]            bump allocator for per-call scratch

dev_data[dev] (lifetime: single mul_mat call)
    │
    ├── src0_dd               weight row slice pointer (into split tensor)
    ├── src1_ddf              activation float copy (per-device VRAM)
    ├── src1_ddq              activation Q8_1 copy (per-device VRAM)
    ├── dst_dd                partial result buffer (per-device VRAM)
    └── host_staging_buf      pinned host memory (allocated at call entry,
                              freed by pool at call exit)
```

### Key Data Flows

1. **src0 (weights) flow:** Already resident on each device's VRAM at model load time (split tensor). Zero copies in the hot path. Each device reads only its assigned row slice via `data_device[i]` pointer in `ggml_tensor_extra_gpu`.

2. **src1 (activation) flow — hot path:** Resident on main device (device 0). Must be copied to each non-main device per matmul. Flow: GPU0 VRAM → pinned host staging → GPU_i VRAM. Two PCIe transfers per device per matmul. The src1 cache skips this when consecutive matmuls share the same activation (common for attn_q/k/v).

3. **dst (result) flow:** Each device computes a row slice into its local `dst_dd` buffer. Non-main device results must be copied back to the main device (or host) in Phase 3. Flow: GPU_i VRAM → pinned staging → GPU0 VRAM (or host RAM directly if `dst_is_host`).

---

## Component Boundaries

### What Lives Where

| Concern | Location | Boundary |
|---------|----------|----------|
| Queue lifecycle | `ggml_backend_sycl_context` | Owned by backend context; one OOO queue per device per context |
| Event lifecycle | Local to `mul_mat` call | `dev_events[]` stack-allocated per call; flushed at end of call via OOO wait |
| Staging buffer allocation | `dev_data[]` in `mul_mat` | Pre-allocated at call entry from `ggml_sycl_host_malloc`; freed at exit |
| Staging buffer reuse tracking | Local to Phase 1 per-device loop | `e_prev_staging_free` / `has_prev_staging` scoped to per-device `i` |
| src1 copy cache | `ggml_backend_sycl_context::row_src1_cache` | Persists across matmul calls within one graph execution |
| Kernel dispatch | `ggml_sycl_op_mul_mat_q()` / `mmvq_q4_k_q8_1` | Unchanged — receives queue pointer, unaware of event sync above it |
| Weight row slices | `ggml_tensor_extra_gpu::data_device[]` | Set at model load; read-only in hot path |

### Integration Points with Existing ggml-sycl Dispatch

The event sync layer is **surgically isolated** to the `use_parallel_row_split` code path. The integration seam is:

- **Entry gate:** `use_parallel_row_split && use_event_sync` boolean, checked once per `mul_mat` call
- **Queue selection:** `ctx.ooo_stream(i)` instead of `ctx.stream(i, is)` — same kernel dispatch code receives a different queue pointer
- **No kernel changes:** Kernels (`mmvq`, `fused_mmq`) are queue-agnostic. They receive a `queue_ptr` and submit to it. OOO vs in-order is invisible to the kernel.
- **Legacy path preserved:** `use_parallel_row_split` falls through to the original sequential loop when `ROW_ALLOW_MMVQ=0`. The old loop is unchanged downstream of `if (!use_parallel_row_split)`.

---

## Build Order Implications

The components have a strict dependency chain for safe implementation:

```
1. OOO queue infrastructure (ooo_qptrs[], ooo_stream())
   ALREADY DONE (common.hpp:370, common.hpp:389)
        │
        ▼
2. Per-device pinned staging buffers (host_staging_buf in dev_data)
   ALREADY DONE (ggml-sycl.cpp:3719-3722, 3997-4007)
        │
        ▼
3. Event-based copy path (GPU→host→GPU via handler submit + depends_on)
   ALREADY DONE (ggml-sycl.cpp:4127-4151)
        │
        ▼
4. Pre-op OOO barrier to gate kernel on copy completion
   ALREADY DONE (ggml-sycl.cpp:4204-4206)
        │
        ▼
5. Completion event recording per device (dev_events[i] = barrier())
   ALREADY DONE (ggml-sycl.cpp:4220)
        │
        ▼
6. Phase 2 host wait on dev_events[] (one wait per device)
   ALREADY DONE (ggml-sycl.cpp:4224-4234)
        │
        ▼
7. OOO queue flush for event pool management
   ALREADY DONE (ggml-sycl.cpp:4296-4301)
        │
        ▼
8. Phase 3 merge via event-based copy (NOT YET DONE)
   Phase 3 currently uses dev2dev_memcpy_staged_sync (blocking).
   Needs: same depends_on pattern as Phase 1, but for dst rows.
```

**The only remaining gap is Phase 3.** All other components are implemented and gated behind `GGML_SYCL_ROW_EVENTS=1`. Phase 3 still uses the fully-blocking `dev2dev_memcpy_staged_sync` per column, which adds N_devices * N_columns host waits back after Phase 1 eliminates them.

---

## Anti-Patterns

### Anti-Pattern 1: Relying on In-Order Queue Semantics for Cross-Operation Ordering

**What people do:** Submit copy + kernel to the same in-order queue and assume the kernel sees the copy's output.

**Why it's wrong:** L0 does NOT honor SYCL in-order queue guarantees. Kernels can execute before prior enqueued memcpy completes. This caused DEVICE_LOST crashes before the explicit pre-op `stream->wait()` was added.

**Do this instead:** Use `stream->wait()` between dependent operations on in-order queues, OR use OOO queues with explicit `depends_on()` chaining. The event path uses the latter — each operation in the chain carries its dependency explicitly via `handler::depends_on(event)`.

### Anti-Pattern 2: Sharing a Staging Buffer Across Devices

**What people do:** Allocate one pinned staging buffer and use it for all devices in the Phase 1 loop.

**Why it's wrong:** Phase 1 submits GPU→host copies from multiple devices concurrently. If they share one staging buffer, GPU1's copy overwrites GPU0's data before GPU2's host→GPU transfer completes.

**Do this instead:** One staging buffer per device (`dev[i].host_staging_buf`). Sized to max(src1_copy, merge_copy) for that device. Allocated once per `mul_mat` call, not per iteration.

### Anti-Pattern 3: Using `ext_oneapi_submit_barrier()` Across Queue Types

**What people do:** Record a barrier on an in-order queue and pass the resulting event to an OOO queue's `depends_on()`.

**Why it's wrong:** Cross-queue barrier events between in-order and OOO queues crash after many iterations. This was the "cross-queue barrier event crash" failure mode.

**Do this instead:** Keep each event chain within the same OOO queue. The pre-sync host wait (`ctx.stream()->wait()`) provides the in-order → OOO boundary crossing. After that, all Phase 1 events stay on OOO queues only.

### Anti-Pattern 4: Accumulating Events Without Flushing

**What people do:** Record many barrier events across repeated matmul calls and never flush the OOO queue.

**Why it's wrong:** L0 has a finite event pool. After ~100 matmuls the pool exhausts. This manifests as silent event failures or crashes.

**Do this instead:** Flush each OOO queue after Phase 3 (`ooo_stream(i)->wait()`). This releases all event handles used during the call. One flush per device per matmul — cheap because it happens after all GPU work is already done (Phase 2 already waited).

### Anti-Pattern 5: Batched Command Lists with Cross-Queue Event Sync

**What people do:** Use `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` (batched mode) with the event-based path.

**Why it's wrong:** Batched mode holds commands in a buffer and flushes them lazily. A cross-queue `depends_on()` on the other queue never sees the event fire because the source command list has not yet been flushed to the hardware. The dependent submission hangs indefinitely.

**Do this instead:** `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` is a hard requirement for the event path. This is enforced in `BenchConfig` via `ROW_EVENTS = ROW_BASE.with_(row_events=True, immediate_cmdlists=True)` in the benchmark framework.

---

## Scaling Considerations

This system is fixed at 3 devices and decode throughput targets. No generalization to larger clusters is in scope.

| Concern | Current (3 GPUs) | If Generalized (N GPUs) |
|---------|-----------------|------------------------|
| Staging buffers | 1 per non-main device | 1 per non-main device (already correct) |
| Phase 1 parallelism | 2 independent chains | N-1 independent chains (already correct) |
| Phase 3 latency | 2 sequential merges | N-1 sequential merges — main bottleneck |
| Event pool pressure | 2 flushes per matmul | N flushes per matmul (fine) |
| PCIe bandwidth | 2x src1 + 2x dst per matmul | N-1x src1 + N-1x dst — saturates at large N |

The merge phase (Phase 3) is the next bottleneck: it still does blocking column-by-column copies even with event-based Phase 1. At 3 GPUs with small activation tensors, this is acceptable. At larger N or larger batch sizes it becomes the dominant cost.

---

## Sources

All findings are HIGH confidence — derived directly from source analysis of the existing fork:

- `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 592-690 (copy functions), 3703-4305 (row-split dispatch)
- `ggml/src/ggml-sycl/common.hpp` lines 364-578 (`ggml_backend_sycl_context`)
- `scripts/bench/tests/test_row_split.py` (issue history, benchmark definitions)
- `.claude/projects/-home-ryan-llm-stack/memory/project_row_split_events.md`
- `.claude/projects/-home-ryan-llm-stack/memory/feedback_l0_inorder_queue.md`
- `.planning/PROJECT.md` (project requirements and constraints)

---

*Architecture research for: Event-Based Multi-GPU Sync for SYCL Row-Split*
*Researched: 2026-03-17*
