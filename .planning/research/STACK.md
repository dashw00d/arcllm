# Technology Stack

**Domain:** Multi-GPU event-based synchronization for SYCL row-split inference on Intel Arc A770
**Researched:** 2026-03-17
**Confidence:** HIGH (SYCL APIs verified against official Intel docs + existing working code in this fork)

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Intel oneAPI DPC++/C++ Compiler | 2025.3.2 (installed) | Compile SYCL kernels and host code | Required for SYCL 2020 spec + Intel extensions. This exact version is already in `env.sglang-xpu.sh`. No version change needed. |
| SYCL 2020 (via DPC++) | rev 11 | Language model for queues, events, USM | Standard spec. All APIs used (`handler::depends_on`, `ext_oneapi_submit_barrier`, `sycl::malloc_host`) are in or adjoint to the SYCL 2020 standard. |
| Level Zero backend | (bundled with oneAPI) | Underlying L0 runtime that SYCL submits to on Intel GPU | The only backend on Intel Arc. All event pool behavior, command list modes, and cross-device semantics are L0-specific. |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` | env var (already set) | Force immediate command list submission | Hard requirement for event-based sync. Batched mode holds commands in a buffer; cross-queue `depends_on()` events never fire because the source list has not been flushed. **Confirmed experimentally: batched mode deadlocks the event path.** |

### SYCL Event APIs

#### `handler::depends_on(event)` — Primary Dependency Primitive

**What it is:** A `sycl::handler` method called inside a `queue.submit([&](handler& h){...})` lambda. Instructs the runtime: "this command group must not begin until `event` has reached the complete state."

**API signature:**
```cpp
// Inside a submit lambda:
q.submit([&](sycl::handler& h) {
    h.depends_on(some_event);       // single event
    h.depends_on({ev1, ev2, ev3});  // vector of events
    h.memcpy(dst, src, bytes);
});
```

**Confidence:** HIGH — standard SYCL 2020 spec, verified against Intel DPC++ docs, and used in working code in this fork at `ggml-sycl.cpp:4127-4151`.

**Critical constraint:** The submitting queue must be out-of-order (OOO). In-order queues serialize submissions by construction and the depends_on dependency edge is redundant/ignored for cross-queue chains. The existing `ooo_qptrs[dev]` infrastructure (`common.hpp:370`) already provides OOO queues.

**Cross-queue semantics:** The SYCL spec explicitly states there are no constraints on context from which the waited-on event's queue may come. Cross-device events are supported. An event recorded on GPU0's OOO queue can be passed to GPU1's `depends_on()`. This is the core CUDA-equivalent pattern.

**Historical bug (fixed):** Pre-2022.1 DPC++ had a bug where cross-queue `depends_on` events were silently ignored. The installed compiler (2025.3.2) is unaffected. MEDIUM confidence that it was fixed in 2022.1.0 based on Intel community forum — verified by the working code in this fork.

---

#### `queue::ext_oneapi_submit_barrier()` — Barrier Event Recording

**What it is:** A `sycl::queue` member function that submits a barrier command and returns a `sycl::event`. The barrier gates: commands submitted after it to the same queue cannot start until the barrier's wait conditions are met.

**API signatures:**
```cpp
// Capture all prior work on this queue as a single event:
sycl::event completion = ooo_q.ext_oneapi_submit_barrier();

// Gate on external events from other queues (cross-queue fan-in):
sycl::event gated = ooo_q.ext_oneapi_submit_barrier({e_from_other_queue});
```

**Why use it instead of `queue.wait()`:** `wait()` blocks the host thread until all queued work completes. `ext_oneapi_submit_barrier()` returns immediately to the host, leaving synchronization as a GPU-side dependency edge. This is the difference between blocking 1000 times per token vs. letting the GPU pipeline run.

**Equivalent to:** `cudaEventRecord(event, stream)` for the no-args form. `cudaStreamWaitEvent(stream, event)` for the wait-list form.

**Known issue (2024, fixed):** Multiple calls to `queue::ext_oneapi_submit_barrier` on in-order non-discard queues had exponential slowdown — fixed in 2024 releases. **This affects in-order queues only.** The event path uses OOO queues for all Phase 1 work; this bug does not apply. However, calling `ext_oneapi_submit_barrier` on the main device's in-order queue (for pre-sync) could trigger it. Use `stream->wait()` for the in-order pre-sync, not `ext_oneapi_submit_barrier`.

**Confirmation macro:** Check `SYCL_EXT_ONEAPI_ENQUEUE_BARRIER` == 1 to confirm support. On oneAPI 2025.x + Arc A770 this is always true.

---

#### `sycl::event` Lifetime and Pool Management

**What it is:** A copyable handle to a submitted command group's completion status. Internally backed by a Level Zero event handle from a fixed-size L0 event pool.

**Event pool behavior on L0:**
- L0 allocates events from a fixed pool at context creation time
- Each `ext_oneapi_submit_barrier()` or `submit()` that produces an event consumes one handle
- Handles are NOT automatically returned to the pool when the `sycl::event` C++ object is destroyed
- Pool exhaustion causes submission failures or silent hangs after ~100+ matmuls of accumulated events

**Exhaustion prevention — the only viable pattern:**
```cpp
// After Phase 2 completes (all dev_events[i].wait() have returned):
for (int i = 1; i < n_devices; i++) {
    ctx.ooo_stream(i)->wait();  // ONE host wait per device per matmul
}
// This flushes the OOO queue's pending work and returns event handles to the pool.
```

**Why this is correct:** Phase 2 already waited for all device work to complete (`dev_events[i].wait()`). The OOO flush is a no-op from a compute standpoint — all the work it "waits on" is already done. It purely releases the L0 event handles.

**Confirmed working:** This pattern is implemented and documented in ARCHITECTURE.md (Pattern 4) and `test_row_split.py` (Issue #1 in the row-split history). The alternative (larger initial pool) does not prevent unbounded accumulation; periodic flush is required.

**Counter-based events (L0 extension):** The L0 spec describes counter-based events that auto-reset without explicit reset calls. These are NOT exposed via SYCL's SYCL 2020 API — they require direct L0 interop via `ze_event_pool_flags_t::ZE_EVENT_POOL_FLAG_COUNTER_BASED`. Do not use: adds L0 interop complexity with no benefit over the OOO flush pattern.

---

### Level Zero Event Pool Configuration

> Note: Direct L0 event pool management is not needed for this project — the SYCL runtime manages pools internally. These details are provided for understanding failure modes, not as a usage recipe.

**`ZE_EVENT_POOL_FLAG_HOST_VISIBLE`:** Required for any event that the host will query via `zeEventHostSynchronize` (equivalent to `event.wait()` on the host). The SYCL runtime sets this flag for events it creates. Do not bypass the SYCL event API — there is no need to call `zeEventPoolCreate` directly.

**`ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP`:** Only needed for profiling. Adds overhead. Not used in the hot path.

**`ZE_EVENT_POOL_FLAG_IPC`:** Cross-process sharing. Not applicable — all three Arc GPUs are in the same process.

**Pool sizing:** The SYCL runtime pre-allocates event pools with a fixed `count`. The exact count is runtime-internal and not user-configurable via the SYCL API. The OOO queue flush pattern (above) is the only reliable way to return handles to the pool — pool sizing is not a lever available to application code.

**Reset vs. create:** `zeEventHostReset` resets a signaled event for reuse without destroying it. The SYCL runtime handles this internally when events are released by the queue flush. Application code should not call L0 event APIs directly — the SYCL layer owns the pool.

---

### Memory Allocation: `sycl::malloc_host` for Staging

**Recommendation:** Use `sycl::malloc_host` for the per-device pinned staging buffers. Do not use `sycl::malloc_shared` or system `malloc`.

| Allocation Type | What It Is | Use For |
|-----------------|-----------|---------|
| `sycl::malloc_host(bytes, q)` | Pinned host memory, DMA-accessible by any device | GPU-GPU staging buffers. Enables async PCIe DMA without an extra copy stage. |
| `sycl::malloc_shared(bytes, q)` | USM shared memory, can migrate between host and device | NOT appropriate for staging. Migration is not controlled by `depends_on()` chains. Will degrade to on-demand page migration, adding latency. |
| `sycl::malloc_device(bytes, q)` | VRAM on a specific device | Device-side buffers (src1_ddf, dst_dd, etc.). Already used correctly in the codebase. |
| System `malloc` / `new` | Pageable host memory | NOT DMA-accessible without an extra implicit copy by the runtime. Adds latency to every transfer. |

**Why `malloc_host` specifically:**
- Pinned (non-pageable): the physical page is locked, so the DMA engine can read/write it without going through the OS page table
- Accessible by any device in the SYCL context: GPU0 can write it and GPU1 can read it without any extra copy
- Transfers initiated via `handler.memcpy(malloc_host_ptr, ...)` are async (DMA-driven); the CPU is not involved in the data movement
- Intel optimization guide confirms: "data transfer rate is maximized when both source and destination are in Unified Shared Memory (USM). Allocate data that will be the source or destination of data transfer in host USM memory using the malloc_host API."

**Staging buffer sizing:** Allocate one buffer per non-main device, sized to `max(src1_copy_bytes, merge_copy_bytes)` for that device. The existing implementation in `dev_data[]` does this correctly. The key is pre-allocation at `mul_mat` call entry (not inside the OOO submission lambda).

**Transfer performance on PCIe 3.0 x16:** Theoretical ~16 GB/s per direction. Practical sustained: 10-12 GB/s. For a 4096-row Q4_K_M src1 activation: ~1MB per copy, ~100µs. This is the PCIe transfer cost floor regardless of sync strategy — event-based sync eliminates the host stall overhead but does not reduce this physics cost.

---

### Queue Infrastructure

**OOO queues via `dpct::get_device(i).out_of_order_queue()`:**

The existing `ooo_stream(dev)` accessor in `ggml_backend_sycl_context` (`common.hpp:389`) creates OOO queues on first access using the dpct device manager. These are the correct queues for all Phase 1 event submissions.

```cpp
// Existing accessor (common.hpp:389):
queue_ptr ooo_stream(int device) {
    if (ooo_qptrs[device] == nullptr) {
        ooo_qptrs[device] = &(dpct::get_device(device).out_of_order_queue());
    }
    return ooo_qptrs[device];
}
```

**Shared context requirement:** The dpct device manager creates all queues within a shared SYCL context. This is critical — cross-device `depends_on()` events only work if both queues share a context (or the events carry sufficient inter-context information). The dpct layer handles this correctly: `test_dpct_events.cpp` verifies `same_ctx: YES` for dpct-managed queues.

**Immediate vs. batched command lists:**

| Mode | Flag | Cross-queue event behavior | Use? |
|------|------|---------------------------|------|
| Immediate | `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` | Commands execute immediately; events fire immediately. Cross-queue `depends_on()` sees events correctly. | YES — required |
| Batched | `=0` | Commands accumulate in a buffer. Source command list may not have flushed when dependent queue polls for the event. Dependent submission hangs indefinitely. | NO — deadlocks confirmed |

**The immediate cmdlist note about Arc GPUs (from Intel docs):** The `sycl_ext_intel_queue_immediate_command_list` queue property is "well-tested only on Intel Data Center Max Series GPUs (aka PVC)" — this is about the **SYCL queue property**, not the env var. The env var `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` applies globally and works on Arc A770. The per-queue SYCL property is an advanced API that bypasses the env var; do not use it. Continue using the env var.

---

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `dpct/helper.hpp` (Intel DPCT) | (bundled with oneAPI 2025.3.2) | Device management, event type aliases (`dpct::event_ptr`), queue creation | Already in use throughout ggml-sycl. Do not bypass for core queue/device operations — dpct ensures shared context across devices. |
| Intel oneAPI Math Kernel Library (MKL) / oneDNN | (bundled) | Linear algebra primitives | Not relevant to the sync layer. Used in ggml-sycl for some ops but not in the row-split hot path. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `queue::wait()` in Phase 1 hot path | Full device drain — blocks host until all queued work completes. 9 waits per matmul = 4000+ stalls per token. This is the exact thing being replaced. | `handler::depends_on(event)` + `ext_oneapi_submit_barrier()` |
| `sycl::malloc_shared` for staging buffers | USM shared memory migrates on demand, not under `depends_on()` control. Transfer latency becomes non-deterministic and can be much higher than pinned memory. | `sycl::malloc_host` |
| `dev2dev_memcpy_staged()` on the OOO event path | Its internal `src_stream.wait()` drains the in-order queue. Calling this from inside a Phase 1 OOO chain defeats the entire event mechanism. | New `dev2dev_memcpy_event()` function using handler submit + `depends_on()` |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` (batched) | Cross-queue `depends_on()` events never fire because the source command list is not flushed. Deadlock is silent and looks like a hang. | Always set to `=1` for the event path; enforce in `BenchConfig::row_events=True` |
| Worker threads for device dispatch | Tested and reverted: both workers submitting GPU→host copies to the main device's queue simultaneously causes queue contention. 0.4 t/s, worse than single-threaded 0.6 t/s. | The 3-phase loop structure achieves the same overlap via submission ordering, not threading |
| `ext_oneapi_submit_barrier()` on in-order queues for pre-sync | In-order queues + this function had exponential slowdown bug in 2024 releases (now fixed, but why risk it). Also conceptually wrong — the pre-sync purpose is to drain the in-order queue's pending upstream ops (norm, rope), not to create a dependency token. | `ctx.stream()->wait()` for the one pre-sync per matmul |
| SYCL graph capture for the row-split path | SIGABRT at startup — graph recording does not support all ops in the row-split path (MoE expert routing). Tested and documented in test_row_split.py. | `depends_on()` chains achieve the same dependency semantics for the subset of ops that matter |
| Direct Level Zero event API (`zeEventPoolCreate`, `zeEventCreate`) | Bypasses the SYCL runtime's event management. Creates interop hazards with SYCL's own pool. Adds L0-specific code for no benefit — the SYCL flush pattern handles pool management adequately. | SYCL event API throughout; OOO queue flush for pool reset |
| `sycl_ext_oneapi_discard_queue_events` property | Deprecated. "This extension no longer provides any benefit. Optimizations enabled by these interfaces have already been disabled in the compiler." | Normal queue without this property |

---

## Stack Patterns by Variant

**If `GGML_SYCL_ROW_EVENTS=1` (event-based path):**
- Use `ooo_stream(dev)` OOO queues for all Phase 1 submissions
- Use `handler::depends_on(event)` for copy-to-quantize and quantize-to-kernel dependencies
- Use `ext_oneapi_submit_barrier()` for per-device completion event recording (Phase 1 end) and copy gating
- Use `malloc_host` staging buffers (pre-allocated, per-device)
- Flush OOO queues after Phase 2 for event pool management
- Require `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`

**If `GGML_SYCL_ROW_EVENTS=0` (legacy host-stall path, default):**
- Use `qptrs[dev][stream]` in-order queues (existing)
- Use `dev2dev_memcpy_staged()` (existing)
- Use `stream->wait()` for synchronization (existing)
- Batched or immediate cmdlists both work

**If Phase 3 merge is also event-based (future v1.x):**
- Record `kernel_completion_event = dev_events[i]` from Phase 1
- New merge copy: `stream->submit([&](handler& h){ h.depends_on(kernel_completion_event); h.memcpy(...); })`
- Phase 3 then returns an event instead of blocking
- This overlaps next-token Phase 1 with current-token Phase 3 merge

---

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| Intel oneAPI DPC++/C++ Compiler | 2025.3.2 | Installed. All APIs confirmed available. `handler::depends_on` cross-queue bug fixed in 2022.1.0. |
| SYCL 2020 spec | rev 11 | `ext_oneapi_submit_barrier` is an Intel extension (not core spec), supported since oneAPI 2021+. |
| Level Zero driver | (system-installed) | Immediate cmdlist behavior verified experimentally on this hardware. Event pool exhaustion behavior documented in test_row_split.py. |
| `sycl_ext_oneapi_enqueue_barrier` | (bundled) | Extension macro `SYCL_EXT_ONEAPI_ENQUEUE_BARRIER == 1` confirmed on this platform. |

---

## Sources

- Intel oneAPI DPC++ Programming Guide (2025-1): [`handler::depends_on` API](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2025-1/data-parallelism-in-c-using-sycl.html) — HIGH confidence
- [sycl_ext_oneapi_enqueue_barrier spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc) — HIGH confidence (official Intel LLVM repo)
- [Level Zero Core Programming Guide](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html) — HIGH confidence (L0 spec); event pool flags, reset/reuse patterns, cross-device forward-progress warning
- [Level Zero Immediate Command Lists guide](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html) — HIGH confidence; confirms multi-queue concurrency requires immediate mode
- [Intel oneAPI GPU Optimization Guide: Optimize SYCL Data Transfers (2024-1)](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/optimize-sycl-data-transfers.html) — HIGH confidence; `malloc_host` recommendation for staging
- [sycl_ext_intel_queue_immediate_command_list spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_queue_immediate_command_list.asciidoc) — MEDIUM confidence; "well-tested on PVC only" caveat documented
- [Intel LLVM issue #4791](https://github.com/intel/llvm/issues/4791) — HIGH confidence; event pool destruction bug, fixed in 2022.1.0
- [Intel LLVM issue #15606](https://github.com/intel/llvm/issues/15606) — HIGH confidence; `ext_oneapi_submit_barrier` + in-order queue interaction bug and fix
- `scripts/bench/tests/test_row_split.py` — HIGH confidence (first-hand experimental results on this hardware)
- `scripts/test_cross_device_events.cpp` — HIGH confidence (standing test for cross-device barrier semantics)
- `ggml/src/ggml-sycl/common.hpp` — HIGH confidence (existing working implementation)
- `icpx --version` output — HIGH confidence (exact installed compiler version: 2025.3.2.20260112)

---

*Stack research for: Event-Based Multi-GPU Sync, SYCL row-split, Intel Arc A770*
*Researched: 2026-03-17*
