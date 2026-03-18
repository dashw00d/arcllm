# Feature Research

**Domain:** Multi-GPU LLM inference synchronization — SYCL/Level Zero event-based sync for row-split on Intel Arc A770
**Researched:** 2026-03-17
**Confidence:** HIGH (based on first-hand experimental data in test bench, PROJECT.md, and working implementation)

## Feature Landscape

This is an infrastructure milestone, not a product. "Users" are: the inference pipeline itself (correctness),
throughput benchmarks (performance), and future developers maintaining the code (debuggability).
"Table stakes" means the pipeline is broken without it. "Differentiators" mean better than the naive
host-stall approach that currently exists.

---

### Table Stakes (Pipeline Is Broken Without These)

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| OOO queue pool per device | Event-based `depends_on()` requires out-of-order queues; in-order queues do not support cross-queue chaining | MEDIUM | One OOO queue per device, separate from the main in-order stream. `common.hpp:370` already has `ooo_stream()` accessor stub. |
| Host-pinned staging buffers via `sycl::malloc_host` | `dev2dev_memcpy_staged()` does per-call `malloc`/`free`; that allocation cost adds to every stall. Pinned memory also enables async host-GPU copies without an extra copy stage. | MEDIUM | Pre-allocate at device init; size to max expected activation tensor. One buffer set per device. |
| `depends_on()`-chained GPU-GPU copy path | Current `dev2dev_memcpy_staged()` calls `src_stream.wait()` internally — a full device drain. This defeats OOO queues. A new copy path that chains via `handler::depends_on(event)` eliminates all host stalls in the hot path. | HIGH | The entire value of this milestone. Without this, the OOO queues accomplish nothing. |
| `ext_oneapi_submit_barrier()` completion event per device | Phase 2 of the 3-phase loop needs to wait for each device's work to finish before merge. A barrier event (not `queue.wait()`) is the minimal host touch: one wait per device per token rather than one per matmul. | MEDIUM | Already partially wired — PROJECT.md documents the pattern. Must ensure it captures OOO queue work, not just in-order stream work. |
| Immediate command lists required (`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`) | Batched command lists buffer kernel submissions and never flush cross-queue `depends_on()` events. This causes the event-based path to deadlock silently. Without immediate cmdlists, the entire feature is non-functional. | LOW | Already a `BenchConfig` flag. Must be enforced at the code level (or documented as a hard requirement) — not just in test configs. |
| Event pool exhaustion prevention | L0 has a finite event pool. After ~100+ matmuls in OOO queues without a flush, the pool exhausts and subsequent `submit()` calls fail or deadlock. Periodic OOO queue flush (e.g., after each token's 3-phase cycle) prevents exhaustion. | LOW | "Option A" from PROJECT.md: one line — flush OOO queue at end of Phase 2. Minor stall, but ~1 per token vs ~4000 currently. |
| Correctness: output matches host-stall reference within perplexity tolerance | If the event chains are wired incorrectly, the pipeline produces wrong output silently. Perplexity match (within 0.01 vs single-GPU reference) and 500+ token generation without desync are the correctness gates. | MEDIUM | This is the acceptance criterion. A fast-but-wrong implementation is worse than the current slow-but-correct one. |

---

### Differentiators (Better Than Naive Host-Stall Approach)

These are the features that make this a meaningful improvement over the existing `stream->wait()` path,
and better than a minimal "just make it work" event implementation.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| 3-phase device loop (submit-all / wait-all / merge) | Allows all non-main devices to overlap compute submission in Phase 1 with zero synchronization between them. The old sequential loop (copy→quantize→kernel→sync per device) serializes all device work. This restructuring gives parallel device utilization even before events eliminate host stalls. | MEDIUM | Already implemented as the "3-phase parallel" approach (0.5 t/s vs 0.4 t/s sequential). The event path builds on this structure. |
| src1 copy cache (skip redundant GPU-GPU copies) | Consecutive matmuls sharing the same input (e.g., `attn_q`, `attn_k`, `attn_v` all reading `attn_norm`) only need one copy per device per token. Cache the last-copied src1 pointer per device; skip the copy if the source hasn't changed. | LOW | Already implemented (+20% speedup). Must be preserved and extended to the event path — the cached copy's completion event must still be threaded into the `depends_on()` chain for the next matmul that uses the same input. |
| Host stall counting / sync path instrumentation | After the event-based path is live, the number of remaining host waits per token should be ~device_count (Phase 2 only). A counter that logs host-side `queue.wait()` calls confirms the event path is actually working rather than silently falling back to host sync. | LOW | Critical for validating the implementation. Can be a compile-time opt-in (`#ifdef GGML_SYCL_ROW_EVENTS_DEBUG`) to avoid production overhead. |
| `GGML_SYCL_ROW_EVENTS=1` env-var gate | The event-based path is a behavioral change with a specific hardware requirement (immediate cmdlists). Gating behind an env var allows safe A/B testing against the host-stall baseline without code rebuilds, and prevents accidental regression on hardware where immediate cmdlists are not available. | LOW | Already in `BenchConfig`. Needs to be enforced in the C++ hot path with `if (getenv("GGML_SYCL_ROW_EVENTS"))`. |
| Event-based merge path (compute-to-copy dependency) | The row-split merge step (`dev2dev_memcpy` from non-main to main device) currently requires the non-main device kernel to be fully drained before the copy can start. An event-based merge chains the copy's `depends_on(kernel_completion_event)` directly, removing the pre-merge `stream->wait()` that currently costs ~0.3ms per merge per device. | HIGH | This is the second major stall site after the per-matmul syncs. Requires the new copy path (table stakes above). Enables overlap: while main device starts merge, non-main device can begin the next matmul's quantize. |

---

### Anti-Features (Deliberately Out of Scope)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Worker threads for device dispatch | Intuitive: parallelize device submissions across threads to overlap latency | Tested and reverted (0.4 t/s, worse than single-threaded). Root cause: both workers submit GPU→host copies to the main device's queue simultaneously, causing queue contention. Threading overhead exceeds benefit at this batch size (np=1). | The 3-phase loop structure gives the same overlap benefit with zero threading overhead by restructuring the submission order, not parallelizing it. |
| SYCL graph capture for the row-split path | Graph execution encodes explicit kernel dependencies, could eliminate L0 overlap entirely | SIGABRT at startup — graph recording doesn't support all ops used in the row-split path (e.g., MoE expert routing). Cannot record a graph that contains these ops. | Event-based `depends_on()` achieves the same effect for the specific ops that matter (matmul + copy chains) without needing graph support. |
| Generalized multi-GPU event sync (beyond row-split) | Seems like the right abstraction — fix the L0 in-order queue violation everywhere | The layer-split crash is a different problem (L0 overlaps within a single device's command list, not cross-device). That requires either immediate cmdlists or per-op barriers at the graph level. Mixing both fixes in one milestone risks blocking on the harder problem. | GLM-4.7-Flash row-split first. Layer-split L0 fix is a separate milestone once row-split proves the event pattern. |
| Graceful fallback to host-sync if events fail | Sounds like defensive coding | If the event path silently falls back to host-sync, performance regressions are invisible. The `GGML_SYCL_ROW_EVENTS=1` gate already provides the fallback — turn the flag off. A silent fallback would mask bugs. | Explicit error logging if event submission fails, then crash loudly. The host-stall path remains the default (flag off). |
| Upstream PR polish during this milestone | The code should be mergeable to llama.cpp | Upstream polish (error handling, documentation, cross-platform testing) competes with the performance work. A clean implementation that hits 10+ t/s is the precondition for a PR, not the goal. | After performance target is hit, create a separate cleanup task before submitting PR. |
| Debug trace mode with per-event timing | Useful for diagnosing event latency | Adds overhead to the hot path even when disabled, and the benchmark framework already captures GPU freq/power/temp at the system level. Per-event latency is only needed if the event path is unexpectedly slow. | Use the benchmark framework's existing utilization data. If GPU clock stays at 600MHz, the sync path is still blocking. If it rises to 2100+ MHz, the events are working. |

---

## Feature Dependencies

```
[OOO queue pool per device]
    └──enables──> [depends_on()-chained copy path]
                      └──enables──> [event-based merge path]

[Host-pinned staging buffers]
    └──enables──> [depends_on()-chained copy path]
                      └──requires──> [OOO queue pool per device]

[Immediate command lists (enforcement)]
    └──required-by──> [OOO queue pool per device]
    └──required-by──> [depends_on()-chained copy path]

[3-phase device loop]
    └──enables──> [per-device completion event (barrier)]
                      └──enables──> [event pool exhaustion prevention]

[src1 copy cache]
    └──must-chain-with──> [depends_on()-chained copy path]
                           (cached copy's event must feed next matmul's depends_on)

[GGML_SYCL_ROW_EVENTS gate]
    └──wraps──> [OOO queue pool per device]
    └──wraps──> [depends_on()-chained copy path]
    └──wraps──> [event-based merge path]

[Host stall counter]
    └──validates──> [depends_on()-chained copy path]
    └──validates──> [event-based merge path]

[Correctness: perplexity + 500-token generation]
    └──validates──> all of the above
```

### Dependency Notes

- **`depends_on()` path requires OOO queues:** SYCL's `handler::depends_on(event)` only works when the submitting queue is out-of-order. In-order queues serialize by construction and ignore cross-queue event chains for dependency purposes.

- **`depends_on()` path requires immediate cmdlists:** Batched command lists accumulate kernel submissions and may not flush the event that a dependent kernel is waiting on. This causes silent deadlock, not an error. Confirmed experimentally (test_events_q4km_np1_batched documents the expected deadlock).

- **src1 cache must propagate its completion event:** The cache currently skips the GPU-GPU copy entirely when the src1 pointer matches the last-copied pointer. In the event path, the downstream matmul still needs a `depends_on()` pointing at the completion event from the *last* copy that loaded this src1. The cache must store and return the last copy event per device, not just skip silently.

- **Event pool exhaustion prevention gates long generation:** Without the periodic OOO queue flush, the pipeline works correctly for short sequences but crashes after ~100 matmuls (a few tokens). The flush is cheap (one `ooo_queue.wait()` per token, not per matmul) and must be in place before the 500-token correctness test.

---

## MVP Definition

### Launch With (v1 — hit 10+ t/s on GLM-4.7-Flash row-split)

- [ ] OOO queue pool infrastructure — one OOO queue per device, allocated at device init, accessible via `ooo_stream()`. All event-based submissions use this queue.
- [ ] Host-pinned staging buffers — pre-allocated at device init, replaces per-call malloc in `dev2dev_memcpy_staged()`.
- [ ] `depends_on()`-chained copy path — new `dev2dev_memcpy_event()` function that accepts a `sycl::event` dependency and returns a completion event. The current `dev2dev_memcpy_staged()` remains for the non-events path.
- [ ] Rewritten Phase 1 device loop — submit copy + quantize + kernel for all non-main devices using `depends_on()` chains, zero `stream->wait()` inside Phase 1.
- [ ] Phase 2 completion events — one `ooo_queue.ext_oneapi_submit_barrier()` per device after Phase 1 submits all work. Phase 2 calls `event.wait()` once per device.
- [ ] Event pool flush — at the end of Phase 2 (after all devices complete), flush OOO queues to reset L0 event pool.
- [ ] Correctness gate: perplexity matches single-GPU within 0.01 and 500-token generation succeeds without desync.

### Add After Validation (v1.x — squeeze remaining stalls)

- [ ] Event-based merge path — chain `dev2dev_memcpy` in Phase 3 via `depends_on(kernel_completion_event)` so the merge overlaps with the next token's Phase 1 submission. Add only after v1 proves stable (correct output, no desync over 500 tokens).
- [ ] src1 cache event propagation — extend the existing cache to return the last copy's completion event so downstream matmuls don't over-sync on already-cached inputs.

### Future Consideration (v2+ / upstream PR prep)

- [ ] Upstream PR polish — error handling, cross-platform guard (`#ifdef GGML_SYCL`), code review, documentation, test on non-Arc hardware.
- [ ] Generalize to layer-split — the event pattern is row-split-specific. Layer-split's L0 in-order violation is a different root cause (single-device command list reordering) that needs a separate solution.
- [ ] np > 1 stability at long sequences — np=2 currently crashes at ~56-83s even with host-stall sync. Root cause may be separate from the event work (could be the L0 in-order violation in non-MUL_MAT ops on the main device). Needs its own investigation after single-slot event path is stable.

---

## Feature Prioritization Matrix

| Feature | Throughput Value | Implementation Cost | Priority |
|---------|-----------------|---------------------|----------|
| OOO queue pool per device | HIGH (prerequisite) | LOW | P1 |
| Host-pinned staging buffers | MEDIUM (+20% on copies alone) | LOW | P1 |
| `depends_on()`-chained copy path | HIGH (eliminates 4000 stalls/token) | HIGH | P1 |
| Phase 1 submission loop rewrite | HIGH (structural change) | MEDIUM | P1 |
| Phase 2 barrier completion events | HIGH (replaces per-matmul sync) | MEDIUM | P1 |
| Event pool exhaustion prevention | HIGH (enables long generation) | LOW | P1 |
| Correctness validation (perplexity + 500tok) | HIGH (acceptance criterion) | LOW | P1 |
| Event-based merge path | MEDIUM (removes last stall site) | HIGH | P2 |
| src1 cache event propagation | LOW-MEDIUM (avoids over-sync) | LOW | P2 |
| Host stall counter / instrumentation | MEDIUM (validates correctness of sync) | LOW | P2 |
| `GGML_SYCL_ROW_EVENTS` C++ gate | LOW (already in BenchConfig, needs C++ side) | LOW | P1 |

**Priority key:**
- P1: Must have for v1 — pipeline produces wrong output or hangs without it
- P2: Add after v1 correctness is confirmed — performance improvements
- P3: Not in this milestone

---

## Competitor Feature Analysis

There is no other SYCL multi-GPU event-based sync implementation to compare against — this is novel
territory for llama.cpp on Intel Arc. The reference implementation is CUDA's multi-stream pattern
used in NVIDIA's tensor parallel inference.

| Feature | CUDA (NVIDIA reference) | Current SYCL (host-stall) | Target SYCL (event-based) |
|---------|------------------------|--------------------------|--------------------------|
| Cross-device copy dependency | `cudaStreamWaitEvent` — zero host stall | `stream->wait()` — full device drain per copy | `handler::depends_on(event)` — zero host stall |
| Copy memory type | Pinned `cudaHostAlloc` — async DMA | Per-call `malloc` — sync allocation overhead | Pre-allocated `sycl::malloc_host` — async DMA |
| Multi-device overlap | Implicit via stream independence | None — devices serialized by host | Phase 1 submits all devices before any sync |
| Event pool lifetime | Explicit `cudaEventCreate/Destroy` | N/A | Periodic OOO queue flush (Option A) |
| Completion wait | `cudaStreamSynchronize(stream)` — one per device | `queue.wait()` — N per device per token (where N = matmul count) | `barrier_event.wait()` — one per device per token |
| Host stalls per token (448 matmuls, 3 devices) | ~3 (one per device) | ~4000 (one per matmul per device) | ~3 (Phase 2 only) |
| Correctness mechanism | Stream ordering guarantees | Explicit `wait()` before every read | `depends_on()` chain — same guarantees, no host round-trip |

---

## Sources

- `/home/ryan/llm-stack/.planning/PROJECT.md` — PROJECT.md documents validated experimental results, known failure modes, and the SYCL-to-CUDA mapping. HIGH confidence.
- `/home/ryan/llm-stack/scripts/bench/tests/test_row_split.py` — Living documentation of every fix attempt and result, including the 3-phase loop, staged buffers, src1 cache, and event-based sync plan. HIGH confidence.
- `/home/ryan/llm-stack/scripts/bench/tests/test_q8_1_corruption.py` — Root cause analysis of L0 in-order queue violation; documents all sync approaches tried and their results. HIGH confidence.
- `/home/ryan/llm-stack/scripts/bench/tests/test_glm47.py` — GLM-4.7-Flash results and L0 overlap findings on the production target model. HIGH confidence.
- SYCL `handler::depends_on()` / `ext_oneapi_submit_barrier()` — standard SYCL 2020 / Intel oneAPI extensions, well-documented in Intel's DPC++ documentation. MEDIUM confidence (not re-verified against current docs during this session, but used in existing working code at `common.hpp:370`).

---

*Feature research for: Multi-GPU SYCL event-based sync, row-split inference path, Intel Arc A770*
*Researched: 2026-03-17*
