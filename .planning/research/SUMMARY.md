# Project Research Summary

**Project:** Event-Based Multi-GPU Sync for SYCL Row-Split Inference
**Domain:** SYCL/Level Zero multi-GPU synchronization on Intel Arc A770
**Researched:** 2026-03-17
**Confidence:** HIGH

## Executive Summary

This project implements event-based GPU-to-GPU synchronization for the row-split matrix multiplication path in llama.cpp's SYCL backend, targeting three Intel Arc A770 GPUs (48GB VRAM total). The core problem is that the current host-stall path calls `stream->wait()` approximately 4000 times per token — once per matmul per device — blocking the host CPU while the GPU pipeline could be running. The event-based approach replaces all intra-Phase-1 host waits with `handler::depends_on(event)` chains and `ext_oneapi_submit_barrier()` completion tokens, reducing host stalls from ~4000 to ~3 per token (one per device at Phase 2). This maps directly to CUDA's multi-stream pattern and has a clear, well-validated API path via oneAPI DPC++ 2025.x.

The recommended approach is a 3-phase dispatch loop: Phase 1 submits all device work (copy, quantize, kernel) as a chained dependency graph on per-device OOO queues with zero host waits; Phase 2 waits on one completion barrier event per device; Phase 3 merges partial results back to the main device. The infrastructure for phases 1-2 is already implemented and gated behind `GGML_SYCL_ROW_EVENTS=1`. The one remaining gap is Phase 3: the merge step still uses `dev2dev_memcpy_staged_sync` (fully blocking). Converting Phase 3 to event-based chains is the primary remaining work, along with correctness validation (perplexity match + 500-token generation).

The key risks are not conceptual but operational: Level Zero's batched command list mode silently prevents cross-queue events from firing (requiring `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`), the L0 event pool exhausts after ~100 matmuls if OOO queues are not flushed, and L0 does not honor SYCL in-order queue ordering semantics (requiring explicit barrier calls between dependent ops). All three risks are understood, documented in the benchmark framework, and have confirmed mitigations in the current codebase.

---

## Key Findings

### Recommended Stack

The entire stack is already installed and verified on this hardware. No version changes or new dependencies are needed. The critical constraint is environment configuration, not software selection.

**Core technologies:**
- **Intel oneAPI DPC++ 2025.3.2** — SYCL compiler; `handler::depends_on()` cross-queue bug fixed since 2022.1.0; all required APIs confirmed available
- **SYCL 2020 + `sycl_ext_oneapi_enqueue_barrier`** — primary sync primitives; `ext_oneapi_submit_barrier()` is the `cudaEventRecord` analog for OOO queues
- **Level Zero backend (bundled)** — underlying runtime; immediate command list mode is required (`IMMEDIATE_COMMANDLISTS=1`); batched mode deadlocks cross-queue event paths
- **`sycl::malloc_host` pinned staging buffers** — DMA-accessible by any device; enables async PCIe copies; already pre-allocated per-device in `dev_data[]`
- **`dpct::get_device(i).out_of_order_queue()`** — OOO queue accessor; `ooo_qptrs[]` infrastructure already exists in `common.hpp:370`

**What NOT to use:** `queue::wait()` in Phase 1 (the exact thing being replaced), `sycl::malloc_shared` for staging, worker threads (tested: 0.4 t/s, worse than single-threaded), SYCL graph capture (SIGABRT on MoE ops), direct L0 event pool APIs.

### Expected Features

This is an infrastructure milestone, not a product. "Users" are the inference pipeline (correctness) and throughput benchmarks (performance).

**Must have — v1 (pipeline broken without these):**
- OOO queue pool per device — prerequisite for `depends_on()` chaining (already implemented)
- Host-pinned staging buffers via `sycl::malloc_host` — async DMA without extra copy stage (already implemented)
- `depends_on()`-chained GPU-GPU copy path — new `dev2dev_memcpy_event()` replacing blocking `dev2dev_memcpy_staged()` (already implemented in Phase 1)
- Phase 2 barrier completion events — one `ext_oneapi_submit_barrier()` per device, one `event.wait()` per device (already implemented)
- Event pool flush — `ooo_stream(i)->wait()` after Phase 2 (already implemented)
- Correctness gate — perplexity matches single-GPU within 0.01 over 500 tokens (not yet validated)

**Should have — v1.x (squeeze remaining stalls):**
- Event-based merge path (Phase 3) — converts the last blocking stall site; Phase 3 currently uses `dev2dev_memcpy_staged_sync`
- src1 cache event propagation — cached copy must carry its completion event into downstream `depends_on()` chains

**Defer to v2+:**
- Upstream PR polish — error handling, cross-platform guards, documentation
- Generalize to layer-split — different root cause, separate milestone
- np > 1 stability at long sequences — crashes at 56-83s, likely separate from event work

**Anti-features (out of scope):**
- Worker threads — tested and reverted (queue contention, 0.4 t/s)
- SYCL graph capture — SIGABRT on MoE expert routing ops
- Silent fallback to host-sync — would mask bugs; use explicit `GGML_SYCL_ROW_EVENTS` gate

### Architecture Approach

The event sync layer is surgically isolated to the `use_parallel_row_split` code path. Queue selection (`ooo_stream(i)` vs `ctx.stream(i, is)`) is the only integration seam — kernels (`mmvq`, `fused_mmq`) receive a queue pointer and are unaware of OOO vs in-order. The legacy host-stall path is fully preserved and remains the default when `GGML_SYCL_ROW_EVENTS=0`.

**Major components:**
1. **`ggml_backend_sycl_context`** (`common.hpp:364`) — owns OOO queue pool (`ooo_qptrs[]`), in-order queues, src1 copy cache (`row_src1_cache[]`), and device memory pools; lifetime is the backend context
2. **`dev_data[]` per-call scratch** (`ggml-sycl.cpp:3703`) — per-device weight slice pointers, activation buffers, pinned staging buffers; stack-allocated per `mul_mat` call
3. **3-phase dispatch loop** (`ggml-sycl.cpp:4053-4301`) — Phase 1 submits all device work via OOO event chains; Phase 2 waits on completion barriers; Phase 3 merges results (currently blocking, target: event-based)
4. **`dev_events[]`** — per-device completion tokens from `ext_oneapi_submit_barrier()`; the Phase 1-to-Phase-2 handoff
5. **OOO queue flush** (`ggml-sycl.cpp:4296-4301`) — `ooo_stream(i)->wait()` after Phase 3; releases L0 event handles back to the pool

**Build order:** Components 1-2 are done. Component 3 (Phase 1-2) is done. The only remaining gap is Phase 3 event-based merge (component 3, Phase 3) plus the correctness validation harness.

### Critical Pitfalls

1. **Batched command lists deadlock cross-queue events** — `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` is a hard requirement; batched mode accumulates commands without flushing so cross-queue `depends_on()` events never fire. Silent hang, no error. Enforce in OOO queue constructor; BenchConfig already sets this via `row_events=True`.

2. **L0 in-order queue does NOT enforce submission order** — Kernels on in-order queues can execute before prior `memcpy` completes. `DEVICE_LOST` is the symptom. Always use `ext_oneapi_submit_barrier({prior_event})` between dependent ops; never rely on queue ordering implicitly.

3. **Event pool exhaustion after ~100 matmuls** — L0 has a finite event pool; `ext_oneapi_submit_barrier()` consumes handles that are only released by `queue.wait()`. At 448 matmuls/token × 3 events each, the pool saturates in 2-3 tokens. The `ooo_stream(i)->wait()` flush at `ggml-sycl.cpp:4296-4301` is the fix; do not remove it.

4. **Staging buffer reuse without event chaining corrupts data** — OOO queues allow concurrent iteration submissions; a new GPU→host copy can overwrite the staging buffer before the prior iteration's host→GPU completes. The `e_prev_staging_free` / `has_prev_staging` chaining pattern in Phase 1 (`ggml-sycl.cpp:4082-4148`) is mandatory, not optional.

5. **Q8_1 quantization `ds` field L1 cache visibility bug (Arc A770)** — The `ds` scale field must be written by ALL work items, not only `wi_id==0`. Upstream merges may revert this workaround (`quantize.hpp:116-120`). If reverted, all MMVQ output silently becomes zero. This is a hardware-specific bug; do not "fix" it away.

6. **Cross-context OOO queues silently drop `depends_on()` events** — If `ooo_qptrs[i]` is constructed from a different SYCL context than the in-order streams, cross-device events are silently ignored. Add startup assertion: `ooo_queue.get_context() == stream->get_context()`. The dpct device manager already provides shared-context queues; verify this holds after any dpct changes.

---

## Implications for Roadmap

Based on research, the work is further along than a typical "implement from scratch" milestone. Components 1-7 in the build order (OOO queues through Phase 2 host wait) are implemented. The roadmap should reflect this state: early phases are validation and cleanup of existing work, not new implementation.

### Phase 1: Harden OOO Queue Infrastructure
**Rationale:** Existing `ooo_stream()` accessor may return queues from the wrong SYCL context (Pitfall 6). Before any correctness testing, the foundation must be verified correct. This phase unblocks all subsequent validation.
**Delivers:** Startup assertion on OOO queue context; `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` enforced in code (not just env); explicit event lists in all hot-path `ext_oneapi_submit_barrier` calls; debug `fprintf` removed from Phase 2.
**Addresses:** FEATURES table-stakes items "OOO queue pool per device" and "immediate command lists enforcement"
**Avoids:** Pitfalls 1 (batched cmdlist deadlock), 5 (barrier no-op on in-order queue), 6 (cross-context event silence)
**Research flag:** No additional research needed — all APIs verified, patterns confirmed working.

### Phase 2: Correctness Validation of Existing Event Path
**Rationale:** Phases 1-2 of the 3-phase loop are implemented but not validated end-to-end. A single-matmul correctness test must pass before Phase 3 is added. This is the acceptance gate for the existing implementation.
**Delivers:** Perplexity match (single-GPU reference within 0.01) over 500-token generation with `GGML_SYCL_ROW_EVENTS=1`; host stall counter confirms ~3 waits/token (not 4000); `test_row_split.py` event benchmark reaches baseline correctness
**Addresses:** FEATURES correctness gate "perplexity + 500-token generation"
**Avoids:** Pitfalls 2 (L0 ordering), 3 (event pool exhaustion), 4 (staging buffer race), 7 (Q8_1 ds bug), 8 (pre-op sync removal)
**Research flag:** No additional research needed — test patterns are established in benchmark framework.

### Phase 3: Event-Based Merge Path (Phase 3 of Dispatch Loop)
**Rationale:** Phase 3 (`dev2dev_memcpy_staged_sync`) is the last blocking stall site. After Phase 2 correctness is confirmed, converting merge to event-based chains overlaps the next token's Phase 1 with the current token's merge. This is the second-highest throughput gain after Phase 1.
**Delivers:** New `dev2dev_memcpy_event()` function for merge; Phase 3 returns a completion event instead of blocking; overlap of next-token Phase 1 with current-token merge; updated benchmark results
**Uses:** Same `handler::depends_on(event)` + `sycl::malloc_host` patterns as Phase 1 copy path
**Implements:** "Event-based merge path" differentiator from FEATURES.md
**Avoids:** Pitfall 4 (staging buffer reuse race applies to merge path too)
**Research flag:** No additional research needed — implementation pattern is identical to Phase 1 copy path, already working.

### Phase 4: src1 Cache Event Propagation
**Rationale:** The src1 copy cache (skipping redundant GPU-GPU copies for consecutive attn_q/k/v matmuls) currently skips the copy silently. In the event path, the downstream matmul must still receive the completion event from the last copy. Without this, the cache either over-syncs (defeats the cache benefit) or under-syncs (race condition).
**Delivers:** Cache stores last copy completion event per device; downstream matmuls receive correct `depends_on()` handle; no regression to pre-cache throughput
**Addresses:** FEATURES "src1 copy cache event propagation" (P2)
**Avoids:** Pitfall 4 (staging buffer data dependency) — same pattern applies to cached copies
**Research flag:** No additional research needed — dependency structure is clear.

### Phase 5: Performance Characterization and Cleanup
**Rationale:** After all event-based paths are validated correct, instrument and measure. The host stall counter confirms the event path is working vs. silently falling back. Clean up technical debt (debug printfs, `GGML_SYCL_ROW_EVENTS` gate evaluation, OOO queue context assertion promotion to proper error handling).
**Delivers:** Throughput benchmark results with event path (target: 10+ t/s on GLM-4.7-Flash row-split); updated `CLAUDE.md` key results table; `test_row_split.py` docstring updated with final results; decision on whether to keep or remove `ROW_EVENTS` gate
**Uses:** `scripts/bench/` framework — `rowsplit.*` and `thinking.*` test suites
**Addresses:** FEATURES "host stall counting / sync path instrumentation" and "GGML_SYCL_ROW_EVENTS gate" (both P2)
**Research flag:** No additional research needed — benchmark framework is fully operational.

### Phase Ordering Rationale

- **Foundation before validation:** Phase 1 (context assertion, immediate cmdlist enforcement) must precede Phase 2 (correctness testing) because an incorrect OOO queue context would silently fail tests without revealing the root cause.
- **Existing work before new work:** Phases 1-2 validate what's already implemented before Phase 3 adds new functionality. This prevents building on a broken foundation.
- **Cache after merge:** Phase 4 (src1 cache event propagation) is lower priority than Phase 3 (merge) because the cache skips the copy; the copy path must be event-based before the cache can properly propagate events.
- **Performance last:** Phase 5 characterizes after all correctness is confirmed. Premature optimization on an incorrect implementation wastes time.

### Research Flags

Phases with standard, well-documented patterns (skip `/gsd:research-phase`):
- **Phase 1:** All API behaviors are verified; immediate cmdlist enforcement is a one-line env var check; the only risk is the context assertion which has a known resolution.
- **Phase 2:** Test patterns are fully defined in the benchmark framework; this is execution, not research.
- **Phase 3:** Implementation pattern is identical to the already-working Phase 1 copy path; the function signature and event chaining are established.
- **Phase 4:** The dependency structure is described in FEATURES.md with full detail.
- **Phase 5:** The benchmark framework handles measurement; the only decision is threshold for removing the `ROW_EVENTS` gate.

No phases require deeper research during planning. All unknowns are implementation details with established solutions.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All APIs verified against official Intel docs and existing working code in this fork; installed compiler version confirmed; no new dependencies needed |
| Features | HIGH | Based on first-hand experimental data from `test_row_split.py`; `PROJECT.md` documents all attempted approaches and their outcomes; feature boundaries are empirically validated, not speculative |
| Architecture | HIGH | Derived from direct source code analysis of `ggml-sycl.cpp` and `common.hpp`; build order reflects actual implementation state with specific line numbers; component boundaries are confirmed by working code |
| Pitfalls | HIGH | All 8 pitfalls are from first-hand crash logs or confirmed Intel LLVM issues; each has a verified mitigation already in the codebase; recovery strategies are documented with reproduction steps |

**Overall confidence:** HIGH

### Gaps to Address

- **Phase 3 event-based merge implementation details:** The merge path (`dev2dev_memcpy_staged_sync`) copy direction is GPU_i VRAM → pinned staging → GPU0 VRAM. The staging buffer sizing must accommodate both src1 (activation, ~1MB) and merge (partial result, ~varies) copies. Sizing analysis should be done at Phase 3 start.

- **np > 1 stability root cause:** np=2 crashes at 56-83s even with host-stall sync. This is likely an L0 in-order queue violation in non-MUL_MAT ops on the main device — a separate root cause from the event work. Do not attempt to fix this during the event milestone; document and flag for a separate investigation.

- **`GGML_SYCL_ROW_EVENTS` gate removal criteria:** No specific threshold is defined for when the gate can be removed and the event path becomes the default. Establish: correctness over 500 tokens, throughput > 10 t/s, and no crashes over 10-minute sustained load.

---

## Sources

### Primary (HIGH confidence)

- `scripts/bench/tests/test_row_split.py` — Living documentation of all experiments, failures, and fixes; first-hand experimental results on this hardware
- `scripts/test_ooo_row_split.cpp` — Standalone OOO queue validation; Tests 3 and 4 validate barrier ordering and staging buffer reuse
- `ggml/src/ggml-sycl/ggml-sycl.cpp:3703-4305` — Current implementation with inline explanatory comments; line references verified
- `ggml/src/ggml-sycl/common.hpp:364-578` — `ggml_backend_sycl_context` definition including `ooo_qptrs`, `row_src1_cache`, pool management
- `.planning/PROJECT.md` — Project requirements, constraints, and validated experimental results
- [Intel oneAPI DPC++ Programming Guide (2025-1)](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2025-1/data-parallelism-in-c-using-sycl.html) — `handler::depends_on()` API
- [sycl_ext_oneapi_enqueue_barrier spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc) — `ext_oneapi_submit_barrier()` semantics
- [Level Zero Core Programming Guide](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html) — Event pool flags, reset/reuse, cross-device forward-progress
- [Level Zero Immediate Command Lists guide](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html) — Batched vs immediate command list behavior
- [Intel oneAPI GPU Optimization Guide: SYCL Data Transfers (2024-1)](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/optimize-sycl-data-transfers.html) — `malloc_host` recommendation
- [Intel llvm issue #4791](https://github.com/intel/llvm/issues/4791) — Event pool destruction bug; fixed in 2022.1.0
- [Intel llvm issue #15606](https://github.com/intel/llvm/issues/15606) — `ext_oneapi_submit_barrier` + in-order queue no-op bug

### Secondary (MEDIUM confidence)

- [sycl_ext_intel_queue_immediate_command_list spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_queue_immediate_command_list.asciidoc) — "Well-tested on PVC only" caveat for the per-queue SYCL property (not the env var)
- [Intel community: depends_on with events from different queues](https://community.intel.com/t5/Intel-oneAPI-DPC-C-Compiler/Using-depends-on-with-events-returned-from-different-queues/td-p/1310387) — Cross-context event silent failure

### Tertiary (LOW confidence, for awareness only)

- [llama.cpp discussion #5277](https://github.com/ggml-org/llama.cpp/discussions/5277) — Community reports of Arc A770 stability issues including fence timeout; relevance to this project unconfirmed

---
*Research completed: 2026-03-17*
*Ready for roadmap: yes*
