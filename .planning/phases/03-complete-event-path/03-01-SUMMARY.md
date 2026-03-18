---
phase: 03-complete-event-path
plan: 01
subsystem: infra
tags: [sycl, row-split, events, ggml, intel-arc]

requires:
  - phase: 02-validate-existing-event-path
    provides: "Stall counter instrumentation and CORR-01/CORR-02 correctness baseline (Phase 1+2 of 3-phase loop verified working)"

provides:
  - "Event-based Phase 3 merge: GPU_i->staging->dst via OOO submit with depends_on(dev_events[i])"
  - "src1 cache event propagation: last_copy_event stored on miss, chained via barrier on hit"
  - "Complete event path: zero host waits in Phase 1 + Phase 3 (only Phase 2 waits remain)"
  - "Bench tests documenting SYNC-01, SYNC-02, and end-to-end stall target (<= 4/token)"

affects:
  - 03-complete-event-path

tech-stack:
  added: []
  patterns:
    - "SYNC-01: OOO submit with depends_on(dev_events[i]) for Phase 3 merge copies"
    - "SYNC-02: sycl::event stored in cache entry, propagated via ext_oneapi_submit_barrier({cached_event}) on hit"
    - "Staging buffer reuse via e_prev_merge_staging / has_prev_merge_staging (same pattern as Phase 1 e_prev_staging_free)"
    - "Event path gated behind use_event_sync boolean; legacy else branch identical to prior code"

key-files:
  created:
    - scripts/bench/tests/test_row_split.py (added SYNC-01/02 test methods + complete_event_path gate)
  modified:
    - llama.cpp/ggml/src/ggml-sycl/common.hpp (row_src1_cache_entry: +last_copy_event, +has_event)
    - llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp (Phase 3 merge event path + cache miss/hit event recording)

key-decisions:
  - "Phase 2 dev_events[i].wait() intentionally kept: ensures kernel output readable before merge copies read it; removing it is a Rule 4 architectural change"
  - "OOO flush (ooo_stream(i)->wait()) at end of mul_mat preserved: releases L0 event handles and ensures all OOO-submitted merge copies complete before next matmul"
  - "Phase 3 event path uses src_ooo (source device OOO queue) for GPU->staging, dst_ooo for staging->GPU_dst; for dst_is_host both use src_ooo since host memory is accessible from any queue"
  - "sycl::event default-constructed in struct initializer is valid/safe; invalidate_row_src1_cache() zero-initializes via row_src1_cache[i] = {} which sets has_event = false without change"

patterns-established:
  - "Phase 3 merge event path pattern: sycl::event e_prev; bool has_prev = false; first column chains depends_on(dev_events[i]); subsequent chains depends_on(e_prev)"
  - "Cache event pattern: ext_oneapi_submit_barrier() on miss, ext_oneapi_submit_barrier({cached}) on hit — both on the device OOO stream"

requirements-completed: [SYNC-01, SYNC-02]

duration: 20min
completed: 2026-03-18
---

# Phase 3 Plan 01: Complete Event Path Summary

**Phase 3 merge converted to OOO event-based copies (depends_on kernel completion events); src1 cache now propagates completion event on hit — closing the sync gap that left zero-event chaining for consecutive attn_q/k/v matmuls**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-03-18T03:56:00Z
- **Completed:** 2026-03-18T04:15:23Z
- **Tasks:** 2
- **Files modified:** 3 (common.hpp + ggml-sycl.cpp via gitignored llama.cpp/, bench test)

## Accomplishments

- SYNC-02: `row_src1_cache_entry` gains `last_copy_event` (sycl::event) and `has_event` (bool); cache miss records `ext_oneapi_submit_barrier()` after copy+quantize; cache hit chains `depends_on(last_copy_event)` via barrier submit — closes sync gap for consecutive matmuls sharing src1
- SYNC-01: Phase 3 merge block now branches on `use_event_sync`; event path submits GPU_i->staging + staging->dst as OOO queue operations with `depends_on(dev_events[i])` on first column and staging-buffer reuse (`e_prev_merge_staging`) for subsequent columns — zero host waits inside Phase 3
- Build verified: `cmake --build . --target llama-server -j$(nproc)` exits 0 with only warnings (no errors)
- Bench tests added to test_row_split.py: `sync01_event_merge_q4km_np1_100tok`, `sync02_cache_event_q4km_np1_100tok`, `complete_event_path_glm47_np1_500tok` (stalls/token <= 4 gate)

## Task Commits

1. **Task 1: SYNC-02 — src1 cache event propagation** (C++ in gitignored llama.cpp/, build verified)
2. **Task 2: SYNC-01 — Phase 3 event-based merge** (C++ in gitignored llama.cpp/, build verified)
3. **Combined bench test commit** - `147c9e8` (feat: SYNC-01+02 bench tests added to test_row_split.py)

Note: `llama.cpp/` is in `.gitignore` (external build dependency). C++ changes are local modifications to the SYCL build; bench test file `scripts/bench/tests/test_row_split.py` is tracked and serves as the living documentation per CLAUDE.md convention.

## Files Created/Modified

- `llama.cpp/ggml/src/ggml-sycl/common.hpp` — Added `last_copy_event` (sycl::event) and `has_event` (bool) to `row_src1_cache_entry` struct
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` — Phase 3 merge: event path with `depends_on(dev_events[i])` + staging reuse; src1 cache: event store on miss, barrier chain on hit
- `scripts/bench/tests/test_row_split.py` — Three new test methods: SYNC-01 standalone, SYNC-02 standalone, complete event path stall gate

## Decisions Made

- Phase 2 `dev_events[i].wait()` intentionally kept: it ensures kernel output is readable before merge copies read it; removing it would be a Rule 4 architectural change (potential DEVICE_LOST)
- OOO flush at end of `mul_mat` preserved: `ooo_stream(i)->wait()` releases L0 event handles and ensures all OOO-submitted merge copies complete; without it, events accumulate and exhaust the L0 pool
- Phase 3 event path: `src_ooo = ctx.ooo_stream(i)` for GPU->staging; `dst_ooo = ctx.ooo_stream(ctx.device)` for staging->GPU_dst; if `dst_is_host` both are `src_ooo` since host memory is accessible from any queue
- `sycl::event` default constructor is valid (creates a trivially-satisfied event); `invalidate_row_src1_cache()` already zero-initializes via `row_src1_cache[i] = {}`, which sets `has_event = false` — no change needed to invalidation logic

## Deviations from Plan

None — plan executed exactly as written. The `sycl::event` initializer note in the plan was confirmed accurate: default construction is safe. Phase 2 wait loop and OOO flush were both preserved as specified.

## Issues Encountered

- `llama.cpp/` is in `.gitignore` at the repo root level — C++ source changes cannot be directly committed. This matches prior phase behavior (Phase 01/02 C++ changes were also in gitignored llama.cpp/). Bench test file serves as the committed artifact and living documentation per CLAUDE.md convention.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Complete event path implemented and built successfully
- Runtime test gate: `python3 -m bench rowsplit.complete_event_path_glm47_np1_500tok` — validates stalls/token <= 4 with GGML_SYCL_ROW_EVENTS=1
- If stalls/token still > 4 after running: investigate whether Phase 2 `dev_events[i].wait()` can be converted to event-based (currently intentional host wait)
- np > 1 stability crash remains a separate issue (pre-existing blocker in STATE.md)

## Self-Check: PASSED

- FOUND: `.planning/phases/03-complete-event-path/03-01-SUMMARY.md`
- FOUND: commit `147c9e8` (feat(03-01): SYNC-01+02)
- FOUND: `last_copy_event` field in `common.hpp` (2 occurrences)
- FOUND: `SYNC-01` marker in `ggml-sycl.cpp`
- FOUND: `SYNC-02` marker in `ggml-sycl.cpp` (2 occurrences)
- FOUND: `e_prev_merge_staging` in `ggml-sycl.cpp` (5 occurrences)
- FOUND: `dev2dev_memcpy_staged_sync` still present (legacy path intact)
- FOUND: `ooo_stream(i)->wait()` OOO flush still present

---
*Phase: 03-complete-event-path*
*Completed: 2026-03-18*
