---
phase: 01-harden-infrastructure
plan: 01
subsystem: infra
tags: [sycl, llama-cpp, intel-arc, level-zero, startup-assertions, bench-framework]

# Dependency graph
requires: []
provides:
  - INFRA-01: SYCL abort on ROW_EVENTS=1 without IMMEDIATE_COMMANDLISTS=1
  - INFRA-02: SYCL abort on OOO queue context mismatch with in-order stream
  - bench test suite 'infra' with negative and positive assertion validation
affects:
  - 01-02 (next plan in phase, if any)
  - All future row-split work — assertions protect against env misconfiguration

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GGML_LOG_ERROR + abort() for fatal SYCL env misconfigurations — consistent with existing GGML_ASSERT pattern"
    - "Negative bench test pattern: run with bad env, assert result.error == 'server failed', read log file to confirm abort message"

key-files:
  created:
    - scripts/bench/tests/test_infra.py
  modified:
    - llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp (gitignored — modified in checkout)
    - llama.cpp/ggml/src/ggml-sycl/common.hpp (gitignored — modified in checkout)

key-decisions:
  - "INFRA-01 placed in ggml_check_sycl() (not ooo_stream()) so it fires at backend init, before model loading, ensuring sub-5-second abort"
  - "INFRA-02 placed in ooo_stream() lazy-init path — fires before first matmul without needing a backend context in ggml_check_sycl()"
  - "g_ggml_sycl_row_events declared extern in common.hpp to expose it for the INFRA-02 check"
  - "Use getenv() directly for SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS — it is a system L0 flag, not a GGML flag, so no g_ global needed"

patterns-established:
  - "Pattern: negative bench test — start server with bad env, assert result.error contains 'server failed', read /tmp/bench_logs/{name}.log for abort string"

requirements-completed:
  - INFRA-01
  - INFRA-02

# Metrics
duration: 6min
completed: 2026-03-18
---

# Phase 1 Plan 1: Harden Infrastructure — Startup Assertions Summary

**Two C++ abort() guards in ggml_check_sycl() and ooo_stream() that catch ROW_EVENTS misconfiguration within seconds of startup, plus a bench test suite validating both paths**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-18T03:13:37Z
- **Completed:** 2026-03-18T03:20:20Z
- **Tasks:** 2
- **Files modified:** 3 (2 C++ in gitignored llama.cpp checkout, 1 Python committed)

## Accomplishments

- INFRA-01: `ggml_check_sycl()` now aborts within seconds if ROW_EVENTS=1 is set without IMMEDIATE_COMMANDLISTS=1, preventing indefinite hangs from unflushed command batches
- INFRA-02: `ooo_stream()` now aborts on first OOO queue construction if the queue's SYCL context differs from the in-order stream's context, preventing silent correctness failures from cross-context event dependencies
- Created `scripts/bench/tests/test_infra.py` with two auto-discovered tests: negative test (abort on bad env) and positive test (clean startup with correct env)

## Task Commits

1. **Task 1: INFRA-01 and INFRA-02 startup assertions** — C++ changes in gitignored `llama.cpp/` checkout (no git commit; verified via `cmake --build` exit 0)
2. **Task 2: test_infra.py bench test** — `6fd7aae` (feat)

**Plan metadata:** (created in this summary commit)

## Files Created/Modified

- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` — Added INFRA-01 assert block after line 290 (inside `ggml_check_sycl()` init guard)
- `llama.cpp/ggml/src/ggml-sycl/common.hpp` — Added `extern int g_ggml_sycl_row_events;` declaration; added INFRA-02 assert block inside `ooo_stream()` lazy-init branch
- `scripts/bench/tests/test_infra.py` — New bench test suite; `python3 -m bench help` shows `infra` suite with two tests

## Decisions Made

- INFRA-01 placed in `ggml_check_sycl()` (not in `ooo_stream()`) so it fires at backend init before model loading — abort within seconds of startup, not minutes
- INFRA-02 placed in `ooo_stream()` lazy-init branch — fires before any matmul without requiring a `ggml_backend_sycl_context` instance in `ggml_check_sycl()`, which is not available there
- `extern int g_ggml_sycl_row_events` added to the existing extern block in `common.hpp` alongside other global declarations for the same pattern
- `getenv()` used directly (not `get_sycl_env()`) for `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` — it is a Level Zero system flag, not a GGML env var, so no persistent global is needed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added extern declaration for g_ggml_sycl_row_events in common.hpp**
- **Found during:** Task 1 (INFRA-01 and INFRA-02 assertions)
- **Issue:** `g_ggml_sycl_row_events` was defined in `ggml-sycl.cpp` but not declared `extern` in `common.hpp`, which is needed for the INFRA-02 check inside `ooo_stream()`
- **Fix:** Added `extern int g_ggml_sycl_row_events;` to the existing extern block in `common.hpp:84-85`
- **Files modified:** `llama.cpp/ggml/src/ggml-sycl/common.hpp`
- **Verification:** Build completed without linker or compiler errors
- **Committed in:** (part of C++ changes in gitignored checkout; build verified)

---

**Total deviations:** 1 auto-fixed (missing critical — extern declaration needed for assertion to compile)
**Impact on plan:** Required for correctness — INFRA-02 check cannot reference `g_ggml_sycl_row_events` without this declaration.

## Issues Encountered

- `llama.cpp/` directory is listed in `.gitignore` — C++ changes cannot be committed to this repo. They exist in the local checkout and are verified by `cmake --build` exit 0. The bench test (Task 2) is the committable artifact.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Both assertions compile and are active in the current `llama.cpp/build-sycl/` binary
- `python3 -m bench infra.row_events_no_cmdlist` will validate INFRA-01 (negative path — expect abort)
- `python3 -m bench infra.row_events_correct_env` will validate INFRA-01/02 positive path
- Phase gate: run `python3 -m bench infra` on hardware before `/gsd:verify-work`

---
*Phase: 01-harden-infrastructure*
*Completed: 2026-03-18*
