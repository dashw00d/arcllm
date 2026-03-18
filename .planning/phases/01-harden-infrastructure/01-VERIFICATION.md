---
phase: 01-harden-infrastructure
verified: 2026-03-18T03:25:00Z
status: human_needed
score: 3/3 must-haves verified (automated); 1 item needs hardware run
re_verification: false
human_verification:
  - test: "Run `cd /home/ryan/llm-stack/scripts && python3 -m bench infra.row_events_correct_env`"
    expected: "Server starts cleanly with ROW_EVENTS=1 + IMMEDIATE_COMMANDLISTS=1, completes 1 inference request (completed=1, no error), confirming INFRA-02 does not false-abort on current dpct queue construction"
    why_human: "Requires live 3x Arc A770 hardware and the built llama-server binary with the 0.6b-q8 model present — cannot verify SYCL context equality assertion behavior without executing on actual Level Zero devices"
  - test: "Run `cd /home/ryan/llm-stack/scripts && python3 -m bench infra.row_events_no_cmdlist`"
    expected: "Server exits non-zero within ~10 seconds, BenchResult.error contains 'server failed', /tmp/bench_logs/infra_row_events_no_cmdlist.log contains 'GGML_SYCL_ROW_EVENTS=1 requires'"
    why_human: "Requires live hardware to confirm the abort fires fast (within seconds) and that the log file is written before process exit — timing behavior cannot be verified statically"
---

# Phase 1: Harden Infrastructure Verification Report

**Phase Goal:** The OOO queue infrastructure is provably correct — misconfigured environments crash loudly instead of silently producing wrong results
**Verified:** 2026-03-18T03:25:00Z
**Status:** human_needed (all automated checks passed; 2 hardware runs pending)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Starting llama-server with `GGML_SYCL_ROW_EVENTS=1` but `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` unset or !=1 aborts within seconds with a clear error message | VERIFIED | `ggml-sycl.cpp:295-309` — `if (g_ggml_sycl_row_events)` block with `getenv("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS")`, check `cmdlist_val == nullptr \|\| cmdlist_val[0] != '1'`, `GGML_LOG_ERROR` + `abort()` |
| 2 | Starting llama-server with an OOO queue constructed from a different SYCL context than the in-order streams aborts with a clear error message | VERIFIED | `common.hpp:399-413` — `if (ooo_ctx != io_ctx)` inside `ooo_stream()` lazy-init path, `GGML_LOG_ERROR("%s: device %d: OOO queue context != in-order stream context.\n", ...)` + `abort()` |
| 3 | Both assertions trigger before any matmul is dispatched (startup, not first inference) | VERIFIED | INFRA-01 is inside `ggml_check_sycl()` static-init guard (fires at backend load, before model weights are read); INFRA-02 fires on first `ooo_stream(device)` call per device, which precedes all matmul dispatches |

**Score:** 3/3 truths verified (code-level); hardware confirmation pending for truths 1 and 2

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` | INFRA-01 cmdlist assertion in `ggml_check_sycl()` init block | VERIFIED | Lines 292-309: `// INFRA-01` comment, `if (g_ggml_sycl_row_events)` block, `getenv()` check, `GGML_LOG_ERROR` + `abort()`. Contains required string "GGML_SYCL_ROW_EVENTS=1 requires". Placed after line 290 (last `get_sycl_env`) and before line 317 (`GGML_SYCL_DEBUG`). |
| `llama.cpp/ggml/src/ggml-sycl/common.hpp` | INFRA-02 context mismatch assertion in `ooo_stream()` accessor | VERIFIED | Lines 394-413: `// INFRA-02` comment, `if (g_ggml_sycl_row_events)` guard, `ooo_ctx != io_ctx` check, error message with `%s: device %d:` format, `abort()`. `extern int g_ggml_sycl_row_events;` declared at line 85 alongside other extern globals. |
| `scripts/bench/tests/test_infra.py` | Negative test (abort on bad env) and positive test (clean startup) | VERIFIED | `class TestInfra(BenchTest)` confirmed. `test_row_events_no_cmdlist` with `row_events=True, immediate_cmdlists=False`, asserts `result.error == "server failed"` and log contains "GGML_SYCL_ROW_EVENTS=1 requires". `test_row_events_correct_env` with `row_events=True, immediate_cmdlists=True`, asserts `result.completed > 0` and no error. Auto-discovered by bench framework as suite `infra`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ggml-sycl.cpp` | `g_ggml_sycl_row_events` global | `if (g_ggml_sycl_row_events)` check immediately after line 287-290 env reads | VERIFIED | `g_ggml_sycl_row_events` defined at line 96, read at line 287, assertion block at lines 295-309. Pattern `g_ggml_sycl_row_events` + `getenv("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS")` confirmed present. |
| `common.hpp` | `ooo_qptrs[device]` lazy init | `ooo_ctx != io_ctx` check on first OOO queue construction | VERIFIED | Lines 399-413 inside `if (ooo_qptrs[device] == nullptr)` branch. Context equality check fires before first `return ooo_qptrs[device]`. |
| `test_infra.py` | llama-server binary | subprocess start with bad env, `result.error == "server failed"`, log read | VERIFIED | Line 83: `immediate_cmdlists=False`. Line 93: `assert "server failed" in (result.error or "")`. Line 97: `assert "GGML_SYCL_ROW_EVENTS=1 requires" in log_text`. `LOG_DIR` imported from `bench.runner`. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| INFRA-01 | 01-01-PLAN.md | System asserts `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` in C++ when `GGML_SYCL_ROW_EVENTS=1` is enabled | SATISFIED | `ggml-sycl.cpp:295-309`: assertion inside `ggml_check_sycl()` fires at backend init; `test_infra.py` negative test validates the abort path |
| INFRA-02 | 01-01-PLAN.md | System verifies OOO queue context matches in-order stream context at startup, crashes loudly if mismatched | SATISFIED | `common.hpp:394-413`: assertion inside `ooo_stream()` lazy-init; `extern int g_ggml_sycl_row_events` declared at line 85; `test_infra.py` positive test implicitly validates the pass path |

No orphaned requirements: REQUIREMENTS.md traceability table maps only INFRA-01 and INFRA-02 to Phase 1. Both are claimed in 01-01-PLAN.md and both have implementation evidence.

### Anti-Patterns Found

None detected in any of the three modified files.

Scanned `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` (INFRA-01 block), `llama.cpp/ggml/src/ggml-sycl/common.hpp` (INFRA-02 block + extern), and `scripts/bench/tests/test_infra.py` (full file, 127 lines). No TODOs, FIXMEs, placeholder returns, or stub implementations found. Both C++ assertions use `GGML_LOG_ERROR` + `abort()` — no silent fallback.

### Human Verification Required

#### 1. INFRA-01 Negative Path (hardware)

**Test:** `cd /home/ryan/llm-stack/scripts && python3 -m bench infra.row_events_no_cmdlist`
**Expected:** Test passes — server exits non-zero within ~10 seconds; `BenchResult.error` contains `"server failed"`; `/tmp/bench_logs/infra_row_events_no_cmdlist.log` contains `"GGML_SYCL_ROW_EVENTS=1 requires"`.
**Why human:** Requires live Arc A770 hardware and the built `llama-server` binary. The static check confirms the abort path is coded correctly; hardware confirms the binary was actually built with the assertions and that startup timing is within seconds (not minutes).

#### 2. INFRA-02 Positive Path (hardware)

**Test:** `cd /home/ryan/llm-stack/scripts && python3 -m bench infra.row_events_correct_env`
**Expected:** Test passes — server starts cleanly, completes 1 inference request (`completed=1`, no error), confirming INFRA-02 assertion does not false-abort with current dpct OOO queue construction.
**Why human:** Requires live hardware to execute `ooo_stream()` and trigger the SYCL context equality check against actual Level Zero device queues. Cannot verify `sycl::context operator==` behavior without running on real hardware.

### Gaps Summary

No gaps. All automated checks passed:
- INFRA-01 assertion exists, is substantive, is in the correct location (`ggml_check_sycl()` after env reads, before debug logging), and uses the required `getenv()` + check + `GGML_LOG_ERROR` + `abort()` pattern.
- INFRA-02 assertion exists, is substantive, is in the correct location (`ooo_stream()` lazy-init branch), has the `extern int g_ggml_sycl_row_events` declaration needed for it to compile, and uses the required context comparison + `GGML_LOG_ERROR` + `abort()` pattern.
- `test_infra.py` exists, is substantive (127 lines, full docstring per CLAUDE.md "tests are documentation" rule), auto-discovered by bench framework as suite `infra` with both tests listed, and correctly wires to `bench.base.BenchTest`, `bench.config.BenchConfig`, and `bench.runner.LOG_DIR`.
- Both INFRA-01 and INFRA-02 are accounted for in REQUIREMENTS.md traceability. No orphaned requirements.

Two hardware runs are needed to confirm timing behavior (INFRA-01: abort within seconds) and SYCL context equality behavior (INFRA-02: no false abort). These cannot be verified statically.

---

_Verified: 2026-03-18T03:25:00Z_
_Verifier: Claude (gsd-verifier)_
