# Phase 1: Harden Infrastructure - Research

**Researched:** 2026-03-17
**Domain:** SYCL backend C++ — startup assertions in ggml-sycl.cpp / common.hpp
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Assertion placement**
- Eager, at backend init — not lazy at first `ooo_stream()` call or first matmul
- Both checks must fire before any GPU work is dispatched
- If the row-events env var is set, validate prerequisites immediately during the SYCL init path (where `g_ggml_sycl_row_events` is already read at line 287)

**Failure behavior**
- `GGML_LOG_ERROR` with clear message explaining what's wrong and how to fix it, then `abort()`
- Do NOT gracefully disable ROW_EVENTS — if the user asked for it and the env is broken, crash loudly
- Do NOT throw SYCL exceptions — use the same logging pattern as existing GGML_LOG calls
- Priority: fail within seconds of startup, not minutes into a generation

**Cmdlist detection**
- Read `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` env var directly via `getenv()`
- Matches existing pattern: `g_ggml_sycl_row_events` uses `get_sycl_env()` which wraps `getenv()`
- No need for L0 runtime query — the env var is what controls the behavior

### Claude's Discretion
- Exact assertion messages (as long as they say what's wrong and how to fix it)
- Whether to add the context check to `ooo_stream()` accessor or to a separate init function
- Order of assertions (cmdlist first or context first — doesn't matter)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-01 | System asserts `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` in C++ when `GGML_SYCL_ROW_EVENTS=1` is enabled | Insertion point identified: after `g_ggml_sycl_row_events` is read at ggml-sycl.cpp:287, inside `ggml_check_sycl()` init block. Pattern: `getenv()` + `GGML_LOG_ERROR` + `abort()`. |
| INFRA-02 | System verifies OOO queue context matches in-order stream context at startup, crashes loudly if mismatched | Insertion point identified: inside `ooo_stream()` accessor in common.hpp:389-394 OR in `ggml_backend_sycl_init()` after context construction. `queue::get_context()` comparison is the check. |
</phase_requirements>

---

## Summary

Phase 1 is two C++ assertions added to the SYCL backend init path. The work is minimal in lines of code but critical in safety: both assertions must fire before any matmul is dispatched, catching misconfigured environments that would otherwise produce silent hangs or silent wrong output 5+ minutes into inference.

The key insertion points are already known from the CONTEXT.md canonical references. `ggml_check_sycl()` (ggml-sycl.cpp:247) is the once-guarded init function that reads all env vars and is called from every public SYCL backend API entry point, including `ggml_backend_sycl_init()`. The `g_ggml_sycl_row_events` variable is set at line 287 inside that function — the cmdlist assertion belongs immediately after it. The context mismatch assertion belongs either inside `ooo_stream()` (lazy but called before first matmul) or in a dedicated validation function called from `ggml_check_sycl()` after the env vars are read.

The bench framework already has `ROW_EVENTS` and `immediate_cmdlists` config levers. A new test in `test_row_split.py` (or a dedicated `test_infra.py`) that starts the server without `immediate_cmdlists=True` should observe a clean abort within a few seconds. This is the validation artifact for both requirements.

**Primary recommendation:** Add both assertions inside `ggml_check_sycl()`'s init block, immediately after line 287. The context check requires constructing a test OOO queue for each device during init — if dpct's device-default queues share context (which `test_dpct_events.cpp` confirms they do), the assertion will always pass; it exists to catch future regressions if the queue construction is ever changed to use a different context.

---

## Standard Stack

### Core
| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| `ggml-sycl.cpp` — `ggml_check_sycl()` | project fork | Static-init guarded function, called from all SYCL entry points | Already the established init path; all env vars and device checks happen here |
| `common.hpp` — `ggml_backend_sycl_context` | project fork | Owns `ooo_qptrs[]` and `ooo_stream()` accessor | `ooo_stream()` is where OOO queues are first instantiated |
| `GGML_LOG_ERROR` macro | ggml.h | Structured error logging to stderr | Used throughout ggml backend; consistent with all other error paths |
| `abort()` | C stdlib | Hard process termination after fatal config error | Consistent with `GGML_ASSERT` pattern (which calls abort on failure) |
| `getenv()` | C stdlib | Read `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` value | Already used by `get_sycl_env()` which wraps it; direct `getenv()` is fine for a non-GGML env var |
| `sycl::queue::get_context()` | SYCL 2020 | Compare two queue contexts for identity | Standard SYCL API, `sycl::context` has `operator==` |

### No New Dependencies
This phase requires zero new libraries. All tools are already used in the file being modified.

---

## Architecture Patterns

### Existing Pattern: `ggml_check_sycl()` Init Block

The function is called at the top of every public API entry. It has a `static bool initialized` guard so assertions run exactly once per process:

```cpp
// ggml-sycl.cpp:247 (existing pattern)
static void ggml_check_sycl() try {
    static bool initialized = false;

    if (!initialized) {
        // ... all env var reads including g_ggml_sycl_row_events at line 287 ...
        // INSERT ASSERTIONS HERE (after g_ggml_sycl_row_events is read)

        initialized = true;
        g_sycl_loaded = true;
        ggml_backend_sycl_print_sycl_devices();
    }
}
```

### Pattern 1: INFRA-01 — Cmdlist Assertion

**What:** After `g_ggml_sycl_row_events` is set (line 287), check `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` is `"1"` if row events are enabled.

**Where:** `ggml_check_sycl()` init block, after line 287.

**Example:**
```cpp
// Source: existing getenv() pattern in get_sycl_env() at ggml-sycl.cpp:233-245

if (g_ggml_sycl_row_events) {
    const char * cmdlist_val = getenv("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS");
    if (cmdlist_val == nullptr || cmdlist_val[0] != '1') {
        GGML_LOG_ERROR(
            "GGML_SYCL_ROW_EVENTS=1 requires SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1.\n"
            "Without immediate command lists, cross-queue event dependencies never fire and\n"
            "inference will hang indefinitely. Set the env var before starting the server.\n"
        );
        abort();
    }
}
```

**Why `getenv()` directly:** `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` is not a GGML env var so there is no existing `g_` global for it. Using `getenv()` directly is the right call rather than adding a permanent global for a one-time check.

### Pattern 2: INFRA-02 — Context Mismatch Assertion

**What:** After the env-var block, iterate over all devices, call `ooo_stream(i)` to instantiate each OOO queue, and compare its context to the corresponding in-order stream's context.

**Where:** Option A — end of `ggml_check_sycl()` init block (requires access to a backend context). Option B — inside `ooo_stream()` on first construction. Option C — a dedicated helper called once from `ggml_check_sycl()`.

**Recommended: Option C — dedicated helper.** This keeps `ggml_check_sycl()` the single authoritative location for startup validation. The helper can construct the contexts it needs locally.

However, there is a constraint: `ggml_check_sycl()` does not have access to a `ggml_backend_sycl_context` instance (that's created later in `ggml_backend_sycl_init()`). The `ooo_stream()` accessor lives on `ggml_backend_sycl_context`, not as a free function.

**Resolution:** Do the context check inside `ooo_stream()` itself on first call, guarded by `g_ggml_sycl_row_events`. This is "lazy-but-early" — `ooo_stream()` is first called from inside the row-split matmul path, before any GPU work is dispatched. This satisfies the requirement "before any matmul is dispatched."

Alternatively, call `ooo_stream()` explicitly during `ggml_backend_sycl_init()` after the context is constructed, purely to trigger the check. This is cleaner but requires knowing which devices are in use.

**Simplest correct approach:** Add the check to `ooo_stream()` accessor itself, firing when `ooo_qptrs[device] == nullptr` (first call per device):

```cpp
// Source: common.hpp:389-394 (existing accessor)

queue_ptr ooo_stream(int device) {
    if (ooo_qptrs[device] == nullptr) {
        ooo_qptrs[device] = &(dpct::get_device(device).out_of_order_queue());

        // INFRA-02: Verify OOO queue shares context with in-order stream.
        // Cross-context event dependencies silently fail on L0 (Pitfall 6).
        if (g_ggml_sycl_row_events) {
            sycl::context ooo_ctx = ooo_qptrs[device]->get_context();
            sycl::context io_ctx  = stream(device, 0)->get_context();
            if (ooo_ctx != io_ctx) {
                GGML_LOG_ERROR(
                    "Device %d: OOO queue context does not match in-order stream context.\n"
                    "Cross-queue event dependencies (depends_on) will silently fail.\n"
                    "This is a bug in OOO queue construction — the queues must share a SYCL context.\n",
                    device
                );
                abort();
            }
        }
    }
    return ooo_qptrs[device];
}
```

**Note on expected behavior:** `test_dpct_events.cpp` verifies `same_ctx: YES` for dpct-managed queues on this hardware. The assertion is expected to **always pass** given the current dpct-based construction. Its value is as a regression guard: if someone later changes `ooo_stream()` to use a different context constructor, the abort fires immediately at the first matmul instead of silently corrupting inference.

### Anti-Patterns to Avoid

- **Graceful fallback:** Do NOT fall back to host-sync if cmdlist is batched. The user explicitly enabled ROW_EVENTS; silently undoing it masks the config error.
- **Lazy check at first inference:** Do NOT defer assertions to first matmul dispatch — the requirement is startup, not first inference (the user wants sub-5-second failure, not 20 minutes in).
- **SYCL exception throw:** Do NOT `throw sycl::exception(...)`. The GGML backend uses `GGML_LOG_ERROR + abort()` consistently; exceptions would be uncaught.
- **L0 runtime API query:** Do NOT call L0 APIs to detect command list mode at runtime. `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` controls behavior at queue creation time; the env var is the ground truth.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Logging the error | Custom fprintf/cerr | `GGML_LOG_ERROR` | Consistent formatting, respects debug level, already used everywhere in ggml-sycl |
| Context equality test | Manual pointer comparison | `sycl::context::operator==` | SYCL contexts have identity comparison; pointer comparison on wrapper objects is wrong |
| Reading SYCL env vars | Custom parser | `getenv()` directly | The env var is a system-level L0 flag, not a GGML flag; `getenv()` is correct here |

---

## Common Pitfalls

### Pitfall 1: Cmdlist Check Triggers Too Late
**What goes wrong:** If the assertion is placed inside `ooo_stream()` or the matmul path rather than `ggml_check_sycl()`, it fires 5+ minutes into server startup (after model load).
**Why it happens:** `ggml_check_sycl()` runs at first call to any SYCL backend function (early), but `ooo_stream()` is only called from inside the row-split matmul loop.
**How to avoid:** INFRA-01 (cmdlist check) belongs in `ggml_check_sycl()`. INFRA-02 (context check) in `ooo_stream()` is acceptable because it fires before the first matmul — but if both can go in `ggml_check_sycl()`, that is better.
**Warning signs:** Abort fires after "loading model weights" log line instead of before it.

### Pitfall 2: Context Check Has No Backend Context Available
**What goes wrong:** `ggml_check_sycl()` does not have a `ggml_backend_sycl_context*` in scope, so calling `ctx.ooo_stream(i)` from there requires instantiating a temporary context.
**Why it happens:** `ggml_backend_sycl_context` is constructed in `ggml_backend_sycl_init()` (line 12780), which is called after `ggml_check_sycl()` returns.
**How to avoid:** Place INFRA-02 check inside `ooo_stream()` itself (on first construction) rather than in `ggml_check_sycl()`. The CONTEXT.md already notes this as a discretion area.

### Pitfall 3: `getenv()` Returns "0" But Check Treats It as Set
**What goes wrong:** `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` is technically set but should trigger the assertion.
**Why it happens:** `cmdlist_val != nullptr` would be true even when the value is "0".
**How to avoid:** Check `cmdlist_val[0] != '1'` (the value must be exactly "1"), not just `cmdlist_val != nullptr`. The example code above already does this correctly.

### Pitfall 4: `sycl::context::operator==` Not Available in Older DPC++
**What goes wrong:** Some older DPC++ versions may not have context equality comparison.
**Why it happens:** SYCL 2020 spec includes context equality; pre-2020 implementations vary.
**How to avoid:** The installed compiler is oneAPI 2025.3.2 — SYCL 2020 is fully supported. No compatibility concern here.
**Confidence:** HIGH — verified against installed compiler version in STACK.md.

---

## Code Examples

### INFRA-01: Full Cmdlist Assertion (in `ggml_check_sycl()`)
```cpp
// Placement: ggml-sycl.cpp, inside the `if (!initialized)` block,
// immediately after line 287 where g_ggml_sycl_row_events is read.
//
// Source: existing getenv()/GGML_LOG_ERROR/abort() pattern in this file.

if (g_ggml_sycl_row_events) {
    const char * cmdlist_val = getenv("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS");
    if (cmdlist_val == nullptr || cmdlist_val[0] != '1') {
        GGML_LOG_ERROR(
            "%s: GGML_SYCL_ROW_EVENTS=1 requires "
            "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1.\n"
            "  Without immediate command lists, cross-queue event dependencies\n"
            "  never fire because command batches are not flushed when another\n"
            "  queue polls for the event. Inference will hang indefinitely.\n"
            "  Fix: export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1\n",
            __func__
        );
        abort();
    }
}
```

### INFRA-02: Full Context Assertion (in `ooo_stream()`)
```cpp
// Placement: common.hpp, inside ooo_stream() after ooo_qptrs[device] is set.
// Source: PITFALLS.md Pitfall 6 and STACK.md Queue Infrastructure section.

queue_ptr ooo_stream(int device) {
    if (ooo_qptrs[device] == nullptr) {
        ooo_qptrs[device] = &(dpct::get_device(device).out_of_order_queue());

        if (g_ggml_sycl_row_events) {
            const sycl::context ooo_ctx = ooo_qptrs[device]->get_context();
            const sycl::context io_ctx  = stream(device, 0)->get_context();
            if (ooo_ctx != io_ctx) {
                GGML_LOG_ERROR(
                    "%s: device %d: OOO queue context != in-order stream context.\n"
                    "  Cross-device event dependencies (handler::depends_on) will\n"
                    "  silently fail or be ignored when queues come from different\n"
                    "  SYCL contexts. All queues must share a context.\n"
                    "  This is likely a bug in ooo_stream() construction.\n",
                    __func__, device
                );
                abort();
            }
        }
    }
    return ooo_qptrs[device];
}
```

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `scripts/bench/` — custom BenchTest framework (Python, auto-discovers test_*.py) |
| Config file | `scripts/bench/config.py` — BenchConfig dataclass |
| Quick run command | `cd /home/ryan/llm-stack/scripts && python3 -m bench row_split.events_q4km_np1_100tok` |
| Full suite command | `cd /home/ryan/llm-stack/scripts && python3 -m bench row_split` |

**Note on test framework:** CLAUDE.md mandates use of `python3 -m bench` for all GPU/server testing. Do not use raw bash/curl to test assertions. A test that deliberately misconfigures the environment and confirms a clean abort within a few seconds is the correct validation pattern.

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFRA-01 | `GGML_SYCL_ROW_EVENTS=1` without `IMMEDIATE_COMMANDLISTS=1` aborts with clear message | smoke (negative test — expect abort) | `python3 -m bench infra.row_events_no_cmdlist` | ❌ Wave 0 |
| INFRA-02 | Context mismatch on OOO queue construction aborts with clear message | smoke (assertion path covered by positive test) | `python3 -m bench infra.row_events_correct_env` | ❌ Wave 0 |

**INFRA-01 validation approach:** The bench framework starts llama-server as a subprocess and checks its exit. A test with `row_events=True, immediate_cmdlists=False` should observe the process exit non-zero with the error message in stderr within ~10 seconds of startup. The bench runner already captures stderr — the test just needs to assert on the exit code and message.

**INFRA-02 validation approach:** INFRA-02 asserts on a condition that the current dpct queue construction always satisfies (same context). The positive path test — `row_events=True, immediate_cmdlists=True`, server starts cleanly and runs inference — implicitly validates INFRA-02 passes. A dedicated INFRA-02 failure test would require constructing OOO queues from a different context, which is not practical in the bench framework. The positive-path inference test is sufficient.

### Sampling Rate
- **Per task commit:** `python3 -m bench infra` (new suite, quick abort test)
- **Per wave merge:** `python3 -m bench row_split` (full row-split suite)
- **Phase gate:** Both INFRA-01 negative test and `row_split.events_q4km_np1_100tok` pass before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `scripts/bench/tests/test_infra.py` — covers INFRA-01 (negative test: abort on bad env), INFRA-02 (positive test: clean startup with correct env)

*(No framework install needed — bench framework already present and used by existing tests.)*

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Silent hang on batched cmdlist + event sync | Abort with clear message at startup | Phase 1 (this phase) | Saves 5+ minutes of debugging per misconfiguration |
| Silent wrong results on context mismatch | Abort with clear message before first matmul | Phase 1 (this phase) | Prevents hours of debugging ghost correctness failures |
| Trust user to set correct env vars | Verify env vars at startup and crash loudly if wrong | Phase 1 (this phase) | Core fail-fast principle |

---

## Open Questions

1. **Best placement for INFRA-02 context check**
   - What we know: `ggml_check_sycl()` has no `ggml_backend_sycl_context` instance in scope; `ooo_stream()` is the natural place
   - What's unclear: How early is `ooo_stream()` first called? If it's only called inside the row-split matmul loop, the check fires at first inference, not at model load. This still satisfies "before any matmul is dispatched" — but does it satisfy the user's "fail within seconds" intent?
   - **Recommendation:** Acceptable in `ooo_stream()` — it fires at the first matmul call, which is well before any generation output. If stricter "at model load" timing is desired, call `ooo_stream(i)` explicitly inside `ggml_backend_sycl_init()` for each device when `g_ggml_sycl_row_events` is set, purely to trigger the check.

2. **Bench test for INFRA-01 negative path**
   - What we know: The bench runner starts llama-server as a subprocess and can inspect exit code and stderr
   - What's unclear: Does the current `BenchRunner` have an "expect abort" mode, or does a non-zero exit always count as a test failure?
   - **Recommendation:** Check `runner.py` during Wave 0. If no "expect-failure" mode exists, the test can be a simple subprocess call that asserts exit code != 0 and checks stderr for the error string, outside the normal `BenchTest.run()` path.

---

## Sources

### Primary (HIGH confidence)
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:233-408` — `get_sycl_env()`, `ggml_check_sycl()` full init block; read directly
- `llama.cpp/ggml/src/ggml-sycl/common.hpp:364-394` — `ggml_backend_sycl_context`, `ooo_stream()` accessor; read directly
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:12774-12794` — `ggml_backend_sycl_init()`; read directly
- `.planning/research/PITFALLS.md` — Pitfall 1 (batched cmdlist deadlock), Pitfall 6 (cross-context event silence); read directly
- `.planning/research/STACK.md` — Immediate command list requirement, context semantics, `ooo_stream()` dpct behavior; read directly
- `.planning/phases/01-harden-infrastructure/01-CONTEXT.md` — All locked decisions; read directly

### Secondary (MEDIUM confidence)
- `scripts/bench/tests/test_row_split.py` — Confirmed `ROW_EVENTS = ROW_BASE.with_(row_events=True, immediate_cmdlists=True)` pattern and test structure; read directly
- `scripts/bench/config.py` — `BenchConfig` fields `row_events` and `immediate_cmdlists`; read directly
- `scripts/bench/base.py` — `BenchTest` class structure for Wave 0 test; read directly

---

## Metadata

**Confidence breakdown:**
- INFRA-01 implementation: HIGH — exact insertion point, exact API calls, exact pattern all confirmed from source
- INFRA-02 implementation: HIGH — SYCL context equality API confirmed, insertion point identified, expected-always-pass behavior documented
- Validation approach: MEDIUM — bench framework capabilities for negative-exit tests not fully verified (flagged as Open Question 2)
- Timing of checks: HIGH — both requirements satisfied by identified insertion points

**Research date:** 2026-03-17
**Valid until:** Stable — changes only needed if ggml_check_sycl() or common.hpp ooo_stream() are significantly refactored
