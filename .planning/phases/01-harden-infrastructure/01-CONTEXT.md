# Phase 1: Harden Infrastructure - Context

**Gathered:** 2026-03-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Add startup assertions that catch misconfigured environments before any matmul runs. Two checks: (1) `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` must be set when `GGML_SYCL_ROW_EVENTS=1`, (2) OOO queue SYCL context must match in-order stream context.

</domain>

<decisions>
## Implementation Decisions

### Assertion placement
- Eager, at backend init — not lazy at first `ooo_stream()` call or first matmul
- Both checks must fire before any GPU work is dispatched
- If the row-events env var is set, validate prerequisites immediately during the SYCL init path (where `g_ggml_sycl_row_events` is already read at line 287)

### Failure behavior
- `GGML_LOG_ERROR` with clear message explaining what's wrong and how to fix it, then `abort()`
- Do NOT gracefully disable ROW_EVENTS — if the user asked for it and the env is broken, crash loudly
- Do NOT throw SYCL exceptions — use the same logging pattern as existing GGML_LOG calls
- Priority: fail within seconds of startup, not minutes into a generation

### Cmdlist detection
- Read `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` env var directly via `getenv()`
- Matches existing pattern: `g_ggml_sycl_row_events` uses `get_sycl_env()` which wraps `getenv()`
- No need for L0 runtime query — the env var is what controls the behavior

### Claude's Discretion
- Exact assertion messages (as long as they say what's wrong and how to fix it)
- Whether to add the context check to `ooo_stream()` accessor or to a separate init function
- Order of assertions (cmdlist first or context first — doesn't matter)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### SYCL backend
- `llama.cpp/ggml/src/ggml-sycl/common.hpp:365-394` — `ggml_backend_sycl_context` with `ooo_qptrs[]` and `ooo_stream()` accessor
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:280-290` — `get_sycl_env()` calls where `g_ggml_sycl_row_events` is read
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:4035-4054` — Row-split pre-sync comment documenting IMMEDIATE_COMMANDLISTS requirement

### Research
- `.planning/research/PITFALLS.md` — Pitfall 1 (batched cmdlist deadlock), Pitfall 6 (cross-context event silence)
- `.planning/research/STACK.md` — Immediate command list requirements and SYCL context semantics

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `get_sycl_env()` at ggml-sycl.cpp:280 — existing pattern for reading SYCL env vars, returns int
- `GGML_LOG_INFO/WARN/ERROR` macros — standard logging throughout the SYCL backend
- `ggml_sycl_info()` singleton — device info already printed at init, good place for assertions

### Established Patterns
- Env vars read once at static init time (line 287: `g_ggml_sycl_row_events = get_sycl_env(...)`)
- Device queues are lazily created in `stream()` and `ooo_stream()` accessors
- `SYCL_CHECK(CHECK_TRY_ERROR(...))` macro pattern used for SYCL call error checking

### Integration Points
- `ooo_stream()` accessor at common.hpp:389 — where OOO queues are first created (lazy)
- `ggml_sycl_info()` constructor — where device info is enumerated and env vars are logged
- Row-split entry at ggml-sycl.cpp:4037 — where the pre-sync comment documents the requirement

</code_context>

<specifics>
## Specific Ideas

- User priority: "whatever makes it easier for you to watch without waiting 5m for a 20 second failure" — fail fast is the driving principle
- This is a guard rail phase, not a feature phase — minimal code, maximum safety

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-harden-infrastructure*
*Context gathered: 2026-03-17*
