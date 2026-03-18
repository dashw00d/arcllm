# Phase 2: Validate Existing Event Path - Research

**Researched:** 2026-03-18
**Domain:** SYCL row-split dispatch loop correctness + host stall instrumentation
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CORR-01 | GLM-4.7-Flash Q4_K_M generates 500+ tokens at 2048 context on 3x A770 without desync or crash | Existing bench config `ROW_EVENTS` + model `glm47-q4km` covers this exactly; dispatch loop Phases 1-2 already implemented and routed via `use_event_sync` flag |
| CORR-02 | Host stall count per token is ~3 (one per device at Phase 2 barrier), not ~4000 | SYNC-03 stall counter feeds this; with `use_event_sync=true`, only `dev_events[i].wait()` calls remain in the hot path (one per non-main device); main device also waits once → ~3 total for 3-device setup |
| SYNC-03 | Host stall counter instruments `queue.wait()` calls and reports count per token when `GGML_SYCL_ROW_EVENTS=1` | Two sites to instrument: `ctx.stream()->wait()` pre-sync (line 4062) and `dev_events[i].wait()` Phase 2 (line 4249); counter must reset per-token, accumulate across matmuls, and print at token boundary |
</phase_requirements>

---

## Summary

Phase 2 is a validation-and-instrumentation phase, not new implementation. The 3-phase dispatch loop (Phase 1: OOO event chains, Phase 2: per-device barrier wait, Phase 3: blocking merge) is already implemented and gated behind `GGML_SYCL_ROW_EVENTS=1`. Two requirements (CORR-01, CORR-02) are correctness acceptance criteria that can only be confirmed by running the GLM-4.7-Flash bench test. The third requirement (SYNC-03) is an instrumentation task: add a host stall counter to the C++ dispatch loop that counts `queue.wait()` and `event.wait()` calls per token, then resets and prints when the token is done.

The key planning question is: "what must be TRUE before the correctness bench passes?" The answer is: Phase 1 (harden infrastructure) must be complete (it is), the INFRA assertions must pass on hardware, and the stall counter must exist so CORR-02 can be verified as a number rather than an assumption. This means SYNC-03 implementation comes first within Phase 2, then CORR-01/02 are validated by running `python3 -m bench rowsplit.events_glm47_np1_500tok`.

The debug `fprintf` statements in the dispatch loop (lines 4247-4252) print for the first 5 matmuls only. They confirm Phase 2 fires but are not sufficient for SYNC-03 — that requirement asks for a per-token aggregate count, not per-matmul debug logs.

**Primary recommendation:** Implement the stall counter in C++ first (SYNC-03), then run the GLM correctness bench (CORR-01/02). The counter output in server stderr is the proof of CORR-02.

---

## What Is Already Implemented (Do Not Re-Implement)

HIGH confidence — verified by direct source code inspection.

| Component | Location | Status |
|-----------|----------|--------|
| OOO queue pool per device | `common.hpp:ooo_stream()` | Done (Phase 1 hardened) |
| `use_event_sync` dispatch branch | `ggml-sycl.cpp:4045` | Done |
| Phase 1 OOO event chains (copy, quantize, kernel) | `ggml-sycl.cpp:4079-4239` | Done |
| Phase 2 per-device barrier event wait | `ggml-sycl.cpp:4242-4253` | Done |
| OOO queue flush after Phase 3 (event pool) | `ggml-sycl.cpp:4313-4321` | Done |
| Pre-sync `ctx.stream()->wait()` | `ggml-sycl.cpp:4062` | Done |
| Staging buffer reuse chaining (`e_prev_staging_free`) | `ggml-sycl.cpp:4155-4167` | Done |
| `ROW_EVENTS` + `immediate_cmdlists` bench configs | `test_row_split.py:ROW_EVENTS` | Done |
| GLM model in bench config | `config.py:glm47-q4km` | Done |
| GLM model file on disk | `models/GLM-4.7-Flash-heretic-GGUF/` | Verified present |
| INFRA-01/02 startup assertions | `ggml-sycl.cpp`, `common.hpp` | Done (Phase 1) |

**The only missing pieces for Phase 2:**
1. **SYNC-03**: Host stall counter in C++ (new instrumentation)
2. **CORR-01/02**: A bench test for GLM 500-token generation that verifies and records the result

---

## Standard Stack

No new libraries or tools are needed. Everything is the project's existing stack.

### Core
| Component | Version | Purpose | Notes |
|-----------|---------|---------|-------|
| llama.cpp SYCL backend | local checkout | Dispatch loop to instrument | `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` |
| Intel oneAPI DPC++ | 2025.3.2 | `sycl::event.wait()` API | Already installed, verified |
| bench framework | project local | Test runner + result capture | `scripts/bench/` |
| GLM-4.7-Flash Q4_K_M | on disk | Correctness target model | `models/GLM-4.7-Flash-heretic-GGUF/` |

**Installation:** None needed.

---

## Architecture Patterns

### Pattern 1: Host Stall Counter Design (SYNC-03)

**What:** A static atomic counter in the row-split dispatch path that increments each time a host-blocking wait occurs, then prints and resets at a token boundary.

**When to use:** Only when `use_event_sync` is true (i.e., `GGML_SYCL_ROW_EVENTS=1`).

**What to count:**
- `ctx.stream()->wait()` at line 4062 — the pre-sync host wait (1 per matmul, both event and legacy paths)
- `dev_events[i].wait()` at line 4249 — Phase 2 per-device event wait (event path only)

**What NOT to count:**
- `ctx.ooo_stream(i)->wait()` at line 4318 — this is the OOO queue flush to release L0 event pool handles; it runs after Phase 2 sync and is expected. Including it would inflate the number.
- Phase 3 `dev2dev_memcpy_staged_sync` internal waits — those are Phase 3's problem, fixed in Phase 3.

**Token boundary detection:** The dispatch loop runs once per matmul call within a token. A token boundary is harder to detect from inside the matmul path. The practical approach: count stalls per matmul and report them. The planner can decide between two options:
- **Option A (simpler):** Print stall count at the end of each matmul call (verbosely confirms per-call), suppress after first N
- **Option B (per-token aggregate):** Use `ne11` change detection — when `ne11 == 1` it's decode (one token), so each `ne11==1` call is one token. Print count for each call. Over 500 tokens, you'd see ~3 per call.
- **Option B is correct for CORR-02:** Requirements says "~3 waits per token" — Option B matches.

With 3 devices and `use_event_sync=true`:
- Pre-sync wait: 1 per matmul (unavoidable — drains in-order queue before row-split reads)
- Phase 2 event waits: 3 per matmul (one per device — `dev_events[0].wait()`, `dev_events[1].wait()`, `dev_events[2].wait()`)
- Total expected: **4 per matmul** (not 3 as the requirement states)

**Clarification on CORR-02 "~3":** The requirement says "one per device at Phase 2 barrier." The pre-sync wait is a separate bullet in the requirement. Reading more carefully: CORR-02 says "~3 waits per token", SYNC-03 says it instruments `queue.wait()` calls. The pre-sync `ctx.stream()->wait()` may be what the requirement is counting as "one per device" — since it only fires once per matmul on the main device's stream. The 3 comes from 3 devices × 1 Phase 2 event wait each. The planner should pick the consistent interpretation and document it in the plan.

**Simplest correct implementation:**
```cpp
// In the use_parallel_row_split block, after Phase 2 event waits:
static thread_local int g_stall_count = 0;  // per-token accumulator

// At pre-sync (line 4062):
ctx.stream()->wait();
if (use_event_sync) g_stall_count++;

// At Phase 2 event wait (line 4249):
dev_events[i].wait();
if (use_event_sync) g_stall_count++;

// At end of matmul (after Phase 3), print if use_event_sync:
if (use_event_sync && ne11 == 1) {  // decode mode: one call per token
    fprintf(stderr, "[EV] stalls/token: %d\n", g_stall_count);
    g_stall_count = 0;
}
```

**Note on `thread_local`:** The dispatch loop runs on a single thread per context (confirmed by absence of worker threads in the event path). A plain `static int` scoped inside the `use_parallel_row_split` block is sufficient. `thread_local` is safer for correctness if the model ever uses parallel decode threads.

### Pattern 2: GLM Correctness Bench Test (CORR-01/02)

**What:** A bench test that runs GLM-4.7-Flash Q4_K_M for 500+ tokens at 2048 context with `ROW_EVENTS=True`.

**Existing config to extend:**
```python
# From test_row_split.py:
ROW_EVENTS = ROW_BASE.with_(row_events=True, immediate_cmdlists=True)

# For Phase 2, need 500+ tokens:
def test_events_glm47_np1_500tok(self):
    self.run(ROW_EVENTS.with_(
        model="glm47-q4km", timeout=600,
        name="rowsplit_events_glm47_np1_500tok",
        prompt=THINK_PROMPT,  # long prompt to encourage long output
        n_parallel=1, concurrent=1, context=2048,
        flash_attn=True,
        max_tokens=500))
```

**Acceptance check:** `result.completed == 1` and `result.total_tokens >= 500`. If the model crashes (DEVICE_LOST, desync), `result.failed > 0` and bench auto-resets GPUs.

**Where to put it:** Extend the existing `TestRowSplit` class in `scripts/bench/tests/test_row_split.py`, OR create a new `scripts/bench/tests/test_glm_correctness.py`. The latter is cleaner as a Phase 2 artifact.

### Pattern 3: Reading Stall Count From Bench Output

The stall counter prints to stderr, which llama-server writes to its log file (`/tmp/bench_logs/{name}.log`). After the test completes, the test can read the log and check for `[EV] stalls/token:` lines.

**In the bench test:**
```python
import re
log = Path(f"/tmp/bench_logs/{name}.log")
stall_lines = [l for l in log.read_text().splitlines() if "[EV] stalls/token:" in l]
counts = [int(re.search(r"stalls/token: (\d+)", l).group(1)) for l in stall_lines]
# For CORR-02: assert all(c <= 6 for c in counts), f"Too many stalls: {counts}"
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| GPU reset between tests | Custom reset script | `BenchRunner.reset_gpus()` (already does hw reset, kill, verify) |
| Model file path resolution | Hardcoded path | `MODELS["glm47-q4km"]` in `config.py` |
| Server lifecycle | Custom subprocess management | `BenchRunner.run_test()` |
| Event pool management | Explicit L0 event pool calls | OOO queue flush at line 4313 already handles this |
| Re-implementing dev2dev copy | New staging buffer logic | Event copy path at lines 4146-4167 is already complete |

---

## Common Pitfalls

### Pitfall 1: Counting OOO Queue Flush as a Stall
**What goes wrong:** Including `ooo_stream(i)->wait()` (line 4318) in the stall counter inflates count from ~4 to ~7 per matmul.
**Why it happens:** It looks like a wait call, but it's a flush to release L0 event pool handles, not a host stall blocking inference.
**How to avoid:** Count only `ctx.stream()->wait()` (line 4062) and `dev_events[i].wait()` (line 4249). Document the exclusion.

### Pitfall 2: Running Correctness Test Without Stall Counter First
**What goes wrong:** CORR-01 (generation test) passes but CORR-02 (stall count) cannot be verified because there's no counter.
**How to avoid:** Implement SYNC-03 before running the CORR-01/02 bench. The stall counter output is the evidence for CORR-02.

### Pitfall 3: Short Token Count Hides Crashes
**What goes wrong:** Using `max_tokens=100` passes because most crashes happen at ~200-400 tokens (L0 event pool saturation, buffer reuse race).
**How to avoid:** CORR-01 specifically requires 500+ tokens. Use `max_tokens=500` in the bench config.

### Pitfall 4: Forgetting `flash_attn=True` for GLM
**What goes wrong:** GLM-4.7-Flash uses MLA attention. Without flash attention, it may run much slower or differently.
**How to avoid:** GLM bench configs already use `flash_attn=True` in the existing test_row_split.py tests. Use the same pattern.

### Pitfall 5: Debug fprintf Spam in the Existing Code
**What goes wrong:** The existing `[EV] matmul #N P2 wait dev%d...` debug prints at lines 4247-4252 fire for the first 5 matmuls. These are not SYNC-03. If the planner tries to count these as the stall counter, it's the wrong metric.
**How to avoid:** SYNC-03 needs a new per-token aggregate counter, not the per-matmul debug prints. The plan must add new instrumentation.

### Pitfall 6: np > 1 Crashes Are Out of Scope
**What goes wrong:** np=2 crashes at 56-83s even with host-stall sync — a separate root cause (L0 in-order queue violation in non-MUL_MAT ops on main device). Test with np=1 only.
**How to avoid:** Run CORR-01 with `n_parallel=1`. Do not investigate np>1 crashes in this phase.

---

## Code Examples

Verified from direct source inspection.

### Existing Host Wait Sites to Instrument

```cpp
// Source: ggml-sycl.cpp:4062 — pre-sync (counts as 1 stall per matmul)
ggml_sycl_set_device(ctx.device);
SYCL_CHECK(CHECK_TRY_ERROR(ctx.stream()->wait()));

// Source: ggml-sycl.cpp:4249 — Phase 2 per-device (counts as 1 stall per device per matmul)
SYCL_CHECK(CHECK_TRY_ERROR(dev_events[i].wait()));

// Source: ggml-sycl.cpp:4318 — OOO flush (DO NOT COUNT — pool management, not inference stall)
ctx.ooo_stream(i)->wait();
```

### Existing Debug Print Pattern (lines 4246-4252)

```cpp
// These fire only for first 5 matmuls — fine for startup validation,
// not sufficient for SYNC-03 per-token counter.
if (use_event_sync && ev_matmul_count <= 5) {
    fprintf(stderr, "[EV] #%d P2 wait dev%d...\n", ev_matmul_count, i);
}
SYCL_CHECK(CHECK_TRY_ERROR(dev_events[i].wait()));
if (use_event_sync && ev_matmul_count <= 5) {
    fprintf(stderr, "[EV] #%d P2 dev%d done\n", ev_matmul_count, i);
}
```

### GLM Bench Config (existing — use as-is)

```python
# Source: scripts/bench/tests/test_row_split.py:142
ROW_EVENTS = ROW_BASE.with_(row_events=True, immediate_cmdlists=True)
# ROW_BASE already has: split_mode="row", affinity="0,1,2", tensor_split="1,1,1",
#   sycl_flags=(("ROW_ALLOW_MMVQ", "1"),)
```

### Bench Test Skeleton for Phase 2

```python
# Source: pattern from test_row_split.py existing tests
def test_events_glm47_np1_500tok(self):
    """GLM-4.7-Flash Q4_K_M row-split 500 tokens — CORR-01/02 acceptance gate.
    Requires: GGML_SYCL_ROW_EVENTS=1, immediate_cmdlists=1, ROW_ALLOW_MMVQ=1.
    PASS criteria: completed=1, total_tokens>=500, stalls/token<=6 in server log."""
    self.run(ROW_EVENTS.with_(
        model="glm47-q4km", timeout=600,
        name="rowsplit_events_glm47_np1_500tok",
        prompt=THINK_PROMPT,
        n_parallel=1, concurrent=1, context=2048,
        flash_attn=True,
        max_tokens=500))
```

---

## State of the Art

| Old Approach | Current Approach | When Changed |
|--------------|------------------|--------------|
| Sequential device loop (6000 stalls/token) | 3-phase parallel loop (Phase 1 async, Phase 2 event wait) | 2026-03-17 |
| `dev2dev_memcpy_staged_sync` (blocking copy) | `handler::depends_on()` event chains in Phase 1 | 2026-03-17 |
| Debug probes in mmvq.cpp | Reverted to upstream mmvq.cpp | 2026-03-17 |
| No startup assertions | INFRA-01/02 assertions in `ggml_check_sycl()` and `ooo_stream()` | 2026-03-18 (Phase 1) |

**Deprecated/outdated:**
- Per-matmul `stream->wait()` in non-main device Phase 1: removed from event path; only pre-sync wait remains
- Debug fprintf without condition guard (ev_matmul_count <= 5): should be removed before Phase 3 (v2 VAL-02 requirement)

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | bench framework (project-local Python, `scripts/bench/`) |
| Config file | `scripts/bench/config.py` |
| Quick run command | `cd /home/ryan/llm-stack/scripts && python3 -m bench rowsplit.events_glm47_np1_500tok` |
| Full suite command | `cd /home/ryan/llm-stack/scripts && python3 -m bench rowsplit` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SYNC-03 | Stall counter in C++ increments at `queue.wait()` / `event.wait()` sites; prints per-token count to stderr | unit (build verify) + smoke (server log check) | `cmake --build llama.cpp/build-sycl --target llama-server -j$(nproc)` | C++ changes needed |
| CORR-01 | GLM-4.7-Flash 500+ tokens at 2048 context, no crash | integration (long run) | `python3 -m bench rowsplit.events_glm47_np1_500tok` | bench test needed |
| CORR-02 | Stall count in server log is ~3-6 per token | smoke (log grep) | Read `/tmp/bench_logs/rowsplit_events_glm47_np1_500tok.log` for `[EV] stalls/token:` | depends on SYNC-03 |

### Sampling Rate
- **Per task commit:** Build verify (`cmake --build` exit 0) for C++ changes; `python3 -m bench infra` for env correctness
- **Per wave merge:** Full correctness bench (`python3 -m bench rowsplit.events_glm47_np1_500tok`)
- **Phase gate:** Bench completes with `completed=1`, `total_tokens>=500`, stall count visible in log

### Wave 0 Gaps
- [ ] `scripts/bench/tests/test_glm_correctness.py` — OR extend `test_row_split.py` — covers CORR-01/02
- [ ] C++ stall counter in `ggml-sycl.cpp` — covers SYNC-03

*(No framework install needed — bench framework is fully operational)*

---

## Open Questions

1. **Exact stall count expected: 3 or 4 per matmul?**
   - What we know: 3 devices, `use_event_sync=true`. Pre-sync = 1 wait. Phase 2 = 3 waits (one per device including main). Total = 4.
   - What's unclear: CORR-02 says "~3" and references "one per device at Phase 2." Does this exclude the pre-sync wait from the count?
   - Recommendation: Count all host waits (including pre-sync). Print actual number. If it comes out to 4, update CORR-02's expected value in the requirements to 4. The exact number matters less than confirming it's not 4000.

2. **Where to add the new bench test: extend test_row_split.py or new file?**
   - What we know: test_row_split.py already has GLM tests (`test_glm47_np1_100tok`). Adding 500-token variant is consistent.
   - Recommendation: Add to existing `test_row_split.py` for co-location. If Phase 2 needs its own clearly-named artifact, create `test_glm_correctness.py`.

3. **Will 500 tokens complete within timeout?**
   - What we know: Current GLM throughput at ~0.6 t/s with host-sync. With event path, expected to be higher but unconfirmed.
   - At 0.6 t/s, 500 tokens = 833 seconds. Timeout needs to be at least 900s.
   - With event path working, ~5 t/s or better → 100 seconds. Timeout of 600s is safe.
   - Recommendation: Set timeout=900 to be safe for the worst case where the event path is slower than expected.

---

## Sources

### Primary (HIGH confidence)
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` lines 4029-4324 — Direct inspection of current 3-phase dispatch loop implementation
- `llama.cpp/ggml/src/ggml-sycl/common.hpp` lines 364-578 — `ggml_backend_sycl_context`, `ooo_stream()` accessor
- `scripts/bench/tests/test_row_split.py` — All existing bench configs; `ROW_EVENTS`, `glm47-q4km` usage confirmed
- `scripts/bench/config.py` — `MODELS["glm47-q4km"]` path verified; `sycl_env()` confirms how `row_events=True` maps to env vars
- `scripts/bench/runner.py` — Server lifecycle, log file path (`/tmp/bench_logs/{name}.log`), error handling
- `.planning/phases/01-harden-infrastructure/01-01-SUMMARY.md` — Phase 1 completion confirmed; INFRA-01/02 in binary

### Secondary (MEDIUM confidence)
- `.planning/research/SUMMARY.md` — Project-level research; Phase 2 described as "test patterns established in benchmark framework, no additional research needed"
- `.planning/REQUIREMENTS.md` — CORR-01, CORR-02, SYNC-03 exact text; traceability table confirms all three are Phase 2

---

## Metadata

**Confidence breakdown:**
- What's implemented: HIGH — confirmed by direct source code inspection with line numbers
- What the stall counter should count: HIGH — two clearly identified call sites
- Expected stall count (CORR-02 "~3"): MEDIUM — slight ambiguity on whether pre-sync wait is included; confirmed by implementation logic but number 3 vs 4 needs to be validated by running the test
- Bench test design: HIGH — directly mirrors existing test patterns in test_row_split.py

**Research date:** 2026-03-18
**Valid until:** Stable (SYCL API and bench framework are not fast-moving for this project)
