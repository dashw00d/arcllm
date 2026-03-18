# Pitfalls Research

**Domain:** Multi-GPU event-based SYCL synchronization on Intel Arc A770 / Level Zero
**Researched:** 2026-03-17
**Confidence:** HIGH — Primary sources: project's own crash logs, bench test docstrings, ggml-sycl.cpp source code, Intel official docs (oneAPI optimization guide, Level Zero spec), and verified Intel llvm GitHub issues. All findings cross-referenced against at least two sources.

---

## Critical Pitfalls

### Pitfall 1: Batched Command Lists Silently Drop Cross-Queue Event Signals

**What goes wrong:**
With `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` (batched mode), events submitted via `handler::depends_on()` or `ext_oneapi_submit_barrier()` on one queue are never observed by a waiting queue on a different device. The dependent queue hangs indefinitely or proceeds with stale data. No error is thrown — the operation simply never executes.

**Why it happens:**
In batched command list mode, L0 accumulates commands in a batch and submits them to the GPU as a unit. An event "fired" on Queue A is not flushed from the batch until the batch closes. Queue B, waiting on that event, never sees the signal because the batch on Queue A is still open. The hang is total and silent.

**How to avoid:**
`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` is REQUIRED for any cross-queue `depends_on()`. This is not optional. The project's bench config enforces `immediate_cmdlists=True` when `row_events=True`. The test `test_events_q4km_np1_batched` documents the hang explicitly as a known-bad baseline.

**Warning signs:**
- GPU compute utilization is 0% while inference runs (GPUs idle, never receive work)
- Process hangs at the Phase 1 submission step — never reaches Phase 2
- `strace` or `perf` shows the process blocked in a Level Zero wait call indefinitely
- Memory bandwidth shows no PCIe activity to non-main devices

**Phase to address:** Phase 1 (OOO Queue Infrastructure) — enforce `immediate_cmdlists` in the OOO queue constructor and assert it at startup.

---

### Pitfall 2: L0 In-Order Queue Does NOT Enforce Submission Order

**What goes wrong:**
Operations submitted to a SYCL in-order queue (`sycl::property::queue::in_order()`) with no explicit data dependency are not guaranteed to execute serially on Intel Level Zero. L0 submits operations as batched commands, and if two adjacent kernels have no explicit input/output dependency (even if they share memory by address), L0 may reorder or overlap them. The result is a kernel reading a buffer whose previous writer has not completed — produces `DEVICE_LOST` or silent garbage output.

**Why it happens:**
L0 performs its own dependency analysis on command lists. When a quantize kernel writes to buffer `X` and a matmul kernel reads `X`, but both are submitted on the same in-order queue with no explicit dependency in the command list, L0 may determine (incorrectly for this hardware) that no ordering is required. The in-order queue property is a SYCL-level contract that L0 honors inconsistently.

This was discovered empirically: removing `stream->wait()` between quantize and matmul on non-main devices caused `DEVICE_LOST` even with in-order queues, despite the operations being sequential in submission order (see `feedback_l0_inorder_queue.md` and `test_row_split.py` issue #4).

**How to avoid:**
Never rely on in-order queue semantics for dependent operations on L0/Arc. For the event-based path: use `ext_oneapi_submit_barrier({e_h2g})` before the quantize kernel to explicitly declare data dependency. The in-code comment at `ggml-sycl.cpp:4206` (`stream->ext_oneapi_submit_barrier()`) addresses exactly this.

For OOO queues (which have even weaker ordering guarantees): ALL dependent operations must be chained via `depends_on` or `ext_oneapi_submit_barrier({prior_event})`. Never submit to an OOO queue without explicit dependency tracking.

**Warning signs:**
- `DEVICE_LOST` immediately after removing a `stream->wait()`
- Output is numerically wrong (zeros, garbage) but no exception — indicates stale data
- Bug appears only on non-main devices (device 1, 2) not device 0
- Passes first 10-20 tokens then diverges (L0 timing-dependent race)

**Phase to address:** Phase 1 (OOO Queue Infrastructure) — document and test OOO queue ordering requirements in the standalone `test_ooo_row_split.cpp` before integration. Test 3 (barrier ordering) already validates `ext_oneapi_submit_barrier` correctness.

---

### Pitfall 3: Event Pool Exhaustion After Sustained Inference (~100 Matmuls)

**What goes wrong:**
L0 has a finite event pool per context. Each `ext_oneapi_submit_barrier()` creates a new L0 event handle. Without explicit pool flushing, events accumulate across the `i0` loop (448 matmuls/token × multiple events/matmul) and exhaust the pool. The driver returns `UR_RESULT_ERROR_INVALID_VALUE` (or silently produces a null event handle), causing the next `depends_on` call to reference an invalid event.

**Why it happens:**
SYCL events are reference-counted at the SYCL level but the underlying L0 event handles are only reclaimed when the event pool is flushed. For OOO queues, L0 does not automatically reap completed events — the application must call `queue.wait()` periodically to release completed handles back to the pool. At 448 matmuls/token × 3 events/matmul = ~1344 new event handles per token, the pool saturates after approximately 2-3 tokens of sustained inference.

**How to avoid:**
Flush OOO queues after each complete matmul dispatch cycle. The existing code in `ggml-sycl.cpp:4294-4301` does exactly this:
```cpp
// Flush OOO queues to release L0 event handles. Without this,
// events accumulate across matmul calls and exhaust the L0 pool.
if (use_event_sync) {
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if (dev[i].row_low != dev[i].row_high) {
            ctx.ooo_stream(i)->wait();
        }
    }
}
```
This `wait()` is one host stall per Phase 3 (not per matmul), which is acceptable. Do NOT remove it as an "optimization" — doing so causes event pool exhaustion within 2-3 tokens.

**Warning signs:**
- `UR_RESULT_ERROR_INVALID_VALUE` thrown from `ext_oneapi_submit_barrier()` at inference time
- Crash occurs after a consistent number of tokens (not random) — suggests fixed pool size
- Works for first 1-2 tokens then crashes
- Error manifests on all three devices simultaneously (pool is context-wide)

**Phase to address:** Phase 1 (OOO Queue Infrastructure) — the flush is already in place; ensure it is never removed and add a comment explaining the pool size constraint.

---

### Pitfall 4: Staging Buffer Reuse Without Event Chaining Causes Data Corruption

**What goes wrong:**
When the same `malloc_host` staging buffer is reused across loop iterations (e.g., the `i0` loop over matrix chunks), the second copy can overwrite the buffer before the first copy's `host→GPU` transfer completes. The GPU reads corrupted data — either partial old content or zero-fill from the new write. No exception is thrown; output is silently wrong.

**Why it happens:**
OOO queues make this worse: without explicit chaining, the GPU→host copy of iteration N+1 can start before iteration N's host→GPU copy completes, because OOO queues have no implicit ordering between submissions. The staging buffer is a single shared memory region; concurrent writes to it produce indeterminate results.

The `test_ooo_row_split.cpp` Test 4 (Staging buffer reuse with event chaining) validates the correct pattern: each GPU→host submission depends on the prior host→GPU completing (`e_prev_h2g`), serializing buffer access without any host stalls.

**How to avoid:**
Always chain staging buffer reuse via `depends_on(e_prev_h2g)` on the GPU→host submission. The `e_prev_staging_free` / `has_prev_staging` pattern in `ggml-sycl.cpp:4082-4148` implements this correctly. Never "simplify" the chaining by removing the dependency — it will corrupt data under concurrent load.

For the merge path in Phase 3 (`dev2dev_memcpy_staged_sync`): this currently uses synchronous waits, which is correct but slow. If this path is ever converted to event-based, the same chaining requirement applies.

**Warning signs:**
- Outputs are reproducibly wrong but only at high concurrency or long sequences
- Perplexity diverges significantly from reference at large `i0` (more than ~4 chunks)
- Correct at `i0=1` but wrong at `i0>1` (single-chunk passes, multi-chunk fails)
- Bug disappears when adding `staging_buf.wait()` between iterations (confirms the race)

**Phase to address:** Phase 2 (Event-Based Copy Path) — staging chaining must be implemented and tested before integration. The standalone test already validates it.

---

### Pitfall 5: ext_oneapi_submit_barrier() with Empty Wait List Is a No-Op on In-Order Queues (SYCL 2024 Bug)

**What goes wrong:**
When `queue::ext_oneapi_submit_barrier()` (no arguments) is called on an in-order queue, the SYCL runtime optimizes it to return the last recorded event. If the previous operation was submitted via `submit_without_event()` or experimental enqueue functions that don't track their events, the returned barrier event does not actually depend on that operation. Code waiting on the returned event proceeds while GPU work is still in flight. (Intel llvm issue #15606, fixed in PR #16223 but still present in older toolchains.)

**Why it happens:**
The optimization is a valid shortcut for the common case but breaks when the preceding submission has no tracked event. The barrier returns a stale event reference instead of a true "all prior work complete" signal.

**How to avoid:**
For OOO queues in the event-based path, use `ext_oneapi_submit_barrier({explicit_event_list})` rather than the zero-argument form. Always pass the specific event(s) you want to wait on rather than relying on the runtime's "last event" optimization. The project already does this correctly (e.g., `stream->ext_oneapi_submit_barrier({e_h2g})` at line 4151).

For the barrier recording at end of Phase 1 (`dev_events[i] = stream->ext_oneapi_submit_barrier()`), the zero-argument form is safe here because the OOO queue received explicit submissions (not submit_without_event). However, validate after any toolchain upgrade.

**Warning signs:**
- Phase 2 `dev_events[i].wait()` returns immediately but GPU work is still running
- Data in merged result is from previous iteration (the real computation hasn't written yet)
- Bug appears after upgrading oneAPI toolkit (older optimization behavior changes)
- `SYCL_PI_TRACE=2` shows barrier event being reused rather than newly created

**Phase to address:** Phase 1 (OOO Queue Infrastructure) — establish a policy of always using explicit event lists in `ext_oneapi_submit_barrier` calls in the hot path.

---

### Pitfall 6: Cross-Context Event Dependencies Are Undefined

**What goes wrong:**
If OOO queues on different devices are created from different SYCL contexts (rather than a shared multi-device context), events from one context passed to `depends_on()` on a queue in another context may be silently ignored or throw `invalid_object_error`. The Intel community forum documents cases where `depends_on()` with cross-queue events fails to execute the dependent kernel — the kernel is submitted but never runs. (Intel Community: "Using depends_on with events returned from different queues")

**Why it happens:**
SYCL events are scoped to a context. The specification states events must be from the same platform, but does not require the same context. However, the L0 backend implementation may silently drop cross-context event dependencies when they cannot be represented in a single L0 command list. The `ooo_stream()` accessor in `common.hpp:389` currently uses `dpct::get_device(device).out_of_order_queue()`, which returns a device-default queue. The context of these default queues must be verified to match the context of the in-order streams they depend on.

**How to avoid:**
Verify that all queues participating in `depends_on()` chains share the same SYCL context. In `ggml-sycl.cpp`, the OOO queues must be constructed with the same context as the device's primary in-order streams. Use `queue(context, device, ...)` constructor to explicitly bind context, not the device-default constructor.

Add a startup assertion: `assert(ooo_queue.get_context() == in_order_stream.get_context())`.

**Warning signs:**
- Events submitted to `depends_on()` are silently ignored (dependent work starts immediately without waiting)
- Intermittent correctness: sometimes works, sometimes produces wrong output
- Only manifests at higher concurrency where queues from different init paths are mixed
- `SYCL_PI_TRACE=2` shows no L0 dependency recorded for the cross-context event

**Phase to address:** Phase 1 (OOO Queue Infrastructure) — context binding must be explicit in the `ooo_stream()` constructor. Add assertion before Phase 2 copy path implementation.

---

### Pitfall 7: Q8_1 Quantization ds Field L1 Cache Visibility (Arc A770 Hardware Bug)

**What goes wrong:**
On Intel Arc A770, the Q8_1 quantization kernel writes the `ds` (scale + sum, `sycl::half2`) field from only `wi_id==0` (one work item per subgroup). The subsequent MMVQ kernel on the same device and same buffer reads `ds = 0.0` despite the write completing. The `qs` bytes are correct. All matmul outputs become zero, producing garbage activations through every layer with no error.

**Why it happens:**
Single-work-item writes of `half2` fields on Xe-HPG may not be flushed from L1 cache to global memory before the next kernel launch, even with an implicit in-order queue synchronization between kernels. This is an Intel Arc A770-specific hardware/driver behavior not observed on other GPU architectures. The project's workaround (having all work items write `ds`, `quantize.hpp:116-120`) is correct.

**How to avoid:**
The all-WI write workaround in `quantize.hpp` must remain. If the quantization kernel is ever refactored, the all-WI write pattern must be preserved. An alternative unverified fix: `sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device)` after the `ds` store.

Critical: do NOT re-enable the `if (wi_id == 0) { ds = ... }` pattern even if it appears in upstream commits — it will silently corrupt output on Arc A770.

**Warning signs:**
- Model outputs all zeros or near-zero activations
- Perplexity is NaN or infinite immediately from token 1
- Bug is present only with MMVQ kernel path, not DMMV (DMMV doesn't use `ds`)
- `GGML_SYCL_DEBUG_ROW=1` shows `ds=[0,0]` in probe output

**Phase to address:** Phase 3 (Row-Split Matmul Loop) — before enabling MMVQ with the event path, confirm the all-WI ds write is still present. Add regression test for `ds != 0` in the bench framework.

---

### Pitfall 8: Pre-Op Sync on Non-Main Devices Cannot Be Removed

**What goes wrong:**
Removing the pre-operation `stream->wait()` on non-main devices (device 1, device 2) before the matmul kernel causes `DEVICE_LOST` immediately. This wait was experimentally determined to be REQUIRED — the `ext_oneapi_submit_barrier()` alone (even on OOO queues) does not provide sufficient ordering for the quantize→matmul dependency on non-main devices under L0's command list batching.

**Why it happens:**
The pre-op sync ensures the quantized src1 buffer is fully visible to the matmul kernel. Without it, the non-main device's kernel starts executing (under L0's concurrent scheduling) before the copy and quantize operations on that device have written to the buffer. This is a manifestation of Pitfall 2 (L0 in-order queue violation) but is particularly severe on non-main devices because they receive data from a cross-device copy chain that L0 cannot automatically reason about.

The project confirmed this empirically: `FORCE_SYNC=1` (which only syncs the main device) still crashes. `SYNC_EVERY=1` (which only syncs within MUL_MAT op) still crashes. Only per-device `stream->wait()` before the kernel fixed it. (See `test_row_split.py` issues #4 and `PROJECT.md` constraint: "Host waits: ONE per matmul acceptable (pre-sync).")

**How to avoid:**
Keep `stream->wait()` before the `i0` loop on non-main devices. In the event path, this is the ONE acceptable host wait per matmul. The current code in `ggml-sycl.cpp:4026` (Phase 1 pre-sync) preserves this. Resist optimizing it away — it is load-bearing.

The confusion is that `ext_oneapi_submit_barrier()` looks equivalent to `wait()` in event terms but is not: the barrier is device-side and provides ordering only within the GPU's scheduler, not host-observable completion.

**Warning signs:**
- `DEVICE_LOST` within the first 1-5 matmuls when the wait is removed
- Removing the wait for `i == ctx.device` (main device) is safe — only non-main devices need it
- If a "fix" makes pre-op wait unnecessary, something else is wrong (likely regression to host-blocking path)

**Phase to address:** Phase 2 (Event-Based Copy Path) — document as an explicit architectural constraint, not a workaround to be cleaned up.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `stream->wait()` in Phase 3 merge path | Correctness without rewrite | 1 host stall per merge per column per non-main device | Acceptable until event-based merge is proven correct |
| Using `dpct::get_device().out_of_order_queue()` for OOO queues | One-line accessor | May return queue from wrong context (Pitfall 6) | Replace with explicit context-bound constructor before Phase 3 |
| Debug `fprintf` in Phase 2 (`ev_matmul_count <= 5`) | Easy debugging | ~5 stderr writes per inference, easily forgotten | Remove before performance benchmarking |
| Event pool flush via `ooo_stream(i)->wait()` at Phase 3 end | Prevents pool exhaustion | 1 host stall per matmul dispatch cycle | Permanent — do not remove |
| `GGML_SYCL_ROW_EVENTS=1` env gate | Clean separation of legacy/event paths | Code duplication, two paths to maintain forever | Acceptable until event path is validated; remove gate when stable |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Immediate command lists | Toggling `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` at runtime | Set ONCE at process start before any SYCL queue is constructed; changing mid-process leaves existing queues in undefined mode |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` | Not setting it globally | Required in ALL launch scripts; without it, device buffers >4GB fail silently (model load appears to succeed, inference crashes) |
| `ZES_ENABLE_SYSMAN=1` | Only setting it in `env.sglang-xpu.sh` but not in bench scripts | Required for VRAM usage monitoring; missing it makes GPU health metrics unavailable but does not affect correctness |
| Linux kernel version | Using kernel >6.6.25 without patch | Some kernel versions have fence expiration regressions on Arc A770 (see llama.cpp discussion #5277); kernel `6.17.0-19-generic` on this machine should be monitored |
| ReBar (Resizable BAR) | BIOS disabled | SIGBUS during device-to-host transfers; must be enabled for all Arc A770 compute workloads |
| mmap model loading | Using `--mmap` with SYCL on Arc | Host-to-device copies via mmap occasionally hang; `--no-mmap` flag required |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Adding host waits "for safety" in event path | Throughput stays at ~0.6 t/s despite event code running | Instrument each wait call; any `stream->wait()` inside the `i0` loop defeats the purpose | Immediately — even one extra wait per matmul costs ~300µs × 448 = 134ms/token |
| Using `sycl::event::wait()` on individual barrier events inside the matmul loop | 4000 host stalls/token instead of 3 | Phase 2 waits on COMPLETION events, never on SUBMISSION events inside Phase 1 | Immediately — this is the original bug being fixed |
| OOO queue construction inside the inference loop | JIT recompilation per token (~100ms overhead), queue object churn | Construct OOO queues once at model load (`ooo_qptrs[]` pattern in `common.hpp`), reuse per call | From token 1 — new queue = new context = new JIT compile |
| Removing the event pool flush at end of Phase 3 | Works for first 2-3 tokens, then `UR_RESULT_ERROR_INVALID_VALUE` | Keep the `ooo_stream(i)->wait()` flush; it is not a "cleanup optimization" | At ~100 matmuls (~2-3 tokens at 448 matmuls/token) |
| Using the merge path's `dev2dev_memcpy_staged_sync` for Phase 1 copy | Sync copy defeats event chaining | Phase 1 copy uses async `handler.memcpy()` with `depends_on`; Phase 3 merge keeps sync | Phase 1 degradation: immediate — sync copy costs 3 × 300µs × 448 = 403ms/token |

---

## "Looks Done But Isn't" Checklist

- [ ] **Event path enabled**: Verify `GGML_SYCL_ROW_EVENTS=1` is set in bench config AND `immediate_cmdlists=True`; both required, either alone causes hang or stall-path fallback
- [ ] **Pre-op sync preserved**: Check that `stream->wait()` before the `i0` loop on non-main devices is NOT removed — looks like dead code next to the barrier but is load-bearing
- [ ] **Event pool flush present**: The `ooo_stream(i)->wait()` block at `ggml-sycl.cpp:4296-4301` must survive all refactors; event path without it crashes at ~token 3
- [ ] **All-WI ds write in quantize.hpp**: Lines 116-120 and 88-92 must have all work items writing `ds`, not only `wi_id==0`; upstream merges may revert this
- [ ] **OOO queue context matches in-order stream context**: `ooo_qptrs[device]` must share context with `ctx.stream(device, is)` or cross-device `depends_on` silently fails
- [ ] **Staging buffer chaining active**: `e_prev_staging_free` dependency must be wired to `e_g2h.submit` for iterations `i0 > 0`; missing it produces sporadic data corruption at high `ne13*ne12`
- [ ] **Debug printfs removed before performance testing**: `ev_matmul_count <= 5` guard in Phase 2 — each `fprintf(stderr, ...)` is a host synchronization point

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| DEVICE_LOST after removing pre-op sync | LOW | Restore `stream->wait()` before `i0` loop on non-main devices; re-run `rowsplit.q4km_np1_100tok` to confirm correctness |
| Event pool exhaustion crash | LOW | Verify event pool flush block at end of Phase 3 is present; add `GGML_LOG_DEBUG` call when flush runs to confirm it executes |
| Batched commandlist deadlock (hang) | LOW | Set `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` in env before process start; kill hung process, restart with correct env |
| Cross-context event silence (no error, no work) | MEDIUM | Add `assert(ooo_queue.get_context() == stream->get_context())` at startup; if assertion fires, rebuild OOO queues with explicit shared context |
| Q8_1 ds corruption (all-zero output) | LOW | Check `quantize.hpp:116-120` for `if (wi_id==0)` guard on `ds` write; restore all-WI pattern |
| Staging buffer race (wrong data, no crash) | MEDIUM | Add `has_prev_staging` dependency chain back to `e_g2h` for `i0>0`; reproduce with `test_ooo_row_split` Test 4 to verify |
| DEVICE_LOST after event pool exhaustion | MEDIUM | GPU requires full process restart; add watchdog in proxy to detect `DEVICE_LOST` exit code and restart `llama-server` automatically |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Batched command lists deadlock (Pitfall 1) | Phase 1 (OOO Queue Infrastructure) | `test_events_q4km_np1_batched` must HANG; `test_events_q4km_np1_100tok` must PASS — both with ROW_EVENTS=1 |
| L0 in-order queue violation (Pitfall 2) | Phase 1 | `test_ooo_row_split` Test 3 (barrier ordering) must PASS; correctness test at np=1 must PASS without `stream->wait()` in Phase 1 body |
| Event pool exhaustion (Pitfall 3) | Phase 1 | 500+ token generation must complete without `UR_RESULT_ERROR_INVALID_VALUE`; verify pool flush runs via debug log |
| Staging buffer reuse race (Pitfall 4) | Phase 2 (Event-Based Copy Path) | `test_ooo_row_split` Test 4 must PASS; perplexity vs reference within 0.01 over 500 tokens |
| Barrier no-op on in-order queue (Pitfall 5) | Phase 1 | Use explicit event lists in all hot-path `ext_oneapi_submit_barrier` calls; verify no zero-arg form in the `i0` loop body |
| Cross-context event silence (Pitfall 6) | Phase 1 | Startup assertion on OOO queue context; verify with single matmul correctness test before enabling full inference |
| Q8_1 ds field bug (Pitfall 7) | Phase 3 (Row-Split Matmul Loop) | MMVQ kernel produces non-zero output on first matmul (instrument `ds` probe); perplexity vs reference within 0.01 |
| Pre-op sync removal regression (Pitfall 8) | Phase 2 | np=1 100-token run PASSES after pre-op sync is verified still present; `DEVICE_LOST` is the detection signal if removed |

---

## Sources

- `scripts/bench/tests/test_row_split.py` — First-hand failure logs, reproduction conditions, and all fixes (PRIMARY)
- `scripts/test_ooo_row_split.cpp` — Standalone OOO queue validation, staging buffer reuse test
- `.planning/codebase/CONCERNS.md` — Codebase audit of all known bugs and architectural risks
- `research/qwen3-local/08-multi-gpu-split.md` — In-depth multi-GPU architecture analysis, stability history
- `.claude/projects/.../memory/feedback_l0_inorder_queue.md` — L0 in-order queue violation finding
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:4082-4301` — Current event-based implementation with inline explanatory comments
- [Intel llvm issue #15606](https://github.com/intel/llvm/issues/15606) — ext_oneapi_submit_barrier no-op bug with enqueue functions (MEDIUM confidence — fixed in PR #16223, may be present in older toolchains)
- [Intel llvm issue #8132](https://github.com/intel/llvm/issues/8132) — Event dependency bug in L0/CUDA (event status reports "running" before actually running)
- [Intel llvm issue #4791](https://github.com/intel/llvm/issues/4791) — SYCL L0 event pool destroyed before events, lifecycle rules
- [Intel oneAPI Optimization Guide: Explicit Scaling SYCL (2024)](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/explicit-scaling-sycl.html) — Cross-device context and memory allocation constraints
- [Intel Level Zero Immediate Command Lists guide](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html) — Batched vs immediate command list cross-queue impact
- [Intel LLVM MultiTileCardWithLevelZero docs](https://intel.github.io/llvm/MultiTileCardWithLevelZero.html) — Multi-card memory allocation context rules
- [Intel community: Using depends_on with events from different queues](https://community.intel.com/t5/Intel-oneAPI-DPC-C-Compiler/Using-depends-on-with-events-returned-from-different-queues/td-p/1310387) — Cross-queue event dependency silent failure
- [UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS: >4GB allocation pitfall](https://jjfumero.github.io/posts/2022/04/understanding-memory-allocation-size-limitations-with-levelzero/) — 4GB per-buffer default limit on Arc
- [llama.cpp SYCL long-term issues discussion #5277](https://github.com/ggml-org/llama.cpp/discussions/5277) — Community-reported Arc A770 stability issues including fence timeout and mmap hang

---
*Pitfalls research for: Multi-GPU event-based SYCL synchronization, Intel Arc A770 / Level Zero*
*Researched: 2026-03-17*
