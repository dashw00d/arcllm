# Codebase Concerns

**Analysis Date:** 2026-03-17

## Critical Bugs

### 1. Level Zero In-Order Queue Violation — Multi-Slot Crash

**Issue:** Intel Level Zero runtime does NOT honor in-order queue semantics. Despite queues created with `sycl::property::queue::in_order()`, kernels are submitted in batched command lists and may execute concurrently when L0 determines they have no data dependencies.

**Files:**
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` — graph execution loop (lines 9610-9617)
- `llama.cpp/ggml/src/ggml-sycl/quantize.hpp` — quantization kernel
- Test documentation: `scripts/bench/tests/test_q8_1_corruption.py` (lines 282-347)

**Affected Workloads:**
- Multi-slot (np≥2) generation with >500 tokens/slot
- Single-GPU at batch≥2, multi-GPU MoE at any batch
- Layer-split crashes reliably at np=2 × 1024 tokens (~77s into decode)
- GLM-4.7 MoE on 3x A770: crashes at ~98s with np=4

**Root Cause:** Operations in a single SYCL stream are supposed to execute serially. However, L0 submits work to the GPU as batches of commands. When consecutive operations (e.g., quantize Q8_1 then matmul) lack explicit data dependencies in the command list, L0 may launch them concurrently. The FFN matmul reads quantized data while the quantize kernel is still writing it, causing NaN/Inf results.

**Current Mitigation:**
- `GGML_SYCL_SYNC_EVERY=3` on single GPU: stream->wait() after every 3 MUL_MAT ops (41.4 t/s on 0.6B Q8_0, was 79 t/s baseline)
- `GGML_SYCL_FORCE_SYNC=1` on multi-GPU: stream->wait() after EVERY graph op (too slow, 6.2 t/s on 32B, unusable for production)
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`: Per-kernel submission (avoids batching) but causes 3x VRAM spike (7GB→27GB) and is slower overall

**Why No Workaround:** The bug manifests in the interaction between L0's command list batching and SYCL's kernel submission ordering. Possible fixes:
1. Intel updates L0 to honor in-order queue semantics (needs driver update)
2. Explicit device-side barriers between dependent ops (GGML_SYCL_ROW_EVENTS=1, WIP for row-split only)
3. Per-layer sync (sync once per transformer layer instead of per-op) — untested, may still be insufficient
4. Eliminate multi-slot batching altogether — not viable for production

**Impact:** Blocks production use of multi-slot inference at >500 tokens/slot. Forces choice between:
- Layer-split (high throughput ~17.5 t/s np=16, but crashes on long multi-slot)
- Row-split (lower throughput ~0.6 t/s np=1, event-based sync being developed)
- Single-slot only (np=1) with expensive SYNC_EVERY overhead

---

### 2. Q8_1 Activation Quantization Corruption

**Issue:** F32→Q8_1 activation quantization produces NaN/Inf values when multiple slots process >500 tokens concurrently. The corruption happens upstream of matmul kernels, in the quantization or buffer management.

**Files:**
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:4442` — quantization dispatch
- `llama.cpp/ggml/src/ggml-sycl/quantize.hpp:27-56` — quantize_q8_1 kernel
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:1987-2096` — pool allocator (prime suspect)
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:3722` — f32 source binding
- Test documentation: `scripts/bench/tests/test_q8_1_corruption.py` (lines 45-77, 206-243)

**Symptoms:**
```
raw=nan scaled=nan q6s=[nan,nan] q8ds=[nan,nan]
ds=[nan,nan] q=[0,0,0,0]  # NaN scales, zero quants
```

**Current Theory:** Pool buffer reuse race condition. The ggml_sycl_pool_leg allocator (no mutex) reuses buffers immediately after a kernel submits. Slot A's quantization kernel is still in flight when Slot B reuses the same buffer address, corrupting reads.

**Why Sync Doesn't Fix It:** Even stream->wait() after quantization doesn't prevent the crash at np≥2 (test_q8_1_corruption.py line 136), suggesting the bug is not simple async reuse but L0's kernel overlap within a single stream.

**Impact:** Cannot use FUSED_MMQ kernel at np≥2 × 1024 tokens (the benchmark showed FUSED_MMQ improves np=16 from 17.4→21.7 t/s, but enables a crash workload at lower parallelism). Fused kernel gains are negated by crash risk.

---

### 3. MMVQ NaN Crash at Multi-Slot Deep Sequences

**Issue:** The SYCL MMVQ kernel (mmvq.cpp:1995) produces NaN in attention Q matrix computations at 4+ slots × 1024 tokens. Crashes with UR_RESULT_ERROR_DEVICE_LOST.

**Files:**
- `llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:1995` — crash site
- Test documentation: `scripts/bench/tests/test_mmvq_nan.py` (lines 1-78)

**Current Findings:**
- 1 slot × 1024 tokens: PASS (6.6 t/s)
- 4 slots × 1024 tokens: CRASH ~120s (~600 tokens/slot)
- 16 slots × 200 tokens: PASS (17.5 t/s)
- 16 slots × 512 tokens: UNKNOWN (needs testing)

**Root Cause:** Upstream of MMVQ. The FUSED_MMQ kernel bypass attempt (test_mmvq_nan.py lines 55-72) showed the bug is NOT in MMVQ kernel itself but in Q8_1 activation data corruption before any matmul runs. See issue #2 above.

**Impact:** Thinking-enabled workloads (which generate longer sequences per slot) are extremely risky. The test_mmvq_nan workload is exactly what users would request: "generate with thinking for multiple concurrent requests."

---

## Performance Bottlenecks

### 1. Row-Split Host Stall Overhead

**Issue:** Row-split distributes matrix rows across 3 GPUs but synchronizes via `stream->wait()` between phases, causing massive host stalls.

**Files:**
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:4956` — pre-merge sync
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:4011` — 3-phase loop (Phase 2 sync point)
- Test documentation: `scripts/bench/tests/test_row_split.py` (lines 8-122)

**Symptoms:**
- 0.6 t/s (was 0.4 t/s with sequential dispatch)
- 99% of time is host waits, not compute
- GPUs idle at 600MHz during syncs

**Bottleneck Analysis** (test_row_split.py lines 98-106):
- Theoretical ceiling: ~43 t/s (19ms compute + 4ms PCIe overhead)
- Observed: 0.6 t/s = 1.55s/token
- Per-matmul overhead: ~9 host waits × 300µs = 2.7ms per matmul × 448 = 1.21s total
- Host stalls account for: 1.53s out of 1.55s (99%)

**In-Progress Fix:** Event-based sync (GGML_SYCL_ROW_EVENTS=1) replaces host->wait() with device-side barriers. Eliminates Phase 1 host stalls, keeps only Phase 2 completion events. Expected to unlock row-split as production alternative to layer-split.

**Impact:** Row-split is currently 29x slower than layer-split (0.6 vs 17.5 t/s), making it useless for production. If event-based sync doesn't close the gap significantly, row-split cannot replace layer-split despite its theoretical advantage for multi-slot stability.

---

### 2. Q8_0 Bandwidth Ceiling

**Issue:** Q8_0 quantization is bandwidth-bound on Arc A770, regardless of parallelism.

**Files:**
- `llama.cpp/ggml/src/ggml-sycl/vecdot.hpp` — dot product kernels
- Benchmark result: `scripts/bench/tests/test_baseline.py` — q8 test
- Known result in CLAUDE.md line 93

**Ceiling:** 3.3 t/s max throughput across all batch/parallelism configs.

**Why:** Q8_0 uses 1 byte/value (8-bit quants). With 560 GB/s A770 bandwidth and 32B model weights = ~32GB to read per batch. At best: 560GB/s ÷ 32GB = ~17 passes/sec. With 5-7 matmuls per token, ceiling is (17÷6) ≈ 2.8 t/s. Adding KV cache copies and overhead → 3.3 t/s observed.

**Impact:** Q8_0 models (e.g., Qwen3-32B Q8_0, GLM-4.7 Q8_0) can never exceed 3.3 t/s on Arc A770 alone. Parallelism doesn't help. Q4_K_M (better compression, higher t/s ceiling) is strongly preferred for production. Users requesting Q8_0 for accuracy will be disappointed by single-digit throughput.

---

### 3. Batch/UBatch Tuning Has Zero Effect

**Issue:** Changing batch (-b) and ubatch (-ub) flags produces no measurable throughput difference.

**Files:**
- `scripts/bench/tests/test_sycl_env.py` — syclenv.* tests
- Known result in CLAUDE.md line 96

**Finding:** Tested various batch/ubatch combos on single and multi-GPU. No correlation with throughput. Likely because:
- Arc A770's ROB (reorder buffer) is small; batch size doesn't saturate it
- SYCL/L0 overhead dominates; batching changes make no difference
- GPU utilization already maxed out by compute demand

**Impact:** Configuration space is smaller than expected. Knob-turning on batch doesn't improve performance; users should use defaults and focus on parallelism (np), quantization (Q4_K_M), and fused kernel (FUSED_MMQ).

---

## Fragile Architectures

### 1. FFN Gate Magnitude Overflow at Multi-Slot

**Issue:** At batch≥2, hidden state magnitudes accumulate through residual connections and overflow during FFN gate matmul computation.

**Files:**
- `scripts/bench/tests/test_q8_1_corruption.py:186-204` — magnitude trace evidence
- Test finding: "layer 63 ffn_gate-63 -nan, src1 amax=2.7e7" (line 229-231)

**Evidence:**
- Layer 0: l_out ≈ 157
- Layer 5: l_out ≈ 217
- Layer 6: l_out ≈ 16,600 (100x jump!)
- Layers 7-63: stable at ~17,000

**Why Batch Affects It:** Single-slot computation uses different memory access patterns and numerical accumulation order, keeping values just below overflow threshold. Batch≥2 with long sequences triggers slightly different accumulation order, crossing the threshold.

**Impact:** Not directly fixable without model changes (e.g., better layer norm scaling). Manifests only at multi-slot long generation, which is exactly the production use case. Suggests the model (Qwen3-32B) or quantization (Q4_K_M) may be numerically fragile at scale.

---

### 2. Pool Allocator Without Mutex Protection

**Issue:** The SYCL pool allocator (ggml_sycl_pool_leg, lines 1987-2096) uses best-fit search with no locking, and buffer reuse doesn't account for in-flight kernels.

**Files:**
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:2027-2041` — best-fit search
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:3745` — Q8_1 buffer allocation
- Test hypothesis: `scripts/bench/tests/test_q8_1_corruption.py:45-77`

**Risk:** At np≥2, multiple ggml_sycl_op_mul_mat() calls in rapid succession allocate from the same pool. Buffer A is allocated → quantization kernel submitted → op returns → buffer returned to pool → Buffer B allocated → buffer A reused while quantization kernel still reads it.

**Why This Matters:** With L0's batched command lists, the actual GPU timeline doesn't match submission order. A "returned" buffer may still have kernels in-flight reading it 100ms later.

**Mitigation:** Event-based pool with reference counting (never reuse until event fires), or per-slot buffer pre-allocation. Neither is implemented.

**Impact:** Core stability issue. Every multi-slot test is a dice roll. Makes debugging extremely time-consuming (Heisenbug behavior).

---

### 3. Flash Attention F16 Overflow Paths

**Issue:** Multiple locations in flash attention code perform unbounded F32→F16 conversions without bounds checking.

**Files:**
- `llama.cpp/ggml/src/ggml-sycl/fattn-common.hpp:347-361` — quantize_q8_1_to_shared() (inline Q8_1 quantization)
- `llama.cpp/ggml/src/ggml-sycl/fattn-tile.hpp:648-665` — KQ_max_scale exp() → F16 conversion
- `llama.cpp/ggml/src/ggml-sycl/fattn-vec.hpp:263` — Q value loading without bounds check
- Discovery: `scripts/bench/tests/test_q8_1_corruption.py:251-270`

**Specific Cases:**
1. `d = amax / 127` can produce F16-overflow if amax > 8.3M
2. `exp(KQ_max - KQ_max_new)` unbounded if attention patterns shift dramatically
3. Large Q values from quantization loaded into F16 registers

**Current State:** Only triggers with SYCL_FAST_FP16 enabled (it IS enabled globally in common.hpp:39). F16 clamping was added to cpy_1_f32_f16() but these flash attention paths remain unclamped.

**Impact:** Contributes to the NaN/Inf cascade. Not a primary blocker (was ruled out as primary cause in test_q8_1_corruption.py line 276), but part of the numerical stability issue.

---

## Technical Debt

### 1. Dead MMQ Code Path

**Issue:** Existing MMQ implementation in mmq.cpp (lines 6081-6000+) is disabled by ggml_sycl_supports_mmq() returning false (line 5635), with comment "TODO: accuracy issues".

**Files:**
- `llama.cpp/ggml/src/ggml-sycl/mmq.cpp` — ~2000 lines of dead code
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:5635` — disabling check
- Measurement: `scripts/bench/tests/test_fused_mmq.py:77-85`

**Why:** The old MMQ was 31-63% slower than generic GEMM for batch>8, so Intel disabled it. New fused kernel (fused_mmq.cpp) replaces it but MMQ is never deleted.

**Impact:** Code maintenance burden. New developers may try to optimize MMQ not realizing it's dead. Makes SYCL backend appear to have CUDA-style MMQ when it doesn't.

---

### 2. Incomplete Row-Split Implementation

**Issue:** Row-split has multiple half-finished synchronization strategies:
- Legacy `stream->wait()` on every non-main device (slow, 0.47 t/s)
- 3-phase parallel dispatch with post-op waits on non-main (faster, 0.6 t/s, still bottlenecked)
- Event-based sync (GGML_SYCL_ROW_EVENTS=1) partially implemented, untested at scale

**Files:**
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:4025` — event-based row-split (WIP)
- `llama.cpp/ggml/src/ggml-sycl/common.hpp:370` — ooo_stream() accessor for OOO queues
- Test documentation: `scripts/bench/tests/test_row_split.py:89-122`

**Known Issues:**
- Immediate commandlists required for event sync (batched mode deadlocks — test_row_split.py:221-231)
- Event sync not yet benchmarked at np≥2
- Row-split disabled in production (arcllm-proxy.py uses layer-split for np=16)

**Impact:** Row-split is a known escape path for layer-split's stability issues but is not ready for production. Developers may attempt to use it and hit untested code paths.

---

### 3. NAN_SNIFF Debug Mode Incomplete

**Issue:** Temporary debug mode (GGML_SYCL_NAN_SNIFF=1-3) for catching first NaN-producing op is still in code but incomplete.

**Files:**
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:9610-9617` — conditional sync + NaN check
- Used extensively in `scripts/bench/tests/test_q8_1_corruption.py` to narrow root cause

**Issues:**
- Only checks f32 tensors, misses F16 overflow (discovered after using it)
- Requires manual trigger (NAN_SNIFF env var) — easy to forget to disable
- Host round-trip copy for every tensor is 40-50% throughput penalty
- Only intended for debugging, should be removed before production

**Impact:** Risk of accidentally shipping with NAN_SNIFF enabled. Should be gated behind #ifdef or config-time flag, not runtime env var.

---

## Missing Critical Features

### 1. No Production Monitoring/Observability

**Issue:** llama-server has no built-in metrics, logging, or crash recovery.

**Files:**
- `scripts/arcllm-proxy.py` — reverse proxy with no error tracking
- `scripts/arcllm-server.sh` — wrapper script with basic logging

**Gaps:**
- No request tracing (can't correlate crash to specific prompt)
- No GPU health monitoring (crash leaves card in DEVICE_LOST)
- No automatic restart on crash (would need systemd or supervisor)
- No request queuing (concurrent requests beyond np limit are dropped)
- No auth/rate limiting (reverse proxy is open)

**Impact:** Production deployment would require external monitoring. Crashes are silent to the application. Users have no way to know if their inference failed or succeeded.

---

### 2. No Cross-GPU Request Load Balancing

**Issue:** Layer-split assigns all slots to all GPUs (tensor_split=1,1,1). No way to assign different requests to different GPU pairs.

**Files:**
- `scripts/arcllm-proxy.py:46-56` — model registry with fixed np only

**Gap:** With 3x A770, you could theoretically run 2x1GPU servers (1 per GPU) instead of 1x3GPU for better isolation. But llama-server doesn't support this config.

**Impact:** Cannot scale horizontally. Forced to buy more GPUs to increase np. Row-split would help here (each request uses 1/3 of VRAM) but row-split isn't production-ready.

---

### 3. No KV Cache Compression at Inference Time

**Issue:** KV cache uses F16 by default. Some quantization (Q8_0) is available but not widely tested.

**Files:**
- `scripts/arcllm-proxy.py:89,106,140` — -ctk and -ctv flags set in model configs
- No Q4_0 (4-bit) KV cache option (only Q8_0)

**Gap:** At 30k+ context and np=16, KV cache dominates VRAM. Going from F16 to Q4_0 would cut it 4x, enabling context doubling or higher parallelism.

**Impact:** Context is capped at 32-65k. Longer documents require external summarization. Can't run np>4 on GLM-4.7 MoE (25GB KV cache for np=4 × 64k context fills VRAM).

---

## Test Coverage Gaps

### 1. Row-Split Event-Based Sync Untested at Scale

**Issue:** GGML_SYCL_ROW_EVENTS=1 is implemented but never run at np≥2 in actual benchmarks.

**Files:**
- `scripts/bench/tests/test_row_split.py:208-251` — test suite defined but marked "WIP"
- Test infrastructure: `scripts/bench/tests/test_row_split.py` has np=2 and np=16 tests but they were not run successfully as of 2026-03-17

**Missing Tests:**
- np=2 × 1024 tokens (the crash workload with event sync)
- np=16 × 50 tokens (throughput comparison with event sync)
- Long-running stability (>5 min, >10k tokens/slot)

**Impact:** If event-based sync works, it could be the fix for layer-split crashes. But we have no proof because tests were never completed. Development is blocked waiting for these results.

---

### 2. FUSED_MMQ Crashes Not Fully Characterized

**Issue:** FUSED_MMQ enables +25% throughput but crashes at np≥2 × 1024 tokens (same crash as without it). Scope of the crash unclear.

**Files:**
- `scripts/bench/tests/test_fused_mmq.py:262-281` — production tests defined
- `scripts/bench/tests/test_mmvq_nan.py:87-143` — crash characterization (some results missing)

**Missing Clarity:**
- Does FUSED_MMQ make the crash worse or same?
- np=4 × 1024 without thinking (should be safe per test_q8_1_corruption.py line 493)?
- Crash threshold: is it token count, batch size, or duration?

**Impact:** Can't safely enable FUSED_MMQ in production (arcllm-proxy.py line 38 has FUSED_MMQ="1" but it's experimental). If users hit np≥2 long generation, deployment is at risk.

---

## Security Concerns

### 1. Reverse Proxy Without Authentication

**Issue:** arcllm-proxy.py listens on 0.0.0.0 with no auth, rate limiting, or request validation.

**Files:**
- `scripts/arcllm-proxy.py:200+` — HTTP server handler (full read needed)

**Risks:**
- Anyone on network can send unlimited requests
- No rate limiting prevents DOS
- No request size limits (could OOM server)
- Model loading is synchronous; slow loading leaves endpoint unresponsive

**Impact:** Not suitable for internet-facing deployment. Even internal deployment should have network isolation (firewall rules).

---

### 2. Model Paths Hardcoded

**Issue:** Model registry is built into arcllm-proxy.py with absolute paths. No way to load arbitrary models or control which models users can request.

**Files:**
- `scripts/arcllm-proxy.py:46-151` — static registry

**Risk:** Adding new models requires code changes and restart. Can't delegate model management to non-engineers.

**Impact:** Operational burden. Users requesting new models must wait for deployment cycle.

---

## Recommendations

**Immediate (blocking production):**
1. Resolve Level Zero in-order queue issue (Intel driver update or per-layer sync strategy)
2. Complete row-split event-based sync testing (validate it fixes crashes)
3. Characterize FUSED_MMQ crash boundary (np/tokens thresholds)

**Short-term (next sprint):**
1. Implement event-based KV cache pool (prevent buffer reuse races)
2. Add F16 bounds-checking in flash attention paths
3. Complete row-split performance optimization (eliminate remaining host waits)

**Medium-term:**
1. Add request tracing and GPU health monitoring to proxy
2. Implement Q4_0 KV cache option (reduce memory usage 4x)
3. Delete dead MMQ code (reduce maintenance)
4. Remove NAN_SNIFF debug mode (gate behind compile flag)

**Long-term:**
1. Multi-GPU request load balancing (horizontal scaling)
2. External model registry (decouple model management from proxy)
3. Production auth/rate limiting (security hardening)

---

*Concerns audit: 2026-03-17*
