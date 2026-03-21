# Known Failure Modes — Intel Arc A770 SYCL / Level Zero

- **Created:** 2026-03-18
- **Sources:** scripts/bench/tests/ docstrings, project investigation logs

## 1. Level Zero In-Order Queue Overlap (CRITICAL)

**Symptom:** `UR_RESULT_ERROR_DEVICE_LOST`, NaN in tensor outputs, GPU hang.

**Root cause:** Level Zero runtime overlaps kernel execution within in-order SYCL queues. Despite `sycl::property::queue::in_order()`, L0 batches kernel submissions into command lists and may execute kernels concurrently when it determines they have no data dependencies. Consecutive graph operations sharing buffers get overlapped, causing data races.

**Trigger conditions:**
- 2+ parallel slots (batch > 1)
- 500+ tokens per slot (enough compute depth for overlap to occur)
- Model-size-independent (reproduces on 0.6B, 32B, MoE)

**Evidence:**
- `stream->wait()` after every MUL_MAT op FIXES crash but costs ~40% throughput
- `NAN_SNIFF=3` (sync after every op) FIXES crash
- `NAN_SNIFF=2` (sync before MUL_MAT only) is FLAKY
- `submit_barrier()` alone does NOT fix it (L0 ignores for ordering)

**Workarounds (tested):**

| Approach | Single GPU | Multi-GPU MoE | Throughput Cost |
|----------|-----------|---------------|-----------------|
| SYNC_EVERY=1 (every MUL_MAT) | PASS | CRASH | ~40% |
| SYNC_EVERY=3 | PASS | CRASH | ~25% |
| FORCE_SYNC=1 (every graph op) | PASS | PASS (too slow) | ~60% |
| Immediate cmdlists | PASS (slow) | PASS (too slow) | ~50% + 4x VRAM |
| Row-split + event sync | TBD | TBD | Target: ~0% |

**Test:** `test_q8_1_corruption.py`, `test_mmvq_nan.py`

**Status:** Row-split with OOO queues + event-based sync is the production fix path. Layer-split has no zero-cost workaround.

---

## 2. MMVQ NaN Corruption

**Symptom:** `raw=nan scaled=nan q6s=[nan,nan]` in MMVQ debug output, then `UR_RESULT_ERROR_DEVICE_LOST`.

**Root cause:** Same as #1 (L0 in-order queue overlap). Initially appeared to be MMVQ-specific but investigation traced it to activation data being corrupted before any matmul kernel reads it.

**Crash log pattern:**
```
ggml_sycl_op_mul_mat_vec_q: row-debug attn_q0.mmvq_blocks_consume
    dev=0 col=1 row_low=0 row_high=8192 e=-1..-1
    raw=nan scaled=nan q6s=[nan,nan] q8ds=[nan,nan]
level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
Exception caught at file:mmvq.cpp, line:1995
```

**What was ruled out:**
- OOM (KV cache pre-allocated)
- SYCL graph replay
- Flash attention (crashes with -fa off)
- Fused kernel (crashes with fused off)
- Cross-device issue (crashes on single GPU)
- Thinking-specific (crashes with reasoning_budget=0)

**Test:** `test_mmvq_nan.py`

---

## 3. Q8_1 Activation Quantization Race

**Symptom:** `ds=[nan,nan] q=[0,0,0,0]` — NaN scales AND zero quants in Q8_1 data.

**Root cause:** Part of #1. The Q8_1 activation buffer shows corruption BEFORE any kernel reads it, because L0 overlaps the quantization kernel with subsequent operations.

**Investigation trace:**
1. Initially suspected pool buffer reuse race → stream sync didn't fix
2. Suspected cross-device memory visibility → single GPU also crashes
3. Suspected F16 overflow in CPY ops → found and fixed, but deeper bug remained
4. Finally traced to L0 overlapping kernels within in-order queues

**Test:** `test_q8_1_corruption.py`

---

## 4. Row-Split DMMV Fallback (0.6 t/s, GPUs Idle)

**Symptom:** Row-split mode runs at 0.6 t/s with GPUs idle at 600 MHz (base clock). CPU-bound.

**Root cause:** Row-split disabled MMVQ/MMQ for split tensors as a workaround for a Q8_1 ds half2 visibility bug across devices. This forces DMMV (CPU-computed matrix-vector multiply), which is 30x slower.

**Fix:** `GGML_SYCL_ROW_ALLOW_MMVQ=1` re-enables GPU kernels for split tensors.

**Test:** `test_row_split.py` (q4km_np1_100tok)

---

## 5. Row-Split Merge DEVICE_LOST

**Symptom:** `UR_RESULT_ERROR_DEVICE_LOST` at `row_split_merge_dst` after MMVQ kernel completes correctly.

**Root cause:** L0 doesn't honor in-order queue for cross-device merge reads. The MMVQ kernel output is correct, but `dev2dev_memcpy` reads stale data because the merge copy starts before the kernel has flushed to global memory.

**Fix:** `stream->wait()` on non-main device streams before merge. This works but is slow (0.47 t/s). Event-based sync is the target fix.

**What doesn't work:**
- `SYNC_EVERY=1` (only syncs within MUL_MAT op, not cross-device)
- `FORCE_SYNC=1` (only syncs main device stream)

**Test:** `test_q8_1_corruption.py` (q4km_row_short), `test_row_split.py`

---

## 6. Event-Based Sync Deadlock with Batched Command Lists

**Symptom:** Server hangs indefinitely when using OOO queues + `depends_on()` with `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0`.

**Root cause:** In batched command list mode, command batches are never flushed when another queue polls for an event. Cross-queue `depends_on()` waits forever for an event that's in an unflushed command batch on the source queue.

**Fix:** `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` is REQUIRED for event-based sync.

**Infrastructure guard:** INFRA-01 assertion in `ggml_check_sycl()` aborts at startup if ROW_EVENTS=1 without IMMEDIATE_COMMANDLISTS=1.

**Test:** `test_infra.py` (row_events_no_cmdlist), `test_row_split.py` (events_q4km_np1_batched)

---

## 7. SYCL Graph Recording Crash

**Symptom:** `SIGABRT` at server startup when `GGML_SYCL_DISABLE_GRAPH=0`.

**Root cause:** Graph recording doesn't support some operations used in the compute graph.

**Workaround:** Always run with `GGML_SYCL_DISABLE_GRAPH=1`.

**Test:** `test_q8_1_corruption.py` (06b_2x_1024_graph)

---

## 8. Immediate Command Lists VRAM Explosion

**Symptom:** VRAM usage jumps from 7 GB to 27 GB when using immediate command lists.

**Root cause:** L0 immediate mode allocates more staging buffers internally.

**Impact:** Reduces available VRAM for KV cache and model weights.

**Test:** `test_q8_1_corruption.py` (glm47_np4_500tok_imm)

---

## 9. XMX (joint_matrix) Compiler Crash

**Symptom:** `IGC Internal Compiler Error: floating point exception` when using `joint_matrix` with sub_group_size 16 or 32.

**Root cause:** Intel Graphics Compiler (IGC) bug. XMX on Arc A770 requires sub_group_size=8. Sizes 16 and 32 crash the compiler.

**Impact:** XMX path must use sg=8, separate from llama.cpp's `WARP_SIZE=16`. dp4a path can stay at sg=16.

**Performance:** XMX at pp16 = 24.72 t/s vs dp4a = 141.59 t/s (6x SLOWER due to data staging overhead at small batch sizes).

**Test:** `test_fused_mmq.py` (fused_xmx_pp128)

---

## 10. F16 Overflow in Copy Operations

**Symptom:** `-inf` values (F16 `0xfc00`) appear at graph node 26 (early CPY operation).

**Root cause:** F32 values > 65504 are cast to F16 without clamping, producing -inf. This -inf propagates through the entire computation.

**Fix:** Added F16 clamping to `cpy_1_f32_f16()` and `binbcast safe_narrow<>`.

**Unclamped paths found in flash attention:**
1. `fattn-common.hpp:347-361` — `quantize_q8_1_to_shared()` inline Q8_1 quantization
2. `fattn-tile.hpp:648-665` — `KQ_max_scale = exp(KQ_max - KQ_max_new)` to F16
3. `fattn-vec.hpp:263` — Q values loaded into F16 registers

All triggered when `SYCL_FAST_FP16` is defined (which it is — common.hpp:39).

**Test:** `test_q8_1_corruption.py`

---

## 11. JIT Cache Corruption After DEVICE_LOST

**Symptom:** Server fails to start or produces incorrect results after a previous DEVICE_LOST crash.

**Root cause:** Level Zero JIT-compiled kernel cache may be corrupted by GPU reset during DEVICE_LOST.

**Fix:** Flush corrupt cache, restore from NVMe backup:
- Cache location: `~/.cache/neo_compiler_cache/` (default) or custom path
- Bench framework saves known-good cache to `/home/ryan/llm-stack/cache/neo_compiler_cache`
- After DEVICE_LOST: flush → restore → restart

**Impact on iteration:** Cold JIT compile = ~114s server start. Cached = 3-18s.

---

## 12. Q8_0 Bandwidth Ceiling

**Symptom:** Q8_0 model maxes out at 3.3 t/s regardless of optimization.

**Root cause:** Q8_0 (8 bits/weight) saturates PCIe + VRAM bandwidth across 3 GPUs. With 9.5 GB model spread across 3x16 GB cards, each GPU reads ~3.2 GB/token. Theoretical ceiling ~59 t/s on single A770 but multi-GPU PCIe becomes bottleneck.

**Not a bug** -- hardware bandwidth limit.

**Test:** `test_baseline.py` (q8)

---

## Summary Table

| # | Failure Mode | Severity | Status | Workaround |
|---|-------------|----------|--------|------------|
| 1 | L0 in-order queue overlap | CRITICAL | Active | SYNC_EVERY=3 (single GPU), event sync (multi-GPU in progress) |
| 2 | MMVQ NaN | CRITICAL | Same as #1 | Same as #1 |
| 3 | Q8_1 activation race | CRITICAL | Same as #1 | Same as #1 |
| 4 | DMMV fallback | HIGH | Fixed | ROW_ALLOW_MMVQ=1 |
| 5 | Row-split merge DEVICE_LOST | HIGH | Fixed (slow) | stream->wait() before merge; event sync target |
| 6 | Event deadlock + batched cmdlists | HIGH | Fixed | IMMEDIATE_COMMANDLISTS=1 required |
| 7 | SYCL graph SIGABRT | MEDIUM | Permanent workaround | DISABLE_GRAPH=1 |
| 8 | Immediate cmdlist VRAM explosion | MEDIUM | Known | Use batched mode when possible |
| 9 | XMX compiler crash | LOW | Known | Use dp4a instead (also faster at small batch) |
| 10 | F16 overflow in CPY | MEDIUM | Fixed | Clamping added |
| 11 | JIT cache corruption | LOW | Fixed | Cache backup/restore in bench framework |
| 12 | Q8_0 bandwidth ceiling | INFO | Hardware limit | Use Q4_K_M |
