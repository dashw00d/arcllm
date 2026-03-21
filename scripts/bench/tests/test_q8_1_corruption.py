"""Q8_1 activation quantization corruption — root cause of MMVQ NaN crash.

## Root Cause Analysis (2026-03-16)

The MMVQ NaN crash (test_mmvq_nan.py) is NOT a kernel bug. The Q8_1 activation
data is corrupted BEFORE any matmul kernel reads it. Evidence:

    ds=[nan,nan] q=[0,0,0,0]

- All-zero quants means quantization either never ran or wrote to wrong buffer
- NaN scales means the ds half2 field was never properly written
- This is upstream of both MMVQ and fused kernel — both would read garbage

## Buffer Lifecycle (ggml-sycl.cpp)

The Q8_1 buffer goes through this path:

1. **Allocation** (line 3745):
   `dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), src1_ddq_bytes)`
   Pool allocator (`ggml_sycl_pool_leg`) maintains 256-buffer reuse pool.
   Best-fit search for reusable buffers (lines 2027-2041).
   **No mutex protection on pool allocation/reuse.**

2. **F32 source** (line 3722):
   When `src1_on_device && src1_is_contiguous`:
   `dev[i].src1_ddf = (float *) src1->data`
   Points directly to tensor data — NOT a pool copy. If the tensor is
   overwritten by another operation before quantization reads it, corruption.

3. **Quantization** (line 4442):
   `quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ...)`
   Submitted async to SYCL stream. Within a single stream, operations are
   ordered. But if src1->data is modified by operations on a DIFFERENT stream
   (different device), the f32 source can be stale/corrupted.

4. **Matmul** (line 4575):
   `op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, ...)`
   Reads Q8_1 data. Same stream as quantization, so ordered WITHIN stream.

5. **Buffer free** (RAII):
   `src1_ddq_alloc` destructor returns buffer to pool when `dev_data` scope ends.
   Pool may reuse this buffer for the next operation, while async kernels from
   THIS operation are still in flight on the stream.

## The Race Condition

Pool buffer reuse is the prime suspect:
- Slot A allocates Q8_1 buffer from pool → gets buffer at address X
- Slot A submits quantization kernel (async) to stream
- Slot A's ggml_sycl_op_mul_mat() returns → RAII frees buffer X back to pool
- Slot B calls ggml_sycl_op_mul_mat() → pool gives buffer X again
- Slot B starts writing new Q8_1 data to X while Slot A's kernel is still reading X
- Result: Slot A reads partially-overwritten Q8_1 data = NaN

This only triggers with:
- Multiple slots (need concurrent pool users)
- Long sequences (deeper SYCL command graphs = more async overlap)
- >500 tokens per slot (enough compute depth for buffers to "lap" each other)

## Alternative Hypothesis: Cross-Device src1->data Race

When `src1_on_device=true`, `src1_ddf` points directly to `src1->data` (line 3722).
In multi-slot batch processing, all slots' activations may be in the same tensor.
If the graph executor modifies `src1->data` for the next layer while the current
layer's quantization kernel is still reading from it... corruption.

This would explain why the bug is on `attn_q0` specifically — attention Q is early
in the layer computation, and the previous layer's output (which IS src1) might be
getting overwritten by the next batch's processing.

## Band-Aid Already Exists

`ggml_sycl_repair_q8_1_scales()` (line 751) copies Q8_1 data to host, detects
NaN/zero scales, recomputes from f32 source, copies back. But it only runs on
`blk.0.attn_q.weight` when `split=true` (row-split mode). In our layer-split mode,
it never runs. And it's a host round-trip per matmul — terrible for performance.

## Key Code Locations

- Pool allocator: ggml-sycl.cpp lines 1987-2096 (ggml_sycl_pool_leg)
- Q8_1 buffer alloc: ggml-sycl.cpp line 3745
- F32 source binding: ggml-sycl.cpp line 3722
- Quantization call: ggml-sycl.cpp line 4442
- Matmul op call: ggml-sycl.cpp line 4575
- Repair function: ggml-sycl.cpp line 751
- quantize_q8_1 kernel: quantize.hpp lines 27-56 (the SYCL kernel itself)
  - Line 89: wi_id==0 writes ds field — potential intra-kernel race

## Fix Strategies (from least to most invasive)

1. **Stream sync after quantization** — add `stream->wait()` between quantize and
   matmul. Ensures Q8_1 data is fully written. Costs ~5-15% throughput from sync.

2. **Per-slot Q8_1 buffers** — don't reuse across slots. Allocate N buffers for N
   slots, never return to pool until all slots complete. Memory cost: N × buffer_size.

3. **Pool reference counting** — track in-flight kernels per buffer. Don't reuse
   until refcount drops to 0. Requires integrating with SYCL event tracking.

4. **Fence-based pool** — attach SYCL events to pool buffers. Before reuse, wait
   on the event from the last operation that used this buffer.

5. **Eliminate Q8_1 entirely** — modify fused kernel to accept f32 activations
   directly. No quantization step = no quantization buffer = no race. Costs compute
   (f32 dot products instead of int8 dp4a) but eliminates the entire bug class.

## quantize_q8_1 Kernel Internals (quantize.hpp)

The kernel processes QK8_1=32 elements per work item:
```
1. Read 32 f32 values from src
2. Compute amax = max(|values|)
3. d = amax / 127.0f (scale)
4. Quantize: qs[i] = round(value / d) as int8
5. Write ds = half2(d, d * sum(qs)) — ONLY work item 0 writes this
```

Step 5 is suspicious: only wi_id==0 writes the `ds` field. If the work group size
doesn't evenly divide the number of blocks, the last work group may have wi_id==0
from a different logical block writing to the wrong ds address.

## Xe Architecture Specifics

Arc A770 SIMD execution model:
- Sub-group (warp) = 16 threads (WARP_SIZE=16 in build)
- SLM coherence: 64KB shared local memory per subslice
- L1 cache: per-EU, NOT coherent across EUs within a subslice
- The Q8_1 ds half2 write visibility bug (documented elsewhere) may be related:
  half2 stores from one EU may not be visible to other EUs reading the same
  cache line, especially under high occupancy with many concurrent sub-groups

## Probe Results (2026-03-16)

| Test | Result | What it rules out |
|------|--------|-------------------|
| 16x_512 + stream sync | CRASH | NOT async pool reuse |
| single_gpu 4x_1024 | CRASH | NOT cross-device |
| 2x_1024 thinking | CRASH | NOT high slot count |
| 16x_200 no thinking | PASS (18.5 t/s) | Needs HIGH per-slot token count |
| 1x_1024 thinking | PASS | Needs 2+ slots |

Narrowed: 2+ slots AND >~500 tokens/slot triggers corruption. Single device,
single stream, sync doesn't help.

| Ruled out | Evidence |
|-----------|----------|
| Async pool reuse | Stream sync after quantize doesn't fix |
| Cross-device | Crashes on single GPU too |
| High slot count | Crashes with just 2 slots |
| Flash attention | Crashes with -fa off |
| Thinking-specific | Crashes with reasoning_budget=0 too |
| SYCL graph | Crashes with graph=ON and graph=OFF |

Remaining suspects:
1. **NaN in f32 activations** — the `f=[nan,nan,nan,nan]` is in the SOURCE data
   before Q8_1 quantization. Something upstream produces NaN.
2. Always `attn_q0` (block 0 attention Q), always `e=224..255` (last 32 of Q6_K block)
3. Always `col=0` and `col=1` — the first two batch columns

The NaN likely originates from:
- Layer norm overflow at long sequences (unlikely — 500 tokens isn't extreme)
- RoPE position encoding computation with multi-slot KV cache offsets
- Attention score computation (QK^T / sqrt(d)) overflow before softmax
- A SYCL kernel bug in one of the upstream operations (element-wise, norm, etc.)
  that only manifests with batch>1 and high iteration count

## Sync Level Results (2026-03-17)

| Sync level | np=2 1024tok | np=16 512tok | Speed (np=2) |
|------------|-------------|--------------|-------------|
| None | CRASH | CRASH | 5.9 t/s |
| Every 64 ops (NAN_SNIFF=1) | — | CRASH | — |
| Before every MUL_MAT (NAN_SNIFF=2) | FLAKY | CRASH | 6.6 t/s |
| After EVERY op (NAN_SNIFF=3) | **PASS** | untested | 6.2 t/s |

ONLY full serialization (sync after every op) reliably prevents the crash.
Per-MUL_MAT sync is not enough — the race is between NON-matmul ops.
This means the bug is in how SYCL command submission reorders or overlaps
operations within a single stream's command list. The Level Zero runtime
may be submitting work out of order or overlapping kernels that share data.

## ROOT CAUSE FOUND (2026-03-17)

NAN_SNIFF=3 at np=16 caught it:

    graph[737] op=MUL_MAT name=ffn_gate-63
    idx=2/358400 val=-nan prev=-6.90008e+08
    src0=blk.63.ffn_gate.weight(q4_K)
    src1=ffn_norm-63(f32)

**The first NaN is in FFN gate at layer 63 (LAST layer).** The previous value
is -6.9e+08 — hidden states have grown to ~7e8 by the last layer through
residual connection accumulation. After RMS norm, the values are still huge.
The FFN gate matmul overflows during Q4_K dequant × large f32 activations.

This is a NUMERICAL STABILITY issue, not a driver bug. The residual connections
accumulate differently when processing multiple sequences simultaneously
(batch>1), leading to larger hidden state magnitudes. At batch=1, the values
stay within range. At batch>=2 with long sequences, they overflow.

Full serialization (NAN_SNIFF=3) "fixes" it at np=2 because the serialization
changes the memory access pattern and numerical accumulation order slightly,
keeping values just below the overflow threshold. At np=16, even with full
serialization, the values are too large and overflow anyway.

## DEEPER ROOT CAUSE (2026-03-17)

Enhanced sniffer with F16 overflow detection found the REAL first failure:

    graph[26] op=CPY name=(copy) idx=1/512 raw=0xfc00 (F16 -inf)
    src0=SYCL0#leaf_14#0(f32)

**F16 overflow at graph node 26** — a CPY (copy) operation converting F32 to F16
at the VERY START of the compute graph. This is NOT gradual accumulation through
64 layers. The overflow happens immediately when F32 values >65504 are cast to F16.

The earlier "layer 63 ffn_gate" finding was a RED HERRING — that was only checking
F32 tensors, missing the F16 overflow that happens much earlier.

The copy at graph[26] is likely the token embedding or KV cache copy. With batch=16,
some values in the F32 source exceed F16 range and get truncated to -inf. This -inf
then propagates through the entire computation.

## Post-CPY-Fix Sniffer Results (2026-03-17)

Added F16 clamping to cpy_1_f32_f16() and binbcast safe_narrow<>.
Re-ran NaN sniffer (NAN_SNIFF=3) at np=2:

    graph[2] op=MUL_MAT name=Qcur-0 type=f32
    idx=0/245760 val=nan amax=2.70758e+07
    src0=blk.0.attn_q.weight(q4_K) src1=attn_norm-0(f32)

**First NaN is now in layer 0 attention Q MATMUL, not in CPY.** The CPY fix
prevented the F16 overflow, revealing the DEEPER bug: the Q4_K matmul kernel
itself produces NaN from valid F32 inputs at batch>1.

Magnitude trace (l_out per layer):
  l_out-2: 157 → l_out-5: 217 → l_out-6: 16,600 (100x jump!) → stable at ~17k

The 100x magnitude jump at layer 6 is suspicious — may be a specific layer's
weights causing extreme amplification. But the NaN at layer 0 means the bug
is in the matmul kernel, not magnitude accumulation.

**UPDATE (2026-03-17 evening):** The "matmul produces NaN from valid inputs"
diagnosis was WRONG. The matmul kernel is fine — the Level Zero runtime is
overlapping kernel execution within in-order queues, causing data races between
consecutive graph operations. Adding stream->wait() after every MUL_MAT op
FIXES the crash on all models (0.6B Q8_0, 32B Q4_K, GLM-4.7 MoE) but costs
~40% throughput. Need a lighter synchronization approach.

## Flash Attention Unclamped F16 Paths (2026-03-17)

Code audit found CRITICAL unclamped F32→F16 conversions in flash attention:

1. **fattn-common.hpp:347-361** — `quantize_q8_1_to_shared()` does inline Q8_1
   quantization with `d = amax / 127` and `make_half2(d, sum)` — NO CLAMP.
   This is a SEPARATE path from quantize.hpp (which IS clamped). If Q values
   (from Qcur output) exceed 65504*127=8.3M, d overflows F16 → inf.

2. **fattn-tile.hpp:648-665** — `KQ_max_scale = exp(KQ_max - KQ_max_new)` →
   `sycl::half2(KQ_max_scale, ...)` — exp() is unbounded, exp(11) > F16 max.
   When attention patterns shift between chunks, this overflows.

3. **fattn-vec.hpp:263** — `sycl::half2(tmp[i1].x(), tmp[i1].y())` — Q values
   loaded into F16 registers without bounds check. Large Q values → inf.

All three only trigger when SYCL_FAST_FP16 is defined (it IS — common.hpp:39).

**KEY QUESTION**: Does the crash happen WITHOUT flash attention? Run
`test_2x_1024_no_fa`. If it passes, these FA paths are the root cause.

## Quick-Kill Test Results (2026-03-17)

| # | Test | What it isolates | Result |
|---|------|-----------------|--------|
| 1 | test_2x_1024_no_fa | Flash attention | **CRASH 77s** — NOT flash attention |
| 2 | test_2x_1024_no_fused | Fused kernel vs generic | **CRASH 79s** — NOT fused kernel |
| 3 | test_np16_50tok | Batch size vs step count | **PASS 17.3 t/s** — needs step count |
| 4 | test_06b_2x_1024 | Model size dependency | **CRASH 25s** — model-size-independent |

Ruled out: flash attention, fused kernel, model size, batch size alone.
Required: 2+ slots AND enough decode steps (~500+ tokens/slot on 32B, ~25s on 0.6B).

## Level Zero Command Overlap Bug (2026-03-17)

**ROOT CAUSE CONFIRMED: Level Zero runtime overlaps kernel execution within
in-order SYCL queues.** Despite queues being created with `sycl::property::queue::in_order()`,
the L0 runtime batches kernel submissions into command lists and may execute
kernels concurrently when it determines they have no data dependencies.

Evidence:
- `stream->wait()` after every MUL_MAT op → FIXES crash (0.6B, 32B, GLM-4.7)
- NAN_SNIFF=3 (sync after every op) → FIXES crash
- NAN_SNIFF=2 (sync before MUL_MAT only) → FLAKY (not enough coverage)
- In-order queue should serialize, but L0 optimizes within command lists

**FIX ATTEMPT 1: stream->wait() after every MUL_MAT op (ggml-sycl.cpp:4581)**
- 0.6B: 2/2 ok, 1629 tok, 29.2 t/s (was crash at 25s)
- 32B Q4_K_M: 2/2 ok, 2048 tok, 6.6 t/s (was crash at 77s)
  - 6.6 t/s is ~62% of previous 10.6 t/s — sync after EVERY matmul is too costly
  - Qwen3-32B has 64 layers × ~7 matmuls/layer = ~448 stream->wait() per token
- GLM-4.7 MoE: too slow to complete in 300s timeout (MoE has even more matmuls)

**FIX ATTEMPT 2: GGML_SYCL_FORCE_SYNC=1 (sync before+after every graph op)**
Added env var gated sync in graph_compute loop (ggml-sycl.cpp:9610-9617).
Same mechanism as NAN_SNIFF=3 but without the NaN check copy-to-host overhead.
- 0.6B: 2/2 ok, 1808 tok, 21.5 t/s (slower than MUL_MAT-only sync)
- GLM-4.7 np=4: 0/4, timed out at 270s — too many ops/token in MoE model

**FIX ATTEMPT 3: submit_barrier() after MUL_MAT**
Device-side barrier (no host stall): `stream->ext_oneapi_submit_barrier()`
- 0.6B: CRASH at 22s — L0 ignores submit_barrier for ordering

**FIX ATTEMPT 4: Periodic MUL_MAT sync (GGML_SYCL_SYNC_EVERY=N)**
Binary search on sync frequency (0.6B single-GPU first, then GLM 3-GPU):

| SYNC_EVERY | 0.6B 1-GPU | t/s | GLM-4.7 3-GPU | t/s |
|------------|-----------|-----|---------------|-----|
| 1 | PASS | 29.2 | PASS (32B 6.6) | slow |
| 2 | PASS | 39.0 | CRASH 181s | — |
| 3 | PASS | 41.4 | CRASH 170s | — |
| 4 | CRASH | — | — | — |
| 8 | CRASH | — | — | — |

MUL_MAT-only sync works for single-GPU but NOT for multi-GPU MoE.
The race on multi-GPU involves non-MUL_MAT ops (MoE gating, expert routing).

**FIX ATTEMPT 5: SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1**
Changes L0 from batched command lists to immediate submission per kernel.
- GLM-4.7 np=4: No crash, but 0/4 completed in 300s — very slow
  - VRAM usage jumped from 7GB to 27GB (L0 immediate allocs more staging buffers)
  - No throughput benefit over batched + sync

**CONCLUSION:**
- Single-GPU: SYNC_EVERY=3 works at 41.4 t/s (vs 29.2 for every-1, crash without)
- Multi-GPU MoE: SYNC_EVERY=1..3 all crash. Only FORCE_SYNC=1 (every graph op) works,
  but it's too slow for production.
- Graph execution (GGML_SYCL_DISABLE_GRAPH=0): crashes at startup (ops not supported)

The multi-GPU case needs sync between ALL op types, not just MUL_MAT. The MoE model
has expert routing, gating, top-k selection ops that also trigger the L0 overlap.

**STATUS:** Single-GPU fixed (SYNC_EVERY=3). Multi-GPU MoE blocked — needs either:
1. FORCE_SYNC with much better throughput (unlikely without L0 runtime fix)
2. Per-op event chaining in the graph loop (Approach B from plan)
3. Intel compute-runtime update that fixes in-order queue guarantee
4. Different sync granularity — sync after each transformer LAYER instead of per-op

## JIT Cache Optimization (2026-03-17)

Server load time reduced from ~114s to ~3-18s by preserving the Level Zero
JIT-compiled kernel cache across test runs. Previously _flush_level_zero_cache()
deleted the cache after every test. Now:
- Cache is saved to /home/ryan/llm-stack/cache/neo_compiler_cache after first load
- After DEVICE_LOST crash: flush corrupt cache, restore from NVMe backup
- Result: 0.6B loads in 3s, 32B loads in 18s (vs 114s without cache)

## Row-Split Investigation (2026-03-17)

Row-split (`--split-mode row`) distributes matrix rows across all 3 GPUs.
Goal: use row-split's event-based cross-device sync to solve the L0 overlap crash.

### Issues Found

1. **DMMV fallback** — Row-split disables MMVQ/MMQ for split tensors (Q8_1 ds
   visibility bug workaround), forcing DMMV which is CPU-bound (0.6 t/s, GPUs idle).
   Fix: `GGML_SYCL_ROW_ALLOW_MMVQ=1` env var to skip the fallback.

2. **Block-0 repair overhead** — `row_force_local_q8_attn_qkv0` does synchronous
   host roundtrips (GPU→host→GPU) for every block-0 QKV matmul on non-owner devices.
   Fix: `ROW_ALLOW_MMVQ=1` also disables this repair path.

3. **ROW_PROBE debug spam** — ~1550 lines of unconditional fprintf/wait_and_throw
   probes in mmvq.cpp from previous investigation. Fix: reverted mmvq.cpp to upstream.

4. **MMVQ crashes with split tensors** — DEVICE_LOST at row_split_merge_dst.
   Root cause: L0 doesn't honor in-order queue for cross-device merge reads.
   The MMVQ kernel completes but the merge `dev2dev_memcpy` reads stale data.
   Fix: `stream->wait()` on non-main devices after kernel, before merge.
   This works but is slow (0.47 t/s) — need event-based sync instead.

### Row-Split Results Summary

| Config | Result | Speed |
|--------|--------|-------|
| DMMV (default) | PASS | 0.6 t/s (GPUs idle) |
| MMVQ (ROW_ALLOW_MMVQ) | CRASH | DEVICE_LOST |
| MMVQ + SYNC_EVERY=1 | CRASH | — |
| MMVQ + FORCE_SYNC=1 | CRASH | — (only syncs main device) |
| MMVQ + post-op stream->wait() | PASS | 0.47 t/s |
| MMVQ + pre+post sync (diag) | PASS | 0.4 t/s |

### Next Steps

Replace host-stall `stream->wait()` with event-based sync for the merge step.
Row-split already has `ext_oneapi_submit_barrier` infrastructure — wire it up
so non-main device kernels chain completion events to the merge memcpy.

## Test Plan

These tests probe specific aspects of the corruption to narrow root cause.
"""
from bench.base import BenchTest
from bench.config import BenchConfig

THINK_PROMPT = (
    "Think step by step. What are the top 5 most impactful inventions "
    "of the 20th century? Rank by long-term societal impact."
)

SHORT_PROMPT = "What is 2+2?"

FUSED = BenchConfig(model="q4km", timeout=600).with_flags(FUSED_MMQ="1")


class TestQ81Corruption(BenchTest):
    """Q8_1 activation corruption probes — narrowing root cause."""

    # ── Pool reuse probe ──────────────────────────────────────────────
    # If the bug is pool buffer reuse, smaller context (less VRAM pressure)
    # should make it harder to reproduce (pool has more headroom).

    def test_16x_512_small_ctx(self):
        """np=16 thinking, small context (8192) — less pool pressure."""
        self.run(FUSED.with_(
            name="q81_16x_512_ctx8k",
            prompt=THINK_PROMPT,
            n_parallel=16, concurrent=16, context=8192,
            max_tokens=512, reasoning_budget=512))

    def test_16x_512_large_ctx(self):
        """np=16 thinking, large context (32768) — more pool pressure."""
        self.run(FUSED.with_(
            name="q81_16x_512_ctx32k",
            prompt=THINK_PROMPT,
            n_parallel=16, concurrent=16, context=32768,
            max_tokens=512, reasoning_budget=512))

    # ── Slot count probe ──────────────────────────────────────────────
    # If the bug is cross-slot buffer collision, fewer slots should help.

    def test_2x_1024_thinking(self):
        """2 slots × 1024 tokens thinking — minimal concurrency."""
        self.run(FUSED.with_(
            name="q81_2x_1024",
            prompt=THINK_PROMPT,
            n_parallel=2, concurrent=2,
            max_tokens=1024, reasoning_budget=1024))

    def test_8x_1024_thinking(self):
        """8 slots × 1024 tokens thinking — medium concurrency."""
        self.run(FUSED.with_(
            name="q81_8x_1024",
            prompt=THINK_PROMPT,
            n_parallel=8, concurrent=8,
            max_tokens=1024, reasoning_budget=1024))

    # ── Stream sync probe ─────────────────────────────────────────────
    # FUSED_MMQ=1 adds stream->wait() after Q8_1 quantization.
    # If this fixes the crash, it confirms async buffer reuse as root cause.

    def test_16x_512_with_sync(self):
        """np=16 thinking, fused kernel with stream sync after quantize.
        If this passes (and without sync it crashes), the root cause is
        async buffer reuse — the matmul reads Q8_1 before quantize finishes."""
        self.run(FUSED.with_(
            name="q81_16x_512_sync",
            prompt=THINK_PROMPT,
            n_parallel=16, concurrent=16, context=32768,
            max_tokens=512, reasoning_budget=512))

    # ── Single GPU probe ──────────────────────────────────────────────
    # If the bug is cross-device memory visibility, single GPU should work.

    def test_single_gpu_4x_1024(self):
        """Single A770, 4 slots × 1024 tokens thinking.
        Uses 9B Q8_0 model (fits one GPU). If this also crashes, the bug
        is NOT cross-device — it's within a single device's buffer management."""
        self.run(FUSED.with_(
            name="q81_1gpu_4x_1024",
            prompt=THINK_PROMPT,
            model="9b-q8", affinity="0", tensor_split="1",
            n_parallel=4, concurrent=4, context=8192,
            max_tokens=1024, reasoning_budget=1024))

    # ── No-thinking control ───────────────────────────────────────────
    # Thinking generates many more tokens per slot. Without thinking,
    # 200 tokens × 16 slots works fine. This confirms the bug needs
    # high per-slot token counts, not just high slot counts.

    def test_16x_200_no_think(self):
        """np=16 × 200 tokens, NO thinking — known to work."""
        self.run(FUSED.with_(
            name="q81_16x_200_nothink",
            prompt=SHORT_PROMPT,
            n_parallel=16, concurrent=16, context=32768,
            max_tokens=200, reasoning_budget=0))

    # ── KV quant probe ────────────────────────────────────────────────
    # Q8_0 KV cache uses different memory layout. If the bug is in
    # KV cache read → activation path, different KV quant might help.

    # ── NaN sniffer — find the FIRST op that produces NaN ───────────
    # GGML_SYCL_NAN_SNIFF=1 checks every op's f32 output for NaN/Inf.
    # Slow (copies every tensor to host), but finds the exact op that
    # first produces NaN. Run with smallest reproducer (2 slots × 1024).

    def test_2x_1024_nan_sniff_full(self):
        """NaN sniffer FULL: sync after every op + NaN check.
        NAN_SNIFF=2. Very slow. Finds exact first NaN-producing op.
        RESULT: PASSES at 4.0 t/s — full serialization prevents crash."""
        self.run(FUSED.with_flags(FUSED_MMQ="1", NAN_SNIFF="2").with_(
            name="q81_2x_1024_sniff_full",
            prompt=THINK_PROMPT,
            n_parallel=2, concurrent=2, context=32768,
            max_tokens=1024, reasoning_budget=1024,
            timeout=900))

    def test_2x_1024_nan_sniff_matmul(self):
        """NaN sniffer TARGETED: sync before MUL_MAT ops only.
        NAN_SNIFF=1. Much less overhead. Tests if sync before matmul is enough."""
        self.run(FUSED.with_flags(FUSED_MMQ="1", NAN_SNIFF="1").with_(
            name="q81_2x_1024_sniff_mm",
            prompt=THINK_PROMPT,
            n_parallel=2, concurrent=2, context=32768,
            max_tokens=1024, reasoning_budget=1024,
            timeout=600))

    def test_16x_512_matmul_sync(self):
        """np=16, 512 tokens, sync before every MUL_MAT. Slow but stable?
        NAN_SNIFF=2. Reduced tokens to fit in timeout."""
        self.run(FUSED.with_flags(FUSED_MMQ="1", NAN_SNIFF="2").with_(
            name="q81_16x_512_mmsync",
            prompt=THINK_PROMPT,
            n_parallel=16, concurrent=16, context=32768,
            max_tokens=512, reasoning_budget=512,
            timeout=1800))

    def test_16x_512_kv_q8(self):
        """np=16 thinking, KV cache quantized to Q8_0."""
        self.run(FUSED.with_(
            name="q81_16x_512_kvq8",
            prompt=THINK_PROMPT,
            n_parallel=16, concurrent=16, context=32768,
            max_tokens=512, reasoning_budget=512,
            kv_quant="q8_0"))

    # ── Flash attention probe ─────────────────────────────────────────
    # NaN first appears in f32 activations (BEFORE Q8_1 quantization).
    # If flash attention has a numerical issue at long sequences with
    # multi-slot batching, disabling it should fix the crash.

    def test_2x_1024_no_fa(self):
        """2 slots × 1024 thinking, flash attention OFF.
        RESULT: CRASH at 77s — flash attention is NOT the root cause."""
        self.run(FUSED.with_(
            name="q81_2x_1024_nofa",
            prompt=THINK_PROMPT,
            n_parallel=2, concurrent=2, context=32768,
            max_tokens=1024, reasoning_budget=1024,
            flash_attn=False))

    # ── Long non-thinking probe ───────────────────────────────────────
    # Is it specifically thinking tokens, or just ANY long generation?

    def test_2x_1024_long_nothink(self):
        """2 slots × 1024 tokens, NO thinking — just long generation.
        If this crashes, the bug is about sequence length, not thinking."""
        self.run(FUSED.with_(
            name="q81_2x_1024_nothink",
            prompt="Write a very detailed 1000-word essay about the history of computing.",
            n_parallel=2, concurrent=2, context=32768,
            max_tokens=1024, reasoning_budget=0))

    # ── Quick-kill isolation tests (2026-03-17) ──────────────────────
    # Each test isolates ONE variable to narrow the root cause.
    # Run in order — first one that passes identifies the culprit.

    def test_np16_50tok(self):
        """np=16, only 50 tokens — batch size isolation.
        If this CRASHES: batch size alone triggers the bug (not step count).
        If this PASSES: step count / sequence length is required.
        RESULT: PASS — 50 tokens not enough to trigger crash even at np=16."""
        self.run(FUSED.with_(
            name="q81_np16_50tok",
            prompt=SHORT_PROMPT,
            n_parallel=16, concurrent=16, context=4096,
            max_tokens=50, reasoning_budget=0,
            timeout=90))

    def test_np1_2000tok(self):
        """np=1, 2000 tokens — step count isolation.
        If this CRASHES: single slot can overflow with enough steps.
        If this PASSES: multi-slot batching is required (confirms existing data)."""
        self.run(FUSED.with_(
            name="q81_np1_2000tok",
            prompt=THINK_PROMPT,
            n_parallel=1, concurrent=1, context=4096,
            max_tokens=2000, reasoning_budget=2000,
            timeout=90))

    def test_2gpu_2x_1024(self):
        """2 GPUs only (skip GPU2 PCIe 3.0 x8) — PCIe isolation.
        If this PASSES: GPU2's half-bandwidth x8 slot is involved.
        If this CRASHES: not a GPU2-specific issue."""
        self.run(FUSED.with_(
            name="q81_2gpu_2x_1024",
            prompt=THINK_PROMPT,
            affinity="0,1", tensor_split="1,1",
            n_parallel=2, concurrent=2, context=4096,
            max_tokens=1024, reasoning_budget=1024,
            timeout=90))

    def test_2x_1024_no_fused(self):
        """2 slots × 1024 thinking, fused kernel OFF — kernel isolation.
        If this PASSES: the fused kernel is the bug.
        If this CRASHES: the bug is in the generic MMVQ/GEMM path too.
        RESULT: CRASH at ~79s — bug is in generic path, not fused kernel."""
        self.run(BenchConfig(
            model="q4km", timeout=90,
            name="q81_2x_1024_nofused",
            prompt=THINK_PROMPT,
            n_parallel=2, concurrent=2, context=4096,
            max_tokens=1024, reasoning_budget=1024))

    # ── Fast reproducer with 0.6B model ──────────────────────────────
    # 800MB model loads in <10s on single GPU. Uses Q8_0 (no Q4_K avail).
    # If this crashes: bug is model-size-independent → pure kernel/driver bug.
    # If this passes: bug needs enough layers for magnitude accumulation.

    def test_06b_2x_1024(self):
        """0.6B Q8_0 on single GPU, 2 slots × 1024 — fast reproducer.
        RESULT (no sync): CRASH at ~25s. 3s load with JIT cache.
        RESULT (FORCE_SYNC=1): PASS 29.2 t/s — confirms L0 overlap is root cause.
        RESULT (MUL_MAT sync only): PASS 29.2 t/s — but too slow on larger models."""
        self.run(BenchConfig(
            model="0.6b-q8", timeout=90,
            name="q81_06b_2x_1024",
            prompt=THINK_PROMPT,
            affinity="0", tensor_split="1",
            n_parallel=2, concurrent=2, context=4096,
            max_tokens=1024, reasoning_budget=1024).with_flags(FORCE_SYNC="1"))

    def test_06b_16x_512(self):
        """0.6B Q8_0 on single GPU, 16 slots × 512 — max batch fast repro."""
        self.run(BenchConfig(
            model="0.6b-q8", timeout=90,
            name="q81_06b_16x_512",
            prompt=THINK_PROMPT,
            affinity="0", tensor_split="1",
            n_parallel=16, concurrent=16, context=4096,
            max_tokens=512, reasoning_budget=512).with_flags(FORCE_SYNC="1"))

    def test_06b_2x_1024_imm(self):
        """0.6B with immediate command lists — zero-overhead fix attempt.
        Not yet run. Tests if immediate cmdlists alone fix the overlap on 0.6B."""
        self.run(BenchConfig(
            model="0.6b-q8", timeout=90,
            name="q81_06b_2x_1024_imm",
            prompt=THINK_PROMPT,
            affinity="0", tensor_split="1",
            n_parallel=2, concurrent=2, context=4096,
            immediate_cmdlists=True,
            max_tokens=1024, reasoning_budget=1024))

    def test_06b_2x_1024_graph(self):
        """0.6B with SYCL graph execution — graph encodes explicit dependencies.
        RESULT: SIGABRT at startup — graph recording doesn't support some ops."""
        self.run(BenchConfig(
            model="0.6b-q8", timeout=90,
            name="q81_06b_2x_1024_graph",
            prompt=THINK_PROMPT,
            affinity="0", tensor_split="1",
            n_parallel=2, concurrent=2, context=4096,
            disable_graph=False,
            max_tokens=1024, reasoning_budget=1024))

    def test_06b_sync2(self):
        """0.6B with SYNC_EVERY=2 — sync every 2 MUL_MATs.
        RESULT: PASS 39.0 t/s — works on single GPU."""
        self.run(BenchConfig(
            model="0.6b-q8", timeout=90,
            name="q81_06b_sync2",
            prompt=THINK_PROMPT,
            affinity="0", tensor_split="1",
            n_parallel=2, concurrent=2, context=4096,
            max_tokens=1024, reasoning_budget=1024).with_flags(SYNC_EVERY="2"))

    def test_06b_sync4(self):
        """0.6B with SYNC_EVERY=4 — sync every 4 MUL_MATs.
        RESULT: CRASH at 25s — every 4 is not frequent enough."""
        self.run(BenchConfig(
            model="0.6b-q8", timeout=90,
            name="q81_06b_sync4",
            prompt=THINK_PROMPT,
            affinity="0", tensor_split="1",
            n_parallel=2, concurrent=2, context=4096,
            max_tokens=1024, reasoning_budget=1024).with_flags(SYNC_EVERY="4"))

    def test_06b_sync3(self):
        """0.6B with SYNC_EVERY=3 — sync every 3 MUL_MATs.
        RESULT: PASS 41.4 t/s — optimal for single GPU 0.6B."""
        self.run(BenchConfig(
            model="0.6b-q8", timeout=90,
            name="q81_06b_sync3",
            prompt=THINK_PROMPT,
            affinity="0", tensor_split="1",
            n_parallel=2, concurrent=2, context=4096,
            max_tokens=1024, reasoning_budget=1024).with_flags(SYNC_EVERY="3"))

    # ── Production stability tests with FORCE_SYNC ───────────────────

    def test_glm47_np4_500tok_sync(self):
        """GLM-4.7 MoE np=4 × 500 tokens with FORCE_SYNC — production stability.
        Previous: CRASH at ~98s.
        RESULT (FORCE_SYNC=1): No crash but too slow — 0/4 completed in 300s timeout.
        FORCE_SYNC syncs after every graph op. MoE has ~3000 ops/token → unusable."""
        self.run(BenchConfig(
            model="glm47-q4km", timeout=300,
            name="q81_glm47_np4_500_sync",
            prompt=THINK_PROMPT,
            affinity="0,1,2", tensor_split="1,1,1",
            n_parallel=4, concurrent=4, context=65536,
            flash_attn=True, cache_reuse=256,
            max_tokens=500, reasoning_budget=0).with_flags(FORCE_SYNC="1", FUSED_MMQ="1"))

    def test_glm47_np4_500tok_sync3(self):
        """GLM-4.7 MoE np=4 × 500 tokens with SYNC_EVERY=3.
        RESULT: CRASH at 170s — 3 is too infrequent for multi-GPU MoE."""
        self.run(BenchConfig(
            model="glm47-q4km", timeout=300,
            name="q81_glm47_np4_500_sync3",
            prompt="Explain the difference between TCP and UDP.",
            affinity="0,1,2", tensor_split="1,1,1",
            n_parallel=4, concurrent=4, context=65536,
            flash_attn=True, cache_reuse=256,
            max_tokens=500, reasoning_budget=0).with_flags(SYNC_EVERY="3", FUSED_MMQ="1"))

    def test_glm47_np4_500tok_sync2(self):
        """GLM-4.7 MoE np=4 × 500 tokens with SYNC_EVERY=2.
        RESULT: CRASH at 181s — MUL_MAT-only sync insufficient for multi-GPU MoE."""
        self.run(BenchConfig(
            model="glm47-q4km", timeout=300,
            name="q81_glm47_np4_500_sync2",
            prompt="Explain the difference between TCP and UDP.",
            affinity="0,1,2", tensor_split="1,1,1",
            n_parallel=4, concurrent=4, context=65536,
            flash_attn=True, cache_reuse=256,
            max_tokens=500, reasoning_budget=0).with_flags(SYNC_EVERY="2", FUSED_MMQ="1"))

    # ── Row-split tests ─────────────────────────────────────────────
    # Row-split distributes matrix rows across GPUs so they compute
    # every matmul in parallel. It uses ext_oneapi_submit_barrier +
    # events for cross-device sync — which layer-split lacks. If
    # row-split's event-based sync prevents the L0 overlap crash,
    # it becomes the production path.
    #
    # Previous state: 4 commits (daf5a6f..e0ebf6b) fixed 5 bugs to
    # make row-split functional. Last known: correct output but 9x
    # slower on 32B due to DMMV fallback (Q8_1 ds visibility bug).

    def test_06b_row_1gpu(self):
        """0.6B Q8_0 row-split on single GPU — basic sanity check.
        RESULT: 1/1 ok, 200 tok, 79.5 t/s. BUT: row_split_active=false
        with 1 GPU (needs used_devices>1), so this is just normal 1-GPU speed.
        Not actually testing row-split."""
        self.run(BenchConfig(
            model="0.6b-q8", timeout=90,
            name="q81_06b_row_1gpu",
            prompt=SHORT_PROMPT,
            split_mode="row",
            affinity="0", tensor_split="1",
            n_parallel=1, concurrent=1, context=4096,
            max_tokens=200))

    def test_06b_row_3gpu(self):
        """0.6B Q8_0 row-split on 3 GPUs — cross-device row distribution.
        RESULT (DMMV): 0/2, 0 tok, timeout 90s. GPUs idle 600MHz — CPU-bound.
        RESULT (ROW_ALLOW_MMVQ): TBD — testing with real GPU kernels."""
        self.run(BenchConfig(
            model="0.6b-q8", timeout=90,
            name="q81_06b_row_3gpu",
            prompt=SHORT_PROMPT,
            split_mode="row",
            affinity="0,1,2", tensor_split="1,1,1",
            n_parallel=2, concurrent=2, context=4096,
            max_tokens=200).with_flags(ROW_ALLOW_MMVQ="1"))

    def test_06b_row_1024tok(self):
        """0.6B row-split, np=2 × 1024 tokens — the crash workload.
        This is the workload that crashes layer-split at ~25s. If
        row-split's event-based sync prevents the L0 overlap crash,
        row-split is the fix. No SYNC_EVERY or FORCE_SYNC — raw row-split.
        NOTE: Single GPU, so row_split_active=false — not a real test."""
        self.run(BenchConfig(
            model="0.6b-q8", timeout=120,
            name="q81_06b_row_1024tok",
            prompt=THINK_PROMPT,
            split_mode="row",
            affinity="0,1,2", tensor_split="1,1,1",
            n_parallel=2, concurrent=2, context=4096,
            max_tokens=1024, reasoning_budget=1024).with_flags(ROW_ALLOW_MMVQ="1"))

    def test_q4km_row_short(self):
        """32B Q4_K_M row-split on 3 GPUs — actual multi-GPU workload.
        RESULT (DMMV): 1/1 ok, 100 tok, 0.6 t/s. GPUs idle 600MHz — CPU-bound.
        RESULT (MMVQ, ROW_ALLOW_MMVQ): DEVICE_LOST at row_split_merge_dst.
        RESULT (FUSED_MMQ, ROW_ALLOW_MMVQ): DEVICE_LOST at row_split_merge_dst.
        RESULT (MMVQ, ROW_ALLOW_MMVQ, SYNC_EVERY=1): CRASH — not L0 overlap.
        RESULT (MMVQ + post-op sync): PASS 0.47 t/s — correct output, needs event sync."""
        self.run(BenchConfig(
            model="q4km", timeout=300,
            name="q81_q4km_row_short",
            prompt=SHORT_PROMPT,
            split_mode="row",
            affinity="0,1,2", tensor_split="1,1,1",
            n_parallel=1, concurrent=1, context=4096,
            max_tokens=100).with_flags(ROW_ALLOW_MMVQ="1"))

    def test_q4km_row_np2_200(self):
        """32B Q4_K_M row-split, np=2 × 200 — crash stability test.
        Layer-split crashes at np=2 after ~77s of decode. Uses
        ROW_ALLOW_MMVQ to run real GPU kernels instead of DMMV fallback."""
        self.run(BenchConfig(
            model="q4km", timeout=600,
            name="q81_q4km_row_np2_200",
            prompt=THINK_PROMPT,
            split_mode="row",
            affinity="0,1,2", tensor_split="1,1,1",
            n_parallel=2, concurrent=2, context=8192,
            max_tokens=200, reasoning_budget=200).with_flags(ROW_ALLOW_MMVQ="1"))

    def test_q4km_row_np2_1024(self):
        """32B Q4_K_M row-split, np=2 × 1024 — the production crash test.
        If this passes without SYNC_EVERY, row-split's event-based sync
        solves the L0 overlap bug for production use. Compare throughput
        to layer-split baseline (17.5 t/s at np=16)."""
        self.run(BenchConfig(
            model="q4km", timeout=600,
            name="q81_q4km_row_np2_1024",
            prompt=THINK_PROMPT,
            split_mode="row",
            affinity="0,1,2", tensor_split="1,1,1",
            n_parallel=2, concurrent=2, context=8192,
            max_tokens=1024, reasoning_budget=1024).with_flags(ROW_ALLOW_MMVQ="1"))

    def test_glm47_row_short(self):
        """GLM-4.7 MoE row-split on 3 GPUs — MoE multi-GPU workload.
        MoE was the hardest case for layer-split (SYNC_EVERY=1..3 all crash,
        only FORCE_SYNC works but too slow). If row-split handles MoE
        without explicit sync, it's a major win."""
        self.run(BenchConfig(
            model="glm47-q4km", timeout=300,
            name="q81_glm47_row_short",
            prompt=SHORT_PROMPT,
            split_mode="row",
            affinity="0,1,2", tensor_split="1,1,1",
            n_parallel=1, concurrent=1, context=4096,
            flash_attn=True,
            max_tokens=100).with_flags(ROW_ALLOW_MMVQ="1"))

    def test_glm47_np4_500tok_imm(self):
        """GLM-4.7 MoE np=4 × 500 tokens with immediate command lists.
        RESULT: No crash but 0/4 completed in 300s. VRAM 7GB→27GB.
        Immediate cmdlists prevent overlap but are too slow and memory-hungry."""
        self.run(BenchConfig(
            model="glm47-q4km", timeout=300,
            name="q81_glm47_np4_500_imm",
            prompt=THINK_PROMPT,
            affinity="0,1,2", tensor_split="1,1,1",
            n_parallel=4, concurrent=4, context=65536,
            flash_attn=True, cache_reuse=256,
            immediate_cmdlists=True,
            max_tokens=500, reasoning_budget=0).with_flags(FUSED_MMQ="1"))
