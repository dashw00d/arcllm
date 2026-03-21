"""MMVQ kernel NaN corruption — multi-slot long sequence crash.

## Bug Summary

The SYCL MMVQ kernel (mmvq.cpp:1995) produces NaN values in attention Q
matrix computations when multiple slots generate >~500 tokens concurrently.
This triggers UR_RESULT_ERROR_DEVICE_LOST (GPU hang).

## Reproduction (without fused kernel)

- 1 slot × 1024 tokens with thinking: PASSES (6.6 t/s)
- 4 slots × 1024 tokens with thinking: CRASHES at ~120s (~600 tok/slot)
- 16 slots × 200 tokens (no thinking): PASSES (17.5 t/s)

The crash is NOT:
- OOM (KV cache is pre-allocated at server start, c=32768 fits fine)
- SYCL graph replay (crashes with graph=ON and graph=OFF)
- Cascading GPU corruption (reproduces from clean GPU state)

The crash IS:
- Numerical: `raw=nan scaled=nan q6s=[nan,nan]` in debug output
- In MMVQ: `ggml_sycl_op_mul_mat_vec_q` at mmvq.cpp:1995
- On Q6_K blocks (mixed quant in Q4_K_M model)
- Only when multiple slots are past ~500 tokens simultaneously
- Possibly a race condition or shared buffer corruption in multi-slot MMVQ dispatch

## Crash Log Pattern

    ggml_sycl_op_mul_mat_vec_q: row-debug attn_q0.mmvq_blocks_consume
        dev=0 col=1 row_low=0 row_high=8192 e=-1..-1
        raw=nan scaled=nan q6s=[nan,nan] q8ds=[nan,nan]
    level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
    Exception caught at file:mmvq.cpp, line:1995

## Relevant Files

- llama.cpp/ggml/src/ggml-sycl/mmvq.cpp:1995 — crash site
- llama.cpp/ggml/src/ggml-sycl/fused_mmq.cpp — fused kernel (bypasses MMVQ at batch>=16)
- llama.cpp/ggml/src/ggml-sycl/common.hpp:179 — error handler

## Hypothesis: Fused Kernel Bypass

With FUSED_MMQ=1 and np=16, all 16 slots generate simultaneously → matmul batch=16
→ routes through fused dp4a kernel instead of MMVQ. If the NaN bug is in MMVQ
specifically, the fused kernel might sidestep it entirely at np>=16.

At np=4, batch=4 still routes through MMVQ (fused only triggers at batch>=~9).
So the fused kernel won't help at np=4 unless we lower the MMVQ→MMQ threshold.

## Production Target

np=16, thinking enabled, 30k context, 1024+ tokens/slot. This is the workload
that must work for production.

## Fused Kernel Bypass Result (2026-03-16)

FUSED_MMQ=1 with Q6_K support does NOT fix the crash. The NaN is not in the
weight-matmul kernel (MMVQ vs fused) — it's in the **attention Q8_1 activation
data** being corrupted before any kernel runs.

Evidence from crash logs:
  `ds=[nan,nan] q=[0,0,0,0]` — Q8_1 activations have NaN scales AND zero quants.
  This means the activation quantization produced garbage, not that the matmul
  kernel computed wrong. The corruption happens BEFORE MMVQ/fused is called.

The attention matmul is per-slot (ne[1]=1 or 2), NOT batched across all 16 slots.
So even with np=16, attention goes through MMVQ with small batch, not the fused kernel.
The fused kernel handles the FFN/projection matmuls (which ARE batched at np=16).

Root cause is likely a race condition in Q8_1 activation quantization or KV cache
management when multiple slots have long sequences. This is an upstream llama.cpp
SYCL backend bug, not fixable by kernel replacement.

## Status

Root cause identified: L0 command overlap within in-order queues causes
kernels to execute before input data is visible (see test_q8_1_corruption.py
for full investigation). The Q8_1 activation NaN is a symptom — the actual
bug is that L0 does NOT honor in-order queue guarantees, allowing
concurrent kernel execution to read stale/uninitialized buffers.

Workaround: FUSED_MMQ=1 at np>=16 routes through fused kernel (avoids
MMVQ code path). Row-split with event-based sync (test_row_split.py) is
the long-term fix.
"""
from bench.base import BenchTest
from bench.config import BenchConfig

THINK_PROMPT = (
    "Think step by step. What are the top 5 most impactful inventions "
    "of the 20th century? Rank by long-term societal impact."
)

# All tests use FUSED_MMQ=1 as the new baseline.
FUSED = BenchConfig(
    model="q4km", prompt=THINK_PROMPT, timeout=300,
).with_flags(FUSED_MMQ="1")


class TestMmvqNan(BenchTest):
    """MMVQ NaN crash reproduction and fused kernel bypass tests."""

    # ── Known behavior (reference — bug exists without fused) ─────────

    def test_1x_1024_control(self):
        """Control: 1 slot, 1024 tokens, fused — should always pass."""
        self.run(FUSED.with_(
            name="nan_1x_1024_fused",
            n_parallel=1, concurrent=1,
            max_tokens=1024, reasoning_budget=1024))

    def test_4x_1024_mmvq_bug(self):
        """Known crash: 4 slots × 1024 tokens — MMVQ NaN at ~600 tok/slot.
        With fused kernel, batch=4 still routes through MMVQ. Expect crash."""
        self.run(FUSED.with_(
            name="nan_4x_1024_fused",
            n_parallel=4, concurrent=4,
            max_tokens=1024, reasoning_budget=1024))

    # ── Fused kernel bypass hypothesis ────────────────────────────────

    def test_16x_1024_fused_bypass(self):
        """KEY TEST: np=16, 1024 tokens, thinking — does fused bypass the NaN?
        At batch=16, matmul routes through fused kernel instead of MMVQ.
        If this passes, the fused kernel sidesteps the MMVQ bug."""
        self.run(FUSED.with_(
            name="nan_16x_1024_fused",
            n_parallel=16, concurrent=16,
            max_tokens=1024, reasoning_budget=1024,
            timeout=600))

    def test_16x_512_fused(self):
        """Shorter probe: np=16, 512 tokens, thinking — below crash threshold."""
        self.run(FUSED.with_(
            name="nan_16x_512_fused",
            n_parallel=16, concurrent=16,
            max_tokens=512, reasoning_budget=512,
            timeout=300))

    # ── Production target ─────────────────────────────────────────────

    def test_16x_2048_production(self):
        """Production target: np=16, 2048 tokens, thinking, fused.
        This is the workload we need for production (30k context, long gen)."""
        self.run(FUSED.with_(
            name="nan_16x_2048_prod",
            n_parallel=16, concurrent=16,
            max_tokens=2048, reasoning_budget=2048,
            timeout=900))
