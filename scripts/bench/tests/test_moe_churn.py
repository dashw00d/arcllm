"""MoE data-churning optimization: np=16 with safe max_tokens to avoid L0 corruption.

## Context

The Level Zero runtime overlaps kernel submissions despite in-order queue creation.
With 2+ parallel slots generating ~500+ tokens concurrently, Q8_1 activation buffers
get corrupted (NaN in ds field) causing garbled output or DEVICE_LOST.

The corruption threshold is ~500 tokens per slot at high concurrency. Single slot
never triggers even at 2000+ tokens. The trigger is concurrent compute depth, not
sequence length.

## Strategy

Maximize aggregate throughput by keeping max_tokens safely under the corruption
threshold (~300) while running np=16 for maximum parallelism. For data churning
workloads (classification, summarization, extraction), individual responses are
short — many fast completions beats few long ones.

## Results (2026-03-21)

| Config | Agg t/s | Tokens | Notes |
|--------|---------|--------|-------|
| np=16 c=512  mt=200 | 25.7 | 3200 | Baseline |
| np=16 c=2048 mt=200 | 25.5 | 3200 | No throughput loss vs c=512 |
| np=16 c=4096 mt=200 | 26.0 | 3200 | Fits in VRAM, slightly faster |
| np=16 c=2048 mt=100 | 25.5 | 1600 | Ultra-short, same per-token speed |
| np=16 c=2048 mt=300 | 26.8 | 3680 | Best t/s — longer gen amortizes overhead |
| FUSED_MMQ np=1  | 13.1 | 200 | Works |
| FUSED_MMQ np=4  | 12.4 | 800 | Works |
| FUSED_MMQ np=8  | 16.9 | 1600 | Works |
| FUSED_MMQ np=12 | 21.9 | 2400 | Works |
| FUSED_MMQ np=16 | 25.6 | 3200 | Works — no improvement over non-fused (25.5) |

**Optimal churning config: np=16, c=4096, mt=300 → 26.8 t/s**

Context has zero throughput cost up to c=4096. Longer max_tokens improves aggregate
throughput by amortizing per-request overhead — push to mt=300 (safe under ~500 threshold).

FUSED_MMQ does NOT crash on MoE (earlier crash was stale GPU state). However it provides
zero throughput benefit on MoE — 25.6 vs 25.5 t/s at np=16. MoE expert FFNs use small
768-dim matmuls where the fused kernel's register optimization doesn't help.

## VRAM Budget

Model: 17.3 GB → ~30.7 GB available for KV cache across 3x A770 (48 GB total).
np=16 × c=4096 fits comfortably (GPUs at ~35% utilization, 50% CPU).

## Relevant Files

- test_frontier.py — MoE np=16 baseline (c=512, 39.4 t/s)
- test_q8_1_corruption.py — L0 race condition investigation
- docs/FLAGSHIP.md — verified settings
"""

from bench.base import BenchTest
from bench.config import BenchConfig


class TestMoEChurn(BenchTest):
    """Tune MoE for high-throughput data churning within L0 corruption limits."""

    base = BenchConfig(
        model="qwen30b-ablit-q4km",
        split_mode="layer",
        n_parallel=16,
        concurrent=16,
        context=2048,
        max_tokens=200,
        immediate_cmdlists=False,
        disable_graph=True,
        no_warmup=True,
        flash_attn=False,  # IGC crash on MoE + FA
    )

    # -- Context scaling (does more context cost throughput?) --

    def test_c512(self):
        """Baseline: matches FLAGSHIP np=16 c=512 config (expect ~39.4 t/s)."""
        self.run(self.base.with_(name="churn_c512", context=512))

    def test_c2048(self):
        """Target config: enough context for real prompts."""
        self.run(self.base.with_(name="churn_c2048"))

    def test_c4096(self):
        """Max context push — may OOM. Tests VRAM ceiling."""
        self.run(self.base.with_(name="churn_c4096", context=4096))

    # -- max_tokens tuning (how close to 500 can we push?) --

    def test_mt100(self):
        """Ultra-short: classification, yes/no, extraction tasks."""
        self.run(self.base.with_(name="churn_mt100", max_tokens=100))

    def test_mt300(self):
        """Push toward corruption threshold. Safe or not?"""
        self.run(self.base.with_(name="churn_mt300", max_tokens=300))

    # -- Slot/context tradeoffs --

    def test_np8_c4096(self):
        """Fewer slots, more context per slot. For longer prompts."""
        self.run(self.base.with_(
            name="churn_np8_c4096",
            n_parallel=8,
            concurrent=8,
            context=4096,
        ))

    # -- FUSED_MMQ batch threshold sweep --
    # Initial test crashed at np=16 (stale GPU state from prior test).
    # Full sweep: works at all batch sizes, but provides 0% throughput gain
    # on MoE (25.6 vs 25.5 at np=16). MoE expert FFNs are too small (768-dim)
    # for the fused kernel's register optimization to matter.

    def test_fused_np1(self):
        """FUSED_MMQ sanity check — single slot, should work."""
        self.run(self.base.with_(
            name="churn_fused_np1", n_parallel=1, concurrent=1,
        ).with_flags(FUSED_MMQ="1"))

    def test_fused_np4(self):
        """FUSED_MMQ at np=4 — proxy-level concurrency."""
        self.run(self.base.with_(
            name="churn_fused_np4", n_parallel=4, concurrent=4,
        ).with_flags(FUSED_MMQ="1"))

    def test_fused_np8(self):
        """FUSED_MMQ at np=8 — find the batch size crash threshold."""
        self.run(self.base.with_(
            name="churn_fused_np8", n_parallel=8, concurrent=8,
        ).with_flags(FUSED_MMQ="1"))

    def test_fused_np12(self):
        """FUSED_MMQ at np=12 — narrowing the crash threshold."""
        self.run(self.base.with_(
            name="churn_fused_np12", n_parallel=12, concurrent=12,
        ).with_flags(FUSED_MMQ="1"))

    def test_fused_np16(self):
        """FUSED_MMQ at np=16 — known crash. Baseline for comparison."""
        self.run(self.base.with_(
            name="churn_fused_np16", n_parallel=16, concurrent=16,
        ).with_flags(FUSED_MMQ="1"))
