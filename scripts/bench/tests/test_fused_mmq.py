"""Fused Q4 dequant+matmul kernel — SYCL port of Marlin approach.

## Summary

The SYCL backend has no fused dequant+matmul path. Current kernel dispatch:
- tg1 (single-token): MMVQ (quantized dot product per vector)
- pp (prompt processing): dequant to f16 -> oneDNN GEMM (wasteful intermediate)

The existing MMQ infrastructure in mmq.cpp (~2000 lines) is disabled — returns false
for all types with "TODO: accuracy issues". This test suite tracks development of a
fused kernel that keeps dequantized values in registers (never writes to SLM/global),
eliminating the f16 intermediate buffer.

## Opportunity

B580 user measured 41.7 tok/s on 3.56GB model vs 128 tok/s theoretical ceiling — only
33% hardware utilization. A single A770 has ~560 GB/s VRAM bandwidth; with 9B Q8_0
(~9.5GB), theoretical tg1 ceiling is ~59 tok/s. Gap between current and theoretical
is the fused kernel's target.

## Key Discoveries

- joint_matrix (Intel XMX API) has zero usage in llama.cpp SYCL — only tile size changes
- Intel's sycl-tla (SYCL CUTLASS impl) has INT4/INT8 GEMM templates with XMX plumbing
- Arc A770 sub_group_sizes: 8, 16, 32 (hardware). Build uses WARP_SIZE=16 (CMakeLists).
- Marlin kernel is self-contained: marlin_cuda_kernel.cu, ~800 lines, no deps beyond CUDA
- ext_intel_matrix confirmed present on all 3 A770s (XMX hardware available)

## Experimental Flags

Each kernel change is gated behind a GGML_SYCL_* env var, set via `with_flags()`.
Tests A/B compare baseline (no flags) vs experimental (flag=1). Once a flag is proven
stable and beneficial, the behavior becomes default and the flag is removed.

| Flag              | Phase | What it does                                      |
|-------------------|-------|---------------------------------------------------|
| FUSED_MMQ         | 1     | Re-enable MMQ tiled matmul (disabled "accuracy")   |
| FUSED_XMX         | 2     | TESTED — 6x SLOWER than dp4a. Not viable for batch<32. |
| FUSED_SLM_DBUF    | 2     | Double-buffered SLM loading (overlap load+compute) |
| FUSED_REORDER     | 2     | Pre-reorder weights for XMX access patterns         |

## Relevant Files

- llama.cpp/ggml/src/ggml-sycl/mmq.cpp — dead MMQ code, pattern reference
- llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:6081 — mul_mat dispatch
- llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:5635 — ggml_sycl_supports_mmq (returns false)
- llama.cpp/ggml/src/ggml-sycl/common.hpp — WARP_SIZE=16, XMX detection
- llama.cpp/ggml/src/ggml-sycl/dequantize.hpp — dequant functions to fuse
- llama.cpp/ggml/src/ggml-sycl/vecdotq.hpp — dp4a dot products (inner loop patterns)

## Phase 0 Results (2026-03-16)

Hardware confirmed:
- SLM: 64 KB, sub_group_sizes: 8, 16, 32
- ext_intel_matrix: YES (XMX hardware present)

joint_matrix feasibility (scripts/test_joint_matrix.cpp):
- INT8 8x8x32 sg=8: PASS (294 GOPS on 256x256x256)
- BF16 8x8x16 sg=8: PASS
- sg=16 and sg=32: CRASH (IGC Internal Compiler Error: floating point exception)
- CRITICAL: XMX requires sub_group_size=8 on Arc A770 (driver 1.14.37020+3)
  This means the fused kernel's XMX path must use sg=8, separate from llama.cpp's
  WARP_SIZE=16. The dp4a path can stay at sg=16.

llama-bench baselines (single A770, Qwen3.5-9B Q8_0):
- pp512: 681.68 t/s (batch matmul, compute-bound — MKL GEMM path)
- tg128: 8.92 t/s (MMVQ path, ~14% of theoretical 62 t/s bandwidth ceiling)
- Kernel path confirmed: mul_mat_vec_q8_0_q8_1_sycl (MMVQ)

## Status

Phase 0 complete.

Phase 0.5 complete (2026-03-16): sycl-tla and XeTLA do NOT support Arc A770 (Alchemist).
They target Ponte-Vecchio and B580 only. Decision: manual Marlin port (Path B).

Phase 1 step 1 (2026-03-16): Re-enabled existing MMQ via GGML_SYCL_FUSED_MMQ=1.
Results show existing MMQ is MUCH SLOWER than generic GEMM:
  pp16: 44.34 t/s (MMQ) vs 64.32 t/s (GEMM) — 31% slower
  pp32: 46.69 t/s (MMQ) vs 125.34 t/s (GEMM) — 63% slower
  pp8: 14.03 t/s (both same — MMQ likely not triggered)
  tg128: 8.93 t/s (both same — MMVQ path, unaffected by flag)
This confirms WHY Intel disabled MMQ — performance regression, not just accuracy.
The existing mmq.cpp is unoptimized dead code. A true Marlin-style fused kernel needs
to be written from scratch with proper register-resident dequant and memory patterns.

Phase 1 step 2 (2026-03-16): Wrote new fused kernel (fused_mmq.cpp). Each sub_group
processes one weight row against ALL batch columns in registers (amortizing weight reads).
Uses dp4a for int8 dot products, template-dispatched by MAX_BATCH for unrolling.

Results (single A770, Qwen3.5-9B Q8_0):
  pp8:  14.04 t/s (same — fused not triggered, batch too small)
  pp16: 141.59 t/s vs 64.32 baseline — +120% faster (2.2x!)
  pp32: 149.97 t/s vs 125.34 baseline — +20% faster
  tg128: 8.94 t/s (same — MMVQ path, unaffected)

The fused kernel MASSIVELY beats both old MMQ (3.2x at pp16) and generic GEMM (2.2x).
Correctness validation pending (llama-perplexity check).

Full dp4a fused sweep (with FUSED_MMQ=1):
  pp2:  11.04 t/s (fused not triggered — MMVQ catches batch<=8?)
  pp4:  12.90 t/s (same)
  pp8:  14.03 t/s (same)
  pp16: 141.47 t/s (FUSED — 2.2x vs baseline 64.32)
  pp24: 135.62 t/s (FUSED — vs baseline ~100?)
  pp32: 149.44 t/s (FUSED — 1.2x vs baseline 125.34)
  tg128: 8.96 t/s (MMVQ path, unchanged)

Phase 2 XMX attempt (2026-03-16): joint_matrix INT8 8x8x32 kernel tested.
Result: 24.72 t/s at pp16 — **6x SLOWER than dp4a** (141.59). The data staging
overhead (copying Q8_0 blocks to contiguous arrays, loading into joint_matrix,
extracting results, applying per-block scales) dominates. dp4a reads data
in-place with zero setup. XMX only helps for large GEMM tiles that amortize
the setup cost; at batch=16-32 it's pure overhead.
Decision: dp4a is the winning compute path for these batch sizes.

Batch cap experiment (2026-03-16): Raised FUSED_MMQ_MAX_BATCH to 64.
  pp48: fused=154.84 vs baseline=167.98 — 8% regression
  pp64: fused=153.31 vs baseline=217.99 — 30% regression
The fused kernel plateaus at ~150 t/s regardless of batch (bandwidth-bound, reads
all weights once per batch). MKL GEMM scales linearly with batch. Crossover ~pp40.
Reverted to batch cap 32 where fused consistently wins.

Production workload test (validated 2026-03-17 post-reboot, clean GPU state):
  Baseline: 17.4 t/s (16/16 ok, 3200 tok, 183.6s)
  FUSED_MMQ=1: 21.7 t/s (16/16 ok, 3200 tok, 147.6s) — **+25% throughput**
With 16 parallel slots generating, matmul batch=16 hits fused kernel sweet spot.

Note: Q6_K fused kernel tested but SLOWER than GEMM for those layers (17.8 t/s).
Q6_K removed from fused dispatch — falls through to generic GEMM which handles it well.
Only Q4_K layers use the fused kernel. Q6_K layers (attn norms) use GEMM.
"""
from bench.base import BenchTest
from bench.config import BenchConfig

# Short prompt for tg-dominated workload (small input, measure generation speed)
TG_PROMPT = "What is 2+2?"

# ~128 token prompt for prompt processing measurement
PP128_PROMPT = (
    "You are an expert systems architect. Analyze the following distributed system "
    "design and identify potential bottlenecks, single points of failure, and "
    "optimization opportunities. The system consists of: a load balancer distributing "
    "traffic across 8 application servers, each connected to a shared PostgreSQL "
    "primary with 2 read replicas, a Redis cluster for session storage and caching, "
    "a RabbitMQ message broker for async task processing with 4 worker nodes, an "
    "Elasticsearch cluster for full-text search, and an S3-compatible object store "
    "for media files. The system serves 50,000 concurrent users with a 99.9% uptime "
    "SLA. Recent monitoring shows P99 latency spikes during peak hours, particularly "
    "on write-heavy endpoints. The database connection pool is frequently exhausted "
    "and the message queue depth grows unbounded during traffic spikes."
)


class TestFusedMmq(BenchTest):
    """Fused dequant+matmul kernel development — single A770, Qwen3.5-9B Q8_0."""

    base = BenchConfig(
        model="9b-q8",
        affinity="0",              # single GPU
        tensor_split="1",
        n_parallel=1,
        concurrent=1,
        context=8192,              # smaller context for faster iteration
    )

    # ── Phase 0: baselines (no flags — stock kernel paths) ────────────

    def test_baseline_tg(self):
        """Baseline single-token generation — unfused MMVQ path."""
        self.run(self.base.with_(
            name="fused_baseline_tg",
            prompt=TG_PROMPT,
            max_tokens=200,
        ))

    def test_baseline_pp128(self):
        """Baseline prompt processing — unfused dequant-to-f16 + oneDNN GEMM."""
        self.run(self.base.with_(
            name="fused_baseline_pp128",
            prompt=PP128_PROMPT,
            max_tokens=50,
        ))

    # ── Phase 1: FUSED_MMQ — dp4a fused kernel, no XMX ───────────────

    def test_fused_dp4a_tg(self):
        """FUSED_MMQ=1: fused dp4a dequant+matmul — single-token gen."""
        self.run(self.base.with_flags(FUSED_MMQ="1").with_(
            name="fused_dp4a_tg",
            prompt=TG_PROMPT,
            max_tokens=200,
        ))

    def test_fused_dp4a_pp128(self):
        """FUSED_MMQ=1: fused dp4a dequant+matmul — prompt processing."""
        self.run(self.base.with_flags(FUSED_MMQ="1").with_(
            name="fused_dp4a_pp128",
            prompt=PP128_PROMPT,
            max_tokens=50,
        ))

    # ── Phase 2: FUSED_XMX — joint_matrix acceleration ───────────────

    def test_fused_xmx_tg(self):
        """FUSED_MMQ=1 + FUSED_XMX=1: XMX-accelerated — single-token gen."""
        self.run(self.base.with_flags(FUSED_MMQ="1", FUSED_XMX="1").with_(
            name="fused_xmx_tg",
            prompt=TG_PROMPT,
            max_tokens=200,
        ))

    def test_fused_xmx_pp128(self):
        """FUSED_MMQ=1 + FUSED_XMX=1: XMX-accelerated — prompt processing."""
        self.run(self.base.with_flags(FUSED_MMQ="1", FUSED_XMX="1").with_(
            name="fused_xmx_pp128",
            prompt=PP128_PROMPT,
            max_tokens=50,
        ))

    def test_fused_xmx_dbuf_pp128(self):
        """All Phase 2 flags: XMX + double-buffered SLM — prompt processing."""
        self.run(self.base.with_flags(
            FUSED_MMQ="1", FUSED_XMX="1", FUSED_SLM_DBUF="1",
        ).with_(
            name="fused_xmx_dbuf_pp128",
            prompt=PP128_PROMPT,
            max_tokens=50,
        ))

    # ── Phase 3: multi-GPU (layer-split, 3x A770) ────────────────────

    multigpu_base = BenchConfig(
        model="q4km",
        affinity="0,1,2",
        tensor_split="1,1,1",
        n_parallel=1,
        concurrent=1,
        context=32768,
    )

    def test_multigpu_baseline(self):
        """3x A770 baseline — Qwen3-32B Q4_K_M, no fused kernel."""
        self.run(self.multigpu_base.with_(
            name="fused_multigpu_baseline",
            max_tokens=200,
        ))

    def test_multigpu_fused(self):
        """FUSED_MMQ=1: fused Q4_K kernel on 3x A770 — Qwen3-32B Q4_K_M."""
        self.run(self.multigpu_base.with_flags(FUSED_MMQ="1").with_(
            name="fused_multigpu_q4km",
            max_tokens=200,
        ))

    # ── Production workload: np=16 parallel generation ────────────────
    #
    # With np=16 and all slots generating, the matmul batch size is 16.
    # This is exactly where the fused kernel excels (2.2x at pp16).
    # Cached prompts mean steady-state is almost entirely tg at batch=16.

    prod_base = BenchConfig(
        model="q4km",
        affinity="0,1,2",
        tensor_split="1,1,1",
        n_parallel=16,
        concurrent=16,
        context=32768,
        max_tokens=200,
    )

    def test_np16_baseline(self):
        """Production baseline: np=16 ×16 concurrent, Q4_K_M, 3x A770."""
        self.run(self.prod_base.with_(name="fused_np16_baseline"))

    def test_np16_fused(self):
        """FUSED_MMQ=1: np=16 ×16 concurrent — fused kernel at batch=16."""
        self.run(self.prod_base.with_flags(FUSED_MMQ="1").with_(
            name="fused_np16_fused",
        ))
