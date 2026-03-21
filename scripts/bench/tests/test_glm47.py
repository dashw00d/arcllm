"""GLM-4.7-Flash MoE — 30B/3.6B active, MLA, 64k context on 3x Arc A770.

## Model

GLM-4.7-Flash-ultimate-irrefusable-heretic Q4_K_M (17 GB).
MoE: 47 layers, 64 routed + 1 shared expert, 4 active/token.
MLA attention (kv_lora_rank=512) — compressed KV cache.
Pure transformer — prompt caching works.

## VRAM Budget

| Component | Size |
|-----------|------|
| Weights (Q4_K_M) | 18 GB |
| KV cache (4 slots × 64k ctx, f16) | 25 GB |
| Buffers (~1 GB/GPU) | 3 GB |
| **Total** | **46 / 48 GB (96%)** |
| Free | 2 GB |

## Config Rationale

- np=4, c=65536: fewer slots but massive context per slot (16k each)
- layer-split (row-split broken on SYCL)
- f16 KV cache (f16 beats q8_0 on SYCL — dequant overhead > bandwidth savings)
- cache-reuse=256 for prompt caching
- FUSED_MMQ=1 for fused dequant+matmul on Q4_K layers
- MoE with 3.6B active params — should be fast since only ~10% of weights activate

## Hardware Constraints

- GPU2 is PCIe 3.0 x8 (half bandwidth) — layer-split distributes evenly but GPU2
  will be the bottleneck for any PCIe-bound operation
- No P2P — all cross-GPU copies go through host RAM

## Relevant Files

- models/GLM-4.7-Flash-heretic-GGUF/GLM-4.7-Flash-ultimate-irrefusable-heretic-Q4_K_M.gguf
- scripts/arcllm-proxy.py — production model registry

## Results

Single slot: 4.2 t/s (200 tok, short prompt)
np=4 baseline (no fused): 9.1 t/s (4/4 ok, 800 tok)
np=4 FUSED_MMQ=1: 8.1 t/s — fused kernel doesn't help at batch=4 (MMVQ handles batch<=8)

MoE with 3.6B active params means much less weight per token than dense 32B.
At np=4, batch=4 routes through MMVQ, not the fused kernel. Fused kernel only
triggers at batch>=9 (MMQ dispatch threshold). For this model, the fused flag
should be OFF unless running at np>=16.

GPU utilization is uneven: GPU0=64W, GPU1/2=46W. Expert routing may concentrate
work on GPU0's layers. GPU0 also clocks lower (2109 vs 2366) — possible throttle.

np=4 × 1024 tokens: CRASH (DEVICE_LOST at ~136s) — Level Zero command overlap bug.
np=4 × long prompt + 500 tokens: CRASH at ~98s.

Root cause: Level Zero runtime overlaps kernel execution within in-order SYCL queues.
See test_q8_1_corruption.py for full investigation. NOT an F16 overflow — it's a
command submission race in the L0 runtime.

Fix: stream->wait() after MUL_MAT ops prevents crash but costs ~40% throughput.
Need a lighter barrier (submit_barrier, per-layer sync, or L0 immediate command lists).

Short generation (200 tokens) works fine at np=4.

## Status

Root cause identified: Level Zero runtime overlaps kernel execution in batched
command lists despite in-order queue semantics. Every sync approach (stream->wait,
FORCE_SYNC, immediate cmdlists) prevents the crash but costs 40-60% throughput.
No zero-cost workaround found yet.

See test_q8_1_corruption.py for full investigation and all fix attempts.
"""
from bench.base import BenchTest
from bench.config import BenchConfig

PROMPT_SHORT = "Explain quantum entanglement in simple terms."

PROMPT_LONG = (
    "You are an expert AI researcher. Write a comprehensive analysis of the "
    "transformer architecture's evolution from the original 'Attention Is All "
    "You Need' paper through modern innovations like Mixture of Experts, Multi-"
    "Latent Attention, and speculative decoding. Cover the key breakthroughs, "
    "their motivations, trade-offs, and impact on inference efficiency. Include "
    "specific examples of models that pioneered each technique and discuss how "
    "these innovations interact when combined in a single architecture."
)


class TestGlm47(BenchTest):
    """GLM-4.7-Flash MoE on 3x Arc A770."""

    base = BenchConfig(
        model="glm47-q4km",
        affinity="0,1,2",
        tensor_split="1,1,1",
        n_parallel=4,
        concurrent=4,
        context=65536,
        flash_attn=True,
        cache_reuse=256,
        max_tokens=200,
    ).with_flags(FUSED_MMQ="1")

    # ── Baseline: does it load and run? ───────────────────────────────

    def test_single_short(self):
        """Single request, short prompt — basic sanity check."""
        self.run(self.base.with_(
            name="glm47_single",
            n_parallel=1, concurrent=1,
            prompt=PROMPT_SHORT,
            max_tokens=200))

    # ── Throughput at np=4 ────────────────────────────────────────────

    def test_np4_short(self):
        """4 concurrent requests, short prompts — parallel throughput."""
        self.run(self.base.with_(
            name="glm47_np4_short",
            prompt=PROMPT_SHORT,
            max_tokens=200))

    def test_np4_long(self):
        """4 concurrent requests, long prompt — prompt processing + generation."""
        self.run(self.base.with_(
            name="glm47_np4_long",
            prompt=PROMPT_LONG,
            max_tokens=500))

    # ── Fused vs baseline A/B ─────────────────────────────────────────

    def test_np4_no_fused(self):
        """Baseline without fused kernel — for A/B comparison."""
        self.run(BenchConfig(
            model="glm47-q4km",
            affinity="0,1,2",
            tensor_split="1,1,1",
            n_parallel=4,
            concurrent=4,
            context=65536,
            flash_attn=True,
            cache_reuse=256,
            prompt=PROMPT_SHORT,
            max_tokens=200,
        ).with_(name="glm47_np4_nofused"))

    # ── Long generation stress test ───────────────────────────────────

    def test_np4_1024tok(self):
        """4 slots × 1024 tokens — stress test for NaN/DEVICE_LOST."""
        self.run(self.base.with_(
            name="glm47_np4_1024tok",
            prompt=PROMPT_SHORT,
            max_tokens=1024,
            timeout=600))

    # ── Context utilization ───────────────────────────────────────────

    def test_single_longctx(self):
        """Single slot, long prompt — test large context handling."""
        self.run(self.base.with_(
            name="glm47_longctx",
            n_parallel=1, concurrent=1,
            prompt=PROMPT_LONG,
            max_tokens=1000,
            timeout=600))
