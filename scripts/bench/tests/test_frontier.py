"""Proven frontier configs: Q4_K_M np=16 FUSED_MMQ=1 → 21.7 t/s (dense).

## History

- 2026-03-16: 17.5 t/s — Q4_K_M np=16 cmdlist=0, no fused kernel
- 2026-03-17: 21.7 t/s — FUSED_MMQ=1 fused dequant+matmul kernel (+25%)

The fused kernel (fused_mmq.cpp) keeps dequantized Q4_K weights in registers
during matmul instead of round-tripping through f16 intermediate buffers.
With np=16, all slots generate simultaneously → matmul batch=16 → fused kernel's
sweet spot. Q6_K layers (attn norms) still use generic GEMM (fused Q6_K tested
slower). Only Q4_K layers route through fused kernel.

This is the regression test. If this breaks, something is wrong with the
build, env, or hardware — not the test config.
"""

from bench.base import BenchTest
from bench.config import BenchConfig


class TestFrontier(BenchTest):
    base = BenchConfig(
        model="q4km",
        n_parallel=16,
        concurrent=16,
        context=32768,
        immediate_cmdlists=False,
        disable_graph=True,
    ).with_flags(FUSED_MMQ="1")

    def test_np16(self):
        self.run(self.base.with_(name="frontier_np16"))

    def test_np16_no_fused(self):
        """Baseline without fused kernel — for A/B comparison."""
        self.run(
            BenchConfig(
                model="q4km",
                n_parallel=16,
                concurrent=16,
                context=32768,
                immediate_cmdlists=False,
                disable_graph=True,
            ).with_(name="frontier_np16_nofused")
        )


class TestMoEFrontier(BenchTest):
    """Qwen3-30B-A3B MoE (abliterated) at np=16. ~28 t/s.

    This is the flagship MoE benchmark. Uses split-mode=layer (standard MoE
    dispatch) rather than tensor (which enables EP/expert-parallel splits).

    Model: Qwen3-30B-A3B-abliterated (128 experts, 8 used per token, Q4_K_M)
    - Abliterated = expert-merged model that works correctly
    - DO NOT use Qwen3-30B-A3B-REAM-heretic-i1 (garbled output, quantization bug)

    ## Results

    | Config | Result | Date |
    |-------|--------|------|
    | qwen30b-ablit np=16 | ~28 t/s | 2026-03-20 |

    ## Model Notes

    The REAM-heretic-i1 variant of this model produces garbled output on both
    stable and eptp builds. This is a quantization/merge issue in that specific
    model, not an EP or llama.cpp bug. Always use the abliterated variant.
    """

    base = BenchConfig(
        model="qwen30b-ablit-q4km",
        split_mode="layer",  # MoE dispatch via layer split (not tensor/EP)
        n_parallel=16,
        concurrent=16,
        context=512,  # Short context sufficient for throughput test
        max_tokens=200,
        immediate_cmdlists=False,
        disable_graph=True,
        no_warmup=True,  # MoE warmup can crash on some configs
        flash_attn=False,  # IGC crash on MoE + flash attn (works on dense models)
    )

    def test_np16(self):
        """MoE np=16 throughput — flagship benchmark."""
        self.run(self.base.with_(name="moe_np16"))

    def test_np4(self):
        """MoE np=4 for comparison."""
        self.run(
            self.base.with_(name="moe_np4", n_parallel=4, concurrent=4, no_warmup=False)
        )

    def test_np8(self):
        """MoE np=8 for scaling curve."""
        self.run(
            self.base.with_(name="moe_np8", n_parallel=8, concurrent=8, no_warmup=False)
        )
