"""Qwen3.5-35B-A3B garbled output investigation — unsloth vs HauhauCS quant.

## Context

HauhauCS Qwen3.5-35B-A3B quant produces garbled output on llama.cpp-stable.
Need to determine: is it a code issue or a model/quant issue?

Test plan:
1. Run unsloth/Qwen3.5-35B-A3B Q4_K_M (clean official quant) — baseline
2. Compare output quality between quants

Both use identical server config from docs/QWEN35-35B.md:
- layer-split, fused_gdn_ar=false (hardcoded in stable), -fa off, no_warmup

## Results

(pending)

## Relevant Files

- docs/QWEN35-35B.md — working config, fused_gdn_ar fix
- llama.cpp-stable/src/llama-context.cpp:153 — fused_gdn_ar=false
"""

from bench.base import BenchTest
from bench.config import BenchConfig


class TestQwen35Garbled(BenchTest):
    """Compare unsloth vs HauhauCS Qwen3.5-35B quants for garbled output."""

    base = BenchConfig(
        split_mode="layer",
        n_parallel=1,
        concurrent=1,
        context=8192,
        max_tokens=400,
        flash_attn=False,  # IGC crash on MoE + FA
        disable_graph=True,
        immediate_cmdlists=False,
        no_warmup=True,
        timeout=300,
    )

    def test_unsloth(self):
        """Unsloth official quant — clean baseline. If garbled, it's code."""
        r = self.run(self.base.with_(
            name="qwen35_unsloth",
            model="qwen35-35b-unsloth",
        ))
        self._print_output(r)

    def test_hauhau(self):
        """HauhauCS quant — known garbled. Control comparison."""
        r = self.run(self.base.with_(
            name="qwen35_hauhau",
            model="qwen35-35b-hauhau",
        ))
        self._print_output(r)

    @staticmethod
    def _print_output(r):
        """Print first 500 chars of response to check for garbled text."""
        for req in r.per_request:
            text = req.get("text", "")
            if text:
                print(f"\n  ── Response (first 500 chars) ──")
                print(f"  {text[:500]}")
                print(f"  ── end ──")
            else:
                print(f"\n  ── No response text captured ──")
