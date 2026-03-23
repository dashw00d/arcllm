"""Qwen3.5-35B concurrent slot crash — Q8_1 buffer race on DG2 (Intel Arc).

## Root Cause

When 2+ slots generate simultaneously, the Q8_1 activation quantization buffers
race in ggml_sycl_op_mul_mat (no_quantize_q8_1 path). The SYCL pool allocator
reuses buffers without synchronization — Slot B can overwrite a Q8_1 buffer that
Slot A's async kernel is still reading. Result: NaN in ds field, garbled output
or DEVICE_LOST.

The bug is in the SYCL backend's buffer pool, not the model. However it only
manifests on Qwen3.5-35B (DeltaNet MoE with recurrent state) because:
- Pure transformer MoE (Qwen3-30B) and dense (Qwen3-32B) don't trigger it
  even at np=16 with 3200+ concurrent tokens
- The recurrent state management in Qwen3.5 changes buffer allocation patterns
  enough to hit the race condition
- The Q8_1 corruption threshold is ~500 tokens per slot at ≥2 concurrent slots

See test_q8_1_corruption.py for the full root cause analysis.

## Current Workaround

arcllm-proxy.py sets gate.set_max_active(1) — serializes ALL requests even
with np=4. This preserves prompt caching (4 KV cache slots) but kills aggregate
throughput (no concurrent generation benefit).

## What To Test

1. Verify the crash still occurs at np=2, np=4 with moderate token counts
2. Find if there's a safe concurrent token threshold (maybe np=2 mt=100 works?)
3. Test if FUSED_MMQ changes the crash boundary
4. Test if kv_quant=q8_0 affects the race (changes buffer sizes)

If a safe concurrent config exists, we can lift the serialization in the proxy
for qwen35 specifically. Otherwise, 30B MoE is the pipeline model and 35B
stays serialized for Discord-only use.

## Results

(Fill in after running)

## How to run
    cd /home/ryan/llm-stack/scripts
    python3 -m bench qwen35_np                       # all tests
    python3 -m bench qwen35_np.serial_baseline        # specific test

## Relevant Files

- scripts/arcllm-proxy.py:225 — gate.set_max_active(1) workaround
- scripts/bench/tests/test_q8_1_corruption.py — full race condition analysis
- llama.cpp-stable/ggml/src/ggml-sycl/ggml-sycl.cpp — pool allocator + mul_mat
"""

from bench.base import BenchTest
from bench.config import BenchConfig

# Need to register the qwen35 model in config.py for bench framework,
# or use absolute path. Using absolute path for now since the model
# requires a special binary (llama-server-qwen35-gdn).
QWEN35_MODEL = (
    "/home/ryan/llm-stack/models/Qwen/"
    "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-GGUF/"
    "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
)
QWEN35_BUILD = "/home/ryan/llm-stack/bin"  # frozen binary dir


class TestQwen35Np(BenchTest):
    """Qwen3.5-35B concurrent slot crash investigation."""

    base = BenchConfig(
        model=QWEN35_MODEL,
        split_mode="layer",
        n_parallel=1,
        concurrent=1,
        context=16384,
        max_tokens=200,
        flash_attn=False,
        disable_graph=True,
        immediate_cmdlists=False,
        no_warmup=True,
        batch=256,  # required — OOMs at default 2048
        timeout=300,
    )

    # ── Baseline: serial (should always pass) ──

    def test_serial_baseline(self):
        """np=1 — single slot, no race possible. Control test."""
        self.run(self.base.with_(name="qwen35_serial"))

    # ── np=2: minimal concurrent ──

    def test_np2_mt200(self):
        """np=2 mt=200 — minimum concurrent config. Expected: crash."""
        self.run(self.base.with_(
            name="qwen35_np2_mt200",
            n_parallel=2, concurrent=2, max_tokens=200,
        ))

    def test_np2_mt50(self):
        """np=2 mt=50 — very short output. Does the race need >50 tokens?"""
        self.run(self.base.with_(
            name="qwen35_np2_mt50",
            n_parallel=2, concurrent=2, max_tokens=50,
        ))

    # ── np=4: production config (currently serialized) ──

    def test_np4_mt200(self):
        """np=4 mt=200 — the config we WANT to run. Expected: crash."""
        self.run(self.base.with_(
            name="qwen35_np4_mt200",
            n_parallel=4, concurrent=4, max_tokens=200,
        ))

    def test_np4_mt100(self):
        """np=4 mt=100 — shorter output window. Safe?"""
        self.run(self.base.with_(
            name="qwen35_np4_mt100",
            n_parallel=4, concurrent=4, max_tokens=100,
        ))

    # ── Experimental: does FUSED_MMQ change the crash boundary? ──

    def test_np2_fused(self):
        """np=2 FUSED_MMQ=1 — different matmul path, different buffer pattern?"""
        self.run(self.base.with_(
            name="qwen35_np2_fused",
            n_parallel=2, concurrent=2, max_tokens=200,
        ).with_flags(FUSED_MMQ="1"))

    # ── Experimental: KV quant changes buffer sizes ──

    def test_np2_kvq8(self):
        """np=2 kv_quant=q8_0 — smaller KV buffers, changes pool allocation."""
        self.run(self.base.with_(
            name="qwen35_np2_kvq8",
            n_parallel=2, concurrent=2, max_tokens=200,
            kv_quant="q8_0",
        ))
