"""Find the maximum safe context size for Qwen3.5-35B on 3x Arc A770.

Binary searches for the context ceiling where DG2 starts producing garbled output.
Tests multiple batch sizes (-b 512, 256, 128) to find the best throughput/context tradeoff.

## Results
(Fill in after running)

## How to run
    cd /home/ryan/llm-stack/scripts
    python3 -m bench context_ceiling
"""

from bench.base import BenchTest
from bench.config import BenchConfig


class TestContextCeiling(BenchTest):
    """Find max safe context per batch size."""

    # Base config — layer-split, np=4, no flash attention
    base = BenchConfig(
        model="qwen35-35b",
        split_mode="layer",
        n_parallel=4,
        flash_attn=False,
        concurrent=1,
        max_tokens=100,
        prompt="Explain the theory of relativity in detail. " * 20,  # ~400 tokens
    )

    # Default batch size (512) — known: 32k works, 65k corrupts
    def test_b512_c32k(self):
        self.run(self.base.with_(name="b512_c32k", context=32768, batch=512))

    def test_b512_c40k(self):
        self.run(self.base.with_(name="b512_c40k", context=40960, batch=512))

    def test_b512_c48k(self):
        self.run(self.base.with_(name="b512_c48k", context=49152, batch=512))

    # Reduced batch (256) — should halve compute buffers
    def test_b256_c32k(self):
        self.run(self.base.with_(name="b256_c32k", context=32768, batch=256))

    def test_b256_c48k(self):
        self.run(self.base.with_(name="b256_c48k", context=49152, batch=256))

    def test_b256_c64k(self):
        self.run(self.base.with_(name="b256_c64k", context=65536, batch=256))

    # Small batch (128) — if 256 still corrupts
    def test_b128_c48k(self):
        self.run(self.base.with_(name="b128_c48k", context=49152, batch=128))

    def test_b128_c64k(self):
        self.run(self.base.with_(name="b128_c64k", context=65536, batch=128))
