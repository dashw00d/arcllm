"""Parallel slot scaling on Q4_K_M — throughput vs slot count.

## Context

Measures aggregate throughput as parallel slots increase from 4 to 16.
Uses non-fused MMVQ path (no FUSED_MMQ flag). For fused kernel scaling,
see test_frontier.py (np=16) and test_fused_mmq.py.

## Results

| np | Throughput | Notes |
|----|-----------|-------|
| 4  | ~7 t/s    | MMVQ path |
| 8  | ~12 t/s   | Near-linear scaling |
| 12 | ~15 t/s   | |
| 16 | ~17.5 t/s | Pre-fused baseline (fused = 21.7 t/s) |

Scaling is near-linear up to np=16. With FUSED_MMQ=1 at np=16,
throughput jumps +25% to 21.7 t/s (see test_frontier.py).
"""
from bench.base import BenchTest
from bench.config import BenchConfig


class TestParallel(BenchTest):
    base = BenchConfig(model="q4km")

    def test_np4(self):
        self.run(self.base.with_(name="parallel_np4", n_parallel=4, concurrent=4))

    def test_np8(self):
        self.run(self.base.with_(name="parallel_np8", n_parallel=8, concurrent=8))

    def test_np12(self):
        self.run(self.base.with_(name="parallel_np12", n_parallel=12, concurrent=12))

    def test_np16(self):
        self.run(self.base.with_(name="parallel_np16", n_parallel=16, concurrent=16))
