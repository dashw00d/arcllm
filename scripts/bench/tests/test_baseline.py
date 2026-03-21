"""Serial throughput baselines — single-slot generation on Q4_K_M and Q8_0.

## Context

Measures raw per-slot throughput with no parallel load. These are the floor
numbers for each quantization level on 3x Arc A770 layer-split.

## Results

| Model | Result | Notes |
|-------|--------|-------|
| Q4_K_M np=1 | ~6.6 t/s | Compute-bound, GPUs at 2100MHz |
| Q8_0 np=1 | 3.3 t/s | Bandwidth-bound (PCIe scatter/gather) |

Q8_0 is bandwidth-limited regardless of np — 3.3 t/s is the ceiling.
"""
from bench.base import BenchTest
from bench.config import BenchConfig


class TestBaseline(BenchTest):
    base = BenchConfig(n_parallel=1, concurrent=1)

    def test_q4km(self):
        self.run(self.base.with_(name="baseline_q4km", model="q4km"))

    def test_q8(self):
        self.run(self.base.with_(name="baseline_q8", model="q8"))
