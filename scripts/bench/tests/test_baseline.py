"""Serial throughput baseline — single-slot generation on Q4_K_M.

## Results

| Model | Result | Notes |
|-------|--------|-------|
| Q4_K_M np=1 | ~6.6 t/s | Compute-bound, GPUs at 2100MHz |
"""
from bench.base import BenchTest
from bench.config import BenchConfig


class TestBaseline(BenchTest):
    base = BenchConfig(n_parallel=1, concurrent=1)

    def test_q4km(self):
        self.run(self.base.with_(name="baseline_q4km", model="q4km"))
