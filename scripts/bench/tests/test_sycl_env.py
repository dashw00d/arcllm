"""SYCL environment variable matrix — graph capture and command list mode.

## Context

Tests the 2x2 matrix of GGML_SYCL_DISABLE_GRAPH (on/off) and
SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS (on/off) to measure
throughput impact on Q4_K_M np=4.

## Results

All four combinations produce equivalent throughput (~same t/s within noise).
Batched command lists (immediate=0) give +7.5% at np=16 (measured in
test_frontier.py), but at np=4 the difference is negligible.

Graph capture (disable_graph=0) is untested at long sequences and may
cause crashes. Production uses disable_graph=1 + immediate_cmdlists=0.
"""
from bench.base import BenchTest
from bench.config import BenchConfig


class TestSyclEnv(BenchTest):
    base = BenchConfig(model="q4km", n_parallel=4, concurrent=4)

    def test_nograph_nocmd(self):
        self.run(self.base.with_(name="sycl_nograph_nocmd",
                                 disable_graph=True, immediate_cmdlists=False))

    def test_nograph_cmd(self):
        self.run(self.base.with_(name="sycl_nograph_cmd",
                                 disable_graph=True, immediate_cmdlists=True))

    def test_graph_nocmd(self):
        self.run(self.base.with_(name="sycl_graph_nocmd",
                                 disable_graph=False, immediate_cmdlists=False))

    def test_graph_cmd(self):
        self.run(self.base.with_(name="sycl_graph_cmd",
                                 disable_graph=False, immediate_cmdlists=True))
