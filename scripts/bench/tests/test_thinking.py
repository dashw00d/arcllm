"""Thinking mode — isolate SYCL graph crash at long sequences.

## Context

Tests whether SYCL graph capture causes crashes at long thinking sequences.
Graph replay is suspected of failing when kernel argument shapes change
during extended generation.

## Results (q4km, c=32768)

| Config | Result | Notes |
|--------|--------|-------|
| np=1, 256tok, graph=OFF | PASS, 6.7 t/s | Control |
| np=1, 1024tok           | PASS, 7.8 t/s | ~136s for 1024 tokens |
| np=4, 1024tok           | TIMEOUT | ~524s > 600s timeout |

Note: graph=ON hangs on this build (c=32768). Test removed.
"""

from bench.base import BenchTest
from bench.config import BenchConfig

THINK_PROMPT = (
    "Think step by step. What are the top 5 most impactful inventions "
    "of the 20th century? Rank by long-term societal impact."
)


class TestThinking(BenchTest):
    base = BenchConfig(
        model="q4km",
        max_tokens=256,
        timeout=120,
        prompt=THINK_PROMPT,
        n_parallel=1,
        concurrent=1,
        reasoning_budget=-1,
        no_warmup=True,
    )

    def test_1x_nograph(self):
        self.run(self.base.with_(name="think_1x_nograph", disable_graph=True))

    def test_1x_1024tok(self):
        self.run(
            self.base.with_(
                name="think_1x_1024", max_tokens=1024, reasoning_budget=-1, timeout=300
            )
        )

    def test_4x_1024tok(self):
        self.run(
            self.base.with_(
                name="think_4x_1024",
                n_parallel=4,
                concurrent=4,
                max_tokens=1024,
                reasoning_budget=-1,
                timeout=600,
            )
        )
