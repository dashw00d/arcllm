"""Base test class. Subclass this in tests/*.py files."""

from __future__ import annotations

from .config import BenchConfig
from .runner import BenchRunner, BenchResult


class BenchTest:
    """Define `base` config and test_* methods. That's it.

    Example:
        class TestFoo(BenchTest):
            base = BenchConfig(model="q4km", n_parallel=4, concurrent=4)

            def test_graph_on(self):
                self.run(self.base.with_(name="foo_graph", disable_graph=False))
    """

    base: BenchConfig = BenchConfig()

    def __init__(self, runner: BenchRunner):
        self.runner = runner
        self.results: list[BenchResult] = []

    def run(self, config: BenchConfig) -> BenchResult:
        r = self.runner.run_test(config)
        self.results.append(r)
        return r

    def run_all(self) -> list[BenchResult]:
        methods = sorted(
            m for m in dir(self)
            if m.startswith("test_") and callable(getattr(self, m))
        )
        for m in methods:
            print(f"\n▶ {self.__class__.__name__}.{m}")
            getattr(self, m)()
        return self.results
