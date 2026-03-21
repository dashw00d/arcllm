"""CLI entrypoint: python3 -m bench <suite[.test]> ...

Discovers all BenchTest subclasses from bench/tests/*.py automatically.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import sys
from pathlib import Path

from .base import BenchTest
from .runner import BenchRunner, BenchResult


def _discover_suites() -> dict[str, type[BenchTest]]:
    """Import all bench/tests/test_*.py and collect BenchTest subclasses."""
    import bench.tests as pkg
    for importer, name, _ in pkgutil.iter_modules(pkg.__path__):
        if name.startswith("test_"):
            importlib.import_module(f"bench.tests.{name}")

    suites = {}
    for cls in BenchTest.__subclasses__():
        key = cls.__name__.removeprefix("Test").lower()
        suites[key] = cls
    return suites


def main():
    suites = _discover_suites()

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print("Usage: python3 -m bench <suite[.test]> [suite[.test] ...]")
        print()
        for name, cls in sorted(suites.items()):
            doc = (cls.__doc__ or "").strip().split('\n')[0]
            tests = sorted(m.removeprefix("test_") for m in dir(cls)
                           if m.startswith("test_") and callable(getattr(cls, m)))
            print(f"  {name:<16s} {doc}")
            print(f"    tests: {', '.join(tests)}")
        print()
        print("Examples:")
        print("  python3 -m bench baseline")
        print("  python3 -m bench parallel.np16")
        print("  python3 -m bench thinking.1x_nograph thinking.1x_graph")
        sys.exit(0)

    runner = BenchRunner()
    all_results: list[BenchResult] = []

    for target in sys.argv[1:]:
        suite_name, _, test_suffix = target.partition(".")

        if suite_name not in suites:
            print(f"Unknown suite: {suite_name}")
            print(f"Available: {', '.join(sorted(suites))}")
            sys.exit(1)

        suite = suites[suite_name](runner)

        if test_suffix:
            method = f"test_{test_suffix}"
            if not hasattr(suite, method):
                tests = sorted(m.removeprefix("test_") for m in dir(suite)
                               if m.startswith("test_"))
                print(f"Unknown test: {target}")
                print(f"Available: {', '.join(tests)}")
                sys.exit(1)
            print(f"\n▶ {target}")
            getattr(suite, method)()
        else:
            suite.run_all()

        all_results.extend(suite.results)

    # Summary table
    print(f"\n{'═' * 72}")
    print("  RESULTS")
    print(f"{'═' * 72}")
    for r in all_results:
        print(f"  {r.row()}")

    # Save JSON
    out = Path("/tmp/bench_results.json")
    out.write_text(json.dumps([{
        "name": r.name, "completed": r.completed, "failed": r.failed,
        "tokens": r.total_tokens, "wall_s": r.wall_time, "tps": r.total_tps,
        "error": r.error, "summary": r.config.summary(),
        "utilization": r.utilization.summary() if r.utilization.samples else "",
        "patches": list(r.config.patches), "per_request": r.per_request,
    } for r in all_results], indent=2))
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
