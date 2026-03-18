# Testing Patterns

**Analysis Date:** 2026-03-17

## Test Framework

**Test Infrastructure:**
- **Benchmark framework**: Custom framework at `scripts/bench/` for GPU testing
  - Entrypoint: `python3 -m bench <suite[.test]>`
  - Auto-discovers test classes via `BenchTest` subclass introspection
  - Handles GPU reset, server lifecycle, utilization monitoring

- **Unit tests**: pytest (llama.cpp tools/server)
  - Config: `llama.cpp/tools/server/tests/pytest.ini`
  - Runner: `pytest tools/server/tests/unit/`

**Run Commands:**

```bash
# Benchmark framework (from scripts/)
python3 -m bench help                    # List all suites and tests
python3 -m bench frontier                # Run suite (all tests)
python3 -m bench frontier.np16           # Run specific test
python3 -m bench baseline parallel thinking  # Run multiple suites

# Unit tests (from llama.cpp/)
pytest tools/server/tests/unit/ -v       # Run all tests
pytest tools/server/tests/unit/test_completion.py -k "test_completion" # Specific test
pytest -m "not slow" tools/server/tests/unit/  # Skip marked slow tests
```

## Test File Organization

**Benchmark framework:**
- **Location**: `scripts/bench/tests/test_<name>.py`
- **Naming**: One test class per file, class name `Test<Suite>` (e.g., `TestBaseline`, `TestFrontier`)
- **Discovery**: Auto-discovered at import time from `bench.tests` package
  - Filename must start with `test_`
  - Class must inherit from `BenchTest`
  - No registration needed

**Unit tests (llama.cpp):**
- **Location**: `llama.cpp/tools/server/tests/unit/test_<name>.py`
- **Naming**: One file per API/feature, e.g., `test_completion.py`, `test_embeddings.py`, `test_router.py`
- **Convention**: `test_<feature>()` functions at module level
- **Fixtures**: Per-test setup via `@pytest.fixture(autouse=True)`

**Example benchmark test structure** (`scripts/bench/tests/test_baseline.py`):
```python
"""Serial throughput baselines."""
from bench.base import BenchTest
from bench.config import BenchConfig


class TestBaseline(BenchTest):
    base = BenchConfig(n_parallel=1, concurrent=1)

    def test_q4km(self):
        self.run(self.base.with_(name="baseline_q4km", model="q4km"))

    def test_q8(self):
        self.run(self.base.with_(name="baseline_q8", model="q8"))
```

**Key points:**
- `base = BenchConfig(...)` — Class variable with default config for all tests in suite
- `test_*()` methods — Inherit from base, derive variants with `.with_()`
- `self.run()` — Execute test via runner, collect results

## Test Structure

**Benchmark suite organization:**
```python
"""One-line description. Context. Known results.

## History
- Date: What changed, why, results

## Issues Found & Fixed
1. Issue description
   - Root cause
   - Fix (code location)
   - Result: metrics before/after

## Current Status
- What works, what doesn't

## Results
- Table of configs and outcomes

## Code Changes
- Affected files and lines
"""
from bench.base import BenchTest
from bench.config import BenchConfig

CUSTOM_PROMPT = "..."  # Module-level constants

class TestSuiteName(BenchTest):
    base = BenchConfig(model="q4km", ...)

    def test_variant_1(self):
        self.run(self.base.with_(name="...", ...))

    def test_variant_2(self):
        self.run(self.base.with_(name="...", ...))
```

**Key pattern from** `/home/ryan/llm-stack/scripts/bench/tests/test_frontier.py`:
- Module docstring IS the documentation (issue context, fix approach, results table)
- Each test method is <10 lines (just call `.run()` with variant config)
- Results stored in `BenchResult` dataclass, displayed in summary table

**Unit test structure** (llama.cpp):
```python
import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()

def test_server_start_simple():
    global server
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200

@pytest.mark.parametrize("prompt,n_predict,re_content,n_prompt,n_predicted,truncated,return_tokens", [
    ("I believe the meaning of life is", 8, "(going|bed)+", 18, 8, False, False),
    ("Write a joke about AI...", 64, "(princesses|everyone|kids)+", 46, 64, False, True),
])
def test_completion(prompt: str, n_predict: int, re_content: str, n_prompt: int, n_predicted: int, truncated: bool, return_tokens: bool):
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={...})
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_prompt
    assert res.body["timings"]["predicted_n"] == n_predicted
```

**Key patterns:**
- Global `server` instance (shared across tests in file)
- `@pytest.fixture(autouse=True)` creates fresh server per test
- `@pytest.mark.parametrize()` for data-driven tests
- `server.make_request()` / `server.make_stream_request()` for HTTP calls
- `assert` for validation (pytest collects assertions)

## Mocking

**Framework:** No mocking framework detected; direct HTTP calls to real servers

**Patterns in bench:**
- No mocks in benchmark framework (tests actual GPU, not stubs)
- `subprocess.Popen()` for server lifecycle (real process, not mocked)
- `urllib.request.urlopen()` for health checks (direct HTTP, not mocked)
- GPU metrics read directly from sysfs (not mocked)

**Patterns in llama.cpp tests:**
- `ServerPreset` fixture (from `utils.py`) — manages real test server
- Direct HTTP requests via `requests` library or OpenAI client
- Regex matching for assertions: `match_regex("(going|bed)+", res.body["content"])`
- Stream testing: `make_stream_request()` returns iterator of JSON chunks

**What to Mock (guidelines):**
- External services (Redis, database) — not present in this codebase
- Network timeouts — test with `timeout=` parameters instead

**What NOT to Mock:**
- GPU operations (bench framework stress-tests real hardware)
- llama-server responses (must test actual model output)
- System metrics (sysfs reads are essential to diagnose GPU state)

## Fixtures and Factories

**Test Data (bench framework):**
- **Prompts**: Module-level constants in each test file
  ```python
  SHORT_PROMPT = "What is 2+2?"
  THINK_PROMPT = "Think step by step. What are the top 5 most impactful inventions..."
  DEFAULT_PROMPT = "Explain the difference between TCP and UDP..."
  ```
- **Configs**: Frozen dataclass `BenchConfig` (immutable, derived via `.with_()`)
  ```python
  base = BenchConfig(model="q4km", n_parallel=16, concurrent=16, context=32768)
  variant = base.with_(name="foo", disable_graph=True)  # Returns new instance
  ```
- **Location**: `/home/ryan/llm-stack/scripts/bench/config.py` (centralized)

**Test Data (llama.cpp):**
- **Server presets**: `ServerPreset.tinyllama2()` from `utils.py`
- **Parametrization**: Data-driven via `@pytest.mark.parametrize()`
  ```python
  @pytest.mark.parametrize("prompt,n_predict,re_content", [
      ("I believe...", 8, "(going|bed)+"),
      ("Write a joke...", 64, "(princesses|everyone)+"),
  ])
  ```
- **Location**: Test files (inline) or `utils.py` (shared fixtures)

## Coverage

**Requirements:** Not enforced (no `.coveragerc` or `pytest --cov` seen)

**Approach:**
- Bench framework: 100% of test paths exercised (regression tests for known results)
- llama.cpp: Completion, embedding, chat, streaming endpoints covered
- No formal coverage metrics tracked

## Test Types

**Unit Tests:**
- Scope: Individual API endpoints in llama.cpp (completion, embedding, etc.)
- Approach: Real server instance, parametrized HTTP requests, regex assertion
- Example: `test_completion()` — POST /completion, validate status, tokens, output format

**Integration Tests:**
- Scope: Bench framework — GPU coordination, model loading, multi-slot concurrency
- Approach: Real GPUs, real llama-server, stress testing with async requests
- Examples:
  - `test_frontier.np16()` — 16 concurrent slots on 3x Arc A770, measure throughput
  - `test_row_split.test_q4km_np2_200tok()` — Multi-slot with row-split, 200 token generation
  - `test_mmvq_nan.4x_1024_crashes()` — Detect DEVICE_LOST crash at long sequences

**E2E Tests:**
- Framework: Not explicitly labeled but bench tests are full E2E
- Server lifecycle: GPU reset → startup → requests → metrics → shutdown
- Validation: Throughput (t/s), utilization (freq/power/temp), crash detection

**Performance/Regression Tests:**
- Maintained in bench framework with expected results documented
- `test_frontier.np16()` — Regression baseline: 21.7 t/s (FUSED_MMQ=1)
- `test_baseline.test_q4km()` — Serial throughput baseline
- Results saved to `/tmp/bench_results.json` for trend analysis

## Common Patterns

**Async Testing (bench):**
```python
# From runner.py — fire N concurrent requests while capturing GPU metrics
async def _fire_requests(self, config: BenchConfig) -> tuple[int, int, int]:
    """Fire concurrent requests to backend, return (completed, failed, total_tokens)."""
    # Implemented in runner.py as asyncio coroutine
    # Captures per-request latency, errors, response content
```

**Error Testing:**
- Crash detection: Server exit code non-zero → log error lines → mark FAILED
  ```python
  if self._proc.poll() is not None:
      print(f"server died (exit {self._proc.returncode})")
      self._proc = None
      return False
  ```
- DEVICE_LOST detection: Log parsing for "device_lost" in server stderr
  ```python
  if any(k in lo for k in ("error", "failed", "alloc", "device_lost")):
      print(f"  {line.strip()}")
  ```
- Timeout detection: `deadline = time.monotonic() + timeout`, check each iteration
- Queue timeout: `gate.acquire(priority, timeout=300)` returns False on timeout

**GPU State Management:**
```python
# From runner.py
def reset_gpus(self, wait: int = 5, flush_cache: bool = False) -> bool:
    """Full GPU reset: kill processes → hw reset → verify."""
    self._kill_gpu_consumers()
    self._hw_reset_gpus()  # Write '1' to sysfs reset files
    if flush_cache:
        self._flush_level_zero_cache()
        self._restore_jit_cache()  # Restore JIT cache from NVMe backup
    time.sleep(wait)
    return self.check_gpus()  # Verify sycl-ls sees all 3 devices
```

**Server Health Polling:**
```python
# From runner.py
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    if self._proc.poll() is not None:
        return False  # Process died
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:8400/health", timeout=2)
        if resp.status == 200:
            return True  # Server ready
    except Exception:
        pass
    time.sleep(3)
return False  # Timeout
```

**Streaming Responses (llama.cpp):**
```python
def test_completion_stream(prompt: str, n_predict: int, ...):
    res = server.make_stream_request("POST", "/completion", data={...})
    content = ""
    for data in res:
        if data["stop"]:
            assert data["timings"]["prompt_n"] == n_prompt
        else:
            content += data["content"]
    assert match_regex(re_content, content)
```

## Test Execution Flow

**Benchmark suite (example: test_frontier):**
1. Discovery: `python3 -m bench frontier` → `__main__.py` loads `BenchTest` subclasses
2. Suite init: `TestFrontier(runner)` → inherits `base` config
3. Test method: `test_np16()` → calls `self.run(config)` → `runner.run_test(config)`
4. Runner steps:
   - `reset_gpus()` — Kill servers, hardware reset, verify GPUs responsive
   - `start_server()` — Spawn llama-server with config flags, poll health
   - `_fire_requests()` — Async fire `concurrent` requests, capture metrics
   - `stop_server()` — Terminate process, save KV cache slot
   - `Monitor` collects GPU freq/power/temp, CPU%, RAM% throughout
5. Result: `BenchResult` object with (completed, failed, tokens, wall_time, tps, utilization)
6. Summary: Collect all results, print table, save JSON to `/tmp/bench_results.json`

**Unit test suite (example: test_completion):**
1. Fixture setup: `create_server()` → `ServerPreset.tinyllama2()` spawns tiny model
2. Test runs: Multiple test functions with global `server` instance
3. Each test: `server.start()` → `server.make_request()` → assertions
4. Cleanup: `server.stop()` (via fixture or explicit teardown)
5. Output: pytest summary (PASSED/FAILED), exit code 0/1

## Test Documentation

**Test files ARE the documentation** (per CLAUDE.md):
- Module docstring explains the issue/feature, reproduction steps, results, and status
- NOT separate markdown docs — test file IS the context
- Update docstring when issue is fixed; don't delete test (becomes regression test)

**Example:** `/home/ryan/llm-stack/scripts/bench/tests/test_row_split.py`
- ~123 lines of module docstring covering:
  - Context: why row-split exists
  - 8 issues found & fixed with code locations
  - Current status and limitations (np=2 crashes at 56-83s)
  - Results table (configs and outcomes)
  - Code changes with file:line references
  - Bottleneck analysis (1.55s/token = 99% host stalls)

---

*Testing analysis: 2026-03-17*
