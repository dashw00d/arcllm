# Coding Conventions

**Analysis Date:** 2026-03-17

## Naming Patterns

**Files:**
- Test files: `test_<name>.py` (e.g., `test_baseline.py`, `test_frontier.py`)
- Module files: `<name>.py` (lowercase, underscores for multi-word)
- Script files: `<name>-<name>.py` (e.g., `arcllm-proxy.py`)

**Functions:**
- Lowercase with underscores: `_read_sysfs()`, `ensure_render_group()`, `start_server()`
- Private/internal: leading underscore `_make_env()`, `_hw_reset_gpus()`
- Test methods: `test_<description>()` (e.g., `test_np16()`, `test_q4km_np1_100tok()`)

**Variables:**
- Instance attributes: lowercase with underscores: `self._proc`, `self.current_model`, `self.last_request_time`
- Class variables: UPPERCASE for constants: `LISTEN_PORT`, `IDLE_TIMEOUT`, `SLOT_CACHE`
- Local variables: lowercase with underscores: `model_id`, `content_length`, `deadline`

**Types & Classes:**
- Class names: PascalCase: `BackendManager`, `PriorityGate`, `BenchTest`, `BenchRunner`
- Dataclass fields: lowercase: `completed`, `failed`, `total_tokens`, `wall_time`

## Code Style

**Formatting:**
- No auto-formatter configured (flake8 only)
- Line length limit: 125 characters (per llama.cpp `.flake8`)
- Indentation: 4 spaces (Python standard)
- Imports grouped: stdlib → third-party → local

**Linting:**
- Tool: `flake8` (llama.cpp uses this)
- Config: `/home/ryan/llm-stack/llama.cpp/.flake8`
- Ignored rules: E203, E211, E221, E225, E231, E241, E251, E261, E266, E501, E701, E704, W503
- Max complexity: not enforced

**String formatting:**
- f-strings preferred for dynamic content: `f"model={self.model}"`, `f"{self.name:.1f}"`
- JSON serialization: `json.dumps()`, `json.loads()` with explicit newlines in dumps
- Path formatting: `Path()` for filesystem operations, `str()` when passing to subprocess

## Import Organization

**Order:**
1. `from __future__ import annotations` (Python 3.9+ typing)
2. Standard library: `import os`, `import sys`, `import json`, `import threading`
3. Third-party: `import requests` (in llama.cpp tests), `from dataclasses import dataclass`
4. Local: `from bench.base import BenchTest`, `from .config import BenchConfig`

**Path Aliases:**
- No path aliases configured (no `@alias` in tsconfig or similar)
- Relative imports: `from .config import`, `from ..runner import` (dot notation)

## Error Handling

**Patterns:**
- `try/except Exception:` for broad error handling, then silently continue or log
- Return False on error, return True on success (e.g., `ensure_model()` → bool)
- Return None or empty dict on error: `return {}`, `return False`
- Logging errors before returning: `log.error()`, `log.warning()`
- Exception context preserved in messages: `except Exception as e:` then log or re-raise

**Example from `/home/ryan/llm-stack/scripts/arcllm-proxy.py` (lines 321-337):**
```python
def _slot_restore(self, model_name: str):
    """Restore slot 0 KV cache from disk."""
    cache_file = Path(str(SLOT_CACHE) + f"{model_name}.bin")
    if not cache_file.exists():
        log.info("No cached slot for %s", model_name)
        return False
    try:
        conn = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=120)
        # ... request ...
        if resp.status == 200:
            log.info("Slot restored: %s.bin (%s tokens)", model_name, tokens)
            return True
        else:
            log.warning("Slot restore failed: %s", data)
    except Exception as e:
        log.warning("Slot restore error: %s", e)
    return False
```

## Logging

**Framework:** `logging` stdlib module

**Setup:**
```python
log = logging.getLogger("arcllm")  # or "bench"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
```

**Patterns:**
- `log.info()` for normal operation: `log.info("Model %s ready", canonical)`
- `log.warning()` for recoverable errors: `log.warning("Slot save failed: %s", data)`
- `log.error()` for critical failures: `log.error("Model file not found: %s", model_path)`
- `log.debug()` for verbose debugging (used in proxy for HTTP handler spam suppression)

## Comments

**When to Comment:**
- Explain WHY, not WHAT (code shows the what): "Layer-split crashes at ~77s due to L0 command overlap"
- Algorithm rationale: "All slots generate simultaneously → matmul batch=16 → fused kernel's sweet spot"
- Non-obvious business logic: "High-priority waits only for a free slot; low-priority waits for free slot AND no high-priority pending"
- Known limitations: "No spec decoding: 30B-A3B draft adds marginal benefit at this speed"

**JSDoc/DocStrings:**
- Module docstrings: Explain what file does and key context
  ```python
  """arcllm-proxy — Lazy-loading reverse proxy for llama-server.

  Behaves like Ollama: listens immediately, loads models on first request,
  unloads after idle timeout. No VRAM used until a request arrives.

  Model registry is defined in TOML config or env vars.
  """
  ```
- Class docstrings: Purpose and key behavior
  ```python
  class BackendManager:
      """Manages a single llama-server subprocess, loading/unloading on demand."""
  ```
- Method docstrings: One-liner + purpose
  ```python
  def ensure_model(self, model_id: str, timeout: float = 300) -> bool:
      """Ensure the given model is loaded. Blocks until ready or timeout."""
  ```
- Test docstrings: Context, issue, results, and test outcome
  ```python
  def test_events_q4km_np1_100tok(self):
      """32B Q4_K_M row-split np=1, 100 tokens — OOO + event sync.
      Baseline comparison: 0.6 t/s with stream->wait().
      Expected: compute-bound throughput (no host stalls in Phase 1)."""
  ```

## Function Design

**Size:**
- Aim for <50 lines per function
- Larger functions broken into logical phases with comments: "# 1. Kill GPU consumers", "# 2. Hardware reset"
- Test methods often 5-10 lines (delegate to framework runner)

**Parameters:**
- Type hints for public methods: `def start_server(self, config: BenchConfig, timeout: int = 300) -> bool:`
- Default values at end: `timeout: float = 300`
- Avoid bare `*args`, `**kwargs` (prefer explicit parameters)
- Keyword-only args when order matters: `stream_request(..., data=..., timeout=...)`

**Return Values:**
- Boolean for success/failure: `ensure_model() → bool`
- None or dict for complex results: `status() → dict`
- Tuple for multiple related values: `_read_cpu() → tuple[int, int]`
- List for collections: `list[BenchResult]`

## Module Design

**Exports:**
- Classes and functions at module level (no `__all__` seen in project)
- Private functions prefixed with `_` (used internally within class or module)
- Test discovery by convention: `BenchTest` subclasses auto-discovered via `__subclasses__()`

**Barrel Files:**
- `/home/ryan/llm-stack/scripts/bench/__init__.py` is minimal (just `from __future__`)
- llama.cpp test `utils.py` imports server utilities but not included in main tests

**Module organization in bench:**
- `base.py` — `BenchTest` base class
- `config.py` — `BenchConfig` dataclass with all tunable levers
- `runner.py` — `BenchRunner` orchestrating GPU reset, server lifecycle, monitoring
- `env.py` — Bootstrap render group + SYCL env loading (run at import time)
- `monitor.py` — `Monitor` class capturing GPU/CPU/RAM metrics
- `tests/*.py` — Individual test suites inheriting `BenchTest`

## Environment Variables

**Supported in bench:**
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` — 0 (batched) or 1 (immediate)
- `GGML_SYCL_DISABLE_GRAPH` — 1 (safe) or 0 (untested at long sequences)
- `GGML_SYCL_ROW_EVENTS` — 1 (enable OOO queue + event sync), 0 (legacy stream->wait)
- `ZE_AFFINITY_MASK` — GPU device list (e.g., "0,1,2" for 3 GPUs)
- `ZES_ENABLE_SYSMAN` — 1 (enable sysfs power/temp metrics)
- `XDG_CACHE_HOME` — JIT cache location (fallback to ~/.cache)

**Set via config in bench:**
- `BenchConfig.sycl_env()` → dict of env vars
- `BenchConfig.with_flags(KEY="value")` → sets `GGML_SYCL_KEY=value`

---

*Convention analysis: 2026-03-17*
