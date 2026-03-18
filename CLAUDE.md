# llm-stack

3x Intel Arc A770 (48GB VRAM), i9-7900X, 64GB RAM. Runs Qwen3-32B via llama.cpp with SYCL backend.

## Benchmark Framework — MANDATORY for all GPU/model/server testing

**NEVER run raw bash/curl commands to test llama-server configurations.** Use the benchmark framework at `scripts/bench/`. This is not optional.

### Why

Raw bash testing caused weeks of wasted time:
- Crashed GPUs left in DEVICE_LOST state, corrupting subsequent tests
- No GPU utilization data, so we couldn't tell if GPUs were actually working
- No automatic GPU reset between tests, so failures cascaded
- No record of what was tested, so the same dead ends were re-explored
- Environment issues (render group, conda, SYCL env) caused false negatives that were blamed on the wrong thing

The framework handles all of this automatically.

### How to use

```bash
# From scripts/ directory:
python3 -m bench help                    # list all suites and tests
python3 -m bench frontier                # run the regression baseline (17.5 t/s np=16)
python3 -m bench thinking.1x_1024tok    # run a specific test
python3 -m bench baseline parallel       # run multiple suites
```

The framework automatically:
- Acquires the `render` group (re-execs under `sg render` if needed)
- Sources `env.sglang-xpu.sh` and builds the full SYCL environment
- Resets GPUs between tests (kills stale servers, waits for recovery)
- Starts/stops llama-server with the exact config
- Fires concurrent async requests
- Captures GPU freq/power/temp, CPU%, RAM during the test
- Prints results with utilization data
- Saves JSON to `/tmp/bench_results.json`

### How to add tests

Create `scripts/bench/tests/test_<name>.py`:

```python
"""One-line description of what this tests.

## Context
Why this test exists, what bug/feature it exercises.

## Results
What we found (update as you run it).

## Relevant Files
- path/to/file.cpp — what's there
"""
from bench.base import BenchTest
from bench.config import BenchConfig

class TestName(BenchTest):
    base = BenchConfig(model="q4km", n_parallel=4, concurrent=4)

    def test_the_thing(self):
        self.run(self.base.with_(name="descriptive_name", disable_graph=True))
```

It auto-discovers — no registration needed. Run with `python3 -m bench name`.

### Tests ARE the documentation

Each test file's module docstring is the living record of:
- What the issue/feature is
- How to reproduce it
- What we tried and what happened
- Crash logs and error patterns
- Current status and workarounds

Do NOT create separate markdown docs for GPU/model/SYCL issues. The test file IS the doc. When an issue is fixed, update the docstring — don't delete the test (it becomes the regression test).

### Config levers (scripts/bench/config.py)

`BenchConfig` has every tunable parameter as a field. Use `.with_()` to derive variants:
- Server: `model`, `split_mode`, `n_parallel`, `context`, `batch`, `ubatch`, `reasoning_budget`, `kv_quant`, `flash_attn`, `threads`
- SYCL env: `disable_graph`, `immediate_cmdlists`, `affinity`, `row_events`
- Experimental: `sycl_flags` via `.with_flags(FUSED_MMQ="1")` for A/B testing new GGML_SYCL_* env vars
- Test: `concurrent`, `max_tokens`, `prompt`, `timeout`
- Build: `build` (subdir), `patches` (list of descriptions for patched builds)

### Key known results

| Config | Result | Test |
|--------|--------|------|
| Q4_K_M np=16 cmdlist=0 | 17.5 t/s, 16/16 ok | `frontier.np16` |
| Q8_0 any np | 3.3 t/s max (bandwidth-bound) | `baseline.q8` |
| Row-split stream->wait | 0.6 t/s np=1 (host stall bound) | `rowsplit.q4km_np1_100tok` |
| Row-split OOO+events | TBD (testing) | `rowsplit.events_q4km_np1_100tok` |
| Row-split event merge np=1 | TBD (pending hardware run) | `rowsplit.events_merge_glm47_np1_500tok` |
| Batch/ubatch tuning | Zero effect | `syclenv.*` |
| MMVQ NaN at 4x×1024tok | Crashes ~120s | `mmvqnan.4x_1024_crashes` |
| Thinking 1x×1024tok | Works fine, 6.6 t/s | `thinking.1x_1024tok` |
| Fused Q8_0 pp16 (1xA770) | 141 t/s (2.2x vs baseline 64) | `fusedmmq.fused_dp4a_pp128` |
| Fused Q4_K_M np=16 3xA770 | **21.7 t/s (+25% vs 17.4)** | `fusedmmq.np16_fused` |

## Project structure

- `scripts/arcllm-proxy.py` — Lazy-loading reverse proxy for llama-server (Ollama-like)
- `scripts/arcllm-server.sh` — Start/stop/status wrapper for the proxy
- `scripts/bench/` — Benchmark framework (see above)
- `llama.cpp/build-sycl/` — Clean SYCL build (use this, not `build/`)
- `env.sglang-xpu.sh` — SYCL/conda/runtime environment setup
- `models/` — GGUF model files

## Environment

- Always source `env.sglang-xpu.sh` before running anything SYCL
- User must be in `render` group for GPU access
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` (batched) gives +7.5% throughput
- `GGML_SYCL_DISABLE_GRAPH=1` is safer; `=0` is untested at long sequences
- `GGML_SYCL_ROW_EVENTS=1` enables OOO queue + event-based sync for row-split (CUDA pattern)
