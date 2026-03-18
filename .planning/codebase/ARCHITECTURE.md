# Architecture

**Analysis Date:** 2026-03-17

## Pattern Overview

**Overall:** Lazy-loading proxy with pluggable benchmark framework

**Key Characteristics:**
- Proxy-based architecture: single HTTP reverse proxy wraps llama-server lifecycle
- On-demand model loading: models load only when first requested, unload after idle timeout
- Modular benchmark framework: auto-discovering test suites with composable configurations
- GPU-aware infrastructure: automatic device reset, JIT cache persistence, utilization monitoring
- Registry-driven model management: canonical model entries with aliases and per-model configuration

## Layers

**Proxy Layer (Request Handling):**
- Purpose: Accept HTTP requests, route to loaded backend, or trigger load on-demand
- Location: `scripts/arcllm-proxy.py` (RequestHandler, HTTPHandler)
- Contains: OpenAI API compatibility layer, request routing, response translation
- Depends on: BackendManager, slotting gate, model registry
- Used by: External clients via HTTP/curl, OpenAI-compatible clients

**Backend Manager (Model Lifecycle):**
- Purpose: Start/stop llama-server subprocess, load/unload models, idle timeout
- Location: `scripts/arcllm-proxy.py` (BackendManager class)
- Contains: Process spawning, health checks, model switching, idle detection
- Depends on: Model registry, SYCL environment setup
- Used by: Proxy layer to ensure model is loaded before routing requests

**Slotting Gate (Concurrency Control):**
- Purpose: Rate-limit concurrent requests based on model's -np (parallel slots)
- Location: `scripts/arcllm-proxy.py` (SlottingGate class)
- Contains: Semaphore-based queue, max_active tracking
- Depends on: None (stateless gate)
- Used by: RequestHandler to throttle incoming requests

**Benchmark Framework (Testing Infrastructure):**
- Purpose: Discover test suites, configure servers, fire requests, capture utilization
- Location: `scripts/bench/__main__.py`, `scripts/bench/base.py`, `scripts/bench/runner.py`
- Contains: CLI entry point, base test class, benchmark runner, GPU/CPU monitor
- Depends on: BenchConfig, environment bootstrap, Monitor
- Used by: Developers running `python3 -m bench suite.test`

**Configuration System (Immutable Config):**
- Purpose: Define all tunable parameters in one frozen dataclass, derive variants with .with_()
- Location: `scripts/bench/config.py` (BenchConfig)
- Contains: Model selection, server flags, SYCL env vars, test params, build metadata
- Depends on: None (purely data)
- Used by: BenchRunner, test files, server startup

**Environment Bootstrap (Setup & Verification):**
- Purpose: Ensure render group access, source SYCL environment, initialize base env dict
- Location: `scripts/bench/env.py`
- Contains: render group addition, env script sourcing, SYCL path setup
- Depends on: None (pure shell interaction)
- Used by: BenchRunner at import time (runs in __main__)

**Monitoring (Utilization Capture):**
- Purpose: Sample GPU freq/power/temp, CPU%, RAM during tests
- Location: `scripts/bench/monitor.py` (Monitor, Utilization)
- Contains: sysfs reading for Intel Arc A770, per-GPU metrics, background thread
- Depends on: None (reads /sys/class/drm/, /proc/stat, /proc/meminfo)
- Used by: BenchRunner to annotate results with hardware state

## Data Flow

**Proxy Request Flow:**

1. Client sends HTTP request (POST /v1/chat/completions with model ID)
2. RequestHandler receives request, parses JSON
3. Handler calls backend.ensure_model(model_id)
4. BackendManager checks if model is loaded; if not, starts llama-server subprocess
5. Handler calls gate.acquire(timeout) — blocks if all slots full
6. Handler fires request to backend at 127.0.0.1:18400
7. Backend returns completion, handler translates to OpenAI format
8. Handler releases gate slot
9. RequestHandler returns response to client
10. BackendManager's idle watcher tracks time since last request; unloads after timeout

**Benchmark Test Flow:**

1. CLI entry point (`python3 -m bench suite.test`) loads env.sglang-xpu.sh
2. __main__.py discovers all BenchTest subclasses from tests/*.py
3. User selects suite and test method
4. BenchRunner.run_test(config) is called with BenchConfig variant
5. Runner resets GPUs (kill processes, hw reset, flush caches if needed)
6. Runner starts llama-server with config.server_args()
7. Monitor spawns background thread to sample GPU/CPU/RAM every 0.1s
8. Runner fires N concurrent async requests with config.concurrent
9. Each request waits up to config.timeout seconds, counts tokens
10. Monitor stops, returns Utilization summary (avg freq, power, temp per GPU)
11. Runner returns BenchResult with TPS, token counts, utilization
12. Results saved to /tmp/bench_results.json

**State Management:**

- **Proxy state**: Current model name, process PID, last_request_time (in BackendManager.lock)
- **Benchmark state**: Server process, Monitor thread, results list (in BenchRunner)
- **GPU state**: Reset between tests via sysfs, JIT cache saved to NVMe backup
- **Test discovery**: BenchTest subclasses registered at import time via __subclasses__()

## Key Abstractions

**Model Entry (Registry):**
- Purpose: Single canonical definition of a model with all its flags
- Examples: `scripts/arcllm-proxy.py` lines 59–195 (_register calls)
- Pattern: Dict with keys "name", "path", "flags", "aliases", "n_parallel"
- Usage: Aliases allow curl to request "32b" or "qwen3-32b-q4" and get the same model

**BenchConfig (Immutable Variant):**
- Purpose: Capture all tunable parameters in one object; derive A/B variants with .with_()
- Examples: `scripts/bench/config.py` (BenchConfig dataclass), test files use .with_flags()
- Pattern: @dataclass(frozen=True), never mutate, only call .with_() to derive new config
- Usage: Prevents config state explosions in test files; enables easy variant A/B testing

**BenchTest Subclass (Test Definition):**
- Purpose: Define a test suite with a base config and multiple test_* methods
- Examples: `scripts/bench/tests/test_frontier.py`, `scripts/bench/tests/test_baseline.py`
- Pattern: Set class variable `base = BenchConfig(...)`, define test_* methods, call self.run(config)
- Usage: Auto-discovery finds all subclasses; CLI maps class name → suite name

**Utilization (Hardware Metrics):**
- Purpose: Capture average GPU/CPU/RAM state during a test window
- Examples: `scripts/bench/monitor.py` (Utilization dataclass)
- Pattern: Per-GPU lists of freq/power/temp, plus CPU% and RAM GB, plus sample count
- Usage: Appended to BenchResult so each test run has hardware context

## Entry Points

**arcllm-proxy:**
- Location: `scripts/arcllm-proxy.py` (lines 350+ main entry point)
- Triggers: User runs `python3 scripts/arcllm-proxy.py` or via arcllm-server.sh
- Responsibilities: Parse ARCLLM_PORT/HOST/BACKEND_PORT env vars, start HTTPServer, handle SIGINT

**arcllm-server.sh:**
- Location: `scripts/arcllm-server.sh`
- Triggers: User runs `./scripts/arcllm-server.sh {start|stop|status|logs|models|load}`
- Responsibilities: Manage PID file, source env, spawn proxy, provide shell interface

**Benchmark CLI:**
- Location: `scripts/bench/__main__.py` main()
- Triggers: User runs `python3 -m bench suite[.test]`
- Responsibilities: Parse argv, discover suites, instantiate test class, run selected tests, print summary

## Error Handling

**Strategy:** Fail fast with explicit logging; no silent failures

**Patterns:**

- **Proxy model load failures**: BackendManager._start_backend_locked() returns bool; handler returns HTTP 500 on false
- **Server startup timeout**: BenchRunner.start_server() waits up to 300s; returns bool; test aborts if false
- **GPU DEVICE_LOST**: BenchRunner.reset_gpus() flushes JIT cache, restores from NVMe backup, re-checks sycl-ls
- **Request timeout**: BenchRunner fires async requests with asyncio.wait_for(timeout=config.timeout)
- **Model file not found**: BackendManager logs error, returns False; proxy refuses request
- **Environment load failure**: env.py exits with sys.exit(1) if env.sglang-xpu.sh sources with non-zero exit code

## Cross-Cutting Concerns

**Logging:**
- Proxy: Python logging module, to /tmp/arcllm-server.log (log_fh file handle appended per model)
- Benchmark: print() to stdout; test logs saved to /tmp/bench_logs/*.log per test
- Approach: Explicit per-component; messages include timestamps, PIDs, model names

**Validation:**
- Proxy: Model ID validation via MODELS.get(model_id); path existence check before spawn
- Benchmark: BenchConfig fields validated by Python dataclass (immutable); model/binary path checks at start_server()
- Approach: Fail at entry point, not mid-operation

**Authentication:**
- Proxy: None enforced (local reverse proxy assumes trusted network)
- Benchmark: render group membership verified in env.py; re-execs under `sg render` if needed

**SYCL Environment:**
- Proxy: SYCL_ENV dict (lines 34–39) merged into subprocess.Popen env before llama-server starts
- Benchmark: BASE_ENV loaded at import time from env.sglang-xpu.sh; BenchConfig.sycl_env() method returns env dict
- Pattern: All GPU code runs with explicit GGML_SYCL_*, SYCL_PI_*, ZE_* variables set

---

*Architecture analysis: 2026-03-17*
