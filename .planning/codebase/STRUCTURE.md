# Codebase Structure

**Analysis Date:** 2026-03-17

## Directory Layout

```
/home/ryan/llm-stack/
├── scripts/                     # Operational tools and benchmarks
│   ├── arcllm-proxy.py          # Lazy-loading reverse proxy for llama-server
│   ├── arcllm-server.sh         # Shell interface for proxy (start/stop/status)
│   ├── bench/                   # Benchmark framework
│   │   ├── __main__.py          # CLI entry point (python3 -m bench)
│   │   ├── __init__.py          # Package marker
│   │   ├── base.py              # BenchTest base class
│   │   ├── config.py            # BenchConfig immutable dataclass
│   │   ├── runner.py            # BenchRunner (GPU reset, server lifecycle, monitor)
│   │   ├── monitor.py           # Monitor (GPU/CPU/RAM utilization capture)
│   │   ├── env.py               # Environment bootstrap (render group, SYCL sourcing)
│   │   └── tests/               # Auto-discovered test suites
│   │       ├── __init__.py      # Package marker
│   │       ├── test_baseline.py # Serial throughput baselines (q4km, q8)
│   │       ├── test_frontier.py # Regression test: Q4_K_M np=16 FUSED_MMQ=1 (21.7 t/s)
│   │       ├── test_parallel.py # Parallel throughput variants (np=2/4/8/16)
│   │       ├── test_sycl_env.py # SYCL environment tuning (graph, cmdlist, affinity)
│   │       ├── test_thinking.py # Reasoning mode (--reasoning-budget) variants
│   │       ├── test_row_split.py # Row-split architecture tests
│   │       ├── test_fused_mmq.py # Fused dequant+matmul kernel variants
│   │       ├── test_mmvq_nan.py # MMVQ NaN overflow bug reproduction
│   │       ├── test_q8_1_corruption.py # Q8_1 quantization bug
│   │       └── test_glm47.py    # GLM-4.7-Flash MoE variant
│   ├── bench_fire_requests.py   # Standalone request firing tool
│   ├── bench_gpu_monitor.py     # Standalone GPU monitoring tool
│   ├── bench_parallel.sh        # Legacy shell-based parallel benchmark
│   ├── build_llama_sycl.sh      # Build llama.cpp with SYCL backend
│   └── [other legacy scripts]   # Deprecated tools (benches, prompts, host watch)
├── llama.cpp/                   # Submodule: llama.cpp source tree
│   ├── build-sycl/              # SYCL build output (use this, not build/)
│   │   └── bin/llama-server     # Binary: HTTP server for inference
│   ├── src/                     # Source: model loading, inference kernels
│   ├── ggml/                    # GGML library: quantization, operations
│   ├── gguf-py/                 # Python GGUF tools (conversion, inspection)
│   └── [build/ tests/ docs/]    # Other directories
├── models/                      # Model weight files (GGUF format)
│   ├── Qwen/                    # Qwen models
│   │   ├── Qwen3-32B-GGUF/      # Qwen3-32B Q4_K_M (19 GB) — default model
│   │   ├── Qwen3-0.6B-GGUF/     # Qwen3-0.6B (speculative draft)
│   │   ├── Qwen3.5-9B-GGUF/     # Qwen3.5-9B baseline
│   │   ├── Qwen3.5-27B-GGUF/    # Qwen3.5-27B mid-size
│   │   └── [other variants]     # Abliterated, Q8_0, MoE variants
│   ├── GLM-4.7-Flash/           # GLM-4.7-Flash MoE 30B/3.6B
│   └── NVIDIA/                  # Nemotron-120B models
├── cache/                       # Runtime caches (persisted across test runs)
│   ├── slots/                   # KV cache snapshots (--slot-save-path)
│   └── neo_compiler_cache/      # JIT-compiled GPU code backup
├── config/                      # Configuration files
│   ├── nemotron_120b_optimal.env # Environment setup for Nemotron
│   └── config.env.example       # Template (deleted from git)
├── env.sglang-xpu.sh            # SYCL environment bootstrap (conda, GGML_SYCL_*, ZE_*)
├── CLAUDE.md                    # Project instructions (mandatory for GPU testing)
├── README.md                    # Project overview
├── .planning/                   # GSD planning outputs (generated)
│   └── codebase/                # Codebase analysis documents
│       ├── ARCHITECTURE.md      # Architecture patterns
│       └── STRUCTURE.md         # This file
├── docs/                        # Project documentation
├── research/                    # Research artifacts and notes
└── OBLITERATUS/                 # Separate project: model evaluation framework
```

## Directory Purposes

**scripts/:**
- Purpose: Operational tools and benchmarks (not core inference)
- Contains: Proxy, shell wrappers, benchmark framework, standalone tools
- Key files: `arcllm-proxy.py`, `arcllm-server.sh`, `bench/__main__.py`

**scripts/bench/:**
- Purpose: Test harness with auto-discovery and GPU infrastructure
- Contains: Base test class, config system, runner (GPU reset + monitor), environment bootstrap
- Key files: `base.py` (BenchTest), `config.py` (BenchConfig), `runner.py` (lifecycle)

**scripts/bench/tests/:**
- Purpose: Individual test suites (auto-discovered by __main__.py)
- Contains: test_*.py files; each is a BenchTest subclass with multiple test_* methods
- Key files: `test_frontier.py` (regression baseline), `test_baseline.py` (serial), `test_parallel.py` (scaling)

**llama.cpp/build-sycl/bin/:**
- Purpose: Compiled llama-server binary (inference server)
- Contains: Single executable; started by BackendManager/BenchRunner with model + flags
- Key files: `llama-server` (binary; ~20 MB)

**models/:**
- Purpose: Downloaded GGUF model weight files (not version-controlled)
- Contains: Qwen, GLM, Nemotron models; organized by provider then quantization
- Key files: Qwen3-32B-Q4_K_M.gguf (19 GB, default), others up to 235B

**cache/:**
- Purpose: Runtime state (KV cache snapshots, JIT cache)
- Contains: Slot save files (--slot-save-path), neo_compiler_cache (backup from crashes)
- Generation: Slot files created by llama-server; JIT cache backed up by BenchRunner

**config/:**
- Purpose: Environment and server configuration
- Contains: env files for specific models (e.g., nemotron_120b_optimal.env)
- Generation: User-created or template-based; not committed for secrets

## Key File Locations

**Entry Points:**

- `scripts/arcllm-proxy.py`: Start proxy (can be run directly or via arcllm-server.sh)
- `scripts/arcllm-server.sh`: Shell interface for proxy (start, stop, status, load, unload)
- `scripts/bench/__main__.py`: Benchmark CLI entry point (python3 -m bench)

**Configuration:**

- `env.sglang-xpu.sh`: SYCL environment bootstrap (mandatory; sources conda, sets GGML_* vars)
- `scripts/bench/config.py`: All tunable benchmark parameters in BenchConfig dataclass
- `scripts/arcllm-proxy.py` lines 41–195: Model registry (_register calls)

**Core Logic:**

- `scripts/arcllm-proxy.py` lines 202–350: BackendManager (model lifecycle)
- `scripts/arcllm-proxy.py` lines 350–450: RequestHandler (HTTP routing)
- `scripts/bench/runner.py` lines 47–250: BenchRunner (GPU reset, server start, request firing)
- `scripts/bench/monitor.py` lines 1–150: Monitor (utilization sampling)

**Testing:**

- `scripts/bench/tests/test_frontier.py`: Regression baseline (21.7 t/s)
- `scripts/bench/tests/test_baseline.py`: Serial q4km and q8 baselines
- `scripts/bench/tests/test_parallel.py`: Parallel scaling (np=2/4/8/16)

## Naming Conventions

**Files:**

- `arcllm-*.py` or `arcllm-*.sh`: Proxy-related (arcllm = Arc LLM)
- `bench_*.py` or `bench_*.sh`: Legacy benchmarking tools
- `test_*.py`: Benchmark test suites (auto-discovered in scripts/bench/tests/)
- `test_*.cpp`: Standalone GPU/SYCL testing utilities

**Directories:**

- `build-sycl/`: SYCL-specific build output (use this, not `build/`)
- `tests/`: Auto-discovered test modules in scripts/bench/tests/
- `cache/`: Runtime-generated cache (slots/, neo_compiler_cache/)
- `bench/`: Framework code (not tests; tests go in bench/tests/)

**Python Classes:**

- `Test*`: BenchTest subclasses (e.g., TestFrontier, TestBaseline)
- `Bench*`: Utility classes (BenchConfig, BenchRunner, BenchTest, BenchResult)
- `*Manager`: Long-lived lifecycle managers (BackendManager, Monitor)
- `*Handler`: HTTP request handlers (RequestHandler)

**Python Functions:**

- `ensure_*`: Idempotent setup (ensure_render_group, ensure_model)
- `cmd_*`: CLI commands in arcllm-server.sh
- `test_*`: Test methods in BenchTest subclasses
- `_*`: Private methods (single leading underscore)

## Where to Add New Code

**New Model in Registry:**
- File: `scripts/arcllm-proxy.py` lines 41–195
- Pattern: Call `_register(name, path, flags, aliases=[...])` with canonical name, model path, llama-server flags, optional aliases
- Example: See lines 59–68 (nemotron-120b), 101–109 (qwen3-32b default)

**New Benchmark Test Suite:**
- Primary code: `scripts/bench/tests/test_<name>.py`
- Tests: Methods like `def test_variant1(self):` in the BenchTest subclass
- Pattern: Create TestName(BenchTest) with base = BenchConfig(...), then call self.run(self.base.with_(...))
- Example: See `scripts/bench/tests/test_frontier.py` (lines 21–43)

**New SYCL Environment Variable:**
- Configuration: `scripts/bench/config.py` BenchConfig dataclass (add field if it's tunable)
- Usage: In test file, call `.with_flags(GGML_SYCL_FLAG="value")` or set field directly with `.with_(...)`
- Pattern: Frozen dataclass; derive variants with .with_(), never mutate
- Example: See config.py lines 52–59 (sycl_flags tuple, with_flags method)

**Utilities/Helpers:**
- Shared helpers: `scripts/bench/runner.py` (BenchRunner methods) or `scripts/bench/monitor.py` (Monitor)
- Per-benchmark: Custom methods in BenchTest subclass
- Pattern: Avoid global state; encapsulate in classes

**Shell Scripts:**
- Operational wrappers: `scripts/*.sh`
- Pattern: Source `env.sglang-xpu.sh`, set SYCL_* vars, spawn Python or C++ binary
- Example: `arcllm-server.sh` (lines 36–86)

## Special Directories

**cache/:**
- Purpose: Runtime caches (KV slots, JIT compilation)
- Generated: By llama-server (--slot-save-path) and BenchRunner (_save_jit_cache)
- Committed: neo_compiler_cache/ is backed up; slots/ are transient

**build-sycl/:**
- Purpose: SYCL-compiled llama.cpp binary and dependencies
- Generated: By build_llama_sycl.sh or CMake
- Committed: No; binary + build artifacts ignored by .gitignore

**/tmp/bench_logs/, /tmp/bench_results.json:**
- Purpose: Test output logs and JSON results summary
- Generated: By BenchRunner per test
- Committed: No; ephemeral

**models/:**
- Purpose: GGUF weight files (not version-controlled)
- Generated: Downloaded manually or via scripts
- Committed: No; too large; listed in .gitignore

---

*Structure analysis: 2026-03-17*
