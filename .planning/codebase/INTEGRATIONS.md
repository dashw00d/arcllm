# External Integrations

**Analysis Date:** 2026-03-17

## APIs & External Services

**Model Downloads:**
- Hugging Face Hub (`huggingface_hub` library)
  - SDK/Client: `huggingface_hub` >=0.34.0
  - Auth: `.huggingface/token` (implicit from HF_TOKEN env var if needed)
  - Purpose: Download GGUF model files from community repositories
  - Environment: `HF_HUB_ENABLE_HF_TRANSFER=1` enables fast HTTP transfer via `hf-transfer` binary

**OpenAI Compatibility Layer:**
- arcllm-proxy exposes OpenAI-compatible `/v1/chat/completions` endpoint
  - SDK/Client: Any OpenAI client library (tested with `openai` ^2.14.0)
  - Purpose: Act as drop-in replacement for OpenAI API
  - Base URL: `http://localhost:11435` (configurable via `ARCLLM_PORT` env var)
  - Backend URL: `http://127.0.0.1:18400` (internal llama-server, configurable via `ARCLLM_BACKEND_PORT`)

**Ollama API Compatibility:**
- arcllm-proxy mimics Ollama API structure:
  - `/v1/models` - List available models
  - `/v1/chat/completions` - Chat endpoint (streaming and non-streaming)
  - `/v1/completions` - Raw completion endpoint
  - Lazy-loading: Models loaded on first request, unloaded after idle timeout (`ARCLLM_IDLE_TIMEOUT`)

## Data Storage

**Model Files:**
- Local filesystem only: `models/` directory
  - Format: GGUF (GGML Unified Format)
  - Storage: `/home/ryan/llm-stack/models/`
  - Structure:
    - `Qwen/Qwen3-32B-GGUF/` - Qwen3 32B variants
    - `Qwen/Qwen3.5-*B-GGUF/` - Qwen3.5 variants (9B, 27B)
    - `Qwen/Qwen3-0.6B-GGUF/` - Draft model
    - `NVIDIA/Nemotron-3-Super-120B-A12B-GGUF/` - Large model (multi-file)
    - `GLM-4.7-Flash-heretic-GGUF/` - MoE model
    - `Qwen/Qwen3-abliterated-GGUF/` - Abliterated variants for specific use cases

**KV Cache Persistence (Slot Cache):**
- Path: `cache/slots/` directory
  - Purpose: Store serialized KV cache states for reuse across requests
  - Used when `--slot-save-path` flag is set on llama-server
  - Format: Binary serialized ggml tensors

**Benchmark Results:**
- Path: `/tmp/bench_results.json`
  - Format: JSON with per-test metrics (tokens, throughput, utilization)
  - Written by benchmark framework after each run
  - Contains: timing, utilization snapshots (GPU freq/power/temp, CPU%, RAM)

**Logs:**
- arcllm proxy: `/tmp/arcllm-server.log` (configurable via `ARCLLM_LOGFILE`)
- Benchmark logs: `/tmp/bench_logs/` directory (auto-created)

**File Storage:**
- None (all inference in-memory, KV cache to disk via slot mechanism)

**Caching:**
- In-memory KV cache: Managed by llama.cpp server per slot
- GPU VRAM cache: Distributed across 3 Arc A770s via tensor-split (1,1,1)
- Slot reuse: Up to 16 concurrent slots per model (configurable via `-np` flag)

## Authentication & Identity

**Auth Provider:**
- None (local, unauthenticated)
- OpenAI-compatible API accepts dummy `OPENAI_API_KEY=local` for client libraries

**Access Control:**
- Render group membership required for GPU access (enforced by OS, not app)
- No HTTP auth on proxy (localhost-only or restricted network deployment recommended)
- arcllm-proxy listens on `0.0.0.0:11435` by default (configurable via `ARCLLM_HOST`)

## Monitoring & Observability

**Error Tracking:**
- None (custom error handling in proxy)

**Logs:**
- Approach: Structured logging via Python's `logging` module
- Format: Text logs to `/tmp/arcllm-server.log`
- Levels: DEBUG, INFO, WARNING, ERROR (configurable)

**Metrics/Monitoring:**
- GPU Metrics:
  - Source: `/sys/class/drm/cardN/gt_cur_freq_mhz` (frequency)
  - Source: `/sys/class/drm/cardN/device/hwmon/*/power1_energy_input` (energy/power)
  - Source: `/sys/class/drm/cardN/device/hwmon/*/temp1_input` (temperature in mC)
  - Captured by `scripts/bench/monitor.py` during test execution
  - Per-GPU: frequency (MHz), power draw (watts), temperature (°C)

- CPU & Memory Metrics:
  - Source: `/proc/stat` (CPU utilization)
  - Source: `/proc/meminfo` (RAM usage)
  - Captured during benchmark runs

- Optional: `prometheus-client` support planned (not currently integrated)

## CI/CD & Deployment

**Hosting:**
- Local only: No remote hosting
- Deployment pattern: Manual `scripts/arcllm-server.sh start` on target machine

**CI Pipeline:**
- None configured (manual testing only via benchmark framework)

**Deployment Mechanism:**
- Wrapper script: `scripts/arcllm-server.sh` (start/stop/status/logs/models/load/unload)
- Environment bootstrap: `env.sglang-xpu.sh` (sources conda, sets SYCL env)
- No container/Docker (native Linux only)

## Environment Configuration

**Required env vars:**
- `ZE_AFFINITY_MASK` - GPU selection (default: "0,1,2" for 3 Arc A770s)
- `GGML_SYCL_DISABLE_GRAPH` - Command graph mode (default: "1" for safety)
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` - Queue batching (default: "0" for throughput)
- `GGML_SYCL_FUSED_MMQ` - Fused kernel optimization (default: "1" for production)
- `ARCLLM_PORT` - Proxy listen port (default: 11435)
- `ARCLLM_HOST` - Proxy listen host (default: "0.0.0.0")
- `ARCLLM_BACKEND_PORT` - Internal llama-server port (default: 18400)
- `ARCLLM_IDLE_TIMEOUT` - Model unload timeout in seconds (default: 0 = never)

**Secrets location:**
- `.huggingface/token` - HF Hub auth (optional, used if downloading private models)
- No other secrets managed by app (environment-variable based)

## Webhooks & Callbacks

**Incoming:**
- None (HTTP API is pull-only)

**Outgoing:**
- None (no external callbacks)

## Model Registry & Loading

**Registry Location:**
- `scripts/arcllm-proxy.py` - Hardcoded model definitions (lines 46-195)
  - Format: `_register(name, path, flags, aliases=[...])`
  - Parsed: Model name, file path, server flags, OpenAI-style aliases

**Default Models:**
- `qwen3-32b` (aliases: `32b`, `qwen3-32b-q4`, `default`) - Primary production model
  - Path: `models/Qwen/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf`
  - Config: `-np 16` (16 slots), `-c 32768` (context), `-fa on` (flash attention)
  - Throughput: 22.7 t/s at 16 concurrent requests

- `qwen3-32b-fast` (aliases: `qwen3-32b-think`, `qwen3-32b-reasoning`)
  - Config: `-np 1` (single slot), `--reasoning-budget 200` (thinking tokens)
  - Use case: Interactive with reasoning

- `qwen3-32b-q8` (aliases: `qwen3-32b-abliterated`)
  - Higher quality, lower throughput (3.3 t/s, bandwidth-limited)

- Other available: `nemotron-120b`, `qwen35-27b`, `qwen35-9b`, `glm47-flash`, `qwen3-30b-test`

---

*Integration audit: 2026-03-17*
