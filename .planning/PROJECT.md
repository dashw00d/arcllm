# Event-Based Multi-GPU Sync for SYCL Row-Split

## What This Is

NVIDIA-style event streaming for llama.cpp's SYCL backend on 3x Intel Arc A770. Replaces host-blocking `queue.wait()` calls in the multi-GPU row-split inference path with SYCL event-based dependencies (`handler::depends_on()`), eliminating ~1000+ host stalls per token. Target model is GLM-4.7-Flash Q4_K_M (MoE with MLA attention).

## Core Value

GPU pipeline stays full: GPU 1 begins computing the moment GPU 0's output is available, without ever draining either device. Performance, not correctness abstractions.

## Requirements

### Validated

- Multi-GPU row-split inference works (basic, with host stalls) — existing
- Single-GPU inference at 23 t/s (GLM-4.7-Flash Q4_K_M) — existing
- Fused MMQ kernels (+25% throughput on Qwen3-32B) — existing
- Benchmark framework with GPU monitoring — existing
- Lazy-loading proxy with OpenAI-compatible API — existing
- SYCL environment management (env.sglang-xpu.sh) — existing

### Active

- [ ] OOO queue pool infrastructure per device (dedicated to row-split work)
- [ ] Host-pinned staging buffers via `sycl::malloc_host` for GPU-GPU copies
- [ ] Event-based GPU-GPU copy replacing `dev2dev_memcpy_staged()`
- [ ] Rewritten row-split matmul loop with event chains (not host waits)
- [ ] Event pool exhaustion prevention (periodic OOO queue flush)
- [ ] GLM-4.7-Flash at 10+ t/s on 3x A770 (row-split)
- [ ] 500+ token generation at 2048 context without desync
- [ ] Perplexity matches single-GPU reference within 0.01
- [ ] Host stalls reduced from ~1000+/token to ~18/token

### Out of Scope

- Generalizing to other models — GLM-4.7-Flash only for now, generalize later
- Upstream PR polish — prioritize t/s; clean up for upstream later
- Layer-split mode — this is row-split only
- CUDA backend changes — SYCL only
- Batched command list support — immediate command lists required, batched deadlocks

## Context

### The Problem
llama.cpp's SYCL backend does `queue.wait()` (full device drain) between every GPU during multi-GPU row-split inference. On 3x Arc A770 with GLM-4.7-Flash Q4_K_M: 0.6 t/s when hardware should deliver 20+ t/s. The model works at 23 t/s single-GPU but desyncs after ~200 tokens in multi-GPU mode.

### The NVIDIA Reference Pattern
- Each GPU has its own CUDA stream
- `cudaEventRecord(event, stream)` marks kernel completion
- `cudaStreamWaitEvent(other_stream, event)` creates cross-GPU dependency without host stall
- Activations flow GPU->host->GPU via pinned memory, overlapped with compute

### SYCL Direct Equivalents
- `sycl::queue` (out-of-order) = CUDA stream
- `sycl::event` from `queue.submit()` = CUDA event
- `handler::depends_on(event)` = `cudaStreamWaitEvent`
- `queue.ext_oneapi_submit_barrier()` = barrier event capturing all pending work
- `sycl::malloc_host()` = CUDA pinned memory

### Known Failure Modes
1. **Event pool exhaustion** after ~100 matmuls — L0 finite event pool. Fix: periodic OOO queue flush.
2. **Cross-queue barrier events (in-order -> OOO)** — crashes after many iterations. Fix: host wait for pre-sync, event-based after.
3. **OOO queues + legacy dev2dev_memcpy_staged** — internal `src_stream.wait()` deadlocks OOO. Fix: new copy path with `depends_on` chaining.
4. **Regression to queue.wait()** — defeats the entire purpose. Fix: discipline.

### Key Files
- `ggml/src/ggml-sycl/ggml-sycl.cpp` — Main dispatch, multi-GPU split logic
- `ggml/src/ggml-sycl/common.hpp` — Queue management, device context
- `ggml/src/ggml-sycl/mmvq.cpp` — Quantized mat-vec kernels
- `ggml/src/ggml-sycl/dequantize.hpp` — Dequant functions

## Constraints

- **Hardware**: 3x Intel Arc A770 16GB, i9-7900X, PCIe 3.0 x16 per slot
- **Environment**: `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` required — batched mode deadlocks cross-queue `depends_on()` (confirmed experimentally)
- **Gating**: New code behind `GGML_SYCL_ROW_EVENTS=1` env var — don't modify legacy path
- **Host waits**: ONE per matmul acceptable (pre-sync). Everything after must be event-based.
- **Performance floor**: Anything under 10 t/s means sync path is still broken

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Immediate command lists only | Batched mode deadlocks cross-queue depends_on | — Pending |
| GLM-4.7-Flash as sole target | MoE/MLA = small active params, sync overhead proportionally larger | — Pending |
| Option A for event pool (periodic flush) | One line fix, minor stalls, eliminates crash | — Pending |
| New event-based path, not patches | Clean 300-line path > 50 lines of patches on broken legacy | — Pending |
| Performance first, upstream later | Can clean up code for PR after hitting t/s target | — Pending |

---
*Last updated: 2026-03-17 after initialization*
