# Active Experiments

Multi-GPU inference on 3x Intel Arc A770 — three parallel approaches.

## Worktree Layout

```
/home/ryan/llm-stack/
├── llama.cpp/              master + SYCL row-split/event mods (original work)
├── llama.cpp-expert/       expert-parallelism branch (profiling done)
├── llama.cpp-tp/           PR #19378 tensor-parallelism (WORKING — 1.99 t/s)
├── scripts/bench/          shared benchmark framework
├── models/                 shared GGUF models
└── research/sycl-reference/ shared SYCL/L0 reference docs
```

## 1. Tensor Parallelism — WORKING (2026-03-18)

**Worktree:** `llama.cpp-tp/` | **Branch:** `tensor-parallelism-upstream`
**Docs:** `llama.cpp-tp/EXPERIMENT.md`

PR #19378's meta backend with SYCL `cpy_tensor_async` implementation. First tensor parallelism on Intel Arc SYCL.

**Result:** Qwen3-32B Q4_K_M at **1.99 t/s** (3.3x faster than legacy row-split 0.6 t/s).

**Remaining optimizations:**
- Pre-allocate pinned staging buffers (avoid per-copy malloc)
- Overlap GPU→host and host→GPU copies
- Implement `allreduce_tensor_async` (direct sum vs pairwise exchange)
- BF16 AllReduce compression (2x less PCIe traffic)
- Profile actual bottleneck (copies vs compute vs overhead)

## 2. Expert Parallelism — Profiled, Not Yet Implemented

**Worktree:** `llama.cpp-expert/` | **Branch:** `expert-parallelism`
**Docs:** `llama.cpp-expert/EXPERIMENT.md`

Split MoE models by expert assignment across GPUs. Profiling shows GLM-4.7-Flash has near-uniform expert routing (not 80/20), but 55/64 GPU placement still gives 62.5% fully-local tokens.

**Remaining work:**
- Expert-aware weight loading
- Routing dispatch hook in `ggml_mul_mat_id`
- Cross-GPU expert call (memcpy hidden state → compute → return)
- Test with models that have stronger expert locality (DeepSeek-V2/V3)

## 3. Event-Based Row Split — Blocked

**Worktree:** `llama.cpp/` (master) | **Branch:** `master`
**Docs:** `.planning/PROJECT.md`

NVIDIA-style event streaming. 52x fewer host stalls but blocked by SYCL runtime event pool leaks (#14797). No production system has shipped cross-device SYCL events.

**Status:** Blocked — SYCL runtime bug, not fixable at application level.

## Shared Resources

- **Bench framework:** `scripts/bench/` — works with any build dir via config
- **Models:** `models/` — GGUF files shared across all worktrees
- **SYCL reference:** `research/sycl-reference/` — L0 events, env vars, known failures
- **Environment:** `env.sglang-xpu.sh` — SYCL/conda setup (shared)
