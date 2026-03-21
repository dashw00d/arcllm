# llama.cpp SYCL/oneAPI Build for Intel Arc A770 — Research Report

**Date:** 2026-03-15
**Scope:** Current state of SYCL backend, multi-GPU build configuration, known bugs, env vars, and fork alternatives for 3x Arc A770 (48 GB total VRAM).

---

## 1. Current State of SYCL Backend (2025–2026)

The SYCL backend is actively maintained but is significantly under-resourced compared to CUDA. A maintainer explicitly acknowledged in [issue #19918](https://github.com/ggml-org/llama.cpp/issues/19918) (Feb 2026): *"There is no commercial support for SYCL backend and less developer for it."* The same issue documents a ~6.8x performance gap between SYCL and Vulkan for MoE token generation on A770 (10 t/s vs 68 t/s), attributed to performance regression in 2024–2025 without a short-term fix.

### Recent upstream improvements (2025–2026):

| Date | PR/Commit | Change |
|------|-----------|--------|
| Feb 2025 | [PR #12035](https://github.com/ggml-org/llama.cpp/pull/12035) | Q4_0 MUL_MAT block reordering: +30% on A770 (42→55 t/s on llama-2-7B) |
| Feb 2026 | [PR #19889](https://github.com/ggml-org/llama.cpp/pull/19889) | Fix binbcast assertion `s10 == 1` — Qwen3-Coder regression, Mar 2026 |
| Feb 2026 | [PR #19920](https://github.com/ggml-org/llama.cpp/pull/19920) | Remove hardcoded work-group size 768 — fixes iGPU/dGPU compatibility |
| Mar 2026 | [PR #20190](https://github.com/ggml-org/llama.cpp/pull/20190) | Flash Attention for SYCL: PP +19–77%, TG −2–21%, memory −38–463 MB |
| Mar 2026 | [PR #20283](https://github.com/ggml-org/llama.cpp/pull/20283) | Fix ACC, L2_NORM, UPSCALE, fused_glu, unary op bugs |
| Mar 2026 | [PR #20293](https://github.com/ggml-org/llama.cpp/pull/20293) | Fix ROPE op, add ROPE_BACK |
| Mar 2026 | [PR #20455](https://github.com/ggml-org/llama.cpp/pull/20455) | Add GATED_DELTA_NET op — fixes Qwen3.5-9B (PP 90→339 t/s on A770) |
| Mar 2026 | [PR #20583](https://github.com/ggml-org/llama.cpp/pull/20583) | Fix untransposed GDA recurrent state (Qwen3.5 hybrid models) |

The official SYCL.md ([docs/backend/SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)) still lists **`Split-mode:[row] is not supported`** as a known issue — this reflects the upstream branch. The local branch (`/home/ryan/llm-stack/llama.cpp`) contains 4 additional commits (Mar 2026) that implement row-split from scratch for Arc A770:

- `daf5a6f`: Fix 5 independent bugs enabling SYCL row-split
- `8b6d3cb`: SSM pointer resolution + 256 KB replication threshold
- `7423303`: Full tensor replication even for partially-split tensors
- `e0ebf6b`: Clear recurrent state after reservation; disable MMVQ/MMQ for all multi-GPU configs

These commits are not in upstream `ggml-org/llama.cpp` as of 2026-03-15.

---

## 2. Correct CMake Configuration for Arc A770

### Official recommended (Linux, from SYCL.md):

```sh
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release -j
```

The docs specify `icx` for C and `icpx` for C++ on Linux — both are oneAPI Release compilers. The open-source LLVM SYCL compiler (`clang++`) is accepted as fallback but generates a warning.

### Local build vs. docs differences:

The local build script (`/home/ryan/llm-stack/scripts/build_llama_sycl.sh`) uses `gcc` for `CMAKE_C_COMPILER` and `icpx` for `CMAKE_CXX_COMPILER`. The official docs say `icx` for C — this discrepancy is benign for SYCL because only the CXX compiler matters for SYCL translation units, but switching C to `icx` is the recommended canonical form.

### Key CMake options and what they do:

| Option | Default | Notes |
|--------|---------|-------|
| `-DGGML_SYCL=ON` | OFF | Mandatory |
| `-DGGML_SYCL_F16=ON/OFF` | OFF | FP16 path; test both — impact is model-dependent |
| `-DGGML_SYCL_GRAPH=ON` | OFF | Enables SYCL command graph extension; **disabled at runtime for multi-GPU** |
| `-DGGML_SYCL_DNN=ON/OFF` | ON | oneDNN GEMM; disabled with `OFF` to force oneMKL |
| `-DGGML_SYCL_DEVICE_ARCH=<arch>` | unset | AOT target: reduces JIT startup, see below |

The local build explicitly sets `-DGGML_SYCL_GRAPH=ON` and `-DGGML_SYCL_DNN=OFF`. This is a valid configuration: SYCL graphs are built in at compile time but automatically disabled at runtime when `device_count > 1` (hardcoded in `check_graph_compatibility()` in `ggml-sycl.cpp:11477`).

### AOT compilation with GGML_SYCL_DEVICE_ARCH:

The Arc A770 uses the DG2/Alchemist architecture (also referred to as `xe_hpg` or `acm-g10`). Setting `GGML_SYCL_DEVICE_ARCH` compiles device kernels ahead-of-time, eliminating JIT compilation on first run:

```sh
-DGGML_SYCL_DEVICE_ARCH=acm-g10   # for Arc A770
```

The arch string is passed as `--offload-arch=${GGML_SYCL_DEVICE_ARCH}` to the backend via `-Xsycl-target-backend`. See [Intel LLVM OffloadDesign.md](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/OffloadDesign.md) for the full list of valid strings. The local build leaves this unset (blank string = JIT), which is why first-run startup is slow.

### oneAPI version compatibility:

From `docs/backend/SYCL.md`, verified releases: **2025.2.1** and **2025.1** (both Linux). The 2024.11 changelog notes syclcompat is required from **oneAPI 2025.0+**. The current local build installs MKL from conda (`mkl-devel-dpcpp==2025.2.0`), matching the verified matrix. A NixOS-specific macro conflict with oneAPI 2025.2 was filed as [issue #14440](https://github.com/ggml-org/llama.cpp/issues/14440) and closed as fixed — not relevant for Ubuntu.

---

## 3. Multi-GPU SYCL: Known Issues and Status

### Architecture constraint on SYCL graphs:

The `check_graph_compatibility()` function in `ggml-sycl.cpp` unconditionally returns `false` and logs *"disabling SYCL graphs due to multiple devices"* when `ggml_sycl_info().device_count > 1`. This is a fundamental SYCL spec limitation: a `sycl_ex::command_graph` object can only be created for a single device. This means **SYCL graphs are inherently incompatible with multi-GPU** regardless of the `GGML_SYCL_DISABLE_GRAPH` env var value.

The env var behavior:
- `GGML_SYCL_DISABLE_GRAPH=1` (default as of current codebase, line 268): graph never attempted
- `GGML_SYCL_DISABLE_GRAPH=0` with multi-GPU: `check_graph_compatibility()` disables it anyway

Net result: on 3-GPU setup, graph mode is always off regardless of this variable. The local build confirmed this: `GGML_SYCL_DISABLE_GRAPH=0` gave +22% throughput for single-GPU (3.02→3.70 t/s for Qwen3-32B Q8_0), but multi-GPU is unaffected.

### Historical multi-GPU bug (2024):

[PR #8554](https://github.com/ggml-org/llama.cpp/pull/8554) (Jul 2024) fixed a crash with multi-GPU by correcting platform enumeration in the SYCL queue management. Before this, running inference with 2+ GPUs caused crashes at initialization.

### Reported performance issue (2025):

A user comment in [issue #5282](https://github.com/ggml-org/llama.cpp/issues/5282) (Mar 2025) reported that *"the performance of 2+ devices is always below that of 1 device in my testing"* for layer-split. The maintainer requested logs; no resolution was documented. The local benchmarks independently confirmed this for dense transformer models: layer-split is **sequential** (one GPU active per layer), not parallel — throughput scales with per-GPU bandwidth relative to model size, not GPU count. Multi-GPU only helps when the model does not fit on a single GPU.

### P2P memory transfer:

Intel Arc A770 (PCIe, no NVLink equivalent) does not support peer-to-peer direct device memory copies via `dpct::async_dpct_memcpy`. The local row-split fix (`daf5a6f`) replaced this with per-column `dev2dev_memcpy` that goes through host. This is necessary but adds latency for row-split cross-device merging.

---

## 4. `--split-mode row` vs `layer` vs `none` — SYCL Behavior

### Official status:

The upstream `SYCL.md` states (as of current HEAD):
> "**Split-mode:[row] is not supported.**"
> "Support multiple cards: `--split-mode`: [none|layer]; not support [row], it's on developing." — 2024.3 news section

### Local branch status:

Row-split was implemented locally across 4 commits (Mar 14 2026). It is functional for Qwen3 dense models but has caveats:

| Split mode | Status | Notes |
|------------|--------|-------|
| `--split-mode none` | Fully supported | Single GPU, no distribution |
| `--split-mode layer` | Fully supported upstream | Sequential per-layer; good for models not fitting 1 GPU |
| `--split-mode row` | Local patches only | Works for Qwen3-32B; not upstreamed; DMMV fallback enforced |

### SYCL-specific differences from CUDA for row-split:

1. **No MMVQ/MMQ with row-split**: The Q8_1 `ds` (half2 scale store) is not visible to subsequent kernels on Arc A770 due to an L1 cache coherency bug. The local fix disables MMVQ and MMQ entirely when `device_count > 1`, forcing DMMV (dequantize + matrix-vector multiply). This is ~8.8x slower for row-split dense models (0.77 t/s vs 6.96 t/s for Qwen3-32B Q4_K_M).

2. **No async cross-device memcpy**: Arc A770 requires synchronous host-staged device-to-device copies for row merge. CUDA uses NVLink or direct PCIe P2P — neither available on Arc.

3. **Tensor replication**: Norm weights and small tensors (<256 KB) are replicated to all devices on upload. CUDA's row-split relies on same-device access patterns that don't apply to SYCL without P2P.

4. **`--tensor-split 1,1,1`**: This flag specifies relative allocation ratios across GPUs. With 3 equal A770s, `1,1,1` divides rows evenly. SYCL behavior matches CUDA semantics for this flag — ratios are normalized internally. No SYCL-specific bugs reported for tensor-split values.

### `--split-mode row` use case:

Row-split is only viable for MoE models where expert matrices are large enough that the DMMV fallback overhead is amortized by memory savings. The local setup uses `--split-mode row -cmoe` for MoE models. For dense transformers, layer-split is strictly better.

---

## 5. ik_llama.cpp Fork — SYCL Support?

**ik_llama.cpp does not support SYCL.** The project's README explicitly states:

> "The only fully functional and performant compute backends are CPU (AVX2 or better, ARM_NEON or better) and CUDA."

It further notes: "ROCm, Vulkan, Metal, etc. They will not get resolved unless you roll up your sleeves and help." No mention of SYCL, oneAPI, or Intel Arc anywhere in the codebase.

The fork's primary value is:
- Advanced quantization kernels (IQK variants, Trellis quants)
- DeepSeek MLA/FlashMLA optimizations
- CPU inference speed (AVX2, ARM NEON)
- Hybrid GPU/CPU with CUDA only

**For Intel Arc A770: ik_llama.cpp provides no benefit.** Upstream llama.cpp with the SYCL backend is the only viable path. The Vulkan backend (upstream) is available as a fallback and was shown to outperform SYCL on MoE by ~6.8x in token generation ([issue #19918](https://github.com/ggml-org/llama.cpp/issues/19918)), though Flash Attention and multi-GPU are less mature in Vulkan.

---

## 6. Intel-Specific Environment Variables — Recommended Values

### Variables in use (local `env.sglang-xpu.sh`):

| Variable | Current Value | Effect |
|----------|--------------|--------|
| `ZE_AFFINITY_MASK` | `0,1,2` | Selects Level Zero devices 0, 1, 2 (all 3 A770s) |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` | `1` | Allows GPU allocations >4 GB (required for 16 GB VRAM) |
| `ZES_ENABLE_SYSMAN` | `1` | Enables `sycl::aspect::ext_intel_free_memory` for VRAM reporting; needed for `--split-mode layer` memory balancing |

### SYCL runtime variables (from `ggml-sycl.cpp` source):

| Variable | Default | Recommended (3x A770) | Notes |
|----------|---------|----------------------|-------|
| `GGML_SYCL_DISABLE_GRAPH` | `1` | `1` | Irrelevant for multi-GPU (always disabled by `check_graph_compatibility()`); set `0` only for single-GPU to gain +22% |
| `GGML_SYCL_ENABLE_FLASH_ATTN` | `1` | `1` (leave default) | Enabled by default after [PR #20190](https://github.com/ggml-org/llama.cpp/pull/20190); improves PP, slightly hurts TG; disable with `0` if crashes occur |
| `GGML_SYCL_DISABLE_OPT` | `0` | `0` | Disabling removes the Q4_0 block-reorder optimization (+30% on A770); leave enabled |
| `GGML_SYCL_DISABLE_DNN` | `0` | `0` (if DNN built) | oneDNN vs oneMKL for GEMM; test both — local build uses `OFF` at CMake level |
| `GGML_SYCL_DISABLE_DMMV` | `0` | `0` | Do not disable; DMMV is the workaround for row-split Q8_1 bug |
| `GGML_SYCL_DISABLE_MMVQ` | `0` | `0` (layer-split); auto-disabled for row-split | Local code disables MMVQ for all multi-GPU configs |
| `GGML_SYCL_DISABLE_BATCHED_MM` | `1` | `1` (default) | Batched matmul disabled by default; no reason to change |
| `GGML_SYCL_DEBUG` | `0` | `0` | Verbose kernel logging; performance impact |
| `GGML_SYCL_FORCE_SYNC_SPLIT` | `0` | `0` | Forces synchronous cross-device syncs in row-split; only useful for debugging |

### Level Zero / oneAPI runtime variables:

| Variable | Status | Notes |
|----------|--------|-------|
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` | **Not recommended** | Local benchmarks showed `ICL=1` gave −7% throughput (6.67→6.38 t/s for Qwen3-32B Q4_K_M). Default (0) is optimal. |
| `ONEAPI_DEVICE_SELECTOR` | Alternative to `ZE_AFFINITY_MASK` | `level_zero:0;level_zero:1;level_zero:2` selects devices; `ZE_AFFINITY_MASK=0,1,2` is equivalent |
| `UR_ADAPTERS_SEARCH_PATH` | Set to conda lib | Ensures Unified Runtime finds the correct Level Zero adapter |

### `ZE_AFFINITY_MASK` notes:

This controls which Level Zero sub-devices/tiles are visible. For PCIe multi-card (not tile-based), `0,1,2` maps to card indices directly. Do not use `ONEAPI_DEVICE_SELECTOR` and `ZE_AFFINITY_MASK` simultaneously — they interact and can produce unexpected device lists.

---

## 7. Recent SYCL Bug Fixes (2025–2026 Upstream)

Key upstream bug fixes relevant to the 3x A770 setup:

### Feb 2026 — Binbcast assertion regression ([PR #19889](https://github.com/ggml-org/llama.cpp/pull/19889)):
A prior Qwen3-Next graph optimization (commit `1725e31`) changed operand shapes in `binbcast`, triggering `GGML_ASSERT(s10 == 1)` at line 200 of `binbcast.cpp`. Fix: removed strict shape validation, added permuted tensor layout support. Affected builds b8053–b8121. Reported in [issue #19779](https://github.com/ggml-org/llama.cpp/issues/19779).

### Feb 2026 — Dynamic work-group size ([PR #19920](https://github.com/ggml-org/llama.cpp/pull/19920)):
Hardcoded work-group size of 768 caused crashes on older iGPUs (max 512). Replaced with runtime query of `max_work_item_sizes`. Relevant for A770 only if running mixed iGPU+dGPU setups.

### Mar 2026 — GATED_DELTA_NET op ([PR #20455](https://github.com/ggml-org/llama.cpp/pull/20455)):
Qwen3.5-9B (and Qwen3.5-27B) use this fused op for hybrid recurrent layers. Without this PR, the op fell back to CPU causing "Qwen3.5 produces gibberish" ([issue #20423](https://github.com/ggml-org/llama.cpp/issues/20423)). With it, A770 PP improved from 90 to 339 t/s for Qwen3.5-9B.

### Mar 2026 — GDA recurrent state fix ([PR #20583](https://github.com/ggml-org/llama.cpp/pull/20583)):
Untransposed GDA (Gated Delta Accumulation) recurrent state — a follow-on fix for Qwen3.5 model correctness.

### Dec 2025 — Async kernel synchronization ([issue #15580](https://github.com/ggml-org/llama.cpp/issues/15580)):
`argsort_f32_i32_sycl` lacked `stream->wait()` at the end, causing race conditions with async kernel execution on Arc (confirmed on A770 16 GB). Fixed by adding explicit synchronization. Affects MoE models (OLMoE, Qwen3 variants).

### Jan 2026 — OLMo 3 MMVQ assertion ([issue #18240](https://github.com/ggml-org/llama.cpp/issues/18240)):
`GGML_ASSERT(block_num_y % num_subgroups == 0)` in `mmvq.cpp:811` crashed A770 with OLMo 3. Fixed by adding padding/rounding for non-divisible dimensions.

---

## 8. `--split-mode` Behavioral Matrix for SYCL

| Mode | Parallelism | SYCL Specific Behavior | Recommended Use |
|------|-------------|----------------------|----------------|
| `none` | None | Single device; SYCL graphs work; fastest per-token if model fits | Single-card inference or debugging |
| `layer` | Sequential | One GPU processes each layer in sequence; no P2P needed; SYCL graphs disabled for multi-device | Dense models too large for 1 GPU (best current option for 3x A770) |
| `row` | Parallel (theoretical) | DMMV forced (Q8_1 ds bug); P2P via host staging; all-reduce after each layer; MMVQ/MMQ disabled; local patches required | MoE models with `-cmoe`; not viable for dense models |

### Why layer-split is not parallel on SYCL:

In the layer-split implementation, each SYCL device processes its assigned layers via `ggml_backend_sycl_graph_compute()` called sequentially for each backend. The ggml scheduler synchronizes between backends at each tensor boundary. Intel Arc lacks hardware context switching that would allow true pipeline overlap. The result is that 3x A770 with layer-split gives roughly the same t/s as a single A770 for the same number of layers per device — confirmed at 6.96 t/s for Qwen3-32B Q4_K_M (3 GPUs) vs. ~6.7 t/s for Q4_K_M on 2 GPUs (marginal difference due to context size).

---

## 9. Additional Findings Not in llama.cpp README

### Flash Attention on A770 — TG regression:

[PR #20190](https://github.com/ggml-org/llama.cpp/pull/20190) was ported from CUDA via DPCT migration. Token generation regresses 2–21% with FA enabled on A770. Flash Attention primarily helps prompt processing (large context). For generation-heavy workloads (chatbots), disabling with `GGML_SYCL_ENABLE_FLASH_ATTN=0` may be faster. The runtime default is `1` (enabled) after this PR merged.

### Vulkan as fallback for MoE:

The A770 Vulkan backend shows ~6.8x better TG for MoE models compared to SYCL ([issue #19918](https://github.com/ggml-org/llama.cpp/issues/19918)). Vulkan added GATED_DELTA_NET in [PR #20334](https://github.com/ggml-org/llama.cpp/pull/20334) (Mar 2026). For Qwen3.5 or other hybrid SSM/attention MoE models on A770, the Vulkan backend is worth benchmarking. Build with `-DGGML_VULKAN=ON`. Limitation: Vulkan multi-GPU for large models is less tested.

### ReBar requirement:

The SYCL discussion thread [#5277](https://github.com/ggml-org/llama.cpp/discussions/5277) documented that Resizable BAR (ReBar/SAM) is required for stable SYCL operation on Arc. Without it, SIGBUS errors occur. This should be enabled in BIOS for the PCIe A770 cards.

### Nvidia/AMD SYCL support dropped:

As of Feb 2026 (from `SYCL.md`): *"Remove support for Nvidia & AMD GPU, because the oneAPI plugin for Nvidia & AMD GPU is unavailable: download/installation channels are out of work."* SYCL is now Intel-only.

### syclcompat and oneAPI 2025.0+ requirement:

The 2024.11 news in SYCL.md: *"Use syclcompat to improve the performance on some platforms. This requires to use oneAPI 2025.0 or newer."* Running with oneAPI <2025.0 may give degraded performance or compilation failures for syclcompat-dependent kernels.

### Q6_K warmup segfault:

Locally confirmed: Q6_K crashes during warmup on 3x A770 layer-split. No upstream issue specifically for Q6_K+SYCL+A770 was found in the tracker. Most likely a kernel-specific bug in the Q6_K dequantize path under row-split or multi-device init. No known upstream fix.

### Q2_K SYCL graph crash on single GPU:

Locally confirmed: Q2_K crashes in graph recording mode. Root cause likely a `wait_and_throw()` inside a kernel that is incompatible with graph recording. Setting `GGML_SYCL_DISABLE_GRAPH=1` may resolve this for single-GPU; not tested.

---

## Recommended Changes to Deployment Plan

### 1. Set `GGML_SYCL_ENABLE_FLASH_ATTN=0` for generation-heavy workloads

Flash Attention was enabled by default after PR #20190 but regresses TG by 2–21% on A770. For the production chatbot use case (mostly TG), disable it:
```sh
export GGML_SYCL_ENABLE_FLASH_ATTN=0
```
For large-context prompt processing (>32k tokens, single user), re-enable per-request or test the tradeoff.

### 2. Add AOT compilation for A770

Eliminate the JIT startup lag by specifying the device architecture at build time. Add to `build_llama_sycl.sh`:
```sh
-DGGML_SYCL_DEVICE_ARCH=acm-g10
```
This compiles device kernels ahead-of-time for the DG2/Alchemist architecture, removing the first-run warm-up delay (typically 10–30 seconds of JIT compilation at startup).

### 3. Switch `CMAKE_C_COMPILER` from `gcc` to `icx`

The current `build_llama_sycl.sh` uses `gcc` for C. The official recommendation is `icx`. Switch to match the verified configuration:
```sh
-DCMAKE_C_COMPILER=icx \
-DCMAKE_CXX_COMPILER=icpx \
```
This avoids potential ABI or header compatibility issues between gcc-compiled C units and icpx-compiled SYCL units, particularly as oneAPI versions advance.

### 4. Keep `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` (do not set to 1)

Local benchmark showed ICL=1 gives −7% throughput. Do not set this variable; the default (0) is optimal for batch=1 inference on Arc A770.

### 5. Set `GGML_SYCL_DISABLE_GRAPH=0` only for single-GPU launches

For single-GPU use cases (e.g., small models on one A770), enable graph mode for +22%:
```sh
GGML_SYCL_DISABLE_GRAPH=0 ZE_AFFINITY_MASK=0 llama-server ...
```
For 3-GPU launches, the value is irrelevant — `check_graph_compatibility()` disables graphs automatically when `device_count > 1`. The current default of `1` is safe for all cases.

### 6. Benchmark Vulkan for Qwen3.5 / MoE models

The upstream SYCL MoE performance gap vs Vulkan (~6.8x worse TG) is an open issue with no near-term fix ([issue #19918](https://github.com/ggml-org/llama.cpp/issues/19918)). If a MoE model is needed at production quality, build a parallel Vulkan binary and benchmark:
```sh
cmake -B build-vulkan -DGGML_VULKAN=ON
```
Note: Vulkan multi-GPU for models requiring 48 GB is less tested, and the row-split local patches do not apply to Vulkan.

### 7. Avoid Q6_K and Q2_K quantizations on SYCL

Q6_K produces a warmup segfault on 3x A770 layer-split (confirmed locally, no upstream fix found). Q2_K triggers SYCL graph crashes. Stick to Q8_0 and Q4_K_M for production. If Q6_K is specifically needed, investigate `GGML_SYCL_DISABLE_OPT=1` as a mitigation or file an upstream bug report.

### 8. Monitor PR for row-split upstreaming

The 4 local row-split commits (Mar 14 2026) are not in upstream. If the implementation matures to support Nemotron Nano row-split and passes broader testing, submitting upstream as `[SYCL] add row-split multi-GPU support for Intel Arc` would benefit the community and enable tracking of any regressions as upstream evolves. The current state (Qwen3 working, Nemotron partial) may be upstreamable as an experimental flag.

### 9. Enable ReBar if not already done

Verify Resizable BAR is enabled in BIOS for all 3 Arc A770 PCIe cards. Absence causes SIGBUS errors per [discussion #5277](https://github.com/ggml-org/llama.cpp/discussions/5277). Check with:
```sh
sudo lspci -vv | grep -A5 "Arc A770" | grep "PrefetchableMem"
# Should show a large (>4GB) prefetchable BAR
```

---

*Sources: [llama.cpp SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md), [issue #19918](https://github.com/ggml-org/llama.cpp/issues/19918), [issue #20423](https://github.com/ggml-org/llama.cpp/issues/20423), [issue #19779](https://github.com/ggml-org/llama.cpp/issues/19779), [issue #18240](https://github.com/ggml-org/llama.cpp/issues/18240), [issue #15580](https://github.com/ggml-org/llama.cpp/issues/15580), [issue #5282](https://github.com/ggml-org/llama.cpp/issues/5282), [issue #9505](https://github.com/ggml-org/llama.cpp/issues/9505), [PR #12035](https://github.com/ggml-org/llama.cpp/pull/12035), [PR #19889](https://github.com/ggml-org/llama.cpp/pull/19889), [PR #19920](https://github.com/ggml-org/llama.cpp/pull/19920), [PR #20190](https://github.com/ggml-org/llama.cpp/pull/20190), [PR #20283](https://github.com/ggml-org/llama.cpp/pull/20283), [PR #20293](https://github.com/ggml-org/llama.cpp/pull/20293), [PR #20455](https://github.com/ggml-org/llama.cpp/pull/20455), [PR #20583](https://github.com/ggml-org/llama.cpp/pull/20583), [PR #8554](https://github.com/ggml-org/llama.cpp/pull/8554), [discussion #5277](https://github.com/ggml-org/llama.cpp/discussions/5277), local commits daf5a6f–e0ebf6b, local memory files project_qwen35_benchmarks.md and project_qwen3_32b_optimization.md, ik_llama.cpp README, Intel SYCL OffloadDesign.md*
