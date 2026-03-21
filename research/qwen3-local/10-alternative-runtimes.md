# Alternative Inference Runtimes for Intel Arc A770 (Beyond llama.cpp SYCL)

**Date:** 2026-03-15
**Context:** 3x Intel Arc A770 (16 GB VRAM each, 48 GB total), currently running llama.cpp SYCL with row-split tensor parallel. The primary pain point is the `quantize_row_q8_1` kernel L1 cache visibility issue in the SYCL backend: the `sycl::half2` d-s scale fields are not reliably visible across sub-groups during MMVQ, forcing a DMMV (de-quantize then matrix-vector multiply) fallback on row-split workloads. DMMV is significantly slower than MMVQ and caps effective multi-GPU scaling for `Q8_0` and higher quants.

---

## 1. The Bug Context

The SYCL MMVQ kernel (`ggml/src/ggml-sycl/mmvq.cpp`) stores Q8\_1 scale/offset as `sycl::half2* q8_1_ds_ptr`. Under row-split across multiple GPUs, partial rows land on different devices and the host-side re-quantize pass re-encodes them as Q8\_1 before the gemv. A GPU-resident visibility issue — likely a missing sub-group barrier around the `q8_1_ds_ptr` read — causes garbage scales on some Arc sub-slices, so the code falls through to the DMMV path.

- **Relevant PR:** [#12858 — sycl: reordered Q4\_0 MMVQ for Intel GPUs](https://github.com/ggml-org/llama.cpp/pull/12858) documents the same DMMV→MMVQ transition for Q4\_0 and shows that switching code paths yielded 0–21% PP gains on Arc 770, with mixed TG results; the Q8\_1 path was not addressed in that PR.
- **Open issue:** [#20338 — SYCL backend fails on Intel Arc A770 (Ubuntu 24.04)](https://github.com/ggml-org/llama.cpp/issues/20338) is open as of March 2026.
- **Open issue:** [#19543 — Missing kernels for Qwen3-VL on Arc A770](https://github.com/ggml-org/llama.cpp/issues/19543) shows ongoing kernel gaps.
- No public issue specifically names the `q8_1` half2 cache coherence bug, suggesting it may be observed empirically without an upstream report yet.

---

## 2. Candidate Runtimes

### 2.1 IPEX-LLM (intel-analytics/ipex-llm)

**Status: ARCHIVED January 28, 2026. Read-only. Intel has ceased support.**

Despite the archive, this was the most Arc-native alternative through late 2025 and is still the only project with documented Qwen3-235B-A22B MoE runs on Arc A770.

#### Architecture & Kernel Approach

IPEX-LLM wraps PyTorch with a custom `xe_linear` / `xe_addons` / `xe_batch` op library (registered via `xpu_ops.py`). Rather than calling llama.cpp's SYCL kernels, it routes quantized matmul through **oneDNN** primitives compiled for XPU, which internally use Intel's XMX (Xe Matrix Extensions) hardware when available. The quantization intermediate is **not** Q8\_1 in the llama.cpp sense — it uses symmetric INT4 weight packing with FP16/BF16 activations and calls `dequantize_rows` to a native float before the matmul. This **entirely avoids the `quantize_row_q8_1` / half2 SYCL kernel path** that causes the DMMV fallback.

- Sources: [`xpu_ops.py`](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/src/ipex_llm/transformers/xpu_ops.py), [`low_bit_linear.py`](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/src/ipex_llm/transformers/low_bit_linear.py), README quantization summary.

#### MoE Support (Qwen3-235B-A22B)

Yes — **FlashMoE** was the headline feature of the final active development period:

> "You can now run DeepSeek V3/R1 671B and Qwen3MoE 235B models with just 1 or 2 Intel Arc GPUs." — [IPEX-LLM README](https://github.com/intel-analytics/ipex-llm/blob/main/README.md)

- Qwen3MoE 235B INT4: requires **128 GB CPU memory**, 1–8 Arc A770/B580 GPUs, 500 GB disk.
- DeepSeek V3/R1 671B INT4: requires 380 GB CPU memory.
- On a single A770, context must be reduced (`-c 1024`) to avoid OOM.
- Source: [`flashmoe_quickstart.md`](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/flashmoe_quickstart.md)

The FlashMoE design offloads inactive expert weights to CPU DRAM and streams them to the GPU on demand — functionally equivalent to llama.cpp's `-cmoe` expert-offload flag. On a 3x A770 setup (48 GB GPU, assuming 128+ GB system RAM) Qwen3-235B-A22B should be runnable but TPS will be memory-bandwidth bound on the CPU-to-GPU transfer.

#### Multi-GPU

- Tensor Parallel via **DeepSpeed AutoTP** (tested on 2x Arc A770). Source: [`deepspeed_autotp_fastapi_quickstart.md`](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/deepspeed_autotp_fastapi_quickstart.md).
- Pipeline Parallel also available.
- Requires Intel oneAPI Base Toolkit 2024.0, OneCCL, MPI4py.

#### OpenAI API Server

Yes — served via **FastAPI + Uvicorn** (custom, not OpenAI-spec vLLM). IPEX-LLM also has a vLLM integration path (`serving_with_vllm.md`) that exposes an OpenAI-compatible endpoint using IPEX-LLM as the backend engine under vLLM.

#### GGUF Support

Yes — direct GGUF loading is supported for FlashMoE. Users download GGUF-format models from HuggingFace and point the CLI at them. The GGUF loading uses the project's own reader, not llama.cpp's backend.

#### Speculative Decoding

A `speculative.py` module exists in the transformers directory, suggesting draft-target speculative decoding is implemented, but no public benchmark data on Arc was found.

#### Quantization Formats

INT4 (sym\_int4, asym\_int4), INT5, INT6, FP6, FP8, FP8\_e4m3, INT8, FP16. AWQ and GPTQ loading supported. Source: README, vLLM quickstart.

#### Licensing

Apache 2.0.

#### Critical Caveat

The project was archived January 28, 2026. Intel explicitly states: *"Intel will not provide or guarantee development of or support for this project… patches are no longer accepted."* The final stable release is v2.2.0 (April 7, 2026, post-archive nightly build). Using this for production means accepting an unsupported, security-flagged codebase.

---

### 2.2 vLLM with Intel XPU Backend

**Status: Functional but buggy; paged-attention kernels incomplete for Arc discrete GPUs.**

#### Intel XPU Support

vLLM has an `xpu` backend maintained by Intel contributors. PRs show ongoing work but also persistent regressions:

- [Issue #28650](https://github.com/vllm-project/vllm/issues/28650) — "Unsupported gpu\_arch of paged\_attention\_vllm!!" on **Arc A770 LE** running `vllm serve Qwen/Qwen3-4B`. Open since November 2025, marked stale February 2026 with no fix assigned.
- [Issue #14295](https://github.com/vllm-project/vllm/issues/14295) — Same `paged_attention_v1` warning on Arc iGPU, closed as NOT_PLANNED. The NOT_PLANNED closure applies to iGPU but the dGPU issue (#28650) remains open.
- XPU-specific components exist: `xpu_communicator`, `xpu_fused_moe`, `xpu_ops`. MoE data-parallel execution is documented.

#### Arc A770 Compatibility

Discrete Arc A770 **can** launch serving with `dtype half` but triggers repeated paged-attention warnings and produces potentially incorrect outputs depending on vLLM version. The root cause is missing hardware-specific paged-attention dispatch for Arc's GPU architecture ID. This is distinct from the llama.cpp Q8\_1 bug — vLLM does not use GGUF or Q8\_1 at all by default; it operates in BF16/FP16/INT4 (AWQ/GPTQ) weight formats via PyTorch XPU.

#### GGUF Support

GGUF quantization layers exist in the vLLM codebase but GGUF inference via the XPU backend specifically has **not been validated** in any public issue or PR found.

#### MoE Support

`xpu_fused_moe` module is present. DeepSeek and Qwen3-MoE are listed target models in the [2025 H2 XPU roadmap (issue #8309)](https://github.com/sgl-project/sglang/issues/8309) — note that issue is in the SGLang repo, not vLLM, but vLLM has equivalent MoE infrastructure.

#### Multi-GPU

Tensor parallel via OneCCL. Requires setting OneCCL environment variables. vLLM's `--tensor-parallel-size` works with XPU. Source: IPEX-LLM vLLM quickstart.

#### CPU Expert Offload (`-cmoe` equivalent)

No equivalent in stock vLLM XPU. CPU offload for MoE experts is not a supported vLLM feature.

#### Speculative Decoding

Supported in vLLM generally; XPU-specific validation unknown.

#### Licensing

Apache 2.0.

#### Kernel Difference from llama.cpp

vLLM XPU uses PyTorch's `torch.xpu` device, which routes through Intel Extension for PyTorch (IPEX). Matmul uses oneDNN / Intel MKL-DNN primitives. The `quantize_row_q8_1` SYCL kernel from llama.cpp is **not used**. However, the paged-attention kernel path is incomplete, which is a different class of bug.

---

### 2.3 OpenVINO GenAI (openvinotoolkit/openvino.genai)

**Status: Production-quality for small-medium models on single Arc GPU; multi-GPU and 100B+ class models not supported.**

#### Arc A770 Support

Confirmed. The README contains a demonstration image captioned *"Text generation using LLaMa 3.2 model running on Intel ARC770 dGPU."* Source: [openvino.genai README](https://raw.githubusercontent.com/openvinotoolkit/openvino.genai/master/README.md).

Inference device is set by passing `"GPU"` to `LLMPipeline` — the same code works on CPU/GPU/NPU without changes. OpenVINO's GPU plugin uses **OpenCL** (not SYCL, not PyTorch XPU) as its runtime, built on Intel's NEO compute runtime.

#### Quantization Approach

Models must be exported to **OpenVINO IR format** using `optimum-cli` (which calls the Hugging Face `optimum-intel` exporter):

```
optimum-cli export openvino --model Qwen/Qwen3-... --weight-format int4
```

Weight-only quantization (INT4/INT8) is applied at export time via Neural Network Compression Framework (NNCF). Inference-time quantization (Q8\_1-style) does **not exist** — the GPU receives INT4 weights and computes in FP16. The `quantize_row_q8_1` kernel path is **architecturally absent**.

#### Qwen3 Support

The README lists "Qwen" as a supported model family alongside Llama, Phi, Mistral, etc. No Qwen3-specific documentation was found, but Qwen3 uses the standard transformer architecture and should export cleanly via `optimum-intel`. Source: [openvino.genai README](https://raw.githubusercontent.com/openvinotoolkit/openvino.genai/master/README.md).

#### Model Size Limits and Multi-GPU

**Critical limitation:** Multi-GPU is not currently supported for LLM inference. PR [#3496](https://github.com/openvinotoolkit/openvino.genai/pull/3496) ("Sum GPU memory across devices for multi-GPU KV cache validation") is open as of March 14, 2026 — multi-GPU KV management is still in development. OpenVINO does support a `MULTI:GPU.0,GPU.1` device string for throughput scaling but tensor-parallel LLM inference across Arc cards is **not available**.

This makes OpenVINO GenAI unsuitable for Qwen3-235B-A22B on 3x A770: the single A770 has 16 GB VRAM, insufficient for any meaningful context with a 235B model even at INT4.

#### 100B+ Model Reports

None found in public issues or documentation. The 16 GB single-GPU constraint makes it impractical without multi-GPU support.

#### OpenAI-Compatible API Server

Not present natively. OpenVINO GenAI is a C++/Python pipeline library, not a serving framework. Third-party wrappers (e.g., via llama.cpp's OpenVINO backend or custom FastAPI wrappers) exist but are not part of the project.

#### CPU Expert Offload

Not applicable — OpenVINO GenAI does not implement MoE expert offload.

#### Speculative Decoding

Available in OpenVINO GenAI via `SpeculativeDecodingPipeline`. Both draft and main models run on the same device.

#### Licensing

Apache 2.0.

---

### 2.4 Intel Extension for PyTorch (IPEX / `intel-extension-for-pytorch`)

**Status: RETIRED. Final release 2.8 (Q4 2025). Maintenance-only until March 2026.**

IPEX is the upstream library that IPEX-LLM is built on. Intel has discontinued active development:

> "We have discontinued active development of the Intel® Extension for PyTorch\* and ceased official quarterly releases following the 2.8 release." — [intel/intel-extension-for-pytorch README](https://github.com/intel/intel-extension-for-pytorch)

`ipex.llm` within IPEX supports Qwen3 and provides INT4/INT8 weight-only quantization for XPU inference, but as a raw library (not a serving framework). No OpenAI API server, no GGUF support, no MoE expert offload.

Intel's recommended migration is to use PyTorch's native XPU device support directly, with community frameworks (vLLM, SGLang) for serving.

---

### 2.5 SGLang with Intel XPU Backend

**Status: Active development targeting Arc B-Series (B580); Arc A770 (Alchemist) not explicitly validated.**

#### Intel XPU Support State

SGLang has a dedicated [`docs/platforms/xpu.md`](https://raw.githubusercontent.com/sgl-project/sglang/main/docs/platforms/xpu.md) and an active development roadmap ([issue #8309, closed February 2026](https://github.com/sgl-project/sglang/issues/8309) — "Intel GPU XPU Backend Optimization 2025 H2").

Merged capabilities as of March 2026:
- MoE integration merged February 5, 2026 ([PR #13561](https://github.com/sgl-project/sglang/pulls?q=intel+xpu))
- DeepSeek R1 inference on XPU (PR #18461)
- PyTorch 2.10+xpu ([PR #20254, merged March 11, 2026](https://github.com/sgl-project/sglang/pulls?q=intel+xpu))
- Qwen3 layernorm/MRoPE optimization (in-progress PR)
- Multi-GPU via `--tp 2` tensor parallel

#### Hardware Targeting

The XPU docs state SGLang is *"optimized for Intel® Arc™ Pro B-Series Graphics and Intel® Arc™ B-Series Graphics"* and verified on Arc B580. Arc A770 (Alchemist / Xe-HPG architecture, older than Battlemage/B-Series) is **not explicitly listed**.

The critical difference: Arc B580 uses the Battlemage (Xe2) architecture with improved XMX hardware. Arc A770 uses Alchemist (Xe-HPG / Xe-HPC mixed). PyTorch 2.10+xpu should support both, but optimized kernel dispatch may not fully cover Alchemist sub-architecture variants.

#### Installation

Source-only install required (`pyproject_xpu.toml`). Docker under active development. Requires:
- Python 3.12 (conda)
- PyTorch 2.10.0+xpu
- triton-xpu 3.6.0

#### MoE Support

Yes — `xpu_fused_moe` (equivalent to CUDA's fused MoE kernel) is merged. Expert CPU offload equivalent is not documented.

#### OpenAI API

Yes — SGLang exposes a full OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`).

#### GGUF Support

No — SGLang operates on HuggingFace model formats (BF16/FP16/INT4 via AWQ/GPTQ). No GGUF support.

#### Quantization Formats

INT4 AWQ/GPTQ via `torch.xpu`; FP8 on roadmap.

#### CPU Expert Offload

Not currently available for XPU.

#### Speculative Decoding

Yes, supported in SGLang generally. XPU validation status unknown.

#### Kernel Difference

SGLang on XPU routes through `torch.xpu` → IPEX (where still active) → oneDNN. The Q8\_1 SYCL kernel is not used. This is a categorically different code path from llama.cpp SYCL.

#### Licensing

Apache 2.0.

---

### 2.6 Ollama on Intel Arc A770

**Status: Not yet production-ready for Arc dGPU.**

Ollama has two competing Intel GPU PRs:
- [PR #10322](https://github.com/ollama/ollama/issues?q=intel+arc+a770+sycl) — SYCL backend for Intel GPU, open since April 2025.
- [PR #11160](https://github.com/ollama/ollama/issues?q=intel+arc+a770+sycl) — Intel GPU with OneAPI/SYCL, open since June 2025.
- Vulkan backend merged ([PR #11835](https://github.com/ollama/ollama/issues?q=intel+arc+a770+sycl), October 2025) but produces garbage output on Intel Arrow Lake GPUs with models >3B.
- [Issue #1590](https://github.com/ollama/ollama/issues?q=intel+arc+a770+sycl) — "Add support for Intel Arc GPUs" remains open since December 2023.

Ollama wraps llama.cpp, so any Ollama Intel SYCL path would inherit the same Q8\_1 half2 DMMV fallback bug.

---

### 2.7 LM Studio / Jan / Anything that wraps llama.cpp SYCL

Any UI or server that delegates to llama.cpp's SYCL backend inherits the same kernel stack and the same Q8\_1 DMMV fallback bug. These are not alternative runtimes — they are wrappers.

---

## 3. Runtime Comparison Table

| Feature | llama.cpp SYCL | IPEX-LLM | vLLM XPU | OpenVINO GenAI | SGLang XPU |
|---|---|---|---|---|---|
| **Arc A770 verified** | Yes | Yes | Partially (bugs) | Yes | B580 verified; A770 not explicit |
| **Qwen3-235B MoE** | Yes (`-cmoe`) | Yes (FlashMoE) | Roadmap (no CPU offload) | No (single-GPU limit) | Untested |
| **Multi-GPU (3x A770)** | Yes (row-split) | Yes (TP+PP) | Yes (TP) | No (in development) | Yes (`--tp`) |
| **GGUF model format** | Yes | Yes | No | No (IR only) | No |
| **HuggingFace format** | No | Yes | Yes | Yes (via export) | Yes |
| **OpenAI API server** | Yes (`--server`) | Yes (FastAPI / vLLM) | Yes | No (library only) | Yes |
| **CPU MoE expert offload** | Yes (`-cmoe`) | Yes (FlashMoE) | No | No | No |
| **Speculative decoding** | Yes | Likely (code exists) | Yes | Yes | Yes |
| **Q8\_1 half2 SYCL kernel** | **Yes (bug source)** | **No (oneDNN path)** | **No (torch.xpu path)** | **No (OpenCL path)** | **No (torch.xpu path)** |
| **Quantization formats** | GGUF quants (Q4-Q8, IQ) | INT4/INT8/FP8/FP6 | AWQ/GPTQ/FP8 | INT4/INT8 (NNCF export) | AWQ/GPTQ |
| **Project status** | Active | **Archived Jan 2026** | Active (Intel bugs open) | Active | Active |
| **License** | MIT | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Community maturity** | High | Medium (now dead) | High (Intel support patchy) | Medium | Growing |

---

## 4. Does Any Runtime Avoid the Q8\_1 Kernel Bug?

**Yes — all alternatives except llama.cpp wrappers avoid it entirely.**

The bug is specific to llama.cpp's `ggml-sycl` backend, in the `quantize_row_q8_1` SYCL kernel and its use in `mmvq.cpp`. The alternatives take fundamentally different paths:

- **IPEX-LLM:** Uses `xe_linear` / oneDNN primitives on XPU. No GGUF-style intermediate quantization in the hot path. INT4 weights are stored in a custom packed format; the dequantize-to-float step uses a separate op, not the `q8_1` dynamic quantize-for-gemv path.
- **vLLM XPU:** PyTorch native matmul on `torch.xpu` device. No Q8\_1 intermediate. Bug surface: the paged-attention dispatch is incomplete (issue #28650), a different but currently worse bug for production use.
- **OpenVINO GenAI:** OpenCL runtime, no SYCL user-space code, no Q8\_1 format. Bug surface: single-GPU only, no multi-GPU LLM.
- **SGLang XPU:** PyTorch 2.10+xpu → oneDNN. No Q8\_1. Bug surface: A770 architecture not validated vs. B580.

None of the non-llama.cpp runtimes use GGUF's dynamic `quantize_row_q8_1` as an in-kernel intermediate, because they do not use GGUF's computational model at all. They perform weight-only quantization at model-load time and use full-precision (or packed INT4) activations during inference.

---

## 5. Community Reports: 200B+ Models on Arc A770 (Not llama.cpp)

No verified public benchmark reports were found for running 200B+ class models on Arc A770 with any runtime other than llama.cpp + IPEX-LLM's FlashMoE. The IPEX-LLM FlashMoE documentation shows a demo GIF of Qwen3MoE-235B on a single A770 but **does not publish TPS numbers**. The memory requirement (128 GB CPU DRAM) means such runs are constrained to servers, not consumer desktops.

The only vLLM + Arc A770 public report for a large model found is the Qwen3-4B paged-attention error report (issue #28650), which suggests vLLM is not yet reliable even for smaller Qwen3 models on Arc.

---

## 6. Recommended Changes to Deployment Plan

### Immediate (within current llama.cpp setup)

1. **Patch or work around the Q8\_1 DMMV fallback.** PR #12858 shows the approach for Q4\_0: implement a reordered MMVQ variant for Q8\_1 that avoids the half2 cache-visibility read. If upstream won't merge it, carry it as a local patch. The performance impact on 3x A770 row-split for MoE models is material.

2. **File a focused upstream issue** describing the Q8\_1 `ds_ptr` visibility bug with a minimal repro — it does not currently have a canonical tracking issue in the llama.cpp repo.

3. **Evaluate `Q6_K` instead of `Q8_0`/`Q8_1`** for Qwen3-235B-A22B. Q6\_K uses a different quantization block structure and may not trigger the same DMMV fallback, potentially hitting MMVQ instead. Trade-off: ~12% larger model size.

### Medium-term Migration Options

4. **SGLang XPU is the most credible active alternative** for HuggingFace-format models on Arc. It avoids the Q8\_1 kernel path entirely, has active Intel GPU development, exposes an OpenAI API, and supports tensor parallel multi-GPU. **Action:** Test SGLang XPU with Qwen3-7B/14B BF16 on A770 to validate A770 (Alchemist) compatibility. If it works, the source-install path is the correct starting point: PyTorch 2.10+xpu + triton-xpu 3.6.0.
   - Source: [sgl-project/sglang xpu.md](https://raw.githubusercontent.com/sgl-project/sglang/main/docs/platforms/xpu.md)

5. **IPEX-LLM FlashMoE for Qwen3-235B-A22B**, if you have ≥128 GB system RAM, is the only validated path for that specific model size on Arc outside of llama.cpp. The archive status is a real risk, but the code is stable and the v2.2.0 release is available. Consider forking it rather than depending on the upstream. Use the `--low-bit sym_int4` path for weight-only quantization; avoid the FlashMoE GGUF path if you suspect lingering Q8\_1 issues in its GGUF loader.
   - Source: [flashmoe_quickstart.md](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/flashmoe_quickstart.md)

6. **Avoid vLLM XPU for Arc A770 in production** until issue [#28650](https://github.com/vllm-project/vllm/issues/28650) is resolved. The paged-attention dispatch bug on Arc dGPU is more disruptive than the llama.cpp Q8\_1 DMMV bug — it produces incorrect outputs silently rather than just reducing throughput.

7. **OpenVINO GenAI is not viable for your workload.** Single-GPU only for LLM inference means you cannot use 3x A770, and 16 GB per card is insufficient for any 235B model. Suitable only for models ≤7B on a single card where ease of deployment matters more than performance.

### Long-term

8. **Monitor PyTorch 2.10+ native XPU support.** Intel's strategy post-IPEX-LLM archive is to upstream Arc support directly into PyTorch. As that matures, SGLang and vLLM XPU will improve without requiring Intel-specific library forks.

9. **Track Ollama SYCL PRs** (#10322, #11160) — if either merges, Ollama would provide the easiest user-facing serving path but would inherit llama.cpp's kernel bugs. Worth watching but not acting on yet.

10. **Consider the Vulkan backend in llama.cpp as a parallel test.** Issue [#10879 (Vulkan benchmark thread)](https://github.com/ggml-org/llama.cpp/issues/10879) shows Arc A770 getting ~314 t/s (pp512) on Llama-2-7B Q4\_0 on Windows, with Linux Mesa lagging due to missing coopmat support. The Vulkan backend does not use the Q8\_1 SYCL kernel at all, so the DMMV fallback bug is absent — but Vulkan coopmat support on Arc/Linux/Mesa is incomplete ([issue #18808](https://github.com/ggml-org/llama.cpp/issues/18808) documents Vulkan flash-attention slowdowns). For MoE row-split, Vulkan multi-GPU support in llama.cpp is less mature than SYCL.
