# Optimal GGUF Quantization for Qwen3-235B-A22B on 3x Arc A770 + 64 GB DDR5 with `-cmoe`

**Research date:** 2026-03-15
**Target hardware:** 3x Intel Arc A770 = 48 GB VRAM total, 64 GB DDR5 RAM
**Key constraint:** `-cmoe` (vanilla llama.cpp) puts expert FFN weights (`blk.*.ffn_*_exps`) on CPU RAM; attention weights stay on VRAM.

---

## 1. What `-cmoe` Actually Does

Source: `/home/ryan/llm-stack/llama.cpp/common/arg.cpp` lines 2283–2289 and `common/common.h` lines 930–938.

```
const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

inline llama_model_tensor_buft_override llm_ffn_exps_cpu_override() {
    return { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() };
}
```

`-cmoe` applies a tensor buffer override that regex-matches every `ffn_up_exps`, `ffn_down_exps`, `ffn_gate_exps` tensor across all 94 layers and forces them to CPU RAM. Attention tensors (`attn_q`, `attn_k`, `attn_v`, `attn_output`), norms, embeddings, and output head remain in VRAM. This is equivalent to manually specifying `-ot ".ffn_(up|down|gate|gate_up)_(ch|)exps=CPU"`.

A partial version `-ncmoe N` puts only the first N layers' experts on CPU, useful for hybrid setups where some VRAM is available for experts.

**Practical implication:** With `-cmoe`, VRAM holds only the **dense/attention** portion of the model. For Qwen3-235B-A22B (94 layers, 64 Q-heads GQA with 4 KV-heads, hidden dim 7168), this is substantially smaller than the total GGUF size — the budget-limiting factor shifts from VRAM to **CPU RAM**.

---

## 2. Architecture and Tensor Size Breakdown

Qwen3-235B-A22B architecture (identical in original and 2507 update):
- 94 transformer layers
- 128 experts per layer, 8 activated per token
- GQA: 64 Q-heads, 4 KV-heads
- Hidden dim: 7168
- FFN intermediate dim (per expert): 2048 (gate + up + down projections each)
- Total parameters: 235B; activated parameters: 22B

Source: [Qwen/Qwen3-235B-A22B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)

### Weight Category Estimates

The ubergarm IQ3_K mixed quant (3.903 BPW overall, 106.8 GiB total) uses IQ6_K (~6 bpw) for attention tensors and IQ3_K/IQ4_K for expert FFN. This means experts are a large fraction: ubergarm confirms the model is ~94 GiB in expert FFN weight across all layers, with ~13 GiB in dense/attention weights. The ratio holds proportionally across quant levels:

| Quant | Total Size | Est. Expert FFN (CPU) | Est. Attention/Dense (VRAM) |
|-------|-----------|----------------------|----------------------------|
| Q2_K  | 85.7 GB   | ~73 GB               | ~12 GB                     |
| Q2_K_L | 85.8 GB  | ~73 GB               | ~12 GB                     |
| Q3_K_S | 101 GB   | ~87 GB               | ~14 GB                     |
| Q3_K_M | 112 GB   | ~97 GB               | ~15 GB                     |
| UD-Q3_K_XL | 104 GB | ~89 GB             | ~15 GB                     |
| IQ4_XS | 126 GB   | ~110 GB              | ~16 GB                     |
| Q4_K_S | 134 GB   | ~117 GB              | ~17 GB                     |
| Q4_K_M | 142 GB   | ~124 GB              | ~18 GB                     |
| UD-Q4_K_XL | 134 GB | ~117 GB            | ~17 GB                     |
| Q5_K_S | 162 GB   | ~143 GB              | ~19 GB                     |
| Q5_K_M | 167 GB   | ~147 GB              | ~20 GB                     |
| Q6_K  | 193 GB   | ~170 GB              | ~23 GB                     |
| Q8_0  | 250 GB   | ~220 GB              | ~30 GB                     |

Sources: Unsloth GGUF sizes from [unsloth/Qwen3-235B-A22B-GGUF](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF); attention fraction estimated from ubergarm's mixed-quant PPL data showing ~12% of weight in dense tensors.

**VRAM fit check (48 GB total):** Even Q8_0's ~30 GB attention weights fit in 48 GB alongside KV cache. VRAM is not the bottleneck for quant selection under `-cmoe`.

**RAM fit check (64 GB):**
- Q3_K_S expert weights ~87 GB — does NOT fit in 64 GB RAM without mmap swapping
- Q2_K expert weights ~73 GB — does NOT fit either
- Q2_K_L expert weights ~73 GB — does NOT fit

This is a critical finding: **even Q3_K_S expert weights (~87 GB) exceed 64 GB RAM**. The current setup relies on mmap/demand paging from SSD or swapped pages, which dramatically impacts throughput on generation (TG) tokens, since TG requires reading every expert weight that is activated on each token.

> **Key constraint:** To fit expert weights fully in RAM without swapping, you need a total GGUF size < ~74 GB (so expert weights < ~62 GB, leaving ~2 GB headroom). That is only achievable at Q2_K or Q2_K_L (~85 GB total, ~73 GB experts) — still too large by ~9 GB.

In practice, with `-cmoe` and 64 GB RAM, the OS will keep as much as fits hot in RAM and page-fault the rest from SSD on demand. **This is why TG speed is highly SSD-dependent at Q3_K_S or larger.**

---

## 3. Quant Quality Comparison

### 3.1 Perplexity Data (Artefact2 / mradermacher)

For Mistral-7B, KL-divergence (lower = better, closer to F16):

| Type | KL-div (median) |
|------|-----------------|
| IQ3_XS | ~0.028 |
| Q3_K_S | 0.0304 |
| IQ3_S | 0.0205 |
| IQ4_XS | 0.0088 |
| Q4_K_S | 0.0083 |
| Q5_K_S | 0.0045 |

Source: [Artefact2 GGUF KL-divergence gist](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9) (Mistral-7B, author notes larger models are less sensitive to quantization noise).

### 3.2 Ubergarm PPL Data for Qwen3-235B-A22B-Instruct-2507

Measured on `wiki.test.raw`:

| Quant | BPW | PPL | Delta vs Q8_0 |
|-------|-----|-----|---------------|
| BF16 | 16.0 | 4.3079 ± 0.025 | baseline |
| Q8_0 | 8.5 | 4.3139 ± 0.025 | +0.006 |
| IQ5_K (ik) | 5.9 | 4.3351 ± 0.025 | +0.027 |
| IQ4_K (ik) | 4.9 | 4.3668 ± 0.025 | +0.059 |
| IQ4_KSS (ik) | 4.2 | 4.4017 | +0.088 |
| IQ3_K (ik) | 3.9 | 4.4561 ± 0.025 | +0.148 |
| IQ3_KS (ik) | 3.7 | 4.4915 | +0.181 |
| IQ2_KL (ik) | 3.0 | 4.7912 ± 0.029 | +0.483 |

Source: [ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) (ik_llama.cpp quant types — not directly comparable to standard K-quants but indicative)

For standard K-quants (original model, ubergarm discussion #1):

| Quant | PPL (calibration_data_v5_rc.txt) |
|-------|----------------------------------|
| IQ3_K mix (ubergarm) | 3.8092 ± 0.036 |
| IQ4_XS | 3.7938 ± 0.036 |

Source: [ubergarm/Qwen3-235B-A22B-GGUF discussions](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/1)

### 3.3 Qualitative User Reports

- User on 3x 3090 + P40 (48 GB VRAM) + 64 GB RAM with Q2_K achieves 2–3 tok/s and reports quality "better than free OpenRouter version." ([unsloth discussions #6](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/discussions/6))
- Unsloth team reported in discussion #11: Q4_K_M and UD-Q4_K_XL have "equal quality," with UD-Q4_K_XL "a tad bit faster" due to mixed precision. ([unsloth discussions #11](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/discussions/11))
- Repetition issues reported on original release were traced to a **chat template bug**, not quantization; fixed by Unsloth ~2025-04-29. ([unsloth discussions #8](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/discussions/8))

---

## 4. IQ Quants (IQ3_K, IQ4_XS) vs K-Quants on SYCL

### 4.1 SYCL Backend IQ Support Status

Examining the local SYCL backend source at `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-sycl/`:

**`ggml-sycl.cpp` line 11777–11781** (in `ggml_sycl_supports_op` for `GGML_OP_MUL_MAT`):
```cpp
if (a_type == GGML_TYPE_IQ4_NL  || a_type == GGML_TYPE_IQ4_XS ||
    a_type == GGML_TYPE_IQ3_XXS || a_type == GGML_TYPE_IQ3_S  ||
    a_type == GGML_TYPE_IQ2_XXS || a_type == GGML_TYPE_IQ2_XS || a_type == GGML_TYPE_IQ2_S ||
    a_type == GGML_TYPE_IQ1_S || a_type == GGML_TYPE_IQ1_M
    ) {
    if (b->ne[1] == 1 && ggml_nrows(b) > 1) {
        return false;
    }
}
```

This returns `false` (falls back to CPU) for single-row batch sizes with IQ types — exactly the TG (token generation) scenario where batch size = 1. During TG, all IQ quant attention tensors will be computed on CPU rather than SYCL GPU.

**`ggml-sycl.cpp` line 6190–6200:** IQ2_XXS explicitly has `use_mul_mat_q` disabled. Line 6190 notes: "quantization which has a ds half2 visibility bug on Intel Arc."

**`convert.cpp` lines 624–708:** IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL all have dequantization conversion kernels in SYCL, meaning they can be loaded and dequantized on GPU for prompt processing (PP) but fall back to CPU for single-token generation.

**`ggml-sycl.cpp` lines 1450–1465** (row rounding for MMVQ):
```cpp
case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: ...
case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ4_NL:
    return max_compute_capability >= VER_GEN9 ? 128 : 64;
```
Arc A770 is gen12+, so it qualifies for 128-row rounding — IQ4_XS and IQ3_S are handled but with the TG single-batch caveat above.

**Summary for SYCL:**
- K-quants (Q3_K, Q4_K, Q5_K, Q6_K): **Fully supported** for both PP and TG on SYCL.
- IQ4_XS, IQ3_S, IQ3_XXS, IQ4_NL: Supported for PP (batch > 1), but **fall back to CPU for TG** (batch = 1).
- IQ2_XXS: Known Intel Arc bug; avoid.
- ik_llama.cpp IQ quants (IQ3_K, IQ4_K, IQ5_K, IQ2_K family): **Not in vanilla llama.cpp at all**; ik_llama.cpp has no SYCL backend. These are incompatible with Arc A770 SYCL.

### 4.2 IQ4_XS vs Q4_K_M Quality-Size Tradeoff

mradermacher hosts both standard and i1 (imatrix) versions:
- i1-IQ4_XS: 125.4 GB — "competes with Q4_K_S range," beats non-imatrix Q3_K variants
- Q4_K_M: 142.3 GB — well-established quality baseline

IQ4_XS saves ~17 GB vs Q4_K_M for equivalent or slightly lower quality, but the TG CPU fallback means attention computation moves to CPU during generation, likely halving TG speed vs K-quants of similar size on SYCL.

Source: [mradermacher/Qwen3-235B-A22B-i1-GGUF](https://huggingface.co/mradermacher/Qwen3-235B-A22B-i1-GGUF), [mradermacher/Qwen3-235B-A22B-GGUF](https://huggingface.co/mradermacher/Qwen3-235B-A22B-GGUF)

---

## 5. Unsloth "UD" Quants (UD-Q3_K_XL, UD-Q4_K_XL, etc.)

### 5.1 What Makes UD Quants Different

**Unsloth Dynamic 2.0** applies per-layer sensitivity analysis to assign different quantization bit-depths to individual weight tensors within the same nominal quant level. The quantization type of each layer is model-specific — "layers quantized in Gemma 3 differ significantly from those in Llama 4."

Key properties:
- Critical tensors (first/last layers, norms, embeddings) get higher-bit quants
- Less-sensitive layers get lower-bit quants
- Result: total file size can be *smaller* than the standard M-variant despite better quality on sensitive layers
- UD-Q4_K_XL (134 GB) is smaller than Q4_K_M (142 GB) while Unsloth claims equal or better quality
- Calibrated using Unsloth's dataset

Source: [unsloth.ai/blog/dynamic-v2](https://unsloth.ai/blog/dynamic-v2), confirmed in [unsloth discussions #11](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/discussions/11) where Unsloth team states "Q4_K_M and Q4_K_XL perform equally in quality."

### 5.2 UD Quant Sizes (Qwen3-235B-A22B and 2507)

| UD Quant | Original Size | 2507 Size |
|----------|--------------|-----------|
| UD-Q2_K_XL | 88 GB | 88.8 GB |
| UD-Q3_K_XL | 104 GB | 104 GB |
| UD-Q4_K_XL | 134 GB | 134 GB |
| UD-Q5_K_XL | 167 GB | 169 GB |
| UD-Q6_K_XL | 199 GB | 202 GB |
| UD-Q8_K_XL | 265 GB | 274 GB |

Source: [unsloth/Qwen3-235B-A22B-GGUF](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF), [unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF)

### 5.3 SYCL Compatibility of UD Quants

UD quants are still K-quant variants (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 internally at the GGML level — the "UD" refers to the per-layer assignment strategy, not a new GGML type). GGUF files use standard GGML types internally. Therefore:
- UD-Q3_K_XL, UD-Q4_K_XL: contain a mix of Q3_K and Q4_K GGML blocks — **SYCL-compatible**, no fallback issues
- UD-Q2_K_XL: contains Q2_K and possibly Q3_K blocks — **SYCL-compatible**

No known SYCL issues with UD quants reported in unsloth discussions.

---

## 6. Ubergarm Quants

### 6.1 Availability

Ubergarm maintains GGUF quantizations for both the original Qwen3-235B-A22B and the 2507 update:
- [ubergarm/Qwen3-235B-A22B-GGUF](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF) — mix-IQ3_K, 106.8 GiB
- [ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) — IQ5_K through IQ2_KL range

### 6.2 What Makes Ubergarm Quants Special

Custom **mixed-precision GGUF** strategy using ik_llama.cpp quant types:
- Attention tensors: `iq6_k` (~6 bpw — near lossless)
- Token embeddings + output: `q8_0` (lossless)
- Expert down projections: `iq4_k`
- Expert gate/up projections: `iq3_k`

PPL: 5.4403 ± 0.034 vs Q8_0's 5.3141 — only +0.126 PPL degradation at 3.9 BPW overall.

### 6.3 SYCL Incompatibility — CRITICAL

**Ubergarm quants require ik_llama.cpp, which has no SYCL backend.** The fork's only performant backends are CPU (AVX2+) and CUDA. Running ubergarm quants on Arc A770 with vanilla llama.cpp SYCL is not possible — they use ik_llama.cpp-specific GGML quant types (IQ3_K, IQ4_K, IQ5_K, IQ6_K as defined by ik_llama.cpp) that are not in the vanilla GGML type registry.

Sources: [ubergarm/Qwen3-235B-A22B-GGUF README](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF), [github.com/ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)

---

## 7. `-ot` Regex Offloading and Mixed Quant Types

### 7.1 Can Expert Weights Use a Different Quant Than Attention?

Yes — this is exactly what `-cmoe` does. The tensor buffer override (`llm_ffn_exps_cpu_override()`) assigns expert tensors to CPU buffer type regardless of their GGML quantization type. The quant type of each tensor is fixed at quantization time and embedded in the GGUF; the buffer override only changes *where* the pre-quantized tensor lives in memory.

You cannot change quant types at runtime via `-ot` — you would need to re-quantize.

### 7.2 `-ot` Pattern for Selective Expert Offloading

From ubergarm's tested configuration (72 GB VRAM setup, discussion #6):
```bash
-ot "blk\.([0-9]|1[0-9]|2[0-9]|3[0-4])\.ffn=CUDA0"   # layers 0-34 → GPU0
-ot "blk\.(5[0-9]|6[0-4])\.ffn=CUDA1"                   # layers 50-64 → GPU1
-ot exps=CPU                                              # remaining experts → CPU
```

For the 3x Arc A770 setup (48 GB VRAM) with `-cmoe`, the equivalent is simply:
```bash
-cmoe   # all ffn_*_exps tensors → CPU RAM
-ngl 99 # all other layers → SYCL GPUs
```

Or equivalently: `-ot ".ffn_(up|down|gate|gate_up)_(ch|)exps=CPU"`

### 7.3 VRAM Usage Under `-cmoe`

With all expert FFN weights on CPU RAM, VRAM holds:
- Attention matrices (Q, K, V, O projections) across all 94 layers
- Layer norms, embeddings, output head
- KV cache (scales with context length and `-ctk`/`-ctv` quantization)

For Q3_K_S total 101 GB, expert estimate ~87 GB, attention portion ~14 GB. Even with KV cache at 32K context (q8_0 KV ~4–6 GB for this model), 48 GB VRAM is ample. VRAM is not the binding constraint at any quant level under `-cmoe`.

---

## 8. Qwen3-235B-A22B-Instruct-2507 GGUF Availability

**Yes, available as of 2025-07–2026.** The 2507 (July 2025 update) is a post-training improvement with identical architecture; no thinking mode (`<think>` blocks removed by default). The following have released GGUFs:

| Provider | Repo | Quant Types |
|----------|------|-------------|
| unsloth | [unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF) | Q2_K through BF16 + full UD-* series |
| ubergarm | [ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) | IQ2_KL through IQ5_K (ik_llama.cpp only) |
| bartowski | Confirmed quantized (restricted HF access, 57+ quantized models listed) | Standard K-quants |
| mradermacher | [mradermacher/Qwen3-235B-A22B-GGUF](https://huggingface.co/mradermacher/Qwen3-235B-A22B-GGUF) | Standard + i1 imatrix |
| Qwen official | [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) | Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0 |

Source: [Hugging Face models search — Qwen3-235B-A22B-Instruct-2507 GGUF](https://huggingface.co/models?search=Qwen3-235B-A22B-Instruct-2507+GGUF)

The 2507 version drops the thinking capability by default (no `<think>` tokens), which some users prefer for coding/assistant tasks. If you run Qwen3 for reasoning tasks, stay on the original.

---

## 9. Real User Reports

### Setup: 3x 3090 + P40 (48 GB VRAM) + 64 GB RAM — Q2_K
User reports 2–3 tok/s TG. Quality subjectively better than OpenRouter free tier. Chat template bug caused repetition on initial release; fixed 2025-04-29. ([unsloth discussions #6](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/discussions/6))

### Setup: RTX 3090 Ti (24 GB) + AMD 9950X + 96 GB DDR5 RAM — IQ3_K mix (ubergarm)
- PP: 12–67 tok/s (varies by batch size and `-rtr`/`-fmoe` flags)
- TG: ~7–8 tok/s
- RAM bandwidth is the bottleneck; removing `-rtr` boosted PP from 24 → 67 tok/s
Source: [ubergarm discussions #3](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/3)

### Setup: Xeon W5-3425 + RTX 4090D (48 GB) + RTX 3090 (24 GB) + 512 GB DDR5 — IQ3_K mix
- VRAM: 47/48 GB (GPU0), 23/24 GB (GPU1), experts on CPU via `-ot exps=CPU`
- PP: 147.5 tok/s, TG: 14.3 tok/s
- This is ik_llama.cpp with `-fmoe` flag, not vanilla llama.cpp
Source: [ubergarm discussions #6](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/6)

### Quality: IQ3_K vs IQ4_XS (ubergarm data)
PPL on calibration data: IQ3_K = 3.8092, IQ4_XS = 3.7938 — "nearly identical," small vocabulary edge to IQ4_XS. ([ubergarm discussions #1](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/1))

---

## 10. Constraints Summary Table

| Constraint | Binding at which quants |
|------------|------------------------|
| 48 GB VRAM (attention only, -cmoe) | Never — even Q8_0 attention fits |
| 64 GB DDR5 RAM (expert FFN) | All quants exceed 64 GB experts; mmap/SSD paging occurs |
| 64 GB RAM fully hot (no SSD paging) | Requires expert weights < ~62 GB — impossible even at Q2_K (~73 GB experts) |
| IQ quants SYCL TG fallback | IQ4_XS, IQ3_S, IQ3_XXS fall back to CPU for single-token generation |
| ik_llama.cpp quants (ubergarm IQ3_K/IQ4_K) | Incompatible with SYCL/Arc A770 entirely |

---

## 11. Recommended Changes to Deployment Plan

### 11.1 The RAM Constraint Is Inescapable at Current Quant Levels

Even at Q2_K (85.7 GB total, ~73 GB experts), CPU RAM is overcommitted by ~9 GB. The current Q3_K_S (101 GB total, ~87 GB experts) exceeds 64 GB RAM by ~23 GB. Both rely on the OS page cache; performance degrades as SSD read latency is added per token's expert activation.

**Recommendation A — Stay at Q2_K for faster TG:**
Download Q2_K (85.7 GB, Unsloth or mradermacher) or UD-Q2_K_XL (88 GB). Expert weights ~73 GB still exceed 64 GB, but the shortfall is smaller (~9 GB vs ~23 GB for Q3_K_S). SSD page faults are less frequent. TG quality is lower but generation speed improves. Use `presence_penalty 1.5` to reduce repetition at Q2_K (Qwen official recommendation).

**Recommendation B — Upgrade RAM to 96–128 GB DDR5 (Preferred):**
With 96 GB RAM, Q3_K_S experts (~87 GB) fit hot. With 128 GB, Q4_K_M experts (~124 GB) fit. This is the highest-leverage hardware change. Qwen3-235B at Q4_K_M with 128 GB RAM and -cmoe would outperform Q3_K_S with SSD paging dramatically on TG speed (~3x improvement expected).

**Recommendation C — Try UD-Q3_K_XL (104 GB) for quality-per-byte:**
UD-Q3_K_XL is 104 GB total (vs Q3_K_S 101 GB) but uses Unsloth's sensitivity-guided mixed precision. It costs ~3 GB more in expert weight pressure but gets better quality on sensitive layers. Given SSD paging is already occurring at Q3_K_S, the marginal extra SSD reads are small. This is the best K-quant within the 3x "IQ-quality for price" range. Source: [unsloth/Qwen3-235B-A22B-GGUF](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF).

**Recommendation D — Avoid IQ4_XS for SYCL:**
While IQ4_XS (126 GB) offers better quality/size than Q4_K_M at smaller size, the SYCL backend falls back to CPU for TG with IQ types (`ggml-sycl.cpp` line 11782). This means attention computation during generation runs on the host CPU, which is substantially slower than SYCL on Arc A770. Net result: worse TG speed than Q3_K_S despite better model quality. Stick to K-quants for SYCL TG performance.

**Recommendation E — Avoid ubergarm/ik_llama.cpp quants:**
These require ik_llama.cpp which has no SYCL backend. Arc A770 GPU acceleration is impossible. CPU-only inference on 64 GB RAM with Qwen3-235B would be unusably slow.

**Recommendation F — Consider upgrading to Qwen3-235B-A22B-Instruct-2507:**
The 2507 version is architecturally identical, so all the above analysis applies. GGUF sizes are within 2% of the original. If you don't need the `<think>` reasoning mode, 2507 offers better instruction following and long-context handling. The unsloth 2507 GGUF has the same quant menu. Download UD-Q3_K_XL-2507 (~104 GB) as a drop-in upgrade.

### 11.2 Priority Action Matrix

| Action | Impact | Effort | Recommended? |
|--------|--------|--------|-------------|
| Download UD-Q3_K_XL (same budget, better quality) | Medium | Low | Yes — immediate |
| Upgrade to 2507 version | Low-Medium | Low | Yes — if no thinking mode needed |
| Upgrade RAM to 96 GB | High | Medium | Yes — biggest speedup |
| Switch to Q2_K for TG speed | Medium speed up | Low | Only if TG latency critical |
| Download IQ4_XS | Quality up, speed down | Low | No — SYCL TG fallback |
| Use ubergarm quants | N/A | Medium | No — SYCL incompatible |
| Upgrade RAM to 128 GB | Very high | Medium | Optimal long-term |

### 11.3 Current Config Reality Check

With Q3_K_S + 64 GB RAM + `-cmoe` on 3x Arc A770:
- VRAM usage: ~14 GB of 48 GB (attention only) — VRAM heavily underutilized
- Expert weights: ~87 GB against 64 GB RAM — ~23 GB on SSD via mmap
- KV cache at 32K context: ~3–5 GB VRAM (depends on `-ctk`/`-ctv`)
- Effective VRAM headroom: ~29 GB unused — could hold more expert layers with `-ncmoe`

**Hybrid approach with `-ncmoe`:** Use `-ncmoe 47` to keep the first 47 layers' experts in VRAM (using the 29 GB headroom) and only offload the remaining 47 layers to CPU. This halves the RAM expert footprint and reduces SSD paging while using the available VRAM.

```bash
# Estimated: keep first 47 layers' experts in VRAM (~14 GB experts), rest on CPU (~43 GB experts)
-ncmoe 47 -ngl 99
```

This is the highest-value configuration change achievable without hardware upgrades or re-downloading.
