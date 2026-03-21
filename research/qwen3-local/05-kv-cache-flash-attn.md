# KV Cache Quantization and Flash Attention on Intel Arc A770 (SYCL)

**Research date:** 2026-03-15
**Hardware context:** 3x Intel Arc A770 (16 GB each, 48 GB total VRAM), llama.cpp SYCL backend
**Model:** Qwen3-235B-A22B Q3_K_S with `-cmoe` (experts on CPU)
**Active config under evaluation:** `-fa on -np 2 --cache-reuse 256`

---

## 1. Flash Attention (`-fa on`) on SYCL/Arc A770

### Status: Working, Merged 2026-03-08

Flash attention for SYCL was merged via [PR #20190](https://github.com/ggml-org/llama.cpp/pull/20190) on 2026-03-08. The implementation is native SYCL (not a wrapper), adapted from the CUDA backend using Intel's `dpct` migration tool with significant manual debugging. The PR author (Intel contractor NeoZhangJianyu) confirmed: *"The SYCL code is migrated from CUDA backend. Using tool dpct, most of CUDA code can be migrated to SYCL code. To get the correct result, there is lots of workload to debug the code."*

Source: `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-sycl/common.hpp` line 38:
```cpp
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
```

This `#define` is hard-coded in the common header, meaning flash attention is **always compiled in** for SYCL builds. It is enabled at runtime by default and can be disabled via the environment variable `GGML_SYCL_ENABLE_FLASH_ATTN=0`. Source: `docs/backend/SYCL.md` line 702:
```
| GGML_SYCL_ENABLE_FLASH_ATTN | 1 (default) or 0 | Enable Flash-Attention. It can reduce memory usage. ...
```

**Auto-detection:** With `-fa auto` (default), the scheduler tests whether the FA tensor can be assigned to the same device as the KV cache. If there is a device mismatch (e.g., missing GPU support), it falls back to standard attention with a warning:
```
llama_context: Flash Attention was auto, set to disabled
```
This was the behavior seen on Iris Xe in [issue #19656](https://github.com/ggml-org/llama.cpp/issues/19656) — a confirmed Arc A770 dGPU does NOT hit this path.

### Performance on Arc A770 (from PR #20190 benchmarks)

| Model | PP delta | TG delta | Memory delta |
|-------|----------|----------|--------------|
| DeepSeek-R1-Distill-Llama-8B-Q4_0 | -62.7% (faster) | +21.8% (faster) | -38 MB |
| gpt-oss-20b-Q4_0 | -41.2% | +11.0% | -150 MB |
| gpt-oss-20b-Q8_0 | -41.3% | +10.7% | -150 MB |
| Qwen3-14B-Q4_K_M | -53.0% | +8.7% | -62 MB |
| Qwen3.5-35B-A3B-UD-IQ2_XXS | -27.1% | **-13.4% (regression)** | 0 MB |

Negative PP delta = PP is slower with FA. Positive TG delta = TG is faster with FA. **PP consistently regresses 20-77% with FA on Arc A770.** TG improves 10-22% for dense models but can regress for MoE models. The large PP regression is a known characteristic of the SYCL FA implementation — the current kernel is not optimized for batch processing. The TODO in the PR explicitly notes "performance optimization" as remaining work.

**Implication for Qwen3-235B-A22B:** TG performance is the bottleneck, and FA is expected to improve it. PP performance during prompt ingestion will be worse with FA on than without. For interactive use (short prompts, long generation), `-fa on` is still preferable.

---

## 2. KV Cache Quantization on SYCL: Which Types Work

### Supported KV type combinations

The SYCL flash attention implementation supports a **default set** of KV type combinations and an expanded set under the compile flag `GGML_SYCL_FA_ALL_QUANTS` (which has **no cmake option** in the SYCL build — it only exists for CUDA/HIP via `GGML_CUDA_FA_ALL_QUANTS`). Without the flag, the runtime dispatch in `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-sycl/fattn.cpp` lines 149-153 restricts the VEC kernel to:

```cpp
// Without GGML_SYCL_FA_ALL_QUANTS (the current build):
FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
```

Additionally, `fattn.cpp` line 224-228 enforces: without `GGML_SYCL_FA_ALL_QUANTS`, **K and V must be the same type**. Mismatched types (e.g., `-ctk q8_0 -ctv f16`) cause `BEST_FATTN_KERNEL_NONE` to be returned, which aborts with `"Not support Flash-Attention"`.

The TILE kernel (used for prefill batches) operates on `sycl::half2` pointers directly (`fattn-tile.hpp` lines 845-847), meaning it **only works correctly with F16/F32 K and V tensors**. The kernel does not have quantized K/V paths.

**Supported combinations on SYCL (current build):**

| `-ctk` | `-ctv` | `-fa on` | Notes |
|--------|--------|----------|-------|
| `f16` (default) | `f16` (default) | required for V quant, optional for K | Works correctly |
| `q8_0` | `q8_0` | required | Works; VEC kernel used for TG |
| `q4_0` | `q4_0` | required | Works; VEC kernel used for TG |
| `q8_0` | `f16` | **WILL ABORT** | K≠V type, no FA_ALL_QUANTS |
| `f16` | `q8_0` | **WILL ABORT** | K≠V type; also V quant requires FA |
| `q8_0` | (default f16) | **WILL ABORT** | Same as above |
| `bf16`, `q4_1`, `q5_0`, `q5_1`, `iq4_nl` | any | **BEST_FATTN_KERNEL_NONE** | Not in switch-case, abort |

**Critical rule:** On SYCL without recompiling with `GGML_SYCL_FA_ALL_QUANTS` defined, you must specify **both** `-ctk` and `-ctv` as the same type if using any quantized KV. Specifying only `-ctk q8_0` without `-ctv q8_0` will crash.

### KV quantization requires Flash Attention for V

`llama-context.cpp` line 2966-2967:
```cpp
if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
    LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
```
V cache quantization is **impossible** without `-fa on`. K cache quantization without V quantization also fails on SYCL (K≠V constraint). Therefore **any KV quantization on SYCL requires `-fa on` and matching K=V types**.

### Why f16 KV outperforms q8_0 KV on Arc A770

The observed speed difference (6.96 vs 6.67 t/s for 32B models) is consistent with the architecture: the A770 has 16 GB of GDDR6 at ~512 GB/s. F16 KV with the TILE GQA kernel path is a highly-optimized path; Q8_0 KV uses the VEC kernel which loads int8 data and dequantizes on-the-fly. The extra arithmetic for dequantization offsets the memory bandwidth savings for this GPU. The A770's memory bandwidth is not a bottleneck for Qwen3-235B-A22B KV cache at the context lengths in use (KV cache is tiny relative to model weights — see Section 3 below).

---

## 3. KV Cache Size Calculations for Qwen3-235B-A22B

### Architecture (from [config.json](https://huggingface.co/Qwen/Qwen3-235B-A22B/resolve/main/config.json))

| Parameter | Value |
|-----------|-------|
| `num_hidden_layers` | 94 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | **4** (GQA ratio = 64/4 = 16) |
| `head_dim` | 128 |
| `max_position_embeddings` | 40,960 |

**Formula:** `KV_bytes = n_layers × n_kv_heads × head_dim × 2 × ctx_len × bytes_per_element`
= `94 × 4 × 128 × 2 × ctx_len × bpe`

### KV cache per slot (single sequence)

| Context | f16 (2 B/elem) | q8_0 (1 B/elem) | q4_0 (0.5 B/elem) |
|---------|----------------|-----------------|-------------------|
| 1,024   | 0.184 GB | 0.092 GB | 0.046 GB |
| 4,096   | 0.734 GB | 0.367 GB | 0.184 GB |
| 8,192   | 1.469 GB | 0.734 GB | 0.367 GB |
| 16,384  | 2.938 GB | 1.469 GB | 0.734 GB |
| 32,768  | 5.875 GB | 2.938 GB | 1.469 GB |
| **40,960** | **7.344 GB** | **3.672 GB** | **1.836 GB** |

### With `-np 2` (two parallel slots, KV cache doubled)

| Context | f16 × 2 slots | q8_0 × 2 slots |
|---------|---------------|----------------|
| 8,192   | 2.94 GB | 1.47 GB |
| 16,384  | 5.88 GB | 2.94 GB |
| 32,768  | 11.75 GB | 5.88 GB |
| 40,960  | 14.69 GB | 7.34 GB |

### VRAM budget with `-cmoe` on 3x A770 (48 GB total)

With `-cmoe`, expert FFN weights (227B parameters out of 235B total) go to CPU RAM. Only attention + embeddings + router weights remain on GPU:

| Component | Estimated GPU VRAM (Q3_K_S, ~3.44 bits/weight) |
|-----------|------------------------------------------------|
| Attention weights (94 layers × 71.3M params) | ~2.88 GB |
| Token embedding (151,936 × 4096, f16) | ~1.24 GB |
| Router weights (94 × 128 × 4096) | ~0.02 GB |
| RMS norms + misc | ~0.01 GB |
| **Total model on GPU** | **~4.15 GB** |
| Activation buffers + overhead | ~2 GB |
| **Available for KV cache** | **~41.8 GB** |

**Conclusion:** At f16 KV with `-np 2`, even the full 40,960 token context (14.69 GB) fits comfortably within the ~41.8 GB available. KV cache size is **not a limiting factor** for this model with `-cmoe`. Context length is only limited by model capability (max_position_embeddings = 40,960) and the O(N²) prefill cost.

### Flash attention memory savings during prefill

Flash attention does **not** reduce the persistent KV cache storage size. It eliminates the need to materialize the full Q×K^T attention score matrix during computation, reducing peak working memory during prefill from O(N²) to O(N×D):

| Context | QK^T working memory (standard attention, f32, 64 heads) |
|---------|-------------------------------------------------------|
| 2,048   | 1.00 GB |
| 4,096   | 4.00 GB |
| 8,192   | **16.0 GB** |
| 16,384  | **64.0 GB** (impossible without FA) |
| 32,768  | **256 GB** (impossible without FA) |

Flash attention is therefore **mandatory** for contexts beyond ~4K on Arc A770 (16 GB per device). At 8K context, standard attention alone would require 16 GB of working memory per device — exhausting a single A770. With FA, the working memory per block is O(head_dim × block_size) ≈ a few MB regardless of context length.

---

## 4. Flash Attention Kernel Selection for Qwen3-235B-A22B (GQA ratio = 16)

### GQA support in SYCL flash attention: Confirmed working

The SYCL flash attention implementation handles GQA via `gqa_ratio = Q->ne[2] / K->ne[2]` in both kernels. Source: `fattn-tile.hpp` line 843 and `fattn.cpp` line 180.

For Qwen3-235B-A22B:
- `gqa_ratio = 64 / 4 = 16`
- `K->ne[0] = 128` (head_dim) — falls into `case 128` in the dispatcher (`fattn.cpp` line 205)
- `V->ne[0] = K->ne[0] = 128` — passes the mismatch check

**GQA optimization path** (from `fattn-tile.hpp` lines 1266-1301):
```
gqa_limit = gqa_ratio <= 4 && DV <= 256 ? 16 : INT_MAX
         = 16 > 4, so gqa_limit = INT_MAX  (all batch sizes get GQA opt)
use_gqa_opt = mask && max_bias == 0.0f && Q->ne[1] <= INT_MAX && ctx_len % 256 == 0
DV = 128 <= 256 → checks: gqa_ratio % 8 == 0? → 16 % 8 = 0 → TRUE → ncols2 = 8
```

With `ncols2=8`, the tile kernel processes **8 Q-heads per K/V-head simultaneously** (reducing the 16 Q-heads per KV-head to 2 outer iterations). All standard context lengths (8192, 16384, 32768, 40960) are multiples of 256 (`FATTN_KQ_STRIDE`), so `use_gqa_opt` applies unconditionally.

**Kernel dispatch per phase:**

| Phase | KV type | Kernel selected |
|-------|---------|-----------------|
| TG (Q->ne[1]=1), F16/F16 | F16 | TILE (gqa_opt applies → ncols2=8) |
| TG, Q8_0/Q8_0 | Q8_0 | VEC (quantized, Q->ne[1]<=2) |
| PP (prefill batch) | any | TILE |

The TILE kernel is a general-purpose blocked flash attention kernel for multi-token batches. The VEC kernel is a vector-based kernel optimized for decode-time single-token queries.

### No known GQA correctness issues on SYCL

No filed issues as of 2026-03-15 report incorrect output specifically attributable to GQA + flash attention on SYCL/Arc A770. The GQA indexing in the TILE kernel (`K + nb13 * sequence + nb12 * (head0 / gqa_ratio)`) mirrors the CUDA implementation exactly.

---

## 5. Known Correctness Issues: Flash Attention + KV Quantization on SYCL

### Garbage output with KV quantization (issue #19847, Vulkan; SYCL adjacent)

A user reported in [issue #19847](https://github.com/ggml-org/llama.cpp/issues/19847) garbage output with SYCL backend under long prompts (~5000 tokens) starting around build b8132. A commenter identified that the garbage appears specifically when **KV quantization is set to q4_0, q8_0, or f16** and disappears with q5_0 or bf16. This was observed on Vulkan (AMD), not SYCL directly, but the comment is relevant:

> *"The garbage is being shown when the quantization of tk/tv is set to q4_0, q8_0, f16, and disappears when I'm changing quantization to other values — eg q5_0 or bf16. I have the app built with Vulkan"* — commenter @eliah

The SYCL reporter (iGPU Intel Ultra 7) never isolated a commit but confirmed it did not occur in b6527. This is likely a separate regression from the SYCL flash attention merge (which was b8184/2026-03-08). **No confirmed instance of this affecting Arc A770 dGPU with SYCL + flash attention.**

### No correctness issues specific to Arc A770 dGPU with `-fa on`

The PR #20190 states "All supported Flash Attention UT cases are passed" and was tested explicitly on "Arc770 and iGPU on i7-13700K." No post-merge correctness regressions have been filed against Arc A770 dGPU with `-fa on` as of 2026-03-15.

### The unmerged PR #16969 was a different, broken implementation

[PR #16969](https://github.com/ggml-org/llama.cpp/issues/16969) was an earlier SYCL flash attention attempt (Nov 2025) that had GPU enablement failures and UT crashes. It was **not merged**. The current codebase uses the PR #20190 implementation exclusively.

---

## 6. `-np 2` with `--cache-reuse 256`: KV Slot Handling on SYCL

### What `--cache-reuse 256` actually means

Source: `common/arg.cpp` lines 2954-2961 (documentation string):
```
"min chunk size to attempt reusing from the cache via KV shifting,
 requires prompt caching to be enabled (default: 0)"
```

The number **256 is the minimum token overlap threshold** — if an incoming request shares at least 256 consecutive tokens (starting from `n_past`) with the cached prompt in a slot, the server shifts the existing KV cache data via `llama_memory_seq_add` rather than recomputing it. It is NOT a count of tokens to save.

Implementation from `server-context.cpp` lines 2262-2283:
```
for each chunk of matching tokens:
  if n_match >= n_cache_reuse (256):
    shift KV cache: llama_memory_seq_rm + llama_memory_seq_add
```

This requires `cache_prompt: true` on requests. With `-np 2`, each of the 2 slots maintains its own KV cache independently. Cache reuse applies per-slot.

### SYCL handles multi-slot KV correctly

The KV cache slot mechanism is backend-agnostic — it operates through `llama_memory_seq_rm`, `llama_memory_seq_add`, and the standard GGML backend tensor operations. There are no SYCL-specific slot management code paths. The SYCL backend implements the required `ggml_backend_tensor_get`/`set` synchronously (`ggml-sycl.cpp` lines 551-566):
```cpp
stream.memcpy(data, (const char *)tensor->data + offset, size).wait()
```
No async issues with KV slot operations.

**Warning in server code** (`server-context.cpp` line 2246-2247):
```cpp
if (!can_cache_reuse && n_cache_reuse > 0) {
    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
```
This fires when `n_cache_reuse` is set but `can_cache_reuse` is false. `can_cache_reuse` requires `cache_prompt` to be enabled in the request params. If requests do not include `"cache_prompt": true`, the 256-token reuse will be silently ignored per-slot.

---

## 7. Slot Save/Restore (`/slots/{id}/action=save|restore`) on SYCL

### Status: Works correctly on SYCL

The slot save/restore API (enabled by `--slot-save-path`) serializes KV cache state to disk using `ggml_backend_tensor_get` (device→host copy) and deserializes using `ggml_backend_tensor_set` (host→device copy). Both are implemented in the SYCL backend with synchronous `sycl::queue::memcpy().wait()` calls.

The save format records KV types, layer counts, and raw tensor bytes — it is backend-agnostic and will correctly serialize quantized KV (q8_0, q4_0) or f16 KV.

Source chain: `server-context.cpp:1819` → `llama_state_seq_save_file` → `state_seq_write_data` → `llama_kv_cache::state_write_data` (`llama-kv-cache.cpp:1773`) → `io.write_tensor` → `ggml_backend_tensor_get`.

**No SYCL-specific issues with slot save/restore have been filed.**

---

## 8. Summary Table: Flag Compatibility on SYCL/Arc A770

| Configuration | Works? | Notes |
|---------------|--------|-------|
| `-fa on` | Yes | Compiled in by default (`#define SYCL_FLASH_ATTN`) |
| `-fa off` | Yes | Disables SYCL fattn dispatch |
| `-fa auto` | Yes | Auto-enables if SYCL device supports it |
| `-ctk f16 -ctv f16 -fa on` | Yes | Default, fastest for A770 |
| `-ctk q8_0 -ctv q8_0 -fa on` | Yes | Slower than f16 on A770; needs both K=V |
| `-ctk q4_0 -ctv q4_0 -fa on` | Yes | Compiles + dispatches correctly |
| `-ctk q8_0` (no `-ctv`) | **ABORTS** | K≠V type without GGML_SYCL_FA_ALL_QUANTS |
| `-ctk q4_1` or `-ctk q5_0` etc. | **ABORTS** | Not in default switch-case |
| `-ctv q8_0` without `-fa on` | **ERROR** at init | "V cache quantization requires flash_attn" |
| `-np 2 --cache-reuse 256` | Yes | Slots managed backend-agnostically |
| Slot save/restore (`--slot-save-path`) | Yes | Uses sync SYCL memcpy |
| GQA (Qwen3-235B, ratio=16) | Yes | Optimized path with ncols2=8 |

---

## 9. Practical Context Length Limits on 3x A770 with Qwen3-235B-A22B Q3_K_S

Given ~41.8 GB available for KV cache after model weights with `-cmoe`:

| Config | Max context (f16, 2 slots) | Max context (q8_0, 2 slots) |
|--------|---------------------------|------------------------------|
| `-np 2` | 40,960 (7.34 GB × 2 = 14.7 GB — fits) | 40,960 (3.67 GB × 2 = 7.34 GB — fits) |
| `-np 1` | 40,960 | 40,960 |

Both configurations fit the model's maximum context window (40,960 tokens) in VRAM even at f16 precision. The VRAM constraint does not limit context length for this model with `-cmoe`. The practical limit is:
1. **Prefill throughput** — PP with FA on A770 is 20-77% slower than standard attention; long prompts will be slow
2. **Max position embeddings** — hard limit at 40,960 tokens
3. **Flash attention is mandatory** above ~4K context (without FA, QK^T matrix exceeds single A770 VRAM)

---

## Recommended Changes to Deployment Plan

1. **Keep `-fa on`** — mandatory for contexts above ~4K, and TG speed improves ~10-20% for dense attention models. The PP regression is acceptable for interactive use where prefill is a one-time cost.

2. **Keep `-ctk f16 -ctv f16` (default)** — do NOT set `-ctk q8_0` alone. On SYCL without `GGML_SYCL_FA_ALL_QUANTS`, mismatched K/V types abort. Even matched q8_0/q8_0 is slower than f16/f16 on Arc A770 due to dequantization overhead vs. bandwidth savings. Your empirical result (6.96 vs 6.67 t/s) confirms f16 is faster.

3. **Extend context if needed** — the KV cache is only ~7.3 GB at 40,960 tokens f16 with 2 slots. You have ~42 GB available. There is no VRAM pressure from KV cache; consider `-c 40960` if the model's reasoning requires long context.

4. **`--cache-reuse 256` is fine** — ensure client requests include `"cache_prompt": true` or the reuse threshold is never triggered. The 256-token minimum is reasonable for multi-turn conversations.

5. **Do not use `-ctk q8_0` without also setting `-ctv q8_0`** — the mismatch will crash. If you want to experiment with KV quantization to save memory (currently not needed given VRAM budget), use `q8_0/q8_0` or `q4_0/q4_0` pairs only, always with `-fa on`.

6. **Slot save/restore is safe** — can use `--slot-save-path /path/to/slots/` without SYCL-specific concerns. The synchronous SYCL memcpy correctly serializes GPU KV state to disk.

7. **GQA is handled correctly** — no action required. The ncols2=8 optimization path is active for Qwen3-235B-A22B's GQA ratio of 16, which is confirmed by source inspection.

8. **Monitor for PP performance** — if prompt preprocessing becomes a bottleneck (e.g., large system prompts being re-ingested frequently), consider `GGML_SYCL_ENABLE_FLASH_ATTN=0` and restricting context to ≤4K for those workloads only. This is a niche optimization.

9. **GGML_SYCL_FA_ALL_QUANTS not available** — if you want mixed KV types (e.g., q8_0 K with f16 V) in the future, you would need to define this at compile time in the SYCL CMakeLists. Currently no cmake option exists for it (unlike CUDA's `GGML_CUDA_FA_ALL_QUANTS`). This is a gap in the SYCL backend.

---

## Sources

| Source | URL/Path |
|--------|----------|
| SYCL FA implementation (merged) | [PR #20190](https://github.com/ggml-org/llama.cpp/pull/20190) |
| SYCL flash attention kernel | `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-sycl/fattn.cpp` |
| SYCL FA common types/kernels | `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-sycl/fattn-common.hpp` |
| SYCL FA tile kernel (GQA) | `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-sycl/fattn-tile.hpp` |
| SYCL enable flag | `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-sycl/common.hpp` line 38 |
| FA compile flag | `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` lines 279-282 |
| KV type constraints | `/home/ryan/llm-stack/llama.cpp/src/llama-context.cpp` lines 2944-2967 |
| cache-reuse semantics | `/home/ryan/llm-stack/llama.cpp/common/arg.cpp` lines 2954-2961 |
| cache-reuse implementation | `/home/ryan/llm-stack/llama.cpp/tools/server/server-context.cpp` lines 2240-2283 |
| Slot save/restore | `/home/ryan/llm-stack/llama.cpp/src/llama-kv-cache.cpp` lines 1773-1830 |
| SYCL tensor_get | `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` lines 551-566 |
| SYCL garbage output report | [Issue #19847](https://github.com/ggml-org/llama.cpp/issues/19847) |
| SYCL FA fallback on Iris Xe | [Issue #19656](https://github.com/ggml-org/llama.cpp/issues/19656) |
| SYCL FA (unmerged older attempt) | [Issue/PR #16969](https://github.com/ggml-org/llama.cpp/issues/16969) |
| Qwen3-235B-A22B config.json | [HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B/resolve/main/config.json) |
| SYCL docs env vars | `/home/ryan/llm-stack/llama.cpp/docs/backend/SYCL.md` line 702 |
