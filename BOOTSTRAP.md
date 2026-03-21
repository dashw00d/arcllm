# BOOTSTRAP: Expert Parallelism on 3x Intel Arc A770

## Current State

**Expert padding: SOLVED.** The `pad-experts-gguf.py` script pads any MoE GGUF model for GPU divisibility. Proven working on Qwen2MoE (60→61) and Qwen3MoE (128→129) with layer-split.

**EP (tensor-split): BROKEN.** The `llama.cpp-eptp` build has never produced clean output with `--split-mode tensor` on any MoE model. This is the next priority.

## Performance

| Config | Speed | Status |
|--------|-------|--------|
| 30B MoE abliterated, layer-split, np=16 | 39.4 t/s | ✅ Working (129 experts) |
| 30B MoE abliterated, layer-split, np=1 | 15 t/s | ✅ Working (129 experts) |
| 30B MoE abliterated, tensor-split EP | crashes/garbles | ❌ EP code broken |

## The EP Bug

EP code lives ONLY in `llama.cpp-eptp` (branch `ep-tp-combined`). It has never produced clean MoE output:

1. **Original investigation (March 19-20):** 96-expert REAM model garbled with EP → turned out the REAM model itself was broken (garbled on all builds). Red herring.

2. **128-expert abliterated model:** crashes with `GGML_ASSERT(split_state.ne[j] % tensor->src[i]->ne[src_split_states[i].axis] == 0)` because 128 ÷ 3 ≠ int.

3. **129-expert padded model:** crashes in `ggml_backend_meta_get_split_state` on the gate bias tensor, then even after fixing that, **still garbles output**.

4. **Key finding:** The Opus agent confirmed that even the **original 128-expert model without padding** garbles with EP tensor-split on the eptp build. The EP code is fundamentally broken.

### What's wrong with the EP code

The `llama.cpp-eptp` build has accumulated several experimental changes:

1. **`qwen3moe.cpp` deferred-PARTIAL fusion (commit 9fc046d19):**
   - `ffn_norm` operates on `inpSA` instead of `ffn_inp` (mathematical change)
   - Residual restructured: `combined_partial = wo_partial + moe_out`, then `cur = combined_partial + inpSA`
   - `norm_w` changed to `true`
   - Shared expert code removed
   - **This was never validated on GPU (task #31 never ran)**

2. **EP dispatch in `ggml-backend-meta.cpp`:**
   - `handle_mul_mat_id` with SPLIT_AXIS_2
   - `expert_offset` computation via op_params
   - Subgraph boundary deferral for gate/up MUL_MAT_ID
   - AllReduce for EP MoE path

3. **EP-aware MUL_MAT_ID in `ggml-sycl.cpp`:**
   - Pre-zero output for EP mode
   - Expert filtering by ownership
   - Local index remapping

4. **Debug instrumentation (commit 76b96dd1e):**
   - WIP printf logging for ep_flag, AllReduce values, META_SPLITS
   - Should be stripped before production

### Root cause candidates

1. **Deferred-PARTIAL math is wrong:** The `ffn_norm(inpSA)` change is mathematically different from `ffn_norm(wo_out + inpSA)`. This was designed to enable AllReduce fusion but may produce numerically incorrect results. **Task #31 was supposed to validate this but never ran.**

2. **EP split state inference bugs:** The meta backend's `get_split_state` may not correctly infer split states for all ops in the MoE path (RESHAPE, VIEW, ADD with PARTIAL tensors).

3. **AllReduce boundary placement:** The deferred AllReduce boundaries may fire at the wrong positions, causing partial results to be consumed before reduction.

4. **op_params encoding:** EP flag written to op_params[1]/[2] may conflict with other uses of those slots.

## How to Debug EP

### Step 1: Isolate deferred-PARTIAL from EP
Test if the ORIGINAL `qwen3moe.cpp` (without deferred-PARTIAL changes) works with EP:
```bash
# Revert qwen3moe.cpp to match qwen2moe.cpp structure (keep EP support, remove deferred-PARTIAL)
# Build and test with --split-mode tensor
```
If this works → deferred-PARTIAL is the bug. If not → EP dispatch itself is broken.

### Step 2: Test EP on a non-MoE tensor-split model
Test EP tensor-split on the dense Qwen3-32B to verify TP itself works on the eptp build:
```bash
GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  ./llama.cpp-eptp/build-sycl/bin/llama-server \
    -m models/Qwen/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf \
    --split-mode tensor -ngl 99 -np 1 -c 512 --port 18404 --no-warmup
```

### Step 3: Debug with META_SPLITS logging
```bash
GGML_META_DEBUG_SPLITS=1 GGML_SYCL_DISABLE_GRAPH=1 \
  ./llama.cpp-eptp/build-sycl/bin/llama-server \
    -m models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m-129experts.gguf \
    --split-mode tensor -ngl 99 -np 1 -c 256 --port 18404 --no-warmup -fa off
```
Check: subgraph count, AllReduce boundaries, PARTIAL/MIRRORED transitions.

## Key Source Files

| File | What |
|------|------|
| `llama.cpp-eptp/src/models/qwen3moe.cpp` | Modified MoE graph (deferred-PARTIAL) |
| `llama.cpp-eptp/ggml/src/ggml-backend-meta.cpp` | EP dispatch, split state, AllReduce |
| `llama.cpp-eptp/ggml/src/ggml-sycl/ggml-sycl.cpp` | EP-aware MUL_MAT_ID + debug logging |
| `llama.cpp-eptp/src/llama-model.cpp` | SPLIT_AXIS_2 for expert tensors |
| `llama.cpp-stable/src/models/qwen3moe.cpp` | Working reference (with gate bias, no EP) |
| `scripts/pad-experts-gguf.py` | Expert padding script (working) |

## Expert Padding (SOLVED — reference)

Script: `scripts/pad-experts-gguf.py`
- Pads expert count to next multiple of N GPUs
- Gate weights = zero, gate bias = -1e30 for fake experts
- Handles alignment (GGML_PAD), shared expert filtering, all quant types
- Tested: Qwen2MoE 60→61 ✅, Qwen3MoE 128→129 ✅ (layer-split)

## Environment

```bash
source /home/ryan/llm-stack/env.sglang-xpu.sh
# Stable build (layer-split, working): llama.cpp-stable/build-sycl/
# EP build (tensor-split, broken): llama.cpp-eptp/build-sycl/
# Build: cd build-sycl && cmake --build . --target llama-server -j$(nproc)
```
