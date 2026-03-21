# BOOTSTRAP: Expert Padding for ÷3 GPU Divisibility

## Goal

Pad a GGUF MoE model's expert count from 128 → 129 (43×3) so it can be evenly split across 3x Intel Arc A770 GPUs for Expert Parallelism. The fake expert(s) should never fire and produce zero quality loss.

## The Model

- **Qwen3-30B-A3B-abliterated** Q4_K_M — 128 experts, 8 active per token, 48 layers
- Path: `models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf` (17.28 GB)
- This model works perfectly at 25.7 t/s with layer-split np=16
- 128 ÷ 3 = 42.67 — doesn't divide cleanly, so EP (`--split-mode tensor`) crashes

## The Approach

Pad at the GGUF binary level (no safetensors conversion needed):

1. **Expert FFN tensors** (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) — shape `[dim, dim, 128]` → `[dim, dim, 129]`. Append one expert's worth of zero bytes (quantized Q4_K/Q6_K zero blocks).

2. **Router gate** (`ffn_gate_inp`) — shape `[2048, 128]` F32 → `[2048, 129]`. Append one row of `-1e9` so top-k never selects expert 128.

3. **Metadata** — update `qwen3moe.expert_count: 128 → 129`

4. **Tensor offsets** — recalculate since padded tensors are larger, write actual offsets back into header.

## What We Built

Script: `scripts/pad-experts-gguf.py` — binary-level GGUF patcher. Reads/writes raw GGUF format without the gguf-py library's shape validation (which rejects non-aligned quantized tensors like output.weight Q6_K with 151936 elements).

## What We Verified (All Correct)

### Data integrity
- **387 non-expert tensors**: byte-identical between original and padded ✅
- **144 expert FFN tensors**: first 128 experts byte-identical ✅  
- **48 router gate tensors**: experts 0-127 have original weights, expert 128 = all -1e9 ✅

### GGUF format
- **Tensor offsets**: GGUF C reader validates stored offsets against recomputed sizes at load time. If the file loads without error, offsets are correct. Our file loads cleanly. ✅
- **Strides (nb[])**: `nb[2] = nb[1] * ne[1]` — independent of `ne[2]`. Changing from 128→129 experts does NOT change any stride. ✅
- **Expert_count metadata**: correctly reads 129 from the padded file ✅
- **Tensor shapes**: all expert tensors show ne[2]=129 in the loader ✅

### Math
- **Softmax**: `exp(-1e9) = 0.0` in float32. Adding a -1e9 element to softmax produces ZERO change in the first 128 probabilities. Verified in numpy. ✅
- **Top-k selection**: identical indices for 128 vs 129 experts ✅
- **Argsort stride**: the `argsort_top_k` view has `nb[1] = 129*4 = 516` (vs 512 for 128). Both `get_rows` and `MUL_MAT_ID` use `ids->nb[1]` correctly ✅

### Softmax template hypothesis (DISPROVEN)
- Tried padding to 256 experts (power-of-2, hits specialized `soft_max_f32<true, 256, 256>` template) — **still garbage** ❌
- Tried CPU-only inference (`-ngl 0`) with 129 experts — **still garbage** ❌
- This rules out SYCL-specific kernel issues entirely

## The Bug (UNSOLVED)

### Symptoms
- Model loads cleanly, no errors or warnings
- First few tokens are **coherent** (e.g., `<think>\nOkay. Do not use thinking.`) then degenerates into garbage (`200001000`, `22222...`)
- Happens on both GPU (SYCL) and CPU (`-ngl 0`)
- Happens with both 129 and 256 experts
- Original 128-expert model produces perfect output

### What this means
- The corruption is **progressive** — gets worse over tokens, not instant garbage
- It happens in the **CPU math path**, not just SYCL kernels
- It's NOT a stride, offset, or data alignment issue (all verified)
- It's NOT a softmax numerical issue (mathematically identical)
- Something about having `ne[2] > 128` in the expert tensors triggers a logic bug in llama.cpp's inference

### Top remaining hypotheses

1. **ggml_mul_mat_id internal buffer sizing**: The `MUL_MAT_ID` kernel may allocate internal scratch buffers sized by `n_as` (expert count). With 129 experts, a buffer might be sized differently, causing overflow or underflow in scratch space that corrupts subsequent operations. Check `ggml_compute_forward_mul_mat_id` in `ggml-cpu/ops.cpp`.

2. **Warmup pass with n_expert_used = n_expert**: During warmup (even with `--no-warmup`), `n_expert_used = cparams.warmup ? hparams.n_expert : hparams.n_expert_used`. If any internal warmup-like pass uses `n_expert=129` as `n_expert_used`, it would route tokens through expert 128 (zero weights), producing zero outputs that corrupt the model state.

3. **Quantized zero blocks are not true zeros**: Our padding uses `\x00` bytes for the fake expert. For Q4_K, a block of all-zero bytes should dequantize to all zeros (scale=0, all nibbles=0). But verify: does Q4_K `\x00` actually produce `0.0f` after dequantization? If scale bytes happen to produce NaN or denormals, the zero expert would corrupt any AllReduce sum it participates in.

4. **Graph computation scratch buffer**: `ggml_graph_compute` allocates work buffers based on the graph's tensor sizes. If any intermediate tensor's size is computed from `n_expert` rather than `n_expert_used`, the scratch buffer layout might differ, causing memory corruption in subsequent operations.

5. **KV cache corruption via residual accumulation**: If the MoE FFN output is even slightly wrong (e.g., tiny numerical difference from Q4_K zero-block dequant), the residual connection accumulates this error. After 48 layers × N tokens, small errors compound into garbage.

## How to Debug

### Quick test
```bash
# Rebuild 129-expert model
python3 scripts/pad-experts-gguf.py \
  models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf \
  models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m-129experts.gguf

# Test (should show garbage after a few tokens)
source env.sglang-xpu.sh
GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  ./llama.cpp-stable/build-sycl/bin/llama-server \
    -m models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m-129experts.gguf \
    --split-mode layer -ngl 99 -np 1 -c 512 --port 18404 --no-warmup -fa off

# Compare with original (should be clean)
GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  ./llama.cpp-stable/build-sycl/bin/llama-server \
    -m models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf \
    --split-mode layer -ngl 99 -np 1 -c 512 --port 18404 --no-warmup -fa off
```

### Diagnostic approaches

1. **Verify Q4_K zero dequantization**: Write a test that creates a Q4_K block of all zeros and dequantizes it. Confirm output is all 0.0f, not NaN/denormal.

2. **Add debug logging to MUL_MAT_ID**: In `ggml_sycl_mul_mat_id` (ggml-sycl.cpp:3880), log which expert IDs are being dispatched. Confirm expert 128 is never selected.

3. **Compare intermediate tensors**: Run both 128 and 129 models with debug output after each MoE layer. Find the first layer where outputs diverge.

4. **Try with `ngl 0` and single-threaded**: `./llama-server -ngl 0 -t 1 ...` to eliminate any threading issues.

5. **Check scratch buffer allocation**: Search for `n_expert` in buffer sizing code in `ggml.c`, `ggml-cpu/`, `ggml-alloc.c`.

## Key Source Files

| File | What to look at |
|------|----------------|
| `ggml/src/ggml.c` | `ggml_new_tensor_impl` (stride computation), `ggml_nbytes` |
| `ggml/src/gguf.cpp` | `gguf_init_from_file_impl` (tensor loading, offset validation) |
| `ggml/src/ggml-cpu/ops.cpp` | `ggml_compute_forward_mul_mat_id` (CPU MoE dispatch) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:3880` | SYCL `ggml_sycl_mul_mat_id` |
| `src/llama-graph.cpp:1192` | `build_moe_ffn` (MoE graph construction) |
| `src/llama-model.cpp:3839` | QWEN3MOE tensor creation |
| `scripts/pad-experts-gguf.py` | The padding script |

## Environment

```bash
source /home/ryan/llm-stack/env.sglang-xpu.sh
# Worktrees: llama.cpp-stable (flagship), llama.cpp-eptp (EP experiment)
# Build: cd llama.cpp-stable/build-sycl && cmake --build . --target llama-server -j$(nproc)
# Bench: cd scripts && python3 -m bench help
```

## What NOT to Do

- Don't test with REAM-heretic models — they're fundamentally broken (garble on all builds)
- Don't pursue safetensors padding — we want a generic GGUF patcher
- Don't assume it's SYCL-specific — CPU also garbles
- Don't try to fix the unequal EP split in llama.cpp code — padding is the cleaner approach if we can make it work
