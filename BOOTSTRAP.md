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

Script: `scripts/pad-experts-gguf.py` — binary-level GGUF patcher.

## Current Status: BUG NOT IN SCRIPT

**IMPORTANT**: The padding script was already correct. The Q4_K block size in the script is 144 bytes (confirmed against `ggml-common.h`):
```python
12: (256, 144),    # Q4_K  ← CORRECT
```

The bug is NOT in the padding script - the padded model still produces garbage.

## What We Verified (All Correct)

### Data integrity
- **387 non-expert tensors**: byte-identical between original and padded ✅
- **144 expert FFN tensors**: first 128 experts byte-identical ✅  
- **48 router gate tensors**: experts 0-127 have original weights, expert 128 = all -1e9 ✅
- **Router gate padding**: Verified all -1e9 for expert 128 across all 48 layers ✅
- **FFN expert padding**: Verified all zero bytes for expert 128 (Q4_K blocks) ✅
- **Q4_K zero block output**: Verified `d=0, min=0` → all dequantized values are 0 ✅

### GGUF format
- **Tensor offsets**: GGUF C reader validates stored offsets against recomputed sizes at load time. ✅
- **Strides (nb[])**: `nb[2] = nb[1] * ne[1]` — independent of `ne[2]`. ✅
- **Expert_count metadata**: correctly reads 129 from the padded file ✅
- **Tensor shapes**: all expert tensors show ne[2]=129 in the loader ✅

### Math
- **Softmax**: `exp(-1e9) = 0.0` in float32. ✅
- **Top-k selection**: identical indices for 128 vs 129 experts ✅
- **Argsort stride**: `nb[1] = 129*4 = 516` ✅

## Test Results

**Original model (128 experts)**:
```
reasoning: '<think>\nOkay, the user is asking "What is 2+2?". Let me check that. So, it\'s a simple math problem'
```
→ Clean, coherent output ✅

**Padded model (129 experts)**:
```
22222
is is is is is
the the the the the
```
→ **Repetitive output, NOT random garbling** ❌

The pattern is distinctly different from random garbage — the model outputs the same token repeatedly. This suggests the logits are collapsing to identical values for all positions (not randomness), or FFN output is zero/constant for all experts.

## Root Cause Analysis

The issue is NOT (all verified):
- ❌ Q4_K block size (144 bytes correct)
- ❌ Tensor offsets (GGUF loader validates cleanly)
- ❌ Router gate -1e9 padding (mathematically correct, verified all -1e9)
- ❌ FFN expert zero-padding (verified all zeros, Q4_K zeros produce 0 output)
- ❌ SYCL kernel (happens on CPU too)
- ❌ Power-of-2 template (256 experts also fails)
- ❌ Model type detection (Qwen3MoE type determined by n_layer=48)
- ❌ `cur_experts[LLAMA_MAX_EXPERTS]` stack array (512 pointers only)
- ❌ `ggml_mul_mat_id` scratch buffer (dynamic sizing)
- ❌ `ggml_view_2d` in expert aggregation (stride independent of ne[2])
- ❌ `ggml_argsort_top_k` (uses dynamic nb[1])
- ❌ `ggml_softmax` (generic implementation)

**Unknown**: The padding is byte-perfect yet output is repetitive. The bug must be in how llama.cpp's MoE routing code handles a non-128 expert count somewhere that's not obvious from static code analysis. Expert profiling (`GGML_SYCL_PROFILE_EXPERTS=1`) is the next debug step.

## How to Debug

### Quick test
```bash
# Regenerate padded model
python3 scripts/pad-experts-gguf.py \
  models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf \
  models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m-129experts.gguf

# Test padded model (garbled)
source env.sglang-xpu.sh
./llama.cpp-stable/build-sycl/bin/llama-server \
  -m models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m-129experts.gguf \
  --split-mode layer -ngl 99 -np 1 -c 512 --port 18404 --no-warmup -fa off
```

### Expert profiling (SYCL)
```bash
# Enable expert profiling to see which experts are selected
GGML_SYCL_PROFILE_EXPERTS=1 ./llama.cpp-stable/build-sycl/bin/llama-server \
  -m models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m-129experts.gguf \
  --split-mode layer -ngl 99 -np 1 -c 512 --port 18404 --no-warmup -fa off
```

### Expert profiling output interpretation
- If expert 128 is selected: the -1e9 padding isn't working (unlikely given data verification)
- If only experts 0-127 are selected but output is wrong: FFN computation issue
- If no experts (all zeros): routing collapsed

**Note**: Expert profiling requires a full server run (not raw curl) since SYCL device access fails in non-interactive mode. Use the bench framework or a long-running server instance.

## Key Source Files

| File | What to look at |
|------|----------------|
| `ggml/src/ggml.c` | `ggml_new_tensor_impl` (stride computation) |
| `ggml/src/ggml-cpu/ops.cpp` | `ggml_compute_forward_mul_mat_id` (CPU MoE dispatch) |
| `src/llama-graph.cpp:1192` | `build_moe_ffn` (MoE graph construction) |
| `src/llama-graph.h:717` | `n_expert` member - how it's used |

## What NOT to Do

- Don't test with REAM-heretic models — fundamentally broken
- Don't pursue safetensors padding — GGUF approach is cleaner if we can fix it
- Don't assume it's SYCL-specific — happens on CPU too
- Don't waste time on padding script bugs — verified correct (data integrity passes)
- Don't waste time on 128-hardcoded checks — verified absent or irrelevant

## Critical Finding: Not About Expert Weights

Tested duplicating expert 0 as expert 128 (valid weights, not zeros). **Still garbled** — different pattern (LaTeX-like gibberish instead of repetition), but still broken. This proves:

- The fake expert's weights (zero or valid) are **irrelevant** — the router never selects expert 128
- The corruption comes from having `n_expert=129` in the model structure itself
- Something in the compute graph, tensor allocation, or MoE dispatch treats n_expert != 128 incorrectly
- This is a **structural/code bug in llama.cpp**, not a data bug

With zero padding: repetitive output ("22222", "is is is")
With expert-0 duplication: varied gibberish ("equiv[\n{equiv(\\]")
Both wrong. Both caused by n_expert=129.

## Remaining Hypotheses (Post Zero-vs-Duplicate Test)

1. **ggml_compute_forward_mul_mat_id scratch buffer**: The CPU `mul_mat_id` allocates `matrix_rows` as `n_as * ids->ne[0] * ids->ne[1] * sizeof(mmid_row_mapping)`. With n_as=129, this is slightly larger. If the wdata buffer wasn't sized for 129, it could overflow.

2. **Graph scheduler buffer estimation**: `ggml_graph_compute` pre-allocates work buffers. If the estimation uses n_expert somewhere, a wrong size could cause silent memory corruption.

3. **Qwen3MoE model code implicit assumptions**: Something in the Qwen3MoE model builder (`llama-model.cpp:3839`) might implicitly depend on n_expert matching the original model config in a way that's not obvious from tensor shapes.

4. **The `n_ff_exp` computation**: Line 3882: `n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used`. If `n_ff_exp` is not set in metadata and computed from `n_ff / n_expert_used`, changing expert count could affect this. But we didn't change n_expert_used (still 8), so this should be fine. Verify: is `n_ff_exp` set in the GGUF metadata or computed?

5. **Work buffer sizing in ggml-alloc**: The graph allocator sizes temporary buffers for each tensor. If any MoE-related tensor's size computation uses n_expert (the weight dimension) instead of n_expert_used (the dispatch dimension), the buffer could be wrong.

## Next Diagnostic Steps

1. **Add printf to CPU mul_mat_id**: In `ggml-cpu.c:1503`, print `n_as`, `n_ids`, buffer sizes. Compare 128 vs 129.

2. **Check wdata sizing**: In `ggml_graph_plan` or `ggml_graph_compute`, find where wdata/work buffer is sized. Search for `MUL_MAT_ID` work size estimation.

3. **Check n_ff_exp**: Verify the GGUF metadata has `qwen3moe.expert_feed_forward_length = 768` and it's being read correctly for both 128 and 129 expert models.

4. **Binary comparison approach**: Run both models to generate 1 token, dump ALL intermediate tensor values after each layer, diff to find first divergence.

## Alternative Approaches to Consider

1. **Fix llama.cpp code** — instead of padding, implement proper unequal expert split (43/43/42) in the meta backend
2. **Pad at safetensors level** — download FP16 safetensors, pad, re-quantize with llama.cpp's convert script (guaranteed correct metadata)
3. **Find the llama.cpp bug** — the fact that n_expert≠128 breaks inference is likely a bug worth fixing upstream
4. **Test with a smaller MoE model** — try padding Qwen1.5-MoE-A2.7B (60 experts) to 63 to see if the bug reproduces with any non-native expert count
