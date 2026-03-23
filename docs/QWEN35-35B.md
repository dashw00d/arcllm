# Qwen3.5-35B-A3B on llama.cpp

## Overview

Qwen3.5-35B-A3B (256 experts, 35B-A3B MoE) successfully runs on llama.cpp-stable with multi-GPU support via layer-split.

## Status

| Config | Status | Notes |
|--------|--------|-------|
| Single GPU | ✅ Works | ~7 t/s |
| Multi-GPU (np=4, layer-split) | ✅ Works | ~7 t/s |
| Tensor-split | ❌ Not supported | "not implemented for architecture 'qwen35moe'" |
| fused_gdn_ar=true | ❌ Hangs | Must be disabled |

## Working Configuration

```bash
cd /home/ryan/llm-stack/llama.cpp-stable
source ../env.sglang-xpu.sh
cd build-sycl && cmake --build . --target llama-server -j$(nproc)

./bin/llama-server \
  --model "/home/ryan/llm-stack/models/Qwen/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-GGUF/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf" \
  --host 127.0.0.1 --port 8400 \
  --split-mode layer -ngl 999 -c 8192 -np 4 -fa off --no-warmup
```

## Critical Fix: fused_gdn_ar=false

**The model HANGING after prompt processing was caused by `fused_gdn_ar=true`.**

### Root Cause

The model was quantized with different tensor naming than stable expects:
- Model tensors: `state_predelta-*`, `attn_gated-*`
- Stable expects: `__fgdnar__-*` (fused Gated Delta Net naming)

This mismatch causes the fused kernel path to hang after prompt processing.

### Fix Applied

In `src/llama-context.cpp:153`:

```cpp
// BEFORE (hangs):
cparams.fused_gdn_ar = true;

// AFTER (works):
cparams.fused_gdn_ar = false;
```

This forces the non-fused element-wise Gated Delta Net path, which works correctly.

### Why It Works

The non-fused path performs Gated Delta Net attention using individual operations (exp, mul, sum, etc.) that work regardless of tensor naming. The fused kernel path expects specific tensor names that don't match this model.

## Performance

### Timing (np=4, layer-split, 8k context)

| Metric | Value |
|--------|-------|
| Prompt eval | ~97 ms/token (10.35 t/s) |
| Generation | ~143 ms/token (6.98 t/s) |
| 500 tokens | ~72 seconds |

### Memory Usage

```
llama_params_fit: projected memory use with initial parameters [MiB]:
  - SYCL0 (Intel(R) Arc(TM) A770 Graphics): 15473 total, 9539 used, 5933 free
  - SYCL1 (Intel(R) Arc(TM) A770 Graphics): 15473 total, 9403 used, 6070 free
  - SYCL2 (Intel(R) Arc(TM) A770 Graphics): 15473 total, 9130 used, 6343 free
  - Total projected: 28073 MiB vs 46420 MiB free
```

### Compute Buffers (layer-split np=4)

```
sched_reserve:
  - SYCL0 compute buffer size = 2502.85 MiB
  - SYCL1 compute buffer size = 2613.54 MiB
  - SYCL2 compute buffer size = 2608.86 MiB
  - SYCL_Host compute buffer size = 5552.72 MiB
```

### KV Cache (layer-split np=4, 2048 context)

```
llama_kv_cache:
  - SYCL0 KV buffer size = 48.00 MiB
  - SYCL1 KV buffer size = 64.00 MiB
  - SYCL2 KV buffer size = 48.00 MiB
  - Total: 160.00 MiB (2048 cells, 10 layers, 4/4 seqs)

llama_memory_recurrent:
  - SYCL0 RS buffer size = 92.12 MiB
  - SYCL1 RS buffer size = 83.75 MiB
  - SYCL2 RS buffer size = 75.38 MiB
  - Total: 251.25 MiB (4 cells, 40 layers, 4 seqs)
```

## Prompt Cache Behavior

**The prompt cache does NOT work for this model.**

```
slot update_slots: forcing full prompt re-processing due to lack of cache data
(likely due to SWA or hybrid/recurrent memory)
```

This is because Qwen3.5-35B uses Sliding Window Attention (SWA) + DeltaNet recurrent states. The hybrid architecture prevents KV cache reuse.

### Implications

- Every request reprocesses the system prompt (~50-60 tokens for 200 words)
- ~5-7 seconds overhead per request for prompt processing
- No benefit from repeated system prompts

## Architecture Notes

### Model Specs

- Architecture: `qwen35moe`
- Experts: 256 total, 8 used per token
- Context length: 262144 (train), 8192 (default)
- Attention: Gated Delta Net (SSM-like)
- Rope type: 40 (linear scaling)

### Tensor Naming

The model uses custom tensor names that differ from llama.cpp-stable's expectations:

| Actual Tensor | Used For |
|--------------|----------|
| `state_predelta-*` | Recurrent state for DeltaNet |
| `attn_gated-*` | Gated normalization |
| `q_conv_predelta-*` | Q after convolution |
| `ffn_shexp_gated-*` | Shared expert gating |

| Stable Expects | Purpose |
|---------------|---------|
| `__fgdnar__-*` | Fused Delta Net autoregressive |
| `__fgdnch__-*` | Fused Delta Net chunked |

## Comparison with Other Models

| Model | Experts | Split Mode | Performance |
|-------|---------|------------|-------------|
| Qwen3-30B-A3B (abliterated) | 128 | tensor-split np=16 | 25.7 t/s |
| Qwen3.5-35B | 256 | layer-split np=4 | ~7 t/s |
| Qwen3-32B (dense) | N/A | tensor-split np=16 | 21.7 t/s |

The MoE models are slower than dense due to expert routing overhead and the lack of FUSED_MMQ in stable.

## Debugging Tips

### Check fused_gdn_ar status

Look for these log messages:
- `sched_reserve: resolving fused Gated Delta Net support:` — if present, fused path is being checked
- `sched_reserve: fused Gated Delta Net (autoregressive) enabled` — fused path active (may hang)
- No fused GDN messages — non-fused path used (should work)

### Kernel build errors

If you see IGC errors during kernel compilation:
```
IGC: Internal Compiler Error: Termination request sent to the program
Build program log for 'Intel(R) Arc(TM) A770 Graphics':
Exception caught at ggml-sycl.cpp, line 3232
```

This is likely a WARP_SIZE or subgroup size mismatch. Check `ggml/src/ggml-sycl/presets.hpp` for `WARP_SIZE` definition.

### SYCL debug logging

```bash
GGML_SYCL_DEBUG=1 ./bin/llama-server [args]
```

Shows tensor initialization and kernel dispatch.

## Files Modified

### llama-context.cpp

```cpp
// Line 153 - disabled fused_gdn_ar to fix hang
cparams.fused_gdn_ar = false;
cparams.fused_gdn_ch = false; // TODO: implement
```

## Related Files

- `src/models/delta-net-base.cpp` — DeltaNet implementation (both fused and non-fused paths)
- `src/models/qwen35moe.cpp` — Qwen3.5 MoE architecture
- `ggml/src/ggml-sycl/gated_delta_net.cpp` — Fused kernel (ported from master)
- `ggml/src/ggml-sycl/gated_delta_net.hpp` — Kernel header

## TODO

1. **Investigate fused_gdn_ar hang** — kernel causes hang after prompt processing; non-fused path works
2. **Implement fused_gdn_ch** — chunked variant not yet implemented in stable
3. **Tensor naming alignment** — could rename model tensors to match stable expectations
4. **Expert padding for tensor-split** — 256 experts doesn't divide evenly for EP
