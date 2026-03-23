# Minimax Session Log

## Date: 2026-03-21

## Context
User has 3x Intel Arc A770 (48GB VRAM), running Qwen3-30B-A3B MoE (abliterated, Q4_K_M) with llama.cpp SYCL.
Goal: Get the 35b-a3b model working and fix MoE bugs (context bugs, L0 corruption, etc.)

## L0 Queue Corruption — Investigation & Fix Attempt

### Root Cause (Previously Known)
Level Zero runtime overlaps kernel execution within in-order SYCL queues.
With 2+ parallel slots generating deeply (~500+ tokens), Q8_1 activation buffers get corrupted.
Trigger is **concurrent compute depth**, not just sequence length.

### Fix Attempted: GGML_SYCL_SYNC_AFTER_QUANT

Added new env var `GGML_SYCL_SYNC_AFTER_QUANT` that adds `stream->wait()` after Q8_1 quantization kernel.

**Changes made to ggml-sycl.cpp:**
1. Line 94: Added `int g_ggml_sycl_sync_after_quant = 0;`
2. Line 291: Added `g_ggml_sycl_sync_after_quant = get_sycl_env("GGML_SYCL_SYNC_AFTER_QUANT", 0);`
3. Line 378: Added logging for the new env var
4. Line 3837-3838: Added sync after quantize call in init section

### Build: SUCCESS

### Test Results

| Config | Result | t/s | Notes |
|--------|--------|-----|-------|
| moefrontier.np16 baseline | 16/16 3200 tok | **25.2-25.8** | No sync, stable |
| moefrontier.np16 SYNC_AFTER_QUANT=1 | 16/16 3200 tok | **25.3** | ~2% slower |
| mt=800 no thinking baseline | 16/16 3696 tok | **27.6** | Passes |
| mt=600 no thinking baseline | 16/16 3696 tok | **26.1** | Passes |

### Critical Finding: `--reasoning-budget` Argument

**Discovery:** Server crashes with `--reasoning-budget` argument error. The llama-server
in stable build expects only `-1` (unlimited) or `0` (disabled), NOT positive integers.

```bash
error: invalid value
usage: --reasoning-budget N  (only -1 or 0)
```

### Thinking Mode Tests (with reasoning_budget=-1)

**Key Finding: Qwen3-30B-A3B MoE is STABLE with thinking up to mt=500 at np=16!**

| Config | Result | t/s | Notes |
|--------|--------|-----|-------|
| dense q4km np=1 mt=200 | 1/1 200 tok | **6.3** | Works |
| MoE np=1 mt=200 | 1/1 200 tok | **12.6** | Works |
| MoE np=2 mt=200 | 2/2 400 tok | **8.9** | Works |
| MoE np=4 mt=200 | 4/4 800 tok | **12.0** | Works |
| MoE np=16 mt=200 | 16/16 3200 tok | **26.4** | Works |
| MoE np=16 mt=300 | 16/16 3504 tok | **25.3** | Works |
| MoE np=16 mt=500 | 16/16 3504 tok | **25.8** | Works |

### Conclusion on SYNC_AFTER_QUANT

SYNC_AFTER_QUANT alone does NOT fix any current problem.
- Baseline throughput unaffected (25.8 vs 25.3 t/s)
- Qwen3-30B-A3B is more stable than GLM-4.7 (doesn't crash at high token counts)

## EP (Expert Parallelism) Investigation

### Key Clarification: Layer-Split vs Expert-Parallelism

**Current setup:** `--split-mode layer` = each GPU computes a SUBSET OF LAYERS
- All experts computed on each GPU (for layers that GPU owns)
- NOT expert parallelism

**EP experiment:** Each GPU computes a SUBSET OF EXPERTS for ALL layers
- Requires `--split-mode tensor`
- On `llama.cpp-eptp` branch

### EP Architecture

In EP mode:
1. Each GPU pre-zeros non-local expert output slots (ggml-sycl.cpp:4218-4224)
2. Each GPU computes only its LOCAL experts via MUL_MAT_ID
3. Non-local slots remain zeros
4. AllReduce SUM across GPUs combines partial results

### Fused Expert Aggregation Kernel

Location: `llama.cpp-eptp/ggml/src/ggml-sycl/fused-expert-agg.cpp`

**Current state:** Kernel SKIPS for EP mode (returns 0):
```cpp
// Skip fusion in EP mode — the kernel produces zeros when expert slots are sparse.
if (t->op == GGML_OP_MUL_MAT_ID && t->op_params[2] > 0) {
    return 0;  // Skip fusion
}
```

**BOOTSTRAP says:** "fused-expert-agg.cpp produces zeros despite having 2 nonzero expert slots"

### Verified Correct (per BOOTSTRAP)
- Weight data ✅
- Expert routing ✅
- Pre-zeroing ✅
- Per-expert matmul outputs (per-slot) ✅
- Buffer pointers ✅
- Dense TP on eptp build ✅

**But:** AllReduce input shows GPU0 = all zeros despite having 2 nonzero expert slots

### The Real Issue

The fused kernel is SKIPPED for EP, but the NON-FUSED ADD chain ALSO has issues.
In EP mode with fusion disabled:
1. MUL_MAT_ID computes each expert's FFN → correct
2. MUL node multiplies experts by weights → ffn_moe_weighted
3. VIEW nodes split into per-expert views
4. ADD chain sums them up — BUT each GPU only has LOCAL expert data

**Aggregation happens at higher level** (ggml-backend-meta.cpp AllReduce), but only
after each GPU finishes its entire subgraph. The issue is the data is correct at
MUL_MAT_ID output but zeros at AllReduce input.

## Summary

1. **SYNC_AFTER_QUANT=1**: No throughput benefit (~2% slower)
2. **Thinking mode**: Works on Qwen3-30B-A3B up to mt=500 np=16 (stable)
3. **Corruption threshold**: Higher than expected for this model/config
4. **The SYNC_AFTER_QUANT fix doesn't hurt**, but doesn't solve current problems

**Next priority: EP needs clean rebase + debug the non-fused aggregation path**

## Qwen3.5 35B — WORKS on Fresh Master (Single GPU)

### Model
- Path: `models/Qwen/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-GGUF/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`
- Size: 19.71 GiB
- Architecture: `qwen35moe` (256 experts, 8 active/token, Gated Delta Net attention)

### What We Tried

1. **qwen35 branch (llama.cpp-qwen35):** Manually added `gated_delta_net.hpp` and `gated_delta_net.cpp` from master. Build succeeded but server crashed during model loading (`common_init_result: fitting params to device memory`).

2. **Fresh master clone (llama.cpp-master):** Cloned clean master into `llama.cpp-master/`. Master has native `qwen35moe` support.

### Working Config (Single GPU)

```bash
cd /home/ryan/llm-stack/llama.cpp-master
source ../env.sglang-xpu.sh
mkdir -p build-sycl && cd build-sycl
cmake .. -DGGML_SYCL=ON -DGGML_SYCL_FAT=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build . --target llama-server -j$(nproc)

# Run with:
./bin/llama-server \
  -m models/Qwen/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-GGUF/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  -fa off -np 1 -ngl 99 -t 32 --host 0.0.0.0 --port 8080 \
  -fit off -c 8192
```

**Key flags:** `-fit off` required for model to load. `-np 1` (single GPU) required — multi-GPU crashes.

### Test Results

| Test | Tokens | Result |
|------|--------|--------|
| "What is 2+2?" | 147 | Coherent reasoning + "4" |
| Photosynthesis explanation | 600 | Full explanation with light reactions & Calvin cycle |
| Robot story | 600 | Generated coherent narrative |
| France capital | 127 | Direct answer "Paris" |

### Gated Delta Net Status

**IT WORKS!** Master branch has proper SYCL kernel support:
```
sched_reserve: resolving fused Gated Delta Net support:
sched_reserve: fused Gated Delta Net (autoregressive) enabled
sched_reserve: fused Gated Delta Net (chunked) enabled
```

### Multi-GPU Issue

- `np=16`, `np=3`: Crashes during initialization after model metadata dump
- Single GPU (`np=1`): Works reliably at ~7 t/s
- Likely needs expert padding (256 experts ÷ 3 GPUs doesn't divide cleanly) + TP work

### Conclusion

**Qwen3.5 35B WORKS on fresh master, single GPU.** Multi-GPU needs the expert padding/TP fixes from stable build.

## Next Steps

1. **Port expert padding + TP logic from stable to master** — enables multi-GPU for Qwen3.5
2. **EP needs clean rebase + debug the non-fused aggregation path**
3. **Qwen3.5 now WORKS** — single GPU at ~7 t/s, needs multi-GPU enablement

## Files Modified

- `/home/ryan/llm-stack/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp`
  - Added `g_ggml_sycl_sync_after_quant` global var (line 94)
  - Added env parsing for `GGML_SYCL_SYNC_AFTER_QUANT` (line 291)
  - Added logging for the new env var (line 378)
  - Added `stream->wait()` after Q8_1 quantization in init section (lines 3837-3838)

- `/home/ryan/llm-stack/llama.cpp-master/` — Fresh clone of upstream master

## Commands Run

```bash
# Clone fresh master
cd /home/ryan/llm-stack
git clone --depth=1 https://github.com/ggml-org/llama.cpp.git llama.cpp-master

# Build master
cd llama.cpp-master
source ../env.sglang-xpu.sh
mkdir -p build-sycl && cd build-sycl
cmake .. -DGGML_SYCL=ON -DGGML_SYCL_FAT=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build . --target llama-server -j$(nproc)

# Run (single GPU)
./bin/llama-server \
  -m models/Qwen/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-GGUF/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  -fa off -np 1 -ngl 99 -t 32 --host 0.0.0.0 --port 8080 \
  -fit off -c 8192

# Test
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":150,"stream":false}'
```
