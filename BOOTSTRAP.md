# BOOTSTRAP: Expert Parallelism — Fused Aggregation Bug

## Status
- Expert padding: ✅ solved
- EP AllReduce deferral: ✅ fixed
- EP rotation: ✅ fixed
- EP GLU split state: ✅ fixed
- EP MUL_MAT_ID compute: ✅ verified correct per-expert
- Dense TP on eptp build: ✅ works
- **Fused expert aggregation kernel: ❌ THE BUG**

## The Bug

The per-expert matmul outputs are correct (verified). The aggregation step uses a **fused SYCL kernel** (`fused-expert-agg.cpp`) that replaces the MUL(weights) + VIEW + ADD chain. This kernel produces ALL ZEROS on GPU0 despite the input buffer having nonzero data at slots 2 and 6.

### Evidence

```
MUL_MAT_ID writes:
  dev=0 expert=37 dst=0x...a65800 (slot 2, offset +16384) → NONZERO ✅
  dev=0 expert=33 dst=0x...a6d800 (slot 6, offset +49152) → NONZERO ✅

FUSED_AGG reads from: experts=0x...a61800 (same buffer base) ✅
FUSED_AGG writes to:  last_add=0x...1f000000

AllReduce input: GPU0 = ALL ZEROS ❌
```

The fused aggregation kernel reads from the correct buffer but produces zeros. It was likely written for TP mode where all 8 expert slots have nonzero data. In EP mode, only 2/8 slots are nonzero (the rest are pre-zeroed for non-owned experts).

### Likely root cause

The fused kernel in `fused-expert-agg.cpp` might:
1. Skip slots with zero data (assumes TP where all slots are populated)
2. Have a race condition with the pre-zeroing memset
3. Read weights incorrectly in EP mode (weights shape/stride differs)
4. Not handle the case where n_local < n_expert_used

## Fix Approach

**Option A (quick test):** Disable the fused aggregation kernel for EP mode. The fallback MUL + VIEW + ADD chain should work correctly. Search for `try_fused_expert_agg` in `ggml-sycl.cpp` and skip it when EP flag is set.

**Option B (proper fix):** Read `fused-expert-agg.cpp` and fix the kernel to handle sparse expert slots (zeros for non-owned experts in EP mode).

## Key Files

| File | What |
|------|------|
| `ggml/src/ggml-sycl/fused-expert-agg.cpp` | The fused aggregation kernel |
| `ggml/src/ggml-sycl/fused-expert-agg.hpp` | Header |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:4689` | Where MUL dispatches (has debug dumps) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp:4194` | EP MUL_MAT_ID with debug dumps |

## Build & Test

```bash
cd /home/ryan/llm-stack/llama.cpp-eptp/build-sycl
source /home/ryan/llm-stack/env.sglang-xpu.sh
cmake --build . --target llama-server -j$(nproc)

pkill -x llama-server; sleep 2
GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  ./bin/llama-server \
    -m /home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf \
    --split-mode tensor -ngl 99 -np 1 -c 256 --port 18404 --no-warmup

curl -s --max-time 60 http://127.0.0.1:18404/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":20,"temperature":0}'
```

## Verified Correct (Don't Re-Investigate)
- Expert padding ✅, AllReduce deferral ✅, Rotation ✅, GLU split state ✅
- Per-expert matmul outputs ✅ (nonzero at correct slots)
- Buffer pointers ✅ (FUSED_AGG reads same buffer MUL_MAT_ID wrote to)
- Dense TP ✅, get_i_delayed boundary ✅, Weight data ✅, Expert routing ✅
