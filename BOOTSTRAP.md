# BOOTSTRAP: Expert Parallelism — Garbled Output

## Status
- Expert padding: ✅ solved
- EP AllReduce deferral: ✅ fixed
- EP rotation: ✅ fixed
- EP GLU split state: ✅ fixed
- EP MUL_MAT_ID compute: ✅ verified correct per-expert
- Dense TP on eptp build: ✅ works (Qwen3-32B coherent output)
- Fused expert aggregation kernel: ✅ bypassed for EP (was producing zeros)
- Fused topk-select kernel: ✅ ruled out (disabled, still garbled)
- Fused add-rmsnorm kernel: ✅ ruled out (disabled, still garbled)
- Per-layer AllReduce sums: ✅ verified correct (GPU0+GPU1+GPU2 sums check out)
- Per-layer MoE aggregation (MUL+VIEW+ADD): ✅ verified nonzero output per GPU
- **Overall EP output: ❌ GARBLED — not a SYCL kernel bug, likely meta backend graph splitting**

## What Was Done This Session

### Fused kernel bypass (committed in working tree)
- `fused-expert-agg.cpp`: Added EP detection — walks up src chain to find MUL_MAT_ID, checks `op_params[2] > 0`. Returns 0 (skip fusion) in EP mode.
- `fused-topk-select.cpp`: Added `return 0;` at top (disabled for testing)
- `fused-add-rmsnorm.cpp`: Added `return 0;` at top (disabled for testing)
- **Result:** Disabling ALL three custom fused kernels still produces garbled output. The bug is NOT in any fused kernel.

### Extensive debug instrumentation added to ggml-sycl.cpp
- `EP_MUL_IN` / `EP_MUL_OUT`: Per-slot dumps of MUL (ffn_moe_weighted) input/output
- `EP_ADD_PRE` / `EP_ADD_POST`: ADD chain input/output for expert aggregation
- `EP_PATH`: Logs whether MUL_MAT_ID takes DECODE (ne12=1) or BATCH (ne12>1) path
- `EP_BATCH_EXPERTS`: Prints which experts were dispatched in batch path
- `EP_BATCH_PROBE`: Post-completion buffer probe after batch path stream->wait()
- `WATCH`: Buffer watchpoint — arms on down_exps buffer, checks every subsequent dispatch
- `stream->wait()` added after batch path loop (didn't help — not a queue ordering issue)

### Key Finding: The bug is NOT in the SYCL kernels

Every individual component verified correct:
1. MUL_MAT_ID writes nonzero data to correct slots ✅
2. Routing dispatches correct experts to correct GPUs ✅
3. AllReduce produces correct sums across GPUs ✅
4. Dense TP on same build produces coherent output ✅
5. All three custom fused kernels disabled — still garbled ✅

## The Real Bug (Narrowed Down)

The garbled output persists with ALL custom SYCL code disabled. The issue is in the **meta backend** (`ggml-backend-meta.cpp`) — specifically how it:
1. Determines split states (PARTIAL vs MIRRORED) for MoE graph nodes
2. Places AllReduce boundaries via `get_i_delayed()` (line 961-1025)
3. Constructs per-GPU subgraphs and dispatches them

### Specific area to investigate: `get_i_delayed` + subgraph construction

The `get_i_delayed` function (line 961) walks the MUL → VIEW×N → ADD×(N-1) aggregation chain to extend the AllReduce boundary past it. It also handles ADD_ID (bias) and MUL (weighting). The `defer_ep_allreduce` logic (line 1055-1080) looks ahead for more MUL_MAT_ID ops to merge gate/up/down into one subgraph.

**Debug approach:** Run with `GGML_META_DEBUG_SPLITS=1` to print split state for every node. Compare the split state trace between:
- Dense TP (works) — should show PARTIAL → AllReduce → MIRRORED pattern
- MoE EP (broken) — check if any node has wrong split state

```bash
GGML_META_DEBUG_SPLITS=1 GGML_SYCL_DISABLE_GRAPH=1 ... ./bin/llama-server ...
```

### What to look for
- A node marked MIRRORED that should be PARTIAL (would cause data broadcasting, overwriting partial results)
- A node marked PARTIAL that should be MIRRORED (would cause AllReduce on wrong data)
- AllReduce boundary at wrong position (e.g., between MUL_MAT_ID and MUL instead of after ADD chain)
- Subgraph containing nodes from different layers (buffer reuse + async could corrupt)

## Key Files

| File | What |
|------|------|
| `ggml/src/ggml-backend-meta.cpp:961` | `get_i_delayed` — AllReduce boundary extension |
| `ggml/src/ggml-backend-meta.cpp:1055` | `defer_ep_allreduce` — MoE subgraph merging |
| `ggml/src/ggml-backend-meta.cpp:1087` | Subgraph boundary decision |
| `ggml/src/ggml-backend-meta.cpp:1288` | Subgraph dispatch loop (async per-GPU) |
| `ggml/src/ggml-backend-meta.cpp:912` | EP op_params encoding for per-GPU MUL_MAT_ID |
| `ggml/src/ggml-sycl/fused-expert-agg.cpp` | Bypassed for EP (EP detection added) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Heavy debug instrumentation (can be cleaned up) |

## Build & Test

```bash
cd /home/ryan/llm-stack/llama.cpp-eptp/build-sycl
source /home/ryan/llm-stack/env.sglang-xpu.sh
cmake --build . --target llama-server -j$(nproc)

pkill -x llama-server; sleep 2
GGML_META_DEBUG_SPLITS=1 GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
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
- Dense TP ✅, Weight data ✅, Expert routing ✅
- All three custom fused SYCL kernels ✅ (disabled all three, still garbled)
- AllReduce sums ✅ (verified GPU0+GPU1+GPU2 partial sums add correctly)
- Queue ordering ✅ (stream->wait() after batch path didn't help)
- Individual MoE layer outputs ✅ (ADD chain produces nonzero, AllReduce sums correctly)
