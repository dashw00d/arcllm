# BOOTSTRAP: Expert Parallelism — Almost There

## Current State

**Expert padding: SOLVED.** Script `scripts/pad-experts-gguf.py` works on any MoE GGUF. Proven on Qwen2MoE (60→61) and Qwen3MoE (128→129) with layer-split.

**EP first bug FIXED.** AllReduce deferral was breaking the MoE block in half — GLU op between gate and up projections couldn't resolve its split state during peek-ahead, causing early AllReduce. Fixed by looking ahead for MUL_MAT_ID ops instead of resolving next-node split state.

**EP second bug: ACTIVE.** Output went from all-zeros (`GGGGG...`) to varied garbled text at 3.74 t/s. Experts ARE computing. Opus agent deployed to find remaining correctness issue.

## What Works

| Config | Speed | Status |
|--------|-------|--------|
| 30B MoE layer-split np=16 (stable build) | 39.4 t/s | ✅ Flagship |
| 30B MoE layer-split np=1 (stable build) | 15 t/s | ✅ Working |
| Dense TP tensor-split (eptp build) | 20.7 t/s | ✅ TP infrastructure works |
| MoE EP tensor-split (eptp build) | 3.74 t/s | ❌ Varied garbled output |

## EP Bug History (Tonight)

1. **AllReduce deferral (FIXED):** The peek-ahead logic called `get_split_state()` on the GLU node whose inputs hadn't been processed yet. GLU returned MIRRORED instead of PARTIAL, causing AllReduce to fire between up and GLU — splitting the MoE block. Fix: look ahead for `MUL_MAT_ID` with `SPLIT_AXIS_2` instead of resolving intermediate split states.

2. **Remaining correctness issue (INVESTIGATING):** After fix #1, expert outputs are no longer zero but still produce garbled text. Possibilities:
   - Deferred-PARTIAL math (`ffn_norm(inpSA)` vs `ffn_norm(ffn_inp)`) — untested, mathematically different
   - Expert index remapping: `local_index = expert_id - expert_offset` might be off
   - AllReduce boundary still not at the right position
   - op_params encoding issue

## EP Code Locations

| File | What |
|------|------|
| `llama.cpp-eptp/ggml/src/ggml-backend-meta.cpp` | EP dispatch, split state, AllReduce deferral (just fixed) |
| `llama.cpp-eptp/ggml/src/ggml-sycl/ggml-sycl.cpp:4194` | SYCL EP-aware MUL_MAT_ID |
| `llama.cpp-eptp/src/models/qwen3moe.cpp` | Modified MoE graph (deferred-PARTIAL) |
| `llama.cpp-eptp/src/llama-model.cpp:127` | SPLIT_AXIS_2 for expert tensors |

## Debug Commands

```bash
# Build
cd /home/ryan/llm-stack/llama.cpp-eptp/build-sycl
source /home/ryan/llm-stack/env.sglang-xpu.sh
cmake --build . --target llama-server -j$(nproc)

# Test EP with debug
pkill -x llama-server; sleep 2
GGML_META_DEBUG_SPLITS=1 GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  ./bin/llama-server \
    -m /home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf \
    --split-mode tensor -ngl 99 -np 1 -c 256 --port 18404 --no-warmup \
    > /tmp/ep-debug.log 2>&1 &

# Query
curl -s -X POST http://127.0.0.1:18404/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":20,"temperature":0}'

# Check split states
grep "META_SPLITS" /tmp/ep-debug.log | head -60

# NOTE: -fa off crashes with tensor-split (V-cache reshape bug, separate issue)
# Flash attention ON works fine
```

## Key Facts

- EP has NEVER produced clean output on any valid model (the 3.95 t/s REAM result was on a broken model)
- Dense TP works perfectly on the eptp build — the bug is MoE-specific
- GPU0 gets expert_offset=0, n_local=42 (128÷3 rotating 42/43/43) — split looks correct
- Expert padding (128→129) is orthogonal to the EP bug — both 128 and 129 garble
- The deferred-PARTIAL changes in qwen3moe.cpp (ffn_norm on inpSA, residual restructure) were NEVER GPU-validated (task #31)

## Debug Methodology: EP printf-based Instrumentation

The EP code runs on 3 GPUs simultaneously. Traditional debuggers don't work well. Use targeted `fprintf(stderr, ...)` instrumentation with build-test-read cycles.

### Step 1: Add debug prints to the SYCL EP path

The MUL_MAT_ID EP code is in `ggml/src/ggml-sycl/ggml-sycl.cpp` around line 4194. Key places to instrument:

**Expert dispatch (which experts are selected per token):**
```cpp
// In the expert loop, after reading expert ID from routing:
static int ep_dbg = 0;
if (ep_dbg < 20) {
    ep_dbg++;
    fprintf(stderr, "EP_DISPATCH dev=%d expert_id=%d offset=%d n_local=%d %s\n",
            ctx.device, expert_id, expert_offset, n_local_experts,
            (expert_id >= expert_offset && expert_id < expert_offset + n_local_experts) ? "PROCESS" : "SKIP");
}
```

**Post-compute output values (are expert outputs nonzero?):**
```cpp
// After the MUL_MAT_ID computation, for down_exps layers:
if (ep_flag && strstr(src0->name, "ffn_down_exps")) {
    stream->wait();
    std::vector<float> buf(256);
    stream->memcpy(buf.data(), dst->data, 256 * sizeof(float)).wait();
    float sum = 0; int nz = 0;
    for (int k = 0; k < 256; k++) { if (buf[k] != 0) nz++; sum += buf[k]; }
    fprintf(stderr, "EP_POST_DOWN dev=%d [%s] first4=[%.4f,%.4f,%.4f,%.4f] nz=%d/256 sum=%.2f\n",
            ctx.device, src0->name, buf[0], buf[1], buf[2], buf[3], nz, sum);
}
```

**AllReduce values (what goes in and comes out):**
```cpp
// In ggml_backend_sycl_allreduce_tensor, after D2H copies:
fprintf(stderr, "AR dev=%d [%s] first4=[%.4f,%.4f,%.4f,%.4f] nz=%d\n", ...);
```

### Step 2: Build and test
```bash
cd /home/ryan/llm-stack/llama.cpp-eptp/build-sycl
source /home/ryan/llm-stack/env.sglang-xpu.sh
cmake --build . --target llama-server -j$(nproc)

pkill -x llama-server; sleep 2
GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  ./bin/llama-server \
    -m /home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf \
    --split-mode tensor -ngl 99 -np 1 -c 256 --port 18404 --no-warmup \
    > /tmp/ep-debug.log 2>&1 &

for i in $(seq 1 60); do curl -sf http://127.0.0.1:18404/health > /dev/null 2>&1 && break; sleep 3; done
curl -s --max-time 120 -X POST http://127.0.0.1:18404/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}'
```

### Step 3: Read the output
```bash
# Expert dispatch — are experts being processed or all skipped?
grep "EP_DISPATCH" /tmp/ep-debug.log | head -30

# Post-compute — are outputs nonzero?
grep "EP_POST_DOWN" /tmp/ep-debug.log | head -30

# AllReduce — are partial results being summed correctly?
grep "AR " /tmp/ep-debug.log | head -20

# Split state structure — correct subgraph boundaries?
grep "META_SPLITS" /tmp/ep-debug.log | head -60
```

### Step 4: Interpret
- All SKIP, no PROCESS → expert_offset or filtering check is wrong
- PROCESS but output all zeros → pre-zeroing overwrites results, or local_index remapping is wrong
- Output nonzero per-GPU but AllReduce produces garbage → AllReduce implementation bug
- Wrong number of subgraphs → AllReduce deferral still incorrect

### Important notes
- Use `static int count` guards to limit output (hot paths execute thousands of times)
- Always `stream->wait()` before reading GPU memory with `stream->memcpy(...).wait()`
- The `ctx.device` field tells you which GPU (0, 1, 2)
- Expert tensors have names like `blk.0.ffn_gate_exps.weight` — parse layer from name
- Don't forget to strip debug prints before benchmarking (they kill performance)

## What NOT to Do

- Don't test with REAM models — broken
- Don't use `-fa off` with tensor-split — crashes (separate bug)
- Don't assume the deferred-PARTIAL math is correct — it hasn't been validated
- Don't look at expert padding as the cause — proven working on layer-split
