# BOOTSTRAP: Expert Parallelism — GLU Split State Bug

## Status
- Expert padding: ✅ solved
- EP AllReduce deferral: ✅ fixed
- EP rotation: ✅ fixed (consistent expert distribution across layers)
- Dense TP on eptp build: ✅ works (20.7 t/s)
- **GLU split state: ❌ THE REMAINING BUG**

## The Bug

GLU op (node 45) shows `axis=MIRRORED` when its inputs (gate MUL_MAT_ID, up MUL_MAT_ID) are both PARTIAL:

```
node[43] op=MUL_MAT_ID  axis=PARTIAL   defer=true   ← gate
node[44] op=MUL_MAT_ID  axis=PARTIAL   defer=true   ← up
node[45] op=GLU          axis=MIRRORED               ← WRONG! Should be PARTIAL
node[46] op=MUL_MAT_ID  axis=PARTIAL   defer=false   ← down → AllReduce at node 62
```

GLU uses `handle_generic(src_split_states, false)` which checks if all sources have the same split state. But `get_split_state` for GLU's sources (the MUL_MAT_ID outputs) doesn't return PARTIAL because the **subgraph sweep doesn't cache computed split states**. When GLU asks "what's my src[0]'s split state?", it recomputes from scratch and the MUL_MAT_ID output falls back to MIRRORED.

## Root Cause: Split State Not Cached

The subgraph boundary loop in `ggml-backend-meta.cpp` (around line 1030) calls `ggml_backend_meta_get_split_state(node)` for each graph node. This function computes the split state recursively from the node's sources. But it does NOT cache the result — each call recomputes.

For GLU at node 45:
1. GLU calls `handle_generic` which calls `get_split_state(src[0])` for the gate MUL_MAT_ID output
2. `get_split_state` for the MUL_MAT_ID output has to trace back to the MUL_MAT_ID op
3. MUL_MAT_ID → `handle_mul_mat_id` → returns PARTIAL (correct)
4. But somewhere in this recursive chain, the resolution fails and returns MIRRORED

The fix needs to either:
- **Cache split states** so GLU can look up its sources' already-computed states
- **Special-case GLU** to propagate PARTIAL from any PARTIAL source
- **Change the graph structure** to avoid GLU between PARTIAL MUL_MAT_IDs

## What Was Already Tried (Didn't Work)

```cpp
// Attempted: force PARTIAL propagation from src[0]
case GGML_OP_GLU: {
    if (src_split_states[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
        split_state = src_split_states[0];
    } else {
        split_state = handle_generic(src_split_states, false);
    }
}
```
This didn't work because `src_split_states[0].axis` is already MIRRORED by the time GLU processes it — the recomputation doesn't return PARTIAL.

## Proposed Fix: Cache Split States in the Subgraph Loop

The subgraph boundary loop already computes split states for every node. Add a `std::unordered_map<ggml_tensor*, ggml_backend_meta_split_state>` to cache results:

```cpp
// In ggml_backend_meta_graph_compute, before the subgraph loop:
std::unordered_map<ggml_tensor*, ggml_backend_meta_split_state> split_state_cache;

// In the loop, after computing split_state for each node:
split_state_cache[node] = split_state;

// In get_split_state (or wherever src split states are resolved):
// Check cache first before recomputing
auto it = split_state_cache.find(tensor);
if (it != split_state_cache.end()) return it->second;
```

The challenge: `get_split_state` is a standalone function, not part of the loop. The cache would need to be passed as context or made a member of the buffer context.

## Alternative: Fix get_split_state for MUL_MAT_ID Outputs

The real question is: why does `get_split_state(MUL_MAT_ID_output)` return MIRRORED instead of PARTIAL?

The function `get_split_state` works by looking at the TENSOR's properties:
- If it's a weight tensor → look up split config by name
- If it's a compute result → look at its op and sources recursively

For MUL_MAT_ID output (the dst tensor), `get_split_state` should call itself recursively on the MUL_MAT_ID op and return PARTIAL. **Debug this recursive chain to find where it fails.**

Add logging:
```cpp
// In get_split_state, at the top:
static int gs_depth = 0;
gs_depth++;
if (gs_depth < 50 && strstr(tensor->name, "moe") != nullptr) {
    fprintf(stderr, "GET_SPLIT_STATE depth=%d op=%s name=%s\n", gs_depth, ggml_op_name(tensor->op), tensor->name);
}
// ... compute split_state ...
if (gs_depth < 50 && strstr(tensor->name, "moe") != nullptr) {
    fprintf(stderr, "  -> axis=%d\n", split_state.axis);
}
gs_depth--;
```

## Files to Edit

| File | Line | What |
|------|------|------|
| `ggml/src/ggml-backend-meta.cpp` ~1030 | Subgraph boundary loop | Add split state cache |
| `ggml/src/ggml-backend-meta.cpp` ~1391 | `handle_generic` | Where GLU resolves its sources |
| `ggml/src/ggml-backend-meta.cpp` ~1485 | `handle_mul_mat_id` | Returns PARTIAL for EP |
| `ggml/src/ggml-backend-meta.cpp` ~620 | `get_split_state` function | Recursive split state resolution |

## Build & Test

```bash
cd /home/ryan/llm-stack/llama.cpp-eptp/build-sycl
source /home/ryan/llm-stack/env.sglang-xpu.sh
cmake --build . --target llama-server -j$(nproc)

pkill -x llama-server; sleep 2
GGML_META_DEBUG_SPLITS=1 GGML_SYCL_DISABLE_GRAPH=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  ./bin/llama-server \
    -m /home/ryan/llm-stack/models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf \
    --split-mode tensor -ngl 99 -np 1 -c 256 --port 18404 --no-warmup \
    > /tmp/ep-debug.log 2>&1 &
for i in $(seq 1 60); do curl -sf http://127.0.0.1:18404/health > /dev/null 2>&1 && break; sleep 3; done

curl -s --max-time 120 -X POST http://127.0.0.1:18404/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":20,"temperature":0}'

# Check GLU split state (should be PARTIAL after fix):
grep "META_SPLITS.*node\[45\]" /tmp/ep-debug.log
# Check expert distribution (should be consistent):
grep "EP META SET" /tmp/ep-debug.log | head -9
```

## Verified Correct (Don't Re-Investigate)
- Expert padding script ✅
- Expert rotation (now disabled for SPLIT_AXIS_2) ✅
- AllReduce deferral (looks ahead for MUL_MAT_ID) ✅
- Weight data integrity ✅
- Expert routing selection ✅
- Pre-zeroing ✅
- Individual expert compute outputs ✅
- Dense TP on eptp build ✅
