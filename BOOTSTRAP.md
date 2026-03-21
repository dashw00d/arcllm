# BOOTSTRAP: Expert Parallelism — Aggregation Chain Bug

## Status
- Expert padding: ✅ solved
- EP AllReduce deferral: ✅ fixed
- EP rotation: ✅ fixed
- EP GLU split state: ✅ fixed (assume_sync=false for GLU sources)
- Dense TP on eptp build: ✅ works (20.7 t/s)
- EP MUL_MAT_ID compute: ✅ verified correct (matmul outputs nonzero per-expert)
- **EP aggregation chain: ❌ THE REMAINING BUG**

## The Bug

The per-expert matmul outputs are **correct** on all GPUs. But the aggregation chain (MUL weights → VIEW → ADD) that combines 8 expert outputs into a single [2048] vector **zeroes GPU0's contribution**.

### Evidence

**Raw MUL_MAT_ID output (CORRECT):**
```
GPU0 blk.0 down_exps:
  slot[0] nz=0     (expert 60 → GPU1, correctly zero)
  slot[2] nz=2048  (expert 37 → GPU0, correctly nonzero!)  ← COMPUTED
  slot[6] nz=2048  (expert 33 → GPU0, correctly nonzero!)  ← COMPUTED
  (other slots correctly zero)

GPU1 blk.0 down_exps:
  slot[0] nz=2048  (expert 60 → GPU1)
  slot[3] nz=2048  (expert 78 → GPU1)
  slot[4] nz=2048  (expert 82 → GPU1)
  slot[7] nz=2048  (expert 56 → GPU1)
```

**After aggregation (AllReduce input — WRONG):**
```
GPU0: nz=0/4096, sum=0.00, absmax=0.0000   ← ALL ZEROS despite having 2 nonzero slots!
GPU1: nz=4096/4096, sum=-3.83, absmax=0.4947
GPU2: nz=4096/4096, sum=-4.42, absmax=0.9897
```

### What this means

The matmul writes correct values to slots 2 and 6 in GPU0's output buffer. Then the aggregation ops (MUL by router weights, VIEW per slot, sequential ADD) run and produce a zero result. Either:

1. **The MUL/VIEW/ADD ops on GPU0 read from a different buffer** than what the matmul wrote to (meta backend buffer allocation issue)
2. **The MUL zeros the data** — router weights on GPU0 might be wrong (but they're MIRRORED, same on all GPUs)
3. **The VIEW offsets are wrong** on GPU0, reading from zeroed slots instead of slots 2 and 6
4. **Buffer reuse** — a subsequent op's pre-zeroing overwrites the matmul output before aggregation reads it

### Most likely: buffer pointer mismatch

The subgraph contains both the MUL_MAT_ID and the aggregation chain. The meta backend allocates per-GPU buffers for all tensors in the subgraph. If the intermediate tensor (down output) is allocated at a different address than where the matmul writes (`dst->data`), the aggregation reads stale/zeroed memory.

## Debug: Trace the aggregation chain

Add dumps at each step of the aggregation on GPU0:

```cpp
// In the MoE aggregation code (build_moe_ffn in llama-graph.cpp):
// After: experts = ggml_mul(ctx0, experts, weights);
// The result 'experts' should still have nonzero at slots 2 and 6

// Key: verify that experts->data == the MUL_MAT_ID dst->data on GPU0
// If they differ, the meta backend allocated a new buffer for the MUL result
```

Or add SYCL-side dump in the MUL kernel when one input is `ffn_moe_down`:
```cpp
// Check if MUL(down_output, weights) reads from the correct buffer
```

## Fixed Bugs (Don't Re-Investigate)

1. **AllReduce deferral** — GLU between gate/up couldn't resolve split state during peek-ahead. Fixed: look ahead for MUL_MAT_ID with SPLIT_AXIS_2 instead.

2. **Rotation** — `il % n_devices` rotated expert distribution per layer, making expert_offset inconsistent. Fixed: `effective_rotation=0` for SPLIT_AXIS_2.

3. **GLU split state** — GLU returned MIRRORED because `handle_mul_mat_id` returns MIRRORED when `assume_sync=true`. Fixed: GLU source resolution uses `assume_sync=false`.

4. **handle_mul_mat_id assertion** — Down projection src1 is PARTIAL (from GLU), not MIRRORED. Fixed: relaxed assertion.

## Verified Correct (Don't Re-Investigate)

- Expert padding script ✅
- Weight data integrity ✅
- Expert routing selection ✅
- Pre-zeroing ✅
- Per-expert matmul outputs ✅ (verified per-slot, nonzero for owned experts)
- get_i_delayed boundary extension ✅ (nodes 46→62, verified correct)
- Dense TP ✅

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

# Check per-slot output (should show nonzero at owned expert slots):
grep "EP_RAW_DOWN" /tmp/ep-debug.log

# Check AllReduce input (GPU0 should be nonzero — currently broken):
grep -A4 "AR #2" /tmp/ep-debug.log

# Check matmul I/O (should be nonzero for processed experts):
grep "EP_MM_OUT" /tmp/ep-debug.log
```

## Key Files

| File | What |
|------|------|
| `ggml/src/ggml-sycl/ggml-sycl.cpp:4194` | EP MUL_MAT_ID dispatch + debug dumps |
| `ggml/src/ggml-backend-meta.cpp:961` | get_i_delayed (aggregation chain walk) |
| `ggml/src/ggml-backend-meta.cpp:1098` | Subgraph boundary + AllReduce dispatch |
| `src/llama-graph.cpp:1350` | `experts = ggml_mul(experts, weights)` aggregation |
| `src/llama-graph.cpp:1380-1400` | VIEW + ADD expert sum chain |
