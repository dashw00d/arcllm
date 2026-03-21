# Multi-GPU Tensor Splitting for 3x Intel Arc A770 in llama.cpp SYCL

**Date:** 2026-03-15
**Hardware:** 3x Intel Arc A770 (16 GB VRAM each, 48 GB total), PCIe 3.0, no P2P
**Software:** llama.cpp custom branch (upstream `d417bc4` + 4 local row-split commits)
**Sources:** local source code analysis, llama.cpp issue/PR history, benchmarks in `/home/ryan/llm-stack/docs/SYCL_TUNING_GUIDE.md`

---

## 1. Inter-GPU Tensor Transfer Without P2P: How It Works

Intel Arc A770 connected via PCIe has no peer-to-peer direct device memory access. There is no Intel equivalent of NVLink or PCIe P2P CUDA extension for Arc. The `ggml_sycl_set_peer_access()` function in `ggml-sycl.cpp:3551` was ported from the original CUDA backend but its body is entirely commented out — all `syclDeviceEnablePeerAccess` calls are dead code for the SYCL path.

**What actually happens for cross-device copies** is implemented in `dev2dev_memcpy()` at `ggml-sycl.cpp:573`:

```cpp
// Step 1: src queue memcpy to host buffer (with .wait_and_throw())
q_src.memcpy(host_buf, ptr_src, size).wait_and_throw();
// Step 2: dst queue memcpy from host buffer (with .wait_and_throw())
q_dst.memcpy(ptr_dst, host_buf, size).wait_and_throw();
```

This is a synchronous two-step copy through a host-side buffer. If `ggml_sycl_host_malloc()` succeeds (pinned memory via Level Zero USM), the buffer is pinned and benefits from higher DMA bandwidth. If not, it falls back to regular malloc. Both legs wait synchronously.

**Bandwidth cost:** PCIe 3.0 x16 theoretical is 16 GB/s. For host-staged copy, the data crosses PCIe twice: once device→host, once host→device. Effective max throughput for a cross-device tensor copy is therefore approximately **8–12 GB/s** (limited by the slower direction and PCIe contention). This contrasts with in-device bandwidth of ~560 GB/s per A770.

**Latency cost:** Each cross-device copy incurs two synchronous `.wait_and_throw()` calls. Even for small tensors (say, 1 MB), the latency floor is ~0.1 ms minimum due to queue submission and PCIe transaction overhead. This matters more for row-split where many small per-column merges happen per token.

**Discovery context:** The original row-split code used `dpct::async_dpct_memcpy` with `dpct::device_to_device` direction. On Intel Arc without P2P, this call **silently produced garbage output** — the destination buffer received stale data without raising an error. This was Bug #4 fixed in commit `daf5a6f` ("SYCL row-split: fix 5 bugs enabling multi-GPU row-split on Intel Arc A770").

---

## 2. `--tensor-split 1,1,1` vs `0.33,0.33,0.34` — Format and Behavior

**They are identical in practice.** The tensor-split input is treated as a list of *relative weights*, not absolute fractions. In `ggml_backend_sycl_split_buffer_type()` at `ggml-sycl.cpp:1888`:

```cpp
float split_sum = 0.0f;
for (int i = 0; i < device_count; ++i) {
    tensor_split_arr[i] = split_sum;  // cumulative sum BEFORE this device
    split_sum += tensor_split[i];
}
for (int i = 0; i < device_count; ++i) {
    tensor_split_arr[i] /= split_sum;  // normalize to [0,1]
}
```

After normalization, `1,1,1` → `[0.0, 0.333, 0.667]` and `0.33,0.33,0.34` → `[0.0, 0.330, 0.660]`, which differ only in floating-point rounding at the third decimal. With 3 identical 16 GB GPUs, `1,1,1` is the natural expression and is what the code's examples use.

The **stored tensor_split** is the cumulative normalized array (boundary positions), not proportional weights. `tensor_split[0]` is always 0.0 (device 0 starts at row 0). `tensor_split[1]` is the fraction where device 1 starts. `tensor_split[2]` is where device 2 starts. The last device always ends at `nrows`.

**Row boundary computation** (`ggml_sycl_get_row_split` at `ggml-sycl.cpp:1471`):

```cpp
*row_low  = id == 0 ? 0 : nrows * tensor_split[id];
*row_low -= *row_low % rounding;   // align to quant block boundary
*row_high = id == last ? nrows : nrows * tensor_split[id+1];
*row_high -= *row_high % rounding;
```

The rounding is type-specific: Q4_K/Q5_K/Q6_K round to 64 rows, Q4_0/Q4_1 round to 128 rows on Gen9+ hardware. For a 7168-row tensor with 3 devices, device 0 gets rows 0–2388, device 1 gets 2389–4735, device 2 gets 4736–7168 (approximately — actual boundary depends on quantization alignment rounding).

**Default (no `--tensor-split`):** If all zeros are passed, `ggml_sycl_info().default_tensor_split` is used, which is initialized at startup proportional to each device's VRAM capacity (`total_vram` accumulation). For 3 identical 16 GB A770s, this defaults to equal thirds — same as `1,1,1`. No practical reason to specify `1,1,1` explicitly except documentation clarity.

---

## 3. Row-Split Execution: Scatter/Gather vs All-Reduce

Row-split in llama.cpp SYCL is **scatter-then-gather**, not a ring all-reduce. The pattern for each MUL_MAT in `ggml_sycl_op_mul_mat()`:

**Phase 1 — Scatter src1:** The activation tensor (`src1`, float32, same shape for all GPUs) is copied to each non-owning device via `dev2dev_memcpy`. This is the host-staged copy bottleneck: one copy per additional GPU.

**Phase 2 — Parallel matmul:** All `used_devices` GPUs execute their respective row shard of the matmul independently and simultaneously. Device `i` computes rows `[row_low_i, row_high_i)` of src0 × src1. There is no cross-GPU communication during computation. SYCL barrier events (`ext_oneapi_submit_barrier`) synchronize device `i` on the barrier event posted by the main device after scatter.

**Phase 3 — Gather results:** Each non-main device's partial result (`dst_dd_i`, float32) is copied back to the main device's dst buffer via `dev2dev_memcpy`, per-column. Code comment at `ggml-sycl.cpp:5029`: *"Use dev2dev_memcpy with both source and destination queues for correct cross-device copy on Intel Arc (no P2P access)."*

**No all-reduce:** The partial results from different GPUs cover *disjoint row ranges* of the output, so they can simply be assembled by placement — no summation needed. This differs from tensor parallelism where GPUs compute the same output rows in parallel (requiring reduction). Row-split is more precisely "data sharding": device `i` owns rows `[row_low_i, row_high_i)` of every weight matrix.

**SYCL primitives used:**
- `queue::ext_oneapi_submit_barrier({event_list})` — cross-device synchronization (Level Zero `zeCommandListAppendBarrier`)
- `queue::memcpy(dst, src, size)` — the underlying DMA transfer
- `dev2dev_memcpy` — host-staged wrapper for cross-device copies (no Level Zero native mechanism)

No `sycl::group_barrier`, no `sycl::reduce_over_group`, and no `allreduce_tensor_async` (the latter is defined as an interface in [PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378) but not implemented for SYCL).

---

## 4. Optimal Tensor-Split for 3 Identical GPUs

**`1,1,1` is optimal for 3 identical 16 GB A770s**, full stop. No reason to weight differently because:

1. All three GPUs have the same VRAM (16 GB), same memory bandwidth (~560 GB/s), same compute (512 Xe-cores/32 Xe-HPG, 24 Xe-LP in Arc A770).
2. Unequal splits would give one GPU fewer rows (less computation), while another GPU does more. Since all GPUs must finish before the gather, the total time is gated by the slowest device. Equal split minimizes max device time.
3. The only exception would be if one GPU has significantly different available VRAM after system allocations (e.g., a display output consuming iGPU shared memory). For 3 discrete A770s with no display outputs, all VRAM is available equally.

**Weighting differently would help if:**
- The main device (device 0 by default) has extra work in the gather phase. In practice, the gather cost is the same regardless of split — it's proportional to the number of row shards, not their size.
- One GPU is slower than others (different hardware, thermal throttling). Not applicable for identical A770s in steady state.

The default behavior with no `--tensor-split` also produces equal thirds for 3 identical GPUs, so this flag is mostly redundant but harmless.

---

## 5. Layer-Split with 94-Layer Qwen3-235B: Layer Distribution

**Qwen3-235B-A22B has 94 transformer layers** (confirmed in local benchmark documents and Qwen3-235B model card). With `--split-mode layer` and 3 GPUs, the assignment follows `get_layer_buft_list()` in `src/llama-model.cpp:2644`:

```cpp
const int layer_gpu = std::upper_bound(
    splits.begin(), splits.begin() + n_devices(),
    float(il - i_gpu_start) / act_gpu_layers
) - splits.begin();
```

For 3 equal GPUs (splits = [0.0, 0.333, 0.667]):
- Layer 0–31: device 0 (32 layers)
- Layer 32–62: device 1 (31 layers)
- Layer 63–93: device 2 (31 layers)

(The exact distribution shifts based on `i_gpu_start` from `--n-gpu-layers`, but with `-ngl 999` all 94 layers are GPU-assigned.)

**The input embedding layer (`tok_embeddings`) is always on CPU** — the code explicitly comments: *"there is very little benefit to offloading the input layer, so always keep it on the CPU"* at `llama-model.cpp:2651`. The output layer (LM head) is placed according to `get_layer_buft_list(n_layer)`, same as a regular transformer layer, so it falls on device 2.

**For Qwen3-235B-A22B as a MoE model**, the distinction between "attention-only" and "expert" layers matters: all 94 layers have both attention and MoE FFN components. There are no "attention-only" layers in Qwen3-235B — unlike some hybrid models (e.g., Qwen3.5-27B which interleaves full attention and GDA recurrent layers). With `-cmoe`, the expert FFN weights (`blk.*.ffn_up_exps`, `blk.*.ffn_down_exps`, `blk.*.ffn_gate_exps`) are moved to CPU regardless of layer-split; only the attention tensors remain in VRAM.

**Practical fit check:** At Q3_K_S (~14 GB of attention weight per GPU in layer-split), 32 layers per GPU occupies ~13–15 GB of the 16 GB VRAM, leaving ~1–3 GB for KV cache and compute buffers. This is tight. Without `-cmoe` (full model in VRAM), Qwen3-235B doesn't fit across 48 GB total at any practical quantization level.

---

## 6. PCIe Bandwidth Constraint: Row-Split vs Layer-Split

### Layer-split bandwidth model

In layer-split, each GPU independently reads its weight shards from VRAM and processes its layers sequentially. There is **no inter-GPU communication during inference**. Activations move device→device only at layer boundaries, and because the scheduler assigns contiguous layer blocks to each GPU, cross-device activation transfers happen only once per block boundary (twice total for 3 GPUs). These transfers are small (embedding dimension × batch_size × sizeof(float) ≈ 7168 × 1 × 4 = 28 KB for single-token decode on Qwen3-235B), negligible at ~12 GB/s effective rate (~2.3 µs).

**Effective bandwidth for TG:** ~560 GB/s on whichever single GPU is currently active. Multi-GPU layer-split is **not parallel** — only one GPU is active per layer, the others idle. This is confirmed by the local benchmarks: Qwen3-32B Q4_K_M at 7.0 t/s on 3 GPUs vs 6.7 t/s on 2 GPUs — barely any difference because the bottleneck is per-GPU bandwidth, not GPU count.

### Row-split bandwidth model

In row-split, every matmul requires:
1. **Scatter:** src1 (activation, ~28 KB for single token) copied from main device to 2 other devices via host. At 12 GB/s: ~2 × (28 KB / 12 GB/s) ≈ 4.7 µs per scatter.
2. **Compute:** All 3 GPUs run their shard matmul in parallel. Ideal 3× speedup.
3. **Gather:** 2 partial results (each ~row_diff × output_dim × 4 bytes) copied from off-devices to main device via host. For a 7168-dim weight with 1/3 rows: 2388 × output_dim × 4 bytes. For ffn_down with output=7168: 2388 × 7168 × 4 = ~68 MB per call, at 12 GB/s = ~5.7 ms per gather.

At 7.0 t/s for layer-split, one token decode takes ~143 ms. Qwen3-235B-A22B has 94 attention output projections (7168×7168) and 94 × 8 active expert FFN layers (7168×2048 each). The gather cost alone for attention output over 94 layers: 94 × 2 × 5.7 ms ≈ 1.07 s — more than 7× the total token time. This explains why row-split TG for dense models is catastrophically slow: **PCIe bandwidth saturates on result merging**.

The mitigation is `-cmoe`: with expert FFN on CPU, the row-split GPUs only hold attention tensors. Expert matmuls run on CPU directly (via `mul_mat_id` CPU path), bypassing the inter-GPU merge entirely. The gather cost is reduced to attention-only operations, shrinking the PCIe-bound phase.

**Quantitative summary from local benchmarks:**
| Config | TG t/s | Reason |
|--------|--------|--------|
| Layer-split, Qwen3-32B Q4_K_M | 6.96 | 560 GB/s per GPU, sequential |
| Row-split, Qwen3-32B Q4_K_M | 0.77 | PCIe gather + DMMV fallback |
| Row-split + -cmoe, 30B-A3B | 0.95 | Expert FFN on CPU, attention row-split |

---

## 7. Stability Issues: 3-Way SYCL Multi-GPU

**Layer-split is stable** on the local branch for tested configurations (Qwen3-32B Q4_K_M, Q8_0 at 65k context). No hangs, crashes, or garbage output observed in the current build.

**Known stability issues (historical/upstream):**

- **PR #8554** (Jul 2024): Fixed crash with 2+ GPUs caused by incorrect SYCL platform enumeration in queue management. Before this, multi-GPU initialization crashed.

- **Issue #19847** (Feb 2026): Garbage output after large prompts using SYCL backend (single iGPU, not multi-GPU). Root cause unconfirmed; may be related to Q8_1 ds visibility bug manifesting differently under large context.

- **Issue #15580** (Oct 2025): MoE row-sort assertion `GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as) failed` during async kernel execution for MoE models on iGPU. Fixed by adding `.wait()` after argsort kernel. Multi-GPU exacerbates async ordering issues.

- **Row-split before daf5a6f:** Upstream llama.cpp crashes immediately with `--split-mode row` on Intel Arc SYCL. The local branch required 5 independent bug fixes and 3 follow-up commits to reach stability.

**Current local branch status** (row-split, validated 2026-03-14):
- Qwen3-0.6B: correct output, validated against expected text, 1/2/3 GPU configurations all produce identical output
- Qwen3-30B-A3B: loads and runs, `llama-simple` functional
- `llama-cli`/`llama-server` with row-split: pre-existing issue (separate from the 5 bugs) under investigation

**MMVQ/MMQ disabled globally for multi-GPU:** Commit `e0ebf6b` disabled MMVQ and MMQ for **all** `device_count > 1` configurations, not just split tensors. This was necessary because MoE expert matmuls use `MUL_MAT_ID` views of split tensors, where the Q8_1 ds visibility bug applies even to tensors that appear non-split from the dispatch path.

---

## 8. ZE_AFFINITY_MASK: Current Correct Usage

**Yes, `ZE_AFFINITY_MASK=0,1,2` is still the correct way to select all 3 Arc GPUs for Intel Level Zero.**

The variable is set in `/home/ryan/llm-stack/env.sglang-xpu.sh:77`:
```bash
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1,2}"
```

**How it works:** `ZE_AFFINITY_MASK` is a Level Zero runtime environment variable (not a SYCL variable directly). The Level Zero driver respects it during device enumeration: only devices whose index appears in the mask are exposed to the process. `SYCL` over `level_zero` backend then sees only those devices.

**ONEAPI_DEVICE_SELECTOR vs ZE_AFFINITY_MASK:**

The SYCL.md (`docs/backend/SYCL.md`) documents two approaches:

| Goal | Method |
|------|--------|
| Use all 3 A770s | `ZE_AFFINITY_MASK=0,1,2` (Level Zero layer) |
| Use all 3 A770s | `ONEAPI_DEVICE_SELECTOR="level_zero:*"` (SYCL layer) |
| Use specific 2 of 3 | `ZE_AFFINITY_MASK=0,2` or `ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:2"` |

The preferred modern approach per Intel's documentation is `ONEAPI_DEVICE_SELECTOR` as it works at the SYCL layer and supports multiple backends. `ZE_AFFINITY_MASK` is a lower-level Level Zero variable. Both work; `ZE_AFFINITY_MASK` is slightly more established and is what the local env uses.

**Caution:** The SYCL backend documentation notes (PR #8014 discussion): *"One physical SYCL device could be mapped to two logical devices: level-zero and openCL devices."* Without `ZE_AFFINITY_MASK` or `ONEAPI_DEVICE_SELECTOR` filtering, an Arc A770 system may expose both `level_zero:0` and `opencl:0` representing the same physical GPU. The SYCL backend by default selects Level Zero devices with the highest compute unit count, which handles this correctly — but explicit `ZE_AFFINITY_MASK` prevents the opencl duplicates from appearing at all.

**Additional variable:** `ZES_ENABLE_SYSMAN=1` (set in local env at line 46) is required for VRAM usage reporting via `sycl::aspect::ext_intel_free_memory`, which the SYCL.md recommends when using `--split-mode layer`. Without it, the scheduler cannot query free memory per device.

---

## 9. Verifying Which Tensors Land on Which GPU

**At load time (verbose logging):** Pass `--verbose` to `llama-server` or set `GGML_LOG_LEVEL=DEBUG`. The tensor assignment is logged for each tensor in `llama-model.cpp:2641`:
```
load_tensors: layer  31 assigned to device SYCL0, is_swa = 0
load_tensors: layer  32 assigned to device SYCL1, is_swa = 0
load_tensors: layer  63 assigned to device SYCL2, is_swa = 0
```

**At runtime (split buffer inspection):** Set `GGML_SYCL_DEBUG=1` which enables `GGML_SYCL_DEBUG` macro logging. This prints device-level activity per operation but is extremely verbose.

**Row-split specific debug:** The local build has extensive row-split debug instrumentation controlled by `g_ggml_sycl_debug_row` (set via `GGML_SYCL_DEBUG_ROW=1` env var, guarded by `GGML_SYCL_ROW_DEBUG_DISABLED` compile-time flag). The instrumentation emits per-tensor/per-device trace logs including which tensor pointer maps to which device, quantized values, and merge column data. As of commit `e0ebf6b`, the debug code is compiled in but gated by the env var.

**Quick check using llama-ls-sycl-device:**
```bash
build/bin/llama-ls-sycl-device
```
Prints all visible Level Zero devices with their IDs, confirming the mapping between llama.cpp SYCL device indices and physical GPUs.

**Graph splits count:** The server output includes `graph splits = N` which indicates how many separate GPU-specific subgraphs the scheduler created. For 3-GPU layer-split, `graph splits = 4` is typical (3 GPU subgraphs + 1 CPU subgraph for embedding/output).

---

## 10. The Q8_1 `ds` half2 L1 Cache Visibility Bug

### What it is

The Q8_1 quantize kernel (`ggml/src/ggml-sycl/quantize.hpp:94–117`) produces Q8_1 blocks: 32 `int8` quantized values (`qs`) plus a `sycl::half2` scale-and-sum pair (`ds`). Only work-item `wi_id==0` in each sub-group writes the `ds` value:

```cpp
if (wi_id == 0) {
    quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
}
```

On Intel Arc A770, when the subsequent MMVQ kernel reads `ds` from the same device buffer, it reads `d=0.0` and `s=0.0` (zero). The `qs` bytes are correct. The result is that all dot products compute as zero, producing entirely silent garbage output. This manifests as the model outputting empty or near-zero activations through every layer.

### Scope: intra-device bug, not inter-GPU

This is a **per-device kernel issue**, not related to inter-GPU communication. The same queue, same device, same buffer, same in-order command queue — the store from the quantize kernel is simply not visible to the subsequent read in the MMVQ kernel on Intel Arc hardware.

The bug was found during single-GPU row-split debugging (`g_ggml_sycl_debug_row` instrumentation) when the main device was running MMVQ on its own shard. The DMMV kernel (which dequantizes weights to float32 rather than using Q8_1) is unaffected because it never reads `ds` from a Q8_1 activation buffer.

### Why it appears in multi-GPU but not single-GPU non-split

In non-split single-GPU mode, `g_ggml_sycl_disable_graph=1` forces the code path to use DMMV instead of MMVQ (`can_use_dequantize_mul_mat_vec` returns `false` when `g_ggml_sycl_disable_graph && !split` at `ggml-sycl.cpp:6066`). This was coincidentally masking the bug. When SYCL graphs are enabled (single-GPU, `GGML_SYCL_DISABLE_GRAPH=0`), MMVQ runs and the bug would appear — but SYCL graph mode has its own incompatibility with MMVQ (blocking `wait()` calls cannot be recorded into a graph). So in practice, MMVQ in single-GPU mode was either suppressed by the graph-disable fallback or gated out by graph-recording incompatibility.

Row-split mode sets `g_ggml_sycl_disable_graph=1` (sync mode for multi-GPU), which *would* have re-enabled MMVQ for the non-split check, exposing the Q8_1 bug.

### Intel bug report status

**No Intel bug report has been filed as of 2026-03-15.** The bug was discovered and characterized entirely within the local development process (commit `daf5a6f`, 2026-03-14). No Intel developer response, no Intel forum thread, no known public acknowledgment.

The root cause hypothesis (L1 cache coherency for single-work-item `half2` stores) is consistent with known Intel Arc GPU microarchitecture characteristics but has not been confirmed by Intel. Attempted fixes:
- `sycl::atomic_ref` store: crashes (likely alignment issue or unsupported half2 atomic)
- Sub-group barrier after `ds` write: no effect (barrier synchronizes work items, not cache lines between kernels)
- `volatile` qualifier: compile error in device code path

### Workaround in place

The fix in `can_use_dequantize_mul_mat_vec()` at `ggml-sycl.cpp:6057–6070`:
```cpp
const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
if (g_ggml_sycl_disable_graph && !split) {
    return false;  // normal graph-disabled path: no DMMV
}
// For split tensors: allow DMMV to bypass Q8_1 ds visibility bug
```

Then in `ggml_sycl_mul_mat()` around line 6212: *"For split tensors, prefer DMMV over MMVQ to avoid Q8_1 ds visibility bug."*

Additionally, commit `e0ebf6b` broadened this to disable MMVQ/MMQ for **all `device_count > 1`** configurations (`g_ggml_sycl_disable_graph && device_count > 1`), covering MoE expert matmul views that don't appear as split buffers to the dispatch path.

### Alternative fix approaches

1. **Use a separate memcpy to commit ds:** After the Q8_1 kernel, issue an explicit `queue.memcpy(ds_dst, ds_src, sizeof(half2))` per block. Extremely slow but guaranteed visibility. Not viable for production.

2. **Two-pass quantization:** Separate kernel pass 1: compute and store `qs`. Pass 2: compute and store `ds`. Add an explicit queue synchronization between passes. Adds kernel launch overhead but may be fast enough.

3. **Avoid half2 store from single work item:** All work items in the sub-group atomically write `ds` (only one value, they all agree). Atomics for half2 may not be supported on Arc, but float atomics are. Convert: `*reinterpret_cast<float*>(&ds_location) = bit_cast<float>(half2(d, sum))`. This avoids the single-wi store pattern.

4. **Use global memory store instead of sub-group store:** The root cause may be that the `if (wi_id == 0)` store goes into a register-backed location that doesn't flush to global memory before the next kernel. A `sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device)` **after** the store might force visibility. Not yet tested.

5. **File Intel bug report:** Reproduce with a minimal test kernel (quantize → read ds), attach to [Intel oneAPI GitHub](https://github.com/intel/llvm/issues) or Intel Developer Forum. Required info: IGC version, level-zero runtime version, exact kernel, observed vs expected behavior.

---

## 11. SYCL Graph Capture and Multi-GPU: Why Fundamentally Incompatible

**The core limitation** (from `ggml-sycl.cpp:11477–11480`):
```cpp
static bool check_graph_compatibility(ggml_cgraph * cgraph) {
    if (ggml_sycl_info().device_count > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        return false;
    }
```

A `sycl_ex::command_graph` (via the `ext_oneapi_graph` SYCL extension) is created per-device. There is no SYCL API to create a graph that spans multiple devices with cross-device synchronization. This is a spec limitation, not an implementation choice.

For row-split to benefit from SYCL graphs, the entire multi-GPU execution would need to be recorded in separate per-device graphs plus explicit cross-device event coordination — a significant engineering effort that Intel has not implemented in llama.cpp.

**What SYCL graphs give you (single GPU only):**
- Eliminates kernel launch overhead (~10–100 µs per kernel) by replaying a pre-compiled command sequence
- Reduces Level Zero API call overhead
- Measured benefit: +22% TG throughput for Qwen3-32B Q8_0 single-GPU (3.02→3.70 t/s)

**Additional graph incompatibilities** (even single-GPU):
- `GGML_OP_CONCAT`: uses blocking `wait()` after memcpy — cannot record into graph
- `GGML_OP_MUL_MAT_ID`: uses blocking `wait()` after memcpy — cannot record
- `GGML_OP_MUL_MAT` without async mem: cannot record unless `g_ggml_sycl_use_async_mem_op` is true (requires oneAPI async allocation extension, not always available)

**Practical implication for row-split:** Fixing row-split to also benefit from SYCL graphs would require: (a) porting to per-device graph recording with cross-device event injection; (b) resolving the `MUL_MAT_ID` blocking issue; (c) the Q8_1 ds visibility bug fix. All three need to be fixed before row-split can use SYCL graphs.

---

## 12. Recent (2025–2026) llama.cpp SYCL Multi-GPU Improvements

| Date | Commit/PR | Change | Impact |
|------|-----------|--------|--------|
| Jul 2024 | [PR #8554](https://github.com/ggml-org/llama.cpp/pull/8554) | Fix multi-GPU crash (platform enumeration) | Enables 2+ GPU at all |
| Feb 2025 | [PR #12035](https://github.com/ggml-org/llama.cpp/pull/12035) | Q4_0 block reordering (+30% on A770) | Single-GPU speedup |
| Feb 2025 | [PR #14815](https://github.com/ggml-org/llama.cpp/pull/14815) | Refactor Q8_1 quantization kernels | Code quality, no behavioral change |
| Nov 2025 | [PR #18992](https://github.com/ggml-org/llama.cpp/pull/18992) | Use malloc to support iGPU and dGPU simultaneously | Device selection fix |
| Feb 2026 | [PR #19889](https://github.com/ggml-org/llama.cpp/pull/19889) | Fix binbcast `s10==1` assertion for Qwen3-Coder | Stability fix |
| Feb 2026 | [PR #19920](https://github.com/ggml-org/llama.cpp/pull/19920) | Remove hardcoded work-group 768 (iGPU/dGPU compat) | Compatibility |
| Mar 2026 | [PR #20190](https://github.com/ggml-org/llama.cpp/pull/20190) | Flash Attention for SYCL (PP +19–77%) | PP speedup |
| Mar 2026 | [PR #20283](https://github.com/ggml-org/llama.cpp/pull/20283) | Fix ACC/L2_NORM/UPSCALE/fused_glu/unary bugs | Stability |
| Mar 2026 | [PR #20293](https://github.com/ggml-org/llama.cpp/pull/20293) | Fix ROPE op, add ROPE_BACK | Correctness |
| Mar 2026 | [PR #20455](https://github.com/ggml-org/llama.cpp/pull/20455) | GATED_DELTA_NET op (Qwen3.5: PP 90→339 t/s) | Hybrid model support |
| Mar 2026 | [PR #20583](https://github.com/ggml-org/llama.cpp/pull/20583) | Fix untransposed GDA recurrent state | Hybrid model fix |
| Mar 2026 | `daf5a6f` (local) | Row-split: 5-bug fix enabling multi-GPU row-split | **Row-split enabled** |
| Mar 2026 | `8b6d3cb` (local) | Row-split: SSM pointer + 256 KB replication | SSM/Mamba support |
| Mar 2026 | `7423303` (local) | Row-split: full tensor replication for partial shards | Correctness fix |
| Mar 2026 | `e0ebf6b` (local) | Row-split: disable MMVQ/MMQ for all multi-GPU | Q8_1 bug workaround |

**What upstream still lacks (as of 2026-03-15):**
- Upstream `SYCL.md` still lists "Split-mode:[row] is not supported" as a known issue
- No upstream multi-GPU row-split support for Arc
- The 4 local commits are not yet submitted to upstream

---

## Summary Table: Research Questions

| Question | Answer |
|----------|--------|
| P2P between Arc GPUs? | No — host-staged 2-leg copy via `dev2dev_memcpy`, ~8–12 GB/s effective |
| `1,1,1` vs `0.33,0.33,0.34`? | Identical after normalization; `1,1,1` is preferred form |
| Row-split mechanism? | Scatter src1 → parallel per-shard matmul → gather results; no all-reduce |
| Optimal split for 3 identical GPUs? | `1,1,1` (equal) always optimal for identical hardware |
| Layer distribution for 94-layer model? | 32/31/31 layers across GPUs 0/1/2 (with `-ngl 999`) |
| PCIe impact on row-split? | Catastrophic for TG on dense models; mitigated by `-cmoe` for MoE |
| Stability of 3-way row-split? | Functional with local patches; `llama-cli`/`llama-server` have secondary bug |
| `ZE_AFFINITY_MASK=0,1,2`? | Correct; sets Level Zero device visibility; also works via `ONEAPI_DEVICE_SELECTOR` |
| Debug tensor placement? | `--verbose` at load; `GGML_SYCL_DEBUG=1` at runtime; `llama-ls-sycl-device` |
| Q8_1 ds visibility bug scope? | Per-device intra-kernel issue; no inter-GPU component; no Intel bug report filed |
| SYCL graph + multi-GPU? | Fundamentally incompatible (spec limitation); graph disabled by `check_graph_compatibility` |
| Recent improvements? | Flash Attention (Mar 2026), GATED_DELTA_NET (Mar 2026), 5-bug row-split fix (local, Mar 2026) |

---

## Recommended Changes to Deployment Plan

### 1. Keep layer-split for all dense models (no change needed)
Layer-split at ~7 t/s for Qwen3-32B Q4_K_M is already near the theoretical bandwidth ceiling for a 19 GB model at 560 GB/s per GPU. Row-split cannot exceed this until the Q8_1 ds visibility bug is fixed and MMVQ is re-enabled — current row-split TG is 9× slower than layer-split for dense models.

### 2. Keep row-split + `-cmoe` for Qwen3-235B-A22B (no change needed)
This is the only viable configuration for 235B on 48 GB VRAM. Row-split VRAM efficiency allows the attention weights (~12–15 GB at Q3), KV cache, compute buffers, and optionally a draft model to coexist. The DMMV penalty on attention matmuls is acceptable (~0.9–1.0 t/s) given the expert FFN is CPU-bound regardless.

### 3. Upgrade `--tensor-split 1,1,1` documentation but keep the value
The format is fine. Consider adding a comment explaining it's relative weights, not absolute fractions, and that the default behavior (no flag) is identical for 3 equal-VRAM GPUs.

### 4. Do not migrate from `ZE_AFFINITY_MASK` to `ONEAPI_DEVICE_SELECTOR`
`ZE_AFFINITY_MASK=0,1,2` works, is validated, and is what the Intel SYCL.md shows as a primary example. A migration to `ONEAPI_DEVICE_SELECTOR="level_zero:*"` would be equivalent but introduces no benefit.

### 5. Keep `GGML_SYCL_DISABLE_GRAPH=0` for single-GPU, set `1` explicitly for 3-GPU launches
The env is currently `GGML_SYCL_DISABLE_GRAPH=0` in `arcllm-server.sh:31`. For multi-GPU (all production configs), `check_graph_compatibility()` disables graphs anyway, so this value is harmless but misleading. Setting it to `1` explicitly for multi-GPU configs avoids the irrelevant `check_graph_compatibility` log spam. For single-GPU configs (if any are added), keep `0` for the +22% benefit.

### 6. Investigate and file Intel bug report for Q8_1 ds visibility
This is the single highest-impact open issue. A minimal repro kernel (quantize_q8_1 → read ds in next kernel → assert non-zero) run as a standalone SYCL program would confirm the bug in isolation. File at [https://github.com/intel/llvm/issues](https://github.com/intel/llvm/issues) with IGC version (`ocloc --version`), level-zero runtime version, and the minimal repro. If Intel acknowledges it as an L1 cache flushing issue, a driver fix or `sycl::atomic_fence` solution may become available. Fixing this would re-enable MMVQ for split tensors and potentially bring row-split TG to parity with or better than layer-split.

### 7. Try explicit fence before MMVQ as an alternative Q8_1 fix
Before filing the Intel bug, test `sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device)` at the end of `quantize_q8_1` kernel (after the `if (wi_id==0) { ds = ... }` write). If this fixes the visibility issue, it's a low-overhead kernel-side fix that doesn't require disabling MMVQ. Current code is at `quantize.hpp:114–116` and `quantize.hpp:87–90` (reorder variant). Both need the fence.

### 8. Consider submitting the 5-bug row-split fix upstream
The 5 bugs fixed in `daf5a6f` are independently valid fixes for the SYCL multi-GPU path. Bugs 1 (binbcast split pointer), 2 (1D tensor replication), 4 (dev2dev_memcpy for no-P2P), and 5 (pool allocation on wrong device) are clean standalone fixes with no Intel-Arc-specific assumptions. Bug 3 (Q8_1 workaround) could be contributed as a temporary workaround with an inline comment describing the Intel-specific root cause. The upstream SYCL team has expressed interest in row-split support (SYCL.md: "it's on developing").

### 9. Set `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` as explicit default
This is already set in `env.sglang-xpu.sh:45` but not in `arcllm-server.sh`. Without it, Level Zero will refuse to allocate device buffers >4 GB, silently failing for models that need large single-tensor allocations. Should be set universally for all llama.cpp SYCL launches on Arc A770.

### 10. Monitor for upstream Flash Attention multi-GPU regression
[PR #20190](https://github.com/ggml-org/llama.cpp/pull/20190) added Flash Attention to SYCL (March 2026). The PR notes "FA disabled when a layer uses SYCL0 but FA tensor is on CPU" — this is the FA auto-disable behavior already seen in issue #19656. For multi-GPU layer-split, if the FA tensor assignment mismatches with the layer assignment, FA may silently fall back to non-FA mode. Monitor `sched_reserve: Flash Attention was auto, set to disabled` in logs and test PP throughput after any llama.cpp upstream merge.
