# BOOTSTRAP — Current State & Path Forward

3x Intel Arc A770 (48GB VRAM), Qwen3-30B-A3B MoE, llama.cpp SYCL.

## What Works Right Now

**Flagship config:** np=16, layer-split, ~26 t/s aggregate generation.

| Setting | Value | Why |
|---------|-------|-----|
| Model | Qwen3-30B-A3B-abliterated Q4_K_M (17.3 GB) | 128 experts, 8 active/token |
| Split | `--split-mode layer` | Only stable multi-GPU mode |
| Context | np-dependent (see below) | c=8192 at np≤4, c=4096 at np=16 verified |
| Slots | np=16 | Max aggregate throughput |
| max_tokens | **≤300** | L0 corruption above ~500 tokens/slot |
| Flash attn | OFF | IGC compiler crash on MoE + FA |
| Graph | OFF | MoE unstable pointers break replay |
| FUSED_MMQ | Works at all batch sizes, **0% throughput gain on MoE** | np=16: 25.6 vs 25.5 t/s without. MoE expert FFNs too small (768-dim) for fused kernel to help. |
| Cmdlists | Batched (=0) | +7.5% over immediate |
| `--reasoning-budget` | 0 | Without this, Qwen3 spends 150+ tokens on `<think>` reasoning before any visible output. `/think` Discord command enables it on-demand (budget=4096). |

**Context limits (np × context dependent, not a flat cap):**

| np | Max verified context | Status |
|----|---------------------|--------|
| 1 | c=8192 | 13.7 t/s gen, 37.2 t/s pp |
| 4 | c=8192 | Works (proxy config) |
| 4 | c=32768 | VRAM exhaustion crash |
| 16 | c=4096 | 26.0 t/s, verified |

**Production path:** Discord bot → arcllm-proxy (:11435) → llama-server (:18400). Proxy runs with FUSED_MMQ=1 in its SYCL env (low-concurrency Discord usage is fine).

**Data churning (test_moe_churn.py, 2026-03-21):**

| Config | Agg t/s | Status |
|--------|---------|--------|
| np=16 c=512 mt=200 | 25.7 | Baseline |
| np=16 c=2048 mt=200 | 25.5 | No throughput loss |
| np=16 c=4096 mt=200 | 26.0 | Fits in VRAM, ~35% GPU util |
| np=16 c=2048 mt=100 | 25.5 | Ultra-short, same per-token speed |
| **np=16 c=4096 mt=300** | **26.8** | **Best — longer gen amortizes overhead** |
| np=16 FUSED_MMQ | 25.6 | Works — no improvement over non-fused (25.5) |

Optimal churning: **np=16, c=4096, max_tokens=300 → 26.8 t/s** (~8 requests/min aggregate: 26.8 t/s ÷ 300 tokens × 60s ≈ 5.4 req/min, or ~0.34 req/min/slot completing every ~3 min).

## The Bug That Caps Us

**Level Zero in-order queue race condition.**

L0 batches kernel submissions and overlaps ops it thinks are independent. With 2+ slots generating deeply (~500+ tokens), Q8_1 activation buffers get corrupted (NaN in ds field) before downstream matmul reads them. Result: garbled output or DEVICE_LOST.

**Note:** Journey.md Chapter 12 documents F16 overflow (`graph[26] op=CPY raw=0xfc00 F16 -inf`) as related/contributing. F16 clamping was added to `cpy_1_f32_f16` and `binbcast safe_narrow`, but whether this fully resolved the F16 path or the L0 race is the deeper cause (or both interact) is not definitively proven. Both may contribute.

| Scenario | Result |
|----------|--------|
| 1 slot, 2000+ tokens | Always passes |
| 16 slots × 200 tokens | Passes |
| 16 slots × 512 tokens | Crashes |
| 2 slots × 1024 tokens | Crashes at ~120s |

**Trigger:** Concurrent compute depth across slots, not sequence length.

**Workaround in production:** Cap max_tokens ≤ 300. Zero throughput cost.

**Sync-based fixes tested (test_q8_1_corruption.py):**

| Fix | Single GPU | Multi-GPU MoE | Throughput hit |
|-----|-----------|---------------|----------------|
| SYNC_EVERY=3 | Pass | Crash | -17% |
| SYNC_EVERY=2 | Pass | Crash | -25% |
| FORCE_SYNC=1 | Pass | Pass | -60% |
| stream->wait() per MUL_MAT | Pass | Pass (6.6 t/s) | -40% |

None are viable for production multi-GPU. Capping max_tokens is the only zero-cost fix.

## Solved Problems (March 20-21)

### Expert Padding (128 ÷ 3 GPU Divisibility) — SOLVED

Script `scripts/pad-experts-gguf.py` pads any MoE GGUF model's expert count for GPU divisibility. Binary-level GGUF patcher.

**How it works:**
- Expert FFN tensors: append zero-weight quant blocks for fake experts
- Router gate weights: set to **zero** (NOT -1e9 — dot product sign issue)
- New `ffn_gate_inp.bias` tensors: -1e30 for fake experts (sign-invariant suppression)
- Tensor count updated in GGUF header, offsets via `GGML_PAD(nbytes, alignment)`

**Bugs found and fixed:**
- Gate weights -1e9 → `dot([-1e9,...], hidden_state)` produces POSITIVE logits when `sum(hidden_state) < 0` (~50% of tokens). Fake expert selected with weight ≈ 1.0, zeroing MoE output. **Fix:** zero weights + bias tensor with -1e30.
- GGUF alignment: requires `GGML_PAD(nbytes, alignment)` between tensors, not contiguous packing. 30B passed by coincidence (all sizes 32-byte aligned), 2.7B exposed it.
- `ffn_gate_inp_shexp` must NOT be padded (shared experts are separate in Qwen2MoE).

**llama.cpp changes:** `llama-model.cpp` loads `ffn_gate_inp_b` (`TENSOR_NOT_REQUIRED`), `qwen3moe.cpp` uses extended `build_moe_ffn` with `gate_inp_b`. Applied to both stable and eptp builds.

**Verified:** 2.7B (60→61 experts) ✅, 30B (128→129 experts) ✅

### REAM Model Red Herring — ~12 Hours Wasted

EP was originally tested on `Qwen3-30B-A3B-REAM-heretic-i1` (96 experts). After extensive investigation, the REAM model garbles on ALL builds including stable TP. The "EP bug" was actually a broken model. The 3.95 t/s EP result from the original EP commit was measured on the broken REAM model and is meaningless.

All REAM/REAP variants tested and rejected (deleted).

## What We're Working On

### 1. MoE Churn Tuning (this session)

See data churning table above. Optimal config identified: np=16, c=4096, mt=300.

### 2. EP (Expert Parallelism) on Abliterated Model

**Worktree:** `llama.cpp-eptp` (branch: `ep-tp-combined`)

Dense TP works perfectly on the eptp build (20.7 t/s). MoE EP garbles. Systematic elimination isolated the bug to the fused aggregation kernel.

**Bugs fixed (March 20-21):**
1. **AllReduce deferral:** peek-ahead called `get_split_state()` on GLU node whose inputs hadn't been processed → returned MIRRORED → AllReduce fired between up and GLU, splitting MoE block in half. **Fix:** look ahead for `MUL_MAT_ID` with `SPLIT_AXIS_2` instead.
2. **Expert distribution rotation:** `rotation = il % n_devices` gave different per-GPU expert distributions per layer, wrong `expert_offset` for 2/3 of layers. **Fix:** `effective_rotation = 0` for `SPLIT_AXIS_2`.
3. **GLU split state:** `handle_mul_mat_id` returned MIRRORED with `assume_sync=true`. GLU source resolution used `assume_sync=true`. **Fix:** use `assume_sync=false` for GLU source resolution; relaxed `handle_mul_mat_id` assertion to accept PARTIAL src1.

**Verification chain (all confirmed correct):** weight data ✅, expert routing ✅, pre-zeroing ✅, per-expert matmul outputs (per-slot) ✅, `get_i_delayed` boundary (nodes 46→62) ✅, buffer pointers ✅, dense TP on eptp build ✅.

**Current blocker:** `fused-expert-agg.cpp` — reads correct buffer pointer (same as MUL_MAT_ID wrote to) but produces zeros. Likely doesn't handle EP's sparse expert slots. AllReduce input shows GPU0 = all zeros despite having 2 nonzero expert slots.

**Status:** Needs clean rebase, then fix fused aggregation kernel for sparse EP slots.

### 3. Qwen3.5 35B MoE — New Architecture

**Worktree:** `llama.cpp-qwen35` (branch: `qwen35-support`)

Architecture: `qwen35moe` — 256 experts, Gated Delta Net attention. Two models tested (`HauhauCS-Aggressive`, `heretic-v2`), both garble at np=1 on clean build from stable-baseline. Upstream SYCL issue with Gated Delta Net, not our code. Separate workstream.

## Big Unlocks (What Could Fix Things)

### A. Fix L0 Queue Corruption — removes the 300-token cap

**Impact:** Unlimited generation length at np=16. Long-form content, full reasoning, no more chaining short requests.

**Path:** Event-based sync with OOO queues (the CUDA pattern). Explicit event dependencies between ops instead of trusting L0's broken in-order execution.

**Status:** Architecture designed (test_row_split.py), blocked by SYCL runtime event pool leak (#14797). No production SYCL system has shipped cross-device events.

**Fallback:** Targeted `stream->wait()` only around Q8_1 quantization (not every MUL_MAT) might recover most throughput. Untested.

### B. ~~FUSED_MMQ on MoE~~ — TESTED, NO BENEFIT

**Result:** Full sweep np=1 through np=16, all pass. 25.6 vs 25.5 t/s at np=16 — within noise. The fused kernel optimizes large dense matmuls by keeping dequantized weights in registers. MoE expert FFNs are 768-dim — too small for this to matter. Earlier crash was stale GPU state, not a real bug.

**Not a path forward for MoE throughput.** Still valuable for dense (Qwen3-32B: +25%).

### C. Fix Fused Expert Aggregation — unblocks EP

**Impact:** Unblocks EP entirely. With EP, each GPU only computes its assigned experts instead of all 128. Potentially 2-3x throughput for MoE.

**Current state:** Per-expert matmul outputs verified correct. `fused-expert-agg.cpp` produces zeros for EP's sparse expert slots. This is the single remaining blocker for EP.

**Path:** The kernel likely assumes dense expert outputs (all slots populated). EP produces sparse outputs (only owned experts have data). Fix the kernel to handle sparse slots, or replace with a non-fused gather+sum path for initial correctness.

### D. N-gram Speculative Decoding — potential +15% (zero-cost test)

**Impact:** ~26.8 → ~30.8 t/s. N-gram spec decoding reuses tokens from prompt/context to speculate future tokens. No draft model needed.

**Path:** llama-server has `--lookup-ngram` support. Needs testing with MoE. Risk: more tokens per step = deeper compute graph, may interact with L0 corruption bug.

## What's Dead / Don't Bother

| Thing | Why |
|-------|-----|
| REAM-heretic models (all variants) | Broken quantization, garbles on all builds. Deleted. |
| REAP pruned 15B | Pruning destroyed quality. Deleted. |
| `--split-mode row` | L0 queue corruption + MMVQ merge crash, worse than layer-split |
| `-fa on` with MoE | IGC compiler internal error, hard crash |
| `-c 32768` with MoE np≥4 | VRAM exhaustion |
| Event-based row-split | SYCL runtime bug #14797, blocked at platform level |
| Q8_0 quant | ~3.3 t/s, bandwidth-bound, not worth optimizing |
| `llama.cpp-tp` worktree | Deleted March 20. Branch `tensor-parallelism-upstream` preserved in git. TP work lives in `llama.cpp-eptp` now. |
| `llama.cpp-expert` worktree | Deleted March 20. Branch preserved in git. |
| Qwen3.5 35B MoE on current build | Gated Delta Net attention garbles, upstream SYCL issue |
