# BOOTSTRAP.md Corrections

Audit by Orion (OpenClaw main agent) — has full session context from March 20-21 debugging.

## 0. OVERALL: Missing Major Work Streams

The BOOTSTRAP is shallow on recent March 20-21 work. Major missing topics:

### Expert Padding (÷3 GPU Divisibility) — SOLVED
Script `scripts/pad-experts-gguf.py` pads any MoE GGUF model's expert count for GPU divisibility. Binary-level GGUF patcher — no safetensors conversion needed.

**How it works:**
- Expert FFN tensors: append zero-weight quant blocks for fake experts
- Router gate weights: set to zero (NOT -1e9 — dot product sign issue)
- New `ffn_gate_inp.bias` tensors: -1e30 for fake experts (sign-invariant suppression)
- Tensor count updated in GGUF header
- Offsets computed using `GGML_PAD(nbytes, alignment)` per tensor

**Critical bug we found and fixed:** Setting gate weights to -1e9 caused `dot([-1e9,...], hidden_state)` to produce POSITIVE logits when `sum(hidden_state) < 0` (~50% of tokens). The fake expert got selected with weight ≈ 1.0, zeroing MoE output. Fix: zero weights + bias tensor with -1e30.

**Alignment bug fixed:** GGUF requires `GGML_PAD(nbytes, alignment)` between tensors, not contiguous packing. The 30B model passed validation by coincidence (all tensor sizes 32-byte aligned). The 2.7B exposed it (IQ4_NL tensors not aligned).

**Shared expert filter:** `ffn_gate_inp_shexp` must NOT be padded (Qwen2MoE has separate shared experts).

**llama.cpp changes for padding support:**
- `llama-model.cpp`: Added `ffn_gate_inp_b` loading (`TENSOR_NOT_REQUIRED`) for Qwen3MoE
- `qwen3moe.cpp`: Switched to extended `build_moe_ffn` with `gate_inp_b` parameter
- Applied to both `llama.cpp-stable` and `llama.cpp-eptp` builds

**Verified:**
- Qwen2MoE 2.7B (60→61): ✅ Clean output, 26.1 t/s
- Qwen3MoE 30B (128→129): ✅ Clean output, 15 t/s np=1, 39.4 t/s np=16 layer-split

### EP Debugging Deep Dive — 4 Bugs Found and Fixed

Dense TP works perfectly on the eptp build (20.7 t/s). MoE EP garbles. We isolated the bug to the MoE EP dispatch code through systematic elimination.

**Bug 1: AllReduce deferral (FIXED)**
The peek-ahead logic for defer_ep_allreduce called `get_split_state()` on the GLU node whose inputs hadn't been processed yet. GLU returned MIRRORED → AllReduce fired between up and GLU, splitting the MoE block in half.
Fix: Look ahead for `MUL_MAT_ID` with `SPLIT_AXIS_2` instead of resolving intermediate split states.

**Bug 2: Expert distribution rotation (FIXED)**
`rotation = il % n_devices` caused different layers to have different per-GPU expert distributions. Layer 0: GPU1 owns experts 42-84. Layer 1: GPU1 owns experts 43-84. `expert_offset` was wrong for 2/3 of layers.
Fix: `effective_rotation = 0` when `split_state.axis == SPLIT_AXIS_2`.

**Bug 3: GLU split state (FIXED)**
`handle_mul_mat_id` returns MIRRORED when `assume_sync=true` (correct for weight init). But GLU's source resolution in the subgraph sweep used `assume_sync=true`, so MUL_MAT_ID outputs resolved as MIRRORED instead of PARTIAL.
Fix: GLU op source resolution uses `assume_sync=false`. Also relaxed `handle_mul_mat_id` assertion to accept PARTIAL src1 (down projection input).

**Bug 4: Fused aggregation kernel (CURRENT BLOCKER)**
Per-expert matmul outputs are verified correct (nonzero at correct slots, zero for non-owned experts). The `fused-expert-agg.cpp` kernel that replaces MUL+VIEW+ADD reads from the correct buffer but produces zeros. Likely doesn't handle EP's sparse expert slots.
Evidence: `FUSED_AGG` kernel fires, reads same buffer pointer as MUL_MAT_ID wrote to, but AllReduce input shows GPU0 = all zeros despite having 2 nonzero expert slots.

**Verification chain (all confirmed correct):**
- Weight data integrity ✅
- Expert routing selection ✅
- Pre-zeroing ✅
- Per-expert matmul outputs (verified per-slot) ✅
- get_i_delayed boundary extension (nodes 46→62) ✅
- Buffer pointers (FUSED_AGG reads same buffer MUL_MAT_ID wrote to) ✅
- Dense TP on eptp build ✅

### REAM Model Red Herring — ~12 Hours Wasted

EP was originally tested on `Qwen3-30B-A3B-REAM-heretic-i1` (96 experts). It appeared to garble only with EP. After extensive investigation, MiniMax discovered the REAM model garbles on ALL builds including stable TP. The "EP bug" was actually a broken model.

Models tested and rejected:
- REAM-heretic-i1 Q4_K_M: garbled on all builds — DELETED
- REAM-heretic Instruct-2507 Q4_K_M: also garbled — DELETED
- REAP pruned 15B Q4_K_M: pruning destroyed quality — DELETED

The 3.95 t/s EP result from the original EP commit was measured on the broken REAM model and is meaningless. EP has never produced clean output on any valid model.

### Qwen3.5 35B MoE — New Architecture, Doesn't Work Yet

Worktree: `llama.cpp-qwen35` (branch: `qwen35-support`)
Architecture: `qwen35moe` — 256 experts, Gated Delta Net attention

Two models tested (`HauhauCS-Aggressive`, `heretic-v2`), both garble at np=1 on a clean build from stable-baseline. The garbling is an upstream SYCL issue with Gated Delta Net attention, not our code. This is a separate workstream.

### Project Reorganization (March 20)

- Removed stale worktrees: `llama.cpp-expert`, `llama.cpp-tp` (branches preserved in git)
- Consolidated CLAUDE.md as single project overview
- Cleaned task board: 32 done tasks archived, fresh priorities
- Gitignored worktrees, caches, agent memory
- Updated journey.md with Chapter 14 (TP optimization marathon)

### Thinking Mode Issue

Qwen3 MoE has built-in `<think>` reasoning that consumes most token budget before producing visible content. Without `--reasoning-budget 0`, the model generates 150+ tokens of thinking and no visible reply. Critical for Discord bot use. The `/think` slash command enables reasoning on-demand with `reasoning_budget=4096`.

## 1. FUSED_MMQ + MoE: CONTRADICTS FLAGSHIP.md

BOOTSTRAP says: "FUSED_MMQ crashes MoE at np=16 (server dies at 9.3s)"
FLAGSHIP.md says: "effect on MoE untested but shouldn't hurt"

The proxy currently runs with `GGML_SYCL_FUSED_MMQ=1` and MoE loads fine through it (verified — proxy served MoE at 14 t/s through FUSED_MMQ=1 env). Either the crash was in a specific test config that isn't documented, or it's been fixed. **Needs verification:** run the bench framework `test_moe_churn.py` with FUSED_MMQ=1 explicitly and capture the crash log if it fails.

## 2. llama.cpp-tp Worktree: DELETED

Section 3 references `llama.cpp-tp` worktree. We deleted this worktree on March 20 during the project reorganization. The branch still exists in git (`tensor-parallelism-upstream`) but the worktree is gone. Remove the section or note it's archived. The TP meta backend work now lives in `llama.cpp-eptp` (ep-tp-combined branch).

## 3. Context Limits: INCOMPLETE

BOOTSTRAP says "c=2048-4096 verified." Actual test results:
- c=512 np=16: ✅ 39.4 t/s (benchmark)
- c=8192 np=1: ✅ 13.7 t/s gen, 37.2 t/s prompt processing (tested 500 tokens)
- c=8192 np=4: ✅ works (proxy config, tested)
- c=4096 np=4: ✅ works (proxy direct test)
- c=32768 np=4: ❌ crashes (VRAM exhaustion — 17GB model + 4×32K KV > 48GB)

The limit is **np × context dependent**, not a flat cap. c=8192 is safe at np=4. The doc should say "c=8192 verified at np≤4" not "c=2048-4096."

## 4. "NOT F16 overflow (that was fixed)": NEEDS SOURCING

BOOTSTRAP claims the F16 overflow bug was fixed and the real issue is L0 queue race. Journey.md Chapter 12 documents the F16 overflow as root cause: `graph[26] op=CPY idx=1/512 raw=0xfc00 (F16 -inf)`. I have no record of a fix landing for this in any commit. If you fixed it during the MiniMax session or separate testing, document WHICH commit fixed it. Otherwise the claim "that was fixed" is unverified and the two bugs may be the same thing described differently.

## 5. "~480 requests/min": MATH DOESN'T ADD UP

BOOTSTRAP says "np=16 × c=2048 × max_tokens=200 → 25.5 t/s, ~480 requests/min."

Calculation: 25.5 t/s aggregate. Each request generates ~200 tokens. Time per request per slot ≈ 200/25.5×16 ≈ 125s... that's not right either. The aggregate 25.5 t/s means all 16 slots together produce 25.5 tokens/second. So: 25.5 t/s ÷ 200 tokens/request = 0.1275 requests/second = **7.65 requests/minute**. Not 480. Even if you meant per-slot with prompt caching and short prompts, 480/min is way off. Check the math.

## 6. EP Section: STALE

The EP section says "Blocked on fused aggregation kernel bug." This is current but missing critical progress from the March 20-21 session:

**Fixed bugs (not mentioned):**
- AllReduce deferral: GLU op between gate/up couldn't resolve split state → fixed by looking ahead for MUL_MAT_ID instead of resolving next-node
- Rotation: `il % n_devices` rotated expert distribution per layer → fixed with `effective_rotation=0` for SPLIT_AXIS_2
- GLU split state: `handle_mul_mat_id` returned MIRRORED when `assume_sync=true` → fixed by using `assume_sync=false` for GLU source resolution
- `handle_mul_mat_id` assertion: relaxed to accept PARTIAL src1 (down projection input from GLU)

**Current state:** Per-expert matmul outputs verified correct on all GPUs. The fused aggregation kernel (`fused-expert-agg.cpp`) produces zeros — this is the remaining blocker. All other EP data paths verified correct. See `llama.cpp-eptp` commits for details.

## 7. Missing: Thinking Mode / reasoning-budget

No mention of `--reasoning-budget`. The MoE model (Qwen3) has built-in `<think>` mode that consumes most of the token budget before producing visible content. Without `--reasoning-budget 0`, the model is unusable for conversational Discord — it thinks for 150+ tokens and produces no visible reply. FLAGSHIP.md documents this; BOOTSTRAP should too.

## 8. Missing: qwen35moe Architecture

We set up a new worktree `llama.cpp-qwen35` (branch: `qwen35-support`) for Qwen3.5 35B MoE models. These use `qwen35moe` architecture (256 experts, Gated Delta Net attention) and garble on the current build — it's an upstream SYCL issue, not our code. Two 35B models tested and both garble at np=1. This is a separate workstream from the Qwen3 30B MoE.
