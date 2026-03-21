# BOOTSTRAP.md Corrections

Audit by Orion (OpenClaw main agent) — has full session context from March 20-21 debugging.

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
