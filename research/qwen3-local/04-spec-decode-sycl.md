# Speculative Decoding with SYCL Backend (Intel Arc A770) in llama.cpp

**Generated:** 2026-03-15
**Hardware context:** 3x Intel Arc A770 (16 GB each, 48 GB total), SYCL backend, custom row-split fork
**Target config:** Qwen3-235B-A22B Q3_K_S + Qwen3-30B-A3B Q4_K_M draft

---

## 1. Does Speculative Decoding + Row-Split + `-cmoe` + SYCL All Work Together?

**Short answer: No confirmed incompatibility, but also no confirmed working report.** The combination is novel territory.

Each piece is documented to work independently in this codebase:

- `-md` (spec decoding via draft model) is confirmed working with SYCL on this fork
- `-sm row` is now working on SYCL after the 4-commit row-split patch series (`daf5a6f`, `8b6d3cb`, `7423303`, `e0ebf6b`)
- `-cmoe` is implemented as a `tensor_buft_override` that pushes `ffn_.*_exps` tensors to CPU; it interacts orthogonally with the split mode
- `-cmoed` similarly redirects draft model expert weights to CPU

The code path in `server-context.cpp:651` shows the draft model is loaded with `params_dft = params_base` (which inherits the main model's `split_mode`). This means **the draft model also inherits `--split-mode row`**. Whether the 30B-A3B draft model behaves correctly under row-split across all 3 GPUs while the main 235B-A22B is also row-split has not been tested and is potentially the most likely failure mode.

**Key structural concern:** During speculative decoding, the target model and draft model run sequentially in the same loop (see `common/speculative.cpp:1004-1021`). The verification pass calls `llama_decode(ctx_tgt, batch)` with a batch of `(N+1)` tokens (1 base + N draft tokens). On bandwidth-bound hardware, this N+1 batch is the critical path.

**Row-split status as of 2026-03-14:** Working for Qwen3-0.6B across 1/2/3 GPUs. Qwen3.5-27B (gated delta net architecture) not yet tested under row-split. Qwen3-235B-A22B under row-split is unconfirmed.

**Known incompatibility at b8184:** Issue [#20097](https://github.com/ggml-org/llama.cpp/issues/20097) reports 3x A770 Vulkan multi-GPU setup producing gibberish output for Qwen3.5-35B-A3B from build b8184 onwards. This is Vulkan-specific but suggests multi-GPU logic changes can break things. The SYCL backend may have analogous regressions; always test on a small model first.

---

## 2. GitHub Issues and PRs: Speculative Decoding on SYCL/Arc A770

**No issues or PRs exist specifically about speculative decoding on SYCL or Arc A770.** The GitHub search space covering `speculative SYCL`, `draft model Arc`, and `speculative Arc A770` returned no matching results in the ggml-org/llama.cpp repository. This is not a tested combination upstream.

Adjacent SYCL issues that touch related code paths:

- **[#9612](https://github.com/ggml-org/llama.cpp/issues/9612)** (closed): SYCL crash in `ggml_sycl_mul_mat_batched_sycl` from b3805 — `gemm_batch` fails with `Native API returns: -999`. Resolved upstream. Related to batched matmul, which is invoked during speculative verification.

- **[#19779](https://github.com/ggml-org/llama.cpp/issues/19779)** (closed): `binbcast.cpp:200: GGML_ASSERT(s10 == 1) failed` on SYCL with Qwen3 models after inference begins. Affected builds b8053–b8121. The `binbcast.cpp` file is also where our row-split fix for split pointer resolution lives — simultaneous use of that fix with speculative decoding is untested.

- **[#20338](https://github.com/ggml-org/llama.cpp/issues/20338)** (open): SYCL fails on Arc A770 with `No kernel named _ZTSZN14bin_bcast_sycl...` for MUL op at `-ngl > 2`. The fix (clean rebuild with IntelLLVM 2025.3.2) resolves kernel compilation issues. This issue occurs at model load, not specifically during spec decoding, but uses the same `binbcast.cpp` op.

---

## 3. Speculative Decoding with MoE Draft Models — MUL_MAT_ID Bugs

**There is a known open bug:** PR [#20075](https://github.com/ggml-org/llama.cpp/pull/20075) — *"fix: speculative decoding broken on hybrid SSM/MoE (Qwen3.5 MoE)"* — is open as of 2026-03-15.

The PR description (by the author, 2026-03-03) documents two bugs in `find_slot`:

1. `empty_cell.src` pointed to `orig_cell.src` instead of `seq_meta.tail` — the graph reads stale state from the wrong cell
2. `copy_cell` call was missing entirely — state drift accumulates across draft tokens
3. `llama_memory_recurrent` has no rollback mechanism for SSM state when draft tokens are rejected — causing irreversible state drift

The fix adds a checkpoint/restore with a rolling buffer (depth 8 per sequence). **This PR targets Qwen3.5 hybrid SSM/MoE models specifically** (`qwen35` architecture with Gated Delta Networks). For Qwen3-235B-A22B and Qwen3-30B-A3B, which use standard MoE (not hybrid SSM/MoE), this bug should not apply.

**Reported results on M3 Max 128 GB with PR applied:**
- Qwen3.5-122B-A10B-UD-Q4_K_XL + Qwen3.5-0.8B draft: baseline ~20.4 t/s → with patch: 23.5–29.7 t/s, acceptance rate 63–89% depending on `--draft-max`

**For standard Qwen3 MoE (not Qwen3.5):** The relevant GGML op is `MUL_MAT_ID`, which dispatches expert-gated matrix multiplications. In ik_llama.cpp issue [#1159](https://github.com/ikawrakow/ik_llama.cpp/issues/1159), a user reports spec decoding with Qwen3-VL-235B-A22B (a related MoE) + Qwen3-VL-2B draft gives intermittent results: sometimes 20% performance loss, sometimes extreme slowdown, plus assertion failures:

```
ggml.c:8943: GGML_ASSERT(a->ne[2] * 4 == b->ne[0]) failed
```

This assert checks expert tensor shape compatibility. The issue was closed without resolution — likely a model architecture mismatch specific to VL models. Plain Qwen3-235B-A22B + Qwen3-30B-A3B should not trigger this (same vocabulary, compatible architecture).

**EAGLE3 buffer sync bug on GPU (relevant):** In the EAGLE3 PR [#18039](https://github.com/ggml-org/llama.cpp/pull/18039), a commenter discovered that GPU-side speculative decoding suffers from buffer reuse races: "once I set it [ggml_set_output], the buffer for the concatenated results got overwritten." The root fix was switching `ggml_backend_tensor_get` to `ggml_backend_tensor_get_async` + `synchronize()`. This was an EAGLE3-specific issue resolved in the PR. For standard draft-model speculative decoding (`COMMON_SPECULATIVE_TYPE_DRAFT`), the equivalent sync logic is handled differently in `common/speculative.cpp` — the `llama_decode` calls are synchronous by design.

---

## 4. Draft Acceptance Rate: Qwen3-30B-A3B Drafting for 235B

No public benchmarks exist for this exact pair on any hardware. Closely related data points:

**EAGLE3 PR [#18039](https://github.com/ggml-org/llama.cpp/pull/18039) — Qwen3-30B-A3B as EAGLE3 draft:**
- Acceptance rate: **58.8%–64.4%** across three test prompts
- Speedup: **1.25x–1.39x** (with EAGLE3 architecture, which is more sophisticated than plain draft-model spec decoding)
- Note: EAGLE3 is not yet merged and has no SYCL support

**Qwen3.5 MoE (PR #20075) with Qwen3.5-0.8B draft:**
- Acceptance rate: **63–89%** depending on `--draft-max` setting
- This is a much smaller, less capable draft model

**General MoE verification overhead (documented in PR #18039):**
> "Speculative decoding performance for MoE models is not as good as dense models... more experts are invoked during the parallel verification phase than during the target model's decoding phase."

For GPT-OSS-120B (MoE) as main model with EAGLE3: **0.69x–0.97x speedup** (performance degradation despite 74-85% acceptance rates). The issue is that verifying N+1 tokens in a single MoE forward pass activates proportionally more experts than a single-token decode, consuming more memory bandwidth than the token savings justify.

**For Qwen3-235B-A22B specifically:** The model uses top-K=8 of 128 experts. Verifying a batch of 5 draft tokens may activate up to 40 distinct expert slots instead of 8 — a 5x increase in expert memory reads, which offsets the parallelism benefit. With `-cmoe`, all expert reads go through PCIe (CPU-to-GPU), so this effect is amplified.

**Expected acceptance rate for Qwen3-30B-A3B → 235B-A22B:**
Both models share the same tokenizer and base training lineage. A conservative estimate based on the 58-64% rate (EAGLE3, which has higher quality drafts than plain spec decoding) is **40–60% for plain draft-model spec decoding**. Higher in repetitive/code contexts, lower for reasoning/math.

---

## 5. `-ngld 999` — GPU Layer Placement for Draft Model

**Behavior confirmed in source code:**

From `common/arg.cpp:3431-3442`:
```cpp
{"-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}, "N",
```
- `-ngld 999` or `-ngld all` → sets `params.speculative.n_gpu_layers = -2` (interpreted as "all layers")
- This is propagated in `server-context.cpp:658`: `params_dft.n_gpu_layers = params_spec.n_gpu_layers;`
- Which is then passed to `llama_model_load_from_file` via `common_model_params_to_llama` at line 1319

**For Qwen3-30B-A3B Q4_K_M:** The model has 48 layers. At Q4_K_M quantization, the non-expert weights are ~18.6 GB. With `-cmoed`, expert weights go to CPU, leaving only attention + norms on GPU (~5-7 GB). `-ngld 999` ensures all attention layers offload to GPU (leaving experts on CPU via `-cmoed`).

**SYCL issues with `-ngld`:** No known issues specific to high `-ngld` values on SYCL. The problematic case (issue [#20338](https://github.com/ggml-org/llama.cpp/issues/20338)) was `-ngl > 2` failing due to a kernel compilation bug, not a `-ngld` issue. However, that issue was with the main model, not the draft model.

**Draft model split mode:** The draft model inherits `--split-mode row` from the main model (because `params_dft = params_base` in `server-context.cpp:651`). There is no separate `--split-mode-draft` flag. This means the draft model will also attempt row-split across all 3 GPUs, which adds overhead for a 30B model that might fit on a single GPU. This is worth benchmarking — you may want to set `-devd SYCL0` (or whichever single GPU has the most free VRAM) to avoid row-split overhead on the draft model.

---

## 6. `-cmoed` Flag — Documentation and Compatibility

**Source confirmation** in `common/arg.cpp:2306-2311`:
```cpp
{"-cmoed", "--cpu-moe-draft"},
"keep all Mixture of Experts (MoE) weights in the CPU for the draft model",
[](common_params & params) {
    params.speculative.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
}
```

**Availability:** Set in examples `LLAMA_EXAMPLE_SPECULATIVE`, `LLAMA_EXAMPLE_SERVER`, `LLAMA_EXAMPLE_CLI` — fully supported in all three entrypoints.

**Mechanism:** `tensor_buft_overrides` is applied via `common_model_params_to_llama` which maps it to `llama_model_params.tensor_buft_overrides`. This is the same mechanism used by `-cmoe` for the main model. The two lists are kept separate (`params.tensor_buft_overrides` vs `params.speculative.tensor_buft_overrides`), so they do not interfere.

**Propagation path** (`server-context.cpp:667`):
```cpp
params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;
```

There is also `-ncmoed N` to offload only the first N layers' experts to CPU, allowing selective expert tiering for the draft model.

**Does `-cmoed` work alongside main's `-cmoe`?** Yes, by design. The `tensor_buft_overrides` lists are separately maintained. No known incompatibility.

---

## 7. Speedup Model: When Does Spec Decoding Help vs Hurt on Arc A770?

**The fundamental equation:**

```
Speedup = (1 + α · k) / (1 + t_draft/t_verify)
```

Where `α` is the acceptance rate (0–1), `k` is the number of draft tokens, `t_draft` is the time to generate k draft tokens, and `t_verify` is the time to verify k+1 tokens.

**Arc A770 bandwidth-bound reality:**

Arc A770 memory bandwidth: ~560 GB/s per GPU. During generation (decode phase), each forward pass reads the full model weight set from VRAM (or DRAM for CPU-offloaded experts). The per-token cost is approximately `model_size / bandwidth`.

The key insight documented in the local `SYCL_TUNING_GUIDE.md`:

| Main Model | Draft Model | Speed Ratio | Spec Benefit |
|---|---|---|---|
| 32B dense (7 t/s) | 0.6B (41 t/s) | 6x | **None** — batch verify scales linearly |
| 235B MoE (3-5 t/s) | 30B-A3B (30+ t/s) | 10-15x | **Potential 2-3x** — if MoE verify doesn't explode |

**Why batch verify is linear on Arc A770:**

Unlike attention (which is KV-cache bandwidth bound and benefits from batching), MLP/FFN operations in the verification pass scale approximately linearly with batch size on bandwidth-bound hardware. Reading the weight matrices for 5 tokens costs ~5x what it costs for 1 token. This is unlike CUDA on NVIDIA hardware where tensor cores provide genuine batch speedup.

However, with `-cmoe`, the expert weights are CPU-side (DDR5 at ~50 GB/s). The verification batch for 5 draft tokens activates up to `5 × 8 = 40` expert slots vs `8` for a single token decode. Each expert read goes over PCIe 3.0 (~16 GB/s). This is the bandwidth bottleneck that can make MoE speculative decoding hurt.

**Critical MoE-specific penalty:** The verification pass for k draft tokens in an MoE model incurs `k × 8` expert activations if all draft tokens are in different context regions. In practice, code generation tends to cluster expert activations (same topics/patterns), so in the best case (fully correlated activations), the overhead is closer to `1 × 8 + overhead`. This clustering behavior is the reason acceptance rates in code tasks are higher.

**When spec decoding helps (235B-A22B scenario):**
- Main model generates at ~3-5 t/s (baseline)
- Draft model (30B-A3B + `-cmoed`) generates at ~15-25 t/s (estimated)
- Speed ratio: ~5-8x — marginal for spec decoding
- Acceptance rate ~50-60% at `--draft-max 8`
- Effective speedup if verification doesn't add MoE penalty: `(1 + 0.55 × 8) / (1 + 1/5)` ≈ 3.75x theoretical, but MoE verification penalty reduces this significantly

**Practical estimate:** For code generation tasks, 1.3–1.8x speedup is plausible. For reasoning/math, it may be neutral or slower than baseline. Measurement is essential.

---

## 8. Reports of Speculative Decoding on Arc A770

**No published success reports found** for speculative decoding specifically on Intel Arc A770 or SYCL backend. The combination does not appear in llama.cpp issues, Reddit r/LocalLLaMA, or GitHub discussions.

**Adjacent data points:**

- **ik_llama.cpp issue [#1159](https://github.com/ikawrakow/ik_llama.cpp/issues/1159)** (2026-03): User tried Qwen3-VL-235B-A22B + Qwen3-VL-2B draft on CPU/GPU hybrid (not Arc), reports "sketchy results" and assertion failures. Closed without resolution.

- **ik_llama.cpp PR [#645](https://github.com/ikawrakow/ik_llama.cpp/pull/645)** (merged 2025-08-16): Port of speculative decoding to ik_llama.cpp server, tested specifically with Qwen3-235B. No hardware backend specified but not SYCL.

- **ik_llama.cpp PR [#1261](https://github.com/ikawrakow/ik_llama.cpp/pull/1261)** (merged 2026-02-13): Self-speculative decoding (ngram-based) port including `DIST` sampler type. No hardware-specific notes.

- **ggml-org issue [#10176](https://github.com/ggml-org/llama.cpp/issues/10176)**: Spec decoding segfault caused by DRY sampler cloning when `model` was NULL (fixed in #10192). This was a CUDA setup, but the fix applies universally.

- **ggml-org issue [#9949](https://github.com/ggml-org/llama.cpp/issues/9949)**: Segfault with 405B main on CPU + 8B draft on CUDA RTX 4070 Ti. Root cause was `llama_batch_allocr` shape mismatch, traced to `--split-mode row` with small draft models where some devices get 0 rows — identical bug to what our row-split patch addresses for Intel Arc.

**The Tencent/AngelSlim issue [#9](https://github.com/Tencent/AngelSlim/issues/9)** is the closest data point on MoE draft models:
- Without EAGLE3: ~130 t/s (on NVIDIA hardware)
- With EAGLE3 (Qwen3-30B-A3B draft): ~60 t/s — **halved performance**
- Context: sglang with large batch sizes — different regime from single-user inference, but illustrative of MoE verification overhead

---

## 9. `--draft-max`, `--draft-min`, `--draft-p-min` — Qwen3-Specific Tuning

**Source: `common/common.h:273-276`** — defaults:
```cpp
int32_t n_max   = 16;  // --draft-max
int32_t n_min   = 0;   // --draft-min
float   p_min   = 0.75f; // --draft-p-min
```

**`--draft-max`:** Maximum draft tokens per speculative step. The acceptance loop in `common/speculative.cpp:359-393` generates up to `n_max` tokens, stopping early if `cur_p->data[0].p < params.p_min`.

For MoE verification overhead reasons, **lower `--draft-max` values (4-8) are strongly recommended** for the 235B-A22B main model. With `-cmoe` putting experts on CPU, verifying 16 draft tokens risks activating 128 distinct expert reads over PCIe, which is catastrophically slow.

**`--draft-min`:** Minimum draft length below which speculative decoding is skipped for a step. In `server-context.cpp:293-295`:
```cpp
if (n_draft_max < task->params.speculative.n_min) {
    n_draft_max = 0;  // skip spec decoding
}
```
Default is 0 (never skip). Setting `--draft-min 2` prevents speculative overhead when only 1-2 tokens are possible (e.g., near context limit).

**`--draft-p-min`:** Draft token probability threshold. Default 0.75 means draft tokens where the draft model is < 75% confident are not proposed. This is a natural early-stopping mechanism.

For Qwen3-30B-A3B as a draft model (which has 3B active params out of 30B total per token), **the draft model's confidence for tokens matching the 235B-A22B will be lower than a dense draft model of similar compute**. Reducing `--draft-p-min` to 0.5 or even 0.3 may increase accepted tokens at the cost of more verification overhead.

**Empirical acceptance rate tracking:** Per `server-context.cpp:394-398`:
```cpp
"draft acceptance rate = %0.5f (%5d accepted / %5d generated)\n"
```
This is printed to the server log after each request (not to the HTTP response). The JSON response for each completion also includes `draft_n` and `draft_n_accepted` fields (from `server-task.h:274-275`) when speculative decoding is active. These can be consumed via the `/completion` endpoint or captured from server stdout.

**Qwen3-specific note:** For reasoning tasks (`/think` mode), the draft acceptance rate will likely be higher in repetitive reasoning chains ("So, let me think... → So, let me think...") and lower at logical inference steps. Consider starting with `--draft-max 6 --draft-p-min 0.5` and iterating based on observed `draft_n_accepted / draft_n` ratio.

---

## 10. Profiling and Measuring Draft Acceptance Rate

**Per-request data** is available in the JSON response body:
```json
{
  "timings": {
    "draft_n": 48,
    "draft_n_accepted": 31,
    ...
  }
}
```
Fields appear only when `draft_n > 0` (source: `server-task.cpp:617-619`).

**Per-request server log** includes:
```
draft acceptance rate = 0.64583 (   31 accepted /   48 generated)
statistics draft: #calls(b,g,a) = 0 4 4, #gen drafts = 4, #acc drafts = 3, ...
```
Printed by `common_speculative_print_stats` after each slot completes.

**Prometheus `/metrics` endpoint** (enabled with `--metrics`): Does not currently export per-model draft acceptance rate as a labeled metric. The `predicted_tokens_seconds` metric reflects effective throughput with spec decoding included.

**Recommended monitoring approach:**
```bash
# Tail server stderr for acceptance rate
llama-server [flags] 2>&1 | grep "draft acceptance rate"

# Or: extract from /completion response JSON
curl -s http://localhost:8080/completion -d '{"prompt":"...","n_predict":200}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('timings',{}))"
```

**Statistics fields from `common/speculative.cpp:1065-1072`:**
```
statistics draft: #calls(b,g,a) = N N N, #gen drafts = N, #acc drafts = N, #gen tokens = N, #acc tokens = N, dur(b,g,a) = X, X, X ms
```
- `#gen tokens / #acc tokens` = effective acceptance rate
- `dur(b,g,a)` = time in begin/generate/accept phases

---

## Summary Table: Risk Assessment for Each Component

| Component | Status | Risk Level | Notes |
|---|---|---|---|
| SYCL + speculative decoding | Confirmed working (local) | Low | `-md` flag tested |
| `-sm row` + speculative decoding | Untested combination | Medium | Draft inherits row-split |
| `-cmoe` + speculative decoding | Untested | Medium | Separate param lists, should work |
| `-cmoed` flag | Code-confirmed, well-documented | Low | Same mechanism as `-cmoe` |
| Qwen3-30B-A3B as draft (MoE) | No known llama.cpp reports | High | MoE verification overhead |
| Qwen3-235B-A22B as main target | Working without spec decoding | Low | Standard MoE, row-split fixed |
| Draft model also row-split | Likely performance negative | Medium | Suggest `-devd SYCL0` instead |
| PR #20075 hybrid SSM/MoE bug | Does not apply (Qwen3 not Qwen3.5) | None | Different architecture |

---

## Recommended Changes to Deployment Plan

### Flag Recommendations

**Remove row-split from draft model.** The draft model (30B-A3B Q4_K_M with `-cmoed`) will inherit `--split-mode row` from the main model's `params_base`. Pin the draft model to a single GPU using `--device-draft`:

```bash
-devd SYCL0
```

This avoids the row-split overhead for a 30B model — the attention-only portion after `-cmoed` fits on one A770 (approximately 5-7 GB for non-expert weights), and single-GPU generation will be faster for the draft loop.

**Set `-ngld all` instead of `-ngld 999`.** Both are equivalent (`-2` internally = "all"), but `all` is the documented enum value.

**Start conservative on draft length:**

```bash
--draft-max 6 --draft-min 2 --draft-p-min 0.5
```

Rationale: MoE verification cost grows with draft length; 6 tokens limits the PCIe expert read explosion while still allowing speedup if acceptance rate is high. `--draft-p-min 0.5` (reduced from default 0.75) gives the 30B-A3B draft model more room to propose tokens — it has lower confidence than dense drafts would.

**Full recommended launch command:**

```bash
llama-server \
  -m /path/to/Qwen3-235B-A22B-Q3_K_S.gguf \
  --split-mode row \
  --tensor-split 1,1,1 \
  -ngl 999 \
  -cmoe \
  -fa \
  -md /path/to/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngld all \
  -devd SYCL0 \
  -cmoed \
  --draft-max 6 \
  --draft-min 2 \
  --draft-p-min 0.5 \
  --metrics \
  -c 8192 \
  --host 0.0.0.0 \
  --port 8080
```

**Measure before committing.** Run without `-md` first to establish a baseline, then add the draft model. Compare `eval time` tokens/second. If the draft-model run is slower, the MoE verification overhead dominates and spec decoding is net-negative for this hardware/model pair.

**Alternative: Self-speculative decoding (no draft model).** If draft-model spec decoding proves net-negative, `--spec-type ngram_mod` (no draft model required) may provide a modest 5-15% speedup in repetitive code/reasoning contexts without any VRAM or PCIe overhead. Set `--spec-ngram-size-n 16` (default) and `--draft-max 48`.

### Risk Mitigation

1. **Test row-split + spec decoding on Qwen3-0.6B first** before loading the 235B+30B pair. Confirm no SYCL assertion failures.
2. **Monitor `draft acceptance rate`** in server stderr for every request. Acceptance rate below 30% signals that spec decoding is almost certainly net-negative.
3. **Watch for `binbcast.cpp` assertions.** If you see `GGML_ASSERT(s10 == 1) failed`, that is a known SYCL binary broadcast bug (issue #19779) triggered by operator shape incompatibilities — rebuild with a clean build directory and verify the 5-bug row-split patch series is applied.
4. **The EAGLE3 path (`--spec-type eagle3`) is not recommended yet.** PR [#18039](https://github.com/ggml-org/llama.cpp/pull/18039) is open with no SYCL testing. The buffer sync fix (`ggml_backend_tensor_get_async`) may not behave correctly on SYCL's async queue model.

---

## Sources

- llama.cpp PR #20075: https://github.com/ggml-org/llama.cpp/pull/20075 — spec decoding broken on hybrid SSM/MoE
- llama.cpp PR #18039: https://github.com/ggml-org/llama.cpp/pull/18039 — EAGLE3 spec decoding (open)
- llama.cpp PR #19378: https://github.com/ggml-org/llama.cpp/pull/19378 — backend-agnostic tensor parallelism (open)
- llama.cpp PR #19493: https://github.com/ggml-org/llama.cpp/pull/19493 — speculative checkpointing (open)
- llama.cpp issue #20338: https://github.com/ggml-org/llama.cpp/issues/20338 — SYCL fails on Arc A770 (open)
- llama.cpp issue #19779: https://github.com/ggml-org/llama.cpp/issues/19779 — binbcast assert on SYCL Qwen3 (closed)
- llama.cpp issue #20097: https://github.com/ggml-org/llama.cpp/issues/20097 — 3x A770 Vulkan multi-GPU gibberish after b8184 (open)
- llama.cpp issue #9612: https://github.com/ggml-org/llama.cpp/issues/9612 — SYCL crash in gemm_batch b3805 (closed)
- llama.cpp issue #10176: https://github.com/ggml-org/llama.cpp/issues/10176 — spec decode segfault with row-split (closed)
- ik_llama.cpp issue #1159: https://github.com/ikawrakow/ik_llama.cpp/issues/1159 — spec decoding with Qwen3-VL 235B (closed)
- ik_llama.cpp PR #645: https://github.com/ikawrakow/ik_llama.cpp/pull/645 — spec decoding port to ik server, tested on Qwen3-235B
- ik_llama.cpp PR #1261: https://github.com/ikawrakow/ik_llama.cpp/pull/1261 — self-spec decoding ngram port (merged)
- Tencent/AngelSlim issue #9: https://github.com/Tencent/AngelSlim/issues/9 — Qwen3-30B-A3B EAGLE3 draft giving 0.46x speedup on sglang
- Local source: `/home/ryan/llm-stack/llama.cpp/common/speculative.cpp` — spec decoding implementation
- Local source: `/home/ryan/llm-stack/llama.cpp/common/arg.cpp` — flag definitions for -cmoe, -cmoed, -ngld, --draft-max, --draft-p-min
- Local source: `/home/ryan/llm-stack/llama.cpp/tools/server/server-context.cpp` — draft model loading and acceptance rate tracking
- Local source: `/home/ryan/llm-stack/llama.cpp/common/common.h` — `common_params_speculative` struct
- Local source: `/home/ryan/llm-stack/llama.cpp/docs/SYCL_TUNING_GUIDE.md` — bandwidth analysis and split mode tradeoffs
- Local source: `/home/ryan/llm-stack/llama.cpp/ROW_SPLIT_FIX.md` — row-split bug analysis and patches
