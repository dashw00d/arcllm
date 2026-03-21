# Prompt Caching / KV Cache Reuse on SYCL Backend in llama.cpp

**Date:** 2026-03-15
**Hardware:** 3x Intel Arc A770 16 GB, SYCL backend
**Config under analysis:** `--cache-reuse 256 -np 2`, `id_slot` pinning, frozen system prompts
**Sources:** local llama.cpp source at `/home/ryan/llm-stack/llama.cpp` (main branch, commit e0ebf6b)

---

## 1. What Does `--cache-reuse N` Mean?

**N is the minimum contiguous matching token run required to trigger a KV shift-and-reuse.**

Source: `common/common.h:535`
```
int32_t n_cache_reuse = 0;  // min chunk size to reuse from the cache via KV shifting
```

Source: `common/arg.cpp:2954-2962`
```
"min chunk size to attempt reusing from the cache via KV shifting, requires prompt caching to be enabled"
```

**Algorithm** (`tools/server/server-context.cpp:2251-2298`):

After the simple prefix match (`cache_prompt` path) locates `n_past` common prefix tokens, the KV-shift-reuse scan kicks in:

1. Walk two cursors, one through the *old* cached prompt (`head_c`) and one through the *new* prompt (`head_p`), both starting at `n_past` (where the prefix diverged).
2. Count consecutive matching tokens (`n_match`).
3. If `n_match >= n_cache_reuse`, the matching KV window is *shifted* to its new position in the prompt via `llama_memory_seq_rm` + `llama_memory_seq_add` with a position delta (`kv_shift = head_p - head_c`).
4. Advance both cursors by `n_match`; if the run is too short, advance `head_c` by 1 and re-try.

**Practical meaning with `--cache-reuse 256`:**
Any run of 256+ tokens that appears verbatim in the new prompt but was displaced (e.g. the system prompt now has user-message prefix before it) will be reused via a RoPE K-shift rather than re-evaluated. Smaller diverged runs are discarded.

**This is NOT prefix-only caching.** Standard `cache_prompt` (always enabled by default) handles prefix reuse without shifting. `cache_reuse` adds non-prefix chunk reuse via K-shift. They are complementary.

---

## 2. Does KV Cache Reuse Work on SYCL?

**Yes, it works, with one important gating check.**

`cache_reuse` is gated on `llama_memory_can_shift()` (`tools/server/server-context.cpp:2242-2248`):

```cpp
const bool can_cache_reuse =
    llama_memory_can_shift(llama_get_memory(ctx)) &&
    !slot.prompt.tokens.has_mtmd;
```

`llama_memory_can_shift` calls `get_can_shift()` on the KV cache object (`src/llama-kv-cache.cpp:976-985`):

```cpp
bool llama_kv_cache::get_can_shift() const {
    if (model.arch == LLM_ARCH_STEP35) return false;
    if (hparams.n_pos_per_embd() > 1) return false;
    return true;
}
```

For Qwen3 (standard transformer, single position per embedding), this returns **true**. The SYCL backend has full `GGML_OP_ROPE` support (`ggml/src/ggml-sycl/ggml-sycl.cpp:7384-7388`) which is what K-shift uses to re-apply RoPE at new positions.

**Also disabled for:** multimodal contexts and any recurrent-state model. Qwen3 triggers neither.

**Runtime sanity check:** at server startup, if `get_can_shift()` returns false, both `ctx_shift` and `n_cache_reuse` are zeroed with warnings (`server-context.cpp:720-729`). If you see no such warnings in your logs, reuse is active.

**Known SYCL-specific issues:** No cache-reuse–specific SYCL bugs exist in the current codebase. The recently fixed row-split bugs (`daf5a6f`, 2026-03-14) affect tensor layout and GEMM correctness but are separate from the KV shift path. The K-shift graph (`build_graph_shift`) only calls `GGML_OP_ROPE` which has a working SYCL implementation.

---

## 3. Cache Persistence with `-np 2`: Both Slots in VRAM?

**Yes — both slot KV caches are in VRAM with no automatic eviction between requests.**

`-np 2` sets `n_parallel = 2`, which maps directly to `cparams.n_seq_max = 2` (`common/common.cpp:1354`). The KV cache is allocated once at startup with `n_ctx * n_seq_max` cells across all layers. For your setup (3x A770, `--kv-offload` enabled by default), all KV cells are on GPU VRAM.

Each slot's KV state lives in its own sequence `seq_id == slot.id`. Sequence data is not evicted between turns. When a slot finishes a request and goes `SLOT_STATE_IDLE`, its KV tensor rows remain in place. The next request to that slot finds `n_past` tokens already cached via the prefix match at `server-context.cpp:2232`.

**What does cause cache clearing:**
- Slot is stolen for a different conversation (only happens with `id_slot` not pinned)
- Context overflow → context-shift discards the oldest `n_discard` tokens
- Explicit `/slots/{id}/action=erase` call
- Server restart

**`--cache-ram N` (new feature, `pr/16391`)** is a *separate* cross-request prompt-state cache in system RAM. This is unrelated to the in-VRAM per-slot KV cache. With `-np 2` and `id_slot` pinning your two consumers are already getting dedicated VRAM slots; `--cache-ram` would add benefit only if the same slot served many different conversations.

---

## 4. `--slot-save-path` and Slot Save/Restore on SYCL

**Slot save/restore works on SYCL via `llama_state_seq_save_file` / `llama_state_seq_load_file`.**

The API is `/slots/{id}/action=save` and `/slots/{id}/action=restore` (requires `--slot-save-path` directory at startup). Source: `server-context.cpp:3432-3453`, `server-context.cpp:3997-4065`.

Save serializes the VRAM KV state for one sequence using `llama_state_seq_get_data_ext`, which internally copies GPU→CPU then writes bytes. Load does the reverse. The mechanism is backend-agnostic: it goes through `llama_io_write_i`/`llama_io_read_i` which do not have SYCL-specific code paths.

**Practical note:** Save/restore involves a full GPU→CPU copy of the KV state (multiple GiB for large contexts). For your 3-GPU row-split topology, the KV tensors live in a row-split split buffer; the state serialization code uses `llama_state_seq_get_size_ext` / `llama_state_seq_get_data_ext`, which pull through the normal KV copy path. If there are issues with row-split KV serialization they would surface here, but no such bugs are currently known.

**Slot save is most useful for** cold-starting long system prompts rather than per-request persistence. Given your frozen system prompts, saving after first warmup and restoring on server restart can eliminate first-request latency.

---

## 5. What Triggers a Cache Miss?

In order of severity:

| Trigger | Effect |
|---|---|
| Token inserted before cached content | Prefix match length drops to the insertion point; `cache_reuse` may recover later chunks if `n_match >= 256` |
| System prompt byte change | Prefix match collapses to 0 or BOS only; entire prompt re-evaluates |
| Slot is assigned to a different conversation | `n_past` = 0 (all new tokens) |
| `cache_prompt = false` sent in request | `n_past` forced to 0 (`server-context.cpp:2299-2302`) |
| Context overflow with `--context-shift` | Oldest tokens discarded; prefix match recalculated against shifted cache |
| `memory_seq_rm` fails on truncation | Cache cleared, `n_prompt_tokens_cache = 0` (`server-context.cpp:2449-2455`) |

**For your frozen system-prompt setup:** as long as system prompt bytes are identical and `id_slot` pinning holds, the prefix match will recover the full system prompt on every turn (prefix = system prompt + prior turns). `cache_reuse` 256 is useful only when the *non-prefix* part of the new prompt contains a 256+ token verbatim block from the prior cached prompt — for example if the conversation turns are restructured without the system prompt prefix changing.

---

## 6. Full Slots with `id_slot`: Queue or Reject?

**Queue (deferred), not reject.**

Source: `server-context.cpp:1706-1711`:
```cpp
if (slot->is_processing()) {
    // if requested slot is unavailable, we defer this task for processing later
    SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
    queue_tasks.defer(std::move(task));
    break;
}
```

The task goes into `queue_tasks_deferred` (`server-queue.h:23`), a `std::deque<server_task>`. When the slot finishes and calls `slot.release()`, `callback_on_release` fires (`server-context.cpp:787-789`), which calls `queue_tasks.pop_deferred_task(id_slot)` — that pops the oldest deferred task for that specific slot id and promotes it to the main queue.

**There is no maximum deferred queue size and no timeout-based rejection.** A client will block (or receive SSE stream silence) until its slot becomes free. For your two-consumer scenario this means the Discord bot never steals the churner's slot and vice versa, but also that a slow churner job will block subsequent churner requests indefinitely.

**Practical implication:** if churner takes 60 s and a second churner request arrives at t=30s, it will wait in the deferred queue until slot 1 is free at t=60s, then immediately start. The HTTP connection stays open.

---

## 7. SYCL `--split-mode row` and KV Cache Location

**KV cache lives on GPU(s), row-split mode changes where model weights are, not where KV lives.**

The KV cache allocation is controlled by `--kv-offload` (default: enabled, `common.h:488`). This is orthogonal to `--split-mode`. With row-split, weight tensors are sharded across 3 GPUs row-wise, but KV tensors are allocated as regular (non-split) buffers on the main device or across devices depending on `offload_kqv`.

Source: `src/llama-context.cpp:314-319`:
```cpp
bool pipeline_parallel =
    model.n_devices() > 1 &&
    model.n_gpu_layers() > model.hparams.n_layer &&
    model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&  // only layer-split, not row-split
    cparams.offload_kqv && ...
```

Row-split does NOT enable pipeline parallelism. KV offload with row-split means the KV cache is on the main GPU (device 0) unless overridden. For 3x A770, this means ~16 GB VRAM on GPU 0 holds the KV cache, while model weights are row-split across all three GPUs.

**Cache reuse path with row-split:** The K-shift graph (`build_graph_shift`) runs RoPE on the KV tensors. Since KV is on a single device (not a split buffer), this avoids the row-split cross-device complications documented in ROW_SPLIT_FIX.md. No known issues.

---

## 8. `--cache-reuse 256` with `--draft-max 8` Speculative Decoding

**Speculative decoding and cache-reuse affect different states and do not conflict directly, but there is a subtle interaction.**

Cache-reuse runs at the **start of a new request** (prompt evaluation phase) when computing `n_past`. Speculative decoding runs during **generation** after `n_past` is established.

When draft tokens are rejected at position `k`, the server does:
```cpp
llama_memory_seq_rm(llama_get_memory(ctx), slot.id, slot.prompt.n_tokens(), -1);
```
(`server-context.cpp:2898`)

This removes only the *speculative tokens beyond the accepted prefix* from the KV cache. The accepted tokens (and all previously cached prompt tokens) remain intact. On the **next request** to the same slot, the prefix match will see the prompt + accepted tokens, and `n_past` will reflect the correct rollback position. Cache-reuse is not involved in the rollback.

**Draft model KV cache** is separate (a different `ctx_dft` context) and uses its own prompt `prompt_dft`. The draft context does its own prefix reuse with `llama_memory_seq_rm`/`llama_memory_seq_add` internally (`common/speculative.cpp:313-323`). The main model's `n_cache_reuse` setting does NOT propagate to the draft context.

**Risk area:** If `--draft-max 8` is combined with very short turns (< 256 tokens), the cache-reuse scan at the start of the next request will not find a 256-token matching chunk and will do no KV-shifting — this is correct behavior, not a bug.

---

## 9. Recent llama.cpp Commits Affecting Cache Reuse or Slot Management on SYCL

Based on `git log` of the local repo (2025-2026):

| Commit | Date | Relevance |
|---|---|---|
| `e0ebf6b` | 2026-03-14 | SYCL row-split: clear recurrent state after reservation |
| `8b6d3cb` | 2026-03-14 | SYCL row-split: SSM pointer resolution + broader tensor replication |
| `daf5a6f` | 2026-03-14 | **SYCL row-split: fix 5 bugs enabling multi-GPU row-split on A770** — does not affect KV cache path |
| `a7b3dee` | 2026-03-10 | `server: make 2 checkpoints near the end of the prompt` — checkpoint placement for long prompts |
| `96cfc49` | 2026-03-09 | `server: fix checkpoints n_tokens calculation` — off-by-one in checkpoint token count |
| `d6e1556` | 2026-03-09 | `server: fix off-by-1 in server_tokens::size_up_to_pos()` — could affect n_past calculation |
| `107d599` | 2026-03-09 | `server: add kill switch when server is stuck` — watchdog, not cache-related |
| `d417bc4` | 2026-03-08 | `server: do not create checkpoints right after mtmd chunks` — multimodal only |
| `59db9a3` | 2026-03-09 | `llama: dynamic head_dim and n_rot for SWA` — affects `get_can_shift()` indirectly for SWA models |

**The `d6e1556` fix (`size_up_to_pos` off-by-one) is worth noting:** an incorrect `n_past` calculation could cause the prefix match to be off by one token, forcing unnecessary re-evaluation of the final cached token. Confirm your build includes this commit.

No upstream commits have landed specifically fixing `--cache-reuse` bugs for SYCL in the 2024-2026 window reviewed. The mechanism is backend-agnostic (pure C++ token comparison + `llama_memory_seq_add` which maps to ROPE op).

---

## 10. Cache Corruption / Incorrect Output Risks on SYCL

No SYCL-specific cache corruption reports are known in the codebase. The mechanisms most likely to cause incorrect output are:

**A. RoPE K-shift on non-standard arch:** `get_can_shift()` returns false for architectures with per-layer RoPE dims (STEP35) or `n_pos_per_embd > 1`. Qwen3 passes both checks. If you load a future model that fails these checks, you will see a logged warning and reuse will be silently disabled.

**B. Quantized K cache with flash attention disabled:** `src/llama-context.cpp:357-360` aborts context creation if quantized V cache is requested without flash attention. No silent corruption.

**C. Row-split MMVQ Q8_1 ds visibility bug (Bug 3 from ROW_SPLIT_FIX.md):** This affects weight-side quantized matmuls under row-split, not the KV cache path. The workaround (force DMMV for split tensors) is applied. Not a cache correctness issue.

**D. `pos_min == -1` abort:** `server-context.cpp:2314-2317` aborts with a clear error message if `n_past > 0` but the sequence has no cells (detects internally inconsistent KV state). This is a defensive check introduced upstream (PR #13833). You will get a hard crash with an informative message rather than silent garbage if this state is reached.

---

## 11. Flash Attention (`-fa`) and Cache Reuse

**Flash attention and cache reuse are independent; `-fa` does not affect `get_can_shift()`.**

`get_can_shift()` checks only model arch and `n_pos_per_embd`. Flash attention is not consulted (`src/llama-kv-cache.cpp:976-985`).

**SYCL flash attention status:** `SYCL_FLASH_ATTN` is compiled in by default (`ggml/src/ggml-sycl/common.hpp:38`). At runtime it can be disabled via `GGML_SYCL_ENABLE_FLASH_ATTN=0`. The SYCL FA implementation (`fattn.cpp`, `fattn-vec.hpp`, `fattn-tile.hpp`) handles standard GQA ratios and non-quantized KV types. For quantized KV (`-ctk q8_0` etc.), the FA kernel falls back if `ggml_is_quantized(K->type)` — quantized KV tensors are filtered at `fattn.cpp:188`.

**Context shift + flash attention:** The `build_graph_shift` (K-shift) uses `GGML_OP_ROPE` independent of whether FA is active for normal decoding. FA is only used in the forward pass, not in the K-shift graph. No interaction.

**Cache reuse + context shift conflict:** Both require `llama_memory_can_shift()`. They are checked at the same gate (`server-context.cpp:720-729`). When active together, context shift discards old tokens (via `seq_rm`/`seq_add`) and cache-reuse shifts chunks for the new prompt. Both call `llama_memory_seq_add` and both work correctly on SYCL.

---

## OpenAI Responses API (`/v1/responses`) and Caching

**Yes, llama.cpp supports `/v1/responses`** (merged, documented in `tools/server/README.md:1286-1326`).

Implementation (`server-context.cpp:3669-3684`): The endpoint converts the Responses request body to a Chat Completions body via `convert_responses_to_chatcmpl()` (`server-common.cpp:1144`), then calls the same `handle_completions_impl` path. **All caching behavior is identical to `/v1/chat/completions`** — prefix matching, `cache_reuse`, and KV slot assignment all apply.

**Limitations in this implementation:**
- `previous_response_id` is explicitly unsupported (throws `invalid_argument`)
- Multi-turn Responses API semantics (where the server manages conversation state) are not implemented — the caller must replay full message history, same as Chat Completions

**`id_slot` injection with Responses API:** The `id_slot` field is passed per-request in the JSON body. The converted body passes it through `handle_completions_impl`. You can pin Responses API calls to slots the same way as Chat Completions calls.

---

## Prefix Caching vs KV Cache Reuse: Terminology

In llama.cpp these are **two different mechanisms**, not synonyms:

| Term | Flag | What it does |
|---|---|---|
| **Prefix caching** | `cache_prompt = true` (default) | Reuses the longest common *prefix* of the new prompt vs. the cached prompt. Pure token matching, no KV shifting. `n_past = common_prefix_len`. |
| **KV cache reuse** | `--cache-reuse N` | After prefix matching, scans the *non-prefix* tail for matching chunks of length >= N and shifts their KV positions via RoPE re-application. Requires `can_shift`. |

Both are active in your configuration. "OpenAI-style prefix caching" (where the provider caches a shared prefix block server-wide) does not apply here — llama.cpp KV caches are per-slot.

---

## Monitoring Cache Hit Rate

**Per-response:** The `timings` object in every completion response includes:
```json
{
  "timings": {
    "cache_n": 236,   // tokens reused from cache (n_past at start of request)
    "prompt_n": 1,    // tokens actually evaluated this request
    "predicted_n": 35
  }
}
```
`cache_n` = `n_prompt_tokens_cache` = `n_past` after prefix match + any KV-shift reuse. Source: `server-context.cpp:2425`, `server-task.cpp:604`, `README.md:1263`.

**Total context** = `cache_n + prompt_n + predicted_n` (from prior turns).

**Effective cache hit rate** = `cache_n / (cache_n + prompt_n)` per request.

**Log-level monitoring:** Set `--log-verbosity 4` (debug) to see `SLT_DBG` messages:
```
trying to reuse chunks with size > 256, n_past = N
reusing chunk with size M, shifting KV cache [a, a+M) -> [b, b+M)
after context reuse, new n_past = K
```

**Prometheus endpoint** (`--metrics`): Exposes `prompt_tokens_total`, `prompt_tokens_seconds`, `requests_deferred` but **does not** expose a dedicated cache-hit-rate counter. You must compute it from `cache_n` in individual response timings. The `/metrics` endpoint was not extended to include `n_prompt_tokens_cache` in total.

**`/slots` endpoint** (enabled by default): Each slot object includes `n_ctx` and current processing state. Combine with `cache_n` from responses to track per-slot hit rates.

---

## Recommended Changes to Deployment Plan

### 1. Confirm `can_shift` is active (no-op if already verified)
Check startup logs for:
```
cache_reuse is not supported by this context, it will be disabled
```
If absent, cache_reuse is operational. Add `--log-prefix` for easier grep.

### 2. Lower `--cache-reuse` threshold or remove it for your use case
With frozen system prompts and `id_slot` pinning, the system prompt is always the common prefix — it is recovered by the `cache_prompt` prefix match without any KV shifting. `--cache-reuse 256` only activates when non-prefix chunks of 256+ tokens match. For typical Discord-bot short turns, this threshold is rarely reached. Consider `--cache-reuse 64` or simply relying on the default prefix caching (`--cache-reuse 0`) unless you have evidence from `cache_n`/`prompt_n` ratios that shifting is firing.

### 3. Pin KV cache to GPU, confirm `--kv-offload` is active
With `--split-mode row`, KV is on the main device (GPU 0) by default. For 3x A770 you have 48 GB total but KV is constrained to 16 GB unless you override. If context is large, consider `--no-kv-offload` to put KV in system RAM and free GPU 0 VRAM for weights (slower, but avoids KV competing with weight compute buffers on GPU 0).

### 4. Add `--slot-save-path /path/to/slots/` for restart warmup
After first request to each slot, save state:
```
POST /slots/0/action=save
POST /slots/1/action=save
```
On server restart, restore both slots before opening to traffic. This converts the cold-start penalty (full system prompt evaluation) into a file read. Files will be large (proportional to context fill) but startup reads are fast.

### 5. Monitor `cache_n` vs `prompt_n` per slot
Instrument your proxy to log `timings.cache_n / (timings.cache_n + timings.prompt_n)` per response. A hit rate below 90% on the second and subsequent turns indicates either a system prompt mismatch, an accidental `cache_prompt=false`, or a slot-steal. This is the most actionable signal.

### 6. Row-split + speculative decoding: check draft model device placement
If using `--draft` with row-split, the draft model defaults to a separate context (`params_dft.n_parallel = 1`) on the main device. Confirm the draft model fits on GPU 0 after the KV cache and weight shards are allocated. If memory is tight on GPU 0, consider moving the draft model to system RAM with `-ngl 0` for the draft context (slower draft, but prevents OOM).

### 7. Be aware of the deferred-queue backpressure behavior
With `id_slot` pinning, a slow request to slot 1 will block subsequent slot-1 requests indefinitely (they queue in `queue_tasks_deferred` with no timeout). If churner jobs can be long, add an application-level timeout and cancel stale requests via `POST /slots/1/cancel` or the task cancel API to prevent queue buildup.

### 8. Do not use `--context-shift` with `--cache-reuse` on the same slot
Context shift is valid and still works on SYCL (it calls the same `can_shift` path). However, after a context shift the cached KV positions are renumbered, and the next `cache_reuse` scan compares against the post-shift token list. This is handled correctly in the code but produces confusing `cache_n` values (the shifted-in content counts as cached). Recommend keeping context large enough to avoid shifts (`--ctx-size` = system prompt + max conversation depth).

### 9. Flash attention (`-fa auto`) is safe with cache reuse
`-fa auto` (the default) on SYCL with Qwen3 will use the SYCL FA kernels for non-quantized KV. This does not interfere with KV shifting. The K-shift graph bypasses FA entirely. No action needed.

### 10. No SYCL-specific KV cache bugs to work around
As of commit `e0ebf6b`, all known SYCL correctness bugs are in the weight-side matmul path (row-split pointer resolution, MMVQ ds visibility) and are fixed. The KV cache path (allocation, shift, save/restore) is backend-agnostic and has no open SYCL-specific issues.
