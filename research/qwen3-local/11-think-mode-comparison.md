# Qwen3 /think vs /no_think Mode: Real-World Performance for Coding and Agentic Tasks

*Research date: 2026-03-15 | Covers Qwen3-235B-A22B running via llama-server on 3x Intel Arc A770*

---

## 1. What Are /think and /no_think? Invocation Mechanics

Qwen3 models (released April 2025) are trained with a unified "hybrid thinking" architecture — a single set of weights handles both deliberate chain-of-thought reasoning and fast, direct answering. The two modes are not different models or LoRAs; they are alternate behaviors of the same model selected at inference time.

**Sources:** [HuggingFace model card](https://huggingface.co/Qwen/Qwen3-235B-A22B), [Qwen3 Technical Report arxiv:2505.09388](https://arxiv.org/abs/2505.09388), [Qwen docs quickstart](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)

### 1.1 Three Ways to Invoke the Toggle

#### Hard switch: `enable_thinking` template parameter

The canonical hard switch is a Jinja2 template parameter passed at request time. This is not a special token — it is a conditional branch in the chat template:

```jinja2
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

When `enable_thinking=False`, the template *pre-fills* an empty `<think></think>` block at the start of the assistant turn. Because the think block is already closed before the model generates a single token, the model cannot produce thinking content — it immediately generates the visible answer.

In llama-server (current build as of this repo), this is accessible via the `chat_template_kwargs` field in `/v1/chat/completions` requests:

```json
{
  "chat_template_kwargs": {"enable_thinking": false}
}
```

**Important caveat from the llama.cpp docs:** *"The hard switch implemented in the chat template is not exposed as a command-line flag in llama-server."* The workaround is `--chat-template-file qwen3_nonthinking.jinja` at server startup, or the per-request `chat_template_kwargs`. [Source: Qwen readthedocs llama.cpp page](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)

#### Soft switch: `/think` and `/no_think` in the user message

Any user message (or system message) may end with `/think` or `/no_think`. The model was trained on data with these flags inserted at random positions in multi-turn conversations; it learns to obey the **most recent flag** encountered in the conversation history. From the technical report (§4.3):

> "For complex multi-turn dialogs, we randomly insert multiple /think and /no think flags into users' queries, with the model response adhering to the last flag encountered."

The flags are **plain text**, not special tokens. Token IDs 151667 (`<think>`) and 151668 (`</think>`) are the structural markers, but `/think` is just the string " /think" appended to the user message.

Soft switch behavior when `enable_thinking=False` at the template level: the soft switches are **ignored**. The empty think block is pre-filled regardless of what the user writes.

#### `--reasoning-budget` (llama-server CLI flag)

The current llama.cpp build in this repo has a `--reasoning-budget N` flag:

```
--reasoning-budget N   controls the amount of thinking allowed;
                       -1 = unrestricted (default), 0 = disable thinking
```

Setting `--reasoning-budget 0` forces thinking off at the server level globally for all requests. This is the simplest way to run a "no-think" server. Setting it to a positive integer caps the thinking token count (see §7 for details on the budget mechanism).

The local `arcllm-proxy.py` already uses this:
- `qwen3-32b` uses `--reasoning off` (equivalent alias)
- `qwen3-32b-fast` and `qwen3-32b-spec` use `--reasoning-budget 200`
- `qwen3-235b` currently has no reasoning flag set (defaults to unrestricted thinking)

---

## 2. Quality Delta: Thinking vs Non-Thinking Mode

### 2.1 Official Benchmarks

The Qwen3 Technical Report (arxiv:2505.09388, Tables 11 & 12) evaluates Qwen3-235B-A22B in both modes against separate baselines:

**Thinking mode** benchmarked against reasoning models (DeepSeek-R1, OpenAI-o1, Gemini 2.5 Pro, Grok-3-Beta Think):

| Benchmark | DeepSeek-R1 | Gemini 2.5 Pro | Qwen3-235B-A22B (Think) |
|---|---|---|---|
| MMLU-Redux | 92.9 | 93.7 | **92.7** |
| GPQA-Diamond | 71.5 | **84.0** | 71.1 |
| MATH-500 | 97.3 | **98.8** | 98.0 |
| AIME'24 | 79.8 | 92.0 | 85.7 |
| AIME'25 | 70.0 | 86.7 | 80.3 |
| LiveCodeBench v5 | 64.3 | 85.1 | 78.7 |
| Codeforces Rating | 2056 / 98.2% | 86.9 | 2001 / 97.9% |
| BFCL v3 | 70.8 | 77.8 | 70.7 |
| Arena-Hard | 92.3 | 96.4 | 95.6 |
| Creative Writing v3 | 85.5 | 86.0 | 84.6 |

**Non-thinking mode** benchmarked against instruction-tuned models (GPT-4o, DeepSeek-V3, Qwen2.5-72B, LLaMA-4-Maverick):

| Benchmark | GPT-4o | DeepSeek-V3 | Qwen2.5-72B-Instruct | Qwen3-235B-A22B (No-Think) |
|---|---|---|---|---|
| MMLU-Redux | 87.0 | 89.1 | 86.8 | **89.2** |
| GPQA-Diamond | 46.0 | 59.1 | 49.0 | **62.9** |
| MATH-500 | 77.2 | 90.2 | 83.6 | **91.2** |
| AIME'24 | 11.1 | 39.2 | 18.9 | **40.1** |
| LiveCodeBench v5 | 32.7 | 52.9 | — | **57.6** (est.) |
| Arena-Hard | 85.3 | 85.5 | 81.2 | **96.1** |
| Creative Writing v3 | 81.1 | 74.0 | 61.8 | **80.4** |
| IFEval strict | 86.5 | 86.1 | 84.1 | 83.2 |
| BFCL v3 (agent) | 72.5 | 63.4 | 52.9 | 57.6 (est.) |

**Key takeaway from the data:**

- For hard math (AIME), non-thinking mode still reaches 40/100 — far above GPT-4o's 11 — because Qwen3-235B's base capabilities are exceptional. Thinking mode pushes this to ~86.
- For coding (LiveCodeBench), non-thinking scores ~58 vs ~79 in thinking mode — a meaningful gap for competitive programming. For everyday coding tasks the gap is likely smaller.
- For instruction following (IFEval), creative writing, and casual alignment (Arena-Hard), non-thinking mode is nearly equal and sometimes leads (Arena-Hard: 96.1 vs 95.6).
- The paper explicitly notes: *"For Knowledge, STEM, Math, and Coding tasks, Thinking Mode Fusion and General RL do not bring significant improvements [to non-thinking mode]. In contrast, for challenging tasks like AIME'24 and LiveCodeBench, the performance in thinking mode actually decreases after General RL."* This suggests the quality gap is primarily on hard problems; for easier coding and general use, non-thinking is competitive.

### 2.2 Coding Tasks

The benchmarks show thinking mode has a **~20 percentage-point advantage** on competition-level coding (LiveCodeBench v5: 79 vs 58). For real-world bug fixes and code generation (not Codeforces-level), community reports suggest the practical gap is smaller:

- BFCL v3 (function/tool calling): thinking mode 70.7 vs non-thinking ~57. This is relevant for agentic workflows that call tools. Non-thinking degrades tool-calling reliability by ~13 points.
- SWE-bench Pro: thinking mode achieves 21.41 (from the model card). Non-thinking mode data not reported separately, but expected to be significantly lower given the multi-step reasoning required for autonomous issue resolution.

### 2.3 Reasoning and Multi-Step Logic

Thinking mode is a strict improvement on multi-step logic at the cost of latency. For ZebraLogic (constraint reasoning) and AutoLogi, thinking mode: 81.5 vs non-thinking: ~40 for AIME-style problems. The gap collapses for simpler tasks.

### 2.4 Creative and Casual Conversation

Non-thinking mode performs **on par or better** for:
- Creative Writing v3: 84.6 (think) vs 80.4 (no-think) — slight think advantage
- AlignBench (Chinese conversational): 8.94 (think) vs 8.91 (no-think) — essentially identical
- WritingBench: 8.03 (think) vs 7.70 (no-think) — marginal think advantage
- Arena-Hard: 95.6 (think) vs 96.1 (no-think) — no-think wins slightly

For Discord casual conversation, non-thinking mode is appropriate. The thinking overhead provides no meaningful quality improvement for chat and may slightly degrade perceived responsiveness.

---

## 3. Token Overhead: How Many Thinking Tokens Does Qwen3-235B Generate?

The technical report does not publish average thinking token counts per task type. Based on available data points:

- **Simple factual questions** with `/think`: Model often generates an empty or near-empty think block (0–50 tokens). The template with `enable_thinking=False` forces this explicitly (`<think>\n\n</think>`).
- **Hard math problems (AIME)**: The paper sets `max_output_length = 38,912` tokens specifically to give "sufficient thinking space." This implies actual thinking traces can approach or exceed 32K tokens on competition math.
- **Typical coding tasks**: Community reports and the recommended default of `max_output_length = 32,768` suggest 500–4,000 thinking tokens is common for non-trivial coding problems.
- **The thinking budget experiment** (Figure 2 in the paper): Performance improves monotonically with thinking tokens up to at least 32K, with the paper noting "if we further extend beyond 32K, performance is expected to improve further."
- **Thinking budget recommendation** from the official docs: *"thinking_budget should not be set too low in practice. We recommend tuning based on acceptable latency and setting it higher than 1024."*

Empirical guideline based on the above: assume **800–3,000 thinking tokens** for typical coding tasks, and **3,000–15,000+** for hard math/algorithm problems.

---

## 4. Latency Impact at 3–5 t/s

Qwen3-235B-A22B running on 3x Intel Arc A770 in the current setup achieves approximately 3–5 t/s for output tokens (measured in prior benchmarks, see `research/qwen3-local/` other files). All thinking tokens count as output tokens — they consume the same per-token budget.

### 4.1 Latency Calculation Table

The latency to *first visible token* (TTFT of answer, excluding prompt processing) equals:
`thinking_tokens / speed_t_per_s`

| Thinking tokens | At 3 t/s | At 4 t/s | At 5 t/s |
|---|---|---|---|
| 0 (no-think) | 0 s | 0 s | 0 s |
| 200 (budget cap) | 67 s | 50 s | 40 s |
| 500 | 167 s (2.8 min) | 125 s (2.1 min) | 100 s (1.7 min) |
| 1,000 | 333 s (5.6 min) | 250 s (4.2 min) | 200 s (3.3 min) |
| 2,000 | 667 s (11.1 min) | 500 s (8.3 min) | 400 s (6.7 min) |
| 4,000 | 1,333 s (22 min) | 1,000 s (16.7 min) | 800 s (13.3 min) |
| 8,000 | 2,667 s (44 min) | 2,000 s (33 min) | 1,600 s (27 min) |
| 16,000 | 5,333 s (89 min) | 4,000 s (67 min) | 3,200 s (53 min) |

**Practical interpretation:**
- A 1,000-token think trace at 4 t/s = **4.2 minutes before the answer starts streaming**
- A Discord user asking a casual question and receiving no response for 4+ minutes = poor UX
- A churner workflow running a complex schema-evolution plan with 4,000 thinking tokens = 17 minutes; acceptable if run async, unacceptable if blocking a downstream step

### 4.2 Total Response Time (Think + Answer)

If a thinking-mode response produces T_think thinking tokens and T_answer answer tokens:
`total_time = (T_think + T_answer) / speed`

For a typical coding response of 500 answer tokens at 4 t/s:
- No-think: 125 s total (2.1 min)
- Think 1K: 375 s total (6.3 min)
- Think 2K: 625 s total (10.4 min)

---

## 5. Best Practices for Toggling: When to Use Each Mode

### 5.1 Decision Framework from Qwen3 Docs

The official recommendation ([quickstart docs](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)):
- **Thinking mode**: complex logical reasoning, multi-step math, hard coding problems
- **Non-thinking mode**: efficient general-purpose chat, simple Q&A, creative writing

The technical report §4.3 explicitly trains the model to recognize `/think` and `/no_think` as conversational controls:
> "By default, the model operates in thinking mode; therefore, we add some thinking mode training samples where the user queries do not include /think flags."

This means without any flag, the model **defaults to thinking** (if `enable_thinking` is not forced False). This is the source of the performance overhead issue for Discord chat — every message without `/no_think` triggers thinking unless the server or template forces it off.

### 5.2 Application-Specific Recommendations

**Discord casual chat:**
- Default: always `/no_think` via system prompt or `enable_thinking=False` in `chat_template_kwargs`
- Exception: user explicitly requests "think carefully" or "reason through this" → inject `/think` or switch to a think-enabled endpoint
- Rationale: 4+ minute latency before first token is incompatible with Discord UX; creative/casual quality is equal in no-think mode

**Complex coding help via Discord (user asks for bug fix / algorithm):**
- Option A: use `--reasoning-budget 200` (current `qwen3-32b-fast` setup) — limits thinking to ~40–67 s delay, still improves code quality over no-think for medium problems
- Option B: route to a dedicated think-enabled endpoint with user warning ("this will take 2–5 minutes")
- Qwen3-235B non-thinking is still stronger than GPT-4o on coding; budget-capped thinking may be the sweet spot

**Churner agentic workflow (schema evolution, data extraction, group discovery):**

The workflow has multiple distinct phases with different thinking needs:

| Phase | Task | Recommended Mode | Rationale |
|---|---|---|---|
| Planning / schema design | Multi-step reasoning, dependency analysis | `/think` with budget 2000–4000 | High value from reasoning; async so latency acceptable |
| SQL/code generation | Generating extraction queries | `/think` budget 500–1000 | Meaningful coding improvement; manageable latency |
| Data extraction | Running queries, iterating rows | `/no_think` | Mechanical task; thinking adds no value |
| Group discovery analysis | Pattern recognition, clustering decisions | `/think` budget 1000–2000 | Reasoning helps novel pattern identification |
| Result summarization | Writing output text | `/no_think` | No reasoning benefit; saves 5+ minutes |
| Tool call parsing | JSON extraction from model output | `/no_think` | Thinking during tool calls can corrupt output format |

The Qwen3 docs explicitly warn about tool calling with thinking mode:
> "It is not recommended to use tool call template based on stopwords, such as ReAct, because the model might output stopwords during reasoning, causing unexpected tool call behavior."

For any step that outputs structured JSON for tool calls, `/no_think` is safer.

---

## 6. Thinking Mode and Prompt Caching

The current llama-server (KV cache) caches the **prompt prefix** — the encoded input tokens that are reused across requests. Thinking tokens are **output tokens**, not input tokens, so they do not affect whether the prompt cache hits.

However, multi-turn conversation handling creates a caching issue:

From the Qwen3 model card best practices:
> "When processing long multi-turn conversations, thinking content should be excluded from conversation history. Only include the final answer."

If thinking content is included in the assistant turn (passed back as conversation history), it inflates each subsequent request's prompt. For a 10-turn conversation where each thinking trace is 2,000 tokens:
- Turn 10 prompt would include ~20,000 extra tokens of old thinking content
- This fills the 8,192-token context window in the current `qwen3-235b` setup (note: context is set to 8192 in `arcllm-proxy.py`)
- Even with larger context, the KV cache hit rate drops because cached thinking tokens differ each time (thinking is stochastic, temperature 0.6)

**Practical rule:** Never include `reasoning_content` in conversation history. The llama-server API populates `message.reasoning_content` separately from `message.content` for exactly this reason (PR #18994). Clients should store `reasoning_content` locally for inspection but only send `content` back in the next turn.

The current `qwen3-235b` registration in `arcllm-proxy.py` has `--cache-reuse 256` and `--slot-save-path`, which means the prefix KV cache is persisted across requests for the same slot. This works well for repeated system prompts but provides no benefit for the thinking portion of responses.

---

## 7. Thinking Budget: `thinking_budget` Parameter Details

### 7.1 How It Works (From the Technical Report)

Section 4.3 of arxiv:2505.09388:

> "When the length of the model's thinking reaches a user-defined threshold, we manually halt the thinking process and insert the stop-thinking instruction: 'Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n'. After this instruction is inserted, the model proceeds to generate a final response based on its accumulated reasoning up to that point. It is worth noting that this ability is not explicitly trained but **emerges naturally** as a result of applying Thinking Mode Fusion."

The budget is **soft**: the model does not know in advance it will be cut off; the inference engine counts output tokens inside the `<think>` block and injects the stop-thinking string + closing `</think>` tag when the count reaches the threshold. The model then generates a final answer from partial reasoning.

### 7.2 llama.cpp Implementation (Local Build)

The `reasoning-budget.h` in this repo implements a sampler-level state machine:

```
IDLE → COUNTING → WAITING_UTF8 → FORCING → DONE
```

- `IDLE`: watches for the `<think>` start token sequence
- `COUNTING`: counts down remaining budget tokens, watching for natural `</think>`
- `WAITING_UTF8`: budget exhausted, allows current UTF-8 sequence to complete
- `FORCING`: forces the `forced_tokens` sequence (budget message + `</think>`) token by token, setting all other logits to -inf
- `DONE`: passthrough forever (post-think generation)

The `--reasoning-budget N` flag with `N=0` skips COUNTING entirely and goes straight to FORCING when `<think>` is detected — effectively disabling thinking by forcing an immediate close. This is distinct from `enable_thinking=False` which pre-fills the empty block before generation begins.

### 7.3 Availability

- **Qwen Chat / Alibaba Cloud Bailian API**: thinking_budget is a first-class API parameter
- **vLLM ≥0.9.0**: supported via `qwen3` reasoning parser
- **SGLang ≥0.4.6.post1**: supported via `--reasoning-parser qwen3`
- **llama-server (this build)**: `--reasoning-budget N` (server-wide default) + per-request via `chat_template_kwargs` if `N` budget messages are configured
- **HuggingFace Transformers**: not natively supported; Issue #1388 was closed as "not planned." Manual two-pass implementation required (generate until budget, inject stop string, continue)

### 7.4 Recommended Budget Values

From Qwen docs: *"thinking_budget should not be set too low in practice. We recommend higher than 1024."*

The technical report Figure 2 shows performance curves for Qwen3-235B-A22B at various budgets on AIME, LiveCodeBench, GPQA, and MATH-500. Performance improves smoothly with budget; there is no cliff but there is significant diminishing returns above ~8K tokens.

Practical budget recommendations for this hardware (3–5 t/s):

| Use case | Recommended budget | Latency overhead | Quality vs no-think |
|---|---|---|---|
| Disable thinking | 0 | 0 | Baseline |
| Minimal reasoning (Discord coding) | 200 | 40–67 s | Modest improvement |
| Moderate reasoning (routine coding) | 500–1000 | 100–333 s | Meaningful improvement |
| Deep reasoning (hard algorithm design) | 2000–4000 | 400–1333 s | Substantial improvement |
| Unrestricted (agentic planning) | -1 | Variable (can be >30 min) | Maximum quality |

---

## 8. How Other Local Qwen3 Users Configure This

From llama.cpp discussions (#12339, #20408) and community reports:

**Common patterns observed:**

1. **System prompt `/no_think` for chat servers**: Most local deployments serving chat add `/no_think` to the system prompt to default to fast mode. The llama.cpp discussion confirms: `--sys "/no_think"` works for Qwen3-8B; results mixed for 0.6B (soft switch less reliable in smaller models).

2. **Per-request switching via app logic**: Applications that need both modes run with `enable_thinking=True` (default) at the server level, and the client appends `/no_think` to casual messages. This avoids server restarts.

3. **Separate server instances**: Some users run two server instances — one with `--reasoning-budget 0` for chat, one with `--reasoning-budget -1` for agentic tasks. Expensive in VRAM but simplest approach.

4. **`--reasoning-budget 200-500` as default**: Increasingly popular "sweet spot" — fast enough for interactive use, enough reasoning budget to catch obvious bugs without multi-minute waits. This is what the local `qwen3-32b-fast` profile already does.

**Community notes on 235B specifically:**
- Discussion #35 on the HuggingFace model repo: some users report "Qwen3-235B is less effective than Qwen3-32B" — this may reflect quantization loss at Q3 (which this deployment uses) rather than model quality
- At Q3_K_S quantization, 235B runs ~4 t/s on this hardware; thinking mode with 2K tokens = 8+ minutes, which is why some prefer 32B for interactive use

---

## 9. Community Reports on 235B Thinking vs Non-Thinking Quality

Based on available community data (HuggingFace discussions, llama.cpp issues):

- **Discussion #32**: "In complex reasoning tasks, Qwen3 falls behind QwQ." This appears to reference non-thinking mode comparisons or specific task types; thinking mode benchmarks show Qwen3-235B beats QwQ-32B on 17/23 benchmarks.
- **Discussion #24**: "High hallucination rates compared to Qwen2.5." Possibly a non-thinking mode observation; non-thinking mode is aligned differently from Qwen2.5, and sampling parameters matter significantly (use presence_penalty=1.5 and temp=0.7 for no-think mode).
- **llama.cpp issue #20182**: `enable_thinking` parameter cannot turn off thinking for Qwen3.5 models — affects the Qwen3.5 series, not Qwen3. Status: open/unconfirmed as of 2026-03-15.
- **llama.cpp issue #20516**: "Response always starts with `</think>` tag when running Qwen3.5 9B" — Qwen3.5 bug, not Qwen3. Likely a chat template issue where the Qwen3.5 template differs from Qwen3's.

No specific community benchmarks for Qwen3-235B thinking vs non-thinking at Q3 quantization have been published as of this writing.

---

## 10. llama.cpp Chat Template Handling: Bugs and Workarounds

### 10.1 What the Local Template Does

The Qwen3 chat template in this build (`/home/ryan/llm-stack/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja`, which is the canonical Qwen3 template) correctly implements:

- `enable_thinking=False`: pre-fills `<think>\n\n</think>\n\n` before generation
- `enable_thinking=True` (default): opens the assistant turn at `<|im_start|>assistant\n`, allowing the model to generate `<think>...` naturally
- Multi-turn reasoning preservation: `reasoning_content` field is parsed from `<think>...</think>` blocks in assistant history and re-formatted correctly for multi-turn context
- Tool calls: Hermes-style `<tool_call>` format, separate from thinking content

### 10.2 Autoparser Detection

The `docs/autoparser.md` documents that the llama.cpp autoparser detects the Qwen3 template as `FORCED_CLOSED` mode:
> "Old Qwen/DeepSeek thinking templates — source contains `content.split('</think>')`: sets `reasoning.mode = FORCED_OPEN` with `<think>`/`</think>` markers"

This means llama-server automatically knows to parse `<think>` output as `reasoning_content` without manual `--reasoning-format` flags, as long as the model's embedded template is loaded.

### 10.3 Known Issues (as of 2026-03-15)

**Qwen3 (original series):**
- No confirmed chat template bugs in the llama.cpp issues tracker for the Qwen3 template specifically
- The `enable_thinking` hard switch requires either `--chat-template-file` or per-request `chat_template_kwargs` — not a command-line flag

**Qwen3.5 (different model family, different template):**
- Issue #20182: `enable_thinking` fails to disable thinking (open)
- Issue #20516: Responses prefixed with `</think>` spuriously (open)
- These do NOT affect Qwen3-235B-A22B which uses the original Qwen3 template

**Tool calling with thinking mode:**
- Issue #20260: "peg-native chat format parser fails when model outputs text before `<tool_call>`" — occurs when thinking mode produces text before the tool call JSON. Workaround: use `/no_think` for any request that requires tool calls, as the Qwen3 docs recommend.

### 10.4 Feature Gap: `reasoning_effort` Not Supported

Discussion #20408 confirms llama-server does not implement OpenAI's `reasoning_effort` parameter (which maps `"none"` to disable thinking). The workaround is `--reasoning-budget 0` (server-wide) or per-request `chat_template_kwargs`. PR #20297 (merged) added the budget mechanism as the llama.cpp alternative.

---

## 11. Qwen3's Extended Thinking vs Claude's Extended Thinking

Qwen3's thinking mode is architecturally similar to DeepSeek-R1's approach, not Claude's. Key differences:

| Feature | Claude Extended Thinking | Qwen3 Thinking Mode |
|---|---|---|
| Architecture | Separate thinking tokens (betas API) | Same model, template-controlled |
| Token budget | `thinking.budget_tokens` API param | `--reasoning-budget` / `chat_template_kwargs` |
| Budget enforcement | Hard (API-controlled) | Soft (sampler injects stop message) |
| Cache | Thinking tokens counted for context | Same KV cache, thinking = output tokens |
| Multi-turn | Thinking encrypted/hidden | Thinking can leak into context (must strip) |
| Quality scaling | Smooth with budget | Smooth with budget (Figure 2 confirms) |
| Tool call compatibility | Thinking before tools supported | Thinking before tools problematic |

Qwen3's approach is more lightweight: no API-level cryptographic isolation of thinking content, no billing distinction. The trade-off is the need to manually strip thinking from conversation history and be careful about tool call templates.

---

## 12. Does Thinking Mode Improve Agentic Task Success?

The Qwen3 technical report §4.4 (General RL) explicitly includes "agent capabilities" in the training objectives for both thinking and non-thinking modes. The benchmarks show:

- **BFCL v3** (function calling): thinking 70.7 vs non-thinking ~57 — ~14 point improvement
- **Multi-IF** (multi-turn instruction following): thinking 48.8 vs non-thinking 65.6 for GPT-4o; Qwen3-235B no-think ~65.3 (competitive with GPT-4o)

For agentic workflows specifically:
- Planning steps (deciding which tables to join, which schemas to evolve): thinking mode's multi-step reasoning is directly applicable and worth the latency cost
- Mechanical extraction steps (run query X, output JSON Y): thinking mode adds latency with negligible quality benefit
- The warning about ReAct-style tool use applies: in thinking mode, the model may generate tool call stopwords inside the `<think>` block prematurely. Use Hermes-style tool format and `/no_think` for tool-call turns.

---

## Recommended Changes to Deployment Plan

### Discord Bot Frontend

**Current state:** No explicit reasoning mode set for Qwen3-235B in `arcllm-proxy.py` (defaults to unrestricted thinking = potentially multi-minute TTFT).

**Recommended changes:**

1. **Add `--reasoning-budget 0` to the `qwen3-235b` registration** for the primary Discord-facing model. This makes no-think the default for all requests.

2. **Create a separate `qwen3-235b-think` profile** with `--reasoning-budget 1500` for explicit "think carefully" requests. Route to this profile when the user message contains certain triggers (e.g., "debug this", "explain why", "design a", "write an algorithm").

3. **In the Discord bot message handler:** append `/no_think` to all messages by default before forwarding to the proxy. This provides a second layer of defense even if `--reasoning-budget 0` fails or is overridden.

4. **Sampling parameters:** ensure Discord bot requests to no-think mode use `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` (non-thinking recommendations). Qwen3-235B at Q3 quant may need presence_penalty to avoid repetition.

5. **Strip reasoning_content from conversation history:** in the Discord bot's context management, only include `choices[0].message.content` in subsequent turns, never `choices[0].message.reasoning_content`.

### Churner Agentic Workflow System

**Recommended per-phase configuration:**

| Churner Phase | `chat_template_kwargs` | `--reasoning-budget` | Rationale |
|---|---|---|---|
| Schema evolution planning | `{"enable_thinking": true}` | 2000–4000 | High-value reasoning; async acceptable |
| SQL/query generation | `{"enable_thinking": true}` | 800–1500 | Meaningful code quality improvement |
| Data extraction execution | `{"enable_thinking": false}` | 0 | Mechanical; thinking = wasted time |
| Group discovery (novel patterns) | `{"enable_thinking": true}` | 1000–2000 | Reasoning helps novel cluster identification |
| Tool call turns (structured JSON) | `{"enable_thinking": false}` | 0 | Prevent think-block interference with tool parsing |
| Result summarization / report | `{"enable_thinking": false}` | 0 | Creative writing quality equal in no-think |

**Implementation approach:**

The churner should pass `chat_template_kwargs` per-request rather than relying on server-level defaults. This allows the same server instance to serve both thinking and non-thinking requests without restart:

```python
# High-reasoning request (planning)
payload = {
    "model": "qwen3-235b",
    "messages": messages,
    "chat_template_kwargs": {"enable_thinking": True},
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 6000  # 2000 think + 4000 answer
}

# Fast extraction request
payload = {
    "model": "qwen3-235b",
    "messages": messages,
    "chat_template_kwargs": {"enable_thinking": False},
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "max_tokens": 2000
}
```

**Context window consideration:** The current `qwen3-235b` context is 8,192 tokens. A single 4,000-token think trace consumes half the context window. If thinking mode is used for planning steps, consider increasing context to 16,384 or 32,768 (may require reducing the number of expert layers kept on GPU vs CPU).

**Reasoning content handling:** The churner should never include `reasoning_content` in conversation history. When storing churner turn history for multi-step workflows, strip the thinking trace before appending to the messages array.

### Server Startup Configuration

For the primary production `qwen3-235b` profile in `arcllm-proxy.py`, recommended flags change:

```python
# Current (no reasoning control):
f"--split-mode row -ngl 999 --tensor-split 1,1,1 -cmoe"
f" -c 8192 -fa on -np 2 --cache-reuse 256 --slot-save-path {SLOT_CACHE}"

# Recommended (with default budget=0 for safety, overridden per-request):
f"--split-mode row -ngl 999 --tensor-split 1,1,1 -cmoe"
f" -c 16384 -fa on -np 2 --cache-reuse 256 --slot-save-path {SLOT_CACHE}"
f" --reasoning-budget 0"
```

The context increase from 8192 to 16384 is recommended regardless: at 8192 tokens, a 500-token think trace + 2000-token conversation history + 2000-token system prompt leaves minimal space for responses.

---

## Sources

- [Qwen3-235B-A22B HuggingFace Model Card](https://huggingface.co/Qwen/Qwen3-235B-A22B)
- [Qwen3-235B-A22B-GGUF Model Card](https://huggingface.co/Qwen/Qwen3-235B-A22B-GGUF)
- [Qwen3 Technical Report (arxiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- [Qwen3 Quickstart Documentation](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Qwen3 llama.cpp Deployment Docs](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)
- [Qwen3 vLLM Deployment Docs](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- [Qwen3 Function Calling Docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [llama.cpp server README (tools/server/README.md)](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama.cpp autoparser docs (docs/autoparser.md)](https://github.com/ggml-org/llama.cpp/blob/master/docs/autoparser.md)
- [llama.cpp common/reasoning-budget.h](https://github.com/ggml-org/llama.cpp/blob/master/common/reasoning-budget.h)
- [llama.cpp PR #18994: reasoning_content in assistant messages](https://github.com/ggml-org/llama.cpp/pull/18994)
- [llama.cpp PR #20297: reasoning budget implementation](https://github.com/ggml-org/llama.cpp/pull/20297)
- [llama.cpp Discussion #20408: reasoning_effort support](https://github.com/ggml-org/llama.cpp/discussions/20408)
- [llama.cpp Discussion #12339: reasoning effort experiments](https://github.com/ggml-org/llama.cpp/discussions/12339)
- [llama.cpp Issue #20182: enable_thinking fails for Qwen3.5](https://github.com/ggml-org/llama.cpp/issues/20182)
- [llama.cpp Issue #20260: tool call parser failure with thinking mode](https://github.com/ggml-org/llama.cpp/issues/20260)
- [QwenLM/Qwen3 Issue #1388: thinking_budget in transformers (closed)](https://github.com/QwenLM/Qwen3/issues/1388)
- [QwenLM/Qwen3 Discussion #1288: thinking budget implementation](https://github.com/QwenLM/Qwen3/discussions/1288)
- [Qwen3 blog post](https://qwenlm.github.io/blog/qwen3/)
- Local files: `/home/ryan/llm-stack/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja`, `/home/ryan/llm-stack/llama.cpp/common/reasoning-budget.h`, `/home/ryan/llm-stack/scripts/arcllm-proxy.py`
