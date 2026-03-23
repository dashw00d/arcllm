# Context Budget — Maximizing Per-Slot Context for Qwen3.5-35B

## The Problem

The site auditor (Henry + agent-browser) overflows at ~9 browsing steps with 8k/slot context. Each agent-browser snapshot is ~800 tokens. With tool call overhead, thinking tokens, and message history, 8k fills in under 10 exchanges.

16k/slot would give ~20+ steps — enough for a full site audit. But 16k/slot OOM'd on host memory due to compute buffer explosion.

## What Happened

```
np=4, c=65536 (16k/slot):
  SYCL0 compute buffer: 10,939 MB  ← HUGE
  SYCL1 compute buffer: 10,935 MB
  SYCL_Host compute:     3,532 MB
  → UR_RESULT_ERROR_OUT_OF_HOST_MEMORY

np=4, c=32768 (8k/slot):
  SYCL0 compute buffer:  3,451 MB  ← reasonable
  SYCL1 compute buffer:  4,475 MB
  SYCL_Host compute:     1,932 MB
  → WORKS
```

Compute buffers jumped 3x when context doubled. This is because the default batch size (`-b 512`) allocates scratch space proportional to `batch_size × context`. The model weights (22GB) and KV cache (1.3GB) are fine — it's the scratch space that kills it.

## Levers to Test

### 1. Reduce batch size: `-b 256` or `-b 128`

Compute buffers scale with batch size. Halving batch should halve compute buffers.

**Trade-off:** Slower prompt eval (processes fewer tokens per batch). Generation speed unaffected (always batch=1 during generation).

**Test command:**
```bash
# In arcllm-proxy.py, modify qwen35 config:
f"--split-mode layer -ngl 999 --tensor-split 1,1,1"
f" -c 65536 -b 256 -fa off"
f" -np 4 --no-warmup"
f" --slot-save-path {SLOT_CACHE}",
```

**What to check:** Server loads without OOM. Then run site auditor and verify it can do 15+ steps.

### 2. Reduce ubatch size: `-ub 256` or `-ub 128`

The "micro-batch" controls actual GPU kernel execution batch. Default is same as `-b`. Reducing this may help without affecting throughput as much.

### 3. KV quantization: `--kv-quant q8_0`

Halves KV cache VRAM (1.3GB → 0.65GB). Small gain but might free just enough for compute buffers.

### 4. np=2 with 16k/slot: `-np 2 -c 32768`

Fewer slots = less total context allocation = smaller compute buffers. The auditor only needs 1 slot anyway. Discord + auditor = 2 slots is fine.

### 5. Dedicated auditor model config

Register a separate model in the proxy for auditor use:
```python
_register(
    "qwen35-auditor",
    ...,
    f"--split-mode layer -ngl 999 --tensor-split 1,1,1"
    f" -c 32768 -b 256 -fa off"
    f" -np 2 --no-warmup",
    aliases=["auditor"],
    server_bin=Path("/home/ryan/llm-stack/bin/llama-server-qwen35-gdn"),
)
```

Site auditor sends `"model": "qwen35-auditor"`, proxy loads the big-context config. Discord uses regular `qwen35`. Model swap takes ~20s (slot save/restore).

## How to Test

### Quick test (does it load?)
```bash
# Modify proxy config, restart, trigger load:
/home/ryan/llm-stack/scripts/arcllm-server.sh stop
/home/ryan/llm-stack/scripts/arcllm-server.sh start

curl -s http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen35","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'

# Wait for load, then check:
grep "compute buffer" /tmp/arcllm-server.log
# Should show reasonable numbers (< 5GB per GPU)
```

### Context fill test (does 16k work?)
```bash
# Send a request that fills context — long system prompt + conversation
python3 -c "
import httpx, json

# ~12k tokens of context
msgs = [
    {'role': 'system', 'content': 'You are helpful. ' * 500},  # ~2k tokens
    {'role': 'user', 'content': 'Summarize everything. ' * 200},  # ~1k tokens
]
# Add 10 rounds of assistant/user (simulating agent loop)
for i in range(10):
    msgs.append({'role': 'assistant', 'content': f'Step {i}: I observed the page has many elements. ' * 50})
    msgs.append({'role': 'user', 'content': f'Continue with step {i+1}.'})

r = httpx.post('http://localhost:11435/v1/chat/completions',
    json={'model': 'qwen35', 'messages': msgs, 'max_tokens': 100},
    timeout=120)
d = r.json()
if 'error' in d:
    print(f'ERROR: {d[\"error\"][\"message\"][:200]}')
else:
    print(f'OK: {d[\"usage\"][\"prompt_tokens\"]} prompt tokens, generated {d[\"usage\"][\"completion_tokens\"]}')
"
```

### Site auditor integration test
```bash
cd /home/ryan/llm-stack/arc-tools/site-auditor
agent-browser close
python3 auditor.py \
  --url "https://wehelpyouparty.com" \
  --entity-type "wedding venues" \
  --description "venues and locations that host weddings" \
  --output /tmp/audit_test.json
```

**Success criteria:** Phase 2 (relevance) completes with a `done` call containing `relevant: true`.

## Current Baseline (working)

```
np=4, c=32768 (8k/slot), b=512 (default)
  SYCL0 model: 6,897 MB
  SYCL1 model: 6,626 MB
  SYCL2 model: 6,382 MB
  SYCL0 compute: 3,451 MB
  SYCL1 compute: 4,475 MB
  SYCL2 compute: 3,664 MB
  Host compute: 1,932 MB
  KV cache: 768 MB (12 attn layers, 4 seqs)
  RS: 251 MB (40 recurrent layers, 4 seqs)
  Total: ~35 GB used, ~13 GB free
```

## Files to Modify

| File | Change |
|------|--------|
| `scripts/arcllm-proxy.py` | Add `-b 256` to qwen35 config (or register qwen35-auditor) |
| `arc-tools/site-auditor/agent.py` | Update HENRY_MODEL if using separate auditor config |
