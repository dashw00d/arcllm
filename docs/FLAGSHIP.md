# Flagship Settings — Qwen3-30B-A3B MoE on 3x Intel Arc A770

## The Model

**Qwen3-30B-A3B-abliterated** Q4_K_M
- Architecture: `qwen3moe` — 128 routed experts, 8 active per token, no shared experts
- Size: 17.3 GB (fits in VRAM with room for context)
- Path: `models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf`
- Thinking mode: built-in `<think>` reasoning (Qwen3 feature)

## Verified Performance

All numbers from testing on March 20-21, 2026 with the stable build.

| Config | Speed | Context | Notes |
|--------|-------|---------|-------|
| np=1, c=512 | 15.0 t/s | 512 | Single user baseline |
| np=1, c=8192 | 13.7 t/s gen, 37.2 pp | 8192 | Long context works |
| np=4, c=8192 | ~14 t/s per-slot | 8192 | Multi-slot verified |
| np=16, c=512 | 39.4 t/s aggregate | 512 | 16 concurrent requests |

**Prompt processing:** 37.2 t/s (141-token prompt)

## Server Flags (Tested & Verified)

```bash
GGML_SYCL_DISABLE_GRAPH=1 \
SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
ZE_AFFINITY_MASK=0,1,2 \
GGML_SYCL_FUSED_MMQ=1 \
  llama-server \
    -m qwen3-30b-a3b-abliterated-q4_k_m.gguf \
    --split-mode layer \
    -ngl 99 \
    --tensor-split 1,1,1 \
    -c 8192 \
    -fa off \
    -np 4 \
    --no-warmup \
    --reasoning-budget 0
```

### Why each flag:

| Flag | Value | Why |
|------|-------|-----|
| `--split-mode layer` | layer | Layer-split is production-proven. Tensor-split (EP) is experimental. |
| `-ngl 99` | 99 | Offload all layers to GPU |
| `--tensor-split 1,1,1` | equal | Even distribution across 3 GPUs |
| `-c 8192` | 8192 | Fits in VRAM. c=32768 crashes (VRAM exhaustion with 17GB model + KV). Tested up to 8192 verified. |
| `-fa off` | off | **IGC crashes on MoE + flash attention.** This is a known Intel Graphics Compiler bug. Do NOT use `-fa on` with MoE models. |
| `-np 4` | 4 | 4 parallel slots. np=16 works for benchmarks but np=4 is safer for production (less KV cache pressure). |
| `--no-warmup` | set | Skip warmup pass (saves time, model loads in ~6s) |
| `--reasoning-budget 0` | 0 | Disable thinking mode for conversational use. Model responds directly without `<think>` blocks. Use `/think` in Discord for reasoning. |
| `GGML_SYCL_DISABLE_GRAPH=1` | 1 | Graph recording is disabled — MoE models have unstable pointers that break graph replay. |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` | 0 | Batched command lists — +7.5% throughput over immediate mode. |
| `GGML_SYCL_FUSED_MMQ=1` | 1 | Fused dequant+matmul — +25% on dense. Effect on MoE untested but shouldn't hurt. |

### Known Issues

- **c=32768 crashes** — VRAM exhaustion. The 17GB model + 4 slots × 32K context exceeds 48GB VRAM.
- **-fa on crashes** — Intel Graphics Compiler internal error on MoE + flash attention.
- **500+ token generation degenerates** — F16 overflow bug (documented in journey.md Chapter 12). Cap max_tokens or use reasoning_budget.
- **Thinking eats token budget** — Without `--reasoning-budget 0`, model spends most tokens on `<think>` reasoning before producing visible content.

### What NOT to Use

- `--split-mode tensor` — EP is experimental, still garbles output
- `-fa on` — crashes the IGC compiler on MoE
- `-c 32768` or higher — VRAM exhaustion
- REAM-heretic models — broken quantization, garbles on all builds

## SYCL Environment

Set by `env.sglang-xpu.sh` before any SYCL operation:
```bash
source /home/ryan/llm-stack/env.sglang-xpu.sh
```

The proxy (`arcllm-proxy.py`) sets SYCL env vars in `SYCL_ENV` dict before spawning llama-server.

## Build

Binary: `llama.cpp-stable/build-sycl/bin/llama-server`
Commit: `5ea15e776` (feat: ffn_gate_inp bias for expert padding)
Compiler: `icpx` (Intel DPC++)
Flags: `GGML_SYCL=ON`, `GGML_SYCL_GRAPH=ON`

The proxy uses a symlink: `llama.cpp/build-sycl → llama.cpp-stable/build-sycl`

```bash
cd llama.cpp-stable/build-sycl
source /home/ryan/llm-stack/env.sglang-xpu.sh
cmake --build . --target llama-server -j$(nproc)
```

---

# Discord Bot Setup

## Architecture

```
Discord → Henry bot (Python) → arcllm-proxy (:11435) → llama-server (:18400) → 3x A770
```

## Components

### 1. arcllm-proxy (`scripts/arcllm-proxy.py`)
- Lazy-loading reverse proxy on port **11435**
- Loads/unloads models on demand (Ollama-like)
- Manages SYCL env, compiler cache, GPU reset on DEVICE_LOST
- OpenAI-compatible API

**Start/stop:**
```bash
bash scripts/arcllm-server.sh start   # starts proxy on :11435
bash scripts/arcllm-server.sh stop
bash scripts/arcllm-server.sh status
```

**Model config** in `scripts/arcllm-proxy.py`:
- `qwen3-30b-moe` — the MoE model (aliases: `30b-moe`, `moe`)
- `qwen3-32b` — dense model, current default (aliases: `32b`, `default`)
- Others: nemotron-120b, qwen35-27b, glm47-flash, qwen3-235b

### 2. Discord bot (`/home/ryan/project/discord-bot/`)

| File | What |
|------|------|
| `bot.py` | Main bot — message handling, slash commands |
| `config.py` | Reads env vars: DISCORD_TOKEN, LLAMA_SERVER_URL, MODEL |
| `inference.py` | OpenAI client → proxy. Sends reasoning_budget for /think |
| `prompts.py` | Frozen system prompt (for KV cache reuse) |
| `history.py` | Per-channel conversation history |
| `rate_limiter.py` | Per-user cooldown |

**Config (env vars):**
```bash
export DISCORD_TOKEN="..."
export LLAMA_SERVER_URL="http://localhost:11435"
export MODEL="qwen3-30b-moe"     # or "qwen3-32b" for dense
export MAX_TOKENS="2048"
export MAX_HISTORY="40"
```

**Run:**
```bash
cd /home/ryan/project/discord-bot
DISCORD_TOKEN=... MODEL=qwen3-30b-moe python3 bot.py
```

**Features:**
- Responds to all messages in `henrys-hotdog-stand` channel
- Responds to @mentions in other channels
- DMs supported
- `/think` command — enables reasoning mode (reasoning_budget=4096)
- `/reset` — clear channel history
- `/status` — check server status
- Rate limiting per user (2s cooldown)

### 3. Same binary & env

The proxy uses the **exact same binary** as our test runs:
- Binary: `llama.cpp-stable/build-sycl/bin/llama-server` (md5: `f2a767c182a4a9abc1c63458952214b6`)
- Accessed via symlink: `llama.cpp/build-sycl → llama.cpp-stable/build-sycl`
- SYCL env: same `GGML_SYCL_DISABLE_GRAPH=1`, `CMDLISTS=0`, `FUSED_MMQ=1`
- Build commit: `5ea15e776`

## To Switch Discord to MoE

1. Start the proxy: `bash scripts/arcllm-server.sh start`
2. Start the bot with MoE model:
   ```bash
   cd /home/ryan/project/discord-bot
   MODEL=qwen3-30b-moe python3 bot.py
   ```
3. First message will trigger model load (~6s)
4. Responses at ~14 t/s without thinking, or use `/think` for reasoning

## To Switch Back to Dense

```bash
# Just restart the bot with the default model
MODEL=qwen3-32b python3 bot.py
```
The proxy auto-unloads the MoE and loads the dense model on first request.
