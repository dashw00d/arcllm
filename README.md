# ArcLLM

This directory contains a user-space XPU setup and local API stack for a 3x Arc A770 machine.

## Layout

- `env.xpu.sh`: runtime environment for the local Intel GPU userspace stack
- `scripts/check_xpu.py`: PyTorch XPU detection and tensor smoke test
- `scripts/smoke_vllm_qwen.py`: one-prompt `vLLM` smoke test with a small Qwen model
- `scripts/smoke_transformers_qwen.py`: one-prompt `transformers` smoke test with a small Qwen model
- `api/worker.py`: single-GPU OpenAI-style worker on `transformers`
- `api/router.py`: round-robin router for multiple workers
- `scripts/run_worker.sh`: launch one worker pinned to one Arc card
- `scripts/run_router.sh`: launch the router
- `scripts/start_3gpu_cluster.sh`: launch 3 workers plus router
- `arcllm/cli.py`: global CLI for start/stop/status/chat

## First load

```bash
./scripts/install.sh
source ./env.xpu.sh
python ./scripts/check_xpu.py
python ./scripts/smoke_transformers_qwen.py
```

## Stable API path

```bash
arcllm start
arcllm status
arcllm chat "Say hello from Arc."
arcllm set-model Qwen/Qwen3.5-27B
arcllm set TORCH_DTYPE float16
arcllm stop
```

OpenAI-compatible usage:

```bash
eval "$(arcllm env)"
```

Then point OpenAI-compatible tools at:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=local
```

Streaming chat works through the router:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Say hello from Arc in one sentence."}
    ]
  }'
```

## Notes

- The global command is `arcllm` and is installed at `~/.local/bin/arcllm`.
- User service file: `~/.config/systemd/user/arcllm.service`.
- Config file: `~/.config/arcllm/config.env`.
- Install script: `./scripts/install.sh`.
- Shell completions are installed for zsh via `~/.config/shell/local.sh`.
- `ENABLE_THINKING=false` is the default for the Qwen chat template.
- The router and workers support OpenAI-style `stream: true` SSE responses.
- Conversation persistence is currently client-managed; agent frameworks should send full message history each turn.
- `ZE_AFFINITY_MASK` defaults to `0` in `env.xpu.sh`.
- Override `ZE_AFFINITY_MASK` or pass `--tensor-parallel-size` when you start testing multi-GPU layouts.
- The current Python environment lives in `.venv`.
- `transformers` inference is currently the known-good Arc path on this machine.
- `vLLM` installs and initializes, but the current XPU attention backends do not complete generation on DG2/A770.
