# Development Guide

How to work on this project. Read CLAUDE.md first for current state.

## Repo Structure

Two repos live in this directory:

1. **`dashw00d/arcllm`** — this repo. Scripts, proxy, bench framework, docs, Discord bot.
2. **`dashw00d/llama.cpp`** — fork of ggml-org/llama.cpp. Our SYCL kernel work lives here as branches, checked out as worktrees.

## Worktrees

llama.cpp uses git worktrees so multiple branches coexist as directories:

```
llama.cpp/           → master (upstream tracking)
llama.cpp-stable/    → stable-baseline (FLAGSHIP — working build)
llama.cpp-eptp/      → ep-tp-combined (EP experiment)
llama.cpp-qwen35/    → qwen35-support (Qwen3.5 architecture)
```

The `llama.cpp/` dir is the main git repo. All worktrees are children of it.

### Key symlink

`llama.cpp/build-sycl` → `llama.cpp-stable/build-sycl`

The bench framework uses `llama.cpp/build-sycl/bin/llama-server`. This symlink makes it always use the flagship build.

### Creating a new worktree (for experiments)

```bash
cd llama.cpp
git worktree add -b my-experiment ../llama.cpp-my-experiment ep-tp-combined
cd ../llama.cpp-my-experiment
mkdir build-sycl && cd build-sycl
source ../../env.sglang-xpu.sh
cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build . --target llama-server -j$(nproc)
```

This creates a new branch forked from `ep-tp-combined` with its own build dir. If it works, cherry-pick into the real branch. If not, `git worktree remove llama.cpp-my-experiment`.

### Pushing llama.cpp branches

```bash
cd llama.cpp
git push fork stable-baseline ep-tp-combined qwen35-support
```

`origin` = upstream ggml-org/llama.cpp, `fork` = dashw00d/llama.cpp.

## Environment

```bash
source /home/ryan/llm-stack/env.sglang-xpu.sh   # ALWAYS first
```

Sets up conda, SYCL, Level Zero. Required for building and running.

User must be in `render` group for GPU access.

## Building

```bash
cd llama.cpp-stable/build-sycl   # or whichever worktree
source ../../env.sglang-xpu.sh
cmake --build . --target llama-server -j$(nproc)
```

## Testing — Use the Bench Framework

**NEVER run raw bash/curl commands to test llama-server.** Use `scripts/bench/`.

```bash
cd scripts
python3 -m bench help                    # list all suites
python3 -m bench frontier                # dense regression (21.7 t/s)
python3 -m bench moefrontier.np16        # MoE regression (25.7 t/s)
python3 -m bench moechurn.c2048          # specific test
```

The framework: resets GPUs between tests, captures GPU freq/power/temp, saves JSON results. Raw testing crashes GPUs with no recovery.

### Adding a test

Create `scripts/bench/tests/test_<name>.py`. The module docstring IS the documentation.

```python
"""One-line description.

## Context / Results
...
"""
from bench.base import BenchTest
from bench.config import BenchConfig

class TestName(BenchTest):
    base = BenchConfig(model="q4km", n_parallel=4, concurrent=4)

    def test_the_thing(self):
        self.run(self.base.with_(name="descriptive_name"))
```

### Testing a different build

To bench a non-stable build, override the build path:

```python
base = BenchConfig(model="q4km", build="../../llama.cpp-experiment/build-sycl")
```

Or symlink `llama.cpp/build-sycl` temporarily.

## Proxy (arcllm-proxy)

Lazy-loading reverse proxy. Listens on `:11435`, spawns llama-server on first request.

```bash
# Systemd service
systemctl --user start arcllm
systemctl --user status arcllm
systemctl --user restart arcllm    # picks up proxy code changes

# Manual
scripts/arcllm-server.sh start
scripts/arcllm-server.sh stop
scripts/arcllm-server.sh status
```

Model registrations are in `scripts/arcllm-proxy.py` (the `_register()` calls).

SYCL env vars are set in `SYCL_ENV` dict at the top of the proxy.

### Install systemd service (fresh machine)

```bash
cp config/arcllm.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable arcllm
systemctl --user start arcllm
```

## Discord Bot + Churner

```bash
cd arc-tools
cp .env.example .env              # add DISCORD_TOKEN
docker compose up -d              # starts temporal, postgres, bot, workers
docker compose logs discord-bot   # check bot status
docker compose down               # stop everything
```

Bot talks to proxy on `:11435`. High-priority (Discord) gets slot 0, low-priority (churner) gets any available slot.

## GPU Recovery

If GPUs go DEVICE_LOST (crash, bad kernel, etc):

```bash
# Step 1: Kill all GPU consumers
pkill -9 -f llama-server; pkill -9 -f arcllm-proxy; sleep 2

# Step 2: Hardware reset via sysfs
for p in /sys/class/drm/card*/device/reset; do echo 1 | sudo tee "$p"; done
sleep 3

# Step 3: Verify — if sycl-ls sees 3 GPUs, you're good
source env.sglang-xpu.sh && sycl-ls | grep -c level_zero:gpu

# Step 4: If sycl-ls sees 0 GPUs, unbind/rebind the i915 driver
for pci in 0000:19:00.0 0000:67:00.0 0000:b5:00.0; do
  echo "$pci" | sudo tee /sys/bus/pci/drivers/i915/unbind
done
sleep 2
for pci in 0000:19:00.0 0000:67:00.0 0000:b5:00.0; do
  echo "$pci" | sudo tee /sys/bus/pci/drivers/i915/bind
done
sleep 3
sycl-ls | grep -c level_zero:gpu  # should show 3
```

The bench framework (`scripts/bench/runner.py`) does this automatically with
fallback — sysfs reset first, then driver rebind if L0 still can't see GPUs.

If card0 disappears entirely after all of this, reboot.

## Models

Downloaded via HuggingFace CLI:

```bash
source env.sglang-xpu.sh
huggingface-cli download <repo> --include "*Q4_K_M*" --local-dir models/Qwen/<name>
```

See CLAUDE.md for which models are active and their paths.
