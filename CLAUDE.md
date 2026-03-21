# llm-stack

3x Intel Arc A770 (48GB VRAM), i9-7900X, 64GB RAM.

## Current State

**Flagship:** `llama.cpp-stable` (branch: `stable-baseline`) — the working build.  
**EP experiment:** `llama.cpp-eptp` (branch: `ep-tp-combined`) — EP implementation, currently has WIP debug instrumentation, needs clean rebase before next EP work.

## Performance Baseline

| Config | Result | Notes |
|--------|--------|-------|
| Qwen3-30B-A3B ablit, layer-split, np=16 | **25.7 t/s** | Current flagship result |
| Qwen3-32B Q4_K_M, tensor-split, np=16, FUSED_MMQ=1 | **21.7 t/s** | Dense model |
| Qwen3-32B Q4_K_M, tensor-split, np=16, FUSED_MMQ=0 | **17.7 t/s** | Stable build (FUSED_MMQ not yet ported) |
| MoE 2.7B, single GPU, np=1 | **20.7 t/s** | Fast iteration model |
| Q8_0, any np | ~3.3 t/s | Bandwidth-bound, not worth optimizing |

## Worktrees

| Dir | Branch | Purpose |
|-----|--------|---------|
| `llama.cpp-stable/` | `stable-baseline` | **FLAGSHIP** — all working optimizations, bench framework docs |
| `llama.cpp-eptp/` | `ep-tp-combined` | EP experiment — SPLIT_AXIS_2, EP dispatch, needs rebase |
| `llama.cpp/` | `master` | Upstream tracking (row-split Nemotron experiments) |

## Models

**DO NOT use `Qwen3-30B-A3B-REAM-heretic-i1`** — garbled output on all builds. Quantization/merge issue, not an EP bug.

| Model | Path | Status |
|-------|------|--------|
| Qwen3-30B-A3B-abliterated Q4_K_M | `models/Qwen/Qwen3-30B-A3B-abliterated-GGUF/qwen3-30b-a3b-abliterated-q4_k_m.gguf` | ✅ Works, 128 experts |
| Qwen3-32B Q4_K_M | `models/Qwen/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf` | ✅ Works, dense |
| Qwen1.5-MoE-A2.7B Q2_K | `models/Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF/Qwen1.5-MoE-A2.7B-Chat.Q2_K.gguf` | ✅ Works, fast iteration |
| Qwen3-30B-A3B-REAM-heretic-i1 | `models/Qwen/Qwen3-30B-A3B-REAM-heretic-i1-GGUF/` | ❌ Broken, deleted |

## Build

```bash
# Flagship build (use this)
cd /home/ryan/llm-stack/llama.cpp-stable
source ../env.sglang-xpu.sh
cd build-sycl && cmake --build . --target llama-server -j$(nproc)

# EP build
cd /home/ryan/llm-stack/llama.cpp-eptp
source ../env.sglang-xpu.sh
cd build-sycl && cmake --build . --target llama-server -j$(nproc)
```

`llama.cpp/build-sycl` symlinks to `llama.cpp-stable/build-sycl` — the bench framework uses it.

## Benchmark Framework — MANDATORY for all GPU/model/server testing

**NEVER run raw bash/curl commands to test llama-server configs.** Use `scripts/bench/`. This is not optional.

Raw bash testing caused cascading failures: crashed GPUs left in DEVICE_LOST, no GPU utilization data, no records of what was tested.

```bash
cd /home/ryan/llm-stack/scripts
python3 -m bench help                       # list all suites and tests
python3 -m bench frontier                   # regression baseline (21.7 t/s)
python3 -m bench moefrontier.np16           # MoE baseline (25.7 t/s)
python3 -m bench thinking.1x_1024tok        # specific test
```

The framework: acquires `render` group, sources env, resets GPUs between tests, captures GPU freq/power/temp, saves JSON to `/tmp/bench_results.json`.

### Adding tests

Create `scripts/bench/tests/test_<name>.py`. The module docstring IS the documentation — update it as you run tests. No separate markdown docs for GPU/SYCL issues.

```python
"""One-line description.

## Context / Results / Relevant Files
...
"""
from bench.base import BenchTest
from bench.config import BenchConfig

class TestName(BenchTest):
    base = BenchConfig(model="q4km", n_parallel=4, concurrent=4)

    def test_the_thing(self):
        self.run(self.base.with_(name="descriptive_name", disable_graph=True))
```

### Config levers (scripts/bench/config.py)

- Server: `model`, `split_mode`, `n_parallel`, `context`, `batch`, `flash_attn`, `kv_quant`
- SYCL env: `disable_graph`, `immediate_cmdlists`, `affinity`
- Experimental: `.with_flags(FUSED_MMQ="1")` for A/B env var testing
- Test: `concurrent`, `max_tokens`, `prompt`, `timeout`

## Environment

```bash
source /home/ryan/llm-stack/env.sglang-xpu.sh   # always first
```

- User must be in `render` group for GPU access
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` (batched) — +7.5% throughput
- `GGML_SYCL_DISABLE_GRAPH=1` — safer; `=0` untested at long sequences
- **IGC crashes on MoE + flash attention** — always use `flash_attn=False` for MoE tests
- GPU card0 can go missing after hard resets — reboot restores it

## Known Issues / Anti-Patterns

- FUSED_MMQ not ported to stable build — MoE gets 25.7 t/s without it; dense loses ~4 t/s
- EP (tensor-split) crashes in stable build — EP code only in `llama.cpp-eptp`
- Don't trust `--split-mode tensor` in stable for MoE (GGML_ASSERT on split_state)
- 128 experts ÷ 3 GPUs doesn't divide cleanly — EP needs padding/masking logic for abliterated model
- REAM-heretic-i1 looks like an EP bug but isn't — it's broken on all builds

## Next Priorities

1. **Port FUSED_MMQ to stable** — recover 4 t/s on dense, probably free on MoE too
2. **EP on abliterated model** — fix 128÷3 unequal split (padding/masking in meta backend)
3. **N-gram speculative decoding** — zero-cost test, potentially +15%
4. **Fused expert aggregation kernel** — designed, not built, estimated -55ms/token

## Project Layout

```
llm-stack/
├── CLAUDE.md                   ← This file
├── env.sglang-xpu.sh           ← Always source this first
├── llama.cpp-stable/           ← FLAGSHIP worktree
│   ├── build-sycl/             ← Working binary
│   └── docs/                   ← ROADMAP, TECHNIQUES, EP-DEBUG, etc.
├── llama.cpp-eptp/             ← EP experiment worktree
├── llama.cpp/                  ← Upstream tracking (master)
├── models/                     ← GGUF model files
├── scripts/
│   ├── bench/                  ← Python benchmark framework
│   ├── arcllm-proxy.py         ← Lazy-loading reverse proxy
│   └── arcllm-server.sh        ← Start/stop wrapper
└── cache/                      ← L0 compiler cache, KV slot saves
```
