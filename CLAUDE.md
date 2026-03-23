# llm-stack

3x Intel Arc A770 (48GB VRAM), i9-7900X, 64GB RAM.

## Quick Start

```bash
source /home/ryan/llm-stack/env.sglang-xpu.sh   # always first
bash scripts/arcllm-server.sh start              # start proxy
bash scripts/arcllm-server.sh load qwen3-32b     # load a model
bash scripts/arcllm-server.sh dashboard          # full status
bash scripts/arcllm-server.sh canary             # verify output is coherent
```

## Models — When to Use What

| Model | Speed | np | Use For | Limitation |
|-------|-------|-----|---------|------------|
| **Qwen3-32B** (dense) | 17-21 t/s | 16 | Pipeline (JSON, tools, structured output) | Slower than MoE |
| **Qwen3-30B-A3B** (MoE) | 20-25 t/s | 16 | Discord chat, unstructured tasks | Can't produce JSON (abliterated, leaks thinking) |
| **Qwen3.5-35B** | 13 t/s | 4 serialized | Best quality, Discord | Q8_1 concurrent bug — np>1 crashes |

**Proxy model names:** `qwen3-32b`, `qwen3-30b-moe`, `qwen35`, `qwen35-122b`

## Operations — arcllm-server.sh

```bash
# Server
start / stop / restart / status / logs / models / load <model> / unload

# Health
canary          # Verify output coherence (2+2=4 check)
dashboard       # Full system: GPU temps, model, queue, cache

# GPU Recovery (escalating severity)
gpu-check       # Verify 3/3 GPUs visible
gpu-reset       # Sysfs reset + i915 driver rebind
gpu-nuke        # PCI remove+rescan (LAST RESORT — can lose a GPU until reboot)

# Cache
cache-status    # Show JIT + slot cache sizes
cache-flush     # Clear all caches

# Full recovery
recover         # stop → flush → nuke → start → canary
```

## Benchmark Framework — MANDATORY

**NEVER use raw curl/bash to test server configs.** Use `scripts/bench/`.

```bash
cd /home/ryan/llm-stack/scripts
python3 -m bench help                       # list all suites
python3 -m bench frontier                   # regression baseline
python3 -m bench moepipeline.np16_c24k      # specific test
```

GPU reset between tests uses `sudo` (passwordless via `/etc/sudoers.d/gpu-reset`).
Results saved to `/tmp/bench_results.json`. Test docstrings ARE the docs.

### Key Suites

| Suite | What It Tests |
|-------|---------------|
| `frontier` | 32B dense baseline (21.7 t/s FUSED_MMQ) |
| `moefrontier` | 30B MoE baseline (25.7 t/s np=16) |
| `moepipeline` | MoE context × concurrency for pipeline |
| `densepipeline` | 32B context × concurrency for pipeline |
| `qwen35np` | Qwen3.5-35B Q8_1 concurrent crash investigation |
| `moechurn` | MoE data-churning optimization |

### Adding Tests

```python
"""One-line description. Docstring IS the documentation."""
from bench.base import BenchTest
from bench.config import BenchConfig

class TestName(BenchTest):
    base = BenchConfig(model="q4km", n_parallel=4, concurrent=4)
    def test_the_thing(self):
        self.run(self.base.with_(name="descriptive_name"))
```

## Data Pipeline (arc-tools/)

**Flow:** prospect search → site auditor → grabber → churner → golden entities

| Tool | Purpose | Model |
|------|---------|-------|
| **Site Auditor** | Browse + pattern websites (4-stage: triage → discovery → patterning → extraction) | qwen3-32b (needs JSON) |
| **Grabber** | Apply patterns at scale on Vultr fleet (no LLM during scaling) | — |
| **Churner** | DOM chunks → structured entities, cross-site dedup | qwen3-32b |
| **Discord Bot** | Chat interface | qwen3-30b-moe or qwen35 |

```bash
# Run site auditor
cd arc-tools/site-auditor
HENRY_MODEL=qwen3-32b python3 auditor.py --url https://example.com --entity-type "wedding venues"

# Docker stack (db, temporal, discord, grabber)
cd arc-tools && docker compose up -d
```

State in Postgres (ghostgraph DB): `sites`, `entity_audits`, `page_scans`, `url_patterns`, `dom_patterns`.

## Known Bugs

- **Q8_1 concurrent crash** — Qwen3.5-35B only, 2+ simultaneous slots. Proxy serializes with `max_active=1`. See `test_qwen35_np.py`.
- **MoE + flash attention** — IGC crash. Always `-fa off` for MoE.
- **GPU card0 disappears** after PCI nuke — only reboot restores it. Don't auto-nuke.
- **FUSED_MMQ unstable** after repeated GPU crashes — works clean, fails after crash cascades.

## Build

```bash
cd /home/ryan/llm-stack/llama.cpp-stable && source ../env.sglang-xpu.sh
cd build-sycl && cmake --build . --target llama-server -j$(nproc)
```

`llama.cpp/build-sycl` symlinks to `llama.cpp-stable/build-sycl`.

## SYCL Environment

- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` — batched mode, +7.5%
- `GGML_SYCL_DISABLE_GRAPH=1` — safer, graph replay bugged on DG2
- `GGML_SYCL_FUSED_MMQ=1` — +25% on dense models at np=16 (no benefit on MoE)

## Layout

```
llm-stack/
├── CLAUDE.md                   ← This file
├── env.sglang-xpu.sh           ← Source first
├── scripts/
│   ├── arcllm-proxy.py         ← Lazy-loading reverse proxy (port 11435)
│   ├── arcllm-server.sh        ← Operations toolkit
│   └── bench/                  ← Benchmark framework
├── arc-tools/
│   ├── site-auditor/           ← Henry-powered site analysis
│   ├── grabber/                ← Scale scraping (Temporal)
│   └── docker-compose.yml      ← Full stack
├── llama.cpp-stable/           ← Flagship build (stable-baseline)
├── llama.cpp-eptp/             ← EP experiment (ep-tp-combined)
├── llama.cpp/                  ← Upstream tracking (build-sycl → stable symlink)
├── bin/                        ← Frozen binaries (qwen35-gdn, 122b)
├── models/                     ← GGUF files
└── cache/                      ← L0 JIT cache, KV slot saves
```
