# Testing Guide — llm-stack

## Golden Rule

**All tests go through the live proxy at localhost:11435.** Never launch llama-server directly for testing. The proxy sets the correct env vars (FUSED_MMQ, DISABLE_GRAPH, CMDLISTS) — launching without it produces different behavior.

## Quick Reference

```bash
source env.sglang-xpu.sh                         # always first
bash scripts/arcllm-server.sh start               # start proxy
bash scripts/arcllm-server.sh load qwen3-32b      # load model
bash scripts/arcllm-server.sh canary              # verify coherent output

cd scripts
python3 -m bench productionload                   # quality + throughput
python3 -m bench np4optimize.baseline             # throughput only
```

## Test Types

### 1. Production Load Tests (RECOMMENDED)

Tests quality AND throughput against the live proxy. No GPU touching.

```bash
python3 -m bench productionload
```

Tests:
- `json_extraction` — churner: structured JSON from HTML
- `tool_calling` — auditor: browser tool selection
- `classification` — triage: relevance + confidence
- `context_capacity` — 1179 prompt tokens handled
- `concurrent_4x` — all 4 slots active, measures aggregate t/s

**Prerequisites:** Proxy running with model loaded.

### 2. Optimization Tests (proxy-based)

A/B tests for env var and flag changes. Hit proxy directly.

```bash
python3 -m bench np4optimize.baseline
```

To test a change:
1. Edit `scripts/arcllm-proxy.py` (SYCL_ENV or model flags)
2. `arcllm-server.sh stop && arcllm-server.sh start`
3. Load model (first request triggers load)
4. Run `python3 -m bench np4optimize.baseline`
5. Compare numbers, revert if worse

### 3. Bench Framework Tests (GPU-isolated)

These launch their OWN llama-server on port 8400. Used for raw hardware benchmarks.

```bash
python3 -m bench frontier           # 32B np=16 regression baseline
python3 -m bench moefrontier        # 30B MoE baseline
python3 -m bench moepipeline        # MoE context scaling
```

**WARNING:** These kill any llama-server on port 8400 between tests. They do NOT touch the proxy (port 11435/18400) anymore.

**WARNING:** If these crash, they may leave GPUs in a bad state. The bench runner does a sysfs reset between tests but will NOT do driver rebind or PCI nuke automatically. If GPUs don't recover, run `arcllm-server.sh gpu-reset` manually.

### 4. Site Auditor (end-to-end)

```bash
cd arc-tools/site-auditor
HENRY_MODEL=qwen3-32b python3 auditor.py \
  --url https://wehelpyouparty.com \
  --entity-type "wedding venues"
```

Requires: proxy + model loaded, Postgres (arc-tools-db-1 container), agent-browser.

## What NOT To Do

### Never automatically:
- **PCI remove/rescan** (`gpu-nuke`) — can lose a GPU that only comes back on reboot
- **i915 driver unbind/rebind** — can break display state, cause kernel flip_done errors
- **Kill arcllm-proxy** — the proxy manages model lifecycle; killing it orphans the backend
- **Launch llama-server directly** — env mismatch (FUSED_MMQ, CMDLISTS) causes different behavior

### These are manual-only (last resort):
```bash
arcllm-server.sh gpu-reset    # sysfs reset + driver rebind (safe-ish)
arcllm-server.sh gpu-nuke     # PCI remove+rescan (LAST RESORT)
arcllm-server.sh cache-flush  # clear JIT + slot caches
arcllm-server.sh recover      # full: stop → flush → nuke → start
```

## GPU Recovery Decision Tree

```
Model producing garbled output?
├── YES → arcllm-server.sh unload, then reload
│   ├── Still garbled? → arcllm-server.sh cache-flush all
│   │   └── Still garbled? → reboot
│   └── Fixed → corrupted slot cache (auto-cleared on canary fail)
│
Server won't start (alloc failure)?
├── Check: is another llama-server running? (ps aux | grep llama)
├── Check: is a monitor plugged into a compute GPU?
├── Try: arcllm-server.sh gpu-reset
│   └── Still fails? → reboot
│
DEVICE_LOST during inference?
├── arcllm-server.sh stop
├── arcllm-server.sh gpu-reset
├── arcllm-server.sh start
│   └── Still crashes? → reboot
```

## Current Production Config

```
Model:     Qwen3-32B Q4_K_M (19GB)
Proxy:     np=4, c=32768 (8192 tok/slot)
Env:       FUSED_MMQ=1, CMDLISTS=2, DISABLE_GRAPH=1
Speed:     ~7 t/s single, ~8.6 t/s 4-concurrent
Quality:   JSON ✓, tool calling ✓, classification ✓
```

## Known Env Dependencies

| Var | Value | Effect if Wrong |
|-----|-------|-----------------|
| GGML_SYCL_FUSED_MMQ | 1 | **Required.** Without it: DEVICE_LOST on concurrent, 3x slower |
| SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS | 2 | Per-thread immediate. 0=batched (slower concurrent), 1=single (crashes) |
| GGML_SYCL_DISABLE_GRAPH | 1 | Graph replay broken on DG2. 0=crashes |
| ZE_AFFINITY_MASK | 0,1,2 | Must match GPU count |

## Adding New Tests

For proxy-based tests (recommended):
```python
# scripts/bench/tests/test_my_thing.py
from bench.base import BenchTest
from bench.config import BenchConfig

class TestMyThing(BenchTest):
    base = BenchConfig(name="dummy")  # no server launched

    def test_it(self):
        import urllib.request, json
        # Hit proxy directly
        resp = urllib.request.urlopen(
            urllib.request.Request(
                "http://localhost:11435/v1/chat/completions",
                json.dumps({"model": "qwen3-32b", ...}).encode(),
                {"Content-Type": "application/json"}),
            timeout=120)
        data = json.loads(resp.read())
        # Assert on data
```

For GPU-isolated tests (bench framework):
```python
class TestHardware(BenchTest):
    base = BenchConfig(model="q4km", n_parallel=4, concurrent=4)

    def test_np4(self):
        self.run(self.base.with_(name="my_np4_test"))
```

These launch their own server on port 8400. They do NOT affect the proxy.
