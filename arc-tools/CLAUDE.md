# arc-tools — Local AI Data Pipeline

3x Intel Arc A770, i9-7900X, 64GB RAM. Henry (Qwen3.5-35B-A3B) runs locally at 13 t/s.

## What This Does

Turn a product idea into the most comprehensive structured dataset for that entity type by combining data from many sources.

```
Idea: "wedding venues in Austin"
    ↓
Grabber: search engines find prospect sites (Bing/DuckDuckGo)
    → wehelpyouparty.com, theknot.com, weddingwire.com, yelp.com/austin-venues, ...
    ↓
Site Auditor: Henry audits each site (is it relevant? where's the data? what patterns?)
    → patterns stored per-site in Postgres
    ↓
Grabber: Vultr fleet applies patterns at scale, grabs labeled DOM chunks
    → raw chunks stored per-page per-site
    ↓
Churner: Henry extracts structured fields from chunks, resolves duplicates across sites
    → "LBJ Library Lawn" entity merges data from 9 sites = more complete than any single source
    ↓
Output: golden entities with traits, facets, groups
```

## Services

```
docker compose up -d

db              Postgres 16 + pgvector (temporal, churner, ghostgraph DBs)
redis           Grabber pattern cache + streams
temporal        Workflow engine
temporal-ui     :8233
worker-fast     Churner — fast extraction (4 concurrent)
worker-heavy    Churner — LLM reasoning (1 concurrent)
discord-bot     Henry chat (MODEL=qwen35)
grabber-api     :8000 — scraping API + pattern store
grabber-worker  Temporal worker (task_queue=grabber)
```

Plus standalone:
- `vultr-clone/` — local Docker that mimics Vultr worker (SPOOF_IPV6=true)
- `site-auditor/` — Henry + agent-browser, runs locally (not in Docker)

## The 3 Phases

### 1. Site Auditor (`site-auditor/`)

Henry explores websites using agent-browser CLI. Multi-fidelity pipeline — each stage sees only the detail level it needs:

| Stage | What | Data Format | LLM Calls |
|-------|------|-------------|-----------|
| Triage | Is site relevant? | Markdown (Jina Reader) | 1, thinking OFF |
| Discovery | Find list/detail pages | A11y tree (@refs) | Agent loop, thinking ON |
| Patterning | Find repeating DOM structure | Skeletal HTML (text stripped) | 1 per page, thinking OFF |
| Extraction | Label data chunks | Scoped HTML fragment | 1 per page, thinking OFF |

All state in Postgres: `sites`, `entity_audits`, `page_scans`, `url_patterns`, `dom_patterns`. Resumable after crash. Never re-scans same page.

### 2. Grabber (`grabber/`)

Applies patterns from Site Auditor at scale on Vultr fleet. Dumb and fast — no LLM needed during grabbing. Just CSS selectors → DOM chunks.

Output: semi-structured data per entity per site:
```json
{"url": "...", "domain": "...", "entity_type": "wedding venue",
 "chunks": [{"label": "reviews", "html": "...", "size": 4200}, ...]}
```

Chunks bridge into `churner.raw_ingests` via Temporal.

### 3. Churner (`churner/`)

Henry processes DOM chunks into structured entities. The key value: **cross-site entity resolution**. LBJ Library Lawn on wehelpyouparty has pricing, on theknot has reviews, on yelp has photos. Churner merges all into one golden entity.

Temporal workflows:
- `ChurnWorkflow` — extract → resolve/merge → traits (continuous)
- `SchemaEvolutionWorkflow` — evolve schema from data (every 4h)
- `GroupDiscoveryWorkflow` — generate faceted groups for SEO (every 2h)

## Henry Config

```
Model: Qwen3.5-35B-A3B (bin/llama-server-qwen35-gdn)
Proxy: localhost:11435
Config: np=4, c=32768 (8k/slot), split-mode layer, 3 GPUs
```

Key per-request params:
- `chat_template_kwargs: {"enable_thinking": false}` — disables reasoning (6 tokens vs 2000)
- Tool calling works (confirmed: JSON args, multi-step, function selection)

Context budget doc: `docs/CONTEXT-BUDGET.md` (test `-b 256` for 12k/slot after reboot)

## LLM Integration

All LLM calls go through Henry at `HENRY_URL` / `HENRY_MODEL`. No OpenRouter, no API keys.

- **Site Auditor**: `agent.py` → `henry_call()` (single-shot) + `run_agent()` (tool loop)
- **Grabber**: Henry only for quality checks (spot-check chunks, flag stale patterns)
- **Churner**: `activities.py` → `_llm_call()` via AsyncOpenAI client
- **Discord**: `inference.py` → high priority, same proxy

## Key Files

| What | Where |
|------|-------|
| Docker stack | `docker-compose.yml` |
| Henry proxy config | `../scripts/arcllm-proxy.py` |
| Site auditor | `site-auditor/auditor.py` |
| Auditor state | `site-auditor/state.py` + `schema.sql` |
| Browser tools | `site-auditor/tools.py` |
| Grabber brain | `grabber/vendor/` (transport, proxy, extraction) |
| Grabber Temporal | `grabber/temporal_*.py` |
| Churner workflows | `churner/workflows.py` |
| Churner activities | `churner/activities.py` |
| Vultr clone | `vultr-clone/` |

## GPU Recovery

```bash
pkill -9 -f llama-server; pkill -f arcllm-proxy; sleep 2
for pci in 0000:19:00.0 0000:67:00.0 0000:b5:00.0; do
  echo "$pci" | sudo tee /sys/bus/pci/drivers/i915/unbind; done
sleep 2
for pci in 0000:19:00.0 0000:67:00.0 0000:b5:00.0; do
  echo "$pci" | sudo tee /sys/bus/pci/drivers/i915/bind; done
```

If garbled output persists: `rm -rf /tmp/neo_compiler_cache` then reboot.
