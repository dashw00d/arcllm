# Grabber

Web scraping intelligence for the arc-tools stack. Learns extraction patterns from any website with 1-2 Henry (local LLM) calls per domain, then extracts at scale with zero LLM cost. Output feeds into the churner for curation.

Forked from GhostGraph. LLM hardwired to Henry (Qwen3.5-35B via arcllm-proxy). Orchestration migrating from Redis Streams → Temporal.

## Architecture

### Pipeline: gather → curate

```
Temporal: GrabWorkflow
  ├── explore_urls     → vendor/discovery/discoverer.py (5-phase URL discovery)
  ├── generate_schema  → vendor/schema_generator/generator.py (4-tier cascade)
  ├── extract_batch    → vendor/extraction/extractor.py (batch extraction)
  └── bridge_to_churner → churner.raw_ingests (cross-DB insert)
                           ↓
                   ChurnWorkflow (churner):
                     extract → resolve/merge → traits → facets → groups
```

### LLM: Henry (local, free, fast)

All LLM calls go to Henry at `HENRY_URL` (default: `http://localhost:11435/v1/chat/completions`).
Model: `HENRY_MODEL` (default: `qwen35` — Qwen3.5-35B-A3B at 13 t/s).
No OpenRouter, no API keys, no cost tracking. Henry is unlimited.

### Extraction Cascade (unchanged from GhostGraph)

```
Tier 1:   extruct (JSON-LD, microdata, OpenGraph) — free
Tier 1.5: ValueDiscoverer — 600+ regex patterns — free
Tier 2:   SelectorBuilder — CSS selectors from known values — free
Tier 3:   LLM region discovery — Henry identifies missing data regions
```

All tiers always run. Cross-validation across 3 sample pages.

### Orchestration

**Temporal** (new): `temporal_worker.py` → task queue `grabber`
- `GrabWorkflow`: full pipeline (explore → schema → extract → bridge)
- Activities wrap vendor/ code directly

**Redis Streams** (legacy, still used by API): `workers/unified_worker.py`
- Still runs for the API's internal job dispatching
- Will be fully replaced by Temporal

## Key Files

| What | Where |
|------|-------|
| Temporal workflow | `temporal_workflows.py` |
| Temporal activities | `temporal_activities.py` |
| Temporal worker | `temporal_worker.py` |
| Schema cascade | `vendor/schema_generator/generator.py` |
| URL discovery | `vendor/discovery/discoverer.py` |
| Extraction | `vendor/extraction/extractor.py` |
| LLM calls (URL classify) | `vendor/discovery/url_classifier.py` |
| LLM calls (region discovery) | `vendor/schema_generator/llm_tier.py` |
| Config | `app/core/config.py` |
| API health | `http://localhost:8000/health` |

## Docker Services

```yaml
grabber-api:     FastAPI on :8000 (scraping API + REST endpoints)
grabber-worker:  Temporal worker (task_queue=grabber, max_concurrent=2)
```

Both use: `HENRY_URL`, `HENRY_MODEL`, `POSTGRES_URL`, `REDIS_URL`

## Style

- Python 3.12
- vendor/ is the extraction brain — pure logic, no framework deps
- app/ is the API shell — thin routers over queries
- Don't add tests, docstrings, or type annotations to code you didn't change
- Keep it simple
