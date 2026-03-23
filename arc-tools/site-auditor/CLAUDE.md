# Site Auditor

Henry-powered site intelligence tool. Explores websites using agent-browser, discovers patterns for the grabber.

## Pipeline

```
audit_site("wehelpyouparty.com", "wedding venues")
  │
  ├─ Stage 0: Recon (no LLM)
  │   Check robots.txt, IPv6, WAF → sites table
  │
  ├─ Stage 1: Triage (markdown, 1 LLM call, thinking OFF)
  │   Jina Reader → markdown → Henry: "is this relevant?" → entity_audits table
  │
  ├─ Stage 2: Discovery (a11y tree, agent loop, thinking ON)
  │   Henry browses with agent-browser → finds list + detail pages
  │   Each page visit → page_scans table
  │
  ├─ Stage 3: Patterning (skeletal HTML, 1 call per page, thinking OFF)
  │   Skeletonize DOM → Henry identifies repeating selectors
  │   Results → url_patterns + dom_patterns tables
  │
  └─ Stage 4: Extraction (scoped HTML, 1 call per page, thinking OFF)
      Isolate entity container → Henry labels data chunks
      Results → dom_patterns.chunks

Output: patterns ready for grabber to use at scale
```

## State Management

Everything in Postgres (`ghostgraph` database). See `schema.sql`.

- `sites` — one row per domain, recon data, status
- `entity_audits` — one row per (site × entity_type), pipeline stage tracker
- `page_scans` — every page visited, with description and outcome
- `url_patterns` — discovered URL patterns (index/detail page templates)
- `dom_patterns` — CSS selectors for entity containers and data chunks

**Resumable:** If the auditor crashes, `get_resumable_audit()` finds the last incomplete audit and `was_scanned()` skips already-visited pages.

## Key Findings

- `chat_template_kwargs: {"enable_thinking": false}` — disables reasoning per-request (6 tokens vs 2000)
- Stage 1 (triage) works at 8k context: 28s, correct results
- Stage 2 (discovery) needs 12k+ context — test `-b 256 -c 49152` after GPU reboot
- `agent-browser snapshot -i` returns ~300-400 tokens per page (compact refs)
- Skeletonize JS strips text, keeps DOM structure: ~500-800 tokens

## Files

| File | Purpose |
|------|---------|
| `auditor.py` | 4-stage pipeline orchestrator |
| `agent.py` | `henry_call()` (single-shot) + `run_agent()` (tool loop) |
| `tools.py` | agent-browser CLI wrappers + markdown fetcher + skeletonize JS |
| `recon.py` | robots.txt, IPv6, WAF checks |
| `context.py` | snapshot compression, noise stripping |
| `state.py` | Postgres state manager (sites, audits, scans, patterns) |
| `schema.sql` | Database schema |

## Usage

```bash
cd /home/ryan/llm-stack/arc-tools/site-auditor
python3 auditor.py --url https://wehelpyouparty.com --entity-type "wedding venues"
```
