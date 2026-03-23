# GhostGraph API Reference

This is the API reference for GhostGraph, a distributed scraping engine. You submit URLs, it discovers sub-pages, generates extraction schemas, and extracts structured entities.

Base URL: `http://localhost:8000` (or the configured API host)

---

## System Overview

GhostGraph processes URLs through a 5-stage pipeline:

```
PENDING -> EXPLORING -> SCHEMING -> EXTRACTING -> COMPLETED
```

Any non-terminal state can transition to `FAILED` or `CANCELLED`.

- **PENDING**: Task created, waiting for a worker to claim it.
- **EXPLORING**: Worker is crawling the URL to discover detail pages (pagination, links, sitemaps).
- **SCHEMING**: Worker samples 3 detail pages and generates an extraction schema via LLM.
- **EXTRACTING**: Workers apply the schema to every discovered page, producing entities. Work items are processed in parallel.
- **COMPLETED**: All work items finished. Entities are stored.
- **FAILED**: Something went wrong. Check `failed_stage` and `error_message`.
- **CANCELLED**: Manually cancelled via API.

You orchestrate by creating projects, submitting tasks, monitoring progress, and merging results.

---

## Quick Start

```
1. POST /api/projects                              -- create a project
2. GET  /api/fleet/status                           -- check if workers are running
3. POST /api/fleet/deploy  {"count": 5}             -- deploy workers if none exist
4. POST /api/tasks/batch   {"tasks": [...]}         -- submit URLs
5. GET  /api/dashboard                              -- monitor progress
6. GET  /api/entities?entity_type=venue              -- get results
7. POST /api/entities/find-duplicates               -- deduplicate
8. POST /api/entities/merge                         -- merge duplicates
```

---

## Projects

Projects group related tasks and entities into a campaign.

### POST /api/projects

Create a project.

**Body:**
```json
{
  "name": "string (required)",
  "job_type": "string (required, e.g. 'wedding_venues')",
  "description": "string (optional)",
  "config": {}
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "name": "string",
  "job_type": "string",
  "description": "string|null",
  "config": {},
  "status": "active",
  "stats": {},
  "task_count": 0,
  "entity_count": 0,
  "created_at": "iso8601",
  "updated_at": "iso8601"
}
```

### GET /api/projects

List projects.

**Query params:** `status` (filter), `limit` (1-200, default 50), `offset`

### GET /api/projects/{id}

Get project detail with task and entity counts.

### PATCH /api/projects/{id}

Update project.

**Body (all optional):**
```json
{
  "name": "string",
  "description": "string",
  "status": "active|paused|completed|failed",
  "config": {},
  "stats": {}
}
```

### GET /api/projects/{id}/summary

Aggregated campaign view. Most useful endpoint for monitoring.

**Response:**
```json
{
  "id": "uuid",
  "name": "string",
  "status": "string",
  "task_states": {
    "pending": 0,
    "exploring": 0,
    "scheming": 0,
    "extracting": 0,
    "completed": 0,
    "failed": 0,
    "cancelled": 0
  },
  "entity_count": 0,
  "recent_failures": [
    {
      "task_id": "uuid",
      "url": "string",
      "error_message": "string|null",
      "failed_stage": "string|null",
      "updated_at": "iso8601|null"
    }
  ]
}
```

---

## Tasks

Tasks are the unit of work. One task = one URL to process through the pipeline.

### POST /api/tasks

Create a single task.

**Body:**
```json
{
  "url": "string (required)",
  "entity_type": "string (required)",
  "deadline_minutes": "int (optional, default 30)",
  "project_id": "uuid (optional)"
}
```

`entity_type` is freeform text that tells the schema generator what to extract. Examples: `venue`, `venue_listing`, `restaurant`, `business`, `job_listing`, `real_estate`, `product`.

**Response (201):**
```json
{
  "task_id": "uuid",
  "state": "pending",
  "url": "string",
  "entity_type": "string",
  "deadline": "iso8601"
}
```

### POST /api/tasks/batch

Create multiple tasks. Max 100 per request.

**Body:**
```json
{
  "tasks": [
    {"url": "string", "entity_type": "string", "project_id": "uuid"},
    ...
  ]
}
```

**Response (201):**
```json
{
  "created": [{"task_id": "uuid", "state": "pending", ...}],
  "count": 50
}
```

If backpressure is hit mid-batch, the response returns only the tasks that were created. Check `count` vs what you submitted.

### GET /api/tasks

List tasks.

**Query params:** `state`, `entity_type`, `created_after` (ISO datetime), `limit` (1-200, default 50), `offset`

### GET /api/tasks/{id}

Full task detail.

**Response:**
```json
{
  "task_id": "uuid",
  "url": "string",
  "entity_type": "string",
  "state": "pending|exploring|scheming|extracting|completed|failed|cancelled",
  "failed_stage": "string|null",
  "error_message": "string|null",
  "deadline": "iso8601|null",
  "urls_discovered": "int|null",
  "schema_fields": "int|null",
  "items_total": 0,
  "items_completed": 0,
  "items_failed": 0,
  "created_at": "iso8601",
  "explore_completed_at": "iso8601|null",
  "schema_completed_at": "iso8601|null",
  "completed_at": "iso8601|null",
  "updated_at": "iso8601"
}
```

### GET /api/tasks/{id}/work-items

List work items for a task. Work items are the individual pages being extracted during the EXTRACTING phase.

**Query params:** `state` (filter), `limit` (1-500, default 100), `offset`

**Response:**
```json
{
  "work_items": [
    {
      "id": "uuid",
      "task_id": "uuid",
      "url": "string",
      "state": "pending|processing|completed|failed|cancelled",
      "claimed_by": "worker_uuid|null",
      "attempts": 0,
      "max_attempts": 3,
      "error_message": "string|null",
      "entity_id": "uuid|null"
    }
  ],
  "count": 50
}
```

### POST /api/tasks/{id}/cancel

Cancel a task. Works from any non-terminal state. Also cancels pending work items.

**Response:** `{"message": "Task cancelled", "task_id": "uuid", "state": "cancelled"}`

**Errors:** 409 if task is already completed/failed/cancelled.

### POST /api/tasks/{id}/retry

Create a new task from a failed or cancelled task's URL and entity_type. Returns a new task.

**Errors:** 409 if original task is not in failed/cancelled state.

---

## Entities

Entities are the structured data extracted from pages.

### GET /api/entities

List entities.

**Query params:** `entity_type`, `source_domain`, `status` (active|merged), `limit` (1-500, default 50), `offset`

### GET /api/entities/{id}

Single entity detail.

**Response:**
```json
{
  "id": "uuid",
  "entity_type": "string",
  "project_id": "uuid|null",
  "data": {"name": "...", "address": "...", ...},
  "meta": {"source_url": "...", ...},
  "source_type": "string|null",
  "source_ref": "string|null",
  "source_domain": "string|null",
  "content_hash": "sha256",
  "status": "active|merged",
  "created_at": "iso8601",
  "updated_at": "iso8601"
}
```

The `data` field contains the extracted structured data. Fields depend on the entity_type and what the schema generator found.

### GET /api/entities/stats

Aggregate counts by domain and entity type.

**Response:**
```json
{
  "total": 5000,
  "by_domain": [
    {"source_domain": "theknot.com", "count": 1200},
    {"source_domain": "weddingwire.com", "count": 800},
    ...
  ],
  "by_type": [
    {"entity_type": "venue", "count": 4000},
    {"entity_type": "venue_listing", "count": 1000}
  ]
}
```

Use this to identify which aggregator domains yielded the most data.

### POST /api/entities/search

Text search across entity JSONB data.

**Body:**
```json
{
  "query": "string (required)",
  "entity_type": "string (optional)",
  "source_domain": "string (optional)",
  "limit": 50
}
```

Searches the `data` JSONB column using ILIKE. Useful for finding entities by name, address, etc.

### POST /api/entities/find-duplicates

Find potential duplicate entity groups by comparing specified data fields.

**Body:**
```json
{
  "project_id": "uuid (optional)",
  "entity_type": "string (optional)",
  "match_fields": ["address"],
  "threshold": 0.8
}
```

`match_fields`: list of keys in `data` to compare. Common choices: `["address"]`, `["name", "city"]`, `["name", "state"]`.

`threshold`: minimum average similarity (0.0-1.0). Default 0.8. Uses normalized string comparison: 1.0 for exact match, 0.5 for containment, 0.0 otherwise.

Caps at 1000 entities for pairwise comparison. Filter by project_id or entity_type to stay within that limit.

**Response:**
```json
{
  "groups": [
    {
      "entities": [{"id": "...", "data": {...}}, ...],
      "confidence": 0.95
    }
  ],
  "count": 25
}
```

### POST /api/entities/merge

Merge multiple entities into one.

**Body:**
```json
{
  "entity_ids": ["uuid1", "uuid2", "uuid3"],
  "primary_id": "uuid1"
}
```

`primary_id` (optional): which entity's data takes precedence. If omitted, the entity with the most data fields wins.

Merge behavior:
- Scalars: primary entity's values win.
- Lists: unioned (no duplicates).
- Dicts: recursively merged.
- Non-primary entities get `status: "merged"` and `meta.merged_into: "<primary_id>"`.
- Primary entity gets `meta.merged_from: ["<id2>", "<id3>"]`.

**Response:** The updated primary entity.

---

## Patterns

Patterns are cached extraction schemas for specific domains. The system creates them automatically. You can inspect and clean them up.

### GET /api/patterns

List patterns.

**Query params:** `domain`, `trust_level` (provisional|trusted), `quarantined` (true|false), `limit`, `offset`

### GET /api/patterns/domain/{domain}

Get all patterns for a domain.

**Response:**
```json
{
  "patterns": [
    {
      "id": "uuid",
      "domain": "theknot.com",
      "pattern_type": "extraction",
      "config": {...},
      "trust_level": "provisional|trusted",
      "confidence": 0.85,
      "usage_count": 42,
      "success_rate": 0.92,
      "quarantined": false
    }
  ],
  "count": 1
}
```

Patterns with low `success_rate` or `quarantined: true` will be regenerated.

### DELETE /api/patterns/{id}

Delete a pattern. Forces schema regeneration next time the domain is scraped.

---

## Fleet Management

Deploy and manage Vultr VPS worker instances.

### GET /api/fleet/status

List all fleet instances.

**Response:**
```json
{
  "total": 10,
  "running": 8,
  "instances": [
    {
      "id": "vultr-instance-id",
      "label": "gg-worker-ewr-000",
      "main_ip": "1.2.3.4",
      "region": "ewr",
      "status": "active",
      "power_status": "running"
    }
  ]
}
```

### POST /api/fleet/deploy

Deploy worker instances.

**Body:**
```json
{
  "count": 5,
  "regions": ["ewr", "ord", "lax"],
  "branch": "master"
}
```

`count`: 1-50 instances. `regions` (optional): Vultr region codes. Default: ewr, ord, lax, dfw, atl, mia, sea, sjc. `branch` (optional): git branch to deploy.

Workers auto-provision via cloud-init: install dependencies, clone repo, start systemd service.

### DELETE /api/fleet/destroy

Destroy all fleet instances. Requires `confirm: true`.

**Body:**
```json
{"confirm": true}
```

### POST /api/fleet/restart

Rolling restart: SSH git pull + systemctl restart on all workers.

**Body:**
```json
{
  "rolling": true,
  "batch_size": 5
}
```

### GET /api/fleet/verify

SSH health check across all running workers. Checks service status, IPv6, memory, camoufox.

**Response:**
```json
{
  "healthy": 8,
  "total": 10,
  "results": [
    {
      "label": "gg-worker-ewr-000",
      "ip": "1.2.3.4",
      "region": "ewr",
      "service": "active",
      "ipv6": "2",
      "memory": "512/2048MB (25%)",
      "camoufox": "ok",
      "healthy": true
    }
  ]
}
```

---

## Workers (Internal Fleet Status)

Low-level worker monitoring. Reads from Postgres worker registrations and Redis heartbeats.

### GET /workers/health

Postgres/Redis connectivity and latency, worker counts.

**Response:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "postgres_latency_ms": 1.5,
  "redis_latency_ms": 0.8,
  "total_workers": 10,
  "active_workers": 8,
  "stale_workers": 1
}
```

### GET /workers/fleet-status

Detailed utilization breakdown.

**Response:**
```json
{
  "total_workers": 10,
  "busy": 6,
  "idle": 2,
  "stale": 1,
  "dead": 1,
  "utilization_pct": 60.0,
  "streams": {
    "pipeline": {"pending": 50, "pel_size": 3, "lag": 47},
    "exploration": {"pending": 10, "pel_size": 2, "lag": 8},
    "schema": {"pending": 5, "pel_size": 1, "lag": 4},
    "extract": {"pending": 200, "pel_size": 8, "lag": 192}
  },
  "assessment": "healthy|degraded|critical|no_workers"
}
```

Use this to decide whether to scale up or down:
- `assessment: "no_workers"`: deploy workers immediately.
- `utilization_pct > 80%` with growing backlogs: deploy more workers.
- `utilization_pct < 20%` with empty queues: safe to destroy fleet.

### GET /workers/stream-health

Per-stream PEL size, consumer count, lag.

### POST /workers/{id}/hard-reset

Mark a specific worker as dead and reset its in-progress jobs to pending.

### POST /workers/nuke-everything

Emergency reset. Clears all Redis state, fails all active tasks, cancels all work items, marks all workers dead. Destructive and irreversible.

---

## Dashboard

### GET /api/dashboard

Single-call system overview. Designed for AI orchestrator polling.

**Response:**
```json
{
  "task_summary": {
    "pending": 100,
    "exploring": 5,
    "scheming": 2,
    "extracting": 10,
    "completed": 500,
    "failed": 15,
    "cancelled": 3
  },
  "recent_failures": [
    {
      "task_id": "uuid",
      "url": "string",
      "error_message": "string|null",
      "failed_stage": "exploring|scheming|extracting|null",
      "updated_at": "iso8601"
    }
  ],
  "worker_summary": {
    "total": 10,
    "active": 8,
    "stale": 1,
    "dead": 1
  },
  "stream_backlog": {
    "pipeline": 50,
    "exploration": 10,
    "schema_gen": 5,
    "extract": 200
  },
  "entity_count": 5000,
  "throughput": {
    "tasks_completed_24h": 450,
    "entities_extracted_24h": 3200
  }
}
```

Poll this every 30-60 seconds during active campaigns.

---

## Error Handling

| Code | Meaning | Action |
|------|---------|--------|
| 201  | Created | Success |
| 400  | Bad request | Check request body |
| 404  | Not found | Invalid ID |
| 409  | Conflict | Task in terminal state, can't cancel/retry |
| 429  | Backpressure | Stream at hard limit. Wait 30-60s and retry |
| 500  | Server error | Check API logs |

### Task Failures

When a task has `state: "failed"`:
- `failed_stage`: which pipeline stage failed (exploring, scheming, extracting)
- `error_message`: human-readable error description

Common failure patterns:
- `failed_stage: "exploring"`, "0 detail URLs found": site blocks crawlers or has no listing pages at that URL.
- `failed_stage: "scheming"`, "0 fields": LLM couldn't generate a schema from the sample pages.
- `failed_stage: "extracting"`, "Circuit breaker": 60%+ of work items failed. Extraction schema may be wrong or site is blocking.

### Retry Strategy

```
POST /api/tasks/{id}/retry
```

Creates a fresh task with the same URL and entity_type. Only works on failed/cancelled tasks.

---

## Scaling Strategy

### Scale Up
1. Check `GET /api/dashboard` -- look at `stream_backlog` and `worker_summary`.
2. If `stream_backlog.pipeline > 100` and `worker_summary.active < 5`, deploy more workers.
3. `POST /api/fleet/deploy {"count": 5}`
4. Wait 3-5 minutes for cloud-init to complete.
5. Verify: `GET /api/fleet/verify`

### Scale Down
1. Check `GET /api/dashboard` -- all stream backlogs near 0, all tasks completed/failed.
2. `DELETE /api/fleet/destroy {"confirm": true}`

### Rolling Update
1. Push code changes to git.
2. `POST /api/fleet/restart {"rolling": true, "batch_size": 3}`
3. Workers pull latest code and restart without downtime.

---

## Entity Types

`entity_type` is a freeform string you set when creating a task. It guides the LLM schema generator on what fields to extract. Use descriptive, specific types:

| entity_type | Use for | Expected fields |
|-------------|---------|-----------------|
| `venue` | Full venue profile pages | name, address, phone, website, capacity, pricing, amenities |
| `venue_listing` | Search/directory listing results | name, address, phone, rating |
| `restaurant` | Restaurant pages | name, address, phone, cuisine, hours, price_range |
| `business` | Generic business pages | name, address, phone, website, category |
| `job_listing` | Job postings | title, company, location, salary, description |
| `real_estate` | Property listings | address, price, bedrooms, bathrooms, sqft |
| `product` | E-commerce product pages | name, price, description, images, sku |

The schema generator adapts to whatever entity_type you provide. These are suggestions, not a fixed list.
