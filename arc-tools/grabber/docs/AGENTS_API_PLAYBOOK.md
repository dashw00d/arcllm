# Agents API Playbook

This is the practical guide for agent clients that call GhostGraph APIs.

## 1) Quick Setup

- Base URL: `http://localhost:8000`
- Required headers:
  - `X-API-Key: <key>` (unless `API_REQUIRE_KEYS=false`)
  - `X-Request-ID: <uuid-or-stable-run-id>` (recommended)
- Main routes:
  - `GET /api/job-types`
  - `POST /api/jobs/validate`
  - `POST /api/jobs`
  - `GET /api/jobs/{job_id}?include_events=true`
  - `POST /api/jobs/{job_id}/retry`
  - `POST /api/jobs/{job_id}/cancel`
  - `POST /api/jobs/uploads`
  - `POST /api/jobs/recipe-pipeline/from-upload`

## 2) Standard Agent Flow

1. Discover supported job types.
2. Validate payload before submit.
3. Submit with an idempotency key.
4. Poll until terminal state.
5. Retry only if needed.

### Discover

```bash
curl -sS "$BASE/api/job-types" \
  -H "X-API-Key: $API_KEY"
```

### Validate

```bash
curl -sS "$BASE/api/jobs/validate" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "pipeline",
    "version": "v1",
    "payload": {
      "url": "https://example.com/listing/123",
      "entity_type": "venue"
    }
  }'
```

### Submit

```bash
curl -sS "$BASE/api/jobs" \
  -H "X-API-Key: $API_KEY" \
  -H "X-Request-ID: run-2026-02-11-001" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "pipeline",
    "version": "v1",
    "idempotency_key": "pipeline:venue:example-123",
    "deadline_minutes": 30,
    "priority": "normal",
    "payload": {
      "url": "https://example.com/listing/123",
      "entity_type": "venue"
    },
    "metadata": {
      "source": "agent"
    }
  }'
```

## 3) Job States

Canonical job states:

- `pending`
- `enqueued`
- `running`
- `completed`
- `failed`
- `cancelled`

Poll:

```bash
curl -sS "$BASE/api/jobs/$JOB_ID?include_events=true" \
  -H "X-API-Key: $API_KEY"
```

List/filter:

```bash
curl -sS "$BASE/api/jobs?state=failed&job_type=pipeline&limit=50&offset=0" \
  -H "X-API-Key: $API_KEY"
```

## 4) Retry and Cancel

Retry a terminal job:

```bash
curl -sS -X POST "$BASE/api/jobs/$JOB_ID/retry" \
  -H "X-API-Key: $API_KEY"
```

Cancel an active job:

```bash
curl -sS -X POST "$BASE/api/jobs/$JOB_ID/cancel" \
  -H "X-API-Key: $API_KEY"
```

## 5) File Upload Flow (Zip Example)

Upload CSV/TXT with zip codes:

```bash
curl -sS -X POST "$BASE/api/jobs/uploads" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@./zipcodes.csv" \
  -F "job_type=recipe_pipeline"
```

Then submit recipe pipeline from that upload:

```bash
curl -sS -X POST "$BASE/api/jobs/recipe-pipeline/from-upload" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "upload_id": "'"$UPLOAD_ID"'",
    "domain": "www.bing.com",
    "page_type": "maps_overlay",
    "query": "wedding venues",
    "entity_type": "venue",
    "project_name": "bing-wedding-us",
    "dry_run": false
  }'
```

## 6) Agent Reliability Rules

- Always send explicit `idempotency_key` for repeatable runs.
- Always run `/api/jobs/validate` before large fanout.
- Use `X-Request-ID` so one run maps to one traceable request lineage.
- Handle `429` with backoff.
- Treat `pending`/`enqueued` as in-flight, not stuck (unless long timeout exceeded in your own SLA).
- Use `include_events=true` for debugging and audit.

## 7) Generic Upload Job Example (`file_transform`)

Upload:

```bash
curl -sS -X POST "$BASE/api/jobs/uploads" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@./contacts.csv" \
  -F "job_type=file_transform"
```

Submit:

```bash
curl -sS -X POST "$BASE/api/jobs" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "file_transform",
    "version": "v1",
    "idempotency_key": "file-transform:contacts:run-1",
    "payload": {
      "upload_id": "'"$UPLOAD_ID"'",
      "mode": "extract_column",
      "text_column": "name"
    }
  }'
```
