# Job Schema + Upload Guide

This guide covers:

- creating a new dynamic job type
- defining payload schema
- wiring worker handler + stream
- validating/submitting jobs
- handling file uploads (`upload_id`) for file-driven jobs

## 1) What a Job Type Needs

Each job type is defined in `job_types` with:

- `job_type`, `version`
- `stream_name`
- `handler_module`, `handler_class`
- `payload_schema` (JSON schema-lite)
- `requires_task` (`true` = task-backed, `false` = direct job-backed)
- `enabled`

Schema support is intentionally lightweight in `job_service.validate_payload_schema`:

- root `type=object`
- `required`
- `properties.<field>.type`

## 2) Build the Worker Handler

Create a handler class under `workers/handlers/`, e.g.:

- `workers/handlers/my_job_handler.py`
- class: `MyJobHandler`

The unified worker resolves handlers dynamically from `job_types.handler_module` + `job_types.handler_class`.

## 3) Register the Job Type

Run SQL migration or insert manually:

```sql
INSERT INTO job_types (
  job_type,
  version,
  stream_name,
  handler_module,
  handler_class,
  payload_schema,
  requires_task,
  enabled,
  description
)
VALUES (
  'my_job',
  'v1',
  'stream:jobs:my_job',
  'workers.handlers.my_job_handler',
  'MyJobHandler',
  '{
    "type": "object",
    "required": ["url", "entity_type"],
    "properties": {
      "url": {"type": "string"},
      "entity_type": {"type": "string"}
    }
  }'::jsonb,
  TRUE,
  TRUE,
  'My custom dynamic job'
)
ON CONFLICT (job_type, version) DO UPDATE
SET stream_name = EXCLUDED.stream_name,
    handler_module = EXCLUDED.handler_module,
    handler_class = EXCLUDED.handler_class,
    payload_schema = EXCLUDED.payload_schema,
    requires_task = EXCLUDED.requires_task,
    enabled = EXCLUDED.enabled,
    description = EXCLUDED.description,
    updated_at = NOW();
```

At app startup:

- consumer groups are created for enabled streams
- unified worker picks up enabled `job_types`

## 4) Task-Backed vs Non-Task Jobs

### `requires_task = TRUE`

- Submit API creates `tasks` row + related `jobs` row.
- Worker receives `task_id` on stream.
- For non-`recipe_pipeline` task-backed jobs, payload should include:
  - `url` (required by current materializer)
  - `entity_type` (recommended)

### `requires_task = FALSE`

- No `tasks` row needed.
- Worker receives `job_id` on stream with JSON `config`.
- Good default for control/orchestration/file-processing jobs.

## 5) Validate + Submit

Validate:

```bash
curl -sS "$BASE/api/jobs/validate" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "my_job",
    "version": "v1",
    "payload": {
      "url": "https://example.com",
      "entity_type": "venue"
    }
  }'
```

Submit:

```bash
curl -sS "$BASE/api/jobs" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "my_job",
    "version": "v1",
    "idempotency_key": "my-job:example-001",
    "payload": {
      "url": "https://example.com",
      "entity_type": "venue"
    }
  }'
```

## 6) File Upload Jobs

GhostGraph stores uploaded files in `job_uploads`.

Upload file:

```bash
curl -sS -X POST "$BASE/api/jobs/uploads" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@./input.csv" \
  -F "job_type=my_job"
```

Response includes `upload_id`.

## 7) Two Upload Patterns

### A) Existing built-in path (zip code recipe fanout)

- `POST /api/jobs/recipe-pipeline/from-upload`
- Parses zip list, resolves rows, fans out recipe pipeline jobs.

### B) Generic custom job path (recommended for new jobs)

1. Put `upload_id` in your payload schema and mark it required.
2. Submit job with that `upload_id`.
3. In handler, fetch bytes from `job_uploads` and process.

Example schema fragment:

```json
{
  "type": "object",
  "required": ["upload_id"],
  "properties": {
    "upload_id": { "type": "string" }
  }
}
```

Example query in handler (using tenant safety checks):

```sql
SELECT u.data, u.filename, u.content_type
FROM job_uploads u
JOIN jobs j ON j.id = %s::uuid
WHERE u.id = %s::uuid
  AND u.tenant_id = j.tenant_id;
```

If your handler only has `task_id`, join via `jobs.related_task_id` first.

## 8) Ready Checklist

- Handler file exists and imports cleanly.
- `job_types` row inserted and `enabled=true`.
- `payload_schema` validates expected payload shape.
- `requires_task` choice matches handler behavior.
- Submit + poll path tested with one sample payload.
- Upload flow tested end-to-end if job uses files.

## 9) Working Reference Job Type: `file_transform`

This repo now includes a reference upload-driven non-task job type:

- `job_type`: `file_transform`
- `version`: `v1`
- `stream_name`: `stream:jobs:file_transform`
- `handler`: `workers.handlers.file_transform_handler.FileTransformHandler`
- `requires_task`: `false`

Payload contract:

```json
{
  "upload_id": "uuid",
  "mode": "count_rows | extract_column",
  "text_column": "required when mode=extract_column"
}
```
