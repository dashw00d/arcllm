"""
Task router: CRUD and lifecycle endpoints for scraping tasks.

POST /api/tasks          - create task, enqueue to Redis Stream
GET  /api/tasks          - list with filters (state, entity_type, created_after)
GET  /api/tasks/{id}     - full task state including progress
GET  /api/tasks/{id}/work-items - list work items for a task
POST /api/tasks/{id}/cancel  - cancel a task
POST /api/tasks/{id}/retry   - create new task from failed task's config
POST /api/tasks/batch    - create multiple tasks
POST /api/tasks/recipe-pipeline - submit recipe pipeline tasks by zip code
"""

from __future__ import annotations

import csv
import json
import logging
import pathlib
import uuid
from datetime import datetime
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.db.client import get_cursor
from app.services import job_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

# Stream names for task dispatch
TASK_STREAM = "stream:jobs:pipeline"
RECIPE_PIPELINE_STREAM = "stream:jobs:recipe_pipeline"

# Zip CSV path (relative to project root)
ZIP_CSV = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "us_zipcodes.csv"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TaskCreate(BaseModel):
    url: str
    entity_type: str
    deadline_minutes: Optional[int] = None
    project_id: Optional[str] = None


class TaskCreateResponse(BaseModel):
    task_id: str
    state: str
    url: str
    entity_type: str
    deadline: str


class BatchTaskCreate(BaseModel):
    tasks: list[TaskCreate]


class BatchTaskCreateResponse(BaseModel):
    created: list[TaskCreateResponse]
    count: int


class TaskDetail(BaseModel):
    task_id: str
    url: str
    entity_type: str
    state: str
    failed_stage: Optional[str] = None
    error_message: Optional[str] = None
    deadline: Optional[str] = None
    urls_discovered: Optional[int] = None
    schema_fields: Optional[int] = None
    items_total: int = 0
    items_completed: int = 0
    items_failed: int = 0
    created_at: Optional[str] = None
    explore_completed_at: Optional[str] = None
    schema_completed_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: Optional[str] = None


class TaskListResponse(BaseModel):
    tasks: list[TaskDetail]
    count: int


class MessageResponse(BaseModel):
    message: str
    task_id: str
    state: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(dt: Any) -> Optional[str]:
    """Convert a datetime or None to ISO string."""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _row_to_detail(row: dict) -> TaskDetail:
    """Convert a DB row dict to TaskDetail."""
    return TaskDetail(
        task_id=str(row["id"]),
        url=row["url"],
        entity_type=row["entity_type"],
        state=row["state"],
        failed_stage=row.get("failed_stage"),
        error_message=row.get("error_message"),
        deadline=_ts(row.get("deadline")),
        urls_discovered=row.get("urls_discovered"),
        schema_fields=row.get("schema_fields"),
        items_total=row.get("items_total", 0) or 0,
        items_completed=row.get("items_completed", 0) or 0,
        items_failed=row.get("items_failed", 0) or 0,
        created_at=_ts(row.get("created_at")),
        explore_completed_at=_ts(row.get("explore_completed_at")),
        schema_completed_at=_ts(row.get("schema_completed_at")),
        completed_at=_ts(row.get("completed_at")),
        updated_at=_ts(row.get("updated_at")),
    )


def _set_deprecation_headers(response: Any) -> None:
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "Wed, 30 Jun 2027 00:00:00 GMT"
    response.headers["Link"] = '</api/jobs>; rel="successor-version"'


def _coerce_config_object(raw_config: Any) -> dict[str, Any]:
    """Best-effort parse of task/project config payloads into a dict."""
    config: Any = raw_config
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            config = {}
    if not isinstance(config, dict):
        return {}
    return dict(config)


def _build_recipe_requeue_config(row: dict[str, Any], project_name: str) -> dict[str, Any]:
    """
    Build stream config for recipe requeue.

    Uses stored task config when present, with legacy reconstruction fallback
    from task URL + project config for rows created before config backfill.
    """
    config = _coerce_config_object(row.get("config"))
    project_config = _coerce_config_object(row.get("project_config"))

    parsed = urlparse(str(row.get("url") or ""))
    params = parse_qs(parsed.query)
    parsed_page_type = parsed.path.lstrip("/")
    query_from_url = params.get("query", [""])[0] or params.get("q", [""])[0]
    zip_code = params.get("zip", [""])[0]

    if not config.get("domain"):
        config["domain"] = project_config.get("domain") or parsed.netloc
    if not config.get("page_type"):
        config["page_type"] = project_config.get("page_type") or parsed_page_type or "default"
    if not config.get("query"):
        config["query"] = project_config.get("query") or query_from_url or "wedding venues"
    if not config.get("entity_type"):
        config["entity_type"] = row.get("entity_type")
    if zip_code and not config.get("zip_code"):
        config["zip_code"] = zip_code

    # Keep legacy defaults so old tasks without config remain recoverable.
    config.setdefault("dedup_key", "ypid")
    config.setdefault("count", 10)
    config.setdefault("max_pages", 5)
    config.setdefault("batch_size", 5)
    config.setdefault("rate_limit_seconds", 0.1)
    config.setdefault("project_name", project_name)
    return config


async def _enqueue_task(request: Request, task_id: str, url: str, entity_type: str) -> None:
    """Enqueue a task to the Redis Stream for worker pickup."""
    redis = request.app.state.redis
    settings = get_settings()

    # Backpressure check
    stream_len = await redis.xlen(TASK_STREAM)
    if stream_len >= settings.stream_hard_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Stream at hard limit ({settings.stream_hard_limit}). Try again later.",
        )

    await redis.xadd(
        TASK_STREAM,
        {
            "task_id": task_id,
            "url": url,
            "entity_type": entity_type,
        },
        maxlen=10000,
        approximate=True,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=TaskCreateResponse, status_code=201)
async def create_task(body: TaskCreate, request: Request, response: Response) -> TaskCreateResponse:
    """Create a new scraping task and enqueue it for processing."""
    _set_deprecation_headers(response)
    settings = get_settings()
    deadline_minutes = body.deadline_minutes or settings.default_deadline_minutes
    payload = {
        "url": body.url,
        "entity_type": body.entity_type,
    }
    if body.project_id:
        payload["project_id"] = body.project_id

    job_row = await job_service.submit_job(
        redis_client=request.app.state.redis,
        tenant_id="public",
        job_type="pipeline",
        version="v1",
        payload=payload,
        idempotency_key=str(uuid.uuid4()),
        request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
        project_id=body.project_id,
        priority="normal",
        metadata={"source": "legacy:/api/tasks"},
        deadline_minutes=deadline_minutes,
        actor="legacy_api",
    )
    task_id = job_row.get("related_task_id")
    if not task_id:
        raise HTTPException(status_code=500, detail="Failed to materialize task for job")

    with get_cursor(commit=False) as cur:
        cur.execute("SELECT id, state, deadline FROM tasks WHERE id = %s::uuid", (task_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=500, detail="Task row missing after submission")

    return TaskCreateResponse(
        task_id=str(row["id"]),
        state=row["state"],
        url=body.url,
        entity_type=body.entity_type,
        deadline=_ts(row["deadline"]) or "",
    )


@router.get("/{task_id}", response_model=TaskDetail)
async def get_task(task_id: str) -> TaskDetail:
    """Get full task state including progress."""
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")

    with get_cursor(commit=False) as cur:
        cur.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return _row_to_detail(row)


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    state: Optional[str] = Query(None, description="Filter by state"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    created_after: Optional[str] = Query(None, description="ISO datetime, filter tasks created after"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> TaskListResponse:
    """List tasks with optional filters."""
    conditions: list[str] = []
    params: list[Any] = []

    if state:
        conditions.append("state = %s")
        params.append(state)
    if entity_type:
        conditions.append("entity_type = %s")
        params.append(entity_type)
    if created_after:
        conditions.append("created_at > %s")
        params.append(created_after)

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    query = f"SELECT * FROM tasks {where} ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_cursor(commit=False) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    tasks = [_row_to_detail(row) for row in rows]
    return TaskListResponse(tasks=tasks, count=len(tasks))


@router.post("/{task_id}/cancel", response_model=MessageResponse)
async def cancel_task(task_id: str) -> MessageResponse:
    """Cancel a task. Works from any non-terminal state."""
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")

    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE tasks
            SET state = 'cancelled', updated_at = NOW()
            WHERE id = %s
              AND state NOT IN ('completed', 'failed', 'cancelled')
            RETURNING id, state
            """,
            (task_id,),
        )
        row = cur.fetchone()

    if row is None:
        # Check if task exists at all
        with get_cursor(commit=False) as cur:
            cur.execute("SELECT state FROM tasks WHERE id = %s", (task_id,))
            existing = cur.fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="Task not found")
        raise HTTPException(
            status_code=409,
            detail=f"Task is already in terminal state: {existing['state']}",
        )

    # Also cancel pending work items
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE work_items
            SET state = 'cancelled', updated_at = NOW()
            WHERE task_id = %s AND state = 'pending'
            """,
            (task_id,),
        )
        cur.execute(
            """
            UPDATE jobs
            SET state = 'cancelled',
                cancelled_at = NOW(),
                updated_at = NOW()
            WHERE related_task_id = %s::uuid
              AND state NOT IN ('completed', 'failed', 'cancelled')
            """,
            (task_id,),
        )

    return MessageResponse(
        message="Task cancelled",
        task_id=str(row["id"]),
        state=row["state"],
    )


@router.post("/{task_id}/retry", response_model=TaskCreateResponse, status_code=201)
async def retry_task(task_id: str, request: Request, response: Response) -> TaskCreateResponse:
    """Create a new task from a failed task's configuration."""
    _set_deprecation_headers(response)
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")

    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, url, entity_type, state FROM tasks WHERE id = %s",
            (task_id,),
        )
        original = cur.fetchone()

    if original is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if original["state"] not in ("failed", "cancelled"):
        raise HTTPException(
            status_code=409,
            detail=f"Can only retry failed or cancelled tasks. Current state: {original['state']}",
        )

    settings = get_settings()
    deadline_minutes = settings.default_deadline_minutes
    payload = {
        "url": original["url"],
        "entity_type": original["entity_type"],
    }

    job_row = await job_service.submit_job(
        redis_client=request.app.state.redis,
        tenant_id="public",
        job_type="pipeline",
        version="v1",
        payload=payload,
        idempotency_key=str(uuid.uuid4()),
        request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
        project_id=None,
        priority="normal",
        metadata={"source": "legacy:/api/tasks/{id}/retry", "retry_of": task_id},
        deadline_minutes=deadline_minutes,
        actor="legacy_api",
    )
    new_task_id = job_row.get("related_task_id")
    if not new_task_id:
        raise HTTPException(status_code=500, detail="Failed to materialize retry task")

    with get_cursor(commit=False) as cur:
        cur.execute("SELECT id, state, deadline FROM tasks WHERE id = %s::uuid", (new_task_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=500, detail="Task row missing after retry submission")

    return TaskCreateResponse(
        task_id=str(row["id"]),
        state=row["state"],
        url=original["url"],
        entity_type=original["entity_type"],
        deadline=_ts(row["deadline"]) or "",
    )


@router.post("/batch", response_model=BatchTaskCreateResponse, status_code=201)
async def create_batch(body: BatchTaskCreate, request: Request, response: Response) -> BatchTaskCreateResponse:
    """Create multiple tasks in a single request."""
    _set_deprecation_headers(response)
    if not body.tasks:
        raise HTTPException(status_code=400, detail="No tasks provided")
    if len(body.tasks) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tasks per batch")

    created: list[TaskCreateResponse] = []

    for task_input in body.tasks:
        settings = get_settings()
        deadline_minutes = task_input.deadline_minutes or settings.default_deadline_minutes
        payload = {
            "url": task_input.url,
            "entity_type": task_input.entity_type,
        }
        if task_input.project_id:
            payload["project_id"] = task_input.project_id

        try:
            job_row = await job_service.submit_job(
                redis_client=request.app.state.redis,
                tenant_id="public",
                job_type="pipeline",
                version="v1",
                payload=payload,
                idempotency_key=str(uuid.uuid4()),
                request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
                project_id=task_input.project_id,
                priority="normal",
                metadata={"source": "legacy:/api/tasks/batch"},
                deadline_minutes=deadline_minutes,
                actor="legacy_api",
            )
            task_id = job_row.get("related_task_id")
            if not task_id:
                raise HTTPException(status_code=500, detail="Failed to materialize task")
            with get_cursor(commit=False) as cur:
                cur.execute("SELECT id, state, deadline FROM tasks WHERE id = %s::uuid", (task_id,))
                row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=500, detail="Task row missing after submission")
        except HTTPException as exc:
            if exc.status_code == 429:
                # If we hit backpressure, still return what we created so far
                logger.warning("Backpressure hit during batch at task %d", len(created) + 1)
                break
            raise

        created.append(
            TaskCreateResponse(
                task_id=str(row["id"]),
                state=row["state"],
                url=task_input.url,
                entity_type=task_input.entity_type,
                deadline=_ts(row["deadline"]) or "",
            )
        )

    return BatchTaskCreateResponse(created=created, count=len(created))


# ---------------------------------------------------------------------------
# Recipe pipeline submission
# ---------------------------------------------------------------------------

class RecipePipelineSubmit(BaseModel):
    domain: str
    page_type: str = "default"
    query: str
    entity_type: str = "venue"
    dedup_key: str = "ypid"
    project_name: str = "recipe-pipeline"
    rate_limit: float = 0.1
    max_pages: int = 5
    batch_size: int = 5
    count: int = 10
    deadline_minutes: int = 120
    # Zip filters (mutually exclusive: zip vs min_population+limit)
    zip: Optional[str] = None
    zips: Optional[list[str]] = None
    min_population: int = 0
    limit: int = 0
    offset: int = 0
    dry_run: bool = False


class RecipePipelineResponse(BaseModel):
    project_id: str
    project_name: str
    submitted: int
    skipped: int = 0
    zips_matched: int
    next_offset: int = 0
    dry_run: bool
    tasks: list[dict[str, Any]] = []


def _load_zips(
    zip_code: str | None,
    zip_list: list[str] | None,
    min_population: int,
    limit: int,
    offset: int = 0,
) -> list[dict]:
    """Load zip codes from CSV with filters."""
    if not ZIP_CSV.exists():
        raise HTTPException(status_code=500, detail=f"Zip CSV not found at {ZIP_CSV}")

    with open(ZIP_CSV) as f:
        reader = csv.DictReader(f)

        if zip_code:
            for row in reader:
                if row["zip"] == zip_code:
                    return [row]
            raise HTTPException(status_code=404, detail=f"Zip {zip_code} not found in CSV")

        if zip_list:
            wanted = set(zip_list)
            matched = []
            for row in reader:
                if row["zip"] in wanted:
                    matched.append(row)
                    wanted.discard(row["zip"])
                    if not wanted:
                        break
            if not matched:
                raise HTTPException(status_code=404, detail="None of the provided zips found")
            return matched

        # Population filter
        zips = []
        for row in reader:
            pop = int(row["population"]) if row["population"] else 0
            if pop >= min_population:
                zips.append(row)

        if offset > 0:
            zips = zips[offset:]

        if limit > 0:
            zips = zips[:limit]

        return zips


def _get_or_create_project(name: str, config: dict) -> str:
    """Get existing project by name or create one. Returns UUID."""
    return job_service.get_or_create_project(name, config)


@router.post("/recipe-pipeline", response_model=RecipePipelineResponse, status_code=201)
async def submit_recipe_pipeline(
    body: RecipePipelineSubmit,
    request: Request,
    response: Response,
) -> RecipePipelineResponse:
    """Submit recipe pipeline tasks for US zip codes.

    Bulk-inserts tasks into Postgres. SyncService feeder pushes to Redis in chunks.
    """
    _set_deprecation_headers(response)
    zips = _load_zips(body.zip, body.zips, body.min_population, body.limit, body.offset)
    if not zips:
        raise HTTPException(status_code=400, detail="No zip codes matched filters")

    project_id = _get_or_create_project(body.project_name, {
        "domain": body.domain,
        "page_type": body.page_type,
        "query": body.query,
    })

    if body.dry_run:
        preview = []
        for z in zips[:20]:
            preview.append({
                "zip": z["zip"],
                "city": z["city"],
                "state": z["state"],
                "population": z["population"],
                "lat": z["lat"],
                "lng": z["lng"],
            })
        return RecipePipelineResponse(
            project_id=project_id,
            project_name=body.project_name,
            submitted=0,
            zips_matched=len(zips),
            dry_run=True,
            tasks=preview,
        )

    result = await job_service.submit_recipe_pipeline_jobs(
        redis_client=request.app.state.redis,
        tenant_id="public",
        project_id=project_id,
        project_name=body.project_name,
        domain=body.domain,
        page_type=body.page_type,
        query=body.query,
        entity_type=body.entity_type,
        dedup_key=body.dedup_key,
        rate_limit_seconds=body.rate_limit,
        max_pages=body.max_pages,
        batch_size=body.batch_size,
        count=body.count,
        deadline_minutes=body.deadline_minutes,
        zip_rows=zips,
        request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
        actor="legacy_api",
        metadata_source="legacy:/api/tasks/recipe-pipeline",
    )

    submitted = result["submitted"]
    skipped = result["skipped"]

    return RecipePipelineResponse(
        project_id=project_id,
        project_name=body.project_name,
        submitted=submitted,
        skipped=skipped,
        zips_matched=len(zips),
        next_offset=body.offset + len(zips),
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# Work items
# ---------------------------------------------------------------------------

class WorkItemDetail(BaseModel):
    id: str
    task_id: str
    url: str
    state: str
    claimed_by: Optional[str] = None
    claimed_at: Optional[str] = None
    attempts: int
    max_attempts: int
    error_message: Optional[str] = None
    entity_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class WorkItemListResponse(BaseModel):
    work_items: list[WorkItemDetail]
    count: int


def _row_to_work_item(row: dict) -> WorkItemDetail:
    return WorkItemDetail(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        url=row["url"],
        state=row["state"],
        claimed_by=row.get("claimed_by"),
        claimed_at=_ts(row.get("claimed_at")),
        attempts=row.get("attempts", 0),
        max_attempts=row.get("max_attempts", 3),
        error_message=row.get("error_message"),
        entity_id=str(row["entity_id"]) if row.get("entity_id") else None,
        created_at=_ts(row.get("created_at")),
        updated_at=_ts(row.get("updated_at")),
    )


@router.get("/{task_id}/work-items", response_model=WorkItemListResponse)
async def list_work_items(
    task_id: str,
    state: Optional[str] = Query(None, description="Filter by work item state"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> WorkItemListResponse:
    """List work items for a task."""
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")

    # Verify task exists
    with get_cursor(commit=False) as cur:
        cur.execute("SELECT id FROM tasks WHERE id = %s", (task_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Task not found")

    conditions = ["task_id = %s"]
    params: list[Any] = [task_id]

    if state:
        conditions.append("state = %s")
        params.append(state)

    where = "WHERE " + " AND ".join(conditions)
    query = f"SELECT * FROM work_items {where} ORDER BY created_at LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_cursor(commit=False) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    items = [_row_to_work_item(row) for row in rows]
    return WorkItemListResponse(work_items=items, count=len(items))


# ---------------------------------------------------------------------------
# Requeue orphaned pending tasks back into the stream
# ---------------------------------------------------------------------------
@router.post("/recipe-pipeline/requeue")
async def requeue_pending_tasks(
    request: Request,
    project_name: str = "bing-wedding-us",
    limit: int = 1000,
):
    """Find pending tasks with no stream entry and re-add them.

    Reads config from each task's config column (set at submission time)
    rather than hardcoding values — works for any recipe pipeline project.
    """
    redis = request.app.state.redis
    settings = get_settings()
    stream_len = await redis.xlen(RECIPE_PIPELINE_STREAM)

    if stream_len >= settings.stream_hard_limit:
        raise HTTPException(status_code=429, detail="Stream at hard limit")

    room = settings.stream_hard_limit - stream_len
    actual_limit = min(limit, room)

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT t.id, t.url, t.entity_type, t.config, p.config AS project_config
            FROM tasks t
            JOIN projects p ON t.project_id = p.id
            WHERE p.name = %s AND t.state = 'pending'
            ORDER BY t.created_at
            LIMIT %s
            """,
            (project_name, actual_limit),
        )
        rows = cur.fetchall()

    enqueued = 0
    enqueued_task_ids: list[str] = []
    for row in rows:
        task_id = str(row["id"])
        config = _build_recipe_requeue_config(row, project_name)

        await redis.xadd(
            RECIPE_PIPELINE_STREAM,
            {"task_id": task_id, "config": json.dumps(config)},
        )
        enqueued += 1
        enqueued_task_ids.append(task_id)

    if enqueued_task_ids:
        with get_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE tasks SET enqueued_at = NOW() WHERE id = ANY(%s::uuid[])",
                (enqueued_task_ids,),
            )
            cur.execute(
                """
                UPDATE jobs
                SET state = 'enqueued',
                    enqueued_at = NOW(),
                    updated_at = NOW()
                WHERE related_task_id = ANY(%s::uuid[])
                  AND state IN ('pending', 'enqueued')
                """,
                (enqueued_task_ids,),
            )

    return {"enqueued": enqueued, "pending_total": len(rows), "stream_len_before": stream_len}
