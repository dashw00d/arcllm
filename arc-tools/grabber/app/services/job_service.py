"""
Canonical job submission and validation service.

This module provides:
  - job type lookup
  - lightweight JSON-schema-like payload validation
  - idempotent job insertion
  - optional API key auth + quota checks
  - enqueue + task materialization for task-backed job types
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
import csv
import io
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import HTTPException
import psycopg2.extras

from app.core.config import get_settings
from app.db.client import get_cursor

logger = logging.getLogger(__name__)
ZIP_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "us_zipcodes.csv"


@dataclass
class ApiAuthContext:
    api_key_id: str
    tenant_id: str
    is_admin: bool
    submit_rpm: int
    pending_limit: int
    payload_bytes_limit: int


def _json_load_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def canonical_payload_hash(job_type: str, version: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    data = f"{job_type}:{version}:{raw}".encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def resolve_job_type(job_type: str, version: str = "v1") -> Optional[dict[str, Any]]:
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT job_type, version, stream_name, handler_module, handler_class,
                   payload_schema, requires_task, enabled, description
            FROM job_types
            WHERE job_type = %s AND version = %s
            """,
            (job_type, version),
        )
        row = cur.fetchone()
    if not row:
        return None

    row["payload_schema"] = _json_load_maybe(row.get("payload_schema")) or {}
    return row


def list_job_types() -> list[dict[str, Any]]:
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT job_type, version, stream_name, handler_module, handler_class,
                   payload_schema, requires_task, enabled, description
            FROM job_types
            ORDER BY job_type, version
            """
        )
        rows = cur.fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        row["payload_schema"] = _json_load_maybe(row.get("payload_schema")) or {}
        result.append(row)
    return result


def _validate_type(name: str, value: Any, expected: str) -> None:
    if expected == "string" and not isinstance(value, str):
        raise HTTPException(status_code=422, detail=f"payload.{name} must be a string")
    if expected == "number" and not isinstance(value, (int, float)):
        raise HTTPException(status_code=422, detail=f"payload.{name} must be a number")
    if expected == "integer" and not isinstance(value, int):
        raise HTTPException(status_code=422, detail=f"payload.{name} must be an integer")
    if expected == "boolean" and not isinstance(value, bool):
        raise HTTPException(status_code=422, detail=f"payload.{name} must be a boolean")
    if expected == "object" and not isinstance(value, dict):
        raise HTTPException(status_code=422, detail=f"payload.{name} must be an object")
    if expected == "array" and not isinstance(value, list):
        raise HTTPException(status_code=422, detail=f"payload.{name} must be an array")


def validate_payload_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    """
    Lightweight validator for the subset of schema semantics we use.

    Supported keys:
      - type=object at root
      - required=[...]
      - properties.{field}.type
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be an object")

    if not isinstance(schema, dict):
        return

    expected_root = schema.get("type")
    if expected_root and expected_root != "object":
        raise HTTPException(status_code=500, detail="job type schema must have root type=object")

    required = schema.get("required") or []
    if isinstance(required, list):
        missing = [field for field in required if field not in payload]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"payload missing required fields: {', '.join(missing)}",
            )

    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return

    for key, prop in properties.items():
        if key not in payload or not isinstance(prop, dict):
            continue
        expected = prop.get("type")
        if expected:
            _validate_type(key, payload[key], expected)


def _hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def authenticate_api_key(raw_key: str | None, required: bool = True) -> Optional[ApiAuthContext]:
    settings = get_settings()
    if not settings.api_require_keys:
        return ApiAuthContext(
            api_key_id="anonymous",
            tenant_id="public",
            is_admin=True,
            submit_rpm=100000,
            pending_limit=1000000,
            payload_bytes_limit=2 * 1024 * 1024,
        )

    if not raw_key:
        if required:
            raise HTTPException(status_code=401, detail="Missing X-API-Key")
        return None

    key_hash = _hash_api_key(raw_key)
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            SELECT
                k.id,
                k.tenant_id,
                k.is_admin,
                COALESCE(l.submit_rpm, 120) AS submit_rpm,
                COALESCE(l.pending_limit, 10000) AS pending_limit,
                COALESCE(l.payload_bytes_limit, 262144) AS payload_bytes_limit
            FROM api_keys k
            LEFT JOIN api_key_limits l ON l.api_key_id = k.id
            WHERE k.key_hash = %s
              AND k.is_active = TRUE
            """,
            (key_hash,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Invalid API key")
        cur.execute("UPDATE api_keys SET last_used_at = NOW() WHERE id = %s", (row["id"],))

    return ApiAuthContext(
        api_key_id=str(row["id"]),
        tenant_id=row["tenant_id"],
        is_admin=bool(row.get("is_admin", False)),
        submit_rpm=int(row["submit_rpm"]),
        pending_limit=int(row["pending_limit"]),
        payload_bytes_limit=int(row["payload_bytes_limit"]),
    )


def enforce_quota_limits(auth: ApiAuthContext, payload_bytes: int) -> None:
    if payload_bytes > auth.payload_bytes_limit:
        raise HTTPException(
            status_code=413,
            detail=f"Payload exceeds limit ({auth.payload_bytes_limit} bytes)",
        )

    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM jobs
            WHERE tenant_id = %s
              AND created_at > NOW() - INTERVAL '1 minute'
            """,
            (auth.tenant_id,),
        )
        rpm = int((cur.fetchone() or {}).get("cnt", 0))
        if rpm >= auth.submit_rpm:
            raise HTTPException(
                status_code=429,
                detail=f"Submit rate limit exceeded ({auth.submit_rpm}/minute)",
            )

        cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM jobs
            WHERE tenant_id = %s
              AND state IN ('pending', 'enqueued', 'running')
            """,
            (auth.tenant_id,),
        )
        pending = int((cur.fetchone() or {}).get("cnt", 0))
        if pending >= auth.pending_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Pending job limit exceeded ({auth.pending_limit})",
            )


async def _ensure_stream_capacity(redis_client: Any, stream_name: str, required: int = 1) -> None:
    settings = get_settings()
    if required <= 0:
        return

    try:
        stream_len = await redis_client.xlen(stream_name)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Unable to check stream capacity") from exc

    if stream_len + required > settings.stream_hard_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Stream at hard limit ({settings.stream_hard_limit}). Try again later.",
        )


def _insert_event(job_id: str, event_type: str, actor: str, data: Optional[dict[str, Any]] = None) -> None:
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO job_events (job_id, event_type, actor, data)
            VALUES (%s, %s, %s, %s::jsonb)
            """,
            (job_id, event_type, actor, json.dumps(data or {})),
        )


def _insert_or_get_job(
    tenant_id: str,
    job_type: str,
    version: str,
    payload: dict[str, Any],
    idempotency_key: str,
    request_id: str,
    project_id: Optional[str],
    priority: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO jobs (
                tenant_id, job_type, version, state, payload, idempotency_key,
                request_id, project_id, priority, metadata
            )
            VALUES (
                %s, %s, %s, 'pending', %s::jsonb, %s,
                %s, %s::uuid, %s, %s::jsonb
            )
            ON CONFLICT (tenant_id, job_type, idempotency_key)
            DO UPDATE SET updated_at = NOW()
            RETURNING id, tenant_id, job_type, version, state, payload, idempotency_key,
                      request_id, project_id, priority, metadata, related_task_id,
                      error_message, enqueued_at, completed_at, cancelled_at,
                      created_at, updated_at, (xmax = 0) AS inserted
            """,
            (
                tenant_id,
                job_type,
                version,
                json.dumps(payload),
                idempotency_key,
                request_id,
                project_id,
                priority,
                json.dumps(metadata),
            ),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=500, detail="Failed to persist job")
    row["payload"] = _json_load_maybe(row.get("payload")) or {}
    row["metadata"] = _json_load_maybe(row.get("metadata")) or {}
    return row


async def _materialize_task_and_enqueue(
    redis_client: Any,
    job_row: dict[str, Any],
    definition: dict[str, Any],
    payload: dict[str, Any],
    deadline_minutes: int,
    actor: str,
) -> dict[str, Any]:
    job_id = str(job_row["id"])
    job_type = job_row["job_type"]
    stream_name = definition["stream_name"]

    payload = dict(payload)
    entity_type = str(payload.get("entity_type") or "unknown")
    project_id = job_row.get("project_id")
    task_id = str(uuid.uuid4())
    deadline = datetime.now(timezone.utc) + timedelta(minutes=deadline_minutes)

    if job_type == "recipe_pipeline":
        if not payload.get("url"):
            domain = payload.get("domain", "")
            page_type = payload.get("page_type", "default")
            if not domain:
                raise HTTPException(status_code=422, detail="payload.domain is required for recipe_pipeline")
            url = f"recipe://{domain}/{page_type}"
            zip_code = payload.get("zip_code")
            if zip_code:
                url = f"{url}?zip={zip_code}"
            payload["url"] = url
        payload.setdefault("entity_type", entity_type)
        url = payload["url"]
    else:
        url = payload.get("url")
        if not isinstance(url, str) or not url:
            raise HTTPException(status_code=422, detail="payload.url is required for task-backed jobs")
        payload.setdefault("entity_type", entity_type)

    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO tasks (id, url, entity_type, state, deadline, project_id, config)
            VALUES (%s, %s, %s, 'pending', %s, %s::uuid, %s::jsonb)
            RETURNING id
            """,
            (task_id, url, entity_type, deadline, project_id, json.dumps(payload)),
        )
        cur.execute(
            """
            UPDATE jobs
            SET related_task_id = %s::uuid,
                updated_at = NOW()
            WHERE id = %s::uuid
            """,
            (task_id, job_id),
        )

    message = {"task_id": task_id}
    if job_type == "recipe_pipeline":
        message["config"] = json.dumps(payload)
    else:
        message["url"] = url
        message["entity_type"] = entity_type
        message["config"] = json.dumps(payload)

    _insert_event(job_id, "task_created", actor, {"task_id": task_id})
    job_row["related_task_id"] = task_id
    try:
        await _ensure_stream_capacity(redis_client, stream_name, required=1)
        await redis_client.xadd(stream_name, message, maxlen=10000, approximate=True)
    except HTTPException as exc:
        if exc.status_code == 429:
            raise
        logger.warning(
            "Deferred enqueue for job %s task %s to stream %s: %s",
            job_id,
            task_id,
            stream_name,
            exc.detail,
        )
        _insert_event(
            job_id,
            "enqueue_deferred",
            actor,
            {"stream_name": stream_name, "reason": str(exc.detail)[:200]},
        )
        job_row["state"] = "pending"
        return job_row
    except Exception as exc:
        logger.warning(
            "Deferred enqueue for job %s task %s to stream %s: %s",
            job_id,
            task_id,
            stream_name,
            exc,
        )
        _insert_event(
            job_id,
            "enqueue_deferred",
            actor,
            {"stream_name": stream_name, "reason": str(exc)[:200]},
        )
        job_row["state"] = "pending"
        return job_row

    with get_cursor(commit=True) as cur:
        cur.execute("UPDATE tasks SET enqueued_at = NOW() WHERE id = %s::uuid", (task_id,))
        cur.execute(
            """
            UPDATE jobs
            SET state = 'enqueued',
                enqueued_at = NOW(),
                updated_at = NOW()
            WHERE id = %s::uuid
            """,
            (job_id,),
        )
    _insert_event(job_id, "enqueued", actor, {"stream_name": stream_name, "task_id": task_id})
    job_row["state"] = "enqueued"
    return job_row


async def _enqueue_non_task_job(
    redis_client: Any,
    job_row: dict[str, Any],
    definition: dict[str, Any],
    payload: dict[str, Any],
    actor: str,
) -> dict[str, Any]:
    job_id = str(job_row["id"])
    stream_name = definition["stream_name"]
    try:
        await _ensure_stream_capacity(redis_client, stream_name, required=1)
        await redis_client.xadd(
            stream_name,
            {"job_id": job_id, "config": json.dumps(payload)},
            maxlen=10000,
            approximate=True,
        )
    except HTTPException as exc:
        if exc.status_code == 429:
            raise
        logger.warning(
            "Deferred enqueue for non-task job %s to stream %s: %s",
            job_id,
            stream_name,
            exc.detail,
        )
        _insert_event(
            job_id,
            "enqueue_deferred",
            actor,
            {"stream_name": stream_name, "reason": str(exc.detail)[:200]},
        )
        job_row["state"] = "pending"
        return job_row
    except Exception as exc:
        logger.warning("Deferred enqueue for non-task job %s to stream %s: %s", job_id, stream_name, exc)
        _insert_event(
            job_id,
            "enqueue_deferred",
            actor,
            {"stream_name": stream_name, "reason": str(exc)[:200]},
        )
        job_row["state"] = "pending"
        return job_row

    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE jobs
            SET state = 'enqueued',
                enqueued_at = NOW(),
                updated_at = NOW()
            WHERE id = %s::uuid
            """,
            (job_id,),
        )
    _insert_event(job_id, "enqueued", actor, {"stream_name": stream_name})
    job_row["state"] = "enqueued"
    return job_row


async def submit_job(
    redis_client: Any,
    tenant_id: str,
    job_type: str,
    version: str,
    payload: dict[str, Any],
    idempotency_key: Optional[str],
    request_id: str,
    project_id: Optional[str],
    priority: str,
    metadata: Optional[dict[str, Any]],
    deadline_minutes: int,
    actor: str = "api",
) -> dict[str, Any]:
    definition = resolve_job_type(job_type=job_type, version=version)
    if not definition or not definition.get("enabled", False):
        raise HTTPException(status_code=404, detail=f"Unsupported job_type/version: {job_type}/{version}")

    if project_id:
        try:
            uuid.UUID(project_id)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="project_id must be a valid UUID") from exc

    validate_payload_schema(payload, definition.get("payload_schema") or {})

    normalized_metadata = metadata or {}
    if not isinstance(normalized_metadata, dict):
        raise HTTPException(status_code=422, detail="metadata must be an object")

    idem = idempotency_key or canonical_payload_hash(job_type, version, payload)
    job_row = _insert_or_get_job(
        tenant_id=tenant_id,
        job_type=job_type,
        version=version,
        payload=payload,
        idempotency_key=idem,
        request_id=request_id,
        project_id=project_id,
        priority=priority,
        metadata=normalized_metadata,
    )

    if job_row.get("inserted"):
        _insert_event(str(job_row["id"]), "submitted", actor, {"request_id": request_id})
    else:
        _insert_event(str(job_row["id"]), "deduplicated", actor, {"request_id": request_id})
        return job_row

    requires_task = bool(definition.get("requires_task", True))
    if requires_task:
        return await _materialize_task_and_enqueue(
            redis_client=redis_client,
            job_row=job_row,
            definition=definition,
            payload=payload,
            deadline_minutes=deadline_minutes,
            actor=actor,
        )

    return await _enqueue_non_task_job(
        redis_client=redis_client,
        job_row=job_row,
        definition=definition,
        payload=payload,
        actor=actor,
    )


def _map_task_state_to_job_state(task_state: str | None, enqueued_at: Any) -> str:
    if not task_state:
        return "pending"
    if task_state in ("completed", "failed", "cancelled"):
        return task_state
    if task_state in ("exploring", "scheming", "extracting"):
        return "running"
    if task_state == "pending" and enqueued_at is not None:
        return "enqueued"
    return "pending"


def get_job(job_id: str, tenant_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    where = "WHERE j.id = %s::uuid"
    params: list[Any] = [job_id]
    if tenant_id:
        where += " AND j.tenant_id = %s"
        params.append(tenant_id)

    with get_cursor(commit=False) as cur:
        cur.execute(
            f"""
            SELECT
                j.*,
                t.state AS task_state,
                t.error_message AS task_error_message,
                t.updated_at AS task_updated_at
            FROM jobs j
            LEFT JOIN tasks t ON t.id = j.related_task_id
            {where}
            """,
            params,
        )
        row = cur.fetchone()
    if not row:
        return None

    row["payload"] = _json_load_maybe(row.get("payload")) or {}
    row["metadata"] = _json_load_maybe(row.get("metadata")) or {}
    derived = _map_task_state_to_job_state(row.get("task_state"), row.get("enqueued_at"))
    if row.get("state") != derived and row.get("state") not in ("failed", "cancelled", "completed"):
        row["state"] = derived
    if row.get("task_error_message") and not row.get("error_message"):
        row["error_message"] = row["task_error_message"]
    return row


def list_jobs(
    tenant_id: Optional[str],
    state: Optional[str],
    job_type: Optional[str],
    project_id: Optional[str],
    created_after: Optional[str],
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    conditions: list[str] = []
    params: list[Any] = []

    if tenant_id:
        conditions.append("j.tenant_id = %s")
        params.append(tenant_id)
    if state:
        conditions.append("j.state = %s")
        params.append(state)
    if job_type:
        conditions.append("j.job_type = %s")
        params.append(job_type)
    if project_id:
        conditions.append("j.project_id = %s::uuid")
        params.append(project_id)
    if created_after:
        conditions.append("j.created_at > %s")
        params.append(created_after)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_cursor(commit=False) as cur:
        cur.execute(
            f"""
            SELECT
                j.*,
                t.state AS task_state,
                t.error_message AS task_error_message,
                t.updated_at AS task_updated_at
            FROM jobs j
            LEFT JOIN tasks t ON t.id = j.related_task_id
            {where}
            ORDER BY j.created_at DESC
            LIMIT %s OFFSET %s
            """,
            [*params, limit, offset],
        )
        rows = cur.fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        row["payload"] = _json_load_maybe(row.get("payload")) or {}
        row["metadata"] = _json_load_maybe(row.get("metadata")) or {}
        derived = _map_task_state_to_job_state(row.get("task_state"), row.get("enqueued_at"))
        if row.get("state") not in ("failed", "cancelled", "completed"):
            row["state"] = derived
        if row.get("task_error_message") and not row.get("error_message"):
            row["error_message"] = row["task_error_message"]
        result.append(row)
    return result


def list_job_events(job_id: str, limit: int = 100) -> list[dict[str, Any]]:
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT id, job_id, event_type, actor, data, created_at
            FROM job_events
            WHERE job_id = %s::uuid
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (job_id, limit),
        )
        rows = cur.fetchall()
    for row in rows:
        row["data"] = _json_load_maybe(row.get("data")) or {}
    return rows


def cancel_job(job_id: str, tenant_id: Optional[str], actor: str = "api") -> Optional[dict[str, Any]]:
    where = "WHERE id = %s::uuid"
    params: list[Any] = [job_id]
    if tenant_id:
        where += " AND tenant_id = %s"
        params.append(tenant_id)

    with get_cursor(commit=True) as cur:
        cur.execute(
            f"""
            UPDATE jobs
            SET state = 'cancelled',
                cancelled_at = NOW(),
                updated_at = NOW()
            {where}
              AND state NOT IN ('completed', 'failed', 'cancelled')
            RETURNING *
            """,
            params,
        )
        row = cur.fetchone()
    if row:
        _insert_event(str(row["id"]), "cancelled", actor, {})
    return row


def require_admin(auth: ApiAuthContext) -> None:
    if not auth.is_admin:
        raise HTTPException(status_code=403, detail="Admin API key required")


def _generate_api_key_value() -> str:
    return f"gg_{uuid.uuid4().hex}{uuid.uuid4().hex}"


def create_api_key(
    auth: ApiAuthContext,
    name: str,
    tenant_id: str,
    is_admin: bool,
    submit_rpm: int,
    pending_limit: int,
    payload_bytes_limit: int,
) -> dict[str, Any]:
    require_admin(auth)
    raw_key = _generate_api_key_value()
    key_hash = _hash_api_key(raw_key)

    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO api_keys (tenant_id, name, key_hash, is_active, is_admin)
            VALUES (%s, %s, %s, TRUE, %s)
            RETURNING id, tenant_id, name, is_active, is_admin, created_at, last_used_at
            """,
            (tenant_id, name, key_hash, is_admin),
        )
        row = cur.fetchone()
        cur.execute(
            """
            INSERT INTO api_key_limits (api_key_id, submit_rpm, pending_limit, payload_bytes_limit)
            VALUES (%s, %s, %s, %s)
            """,
            (row["id"], submit_rpm, pending_limit, payload_bytes_limit),
        )
    row["api_key"] = raw_key
    return row


def list_api_keys(auth: ApiAuthContext, tenant_id: Optional[str] = None) -> list[dict[str, Any]]:
    require_admin(auth)
    where = ""
    params: list[Any] = []
    if tenant_id:
        where = "WHERE k.tenant_id = %s"
        params.append(tenant_id)

    with get_cursor(commit=False) as cur:
        cur.execute(
            f"""
            SELECT
                k.id, k.tenant_id, k.name, k.is_active, k.is_admin, k.created_at, k.last_used_at,
                COALESCE(l.submit_rpm, 120) AS submit_rpm,
                COALESCE(l.pending_limit, 10000) AS pending_limit,
                COALESCE(l.payload_bytes_limit, 262144) AS payload_bytes_limit
            FROM api_keys k
            LEFT JOIN api_key_limits l ON l.api_key_id = k.id
            {where}
            ORDER BY k.created_at DESC
            """,
            params,
        )
        rows = cur.fetchall()
    return rows


def deactivate_api_key(auth: ApiAuthContext, key_id: str) -> Optional[dict[str, Any]]:
    require_admin(auth)
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE api_keys
            SET is_active = FALSE
            WHERE id = %s::uuid
            RETURNING id, tenant_id, name, is_active, is_admin, created_at, last_used_at
            """,
            (key_id,),
        )
        return cur.fetchone()


def rotate_api_key(auth: ApiAuthContext, key_id: str) -> Optional[dict[str, Any]]:
    require_admin(auth)
    new_key = _generate_api_key_value()
    key_hash = _hash_api_key(new_key)

    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE api_keys
            SET key_hash = %s,
                is_active = TRUE
            WHERE id = %s::uuid
            RETURNING id, tenant_id, name, is_active, is_admin, created_at, last_used_at
            """,
            (key_hash, key_id),
        )
        row = cur.fetchone()
    if row:
        row["api_key"] = new_key
    return row


def create_job_upload(
    auth: ApiAuthContext,
    filename: str,
    content_type: str,
    data: bytes,
    job_type: Optional[str] = None,
) -> dict[str, Any]:
    max_bytes = 10 * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload exceeds limit ({max_bytes} bytes)")

    digest = hashlib.sha256(data).hexdigest()
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO job_uploads (
                tenant_id, uploaded_by_api_key, job_type, filename, content_type,
                size_bytes, sha256, data
            )
            VALUES (%s, %s::uuid, %s, %s, %s, %s, %s, %s)
            RETURNING id, tenant_id, uploaded_by_api_key, job_type, filename,
                      content_type, size_bytes, sha256, created_at
            """,
            (
                auth.tenant_id,
                auth.api_key_id if auth.api_key_id != "anonymous" else None,
                job_type,
                filename,
                content_type,
                len(data),
                digest,
                data,
            ),
        )
        return cur.fetchone()


def get_job_upload(upload_id: str, tenant_id: str, include_data: bool = False) -> Optional[dict[str, Any]]:
    select_data = ", data" if include_data else ""
    with get_cursor(commit=False) as cur:
        cur.execute(
            f"""
            SELECT id, tenant_id, uploaded_by_api_key, job_type, filename,
                   content_type, size_bytes, sha256, created_at
                   {select_data}
            FROM job_uploads
            WHERE id = %s::uuid
              AND tenant_id = %s
            """,
            (upload_id, tenant_id),
        )
        return cur.fetchone()


def resolve_upload_for_job(upload_id: str, job_id: str, include_data: bool = True) -> dict[str, Any]:
    """
    Resolve an upload for a job with tenant-safe scoping.

    The upload must belong to the same tenant as the job row.
    Raises ValueError when IDs are invalid or row is not found.
    """
    try:
        uuid.UUID(upload_id)
        uuid.UUID(job_id)
    except ValueError as exc:
        raise ValueError("Invalid upload_id or job_id format") from exc

    select_data = ", u.data" if include_data else ""
    with get_cursor(commit=False) as cur:
        cur.execute(
            f"""
            SELECT
                u.id,
                u.tenant_id,
                u.uploaded_by_api_key,
                u.job_type,
                u.filename,
                u.content_type,
                u.size_bytes,
                u.sha256,
                u.created_at
                {select_data}
            FROM job_uploads u
            INNER JOIN jobs j
              ON j.tenant_id = u.tenant_id
            WHERE u.id = %s::uuid
              AND j.id = %s::uuid
            """,
            (upload_id, job_id),
        )
        row = cur.fetchone()

    if not row:
        raise ValueError("Upload not found for this job")
    return row


def _normalize_zip(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) < 5:
        return ""
    return digits[:5]


def parse_zip_codes_from_upload(data: bytes, filename: str = "") -> list[str]:
    if isinstance(data, memoryview):
        data = data.tobytes()

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    zip_codes: list[str] = []
    seen: set[str] = set()

    # Try CSV first for .csv files or comma-containing header.
    should_try_csv = filename.lower().endswith(".csv") or ("," in lines[0])
    if should_try_csv:
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames:
            zip_field = None
            for candidate in ("zip", "zipcode", "zip_code", "postal_code"):
                if candidate in reader.fieldnames:
                    zip_field = candidate
                    break
            if zip_field:
                for row in reader:
                    raw = str(row.get(zip_field, "")).strip()
                    zip_code = _normalize_zip(raw)
                    if zip_code and zip_code not in seen:
                        seen.add(zip_code)
                        zip_codes.append(zip_code)
                if zip_codes:
                    return zip_codes

    # Fallback: one zip per line.
    for line in lines:
        zip_code = _normalize_zip(line)
        if zip_code and zip_code not in seen:
            seen.add(zip_code)
            zip_codes.append(zip_code)

    if not zip_codes:
        raise HTTPException(status_code=400, detail="No valid zip codes found in upload")
    return zip_codes


def _load_zip_index() -> dict[str, dict[str, str]]:
    if not ZIP_CSV.exists():
        raise HTTPException(status_code=500, detail=f"Zip CSV not found at {ZIP_CSV}")

    index: dict[str, dict[str, str]] = {}
    with open(ZIP_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("zip"):
                index[row["zip"]] = row
    return index


def resolve_zip_rows(zip_codes: list[str]) -> list[dict[str, str]]:
    index = _load_zip_index()
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for z in zip_codes:
        row = index.get(z)
        if not row:
            missing.append(z)
            continue
        rows.append(row)
    if missing:
        sample = ", ".join(missing[:5])
        raise HTTPException(status_code=404, detail=f"Zip codes not found in dataset: {sample}")
    return rows


def get_or_create_project(name: str, config: dict[str, Any]) -> str:
    with get_cursor(commit=True) as cur:
        cur.execute("SELECT id FROM projects WHERE name = %s", (name,))
        row = cur.fetchone()
        if row:
            return str(row["id"])
        cur.execute(
            """
            INSERT INTO projects (name, job_type, config)
            VALUES (%s, 'recipe_pipeline', %s::jsonb)
            RETURNING id
            """,
            (name, json.dumps(config)),
        )
        return str(cur.fetchone()["id"])


async def submit_recipe_pipeline_jobs(
    redis_client: Any,
    tenant_id: str,
    project_id: str,
    project_name: str,
    domain: str,
    page_type: str,
    query: str,
    entity_type: str,
    dedup_key: str,
    rate_limit_seconds: float,
    max_pages: int,
    batch_size: int,
    count: int,
    deadline_minutes: int,
    zip_rows: list[dict[str, Any]],
    request_id: str,
    actor: str,
    metadata_source: str,
) -> dict[str, int]:
    if not zip_rows:
        return {"submitted": 0, "skipped": 0}

    try:
        uuid.UUID(project_id)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="project_id must be a valid UUID") from exc

    # Deduplicate zip rows in-request for deterministic idempotent behavior.
    unique_rows: list[dict[str, Any]] = []
    seen_zip: set[str] = set()
    for row in zip_rows:
        zip_code = str(row.get("zip", "")).strip()
        if not zip_code or zip_code in seen_zip:
            continue
        seen_zip.add(zip_code)
        unique_rows.append(row)

    records: list[dict[str, Any]] = []
    for row in unique_rows:
        lat = float(row["lat"])
        lng = float(row["lng"])
        zip_code = str(row["zip"])
        payload = {
            "domain": domain,
            "page_type": page_type,
            "query": query,
            "lat": lat,
            "lng": lng,
            "count": count,
            "zip_code": zip_code,
            "city": row.get("city", ""),
            "state": row.get("state", ""),
            "entity_type": entity_type,
            "dedup_key": dedup_key,
            "project_name": project_name,
            "project_id": project_id,
            "rate_limit_seconds": rate_limit_seconds,
            "max_pages": max_pages,
            "batch_size": batch_size,
        }
        idem = f"recipe:{project_id}:{domain}:{page_type}:{query}:{entity_type}:{zip_code}"
        url = f"recipe://{domain}/{page_type}?zip={zip_code}"
        records.append(
            {
                "idem": idem,
                "payload": payload,
                "url": url,
                "zip_code": zip_code,
                "entity_type": entity_type,
            }
        )

    if not records:
        return {"submitted": 0, "skipped": 0}

    idempotency_keys = [r["idem"] for r in records]
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT idempotency_key
            FROM jobs
            WHERE tenant_id = %s
              AND job_type = 'recipe_pipeline'
              AND idempotency_key = ANY(%s::text[])
            """,
            (tenant_id, idempotency_keys),
        )
        existing = {row["idempotency_key"] for row in cur.fetchall()}

    candidates = [r for r in records if r["idem"] not in existing]
    if not candidates:
        return {"submitted": 0, "skipped": len(records)}

    definition = resolve_job_type("recipe_pipeline", "v1")
    if not definition or not definition.get("enabled", False):
        raise HTTPException(status_code=404, detail="Unsupported job_type/version: recipe_pipeline/v1")
    stream_name = definition["stream_name"]

    await _ensure_stream_capacity(redis_client, stream_name, required=len(candidates))

    job_rows = [
        (
            tenant_id,
            "recipe_pipeline",
            "v1",
            json.dumps(r["payload"]),
            r["idem"],
            request_id,
            project_id,
            "normal",
            json.dumps({"source": metadata_source}),
        )
        for r in candidates
    ]

    with get_cursor(commit=True) as cur:
        inserted_jobs = psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO jobs (
                tenant_id, job_type, version, state, payload, idempotency_key,
                request_id, project_id, priority, metadata
            )
            VALUES %s
            ON CONFLICT (tenant_id, job_type, idempotency_key) DO NOTHING
            RETURNING id, idempotency_key
            """,
            job_rows,
            template="(%s, %s, %s, 'pending', %s::jsonb, %s, %s, %s::uuid, %s, %s::jsonb)",
            page_size=1000,
            fetch=True,
        ) or []

    if not inserted_jobs:
        return {"submitted": 0, "skipped": len(records)}

    inserted_map = {row["idempotency_key"]: str(row["id"]) for row in inserted_jobs}
    inserted_records = [r for r in candidates if r["idem"] in inserted_map]

    deadline = datetime.now(timezone.utc) + timedelta(minutes=deadline_minutes)
    task_rows: list[tuple[str, str, str, datetime, str, str, str]] = []
    job_task_pairs: list[tuple[str, str]] = []
    for r in inserted_records:
        task_id = str(uuid.uuid4())
        payload_json = json.dumps(r["payload"])
        task_rows.append(
            (
                task_id,
                r["url"],
                r["entity_type"],
                deadline,
                project_id,
                payload_json,
                inserted_map[r["idem"]],
            )
        )
        job_task_pairs.append((inserted_map[r["idem"]], task_id))

    with get_cursor(commit=True) as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO tasks (id, url, entity_type, state, deadline, project_id, config)
            VALUES %s
            """,
            [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in task_rows],
            template="(%s, %s, %s, 'pending', %s, %s::uuid, %s::jsonb)",
            page_size=1000,
        )
        psycopg2.extras.execute_values(
            cur,
            """
            UPDATE jobs AS j
            SET related_task_id = v.task_id::uuid,
                updated_at = NOW()
            FROM (VALUES %s) AS v(job_id, task_id)
            WHERE j.id = v.job_id::uuid
            """,
            job_task_pairs,
            template="(%s, %s)",
            page_size=1000,
        )

    # Enqueue in Redis with one pipeline.
    pipe = redis_client.pipeline(transaction=False)
    for task_id, _, _, _, _, payload_json, _ in task_rows:
        pipe.xadd(stream_name, {"task_id": task_id, "config": payload_json}, maxlen=10000, approximate=True)
    await pipe.execute()

    task_ids = [r[0] for r in task_rows]
    job_ids = [r[6] for r in task_rows]
    with get_cursor(commit=True) as cur:
        cur.execute("UPDATE tasks SET enqueued_at = NOW() WHERE id = ANY(%s::uuid[])", (task_ids,))
        cur.execute(
            """
            UPDATE jobs
            SET state = 'enqueued',
                enqueued_at = NOW(),
                updated_at = NOW()
            WHERE id = ANY(%s::uuid[])
              AND state IN ('pending', 'enqueued')
            """,
            (job_ids,),
        )
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO job_events (job_id, event_type, actor, data)
            VALUES %s
            """,
            [(jid, "submitted", actor, json.dumps({"request_id": request_id})) for jid in job_ids],
            template="(%s::uuid, %s, %s, %s::jsonb)",
            page_size=1000,
        )
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO job_events (job_id, event_type, actor, data)
            VALUES %s
            """,
            [
                (
                    jid,
                    "enqueued",
                    actor,
                    json.dumps({"stream_name": stream_name, "task_id": tid}),
                )
                for tid, _, _, _, _, _, jid in task_rows
            ],
            template="(%s::uuid, %s, %s, %s::jsonb)",
            page_size=1000,
        )

    submitted = len(task_rows)
    skipped = len(records) - submitted
    return {"submitted": submitted, "skipped": skipped}
