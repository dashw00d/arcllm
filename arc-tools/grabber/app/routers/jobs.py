"""
Canonical dynamic job router.

POST /api/jobs              - submit job envelope
POST /api/jobs/validate     - validate envelope/payload without enqueue
GET  /api/jobs              - list jobs
GET  /api/jobs/{id}         - get one job with latest state
POST /api/jobs/{id}/cancel  - cancel job
POST /api/jobs/{id}/retry   - retry from prior payload
GET  /api/job-types         - list job type registry
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile
from pydantic import BaseModel, Field

from app.services import job_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
types_router = APIRouter(prefix="/api/job-types", tags=["jobs"])


def _ts(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _request_id(request: Request, response: Response) -> str:
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    response.headers["X-Request-ID"] = rid
    return rid


def _auth_required(request: Request) -> job_service.ApiAuthContext:
    raw_key = request.headers.get("x-api-key")
    return job_service.authenticate_api_key(raw_key, required=True)


class JobSubmitRequest(BaseModel):
    job_type: str
    version: str = "v1"
    payload: dict[str, Any]
    idempotency_key: Optional[str] = None
    deadline_minutes: int = Field(default=30, ge=1, le=10080)
    project_id: Optional[str] = None
    priority: str = Field(default="normal", pattern="^(normal|high)$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobSubmitResponse(BaseModel):
    job_id: str
    job_type: str
    version: str
    state: str
    idempotency_key: str
    created: bool
    related_task_id: Optional[str] = None
    request_id: str


class JobTypeDetail(BaseModel):
    job_type: str
    version: str
    stream_name: str
    handler_module: str
    handler_class: str
    payload_schema: dict[str, Any]
    requires_task: bool
    enabled: bool
    description: str = ""


class JobTypeListResponse(BaseModel):
    job_types: list[JobTypeDetail]
    count: int


class JobEvent(BaseModel):
    event_id: str
    event_type: str
    actor: str
    data: dict[str, Any]
    created_at: Optional[str] = None


class JobDetail(BaseModel):
    job_id: str
    tenant_id: str
    job_type: str
    version: str
    state: str
    idempotency_key: str
    priority: str
    payload: dict[str, Any]
    metadata: dict[str, Any]
    request_id: Optional[str] = None
    related_task_id: Optional[str] = None
    project_id: Optional[str] = None
    error_message: Optional[str] = None
    enqueued_at: Optional[str] = None
    completed_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    events: list[JobEvent] = []


class JobListResponse(BaseModel):
    jobs: list[JobDetail]
    count: int


class JobUploadDetail(BaseModel):
    upload_id: str
    tenant_id: str
    job_type: Optional[str] = None
    filename: str
    content_type: Optional[str] = None
    size_bytes: int
    sha256: str
    created_at: Optional[str] = None


class RecipeUploadSubmitRequest(BaseModel):
    upload_id: str
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
    dry_run: bool = False


def _row_to_job_detail(row: dict[str, Any], events: list[dict[str, Any]] | None = None) -> JobDetail:
    return JobDetail(
        job_id=str(row["id"]),
        tenant_id=row["tenant_id"],
        job_type=row["job_type"],
        version=row["version"],
        state=row["state"],
        idempotency_key=row["idempotency_key"],
        priority=row.get("priority", "normal"),
        payload=row.get("payload") or {},
        metadata=row.get("metadata") or {},
        request_id=row.get("request_id"),
        related_task_id=str(row["related_task_id"]) if row.get("related_task_id") else None,
        project_id=str(row["project_id"]) if row.get("project_id") else None,
        error_message=row.get("error_message"),
        enqueued_at=_ts(row.get("enqueued_at")),
        completed_at=_ts(row.get("completed_at")),
        cancelled_at=_ts(row.get("cancelled_at")),
        created_at=_ts(row.get("created_at")),
        updated_at=_ts(row.get("updated_at")),
        events=[
            JobEvent(
                event_id=str(e["id"]),
                event_type=e["event_type"],
                actor=e["actor"],
                data=e.get("data") or {},
                created_at=_ts(e.get("created_at")),
            )
            for e in (events or [])
        ],
    )


def _row_to_upload_detail(row: dict[str, Any]) -> JobUploadDetail:
    return JobUploadDetail(
        upload_id=str(row["id"]),
        tenant_id=row["tenant_id"],
        job_type=row.get("job_type"),
        filename=row["filename"],
        content_type=row.get("content_type"),
        size_bytes=int(row.get("size_bytes", 0)),
        sha256=row["sha256"],
        created_at=_ts(row.get("created_at")),
    )


@types_router.get("", response_model=JobTypeListResponse)
async def list_registered_job_types(request: Request) -> JobTypeListResponse:
    _auth_required(request)
    rows = job_service.list_job_types()
    return JobTypeListResponse(
        job_types=[JobTypeDetail(**row) for row in rows],
        count=len(rows),
    )


@router.post("/uploads", response_model=JobUploadDetail, status_code=201)
async def upload_job_file(
    request: Request,
    file: UploadFile = File(...),
    job_type: Optional[str] = Form(None),
) -> JobUploadDetail:
    auth = _auth_required(request)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    row = job_service.create_job_upload(
        auth=auth,
        filename=file.filename or "upload.bin",
        content_type=file.content_type or "application/octet-stream",
        data=raw,
        job_type=job_type,
    )
    return _row_to_upload_detail(row)


@router.get("/uploads/{upload_id}", response_model=JobUploadDetail)
async def get_upload(upload_id: str, request: Request) -> JobUploadDetail:
    auth = _auth_required(request)
    try:
        uuid.UUID(upload_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid upload ID format")
    row = job_service.get_job_upload(upload_id, tenant_id=auth.tenant_id, include_data=False)
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")
    return _row_to_upload_detail(row)


@router.post("/recipe-pipeline/from-upload")
async def submit_recipe_from_upload(
    body: RecipeUploadSubmitRequest,
    request: Request,
    response: Response,
) -> dict[str, Any]:
    auth = _auth_required(request)
    req_id = _request_id(request, response)
    try:
        uuid.UUID(body.upload_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid upload ID format")

    upload = job_service.get_job_upload(body.upload_id, tenant_id=auth.tenant_id, include_data=True)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    zip_codes = job_service.parse_zip_codes_from_upload(upload["data"], filename=upload["filename"])
    zip_rows = job_service.resolve_zip_rows(zip_codes)
    project_id = job_service.get_or_create_project(
        body.project_name,
        {"domain": body.domain, "page_type": body.page_type, "query": body.query},
    )

    if body.dry_run:
        preview = [
            {
                "zip": z["zip"],
                "city": z.get("city", ""),
                "state": z.get("state", ""),
                "population": z.get("population", ""),
                "lat": z.get("lat", ""),
                "lng": z.get("lng", ""),
            }
            for z in zip_rows[:20]
        ]
        return {
            "project_id": project_id,
            "project_name": body.project_name,
            "submitted": 0,
            "skipped": 0,
            "zips_matched": len(zip_rows),
            "dry_run": True,
            "tasks": preview,
            "request_id": req_id,
        }

    result = await job_service.submit_recipe_pipeline_jobs(
        redis_client=request.app.state.redis,
        tenant_id=auth.tenant_id,
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
        zip_rows=zip_rows,
        request_id=req_id,
        actor=f"api_key:{auth.api_key_id}",
        metadata_source="api:/api/jobs/recipe-pipeline/from-upload",
    )
    return {
        "project_id": project_id,
        "project_name": body.project_name,
        "submitted": result["submitted"],
        "skipped": result["skipped"],
        "zips_matched": len(zip_rows),
        "dry_run": False,
        "request_id": req_id,
    }


@router.post("", response_model=JobSubmitResponse, status_code=201)
async def submit_job(body: JobSubmitRequest, request: Request, response: Response) -> JobSubmitResponse:
    auth = _auth_required(request)
    req_id = _request_id(request, response)

    payload_bytes = len(json.dumps(body.payload, separators=(",", ":")).encode("utf-8"))
    job_service.enforce_quota_limits(auth, payload_bytes)

    row = await job_service.submit_job(
        redis_client=request.app.state.redis,
        tenant_id=auth.tenant_id,
        job_type=body.job_type,
        version=body.version,
        payload=body.payload,
        idempotency_key=body.idempotency_key,
        request_id=req_id,
        project_id=body.project_id,
        priority=body.priority,
        metadata=body.metadata,
        deadline_minutes=body.deadline_minutes,
        actor=f"api_key:{auth.api_key_id}",
    )

    created = bool(row.get("inserted", False))
    return JobSubmitResponse(
        job_id=str(row["id"]),
        job_type=row["job_type"],
        version=row["version"],
        state=row["state"],
        idempotency_key=row["idempotency_key"],
        created=created,
        related_task_id=str(row["related_task_id"]) if row.get("related_task_id") else None,
        request_id=req_id,
    )


@router.post("/validate")
async def validate_job(body: JobSubmitRequest, request: Request, response: Response) -> dict[str, Any]:
    _auth_required(request)
    req_id = _request_id(request, response)
    definition = job_service.resolve_job_type(body.job_type, body.version)
    if not definition or not definition.get("enabled"):
        raise HTTPException(status_code=404, detail=f"Unsupported job_type/version: {body.job_type}/{body.version}")
    job_service.validate_payload_schema(body.payload, definition.get("payload_schema") or {})
    return {
        "valid": True,
        "job_type": body.job_type,
        "version": body.version,
        "request_id": req_id,
    }


@router.get("", response_model=JobListResponse)
async def list_jobs(
    request: Request,
    state: Optional[str] = Query(None),
    job_type: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    created_after: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> JobListResponse:
    auth = _auth_required(request)
    rows = job_service.list_jobs(
        tenant_id=auth.tenant_id,
        state=state,
        job_type=job_type,
        project_id=project_id,
        created_after=created_after,
        limit=limit,
        offset=offset,
    )
    jobs = [_row_to_job_detail(row) for row in rows]
    return JobListResponse(jobs=jobs, count=len(jobs))


@router.get("/{job_id}", response_model=JobDetail)
async def get_job(job_id: str, request: Request, include_events: bool = Query(False)) -> JobDetail:
    auth = _auth_required(request)
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    row = job_service.get_job(job_id, tenant_id=auth.tenant_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    events = job_service.list_job_events(job_id, limit=100) if include_events else []
    return _row_to_job_detail(row, events)


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request, response: Response) -> dict[str, Any]:
    auth = _auth_required(request)
    _request_id(request, response)
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    row = job_service.cancel_job(job_id, tenant_id=auth.tenant_id, actor=f"api_key:{auth.api_key_id}")
    if not row:
        raise HTTPException(status_code=404, detail="Job not found or already terminal")
    return {"job_id": str(row["id"]), "state": row["state"], "message": "Job cancelled"}


@router.post("/{job_id}/retry", response_model=JobSubmitResponse, status_code=201)
async def retry_job(job_id: str, request: Request, response: Response) -> JobSubmitResponse:
    auth = _auth_required(request)
    req_id = _request_id(request, response)
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    original = job_service.get_job(job_id, tenant_id=auth.tenant_id)
    if not original:
        raise HTTPException(status_code=404, detail="Job not found")
    if original["state"] not in ("failed", "cancelled", "completed"):
        raise HTTPException(status_code=409, detail=f"Can only retry terminal jobs. Current state: {original['state']}")

    payload = dict(original.get("payload") or {})
    metadata = dict(original.get("metadata") or {})
    metadata["retry_of"] = job_id

    payload_bytes = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    job_service.enforce_quota_limits(auth, payload_bytes)

    row = await job_service.submit_job(
        redis_client=request.app.state.redis,
        tenant_id=auth.tenant_id,
        job_type=original["job_type"],
        version=original["version"],
        payload=payload,
        idempotency_key=f"retry:{uuid.uuid4()}",
        request_id=req_id,
        project_id=str(original["project_id"]) if original.get("project_id") else None,
        priority=original.get("priority", "normal"),
        metadata=metadata,
        deadline_minutes=30,
        actor=f"api_key:{auth.api_key_id}",
    )

    return JobSubmitResponse(
        job_id=str(row["id"]),
        job_type=row["job_type"],
        version=row["version"],
        state=row["state"],
        idempotency_key=row["idempotency_key"],
        created=bool(row.get("inserted", False)),
        related_task_id=str(row["related_task_id"]) if row.get("related_task_id") else None,
        request_id=req_id,
    )
