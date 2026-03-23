"""
Projects router: CRUD and summary endpoints for project/campaign management.

POST  /api/projects              - create project
GET   /api/projects              - list projects with status filter
GET   /api/projects/{id}         - project detail with task/entity counts
PATCH /api/projects/{id}         - update status, config, stats
GET   /api/projects/{id}/summary - aggregated view of tasks and entities
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.db.client import get_cursor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    name: str
    job_type: str
    description: Optional[str] = None
    config: Optional[dict[str, Any]] = Field(default_factory=dict)


class ProjectDetail(BaseModel):
    id: str
    name: str
    job_type: str
    description: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)
    status: str
    stats: dict[str, Any] = Field(default_factory=dict)
    task_count: int = 0
    entity_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ProjectListResponse(BaseModel):
    projects: list[ProjectDetail]
    count: int


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    config: Optional[dict[str, Any]] = None
    stats: Optional[dict[str, Any]] = None


class TaskStateSummary(BaseModel):
    pending: int = 0
    exploring: int = 0
    scheming: int = 0
    extracting: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0


class RecentFailure(BaseModel):
    task_id: str
    url: str
    error_message: Optional[str] = None
    failed_stage: Optional[str] = None
    updated_at: Optional[str] = None


class ProjectSummary(BaseModel):
    id: str
    name: str
    status: str
    task_states: TaskStateSummary
    entity_count: int = 0
    recent_failures: list[RecentFailure]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _validate_uuid(project_id: str) -> None:
    try:
        uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID format")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=ProjectDetail, status_code=201)
async def create_project(body: ProjectCreate) -> ProjectDetail:
    """Create a new project."""
    project_id = str(uuid.uuid4())
    config_json = json.dumps(body.config or {})

    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO projects (id, name, job_type, description, config)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
            """,
            (project_id, body.name, body.job_type, body.description, config_json),
        )
        row = cur.fetchone()

    return ProjectDetail(
        id=str(row["id"]),
        name=row["name"],
        job_type=row["job_type"],
        description=row["description"],
        config=row["config"] if isinstance(row["config"], dict) else {},
        status=row["status"],
        stats=row["stats"] if isinstance(row["stats"], dict) else {},
        created_at=_ts(row["created_at"]),
        updated_at=_ts(row["updated_at"]),
    )


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> ProjectListResponse:
    """List projects with optional status filter."""
    conditions: list[str] = []
    params: list[Any] = []

    if status:
        conditions.append("p.status = %s")
        params.append(status)

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT p.*,
               COALESCE(tc.task_count, 0) AS task_count,
               COALESCE(ec.entity_count, 0) AS entity_count
        FROM projects p
        LEFT JOIN (
            SELECT project_id, COUNT(*) AS task_count
            FROM tasks
            WHERE project_id IS NOT NULL
            GROUP BY project_id
        ) tc ON tc.project_id = p.id
        LEFT JOIN (
            SELECT project_id, COUNT(*) AS entity_count
            FROM entities
            WHERE project_id IS NOT NULL
            GROUP BY project_id
        ) ec ON ec.project_id = p.id
        {where}
        ORDER BY p.created_at DESC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])

    with get_cursor(commit=False) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    projects = [
        ProjectDetail(
            id=str(row["id"]),
            name=row["name"],
            job_type=row["job_type"],
            description=row["description"],
            config=row["config"] if isinstance(row["config"], dict) else {},
            status=row["status"],
            stats=row["stats"] if isinstance(row["stats"], dict) else {},
            task_count=row["task_count"],
            entity_count=row["entity_count"],
            created_at=_ts(row["created_at"]),
            updated_at=_ts(row["updated_at"]),
        )
        for row in rows
    ]
    return ProjectListResponse(projects=projects, count=len(projects))


@router.get("/{project_id}", response_model=ProjectDetail)
async def get_project(project_id: str) -> ProjectDetail:
    """Get project detail with task and entity counts."""
    _validate_uuid(project_id)

    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT p.*,
                   COALESCE(tc.task_count, 0) AS task_count,
                   COALESCE(ec.entity_count, 0) AS entity_count
            FROM projects p
            LEFT JOIN (
                SELECT project_id, COUNT(*) AS task_count
                FROM tasks
                WHERE project_id = %s
                GROUP BY project_id
            ) tc ON tc.project_id = p.id
            LEFT JOIN (
                SELECT project_id, COUNT(*) AS entity_count
                FROM entities
                WHERE project_id = %s
                GROUP BY project_id
            ) ec ON ec.project_id = p.id
            WHERE p.id = %s
            """,
            (project_id, project_id, project_id),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectDetail(
        id=str(row["id"]),
        name=row["name"],
        job_type=row["job_type"],
        description=row["description"],
        config=row["config"] if isinstance(row["config"], dict) else {},
        status=row["status"],
        stats=row["stats"] if isinstance(row["stats"], dict) else {},
        task_count=row["task_count"],
        entity_count=row["entity_count"],
        created_at=_ts(row["created_at"]),
        updated_at=_ts(row["updated_at"]),
    )


@router.patch("/{project_id}", response_model=ProjectDetail)
async def update_project(project_id: str, body: ProjectUpdate) -> ProjectDetail:
    """Update project status, config, or stats."""
    _validate_uuid(project_id)

    sets: list[str] = []
    params: list[Any] = []

    if body.name is not None:
        sets.append("name = %s")
        params.append(body.name)
    if body.description is not None:
        sets.append("description = %s")
        params.append(body.description)
    if body.status is not None:
        valid = ("active", "paused", "completed", "failed")
        if body.status not in valid:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid}")
        sets.append("status = %s")
        params.append(body.status)
    if body.config is not None:
        sets.append("config = %s")
        params.append(json.dumps(body.config))
    if body.stats is not None:
        sets.append("stats = %s")
        params.append(json.dumps(body.stats))

    if not sets:
        raise HTTPException(status_code=400, detail="No fields to update")

    sets.append("updated_at = NOW()")
    params.append(project_id)

    with get_cursor(commit=True) as cur:
        cur.execute(
            f"""
            UPDATE projects
            SET {', '.join(sets)}
            WHERE id = %s
            RETURNING *
            """,
            params,
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectDetail(
        id=str(row["id"]),
        name=row["name"],
        job_type=row["job_type"],
        description=row["description"],
        config=row["config"] if isinstance(row["config"], dict) else {},
        status=row["status"],
        stats=row["stats"] if isinstance(row["stats"], dict) else {},
        created_at=_ts(row["created_at"]),
        updated_at=_ts(row["updated_at"]),
    )


@router.get("/{project_id}/summary", response_model=ProjectSummary)
async def get_project_summary(project_id: str) -> ProjectSummary:
    """Aggregated view: task counts by state, entity count, recent failures."""
    _validate_uuid(project_id)

    with get_cursor(commit=False) as cur:
        # Verify project exists
        cur.execute("SELECT id, name, status FROM projects WHERE id = %s", (project_id,))
        project = cur.fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        # Task state counts
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE state = 'pending')    AS pending,
                COUNT(*) FILTER (WHERE state = 'exploring')  AS exploring,
                COUNT(*) FILTER (WHERE state = 'scheming')   AS scheming,
                COUNT(*) FILTER (WHERE state = 'extracting') AS extracting,
                COUNT(*) FILTER (WHERE state = 'completed')  AS completed,
                COUNT(*) FILTER (WHERE state = 'failed')     AS failed,
                COUNT(*) FILTER (WHERE state = 'cancelled')  AS cancelled
            FROM tasks
            WHERE project_id = %s
            """,
            (project_id,),
        )
        state_row = cur.fetchone()

        # Entity count
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM entities WHERE project_id = %s",
            (project_id,),
        )
        entity_row = cur.fetchone()

        # Recent failures
        cur.execute(
            """
            SELECT id, url, error_message, failed_stage, updated_at
            FROM tasks
            WHERE project_id = %s AND state = 'failed'
            ORDER BY updated_at DESC
            LIMIT 10
            """,
            (project_id,),
        )
        failure_rows = cur.fetchall()

    task_states = TaskStateSummary(
        pending=state_row["pending"] or 0,
        exploring=state_row["exploring"] or 0,
        scheming=state_row["scheming"] or 0,
        extracting=state_row["extracting"] or 0,
        completed=state_row["completed"] or 0,
        failed=state_row["failed"] or 0,
        cancelled=state_row["cancelled"] or 0,
    )

    failures = [
        RecentFailure(
            task_id=str(r["id"]),
            url=r["url"],
            error_message=r.get("error_message"),
            failed_stage=r.get("failed_stage"),
            updated_at=_ts(r.get("updated_at")),
        )
        for r in failure_rows
    ]

    return ProjectSummary(
        id=str(project["id"]),
        name=project["name"],
        status=project["status"],
        task_states=task_states,
        entity_count=entity_row["cnt"] or 0,
        recent_failures=failures,
    )
