"""
Dashboard router: single-call system overview for AI orchestrator.

GET /api/dashboard - complete system state snapshot
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.db.client import get_cursor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Stream names to monitor
STREAMS = {
    "pipeline": "stream:jobs:pipeline",
    "exploration": "stream:jobs:exploration",
    "schema_gen": "stream:jobs:schema",
    "extract": "stream:jobs:extract",
    "recipe_pipeline": "stream:jobs:recipe_pipeline",
}
CONSUMER_GROUP = "workers"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class TaskSummary(BaseModel):
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


class WorkerSummary(BaseModel):
    total: int = 0
    active: int = 0
    stale: int = 0
    dead: int = 0


class StreamBacklog(BaseModel):
    pipeline: int = 0
    exploration: int = 0
    schema_gen: int = 0
    extract: int = 0
    recipe_pipeline: int = 0


class Throughput(BaseModel):
    tasks_completed_24h: int = 0
    entities_extracted_24h: int = 0


class DashboardResponse(BaseModel):
    task_summary: TaskSummary
    recent_failures: list[RecentFailure]
    worker_summary: WorkerSummary
    stream_backlog: StreamBacklog
    entity_count: int
    throughput: Throughput


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


def _get_task_summary() -> TaskSummary:
    """Count tasks by state (using GROUP BY for index efficiency)."""
    try:
        with get_cursor(commit=False) as cur:
            # GROUP BY uses idx_tasks_state efficiently
            cur.execute("SELECT state, COUNT(*) as count FROM tasks GROUP BY state")
            counts = {row["state"]: row["count"] for row in cur.fetchall()}
            return TaskSummary(
                pending=counts.get("pending", 0),
                exploring=counts.get("exploring", 0),
                scheming=counts.get("scheming", 0),
                extracting=counts.get("extracting", 0),
                completed=counts.get("completed", 0),
                failed=counts.get("failed", 0),
                cancelled=counts.get("cancelled", 0),
            )
    except Exception:
        logger.exception("Failed to get task summary")
    return TaskSummary()


def _get_recent_failures() -> list[RecentFailure]:
    """Get the last 10 failed tasks."""
    try:
        with get_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT id, url, error_message, failed_stage, updated_at
                FROM tasks
                WHERE state = 'failed'
                ORDER BY updated_at DESC
                LIMIT 10
                """
            )
            rows = cur.fetchall()
            return [
                RecentFailure(
                    task_id=str(row["id"]),
                    url=row["url"],
                    error_message=row.get("error_message"),
                    failed_stage=row.get("failed_stage"),
                    updated_at=_ts(row.get("updated_at")),
                )
                for row in rows
            ]
    except Exception:
        logger.exception("Failed to get recent failures")
    return []


def _get_worker_summary() -> WorkerSummary:
    """Get worker counts by status (uses idx_workers_status and idx_workers_heartbeat)."""
    try:
        with get_cursor(commit=False) as cur:
            # Use approximate count from pg_stat_user_tables for total
            cur.execute(
                """
                SELECT
                    (SELECT n_live_tup FROM pg_stat_user_tables WHERE relname = 'workers') AS total,
                    (SELECT COUNT(*) FROM workers
                     WHERE status = 'active'
                       AND last_heartbeat > NOW() - INTERVAL '120 seconds'
                    ) AS active,
                    (SELECT COUNT(*) FROM workers
                     WHERE status = 'active'
                       AND last_heartbeat <= NOW() - INTERVAL '120 seconds'
                       AND last_heartbeat > NOW() - INTERVAL '600 seconds'
                    ) AS stale,
                    (SELECT COUNT(*) FROM workers
                     WHERE status = 'dead'
                        OR (status = 'active' AND last_heartbeat <= NOW() - INTERVAL '600 seconds')
                    ) AS dead
                """
            )
            row = cur.fetchone()
            if row:
                return WorkerSummary(
                    total=row["total"] or 0,
                    active=row["active"] or 0,
                    stale=row["stale"] or 0,
                    dead=row["dead"] or 0,
                )
    except Exception:
        logger.exception("Failed to get worker summary")
    return WorkerSummary()


def _get_entity_count() -> int:
    """Total entities extracted (approximate from stats for speed)."""
    try:
        with get_cursor(commit=False) as cur:
            # Use pg_stat_user_tables for instant approximate count
            cur.execute(
                "SELECT n_live_tup FROM pg_stat_user_tables WHERE relname = 'entities'"
            )
            row = cur.fetchone()
            if row and row["n_live_tup"]:
                return row["n_live_tup"]
    except Exception:
        logger.exception("Failed to get entity count")
    return 0


def _get_throughput() -> Throughput:
    """Tasks completed and entities extracted in the last 24 hours."""
    try:
        with get_cursor(commit=False) as cur:
            # Now uses idx_tasks_completed_at and idx_entities_created_at
            cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM tasks
                     WHERE state = 'completed'
                       AND completed_at > NOW() - INTERVAL '24 hours'
                    ) AS tasks_completed_24h,
                    (SELECT COUNT(*) FROM entities
                     WHERE created_at > NOW() - INTERVAL '24 hours'
                    ) AS entities_extracted_24h
                """
            )
            row = cur.fetchone()
            if row:
                return Throughput(
                    tasks_completed_24h=row["tasks_completed_24h"] or 0,
                    entities_extracted_24h=row["entities_extracted_24h"] or 0,
                )
    except Exception:
        logger.exception("Failed to get throughput")
    return Throughput()


async def _get_stream_backlog(request: Request) -> StreamBacklog:
    """Pending message counts per stream."""
    redis = request.app.state.redis
    counts = {}
    for name, stream_key in STREAMS.items():
        try:
            counts[name] = await redis.xlen(stream_key)
        except Exception:
            counts[name] = 0
    return StreamBacklog(**counts)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("", response_model=DashboardResponse)
async def get_dashboard(request: Request) -> DashboardResponse:
    """Single-call system overview for AI orchestrator."""
    stream_backlog = await _get_stream_backlog(request)

    return DashboardResponse(
        task_summary=_get_task_summary(),
        recent_failures=_get_recent_failures(),
        worker_summary=_get_worker_summary(),
        stream_backlog=stream_backlog,
        entity_count=_get_entity_count(),
        throughput=_get_throughput(),
    )
