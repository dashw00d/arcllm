"""
SQL queries for Grabber.

All state transitions, work item claims, entity inserts, and
maintenance operations. Every transition enforces valid source state
via WHERE clause -- 0 rows returned means invalid transition.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any, Optional
from uuid import UUID

import psycopg2.extras

from app.db.client import get_cursor

logger = logging.getLogger(__name__)


# ============================================================
# Task creation
# ============================================================

def create_task(
    url: str,
    entity_type: str,
    deadline_minutes: int = 30,
) -> dict[str, Any]:
    """Create a new task in PENDING state."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO tasks (url, entity_type, deadline)
            VALUES (%s, %s, NOW() + %s::interval)
            RETURNING *
            """,
            (url, entity_type, f"{deadline_minutes} minutes"),
        )
        return dict(cur.fetchone())


# ============================================================
# State transitions
# Each enforces valid source state. Returns None if transition invalid.
# ============================================================

def transition_pending_to_exploring(task_id: UUID) -> Optional[dict]:
    """Worker claims a pending task to begin exploration."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE tasks
            SET state = 'exploring',
                updated_at = NOW()
            WHERE id = %s
              AND state = 'pending'
            RETURNING id
            """,
            (str(task_id),),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def transition_exploring_to_scheming(
    task_id: UUID,
    urls_discovered: int,
    detail_urls: list[str],
    schema_sample_urls: list[str],
) -> Optional[dict]:
    """Exploration complete, move to schema generation."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE tasks
            SET state = 'scheming',
                explore_completed_at = NOW(),
                urls_discovered = %s,
                detail_urls = %s,
                schema_sample_urls = %s,
                updated_at = NOW()
            WHERE id = %s
              AND state = 'exploring'
            RETURNING id
            """,
            (
                urls_discovered,
                json.dumps(detail_urls),
                json.dumps(schema_sample_urls),
                str(task_id),
            ),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def transition_scheming_to_extracting(
    task_id: UUID,
    schema: dict,
    schema_fields: int,
    work_item_urls: list[str],
) -> Optional[dict]:
    """
    Schema validated. Store schema AND create all work items in a single
    transaction, then transition to EXTRACTING.

    Work items only exist after schema succeeds -- no orphaned extract jobs.
    """
    with get_cursor(commit=False) as cur:
        conn = cur.connection
        try:
            # Update task
            cur.execute(
                """
                UPDATE tasks
                SET state = 'extracting',
                    schema_completed_at = NOW(),
                    schema = %s,
                    schema_fields = %s,
                    items_total = %s,
                    updated_at = NOW()
                WHERE id = %s
                  AND state = 'scheming'
                RETURNING id
                """,
                (
                    json.dumps(schema),
                    schema_fields,
                    len(work_item_urls),
                    str(task_id),
                ),
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return None

            # Bulk insert work items
            if work_item_urls:
                values = [(str(task_id), url) for url in work_item_urls]
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO work_items (task_id, url)
                    VALUES %s
                    """,
                    values,
                    template="(%s, %s)",
                )

            conn.commit()
            return dict(row)
        except Exception:
            conn.rollback()
            raise


def transition_extracting_to_completed(task_id: UUID) -> Optional[dict]:
    """All work items resolved, task is done."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE tasks
            SET state = 'completed',
                completed_at = NOW(),
                updated_at = NOW()
            WHERE id = %s
              AND state = 'extracting'
            RETURNING id
            """,
            (str(task_id),),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def transition_to_failed(
    task_id: UUID,
    error_message: str,
    failed_stage: Optional[str] = None,
) -> Optional[dict]:
    """Fail a task from any non-terminal state."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE tasks
            SET state = 'failed',
                error_message = %s,
                failed_stage = COALESCE(%s, state),
                updated_at = NOW()
            WHERE id = %s
              AND state NOT IN ('completed', 'failed', 'cancelled')
            RETURNING id
            """,
            (error_message, failed_stage, str(task_id)),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def transition_to_cancelled(task_id: UUID) -> Optional[dict]:
    """Cancel a task from any non-terminal state."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE tasks
            SET state = 'cancelled',
                updated_at = NOW()
            WHERE id = %s
              AND state NOT IN ('completed', 'failed', 'cancelled')
            RETURNING id
            """,
            (str(task_id),),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_task(task_id: UUID) -> Optional[dict]:
    """Fetch a task by ID."""
    with get_cursor(commit=False) as cur:
        cur.execute("SELECT * FROM tasks WHERE id = %s", (str(task_id),))
        row = cur.fetchone()
        return dict(row) if row else None


# ============================================================
# Work item operations
# ============================================================

def claim_work_items(
    task_id: UUID,
    worker_id: str,
    batch_size: int = 10,
) -> list[dict]:
    """
    Claim a batch of pending work items via FOR UPDATE SKIP LOCKED.
    Multiple workers can run this concurrently -- each gets different items.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE work_items
            SET state = 'processing',
                claimed_by = %s,
                claimed_at = NOW(),
                attempts = attempts + 1,
                updated_at = NOW()
            WHERE id IN (
                SELECT id FROM work_items
                WHERE task_id = %s AND state = 'pending'
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
            """,
            (worker_id, str(task_id), batch_size),
        )
        return [dict(row) for row in cur.fetchall()]


def complete_work_item(
    item_id: UUID,
    entity_id: Optional[UUID] = None,
) -> Optional[dict]:
    """Mark a work item as completed, optionally linking the entity."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE work_items
            SET state = 'completed',
                entity_id = %s,
                updated_at = NOW()
            WHERE id = %s
              AND state = 'processing'
            RETURNING *
            """,
            (str(entity_id) if entity_id else None, str(item_id)),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def fail_work_item(item_id: UUID, error_message: str) -> Optional[dict]:
    """Mark a work item as failed."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE work_items
            SET state = 'failed',
                error_message = %s,
                updated_at = NOW()
            WHERE id = %s
              AND state = 'processing'
            RETURNING *
            """,
            (error_message, str(item_id)),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_task_progress(task_id: UUID) -> dict[str, int]:
    """Get completion counts for a task's work items."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE state = 'completed') AS completed,
                COUNT(*) FILTER (WHERE state = 'failed') AS failed,
                COUNT(*) FILTER (WHERE state = 'pending') AS pending,
                COUNT(*) FILTER (WHERE state = 'processing') AS processing,
                COUNT(*) AS total
            FROM work_items
            WHERE task_id = %s
            """,
            (str(task_id),),
        )
        return dict(cur.fetchone())


def update_task_counters(task_id: UUID) -> None:
    """Sync task item counters from actual work_items state."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE tasks
            SET items_completed = sub.completed,
                items_failed = sub.failed,
                updated_at = NOW()
            FROM (
                SELECT
                    COUNT(*) FILTER (WHERE state = 'completed') AS completed,
                    COUNT(*) FILTER (WHERE state = 'failed') AS failed
                FROM work_items
                WHERE task_id = %s
            ) sub
            WHERE id = %s
            """,
            (str(task_id), str(task_id)),
        )


# ============================================================
# Entity operations
# ============================================================

def insert_entity(
    entity_type: str,
    data: dict,
    content_hash: str,
    source_type: Optional[str] = None,
    source_ref: Optional[str] = None,
    source_domain: Optional[str] = None,
    project_id: Optional[UUID] = None,
    meta: Optional[dict] = None,
) -> Optional[dict]:
    """
    Insert an entity with SHA-256 dedup.
    ON CONFLICT DO NOTHING -- returns None if duplicate.
    """
    pid = str(project_id) if project_id else None
    # Two unique indexes exist:
    #   idx_entities_dedup           (entity_type, project_id, content_hash)  -- when project_id IS NOT NULL
    #   idx_entities_dedup_no_project (entity_type, content_hash) WHERE project_id IS NULL
    # ON CONFLICT can only target one, so branch on project_id.
    if pid:
        conflict = "ON CONFLICT (entity_type, project_id, content_hash) DO NOTHING"
    else:
        conflict = "ON CONFLICT (entity_type, content_hash) WHERE project_id IS NULL DO NOTHING"

    with get_cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO entities (
                entity_type, project_id, data, meta,
                source_type, source_ref, source_domain, content_hash
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            {conflict}
            RETURNING *
            """,
            (
                entity_type,
                pid,
                json.dumps(data),
                json.dumps(meta or {}),
                source_type,
                source_ref,
                source_domain,
                content_hash,
            ),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def bulk_insert_entities(entities: list[dict]) -> int:
    """
    Bulk insert entities with dedup. Returns count of newly inserted rows.
    Each dict must have: entity_type, data, content_hash.
    Optional: source_type, source_ref, source_domain, project_id, meta.

    Two separate inserts are used because PostgreSQL ON CONFLICT can only
    target one unique index, and Grabber has two dedup indexes:
      - idx_entities_dedup           (entity_type, project_id, content_hash) — project_id NOT NULL
      - idx_entities_dedup_no_project (entity_type, content_hash) WHERE project_id IS NULL
    """
    if not entities:
        return 0

    def _row(e: dict) -> tuple:
        return (
            e["entity_type"],
            str(e["project_id"]) if e.get("project_id") else None,
            json.dumps(e["data"]),
            json.dumps(e.get("meta", {})),
            e.get("source_type"),
            e.get("source_ref"),
            e.get("source_domain"),
            e["content_hash"],
        )

    no_project = [_row(e) for e in entities if not e.get("project_id")]
    with_project = [_row(e) for e in entities if e.get("project_id")]

    total = 0
    with get_cursor() as cur:
        if no_project:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO entities (
                    entity_type, project_id, data, meta,
                    source_type, source_ref, source_domain, content_hash
                )
                VALUES %s
                ON CONFLICT (entity_type, content_hash) WHERE project_id IS NULL DO NOTHING
                """,
                no_project,
                template="(%s, %s, %s, %s, %s, %s, %s, %s)",
            )
            total += cur.rowcount

        if with_project:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO entities (
                    entity_type, project_id, data, meta,
                    source_type, source_ref, source_domain, content_hash
                )
                VALUES %s
                ON CONFLICT (entity_type, project_id, content_hash) DO NOTHING
                """,
                with_project,
                template="(%s, %s, %s, %s, %s, %s, %s, %s)",
            )
            total += cur.rowcount

    return total


# ============================================================
# Pattern operations
# ============================================================

def upsert_pattern(
    domain: str,
    pattern_type: str,
    config: dict,
    trust_level: str = "provisional",
) -> dict:
    """Insert or update an extraction pattern."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO patterns (domain, pattern_type, config, trust_level)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (domain, pattern_type)
            DO UPDATE SET
                config = EXCLUDED.config,
                trust_level = EXCLUDED.trust_level,
                updated_at = NOW()
            RETURNING *
            """,
            (domain, pattern_type, json.dumps(config), trust_level),
        )
        return dict(cur.fetchone())


def get_pattern(domain: str, pattern_type: str) -> Optional[dict]:
    """Fetch a pattern by domain and type."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT * FROM patterns
            WHERE domain = %s AND pattern_type = %s AND NOT quarantined
            """,
            (domain, pattern_type),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def update_pattern_stats(
    pattern_id: UUID,
    success: bool,
) -> None:
    """Update pattern usage_count, success_rate, and confidence after extraction."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE patterns
            SET usage_count = usage_count + 1,
                success_rate = (success_rate * usage_count + %s) / (usage_count + 1),
                confidence = CASE
                    WHEN (success_rate * usage_count + %s) / (usage_count + 1) > 0.85
                        AND usage_count + 1 >= 5
                    THEN GREATEST(confidence, 0.85)
                    ELSE (success_rate * usage_count + %s) / (usage_count + 1)
                END,
                updated_at = NOW()
            WHERE id = %s
            """,
            (
                1.0 if success else 0.0,
                1.0 if success else 0.0,
                1.0 if success else 0.0,
                str(pattern_id),
            ),
        )


def quarantine_pattern(pattern_id: UUID) -> None:
    """Quarantine a failing pattern."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE patterns
            SET quarantined = TRUE, updated_at = NOW()
            WHERE id = %s
            """,
            (str(pattern_id),),
        )


# ============================================================
# Worker operations
# ============================================================

def upsert_worker(
    worker_id: UUID,
    worker_type: str = "unified",
    instance_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    ipv6_prefix: Optional[str] = None,
    region: Optional[str] = None,
) -> dict:
    """Register or update a worker."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO workers (id, worker_type, instance_id, ip_address, ipv6_prefix, region, last_heartbeat)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (id)
            DO UPDATE SET
                status = 'active',
                instance_id = COALESCE(EXCLUDED.instance_id, workers.instance_id),
                ip_address = COALESCE(EXCLUDED.ip_address, workers.ip_address),
                ipv6_prefix = COALESCE(EXCLUDED.ipv6_prefix, workers.ipv6_prefix),
                region = COALESCE(EXCLUDED.region, workers.region),
                last_heartbeat = NOW()
            RETURNING *
            """,
            (str(worker_id), worker_type, instance_id, ip_address, ipv6_prefix, region),
        )
        return dict(cur.fetchone())


def worker_heartbeat(
    worker_id: UUID,
    current_job_id: Optional[UUID] = None,
    items_processed: Optional[int] = None,
    errors_count: Optional[int] = None,
) -> None:
    """Update worker heartbeat timestamp and optional stats."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE workers
            SET last_heartbeat = NOW(),
                current_job_id = %s,
                items_processed = COALESCE(%s, items_processed),
                errors_count = COALESCE(%s, errors_count)
            WHERE id = %s
            """,
            (
                str(current_job_id) if current_job_id else None,
                items_processed,
                errors_count,
                str(worker_id),
            ),
        )


# ============================================================
# Settings operations
# ============================================================

def get_setting(key: str) -> Optional[dict]:
    """Get the active setting for a key."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT * FROM settings WHERE key = %s AND is_active = TRUE",
            (key,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def set_setting(key: str, value: Any) -> dict:
    """
    Set a setting value, incrementing the version.
    Deactivates previous version, inserts new one.
    """
    with get_cursor() as cur:
        # Get current version
        cur.execute(
            "SELECT version FROM settings WHERE key = %s AND is_active = TRUE",
            (key,),
        )
        row = cur.fetchone()
        new_version = (row["version"] + 1) if row else 1

        # Deactivate old
        cur.execute(
            "UPDATE settings SET is_active = FALSE WHERE key = %s AND is_active = TRUE",
            (key,),
        )

        # Insert new
        cur.execute(
            """
            INSERT INTO settings (key, value, version, is_active)
            VALUES (%s, %s, %s, TRUE)
            RETURNING *
            """,
            (key, json.dumps(value), new_version),
        )
        return dict(cur.fetchone())


# ============================================================
# Maintenance / Reaper queries
# ============================================================

def reap_expired_tasks() -> list[dict]:
    """Fail all tasks past their deadline. The single query that replaces
    the watchdog, deferred completion checker, orphan recovery, and
    Postgres-Redis reconciliation loop."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE tasks
            SET state = 'failed',
                error_message = 'deadline exceeded in state: ' || state,
                failed_stage = state,
                updated_at = NOW()
            WHERE state NOT IN ('completed', 'failed', 'cancelled')
              AND deadline < NOW()
            RETURNING id, state
            """
        )
        return [dict(row) for row in cur.fetchall()]


def reclaim_stuck_work_items(stale_minutes: int = 5) -> list[dict]:
    """Reclaim work items stuck in 'processing' past the threshold."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE work_items
            SET state = 'pending',
                claimed_by = NULL,
                claimed_at = NULL,
                updated_at = NOW()
            WHERE state = 'processing'
              AND claimed_at < NOW() - %s::interval
            RETURNING id
            """,
            (f"{stale_minutes} minutes",),
        )
        return [dict(row) for row in cur.fetchall()]


def check_circuit_breaker(task_id: UUID, window: int = 20, threshold: float = 0.6) -> bool:
    """
    Check if extraction failure rate exceeds threshold.
    Returns True if circuit should trip (fail the task).
    """
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT state FROM work_items
            WHERE task_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (str(task_id), window),
        )
        rows = cur.fetchall()
        if len(rows) < window:
            return False
        failure_count = sum(1 for r in rows if r["state"] == "failed")
        return (failure_count / len(rows)) >= threshold
