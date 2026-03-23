"""
Grabber State Machine

Every task transitions through: PENDING -> EXPLORING -> SCHEMING -> EXTRACTING -> COMPLETED
Any non-terminal state can transition to FAILED or CANCELLED.

Every transition is a single UPDATE with WHERE state = 'expected' RETURNING id.
Zero rows returned = invalid transition (log warning, return False).
No backward edges. No deferred completion. No distributed locks.
"""

import json
import logging
import uuid
from datetime import timedelta
from typing import Optional

import psycopg2
import psycopg2.extras

from app.core.config import get_settings
from app.db.client import get_pool

logger = logging.getLogger(__name__)

# Valid states
PENDING = "pending"
EXPLORING = "exploring"
SCHEMING = "scheming"
EXTRACTING = "extracting"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL_STATES = frozenset({COMPLETED, FAILED, CANCELLED})

# Default deadline: 30 minutes from creation
DEFAULT_DEADLINE_MINUTES = 30


class StateMachine:
    """
    Postgres-backed state machine for Grabber tasks.

    All transitions are atomic single-UPDATE operations. The WHERE clause
    enforces the expected source state, and RETURNING id confirms the
    transition succeeded. If zero rows are returned, the transition was
    invalid (task already moved, cancelled, or failed by reaper).
    """

    def __init__(self):
        self._pool = None

    def _get_pool(self):
        if self._pool is None:
            self._pool = get_pool()
        return self._pool

    def _get_conn(self):
        return self._get_pool().getconn()

    def _put_conn(self, conn):
        self._get_pool().putconn(conn)

    def _execute_transition(self, query: str, params: tuple, transition_name: str) -> bool:
        """
        Execute a state transition query. Returns True if the transition
        succeeded (1+ rows returned), False if invalid (0 rows).
        """
        conn = self._get_conn()
        try:
            conn.autocommit = False
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                result = cur.fetchone()
                conn.commit()
                if result:
                    logger.info("Transition %s succeeded for task %s", transition_name, params[0])
                    return True
                else:
                    logger.warning(
                        "Transition %s failed for task %s: zero rows returned (invalid source state)",
                        transition_name,
                        params[0],
                    )
                    return False
        except Exception:
            conn.rollback()
            logger.exception("Transition %s failed for task %s with exception", transition_name, params[0])
            raise
        finally:
            conn.autocommit = True
            self._put_conn(conn)

    def _sync_related_job_state(
        self,
        cur: psycopg2.extras.RealDictCursor,
        task_id: str,
        target_state: str,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Keep canonical jobs.state aligned with task lifecycle transitions.

        This only updates the job linked via related_task_id and never regresses
        terminal job states to non-terminal states.
        """
        if target_state == "running":
            cur.execute(
                """
                UPDATE jobs
                SET state = 'running',
                    error_message = NULL,
                    updated_at = NOW()
                WHERE related_task_id = %s::uuid
                  AND state IN ('pending', 'enqueued', 'running')
                """,
                (task_id,),
            )
            return

        if target_state == "completed":
            cur.execute(
                """
                UPDATE jobs
                SET state = 'completed',
                    completed_at = COALESCE(completed_at, NOW()),
                    error_message = NULL,
                    updated_at = NOW()
                WHERE related_task_id = %s::uuid
                  AND state IN ('pending', 'enqueued', 'running', 'completed')
                """,
                (task_id,),
            )
            return

        if target_state == "failed":
            cur.execute(
                """
                UPDATE jobs
                SET state = 'failed',
                    error_message = %s,
                    completed_at = COALESCE(completed_at, NOW()),
                    updated_at = NOW()
                WHERE related_task_id = %s::uuid
                  AND state IN ('pending', 'enqueued', 'running', 'failed')
                """,
                (error_message, task_id),
            )
            return

        if target_state == "cancelled":
            cur.execute(
                """
                UPDATE jobs
                SET state = 'cancelled',
                    cancelled_at = COALESCE(cancelled_at, NOW()),
                    updated_at = NOW()
                WHERE related_task_id = %s::uuid
                  AND state IN ('pending', 'enqueued', 'running', 'cancelled')
                """,
                (task_id,),
            )
            return

    # ─── Task Creation ───────────────────────────────────────────────

    def create_task(
        self,
        url: str,
        entity_type: str,
        deadline_minutes: Optional[int] = None,
    ) -> str:
        """
        Create a new task in PENDING state.

        Returns the task UUID as a string.
        """
        if deadline_minutes is None:
            deadline_minutes = DEFAULT_DEADLINE_MINUTES

        task_id = str(uuid.uuid4())
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO tasks (id, url, entity_type, state, deadline, created_at, updated_at)
                    VALUES (%s, %s, %s, 'pending', NOW() + %s * INTERVAL '1 minute', NOW(), NOW())
                    RETURNING id
                    """,
                    (task_id, url, entity_type, deadline_minutes),
                )
                row = cur.fetchone()
                conn.commit()
                logger.info("Created task %s for %s (entity_type=%s, deadline=%dm)", task_id, url, entity_type, deadline_minutes)
                return str(row["id"])
        except Exception:
            conn.rollback()
            logger.exception("Failed to create task for %s", url)
            raise
        finally:
            self._put_conn(conn)

    # ─── PENDING -> EXPLORING ────────────────────────────────────────

    def claim_task(self, task_id: str, worker_id: str) -> bool:
        """
        Worker claims a PENDING task, transitioning it to EXPLORING.
        Returns True if claim succeeded, False if task was already claimed/moved.
        """
        conn = self._get_conn()
        try:
            conn.autocommit = False
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE tasks
                    SET state = 'exploring',
                        updated_at = NOW()
                    WHERE id = %s
                      AND state = 'pending'
                    RETURNING id
                    """,
                    (task_id,),
                )
                result = cur.fetchone()
                if not result:
                    conn.commit()
                    logger.warning(
                        "Transition %s failed for task %s: zero rows returned (invalid source state)",
                        f"PENDING->EXPLORING (worker={worker_id})",
                        task_id,
                    )
                    return False

                self._sync_related_job_state(cur, task_id, "running")
                conn.commit()
                logger.info("Transition %s succeeded for task %s", f"PENDING->EXPLORING (worker={worker_id})", task_id)
                return True
        except Exception:
            conn.rollback()
            logger.exception("Transition %s failed for task %s with exception", f"PENDING->EXPLORING (worker={worker_id})", task_id)
            raise
        finally:
            conn.autocommit = True
            self._put_conn(conn)

    # ─── EXPLORING -> SCHEMING ───────────────────────────────────────

    def complete_exploration(
        self,
        task_id: str,
        detail_urls: list[str],
        sample_urls: list[str],
    ) -> bool:
        """
        Mark exploration complete. Stores discovered URLs and sample URLs,
        transitions task to SCHEMING.

        detail_urls: all discovered detail page URLs
        sample_urls: 3 URLs selected for schema generation
        """
        urls_discovered = len(detail_urls)
        if urls_discovered == 0:
            logger.warning("Task %s: exploration found 0 URLs, failing task", task_id)
            return self.fail_task(task_id, "Exploration found 0 detail URLs", EXPLORING)

        conn = self._get_conn()
        try:
            conn.autocommit = False
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE tasks
                    SET state = 'scheming',
                        explore_completed_at = NOW(),
                        urls_discovered = %s,
                        detail_urls = %s::jsonb,
                        schema_sample_urls = %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                      AND state = 'exploring'
                    RETURNING id
                    """,
                    (urls_discovered, json.dumps(detail_urls), json.dumps(sample_urls), task_id),
                )
                result = cur.fetchone()
                if not result:
                    conn.commit()
                    logger.warning(
                        "Transition EXPLORING->SCHEMING failed for task %s: zero rows returned",
                        task_id,
                    )
                    return False

                self._sync_related_job_state(cur, task_id, "running")
                conn.commit()
                logger.info("Transition EXPLORING->SCHEMING succeeded for task %s", task_id)
                return True
        except Exception:
            conn.rollback()
            logger.exception("Transition EXPLORING->SCHEMING failed for task %s with exception", task_id)
            raise
        finally:
            conn.autocommit = True
            self._put_conn(conn)

    # ─── SCHEMING -> EXTRACTING (with work item creation) ────────────

    def complete_schema(
        self,
        task_id: str,
        schema: dict,
        work_item_urls: list[str],
    ) -> bool:
        """
        Store the validated schema and create work items in a SINGLE transaction.
        Transitions task from SCHEMING to EXTRACTING.

        This is the critical transition: work items are ONLY created here,
        inside the same transaction. If schema fails, no orphaned work items.
        """
        schema_fields = len(schema.get("fields", []))
        if schema_fields == 0:
            logger.warning("Task %s: schema has 0 fields, failing task", task_id)
            return self.fail_task(task_id, "Schema generation produced 0 fields", SCHEMING)

        items_total = len(work_item_urls)
        if items_total == 0:
            logger.warning("Task %s: 0 work item URLs, failing task", task_id)
            return self.fail_task(task_id, "No URLs to extract", SCHEMING)

        conn = self._get_conn()
        try:
            conn.autocommit = False
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Transition task state
                cur.execute(
                    """
                    UPDATE tasks
                    SET state = 'extracting',
                        schema = %s::jsonb,
                        schema_fields = %s,
                        schema_completed_at = NOW(),
                        items_total = %s,
                        items_completed = 0,
                        items_failed = 0,
                        updated_at = NOW()
                    WHERE id = %s
                      AND state = 'scheming'
                    RETURNING id
                    """,
                    (json.dumps(schema), schema_fields, items_total, task_id),
                )
                result = cur.fetchone()
                if not result:
                    conn.rollback()
                    logger.warning(
                        "Transition SCHEMING->EXTRACTING failed for task %s: zero rows returned",
                        task_id,
                    )
                    return False

                # Create work items in the SAME transaction
                # Use unnest for efficient bulk insert
                work_item_ids = [str(uuid.uuid4()) for _ in work_item_urls]
                cur.execute(
                    """
                    INSERT INTO work_items (id, task_id, url, state, created_at, updated_at)
                    SELECT unnest(%s::uuid[]), %s, unnest(%s::text[]), 'pending', NOW(), NOW()
                    """,
                    (work_item_ids, task_id, work_item_urls),
                )

                self._sync_related_job_state(cur, task_id, "running")
                conn.commit()
                logger.info(
                    "Task %s: SCHEMING->EXTRACTING, created %d work items (%d schema fields)",
                    task_id,
                    items_total,
                    schema_fields,
                )
                return True
        except Exception:
            conn.rollback()
            logger.exception("Failed SCHEMING->EXTRACTING transition for task %s", task_id)
            raise
        finally:
            conn.autocommit = True
            self._put_conn(conn)

    # ─── EXTRACTING -> COMPLETED ─────────────────────────────────────

    def complete_extraction(self, task_id: str) -> bool:
        """
        Attempt to transition task from EXTRACTING to COMPLETED.
        Only succeeds if there are NO pending or processing work items.

        Multiple workers may call this concurrently. The WHERE clause with
        NOT EXISTS ensures exactly one succeeds.
        """
        conn = self._get_conn()
        try:
            conn.autocommit = False
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE tasks
                    SET state = 'completed',
                        completed_at = NOW(),
                        items_completed = (
                            SELECT count(*) FROM work_items
                            WHERE task_id = %s AND state = 'completed'
                        ),
                        items_failed = (
                            SELECT count(*) FROM work_items
                            WHERE task_id = %s AND state = 'failed'
                        ),
                        updated_at = NOW()
                    WHERE id = %s
                      AND state = 'extracting'
                      AND NOT EXISTS (
                          SELECT 1 FROM work_items
                          WHERE task_id = %s AND state IN ('pending', 'processing')
                      )
                    RETURNING id
                    """,
                    (task_id, task_id, task_id, task_id),
                )
                result = cur.fetchone()
                if result:
                    self._sync_related_job_state(cur, task_id, "completed")
                    conn.commit()
                    logger.info("Task %s: EXTRACTING->COMPLETED", task_id)
                    return True
                else:
                    conn.commit()
                    logger.debug(
                        "Task %s: completion check returned 0 rows (items still in flight or already completed)",
                        task_id,
                    )
                    return False
        except Exception:
            conn.rollback()
            logger.exception("Failed completion check for task %s", task_id)
            raise
        finally:
            conn.autocommit = True
            self._put_conn(conn)

    # ─── Any -> FAILED ───────────────────────────────────────────────

    def fail_task(self, task_id: str, error_message: str, failed_stage: str) -> bool:
        """
        Transition any non-terminal task to FAILED.
        Records which stage it failed in and the error message.
        """
        conn = self._get_conn()
        try:
            conn.autocommit = False
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE tasks
                    SET state = 'failed',
                        error_message = %s,
                        failed_stage = %s,
                        updated_at = NOW()
                    WHERE id = %s
                      AND state NOT IN ('completed', 'failed', 'cancelled')
                    RETURNING id
                    """,
                    (error_message, failed_stage, task_id),
                )
                result = cur.fetchone()
                if not result:
                    conn.commit()
                    logger.warning(
                        "Transition *->FAILED (stage=%s) failed for task %s: zero rows returned (invalid source state)",
                        failed_stage,
                        task_id,
                    )
                    return False

                self._sync_related_job_state(cur, task_id, "failed", error_message=error_message)
                conn.commit()
                logger.info("Transition *->FAILED (stage=%s) succeeded for task %s", failed_stage, task_id)
                return True
        except Exception:
            conn.rollback()
            logger.exception("Transition *->FAILED (stage=%s) failed for task %s with exception", failed_stage, task_id)
            raise
        finally:
            conn.autocommit = True
            self._put_conn(conn)

    # ─── Any -> CANCELLED ────────────────────────────────────────────

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task. Only works on non-terminal tasks.
        Also cancels any pending work items for this task.
        """
        conn = self._get_conn()
        try:
            conn.autocommit = False
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Cancel the task
                cur.execute(
                    """
                    UPDATE tasks
                    SET state = 'cancelled',
                        updated_at = NOW()
                    WHERE id = %s
                      AND state NOT IN ('completed', 'failed', 'cancelled')
                    RETURNING id
                    """,
                    (task_id,),
                )
                result = cur.fetchone()
                if not result:
                    conn.commit()
                    logger.warning("Cancel failed for task %s: already in terminal state", task_id)
                    return False

                # Cancel any pending work items
                cur.execute(
                    """
                    UPDATE work_items
                    SET state = 'cancelled', updated_at = NOW()
                    WHERE task_id = %s
                      AND state IN ('pending', 'processing')
                    """,
                    (task_id,),
                )
                cancelled_items = cur.rowcount

                self._sync_related_job_state(cur, task_id, "cancelled")
                conn.commit()
                logger.info("Task %s: CANCELLED (%d work items cancelled)", task_id, cancelled_items)
                return True
        except Exception:
            conn.rollback()
            logger.exception("Failed to cancel task %s", task_id)
            raise
        finally:
            conn.autocommit = True
            self._put_conn(conn)

    # ─── Work Item Operations ────────────────────────────────────────

    def claim_work_items(
        self,
        task_id: str,
        worker_id: str,
        batch_size: int = 10,
    ) -> list[dict]:
        """
        Claim a batch of pending work items using FOR UPDATE SKIP LOCKED.
        Multiple workers call this concurrently; each gets a different batch.
        Returns list of claimed work item dicts, or empty list when no work remains.
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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
                    (worker_id, task_id, batch_size),
                )
                items = cur.fetchall()
                conn.commit()
                if items:
                    logger.debug(
                        "Worker %s claimed %d items for task %s",
                        worker_id,
                        len(items),
                        task_id,
                    )
                return [dict(row) for row in items]
        except Exception:
            conn.rollback()
            logger.exception("Failed to claim work items for task %s", task_id)
            raise
        finally:
            self._put_conn(conn)

    def complete_work_item(self, item_id: str, entity_id: Optional[str] = None) -> bool:
        """Mark a work item as completed, optionally linking it to an entity."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE work_items
                    SET state = 'completed',
                        entity_id = %s,
                        updated_at = NOW()
                    WHERE id = %s
                      AND state = 'processing'
                    RETURNING id
                    """,
                    (entity_id, item_id),
                )
                result = cur.fetchone()
                conn.commit()
                if result:
                    return True
                else:
                    logger.warning("Complete work item %s: zero rows (not in processing state)", item_id)
                    return False
        except Exception:
            conn.rollback()
            logger.exception("Failed to complete work item %s", item_id)
            raise
        finally:
            self._put_conn(conn)

    def fail_work_item(self, item_id: str, error_message: str) -> bool:
        """Mark a work item as failed with an error message."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE work_items
                    SET state = 'failed',
                        error_message = %s,
                        updated_at = NOW()
                    WHERE id = %s
                      AND state = 'processing'
                    RETURNING id
                    """,
                    (error_message, item_id),
                )
                result = cur.fetchone()
                conn.commit()
                if result:
                    return True
                else:
                    logger.warning("Fail work item %s: zero rows (not in processing state)", item_id)
                    return False
        except Exception:
            conn.rollback()
            logger.exception("Failed to fail work item %s", item_id)
            raise
        finally:
            self._put_conn(conn)

    # ─── Circuit Breaker ─────────────────────────────────────────────

    def check_circuit_breaker(self, task_id: str) -> bool:
        """
        Check if the circuit breaker should trip for a task.
        Returns True if 60%+ of the last 20 resolved work items failed.
        """
        settings = get_settings()
        window = getattr(settings, "circuit_breaker_window", 20)
        threshold = getattr(settings, "circuit_breaker_threshold", 0.6)

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT state FROM work_items
                    WHERE task_id = %s
                      AND state IN ('completed', 'failed')
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (task_id, window),
                )
                recent = cur.fetchall()

                if len(recent) < window:
                    return False

                failure_count = sum(1 for r in recent if r["state"] == "failed")
                failure_rate = failure_count / len(recent)

                if failure_rate >= threshold:
                    logger.warning(
                        "Circuit breaker TRIPPED for task %s: %.0f%% failure rate in last %d items",
                        task_id,
                        failure_rate * 100,
                        window,
                    )
                    return True
                return False
        except Exception:
            logger.exception("Circuit breaker check failed for task %s", task_id)
            return False
        finally:
            self._put_conn(conn)

    def trip_circuit_breaker(self, task_id: str, failure_rate: float) -> bool:
        """
        Trip the circuit breaker: cancel remaining pending work items and
        fail the task.
        """
        conn = self._get_conn()
        try:
            conn.autocommit = False
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Cancel all remaining pending work items
                cur.execute(
                    """
                    UPDATE work_items
                    SET state = 'cancelled', updated_at = NOW()
                    WHERE task_id = %s AND state = 'pending'
                    """,
                    (task_id,),
                )
                cancelled = cur.rowcount

                # Fail the task
                cur.execute(
                    """
                    UPDATE tasks
                    SET state = 'failed',
                        error_message = %s,
                        failed_stage = 'extracting',
                        items_completed = (
                            SELECT count(*) FROM work_items
                            WHERE task_id = %s AND state = 'completed'
                        ),
                        items_failed = (
                            SELECT count(*) FROM work_items
                            WHERE task_id = %s AND state = 'failed'
                        ),
                        updated_at = NOW()
                    WHERE id = %s
                      AND state = 'extracting'
                    RETURNING id
                    """,
                    (
                        f"Circuit breaker: {failure_rate:.0%} failure rate in last 20 items",
                        task_id,
                        task_id,
                        task_id,
                    ),
                )
                result = cur.fetchone()
                if result:
                    self._sync_related_job_state(
                        cur,
                        task_id,
                        "failed",
                        error_message=f"Circuit breaker: {failure_rate:.0%} failure rate in last 20 items",
                    )
                    conn.commit()
                    logger.info(
                        "Task %s: circuit breaker tripped (%.0f%% failures, %d items cancelled)",
                        task_id,
                        failure_rate * 100,
                        cancelled,
                    )
                    return True
                else:
                    conn.commit()
                    logger.warning("Task %s: circuit breaker trip failed (task not in extracting)", task_id)
                    return False
        except Exception:
            conn.rollback()
            logger.exception("Failed to trip circuit breaker for task %s", task_id)
            raise
        finally:
            conn.autocommit = True
            self._put_conn(conn)

    # ─── Query Helpers ───────────────────────────────────────────────

    def get_task(self, task_id: str) -> Optional[dict]:
        """Fetch a task by ID. Returns None if not found."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            self._put_conn(conn)

    def get_task_progress(self, task_id: str) -> Optional[dict]:
        """Get task state plus live work item counts."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        t.id,
                        t.url,
                        t.entity_type,
                        t.state,
                        t.error_message,
                        t.failed_stage,
                        t.items_total,
                        t.created_at,
                        t.completed_at,
                        coalesce(wi_counts.pending, 0) as items_pending,
                        coalesce(wi_counts.processing, 0) as items_processing,
                        coalesce(wi_counts.completed, 0) as items_completed,
                        coalesce(wi_counts.failed, 0) as items_failed
                    FROM tasks t
                    LEFT JOIN LATERAL (
                        SELECT
                            count(*) FILTER (WHERE state = 'pending') as pending,
                            count(*) FILTER (WHERE state = 'processing') as processing,
                            count(*) FILTER (WHERE state = 'completed') as completed,
                            count(*) FILTER (WHERE state = 'failed') as failed
                        FROM work_items
                        WHERE task_id = t.id
                    ) wi_counts ON true
                    WHERE t.id = %s
                    """,
                    (task_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            self._put_conn(conn)
