"""
SyncService: drains Redis buffers into Postgres in the background.

Runs as an asyncio task alongside the FastAPI app. Three loops:
  1. drain_entities()   - RPOPLPUSH reliable queue, batch INSERT with ON CONFLICT DO NOTHING
  2. drain_job_results() - sync job completion summaries from Redis to Postgres
  3. reconcile_counters() - self-heal task counter drift every 60s

All Redis reads use safe patterns: silent fail on errors, bounded batch sizes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import psycopg2.extras
import redis.asyncio as aioredis

from app.core.config import get_settings
from app.db.client import get_cursor

# Stream names for task dispatch
TASK_STREAM = "stream:jobs:pipeline"
RECIPE_PIPELINE_STREAM = "stream:jobs:recipe_pipeline"

logger = logging.getLogger(__name__)

# Redis keys
ENTITIES_PENDING = "entities:pending"
ENTITIES_PROCESSING = "entities:processing"
JOBS_RESULTS = "jobs:results"
JOBS_RESULTS_PROCESSING = "jobs:results:processing"

# Tuning
BATCH_SIZE = 50
DRAIN_INTERVAL = 2.0  # seconds between drain cycles
RECONCILE_INTERVAL = 60.0  # seconds between counter reconciliation
FEED_INTERVAL = 2.0  # seconds between feeder polls
FEED_BATCH_SIZE = 200  # tasks to feed per cycle
FEED_GRACE_SECONDS = 15  # only feed stale pending rows to avoid racing direct enqueues


def _normalize_recipe_config(raw_config: Any, url: str, entity_type: str) -> str:
    """
    Ensure recipe stream payload config is always a JSON object.

    Legacy rows may contain NULL/invalid config values; workers expect a dict.
    """
    config: Any = raw_config
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            logger.warning("Invalid recipe config JSON for %s, defaulting to empty object", url)
            config = {}

    if not isinstance(config, dict):
        config = {}

    if not config.get("url"):
        config["url"] = url
    if entity_type and not config.get("entity_type"):
        config["entity_type"] = entity_type

    return json.dumps(config)


class SyncService:
    """Background service that drains Redis queues into Postgres."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start all drain loops as background tasks."""
        if self._running:
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._entity_loop(), name="sync-entities"),
            asyncio.create_task(self._results_loop(), name="sync-results"),
            asyncio.create_task(self._reconcile_loop(), name="sync-reconcile"),
            asyncio.create_task(self._feed_tasks_loop(), name="sync-feeder"),
        ]
        logger.info("SyncService started (%d background tasks)", len(self._tasks))

    async def stop(self) -> None:
        """Signal all loops to stop and wait for them to finish."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("SyncService stopped")

    # ------------------------------------------------------------------
    # Entity drain: Redis List -> Postgres entities table
    # ------------------------------------------------------------------

    async def _entity_loop(self) -> None:
        """Continuously drain entities from Redis into Postgres."""
        while self._running:
            try:
                drained = await self._drain_entities()
                if drained == 0:
                    await asyncio.sleep(DRAIN_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("SyncService entity drain error")
                await asyncio.sleep(DRAIN_INTERVAL * 2)

    async def _drain_entities(self) -> int:
        """
        Drain up to BATCH_SIZE entities using reliable queue pattern.

        RPOPLPUSH moves items from pending to processing atomically.
        After successful Postgres INSERT, LREM removes from processing.
        On crash, items remain in processing list for recovery.
        """
        batch: list[dict[str, Any]] = []
        raw_items: list[str] = []

        for _ in range(BATCH_SIZE):
            try:
                item = await self._redis.rpoplpush(ENTITIES_PENDING, ENTITIES_PROCESSING)
            except Exception:
                break
            if item is None:
                break
            raw_items.append(item if isinstance(item, str) else item.decode())
            try:
                batch.append(json.loads(raw_items[-1]))
            except (json.JSONDecodeError, TypeError):
                logger.warning("Invalid entity JSON, discarding: %.100s", raw_items[-1])
                await self._safe_lrem(ENTITIES_PROCESSING, raw_items[-1])
                raw_items.pop()

        if not batch:
            return 0

        inserted = self._insert_entities(batch)

        # Remove successfully processed items from the processing list
        for raw in raw_items:
            await self._safe_lrem(ENTITIES_PROCESSING, raw)

        if inserted > 0:
            logger.info("Synced %d entities to Postgres", inserted)
        return len(batch)

    def _insert_entities(self, batch: list[dict[str, Any]]) -> int:
        """Batch INSERT entities with ON CONFLICT DO NOTHING for dedup."""
        inserted = 0
        try:
            with get_cursor(commit=True) as cur:
                for entity in batch:
                    try:
                        cur.execute(
                            """
                            INSERT INTO entities (
                                id, entity_type, project_id, data, meta,
                                source_type, source_ref, source_domain,
                                content_hash, status
                            ) VALUES (
                                COALESCE(%(id)s::uuid, gen_random_uuid()),
                                %(entity_type)s, %(project_id)s::uuid,
                                %(data)s::jsonb, COALESCE(%(meta)s::jsonb, '{}'::jsonb),
                                %(source_type)s, %(source_ref)s, %(source_domain)s,
                                %(content_hash)s, COALESCE(%(status)s, 'active')
                            )
                            ON CONFLICT (entity_type, project_id, content_hash)
                            DO NOTHING
                            """,
                            {
                                "id": entity.get("id"),
                                "entity_type": entity.get("entity_type", "unknown"),
                                "project_id": entity.get("project_id"),
                                "data": json.dumps(entity.get("data", {})),
                                "meta": json.dumps(entity.get("meta", {})),
                                "source_type": entity.get("source_type"),
                                "source_ref": entity.get("source_ref"),
                                "source_domain": entity.get("source_domain"),
                                "content_hash": entity.get("content_hash", ""),
                                "status": entity.get("status"),
                            },
                        )
                        inserted += cur.rowcount
                    except Exception:
                        logger.exception("Failed to insert entity: %.100s", entity.get("id"))
        except Exception:
            logger.exception("Entity batch insert failed")
        return inserted

    # ------------------------------------------------------------------
    # Job results drain: Redis List -> Postgres tasks table
    # ------------------------------------------------------------------

    async def _results_loop(self) -> None:
        """Continuously drain job result summaries."""
        while self._running:
            try:
                drained = await self._drain_job_results()
                if drained == 0:
                    await asyncio.sleep(DRAIN_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("SyncService results drain error")
                await asyncio.sleep(DRAIN_INTERVAL * 2)

    async def _drain_job_results(self) -> int:
        """Drain job completion summaries from Redis to Postgres."""
        batch: list[dict[str, Any]] = []
        raw_items: list[str] = []

        for _ in range(BATCH_SIZE):
            try:
                item = await self._redis.rpoplpush(JOBS_RESULTS, JOBS_RESULTS_PROCESSING)
            except Exception:
                break
            if item is None:
                break
            raw_items.append(item if isinstance(item, str) else item.decode())
            try:
                batch.append(json.loads(raw_items[-1]))
            except (json.JSONDecodeError, TypeError):
                logger.warning("Invalid result JSON, discarding: %.100s", raw_items[-1])
                await self._safe_lrem(JOBS_RESULTS_PROCESSING, raw_items[-1])
                raw_items.pop()

        if not batch:
            return 0

        self._update_job_results(batch)

        for raw in raw_items:
            await self._safe_lrem(JOBS_RESULTS_PROCESSING, raw)

        return len(batch)

    def _update_job_results(self, batch: list[dict[str, Any]]) -> None:
        """Update task rows with completion summaries from workers."""
        try:
            with get_cursor(commit=True) as cur:
                for result in batch:
                    task_id = result.get("task_id")
                    if not task_id:
                        continue
                    try:
                        cur.execute(
                            """
                            UPDATE tasks
                            SET items_completed = COALESCE(%(items_completed)s, items_completed),
                                items_failed = COALESCE(%(items_failed)s, items_failed),
                                updated_at = NOW()
                            WHERE id = %(task_id)s::uuid
                            """,
                            {
                                "task_id": task_id,
                                "items_completed": result.get("items_completed"),
                                "items_failed": result.get("items_failed"),
                            },
                        )
                    except Exception:
                        logger.exception("Failed to update result for task %s", task_id)
        except Exception:
            logger.exception("Job results batch update failed")

    # ------------------------------------------------------------------
    # Counter reconciliation: self-heal drift between work_items and tasks
    # ------------------------------------------------------------------

    async def _reconcile_loop(self) -> None:
        """Periodically reconcile task counters from work_items table."""
        while self._running:
            try:
                await asyncio.sleep(RECONCILE_INTERVAL)
                if not self._running:
                    break
                self._reconcile_counters()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("SyncService reconcile error")

    def _reconcile_counters(self) -> None:
        """
        Recompute items_completed / items_failed / items_total from work_items.

        Only touches tasks in extracting state to avoid interfering with
        state transitions.
        """
        try:
            with get_cursor(commit=True) as cur:
                cur.execute(
                    """
                    UPDATE tasks t
                    SET items_total = sub.total,
                        items_completed = sub.completed,
                        items_failed = sub.failed,
                        updated_at = NOW()
                    FROM (
                        SELECT
                            wi.task_id,
                            COUNT(*) AS total,
                            COUNT(*) FILTER (WHERE wi.state = 'completed') AS completed,
                            COUNT(*) FILTER (WHERE wi.state = 'failed') AS failed
                        FROM work_items wi
                        INNER JOIN tasks t2 ON t2.id = wi.task_id AND t2.state = 'extracting'
                        GROUP BY wi.task_id
                    ) sub
                    WHERE t.id = sub.task_id
                      AND t.state = 'extracting'
                      AND (
                          t.items_total != sub.total
                          OR t.items_completed != sub.completed
                          OR t.items_failed != sub.failed
                      )
                    """
                )
                if cur.rowcount and cur.rowcount > 0:
                    logger.info("Reconciled counters for %d tasks", cur.rowcount)
        except Exception:
            logger.exception("Counter reconciliation failed")

    # ------------------------------------------------------------------
    # Task feeder: Postgres pending tasks -> Redis Streams
    # ------------------------------------------------------------------

    async def _feed_tasks_loop(self) -> None:
        """Poll Postgres for pending unenqueued rows and push to Redis streams."""
        while self._running:
            try:
                fed_tasks = await self._feed_tasks()
                fed_jobs = await self._feed_non_task_jobs()
                fed = fed_tasks + fed_jobs
                if fed == 0:
                    await asyncio.sleep(FEED_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("SyncService feeder error")
                await asyncio.sleep(FEED_INTERVAL * 2)

    async def _feed_tasks(self) -> int:
        """Feed a batch of pending tasks from Postgres into Redis streams."""
        settings = get_settings()

        # Check stream backpressure for both streams
        try:
            recipe_len = await self._redis.xlen(RECIPE_PIPELINE_STREAM)
            pipeline_len = await self._redis.xlen(TASK_STREAM)
        except Exception:
            logger.debug("Failed to check stream lengths")
            return 0

        # Pick the relevant stream length (recipe pipeline is the hot path)
        max_stream_len = max(recipe_len, pipeline_len)
        if max_stream_len >= settings.stream_soft_limit:
            return 0  # Back off until workers drain

        room = settings.stream_soft_limit - max_stream_len
        batch_limit = min(FEED_BATCH_SIZE, room)

        # Feed only stale pending tasks so direct API enqueue paths get first chance
        # to write + mark rows without racing this feeder.
        # Include:
        #  1) recipe tasks, and
        #  2) task-backed canonical jobs still in pending state (fallback path).
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT id, url, entity_type, config
                FROM tasks
                WHERE state = 'pending'
                  AND enqueued_at IS NULL
                  AND created_at < NOW() - (%s * INTERVAL '1 second')
                  AND (
                    url LIKE 'recipe://%%'
                    OR EXISTS (
                        SELECT 1
                        FROM jobs j
                        WHERE j.related_task_id = tasks.id
                          AND j.state = 'pending'
                    )
                  )
                ORDER BY created_at
                LIMIT %s
                """,
                (FEED_GRACE_SECONDS, batch_limit),
            )
            rows = cur.fetchall()

        if not rows:
            return 0

        # Batch XADD via Redis pipeline
        task_ids = []
        pipe = self._redis.pipeline()
        for row in rows:
            task_id = str(row["id"])
            url = row["url"]
            task_ids.append(task_id)

            if url.startswith("recipe://"):
                config = _normalize_recipe_config(
                    row.get("config"),
                    url=url,
                    entity_type=row.get("entity_type", ""),
                )
                pipe.xadd(RECIPE_PIPELINE_STREAM, {"task_id": task_id, "config": config})
            else:
                pipe.xadd(TASK_STREAM, {
                    "task_id": task_id,
                    "url": url,
                    "entity_type": row["entity_type"],
                })

        await pipe.execute()

        # Mark as enqueued
        with get_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE tasks SET enqueued_at = NOW() WHERE id = ANY(%s::uuid[])",
                (task_ids,),
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
                (task_ids,),
            )

        logger.info("Fed %d tasks to Redis streams", len(task_ids))
        return len(task_ids)

    async def _feed_non_task_jobs(self) -> int:
        """
        Feed stale pending non-task jobs directly from jobs -> streams.

        This is a fallback path when direct enqueue in submit_job fails.
        """
        settings = get_settings()

        with get_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT
                    j.id,
                    j.payload,
                    jt.stream_name
                FROM jobs j
                INNER JOIN job_types jt
                    ON jt.job_type = j.job_type
                   AND jt.version = j.version
                WHERE j.state = 'pending'
                  AND j.enqueued_at IS NULL
                  AND j.related_task_id IS NULL
                  AND jt.enabled = TRUE
                  AND jt.requires_task = FALSE
                  AND j.created_at < NOW() - (%s * INTERVAL '1 second')
                ORDER BY j.created_at
                LIMIT %s
                """,
                (FEED_GRACE_SECONDS, FEED_BATCH_SIZE),
            )
            rows = cur.fetchall()

        if not rows:
            return 0

        stream_lengths: dict[str, int] = {}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            stream = row["stream_name"]
            if stream not in grouped:
                grouped[stream] = []
            grouped[stream].append(row)

        enqueued_ids: list[str] = []
        for stream_name, stream_rows in grouped.items():
            try:
                if stream_name not in stream_lengths:
                    stream_lengths[stream_name] = await self._redis.xlen(stream_name)
            except Exception:
                logger.debug("Failed to read stream length for %s", stream_name)
                continue

            room = settings.stream_soft_limit - stream_lengths[stream_name]
            if room <= 0:
                continue

            selected = stream_rows[:room]
            pipe = self._redis.pipeline()
            for row in selected:
                payload = row.get("payload")
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                pipe.xadd(
                    stream_name,
                    {"job_id": str(row["id"]), "config": json.dumps(payload)},
                )
            try:
                await pipe.execute()
            except Exception:
                logger.exception("Failed feeding non-task jobs to stream %s", stream_name)
                continue

            enqueued_ids.extend(str(row["id"]) for row in selected)
            stream_lengths[stream_name] += len(selected)

        if not enqueued_ids:
            return 0

        with get_cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE jobs
                SET state = 'enqueued',
                    enqueued_at = NOW(),
                    updated_at = NOW()
                WHERE id = ANY(%s::uuid[])
                  AND state = 'pending'
                """,
                (enqueued_ids,),
            )

        logger.info("Fed %d non-task jobs to Redis streams", len(enqueued_ids))
        return len(enqueued_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _safe_lrem(self, key: str, value: str) -> None:
        """Remove item from processing list. Silent fail."""
        try:
            await self._redis.lrem(key, 1, value)
        except Exception:
            logger.debug("LREM failed for key %s", key)

    async def recover_processing(self) -> None:
        """
        On startup, move any items left in processing lists back to pending.

        This handles the case where SyncService crashed mid-drain.
        """
        for src, dst in [
            (ENTITIES_PROCESSING, ENTITIES_PENDING),
            (JOBS_RESULTS_PROCESSING, JOBS_RESULTS),
        ]:
            try:
                count = 0
                while True:
                    item = await self._redis.rpoplpush(src, dst)
                    if item is None:
                        break
                    count += 1
                if count > 0:
                    logger.info("Recovered %d items from %s back to %s", count, src, dst)
            except Exception:
                logger.exception("Recovery failed for %s", src)
