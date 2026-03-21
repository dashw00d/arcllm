"""Temporal workflow definitions."""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import (
        fetch_pending_records,
        get_active_schema,
        extract_record,
        resolve_and_merge,
        mine_traits,
        sample_recent_entities,
        propose_schema_changes,
        apply_schema_update,
        refresh_facet_catalog,
        get_facet_stats,
        architect_groups,
        upsert_group,
    )


@workflow.defn
class ChurnWorkflow:
    """Main loop: grab pending raw records, extract, resolve, merge, mine traits."""

    @workflow.run
    async def run(self, mission_id: str, batch_size: int = 50):
        while True:
            # Reset history periodically to prevent replay bloat
            if workflow.info().get_current_history_length() > 1000:
                workflow.continue_as_new(args=[mission_id, batch_size])

            batch = await workflow.execute_activity(
                fetch_pending_records,
                args=[mission_id, batch_size],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue="fast-extraction",
            )

            if not batch:
                await workflow.sleep(timedelta(minutes=5))
                continue

            schema = await workflow.execute_activity(
                get_active_schema,
                args=[mission_id],
                start_to_close_timeout=timedelta(seconds=10),
                task_queue="fast-extraction",
            )

            # Extract entities (parallel fan-out)
            extraction_futures = []
            for record in batch:
                fut = workflow.execute_activity(
                    extract_record,
                    args=[record["id"], record["raw_payload"], schema],
                    start_to_close_timeout=timedelta(minutes=5),
                    task_queue="fast-extraction",
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
                extraction_futures.append(fut)

            extracted = await asyncio.gather(*extraction_futures, return_exceptions=True)

            # Resolve + merge (sequential — heavy reasoning)
            resolved = []
            for entity in extracted:
                if isinstance(entity, Exception) or entity is None:
                    continue
                result = await workflow.execute_activity(
                    resolve_and_merge,
                    args=[mission_id, entity],
                    start_to_close_timeout=timedelta(minutes=10),
                    task_queue="heavy-reasoning",
                )
                if result:
                    resolved.append(result)

            # Mine traits (parallel, fast)
            trait_futures = []
            for entity in resolved:
                entity_id = entity.get("_entity_id")
                if entity_id:
                    fut = workflow.execute_activity(
                        mine_traits,
                        args=[entity_id],
                        start_to_close_timeout=timedelta(minutes=2),
                        task_queue="fast-extraction",
                    )
                    trait_futures.append(fut)

            if trait_futures:
                await asyncio.gather(*trait_futures, return_exceptions=True)


@workflow.defn
class SchemaEvolutionWorkflow:
    """Periodic: analyze recent extractions, propose schema changes."""

    @workflow.run
    async def run(self, mission_id: str):
        while True:
            await workflow.sleep(timedelta(hours=4))

            sample = await workflow.execute_activity(
                sample_recent_entities,
                args=[mission_id, 500],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue="fast-extraction",
            )

            if len(sample) < 50:
                continue

            proposal = await workflow.execute_activity(
                propose_schema_changes,
                args=[mission_id, sample],
                start_to_close_timeout=timedelta(minutes=10),
                task_queue="heavy-reasoning",
            )

            if proposal and proposal.get("confidence", 0) > 0.85:
                await workflow.execute_activity(
                    apply_schema_update,
                    args=[mission_id, proposal],
                    start_to_close_timeout=timedelta(seconds=30),
                    task_queue="fast-extraction",
                )

            workflow.continue_as_new(args=[mission_id])


@workflow.defn
class GroupDiscoveryWorkflow:
    """Periodic: discover facet combos, build SEO groups."""

    @workflow.run
    async def run(self, mission_id: str):
        while True:
            await workflow.sleep(timedelta(hours=2))

            await workflow.execute_activity(
                refresh_facet_catalog,
                args=[mission_id],
                start_to_close_timeout=timedelta(minutes=5),
                task_queue="fast-extraction",
            )

            facet_stats = await workflow.execute_activity(
                get_facet_stats,
                args=[mission_id],
                start_to_close_timeout=timedelta(seconds=30),
                task_queue="fast-extraction",
            )

            new_groups = await workflow.execute_activity(
                architect_groups,
                args=[mission_id, facet_stats],
                start_to_close_timeout=timedelta(minutes=10),
                task_queue="heavy-reasoning",
            )

            for group in (new_groups or []):
                await workflow.execute_activity(
                    upsert_group,
                    args=[mission_id, group],
                    start_to_close_timeout=timedelta(seconds=30),
                    task_queue="fast-extraction",
                )

            workflow.continue_as_new(args=[mission_id])
