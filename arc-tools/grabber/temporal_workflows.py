"""
Temporal workflows for Grabber — full gather → curate pipeline.

GrabWorkflow: seed URL → explore → schema → extract → bridge to churner
The churner's ChurnWorkflow picks up bridged records automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from temporal_activities import (
        explore_urls,
        generate_schema,
        extract_batch,
        bridge_to_churner,
        ExploreResult,
        SchemaResult,
        ExtractResult,
    )

RETRY_FAST = RetryPolicy(maximum_attempts=3, backoff_coefficient=2.0)
RETRY_SCRAPE = RetryPolicy(maximum_attempts=2, backoff_coefficient=2.0)


@dataclass
class GrabParams:
    seed_url: str
    entity_type: str
    mission_id: str  # churner mission to bridge results into
    batch_size: int = 10  # URLs per extract_batch activity


@workflow.defn
class GrabWorkflow:
    """Full pipeline: explore URLs → generate schema → extract → bridge to churner."""

    @workflow.run
    async def run(self, params: GrabParams) -> dict:
        # Phase 1: URL Discovery
        explore_result: ExploreResult = await workflow.execute_activity(
            explore_urls,
            args=[params.seed_url, params.entity_type],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RETRY_SCRAPE,
            task_queue="grabber",
            heartbeat_timeout=timedelta(minutes=2),
        )

        # Phase 2: Schema Generation
        schema_result: SchemaResult = await workflow.execute_activity(
            generate_schema,
            args=[explore_result.sample_urls, params.entity_type, params.seed_url],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RETRY_SCRAPE,
            task_queue="grabber",
            heartbeat_timeout=timedelta(minutes=2),
        )

        # Phase 3: Batch Extraction (fan out in chunks)
        all_entities = []
        urls = explore_result.detail_urls
        for i in range(0, len(urls), params.batch_size):
            batch = urls[i:i + params.batch_size]
            result: ExtractResult = await workflow.execute_activity(
                extract_batch,
                args=[batch, schema_result.fields, params.entity_type],
                start_to_close_timeout=timedelta(minutes=15),
                retry_policy=RETRY_FAST,
                task_queue="grabber",
                heartbeat_timeout=timedelta(minutes=5),
            )
            all_entities.extend(result.entities)

            # Reset history if getting long
            if workflow.info().get_current_history_length() > 5000:
                # Can't continue_as_new mid-extraction, just keep going
                pass

        # Phase 4: Bridge to Churner
        source_tag = schema_result.domain
        bridged = 0
        if all_entities and params.mission_id:
            bridged = await workflow.execute_activity(
                bridge_to_churner,
                args=[params.mission_id, all_entities, source_tag],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RETRY_FAST,
                task_queue="grabber",
            )

        return {
            "urls_discovered": len(urls),
            "schema_fields": schema_result.field_count,
            "entities_extracted": len(all_entities),
            "entities_bridged": bridged,
            "domain": schema_result.domain,
        }
