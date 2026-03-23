"""
Temporal workflows v2 — pattern-based grab pipeline.

PatternGrabWorkflow:
  1. fetch_completed_audits → audit records + patterns from ghostgraph
  2. expand_urls → discover all detail page URLs via pagination + entity selectors
  3. grab_with_patterns → CSS selector extraction (ZERO LLM)
  4. bridge_pattern_grab_to_churner → push chunks to churner.raw_ingests
  5. mark_audit_grabbed / mark_audit_failed
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import uuid

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from temporal_activities_v2 import (
        fetch_completed_audits,
        expand_urls,
        grab_with_patterns,
        bridge_pattern_grab_to_churner,
        mark_audit_grabbed,
        mark_audit_failed,
        FetchCompletedResult,
        ExpandResult,
        GrabResult,
    )

RETRY_FAST = RetryPolicy(maximum_attempts=3, backoff_coefficient=2.0)
RETRY_SCRAPE = RetryPolicy(maximum_attempts=2, initial_interval=5.0, backoff_coefficient=2.0)


@dataclass
class PatternGrabParams:
    entity_type: str | None = None  # None = all entity types
    mission_id: str | None = None   # churner mission to bridge into
    batch_size: int = 10            # URLs per grab_with_patterns batch


@workflow.defn
class PatternGrabWorkflow:
    """
    Consume site-auditor output (dom_patterns, url_patterns) from ghostgraph DB
    and apply pattern-based extraction at scale.

    ZERO LLM calls throughout.
    """

    @workflow.run
    async def run(self, params: PatternGrabParams) -> dict:
        workflow.logger.info("PatternGrabWorkflow starting (entity_type=%s)", params.entity_type or "all")

        # Step 1: fetch completed audits
        fetch_result: FetchCompletedResult = await workflow.execute_activity(
            fetch_completed_audits,
            args=[params.entity_type],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RETRY_FAST,
            task_queue="grabber",
            heartbeat_timeout=timedelta(minutes=2),
        )

        if fetch_result.count == 0:
            workflow.logger.info("No completed audits to grab — exiting")
            return {"audits_processed": 0, "total_chunks": 0, "errors": []}

        total_chunks = 0
        total_bridged = 0
        errors = []

        # Process each audit
        for audit in fetch_result.audits:
            audit_id = audit.audit_id
            workflow.logger.info(
                "Processing audit %s (domain=%s, entity_type=%s)",
                audit_id, audit.domain, audit.entity_type,
            )

            try:
                # Collect ALL index selector sets from ALL dom_patterns where page_type='index'
                index_selector_sets: list[dict] = []
                for dp in audit.dom_patterns:
                    if dp.get("page_type") == "index" and dp.get("entity_selector"):
                        index_selector_sets.append({
                            "entity_selector": dp.get("entity_selector", ""),
                            "link_selector": dp.get("link_selector", ""),
                        })

                workflow.logger.info(
                    "Audit %s: collected %d index selector sets",
                    audit_id, len(index_selector_sets),
                )

                # Step 2: expand_urls — discover detail page URLs
                expand_result: ExpandResult = await workflow.execute_activity(
                    expand_urls,
                    args=[audit_id, audit.domain, audit.url_patterns, index_selector_sets],
                    start_to_close_timeout=timedelta(minutes=15),
                    retry_policy=RETRY_SCRAPE,
                    task_queue="grabber",
                    heartbeat_timeout=timedelta(minutes=5),
                )

                if not expand_result.detail_urls:
                    workflow.logger.warning(
                        "No detail URLs expanded for audit %s — skipping",
                        audit_id,
                    )
                    await workflow.execute_activity(
                        mark_audit_failed,
                        args=[audit_id, "no detail URLs found"],
                        start_to_close_timeout=timedelta(seconds=30),
                        task_queue="grabber",
                    )
                    errors.append({"audit_id": audit_id, "error": "no detail URLs"})
                    continue

                # Collect dom_pattern chunks from all dom_pattern records
                all_chunks = []
                for dp in audit.dom_patterns:
                    chunks_def = dp.get("chunks", [])
                    if isinstance(chunks_def, list):
                        all_chunks.extend(chunks_def)

                if not all_chunks:
                    # Fallback: try generic selectors if no named chunks
                    all_chunks = []

                # Step 3: grab_with_patterns — batch extraction
                grab_result: GrabResult = await workflow.execute_activity(
                    grab_with_patterns,
                    args=[
                        expand_result.detail_urls,
                        audit.domain,
                        audit.entity_type,
                        all_chunks,
                    ],
                    start_to_close_timeout=timedelta(minutes=30),
                    retry_policy=RETRY_FAST,
                    task_queue="grabber",
                    heartbeat_timeout=timedelta(minutes=10),
                )

                if grab_result.chunks:
                    # Step 4: bridge to churner
                    source_tag = audit.domain
                    # Validate mission_id is a proper UUID — raw_ingests.mission_id is FK to missions.id
                    mission_id = params.mission_id
                    if not mission_id:
                        workflow.logger.warning("No mission_id set — cannot bridge to churner, skipping")
                        continue
                    try:
                        uuid.UUID(mission_id)
                    except ValueError:
                        workflow.logger.error("Invalid mission_id UUID '%s' — cannot bridge to churner", mission_id)
                        errors.append({"audit_id": audit_id, "error": f"invalid mission_id: {mission_id}"})
                        continue
                    bridged = await workflow.execute_activity(
                        bridge_pattern_grab_to_churner,
                        args=[mission_id, grab_result.chunks, source_tag],
                        start_to_close_timeout=timedelta(minutes=5),
                        retry_policy=RETRY_FAST,
                        task_queue="grabber",
                    )
                    total_bridged += bridged
                    total_chunks += len(grab_result.chunks)

                # Step 5: mark success
                await workflow.execute_activity(
                    mark_audit_grabbed,
                    args=[audit_id],
                    start_to_close_timeout=timedelta(seconds=30),
                    task_queue="grabber",
                )

                workflow.logger.info(
                    "Audit %s done: %d URLs, %d chunks extracted, %d bridged",
                    audit_id,
                    len(expand_result.detail_urls),
                    len(grab_result.chunks),
                    bridged if grab_result.chunks else 0,
                )

            except Exception as e:
                workflow.logger.error("Audit %s failed: %s", audit_id, e)
                errors.append({"audit_id": audit_id, "error": str(e)})
                try:
                    await workflow.execute_activity(
                        mark_audit_failed,
                        args=[audit_id, str(e)],
                        start_to_close_timeout=timedelta(seconds=30),
                        task_queue="grabber",
                    )
                except Exception:
                    pass

        workflow.logger.info(
            "PatternGrabWorkflow complete: %d audits, %d total chunks, %d bridged",
            fetch_result.count, total_chunks, total_bridged,
        )

        return {
            "audits_processed": fetch_result.count,
            "total_chunks": total_chunks,
            "total_bridged": total_bridged,
            "errors": errors,
        }
