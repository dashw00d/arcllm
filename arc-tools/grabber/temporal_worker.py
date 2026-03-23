#!/usr/bin/env python3
"""Temporal worker for Grabber — replaces Redis Streams unified_worker."""

import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.worker import Worker

from temporal_workflows import GrabWorkflow
from temporal_workflows_v2 import PatternGrabWorkflow
from temporal_activities import (
    explore_urls,
    generate_schema,
    extract_batch,
    bridge_to_churner,
)
from temporal_activities_v2 import (
    fetch_completed_audits,
    expand_urls,
    grab_with_patterns,
    bridge_pattern_grab_to_churner,
    mark_audit_grabbed,
    mark_audit_failed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("grabber.worker")

TEMPORAL_HOST = os.environ.get("TEMPORAL_HOST", "localhost:7233")


async def main():
    client = await Client.connect(TEMPORAL_HOST)
    log.info("Grabber worker connecting to Temporal at %s", TEMPORAL_HOST)

    worker = Worker(
        client,
        task_queue="grabber",
        workflows=[GrabWorkflow, PatternGrabWorkflow],
        activities=[
            # Original workflow activities
            explore_urls,
            generate_schema,
            extract_batch,
            bridge_to_churner,
            # Pattern-grab workflow activities
            fetch_completed_audits,
            expand_urls,
            grab_with_patterns,
            bridge_pattern_grab_to_churner,
            mark_audit_grabbed,
            mark_audit_failed,
        ],
        max_concurrent_activities=2,
    )
    log.info("Grabber worker started (task_queue=grabber, max_concurrent=2)")
    log.info("Registered workflows: GrabWorkflow, PatternGrabWorkflow")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
