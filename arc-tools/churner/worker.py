#!/usr/bin/env python3
"""Temporal worker entry point."""

import asyncio
import logging
import sys

from temporalio.client import Client
from temporalio.worker import Worker

from config import TEMPORAL_HOST
from workflows import ChurnWorkflow, SchemaEvolutionWorkflow, GroupDiscoveryWorkflow
from activities import (
    fetch_pending_records, get_active_schema, extract_record,
    resolve_and_merge, mine_traits, sample_recent_entities,
    propose_schema_changes, apply_schema_update,
    refresh_facet_catalog, get_facet_stats, architect_groups, upsert_group,
    recover_stuck_raw_ingests,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("churner.worker")

FAST_ACTIVITIES = [
    fetch_pending_records, get_active_schema, extract_record,
    mine_traits, sample_recent_entities, apply_schema_update,
    refresh_facet_catalog, get_facet_stats, upsert_group,
    recover_stuck_raw_ingests,
]

HEAVY_ACTIVITIES = [
    resolve_and_merge, architect_groups, propose_schema_changes,
]

ALL_WORKFLOWS = [ChurnWorkflow, SchemaEvolutionWorkflow, GroupDiscoveryWorkflow]


async def run_fast():
    client = await Client.connect(TEMPORAL_HOST)
    log.info("Starting fast worker (max_concurrent=4)")
    worker = Worker(
        client,
        task_queue="fast-extraction",
        workflows=ALL_WORKFLOWS,
        activities=FAST_ACTIVITIES,
        max_concurrent_activities=4,
    )
    await worker.run()


async def run_heavy():
    client = await Client.connect(TEMPORAL_HOST)
    log.info("Starting heavy worker (max_concurrent=1)")
    worker = Worker(
        client,
        task_queue="heavy-reasoning",
        activities=HEAVY_ACTIVITIES,
        max_concurrent_activities=1,
    )
    await worker.run()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if mode == "fast":
        asyncio.run(run_fast())
    elif mode == "heavy":
        asyncio.run(run_heavy())
    else:
        print(f"Usage: {sys.argv[0]} [fast|heavy]")
        sys.exit(1)
