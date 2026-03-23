#!/usr/bin/env python3
"""
Start a PatternGrabWorkflow for a given entity_type.

Usage:
    python3 scripts/start_pattern_grab.py --name 'wedding venues' --entity-type 'wedding venues' --mission-id <uuid>
    python3 scripts/start_pattern_grab.py --name 'wedding venues' --entity-type 'wedding venues'  # auto-creates mission

Environment:
    TEMPORAL_HOST   default: localhost:7233
    POSTGRES_URL    default: postgresql://temporal:temporal@localhost:5432/ghostgraph
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid

import psycopg2
from temporalio.client import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("start_pattern_grab")


def _churner_conn():
    """Connect directly to the churner DB."""
    postgres_url = os.environ.get(
        "POSTGRES_URL", "postgresql://temporal:temporal@localhost:5432/ghostgraph"
    )
    churner_url = postgres_url.rsplit("/", 1)[0] + "/churner"
    return psycopg2.connect(churner_url)


async def main():
    parser = argparse.ArgumentParser(description="Start a PatternGrabWorkflow")
    parser.add_argument("--entity-type", type=str, default=None,
                        help="Entity type to filter audits (e.g. 'wedding venues')")
    parser.add_argument("--name", type=str, default=None,
                        help="Mission name — creates a new mission if --mission-id not given")
    parser.add_argument("--mission-id", type=str, default=None,
                        help="Churner mission ID to bridge results into (UUID)")
    parser.add_argument("--workflow-id", type=str, default=None,
                        help="Optional custom workflow ID (default: auto-generated)")
    parser.add_argument("--temporal-host", type=str, default=None,
                        help="Temporal host (default: from TEMPORAL_HOST env or localhost:7233)")
    args = parser.parse_args()

    temporal_host = args.temporal_host or os.environ.get("TEMPORAL_HOST", "localhost:7233")

    client = await Client.connect(temporal_host)
    log.info("Connected to Temporal at %s", temporal_host)

    # Resolve mission_id
    mission_id = args.mission_id
    if mission_id:
        try:
            uuid.UUID(mission_id)
        except ValueError:
            log.error("--mission-id must be a valid UUID: %s", mission_id)
            return
        log.info("Using existing mission_id=%s", mission_id)
    elif args.name:
        # Auto-create mission in churner DB
        try:
            conn = _churner_conn()
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO missions (name, purpose) VALUES (%s, %s) RETURNING id",
                (args.name, f"PatternGrabWorkflow: {args.name}"),
            )
            mission_id = str(cur.fetchone()[0])
            cur.close()
            conn.close()
            log.info("Created mission: %s (ID: %s)", args.name, mission_id)
            print(f"mission_id={mission_id}")
        except psycopg2.IntegrityError:
            log.error("Mission '%s' already exists — use --mission-id instead", args.name)
            return
        except Exception as e:
            log.error("Failed to create mission: %s", e)
            return
    else:
        log.error("Must provide either --mission-id or --name")
        print("ERROR: Must provide either --mission-id or --name", file=__import__("sys").stderr)
        return

    # Build workflow ID
    wf_id = args.workflow_id
    if not wf_id:
        ts = asyncio.get_event_loop().time()
        entity_suffix = args.entity_type.replace(" ", "-") if args.entity_type else "all"
        wf_id = f"pattern-grab-{entity_suffix}-{int(ts)}"

    log.info("Starting PatternGrabWorkflow id=%s entity_type=%s mission_id=%s",
             wf_id, args.entity_type or "all", mission_id)

    from grabber.temporal_workflows_v2 import PatternGrabWorkflow, PatternGrabParams

    handle = await client.start_workflow(
        PatternGrabWorkflow.run,
        PatternGrabParams(
            entity_type=args.entity_type,
            mission_id=mission_id,
            batch_size=10,
        ),
        id=wf_id,
        task_queue="grabber",
    )

    log.info("Workflow started: %s", handle.id)
    log.info("Check status at: Temporal UI (usually http://localhost:8233)")

    # Optionally wait for result
    log.info("Waiting for workflow to complete (Ctrl-C to exit without waiting)...")
    try:
        result = await handle.result(timeout=3600)
        log.info("Workflow complete: %s", result)
    except asyncio.TimeoutError:
        log.warning("Workflow still running after 1 hour — check Temporal UI")


if __name__ == "__main__":
    asyncio.run(main())
