#!/usr/bin/env python3
"""
Mission orchestrator — ties the full pipeline together.

Usage:
    python3 scripts/orchestrator.py --entity-type 'wedding venues' --name 'wedding-venues-v1'

This script:
1. Creates a churner mission (if name doesn't exist)
2. Checks for queued sites in ghostgraph.sites
3. Monitors audit progress
4. Triggers PatternGrabWorkflow for completed audits
5. Reports stats every iteration

Run as a loop (--loop) or single pass (default).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("orchestrator")

# ── DB helpers ────────────────────────────────────────────────────────────

GHOSTGRAPH_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://temporal:temporal@localhost:5432/ghostgraph"
)
CHURNER_URL = os.environ.get(
    "CHURNER_URL", "postgresql://temporal:temporal@localhost:5432/churner"
)
TEMPORAL_HOST = os.environ.get("TEMPORAL_HOST", "localhost:7233")


def _gf_query(sql: str, args: tuple | None = None) -> list[dict]:
    """Execute a read query on ghostgraph DB."""
    conn = psycopg2.connect(GHOSTGRAPH_URL)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, args) if args else cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def _churner_conn():
    """Return a connection to churner DB."""
    return psycopg2.connect(CHURNER_URL)


# ── Mission management ────────────────────────────────────────────────────

def get_or_create_mission(name: str, purpose: str) -> str:
    """
    Get existing mission by name, or create a new one.
    Returns mission_id (uuid str).
    """
    conn = _churner_conn()
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("SELECT id FROM missions WHERE name = %s", (name,))
    row = cur.fetchone()
    if row:
        mission_id = str(row["id"])
        log.info("Using existing mission %s (%s)", mission_id, name)
        cur.close()
        conn.close()
        return mission_id

    cur.execute(
        "INSERT INTO missions (name, purpose) VALUES (%s, %s) RETURNING id",
        (name, purpose),
    )
    row = cur.fetchone()
    mission_id = str(row["id"])
    log.info("Created mission %s (%s)", mission_id, name)
    cur.close()
    conn.close()
    return mission_id


# ── Stats ─────────────────────────────────────────────────────────────────

def get_stats(entity_type: str | None) -> dict:
    """
    Return a dict of pipeline stats:
    - sites_queued: count in ghostgraph.sites with status='queued'
    - sites_auditing: count in ghostgraph.sites with status='auditing'
    - audits_pending: entity_audits with stage not in (complete, not_relevant)
    - audits_complete: entity_audits with stage='complete' and grab_status='pending'
    - audits_grabbing: entity_audits with grab_status='grabbing'
    - audits_grab_done: entity_audits with grab_status='done'
    - raw_ingests_pending: churner.raw_ingests with status='pending' for this mission
    - raw_ingests_done: churner.raw_ingests with status!='pending'
    - entities_count: churner.entities count
    """
    stats = {}

    # Sites stats
    if entity_type:
        sites_where = f"AND description ILIKE '%{entity_type}%'"
        audit_where = f"AND (ea.description ILIKE '%{entity_type}%' OR s.description ILIKE '%{entity_type}%')"
    else:
        sites_where = ""
        audit_where = ""

    sites_q = _gf_query(
        f"SELECT status, COUNT(*) as cnt FROM sites WHERE true {sites_where} GROUP BY status"
    )
    for row in sites_q:
        stats[f"sites_{row['status']}"] = row["cnt"]

    # Audit stats
    audit_q = _gf_query(
        f"""
        SELECT ea.stage, COALESCE(ea.grab_status, 'none') as grab_status, COUNT(*) as cnt
        FROM entity_audits ea
        JOIN sites s ON s.id = ea.site_id
        WHERE true {audit_where}
        GROUP BY ea.stage, COALESCE(ea.grab_status, 'none')
        """
    )
    for row in audit_q:
        key = f"audits_{row['stage']}_{row['grab_status']}"
        stats[key] = row["cnt"]

    return stats


def print_stats(stats: dict, mission_id: str, entity_type: str | None):
    """Pretty-print stats."""
    print("\n=== Pipeline Stats ===")
    print(f"  Entity type : {entity_type or 'all'}")
    print(f"  Mission ID  : {mission_id}")
    print(f"  Sites queued     : {stats.get('sites_queued', 0)}")
    print(f"  Sites auditing   : {stats.get('sites_auditing', 0)}")
    print(f"  Sites ready      : {stats.get('sites_ready', 0)}")
    print(f"  Audits pending   : {sum(v for k, v in stats.items() if k.startswith('audits_') and not k.startswith('audits_complete_') and not k.startswith('audits_not_relevant_'))}")
    print(f"  Audits complete  : {stats.get('audits_complete_pending', 0)}")
    print(f"  Audits grabbing  : {stats.get('audits_complete_grabbing', 0)}")
    print(f"  Audits done      : {stats.get('audits_complete_done', 0)}")
    print(f"  Raw ingests pend : {stats.get('raw_ingests_pending', 0)}")
    print(f"  Raw ingests done : {stats.get('raw_ingests_done', 0)}")
    print(f"  Entities         : {stats.get('entities_count', 0)}")
    print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
    print()


# ── PatternGrabWorkflow trigger ───────────────────────────────────────────

async def trigger_pattern_grab(mission_id: str, entity_type: str | None) -> int:
    """
    Start a PatternGrabWorkflow for the given entity_type and mission_id.
    Returns the number of workflows started.
    """
    from temporalio.client import Client

    client = await Client.connect(TEMPORAL_HOST)
    ts = int(time.time())
    wf_id = f"pattern-grab-{entity_type or 'all'}-{ts}"

    try:
        from grabber.temporal_workflows_v2 import PatternGrabWorkflow, PatternGrabParams

        handle = await client.start_workflow(
            PatternGrabWorkflow.run,
            PatternGrabParams(entity_type=entity_type, mission_id=mission_id, batch_size=10),
            id=wf_id,
            task_queue="grabber",
        )
        log.info(
            "Started PatternGrabWorkflow %s for entity_type=%s mission=%s",
            handle.id, entity_type, mission_id,
        )
        return 1
    except Exception as e:
        log.error("Failed to start PatternGrabWorkflow (entity_type=%s, mission=%s): %s", entity_type, mission_id, e)
        return 0


# ── Single iteration ───────────────────────────────────────────────────────

async def run_once(mission_id: str, entity_type: str | None) -> bool:
    """
    Run one iteration of the orchestrator loop.
    Returns True if there's more work to do, False if idle.
    """
    stats = get_stats(entity_type)

    # Get mission-specific raw ingest stats
    conn = _churner_conn()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "SELECT status, COUNT(*) as cnt FROM raw_ingests WHERE mission_id = %s GROUP BY status",
        (mission_id,),
    )
    for row in cur.fetchall():
        stats[f"raw_ingests_{row[0]}"] = row[1]
    cur.execute(
        "SELECT COUNT(*) FROM entities WHERE mission_id = %s",
        (mission_id,),
    )
    stats["entities_count"] = cur.fetchone()[0]
    cur.close()
    conn.close()

    print_stats(stats, mission_id, entity_type)

    # Trigger PatternGrabWorkflow if there are completed audits waiting
    complete_audits = stats.get("audits_complete_pending", 0)
    if complete_audits > 0:
        started = await trigger_pattern_grab(mission_id, entity_type)
        if started:
            log.info(
                "Triggered PatternGrabWorkflow (entity_type=%s, pending_audits=%d)",
                entity_type, complete_audits,
            )
            return True

    # Check if there's more work in the pipeline
    queued = stats.get("sites_queued", 0)
    pending_audits = sum(
        v for k, v in stats.items()
        if k.startswith("audits_")
        and not k.startswith("audits_complete_")
        and not k.startswith("audits_not_relevant_")
    )
    grabbing = stats.get("audits_complete_grabbing", 0)

    if queued > 0 or pending_audits > 0 or grabbing > 0 or complete_audits > 0:
        return True

    return False


# ── Main ─────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Mission orchestrator")
    parser.add_argument("--entity-type", type=str, required=True,
                        help="Entity type to orchestrate (e.g. 'wedding venues')")
    parser.add_argument("--name", type=str, required=True,
                        help="Mission name (used for churner.missions)")
    parser.add_argument("--purpose", type=str, default="",
                        help="Mission purpose description")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously every 5 minutes")
    parser.add_argument("--interval", type=int, default=300,
                        help="Loop interval in seconds (default: 300 = 5 min)")
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="Max loop iterations (0=unlimited)")
    args = parser.parse_args()

    log.info("Orchestrator starting for entity_type=%s name=%s", args.entity_type, args.name)

    # Get or create mission
    purpose = args.purpose or f"Data pipeline for: {args.entity_type}"
    mission_id = get_or_create_mission(args.name, purpose)
    log.info("Mission ID: %s", mission_id)

    iterations = 0
    while True:
        has_work = await run_once(mission_id, args.entity_type)
        iterations += 1

        if not args.loop:
            break

        if args.max_iterations > 0 and iterations >= args.max_iterations:
            log.info("Max iterations (%d) reached — exiting", args.max_iterations)
            break

        if not has_work:
            log.info("Pipeline idle — sleeping %ds before next check", args.interval)

        await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
