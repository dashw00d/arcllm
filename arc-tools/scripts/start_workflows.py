#!/usr/bin/env python3
"""Start the eternal churn workflows for a mission."""

import argparse
import asyncio
import sys

from temporalio.client import Client

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "churner"))
from config import TEMPORAL_HOST

import asyncpg
from config import DATABASE_URL


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", required=True, help="Mission name")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        row = await conn.fetchrow("SELECT id FROM missions WHERE name = $1", args.mission)
        if not row:
            print(f"Mission not found: {args.mission}")
            sys.exit(1)
        mission_id = str(row["id"])
    finally:
        await conn.close()

    client = await Client.connect(TEMPORAL_HOST)

    workflows = [
        ("ChurnWorkflow", f"churn-{args.mission}", [mission_id, args.batch_size]),
        ("SchemaEvolutionWorkflow", f"schema-evolution-{args.mission}", [mission_id]),
        ("GroupDiscoveryWorkflow", f"group-discovery-{args.mission}", [mission_id]),
    ]

    for name, wf_id, wf_args in workflows:
        try:
            await client.start_workflow(
                name, args=wf_args, id=wf_id, task_queue="fast-extraction",
            )
            print(f"Started {name} for {args.mission}")
        except Exception as e:
            if "already started" in str(e).lower():
                print(f"{name} already running for {args.mission}")
            else:
                raise


if __name__ == "__main__":
    asyncio.run(main())
