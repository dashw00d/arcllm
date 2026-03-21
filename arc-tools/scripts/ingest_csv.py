#!/usr/bin/env python3
"""Bulk ingest raw data from CSV or JSON files."""

import argparse
import asyncio
import csv
import json
import sys

import asyncpg

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "churner"))
from config import DATABASE_URL
from db import ingest_raw


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", required=True, help="Mission name")
    parser.add_argument("--file", required=True, help="CSV or JSON file path")
    parser.add_argument("--source", default="file_import", help="Source label")
    args = parser.parse_args()

    conn = await asyncpg.connect(DATABASE_URL)
    row = await conn.fetchrow("SELECT id FROM missions WHERE name = $1", args.mission)
    if not row:
        print(f"Mission not found: {args.mission}")
        sys.exit(1)
    mission_id = str(row["id"])
    await conn.close()

    # Read file
    path = args.file
    records = []
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
            records = data if isinstance(data, list) else [data]
    elif path.endswith(".csv"):
        with open(path) as f:
            reader = csv.DictReader(f)
            records = list(reader)
    else:
        print("Unsupported file format. Use .csv or .json")
        sys.exit(1)

    # Ingest
    imported = 0
    dupes = 0
    for record in records:
        result = await ingest_raw(mission_id, args.source, record)
        if result:
            imported += 1
        else:
            dupes += 1

    print(f"Ingested {imported} records ({dupes} duplicates skipped)")


if __name__ == "__main__":
    asyncio.run(main())
