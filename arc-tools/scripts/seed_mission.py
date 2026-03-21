#!/usr/bin/env python3
"""Create a mission with an initial schema."""

import argparse
import asyncio
import json
import sys

import asyncpg

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "churner"))
from config import DATABASE_URL


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--entity-type", default="default")
    parser.add_argument("--schema-file", help="Path to initial JSON Schema file")
    args = parser.parse_args()

    conn = await asyncpg.connect(DATABASE_URL)
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")

    try:
        row = await conn.fetchrow(
            "INSERT INTO missions (name, purpose) VALUES ($1, $2) RETURNING id",
            args.name, args.purpose,
        )
        mission_id = row["id"]
        print(f"Created mission: {args.name} (ID: {mission_id})")

        if args.schema_file:
            with open(args.schema_file) as f:
                schema = json.load(f)
        else:
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            }

        await conn.execute(
            """INSERT INTO schemas (mission_id, entity_type, version, json_schema, created_by)
               VALUES ($1, $2, 1, $3, 'seed')""",
            mission_id, args.entity_type, schema,
        )
        print(f"Created initial schema v1 for entity type: {args.entity_type}")
    except asyncpg.UniqueViolationError:
        print(f"Mission '{args.name}' already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
