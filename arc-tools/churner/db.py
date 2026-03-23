"""Database helpers using asyncpg."""

import json
import hashlib
from uuid import UUID

import asyncpg

from config import DATABASE_URL

_pool: asyncpg.Pool | None = None


async def _init_conn(conn: asyncpg.Connection):
    """Register JSON codecs so asyncpg handles JSONB columns natively."""
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    await conn.set_type_codec("json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10, init=_init_conn)
    return _pool


async def fetch_pending_records(mission_id: str, limit: int = 50) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """UPDATE raw_ingests
           SET status = 'processing'
           WHERE id IN (
               SELECT id FROM raw_ingests
               WHERE mission_id = $1 AND status = 'pending'
               ORDER BY ingested_at
               LIMIT $2
               FOR UPDATE SKIP LOCKED
           )
           RETURNING id, raw_payload""",
        UUID(mission_id), limit,
    )
    return [{"id": str(r["id"]), "raw_payload": r["raw_payload"]} for r in rows]


async def mark_raw_done(record_id: str):
    pool = await get_pool()
    await pool.execute(
        "UPDATE raw_ingests SET status = 'done', processed_at = now() WHERE id = $1",
        UUID(record_id),
    )


async def mark_raw_error(record_id: str):
    pool = await get_pool()
    await pool.execute(
        "UPDATE raw_ingests SET status = 'error' WHERE id = $1",
        UUID(record_id),
    )


async def get_active_schema(mission_id: str, entity_type: str = "default") -> dict:
    pool = await get_pool()
    row = await pool.fetchrow(
        """SELECT json_schema FROM schemas
           WHERE mission_id = $1 AND entity_type = $2
           ORDER BY version DESC LIMIT 1""",
        UUID(mission_id), entity_type,
    )
    if row:
        return row["json_schema"]
    return {}


async def insert_entity(mission_id: str, entity_type: str, schema_version: int,
                        data: dict, source_ids: list[str], confidence: float) -> str:
    pool = await get_pool()
    row = await pool.fetchrow(
        """INSERT INTO entities (mission_id, entity_type, schema_version, data, source_ids, confidence)
           VALUES ($1, $2, $3, $4, $5::uuid[], $6)
           RETURNING id""",
        UUID(mission_id), entity_type, schema_version,
        data, [UUID(s) for s in source_ids], confidence,
    )
    return str(row["id"])


async def mark_as_golden(entity_id: str):
    pool = await get_pool()
    await pool.execute(
        "UPDATE entities SET is_golden = true WHERE id = $1",
        UUID(entity_id),
    )


async def get_entity(entity_id: str) -> dict:
    pool = await get_pool()
    row = await pool.fetchrow("SELECT * FROM entities WHERE id = $1", UUID(entity_id))
    if not row:
        return None
    return {
        "id": str(row["id"]),
        "data": row["data"],
        "traits": row["traits"] or {},
        "is_golden": row["is_golden"],
    }


async def update_entity_traits(entity_id: str, traits: dict):
    pool = await get_pool()
    await pool.execute(
        "UPDATE entities SET traits = $1, updated_at = now() WHERE id = $2",
        traits, UUID(entity_id),
    )


async def find_candidate_entities(mission_id: str, entity_type: str, new_entity_name: str | None = None, limit: int = 20) -> list[dict]:
    """Find golden entities that might match a new extraction.

    Strategy:
    1. If new_entity_name is provided, use trigram similarity to find name-similar candidates
    2. Fall back to recent golden entities if no name-based matches
    """
    pool = await get_pool()
    candidates = []

    if new_entity_name and len(new_entity_name) >= 2:
        # Try trigram similarity match on name field
        rows = await pool.fetch(
            """SELECT id, data FROM entities
               WHERE mission_id = $1 AND entity_type = $2 AND is_golden = true
                 AND data->>'name' IS NOT NULL
                 AND similarity(data->>'name', $3) > 0.3
               ORDER BY similarity(data->>'name', $3) DESC
               LIMIT $4""",
            UUID(mission_id), entity_type, new_entity_name, limit,
        )
        candidates = [{"id": str(r["id"]), "data": r["data"]} for r in rows]

        # Also grab ILIKE prefix matches (handles substring matches)
        if len(candidates) < limit:
            rows = await pool.fetch(
                """SELECT id, data FROM entities
                   WHERE mission_id = $1 AND entity_type = $2 AND is_golden = true
                     AND data->>'name' IS NOT NULL
                     AND (data->>'name' ILIKE '%' || $3 || '%'
                          OR $3 ILIKE '%' || (data->>'name') || '%')
                   LIMIT $4""",
                UUID(mission_id), entity_type, new_entity_name, limit,
            )
            seen = {c["id"] for c in candidates}
            for r in rows:
                rid = str(r["id"])
                if rid not in seen:
                    candidates.append({"id": rid, "data": r["data"]})
                    seen.add(rid)

    # Fall back to recent golden entities if we don't have enough
    if len(candidates) < limit:
        fallback_limit = limit - len(candidates)
        rows = await pool.fetch(
            """SELECT id, data FROM entities
               WHERE mission_id = $1 AND entity_type = $2 AND is_golden = true
               ORDER BY updated_at DESC LIMIT $3""",
            UUID(mission_id), entity_type, fallback_limit,
        )
        seen = {c["id"] for c in candidates}
        for r in rows:
            rid = str(r["id"])
            if rid not in seen:
                candidates.append({"id": rid, "data": r["data"]})

    return candidates[:limit]


async def find_near_duplicates(mission_id: str, entity_type: str = "default", threshold: float = 0.5) -> list[dict]:
    """Find pairs of golden entities with similar names (potential duplicates).

    Returns list of dicts with from_id, to_id, similarity_score.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT e1.id AS id1, e2.id AS id2,
                  similarity(e1.data->>'name', e2.data->>'name') AS sim
           FROM entities e1
           JOIN entities e2 ON e1.mission_id = e2.mission_id
             AND e1.entity_type = e2.entity_type
             AND e1.id < e2.id
             AND e1.is_golden = true AND e2.is_golden = true
             AND e1.data->>'name' IS NOT NULL
             AND e2.data->>'name' IS NOT NULL
           WHERE e1.mission_id = $1 AND e1.entity_type = $2
             AND similarity(e1.data->>'name', e2.data->>'name') > $3
           ORDER BY sim DESC
           LIMIT 500""",
        UUID(mission_id), entity_type, threshold,
    )
    return [{"id1": str(r["id1"]), "id2": str(r["id2"]), "similarity": float(r["sim"])} for r in rows]


async def recover_stuck_raw_ingests(mission_id: str, older_than_minutes: int = 60) -> int:
    """Reset 'processing' records stuck for too long back to 'pending'.

    Returns count of recovered records.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT id FROM raw_ingests
           WHERE mission_id = $1
             AND status = 'processing'
             AND processed_at < now() - interval '1 minute' * $2""",
        UUID(mission_id), older_than_minutes,
    )
    count = len(rows)
    if count > 0:
        await pool.execute(
            """UPDATE raw_ingests
               SET status = 'pending', processed_at = NULL
               WHERE mission_id = $1
                 AND status = 'processing'
                 AND processed_at < now() - interval '1 minute' * $2""",
            UUID(mission_id), older_than_minutes,
        )
    return count


async def merge_into_golden(golden_id: str, new_id: str, merged_data: dict):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE entities SET data = $1, updated_at = now() WHERE id = $2",
                merged_data, UUID(golden_id),
            )
            await conn.execute(
                "UPDATE entities SET golden_id = $1, is_golden = false WHERE id = $2",
                UUID(golden_id), UUID(new_id),
            )
            await conn.execute(
                """INSERT INTO entity_links (from_id, to_id, link_type, confidence)
                   VALUES ($1, $2, 'same_as', 1.0)""",
                UUID(new_id), UUID(golden_id),
            )


async def sample_recent_entities(mission_id: str, limit: int = 500) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT id, data, traits FROM entities
           WHERE mission_id = $1 AND is_golden = true
           ORDER BY updated_at DESC LIMIT $2""",
        UUID(mission_id), limit,
    )
    return [{"id": str(r["id"]), "data": r["data"],
             "traits": r["traits"] or {}} for r in rows]


async def get_facet_stats(mission_id: str) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT facet_key, facet_value, entity_count
           FROM facets WHERE mission_id = $1 AND entity_count > 0
           ORDER BY entity_count DESC LIMIT 500""",
        UUID(mission_id),
    )
    return [dict(r) for r in rows]


async def refresh_facet_catalog(mission_id: str):
    """Rebuild facet counts from entity traits."""
    pool = await get_pool()
    await pool.execute(
        """INSERT INTO facets (mission_id, entity_type, facet_key, facet_value, entity_count, last_updated)
           SELECT e.mission_id, e.entity_type, kv.key, kv.value::text, COUNT(*), now()
           FROM entities e, jsonb_each(e.traits) AS kv
           WHERE e.mission_id = $1 AND e.is_golden = true
           GROUP BY e.mission_id, e.entity_type, kv.key, kv.value::text
           ON CONFLICT (mission_id, entity_type, facet_key, facet_value)
           DO UPDATE SET entity_count = EXCLUDED.entity_count, last_updated = now()""",
        UUID(mission_id),
    )


async def upsert_group(mission_id: str, group: dict):
    pool = await get_pool()
    await pool.execute(
        """INSERT INTO groups (mission_id, entity_type, slug, facet_combo, title, description, member_count, seo_score)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
           ON CONFLICT (slug) DO UPDATE
           SET facet_combo = EXCLUDED.facet_combo, title = EXCLUDED.title,
               description = EXCLUDED.description, member_count = EXCLUDED.member_count,
               seo_score = EXCLUDED.seo_score, refreshed_at = now()""",
        UUID(mission_id), group.get("entity_type", "default"), group["slug"],
        group["facet_combination"], group["title"],
        group.get("description", ""), group.get("member_count", 0),
        group.get("seo_score", 0),
    )


async def apply_schema_update(mission_id: str, proposal: dict):
    pool = await get_pool()
    current = await get_active_schema(mission_id)
    row = await pool.fetchrow(
        "SELECT COALESCE(MAX(version), 0) AS v FROM schemas WHERE mission_id = $1",
        UUID(mission_id),
    )
    new_version = row["v"] + 1
    await pool.execute(
        """INSERT INTO schemas (mission_id, entity_type, version, json_schema, changelog, created_by)
           VALUES ($1, $2, $3, $4, $5, 'schema_architect')""",
        UUID(mission_id), proposal.get("entity_type", "default"), new_version,
        proposal["new_schema"], proposal.get("changelog", {}),
    )


async def ingest_raw(mission_id: str, source: str, payload: dict) -> str | None:
    """Insert a raw record, deduplicating by content hash. Returns ID or None if duplicate."""
    content_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    pool = await get_pool()
    try:
        row = await pool.fetchrow(
            """INSERT INTO raw_ingests (mission_id, source, raw_payload, content_hash)
               VALUES ($1, $2, $3, $4) RETURNING id""",
            UUID(mission_id), source, payload, content_hash,
        )
        return str(row["id"])
    except asyncpg.UniqueViolationError:
        return None
