"""
Entity query router: list, filter, search, merge, dedup.

GET  /api/entities              - list with filters (entity_type, source_domain, status)
GET  /api/entities/stats        - counts by domain and type
GET  /api/entities/{id}         - single entity detail
POST /api/entities/search       - text search across entity data
POST /api/entities/merge        - merge duplicate entities into one
POST /api/entities/find-duplicates - find potential duplicate entity groups
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from itertools import combinations
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.db.client import get_cursor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/entities", tags=["entities"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class EntityDetail(BaseModel):
    id: str
    entity_type: str
    project_id: Optional[str] = None
    data: dict[str, Any]
    meta: dict[str, Any]
    source_type: Optional[str] = None
    source_ref: Optional[str] = None
    source_domain: Optional[str] = None
    content_hash: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class EntityListResponse(BaseModel):
    entities: list[EntityDetail]
    count: int


class DomainStat(BaseModel):
    source_domain: Optional[str] = None
    count: int


class TypeStat(BaseModel):
    entity_type: str
    count: int


class EntityStatsResponse(BaseModel):
    total: int
    by_domain: list[DomainStat]
    by_type: list[TypeStat]


class EntitySearchRequest(BaseModel):
    query: str
    entity_type: Optional[str] = None
    source_domain: Optional[str] = None
    limit: int = 50


class MergeRequest(BaseModel):
    entity_ids: list[str]
    primary_id: Optional[str] = None


class FindDuplicatesRequest(BaseModel):
    project_id: Optional[str] = None
    entity_type: Optional[str] = None
    match_fields: list[str]
    threshold: float = 0.8


class DuplicateGroup(BaseModel):
    entities: list[EntityDetail]
    confidence: float


class DuplicateGroupsResponse(BaseModel):
    groups: list[DuplicateGroup]
    count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    from datetime import datetime
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _normalize(s: str) -> str:
    """Normalize a string for comparison: lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _field_similarity(a: Any, b: Any) -> float:
    """Simple string similarity between two field values.

    Returns 1.0 for exact match (after normalization), 0.5 for containment,
    0.0 otherwise. Non-string values are compared via JSON repr.
    """
    if a is None or b is None:
        return 0.0
    sa = _normalize(str(a) if not isinstance(a, str) else a)
    sb = _normalize(str(b) if not isinstance(b, str) else b)
    if sa == sb:
        return 1.0
    if sa in sb or sb in sa:
        return 0.5
    return 0.0


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep merge overlay into base. Lists are unioned, dicts are recursed,
    scalars from base take precedence (base = primary entity)."""
    merged = dict(base)
    for key, val in overlay.items():
        if key not in merged:
            merged[key] = val
        elif isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        elif isinstance(merged[key], list) and isinstance(val, list):
            # Union: preserve order, add new items
            seen = {json.dumps(v, sort_keys=True, default=str) for v in merged[key]}
            for item in val:
                key_repr = json.dumps(item, sort_keys=True, default=str)
                if key_repr not in seen:
                    merged[key].append(item)
                    seen.add(key_repr)
        # else: keep base value (primary wins)
    return merged


def _row_to_entity(row: dict) -> EntityDetail:
    return EntityDetail(
        id=str(row["id"]),
        entity_type=row["entity_type"],
        project_id=str(row["project_id"]) if row.get("project_id") else None,
        data=row.get("data") or {},
        meta=row.get("meta") or {},
        source_type=row.get("source_type"),
        source_ref=row.get("source_ref"),
        source_domain=row.get("source_domain"),
        content_hash=row.get("content_hash", ""),
        status=row.get("status", "active"),
        created_at=_ts(row.get("created_at")),
        updated_at=_ts(row.get("updated_at")),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/stats", response_model=EntityStatsResponse)
async def entity_stats() -> EntityStatsResponse:
    """Aggregate counts by domain and entity type."""
    with get_cursor(commit=False) as cur:
        cur.execute("SELECT COUNT(*) AS total FROM entities")
        total_row = cur.fetchone()
        total = total_row["total"] if total_row else 0

        cur.execute(
            """
            SELECT source_domain, COUNT(*) AS count
            FROM entities
            GROUP BY source_domain
            ORDER BY count DESC
            LIMIT 50
            """
        )
        domain_rows = cur.fetchall()

        cur.execute(
            """
            SELECT entity_type, COUNT(*) AS count
            FROM entities
            GROUP BY entity_type
            ORDER BY count DESC
            LIMIT 50
            """
        )
        type_rows = cur.fetchall()

    return EntityStatsResponse(
        total=total,
        by_domain=[
            DomainStat(source_domain=r["source_domain"], count=r["count"])
            for r in domain_rows
        ],
        by_type=[
            TypeStat(entity_type=r["entity_type"], count=r["count"])
            for r in type_rows
        ],
    )


@router.post("/search", response_model=EntityListResponse)
async def search_entities(body: EntitySearchRequest) -> EntityListResponse:
    """Search entities by text match on JSONB data.

    # TODO: Upgrade to pgvector semantic similarity search when embeddings
    # are available. Current implementation uses ILIKE text search as fallback.
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    conditions = ["data::text ILIKE %s"]
    params: list[Any] = [f"%{body.query}%"]

    if body.entity_type:
        conditions.append("entity_type = %s")
        params.append(body.entity_type)
    if body.source_domain:
        conditions.append("source_domain = %s")
        params.append(body.source_domain)

    limit = max(1, min(body.limit, 500))
    where = "WHERE " + " AND ".join(conditions)
    query = f"SELECT * FROM entities {where} ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    with get_cursor(commit=False) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    entities = [_row_to_entity(row) for row in rows]
    return EntityListResponse(entities=entities, count=len(entities))


@router.post("/merge", response_model=EntityDetail)
async def merge_entities(body: MergeRequest) -> EntityDetail:
    """Merge multiple entities into one primary entity.

    Deep-merges JSONB data fields, tracks provenance in meta, and marks
    merged entities with status 'merged'.
    """
    if len(body.entity_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 entity IDs required")

    # Validate all IDs are UUIDs
    for eid in body.entity_ids:
        try:
            uuid.UUID(eid)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid entity ID: {eid}")

    if body.primary_id and body.primary_id not in body.entity_ids:
        raise HTTPException(
            status_code=400,
            detail="primary_id must be one of the entity_ids",
        )

    with get_cursor(commit=True) as cur:
        # Fetch all entities
        placeholders = ",".join(["%s"] * len(body.entity_ids))
        cur.execute(
            f"SELECT * FROM entities WHERE id IN ({placeholders}) AND status != 'merged'",
            body.entity_ids,
        )
        rows = cur.fetchall()

        if len(rows) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 non-merged entities to merge",
            )

        entity_map = {str(r["id"]): r for r in rows}

        # Pick the primary: explicit or the one with the most data fields
        if body.primary_id and body.primary_id in entity_map:
            primary_id = body.primary_id
        else:
            primary_id = max(
                entity_map,
                key=lambda eid: len(entity_map[eid].get("data") or {}),
            )

        primary = entity_map[primary_id]
        others = [entity_map[eid] for eid in entity_map if eid != primary_id]
        other_ids = [str(o["id"]) for o in others]

        # Deep merge data from others into primary
        merged_data = dict(primary.get("data") or {})
        for other in others:
            merged_data = _deep_merge(merged_data, other.get("data") or {})

        # Build updated meta with provenance
        merged_meta = dict(primary.get("meta") or {})
        merged_meta["merged_from"] = other_ids

        # Recalculate content_hash for the merged data
        content_hash = hashlib.sha256(
            json.dumps(merged_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Update the primary entity
        cur.execute(
            """
            UPDATE entities
            SET data = %s, meta = %s, content_hash = %s, updated_at = NOW()
            WHERE id = %s
            RETURNING *
            """,
            (json.dumps(merged_data), json.dumps(merged_meta), content_hash, primary_id),
        )
        updated_row = cur.fetchone()

        # Mark other entities as merged
        for oid in other_ids:
            other_meta = dict(entity_map[oid].get("meta") or {})
            other_meta["merged_into"] = primary_id
            cur.execute(
                """
                UPDATE entities
                SET status = 'merged', meta = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (json.dumps(other_meta), oid),
            )

    return _row_to_entity(updated_row)


@router.post("/find-duplicates", response_model=DuplicateGroupsResponse)
async def find_duplicates(body: FindDuplicatesRequest) -> DuplicateGroupsResponse:
    """Find potential duplicate entities by comparing specified data fields.

    Uses simple normalized string comparison (not fuzzy matching).
    Returns groups of potential duplicates for an AI agent to review.
    """
    if not body.match_fields:
        raise HTTPException(status_code=400, detail="match_fields must not be empty")

    conditions = ["status != 'merged'"]
    params: list[Any] = []

    if body.project_id:
        conditions.append("project_id = %s")
        params.append(body.project_id)
    if body.entity_type:
        conditions.append("entity_type = %s")
        params.append(body.entity_type)

    where = "WHERE " + " AND ".join(conditions)
    # Cap at 1000 entities to keep pairwise comparison tractable
    query = f"SELECT * FROM entities {where} ORDER BY created_at DESC LIMIT 1000"

    with get_cursor(commit=False) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    if len(rows) < 2:
        return DuplicateGroupsResponse(groups=[], count=0)

    # Build fingerprints for comparison
    entities = [(r, {f: (r.get("data") or {}).get(f) for f in body.match_fields}) for r in rows]

    # Find pairs that exceed threshold
    # Use union-find to group transitive duplicates
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pair_confidence: dict[tuple[str, str], float] = {}

    for (row_a, fields_a), (row_b, fields_b) in combinations(entities, 2):
        id_a, id_b = str(row_a["id"]), str(row_b["id"])
        # Calculate average similarity across match_fields
        sims = []
        for f in body.match_fields:
            val_a, val_b = fields_a.get(f), fields_b.get(f)
            if val_a is not None and val_b is not None:
                sims.append(_field_similarity(val_a, val_b))
        if not sims:
            continue
        avg_sim = sum(sims) / len(sims)
        if avg_sim >= body.threshold:
            union(id_a, id_b)
            key = (min(id_a, id_b), max(id_a, id_b))
            pair_confidence[key] = avg_sim

    # Collect groups
    groups_map: dict[str, list[dict]] = {}
    for row, _ in entities:
        eid = str(row["id"])
        root = find(eid)
        if root != eid or parent.get(eid, eid) != eid:
            groups_map.setdefault(find(eid), []).append(row)

    # Build response groups with confidence
    result_groups: list[DuplicateGroup] = []
    for root, members in groups_map.items():
        if len(members) < 2:
            continue
        # Group confidence = average of all pair confidences in the group
        member_ids = [str(m["id"]) for m in members]
        confs = []
        for a, b in combinations(member_ids, 2):
            key = (min(a, b), max(a, b))
            if key in pair_confidence:
                confs.append(pair_confidence[key])
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        result_groups.append(
            DuplicateGroup(
                entities=[_row_to_entity(m) for m in members],
                confidence=round(avg_conf, 3),
            )
        )

    # Sort by confidence desc, limit to 100
    result_groups.sort(key=lambda g: g.confidence, reverse=True)
    result_groups = result_groups[:100]

    return DuplicateGroupsResponse(groups=result_groups, count=len(result_groups))


@router.get("/{entity_id}", response_model=EntityDetail)
async def get_entity(entity_id: str) -> EntityDetail:
    """Get a single entity by ID."""
    try:
        uuid.UUID(entity_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid entity ID format")

    with get_cursor(commit=False) as cur:
        cur.execute("SELECT * FROM entities WHERE id = %s", (entity_id,))
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    return _row_to_entity(row)


@router.get("", response_model=EntityListResponse)
async def list_entities(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    source_domain: Optional[str] = Query(None, description="Filter by source domain"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> EntityListResponse:
    """List entities with optional filters."""
    conditions: list[str] = []
    params: list[Any] = []

    if entity_type:
        conditions.append("entity_type = %s")
        params.append(entity_type)
    if source_domain:
        conditions.append("source_domain = %s")
        params.append(source_domain)
    if status:
        conditions.append("status = %s")
        params.append(status)

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    query = f"SELECT * FROM entities {where} ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_cursor(commit=False) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    entities = [_row_to_entity(row) for row in rows]
    return EntityListResponse(entities=entities, count=len(entities))
