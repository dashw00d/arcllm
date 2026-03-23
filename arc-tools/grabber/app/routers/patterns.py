"""
Pattern router: CRUD for extraction patterns and recipes.

GET    /api/patterns              - list with filters (domain, trust_level, quarantined)
GET    /api/patterns/domain/{d}   - patterns for a specific domain
DELETE /api/patterns/{id}         - delete a pattern

POST   /api/patterns/recipes      - create a recipe
GET    /api/patterns/recipes      - list recipes with filters
GET    /api/patterns/recipes/{id} - get a single recipe
PUT    /api/patterns/recipes/{id} - update a recipe
DELETE /api/patterns/recipes/{id} - delete a recipe
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.db.client import get_cursor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/patterns", tags=["patterns"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PatternDetail(BaseModel):
    id: str
    domain: str
    pattern_type: str
    config: dict[str, Any]
    trust_level: str
    confidence: float
    usage_count: int
    success_rate: float
    quarantined: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PatternListResponse(BaseModel):
    patterns: list[PatternDetail]
    count: int


class DeleteResponse(BaseModel):
    message: str
    id: str


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


def _row_to_pattern(row: dict) -> PatternDetail:
    return PatternDetail(
        id=str(row["id"]),
        domain=row["domain"],
        pattern_type=row["pattern_type"],
        config=row.get("config") or {},
        trust_level=row.get("trust_level", "provisional"),
        confidence=row.get("confidence", 0.0) or 0.0,
        usage_count=row.get("usage_count", 0) or 0,
        success_rate=row.get("success_rate", 0.0) or 0.0,
        quarantined=row.get("quarantined", False) or False,
        created_at=_ts(row.get("created_at")),
        updated_at=_ts(row.get("updated_at")),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=PatternListResponse)
async def list_patterns(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    trust_level: Optional[str] = Query(None, description="Filter by trust level"),
    quarantined: Optional[bool] = Query(None, description="Filter by quarantine status"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> PatternListResponse:
    """List patterns with optional filters."""
    conditions: list[str] = []
    params: list[Any] = []

    if domain:
        conditions.append("domain = %s")
        params.append(domain)
    if trust_level:
        conditions.append("trust_level = %s")
        params.append(trust_level)
    if quarantined is not None:
        conditions.append("quarantined = %s")
        params.append(quarantined)

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    query = f"SELECT * FROM patterns {where} ORDER BY updated_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_cursor(commit=False) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    patterns = [_row_to_pattern(row) for row in rows]
    return PatternListResponse(patterns=patterns, count=len(patterns))


@router.get("/domain/{domain}", response_model=PatternListResponse)
async def patterns_by_domain(domain: str) -> PatternListResponse:
    """Get all patterns for a specific domain."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT * FROM patterns WHERE domain = %s ORDER BY pattern_type",
            (domain,),
        )
        rows = cur.fetchall()

    patterns = [_row_to_pattern(row) for row in rows]
    return PatternListResponse(patterns=patterns, count=len(patterns))


@router.delete("/{pattern_id}", response_model=DeleteResponse)
async def delete_pattern(pattern_id: str) -> DeleteResponse:
    """Delete a pattern by ID."""
    try:
        uuid.UUID(pattern_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pattern ID format")

    with get_cursor(commit=True) as cur:
        cur.execute(
            "DELETE FROM patterns WHERE id = %s RETURNING id",
            (pattern_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Pattern not found")

    return DeleteResponse(message="Pattern deleted", id=str(row["id"]))


# ---------------------------------------------------------------------------
# Recipe schemas
# ---------------------------------------------------------------------------

class RecipeCreate(BaseModel):
    domain: str
    page_type: str = "default"
    author: str = ""
    description: str = ""
    recipe: dict[str, Any]


class RecipeDetail(BaseModel):
    id: str
    domain: str
    page_type: str
    author: str
    description: str
    recipe: dict[str, Any]
    trust_level: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RecipeListResponse(BaseModel):
    recipes: list[RecipeDetail]
    count: int


# ---------------------------------------------------------------------------
# Recipe helpers
# ---------------------------------------------------------------------------

def _row_to_recipe(row: dict) -> RecipeDetail:
    import json as _json
    recipe_val = row.get("recipe")
    if isinstance(recipe_val, str):
        recipe_val = _json.loads(recipe_val)
    return RecipeDetail(
        id=str(row["id"]),
        domain=row["domain"],
        page_type=row.get("page_type", "default") or "default",
        author=row.get("author", "") or "",
        description=row.get("description", "") or "",
        recipe=recipe_val or {},
        trust_level=row.get("trust_level", "provisional"),
        created_at=_ts(row.get("created_at")),
        updated_at=_ts(row.get("updated_at")),
    )


# ---------------------------------------------------------------------------
# Recipe endpoints
# ---------------------------------------------------------------------------

@router.post("/recipes", response_model=RecipeDetail, status_code=201)
async def create_recipe(body: RecipeCreate) -> RecipeDetail:
    """Create a new recipe (inserts as PROVISIONAL)."""
    import json as _json
    recipe_id = str(uuid.uuid4())

    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO patterns (id, domain, pattern_type, page_type, config,
                                  recipe, author, description, trust_level,
                                  confidence, usage_count, success_rate, quarantined)
            VALUES (%s, %s, 'recipe', %s, '{}'::jsonb,
                    %s::jsonb, %s, %s, 'provisional',
                    0.0, 0, 0.0, false)
            RETURNING *
            """,
            (
                recipe_id,
                body.domain,
                body.page_type,
                _json.dumps(body.recipe),
                body.author,
                body.description,
            ),
        )
        row = cur.fetchone()

    return _row_to_recipe(row)


@router.get("/recipes", response_model=RecipeListResponse)
async def list_recipes(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    page_type: Optional[str] = Query(None, description="Filter by page type"),
    author: Optional[str] = Query(None, description="Filter by author"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> RecipeListResponse:
    """List recipes with optional filters."""
    conditions: list[str] = ["pattern_type = 'recipe'"]
    params: list[Any] = []

    if domain:
        conditions.append("domain = %s")
        params.append(domain)
    if page_type:
        conditions.append("page_type = %s")
        params.append(page_type)
    if author:
        conditions.append("author = %s")
        params.append(author)

    where = "WHERE " + " AND ".join(conditions)
    query = f"SELECT * FROM patterns {where} ORDER BY updated_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_cursor(commit=False) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    recipes = [_row_to_recipe(row) for row in rows]
    return RecipeListResponse(recipes=recipes, count=len(recipes))


@router.get("/recipes/{recipe_id}", response_model=RecipeDetail)
async def get_recipe(recipe_id: str) -> RecipeDetail:
    """Get a single recipe by ID."""
    try:
        uuid.UUID(recipe_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid recipe ID format")

    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT * FROM patterns WHERE id = %s AND pattern_type = 'recipe'",
            (recipe_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Recipe not found")

    return _row_to_recipe(row)


@router.put("/recipes/{recipe_id}", response_model=RecipeDetail)
async def update_recipe(recipe_id: str, body: RecipeCreate) -> RecipeDetail:
    """Update a recipe (resets trust to PROVISIONAL)."""
    import json as _json

    try:
        uuid.UUID(recipe_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid recipe ID format")

    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE patterns
            SET domain = %s,
                page_type = %s,
                recipe = %s::jsonb,
                author = %s,
                description = %s,
                trust_level = 'provisional',
                confidence = 0.0,
                usage_count = 0,
                success_rate = 0.0,
                updated_at = NOW()
            WHERE id = %s AND pattern_type = 'recipe'
            RETURNING *
            """,
            (
                body.domain,
                body.page_type,
                _json.dumps(body.recipe),
                body.author,
                body.description,
                recipe_id,
            ),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Recipe not found")

    return _row_to_recipe(row)


@router.delete("/recipes/{recipe_id}", response_model=DeleteResponse)
async def delete_recipe(recipe_id: str) -> DeleteResponse:
    """Delete a recipe by ID."""
    try:
        uuid.UUID(recipe_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid recipe ID format")

    with get_cursor(commit=True) as cur:
        cur.execute(
            "DELETE FROM patterns WHERE id = %s AND pattern_type = 'recipe' RETURNING id",
            (recipe_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Recipe not found")

    return DeleteResponse(message="Recipe deleted", id=str(row["id"]))
