"""
Admin API key management.

All endpoints require an admin API key in X-API-Key.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.services import job_service

router = APIRouter(prefix="/api/auth/api-keys", tags=["auth"])


def _ts(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _admin_auth(request: Request) -> job_service.ApiAuthContext:
    auth = job_service.authenticate_api_key(request.headers.get("x-api-key"), required=True)
    job_service.require_admin(auth)
    return auth


class ApiKeyCreateRequest(BaseModel):
    name: str
    tenant_id: str = "public"
    is_admin: bool = False
    submit_rpm: int = Field(default=120, ge=1, le=100000)
    pending_limit: int = Field(default=10000, ge=1, le=10000000)
    payload_bytes_limit: int = Field(default=262144, ge=1024, le=10485760)


class ApiKeyDetail(BaseModel):
    id: str
    tenant_id: str
    name: str
    is_active: bool
    is_admin: bool
    submit_rpm: int
    pending_limit: int
    payload_bytes_limit: int
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None


class ApiKeyCreateResponse(ApiKeyDetail):
    api_key: str


class ApiKeyListResponse(BaseModel):
    keys: list[ApiKeyDetail]
    count: int


def _row_to_detail(row: dict) -> ApiKeyDetail:
    return ApiKeyDetail(
        id=str(row["id"]),
        tenant_id=row["tenant_id"],
        name=row["name"],
        is_active=bool(row.get("is_active", True)),
        is_admin=bool(row.get("is_admin", False)),
        submit_rpm=int(row.get("submit_rpm", 120)),
        pending_limit=int(row.get("pending_limit", 10000)),
        payload_bytes_limit=int(row.get("payload_bytes_limit", 262144)),
        created_at=_ts(row.get("created_at")),
        last_used_at=_ts(row.get("last_used_at")),
    )


@router.get("", response_model=ApiKeyListResponse)
async def list_keys(request: Request, tenant_id: Optional[str] = Query(None)) -> ApiKeyListResponse:
    auth = _admin_auth(request)
    rows = job_service.list_api_keys(auth, tenant_id=tenant_id)
    keys = [_row_to_detail(row) for row in rows]
    return ApiKeyListResponse(keys=keys, count=len(keys))


@router.post("", response_model=ApiKeyCreateResponse, status_code=201)
async def create_key(body: ApiKeyCreateRequest, request: Request) -> ApiKeyCreateResponse:
    auth = _admin_auth(request)
    row = job_service.create_api_key(
        auth=auth,
        name=body.name,
        tenant_id=body.tenant_id,
        is_admin=body.is_admin,
        submit_rpm=body.submit_rpm,
        pending_limit=body.pending_limit,
        payload_bytes_limit=body.payload_bytes_limit,
    )
    detail = _row_to_detail(row)
    return ApiKeyCreateResponse(**detail.__dict__, api_key=row["api_key"])


@router.post("/{key_id}/rotate", response_model=ApiKeyCreateResponse)
async def rotate_key(key_id: str, request: Request) -> ApiKeyCreateResponse:
    try:
        uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid API key ID format")
    auth = _admin_auth(request)
    row = job_service.rotate_api_key(auth, key_id)
    if not row:
        raise HTTPException(status_code=404, detail="API key not found")
    detail = _row_to_detail(row)
    return ApiKeyCreateResponse(**detail.__dict__, api_key=row["api_key"])


@router.delete("/{key_id}")
async def deactivate_key(key_id: str, request: Request) -> dict:
    try:
        uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid API key ID format")
    auth = _admin_auth(request)
    row = job_service.deactivate_api_key(auth, key_id)
    if not row:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"id": str(row["id"]), "is_active": bool(row["is_active"])}
