"""
Grabber FastAPI application.

Lifespan handles: Postgres pool, Redis client, DB migrations, SyncService.
Orchestration is via Temporal (temporal_worker.py), not Redis Streams.
"""

from __future__ import annotations

import logging
import pathlib
import hashlib
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.client import get_pool, close_pool, get_cursor
from app.routers import api_keys, applications, dashboard, entities, jobs, patterns, projects, tasks
from app.services.sync_service import SyncService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: setup and teardown shared resources."""
    settings = get_settings()

    # --- Postgres ---
    logger.info("Initializing Postgres connection pool")
    get_pool()

    # --- Auto-migrate ---
    try:
        migration_dir = pathlib.Path(__file__).resolve().parent / "db" / "migrations"

        with get_cursor(commit=True) as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

        # Backfill: if tasks table exists but schema_migrations is empty
        with get_cursor(commit=True) as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM schema_migrations")
            if cur.fetchone()["cnt"] == 0:
                cur.execute("SELECT to_regclass('public.tasks')")
                if cur.fetchone()["to_regclass"] is not None:
                    cur.execute(
                        "INSERT INTO schema_migrations (filename) VALUES (%s)",
                        ("001_initial.sql",),
                    )
                    logger.info("Backfilled schema_migrations with 001_initial.sql")

        for sql_file in sorted(migration_dir.glob("*.sql")):
            with get_cursor(commit=True) as cur:
                cur.execute(
                    "SELECT 1 FROM schema_migrations WHERE filename = %s",
                    (sql_file.name,),
                )
                if cur.fetchone() is not None:
                    continue
                logger.info("Applying migration: %s", sql_file.name)
                cur.execute(sql_file.read_text())
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (sql_file.name,),
                )
                logger.info("Migration %s applied successfully", sql_file.name)
        app.state.schema_status = "ok"
    except Exception:
        logger.exception("Migration failed")
        app.state.schema_status = "error"
        raise RuntimeError("Database migration failed")

    # --- Redis (still used for pattern store cache) ---
    logger.info("Connecting to Redis at %s", settings.redis_url)
    r = aioredis.from_url(settings.redis_url, decode_responses=True, max_connections=20)
    app.state.redis = r

    try:
        await r.ping()
        logger.info("Redis connected")
    except Exception:
        logger.error("Redis connection failed - starting without Redis")

    # --- SyncService ---
    sync_service = SyncService(redis=r)
    app.state.sync_service = sync_service
    await sync_service.recover_processing()
    await sync_service.start()

    logger.info("Grabber API ready")
    yield

    # --- Shutdown ---
    logger.info("Shutting down Grabber API")
    await sync_service.stop()
    await r.aclose()
    close_pool()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="Grabber",
        description="Web scraping intelligence for arc-tools",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Routers
    app.include_router(jobs.router)
    app.include_router(jobs.types_router)
    app.include_router(api_keys.router)
    app.include_router(tasks.router)
    app.include_router(entities.router)
    app.include_router(patterns.router)
    app.include_router(dashboard.router)
    app.include_router(projects.router)
    app.include_router(applications.router)

    @app.get("/health")
    async def root_health() -> dict:
        checks = {"service": "grabber"}
        try:
            with get_cursor(commit=False) as cur:
                cur.execute("SELECT 1")
            checks["postgres"] = "ok"
        except Exception:
            checks["postgres"] = "error"
        try:
            await app.state.redis.ping()
            checks["redis"] = "ok"
        except Exception:
            checks["redis"] = "error"
        checks["schema_status"] = getattr(app.state, "schema_status", "unknown")
        checks["status"] = "ok" if checks.get("postgres") == "ok" and checks.get("redis") == "ok" else "degraded"
        return checks

    return app


app = create_app()
