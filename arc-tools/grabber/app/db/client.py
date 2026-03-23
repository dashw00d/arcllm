"""
Postgres connection pool for Grabber.

Uses psycopg2 ThreadedConnectionPool (min 1, max 3 per worker).
Includes health check that recreates the pool on dead connections.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_pool: Optional[ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def _create_pool() -> ThreadedConnectionPool:
    settings = get_settings()
    return ThreadedConnectionPool(
        minconn=settings.db_pool_min,
        maxconn=settings.db_pool_max,
        dsn=settings.postgres_url,
    )


def get_pool() -> ThreadedConnectionPool:
    """Return the global connection pool, creating it if needed."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = _create_pool()
                logger.info("Database connection pool created")
    return _pool


def reset_pool() -> None:
    """Close and recreate the connection pool."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.closeall()
            except Exception:
                pass
        _pool = _create_pool()
        logger.info("Database connection pool reset")


def _health_check(conn: Any) -> bool:
    """Return True if the connection is alive."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


@contextmanager
def get_connection() -> Iterator[Any]:
    """
    Yield a connection from the pool.

    Automatically returns the connection on exit.
    If the connection is dead, resets the pool and retries once.
    """
    pool = get_pool()
    conn = None
    try:
        conn = pool.getconn()
        if not _health_check(conn):
            logger.warning("Dead connection detected, resetting pool")
            try:
                pool.putconn(conn, close=True)
            except Exception:
                pass
            reset_pool()
            pool = get_pool()
            conn = pool.getconn()
        yield conn
    except psycopg2.OperationalError:
        logger.warning("Operational error, resetting pool")
        if conn is not None:
            try:
                pool.putconn(conn, close=True)
            except Exception:
                pass
            conn = None
        reset_pool()
        raise
    finally:
        if conn is not None:
            try:
                pool.putconn(conn)
            except Exception:
                pass


@contextmanager
def get_cursor(commit: bool = True) -> Iterator[psycopg2.extras.RealDictCursor]:
    """
    Yield a RealDictCursor from a pooled connection.

    If commit=True (default), auto-commits on success and rolls back on error.
    """
    with get_connection() as conn:
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
                if commit:
                    conn.commit()
        except Exception:
            conn.rollback()
            raise


def close_pool() -> None:
    """Shut down the connection pool."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.closeall()
            except Exception:
                pass
            _pool = None
            logger.info("Database connection pool closed")
