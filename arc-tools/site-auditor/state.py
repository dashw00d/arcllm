"""
Audit state manager — tracks sites, audits, scans, and patterns in Postgres.

Every action is recorded. Nothing is repeated unless forced.
Pipeline can resume from any stage after a crash.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

DB_URL = os.environ.get("POSTGRES_URL", "postgresql://temporal:temporal@localhost:5432/ghostgraph")


def _conn():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    return conn


def _cur(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


# ── Sites ─────────────────────────────────────────────────────────────────

def get_or_create_site(domain: str) -> dict:
    """Get existing site or create new one."""
    conn = _conn()
    cur = _cur(conn)
    cur.execute("SELECT * FROM sites WHERE domain = %s", (domain,))
    site = cur.fetchone()
    if site:
        conn.close()
        return dict(site)

    cur.execute(
        "INSERT INTO sites (domain, status) VALUES (%s, 'new') RETURNING *",
        (domain,),
    )
    site = dict(cur.fetchone())
    conn.close()
    logger.info("Created site: %s", domain)
    return site


def update_site(site_id: str, **kwargs) -> None:
    """Update site fields."""
    conn = _conn()
    cur = _cur(conn)
    sets = ", ".join(f"{k} = %s" for k in kwargs)
    vals = list(kwargs.values()) + [site_id]
    cur.execute(f"UPDATE sites SET {sets} WHERE id = %s", vals)
    conn.close()


# ── Entity Audits ─────────────────────────────────────────────────────────

def get_or_create_audit(site_id: str, entity_type: str, description: str = "") -> dict:
    """Get existing audit or create new one for this entity type on this site."""
    conn = _conn()
    cur = _cur(conn)
    cur.execute(
        "SELECT * FROM entity_audits WHERE site_id = %s AND entity_type = %s",
        (site_id, entity_type),
    )
    audit = cur.fetchone()
    if audit:
        conn.close()
        return dict(audit)

    cur.execute(
        """INSERT INTO entity_audits (site_id, entity_type, description, stage)
           VALUES (%s, %s, %s, 'pending') RETURNING *""",
        (site_id, entity_type, description),
    )
    audit = dict(cur.fetchone())
    conn.close()
    logger.info("Created audit: %s / %s", site_id, entity_type)
    return audit


def update_audit(audit_id: str, **kwargs) -> None:
    """Update audit fields + updated_at."""
    conn = _conn()
    cur = _cur(conn)
    kwargs["updated_at"] = datetime.now(timezone.utc)
    sets = ", ".join(f"{k} = %s" for k in kwargs)
    vals = list(kwargs.values()) + [audit_id]
    cur.execute(f"UPDATE entity_audits SET {sets} WHERE id = %s", vals)
    conn.close()


def advance_stage(audit_id: str, new_stage: str) -> None:
    """Move audit to the next pipeline stage."""
    update_audit(audit_id, stage=new_stage, last_stage_at=datetime.now(timezone.utc))
    logger.info("Audit %s → stage: %s", audit_id, new_stage)


# ── Page Scans ────────────────────────────────────────────────────────────

def was_scanned(audit_id: str, url: str, stage: str) -> bool:
    """Check if this URL was already scanned in this stage."""
    conn = _conn()
    cur = _cur(conn)
    cur.execute(
        "SELECT 1 FROM page_scans WHERE audit_id = %s AND url = %s AND stage = %s",
        (audit_id, url, stage),
    )
    result = cur.fetchone() is not None
    conn.close()
    return result


def record_scan(audit_id: str, url: str, path: str, stage: str, **kwargs) -> str:
    """Record a page scan. Returns scan ID."""
    conn = _conn()
    cur = _cur(conn)
    scan_id = str(uuid4())
    fields = ["id", "audit_id", "url", "path", "stage"] + list(kwargs.keys())
    placeholders = ["%s"] * len(fields)
    vals = [scan_id, audit_id, url, path, stage] + list(kwargs.values())
    cur.execute(
        f"INSERT INTO page_scans ({', '.join(fields)}) VALUES ({', '.join(placeholders)}) "
        f"ON CONFLICT (audit_id, url, stage) DO UPDATE SET "
        + ", ".join(f"{k} = EXCLUDED.{k}" for k in kwargs.keys())
        + " RETURNING id",
        vals,
    )
    result = cur.fetchone()["id"]
    conn.close()

    # Update scan count
    conn = _conn()
    cur = _cur(conn)
    cur.execute("SELECT COUNT(*) as cnt FROM page_scans WHERE audit_id = %s", (audit_id,))
    count = cur.fetchone()["cnt"]
    cur.execute("UPDATE entity_audits SET pages_scanned = %s WHERE id = %s", (count, audit_id))
    conn.close()

    return str(result)


def get_scans(audit_id: str, stage: str = None) -> list[dict]:
    """Get all scans for an audit, optionally filtered by stage."""
    conn = _conn()
    cur = _cur(conn)
    if stage:
        cur.execute("SELECT * FROM page_scans WHERE audit_id = %s AND stage = %s ORDER BY scanned_at", (audit_id, stage))
    else:
        cur.execute("SELECT * FROM page_scans WHERE audit_id = %s ORDER BY scanned_at", (audit_id,))
    results = [dict(r) for r in cur.fetchall()]
    conn.close()
    return results


# ── URL Patterns ──────────────────────────────────────────────────────────

def save_url_pattern(audit_id: str, pattern_type: str, url_pattern: str, **kwargs) -> str:
    """Save a discovered URL pattern."""
    conn = _conn()
    cur = _cur(conn)
    pattern_id = str(uuid4())
    fields = ["id", "audit_id", "pattern_type", "url_pattern"] + list(kwargs.keys())
    placeholders = ["%s"] * len(fields)
    vals = [pattern_id, audit_id, pattern_type, url_pattern] + list(kwargs.values())
    cur.execute(
        f"INSERT INTO url_patterns ({', '.join(fields)}) VALUES ({', '.join(placeholders)}) RETURNING id",
        vals,
    )
    result = cur.fetchone()["id"]
    conn.close()
    return str(result)


def get_url_patterns(audit_id: str, pattern_type: str = None) -> list[dict]:
    """Get URL patterns for an audit."""
    conn = _conn()
    cur = _cur(conn)
    if pattern_type:
        cur.execute("SELECT * FROM url_patterns WHERE audit_id = %s AND pattern_type = %s", (audit_id, pattern_type))
    else:
        cur.execute("SELECT * FROM url_patterns WHERE audit_id = %s", (audit_id,))
    results = [dict(r) for r in cur.fetchall()]
    conn.close()
    return results


# ── DOM Patterns ──────────────────────────────────────────────────────────

def save_dom_pattern(audit_id: str, page_type: str, **kwargs) -> str:
    """Save a DOM extraction pattern."""
    conn = _conn()
    cur = _cur(conn)
    pattern_id = str(uuid4())
    # Convert chunks to JSON if it's a list
    if "chunks" in kwargs and isinstance(kwargs["chunks"], list):
        kwargs["chunks"] = json.dumps(kwargs["chunks"])
    fields = ["id", "audit_id", "page_type"] + list(kwargs.keys())
    placeholders = ["%s"] * len(fields)
    vals = [pattern_id, audit_id, page_type] + list(kwargs.values())
    cur.execute(
        f"INSERT INTO dom_patterns ({', '.join(fields)}) VALUES ({', '.join(placeholders)}) RETURNING id",
        vals,
    )
    result = cur.fetchone()["id"]
    conn.close()
    return str(result)


def get_dom_patterns(audit_id: str) -> list[dict]:
    """Get DOM patterns for an audit."""
    conn = _conn()
    cur = _cur(conn)
    cur.execute("SELECT * FROM dom_patterns WHERE audit_id = %s", (audit_id,))
    results = [dict(r) for r in cur.fetchall()]
    conn.close()
    return results


# ── Resume helpers ────────────────────────────────────────────────────────

def get_resumable_audit(domain: str, entity_type: str) -> dict | None:
    """Find an incomplete audit to resume."""
    conn = _conn()
    cur = _cur(conn)
    cur.execute("""
        SELECT ea.*, s.domain FROM entity_audits ea
        JOIN sites s ON s.id = ea.site_id
        WHERE s.domain = %s AND ea.entity_type = %s AND ea.stage NOT IN ('complete', 'not_relevant')
        ORDER BY ea.updated_at DESC LIMIT 1
    """, (domain, entity_type))
    result = cur.fetchone()
    conn.close()
    return dict(result) if result else None


def init_schema():
    """Apply the schema if tables don't exist."""
    import pathlib
    schema_file = pathlib.Path(__file__).parent / "schema.sql"
    if not schema_file.exists():
        logger.error("schema.sql not found")
        return

    conn = _conn()
    cur = _cur(conn)
    try:
        cur.execute("SELECT 1 FROM sites LIMIT 0")
        logger.debug("Auditor schema already exists")
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        cur.execute(schema_file.read_text())
        logger.info("Applied auditor schema")
    conn.close()
