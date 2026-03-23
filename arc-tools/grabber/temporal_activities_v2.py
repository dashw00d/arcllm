"""
Temporal activities v2 — pattern-based grab workflow.

ZERO LLM calls. Pure CSS selector + HTTP extraction.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import psycopg2
import psycopg2.extras
from bs4 import BeautifulSoup
from temporalio import activity

from app.core.config import get_settings
from app.db.client import get_cursor

logger = logging.getLogger(__name__)

# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class AuditInfo:
    audit_id: str
    site_id: str
    domain: str
    entity_type: str
    url_patterns: list[dict]
    dom_patterns: list[dict]


@dataclass
class FetchCompletedResult:
    audits: list[AuditInfo]
    count: int


@dataclass
class ExpandResult:
    detail_urls: list[str]
    errors: list[str]


@dataclass
class GrabResult:
    chunks: list[dict]
    success_count: int
    fail_count: int


# ── Helpers ─────────────────────────────────────────────────────────────

def _ghostgraph_cursor():
    """Yield a cursor on the ghostgraph DB."""
    with get_cursor(commit=True) as cur:
        yield cur


def _churner_conn():
    """Connect directly to the churner DB."""
    settings = get_settings()
    churner_url = settings.postgres_url.rsplit("/", 1)[0] + "/churner"
    return psycopg2.connect(churner_url)


async def _http_get(session: Any, url: str, timeout: int = 20) -> tuple[bytes | None, int, str]:
    """
    Fetch a URL, return (html_bytes, status_code, error).
    """
    try:
        response = await session.get(url, timeout=timeout)
        return response.content, response.status_code, ""
    except Exception as e:
        return None, 0, str(e)


def _rate_limit(domain: str, last_request: dict) -> float:
    """Return seconds to sleep before next request to domain (1 req/sec)."""
    now = time.monotonic()
    last = last_request.get(domain, 0)
    elapsed = now - last
    if elapsed < 1.0:
        return 1.0 - elapsed
    return 0.0


# ── Activity 1: fetch_completed_audits ─────────────────────────────────

@activity.defn
async def fetch_completed_audits(entity_type: str | None = None) -> FetchCompletedResult:
    """
    Query ghostgraph.entity_audits for audits where stage='complete' and grab_status='pending'.
    Returns list of audit records with their url_patterns and dom_patterns.
    """
    activity.logger.info("Fetching completed audits (entity_type=%s)", entity_type or "all")

    settings = get_settings()
    conn = psycopg2.connect(settings.postgres_url)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    if entity_type:
        cur.execute(
            """
            SELECT ea.id as audit_id, ea.site_id, s.domain, ea.entity_type,
                   ea.grab_status
            FROM entity_audits ea
            JOIN sites s ON s.id = ea.site_id
            WHERE ea.stage = 'complete'
              AND (ea.grab_status = 'pending' OR ea.grab_status IS NULL)
              AND ea.entity_type = %s
            ORDER BY ea.updated_at ASC
            LIMIT 50
            """,
            (entity_type,),
        )
    else:
        cur.execute(
            """
            SELECT ea.id as audit_id, ea.site_id, s.domain, ea.entity_type,
                   ea.grab_status
            FROM entity_audits ea
            JOIN sites s ON s.id = ea.site_id
            WHERE ea.stage = 'complete'
              AND (ea.grab_status = 'pending' OR ea.grab_status IS NULL)
            ORDER BY ea.updated_at ASC
            LIMIT 50
            """,
        )

    audit_rows = cur.fetchall()
    cur.close()
    conn.close()

    if not audit_rows:
        return FetchCompletedResult(audits=[], count=0)

    # Re-connect to fetch patterns for each audit
    conn2 = psycopg2.connect(settings.postgres_url)
    conn2.autocommit = True
    cur2 = conn2.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    audits = []
    for row in audit_rows:
        audit_id = str(row["audit_id"])
        site_id = str(row["site_id"])

        # Fetch url_patterns
        cur2.execute(
            "SELECT * FROM url_patterns WHERE audit_id = %s",
            (row["audit_id"],),
        )
        url_patterns = [dict(r) for r in cur2.fetchall()]

        # Fetch dom_patterns
        cur2.execute(
            "SELECT * FROM dom_patterns WHERE audit_id = %s",
            (row["audit_id"],),
        )
        dom_patterns = [dict(r) for r in cur2.fetchall()]

        # Mark as grabbed
        cur2.execute(
            "UPDATE entity_audits SET grab_status = 'grabbing' WHERE id = %s",
            (row["audit_id"],),
        )

        audits.append(AuditInfo(
            audit_id=audit_id,
            site_id=site_id,
            domain=row["domain"],
            entity_type=row["entity_type"],
            url_patterns=url_patterns,
            dom_patterns=dom_patterns,
        ))

    cur2.close()
    conn2.close()

    activity.logger.info("Found %d completed audits to grab", len(audits))
    return FetchCompletedResult(audits=audits, count=len(audits))


# ── Activity 2: expand_urls ─────────────────────────────────────────────

async def _build_stealth_session():
    """Build a StealthSession (simplified — no Redis needed for basic HTTP)."""
    from vendor.transport import StealthSession
    import redis.asyncio as aioredis

    settings = get_settings()
    redis_client = await aioredis.from_url(settings.redis_url, decode_responses=True)
    session = StealthSession(redis_client=redis_client)
    return session


def _expand_url_pattern(base_url: str, pattern: str, max_pages: int | None) -> list[str]:
    """
    Expand a URL pattern into concrete URLs.
    Handles:
    - /venues?page={N}  → /venues?page=1, /venues?page=2, ...
    - next_link (deferred — caller handles via page fetch)
    - load_more / infinite scroll (deferred)
    """
    urls = []

    if "{N}" in pattern or re.search(r"\{(\d+)\}", pattern):
        max_p = max_pages or 50
        for i in range(1, max_p + 1):
            url = re.sub(r"\{N\}", str(i), pattern)
            url = re.sub(r"\{(\d+)\}", str(i), url)
            url = urljoin(base_url, url)
            urls.append(url)
    else:
        urls.append(urljoin(base_url, pattern))

    return urls


async def _fetch_and_extract_links(
    session: Any,
    url: str,
    entity_selector: str | None,
    link_selector: str | None,
) -> tuple[list[str], str]:
    """
    Fetch a page and extract detail page URLs using CSS selectors.
    Returns (list of detail URLs, error string).
    """
    html, status, err = await _http_get(session, url)
    if err or status >= 400:
        return [], err or f"HTTP {status}"

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        return [], f"parse error: {e}"

    detail_urls = []

    # Extract entity container + link
    if entity_selector:
        containers = soup.select(entity_selector)
        sel = link_selector or "a[href]"
        for container in containers:
            links = container.select(sel)
            for a in links:
                href = a.get("href")
                if href:
                    detail_urls.append(urljoin(url, href))
    elif link_selector:
        links = soup.select(link_selector)
        for a in links:
            href = a.get("href")
            if href:
                detail_urls.append(urljoin(url, href))

    # Deduplicate
    seen = set()
    deduped = []
    for u in detail_urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    return deduped, ""


@activity.defn
async def expand_urls(audit_id: str, domain: str, url_patterns: list[dict]) -> ExpandResult:
    """
    For each url_pattern in audit's url_patterns:
    - Expand pagination patterns
    - Fetch index pages
    - Extract detail page URLs via entity_selector + link_selector from dom_patterns
    """
    activity.logger.info("Expanding URLs for audit %s (domain=%s)", audit_id, domain)

    # Build a session — use simple curl_cffi directly to avoid Redis dependency
    from curl_cffi.requests import AsyncSession
    from vendor.transport import PlatformProfile

    profile = PlatformProfile()
    session = AsyncSession(impersonate=profile.impersonate_profile)

    all_detail_urls = []
    errors = []
    last_request_time = {}

    for up in url_patterns:
        pattern_type = up.get("pattern_type", "index")
        url_pattern = up.get("url_pattern", "")
        pagination_type = up.get("pagination_type", "none")
        pagination_selector = up.get("pagination_selector")
        max_pages = up.get("max_pages")

        if not url_pattern:
            continue

        # Build base URL
        base = f"https://{domain}"
        seed_url = urljoin(base, url_pattern)

        if pagination_type == "none":
            urls_to_fetch = [seed_url]
        else:
            urls_to_fetch = _expand_url_pattern(base, url_pattern, max_pages)

        # Respect rate limit
        await asyncio.sleep(_rate_limit(domain, last_request_time))
        last_request_time[domain] = time.monotonic()

        # Fetch first page to get entity+link selectors
        first_urls = urls_to_fetch[:3]  # sample a few pages

        for page_url in first_urls:
            activity.heartbeat()
            links, err = await _fetch_and_extract_links(
                session, page_url,
                entity_selector=None,  # will be passed via dom_patterns
                link_selector=pagination_selector,
            )
            if err:
                errors.append(f"{page_url}: {err}")
                continue
            all_detail_urls.extend(links)

    await session.close()

    # Deduplicate
    seen = set()
    deduped = []
    for u in all_detail_urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    activity.logger.info(
        "Expanded %d detail URLs for audit %s (%d errors)",
        len(deduped), audit_id, len(errors),
    )
    return ExpandResult(detail_urls=deduped, errors=errors)


# ── Activity 3: grab_with_patterns ──────────────────────────────────────

@activity.defn
async def grab_with_patterns(
    urls: list[str],
    domain: str,
    entity_type: str,
    dom_chunks: list[dict],
) -> GrabResult:
    """
    Fetch each URL and apply dom_pattern chunks (CSS selectors) to extract labeled DOM sections.
    BATCH_SIZE = 10 URLs processed concurrently.

    Output: {url, domain, entity_type, chunks: [{label, html, size}]}
    """
    from curl_cffi.requests import AsyncSession
    from vendor.transport import PlatformProfile

    BATCH_SIZE = 10
    profile = PlatformProfile()
    last_request_time = {}

    all_chunks = []
    success_count = 0
    fail_count = 0

    activity.logger.info("Grabbing %d URLs for domain %s", len(urls), domain)

    for i in range(0, len(urls), BATCH_SIZE):
        batch = urls[i:i + BATCH_SIZE]
        activity.heartbeat()

        # Rate limit per domain
        await asyncio.sleep(_rate_limit(domain, last_request_time))
        last_request_time[domain] = time.monotonic()

        # Create session per batch
        session = AsyncSession(impersonate=profile.impersonate_profile)

        async def fetch_one(url: str) -> dict | None:
            try:
                html, status, err = await _http_get(session, url, timeout=20)
                if err or status >= 400:
                    logger.warning("grab_with_patterns: %s → %s", url, err or f"HTTP {status}")
                    return None

                soup = BeautifulSoup(html, "html.parser")
                page_chunks = []

                for chunk_def in dom_chunks:
                    label = chunk_def.get("label", "unknown")
                    selector = chunk_def.get("selector", "")
                    if not selector:
                        continue
                    elements = soup.select(selector)
                    for el in elements:
                        html_str = str(el)
                        page_chunks.append({
                            "label": label,
                            "html": html_str,
                            "size": len(html_str),
                        })

                return {
                    "url": url,
                    "domain": domain,
                    "entity_type": entity_type,
                    "chunks": page_chunks,
                }

            except Exception as e:
                logger.warning("grab_with_patterns fetch error %s: %s", url, e)
                return None

        results = await asyncio.gather(*[fetch_one(u) for u in batch], return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                fail_count += 1
                logger.warning("batch item exception: %s", res)
            elif res is None:
                fail_count += 1
            else:
                if res.get("chunks"):
                    all_chunks.append(res)
                    success_count += 1
                else:
                    # Page fetched but no chunks matched — still count as success but no output
                    success_count += 1

        await session.close()

    activity.logger.info(
        "grab_with_patterns: %d pages with chunks, %d failures, domain=%s",
        len(all_chunks), fail_count, domain,
    )
    return GrabResult(chunks=all_chunks, success_count=success_count, fail_count=fail_count)


# ── Activity 4: bridge_to_churner ───────────────────────────────────────

@activity.defn
async def bridge_pattern_grab_to_churner(mission_id: str, chunks: list[dict], source_tag: str) -> int:
    """
    Push grabbed chunks (from grab_with_patterns) into churner.raw_ingests.
    Reuses the existing bridge_to_churner pattern but for pattern-grabbed data.
    """
    inserted = 0
    for chunk in chunks:
        payload = json.dumps(chunk, sort_keys=True)
        content_hash = hashlib.sha256(payload.encode()).hexdigest()

        try:
            conn = _churner_conn()
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO raw_ingests (mission_id, source, raw_payload, content_hash, status)
                   VALUES (%s, %s, %s, %s, 'pending')
                   ON CONFLICT (content_hash) DO NOTHING""",
                (
                    mission_id,
                    f"grabber:pattern:{source_tag}:{chunk.get('url', '')}",
                    payload,
                    content_hash,
                ),
            )
            if cur.rowcount > 0:
                inserted += 1
            cur.close()
            conn.close()
        except Exception as e:
            logger.warning("bridge_pattern_grab_to_churner insert failed: %s", e)

    activity.logger.info(
        "Bridged %d/%d pattern-grabbed chunks to churner (mission=%s, source=%s)",
        inserted, len(chunks), mission_id, source_tag,
    )
    return inserted


# ── Activity 5: mark_audit_grabbed ───────────────────────────────────────

@activity.defn
async def mark_audit_grabbed(audit_id: str) -> None:
    """Mark an audit as fully grabbed in ghostgraph."""
    settings = get_settings()
    conn = psycopg2.connect(settings.postgres_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "UPDATE entity_audits SET grab_status = 'done' WHERE id = %s",
        (audit_id,),
    )
    cur.close()
    conn.close()
    activity.logger.info("Marked audit %s as grab_status=done", audit_id)


# ── Activity 6: mark_audit_failed ──────────────────────────────────────

@activity.defn
async def mark_audit_failed(audit_id: str, error: str) -> None:
    """Mark an audit as failed grab."""
    settings = get_settings()
    conn = psycopg2.connect(settings.postgres_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "UPDATE entity_audits SET grab_status = 'failed' WHERE id = %s",
        (audit_id,),
    )
    cur.close()
    conn.close()
    activity.logger.warning("Marked audit %s as grab_status=failed: %s", audit_id, error)
