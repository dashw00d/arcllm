"""
Temporal activities for Grabber — wraps vendor/ extraction brain.

Each activity corresponds to a pipeline stage:
  explore_urls  → vendor.discovery.discoverer
  generate_schema → vendor.schema_generator.generator
  extract_batch → vendor.extraction.extractor
  bridge_to_churner → cross-DB insert into churner.raw_ingests
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import psycopg2
import psycopg2.extras
from temporalio import activity

from app.core.config import get_settings
from app.db.client import get_cursor

logger = logging.getLogger(__name__)


# ── Dataclasses for activity I/O ──────────────────────────────────────────

@dataclass
class ExploreResult:
    detail_urls: list[str]
    sample_urls: list[str]
    total_links: int
    llm_calls: int


@dataclass
class SchemaResult:
    fields: list[dict]
    field_count: int
    domain: str


@dataclass
class ExtractResult:
    entities: list[dict]
    success_count: int
    fail_count: int


# ── Session helper ────────────────────────────────────────────────────────

def _make_session():
    """Build a StealthSession wired to config (IPv6, proxy, etc.)."""
    from vendor.transport import StealthSession
    from vendor.ipv6_utils import IPv6Rotator

    settings = get_settings()
    ipv6 = IPv6Rotator(settings.ipv6_subnet) if settings.ipv6_subnet else None
    return StealthSession(ipv6_rotator=ipv6)


# ── Activities ────────────────────────────────────────────────────────────

@activity.defn
async def explore_urls(seed_url: str, entity_type: str) -> ExploreResult:
    """Run 5-phase URL discovery on a seed URL."""
    from vendor.discovery.discoverer import Discoverer

    activity.logger.info("Exploring %s (entity_type=%s)", seed_url, entity_type)

    async with _make_session() as session:
        discoverer = Discoverer(session=session)
        result = await discoverer.discover(
            seed_url=seed_url,
            entity_type=entity_type,
        )

    if not result or not result.detail_urls:
        raise RuntimeError(f"Exploration found 0 detail URLs for {seed_url}")

    activity.logger.info("Found %d detail URLs, %d samples", len(result.detail_urls), len(result.sample_urls))

    return ExploreResult(
        detail_urls=result.detail_urls,
        sample_urls=result.sample_urls,
        total_links=result.total_links_found,
        llm_calls=result.llm_calls_made,
    )


@activity.defn
async def generate_schema(
    sample_urls: list[str],
    entity_type: str,
    seed_url: str,
) -> SchemaResult:
    """Run 4-tier schema generation cascade on sample pages."""
    from urllib.parse import urlparse
    from vendor.schema_generator.generator import SchemaGenerator

    domain = urlparse(seed_url).netloc

    activity.logger.info("Generating schema for %s (%d samples)", domain, len(sample_urls))

    async with _make_session() as session:
        generator = SchemaGenerator(session=session)
        schema = await generator.generate(
            sample_urls=sample_urls,
            entity_type=entity_type,
        )

    if not schema or not schema.fields:
        raise RuntimeError(f"Schema generation produced 0 fields for {domain}")

    fields = [f.to_dict() for f in schema.fields]
    activity.logger.info("Generated schema: %d fields for %s", len(fields), domain)

    return SchemaResult(fields=fields, field_count=len(fields), domain=domain)


@activity.defn
async def extract_batch(
    urls: list[str],
    schema_fields: list[dict],
    entity_type: str,
) -> ExtractResult:
    """Extract data from a batch of URLs using the generated schema."""
    from vendor.extraction.extractor import Extractor

    activity.logger.info("Extracting %d URLs", len(urls))

    entities = []
    fail_count = 0

    async with _make_session() as session:
        extractor = Extractor(session=session)
        for url in urls:
            activity.heartbeat()
            try:
                result = await extractor.extract(
                    url=url,
                    schema_fields=schema_fields,
                    entity_type=entity_type,
                )
                if result and result.data:
                    entities.append({"url": url, "data": result.data, "entity_type": entity_type})
                else:
                    fail_count += 1
            except Exception as e:
                logger.warning("Extract failed for %s: %s", url, e)
                fail_count += 1

    activity.logger.info("Extracted %d entities, %d failures", len(entities), fail_count)

    return ExtractResult(entities=entities, success_count=len(entities), fail_count=fail_count)


@activity.defn
async def bridge_to_churner(mission_id: str, entities: list[dict], source_tag: str) -> int:
    """Push extracted entities into churner.raw_ingests for curation."""
    settings = get_settings()
    # Connect to churner DB (same Postgres, different database)
    churner_url = settings.postgres_url.rsplit("/", 1)[0] + "/churner"

    conn = psycopg2.connect(churner_url)
    conn.autocommit = True
    cur = conn.cursor()

    inserted = 0
    for entity in entities:
        payload = json.dumps(entity["data"], sort_keys=True)
        content_hash = hashlib.sha256(payload.encode()).hexdigest()

        try:
            cur.execute(
                """INSERT INTO raw_ingests (mission_id, source, raw_payload, content_hash, status)
                   VALUES (%s, %s, %s, %s, 'pending')
                   ON CONFLICT (content_hash) DO NOTHING""",
                (mission_id, f"grabber:{source_tag}:{entity.get('url', '')}", payload, content_hash),
            )
            if cur.rowcount > 0:
                inserted += 1
        except Exception as e:
            logger.warning("Bridge insert failed: %s", e)

    cur.close()
    conn.close()

    activity.logger.info("Bridged %d/%d entities to churner (mission=%s)", inserted, len(entities), mission_id)
    return inserted
