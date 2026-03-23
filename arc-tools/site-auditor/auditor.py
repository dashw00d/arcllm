"""
Site Auditor — Multi-Fidelity 4-Stage Pipeline with persistent state.

Every page visit, pattern, and decision is recorded in Postgres.
Pipeline is resumable — picks up where it left off after crash.
Never re-scans a page in the same stage unless forced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from urllib.parse import urlparse

from agent import henry_call, run_agent
from recon import run_recon
from tools import (
    browser_open, browser_eval, browser_get_html, browser_close,
    get_markdown, SKELETONIZE_JS,
)
import state

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("auditor")


# ── Stage 1: Triage ──────────────────────────────────────────────────────

async def stage_triage(audit: dict, url: str, entity_type: str, description: str) -> dict:
    """Is this site relevant? Single LLM call on markdown."""
    audit_id = str(audit["id"])

    # Skip if already triaged
    if audit["stage"] not in ("pending", "triage"):
        logger.info("TRIAGE: already done (stage=%s)", audit["stage"])
        return {"relevant": audit["relevant"], "confidence": audit["confidence"]}

    state.advance_stage(audit_id, "triage")
    logger.info("TRIAGE — fetching markdown for %s", url)

    markdown = await get_markdown(url)
    logger.info("  Got %d chars of markdown", len(markdown))

    result = await henry_call(
        system="""You evaluate websites for data relevance. Respond with JSON:
{"relevant": true/false, "confidence": 0.0-1.0, "reasoning": "why", "site_purpose": "what the site does", "entity_sections": ["where entities appear"], "suggested_urls": ["URLs to explore"]}""",
        user=f"Does this site contain data about: {entity_type}?\nDescription: {description}\nURL: {url}\n\nA site can be relevant even if its main purpose differs. A catering site listing venues = relevant for 'venues'.\n\nPage content:\n{markdown}",
    )

    if not result:
        state.update_audit(audit_id, error="triage LLM call failed")
        return {"relevant": False}

    # Record triage result
    relevant = result.get("relevant", False)
    confidence = result.get("confidence", 0)
    state.update_audit(
        audit_id,
        relevant=relevant,
        confidence=confidence,
        reasoning=result.get("reasoning", ""),
    )

    # Record the scan
    state.record_scan(
        audit_id, url, urlparse(url).path or "/", "triage",
        page_type="homepage",
        description=result.get("site_purpose", ""),
        outcome="relevant" if relevant else "not_relevant",
        summary=result.get("reasoning", ""),
    )

    if not relevant:
        state.advance_stage(audit_id, "not_relevant")
    else:
        state.advance_stage(audit_id, "discovery")

    logger.info("  Relevant: %s (%.0f%%)", relevant, confidence * 100)
    return result


# ── Stage 2: Discovery ───────────────────────────────────────────────────

DISCOVERY_PROMPT = """You are finding list/index pages for a specific entity type on a website.

You have browser tools. Strategy:
1. Open the starting URL, take a snapshot
2. Look for links to listings, directories, categories
3. Click into promising pages — if it lists many entities, record it
4. Click into 1-2 entity detail pages to verify
5. Call 'done' when you've found list pages and sample detail pages

When done, call 'done' with JSON:
{"list_pages": [{"url": "...", "description": "..."}], "sample_detail_pages": [{"url": "...", "description": "..."}], "pagination": "how to get more", "notes": "site structure"}"""


async def stage_discovery(audit: dict, url: str, entity_type: str, hints: list[str]) -> dict:
    """Browse for list/detail pages using agent-browser."""
    audit_id = str(audit["id"])

    if audit["stage"] not in ("discovery",):
        logger.info("DISCOVERY: skipping (stage=%s)", audit["stage"])
        # Return existing scan data
        scans = state.get_scans(audit_id, "discovery")
        list_pages = [{"url": s["url"], "description": s["description"]} for s in scans if s["page_type"] == "index"]
        detail_pages = [{"url": s["url"], "description": s["description"]} for s in scans if s["page_type"] == "detail"]
        return {"list_pages": list_pages, "sample_detail_pages": detail_pages}

    logger.info("DISCOVERY — browsing for %s pages", entity_type)

    start_urls = hints[:3] if hints else [url]
    result = await run_agent(
        system_prompt=DISCOVERY_PROMPT,
        user_task=f"Find list/index pages for '{entity_type}'.\nStart from: {', '.join(start_urls)}",
        max_steps=20,
    )

    if not result:
        state.update_audit(audit_id, error="discovery agent loop failed")
        return {"list_pages": [], "sample_detail_pages": []}

    # Record discovered pages
    for page in result.get("list_pages", []):
        page_url = page["url"] if isinstance(page, dict) else page
        desc = page.get("description", "") if isinstance(page, dict) else ""
        if not state.was_scanned(audit_id, page_url, "discovery"):
            state.record_scan(
                audit_id, page_url, urlparse(page_url).path, "discovery",
                page_type="index", description=desc, outcome="found_entities",
            )

    for page in result.get("sample_detail_pages", []):
        page_url = page["url"] if isinstance(page, dict) else page
        desc = page.get("description", "") if isinstance(page, dict) else ""
        if not state.was_scanned(audit_id, page_url, "discovery"):
            state.record_scan(
                audit_id, page_url, urlparse(page_url).path, "discovery",
                page_type="detail", description=desc, outcome="found_entities",
            )

    lp = result.get("list_pages", [])
    dp = result.get("sample_detail_pages", [])
    state.update_audit(audit_id, list_pages_found=len(lp), detail_pages_found=len(dp))
    state.advance_stage(audit_id, "patterning")

    logger.info("  Found %d list pages, %d detail pages", len(lp), len(dp))
    return result


# ── Stage 3: Patterning ──────────────────────────────────────────────────

async def stage_pattern_list(audit: dict, url: str, entity_type: str) -> dict:
    """Identify repeating entity patterns via skeletal HTML."""
    audit_id = str(audit["id"])

    if state.was_scanned(audit_id, url, "patterning"):
        logger.info("PATTERNING: already scanned %s", url)
        return {}

    logger.info("PATTERNING list page — %s", url)
    browser_open(url)
    skeleton = browser_eval(SKELETONIZE_JS)

    if not skeleton or len(skeleton) < 50:
        skeleton = browser_get_html("body")[:4000]

    result = await henry_call(
        system="""You analyze DOM structure to find repeating entity patterns. Respond with JSON:
{"entity_selector": "CSS selector", "link_selector": "CSS selector for detail link", "pagination_selector": "next page selector or null", "entity_count": number, "confidence": 0.0-1.0}""",
        user=f"Find the repeating pattern for '{entity_type}' in this page.\nURL: {url}\n\nText replaced with '...' to show structure:\n{skeleton[:4000]}",
    )

    # Record scan
    state.record_scan(
        audit_id, url, urlparse(url).path, "patterning",
        page_type="index",
        description=f"entity_selector: {result.get('entity_selector', '?')}" if result else "failed",
        outcome="found_entities" if result else "error",
        element_count=result.get("entity_count") if result else None,
    )

    # Save URL pattern
    if result and result.get("entity_selector"):
        # Derive URL pattern from the URL
        parsed = urlparse(url)
        path = parsed.path
        url_pat = path  # TODO: Henry should help generalize this
        state.save_url_pattern(
            audit_id, "index", url_pat,
            example_urls=[url],
            description=f"Lists {entity_type}",
            pagination_selector=result.get("pagination_selector"),
            confidence=result.get("confidence", 0),
        )

        # Save DOM pattern
        state.save_dom_pattern(
            audit_id, "index",
            entity_selector=result.get("entity_selector"),
            link_selector=result.get("link_selector"),
            count_on_page=result.get("entity_count"),
            title_selector=result.get("title_selector"),
            confidence=result.get("confidence", 0),
            tested_on_urls=[url],
        )

    return result or {}


async def stage_pattern_detail(audit: dict, url: str, entity_type: str) -> dict:
    """Identify data chunks on a detail page via skeletal HTML."""
    audit_id = str(audit["id"])

    if state.was_scanned(audit_id, url, "extraction"):
        logger.info("EXTRACTION: already scanned %s", url)
        return {}

    logger.info("PATTERNING detail page — %s", url)
    browser_open(url)
    skeleton = browser_eval(SKELETONIZE_JS)

    if not skeleton or len(skeleton) < 50:
        skeleton = browser_get_html("body")[:4000]

    result = await henry_call(
        system="""You identify data sections on entity pages. Respond with JSON:
{"entity_name": "name if visible", "chunks": [{"label": "reviews/pricing/contact/etc", "selector": "CSS selector", "size": "small/medium/large", "contains": "what data"}], "title_selector": "CSS for entity name"}""",
        user=f"Identify data sections on this {entity_type} page.\nURL: {url}\n\nDOM structure:\n{skeleton[:4000]}",
    )

    # Record scan
    chunks = result.get("chunks", []) if result else []
    state.record_scan(
        audit_id, url, urlparse(url).path, "extraction",
        page_type="detail",
        description=f"{len(chunks)} chunks: {', '.join(c.get('label', '?') for c in chunks)}" if chunks else "failed",
        outcome="found_entities" if chunks else "error",
        element_count=len(chunks),
    )

    # Save DOM pattern
    if result and chunks:
        state.save_dom_pattern(
            audit_id, "detail",
            chunks=chunks,
            title_selector=result.get("title_selector"),
            confidence=0.8,
            tested_on_urls=[url],
        )

    return result or {}


# ── Main orchestrator ─────────────────────────────────────────────────────

async def audit_site(url: str, entity_type: str, description: str) -> dict:
    """Run the full audit pipeline with persistent state."""

    # Init schema if needed
    state.init_schema()

    domain = urlparse(url).netloc
    site = state.get_or_create_site(domain)
    site_id = str(site["id"])

    # Check for resumable audit
    existing = state.get_resumable_audit(domain, entity_type)
    if existing:
        audit = existing
        logger.info("Resuming audit %s at stage: %s", audit["id"], audit["stage"])
    else:
        audit = state.get_or_create_audit(site_id, entity_type, description)
        logger.info("Starting new audit %s for %s / %s", audit["id"], domain, entity_type)

    # Stage 0: Recon
    if site["status"] == "new" or not site.get("waf"):
        logger.info("=" * 60)
        logger.info("STAGE 0: RECON — %s", url)
        logger.info("=" * 60)
        recon = await run_recon(url)
        state.update_site(
            site_id,
            description=recon.get("headers", {}).get("server", ""),
            ipv6_supported=recon["ipv6"]["supported"],
            waf=recon["headers"].get("waf"),
            robots_disallow=recon["robots"].get("disallowed", []),
            warnings=recon.get("warnings", []),
            status="auditing",
        )
        logger.info("  IPv6: %s | WAF: %s", recon["ipv6"]["supported"], recon["headers"].get("waf", "none"))

    # Stage 1: Triage
    logger.info("=" * 60)
    logger.info("STAGE 1: TRIAGE")
    logger.info("=" * 60)
    triage = await stage_triage(audit, url, entity_type, description)

    if not triage.get("relevant"):
        logger.info("Not relevant. Done.")
        return {"status": "not_relevant", "audit_id": str(audit["id"]), "reasoning": triage.get("reasoning", "")}

    # Refresh audit from DB (triage advanced the stage)
    audit = state.get_resumable_audit(domain, entity_type) or audit

    # Stage 2: Discovery
    logger.info("=" * 60)
    logger.info("STAGE 2: DISCOVERY")
    logger.info("=" * 60)
    discovery = await stage_discovery(
        audit, url, entity_type,
        triage.get("suggested_urls", [url]),
    )

    list_pages = discovery.get("list_pages", [])
    sample_details = discovery.get("sample_detail_pages", [])

    if not list_pages and not sample_details:
        state.update_audit(str(audit["id"]), error="no pages found during discovery")
        return {"status": "no_pages_found", "audit_id": str(audit["id"])}

    # Stage 3: Pattern list pages
    logger.info("=" * 60)
    logger.info("STAGE 3: PATTERN LIST PAGES")
    logger.info("=" * 60)
    for page in list_pages[:3]:
        page_url = page["url"] if isinstance(page, dict) else page
        await stage_pattern_list(audit, page_url, entity_type)

    # Stage 4: Pattern detail pages
    logger.info("=" * 60)
    logger.info("STAGE 4: PATTERN DETAIL PAGES")
    logger.info("=" * 60)
    detail_urls = []
    for d in sample_details:
        detail_urls.append(d["url"] if isinstance(d, dict) else d)
    detail_urls = list(dict.fromkeys(detail_urls))[:3]

    for detail_url in detail_urls:
        await stage_pattern_detail(audit, detail_url, entity_type)

    # Mark complete
    state.advance_stage(str(audit["id"]), "complete")
    state.update_site(site_id, status="ready", last_audited="now()")
    browser_close()

    # Summary
    url_patterns = state.get_url_patterns(str(audit["id"]))
    dom_patterns = state.get_dom_patterns(str(audit["id"]))
    scans = state.get_scans(str(audit["id"]))

    logger.info("=" * 60)
    logger.info("AUDIT COMPLETE")
    logger.info("  Pages scanned: %d", len(scans))
    logger.info("  URL patterns: %d", len(url_patterns))
    logger.info("  DOM patterns: %d", len(dom_patterns))
    logger.info("  Audit ID: %s", audit["id"])
    logger.info("=" * 60)

    return {
        "status": "complete",
        "audit_id": str(audit["id"]),
        "site_id": site_id,
        "domain": domain,
        "pages_scanned": len(scans),
        "url_patterns": len(url_patterns),
        "dom_patterns": len(dom_patterns),
    }


def main():
    parser = argparse.ArgumentParser(description="Site Auditor")
    parser.add_argument("--url", required=True)
    parser.add_argument("--entity-type", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = asyncio.run(audit_site(
        args.url, args.entity_type,
        args.description or args.entity_type,
    ))

    output = json.dumps(results, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        logger.info("Results written to %s", args.output)
    else:
        print(output)


if __name__ == "__main__":
    main()
