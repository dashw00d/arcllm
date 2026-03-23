"""
Seed site discovery — finds prospect sites for a given entity type via DuckDuckGo.

Usage:
  python3 seed_sites.py --entity-type "wedding venues" [--limit 100]

Output:
  - Inserts domains into ghostgraph DB with status='queued'
  - Writes discovered sites to sites_queue.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    from ddgs import DDGS
    _HAS_DDGS = True
except ImportError:
    _HAS_DDGS = False
    print("WARNING: ddgs not installed. Run: pip3 install --break-system-packages ddgs")

sys.path.insert(0, str(Path(__file__).parent))

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("seed_sites")

# ── DB ─────────────────────────────────────────────────────────────────────

DB_URL = os.environ.get("POSTGRES_URL", "postgresql://temporal:temporal@localhost:5432/ghostgraph")


def _db_conn():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    return conn


# ── Domain extraction ──────────────────────────────────────────────────────

def _extract_domain(url_or_host: str) -> str | None:
    """Extract root domain from a URL or hostname."""
    url_or_host = url_or_host.strip().lower()
    if not url_or_host:
        return None
    # If no scheme, add one to parse correctly
    if not re.match(r"^https?://", url_or_host):
        url_or_host = "https://" + url_or_host
    try:
        parsed = urlparse(url_or_host)
        domain = parsed.netloc or parsed.path.split("/")[0]
        # Strip www. prefix
        domain = re.sub(r"^www\.", "", domain)
        # Remove port
        domain = domain.split(":")[0]
        if domain and "." in domain and not domain.startswith("."):
            return domain
    except Exception:
        pass
    return None


# ── DuckDuckGo search ──────────────────────────────────────────────────────

SEARCH_QUERIES = [
    "{entity} near me",
    "{entity} directory",
    "{entity} listings",
    "best {entity}",
    "{entity} reviews",
    '"{entity}" site list',
]


def _is_valid_url(url: str) -> bool:
    """Check if a string looks like an actual URL (not a search snippet)."""
    url = url.strip().lower()
    if not url:
        return False
    # Must start with http(s)://
    if not url.startswith("http"):
        return False
    # Must have a domain with TLD (at least one dot after the domain part)
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if not netloc or "." not in netloc:
            return False
        # Reject known search/engine domains
        if any(bad in netloc for bad in ["bing.com", "duckduckgo.com", "google.com", "yahoo.com", "baidu.com", "yandex.ru"]):
            return False
        return True
    except Exception:
        return False


def _search_duckduckgo(entity_type: str, max_results: int = 80) -> set[str]:
    """Search DuckDuckGo for domains related to entity_type."""
    if not _HAS_DDGS:
        return set()

    domains: set[str] = set()
    queries = [q.format(entity=entity_type) for q in SEARCH_QUERIES]

    with DDGS() as ddgs:
        for query in queries:
            if len(domains) >= max_results:
                break
            try:
                logger.info("Searching: %s", query)
                results = ddgs.text(query, max_results=max_results)
                for r in results:
                    url = r.get("href", "")
                    # Only use href — never the title text as a URL
                    if _is_valid_url(url):
                        domain = _extract_domain(url)
                        if domain:
                            domains.add(domain)
                # Be polite — small delay between queries
                time.sleep(1.5)
            except Exception as e:
                logger.warning("Search error for '%s': %s", query, e)
                continue

    logger.info("Got %d unique domains from DuckDuckGo", len(domains))
    return domains


# ── DB operations ──────────────────────────────────────────────────────────

def _insert_sites(domains: set[str], entity_type: str) -> int:
    """Insert domains into DB with status='queued'. Returns count inserted."""
    conn = _db_conn()
    cur = conn.cursor()

    inserted = 0
    for domain in sorted(domains):
        try:
            cur.execute(
                """
                INSERT INTO sites (domain, status, description)
                VALUES (%s, 'queued', %s)
                ON CONFLICT (domain) DO UPDATE SET status = EXCLUDED.status
                WHERE sites.status = 'new'
                RETURNING id
                """,
                (domain, f"seeded for: {entity_type}"),
            )
            result = cur.fetchone()
            if result:
                inserted += 1
                logger.debug("Inserted: %s", domain)
        except Exception as e:
            logger.warning("DB insert error for %s: %s", domain, e)
            continue

    conn.close()
    return inserted


def _write_jsonl(domains: set[str], entity_type: str, output_path: Path):
    """Write domains to JSONL file for batch_auditor."""
    with open(output_path, "w") as f:
        for domain in sorted(domains):
            f.write(json.dumps({"url": f"https://{domain}", "entity_type": entity_type}) + "\n")
    logger.info("Wrote %d sites to %s", len(domains), output_path)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed Site Discovery")
    parser.add_argument("--entity-type", required=True, help="Entity type to search for")
    parser.add_argument("--limit", type=int, default=100, help="Max domains to collect (default 100)")
    parser.add_argument("--output", default="sites_queue.jsonl",
                        help="Output JSONL path (default: sites_queue.jsonl)")
    args = parser.parse_args()

    logger.info("Starting seed discovery for: %s", args.entity_type)

    # Discover domains
    domains = _search_duckduckgo(args.entity_type, max_results=args.limit)

    if not domains:
        logger.warning("No domains found for '%s'. Is duckduckgo_search installed?", args.entity_type)
        print("No domains found. Try: pip3 install --break-system-packages duckduckgo_search")
        sys.exit(1)

    # Insert into DB
    try:
        inserted = _insert_sites(domains, args.entity_type)
        logger.info("Inserted %d new domains into DB", inserted)
    except Exception as e:
        logger.error("DB insert failed: %s", e)
        inserted = 0

    # Write to JSONL
    output_path = Path(__file__).parent / args.output
    _write_jsonl(domains, args.entity_type, output_path)

    print(f"\nDiscovered {len(domains)} domains for '{args.entity_type}'")
    print(f"  {inserted} new domains inserted into DB (status=queued)")
    print(f"  {len(domains)} domains written to {output_path}")
    print(f"\nNext: python3 batch_auditor.py --input {output_path}")


if __name__ == "__main__":
    main()
