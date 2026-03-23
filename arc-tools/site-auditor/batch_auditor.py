"""
Batch site auditor — runs audits on a list of sites with fault tolerance.

Supports two input modes:
- JSONL file: one {"url": "...", "entity_type": "..."} per line
- Database: reads queued sites from ghostgraph DB

Features:
- Per-site try/except with 3 retries and exponential backoff (30s/60s/120s)
- Signal handlers for graceful SIGTERM/SIGINT shutdown
- Structured JSON logging to audit.log
- Browser closed between sites to prevent leaks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import state as auditor_state
from auditor import audit_site

# ── Logging ───────────────────────────────────────────────────────────────

LOG_FILE = Path(__file__).parent / "audit.log"
_json_log = logging.getLogger("batch_auditor.json")
_file_handler = logging.FileHandler(LOG_FILE)
_file_handler.setFormatter(logging.Formatter("%(message)s"))
_json_log.addHandler(_file_handler)
_json_log.setLevel(logging.INFO)

# ── Shutdown state ─────────────────────────────────────────────────────────

_shutdown_requested = False


def _request_shutdown(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    _json_log.info(json.dumps({"event": "shutdown_signal", "signal": sig_name}))
    print(f"\nReceived {sig_name} — will finish current audit, then exit.")
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _request_shutdown)
signal.signal(signal.SIGINT, _request_shutdown)


# ── Core ──────────────────────────────────────────────────────────────────

def _log(event: str, **kwargs):
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}
    _json_log.info(json.dumps(entry))


def _retry_backoff(attempt: int) -> float:
    return [30, 60, 120][min(attempt, 2)]


def audit_with_retry(url: str, entity_type: str, description: str, attempt: int = 0) -> dict:
    """Run audit_site with exponential backoff retry."""
    start = time.monotonic()
    try:
        result = {"status": "unknown", "duration_s": 0}
        actual = audit_site(url, entity_type, description)
        duration = time.monotonic() - start
        result = {**actual, "duration_s": round(duration, 1)}
        _log("audit_complete", url=url, entity_type=entity_type, attempt=attempt + 1, **result)
        return result
    except Exception as e:
        duration = time.monotonic() - start
        if attempt < 3:
            wait = _retry_backoff(attempt)
            _log("audit_retry", url=url, entity_type=entity_type, attempt=attempt + 1,
                 wait_s=wait, error=str(e)[:200])
            time.sleep(wait)
            return audit_with_retry(url, entity_type, description, attempt + 1)
        else:
            _log("audit_failed", url=url, entity_type=entity_type, error=str(e)[:200],
                 duration_s=round(duration, 1))
            return {"status": "failed", "url": url, "entity_type": entity_type,
                    "error": str(e), "duration_s": round(duration, 1)}


def _from_jsonl(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                _json_log.warning(json.dumps({"event": "jsonl_skip", "line": line[:100]}))
    return items


def _from_db(entity_type: str) -> list[dict]:
    """Read queued sites from ghostgraph DB.

    Two sources are merged:
    1. entity_audits with stage='pending' (normal pipeline flow)
    2. Sites with status='queued' that have no entity_audit row yet
       (bridges the seed_sites → batch_auditor handoff gap)
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        DB_URL = os.environ.get("POSTGRES_URL", "postgresql://temporal:temporal@localhost:5432/ghostgraph")
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # ── Handoff gap fix: create entity_audit rows for queued sites that
        #    don't have one yet (seed_sites only inserts into `sites` table).
        cur.execute(
            """INSERT INTO entity_audits (site_id, entity_type, description, stage)
               SELECT s.id, %s, 'seeded: ' || s.description, 'pending'
               FROM sites s
               WHERE s.status = 'queued'
                 AND NOT EXISTS (
                     SELECT 1 FROM entity_audits ea
                     WHERE ea.site_id = s.id AND ea.entity_type = %s
                 )""",
            (entity_type, entity_type),
        )

        # ── Normal: pick up all pending audits for this entity type
        cur.execute(
            """SELECT s.domain, ea.entity_type, ea.description
               FROM sites s
               JOIN entity_audits ea ON ea.site_id = s.id
               WHERE ea.stage = 'pending' AND ea.entity_type = %s
               ORDER BY ea.created_at LIMIT 200""",
            (entity_type,),
        )
        rows = cur.fetchall()
        conn.close()

        items = []
        for row in rows:
            domain = row["domain"]
            scheme = "https://" if not domain.startswith("http") else ""
            items.append({
                "url": f"{scheme}{domain}",
                "entity_type": row["entity_type"],
                "description": row.get("description") or "",
            })
        return items
    except Exception as e:
        _json_log.warning(json.dumps({"event": "db_read_error", "error": str(e)}))
        return []


def run_batch(items: list[dict], from_db: bool = False):
    """Run audits for all items. Each item: {"url": "...", "entity_type": "...", "description": "..."}."""
    total = len(items)
    results = {"success": 0, "failed": 0, "skipped": 0, "items": []}
    _log("batch_start", total=total, source="jsonl" if not from_db else "db")

    for idx, item in enumerate(items):
        if _shutdown_requested:
            _log("batch_shutdown", processed=idx + 1, remaining=total - idx - 1)
            break

        url = item.get("url")
        entity_type = item.get("entity_type", "")
        description = item.get("description", entity_type)

        if not url or not entity_type:
            _log("batch_skip", idx=idx, reason="missing url or entity_type", item=str(item)[:100])
            results["skipped"] += 1
            continue

        _log("batch_item_start", idx=idx + 1, total=total, url=url, entity_type=entity_type)

        result = audit_with_retry(url, entity_type, description)

        if result.get("status") == "complete":
            results["success"] += 1
        else:
            results["failed"] += 1

        results["items"].append(result)

        # Close browser between sites to prevent resource leaks
        try:
            from tools import browser_close
            browser_close()
        except Exception:
            pass

        # Progress dot
        status_char = "✓" if result.get("status") == "complete" else "✗"
        print(f"  [{idx + 1}/{total}] {status_char} {url} ({result.get('status', '?')}) — {result.get('duration_s', 0)}s")

    _log("batch_complete", **results)
    print(f"\nBatch complete: {results['success']} succeeded, {results['failed']} failed, {results['skipped']} skipped")
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch Site Auditor")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", metavar="FILE", help="JSONL input file")
    group.add_argument("--from-db", action="store_true", help="Read from ghostgraph DB")
    parser.add_argument("--entity-type", default="wedding venues",
                        help="Entity type for DB mode or when not in JSONL")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max sites to process (0 = unlimited)")
    args = parser.parse_args()

    if args.from_db:
        items = _from_db(args.entity_type)
    elif args.input:
        items = _from_jsonl(args.input)
    else:
        parser.error("Either --input or --from-db is required")
        return

    if args.limit > 0:
        items = items[:args.limit]

    if not items:
        print("No items to process.")
        return

    run_batch(items, from_db=args.from_db)


if __name__ == "__main__":
    main()
