#!/usr/bin/env python3
"""Pipeline metrics collector — queries ghostgraph and churner DBs, outputs JSON to stdout + appends to metrics.jsonl."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

ROOT_DIR = Path(__file__).parent.parent
METRICS_LOG = ROOT_DIR / "metrics.jsonl"

DB_GHOSTGRAPH = "ghostgraph"
DB_CHURNER = "churner"
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_USER = os.environ.get("DB_USER", "temporal")
DB_PASS = os.environ.get("DB_PASS", "temporal")


def get_conn(dbname):
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=dbname,
        user=DB_USER, password=DB_PASS,
    )


def query_ghostgraph(conn):
    cur = conn.cursor()
    metrics = {}

    # Sites count
    cur.execute("SELECT COUNT(*) FROM sites")
    metrics["sites_count"] = cur.fetchone()[0]

    # Audits by stage
    cur.execute("SELECT stage, COUNT(*) FROM entity_audits GROUP BY stage")
    metrics["audits_by_stage"] = {row[0]: row[1] for row in cur.fetchall()}

    # Page scans count
    cur.execute("SELECT COUNT(*) FROM page_scans")
    metrics["page_scans_count"] = cur.fetchone()[0]

    # DOM patterns count
    cur.execute("SELECT COUNT(*) FROM dom_patterns")
    metrics["dom_patterns_count"] = cur.fetchone()[0]

    # Scans rate (scans per hour) — use first and latest scan timestamps
    cur.execute("""
        SELECT MIN(scanned_at), MAX(scanned_at), COUNT(*)
        FROM page_scans
        WHERE scanned_at IS NOT NULL
    """)
    min_ts, max_ts, scan_count = cur.fetchone()
    if min_ts and max_ts and max_ts > min_ts:
        delta_hours = (max_ts - min_ts).total_seconds() / 3600
        metrics["scans_per_hour"] = round(scan_count / delta_hours, 2) if delta_hours > 0 else 0
    else:
        metrics["scans_per_hour"] = 0

    cur.close()
    return metrics


def query_churner(conn):
    cur = conn.cursor()
    metrics = {}

    # Raw ingests by status
    cur.execute("SELECT status, COUNT(*) FROM raw_ingests GROUP BY status")
    metrics["raw_ingests_by_status"] = {row[0]: row[1] for row in cur.fetchall()}

    # Total raw ingests
    cur.execute("SELECT COUNT(*) FROM raw_ingests")
    metrics["raw_ingests_total"] = cur.fetchone()[0]

    # Entities count
    cur.execute("SELECT COUNT(*) FROM entities")
    metrics["entities_count"] = cur.fetchone()[0]

    # Golden entities count
    cur.execute("SELECT COUNT(*) FROM entities WHERE is_golden = true")
    metrics["golden_entities_count"] = cur.fetchone()[0]

    # Entities per hour — use first and latest entity timestamps
    cur.execute("""
        SELECT MIN(created_at), MAX(created_at), COUNT(*)
        FROM entities
        WHERE created_at IS NOT NULL
    """)
    min_ts, max_ts, entity_count = cur.fetchone()
    if min_ts and max_ts and max_ts > min_ts:
        delta_hours = (max_ts - min_ts).total_seconds() / 3600
        metrics["entities_per_hour"] = round(entity_count / delta_hours, 2) if delta_hours > 0 else 0
    else:
        metrics["entities_per_hour"] = 0

    # Missions count
    cur.execute("SELECT COUNT(*) FROM missions")
    metrics["missions_count"] = cur.fetchone()[0]

    cur.close()
    return metrics


def collect_metrics():
    result = {
        "timestamp": time.time(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }

    # Ghostgraph
    try:
        conn_gg = get_conn(DB_GHOSTGRAPH)
        result["ghostgraph"] = query_ghostgraph(conn_gg)
        conn_gg.close()
    except Exception as e:
        result["ghostgraph"] = {"error": str(e)}

    # Churner
    try:
        conn_ch = get_conn(DB_CHURNER)
        result["churner"] = query_churner(conn_ch)
        conn_ch.close()
    except Exception as e:
        result["churner"] = {"error": str(e)}

    return result


def main():
    parser = argparse.ArgumentParser(description="Pipeline metrics collector")
    parser.add_argument("--loop", action="store_true", help="Run in a loop every 60 seconds")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval in seconds")
    args = parser.parse_args()

    if args.loop:
        print(f"[metrics] Running in loop (interval={args.interval}s), log={METRICS_LOG}", flush=True)
        while True:
            metrics = collect_metrics()
            line = json.dumps(metrics)
            print(line)
            try:
                with open(METRICS_LOG, "a") as f:
                    f.write(line + "\n")
            except Exception as e:
                print(f"[metrics] Failed to append to log: {e}", flush=True)
            time.sleep(args.interval)
    else:
        metrics = collect_metrics()
        print(json.dumps(metrics, indent=2))
        try:
            with open(METRICS_LOG, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            print(f"[metrics] Failed to append to log: {e}", flush=True)


if __name__ == "__main__":
    main()
