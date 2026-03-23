#!/usr/bin/env python3
"""Pipeline health monitor — checks all services and auto-restarts Docker containers on failure."""

import asyncio
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
STATUS_FILE = ROOT_DIR / "pipeline_status.json"
COMPOSE_DIR = ROOT_DIR

HENRY_URL = "http://localhost:11435/health"
GRABBER_API_URL = "http://localhost:8000/health"
PG_PORT = 5432
TEMPORAL_PORT = 7233
CHECK_INTERVAL = 60

_shutdown = False


def log(msg):
    print(f"[health-monitor] {msg}", flush=True)


def sigterm_handler(signum, frame):
    global _shutdown
    log("Received SIGTERM, shutting down gracefully...")
    _shutdown = True


signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)


def check_http(url, timeout=5):
    """Returns (ok, latency_ms, error_msg)."""
    import urllib.request
    try:
        start = time.monotonic()
        req = urllib.request.urlopen(url, timeout=timeout)
        req.read()
        latency_ms = (time.monotonic() - start) * 1000
        return (True, latency_ms, None)
    except Exception as e:
        return (False, None, str(e))


def check_tcp(host, port, timeout=3):
    """Returns True if port is open."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except Exception:
        return False


def check_postgres():
    """Returns (ok, error_msg)."""
    ok = check_tcp("localhost", PG_PORT, timeout=3)
    return (ok, None if ok else "postgres port not reachable")


def check_temporal():
    """Returns (ok, error_msg)."""
    ok = check_tcp("localhost", TEMPORAL_PORT, timeout=3)
    return (ok, None if ok else "temporal port not reachable")


def check_docker_service(service):
    """Returns (running, error)."""
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", service],
            cwd=COMPOSE_DIR,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return (False, f"docker compose ps failed: {result.stderr}")
        lines = result.stdout.strip().split("\n")
        for line in lines[1:]:
            if service in line:
                if "Up" in line or "running" in line.lower():
                    return (True, None)
                else:
                    return (False, f"service not Up: {line}")
        return (False, f"service {service} not found in ps output")
    except Exception as e:
        return (False, str(e))


def restart_docker_service(service):
    """Attempt to restart a Docker service. Returns (success, error)."""
    try:
        result = subprocess.run(
            ["docker", "compose", "restart", service],
            cwd=COMPOSE_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            log(f"Restarted {service}")
            return (True, None)
        else:
            return (False, result.stderr)
    except Exception as e:
        return (False, str(e))


DOCKER_SERVICES = [
    "db",
    "redis",
    "temporal",
    "temporal-ui",
    "grabber-api",
    "grabber-worker",
    "worker-fast",
    "worker-heavy",
]


def run_checks():
    """Run all health checks, restart failed Docker services, return status dict."""
    status = {
        "timestamp": time.time(),
        "services": {},
        "overall": "healthy",
        "actions_taken": [],
    }

    # Henry proxy
    ok, latency, err = check_http(HENRY_URL, timeout=5)
    status["services"]["henry"] = {"ok": ok, "latency_ms": latency, "error": err}
    if not ok:
        status["overall"] = "degraded"
        log(f"Henry proxy FAILED: {err}")

    # Grabber API
    ok, latency, err = check_http(GRABBER_API_URL, timeout=5)
    status["services"]["grabber-api"] = {"ok": ok, "latency_ms": latency, "error": err}
    if not ok:
        status["overall"] = "degraded"
        log(f"Grabber API FAILED: {err}")

    # Postgres
    ok, err = check_postgres()
    status["services"]["postgres"] = {"ok": ok, "error": err}
    if not ok:
        status["overall"] = "unhealthy"
        log(f"Postgres FAILED: {err}")

    # Temporal
    ok, err = check_temporal()
    status["services"]["temporal"] = {"ok": ok, "error": err}
    if not ok:
        status["overall"] = "unhealthy"
        log(f"Temporal FAILED: {err}")

    # Docker services
    docker_status = {}
    for svc in DOCKER_SERVICES:
        running, err = check_docker_service(svc)
        docker_status[svc] = {"ok": running, "error": err}
        if not running:
            status["overall"] = "degraded"
            log(f"Docker service {svc} not running: {err}")
            # Attempt restart
            success, restart_err = restart_docker_service(svc)
            if success:
                status["actions_taken"].append(f"restarted {svc}")
                docker_status[svc]["restarted"] = True
            else:
                status["actions_taken"].append(f"FAILED to restart {svc}: {restart_err}")
                status["overall"] = "unhealthy"

    status["services"]["docker"] = docker_status

    # Write status file
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        log(f"Failed to write status file: {e}")

    return status


def main():
    log(f"Starting pipeline health monitor (interval={CHECK_INTERVAL}s)")
    log(f"Status file: {STATUS_FILE}")

    while not _shutdown:
        status = run_checks()
        if status["overall"] == "healthy":
            log("All services healthy")
        else:
            log(f"Status: {status['overall']} — actions: {status['actions_taken']}")

        # Sleep in small increments so we can respond to shutdown quickly
        for _ in range(CHECK_INTERVAL):
            if _shutdown:
                break
            time.sleep(1)

    log("Health monitor stopped.")


if __name__ == "__main__":
    main()
