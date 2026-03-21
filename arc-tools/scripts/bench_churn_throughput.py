#!/usr/bin/env python3
"""Benchmark churner throughput across different parallelism configs.

Sends real extraction prompts to the LLM server with varying concurrency,
max_tokens, and records-per-call to find the optimal configuration.

Usage:
    python scripts/bench_churn_throughput.py [--server-url http://localhost:8090]
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

# ── Config ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
LLM_STACK = ROOT.parent / "llm-stack"
SERVER_BIN = LLM_STACK / "llama.cpp" / "build-sycl" / "bin" / "llama-server"
MODEL_PATH = LLM_STACK / "models" / "Qwen" / "Qwen3-32B-abliterated-GGUF" / "Qwen3-32B-abliterated.Q8_0.gguf"
TEST_DATA = ROOT / "test_data" / "venues.json"

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8090")
SYCL_ENV = {
    "GGML_SYCL_DISABLE_GRAPH": "1",
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "0",
    "ZE_AFFINITY_MASK": "0,1,2",
}

SYSTEM_PROMPT = """You are a data extraction specialist. You receive raw unstructured \
data and a target JSON schema. Your job is to extract every factual claim from the raw data \
and output a single valid JSON object conforming to the schema. Rules: \
1. Only include facts explicitly stated or directly inferable from the source. \
2. Use null for fields with no data — never fabricate. \
3. Normalize formats: phones as E.164, addresses with street/city/state/zip/country, \
dates as ISO 8601, URLs without trailing slashes. \
4. If the raw data contains multiple entities, output an array. \
5. Include a _confidence field (0.0-1.0) for the overall extraction. \
6. Include a _notes field for ambiguities or conflicts in the source data."""

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "address": {"type": "string"},
        "phone": {"type": "string"},
        "description": {"type": "string"},
        "category": {"type": "string"},
        "hours": {"type": "string"},
        "price_range": {"type": "string"},
        "website": {"type": "string"},
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────

def load_records(n=20):
    with open(TEST_DATA) as f:
        data = json.load(f)
    return data[:n]


def make_user_msg(records: list[dict]) -> str:
    """Build extraction prompt for 1 or more records."""
    schema_str = json.dumps(SCHEMA, indent=2)
    if len(records) == 1:
        payload = json.dumps(records[0], indent=2)
        return f"## Target Schema\n```json\n{schema_str}\n```\n\n## Raw Data\n```json\n{payload}\n```\n\nExtract all entities. Output valid JSON only."
    else:
        payloads = "\n---\n".join(json.dumps(r, indent=2) for r in records)
        return (
            f"## Target Schema\n```json\n{schema_str}\n```\n\n"
            f"## Raw Data ({len(records)} records — extract each separately)\n"
            f"```json\n{payloads}\n```\n\n"
            f"Extract entities from ALL {len(records)} records. "
            f"Output a JSON array with one object per record."
        )


server_proc = None

def start_server(np: int, context: int):
    global server_proc
    stop_server()

    env = {**os.environ, **SYCL_ENV}
    cmd = [
        str(SERVER_BIN),
        "-m", str(MODEL_PATH),
        "--split-mode", "layer", "-ngl", "999",
        "-c", str(context), "-fa", "on",
        "-np", str(np),
        "--reasoning-budget", "0",
        "--threads", "10",
        "--host", "0.0.0.0", "--port", "8090",
    ]
    server_proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    # Wait for health
    for i in range(120):
        time.sleep(1)
        try:
            import urllib.request
            urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2)
            return True
        except Exception:
            if server_proc.poll() is not None:
                print(f"  SERVER CRASHED (exit {server_proc.returncode})")
                return False
    print("  SERVER TIMEOUT")
    return False


def stop_server():
    global server_proc
    if server_proc and server_proc.poll() is None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
    server_proc = None
    # Also kill any stray
    os.system("pkill -f llama-server 2>/dev/null")
    time.sleep(2)


async def run_concurrent_requests(client, records, concurrency, max_tokens, records_per_call):
    """Fire `concurrency` extraction requests simultaneously, return stats."""
    # Split records into chunks for each request
    chunks = []
    idx = 0
    for _ in range(concurrency):
        chunk = records[idx:idx + records_per_call]
        if not chunk:
            chunk = records[:records_per_call]  # wrap around
        chunks.append(chunk)
        idx += records_per_call

    total_records = sum(len(c) for c in chunks)

    async def one_request(chunk):
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model="qwen3-32b",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": make_user_msg(chunk)},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.2,
                ),
                timeout=300,
            )
            elapsed = time.monotonic() - t0
            tokens = resp.usage.completion_tokens if resp.usage else 0
            return {"ok": True, "elapsed": elapsed, "tokens": tokens, "records": len(chunk)}
        except Exception as e:
            return {"ok": False, "elapsed": time.monotonic() - t0, "tokens": 0, "records": len(chunk), "error": str(e)[:80]}

    t0 = time.monotonic()
    results = await asyncio.gather(*[one_request(c) for c in chunks])
    wall_time = time.monotonic() - t0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    total_tokens = sum(r["tokens"] for r in ok)
    total_ok_records = sum(r["records"] for r in ok)

    return {
        "wall_time": wall_time,
        "total_records": total_records,
        "ok_records": total_ok_records,
        "failed": len(fail),
        "total_tokens": total_tokens,
        "total_tps": total_tokens / wall_time if wall_time > 0 else 0,
        "rec_per_min": (total_ok_records / wall_time * 60) if wall_time > 0 else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    records = load_records(40)
    print(f"Loaded {len(records)} test records", flush=True)
    print(flush=True)

    # Test matrix — focused on what works: -np 1 (SYCL crashes with concurrent slots)
    # The win is batch-per-prompt: pack N records into one LLM call.
    configs = [
        # (np, context, concurrency, max_tokens, records_per_call, label)
        (1, 8192, 1, 1024, 1, "1 rec/call, 1024 tok"),
        (1, 8192, 1, 512, 1, "1 rec/call, 512 tok"),
        (1, 8192, 1, 2048, 3, "3 rec/call, 2048 tok"),
        (1, 8192, 1, 2048, 5, "5 rec/call, 2048 tok"),
        (1, 8192, 1, 3072, 10, "10 rec/call, 3072 tok"),
        (1, 8192, 1, 4096, 15, "15 rec/call, 4096 tok"),
        (1, 8192, 1, 4096, 20, "20 rec/call, 4096 tok"),
        # Try -np 2 just to confirm it crashes or works
        (2, 8192, 2, 1024, 1, "np2: 2 concurrent, 1 rec each"),
        (2, 8192, 1, 2048, 5, "np2: 1 req, 5 rec batch"),
    ]

    results = []
    current_np = None

    hdr = f"{'Config':<40} {'Wall(s)':>8} {'Rec':>4} {'Fail':>4} {'Tok':>6} {'t/s':>6} {'Rec/min':>8}"
    print(hdr, flush=True)
    print("=" * 82, flush=True)

    for np, ctx, conc, max_tok, rpc, label in configs:
        # Restart server if np changed
        if np != current_np:
            print(f"\n--- Restarting server: -np {np} -c {ctx} ---")
            if not start_server(np, ctx):
                print(f"  SKIP (server failed to start)")
                current_np = None
                continue
            current_np = np
            print(f"  Server ready")

        client = AsyncOpenAI(base_url=f"{SERVER_URL}/v1", api_key="na")

        # Run the test
        stats = await run_concurrent_requests(client, records, conc, max_tok, rpc)

        row = f"{label:<40} {stats['wall_time']:>7.1f}s {stats['ok_records']:>4} {stats['failed']:>4} {stats['total_tokens']:>6} {stats['total_tps']:>5.1f} {stats['rec_per_min']:>7.1f}"
        print(row, flush=True)
        results.append({"label": label, **stats})

    print("\n" + "=" * 82)
    print("DONE. Best configs by rec/min:")
    for r in sorted(results, key=lambda x: x["rec_per_min"], reverse=True)[:5]:
        print(f"  {r['rec_per_min']:>7.1f} rec/min — {r['label']}")

    stop_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        stop_server()
