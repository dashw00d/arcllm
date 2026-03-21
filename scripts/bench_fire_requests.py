#!/usr/bin/env python3
"""Fire N concurrent LLM requests and report timings.

Uses asyncio + aiohttp to send parallel chat completion requests — this is
what the churner actually does, so it's the real-world test.

Usage:
    python3 bench_fire_requests.py \
        --url http://127.0.0.1:8400 \
        --concurrent 4 \
        --max-tokens 200 \
        [--model qwen3-32b] \
        [--prompt "your prompt"] \
        [--records-per-call 1] \
        [--timeout 300]

Output (JSON to stdout):
    {
        "concurrent": 4,
        "completed": 4,
        "failed": 0,
        "wall_time_s": 12.3,
        "total_tokens": 800,
        "total_tps": 65.0,
        "per_request": [
            {"idx": 0, "tokens": 200, "time_s": 11.1, "tps": 18.0, "status": "ok"},
            ...
        ]
    }
"""

import argparse
import asyncio
import json
import sys
import time

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install: pip install aiohttp", file=sys.stderr)
    sys.exit(1)


DEFAULT_PROMPTS = [
    "Explain the difference between TCP and UDP in networking. Include when you would use each.",
    "Write a Python function that finds all prime numbers up to N using the Sieve of Eratosthenes. Include type hints.",
    "Describe the process of photosynthesis in detail, including the light-dependent and light-independent reactions.",
    "What are the key differences between SQL and NoSQL databases? Give specific examples of when to use each.",
    "Explain how a compiler works, from source code to machine code. Cover lexing, parsing, and code generation.",
    "Write a detailed explanation of how hash tables work, including collision resolution strategies.",
    "Describe the architecture of a modern CPU including pipelines, caches, and branch prediction.",
    "Explain the CAP theorem in distributed systems and its practical implications for system design.",
    "Write a comprehensive guide to Git rebasing vs merging, with examples of when to use each.",
    "Describe how TLS/SSL handshake works, step by step, including certificate verification.",
    "Explain memory management in Rust, including ownership, borrowing, and lifetimes.",
    "Describe the differences between processes and threads, including how context switching works.",
    "Explain how DNS resolution works, from typing a URL to getting an IP address.",
    "Write about the different types of load balancing algorithms and when to use each.",
    "Describe how garbage collection works in JVM-based languages, including generational GC.",
    "Explain the architecture of a modern web browser rendering engine.",
    "Describe how B-trees work and why they're used in databases and file systems.",
    "Explain the principles behind eventual consistency and how CRDTs help solve conflicts.",
    "Describe how virtual memory works, including page tables and TLB.",
    "Explain the key concepts of functional programming and how they compare to OOP.",
]

BATCH_TEMPLATE = """Analyze the following records and provide a brief summary for each:

{records}

For each record, provide:
1. Key topic
2. One-sentence summary
3. Complexity rating (1-5)"""


def make_prompt(idx: int, records_per_call: int) -> str:
    """Generate a prompt. If records_per_call > 1, pack multiple into one."""
    if records_per_call <= 1:
        return DEFAULT_PROMPTS[idx % len(DEFAULT_PROMPTS)]

    # Batch mode: pack N records into one prompt
    records = []
    for r in range(records_per_call):
        prompt_idx = (idx * records_per_call + r) % len(DEFAULT_PROMPTS)
        records.append(f"Record {r+1}: {DEFAULT_PROMPTS[prompt_idx]}")
    return BATCH_TEMPLATE.format(records="\n\n".join(records))


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    idx: int,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> dict:
    """Send one chat completion request, return timing info."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }

    start = time.monotonic()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            body = await resp.json()
            elapsed = time.monotonic() - start

            if resp.status != 200:
                error = body.get("error", {}).get("message", str(body))
                return {
                    "idx": idx,
                    "tokens": 0,
                    "time_s": round(elapsed, 2),
                    "tps": 0,
                    "status": f"error:{resp.status}",
                    "error": error,
                }

            usage = body.get("usage", {})
            tokens = usage.get("completion_tokens", 0)
            tps = tokens / elapsed if elapsed > 0 else 0

            return {
                "idx": idx,
                "tokens": tokens,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "time_s": round(elapsed, 2),
                "tps": round(tps, 1),
                "status": "ok",
            }
    except asyncio.TimeoutError:
        return {
            "idx": idx,
            "tokens": 0,
            "time_s": round(time.monotonic() - start, 2),
            "tps": 0,
            "status": "timeout",
        }
    except Exception as e:
        return {
            "idx": idx,
            "tokens": 0,
            "time_s": round(time.monotonic() - start, 2),
            "tps": 0,
            "status": f"error:{e}",
        }


async def run_benchmark(
    url: str,
    concurrent: int,
    model: str,
    max_tokens: int,
    timeout: int,
    records_per_call: int,
    custom_prompt: str | None,
):
    """Fire N concurrent requests and collect results."""
    prompts = []
    for i in range(concurrent):
        if custom_prompt:
            prompts.append(custom_prompt)
        else:
            prompts.append(make_prompt(i, records_per_call))

    async with aiohttp.ClientSession() as session:
        wall_start = time.monotonic()
        tasks = [
            send_request(session, url, i, model, prompts[i], max_tokens, timeout)
            for i in range(concurrent)
        ]
        results = await asyncio.gather(*tasks)
        wall_time = time.monotonic() - wall_start

    completed = sum(1 for r in results if r["status"] == "ok")
    failed = concurrent - completed
    total_tokens = sum(r["tokens"] for r in results)
    total_tps = total_tokens / wall_time if wall_time > 0 else 0

    output = {
        "concurrent": concurrent,
        "completed": completed,
        "failed": failed,
        "wall_time_s": round(wall_time, 2),
        "total_tokens": total_tokens,
        "total_tps": round(total_tps, 1),
        "records_per_call": records_per_call,
        "per_request": sorted(results, key=lambda r: r["idx"]),
    }
    return output


def main():
    parser = argparse.ArgumentParser(description="Fire concurrent LLM requests")
    parser.add_argument("--url", default="http://127.0.0.1:8400", help="Server base URL")
    parser.add_argument("--concurrent", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--model", default="qwen3-32b", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens per response")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout (seconds)")
    parser.add_argument("--records-per-call", type=int, default=1,
                        help="Pack N records into each prompt (for batch testing)")
    parser.add_argument("--prompt", default=None, help="Custom prompt (overrides defaults)")
    args = parser.parse_args()

    result = asyncio.run(run_benchmark(
        url=args.url,
        concurrent=args.concurrent,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        records_per_call=args.records_per_call,
        custom_prompt=args.prompt,
    ))

    # Print results as JSON to stdout
    print(json.dumps(result, indent=2))

    # Also print a human summary to stderr
    print(f"\n--- Summary ---", file=sys.stderr)
    print(f"Concurrent: {result['concurrent']}", file=sys.stderr)
    print(f"Completed:  {result['completed']}/{result['concurrent']}", file=sys.stderr)
    print(f"Wall time:  {result['wall_time_s']:.1f}s", file=sys.stderr)
    print(f"Total tok:  {result['total_tokens']}", file=sys.stderr)
    print(f"Total t/s:  {result['total_tps']:.1f}", file=sys.stderr)
    for r in result["per_request"]:
        status = r["status"]
        if status == "ok":
            print(f"  req[{r['idx']}]: {r['tokens']} tok in {r['time_s']:.1f}s = {r['tps']:.1f} t/s",
                  file=sys.stderr)
        else:
            print(f"  req[{r['idx']}]: {status} ({r['time_s']:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
