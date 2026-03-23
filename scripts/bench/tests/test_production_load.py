"""Production pipeline test — validates the LIVE proxy config end-to-end.

Hits the proxy at localhost:11435, NOT a bench-launched server.
This guarantees the exact same env, flags, and model config as production.
No more env mismatches.

## What it tests

1. JSON extraction (churner) — structured output from HTML
2. Tool calling (auditor discovery) — multi-tool selection
3. Classification (triage) — relevant/not-relevant with confidence
4. Context capacity — can a slot handle a real agent loop (~1500 tok)
5. 4x concurrent — all slots active simultaneously

## Prerequisites

    bash scripts/arcllm-server.sh start
    bash scripts/arcllm-server.sh load qwen3-32b

## How to run

    cd /home/ryan/llm-stack/scripts
    python3 -m bench productionload              # all tests
    python3 -m bench productionload.concurrent_4x  # specific test

## Results

(Fill in after running)
"""

import json
import time
import threading

PROXY_URL = "http://localhost:11435/v1/chat/completions"
MODEL = "qwen3-32b"


def _call(system: str, user: str, max_tokens: int = 300, tools=None) -> dict:
    import urllib.request
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        payload["tools"] = tools
    body = json.dumps(payload).encode()
    req = urllib.request.Request(PROXY_URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    data["_elapsed"] = time.time() - t0
    return data


def _content(data: dict) -> str:
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def _tool_calls(data: dict) -> list:
    return data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])


def _parse_json(content: str) -> dict | None:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        for opener, closer in [("{", "}"), ("[", "]")]:
            start = content.find(opener)
            if start < 0:
                continue
            for end in range(len(content) - 1, start, -1):
                if content[end] == closer:
                    try:
                        return json.loads(content[start:end + 1])
                    except json.JSONDecodeError:
                        continue
    return None


def _check_proxy():
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://localhost:11435/health", timeout=5)
        health = json.loads(resp.read())
        model = health.get("model")
        if not model:
            print("  ✗ No model loaded. Run: arcllm-server.sh load qwen3-32b")
            return False
        print(f"  Proxy: model={model} max_active={health['queue']['max_active']}")
        return True
    except Exception as e:
        print(f"  ✗ Proxy not running: {e}")
        return False


# ── Tests ──────────────────────────────────────────────────────────────

from bench.base import BenchTest
from bench.config import BenchConfig


class TestProductionLoad(BenchTest):
    """Pipeline validation against the live proxy. Does NOT launch a server."""

    # Dummy config — bench framework won't launch a server for these
    base = BenchConfig(name="productionload_dummy")

    def test_json_extraction(self):
        """Churner: extract structured JSON from HTML venue data."""
        if not _check_proxy():
            return

        html = """<div class='venue'><h2>Barton Creek Resort</h2>
        <p>Luxury resort with golf course and spa. Indoor and outdoor spaces.
        Capacity: 50-500 guests. Full catering available.</p>
        <div class='pricing'>Starting at $5,000 for 4 hours</div>
        <div class='rating'>4.7/5 (128 reviews)</div>
        <div class='contact'><a href='tel:5125551234'>512-555-1234</a>
        <a href='mailto:events@bartoncreek.com'>events@bartoncreek.com</a></div></div>"""

        data = _call(
            system="Extract venue data as JSON: name, description, capacity_min, capacity_max, price, rating, review_count, phone, email. ONLY valid JSON.",
            user=f"Extract from:\n{html}",
            max_tokens=500,
        )
        content = _content(data)
        result = _parse_json(content)
        toks = data.get("usage", {}).get("completion_tokens", 0)
        elapsed = data["_elapsed"]

        ok = result and "name" in result
        print(f"  {'✓' if ok else '✗'} JSON extraction: {toks} tok, {elapsed:.1f}s")
        if ok:
            print(f"    name={result.get('name')}, price={result.get('price')}, rating={result.get('rating')}")
        else:
            print(f"    output: {content[:200]}")

    def test_tool_calling(self):
        """Auditor discovery: select correct browser tool."""
        if not _check_proxy():
            return

        tools = [
            {"type": "function", "function": {"name": "browser_open", "description": "Navigate to URL", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
            {"type": "function", "function": {"name": "browser_snapshot", "description": "Get page snapshot", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "browser_click", "description": "Click element by ref", "parameters": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}}},
            {"type": "function", "function": {"name": "done", "description": "Task complete with JSON result", "parameters": {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]}}},
        ]

        data = _call(
            system="You browse websites to find venue listings. Use browser tools.",
            user="Find venue listings on https://wehelpyouparty.com. Start by opening the site.",
            tools=tools,
        )
        tcs = _tool_calls(data)
        toks = data.get("usage", {}).get("completion_tokens", 0)
        elapsed = data["_elapsed"]

        ok = tcs and tcs[0]["function"]["name"] == "browser_open"
        print(f"  {'✓' if ok else '✗'} Tool calling: {toks} tok, {elapsed:.1f}s")
        if ok:
            url = json.loads(tcs[0]["function"]["arguments"]).get("url", "")
            print(f"    browser_open({url})")
        else:
            print(f"    output: {_content(data)[:200]}")

    def test_classification(self):
        """Triage: classify site relevance with confidence."""
        if not _check_proxy():
            return

        data = _call(
            system='Classify site relevance. JSON: {"relevant": true/false, "confidence": 0.0-1.0, "reasoning": "why"}',
            user="Entity: wedding venues\nSite: wehelpyouparty.com\nContent: We Help You Party offers catering, staffing, and venue coordination for events in Austin TX. Browse our venue directory with 50+ locations.",
            max_tokens=200,
        )
        content = _content(data)
        result = _parse_json(content)
        toks = data.get("usage", {}).get("completion_tokens", 0)
        elapsed = data["_elapsed"]

        ok = result and "relevant" in result
        print(f"  {'✓' if ok else '✗'} Classification: {toks} tok, {elapsed:.1f}s")
        if ok:
            print(f"    relevant={result['relevant']}, confidence={result.get('confidence')}")
        else:
            print(f"    output: {content[:200]}")

    def test_context_capacity(self):
        """Slot handles full agent loop: ~1500 tok input."""
        if not _check_proxy():
            return

        big_snapshot = "- heading 'Venues' [level=1]\n" + "".join(
            f"- link 'Venue {i}' [ref=e{i}]\n" for i in range(80)
        )

        import urllib.request
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are finding venue listing pages. Use browser tools."},
                {"role": "user", "content": f"[CONTEXT]\nVisited: /venues, /venues/outdoor\nFindings: /venues has 3 cards, /venues/outdoor has 5\n\nSnapshot:\n{big_snapshot}\n\nContinue. Call done when finished."},
            ],
            "max_tokens": 300, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(PROXY_URL, body, {"Content-Type": "application/json"})
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        elapsed = time.time() - t0

        prompt_tok = data.get("usage", {}).get("prompt_tokens", 0)
        comp_tok = data.get("usage", {}).get("completion_tokens", 0)

        ok = comp_tok > 0
        print(f"  {'✓' if ok else '✗'} Context: {prompt_tok} prompt + {comp_tok} gen tok, {elapsed:.1f}s")
        print(f"    slot limit=8192, used={prompt_tok + comp_tok}")

    def test_concurrent_4x(self):
        """All 4 slots active — pipeline + discord simultaneously."""
        if not _check_proxy():
            return

        tasks = [
            ("Classify: about food? JSON {\"about_food\": bool}", "Pizza is delicious"),
            ("Extract name+price as JSON", "<div><h1>Blue Bonnet Hall</h1><span>$2000</span></div>"),
            ("Summarize in 1 sentence", "The venue has a 5000 sq ft ballroom with crystal chandeliers, hardwood floors, and a built-in stage."),
            ("List 3 pros, 3 cons as JSON", "Outdoor venue, great views, no cover, free parking, 30 min from downtown, seats 200"),
        ]

        results = [None] * 4

        def fire(idx):
            try:
                data = _call(system=tasks[idx][0], user=tasks[idx][1], max_tokens=150)
                results[idx] = (data.get("usage", {}).get("completion_tokens", 0), data["_elapsed"])
            except Exception as e:
                results[idx] = f"ERR: {e}"

        t0 = time.time()
        threads = [threading.Thread(target=fire, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0

        ok_count = sum(1 for r in results if isinstance(r, tuple))
        total_tok = sum(r[0] for r in results if isinstance(r, tuple))

        print(f"  {'✓' if ok_count == 4 else '✗'} Concurrent: {ok_count}/4 ok, {total_tok} tok, {wall:.1f}s, {total_tok / max(wall, 0.01):.1f} t/s")
        for i, r in enumerate(results):
            if isinstance(r, tuple):
                print(f"    slot{i}: {r[0]} tok, {r[1]:.1f}s")
            else:
                print(f"    slot{i}: {r}")
