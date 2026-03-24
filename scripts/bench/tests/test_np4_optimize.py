"""np=4 throughput optimization — A/B tests against the live proxy.

All tests hit localhost:11435 (the proxy). No bench-launched servers.
This guarantees the exact same env as production.

## How to use

1. Edit arcllm-proxy.py SYCL_ENV or model flags
2. Restart proxy: arcllm-server.sh stop && arcllm-server.sh start
3. Load model: curl proxy with model=qwen3-32b
4. Run: python3 -m bench np4optimize

## Results

| Phase | Change | 1x t/s | 4x t/s | Quality | Winner? |
|-------|--------|--------|--------|---------|---------|
| baseline | current config | | | | — |
| 1 | FUSED_MMQ=0 | | | | |
| 2 | CMDLISTS=2 | | | | |
| 3 | FA off | | | | |
| 4 | c=16384 | | | | |
| 5 | SYCL_CACHE_PERSISTENT=1 | | | | |
| 6 | -b 256 | | | | |
| 7 | combined winners | | | | |
"""

import json
import time
import threading

from bench.base import BenchTest
from bench.config import BenchConfig

PROXY = "http://localhost:11435/v1/chat/completions"
MODEL = "qwen3-32b"

CHURNER_PROMPT = (
    "Extract structured data from this DOM chunk for a wedding venue:\n"
    "<div class='venue-detail'><h1>LBJ Library Lawn</h1>"
    "<p class='desc'>Beautiful outdoor space on the UT campus with views of downtown Austin. "
    "Capacity for up to 200 guests. Features include natural lighting, covered pavilion, "
    "and on-site parking. Popular for weddings, receptions, and corporate events.</p>"
    "<div class='reviews'><div class='review'><span class='rating'>5</span>"
    "<p>Amazing venue, the sunset views were incredible for our ceremony.</p></div></div>"
    "<div class='pricing'><span class='price'>$3500</span><span class='period'>per event</span></div>"
    "</div>\n\nRespond with JSON: {name, description, capacity, price, rating}"
)


def _call(system, user, max_tokens=300, tools=None):
    import urllib.request
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        payload["tools"] = tools
    body = json.dumps(payload).encode()
    req = urllib.request.Request(PROXY, body, {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "_elapsed": time.time() - t0}
    data["_elapsed"] = time.time() - t0
    return data


def _content(d):
    return d.get("choices", [{}])[0].get("message", {}).get("content", "")


def _parse_json(content):
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        for o, c in [("{", "}"), ("[", "]")]:
            s = content.find(o)
            if s < 0:
                continue
            for e in range(len(content) - 1, s, -1):
                if content[e] == c:
                    try:
                        return json.loads(content[s:e + 1])
                    except json.JSONDecodeError:
                        continue
    return None


def _check_proxy():
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://localhost:11435/health", timeout=5)
        h = json.loads(resp.read())
        if not h.get("model"):
            print("  ✗ No model loaded")
            return False
        print(f"  proxy: model={h['model']} max_active={h['queue']['max_active']}")
        return True
    except Exception as e:
        print(f"  ✗ Proxy down: {e}")
        return False


def _throughput(n_concurrent, prompt=CHURNER_PROMPT, max_tokens=300):
    """Fire n_concurrent requests, return (ok_count, total_tokens, wall_time, agg_tps)."""
    results = [None] * n_concurrent

    def fire(idx):
        try:
            d = _call("Extract venue data as JSON.", prompt, max_tokens=max_tokens)
            toks = d.get("usage", {}).get("completion_tokens", 0)
            results[idx] = toks if toks > 0 and "error" not in d else f"ERR: {d.get('error', 'empty')}"
        except Exception as e:
            results[idx] = f"ERR: {e}"

    t0 = time.time()
    threads = [threading.Thread(target=fire, args=(i,)) for i in range(n_concurrent)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)
    wall = time.time() - t0

    ok = sum(1 for r in results if isinstance(r, int))
    total = sum(r for r in results if isinstance(r, int))
    tps = total / max(wall, 0.01)
    return ok, total, wall, tps


def _quality():
    """Run 3 quality checks. Returns (passed, failed) counts."""
    passed = 0

    # JSON extraction
    d = _call("Extract venue data as JSON: name, price, rating. ONLY JSON.", CHURNER_PROMPT, 500)
    r = _parse_json(_content(d))
    if r and "name" in r:
        passed += 1
    else:
        print(f"    ✗ JSON: {_content(d)[:100]}")

    # Tool calling
    tools = [{"type": "function", "function": {"name": "browser_open", "description": "Navigate to URL",
              "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}}]
    d = _call("Use browser tools to browse.", "Open https://example.com", tools=tools)
    tcs = d.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
    if tcs and tcs[0]["function"]["name"] == "browser_open":
        passed += 1
    else:
        print(f"    ✗ Tool: {_content(d)[:100]}")

    # Classification
    d = _call('Classify relevance. JSON: {"relevant": bool, "confidence": float}',
              "Entity: wedding venues. Site sells party supplies and lists 50 venues.", 200)
    r = _parse_json(_content(d))
    if r and "relevant" in r:
        passed += 1
    else:
        print(f"    ✗ Class: {_content(d)[:100]}")

    return passed, 3 - passed


def _run_phase(label):
    """Standard test protocol: throughput + quality. Runs sequentially to avoid gate contention."""
    if not _check_proxy():
        return

    # Quality first (sequential, single-slot)
    p, f = _quality()
    print(f"  quality: {p}/3 pass")

    # Single slot throughput
    ok1, tok1, wall1, tps1 = _throughput(1)
    print(f"  1x: {tok1} tok, {wall1:.1f}s, {tps1:.1f} t/s")

    # 4 concurrent throughput
    ok4, tok4, wall4, tps4 = _throughput(4)
    print(f"  4x: {ok4}/4 ok, {tok4} tok, {wall4:.1f}s, {tps4:.1f} t/s")

    print(f"  === {label}: 1x={tps1:.1f} 4x={tps4:.1f} quality={p}/3 ===")


class TestNp4Optimize(BenchTest):
    """np=4 optimization phases. Requires proxy running with qwen3-32b."""

    base = BenchConfig(name="np4opt_dummy")

    def test_baseline(self):
        """Current production config — establish baseline."""
        _run_phase("BASELINE")

    def test_all_phases(self):
        """Run all phases sequentially. Edit proxy between each, restart, then rerun."""
        _run_phase("CURRENT")
