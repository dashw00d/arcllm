"""
Site recon — check blockers before Henry starts browsing.

Checks: robots.txt, IPv6 support, CDN/WAF detection, basic accessibility.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import ssl
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


async def run_recon(url: str) -> dict:
    """Run all recon checks on a URL. Returns a dict of findings."""
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path.split("/")[0]

    results = {
        "domain": domain,
        "url": url,
        "ipv6": await _check_ipv6(domain),
        "robots": await _check_robots(f"{parsed.scheme}://{domain}"),
        "headers": await _check_headers(url),
    }

    # Summarize blockers and warnings
    blockers = []  # hard blockers
    warnings = []  # things to be aware of
    if not results["ipv6"]["supported"]:
        warnings.append("no_ipv6 — will need proxy for grabber fleet")
    if results["robots"].get("disallowed"):
        warnings.append(f"robots_disallow: {results['robots']['disallowed'][:5]}")
    if results["headers"].get("waf"):
        warnings.append(f"waf: {results['headers']['waf']} — browser can bypass, curl may need stealth")
    if results["headers"].get("has_js_challenge"):
        warnings.append("js_challenge — needs real browser for initial access")

    results["blockers"] = blockers
    results["warnings"] = warnings
    results["can_scrape"] = len(blockers) == 0  # warnings don't block, they inform

    return results


async def _check_ipv6(domain: str) -> dict:
    """Check if domain has AAAA (IPv6) DNS records."""
    try:
        loop = asyncio.get_event_loop()
        addrs = await loop.getaddrinfo(domain, 443, family=socket.AF_INET6)
        ipv6_addrs = [a[4][0] for a in addrs]
        return {"supported": len(ipv6_addrs) > 0, "addresses": ipv6_addrs[:3]}
    except (socket.gaierror, OSError):
        return {"supported": False, "addresses": []}


async def _check_robots(base_url: str) -> dict:
    """Fetch and parse robots.txt for relevant disallow rules."""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(f"{base_url}/robots.txt")
            if resp.status_code != 200:
                return {"exists": False, "disallowed": []}

            text = resp.text
            disallowed = []
            for line in text.splitlines():
                line = line.strip().lower()
                if line.startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    if path and path != "/":
                        disallowed.append(path)

            return {
                "exists": True,
                "disallowed": disallowed[:10],  # top 10
                "crawl_delay": _extract_crawl_delay(text),
            }
    except Exception as e:
        return {"exists": False, "error": str(e), "disallowed": []}


def _extract_crawl_delay(robots_text: str) -> float | None:
    for line in robots_text.splitlines():
        if line.strip().lower().startswith("crawl-delay:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return None


async def _check_headers(url: str) -> dict:
    """Check response headers for CDN/WAF indicators."""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url)
            headers = dict(resp.headers)

            waf = None
            server = headers.get("server", "").lower()
            if "cloudflare" in server or "cf-ray" in headers:
                waf = "cloudflare"
            elif "akamai" in server or "akamaighost" in headers.get("x-cdn", "").lower():
                waf = "akamai"
            elif "fastly" in headers.get("via", "").lower():
                waf = "fastly"
            elif "sucuri" in server:
                waf = "sucuri"

            return {
                "status": resp.status_code,
                "server": headers.get("server", ""),
                "waf": waf,
                "content_type": headers.get("content-type", ""),
                "has_js_challenge": "challenge" in resp.text[:1000].lower() if resp.status_code == 200 else False,
            }
    except Exception as e:
        return {"status": None, "error": str(e), "waf": None}
