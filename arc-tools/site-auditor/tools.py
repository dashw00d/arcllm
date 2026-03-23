"""
agent-browser CLI wrappers + markdown fetcher.

Two categories:
- Browser tools (for Stage 2 discovery agent loop)
- Direct tools (for Stage 1 triage, Stage 3-4 patterning)
"""

import subprocess
import logging
import os

import httpx

logger = logging.getLogger(__name__)


def _run(args: list[str], timeout: int = 30) -> str:
    """Run an agent-browser command and return stdout."""
    cmd = ["agent-browser", "--args", "--no-sandbox"] + args
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0 and result.stderr:
            logger.warning("agent-browser error: %s", result.stderr.strip())
        return result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


# ── Browser tools (used in agent loop + direct calls) ────────────────────

def browser_open(url: str) -> str:
    """Navigate the browser to a URL."""
    return _run(["open", url], timeout=15)


def browser_snapshot() -> str:
    """Get a compact accessibility snapshot with @ref identifiers."""
    return _run(["snapshot", "-i"])


def browser_click(ref: str) -> str:
    """Click an element by @ref."""
    return _run(["click", ref])


def browser_scroll(direction: str = "down") -> str:
    """Scroll the page up or down."""
    return _run(["scroll", direction])


def browser_back() -> str:
    """Go back one page."""
    return _run(["back"])


def browser_get_text(ref: str) -> str:
    """Get the text content of an element by @ref."""
    return _run(["get", "text", ref])


def browser_get_html(ref: str = "body") -> str:
    """Get the outer HTML of an element (default: body)."""
    return _run(["get", "html", ref])


def browser_get_url() -> str:
    """Get the current page URL."""
    return _run(["get", "url"])


def browser_eval(js: str) -> str:
    """Execute JavaScript in the page context."""
    return _run(["eval", js], timeout=10)


def browser_close() -> str:
    """Close the browser."""
    return _run(["close"])


# ── Markdown fetcher (Stage 1 triage — no browser needed) ────────────────

def _html_to_text(html_text: str) -> str:
    """Convert HTML to plain text using stdlib only (no beautifulsoup)."""
    import re, html as html_module
    # Remove scripts, styles, SVGs
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", "", html_text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", "", text)
    text = re.sub(r"(?is)<svg[^>]*>.*?</svg>", "", text)
    # Decode HTML entities
    text = html_module.unescape(text)
    # Split on block-level tags to preserve paragraph breaks
    parts = re.split(r"(?i)</?(?:p|div|h[1-6]|br|li|tr|blockquote|pre|ul|ol|table|hr)[^>]*>", text)
    lines = [p.strip() for p in parts if p.strip()]
    # Remove remaining tags
    lines = [re.sub(r"<[^>]+>", "", line).strip() for line in lines]
    text = "\n".join(lines)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def get_markdown(url: str) -> str:
    """Fetch a page as clean markdown via local browser extraction.

    Strategy (in order):
    1. Browser + local HTML→text (most reliable for JS-heavy sites)
    2. Raw HTML fetch + local HTML→text (static sites)
    3. Jina Reader (last resort fallback)
    """
    # Method 1: Browser extraction
    try:
        opened = browser_open(url)
        if opened and not opened.startswith("ERROR"):
            html_content = browser_get_html("body")
            if html_content and len(html_content) > 100 and not html_content.startswith("ERROR"):
                text = _html_to_text(html_content)
                if text:
                    logger.info("Browser extraction got %d chars for %s", len(text), url)
                    return text[:4000]
    except Exception as e:
        logger.warning("Browser extraction failed for %s: %s", url, e)

    # Method 2: Raw HTML + local strip
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                text = _html_to_text(resp.text)
                if text and len(text) > 100:
                    logger.info("Raw HTML extraction got %d chars for %s", len(text), url)
                    return text[:4000]
    except Exception as e:
        logger.warning("Raw HTML fetch failed for %s: %s", url, e)

    # Method 3: Jina Reader (last resort)
    try:
        jina_url = f"https://r.jina.ai/{url}"
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(jina_url, headers={"Accept": "text/markdown"})
            if resp.status_code == 200 and len(resp.text) > 100:
                logger.info("Jina fallback got %d chars for %s", len(resp.text), url)
                return resp.text[:4000]
    except Exception as e:
        logger.warning("Jina Reader failed for %s: %s", url, e)

    return f"ERROR: could not fetch {url}"


# ── Skeletonize JS (Stage 3 patterning) ─────────────────────────────────

SKELETONIZE_JS = """
(() => {
  document.querySelectorAll('script, style, svg, path, noscript, iframe').forEach(e => e.remove());
  const nav = document.querySelectorAll('nav, footer, header, [role="navigation"], [role="banner"], [role="contentinfo"]');
  nav.forEach(e => { e.innerHTML = '<!-- ' + e.tagName.toLowerCase() + ' removed -->'; });
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while (node = walker.nextNode()) {
    const t = node.textContent.trim();
    if (t.length > 20) node.textContent = '...';
  }
  return document.body.innerHTML.substring(0, 8000);
})()
""".strip()


# ── Tool definitions for discovery agent loop (Stage 2 only) ─────────────

DISCOVERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browser_open",
            "description": "Navigate the browser to a URL",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_snapshot",
            "description": "Get compact page snapshot with @ref identifiers for clickable elements",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click element by @ref (e.g. @e1)",
            "parameters": {
                "type": "object",
                "properties": {"ref": {"type": "string"}},
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": "Scroll page up or down",
            "parameters": {
                "type": "object",
                "properties": {"direction": {"type": "string", "enum": ["up", "down"]}},
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_back",
            "description": "Go back to previous page",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_url",
            "description": "Get current page URL",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Task complete. Provide results as JSON string.",
            "parameters": {
                "type": "object",
                "properties": {"result": {"type": "string", "description": "JSON results"}},
                "required": ["result"],
            },
        },
    },
]

DISCOVERY_DISPATCH = {
    "browser_open": browser_open,
    "browser_snapshot": browser_snapshot,
    "browser_click": browser_click,
    "browser_scroll": browser_scroll,
    "browser_back": browser_back,
    "browser_get_url": browser_get_url,
}
