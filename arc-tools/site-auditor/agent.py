"""
LLM interface for the site auditor.

Two modes:
- henry_call(): single-shot LLM call for Stages 1, 3, 4 (no tool loop)
- run_agent(): tool-calling agent loop for Stage 2 discovery
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from tools import DISCOVERY_TOOLS, DISCOVERY_DISPATCH
from context import compress_snapshot

logger = logging.getLogger(__name__)

HENRY_URL = os.environ.get("HENRY_URL", "http://localhost:11435/v1/chat/completions")
HENRY_MODEL = os.environ.get("HENRY_MODEL", "qwen35")


# ── Single-shot call (Stages 1, 3, 4) ────────────────────────────────────

async def henry_call(system: str, user: str, max_tokens: int = 1024) -> dict | None:
    """Single LLM call, no tool loop. Returns parsed JSON or None."""
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            HENRY_URL,
            headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
            json={
                "model": HENRY_MODEL,
                "messages": [
                    {"role": "system", "content": system + "\n\nRespond with ONLY valid JSON."},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        data = resp.json()

        if "error" in data or "choices" not in data:
            err = data.get("error", {}).get("message", str(data))[:200]
            logger.error("henry_call error: %s", err)
            return None

        content = data["choices"][0]["message"].get("content", "")
        # Strip reasoning/thinking if present
        if "<think>" in content:
            content = content.split("</think>")[-1].strip()

        # Parse JSON from response — abliterated models leak thinking text
        # before the JSON, so we need to find and extract the JSON object.
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Find the first { or [ and try progressively shorter substrings
            for opener, closer in [("{", "}"), ("[", "]")]:
                start = content.find(opener)
                if start < 0:
                    continue
                # Try from the last closer back to first closer
                for end in range(len(content) - 1, start, -1):
                    if content[end] == closer:
                        try:
                            return json.loads(content[start:end + 1])
                        except json.JSONDecodeError:
                            continue
            logger.warning("Could not parse JSON from response: %s", content[:200])
            return {"raw": content}


# ── Agent loop (Stage 2 discovery) ────────────────────────────────────────

async def run_agent(
    system_prompt: str,
    user_task: str,
    max_steps: int = 25,
    temperature: float = 0.1,
) -> dict | None:
    """Run Henry with browser tools. Fresh context — no history from other stages."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_task},
    ]

    async with httpx.AsyncClient(timeout=300) as client:
        for step in range(max_steps):
            logger.info("Step %d/%d", step + 1, max_steps)

            # Trim when approaching context limit (~16k slot)
            # Each exchange is ~1800 tokens, so ~20 messages = ~8k tokens (safe for 16k)
            if len(messages) > 24:
                # Keep system + user task + recent exchanges
                # Find safe cut: start of a user message or tool-free assistant
                keep = messages[-12:]
                while keep and keep[0].get("role") == "tool":
                    keep.pop(0)
                while keep and keep[0].get("role") == "assistant" and keep[0].get("tool_calls"):
                    keep.pop(0)
                if not keep:
                    keep = [{"role": "user", "content": "Continue browsing and call 'done' when you have results."}]
                messages = [messages[0]] + keep
                logger.info("  Trimmed to %d messages", len(messages))

            resp = await client.post(
                HENRY_URL,
                headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
                json={
                    "model": HENRY_MODEL,
                    "messages": messages,
                    "tools": DISCOVERY_TOOLS,
                    "temperature": temperature,
                    "max_tokens": 2048,
                },
            )
            data = resp.json()

            if "error" in data or "choices" not in data:
                err = data.get("error", {}).get("message", str(data))[:200]
                logger.warning("LLM error at step %d: %s", step + 1, err)
                # Trim harder — keep only last user message to restart cleanly
                last_user = [m for m in messages if m.get("role") == "user"]
                if last_user:
                    messages = [messages[0], last_user[-1]]
                else:
                    messages = [messages[0], {"role": "user", "content": "Continue with the task using browser tools."}]
                continue

            choice = data["choices"][0]
            msg = choice["message"]
            messages.append(msg)

            # Text response — nudge Henry to use tools
            if choice["finish_reason"] != "tool_calls" or not msg.get("tool_calls"):
                text = msg.get("content", "")
                if text:
                    logger.info("Henry thinking: %s", text[:100])
                    messages.append({
                        "role": "user",
                        "content": "Use the browser tools to continue. Call 'done' when finished.",
                    })
                continue

            # Execute tool calls
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                call_id = tc["id"]

                if fn_name == "done":
                    logger.info("Agent done at step %d", step + 1)
                    try:
                        return json.loads(fn_args.get("result", "{}"))
                    except json.JSONDecodeError:
                        return {"raw": fn_args.get("result", "")}

                fn = DISCOVERY_DISPATCH.get(fn_name)
                if not fn:
                    tool_result = f"ERROR: unknown tool '{fn_name}'"
                else:
                    logger.info("  %s(%s)", fn_name, json.dumps(fn_args)[:80])
                    tool_result = fn(**fn_args)

                # Compress snapshots
                if fn_name == "browser_snapshot":
                    tool_result = compress_snapshot(tool_result)

                messages.append({
                    "role": "tool",
                    "content": tool_result[:3000],
                    "tool_call_id": call_id,
                })

        # Final attempt: force a summary with a clean context
        logger.info("Max steps reached — forcing summary")
        summary_msgs = [messages[0], {
            "role": "user",
            "content": "You've run out of browsing steps. Call 'done' NOW with your best results based on what you observed.",
        }]
        resp = await client.post(
            HENRY_URL,
            headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
            json={"model": HENRY_MODEL, "messages": summary_msgs, "tools": DISCOVERY_TOOLS, "temperature": 0.1, "max_tokens": 2048},
        )
        data = resp.json()
        if "choices" in data:
            msg = data["choices"][0]["message"]
            for tc in msg.get("tool_calls", []):
                if tc["function"]["name"] == "done":
                    try:
                        return json.loads(tc["function"]["arguments"].get("result", "{}"))
                    except (json.JSONDecodeError, AttributeError):
                        pass

    logger.warning("Agent could not complete discovery")
    return None
