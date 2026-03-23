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

# Token estimation: ~4 chars/token
CHARS_PER_TOKEN = 4
# Target trim point: 12k tokens = 48k chars (of 16k = 64k char slot)
TOKEN_BUDGET = 12_000
CHAR_BUDGET = TOKEN_BUDGET * CHARS_PER_TOKEN
# Aggressive trim when approaching budget
TRIM_THRESHOLD_CHARS = int(CHAR_BUDGET * 0.8)


def _estimate_tokens(messages: list[dict]) -> int:
    """Estimate total tokens in the message list."""
    total = 0
    for m in messages:
        total += 50  # role/overhead per message
        total += len(str(m.get("content", "")))
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                total += len(str(tc))
    return total // CHARS_PER_TOKEN


def _find_safe_trim_point(messages: list[dict]) -> list[dict]:
    """Find a safe place to trim messages, preserving tool_call/tool_result pairs.

    Strategy:
    - Always keep: system message (index 0) + original user task (index 1)
    - Walk backward from the end, collecting complete exchanges
    - Never cut in the middle of an assistant+tool_result pair
    - Compress old tool results to summaries instead of dropping entirely
    """
    if len(messages) <= 4:
        return messages

    system = messages[:1]
    user_task = messages[1:2]  # original user task

    # Build safe tail walking backward: collect messages in reverse order,
    # then reverse the list at the end to get correct chronological order.
    # We track "open tool_calls" — assistant messages whose tool results
    # we haven't seen yet. When we encounter a tool result, we pair it with
    # the oldest open tool_call.
    tail_reversed = []  # accumulated in reverse order
    open_tool_calls = []  # stack of assistant msgs with pending tool results
    i = len(messages) - 1

    while i >= 2:  # stop before original user task
        msg = messages[i]
        role = msg.get("role")

        if role == "tool":
            # Tool result — pair with the most recent open tool_call
            if open_tool_calls:
                # Pop the most recent assistant with tool_calls
                assistant_msg = open_tool_calls.pop()
                # Add tool result first (reversed), then assistant (so order is correct after final reverse)
                tail_reversed.append(msg)
                tail_reversed.append(assistant_msg)
            # else: orphaned tool result — drop it
            i -= 1
        elif role == "assistant" and msg.get("tool_calls"):
            # Assistant with tool_calls — push onto stack; we'll pair when we see tool results
            open_tool_calls.append(msg)
            i -= 1
        elif role == "user" or (role == "assistant" and not msg.get("tool_calls")):
            # Safe to cut: user or tool-free assistant
            # First, flush any open tool_calls as summaries
            if open_tool_calls:
                for atc in reversed(open_tool_calls):
                    num_calls = len(atc.get("tool_calls", []))
                    summary = {"role": "user", "content": f"[Previous: {num_calls} tool call(s)]"}
                    tail_reversed.append(summary)
                open_tool_calls = []
            tail_reversed.append(msg)
            i -= 1
        else:
            i -= 1

    # Flush remaining open tool_calls
    if open_tool_calls:
        for atc in reversed(open_tool_calls):
            num_calls = len(atc.get("tool_calls", []))
            summary = {"role": "user", "content": f"[Previous: {num_calls} tool call(s)]"}
            tail_reversed.append(summary)

    # Reverse to get correct chronological order
    tail = list(reversed(tail_reversed))

    # If tail is too long, take only the last N messages
    MAX_TAIL = 14  # ~7 complete exchanges, well within token budget
    if len(tail) > MAX_TAIL:
        tail = tail[-MAX_TAIL:]

    result = system + user_task + tail
    logger.info("  Trimmed %d messages → %d (est. %d tokens)",
                 len(messages), len(result), _estimate_tokens(result))
    return result


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
            logger.info("Step %d/%d (est. %d tokens)", step + 1, max_steps, _estimate_tokens(messages))

            # Trim when approaching token budget
            if _estimate_tokens(messages) > TRIM_THRESHOLD_CHARS // CHARS_PER_TOKEN:
                messages = _find_safe_trim_point(messages)
                # If still over budget, force aggressive trim
                if _estimate_tokens(messages) > TRIM_THRESHOLD_CHARS // CHARS_PER_TOKEN:
                    messages = [messages[0], messages[1]] + messages[-12:]

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

            # Handle context-too-long errors (HTTP 400 with context/length in error)
            if "error" in data:
                err = data["error"]
                err_msg = err.get("message", "") if isinstance(err, dict) else str(err)
                err_type = err.get("type", "") if isinstance(err, dict) else ""
                if "context" in err_msg.lower() or "length" in err_msg.lower() or err_type == "context_too_long":
                    logger.warning("Context too long at step %d — trimming to last 6 messages and retrying", step + 1)
                    # Keep system + last 6 messages
                    messages = [messages[0]] + messages[-6:]
                    continue
                logger.warning("LLM error at step %d: %s", step + 1, err_msg[:200])
                # Trim hard and retry
                last_user = [m for m in messages if m.get("role") == "user"]
                if len(last_user) >= 2:
                    messages = [messages[0], last_user[-1]]
                else:
                    messages = [messages[0], {"role": "user", "content": "Continue browsing and call 'done' when you have results."}]
                continue

            if "choices" not in data:
                logger.warning("No choices at step %d: %s", step + 1, str(data)[:200])
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
