"""
Context management for the agent loop.

Handles:
1. Snapshot compression — strip noise, group similar elements
2. Rolling state — maintain a structured summary of what's been learned
3. History compaction — replace old tool exchanges with summary
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Noise patterns to strip from snapshots
NOISE_PATTERNS = [
    r".*cookie.*",
    r".*consent.*",
    r".*privacy.*policy.*",
    r".*terms.*service.*",
    r".*copyright.*",
    r".*social.*media.*",
    r".*facebook.*",
    r".*twitter.*",
    r".*instagram.*",
    r".*linkedin.*",
    r".*pinterest.*",
    r".*youtube.*",
    r".*tiktok.*",
    r".*subscribe.*newsletter.*",
    r".*sign.?up.*email.*",
    r".*app.?store.*",
    r".*google.?play.*",
    r".*download.*app.*",
]
_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def compress_snapshot(raw: str, max_refs: int = 60) -> str:
    """Compress a browser snapshot by stripping noise and grouping repetitive elements."""
    lines = raw.strip().splitlines()
    if not lines:
        return raw

    # Pass 1: strip noise lines
    clean = []
    stripped = 0
    for line in lines:
        if any(r.search(line) for r in _NOISE_RE):
            stripped += 1
            continue
        clean.append(line)

    # Pass 2: group repetitive elements (e.g. 50 similar links)
    if len(clean) > max_refs:
        groups = _group_similar(clean)
        compressed = []
        for group in groups:
            if len(group) <= 3:
                compressed.extend(group)
            else:
                # Show first 2, summarize rest
                compressed.append(group[0])
                compressed.append(group[1])
                compressed.append(f"  ... (+{len(group) - 2} similar elements)")
        clean = compressed

    if stripped > 0:
        clean.append(f"({stripped} noise elements removed: cookies, social, footer)")

    return "\n".join(clean)


def _group_similar(lines: list[str]) -> list[list[str]]:
    """Group consecutive lines with similar structure."""
    groups: list[list[str]] = []
    current_group: list[str] = []
    current_pattern = ""

    for line in lines:
        # Extract structural pattern (ignore ref numbers and specific text)
        pattern = re.sub(r'ref=e\d+', 'ref=eN', line)
        pattern = re.sub(r'"[^"]{10,}"', '"..."', pattern)

        if pattern == current_pattern:
            current_group.append(line)
        else:
            if current_group:
                groups.append(current_group)
            current_group = [line]
            current_pattern = pattern

    if current_group:
        groups.append(current_group)

    return groups


@dataclass
class AgentState:
    """Rolling state that tracks what the agent has learned."""
    pages_visited: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    current_url: str = ""

    def add_visit(self, url: str, observation: str):
        if url and url not in self.pages_visited:
            self.pages_visited.append(url)
        if observation:
            self.findings.append(observation)

    def to_context(self) -> str:
        """Generate a compact context string for the system prompt."""
        parts = []
        if self.pages_visited:
            parts.append(f"Pages visited: {', '.join(self.pages_visited[-5:])}")
        if self.findings:
            parts.append("Key findings so far:")
            for f in self.findings[-8:]:
                parts.append(f"  - {f}")
        if self.current_url:
            parts.append(f"Currently on: {self.current_url}")
        return "\n".join(parts) if parts else "No browsing history yet."


def compact_messages(
    messages: list[dict],
    state: AgentState,
    keep_recent: int = 6,
) -> list[dict]:
    """Replace old messages with a rolling state summary.

    Keeps: system prompt + state summary + last N messages.
    """
    if len(messages) <= keep_recent + 2:
        return messages

    system = messages[0]

    # Inject state as a context message right after system
    state_msg = {
        "role": "user",
        "content": f"[BROWSING CONTEXT]\n{state.to_context()}\n\nContinue with the task.",
    }

    # Keep last N messages (recent tool calls + results)
    recent = messages[-keep_recent:]

    result = [system, state_msg] + recent
    logger.debug("Compacted %d messages → %d (kept %d recent)", len(messages), len(result), keep_recent)
    return result
