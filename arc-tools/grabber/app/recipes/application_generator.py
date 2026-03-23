"""
Application Generator — produces hyper-personalized cover letters and tailored resumes.

Combines:
  - Company dossier (from dossier_builder.py / Grabber)
  - Evidence package (from git-chronicle /match endpoint)
  - Job posting requirements

Usage:
    from app.recipes.application_generator import generate_application
    result = generate_application(
        job_posting="...",
        company_name="Vercel",
        requirements=["React", "Node.js", "CI/CD"],
    )
    print(result["cover_letter"])
    print(result["resume_markdown"])
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from app.core.config import get_settings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_settings = get_settings()

# git-chronicle is discovered via Lantern domain 'chronicle.glow'.
# Override GIT_CHRONICLE_URL to target a specific host/port (e.g. in tests
# or when Lantern is not running).
GIT_CHRONICLE_URL = os.environ.get("GIT_CHRONICLE_URL", "http://chronicle.glow")
GHOSTGRAPH_URL = os.environ.get("GHOSTGRAPH_URL", _settings.api_base_url)

# LLM config — hardwired to Henry (local Qwen3.5-35B)
LLM_API_URL = os.environ.get("HENRY_URL", "http://localhost:11435/v1/chat/completions")
LLM_API_KEY = "none"
LLM_MODEL = os.environ.get("HENRY_MODEL", "qwen35")

# ---------------------------------------------------------------------------
# Humanize (inline, adapted from twitter-intel)
# ---------------------------------------------------------------------------

_BANNED_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bgame.?changer\b",
        r"\bparadigm shift\b",
        r"\blever(?:age|aging)\b",
        r"\bsynerg(?:y|ies|istic)\b",
        r"\bpassionate about\b",
        r"\bdeeply passionate\b",
        r"\bI'?m thrilled\b",
        r"\bI'?m excited to\b",
        r"\bin today'?s (?:fast-paced|rapidly evolving|dynamic)\b",
        r"\bdelve\b",
        r"\btap(?:ping)? into\b",
        r"\bunlock(?:ing)? (?:the |new )?\b",
        r"\bseamless(?:ly)?\b",
        r"\brobust\b",
        r"\bcutting.?edge\b",
        r"\binnovative solutions?\b",
        r"\bholistic\b",
        r"\bsupercharge\b",
        r"\b10x\b",
        r"\bthought leader\b",
    ]
]


def _humanize(text: str) -> str:
    """Strip obvious AI-isms from generated text."""
    for pat in _BANNED_PATTERNS:
        text = pat.sub("", text)
    # Collapse double spaces / orphan punctuation
    text = re.sub(r"  +", " ", text)
    text = re.sub(r" ,", ",", text)
    text = re.sub(r"\n ", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


def _fetch_dossier(company_name: str, project_id: Optional[str] = None) -> dict[str, Any]:
    """Fetch company dossier from Grabber's dossier builder.

    Falls back to importing locally if the API isn't running.
    """
    if project_id:
        try:
            resp = httpx.get(
                f"{GHOSTGRAPH_URL}/api/dossier/{project_id}",
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

    # Local fallback
    try:
        from app.recipes.dossier_builder import build_dossier
        dossier = build_dossier(project_id=project_id or company_name)
        return dossier.to_dict()
    except Exception as e:
        return {"company_name": company_name, "warnings": [f"Dossier unavailable: {e}"]}


def _fetch_evidence(requirements: list[str], topk: int = 5) -> dict[str, Any]:
    """Call git-chronicle /match to get evidence package."""
    try:
        resp = httpx.post(
            f"{GIT_CHRONICLE_URL}/match",
            json={"requirements": requirements, "topk": topk},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        return {"error": str(e), "results": []}
    return {"error": f"HTTP {resp.status_code}", "results": []}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _llm_generate(system_prompt: str, user_prompt: str) -> str:
    """Call LLM and return text response."""
    resp = httpx.post(
        LLM_API_URL,
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LLM API {resp.status_code}: {resp.text[:500]}")
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

COVER_LETTER_SYSTEM = """You write cover letters for Ryan, a senior full-stack developer.

Rules:
- 3-4 paragraphs max. No fluff.
- Reference 1-2 specific things about the company (a blog post, GitHub repo, recent launch, tech choice). Not more — informed, not obsessive.
- Map each key requirement to a specific project or contribution from Ryan's history. Use concrete details (repo names, what was built, measurable outcomes).
- Sound like a real human who did their research. Conversational but professional.
- NO AI-isms: no "passionate about", "thrilled", "leverage", "synergy", "in today's fast-paced", "delve", "cutting-edge".
- NO generic filler. Every sentence should carry information.
- End with something specific about why THIS company, not just any company.
- CRITICAL: Only mention projects, repos, and accomplishments listed in the evidence. NEVER invent projects, company names, or metrics that aren't in the provided data. If evidence is thin, keep it general but honest."""

COVER_LETTER_USER = """Write a cover letter for this job:

--- JOB POSTING ---
{job_posting}

--- COMPANY INTEL ---
{dossier_summary}

--- RYAN'S MATCHING EXPERIENCE ---
{evidence_summary}

Write the cover letter now. No preamble, no "Here's a cover letter" — just the letter itself."""

RESUME_SYSTEM = """You tailor resumes. Given a job posting and candidate history, reorder and rewrite the resume to emphasize what matters most for THIS role.

Rules:
- Mirror the job posting's terminology where honest (if they say "CI/CD", use "CI/CD" not "deployment automation")
- Lead with the most relevant experience
- Quantify where possible (users, requests/sec, uptime %, lines of code)
- Skills section should front-load what the posting asks for
- Output clean markdown suitable for PDF conversion
- Keep to 1 page equivalent (~600 words max)"""

RESUME_USER = """Tailor this resume for the job below.

--- JOB POSTING ---
{job_posting}

--- CANDIDATE EVIDENCE (from git history) ---
{evidence_summary}

--- BASE RESUME SECTIONS ---
Name: Ryan
Role: Senior Full-Stack Developer
Key areas: Elixir/OTP, Python, TypeScript/React, infrastructure, browser automation, ML pipelines

Output the tailored resume in markdown. No preamble."""


# ---------------------------------------------------------------------------
# Summarizers (keep prompts small)
# ---------------------------------------------------------------------------


def _summarize_dossier(dossier: dict[str, Any]) -> str:
    """Flatten dossier into a concise text block for the prompt."""
    parts = [f"Company: {dossier.get('company_name', 'Unknown')}"]
    if dossier.get("domain"):
        parts.append(f"Domain: {dossier['domain']}")
    if dossier.get("tech_stack"):
        techs = [t.get("name", t.get("technology", str(t))) for t in dossier["tech_stack"][:15]]
        parts.append(f"Tech stack: {', '.join(techs)}")
    if dossier.get("recent_activity"):
        activities = dossier["recent_activity"][:5]
        for a in activities:
            parts.append(f"- Recent: {a.get('title', a.get('description', str(a)))[:120]}")
    if dossier.get("blog_posts"):
        for bp in dossier["blog_posts"][:3]:
            parts.append(f"- Blog: {bp.get('title', '')} ({bp.get('url', '')})")
    if dossier.get("github_repos"):
        for r in dossier["github_repos"][:5]:
            parts.append(f"- Repo: {r.get('name', '')} — {r.get('description', '')[:80]}")
    if dossier.get("glassdoor_summary"):
        gs = dossier["glassdoor_summary"]
        if gs.get("rating"):
            parts.append(f"Glassdoor: {gs['rating']}/5")
    if dossier.get("warnings"):
        parts.append(f"(Warnings: {'; '.join(dossier['warnings'][:3])})")
    return "\n".join(parts)


def _summarize_evidence(evidence: dict[str, Any]) -> str:
    """Flatten git-chronicle match results into text for the prompt."""
    results = evidence.get("results", evidence.get("matches", []))
    if not results:
        return "(No matching evidence found — generate based on general experience)"

    parts = []
    for item in results:
        req = item.get("requirement", "")
        parts.append(f"\n## Requirement: {req}")
        for ev in item.get("evidence", [])[:3]:
            repo = ev.get("repo", "unknown")
            commits = ev.get("commits", 0)
            skills = ", ".join(ev.get("skills", []))
            summary = ev.get("summary", "")[:150]
            parts.append(f"- {repo} ({commits} commits, skills: {skills})")
            if summary:
                parts.append(f"  {summary}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass
class ApplicationResult:
    cover_letter: str
    resume_markdown: str
    dossier_summary: str
    evidence_summary: str
    confidence: float  # 0-1, based on evidence quality
    generation_time_s: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cover_letter": self.cover_letter,
            "resume_markdown": self.resume_markdown,
            "dossier_summary": self.dossier_summary,
            "evidence_summary": self.evidence_summary,
            "confidence": self.confidence,
            "generation_time_s": self.generation_time_s,
            "warnings": self.warnings,
        }


def _compute_confidence(evidence: dict[str, Any]) -> float:
    """Rough confidence score based on evidence coverage."""
    results = evidence.get("results", evidence.get("matches", []))
    if not results:
        return 0.2
    covered = sum(1 for r in results if r.get("evidence"))
    total = len(results) or 1
    return min(0.3 + 0.7 * (covered / total), 1.0)


def generate_application(
    job_posting: str,
    company_name: str,
    requirements: list[str],
    project_id: Optional[str] = None,
    topk: int = 5,
) -> ApplicationResult:
    """
    Generate a hyper-personalized cover letter and tailored resume.

    Args:
        job_posting: Full job posting text
        company_name: Company name for dossier lookup
        requirements: List of key requirements extracted from posting
        project_id: Optional Grabber project ID for dossier
        topk: Number of evidence items per requirement

    Returns:
        ApplicationResult with cover letter, resume, and metadata
    """
    t0 = time.time()
    warnings: list[str] = []

    # 1. Fetch inputs in parallel would be nice, but sequential is fine for now
    dossier = _fetch_dossier(company_name, project_id)
    evidence = _fetch_evidence(requirements, topk)

    if "error" in evidence:
        warnings.append(f"Evidence fetch: {evidence['error']}")

    dossier_text = _summarize_dossier(dossier)
    evidence_text = _summarize_evidence(evidence)
    confidence = _compute_confidence(evidence)

    # 2. Generate cover letter
    cover_prompt = COVER_LETTER_USER.format(
        job_posting=job_posting[:3000],
        dossier_summary=dossier_text,
        evidence_summary=evidence_text,
    )
    cover_letter = _llm_generate(COVER_LETTER_SYSTEM, cover_prompt)
    cover_letter = _humanize(cover_letter)

    # 3. Generate tailored resume
    resume_prompt = RESUME_USER.format(
        job_posting=job_posting[:3000],
        evidence_summary=evidence_text,
    )
    resume_md = _llm_generate(RESUME_SYSTEM, resume_prompt)

    elapsed = time.time() - t0

    return ApplicationResult(
        cover_letter=cover_letter,
        resume_markdown=resume_md,
        dossier_summary=dossier_text,
        evidence_summary=evidence_text,
        confidence=confidence,
        generation_time_s=round(elapsed, 2),
        warnings=warnings + dossier.get("warnings", []),
    )


# ---------------------------------------------------------------------------
# CLI quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    result = generate_application(
        job_posting="Senior Full-Stack Engineer at Vercel. Requirements: React, Next.js, TypeScript, CI/CD, distributed systems.",
        company_name="Vercel",
        requirements=["React", "Next.js", "TypeScript", "CI/CD", "distributed systems"],
    )
    print("=== COVER LETTER ===")
    print(result.cover_letter)
    print("\n=== RESUME ===")
    print(result.resume_markdown)
    print(f"\nConfidence: {result.confidence:.0%} | Time: {result.generation_time_s}s")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
