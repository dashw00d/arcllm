"""
Dossier Builder — aggregates Grabber entity data into a company intelligence dossier.

After company_intel recipes run and store entities, this module queries
the extracted entities by project and assembles a structured dossier with:
  - Tech stack (from GitHub repos + blog mentions + website signals)
  - Recent activity (commits, blog posts in last 90 days)
  - Employee sentiment (Glassdoor summary)
  - Key people (blog authors, active GitHub contributors)

Usage:
    from app.recipes.dossier_builder import build_dossier
    dossier = build_dossier(project_id="...")
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@contextmanager
def _get_cursor():
    """Direct DB connection to avoid circular import issues with the full app."""
    import psycopg2
    import psycopg2.extras
    dsn = os.environ.get("DATABASE_URL", "postgres://postgres:R7Huut2sN5sRpJrZztPE4PbQZBCXdi7WltnPJckTrCdFlqiB8SgE8RtU36ZGNhT1@154.53.46.109:5432/postgres")
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CompanyDossier:
    """Aggregated intelligence on a single company."""

    company_name: str
    domain: str | None = None
    github_org: str | None = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Sections
    tech_stack: list[dict[str, Any]] = field(default_factory=list)
    recent_activity: list[dict[str, Any]] = field(default_factory=list)
    github_repos: list[dict[str, Any]] = field(default_factory=list)
    blog_posts: list[dict[str, Any]] = field(default_factory=list)
    glassdoor_summary: dict[str, Any] = field(default_factory=dict)
    key_people: list[dict[str, Any]] = field(default_factory=list)

    # Meta
    source_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_name": self.company_name,
            "domain": self.domain,
            "github_org": self.github_org,
            "generated_at": self.generated_at,
            "tech_stack": self.tech_stack,
            "recent_activity": self.recent_activity,
            "github_repos": self.github_repos,
            "blog_posts": self.blog_posts,
            "glassdoor_summary": self.glassdoor_summary,
            "key_people": self.key_people,
            "source_count": self.source_count,
            "warnings": self.warnings,
        }

    def to_markdown(self) -> str:
        """Render as a human-readable markdown dossier."""
        lines = [
            f"# Company Dossier: {self.company_name}",
            f"*Generated: {self.generated_at}*",
            "",
        ]
        if self.domain:
            lines.append(f"**Domain:** {self.domain}")
        if self.github_org:
            lines.append(f"**GitHub:** github.com/{self.github_org}")
        lines.append(f"**Sources analyzed:** {self.source_count}")
        lines.append("")

        # Tech Stack
        lines.append("## Tech Stack")
        if self.tech_stack:
            for t in sorted(self.tech_stack, key=lambda x: x.get("confidence", 0), reverse=True):
                conf = t.get("confidence", "?")
                lines.append(f"- **{t['name']}** ({t.get('category', 'unknown')}) — confidence: {conf}")
                if t.get("evidence"):
                    lines.append(f"  - Evidence: {t['evidence']}")
        else:
            lines.append("*No tech stack signals found.*")
        lines.append("")

        # GitHub Repos
        lines.append("## GitHub Repositories")
        if self.github_repos:
            for r in self.github_repos[:15]:
                stars = r.get("stars", 0)
                lang = r.get("language", "?")
                lines.append(f"- **{r.get('name', '?')}** — {lang}, ⭐ {stars}")
                if r.get("description"):
                    lines.append(f"  - {r['description']}")
        else:
            lines.append("*No GitHub repos found.*")
        lines.append("")

        # Blog Posts
        lines.append("## Engineering Blog")
        if self.blog_posts:
            for p in self.blog_posts[:10]:
                date = p.get("date", "?")
                lines.append(f"- [{p.get('title', 'Untitled')}]({p.get('url', '#')}) — {date}")
                if p.get("author"):
                    lines.append(f"  - By: {p['author']}")
        else:
            lines.append("*No blog posts found.*")
        lines.append("")

        # Glassdoor
        lines.append("## Employee Sentiment (Glassdoor)")
        if self.glassdoor_summary:
            gs = self.glassdoor_summary
            if gs.get("overall_rating"):
                lines.append(f"**Overall:** {gs['overall_rating']}/5")
            if gs.get("recommend_pct"):
                lines.append(f"**Recommend:** {gs['recommend_pct']}%")
            if gs.get("top_pros"):
                lines.append(f"**Pros:** {', '.join(gs['top_pros'][:5])}")
            if gs.get("top_cons"):
                lines.append(f"**Cons:** {', '.join(gs['top_cons'][:5])}")
            if gs.get("review_count"):
                lines.append(f"*Based on {gs['review_count']} reviews*")
        else:
            lines.append("*No Glassdoor data available.*")
        lines.append("")

        # Key People
        lines.append("## Key People")
        if self.key_people:
            for p in self.key_people[:10]:
                role = p.get("role", "contributor")
                lines.append(f"- **{p['name']}** — {role} (seen in: {p.get('source', '?')})")
        else:
            lines.append("*No key people identified.*")
        lines.append("")

        # Recent Activity
        lines.append("## Recent Activity (90 days)")
        if self.recent_activity:
            for a in self.recent_activity[:10]:
                lines.append(f"- [{a.get('type', '?')}] {a.get('title', '?')} — {a.get('date', '?')}")
        else:
            lines.append("*No recent activity detected.*")

        if self.warnings:
            lines.append("")
            lines.append("## ⚠️ Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entity fetching
# ---------------------------------------------------------------------------

def _fetch_entities(project_id: str, entity_type: Optional[str] = None) -> list[dict]:
    """Fetch all entities for a project, optionally filtered by type."""
    with _get_cursor() as cur:
        if entity_type:
            cur.execute(
                "SELECT entity_type, data, source_type, source_domain, created_at "
                "FROM entities WHERE project_id = %s AND entity_type = %s "
                "ORDER BY created_at DESC",
                (project_id, entity_type),
            )
        else:
            cur.execute(
                "SELECT entity_type, data, source_type, source_domain, created_at "
                "FROM entities WHERE project_id = %s "
                "ORDER BY created_at DESC",
                (project_id,),
            )
        rows = cur.fetchall()
    return [
        {
            "entity_type": r[0],
            "data": r[1] if isinstance(r[1], dict) else json.loads(r[1]) if r[1] else {},
            "source_type": r[2],
            "source_domain": r[3],
            "created_at": r[4].isoformat() if r[4] else None,
        }
        for r in rows
    ]


def _fetch_project_meta(project_id: str) -> dict[str, Any]:
    """Get project name and config."""
    with _get_cursor() as cur:
        cur.execute(
            "SELECT name, config FROM projects WHERE id = %s", (project_id,)
        )
        row = cur.fetchone()
    if not row:
        return {}
    return {
        "name": row[0],
        "config": row[1] if isinstance(row[1], dict) else json.loads(row[1]) if row[1] else {},
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_tech_stack(entities: list[dict]) -> list[dict[str, Any]]:
    """Merge tech signals from multiple sources, deduplicate, boost confidence."""
    tech_map: dict[str, dict] = {}

    for ent in entities:
        data = ent.get("data", {})

        # Direct tech stack entities
        if ent["entity_type"] in ("technology", "tech_signal", "tech_stack"):
            name = data.get("name", data.get("technology", "")).strip()
            if not name:
                continue
            key = name.lower()
            if key not in tech_map:
                tech_map[key] = {
                    "name": name,
                    "category": data.get("category", "unknown"),
                    "evidence": data.get("evidence", ""),
                    "confidence": float(data.get("confidence", 0.5)),
                    "sources": 1,
                }
            else:
                existing = tech_map[key]
                # Boost confidence when seen from multiple sources
                existing["confidence"] = round(min(1.0, existing["confidence"] + 0.15), 2)
                existing["sources"] += 1
                if data.get("evidence") and data["evidence"] not in existing["evidence"]:
                    existing["evidence"] += f"; {data['evidence']}"

        # Extract tech from GitHub repos (language field)
        if ent["entity_type"] in ("github_repo", "repo"):
            lang = data.get("language", "")
            if lang:
                key = lang.lower()
                if key not in tech_map:
                    tech_map[key] = {
                        "name": lang,
                        "category": "language",
                        "evidence": f"GitHub repo: {data.get('name', '?')}",
                        "confidence": 0.9,
                        "sources": 1,
                    }
                else:
                    tech_map[key]["sources"] += 1
                    tech_map[key]["confidence"] = round(min(1.0, tech_map[key]["confidence"] + 0.05), 2)

            # Topics as tech signals
            for topic in data.get("topics", []):
                tkey = topic.lower().strip()
                if tkey and len(tkey) > 1:
                    if tkey not in tech_map:
                        tech_map[tkey] = {
                            "name": topic,
                            "category": "topic",
                            "evidence": f"GitHub topic on {data.get('name', '?')}",
                            "confidence": 0.4,
                            "sources": 1,
                        }
                    else:
                        tech_map[tkey]["sources"] += 1

    return sorted(tech_map.values(), key=lambda x: x["confidence"], reverse=True)


def _aggregate_repos(entities: list[dict]) -> list[dict[str, Any]]:
    """Deduplicate and sort GitHub repos."""
    seen = set()
    repos = []
    for ent in entities:
        if ent["entity_type"] not in ("github_repo", "repo"):
            continue
        data = ent["data"]
        name = data.get("name", "")
        if name in seen:
            continue
        seen.add(name)
        repos.append({
            "name": name,
            "language": data.get("language"),
            "stars": data.get("stars", 0),
            "forks": data.get("forks", 0),
            "last_updated": data.get("last_updated"),
            "description": data.get("description"),
            "topics": data.get("topics", []),
        })
    return sorted(repos, key=lambda x: x.get("stars", 0), reverse=True)


def _aggregate_blog_posts(entities: list[dict]) -> list[dict[str, Any]]:
    """Collect blog posts, sorted by date desc."""
    posts = []
    for ent in entities:
        if ent["entity_type"] not in ("blog_post", "post", "article"):
            continue
        data = ent["data"]
        posts.append({
            "title": data.get("title", "Untitled"),
            "date": data.get("date"),
            "author": data.get("author"),
            "summary": data.get("summary"),
            "url": data.get("url"),
            "tags": data.get("tags", []),
        })
    return sorted(posts, key=lambda x: x.get("date") or "", reverse=True)


def _aggregate_glassdoor(entities: list[dict]) -> dict[str, Any]:
    """Summarize Glassdoor reviews into a sentiment overview."""
    reviews = []
    overview = {}

    for ent in entities:
        if ent["entity_type"] == "glassdoor_overview":
            overview = ent["data"]
        elif ent["entity_type"] in ("glassdoor_review", "review"):
            reviews.append(ent["data"])

    if not reviews and not overview:
        return {}

    # Aggregate reviews
    ratings = [r.get("rating") for r in reviews if r.get("rating")]
    pros = [r.get("pros", "") for r in reviews if r.get("pros")]
    cons = [r.get("cons", "") for r in reviews if r.get("cons")]

    # Simple keyword frequency for top themes
    def top_themes(texts: list[str], n: int = 5) -> list[str]:
        words = Counter()
        stopwords = {"the", "and", "for", "are", "was", "with", "that", "this", "from",
                     "have", "has", "had", "but", "not", "you", "all", "can", "her",
                     "his", "they", "been", "some", "very", "good", "great", "bad",
                     "work", "company", "people", "really", "there", "would"}
        for text in texts:
            for word in text.lower().split():
                word = word.strip(".,!?;:'\"()-")
                if len(word) > 3 and word not in stopwords:
                    words[word] += 1
        return [w for w, _ in words.most_common(n)]

    result = {
        "review_count": len(reviews),
    }
    if overview:
        result.update({
            "overall_rating": overview.get("overall_rating"),
            "recommend_pct": overview.get("recommend_pct"),
            "ceo_approval": overview.get("ceo_approval"),
        })
    elif ratings:
        result["overall_rating"] = round(sum(float(r) for r in ratings) / len(ratings), 1)

    if pros:
        result["top_pros"] = top_themes(pros)
    if cons:
        result["top_cons"] = top_themes(cons)

    return result


def _extract_key_people(entities: list[dict]) -> list[dict[str, Any]]:
    """Identify key people from blog authors and GitHub contributors."""
    people: dict[str, dict] = {}

    for ent in entities:
        data = ent["data"]
        name = None
        source = None
        role = None

        if ent["entity_type"] in ("blog_post", "post", "article") and data.get("author"):
            name = data["author"]
            source = "blog"
            role = "author"
        elif ent["entity_type"] in ("github_contributor", "contributor"):
            name = data.get("name", data.get("login", ""))
            source = "github"
            role = "contributor"
        elif ent["entity_type"] in ("glassdoor_review", "review") and data.get("role"):
            # Don't use reviewer names (anonymous), but note roles
            continue

        if name:
            key = name.lower().strip()
            if key not in people:
                people[key] = {"name": name, "role": role, "source": source, "mentions": 1}
            else:
                people[key]["mentions"] += 1

    return sorted(people.values(), key=lambda x: x["mentions"], reverse=True)


def _find_recent_activity(entities: list[dict], days: int = 90) -> list[dict[str, Any]]:
    """Find entities with dates within the last N days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    activity = []

    for ent in entities:
        data = ent["data"]
        date = data.get("date") or data.get("last_updated") or ent.get("created_at")
        if not date:
            continue

        date_str = str(date)[:10]  # YYYY-MM-DD
        if date_str >= cutoff[:10]:
            activity.append({
                "type": ent["entity_type"],
                "title": data.get("title") or data.get("name") or "?",
                "date": date_str,
                "url": data.get("url"),
            })

    return sorted(activity, key=lambda x: x["date"], reverse=True)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dossier(project_id: str) -> CompanyDossier:
    """
    Build a company intelligence dossier from all entities in a Grabber project.

    Args:
        project_id: UUID of the Grabber project containing company_intel results.

    Returns:
        CompanyDossier with aggregated intelligence.
    """
    meta = _fetch_project_meta(project_id)
    config = meta.get("config", {})

    dossier = CompanyDossier(
        company_name=config.get("company_name", meta.get("name", "Unknown")),
        domain=config.get("domain"),
        github_org=config.get("github_org"),
    )

    entities = _fetch_entities(project_id)
    dossier.source_count = len(entities)

    if not entities:
        dossier.warnings.append("No entities found — recipes may not have completed yet.")
        return dossier

    dossier.tech_stack = _aggregate_tech_stack(entities)
    dossier.github_repos = _aggregate_repos(entities)
    dossier.blog_posts = _aggregate_blog_posts(entities)
    dossier.glassdoor_summary = _aggregate_glassdoor(entities)
    dossier.key_people = _extract_key_people(entities)
    dossier.recent_activity = _find_recent_activity(entities)

    # Warnings for missing sections
    if not dossier.github_repos:
        dossier.warnings.append("No GitHub repos found — org may not exist or scrape failed.")
    if not dossier.glassdoor_summary:
        dossier.warnings.append("No Glassdoor data — may be blocked or company not listed.")
    if not dossier.blog_posts:
        dossier.warnings.append("No blog posts found — company may not have a public eng blog.")

    return dossier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a company intelligence dossier")
    parser.add_argument("project_id", help="Grabber project UUID")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of markdown")
    parser.add_argument("--out", help="Write to file instead of stdout")
    args = parser.parse_args()

    dossier = build_dossier(args.project_id)

    if args.json:
        output = json.dumps(dossier.to_dict(), indent=2)
    else:
        output = dossier.to_markdown()

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Dossier written to {args.out}")
    else:
        print(output)
