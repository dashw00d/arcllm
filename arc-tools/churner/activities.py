"""Temporal activities — where LLM calls happen."""

import json
import logging

from temporalio import activity
from openai import AsyncOpenAI

from config import LLAMA_SERVER_URL, MODEL, MAX_TOKENS
from prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    RESOLVER_SYSTEM_PROMPT,
    TRAIT_MINER_SYSTEM_PROMPT,
    GROUP_ARCHITECT_SYSTEM_PROMPT,
    SCHEMA_ARCHITECT_SYSTEM_PROMPT,
)
import db

log = logging.getLogger("churner.activities")

llm = AsyncOpenAI(
    base_url=f"{LLAMA_SERVER_URL}/v1",
    api_key="none",
    timeout=300.0,
    default_headers={"X-Priority": "low"},
)


def _parse_json(text: str) -> dict | list | None:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop ```json
        end = next((i for i, l in enumerate(lines) if l.strip() == "```"), len(lines))
        text = "\n".join(lines[:end])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("Failed to parse JSON from LLM response: %.200s", text)
        return None


async def _llm_call(system_prompt: str, user_msg: str, temperature: float = 0.3,
                    max_tokens: int = MAX_TOKENS) -> dict | list | None:
    """Make an LLM call with a frozen system prompt and parse the JSON response."""
    resp = await llm.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return _parse_json(resp.choices[0].message.content)


# ── Data pipeline activities ──────────────────────────────────────────────

@activity.defn
async def fetch_pending_records(mission_id: str, batch_size: int) -> list[dict]:
    return await db.fetch_pending_records(mission_id, batch_size)


@activity.defn
async def get_active_schema(mission_id: str, entity_type: str = "default") -> dict:
    return await db.get_active_schema(mission_id, entity_type)


@activity.defn
async def extract_record(record_id: str, raw_payload: dict, schema: dict) -> dict | None:
    user_msg = f"""## Target Schema
```json
{json.dumps(schema, indent=2)}
```

## Raw Data (Source ID: {record_id})
```json
{json.dumps(raw_payload, indent=2)}
```

Extract all entities from this raw data according to the schema. Output valid JSON only."""

    try:
        result = await _llm_call(EXTRACTOR_SYSTEM_PROMPT, user_msg)
    except Exception as e:
        log.error("extract_record LLM call failed for %s: %s", record_id, e)
        await db.mark_raw_error(record_id)
        return None

    if result:
        # LLM may return a list if raw data contains multiple entities — take first.
        if isinstance(result, list):
            result = result[0] if result else None
        if result and isinstance(result, dict):
            result["_source_id"] = record_id
            return result
    await db.mark_raw_error(record_id)
    return None


@activity.defn
async def resolve_and_merge(mission_id: str, entity: dict) -> dict | None:
    entity_type = entity.get("_type", "default")
    source_ids = [entity["_source_id"]] if entity.get("_source_id") else []
    entity_name = entity.get("name") or entity.get("title") or None

    # Always insert the new entity first so it has a real DB ID
    entity_id = await db.insert_entity(
        mission_id, entity_type, entity.get("_schema_version", 1),
        entity, source_ids, entity.get("_confidence", 0.5),
    )

    # Now that entity is persisted, mark the raw record as done
    for sid in source_ids:
        await db.mark_raw_done(sid)

    candidates = await db.find_candidate_entities(mission_id, entity_type, new_entity_name=entity_name)

    if not candidates:
        await db.mark_as_golden(entity_id)
        entity["_entity_id"] = entity_id
        return entity

    for candidate in candidates:
        if candidate["id"] == entity_id:
            continue  # don't compare to self

        user_msg = f"""## Record A (existing golden record)
```json
{json.dumps(candidate["data"], indent=2)}
```

## Record B (new extraction)
```json
{json.dumps(entity, indent=2)}
```

Are these the same real-world entity? If yes, produce the merged record."""

        try:
            result = await _llm_call(RESOLVER_SYSTEM_PROMPT, user_msg, temperature=0.2)
        except Exception as e:
            log.warning("resolve_and_merge LLM call failed: %s", e)
            result = None

        if result and result.get("verdict") == "same_as":
            await db.merge_into_golden(candidate["id"], entity_id, result["merged_record"])
            # Traits go on the golden record
            return {"_entity_id": candidate["id"], **result["merged_record"]}

    # No match — new golden record
    await db.mark_as_golden(entity_id)
    entity["_entity_id"] = entity_id
    return entity


@activity.defn
async def mine_traits(entity_id: str) -> dict:
    entity = await db.get_entity(entity_id)
    if not entity:
        return {}
    user_msg = f"""## Entity Record
```json
{json.dumps(entity["data"], indent=2)}
```

Extract all searchable traits as key-value pairs."""

    try:
        traits = await _llm_call(TRAIT_MINER_SYSTEM_PROMPT, user_msg, temperature=0.2)
    except Exception as e:
        log.warning("mine_traits LLM call failed for %s: %s", entity_id, e)
        return {}
    if traits and isinstance(traits, dict):
        await db.update_entity_traits(entity_id, traits)
        return traits
    return {}


@activity.defn
async def recover_stuck_raw_ingests(mission_id: str, older_than_minutes: int = 60) -> int:
    """Reset 'processing' records stuck for too long back to 'pending'.

    Returns count of recovered records.
    """
    return await db.recover_stuck_raw_ingests(mission_id, older_than_minutes)


# ── Schema evolution activities ───────────────────────────────────────────

@activity.defn
async def sample_recent_entities(mission_id: str, limit: int) -> list[dict]:
    return await db.sample_recent_entities(mission_id, limit)


@activity.defn
async def propose_schema_changes(mission_id: str, sample: list[dict]) -> dict | None:
    current_schema = await db.get_active_schema(mission_id)
    sample_summary = json.dumps(sample[:20], indent=2)

    user_msg = f"""## Current Schema
```json
{json.dumps(current_schema, indent=2)}
```

## Sample Records ({len(sample)} total, showing first 20)
```json
{sample_summary}
```

Analyze these records and propose schema improvements. Output JSON with:
- new_schema: the complete updated JSON Schema
- changelog: list of changes with rationale
- confidence: 0.0-1.0"""

    try:
        return await _llm_call(SCHEMA_ARCHITECT_SYSTEM_PROMPT, user_msg, temperature=0.3)
    except Exception as e:
        log.warning("propose_schema_changes LLM call failed: %s", e)
        return None


@activity.defn
async def apply_schema_update(mission_id: str, proposal: dict):
    await db.apply_schema_update(mission_id, proposal)


# ── Group discovery activities ────────────────────────────────────────────

@activity.defn
async def refresh_facet_catalog(mission_id: str):
    await db.refresh_facet_catalog(mission_id)


@activity.defn
async def get_facet_stats(mission_id: str) -> list[dict]:
    return await db.get_facet_stats(mission_id)


@activity.defn
async def architect_groups(mission_id: str, facet_stats: list[dict]) -> list[dict] | None:
    user_msg = f"""## Facet Statistics ({len(facet_stats)} facets)
```json
{json.dumps(facet_stats[:100], indent=2)}
```

Discover interesting multi-facet combinations for listing pages. Output a JSON array of groups."""

    try:
        result = await _llm_call(GROUP_ARCHITECT_SYSTEM_PROMPT, user_msg, temperature=0.5)
    except Exception as e:
        log.warning("architect_groups LLM call failed: %s", e)
        return []
    if isinstance(result, list):
        return result
    return []


@activity.defn
async def upsert_group(mission_id: str, group: dict):
    await db.upsert_group(mission_id, group)
