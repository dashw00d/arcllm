-- Site Auditor state schema — tracks every site, audit, page scan, and pattern.
-- Stored in the ghostgraph database alongside grabber's operational data.

-- ── Sites ──────────────────────────────────────────────────────────────
-- One row per root domain. The top-level record.
CREATE TABLE IF NOT EXISTS sites (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain          TEXT UNIQUE NOT NULL,          -- wehelpyouparty.com
    description     TEXT,                          -- "catering staffing service with venue listings"
    first_seen      TIMESTAMPTZ DEFAULT now(),
    last_audited    TIMESTAMPTZ,

    -- Recon results (populated once, updated on re-audit)
    ipv6_supported  BOOLEAN,
    waf             TEXT,                          -- cloudflare, akamai, null
    robots_disallow TEXT[],
    warnings        TEXT[],

    -- Overall status
    status          TEXT DEFAULT 'new'             -- new, queued, auditing, ready, stale, blocked
);

-- ── Entity Audits ──────────────────────────────────────────────────────
-- One row per (site × entity_type). Tracks the audit state for finding
-- "wedding venues" on wehelpyouparty.com separately from "catering rentals".
CREATE TABLE IF NOT EXISTS entity_audits (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id         UUID NOT NULL REFERENCES sites(id),
    entity_type     TEXT NOT NULL,                 -- "wedding venues", "car rentals"
    description     TEXT,                          -- user-provided description of target

    -- Triage result
    relevant        BOOLEAN,
    confidence      FLOAT,
    reasoning       TEXT,

    -- Discovery summary
    pages_scanned   INT DEFAULT 0,
    list_pages_found INT DEFAULT 0,
    detail_pages_found INT DEFAULT 0,

    -- Pipeline state
    stage           TEXT DEFAULT 'pending',        -- pending, triage, discovery, patterning, extraction, complete, not_relevant
    last_stage_at   TIMESTAMPTZ,
    error           TEXT,                          -- last error if failed

    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),

    UNIQUE(site_id, entity_type)
);

-- ── Page Scans ─────────────────────────────────────────────────────────
-- Every page visited during any audit. Never deleted — full history.
CREATE TABLE IF NOT EXISTS page_scans (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id        UUID NOT NULL REFERENCES entity_audits(id),
    url             TEXT NOT NULL,
    path            TEXT NOT NULL,                 -- /venues?page=4
    page_type       TEXT,                          -- index, detail, navigation, other
    description     TEXT,                          -- "lists venues with rating, event count, features"

    -- What happened
    stage           TEXT NOT NULL,                 -- triage, discovery, patterning, extraction
    outcome         TEXT,                          -- found_entities, no_data, blocked, error
    http_status     INT,
    scan_time_ms    INT,

    -- Content summary (not raw HTML — just what Henry observed)
    summary         TEXT,                          -- Henry's description of what's on the page
    element_count   INT,                           -- number of entity elements found
    snapshot_tokens INT,                           -- how many tokens the snapshot used

    scanned_at      TIMESTAMPTZ DEFAULT now(),

    -- Don't rescan the same URL in the same audit unless forced
    UNIQUE(audit_id, url, stage)
);

-- ── URL Patterns ───────────────────────────────────────────────────────
-- Discovered URL patterns for index and detail pages.
-- "index" = pages that list entities. "detail" = individual entity page.
CREATE TABLE IF NOT EXISTS url_patterns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id        UUID NOT NULL REFERENCES entity_audits(id),
    pattern_type    TEXT NOT NULL,                 -- index, detail
    url_pattern     TEXT NOT NULL,                 -- /venues?page=(\d+)
    example_urls    TEXT[],                        -- ["/venues?page=1", "/venues?page=2"]
    description     TEXT,                          -- "paginated venue listing"

    -- Pagination
    pagination_type TEXT,                          -- query_param, next_link, load_more, none
    pagination_selector TEXT,                      -- CSS selector for next page link
    max_pages       INT,                           -- estimated total pages

    -- Confidence
    confidence      FLOAT DEFAULT 0.0,
    validated       BOOLEAN DEFAULT false,         -- cross-validated on multiple pages
    validated_at    TIMESTAMPTZ,

    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ── DOM Patterns ───────────────────────────────────────────────────────
-- CSS selectors for extracting data from pages. Two levels:
--   1. Entity container selector (repeating element on index pages)
--   2. Chunk selectors (data sections within detail pages)
CREATE TABLE IF NOT EXISTS dom_patterns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id        UUID NOT NULL REFERENCES entity_audits(id),
    url_pattern_id  UUID REFERENCES url_patterns(id),
    page_type       TEXT NOT NULL,                 -- index, detail

    -- For index pages: what selector grabs each entity card
    entity_selector TEXT,                          -- div.venue-card
    link_selector   TEXT,                          -- a[href] within entity
    count_on_page   INT,                           -- how many entities per page

    -- For detail pages: labeled DOM chunks
    -- Stored as JSONB array: [{"label": "reviews", "selector": "div.reviews", "size": "large", "contains": "user reviews"}]
    chunks          JSONB,

    -- Title/name selector (for both page types)
    title_selector  TEXT,                          -- h1.venue-name

    -- Confidence + validation
    confidence      FLOAT DEFAULT 0.0,
    tested_on_urls  TEXT[],                        -- URLs where this pattern was tested
    success_rate    FLOAT,                         -- % of test pages where pattern worked
    validated       BOOLEAN DEFAULT false,

    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- ── Indexes ────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_sites_domain ON sites(domain);
CREATE INDEX IF NOT EXISTS idx_entity_audits_site ON entity_audits(site_id);
CREATE INDEX IF NOT EXISTS idx_entity_audits_stage ON entity_audits(stage) WHERE stage != 'complete';
CREATE INDEX IF NOT EXISTS idx_page_scans_audit ON page_scans(audit_id);
CREATE INDEX IF NOT EXISTS idx_page_scans_url ON page_scans(url);
CREATE INDEX IF NOT EXISTS idx_url_patterns_audit ON url_patterns(audit_id);
CREATE INDEX IF NOT EXISTS idx_dom_patterns_audit ON dom_patterns(audit_id);
