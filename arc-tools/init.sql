-- Data Churner schema — PostgreSQL 16 + pgvector
-- Runs against the 'churner' database (created by init-db.sh)
\connect churner;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Missions
CREATE TABLE missions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    purpose TEXT NOT NULL,
    purpose_embedding VECTOR(1024),
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Raw ingests (immutable)
CREATE TABLE raw_ingests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mission_id UUID REFERENCES missions(id),
    source TEXT NOT NULL,
    raw_payload JSONB NOT NULL,
    content_hash TEXT UNIQUE NOT NULL,
    status TEXT DEFAULT 'pending',
    ingested_at TIMESTAMPTZ DEFAULT now(),
    processed_at TIMESTAMPTZ
);
CREATE INDEX idx_raw_status ON raw_ingests(status) WHERE status = 'pending';

-- Schemas (versioned, agent-evolved)
CREATE TABLE schemas (
    id SERIAL PRIMARY KEY,
    mission_id UUID REFERENCES missions(id),
    entity_type TEXT NOT NULL,
    version INT NOT NULL,
    json_schema JSONB NOT NULL,
    changelog JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    created_by TEXT,
    UNIQUE(mission_id, entity_type, version)
);

-- Extracted entities
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mission_id UUID REFERENCES missions(id),
    entity_type TEXT NOT NULL,
    schema_version INT NOT NULL,
    data JSONB NOT NULL,
    traits JSONB DEFAULT '{}',
    embedding VECTOR(1024),
    source_ids UUID[] NOT NULL,
    confidence FLOAT,
    is_golden BOOLEAN DEFAULT false,
    golden_id UUID,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_entities_golden ON entities(mission_id, entity_type) WHERE is_golden = true;
CREATE INDEX idx_entities_traits ON entities USING GIN(traits);
CREATE INDEX idx_entities_embedding ON entities USING hnsw(embedding vector_cosine_ops);
-- Name-based candidate matching (trigram similarity for fuzzy dedup)
CREATE INDEX IF NOT EXISTS idx_entities_name_trgm ON entities USING GIN ((data->>'name') gin_trgm_ops) WHERE data->>'name' IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities ((data->>'name')) WHERE data->>'name' IS NOT NULL;

-- Entity links
CREATE TABLE entity_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_id UUID REFERENCES entities(id),
    to_id UUID REFERENCES entities(id),
    link_type TEXT NOT NULL,
    confidence FLOAT,
    evidence JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Facet catalog
CREATE TABLE facets (
    id SERIAL PRIMARY KEY,
    mission_id UUID REFERENCES missions(id),
    entity_type TEXT NOT NULL,
    facet_key TEXT NOT NULL,
    facet_value TEXT NOT NULL,
    entity_count INT DEFAULT 0,
    interesting_score FLOAT DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT now(),
    UNIQUE(mission_id, entity_type, facet_key, facet_value)
);

-- Dynamic groups (SEO page seeds)
CREATE TABLE groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mission_id UUID REFERENCES missions(id),
    entity_type TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    facet_combo JSONB NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    member_count INT DEFAULT 0,
    seo_score FLOAT DEFAULT 0,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT now(),
    refreshed_at TIMESTAMPTZ
);
CREATE INDEX idx_groups_combo ON groups USING GIN(facet_combo);
