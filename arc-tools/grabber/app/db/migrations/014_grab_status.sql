-- Migration 014: Add grab_status to entity_audits for PatternGrabWorkflow
-- Tracks whether an audit has been consumed by the pattern-based grab workflow

ALTER TABLE entity_audits ADD COLUMN IF NOT EXISTS grab_status TEXT DEFAULT 'pending';

-- Index for efficient querying of unaudited completed audits
CREATE INDEX IF NOT EXISTS idx_entity_audits_grab_status
  ON entity_audits(grab_status)
  WHERE grab_status != 'done';
