-- Production Postgres schema for the NER system.
--
-- Three logical zones:
--
--   1. SYNTHETIC POOLS  (entity_pools / decoy_pools / templates)
--      Same names and shape as the SQLite contract in sql/schema.sql so
--      `ner.data.pools.load_from_postgres` reuses the SQLite query strings.
--
--   2. GOLD STORE       (gold_records / gold_entities)
--      Real, hand-labeled validation and test records. The Python load-time
--      invariant `text[start:end] == entity.surface` is enforced both via
--      DB constraints (CHECKs on offsets, polarity domain) and by an
--      application-level validator before insert.
--
--   3. ANNOTATION TRAIL (annotators / annotations / adjudications /
--                        dataset_versions)
--      Audit log for IAA computation, adjudication history, and
--      version-locked dataset releases.
--
-- The script is idempotent: every CREATE uses IF NOT EXISTS and every
-- function is CREATE OR REPLACE.

BEGIN;

-- =============================================================================
-- 1. SYNTHETIC POOLS
-- =============================================================================

CREATE TABLE IF NOT EXISTS entity_pools (
    id          BIGSERIAL PRIMARY KEY,
    entity_type TEXT             NOT NULL CHECK (entity_type IN ('PERSON','ORG','ADDRESS','COMMODITY')),
    value       TEXT             NOT NULL,
    weight      DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    UNIQUE (entity_type, value)
);
CREATE INDEX IF NOT EXISTS idx_entity_pools_type ON entity_pools (entity_type);

CREATE TABLE IF NOT EXISTS decoy_pools (
    id        BIGSERIAL PRIMARY KEY,
    slot_name TEXT             NOT NULL,
    value     TEXT             NOT NULL,
    weight    DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    UNIQUE (slot_name, value)
);
CREATE INDEX IF NOT EXISTS idx_decoy_pools_slot ON decoy_pools (slot_name);

CREATE TABLE IF NOT EXISTS templates (
    id       BIGSERIAL PRIMARY KEY,
    template TEXT             NOT NULL UNIQUE,
    weight   DOUBLE PRECISION NOT NULL DEFAULT 1.0
);


-- =============================================================================
-- 2. ANNOTATION INFRASTRUCTURE
-- =============================================================================

CREATE TABLE IF NOT EXISTS annotators (
    id         BIGSERIAL    PRIMARY KEY,
    code       TEXT         NOT NULL UNIQUE,    -- short stable id (e.g. "a01")
    name       TEXT,
    role       TEXT         NOT NULL CHECK (role IN ('annotator','adjudicator','admin')),
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id            BIGSERIAL   PRIMARY KEY,
    tag           TEXT        NOT NULL UNIQUE,  -- e.g. "dev-v2.3.0"
    scope         TEXT        NOT NULL CHECK (scope IN ('dev','test','edge_case')),
    manifest_sha  TEXT,                          -- SHA-256 of sorted record manifest
    notes         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    locked_at     TIMESTAMPTZ                    -- non-null = frozen / sealed
);


-- =============================================================================
-- 3. GOLD STORE
-- =============================================================================

CREATE TABLE IF NOT EXISTS gold_records (
    id            BIGSERIAL   PRIMARY KEY,
    text          TEXT        NOT NULL,
    source        TEXT        NOT NULL CHECK (source IN ('invoice','manifest','email','webhook','ocr','other')),
    split         TEXT        NOT NULL CHECK (split IN ('dev','test','edge_case')),
    version_id    BIGINT      REFERENCES dataset_versions(id),
    content_hash  TEXT        NOT NULL,          -- SHA-256 of normalized text
    locked_at     TIMESTAMPTZ,                   -- non-null = vault'd (test set)
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Global content-hash uniqueness prevents accidental dev/test leak.
CREATE UNIQUE INDEX IF NOT EXISTS uq_gold_records_content_hash ON gold_records (content_hash);
CREATE INDEX IF NOT EXISTS idx_gold_records_split   ON gold_records (split);
CREATE INDEX IF NOT EXISTS idx_gold_records_version ON gold_records (version_id);
CREATE INDEX IF NOT EXISTS idx_gold_records_locked  ON gold_records (locked_at)
    WHERE locked_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS gold_entities (
    id            BIGSERIAL PRIMARY KEY,
    record_id     BIGINT    NOT NULL REFERENCES gold_records(id) ON DELETE CASCADE,
    type          TEXT      NOT NULL CHECK (type IN ('PERSON','ORG','ADDRESS','COMMODITY')),
    surface       TEXT      NOT NULL,
    start_offset  INTEGER   NOT NULL CHECK (start_offset >= 0),
    end_offset    INTEGER   NOT NULL CHECK (end_offset > start_offset),
    polarity      TEXT      NOT NULL DEFAULT 'POS' CHECK (polarity IN ('POS','NEG')),
    -- NEG polarity is only meaningful for COMMODITY (Section 5 of the spec).
    CHECK (polarity = 'POS' OR type = 'COMMODITY')
);
CREATE INDEX IF NOT EXISTS idx_gold_entities_record   ON gold_entities (record_id);
CREATE INDEX IF NOT EXISTS idx_gold_entities_type     ON gold_entities (type);
CREATE INDEX IF NOT EXISTS idx_gold_entities_polarity ON gold_entities (polarity);

CREATE TABLE IF NOT EXISTS annotations (
    id           BIGSERIAL   PRIMARY KEY,
    record_id    BIGINT      NOT NULL REFERENCES gold_records(id) ON DELETE CASCADE,
    annotator_id BIGINT      NOT NULL REFERENCES annotators(id),
    -- Raw payload from the labeling tool BEFORE adjudication. Schema is
    -- documented in docs/data_specification.md §19.
    payload      JSONB       NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (record_id, annotator_id)
);
CREATE INDEX IF NOT EXISTS idx_annotations_record    ON annotations (record_id);
CREATE INDEX IF NOT EXISTS idx_annotations_annotator ON annotations (annotator_id);

CREATE TABLE IF NOT EXISTS adjudications (
    id             BIGSERIAL   PRIMARY KEY,
    record_id      BIGINT      NOT NULL REFERENCES gold_records(id) ON DELETE CASCADE,
    adjudicator_id BIGINT      NOT NULL REFERENCES annotators(id),
    decided_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes          TEXT,
    UNIQUE (record_id)                         -- one final adjudication per record
);


-- =============================================================================
-- 4. IMMUTABILITY TRIGGERS
-- =============================================================================
--
-- Once a gold_record is locked (locked_at IS NOT NULL), no update or delete
-- is permitted. This enforces the "test is a vault" contract at the DB level
-- so application bugs can't accidentally mutate sealed records.

CREATE OR REPLACE FUNCTION reject_locked_gold_record_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.locked_at IS NOT NULL THEN
            RAISE EXCEPTION 'gold_records.id=% is locked at % - delete forbidden',
                            OLD.id, OLD.locked_at;
        END IF;
        RETURN OLD;
    END IF;
    IF OLD.locked_at IS NOT NULL THEN
        RAISE EXCEPTION 'gold_records.id=% is locked at % - update forbidden',
                        OLD.id, OLD.locked_at;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_gold_records_locked_update ON gold_records;
CREATE TRIGGER trg_gold_records_locked_update
BEFORE UPDATE ON gold_records
FOR EACH ROW EXECUTE FUNCTION reject_locked_gold_record_change();

DROP TRIGGER IF EXISTS trg_gold_records_locked_delete ON gold_records;
CREATE TRIGGER trg_gold_records_locked_delete
BEFORE DELETE ON gold_records
FOR EACH ROW EXECUTE FUNCTION reject_locked_gold_record_change();


-- =============================================================================
-- 5. CONVENIENCE VIEWS
-- =============================================================================

-- Aggregate per-split counts and entity-type distribution for the Section 15
-- acceptance gates. Cheap to compute; useful for nightly dashboards.
CREATE OR REPLACE VIEW v_split_coverage AS
SELECT
    r.split,
    COUNT(DISTINCT r.id)                              AS record_count,
    COUNT(*) FILTER (WHERE e.type = 'PERSON')         AS person_spans,
    COUNT(*) FILTER (WHERE e.type = 'ORG')            AS org_spans,
    COUNT(*) FILTER (WHERE e.type = 'ADDRESS')        AS address_spans,
    COUNT(*) FILTER (WHERE e.type = 'COMMODITY'
                     AND e.polarity = 'POS')          AS commodity_pos_spans,
    COUNT(*) FILTER (WHERE e.type = 'COMMODITY'
                     AND e.polarity = 'NEG')          AS commodity_neg_spans,
    COUNT(DISTINCT r.id) FILTER (WHERE e.id IS NULL)  AS zero_entity_records
FROM gold_records r
LEFT JOIN gold_entities e ON e.record_id = r.id
GROUP BY r.split;

COMMIT;
