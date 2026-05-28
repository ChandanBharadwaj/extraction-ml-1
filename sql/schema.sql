-- Seed schema for synthetic NER data generation.
--
-- This is the contract the slot-fill generator expects. Drop your seed inserts
-- against these tables and the pipeline will pick them up via
-- `ner.data.pools.load_from_sqlite(path)`.
--
-- Three logical tables:
--   entity_pools     : real entity values, partitioned by entity type
--   decoy_pools      : non-entity filler values for template slots (qty, units, ids...)
--   templates        : sentence templates with {ENTITY_TYPE} and {decoy:slot} markers
--
-- Templates use the following placeholder syntax (case-sensitive):
--   {PERSON}, {ORG}, {ADDRESS}, {COMMODITY}        -- entity slots
--   {decoy:qty}, {decoy:invoice_id}, {decoy:unit}  -- non-entity slots
--   Repeat placeholders use {PERSON#1}, {PERSON#2} to force distinct samples.
--
-- All `weight` columns default to 1.0; the generator does weighted sampling
-- without replacement within a single record.

CREATE TABLE IF NOT EXISTS entity_pools (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT    NOT NULL CHECK(entity_type IN ('PERSON','ORG','ADDRESS','COMMODITY')),
    value       TEXT    NOT NULL,
    weight      REAL    NOT NULL DEFAULT 1.0,
    UNIQUE(entity_type, value)
);

CREATE INDEX IF NOT EXISTS idx_entity_pools_type ON entity_pools(entity_type);

CREATE TABLE IF NOT EXISTS decoy_pools (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_name TEXT    NOT NULL,
    value     TEXT    NOT NULL,
    weight    REAL    NOT NULL DEFAULT 1.0,
    UNIQUE(slot_name, value)
);

CREATE INDEX IF NOT EXISTS idx_decoy_pools_slot ON decoy_pools(slot_name);

CREATE TABLE IF NOT EXISTS templates (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    template TEXT    NOT NULL UNIQUE,
    weight   REAL    NOT NULL DEFAULT 1.0
);
