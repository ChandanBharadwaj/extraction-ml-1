-- Tiny example seed so the pipeline produces something out of the box.
-- Real production seeds should be 10k+ commodities, 5k+ orgs, etc.
--
-- This seed exercises:
--   - positive commodity extraction (status quo)
--   - negation (B-NEG_COMMODITY / I-NEG_COMMODITY via {NEG_COMMODITY} placeholder)
--   - scope (multi-target denial)
--   - target specificity (bare + qualified commodity forms in the pool)
--   - dissolved cues (frozen_compound decoys — never mark as negation)
--   - word order (denial-then-assertion and assertion-then-denial)

INSERT OR IGNORE INTO entity_pools (entity_type, value) VALUES
    ('PERSON', 'Maria Gonzalez'),
    ('PERSON', 'Felix Yu'),
    ('PERSON', 'Aarav Patel'),
    ('PERSON', 'Sofia Rossi'),
    ('PERSON', 'Kenji Tanaka'),

    ('ORG', 'Nordwind Logistics GmbH'),
    ('ORG', 'Acme Trading Co.'),
    ('ORG', 'Delta Packaging'),
    ('ORG', 'Oceanic Freight Co.'),
    ('ORG', 'BlueRiver Commodities Ltd'),

    ('ADDRESS', '7 Canal St, Singapore 049320'),
    ('ADDRESS', '42 Industrial Park Road, Rotterdam, 3011 AB'),
    ('ADDRESS', '15 Marina Bay Drive, Singapore 018956'),
    ('ADDRESS', '88 Wallaby Way, Sydney NSW 2000, Australia'),

    -- Bare and qualified pairs: the model must learn that negation
    -- targets the qualified form when one is present.
    ('COMMODITY', 'Grade A robusta coffee'),
    ('COMMODITY', 'robusta coffee'),
    ('COMMODITY', '304 stainless steel sheet'),
    ('COMMODITY', 'steel sheet'),
    ('COMMODITY', 'Polyethylene resin HDPE'),
    ('COMMODITY', 'refined copper cathode'),
    ('COMMODITY', 'copper cathode'),
    ('COMMODITY', 'galvanized steel coil'),
    ('COMMODITY', 'steel coil'),
    ('COMMODITY', 'anhydrous ammonia'),
    ('COMMODITY', 'ammonia'),
    ('COMMODITY', 'raw cane sugar'),
    ('COMMODITY', 'cane sugar'),
    ('COMMODITY', 'special wood'),
    ('COMMODITY', 'treated wood'),
    ('COMMODITY', 'wood'),
    ('COMMODITY', 'organic cotton'),
    ('COMMODITY', 'cotton');

INSERT OR IGNORE INTO decoy_pools (slot_name, value) VALUES
    ('qty', '500 tons of'),
    ('qty', '12,000 kg of'),
    ('qty', '300 bales of'),
    ('qty', '40 containers of'),
    ('unit', 'kg'),
    ('unit', 'tons'),
    ('unit', 'MT'),
    ('invoice_id', 'PO#88231'),
    ('invoice_id', 'INV-2024-0421'),
    ('invoice_id', 'REF#A-9912'),

    -- Negation cues: characters here are protected by the slot-fill
    -- assembler (PRESERVE_DECOY_SLOTS) so noise can't silently flip
    -- the polarity of adjacent gold labels.
    ('neg_cue', 'does not contain'),
    ('neg_cue', 'no'),
    ('neg_cue', 'without'),
    ('neg_cue', 'lacks'),
    ('neg_cue', 'excludes'),
    ('neg_cue', 'free of'),
    ('neg_cue', 'absent of'),
    ('neg_cue', 'missing'),
    ('neg_cue', 'not'),

    -- Contrast cues mark scope boundaries between denied and asserted
    -- entities in the same sentence.
    ('contrast_cue', ', but only'),
    ('contrast_cue', ', only'),
    ('contrast_cue', ', instead'),
    ('contrast_cue', ', rather than'),
    ('contrast_cue', '; instead of'),
    ('contrast_cue', '; in lieu of'),

    -- Frozen compounds: dissolved negation cues that must NOT be read
    -- as denials. We inject them in POSITIVE templates so the model
    -- learns to ignore the surface negation signal.
    ('frozen_compound', 'sugar-free'),
    ('frozen_compound', 'non-stick'),
    ('frozen_compound', 'stainless'),
    ('frozen_compound', 'lead-free'),
    ('frozen_compound', 'BPA-free'),
    ('frozen_compound', 'gluten-free'),
    ('frozen_compound', 'noise-cancelling');

INSERT OR IGNORE INTO templates (template) VALUES
    -- Existing positive-only templates (unchanged).
    ('Invoice: {decoy:qty} {COMMODITY} for {ORG}.'),
    ('shipment of {COMMODITY} approved by {PERSON} at {ORG}'),
    ('{decoy:invoice_id} | {COMMODITY} | Qty {decoy:qty} | Sold to: {ORG}, {ADDRESS}'),
    ('{PERSON} from {ORG} confirmed {COMMODITY} shipped to {ADDRESS}.'),
    ('Manifest: {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3} — ETA {PERSON}, {ORG}.'),
    ('Please dispatch {COMMODITY} to {ADDRESS} c/o {PERSON}.'),
    ('{ORG} acknowledges receipt of {COMMODITY} from {PERSON}.'),

    -- Negation: pure denial.
    ('{ORG} {decoy:neg_cue} {NEG_COMMODITY}.'),
    ('{decoy:invoice_id}: {decoy:neg_cue} {NEG_COMMODITY}.'),

    -- Negation: denial then assertion (word order A).
    ('{decoy:neg_cue} {NEG_COMMODITY#1}{decoy:contrast_cue} {COMMODITY#2}.'),
    ('Shipment {decoy:neg_cue} {NEG_COMMODITY#1}{decoy:contrast_cue} {COMMODITY#2}, per {PERSON}.'),

    -- Negation: assertion then denial (word order B).
    ('{COMMODITY#1}, but {decoy:neg_cue} {NEG_COMMODITY#2}.'),
    ('Delivery contains {COMMODITY#1}, but {decoy:neg_cue} {NEG_COMMODITY#2} ({ORG}).'),

    -- Negation: multi-target scope (one cue, multiple denied commodities).
    ('{decoy:neg_cue} {NEG_COMMODITY#1}, {NEG_COMMODITY#2}, or {NEG_COMMODITY#3}.'),
    ('{ORG} certifies: {decoy:neg_cue} {NEG_COMMODITY#1}, {NEG_COMMODITY#2}.'),

    -- Frozen compounds: positive templates that embed dissolved-negation cues
    -- so the model learns these surface forms are NOT negations.
    ('{PERSON} confirmed {decoy:frozen_compound} {COMMODITY} delivery.'),
    ('{decoy:qty} {decoy:frozen_compound} {COMMODITY} shipped to {ADDRESS}.'),
    ('{ORG} supplies {decoy:frozen_compound} {COMMODITY}.'),

    -- Mixed: qualified denial + bare positive in same record.
    ('{decoy:neg_cue} {NEG_COMMODITY#1}; ordinary {COMMODITY#2} acceptable.'),
    ('{ORG} requires {COMMODITY#1}, {decoy:neg_cue} {NEG_COMMODITY#2}.');
