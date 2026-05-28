-- Postgres-flavored example seed.
--
-- Mirrors sql/example_seed.sql but uses ON CONFLICT DO NOTHING (Postgres
-- equivalent of SQLite's INSERT OR IGNORE) so re-running this against a
-- partially-populated DB is safe.
--
-- Real production seeds should be 10k+ commodities, 5k+ orgs, etc.

INSERT INTO entity_pools (entity_type, value) VALUES
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

    ('COMMODITY', 'Grade A robusta coffee'),
    ('COMMODITY', '304 stainless steel sheet'),
    ('COMMODITY', 'Polyethylene resin HDPE'),
    ('COMMODITY', 'refined copper cathode'),
    ('COMMODITY', 'galvanized steel coil'),
    ('COMMODITY', 'anhydrous ammonia'),
    ('COMMODITY', 'raw cane sugar'),
    ('COMMODITY', 'wood'),
    ('COMMODITY', 'special wood'),
    ('COMMODITY', 'robusta coffee'),
    ('COMMODITY', 'cotton'),
    ('COMMODITY', 'organic cotton')
ON CONFLICT (entity_type, value) DO NOTHING;

INSERT INTO decoy_pools (slot_name, value) VALUES
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
    ('neg_cue', 'does not contain'),
    ('neg_cue', 'no'),
    ('neg_cue', 'without'),
    ('neg_cue', 'lacks'),
    ('neg_cue', 'excludes'),
    ('neg_cue', 'free of'),
    ('neg_cue', 'absent of'),
    ('neg_cue', 'missing'),
    ('contrast_cue', ', but only'),
    ('contrast_cue', ', only'),
    ('contrast_cue', ', instead'),
    ('contrast_cue', ', rather than'),
    ('frozen_compound', 'sugar-free'),
    ('frozen_compound', 'non-stick'),
    ('frozen_compound', 'stainless'),
    ('frozen_compound', 'lead-free'),
    ('frozen_compound', 'BPA-free'),
    ('frozen_compound', 'gluten-free')
ON CONFLICT (slot_name, value) DO NOTHING;

INSERT INTO templates (template) VALUES
    ('Invoice: {decoy:qty} {COMMODITY} for {ORG}.'),
    ('shipment of {COMMODITY} approved by {PERSON} at {ORG}'),
    ('{decoy:invoice_id} | {COMMODITY} | Qty {decoy:qty} | Sold to: {ORG}, {ADDRESS}'),
    ('{PERSON} from {ORG} confirmed {COMMODITY} shipped to {ADDRESS}.'),
    ('Manifest: {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3} — ETA {PERSON}, {ORG}.'),
    ('Please dispatch {COMMODITY} to {ADDRESS} c/o {PERSON}.'),
    ('{ORG} acknowledges receipt of {COMMODITY} from {PERSON}.'),
    -- Negation templates
    ('{ORG} {decoy:neg_cue} {NEG_COMMODITY}.'),
    ('{decoy:neg_cue} {NEG_COMMODITY#1}{decoy:contrast_cue} {COMMODITY#2}'),
    ('{COMMODITY#1}, but {decoy:neg_cue} {NEG_COMMODITY#2}'),
    ('{decoy:neg_cue} {NEG_COMMODITY#1}, {NEG_COMMODITY#2}, or {NEG_COMMODITY#3}.'),
    ('{PERSON} confirmed {decoy:frozen_compound} {COMMODITY} delivery.'),
    ('no {NEG_COMMODITY#1}; ordinary {COMMODITY#2} acceptable.')
ON CONFLICT (template) DO NOTHING;
