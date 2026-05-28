-- Tiny example seed so the pipeline produces something out of the box.
-- Real production seeds should be 10k+ commodities, 5k+ orgs, etc.

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

    ('COMMODITY', 'Grade A robusta coffee'),
    ('COMMODITY', '304 stainless steel sheet'),
    ('COMMODITY', 'Polyethylene resin HDPE'),
    ('COMMODITY', 'refined copper cathode'),
    ('COMMODITY', 'galvanized steel coil'),
    ('COMMODITY', 'anhydrous ammonia'),
    ('COMMODITY', 'raw cane sugar');

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
    ('invoice_id', 'REF#A-9912');

INSERT OR IGNORE INTO templates (template) VALUES
    ('Invoice: {decoy:qty} {COMMODITY} for {ORG}.'),
    ('shipment of {COMMODITY} approved by {PERSON} at {ORG}'),
    ('{decoy:invoice_id} | {COMMODITY} | Qty {decoy:qty} | Sold to: {ORG}, {ADDRESS}'),
    ('{PERSON} from {ORG} confirmed {COMMODITY} shipped to {ADDRESS}.'),
    ('Manifest: {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3} — ETA {PERSON}, {ORG}.'),
    ('Please dispatch {COMMODITY} to {ADDRESS} c/o {PERSON}.'),
    ('{ORG} acknowledges receipt of {COMMODITY} from {PERSON}.');
