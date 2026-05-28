"""Hand-labeled gold validation records.

These are the *sole* source of truth for early stopping and metric reporting.
Synthetic data should never bleed into this set. The seed examples below cover
the canonical edge cases from the TDD plus a curated set of negation records
that exercise the four hard properties of negation handling:

    Scope          — how far the negation reaches across coordinated commodities.
    Target spec.   — denial sticks to a qualifier ("special wood") while the
                     bare form may still be positively asserted in the same record.
    Dissolved cues — "sugar-free", "stainless", "non-stick" are NOT negations
                     in this domain even though they contain negation-looking
                     surface forms.
    Word order     — denied and asserted entities sit on either side of the cue.

Replace / extend this list as your hand-labeling effort grows. Aim for
500-2,000 examples before production training.
"""
from __future__ import annotations

from ner.schema import Entity, Record


def _r(text: str, *entities: tuple[str, str, int, int]) -> Record:
    """TDD-style 4-tuple constructor (type, text, start, end) — POS polarity."""
    rec = Record(text=text, entities=[Entity(t, s, a, b) for t, s, a, b in entities])
    rec.validate()
    return rec


def _n(text: str, *entities: tuple[str, str, int, int, str]) -> Record:
    """Negation-aware 5-tuple constructor (type, text, start, end, polarity)."""
    rec = Record(text=text, entities=[
        Entity(type=t, text=s, start=a, end=b, polarity=p)
        for t, s, a, b, p in entities
    ])
    rec.validate()
    return rec


# 5 canonical TDD examples (positive-only).
TDD_SEED: list[Record] = [
    _r(
        "Invoice: 500 tons of Grade A robusta coffee for Nordwind Logistics GmbH.",
        ("COMMODITY", "Grade A robusta coffee", 21, 43),
        ("ORG", "Nordwind Logistics GmbH", 48, 71),
    ),
    _r(
        "shipment of 304 stainless steel sheet approved by maria gonzalez at acme trading co",
        ("COMMODITY", "304 stainless steel sheet", 12, 37),
        ("PERSON", "maria gonzalez", 50, 64),
        ("ORG", "acme trading co", 68, 83),
    ),
    _r(
        "PO#88231 | Polyethylene resin HDPE | Qty 12,000 kg | Sold to: Delta Packaging, 7 Canal St, Singapore 049320",
        ("COMMODITY", "Polyethylene resin HDPE", 11, 34),
        ("ORG", "Delta Packaging", 62, 77),
        ("ADDRESS", "7 Canal St, Singapore 049320", 79, 107),
    ),
    _r(
        "Maria Gonzalez from Acme Trading Co. confirmed refined copper cathode shipped to 42 Industrial Park Road, Rotterdam, 3011 AB.",
        ("PERSON", "Maria Gonzalez", 0, 14),
        ("ORG", "Acme Trading Co.", 20, 36),
        ("COMMODITY", "refined copper cathode", 47, 69),
        ("ADDRESS", "42 Industrial Park Road, Rotterdam, 3011 AB", 81, 124),
    ),
    _r(
        "Manifest: galvanized steel coil, anhydrous ammonia, raw cane sugar — ETA Felix Yu, Oceanic Freight Co.",
        ("COMMODITY", "galvanized steel coil", 10, 31),
        ("COMMODITY", "anhydrous ammonia", 33, 50),
        ("COMMODITY", "raw cane sugar", 52, 66),
        ("PERSON", "Felix Yu", 73, 81),
        ("ORG", "Oceanic Freight Co.", 83, 102),
    ),
]


# 30 negation-aware records covering scope / target specificity / dissolved
# cues / word order. Offsets verified by find() over the source string.
NEGATION_SEED: list[Record] = [
    _n('Manifest does not contain wood, lead-free paint, or treated wood.',
        ('COMMODITY', 'wood', 26, 30, 'NEG'),
        ('COMMODITY', 'treated wood', 52, 64, 'NEG'),
    ),
    _n('Acme Trading Co. certifies no asbestos, lead, or mercury in this shipment.',
        ('ORG', 'Acme Trading Co.', 0, 16, 'POS'),
        ('COMMODITY', 'asbestos', 30, 38, 'NEG'),
        ('COMMODITY', 'lead', 40, 44, 'NEG'),
        ('COMMODITY', 'mercury', 49, 56, 'NEG'),
    ),
    _n('PO#88231: does not contain raw cane sugar, cotton, or anhydrous ammonia.',
        ('COMMODITY', 'raw cane sugar', 27, 41, 'NEG'),
        ('COMMODITY', 'cotton', 43, 49, 'NEG'),
        ('COMMODITY', 'anhydrous ammonia', 54, 71, 'NEG'),
    ),
    _n('Shipment contains no special wood, but ordinary wood is acceptable.',
        ('COMMODITY', 'special wood', 21, 33, 'NEG'),
        ('COMMODITY', 'wood', 48, 52, 'POS'),
    ),
    _n('No Grade A robusta coffee; standard robusta coffee shipped instead.',
        ('COMMODITY', 'Grade A robusta coffee', 3, 25, 'NEG'),
        ('COMMODITY', 'robusta coffee', 36, 50, 'POS'),
    ),
    _n('Delivery excludes 304 stainless steel sheet; plain steel sheet is fine.',
        ('COMMODITY', '304 stainless steel sheet', 18, 43, 'NEG'),
        ('COMMODITY', 'steel sheet', 51, 62, 'POS'),
    ),
    _n('Order rejects treated wood, accepts wood.',
        ('COMMODITY', 'treated wood', 14, 26, 'NEG'),
        ('COMMODITY', 'wood', 36, 40, 'POS'),
    ),
    _n('Shipment of sugar-free chocolate and stainless steel sheet to Delta Packaging.',
        ('COMMODITY', 'sugar-free chocolate', 12, 32, 'POS'),
        ('COMMODITY', 'stainless steel sheet', 37, 58, 'POS'),
        ('ORG', 'Delta Packaging', 62, 77, 'POS'),
    ),
    _n('BlueRiver Commodities Ltd supplies non-stick cookware coatings.',
        ('ORG', 'BlueRiver Commodities Ltd', 0, 25, 'POS'),
        ('COMMODITY', 'non-stick cookware coatings', 35, 62, 'POS'),
    ),
    _n('Felix Yu confirmed gluten-free flour delivery.',
        ('PERSON', 'Felix Yu', 0, 8, 'POS'),
        ('COMMODITY', 'gluten-free flour', 19, 36, 'POS'),
    ),
    _n('Delivery of lead-free solder approved by Maria Gonzalez.',
        ('COMMODITY', 'lead-free solder', 12, 28, 'POS'),
        ('PERSON', 'Maria Gonzalez', 41, 55, 'POS'),
    ),
    _n('Does not contain copper cathode, only refined copper cathode shipped.',
        ('COMMODITY', 'copper cathode', 17, 31, 'NEG'),
        ('COMMODITY', 'refined copper cathode', 38, 60, 'POS'),
    ),
    _n('Refined copper cathode shipped, but no copper cathode reserves remaining.',
        ('COMMODITY', 'Refined copper cathode', 0, 22, 'POS'),
        ('COMMODITY', 'copper cathode', 39, 53, 'NEG'),
    ),
    _n('Cotton confirmed; organic cotton not available.',
        ('COMMODITY', 'Cotton', 0, 6, 'POS'),
        ('COMMODITY', 'organic cotton', 18, 32, 'NEG'),
    ),
    _n('No anhydrous ammonia in stock; ammonia substitutes acceptable.',
        ('COMMODITY', 'anhydrous ammonia', 3, 20, 'NEG'),
        ('COMMODITY', 'ammonia', 31, 38, 'POS'),
    ),
    _n('Acme Trading Co. shipment lacks raw cane sugar.',
        ('ORG', 'Acme Trading Co.', 0, 16, 'POS'),
        ('COMMODITY', 'raw cane sugar', 32, 46, 'NEG'),
    ),
    _n('This delivery is free of asbestos.',
        ('COMMODITY', 'asbestos', 25, 33, 'NEG'),
    ),
    _n('Without Polyethylene resin HDPE in the manifest.',
        ('COMMODITY', 'Polyethylene resin HDPE', 8, 31, 'NEG'),
    ),
    _n('Container absent of galvanized steel coil.',
        ('COMMODITY', 'galvanized steel coil', 20, 41, 'NEG'),
    ),
    _n('Maria Gonzalez at Acme Trading Co. confirmed: no special wood, only treated wood.',
        ('PERSON', 'Maria Gonzalez', 0, 14, 'POS'),
        ('ORG', 'Acme Trading Co.', 18, 34, 'POS'),
        ('COMMODITY', 'special wood', 49, 61, 'NEG'),
        ('COMMODITY', 'treated wood', 68, 80, 'POS'),
    ),
    _n('Felix Yu reports Oceanic Freight Co. did not deliver refined copper cathode.',
        ('PERSON', 'Felix Yu', 0, 8, 'POS'),
        ('ORG', 'Oceanic Freight Co.', 17, 36, 'POS'),
        ('COMMODITY', 'refined copper cathode', 53, 75, 'NEG'),
    ),
    _n('Nordwind Logistics GmbH excludes hazardous materials: no anhydrous ammonia, no mercury.',
        ('ORG', 'Nordwind Logistics GmbH', 0, 23, 'POS'),
        ('COMMODITY', 'anhydrous ammonia', 57, 74, 'NEG'),
        ('COMMODITY', 'mercury', 79, 86, 'NEG'),
    ),
    _n('shipment does not contain wood, only plastic toys',
        ('COMMODITY', 'wood', 26, 30, 'NEG'),
        ('COMMODITY', 'plastic toys', 37, 49, 'POS'),
    ),
    _n('no robusta coffee in this batch; arabica beans only.',
        ('COMMODITY', 'robusta coffee', 3, 17, 'NEG'),
        ('COMMODITY', 'arabica beans', 33, 46, 'POS'),
    ),
    _n('Sugar-free chocolate confirmed; no raw cane sugar in any product.',
        ('COMMODITY', 'Sugar-free chocolate', 0, 20, 'POS'),
        ('COMMODITY', 'raw cane sugar', 35, 49, 'NEG'),
    ),
    _n('Stainless steel sheet shipped, but no galvanized steel coil this run.',
        ('COMMODITY', 'Stainless steel sheet', 0, 21, 'POS'),
        ('COMMODITY', 'galvanized steel coil', 38, 59, 'NEG'),
    ),
    _n('Delivery to 7 Canal St, Singapore 049320 does not include cotton.',
        ('ADDRESS', '7 Canal St, Singapore 049320', 12, 40, 'POS'),
        ('COMMODITY', 'cotton', 58, 64, 'NEG'),
    ),
    _n('Shipment to 42 Industrial Park Road, Rotterdam, 3011 AB excludes lead-free solder containers.',
        ('ADDRESS', '42 Industrial Park Road, Rotterdam, 3011 AB', 12, 55, 'POS'),
        ('COMMODITY', 'lead-free solder containers', 65, 92, 'NEG'),
    ),
    _n('Acme Trading Co. ships robusta coffee but lacks Grade A robusta coffee at present.',
        ('ORG', 'Acme Trading Co.', 0, 16, 'POS'),
        ('COMMODITY', 'robusta coffee', 23, 37, 'POS'),
        ('COMMODITY', 'Grade A robusta coffee', 48, 70, 'NEG'),
    ),
    _n('Kenji Tanaka noted: no treated wood, no special wood, no plywood in the order.',
        ('PERSON', 'Kenji Tanaka', 0, 12, 'POS'),
        ('COMMODITY', 'treated wood', 23, 35, 'NEG'),
        ('COMMODITY', 'special wood', 40, 52, 'NEG'),
        ('COMMODITY', 'plywood', 57, 64, 'NEG'),
    ),
]


GOLD_SEED: list[Record] = [*TDD_SEED, *NEGATION_SEED]


def load_gold(path: str | None = None) -> list[Record]:
    """Load gold records: from `path` if provided, else the in-memory seed set."""
    if path is None:
        return list(GOLD_SEED)
    from ner.data.assembler import read_jsonl
    return read_jsonl(path)
