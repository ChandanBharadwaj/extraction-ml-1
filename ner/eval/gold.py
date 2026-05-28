"""Hand-labeled gold validation records.

These are the *sole* source of truth for early stopping and metric reporting.
Synthetic data should never bleed into this set. The seed examples below are
the canonical edge cases from the TDD plus a few extras covering address
formats, ALL-CAPS commodities, and commodity adjacency.

Replace / extend this list as your hand-labeling effort grows. Aim for
500-2,000 examples before production training.
"""
from __future__ import annotations

from ner.schema import Entity, Record


def _r(text: str, *entities: tuple[str, str, int, int]) -> Record:
    rec = Record(text=text, entities=[Entity(t, s, a, b) for t, s, a, b in entities])
    rec.validate()
    return rec


GOLD_SEED: list[Record] = [
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


def load_gold(path: str | None = None) -> list[Record]:
    """Load gold records: from `path` if provided, else the in-memory seed set."""
    if path is None:
        return list(GOLD_SEED)
    from ner.data.assembler import read_jsonl
    return read_jsonl(path)
