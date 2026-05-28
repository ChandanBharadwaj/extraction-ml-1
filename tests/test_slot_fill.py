"""The slot-fill generator is the safety-critical path: gold labels are
computed by Python, so an off-by-one here is a bug in the *training data*.
We exhaustively assert that every generated entity's char span actually
reconstructs the entity's surface form from the record text.
"""
from __future__ import annotations

import random

import pytest

from ner.data.pools import Pools
from ner.data.slot_fill import GenConfig, SlotFillError, fill_template, generate_records


def _pools() -> Pools:
    pools = Pools()
    pools.add_entity("PERSON", "Maria Gonzalez")
    pools.add_entity("PERSON", "Felix Yu")
    pools.add_entity("ORG", "Acme Trading Co.")
    pools.add_entity("ORG", "Nordwind Logistics GmbH")
    pools.add_entity("ADDRESS", "7 Canal St, Singapore 049320")
    pools.add_entity("COMMODITY", "refined copper cathode")
    pools.add_entity("COMMODITY", "anhydrous ammonia")
    pools.add_entity("COMMODITY", "raw cane sugar")
    pools.add_decoy("qty", "500 tons of")
    pools.add_decoy("invoice_id", "PO#88231")
    pools.add_template("{PERSON} from {ORG} shipped {COMMODITY} to {ADDRESS}.")
    pools.add_template("{decoy:invoice_id} | {COMMODITY} | for {ORG}.")
    pools.add_template("Manifest: {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3}.")
    return pools


def test_fill_template_preserves_offsets():
    pools = _pools()
    rng = random.Random(0)
    rec = fill_template(
        "{PERSON} from {ORG} shipped {COMMODITY} to {ADDRESS}.",
        pools, rng,
    )
    for ent in rec.entities:
        assert rec.text[ent.start:ent.end] == ent.text


def test_fill_template_handles_decoys_without_labels():
    pools = _pools()
    rng = random.Random(0)
    rec = fill_template("{decoy:invoice_id} | {COMMODITY} | for {ORG}.", pools, rng)
    assert all(e.type in ("COMMODITY", "ORG") for e in rec.entities)
    assert len(rec.entities) == 2


def test_distinct_within_record_yields_unique_values():
    pools = _pools()
    rng = random.Random(0)
    rec = fill_template(
        "Manifest: {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3}.",
        pools, rng, distinct_within_record=True,
    )
    commodities = [e.text for e in rec.entities if e.type == "COMMODITY"]
    assert len(commodities) == 3
    assert len(set(commodities)) == 3


def test_generate_records_validates_all():
    pools = _pools()
    cfg = GenConfig(seed=7, n_records=200)
    records = generate_records(pools, cfg)
    assert len(records) == 200
    for rec in records:
        rec.validate()  # raises on any span/text mismatch


def test_missing_pool_raises():
    pools = Pools()
    pools.add_entity("PERSON", "X")
    pools.add_entity("ORG", "Y")
    pools.add_entity("ADDRESS", "Z")
    pools.add_entity("COMMODITY", "Q")
    pools.add_template("{decoy:missing} {PERSON}")
    rng = random.Random(0)
    with pytest.raises(SlotFillError):
        fill_template("{decoy:missing} {PERSON}", pools, rng)
