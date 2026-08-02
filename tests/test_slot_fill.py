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


def test_neg_commodity_placeholder_emits_neg_polarity():
    pools = _pools()
    pools.add_decoy("neg_cue", "does not contain")
    rng = random.Random(0)
    rec = fill_template(
        "{ORG} {decoy:neg_cue} {NEG_COMMODITY}.",
        pools, rng,
    )
    rec.validate()
    by_pol = {e.polarity: e for e in rec.entities if e.type == "COMMODITY"}
    assert "NEG" in by_pol
    assert rec.text[by_pol["NEG"].start:by_pol["NEG"].end] == by_pol["NEG"].text


def test_mixed_pos_and_neg_commodities_in_one_record():
    pools = _pools()
    pools.add_decoy("neg_cue", "no")
    pools.add_decoy("contrast_cue", ", only")
    rng = random.Random(1)
    rec = fill_template(
        "{decoy:neg_cue} {NEG_COMMODITY#1}{decoy:contrast_cue} {COMMODITY#2}.",
        pools, rng,
    )
    rec.validate()
    polarities = sorted(e.polarity for e in rec.entities if e.type == "COMMODITY")
    assert polarities == ["NEG", "POS"]


def test_preserve_spans_recorded_for_negation_cues():
    pools = _pools()
    pools.add_decoy("neg_cue", "does not contain")
    rng = random.Random(2)
    rec = fill_template(
        "Manifest {decoy:neg_cue} {NEG_COMMODITY}.",
        pools, rng,
    )
    spans = rec.meta.get("preserve_spans")
    assert spans is not None
    # The recorded span should align with the cue text in the record.
    a, b = spans[0]
    assert rec.text[a:b] == "does not contain"


def test_preserve_spans_NOT_recorded_for_neutral_decoys():
    pools = _pools()
    rng = random.Random(3)
    rec = fill_template("{decoy:invoice_id} | {COMMODITY}", pools, rng)
    assert "preserve_spans" not in rec.meta


def _family_pools() -> Pools:
    """Pools whose COMMODITY values form head-noun families."""
    pools = _pools()
    pools.add_entity("COMMODITY", "wood")
    pools.add_entity("COMMODITY", "treated wood")
    pools.add_entity("COMMODITY", "special wood")
    pools.add_entity("COMMODITY", "cane sugar")  # head for "raw cane sugar"
    return pools


def test_repeated_indexed_slot_reuses_value():
    pools = _pools()
    rng = random.Random(0)
    rec = fill_template("{ORG#1} | {ORG#1}", pools, rng)
    rec.validate()
    orgs = [e.text for e in rec.entities if e.type == "ORG"]
    assert len(orgs) == 2
    assert orgs[0] == orgs[1]


def test_repeated_indexed_slot_alias_pattern():
    pools = _pools()
    rng = random.Random(5)
    rec = fill_template(
        "Supplier has {COMMODITY#1}; buyer requests {COMMODITY#1}.",
        pools, rng,
    )
    rec.validate()
    values = [e.text for e in rec.entities]
    assert len(values) == 2 and values[0] == values[1]


def test_different_indices_stay_distinct():
    pools = _pools()
    rng = random.Random(0)
    rec = fill_template("{PERSON#1} and {PERSON#2}", pools, rng)
    names = [e.text for e in rec.entities]
    assert len(names) == 2 and names[0] != names[1]


def test_pair_slots_share_head_noun_family():
    pools = _family_pools()
    for seed in range(20):
        rng = random.Random(seed)
        rec = fill_template(
            "does not contain {NEG_COMMODITY~1}, {COMMODITY~1} is fine.",
            pools, rng,
        )
        rec.validate()
        neg = next(e for e in rec.entities if e.polarity == "NEG")
        pos = next(e for e in rec.entities if e.polarity == "POS")
        # NEG must be a qualified form; POS is either its bare head or another
        # family member — either way they share the head-noun suffix.
        assert neg.text != pos.text
        shorter, longer = sorted([neg.text.lower(), pos.text.lower()], key=len)
        assert longer.endswith(" " + shorter) or (
            # both qualified members of the same family: common head suffix
            longer.split()[-1] == shorter.split()[-1]
        )


def test_pair_slot_values_are_deterministic_under_seed():
    pools_a = _family_pools()
    pools_b = _family_pools()
    tmpl = "no {NEG_COMMODITY~1} but {COMMODITY~1} ok"
    rec_a = fill_template(tmpl, pools_a, random.Random(11))
    rec_b = fill_template(tmpl, pools_b, random.Random(11))
    assert rec_a.text == rec_b.text


def test_pair_and_index_on_same_slot_rejected():
    pools = _family_pools()
    rng = random.Random(0)
    with pytest.raises(SlotFillError):
        fill_template("{COMMODITY#1~1}", pools, rng)


def test_pair_on_non_commodity_rejected():
    pools = _family_pools()
    rng = random.Random(0)
    with pytest.raises(SlotFillError):
        fill_template("{PERSON~1} met {PERSON~1}", pools, rng)


def test_pair_without_families_raises():
    pools = _pools()  # no value is a token-boundary suffix of another
    rng = random.Random(0)
    with pytest.raises(SlotFillError):
        fill_template("{NEG_COMMODITY~1} vs {COMMODITY~1}", pools, rng)


def test_contrast_cue_also_recorded_in_preserve_spans():
    pools = _pools()
    pools.add_decoy("neg_cue", "no")
    pools.add_decoy("contrast_cue", ", only")
    rng = random.Random(4)
    rec = fill_template(
        "{decoy:neg_cue} {NEG_COMMODITY#1}{decoy:contrast_cue} {COMMODITY#2}.",
        pools, rng,
    )
    spans = rec.meta["preserve_spans"]
    assert len(spans) == 2
    surfaces = {rec.text[a:b] for (a, b) in spans}
    assert surfaces == {"no", ", only"}
