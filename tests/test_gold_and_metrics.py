from __future__ import annotations

from ner.eval.gold import GOLD_SEED, load_gold
from ner.eval.metrics import evaluate


def test_gold_seed_records_are_valid():
    for rec in GOLD_SEED:
        rec.validate()


def test_load_gold_returns_seed_when_no_path():
    recs = load_gold(None)
    assert len(recs) == len(GOLD_SEED)
    assert recs[0].text == GOLD_SEED[0].text


def test_evaluate_perfect_predictions_yields_f1_1():
    preds = [list(r.entities) for r in GOLD_SEED]
    report = evaluate(preds, GOLD_SEED)
    assert report.micro.f1 == 1.0
    for et, m in report.per_type.items():
        if m.support > 0:
            assert m.f1 == 1.0


def test_evaluate_missing_one_span_lowers_recall():
    preds = []
    for r in GOLD_SEED:
        preds.append(list(r.entities[:-1]))  # drop last entity each record
    report = evaluate(preds, GOLD_SEED)
    assert report.micro.recall < 1.0
    assert report.micro.precision == 1.0  # no false positives


def test_evaluate_extra_span_lowers_precision():
    from ner.schema import Entity
    preds = []
    for r in GOLD_SEED:
        extras = list(r.entities)
        extras.append(Entity("PERSON", r.text[:5] or "Aaaaa", 0, 5))
        # Avoid overlap-induced span checks at eval (evaluate doesn't validate records).
        preds.append(extras)
    report = evaluate(preds, GOLD_SEED)
    assert report.micro.precision < 1.0


def test_negation_gold_records_exist_and_validate():
    from ner.eval.gold import NEGATION_SEED
    assert len(NEGATION_SEED) >= 25  # plan target ~30
    for rec in NEGATION_SEED:
        rec.validate()
    # Every NEG entity is a COMMODITY (domain invariant).
    for rec in NEGATION_SEED:
        for e in rec.entities:
            if e.polarity == "NEG":
                assert e.type == "COMMODITY"
    # The seed contains at least one frozen-compound positive (no NEG entity).
    has_dissolved_cue_positive = any(
        not any(e.polarity == "NEG" for e in rec.entities)
        and ("sugar-free" in rec.text or "stainless" in rec.text or "non-stick" in rec.text)
        for rec in NEGATION_SEED
    )
    assert has_dissolved_cue_positive


def test_polarity_mismatch_counts_as_fp_plus_fn():
    """A COMMODITY span with the right type/edges but flipped polarity is a
    miss (FP + FN), not a hit."""
    from ner.schema import Entity
    from ner.eval.gold import NEGATION_SEED

    preds = []
    for rec in NEGATION_SEED:
        flipped = []
        for e in rec.entities:
            if e.type == "COMMODITY":
                new_pol = "POS" if e.polarity == "NEG" else "NEG"
                flipped.append(Entity(e.type, e.text, e.start, e.end, polarity=new_pol))
            else:
                flipped.append(e)
        preds.append(flipped)
    report = evaluate(preds, NEGATION_SEED)
    # COMMODITY F1 should crater (zero tp on COMMODITY of either polarity).
    for bucket in ("COMMODITY(POS)", "COMMODITY(NEG)"):
        if report.per_type[bucket].support > 0:
            assert report.per_type[bucket].f1 == 0.0


def test_per_type_buckets_split_commodity_by_polarity():
    preds = [list(r.entities) for r in GOLD_SEED]
    report = evaluate(preds, GOLD_SEED)
    # Bucket key contract.
    assert "COMMODITY(POS)" in report.per_type
    assert "COMMODITY(NEG)" in report.per_type
    # PERSON / ORG / ADDRESS remain unsplit.
    assert "PERSON" in report.per_type
    assert "ORG" in report.per_type
    assert "ADDRESS" in report.per_type
    # NEG bucket must have non-zero support because NEGATION_SEED includes
    # at least one NEG record.
    assert report.per_type["COMMODITY(NEG)"].support > 0


def test_cargo_gold_coverage_floors():
    """Guard the cargo-register coverage the gold set gained; future edits
    must not silently regress it."""
    from ner.eval.gold import CARGO_SEED, GOLD_SEED, split_gold

    assert len(CARGO_SEED) >= 35

    def caps_record(r):
        letters = [c for c in r.text if c.isalpha()]
        return letters and sum(c.isupper() for c in letters) / len(letters) > 0.9

    assert sum(1 for r in GOLD_SEED if caps_record(r)) >= 5      # telex/B-L caps
    assert any("\n" in r.text for r in GOLD_SEED)                # multi-line
    assert any(len(r.text) > 500 for r in GOLD_SEED)             # windowing pin

    commodity_spans = [
        e for r in GOLD_SEED for e in r.entities if e.type == "COMMODITY"
    ]
    assert sum(1 for e in commodity_spans if len(e.text) > 60) >= 2   # XL
    assert any(len(e.text) > 100 for e in commodity_spans)            # XL+

    addr = sum(1 for r in GOLD_SEED for e in r.entities if e.type == "ADDRESS")
    assert addr >= 8

    # Each deterministic split half keeps enough NEG support to tune on.
    for split in ("earlystop", "tune"):
        half = split_gold(GOLD_SEED, split)
        neg = sum(1 for r in half for e in r.entities if e.polarity == "NEG")
        assert neg >= 15, (split, neg)
