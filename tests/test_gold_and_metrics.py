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
