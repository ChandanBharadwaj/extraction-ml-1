"""Tests for the threshold sweep.

We bypass the ONNX model entirely: the sweep operates over a `ProbCache` (a
plain numpy array of per-token, per-label softmax probabilities), so the tests
synthesize a deterministic cache that mimics what a model would emit on the
gold seed. This keeps the sweep tests fast and independent of a trained
checkpoint.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ner.bio import char_spans_to_bio
from ner.constants import LABEL2ID, NUM_LABELS
from ner.eval.gold import GOLD_SEED
from ner.eval.threshold_sweep import (
    ProbCache,
    decode_with_thresholds,
    load_thresholds_json,
    parse_objective,
    sweep,
    thresholds_dict_to_array,
    write_thresholds_json,
)


def _whitespace_offsets(text: str) -> list[tuple[int, int]]:
    """Same toy tokenizer as test_bio uses, so cache shapes are deterministic."""
    offsets: list[tuple[int, int]] = [(0, 0)]
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        offsets.append((i, j))
        i = j
    offsets.append((0, 0))
    return offsets


def _synthetic_cache(records, *, confidence: float = 0.95) -> ProbCache:
    """Build a ProbCache where each token's softmax peaks at the gold label
    with the given confidence. Rest of the mass is distributed uniformly over
    the other labels."""
    probs_list = []
    offsets_list = []
    texts = []
    other = (1.0 - confidence) / (NUM_LABELS - 1)
    for rec in records:
        offsets = _whitespace_offsets(rec.text)
        labels = char_spans_to_bio(rec.entities, offsets)
        n = len(offsets)
        p = np.full((n, NUM_LABELS), other, dtype=np.float32)
        for i, lid in enumerate(labels):
            if lid == -100:
                # special token: peak on O with the same confidence shape
                p[i, :] = other
                p[i, 0] = confidence
            else:
                p[i, :] = other
                p[i, lid] = confidence
        probs_list.append(p)
        offsets_list.append(offsets)
        texts.append(rec.text)
    return ProbCache(probs=probs_list, offsets=offsets_list, texts=texts)


def test_decode_with_zero_thresholds_recovers_gold_labels():
    cache = _synthetic_cache(GOLD_SEED, confidence=0.95)
    thresholds = np.zeros(NUM_LABELS, dtype=np.float32)
    preds = decode_with_thresholds(cache, thresholds)
    # Whitespace tokenization may not perfectly align with our character-level
    # gold spans (multi-word entities like "Maria Gonzalez"). Verify totals
    # instead of strict equality.
    from ner.eval.metrics import evaluate
    rep = evaluate(preds, GOLD_SEED)
    # With near-one-hot probabilities aligned to gold, micro recall should be
    # high; we don't insist on 1.0 because the toy tokenizer may merge or
    # split entities differently from a real BPE tokenizer.
    assert rep.micro.recall > 0.5


def test_sweep_at_zero_is_a_valid_baseline():
    """With confidence ~0.95 everywhere, the optimal global threshold should be
    near 0 (we get the gold answer with no gating needed)."""
    cache = _synthetic_cache(GOLD_SEED, confidence=0.95)
    result = sweep(cache, GOLD_SEED, "f1_micro", step=0.05)
    # All non-O thresholds are below 0.95 (otherwise everything is demoted).
    for label, t in result.thresholds.items():
        assert t < 0.95, f"{label} threshold too high: {t}"
    assert result.feasible


def test_sweep_demotes_low_confidence_noise():
    """Inject noise: a chunk of records get peak confidence 0.4 on incorrect
    labels. The sweep should raise the threshold for the noisy label to
    suppress its predictions."""
    cache_clean = _synthetic_cache(GOLD_SEED, confidence=0.95)

    # Add a fake "noisy" prediction at a high-frequency wrong label.
    noisy_label = LABEL2ID["B-PERSON"]
    for p in cache_clean.probs:
        # Boost B-PERSON across all tokens to 0.30 (still below O, but close
        # enough that argmax sometimes prefers it depending on what we set O to).
        # We set O to 0.25 so B-PERSON wins on tokens that weren't already
        # assigned by gold.
        for i in range(p.shape[0]):
            if np.argmax(p[i]) == 0:  # currently predicting O
                p[i, noisy_label] = 0.30
                p[i, 0] = 0.25

    result = sweep(cache_clean, GOLD_SEED, "f1_micro", step=0.05)
    assert result.feasible
    # Threshold on B-PERSON should be above the noise level (0.30) to demote
    # the noisy predictions.
    assert result.thresholds["B-PERSON"] >= 0.30


def test_objective_parser_accepts_known_names():
    parse_objective("f1_micro")
    parse_objective("f1_macro")
    parse_objective("f1_per_type:COMMODITY(POS)")
    parse_objective("max_f1_at_precision_floor", precision_floor=0.9)
    parse_objective("max_f1_at_recall_floor", recall_floor=0.5)


def test_objective_parser_rejects_unknown():
    with pytest.raises(ValueError):
        parse_objective("bogus")


def test_infeasible_precision_floor_marks_result_infeasible():
    # No matter the thresholds, perfect precision in this toy setup is unlikely
    # because the toy tokenizer's offsets don't perfectly align with gold
    # character edges. We set a precision floor of 0.99999 which should be
    # unreachable.
    cache = _synthetic_cache(GOLD_SEED, confidence=0.95)
    result = sweep(
        cache, GOLD_SEED, "max_f1_at_precision_floor",
        step=0.1, precision_floor=0.999999,
    )
    assert not result.feasible


def test_thresholds_dict_to_array_round_trip(tmp_path):
    arr = np.zeros(NUM_LABELS, dtype=np.float32)
    arr[LABEL2ID["B-COMMODITY"]] = 0.71
    arr[LABEL2ID["I-COMMODITY"]] = 0.63
    arr[LABEL2ID["B-NEG_COMMODITY"]] = 0.81

    # Synthesize a SweepResult-like object minimally (we only need the labels
    # for round-trip).
    from ner.constants import ID2LABEL
    d = {ID2LABEL[i]: float(arr[i]) for i in range(NUM_LABELS) if ID2LABEL[i] != "O"}
    out_path = tmp_path / "thresholds.json"
    out_path.write_text(json.dumps({"labels": d}))
    loaded = load_thresholds_json(out_path)
    for lid in range(NUM_LABELS):
        assert abs(float(loaded[lid]) - float(arr[lid])) < 1e-6


def test_write_thresholds_json_is_readable_by_runtime_loader(tmp_path):
    from ner.eval.threshold_sweep import SweepResult
    from ner.eval.metrics import EvalReport, TypeMetrics
    fake_rep = EvalReport(
        per_type={"COMMODITY(POS)": TypeMetrics(1.0, 1.0, 1.0, 5, 5)},
        micro=TypeMetrics(1.0, 1.0, 1.0, 5, 5),
    )
    result = SweepResult(
        thresholds={"B-COMMODITY": 0.5, "I-COMMODITY": 0.4,
                    "B-NEG_COMMODITY": 0.7, "I-NEG_COMMODITY": 0.6,
                    "B-PERSON": 0.0, "I-PERSON": 0.0,
                    "B-ORG": 0.0, "I-ORG": 0.0,
                    "B-ADDRESS": 0.0, "I-ADDRESS": 0.0},
        report=fake_rep,
        objective="f1_micro",
        feasible=True,
    )
    out = tmp_path / "thresholds.json"
    write_thresholds_json(result, out, gold_set_path=None, gold_records=5)
    arr = load_thresholds_json(out)
    assert arr[LABEL2ID["B-COMMODITY"]] == pytest.approx(0.5)
    assert arr[LABEL2ID["B-NEG_COMMODITY"]] == pytest.approx(0.7)
    assert arr[0] == 0.0  # O never gated
