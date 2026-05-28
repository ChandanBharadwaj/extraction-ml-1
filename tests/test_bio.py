"""Round-trip tests for BIO encode/decode.

We fake an offset_mapping that matches a whitespace-tokenized version of each
text — this is sufficient to exercise the encoder/decoder without pulling in
the real DeBERTa tokenizer.
"""
from __future__ import annotations

import pytest

from ner.bio import bio_ids_to_spans, char_spans_to_bio
from ner.constants import LABEL2ID
from ner.schema import Entity


def _whitespace_offsets(text: str) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = [(0, 0)]  # CLS
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
    offsets.append((0, 0))  # SEP
    return offsets


def test_encode_decode_roundtrip_simple():
    text = "Maria Gonzalez from Acme Trading Co."
    entities = [
        Entity("PERSON", "Maria Gonzalez", 0, 14),
        Entity("ORG", "Acme Trading Co.", 20, 36),
    ]
    offsets = _whitespace_offsets(text)
    labels = char_spans_to_bio(entities, offsets)
    # PERSON spans 2 tokens; ORG spans 3 tokens (Acme/Trading/Co.).
    assert labels[0] == -100  # CLS
    assert labels[-1] == -100  # SEP
    assert labels[1] == LABEL2ID["B-PERSON"]
    assert labels[2] == LABEL2ID["I-PERSON"]
    assert labels[3] == LABEL2ID["O"]  # "from"
    assert labels[4] == LABEL2ID["B-ORG"]
    assert labels[5] == LABEL2ID["I-ORG"]
    assert labels[6] == LABEL2ID["I-ORG"]

    decoded = bio_ids_to_spans(labels, offsets, text)
    assert {(e.type, e.text) for e in decoded} == {
        ("PERSON", "Maria Gonzalez"),
        ("ORG", "Acme Trading Co."),
    }


def test_decode_handles_orphan_I_tag_lenient():
    text = "X Acme Y"
    offsets = _whitespace_offsets(text)
    # Force an I-ORG with no preceding B-ORG.
    labels = [LABEL2ID["O"]] * len(offsets)
    labels[0] = -100
    labels[-1] = -100
    labels[2] = LABEL2ID["I-ORG"]  # "Acme"
    decoded = bio_ids_to_spans(labels, offsets, text, strict=False)
    assert any(e.type == "ORG" and e.text == "Acme" for e in decoded)


def test_decode_strict_drops_orphan_I_tag():
    text = "X Acme Y"
    offsets = _whitespace_offsets(text)
    labels = [LABEL2ID["O"]] * len(offsets)
    labels[0] = -100
    labels[-1] = -100
    labels[2] = LABEL2ID["I-ORG"]
    decoded = bio_ids_to_spans(labels, offsets, text, strict=True)
    assert not any(e.type == "ORG" for e in decoded)


def test_serial_commodities_decode_as_distinct_spans():
    text = "Manifest galvanized steel coil anhydrous ammonia raw cane sugar"
    offsets = _whitespace_offsets(text)
    entities = [
        Entity("COMMODITY", "galvanized steel coil", 9, 30),
        Entity("COMMODITY", "anhydrous ammonia", 31, 48),
        Entity("COMMODITY", "raw cane sugar", 49, 63),
    ]
    labels = char_spans_to_bio(entities, offsets)
    decoded = bio_ids_to_spans(labels, offsets, text)
    assert len(decoded) == 3
    assert [e.text for e in decoded] == [
        "galvanized steel coil",
        "anhydrous ammonia",
        "raw cane sugar",
    ]


def test_special_tokens_are_ignored_label():
    text = "Hello Acme"
    offsets = _whitespace_offsets(text)
    labels = char_spans_to_bio([Entity("ORG", "Acme", 6, 10)], offsets)
    assert labels[0] == -100
    assert labels[-1] == -100


def test_neg_commodity_encode_decode_roundtrip():
    text = "no special wood"
    offsets = _whitespace_offsets(text)
    entities = [Entity("COMMODITY", "special wood", 3, 15, polarity="NEG")]
    labels = char_spans_to_bio(entities, offsets)
    assert labels[2] == LABEL2ID["B-NEG_COMMODITY"]
    assert labels[3] == LABEL2ID["I-NEG_COMMODITY"]
    decoded = bio_ids_to_spans(labels, offsets, text)
    assert len(decoded) == 1
    assert decoded[0].type == "COMMODITY"
    assert decoded[0].polarity == "NEG"
    assert decoded[0].text == "special wood"


def test_mixed_polarity_same_sentence_decodes_distinctly():
    text = "no wood only steel"
    offsets = _whitespace_offsets(text)
    entities = [
        Entity("COMMODITY", "wood", 3, 7, polarity="NEG"),
        Entity("COMMODITY", "steel", 13, 18, polarity="POS"),
    ]
    labels = char_spans_to_bio(entities, offsets)
    decoded = bio_ids_to_spans(labels, offsets, text)
    by_polarity = {e.polarity: e for e in decoded}
    assert by_polarity["NEG"].text == "wood"
    assert by_polarity["POS"].text == "steel"


def test_polarity_flip_mid_span_closes_and_reopens():
    # Adjacent tokens: one labeled B-COMMODITY, next labeled I-NEG_COMMODITY.
    # Decoder must close at the flip and not merge them into one span.
    text = "alpha beta"
    offsets = _whitespace_offsets(text)
    labels = [LABEL2ID["O"]] * len(offsets)
    labels[0] = -100
    labels[-1] = -100
    labels[1] = LABEL2ID["B-COMMODITY"]
    labels[2] = LABEL2ID["I-NEG_COMMODITY"]
    decoded = bio_ids_to_spans(labels, offsets, text)
    assert len(decoded) == 2
    assert decoded[0].polarity == "POS"
    assert decoded[1].polarity == "NEG"


def test_apply_threshold_gate_demotes_low_confidence_predictions():
    import numpy as np

    from ner.bio import apply_threshold_gate
    from ner.constants import NUM_LABELS

    # Three tokens; force argmax via one-hot-ish distributions of varying confidence.
    probs = np.zeros((3, NUM_LABELS), dtype=np.float32)
    probs[0, LABEL2ID["B-COMMODITY"]] = 0.95  # high confidence
    probs[0, LABEL2ID["O"]] = 0.05
    probs[1, LABEL2ID["B-COMMODITY"]] = 0.40  # low confidence
    probs[1, LABEL2ID["O"]] = 0.60            # but O wins anyway here
    probs[2, LABEL2ID["B-PERSON"]] = 0.55
    probs[2, LABEL2ID["O"]] = 0.45

    # Thresholds: PERSON requires 0.8, COMMODITY requires 0.5. With these,
    # token 2 (PERSON @ 0.55) should be demoted to O; token 0 (COMMODITY @ 0.95)
    # survives.
    thresholds = np.zeros(NUM_LABELS, dtype=np.float32)
    thresholds[LABEL2ID["B-PERSON"]] = 0.80
    thresholds[LABEL2ID["B-COMMODITY"]] = 0.50
    out = apply_threshold_gate(probs, thresholds)

    assert out[0] == LABEL2ID["B-COMMODITY"]
    assert out[1] == LABEL2ID["O"]  # argmax was already O
    assert out[2] == LABEL2ID["O"]  # demoted from B-PERSON


def test_apply_threshold_gate_with_zero_thresholds_is_noop():
    import numpy as np

    from ner.bio import apply_threshold_gate
    from ner.constants import NUM_LABELS

    probs = np.random.RandomState(0).dirichlet(np.ones(NUM_LABELS), size=10).astype(np.float32)
    thresholds = np.zeros(NUM_LABELS, dtype=np.float32)
    out = apply_threshold_gate(probs, thresholds)
    expected = np.argmax(probs, axis=-1)
    assert np.array_equal(out, expected)
