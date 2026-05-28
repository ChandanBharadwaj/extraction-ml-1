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
