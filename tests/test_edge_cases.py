"""Frozen edge-case smoke suite EC-01..EC-10 (docs/data_specification.md §17).

These records live forever and never change; any behavioral shift here is a
regression alert. Everything runs without a trained model: label round-trips
go through the BIO encoder/decoder with a whitespace toy tokenizer, and the
runtime cases use the same stub-session pattern as test_runtime_window.
"""
from __future__ import annotations

import numpy as np
import pytest

from ner.bio import bio_ids_to_spans, char_spans_to_bio
from ner.constants import NUM_LABELS
from ner.infer.runtime import NERRuntime, NERRuntimeConfig
from ner.preprocess import Preprocessor
from ner.schema import Entity, Record


def _whitespace_offsets(text: str) -> list[tuple[int, int]]:
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


def _roundtrip(text: str, entities: list[Entity]) -> list[Entity]:
    rec = Record(text=text, entities=entities)
    rec.validate()
    offsets = _whitespace_offsets(text)
    labels = char_spans_to_bio(entities, offsets)
    return bio_ids_to_spans(labels, offsets, text)


def _fake_runtime(**config_kwargs) -> NERRuntime:
    rt = NERRuntime.__new__(NERRuntime)
    rt.config = NERRuntimeConfig(onnx_path="", tokenizer_dir="", **config_kwargs)
    rt._session = None
    rt._tokenizer = None
    rt.thresholds = None
    rt.preprocessor = Preprocessor()
    return rt


def test_ec01_empty_input_yields_no_entities():
    rt = _fake_runtime()
    assert rt.predict("") == []


def test_ec02_whitespace_only_input_yields_no_entities():
    rt = _fake_runtime()
    assert rt.predict("   ") == []


def test_ec03_bare_acme_resolves_as_org():
    # Documented resolution: a bare company-style token with no person
    # context is labeled ORG. The label space must round-trip it faithfully.
    decoded = _roundtrip("Acme", [Entity("ORG", "Acme", 0, 4)])
    assert [(e.type, e.text) for e in decoded] == [("ORG", "Acme")]


def test_ec04_tdd_canonical_four_entities():
    text = ("Maria Gonzalez from Acme Trading Co. confirmed refined copper "
            "cathode shipped to 42 Industrial Park Road, Rotterdam, 3011 AB.")
    entities = [
        Entity("PERSON", "Maria Gonzalez", 0, 14),
        Entity("ORG", "Acme Trading Co.", 20, 36),
        Entity("COMMODITY", "refined copper cathode", 47, 69),
        Entity("ADDRESS", "42 Industrial Park Road, Rotterdam, 3011 AB", 81, 124),
    ]
    decoded = _roundtrip(text, entities)
    # The trailing period after the ADDRESS is outside every span; all four
    # entities survive with types intact.
    assert sorted(e.type for e in decoded) == ["ADDRESS", "COMMODITY", "ORG", "PERSON"]
    by_type = {e.type: e for e in decoded}
    assert by_type["COMMODITY"].text == "refined copper cathode"
    assert by_type["PERSON"].text == "Maria Gonzalez"


def test_ec05_serial_commodities_manifest():
    text = ("Manifest: galvanized steel coil, anhydrous ammonia, raw cane "
            "sugar — ETA Felix Yu, Oceanic Freight Co.")
    entities = [
        Entity("COMMODITY", "galvanized steel coil", 10, 31),
        Entity("COMMODITY", "anhydrous ammonia", 33, 50),
        Entity("COMMODITY", "raw cane sugar", 52, 66),
        Entity("PERSON", "Felix Yu", 73, 81),
        Entity("ORG", "Oceanic Freight Co.", 83, 102),
    ]
    decoded = _roundtrip(text, entities)
    commodities = [e for e in decoded if e.type == "COMMODITY"]
    assert len(commodities) == 3
    for e in decoded:
        assert text[e.start:e.end] == e.text


def test_ec06_bare_negated_commodity():
    text = "does not contain wood"
    decoded = _roundtrip(
        text, [Entity("COMMODITY", "wood", 17, 21, polarity="NEG")],
    )
    assert len(decoded) == 1
    assert (decoded[0].text, decoded[0].polarity) == ("wood", "NEG")


def test_ec07_qualified_denial_bare_assertion():
    text = "no special wood, but ordinary wood is fine"
    decoded = _roundtrip(text, [
        Entity("COMMODITY", "special wood", 3, 15, polarity="NEG"),
        Entity("COMMODITY", "wood", 30, 34, polarity="POS"),
    ])
    by_pol = {e.polarity: e.text for e in decoded}
    assert by_pol == {"NEG": "special wood", "POS": "wood"}


def test_ec08_frozen_compounds_stay_positive():
    text = "shipment of sugar-free chocolate and stainless steel sheet"
    decoded = _roundtrip(text, [
        Entity("COMMODITY", "sugar-free chocolate", 12, 32),
        Entity("COMMODITY", "stainless steel sheet", 37, 58),
    ])
    assert all(e.polarity == "POS" for e in decoded)
    assert len(decoded) == 2


def test_ec09_zero_width_and_nbsp_project_to_original_coords():
    # "Maria Gonzalez" with a zero-width space inside and an NBSP separator.
    raw = "attn: Maria Gon​zalez"
    pre = Preprocessor().clean(raw)
    assert "​" not in pre.text and " " not in pre.text
    # The cleaned surface is contiguous; find it and project back.
    start = pre.text.find("Maria Gonzalez")
    assert start != -1
    ent = Entity("PERSON", "Maria Gonzalez", start, start + len("Maria Gonzalez"))
    projected = pre.project_entity(ent, raw)
    # Offsets index the RAW caller text and re-slice to the raw surface,
    # zero-width char included.
    assert raw[projected.start:projected.end] == projected.text
    assert projected.text.replace("​", "") == "Maria Gonzalez"


def test_ec10_entity_near_char_cap_is_preserved_and_beyond_cap_warns():
    from ner.constants import LABEL2ID as L

    CLS, SEP, PLAIN, B_TOK = 101, 102, 5, 7

    class _IdSession:
        def run(self, _out_names, feed):
            ids = feed["input_ids"]
            logits = np.zeros((*ids.shape, NUM_LABELS), dtype=np.float32)
            logits[..., L["O"]] = 1.0
            logits[ids == B_TOK] = 0.0
            logits[ids == B_TOK, L["B-COMMODITY"]] = 5.0
            return [logits]

    def make_rt(entity_char: int, **cfg):
        rt = _fake_runtime(**cfg)
        rt._session = _IdSession()

        def fake_encode_full(text: str):
            ids = [CLS] + [
                B_TOK if i == entity_char else PLAIN for i in range(len(text))
            ] + [SEP]
            offsets = [(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)]
            return ids, offsets

        rt._encode_full = fake_encode_full  # type: ignore[method-assign]
        return rt

    # A 500-char record — historically the truncation cliff — now fits well
    # inside the cap; an entity at char 499 survives via windowed inference.
    rt = make_rt(entity_char=499)
    ents = rt.predict("a" * 500)
    assert [(e.start, e.end) for e in ents] == [(499, 500)]

    # Beyond max_input_chars the runtime truncates loudly, and nothing past
    # the cap is scored.
    rt = make_rt(entity_char=460, max_input_chars=450)
    with pytest.warns(UserWarning, match="truncated"):
        ents = rt.predict("a" * 500)
    assert ents == []
