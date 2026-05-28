from __future__ import annotations

import pytest

from ner.data.free_gen import FreeGenEntity, RelocationError, relocate_entities


def test_relocates_exact_match():
    text = "Maria Gonzalez from Acme Trading Co. confirmed shipment."
    ents = [
        FreeGenEntity("PERSON", "Maria Gonzalez"),
        FreeGenEntity("ORG", "Acme Trading Co."),
    ]
    rec = relocate_entities(text, ents)
    rec.validate()
    assert {(e.type, e.start, e.end) for e in rec.entities} == {
        ("PERSON", 0, 14),
        ("ORG", 20, 36),
    }


def test_falls_back_to_case_insensitive():
    text = "shipment by maria gonzalez at acme trading co"
    ents = [FreeGenEntity("PERSON", "Maria Gonzalez")]
    rec = relocate_entities(text, ents)
    rec.validate()
    assert rec.entities[0].text == "maria gonzalez"


def test_distinct_anchors_for_repeated_type():
    text = "ship galvanized steel coil and refined copper cathode"
    ents = [
        FreeGenEntity("COMMODITY", "galvanized steel coil"),
        FreeGenEntity("COMMODITY", "refined copper cathode"),
    ]
    rec = relocate_entities(text, ents)
    rec.validate()
    starts = sorted(e.start for e in rec.entities)
    assert starts == [5, 31]


def test_raises_when_nothing_locates():
    with pytest.raises(RelocationError):
        relocate_entities("hello world", [FreeGenEntity("PERSON", "Nobody Here")])
