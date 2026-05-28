import pytest

from ner.schema import Entity, Record, SpanError


def test_entity_validates_against_source():
    text = "Hello Alice."
    e = Entity("PERSON", "Alice", 6, 11)
    e.validate_against(text)


def test_entity_rejects_wrong_text():
    text = "Hello Alice."
    e = Entity("PERSON", "Bob", 6, 11)
    with pytest.raises(SpanError):
        e.validate_against(text)


def test_record_rejects_overlapping_spans():
    text = "Maria Gonzalez Patel"
    rec = Record(
        text=text,
        entities=[
            Entity("PERSON", "Maria Gonzalez", 0, 14),
            Entity("PERSON", "Gonzalez Patel", 6, 20),
        ],
    )
    with pytest.raises(SpanError):
        rec.validate()


def test_record_round_trip_dict():
    rec = Record(text="Acme.", entities=[Entity("ORG", "Acme", 0, 4)])
    rec.validate()
    rec2 = Record.from_dict(rec.to_dict())
    rec2.validate()
    assert rec2.text == rec.text
    assert rec2.entities[0].to_dict() == rec.entities[0].to_dict()


def test_polarity_defaults_to_pos():
    e = Entity("COMMODITY", "wood", 0, 4)
    assert e.polarity == "POS"


def test_polarity_neg_allowed_on_commodity():
    e = Entity("COMMODITY", "wood", 0, 4, polarity="NEG")
    assert e.polarity == "NEG"


def test_polarity_neg_rejected_on_non_commodity():
    for et in ("PERSON", "ORG", "ADDRESS"):
        with pytest.raises(SpanError):
            Entity(et, "x", 0, 1, polarity="NEG")


def test_unknown_polarity_rejected():
    with pytest.raises(SpanError):
        Entity("COMMODITY", "x", 0, 1, polarity="MAYBE")


def test_legacy_json_without_polarity_round_trips_as_pos():
    rec = Record.from_dict({
        "text": "wood",
        "entities": [{"type": "COMMODITY", "text": "wood", "start": 0, "end": 4}],
    })
    rec.validate()
    assert rec.entities[0].polarity == "POS"
    # Writing back includes polarity.
    assert rec.entities[0].to_dict()["polarity"] == "POS"


def test_neg_entity_round_trips_through_dict():
    e = Entity("COMMODITY", "wood", 0, 4, polarity="NEG")
    e2 = Entity(**e.to_dict())
    assert e2.polarity == "NEG"
    assert e == e2
