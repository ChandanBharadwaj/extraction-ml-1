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
