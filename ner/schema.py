"""Core data types for records, entities, and training examples.

`Entity` uses half-open character intervals [start, end) — the same convention as
Python slicing — so `text[start:end]` always reconstructs the entity surface form.
This matches the offset_mapping returned by Hugging Face fast tokenizers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ner.constants import ENTITY_TYPES, NEGATABLE_TYPES, POLARITIES


class SpanError(ValueError):
    """Raised when an entity span is malformed or inconsistent with its text."""


@dataclass(frozen=True, slots=True)
class Entity:
    type: str
    text: str
    start: int  # inclusive
    end: int    # exclusive (half-open)
    polarity: str = "POS"  # POS / NEG; NEG is only valid for NEGATABLE_TYPES

    def __post_init__(self) -> None:
        if self.type not in ENTITY_TYPES:
            raise SpanError(f"Unknown entity type: {self.type!r}")
        if self.start < 0 or self.end <= self.start:
            raise SpanError(f"Invalid span [{self.start}, {self.end}) for {self.text!r}")
        if self.polarity not in POLARITIES:
            raise SpanError(f"Unknown polarity: {self.polarity!r}")
        if self.polarity == "NEG" and self.type not in NEGATABLE_TYPES:
            raise SpanError(
                f"polarity='NEG' is only valid for {sorted(NEGATABLE_TYPES)}, "
                f"got type={self.type!r}"
            )

    def validate_against(self, source: str) -> None:
        if self.end > len(source):
            raise SpanError(
                f"Span [{self.start},{self.end}) exceeds source length {len(source)}"
            )
        actual = source[self.start:self.end]
        if actual != self.text:
            raise SpanError(
                f"Span text mismatch: stored={self.text!r} actual={actual!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type, "text": self.text,
            "start": self.start, "end": self.end,
            "polarity": self.polarity,
        }


@dataclass(slots=True)
class Record:
    """A text record with zero or more gold entities."""
    text: str
    entities: list[Entity] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for ent in self.entities:
            ent.validate_against(self.text)
        # No overlapping spans (BIO can't represent overlaps).
        sorted_ents = sorted(self.entities, key=lambda e: e.start)
        for prev, curr in zip(sorted_ents, sorted_ents[1:]):
            if curr.start < prev.end:
                raise SpanError(
                    f"Overlapping entities: {prev.to_dict()} vs {curr.to_dict()}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Record":
        return cls(
            text=d["text"],
            entities=[Entity(**e) for e in d.get("entities", [])],
            meta=d.get("meta", {}),
        )
