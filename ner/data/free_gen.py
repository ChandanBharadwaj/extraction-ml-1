"""Free-generation (secondary) data pipeline: LLM emits naturalistic text and a
list of entities; Python relocates each entity in the text by exact match (with
case-insensitive fallback) and computes char offsets deterministically.

This module only handles the relocation step; the LLM call lives in
`ner.llm.claude_generator`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ner.constants import ENTITY_TYPES
from ner.schema import Entity, Record, SpanError


@dataclass
class FreeGenEntity:
    """Untyped, unlocated entity as returned by the LLM."""
    type: str
    text: str


class RelocationError(ValueError):
    pass


def relocate_entities(
    source_text: str,
    entities: list[FreeGenEntity],
    *,
    case_insensitive_fallback: bool = True,
) -> Record:
    """Resolve each (type, surface) -> exact char span in `source_text`.

    Strategy:
      1. Exact, case-sensitive substring search starting from the previous match.
      2. If miss, fall back to case-insensitive search (replacing the surface
         form on the record with the cased version actually in the source).
      3. Skip entities that still can't be located (logged via meta).

    Overlapping/adjacent matches are pushed forward by anchoring the search to
    the end of the last accepted match for that entity type.
    """
    located: list[Entity] = []
    skipped: list[dict[str, str]] = []
    # Anchor cursor per type so two commodities in series both get found.
    cursors: dict[str, int] = {t: 0 for t in ENTITY_TYPES}

    for ent in entities:
        if ent.type not in ENTITY_TYPES:
            skipped.append({"type": ent.type, "text": ent.text, "reason": "unknown_type"})
            continue
        needle = ent.text
        if not needle:
            skipped.append({"type": ent.type, "text": ent.text, "reason": "empty"})
            continue

        anchor = cursors[ent.type]
        idx = source_text.find(needle, anchor)
        actual_text = needle
        if idx == -1 and case_insensitive_fallback:
            m = re.search(re.escape(needle), source_text[anchor:], flags=re.IGNORECASE)
            if m:
                idx = anchor + m.start()
                actual_text = source_text[idx:idx + len(needle)]

        if idx == -1:
            skipped.append({"type": ent.type, "text": needle, "reason": "not_found"})
            continue

        end = idx + len(actual_text)
        try:
            located.append(Entity(ent.type, actual_text, idx, end))
        except SpanError:
            skipped.append({"type": ent.type, "text": needle, "reason": "span_invalid"})
            continue
        cursors[ent.type] = end

    if not located:
        raise RelocationError(
            f"No entities could be relocated in: {source_text[:80]!r}..."
        )

    # Drop overlapping entities (keep the earlier-started one).
    located.sort(key=lambda e: (e.start, e.end))
    deduped: list[Entity] = []
    for ent in located:
        if deduped and ent.start < deduped[-1].end:
            skipped.append({"type": ent.type, "text": ent.text, "reason": "overlap"})
            continue
        deduped.append(ent)

    rec = Record(text=source_text, entities=deduped, meta={"skipped": skipped})
    rec.validate()
    return rec
