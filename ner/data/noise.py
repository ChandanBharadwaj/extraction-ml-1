"""Noise injection for synthetic-to-real gap reduction.

Operates on a `Record` and returns a new `Record` whose entity char offsets
have been recomputed against the transformed text. Each transformation either
preserves character positions (lowercasing, char swaps inside a token) or
deletes characters and shifts downstream offsets deterministically.

Preserve spans (e.g. negation cues registered by the slot-fill assembler) are
respected: characters inside those ranges are never deleted and never altered
by typo injection. The ranges themselves are shifted alongside entities when
upstream deletions occur, so they remain valid against the transformed text.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass

from ner.schema import Entity, Record


@dataclass
class NoiseConfig:
    p_lowercase: float = 0.25
    p_drop_punct: float = 0.25
    drop_punct_rate: float = 0.5
    p_typo: float = 0.15
    typo_rate: float = 0.02
    p_truncate: float = 0.05
    truncate_max_chars: int = 8
    preserve_entity_surface: bool = True


_PUNCT = set(",.;:!?-—|")


def _get_preserve_spans(record: Record) -> list[tuple[int, int]]:
    spans = record.meta.get("preserve_spans") if record.meta else None
    return list(spans) if spans else []


def _set_preserve_spans(meta: dict, spans: list[tuple[int, int]]) -> None:
    if spans:
        meta["preserve_spans"] = spans
    elif "preserve_spans" in meta:
        del meta["preserve_spans"]


def _is_in_any(i: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a <= i < b for a, b in ranges)


def _apply_to_record(
    text: str,
    entities: list[Entity],
    deletions: list[int],
    preserve_spans: list[tuple[int, int]] | None = None,
    meta: dict | None = None,
) -> Record:
    """Rebuild a Record after deleting char positions in `deletions` from `text`.

    Entities and `preserve_spans` are re-projected onto the shorter string via
    a single inverse-position map. Both first/last surviving chars are used to
    define new boundaries so neighboring deletions don't bleed spans into each
    other.
    """
    preserve_spans = preserve_spans or []
    if not deletions:
        out_meta = dict(meta) if meta else {}
        _set_preserve_spans(out_meta, preserve_spans)
        return Record(text=text, entities=list(entities), meta=out_meta)

    drop_set = set(deletions)
    new_chars: list[str] = []
    pos_map: list[int] = []
    for i, ch in enumerate(text):
        if i in drop_set:
            continue
        new_chars.append(ch)
        pos_map.append(i)
    new_text = "".join(new_chars)

    inv = [-1] * len(text)
    for new_i, old_i in enumerate(pos_map):
        inv[old_i] = new_i

    def reproject(start: int, end: int) -> tuple[int, int] | None:
        new_start = -1
        for j in range(start, end):
            if inv[j] != -1:
                new_start = inv[j]
                break
        if new_start == -1:
            return None
        new_end = -1
        for j in range(end - 1, start - 1, -1):
            if inv[j] != -1:
                new_end = inv[j] + 1
                break
        if new_end <= new_start:
            return None
        return new_start, new_end

    new_entities: list[Entity] = []
    for ent in entities:
        reprojected = reproject(ent.start, ent.end)
        if reprojected is None:
            continue
        new_start, new_end = reprojected
        new_text_slice = new_text[new_start:new_end]
        if not new_text_slice:
            continue
        new_entities.append(Entity(
            type=ent.type, text=new_text_slice,
            start=new_start, end=new_end,
            polarity=ent.polarity,
        ))

    new_preserve: list[tuple[int, int]] = []
    for (a, b) in preserve_spans:
        rp = reproject(a, b)
        if rp is not None:
            new_preserve.append(rp)

    out_meta = dict(meta) if meta else {}
    _set_preserve_spans(out_meta, new_preserve)
    rec = Record(text=new_text, entities=new_entities, meta=out_meta)
    rec.validate()
    return rec


def apply_noise(record: Record, config: NoiseConfig, rng: random.Random) -> Record:
    text = record.text
    entities = list(record.entities)
    preserve_spans = _get_preserve_spans(record)
    meta = dict(record.meta) if record.meta else {}
    _set_preserve_spans(meta, preserve_spans)

    # 1) Lowercase (offsets preserved).
    if rng.random() < config.p_lowercase:
        text = text.lower()
        entities = [
            Entity(type=e.type, text=text[e.start:e.end],
                   start=e.start, end=e.end, polarity=e.polarity)
            for e in entities
        ]

    # 2) Drop punctuation (offsets shift). Skip chars inside entities or
    #    preserve_spans (negation/contrast cues).
    if rng.random() < config.p_drop_punct:
        deletions = [
            i for i, ch in enumerate(text)
            if ch in _PUNCT and rng.random() < config.drop_punct_rate
            and not (config.preserve_entity_surface
                     and _is_in_any(i, [(e.start, e.end) for e in entities]))
            and not _is_in_any(i, preserve_spans)
        ]
        record = _apply_to_record(text, entities, deletions, preserve_spans, meta)
        text = record.text
        entities = list(record.entities)
        preserve_spans = _get_preserve_spans(record)
        meta = dict(record.meta)

    # 3) Char-level typos (offsets preserved). Skip chars inside entities or
    #    preserve_spans.
    if rng.random() < config.p_typo:
        ent_ranges = [(e.start, e.end) for e in entities]
        new_chars: list[str] = []
        for i, ch in enumerate(text):
            if config.preserve_entity_surface and _is_in_any(i, ent_ranges):
                new_chars.append(ch)
                continue
            if _is_in_any(i, preserve_spans):
                new_chars.append(ch)
                continue
            if ch.isalpha() and rng.random() < config.typo_rate:
                new_chars.append(rng.choice(string.ascii_lowercase))
            else:
                new_chars.append(ch)
        text = "".join(new_chars)
        entities = [
            Entity(type=e.type, text=text[e.start:e.end],
                   start=e.start, end=e.end, polarity=e.polarity)
            for e in entities
        ]

    # 4) Trailing truncation (may delete entities). Block truncation that would
    #    chop into a preserve_span.
    if rng.random() < config.p_truncate and len(text) > 10:
        drop_n = rng.randint(1, config.truncate_max_chars)
        cut_at = len(text) - drop_n
        straddles_entity = any(e.start < cut_at < e.end for e in entities)
        straddles_preserve = any(a < cut_at < b for a, b in preserve_spans)
        ends_inside_preserve = any(a <= cut_at < b for a, b in preserve_spans)
        if not straddles_entity and not straddles_preserve and not ends_inside_preserve:
            deletions = list(range(cut_at, len(text)))
            record = _apply_to_record(text, entities, deletions, preserve_spans, meta)
            text = record.text
            entities = list(record.entities)
            preserve_spans = _get_preserve_spans(record)
            meta = dict(record.meta)

    _set_preserve_spans(meta, preserve_spans)
    out = Record(text=text, entities=entities, meta=meta)
    out.validate()
    return out
