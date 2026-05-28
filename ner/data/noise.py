"""Noise injection for synthetic-to-real gap reduction.

Operates on a `Record` and returns a new `Record` whose entity char offsets
have been recomputed against the transformed text. Each transformation either
preserves character positions (lowercasing, char swaps inside a token) or
deletes characters and shifts downstream offsets deterministically.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass

from ner.schema import Entity, Record


@dataclass
class NoiseConfig:
    p_lowercase: float = 0.25            # full-record lowercasing
    p_drop_punct: float = 0.25           # drop random punctuation
    drop_punct_rate: float = 0.5         # if dropping, what fraction
    p_typo: float = 0.15                 # single-char typos
    typo_rate: float = 0.02              # per-character typo probability
    p_truncate: float = 0.05             # truncate trailing chars
    truncate_max_chars: int = 8
    preserve_entity_surface: bool = True # never edit chars inside entity spans (keeps gold spans 1:1 matchable)


_PUNCT = set(",.;:!?-—|")


def _apply_to_record(text: str, entities: list[Entity], deletions: list[int]) -> Record:
    """Rebuild a Record after deleting char positions in `deletions` from `text`."""
    if not deletions:
        return Record(text=text, entities=entities)

    drop_set = set(deletions)
    new_chars: list[str] = []
    pos_map: list[int] = []  # new index -> old index
    for i, ch in enumerate(text):
        if i in drop_set:
            continue
        new_chars.append(ch)
        pos_map.append(i)
    new_text = "".join(new_chars)

    # Build inverse map: old index -> new index (or -1 if dropped).
    # We don't include the sentinel for len(text) — entity end is exclusive and
    # is resolved as `inv[last_surviving_char] + 1`.
    inv = [-1] * len(text)
    for new_i, old_i in enumerate(pos_map):
        inv[old_i] = new_i

    new_entities: list[Entity] = []
    for ent in entities:
        # First surviving char inside the entity becomes new_start.
        new_start = -1
        for j in range(ent.start, ent.end):
            if inv[j] != -1:
                new_start = inv[j]
                break
        if new_start == -1:
            continue  # entity fully erased
        # Last surviving char inside the entity defines new_end (exclusive).
        new_end = -1
        for j in range(ent.end - 1, ent.start - 1, -1):
            if inv[j] != -1:
                new_end = inv[j] + 1
                break
        if new_end <= new_start:
            continue
        new_text_slice = new_text[new_start:new_end]
        if not new_text_slice:
            continue
        new_entities.append(Entity(ent.type, new_text_slice, new_start, new_end))

    rec = Record(text=new_text, entities=new_entities)
    rec.validate()
    return rec


def _is_in_entity(i: int, entities: list[Entity]) -> bool:
    return any(e.start <= i < e.end for e in entities)


def apply_noise(record: Record, config: NoiseConfig, rng: random.Random) -> Record:
    text = record.text
    entities = list(record.entities)

    # 1) Lowercase (offsets preserved).
    if rng.random() < config.p_lowercase:
        text = text.lower()
        entities = [
            Entity(e.type, text[e.start:e.end], e.start, e.end) for e in entities
        ]

    # 2) Drop punctuation (offsets shift).
    if rng.random() < config.p_drop_punct:
        deletions = [
            i for i, ch in enumerate(text)
            if ch in _PUNCT and rng.random() < config.drop_punct_rate
            and not (config.preserve_entity_surface and _is_in_entity(i, entities))
        ]
        record = _apply_to_record(text, entities, deletions)
        text = record.text
        entities = list(record.entities)

    # 3) Char-level typos (offsets preserved).
    if rng.random() < config.p_typo:
        new_chars: list[str] = []
        for i, ch in enumerate(text):
            if (
                config.preserve_entity_surface
                and _is_in_entity(i, entities)
            ):
                new_chars.append(ch)
                continue
            if ch.isalpha() and rng.random() < config.typo_rate:
                new_chars.append(rng.choice(string.ascii_lowercase))
            else:
                new_chars.append(ch)
        text = "".join(new_chars)
        entities = [
            Entity(e.type, text[e.start:e.end], e.start, e.end) for e in entities
        ]

    # 4) Trailing truncation (may delete entities).
    if rng.random() < config.p_truncate and len(text) > 10:
        drop_n = rng.randint(1, config.truncate_max_chars)
        deletions = list(range(len(text) - drop_n, len(text)))
        # Only do it if no entity is straddled (clean truncation).
        if not any(e.start < len(text) - drop_n < e.end for e in entities):
            record = _apply_to_record(text, entities, deletions)
            text = record.text
            entities = list(record.entities)

    out = Record(text=text, entities=entities, meta=record.meta)
    out.validate()
    return out
