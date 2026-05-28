"""Deterministic slot-fill generator with mathematically correct char offsets.

The LLM is never trusted to count characters. We sample entity values and
templates separately, then assemble the text in Python while recording the
exact byte position of each entity insertion.

Template syntax:
    {PERSON} {ORG} {ADDRESS} {COMMODITY}    entity slots (label-bearing, POS)
    {NEG_COMMODITY}                          commodity entity with polarity=NEG
    {PERSON#1} {PERSON#2}                    indexed slots for multiple of same type
    {NEG_COMMODITY#1} {NEG_COMMODITY#2}      same, for the NEG case
    {decoy:qty} {decoy:invoice_id} {...}     non-entity filler slots

Preserve spans:
    When a decoy slot whose name is in `PRESERVE_DECOY_SLOTS` (negation /
    contrast cues) is filled, its character range is recorded in
    `record.meta["preserve_spans"]`. The noise injector must not delete
    characters inside these ranges — a dropped "no" would silently flip the
    gold polarity without updating the labels.

Generated entities use the surface form of the sampled value verbatim (after
any noise transformations have been applied), so `text[start:end] == entity.text`
is an invariant that the schema validator enforces.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from ner.data.pools import Pools, WeightedPool
from ner.schema import Entity, Record

# Matches {PERSON}, {NEG_COMMODITY#2}, {decoy:qty}, etc.
# `kind` allows underscores to permit NEG_COMMODITY.
_SLOT_RE = re.compile(r"\{(?P<kind>[A-Za-z_]+)(?:#(?P<idx>\d+))?(?::(?P<sub>[A-Za-z_]+))?\}")

# Decoy slot names whose char ranges must be protected from noise. Drops here
# would change the polarity of nearby commodity entities without re-labeling.
PRESERVE_DECOY_SLOTS: frozenset[str] = frozenset({"neg_cue", "contrast_cue"})


@dataclass
class GenConfig:
    seed: int = 0
    n_records: int = 1000
    # Sample without replacement *within* one record (so {PERSON#1} != {PERSON#2}).
    distinct_within_record: bool = True


class SlotFillError(ValueError):
    pass


def _weighted_choice(rng: random.Random, pool: WeightedPool, exclude: set[str]) -> str:
    """Pick a value from `pool` whose surface form is not already in `exclude`."""
    candidates = [(v, w) for v, w in zip(pool.values, pool.weights) if v not in exclude]
    if not candidates:
        candidates = list(zip(pool.values, pool.weights))
    if not candidates:
        raise SlotFillError("Empty pool")
    values, weights = zip(*candidates)
    return rng.choices(values, weights=weights, k=1)[0]


def _resolve_entity_slot(kind: str) -> tuple[str, str]:
    """Map a slot kind to (entity_type, polarity). `kind` is case-folded."""
    upper = kind.upper()
    if upper.startswith("NEG_"):
        return upper[len("NEG_"):], "NEG"
    return upper, "POS"


def fill_template(
    template: str,
    pools: Pools,
    rng: random.Random,
    *,
    distinct_within_record: bool = True,
) -> Record:
    """Fill one template into a Record with correct char offsets.

    Implementation: walk the template left-to-right, copying literal chunks into
    a buffer and resolving each `{...}` slot to a sampled value. The buffer
    length at slot insertion time is the entity's start offset.
    """
    # Distinct-sample tracking is keyed by entity_type, NOT polarity — drawing
    # the same commodity value as both POS and NEG in one record is degenerate.
    used: dict[str, set[str]] = {}
    out_parts: list[str] = []
    entities: list[Entity] = []
    preserve_spans: list[tuple[int, int]] = []
    cursor = 0
    char_pos = 0

    for m in _SLOT_RE.finditer(template):
        literal = template[cursor:m.start()]
        out_parts.append(literal)
        char_pos += len(literal)

        kind = m.group("kind")
        sub = m.group("sub")

        if kind == "decoy":
            if not sub:
                raise SlotFillError(f"decoy slot missing sub-name: {m.group(0)}")
            pool = pools.decoy_pools.get(sub)
            if pool is None or pool.is_empty():
                raise SlotFillError(f"No decoy pool for {sub!r}")
            value = _weighted_choice(rng, pool, set())
            start = char_pos
            out_parts.append(value)
            char_pos += len(value)
            if sub in PRESERVE_DECOY_SLOTS:
                preserve_spans.append((start, char_pos))
        else:
            entity_type, polarity = _resolve_entity_slot(kind)
            pool = pools.entity_pools.get(entity_type)
            if pool is None or pool.is_empty():
                raise SlotFillError(f"No entity pool for {entity_type!r}")
            exclude = used.setdefault(entity_type, set()) if distinct_within_record else set()
            value = _weighted_choice(rng, pool, exclude)
            if distinct_within_record:
                used[entity_type].add(value)
            start = char_pos
            out_parts.append(value)
            char_pos += len(value)
            entities.append(Entity(
                type=entity_type, text=value,
                start=start, end=char_pos,
                polarity=polarity,
            ))

        cursor = m.end()

    out_parts.append(template[cursor:])
    text = "".join(out_parts)
    meta: dict = {"template": template}
    if preserve_spans:
        meta["preserve_spans"] = preserve_spans
    record = Record(text=text, entities=entities, meta=meta)
    record.validate()
    return record


def generate_records(pools: Pools, config: GenConfig) -> list[Record]:
    pools.validate()
    rng = random.Random(config.seed)
    out: list[Record] = []
    for _ in range(config.n_records):
        tmpl = rng.choices(pools.templates.values, weights=pools.templates.weights, k=1)[0]
        out.append(
            fill_template(
                tmpl, pools, rng,
                distinct_within_record=config.distinct_within_record,
            )
        )
    return out
