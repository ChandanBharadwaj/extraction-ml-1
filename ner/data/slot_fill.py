"""Deterministic slot-fill generator with mathematically correct char offsets.

The LLM is never trusted to count characters. We sample entity values and
templates separately, then assemble the text in Python while recording the
exact byte position of each entity insertion.

Template syntax:
    {PERSON} {ORG} {ADDRESS} {COMMODITY}    entity slots (label-bearing)
    {PERSON#1} {PERSON#2}                   indexed slots for multiple of same type
    {decoy:qty} {decoy:invoice_id} {...}    non-entity filler slots

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

# Matches {PERSON}, {ORG#2}, {decoy:qty}, etc.
_SLOT_RE = re.compile(r"\{(?P<kind>[A-Za-z]+)(?:#(?P<idx>\d+))?(?::(?P<sub>[A-Za-z_]+))?\}")


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
        # Fallback: allow repetition if pool too small.
        candidates = list(zip(pool.values, pool.weights))
    if not candidates:
        raise SlotFillError("Empty pool")
    values, weights = zip(*candidates)
    return rng.choices(values, weights=weights, k=1)[0]


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
    used: dict[str, set[str]] = {}
    out_parts: list[str] = []
    entities: list[Entity] = []
    cursor = 0  # position in template
    char_pos = 0  # position in output buffer

    for m in _SLOT_RE.finditer(template):
        # Copy literal between cursor and m.start()
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
            out_parts.append(value)
            char_pos += len(value)
        else:
            entity_type = kind.upper()
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
            entities.append(Entity(entity_type, value, start, char_pos))

        cursor = m.end()

    # Trailing literal.
    trailing = template[cursor:]
    out_parts.append(trailing)

    text = "".join(out_parts)
    record = Record(text=text, entities=entities, meta={"template": template})
    record.validate()  # invariant check
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
