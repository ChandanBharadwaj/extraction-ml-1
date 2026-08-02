"""Deterministic slot-fill generator with mathematically correct char offsets.

The LLM is never trusted to count characters. We sample entity values and
templates separately, then assemble the text in Python while recording the
exact byte position of each entity insertion.

Template syntax:
    {PERSON} {ORG} {ADDRESS} {COMMODITY}    entity slots (label-bearing, POS)
    {NEG_COMMODITY}                          commodity entity with polarity=NEG
    {PERSON#1} {PERSON#2}                    indexed slots: an index names a
                                             variable — the same index repeats
                                             the same value; different indices
                                             force distinct values
    {NEG_COMMODITY#1} {NEG_COMMODITY#2}      same, for the NEG case
    {NEG_COMMODITY~1} ... {COMMODITY~1}      paired slots: all slots sharing a
                                             pair id draw from one head-noun
                                             family (NEG slots get qualified
                                             members like "treated wood", POS
                                             slots usually get the bare head
                                             "wood") — the negation-scope
                                             pattern the gold set tests
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

# Matches {PERSON}, {NEG_COMMODITY#2}, {COMMODITY~1}, {decoy:qty}, etc.
# `kind` allows underscores to permit NEG_COMMODITY. `idx` (#) and `pair` (~)
# are mutually exclusive, enforced in fill_template (and mirrored by the
# validator in scripts/build_seed.py — keep both copies in sync).
_SLOT_RE = re.compile(
    r"\{(?P<kind>[A-Za-z_]+)(?:#(?P<idx>\d+))?(?:~(?P<pair>\d+))?(?::(?P<sub>[A-Za-z_]+))?\}"
)

# Pair (~) slots only make sense where a head-noun family exists, which is a
# commodity-pool concept ("treated wood" -> head "wood").
PAIRABLE_ENTITY_TYPES: frozenset[str] = frozenset({"COMMODITY"})

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


def build_family_index(values: list[str]) -> dict[str, list[str]]:
    """Group pool values into head-noun families.

    A value `q` is a qualified member of family `h` when both are pool values
    and `q` ends with `h` at a token boundary ("Grade A robusta coffee" is in
    the "robusta coffee" family AND the "coffee" family). Matching is
    case-insensitive; returned strings keep pool casing. Only heads with at
    least one qualified member appear as keys.
    """
    by_lower = {v.lower(): v for v in values}
    families: dict[str, list[str]] = {}
    for q in values:
        ql = q.lower()
        # Successive whitespace-delimited suffixes of q are candidate heads.
        pos = ql.find(" ")
        while pos != -1:
            head = by_lower.get(ql[pos + 1:])
            if head is not None and head.lower() != ql:
                families.setdefault(head, []).append(q)
            pos = ql.find(" ", pos + 1)
    return families


def _family_index(pool: WeightedPool) -> dict[str, list[str]]:
    """`build_family_index` over a pool, cached on the pool instance."""
    cached = getattr(pool, "_slotfill_family_index", None)
    if cached is None:
        cached = build_family_index(pool.values)
        pool._slotfill_family_index = cached  # type: ignore[attr-defined]
    return cached


def _assign_pair_groups(
    matches: list[re.Match],
    pools: Pools,
    rng: random.Random,
) -> dict[int, str]:
    """Pre-assign values for every ~pair slot, keyed by match position.

    Per pair group: pick one head-noun family, give NEG slots qualified
    members and POS slots the bare head (with p=0.3 a different qualified
    member — covers "no special wood, treated wood OK"). All values within a
    group are distinct.
    """
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(matches):
        if m.group("pair") is not None and m.group("kind") != "decoy":
            groups.setdefault(m.group("pair"), []).append(i)
    if not groups:
        return {}

    assignments: dict[int, str] = {}
    for pair_id in sorted(groups):
        slot_indices = groups[pair_id]
        etypes = set()
        for i in slot_indices:
            etype, _ = _resolve_entity_slot(matches[i].group("kind"))
            etypes.add(etype)
        if len(etypes) > 1:
            raise SlotFillError(
                f"pair group ~{pair_id} mixes entity types {sorted(etypes)}"
            )
        etype = next(iter(etypes))
        if etype not in PAIRABLE_ENTITY_TYPES:
            raise SlotFillError(
                f"pair slots are only valid for {sorted(PAIRABLE_ENTITY_TYPES)}, "
                f"got ~{pair_id} on {etype}"
            )
        pool = pools.entity_pools.get(etype)
        if pool is None or pool.is_empty():
            raise SlotFillError(f"No entity pool for {etype!r}")
        families = _family_index(pool)
        if not families:
            raise SlotFillError(
                f"pool for {etype!r} has no head-noun families; ~pair slots "
                "need qualified values whose bare head is also in the pool"
            )
        head = rng.choice(list(families))
        qualified = families[head]
        taken: set[str] = set()

        def _pick_qualified() -> str:
            avail = [q for q in qualified if q not in taken]
            if not avail:
                avail = qualified
            return rng.choice(avail)

        # NEG slots first: the denial sticks to a qualified form.
        for i in slot_indices:
            _, polarity = _resolve_entity_slot(matches[i].group("kind"))
            if polarity == "NEG":
                value = _pick_qualified()
                taken.add(value)
                assignments[i] = value
        # POS slots: usually the bare head, sometimes another qualified form.
        # Never duplicate a taken value unless the family is exhausted.
        for i in slot_indices:
            _, polarity = _resolve_entity_slot(matches[i].group("kind"))
            if polarity == "POS":
                avail_q = [q for q in qualified if q not in taken]
                if rng.random() < 0.3 and avail_q:
                    value = rng.choice(avail_q)
                elif head not in taken:
                    value = head
                elif avail_q:
                    value = rng.choice(avail_q)
                else:
                    value = head
                taken.add(value)
                assignments[i] = value
    return assignments


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
    # Exception: ~pair slots intentionally share a head-noun family, so their
    # values are assigned as a group first and only then recorded here.
    used: dict[str, set[str]] = {}
    # Indexed slots are variables: (KIND, idx) -> value sampled at first use.
    assigned_vars: dict[tuple[str, str], str] = {}
    out_parts: list[str] = []
    entities: list[Entity] = []
    preserve_spans: list[tuple[int, int]] = []
    cursor = 0
    char_pos = 0

    matches = list(_SLOT_RE.finditer(template))
    for m in matches:
        if m.group("idx") is not None and m.group("pair") is not None:
            raise SlotFillError(f"slot mixes #index and ~pair: {m.group(0)}")
        if m.group("pair") is not None and m.group("kind") == "decoy":
            raise SlotFillError(f"~pair is not valid on decoy slots: {m.group(0)}")

    pair_values = _assign_pair_groups(matches, pools, rng)
    for i, value in pair_values.items():
        etype, _ = _resolve_entity_slot(matches[i].group("kind"))
        used.setdefault(etype, set()).add(value)

    for i, m in enumerate(matches):
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
            if i in pair_values:
                value = pair_values[i]
            elif m.group("idx") is not None:
                var_key = (kind.upper(), m.group("idx"))
                if var_key in assigned_vars:
                    value = assigned_vars[var_key]
                else:
                    exclude = (
                        used.setdefault(entity_type, set())
                        if distinct_within_record else set()
                    )
                    value = _weighted_choice(rng, pool, exclude)
                    assigned_vars[var_key] = value
                    if distinct_within_record:
                        used[entity_type].add(value)
            else:
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
