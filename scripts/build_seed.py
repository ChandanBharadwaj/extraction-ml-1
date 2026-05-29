"""Build the production seed SQL from the `scripts.seedgen` data modules.

Emits the Postgres seed from the source-of-truth Python lists:
  - sql/postgres/seed.sql  (Postgres dialect: ON CONFLICT DO NOTHING)

Run:
    python -m scripts.build_seed                 # write the file
    python -m scripts.build_seed --check         # validate only, no write

The seedgen modules hold *plain* surface strings (no SQL escaping). This
builder deduplicates (order-preserving), validates that every {decoy:<slot>}
referenced by a template has a backing pool and that the entity/decoy pools
are non-empty, then escapes and emits the SQL. `build_pools()` returns the same
data as an in-memory `Pools` object (no DB round-trip) for tests / offline use.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ner.constants import ENTITY_TYPES
from ner.data.pools import Pools
from scripts.seedgen import ALLOWED_DECOY_SLOTS
from scripts.seedgen.addresses import ADDRESSES
from scripts.seedgen.commodities import COMMODITIES
from scripts.seedgen.decoys import DECOYS
from scripts.seedgen.orgs import ORGS
from scripts.seedgen.persons import PERSONS
from scripts.seedgen.templates import TEMPLATES

REPO_ROOT = Path(__file__).resolve().parents[1]
POSTGRES_OUT = REPO_ROOT / "sql" / "postgres" / "seed.sql"

# Mirror of ner.data.slot_fill._SLOT_RE so we can validate templates without
# importing the generator's runtime dependencies.
_SLOT_RE = re.compile(r"\{(?P<kind>[A-Za-z_]+)(?:#(?P<idx>\d+))?(?::(?P<sub>[A-Za-z_]+))?\}")

_ROWS_PER_STATEMENT = 200


def _dedup(values: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _entity_pools() -> dict[str, list[str]]:
    return {
        "PERSON": _dedup(PERSONS),
        "ORG": _dedup(ORGS),
        "ADDRESS": _dedup(ADDRESSES),
        "COMMODITY": _dedup(COMMODITIES),
    }


def _decoy_pools() -> dict[str, list[str]]:
    return {slot: _dedup(vals) for slot, vals in DECOYS.items()}


def _templates() -> list[str]:
    return _dedup(TEMPLATES)


def validate() -> dict[str, object]:
    """Validate the pools/templates contract; raise ValueError on any breach.

    Returns a summary dict of counts for reporting.
    """
    entity_pools = _entity_pools()
    decoy_pools = _decoy_pools()
    templates = _templates()
    errors: list[str] = []

    for et in ENTITY_TYPES:
        if not entity_pools.get(et):
            errors.append(f"entity pool {et!r} is empty")

    for slot in ALLOWED_DECOY_SLOTS:
        if not decoy_pools.get(slot):
            errors.append(f"decoy pool {slot!r} is missing or empty")
    extra = set(decoy_pools) - set(ALLOWED_DECOY_SLOTS)
    if extra:
        errors.append(f"decoy pools contain undeclared slots: {sorted(extra)}")

    if not templates:
        errors.append("no templates")

    valid_entity_kinds = set(ENTITY_TYPES) | {f"NEG_{et}" for et in ENTITY_TYPES}
    for tmpl in templates:
        for m in _SLOT_RE.finditer(tmpl):
            kind = m.group("kind")
            sub = m.group("sub")
            if kind == "decoy":
                if not sub:
                    errors.append(f"template has decoy slot without name: {tmpl!r}")
                elif sub not in decoy_pools:
                    errors.append(f"template references unknown decoy slot {sub!r}: {tmpl!r}")
            else:
                if kind.upper() not in valid_entity_kinds:
                    errors.append(f"template references unknown entity kind {kind!r}: {tmpl!r}")

    if errors:
        raise ValueError("seed validation failed:\n  - " + "\n  - ".join(errors))

    return {
        "entity_pools": {et: len(v) for et, v in entity_pools.items()},
        "decoy_pools": {s: len(v) for s, v in decoy_pools.items()},
        "templates": len(templates),
    }


def build_pools() -> Pools:
    """Return the seed as an in-memory `Pools` object (no DB round-trip).

    Validates first, so a returned `Pools` is always slot-fill-ready.
    """
    validate()
    pools = Pools()
    entity_pools = _entity_pools()
    for et in ENTITY_TYPES:
        for v in entity_pools[et]:
            pools.add_entity(et, v)
    for slot, vals in _decoy_pools().items():
        for v in vals:
            pools.add_decoy(slot, v)
    for tmpl in _templates():
        pools.add_template(tmpl)
    return pools


def _sql_str(value: str) -> str:
    """Single-quote a SQL string literal, escaping embedded quotes."""
    return "'" + value.replace("'", "''") + "'"


def _chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _emit_inserts(
    table: str,
    columns: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    conflict_cols: tuple[str, ...],
) -> list[str]:
    """Emit one or more multi-row Postgres INSERT ... ON CONFLICT statements."""
    if not rows:
        return []
    out: list[str] = []
    col_sql = ", ".join(columns)
    prefix = f"INSERT INTO {table} ({col_sql}) VALUES"
    suffix = f"\nON CONFLICT ({', '.join(conflict_cols)}) DO NOTHING;"
    for chunk in _chunked(rows, _ROWS_PER_STATEMENT):
        value_rows = ",\n".join(
            "    (" + ", ".join(_sql_str(c) for c in row) + ")" for row in chunk
        )
        out.append(f"{prefix}\n{value_rows}{suffix}")
    return out


def build_sql() -> str:
    """Render the Postgres seed SQL text."""
    entity_pools = _entity_pools()
    decoy_pools = _decoy_pools()
    templates = _templates()

    header = (
        "-- GENERATED FILE — do not edit by hand.\n"
        "-- Source of truth: scripts/seedgen/*.py ; regenerate with\n"
        "--     python -m scripts.build_seed\n"
        "--\n"
        "-- Complete production seed for the multi-entity NER slot-fill pools,\n"
        "-- covering the scenarios enumerated in docs/data_specification.md\n"
        "-- (PERSON/ORG/ADDRESS/COMMODITY pools, decoy slots, slot-fill templates).\n"
    )

    parts: list[str] = [header]

    entity_rows: list[tuple[str, str]] = []
    for et in ENTITY_TYPES:
        for v in entity_pools[et]:
            entity_rows.append((et, v))
    parts.append("\n-- ===== entity_pools =====")
    parts.extend(
        _emit_inserts(
            "entity_pools", ("entity_type", "value"), entity_rows,
            conflict_cols=("entity_type", "value"),
        )
    )

    decoy_rows: list[tuple[str, str]] = []
    for slot in ALLOWED_DECOY_SLOTS:
        for v in decoy_pools[slot]:
            decoy_rows.append((slot, v))
    parts.append("\n-- ===== decoy_pools =====")
    parts.extend(
        _emit_inserts(
            "decoy_pools", ("slot_name", "value"), decoy_rows,
            conflict_cols=("slot_name", "value"),
        )
    )

    template_rows = [(t,) for t in templates]
    parts.append("\n-- ===== templates =====")
    parts.extend(
        _emit_inserts(
            "templates", ("template",), template_rows,
            conflict_cols=("template",),
        )
    )

    return "\n".join(parts) + "\n"


def build(write: bool = True) -> dict[str, object]:
    summary = validate()
    if write:
        POSTGRES_OUT.write_text(build_sql(), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Postgres seed SQL from seedgen modules")
    ap.add_argument("--check", action="store_true", help="validate only; do not write files")
    args = ap.parse_args()

    summary = build(write=not args.check)
    ep = summary["entity_pools"]
    dp = summary["decoy_pools"]
    total_entities = sum(ep.values())
    total_decoys = sum(dp.values())
    print("Seed validation passed.")
    print(f"  entity_pools : {total_entities} values  {ep}")
    print(f"  decoy_pools  : {total_decoys} values across {len(dp)} slots")
    print(f"  templates    : {summary['templates']}")
    if args.check:
        print("(--check: no files written)")
    else:
        print(f"  wrote {POSTGRES_OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
