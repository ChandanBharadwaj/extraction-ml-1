"""Validate the complete production seed (scripts/seedgen + scripts.build_seed).

Guards the seed contract without needing a database:
  - the seedgen modules satisfy the pool/template validation,
  - `build_pools()` yields a slot-fill-ready `Pools` straight from the Python
    source-of-truth lists,
  - the slot-fill generator produces valid records (offset invariant holds),
  - both POS and NEG commodity polarities are exercised,
  - negation templates populate meta["preserve_spans"] and zero-entity records
    appear,
  - the committed sql/postgres/seed.sql is in sync with the seedgen source.
"""
from __future__ import annotations

from pathlib import Path

from ner.data.slot_fill import GenConfig, generate_records
from scripts import build_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_seed_validation_passes():
    summary = build_seed.validate()
    ep = summary["entity_pools"]
    assert ep["PERSON"] >= 300
    assert ep["ORG"] >= 300
    assert ep["ADDRESS"] >= 150
    assert ep["COMMODITY"] >= 250
    assert summary["templates"] >= 100


def test_build_pools_generates_valid_records():
    pools = build_seed.build_pools()
    pools.validate()
    assert len(pools.entity_pools["COMMODITY"]) >= 250
    assert len(pools.templates) >= 100

    records = generate_records(pools, GenConfig(seed=0, n_records=500))
    assert len(records) == 500
    for rec in records:
        rec.validate()  # text[start:end] == entity.text for every entity


def test_both_polarities_and_preserve_spans():
    pools = build_seed.build_pools()
    records = generate_records(pools, GenConfig(seed=7, n_records=1000))

    polarities = {e.polarity for rec in records for e in rec.entities}
    assert "POS" in polarities
    assert "NEG" in polarities

    # At least one negation/contrast cue range was protected from noise.
    assert any(rec.meta.get("preserve_spans") for rec in records)

    # Zero-entity (true-negative) templates must surface too.
    assert any(len(rec.entities) == 0 for rec in records)


def test_committed_seed_file_in_sync():
    """The checked-in Postgres SQL must match a fresh render of the modules."""
    path = REPO_ROOT / "sql" / "postgres" / "seed.sql"
    assert path.exists(), f"missing generated seed file: {path}"
    expected = build_seed.build_sql()
    actual = path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{path} is stale; regenerate with `python -m scripts.build_seed`"
    )
