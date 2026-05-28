"""Validate the complete production seed (scripts/seedgen + scripts.build_seed).

Guards the seed contract end-to-end:
  - the seedgen modules satisfy the pool/template validation,
  - the emitted SQLite SQL loads into a DB matching sql/schema.sql,
  - the slot-fill generator produces valid records (offset invariant holds),
  - both POS and NEG commodity polarities are exercised,
  - negation templates populate meta["preserve_spans"],
  - the committed sql/seed.sql / sql/postgres/seed.sql are in sync with the
    seedgen source of truth.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ner.data.pools import _ensure_schema, load_from_sqlite
from ner.data.slot_fill import GenConfig, generate_records
from scripts import build_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_sqlite(tmp_path) -> Path:
    db = tmp_path / "seed.sqlite"
    sql = build_seed._build_sql("sqlite")
    with sqlite3.connect(str(db)) as conn:
        _ensure_schema(conn)
        conn.executescript(sql)
    return db


def test_seed_validation_passes():
    summary = build_seed.validate()
    ep = summary["entity_pools"]
    # Each entity type should be well-populated for production coverage.
    assert ep["PERSON"] >= 300
    assert ep["ORG"] >= 300
    assert ep["ADDRESS"] >= 150
    assert ep["COMMODITY"] >= 250
    assert summary["templates"] >= 100


def test_emitted_sql_loads_and_generates(tmp_path):
    db = _build_sqlite(tmp_path)
    pools = load_from_sqlite(db)
    pools.validate()
    assert len(pools.entity_pools["COMMODITY"]) >= 250
    assert len(pools.templates) >= 100

    records = generate_records(pools, GenConfig(seed=0, n_records=500))
    assert len(records) == 500
    for rec in records:
        rec.validate()  # text[start:end] == entity.text for every entity


def test_both_polarities_and_preserve_spans(tmp_path):
    db = _build_sqlite(tmp_path)
    pools = load_from_sqlite(db)
    records = generate_records(pools, GenConfig(seed=7, n_records=1000))

    polarities = {e.polarity for rec in records for e in rec.entities}
    assert "POS" in polarities
    assert "NEG" in polarities

    # At least one negation/contrast cue range was protected from noise.
    assert any(rec.meta.get("preserve_spans") for rec in records)

    # Zero-entity (true-negative) templates must surface too.
    assert any(len(rec.entities) == 0 for rec in records)


def test_postgres_sql_builds():
    sql = build_seed._build_sql("postgres")
    assert "ON CONFLICT" in sql
    assert "INSERT INTO entity_pools" in sql


def test_committed_seed_files_in_sync():
    """The checked-in SQL must match a fresh render of the seedgen modules."""
    for path, dialect in (
        (REPO_ROOT / "sql" / "seed.sql", "sqlite"),
        (REPO_ROOT / "sql" / "postgres" / "seed.sql", "postgres"),
    ):
        assert path.exists(), f"missing generated seed file: {path}"
        expected = build_seed._build_sql(dialect)
        actual = path.read_text(encoding="utf-8")
        assert actual == expected, (
            f"{path} is stale; regenerate with `python -m scripts.build_seed`"
        )
