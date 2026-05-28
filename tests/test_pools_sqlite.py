"""Verify the SQLite seed contract: schema.sql + example_seed.sql produce
working `Pools` that the slot-fill generator accepts.
"""
from __future__ import annotations

from pathlib import Path

from ner.data.pools import build_sqlite_from_sql_files, load_from_sqlite
from ner.data.slot_fill import GenConfig, generate_records

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = REPO_ROOT / "sql" / "schema.sql"
SEED_SQL = REPO_ROOT / "sql" / "example_seed.sql"


def test_build_sqlite_from_sql_files(tmp_path):
    db = tmp_path / "pools.sqlite"
    build_sqlite_from_sql_files(db, [SEED_SQL])
    pools = load_from_sqlite(db)
    pools.validate()
    assert len(pools.entity_pools["COMMODITY"]) >= 5
    assert len(pools.templates) >= 5


def test_pipeline_end_to_end_from_sqlite(tmp_path):
    db = tmp_path / "pools.sqlite"
    build_sqlite_from_sql_files(db, [SEED_SQL])
    pools = load_from_sqlite(db)
    records = generate_records(pools, GenConfig(seed=0, n_records=50))
    assert len(records) == 50
    for rec in records:
        rec.validate()
