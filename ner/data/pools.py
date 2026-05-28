"""Entity / decoy / template pools loaded from SQLite (or in-memory seeds).

The pipeline is decoupled from the source of pools: load_from_sqlite() reads
seed SQL conforming to sql/schema.sql, and `Pools` is also constructable
directly from Python dicts for tests and offline development.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ner.constants import ENTITY_TYPES


@dataclass
class WeightedPool:
    values: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def add(self, value: str, weight: float = 1.0) -> None:
        self.values.append(value)
        self.weights.append(weight)

    def __len__(self) -> int:
        return len(self.values)

    def is_empty(self) -> bool:
        return not self.values


@dataclass
class Pools:
    entity_pools: dict[str, WeightedPool] = field(default_factory=dict)
    decoy_pools: dict[str, WeightedPool] = field(default_factory=dict)
    templates: WeightedPool = field(default_factory=WeightedPool)

    def __post_init__(self) -> None:
        for et in ENTITY_TYPES:
            self.entity_pools.setdefault(et, WeightedPool())

    def add_entity(self, entity_type: str, value: str, weight: float = 1.0) -> None:
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"Unknown entity type: {entity_type!r}")
        self.entity_pools[entity_type].add(value, weight)

    def add_decoy(self, slot_name: str, value: str, weight: float = 1.0) -> None:
        self.decoy_pools.setdefault(slot_name, WeightedPool()).add(value, weight)

    def add_template(self, template: str, weight: float = 1.0) -> None:
        self.templates.add(template, weight)

    def validate(self) -> None:
        for et in ENTITY_TYPES:
            if self.entity_pools[et].is_empty():
                raise ValueError(f"Entity pool {et!r} is empty")
        if self.templates.is_empty():
            raise ValueError("No templates loaded")


SCHEMA_FILE: Path = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    schema_sql = SCHEMA_FILE.read_text()
    conn.executescript(schema_sql)


def load_from_sqlite(path: str | Path) -> Pools:
    """Load pools from a SQLite database matching sql/schema.sql."""
    pools = Pools()
    with sqlite3.connect(str(path)) as conn:
        _ensure_schema(conn)
        cur = conn.execute(
            "SELECT entity_type, value, weight FROM entity_pools"
        )
        for et, value, weight in cur:
            pools.add_entity(et, value, weight)

        cur = conn.execute(
            "SELECT slot_name, value, weight FROM decoy_pools"
        )
        for slot, value, weight in cur:
            pools.add_decoy(slot, value, weight)

        cur = conn.execute("SELECT template, weight FROM templates")
        for tmpl, weight in cur:
            pools.add_template(tmpl, weight)
    return pools


def build_sqlite_from_sql_files(
    target_path: str | Path,
    seed_sql_files: Iterable[str | Path],
) -> Path:
    """Initialize a SQLite DB with our schema and apply seed SQL files in order."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    with sqlite3.connect(str(target)) as conn:
        _ensure_schema(conn)
        for sql_path in seed_sql_files:
            conn.executescript(Path(sql_path).read_text())
    return target


DEFAULT_POSTGRES_DSN: str = "postgresql://ner:ner@localhost:6655/multi_entity_ner"


def load_from_postgres(dsn: str | None = None) -> Pools:
    """Load pools from a Postgres DB matching sql/postgres/schema.sql.

    The table names and columns match the SQLite contract, so this is a
    drop-in alternative for `load_from_sqlite` once the production DB is
    populated (see scripts.init_postgres).

    psycopg is imported lazily so this module remains importable without
    the [data] extra.
    """
    try:
        import psycopg
    except ImportError as exc:
        raise ImportError(
            "psycopg is required for load_from_postgres; install with "
            "`pip install -e .[data]`"
        ) from exc

    pools = Pools()
    with psycopg.connect(dsn or DEFAULT_POSTGRES_DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT entity_type, value, weight FROM entity_pools")
        for et, value, weight in cur.fetchall():
            pools.add_entity(et, value, float(weight))

        cur.execute("SELECT slot_name, value, weight FROM decoy_pools")
        for slot, value, weight in cur.fetchall():
            pools.add_decoy(slot, value, float(weight))

        cur.execute("SELECT template, weight FROM templates")
        for tmpl, weight in cur.fetchall():
            pools.add_template(tmpl, float(weight))
    return pools


def dump_to_json(pools: Pools, path: str | Path) -> None:
    """Serialize pools to JSON for debugging."""
    obj = {
        "entity_pools": {
            et: list(zip(p.values, p.weights)) for et, p in pools.entity_pools.items()
        },
        "decoy_pools": {
            s: list(zip(p.values, p.weights)) for s, p in pools.decoy_pools.items()
        },
        "templates": list(zip(pools.templates.values, pools.templates.weights)),
    }
    Path(path).write_text(json.dumps(obj, indent=2))
