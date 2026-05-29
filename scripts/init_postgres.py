"""Initialize the production Postgres database for the NER system.

Applies sql/postgres/schema.sql and loads sql/postgres/example_seed.sql
(or any other seed file the caller specifies). Idempotent — every CREATE
and INSERT is conflict-safe, so running this multiple times converges to
the same state.

Default DSN matches docker-compose.yml:
    postgresql://ner:ner@localhost:6655/multi_entity_ner

Override with --dsn or DATABASE_URL.

Quickstart:

    docker compose up -d                                          # start the DB
    python -m scripts.init_postgres                               # schema + example seed
    python -m scripts.init_postgres --no-seed                     # schema only
    python -m scripts.init_postgres --seed sql/postgres/prod_seed_2024q4.sql

Verification:

    python -m scripts.init_postgres --verify
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ner.data.pools import DEFAULT_DSN

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = REPO_ROOT / "sql" / "postgres" / "schema.sql"
DEFAULT_SEED = REPO_ROOT / "sql" / "postgres" / "example_seed.sql"


def _connect(dsn: str):
    """Lazy import of psycopg so the module is importable without the dep."""
    try:
        import psycopg  # psycopg3
    except ImportError as exc:
        raise SystemExit(
            "psycopg is not installed. Run: pip install -e .[data]\n"
            "(or `pip install 'psycopg[binary]'`)"
        ) from exc
    return psycopg.connect(dsn, autocommit=False)


def apply_schema(dsn: str, schema_path: Path) -> None:
    sql = schema_path.read_text(encoding="utf-8")
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
    print(f"[init_postgres] applied schema {schema_path.name}")


def load_seed(dsn: str, seed_path: Path) -> dict[str, int]:
    """Apply a seed SQL file. Returns row counts of the three pool tables
    after load so the operator sees what landed."""
    sql = seed_path.read_text(encoding="utf-8")
    with _connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql)
        counts: dict[str, int] = {}
        for table in ("entity_pools", "decoy_pools", "templates"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
        conn.commit()
    print(f"[init_postgres] loaded seed {seed_path.name}")
    for table, n in counts.items():
        print(f"  {table}: {n} rows")
    return counts


def verify(dsn: str) -> None:
    """Run sanity SELECTs that prove the schema is consistent."""
    queries = [
        ("entity_pools by type",
         "SELECT entity_type, COUNT(*) FROM entity_pools "
         "GROUP BY entity_type ORDER BY entity_type"),
        ("decoy_pools by slot",
         "SELECT slot_name, COUNT(*) FROM decoy_pools "
         "GROUP BY slot_name ORDER BY slot_name"),
        ("templates count",
         "SELECT COUNT(*) FROM templates"),
        ("split coverage view",
         "SELECT * FROM v_split_coverage"),
    ]
    with _connect(dsn) as conn, conn.cursor() as cur:
        for label, q in queries:
            print(f"\n-- {label}")
            cur.execute(q)
            for row in cur.fetchall():
                print("  ", row)


def main() -> None:
    p = argparse.ArgumentParser(description="Initialize the NER Postgres DB")
    p.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", DEFAULT_DSN),
        help=f"Postgres DSN (default: {DEFAULT_DSN}; env DATABASE_URL also works)",
    )
    p.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="Schema SQL to apply",
    )
    p.add_argument(
        "--seed",
        type=Path,
        default=DEFAULT_SEED,
        help="Seed SQL to load (use --no-seed to skip)",
    )
    p.add_argument(
        "--no-seed",
        action="store_true",
        help="Apply schema only; do not load any seed",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="After initialization, print row counts and split coverage",
    )
    args = p.parse_args()

    if not args.schema.exists():
        print(f"ERROR: schema file not found: {args.schema}", file=sys.stderr)
        sys.exit(2)

    apply_schema(args.dsn, args.schema)

    if not args.no_seed:
        if not args.seed.exists():
            print(f"ERROR: seed file not found: {args.seed}", file=sys.stderr)
            sys.exit(2)
        load_seed(args.dsn, args.seed)

    if args.verify:
        verify(args.dsn)


if __name__ == "__main__":
    main()
