"""Generate synthetic training data from a SQLite seed.

Usage:
    python -m scripts.generate_data \\
        --sqlite data/pools.sqlite \\
        --out data/train.jsonl \\
        --n 50000 --seed 42

To initialize the SQLite from sql/schema.sql + a seed .sql file:
    python -m scripts.generate_data --init-db data/pools.sqlite --seed-sql sql/example_seed.sql
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ner.data.assembler import AssemblerConfig, assemble_from_sqlite
from ner.data.pools import build_sqlite_from_sql_files


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic NER data generator")
    p.add_argument("--sqlite", help="Path to SQLite pool DB")
    p.add_argument("--out", help="Output JSONL path")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-prob", type=float, default=0.6)
    p.add_argument("--init-db", help="Initialize a fresh DB at this path and exit")
    p.add_argument("--seed-sql", action="append", default=[],
                   help="Seed .sql files to apply when initializing (repeatable)")
    args = p.parse_args()

    if args.init_db:
        seeds = args.seed_sql or [str(Path(__file__).resolve().parents[1] / "sql" / "example_seed.sql")]
        out = build_sqlite_from_sql_files(args.init_db, seeds)
        print(f"Initialized SQLite DB at {out}")
        return

    if not args.sqlite or not args.out:
        p.error("--sqlite and --out are required unless --init-db is given")

    cfg = AssemblerConfig(n_records=args.n, seed=args.seed)
    cfg.apply_noise_prob = args.noise_prob
    out = assemble_from_sqlite(args.sqlite, args.out, cfg)
    print(f"Wrote {args.n} records to {out}")


if __name__ == "__main__":
    main()
