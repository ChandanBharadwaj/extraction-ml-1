"""Generate synthetic training data from the Postgres pool warehouse.

Pools (entity / decoy / template) live in Postgres — see scripts.init_postgres
to apply the schema and load a seed. This reads those pools and writes
synthetic training JSONL (+ preprocess.json) to disk.

Usage:
    python -m scripts.generate_data \\
        --out data/train.jsonl \\
        --n 50000 --seed 42

    # Override the connection (else $DATABASE_URL, else the default local DSN):
    python -m scripts.generate_data --out data/train.jsonl --postgres-dsn \\
        postgresql://ner:ner@localhost:6655/multi_entity_ner
"""
from __future__ import annotations

import argparse

from ner.data.assembler import AssemblerConfig, assemble_from_postgres


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic NER data generator (Postgres pools)")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-prob", type=float, default=0.6)
    p.add_argument(
        "--postgres-dsn",
        default=None,
        help="Postgres DSN for the pool warehouse. Defaults to $DATABASE_URL, "
             "then the local DSN postgresql://ner:ner@localhost:6655/multi_entity_ner",
    )
    args = p.parse_args()

    cfg = AssemblerConfig(n_records=args.n, seed=args.seed)
    cfg.apply_noise_prob = args.noise_prob
    out = assemble_from_postgres(args.postgres_dsn, args.out, cfg)
    print(f"Wrote {args.n} records to {out}")


if __name__ == "__main__":
    main()
