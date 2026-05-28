"""Top-level data pipeline: SQL -> Pools -> slot-fill -> noise -> JSONL."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ner.data.noise import NoiseConfig, apply_noise
from ner.data.pools import Pools, load_from_sqlite
from ner.data.slot_fill import GenConfig, generate_records
from ner.schema import Record


@dataclass
class AssemblerConfig:
    n_records: int = 10_000
    seed: int = 0
    gen: GenConfig = field(default_factory=GenConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    apply_noise_prob: float = 0.6


def assemble(
    pools: Pools,
    config: AssemblerConfig,
) -> list[Record]:
    config.gen.seed = config.seed
    config.gen.n_records = config.n_records
    clean = generate_records(pools, config.gen)
    rng = random.Random(config.seed + 1)
    out: list[Record] = []
    for rec in clean:
        if rng.random() < config.apply_noise_prob:
            out.append(apply_noise(rec, config.noise, rng))
        else:
            out.append(rec)
    return out


def write_jsonl(records: list[Record], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[Record]:
    out: list[Record] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Record.from_dict(json.loads(line)))
    return out


def assemble_from_sqlite(
    sqlite_path: str | Path,
    out_path: str | Path,
    config: AssemblerConfig,
) -> Path:
    pools = load_from_sqlite(sqlite_path)
    records = assemble(pools, config)
    return write_jsonl(records, out_path)
