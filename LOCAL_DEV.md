# Running the NER pipeline locally

A practical, copy-paste guide for getting the multi-entity NER system running on
your own machine — from seeding the pools through generating data, training,
exporting, and serving. For the *why* behind each stage see `README.md` and
`docs/data_specification.md`; this file is the *how*.

---

## 1. Prerequisites

- **Python 3.10+** (3.11 recommended).
- **git**.
- **Docker + Docker Compose** — optional, only needed for the Postgres path
  (§6). The default SQLite path needs no Docker.

Check:

```bash
python --version        # >= 3.10
docker --version        # optional
```

---

## 2. Set up a virtualenv and install

```bash
git clone <repo-url> extraction-ml-1
cd extraction-ml-1

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Dependency groups (defined in pyproject.toml) — install what you need:
pip install -e .[dev]              # pytest + ruff (tests/lint)
pip install -e .[inference]        # numpy + onnxruntime + tokenizers (serving)
pip install -e .[train]            # torch + transformers + datasets (training/export)
pip install -e .[data]             # psycopg (Postgres) + anthropic (optional LLM hook)
```

What each path actually needs:

| You want to…                                  | Install            |
|-----------------------------------------------|--------------------|
| Generate synthetic data / run the seed (§3–5) | nothing extra — the data core (`ner.data.slot_fill`, `ner.data.pools`, `ner.schema`) is **pure stdlib** |
| Run the full test suite                       | `.[dev]` **and** `.[inference]` (a few tests `import numpy`) |
| Train a model                                 | `.[train]`         |
| Export to ONNX                                | `.[train]`         |
| Serve / run inference                         | `.[inference]`     |
| Use the Postgres warehouse                    | `.[data]`          |

> The serving runtime is intentionally torch-free. You only need `.[train]` to
> produce a checkpoint and ONNX file; production serving runs on `.[inference]`.

---

## 3. Seed the pools (SQLite — the default, no Docker)

The data layer is driven by a SQLite DB matching `sql/schema.sql`, populated
from a seed `.sql`. Two seeds ship in the repo:

- `sql/example_seed.sql` — tiny smoke seed (used by `tests/test_pools_sqlite.py`).
- `sql/seed.sql` — **full production seed** covering every scenario in
  `docs/data_specification.md` (≈2,400 entity values, 16 decoy slots, ~200
  templates).

Initialize a pool DB from the full seed:

```bash
python -m scripts.generate_data --init-db data/pools.sqlite --seed-sql sql/seed.sql
```

### Regenerating the full seed

`sql/seed.sql` and `sql/postgres/seed.sql` are **generated** from the Python
source-of-truth modules in `scripts/seedgen/` (one per scenario family:
`persons.py`, `orgs.py`, `addresses.py`, `commodities.py`, `decoys.py`,
`templates.py`). Edit those, then rebuild both SQL dialects:

```bash
python -m scripts.build_seed            # writes sql/seed.sql + sql/postgres/seed.sql
python -m scripts.build_seed --check    # validate only (no files written)
```

The builder dedups, validates that every `{decoy:slot}` a template references
has a backing pool, and escapes apostrophes / newlines / non-ASCII correctly.

---

## 4. Generate synthetic training data

```bash
python -m scripts.generate_data \
    --sqlite data/pools.sqlite \
    --out data/train.jsonl \
    --n 50000 --seed 42 --noise-prob 0.6
```

Every emitted record passes `Record.validate()` — character offsets are computed
in Python, so `text[start:end] == entity.text` is a hard invariant. A
`preprocess.json` is written next to `train.jsonl` so training and serving share
the identical preprocessing config.

Quick peek at a few generated rows (no extra deps required):

```bash
python - <<'PY'
from ner.data.pools import load_from_sqlite
from ner.data.slot_fill import GenConfig, generate_records
pools = load_from_sqlite("data/pools.sqlite")
for r in generate_records(pools, GenConfig(seed=0, n_records=5)):
    spans = ", ".join(f"{e.type}/{e.polarity}:{e.text!r}" for e in r.entities) or "(none)"
    print(r.text[:90], "=>", spans)
PY
```

---

## 5. Train → export → tune → serve

These need the heavier extras (`.[train]` then `.[inference]`).

```bash
# Train (early-stops on the real gold set; synthetic is never used for eval)
python -m scripts.train --train-jsonl data/train.jsonl --output-dir artifacts/ckpt

# Export to FP32 ONNX (graph-optimized; INT8 is prohibited by the TDD)
python -m scripts.export_onnx --model-dir artifacts/ckpt --output artifacts/serve/model.onnx

# Tune per-label confidence thresholds on the gold set
python -m scripts.tune_threshold --artifact-dir artifacts/serve --gold-jsonl data/gold_dev.jsonl

# Serve a single string from the CLI
python -m scripts.infer --artifact-dir artifacts/serve \
    --text "manifest contains no robusta coffee, only cane sugar"
```

Or from Python:

```python
from ner.infer.runtime import from_artifact_dir
runtime = from_artifact_dir("artifacts/serve")
print(runtime.predict("Maria Gonzalez from Acme Trading Co. confirmed refined copper cathode"))
```

---

## 6. Postgres warehouse (optional, needs Docker + `.[data]`)

For the production gold store / annotation / versioning trail
(`docs/data_specification.md` §20):

```bash
docker compose up -d                                   # Postgres on localhost:6655

python -m scripts.init_postgres                        # schema + example seed
python -m scripts.init_postgres --seed sql/postgres/seed.sql   # load the full seed
python -m scripts.init_postgres --no-seed              # schema only
python -m scripts.init_postgres --verify               # row counts + v_split_coverage
```

Connection defaults (override with `--dsn` or `DATABASE_URL`):

| Setting   | Value                                                  |
|-----------|--------------------------------------------------------|
| DSN       | `postgresql://ner:ner@localhost:6655/multi_entity_ner` |
| DB / user | `multi_entity_ner` / `ner` (password `ner`)            |

The Docker volume `ner_pg_data` persists across restarts; `docker compose down`
does **not** delete data. To wipe it: `docker compose down -v`.

---

## 7. Run the tests

```bash
pip install -e .[dev] .[inference]   # numpy is needed by a few tests
pytest -q                            # full deterministic suite (sub-second)

# Subsets:
pytest tests/test_seed_full.py -v    # the full production seed contract
pytest tests/test_slot_fill.py -v    # offset invariants under random seeds
pytest -k threshold                  # by name substring
```

If you only installed `.[dev]` (no numpy), exclude the inference-dependent tests:

```bash
pytest -q --ignore=tests/test_runtime_preprocess.py --ignore=tests/test_threshold_sweep.py
```

Lint:

```bash
ruff check .
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named pytest` | `pip install -e .[dev]` |
| `No module named numpy` during tests | `pip install -e .[inference]` |
| `psycopg is not installed` | `pip install -e .[data]` |
| `scripts.build_seed` says "seed validation failed" | a template references a `{decoy:slot}` with no pool — add the slot in `scripts/seedgen/decoys.py` (must be in `ALLOWED_DECOY_SLOTS`) |
| Postgres connection refused | `docker compose up -d` and wait for the healthcheck (`docker compose ps`) |
| Want a fresh pool DB | delete `data/pools.sqlite` and re-run the `--init-db` step (it is git-ignored) |
