# Running the NER pipeline locally

A practical, copy-paste guide for getting the multi-entity NER system running on
your own machine — from seeding the pools through generating data, training,
exporting, and serving. For the *why* behind each stage see `README.md` and
`docs/data_specification.md`; this file is the *how*.

---

## 1. Prerequisites

- **Python 3.10+** (3.11 recommended).
- **git**.
- **Docker + Docker Compose** — required. Pools *and* gold live in Postgres;
  there is no SQLite fallback.

Check:

```bash
python --version        # >= 3.10
docker --version
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
| Seed Postgres / generate synthetic data (§3–4)| `.[data]` (psycopg) + a running Postgres |
| Inspect pools / generate records offline      | nothing extra — `build_seed.build_pools()` is **pure stdlib** (no DB) |
| Run the full test suite                       | `.[dev]` **and** `.[inference]` (a few tests `import numpy`) |
| Train a model                                 | `.[train]`         |
| Export to ONNX                                | `.[train]`         |
| Serve / run inference                         | `.[inference]`     |

> The serving runtime is intentionally torch-free. You only need `.[train]` to
> produce a checkpoint and ONNX file; production serving runs on `.[inference]`.

---

## 3. Seed the pools (Postgres via Docker)

Pools live in Postgres — the single source of truth (no SQLite). Start the
container and load the seed:

```bash
docker compose up -d                                            # Postgres @ localhost:6655
pip install -e .[data]                                          # psycopg driver
python -m scripts.init_postgres --seed sql/postgres/seed.sql    # schema + full production seed
python -m scripts.init_postgres --verify                        # row counts + coverage view
```

Two seeds ship in the repo:

- `sql/postgres/example_seed.sql` — tiny smoke seed (the default if you omit
  `--seed`).
- `sql/postgres/seed.sql` — **full production seed** covering every scenario in
  `docs/data_specification.md` (≈2,400 entity values, 16 decoy slots, ~200
  templates).

The DSN defaults to `postgresql://ner:ner@localhost:6655/multi_entity_ner`;
override with `--dsn` or the `DATABASE_URL` env var.

### Regenerating the seed

`sql/postgres/seed.sql` is **generated** from the Python source-of-truth modules
in `scripts/seedgen/` (one per scenario family: `persons.py`, `orgs.py`,
`addresses.py`, `commodities.py`, `decoys.py`, `templates.py`). Edit those, then:

```bash
python -m scripts.build_seed            # writes sql/postgres/seed.sql
python -m scripts.build_seed --check    # validate only (no files written)
```

The builder dedups, validates that every `{decoy:slot}` a template references
has a backing pool, and escapes apostrophes / newlines / non-ASCII correctly.
Reload it with `init_postgres --seed sql/postgres/seed.sql` (idempotent —
`ON CONFLICT DO NOTHING`).

---

## 4. Generate synthetic training data

Reads pools from Postgres (connection: `--postgres-dsn`, else `$DATABASE_URL`,
else the default local DSN):

```bash
python -m scripts.generate_data \
    --out data/train.jsonl \
    --n 50000 --seed 42 --noise-prob 0.6
```

Every emitted record passes `Record.validate()` — character offsets are computed
in Python, so `text[start:end] == entity.text` is a hard invariant. A
`preprocess.json` is written next to `train.jsonl` so training and serving share
the identical preprocessing config.

Quick peek at a few generated rows **without a database** (builds `Pools`
straight from the seedgen modules — no extra deps):

```bash
python - <<'PY'
from scripts.build_seed import build_pools
from ner.data.slot_fill import GenConfig, generate_records
pools = build_pools()
for r in generate_records(pools, GenConfig(seed=0, n_records=5)):
    spans = ", ".join(f"{e.type}/{e.polarity}:{e.text!r}" for e in r.entities) or "(none)"
    print(r.text[:90], "=>", spans)
PY
```

---

## 5. Train → export → tune → serve

> **No local GPU?** Use the free Colab notebook at `notebooks/colab_train.ipynb`
> (T4 GPU; a deberta-v3-xsmall run is ~10–15 min). Open it via
> `https://colab.research.google.com/github/ChandanBharadwaj/extraction-ml-1/blob/<branch>/notebooks/colab_train.ipynb`
> or upload it to [colab.research.google.com](https://colab.research.google.com).
> Set `Runtime → Change runtime type → T4 GPU`, then run the cells top to bottom;
> the last cell saves artifacts to Google Drive (Colab disks are ephemeral).

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

### Choosing a smaller / faster model

The default `microsoft/deberta-v3-base` is ~184M params → **~700 MB** on disk
(most of it the 128K-token embedding) and needs a real GPU. Swap the backbone
with `--base-model` — no other change; the BIO head and `LABEL_LIST` contract
are backbone-agnostic. FP32 size ≈ params × 4 bytes:

| `--base-model` | Params | ~Size | Notes |
|---|---|---|---|
| `microsoft/deberta-v3-base` (default) | 184M | ~700 MB | best quality; GPU-only |
| `microsoft/deberta-v3-xsmall` | ~71M | **~280 MB** | best size/quality tradeoff; fits 6 GB easily |
| `distilbert-base-uncased` | ~66M | ~265 MB | fast English NER workhorse |
| `microsoft/MiniLM-L12-H384-uncased` | ~33M | ~133 MB | smallest practical English option |
| `distilbert-base-multilingual-cased` | ~135M | ~540 MB | if non-Latin names matter |

DeBERTa-v3's small variants don't shrink much because the embedding is shared;
to go below ~280 MB use a small-vocab (BERT/DistilBERT/MiniLM) model.

### Training on a small GPU (e.g. GTX 1660 Ti, 6 GB)

The `Trainer` auto-uses CUDA when available — confirm with
`python -c "import torch; print(torch.cuda.is_available())"` (must be `True`;
install the CUDA build of torch). Fit the job into limited VRAM with these
flags (added to `scripts.train`):

| Flag | Purpose |
|---|---|
| `--batch-size` | per-device batch; **lower** to fit VRAM |
| `--grad-accum` | accumulate steps so *effective* batch = batch-size × grad-accum |
| `--fp16` | mixed precision; ~halves memory (needed to fit base models on ≤6 GB) |
| `--max-seq-len` | shorter sequences (128/192) save memory + time; data is ≤500 chars |
| `--max-train-samples` | cap records for a quick smoke run |

**Recommended for a 6 GB GPU** — deberta-v3-xsmall fits comfortably:

```bash
python -m scripts.generate_data --out data/train.jsonl --n 25000 --seed 42
python -m scripts.train \
    --train-jsonl data/train.jsonl --output-dir artifacts/ckpt \
    --base-model microsoft/deberta-v3-xsmall \
    --batch-size 16 --max-seq-len 192 --epochs 3
```

If you insist on `deberta-v3-base` on 6 GB, you must shrink the footprint:
`--batch-size 8 --grad-accum 4 --fp16 --max-seq-len 192` (drop to
`--batch-size 4` if you still hit CUDA OOM). On GTX 16-series, if loss becomes
`nan` under `--fp16`, drop `--fp16` and use `--batch-size 4` (no Tensor Cores,
so fp16's benefit here is memory, not speed).

### Training on CPU (no GPU)

Feasible with `deberta-v3-xsmall` if you keep the job small. CPU specifics:

- **Do not pass `--fp16`** — CPUs don't accelerate fp16 (often slower). Keep fp32.
- The biggest lever is `--max-seq-len` (cost scales with sequence length, and
  the data is short) followed by fewer records / epochs.
- Set `OMP_NUM_THREADS` to your physical core count to use all cores.
- Prefer the machine with more RAM (xsmall itself needs < 4 GB).

```bash
python -m scripts.generate_data --out data/train_cpu.jsonl --n 8000 --seed 42

OMP_NUM_THREADS=$(nproc) python -m scripts.train \
    --train-jsonl data/train_cpu.jsonl --output-dir artifacts/ckpt \
    --base-model microsoft/deberta-v3-xsmall \
    --batch-size 16 --max-seq-len 128 --epochs 2
```

Rough, hardware-dependent CPU timings with xsmall: ~8k records / 2 epochs /
seq-128 ≈ 30–70 min (good first run); ~20k / 3 epochs ≈ 3–6 h. Full 50k on CPU
is possible but slow (~10–20 h) — at that point a free cloud GPU (Colab/Kaggle)
is the better use of time.

---

## 6. Postgres warehouse — gold store & versioning (`.[data]`)

Beyond the pools (§3), the same DB holds the production gold store / annotation
/ versioning trail (`docs/data_specification.md` §20). Useful init variants:

```bash
docker compose up -d                                   # Postgres on localhost:6655

python -m scripts.init_postgres                        # schema + tiny example seed
python -m scripts.init_postgres --seed sql/postgres/seed.sql   # schema + full seed
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
| Want fresh pools | re-run `python -m scripts.init_postgres --seed sql/postgres/seed.sql` (idempotent), or `docker compose down -v` to wipe the volume entirely |
| `KeyError`/empty pool in `generate_data` | Postgres isn't seeded — run the §3 `init_postgres --seed` step first |
