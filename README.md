# Real-Time Multi-Entity NER

Production NER pipeline for `PERSON`, `ORG`, `ADDRESS`, and `COMMODITY` from
messy short commercial text (invoices, manifests, emails). Single DeBERTa-v3
model with a BIO head, deployed as FP32 ONNX behind a Python-only runtime that
hits the 1 s/record SLA on CPU.

See the TDD for the design rationale; this README covers shape, install, and
how to wire your seed data in.

## Layout

```
ner/
  constants.py        BIO label space (frozen contract; do not reorder)
  schema.py           Entity / Record dataclasses with span validation
  bio.py              char_spans <-> per-token BIO ids (uses offset_mapping)
  data/
    pools.py          SQLite seed loader (sql/schema.sql)
    slot_fill.py      Deterministic offset generator
    free_gen.py       Regex relocation of LLM-emitted entity surfaces
    noise.py          Casing (lower/UPPER), punctuation drops, typos, OCR, truncation
    assembler.py      Top-level data pipeline + JSONL I/O
  llm/
    claude_generator.py   Optional Anthropic SDK hook (prompt-cached)
  train/
    dataset.py        Record -> tokenized example with BIO labels
    metrics.py        Span-F1 compute_metrics for HF Trainer
    train.py          Trainer entry point
  export/
    onnx_export.py    FP32 ONNX export with graph optimization
  infer/
    runtime.py        Python-only serving (tokenizers + onnxruntime + numpy)
  eval/
    gold.py           Hand-labeled validation seed (replace at scale)
    metrics.py        Exact-span P/R/F1 evaluator
sql/
  schema.sql          Seed pool DB schema
  example_seed.sql    Tiny example to bootstrap the pipeline
scripts/              CLI entry points (generate_data / train / export_onnx / infer)
tests/                Deterministic-core test suite
```

## Install

```bash
pip install -e .[dev]                 # tests
pip install -e .[train]               # to train (pulls torch)
pip install -e .[inference]           # serving-only (no torch)
pip install -e .[data]                # optional: Anthropic SDK for LLM hook
```

## Seed your pools (SQL)

The data layer is driven by a SQLite DB matching `sql/schema.sql`. Three tables:

* `entity_pools(entity_type, value, weight)` — PERSON / ORG / ADDRESS / COMMODITY
* `decoy_pools(slot_name, value, weight)` — quantity, units, invoice IDs, etc.
* `templates(template, weight)` — sentence templates

Template placeholders:

* `{PERSON}` `{ORG}` `{ADDRESS}` `{COMMODITY}` — entity slots (label-bearing)
* `{PERSON#1}` `{PERSON#2}` — indexed to force distinct samples for repeats
* `{decoy:qty}` `{decoy:invoice_id}` `{decoy:unit}` — non-entity fillers

Bootstrap a DB from a `.sql` file:

```bash
python -m scripts.generate_data --init-db data/pools.sqlite --seed-sql my_seed.sql
```

## Generate training data

```bash
python -m scripts.generate_data \
    --sqlite data/pools.sqlite \
    --out data/train.jsonl \
    --n 50000 --seed 42 --noise-prob 0.6
```

Every emitted record passes `Record.validate()` — char offsets are computed in
Python, not by the LLM, so `text[start:end] == ent.text` is a hard invariant.

## Train

```bash
python -m scripts.train \
    --train-jsonl data/train.jsonl \
    --output-dir artifacts/ckpt \
    --epochs 3 --batch-size 32
```

Early stopping uses the **real** gold set (`ner.eval.gold.GOLD_SEED` or
`--gold-jsonl` if you point at a larger hand-labeled file). Synthetic data is
never used for eval.

## Export to ONNX (FP32, graph-optimized)

```bash
python -m scripts.export_onnx \
    --model-dir artifacts/ckpt \
    --output artifacts/serve/model.onnx
cp artifacts/ckpt/tokenizer.json artifacts/serve/   # or transformers tokenizer files
```

Quantization is intentionally **not** applied — INT8 degrades the boundary
logits the BIO head depends on.

## Serve

```python
from ner.infer.runtime import from_artifact_dir

runtime = from_artifact_dir("artifacts/serve")
entities = runtime.predict("Manifest: galvanized steel coil, anhydrous ammonia...")
```

Threading defaults are tuned for short records on CPU. For backlog scoring of
millions of records, prefer multiprocessing over thread parallelism.

Long inputs (multi-line packing lists, full B/L bodies) are handled by
sliding-window inference: overlapping 256-token windows run as one batched
session call and merge token-wise, so entities beyond the first window are
still extracted. Inputs beyond `MAX_INPUT_CHARS` (4000) are truncated with a
`UserWarning`.

## Retrain runbook (after changing seed data, noise, or the gold set)

The data-side improvements (seed pools, templates, noise transforms, loss
weighting) are inert until a retrain; the serving/eval fixes apply to the
existing artifact immediately. To pick everything up:

```bash
# 1) Regenerate the seed SQL from the seedgen source of truth.
python -m scripts.build_seed

# 2) Rebuild the pool DB and synthetic training set.
python -m scripts.generate_data --init-db data/pools.sqlite --seed-sql sql/seed.sql
python -m scripts.generate_data --sqlite data/pools.sqlite --out data/train.jsonl --n 50000

# 3) Train. --gold-split earlystop keeps the early-stopping half of
#    gold disjoint from the threshold-tuning half. Optional knobs:
#    --class-weights neg_boost, --metric-for-best-model "f1_COMMODITY(NEG)".
python -m scripts.train --train-jsonl data/train.jsonl \
    --output-dir artifacts/ckpt --gold-split earlystop

# 4) Export FP32 ONNX.
python -m scripts.export_onnx --model-dir artifacts/ckpt --output artifacts/serve/model.onnx

# 5) Tune thresholds on the OTHER gold half, cargo-focused. Alternatives:
#    --objective f1_micro, or add --precision-floor-type "COMMODITY(NEG):0.9".
python -m scripts.tune_threshold --artifact-dir artifacts/serve \
    --eval-split tune --objective "f1_per_type:COMMODITY(NEG)"

# 6) SLA spot-check on a long manifest (>2000 chars) — windowed inference
#    should stay well under the 1 s/record CPU budget.
python -m scripts.infer --artifact-dir artifacts/serve --text "$(cat sample_manifest.txt)"
```

### No GPU?

Only step 3 wants one — everything else in this pipeline (data generation,
ONNX export, threshold tuning, serving, the test suite) is CPU-only by
design. Three routes, in order of preference:

1. **Free hosted GPU (recommended).** Google Colab (free T4) or a Kaggle
   notebook (free T4/P100, ~30 h/week) comfortably fits this fine-tune.
   In a fresh notebook:

   ```bash
   !git clone <your-repo-url> && cd extraction-ml-1 && pip install -e .[train]
   !cd extraction-ml-1 && python -m scripts.generate_data --init-db data/pools.sqlite --seed-sql sql/seed.sql
   !cd extraction-ml-1 && python -m scripts.generate_data --sqlite data/pools.sqlite --out data/train.jsonl --n 50000
   !cd extraction-ml-1 && python -m scripts.train --train-jsonl data/train.jsonl \
       --output-dir artifacts/ckpt --gold-split earlystop --class-weights neg_boost
   ```

   Then zip and download `artifacts/ckpt/`, and run steps 4–6 locally on CPU.
   A 50k-record, 3-epoch run on a T4 is roughly 1–2 hours.

2. **CPU fine-tune with a smaller footprint.** The HF Trainer falls back to
   CPU automatically; make the run tractable by shrinking everything:

   ```bash
   python -m scripts.generate_data --sqlite data/pools.sqlite --out data/train_small.jsonl --n 8000
   python -m scripts.train --train-jsonl data/train_small.jsonl \
       --output-dir artifacts/ckpt \
       --base-model microsoft/deberta-v3-xsmall \
       --epochs 2 --batch-size 8 --gold-split earlystop
   ```

   Expect several hours on a modern 8-core machine (deberta-v3-base at the
   full 50k on CPU is days — don't). `deberta-v3-xsmall`/`-small` trade some
   accuracy for a 5–10× smaller backbone; the export and serving layers are
   model-size agnostic, so nothing else changes.

3. **Skip retraining for now.** The runtime and eval fixes on this branch
   (sliding-window long-input handling, span-edge trimming, threshold-tuner
   parity, the enlarged cargo gold set) improve the *existing* artifact
   without any training — re-run step 5 against your current
   `artifacts/serve` to re-tune thresholds on the new gold set and deploy.

## Test

```bash
pytest -q
```

Tests cover: BIO encode/decode round-trip, schema/overlap validation, slot-fill
offset invariants under random seeds, noise-transform invariants, SQLite seed
contract, exact-span metric correctness, and gold-record validity.
