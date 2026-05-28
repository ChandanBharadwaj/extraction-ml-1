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
    noise.py          Lowercasing, punctuation drops, typos, truncation
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

## Test

```bash
pytest -q
```

Tests cover: BIO encode/decode round-trip, schema/overlap validation, slot-fill
offset invariants under random seeds, noise-transform invariants, SQLite seed
contract, exact-span metric correctness, and gold-record validity.
