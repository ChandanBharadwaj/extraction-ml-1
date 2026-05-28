# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A production NER pipeline that extracts `PERSON`, `ORG`, `ADDRESS`, and
`COMMODITY` entities (with POS/NEG polarity on COMMODITY) as exact character
spans from short, messy commercial text. The serving layer is Python-only
(tokenizers + onnxruntime + numpy — no torch); training and ONNX export use
DeBERTa-v3-base with a BIO head. See `README.md` for the full design rationale.

## Common commands

Install (optional-deps groups defined in `pyproject.toml`):

```bash
pip install -e .[dev]            # tests + ruff
pip install -e .[train]          # adds torch / transformers / datasets / seqeval / onnx
pip install -e .[inference]      # serving only (numpy + onnxruntime + tokenizers)
pip install -e .[data]           # adds anthropic SDK for the LLM data-gen hook
```

Tests:

```bash
pytest                                # full deterministic suite (~60 tests, sub-second)
pytest tests/test_bio.py -v           # one file
pytest -k threshold                   # by name substring
```

Pipeline CLIs (all installed as `python -m scripts.<name>`):

```bash
# 1) Initialize a SQLite pool DB from sql/schema.sql + a seed .sql.
python -m scripts.generate_data --init-db data/pools.sqlite --seed-sql sql/example_seed.sql

# 2) Generate synthetic training JSONL from the pool DB.
python -m scripts.generate_data --sqlite data/pools.sqlite --out data/train.jsonl --n 50000

# 3) Fine-tune DeBERTa-v3 on the JSONL; early-stop on real gold.
python -m scripts.train --train-jsonl data/train.jsonl --output-dir artifacts/ckpt

# 4) Export to FP32 ONNX (graph-optimized; INT8 is prohibited by the TDD).
python -m scripts.export_onnx --model-dir artifacts/ckpt --output artifacts/serve/model.onnx

# 5) Tune per-label confidence thresholds on the gold set.
python -m scripts.tune_threshold --artifact-dir artifacts/serve --gold-jsonl data/gold_dev.jsonl
# Also: --objective max_f1_at_precision_floor --precision-floor 0.95

# 6) Serve.
python -m scripts.infer --artifact-dir artifacts/serve --text "manifest contains no robusta coffee, only cane sugar"
```

## Architecture: pipeline stages decoupled by artifacts

Data flows through four stages that talk only through files:

```
SQLite pool DB  --(slot_fill + noise)-->  train.jsonl
                                              |
                                              v
                                        HF Trainer  ----> artifacts/ckpt/  (PyTorch)
                                                            |
                                                            v
                                                      onnx_export  ----> artifacts/serve/model.onnx
                                                                                  |
                                          tune_threshold(gold.jsonl)  ----> thresholds.json
                                                                                  |
                                                                                  v
                                                                          NERRuntime (numpy + ORT)
```

Each stage is a small, testable unit; the serving layer never imports torch
or transformers. The runtime auto-loads `thresholds.json` from the artifact
dir if present (absence is a no-op — classic argmax decoding).

## Frozen contracts you must not break casually

These cross-cut multiple files; touching them requires changing all consumers.

1. **`ner.constants.LABEL_LIST` is append-only.** Index = class id; reordering
   silently invalidates any trained checkpoint. New label types go at the end.
   Currently 11 labels (`O` + 4 entity types × {B,I} + `NEG_COMMODITY` × {B,I}).

2. **`Entity` uses half-open character intervals** `[start, end)`. The invariant
   `text[start:end] == entity.text` is enforced by `Record.validate()` at
   construction time everywhere. Char offsets are computed in **Python**, never
   by the LLM — slot-fill substitutes literal strings and records positions
   from the output buffer length.

3. **`Entity.polarity` defaults to `"POS"`.** `"NEG"` is only valid for types
   in `NEGATABLE_TYPES` (currently `{"COMMODITY"}`). Polarity is part of the
   metric match key — a polarity flip counts as 1 FP + 1 FN, not a partial
   credit. JSON round-trips include the field; legacy JSONL without polarity
   loads as POS.

4. **`Record.meta["preserve_spans"]`** is a list of `(start, end)` char ranges
   the noise injector must not touch. Slot-fill populates it for `neg_cue` and
   `contrast_cue` decoys (see `slot_fill.PRESERVE_DECOY_SLOTS`). The set
   exists because dropping the comma after "does not contain" or the word
   "no" would silently flip gold polarity without re-labeling.

5. **Synthetic data is never used for evaluation.** `ner.eval.gold.GOLD_SEED`
   (35 hand-labeled records, offsets verified by Python) is the sole source
   of truth for early stopping and threshold tuning. Synthetic-to-real
   leakage breaks this guarantee.

## Slot-fill template grammar

Templates live in the `templates` table of the SQLite pool DB and use:

- `{PERSON}` `{ORG}` `{ADDRESS}` `{COMMODITY}` — entity slots (label-bearing).
- `{NEG_COMMODITY}` — samples from the COMMODITY pool, emits `polarity=NEG`.
- `{PERSON#1}` `{PERSON#2}` `{NEG_COMMODITY#1}` — indexed slots that force
  distinct samples when the same type repeats in one template.
- `{decoy:qty}` `{decoy:neg_cue}` `{decoy:contrast_cue}` `{decoy:frozen_compound}` etc. —
  non-entity fillers from the `decoy_pools` table.

The slot regex (`ner/data/slot_fill.py:_SLOT_RE`) is shared by parser and
generator — widening `kind` to allow new prefixes (e.g. for `UNCERTAIN_COMMODITY`
in the future) means updating the regex *and* the dispatch in
`_resolve_entity_slot`.

## Threshold tuner mental model

`ner.eval.threshold_sweep` runs the model **once** over the gold set, caches
softmax probabilities per token per label, then sweeps threshold grids
entirely in numpy — no per-candidate model run. Two-stage search:

1. Global threshold (one value applied to every non-O label) over `[0, 0.99]`.
2. Per-label refinement starting from the global optimum.

Objectives: `f1_micro` (default), `f1_macro`, `f1_per_type:<bucket>` (e.g.
`COMMODITY(NEG)`), `max_f1_at_precision_floor`, `max_f1_at_recall_floor`. The
last two refuse to write `thresholds.json` if no threshold combination
satisfies the floor (non-zero exit with diagnostics).

## Where to look first

- Changing what entities exist or how they're labeled → `ner/constants.py` + `ner/schema.py` (then run the full test suite to surface every consumer).
- Adding a new noise transformation → `ner/data/noise.py` (honor `preserve_spans` and entity-surface guards; reproject offsets via `_apply_to_record`).
- Adding new template patterns or pools → `sql/example_seed.sql` (real seeds drop in against `sql/schema.sql` — no schema change needed for new decoy slot names or qualified commodity values).
- Tuning the serving operating point → `ner/eval/threshold_sweep.py` + `scripts/tune_threshold.py`.
- LLM data generation (Anthropic SDK with prompt caching, optional) → `ner/llm/claude_generator.py`.
