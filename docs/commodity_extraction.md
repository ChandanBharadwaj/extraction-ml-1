# Open-vocabulary commodity extraction

Extract **commodity goods from arbitrary text** as exact character spans, with
POS/NEG polarity, fast on CPU. Open vocabulary: the target is any traded good,
not a maintained product list.

This is a different problem from the rest of the repo, which extracts four entity
types using data generated from a fixed pool. The distinction is not stylistic —
it is measured below, and it determines what can be reused.

## Why the repo's own pipeline cannot serve this goal

| Measurement | Value |
|---|---|
| Repo commodity pool size | 589 strings |
| HS 6-digit subheadings (real traded goods) | 5,613 → 1,396 distinct heads |
| Pool ∩ HS heads, exact | **3.5%** |
| Pool ∩ HS heads, counting shared head nouns | **13.3%** |
| Gold-set spans that are in the pool | **78%** |

The pool has no representation of selenium, bulldozers, transformers, hearing
aids, photographic film, turbo-propellers, sewing machine needles. The synthetic
generator slot-fills *from that pool*, so a model trained on its output learns
those 589 strings. And because the 72-record gold set was hand-authored alongside
the pool, it cannot detect that failure — it shares the vocabulary.

The harness makes this concrete. The same dictionary matcher, same code, two
evaluation sets:

```
                              P      R     F1
repo gold fixture          0.838  0.769  0.802     <- 78% shared vocabulary
HS open-vocabulary probe   0.098  0.027  0.043     <- 2.7% shared vocabulary
```

A 0.76 F1 collapse from vocabulary overlap alone. `TestMemorizationControl` in
`tests/test_commodity_bench.py` pins this property so it cannot silently regress.

## Architecture

Two composable halves, split along the axis that matters:

```
raw text ──► span extractor (open vocabulary, model)  ──► spans
                                                          │
                                    polarity tagger (rules) ──► POS/NEG spans
```

**Spans need a model.** Commodities are an open class; the dominant error mode is
span *extent* on descriptive trade text (`'CANE SUGAR'` where gold is
`'WHITE REFINED CANE SUGAR ICUMSA 45'`). No dictionary solves that.

**Polarity does not.** Negation cues are a small closed class, so rules transfer
where the vocabulary does not:

| Set | Spans | Polarity F1 (given perfect spans) |
|---|---|---|
| Repo gold fixture | 121 | 1.000 |
| HS open-vocabulary probe (unseen vocabulary *and* frames) | 1,032 | 1.000 |

For contrast, ignoring negation entirely scores 0.587 on the gold set — 41% of
its commodity spans are negated. Published negation models do not transfer
(NegBERT: ~90 F1 in-domain, ~70 cross-domain), and every off-the-shelf library
ships a clinical lexicon.

The rules were tuned on the gold fixture, so 1.000 there means "saturates the
fixture". The probe number is the meaningful one: unseen vocabulary, unseen
framing, still 1.000.

## Layout

| Module | Role |
|---|---|
| `ner/commodity/protocol.py` | `CommodityExtractor` Protocol — same shape as `NERRuntime`, so candidates are drop-in. `CommodityFilter` adapts a multi-type model. |
| `ner/commodity/negation.py` | `PolarityTagger` — cue lexicon, frozen-compound masking, scope termination. Stateless, model-agnostic. |
| `ner/commodity/gliner_adapter.py` | GLiNER / GLiNER2 / NuNER behind the Protocol. Lazy torch import. `LabelSweep` sweeps label strings and thresholds over shared weights. |
| `ner/commodity/gazetteer.py` | Dictionary matcher. **A control, not a candidate** — used to measure memorization. |
| `ner/commodity/bench.py` | Scoring: reuses `ner.eval.metrics.evaluate`, adds error taxonomy, bootstrap CIs, per-record latency. |
| `ner/commodity/eval_sets.py` | Builds the HS OOV probe; loads your hand-labeled JSONL **with validation**. |

## Usage

```bash
pip install -e .[dev]                  # harness, control, probe — no model downloads
pip install -e .[commodity]            # adds gliner (pulls torch, ~2GB)

# refresh the HS vocabulary (committed, so only needed to update it)
python -m scripts.build_hs_vocab

# offline sanity: the memorization control on both sets
python -m scripts.bench_commodity --gold gold  --control
python -m scripts.bench_commodity --gold probe --control

# compare model candidates
python -m scripts.bench_commodity --gold probe --models fastino/gliner2-base-v1
python -m scripts.bench_commodity --gold probe --sweep  fastino/gliner2-base-v1
```

### Measuring on your own data

The benchmark above uses independent data by design, so nothing here is fitted to
your text. To get the number that actually decides:

```bash
# 1. pre-annotate (runs locally; your text never leaves the machine)
python -m scripts.label_assist --input my_records.txt \
    --model fastino/gliner2-base-v1 --out to_correct.jsonl

# 2. correct to_correct.jsonl by hand, then check the offsets
python -m scripts.label_assist --verify to_correct.jsonl

# 3. score any candidate against it
python -m scripts.bench_commodity --gold to_correct.jsonl --sweep fastino/gliner2-base-v1
```

**Aim for ~300 records.** Measured on the 72-record fixture, the bootstrap 95% CI
on F1 was ±0.08 — two candidates had to differ by 16 F1 points to be
distinguishable. The harness prints the CI width with every table for this reason.

## Notes and constraints

- **The probe is directional, not absolute.** Its vocabulary is genuinely open,
  but its document framing is hand-written. It answers "does this model
  generalize past a memorized list?" — not "how will it do on your documents?"
- **Sweep label strings before concluding a model fails.** These models take the
  entity type as text; `"commodity"` / `"goods"` / `"cargo"` are different queries
  against the same weights. `LabelSweep` covers the presets over one model load.
- **Do not INT8-quantize GLiNER.** Its issue tracker documents quantized exports
  returning empty results and one fine-tuned export running 50% *slower* than
  PyTorch. Use fp32 ONNX and A/B it.
- **If a fine-tune happens**, split the vocabulary with
  `eval_sets.split_vocabulary` so train and eval share no surface forms.
  Otherwise the evaluation measures memorization again, just with a longer list.
- **Licensing.** HS data is ODC-PDDL (public domain). The `gliner` library is
  Apache-2.0, but v1-era checkpoints were trained on ChatGPT-generated Pile-NER —
  read the `license:` field on the specific model card before commercial use.
