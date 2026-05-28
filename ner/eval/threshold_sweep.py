"""Post-training threshold tuner for the BIO head.

Premise: with thresholds=0 the decoder is `argmax`, which means every non-O
token contributes to some span. In production we want a per-label confidence
floor — predictions whose softmax probability falls below their label's
threshold are demoted to O. This script picks the thresholds that maximize a
chosen objective on a gold (hand-labeled) set.

Pipeline:
  1. `cache_probabilities` — run the model once over the gold set, save
     softmax probabilities + tokenizer offsets per record. The hot inner loop
     in (2)/(3) never re-touches the model.
  2. `decode_with_thresholds` — pure numpy: apply per-label gates, argmax,
     run `bio_ids_to_spans`, return predicted entity lists.
  3. `sweep` — two-stage search:
       (a) global threshold: one value applied to every non-O label
       (b) per-label refinement: starting from the global optimum, sweep each
           label independently and accept any improvement.

Failure modes handled explicitly:
  - Labels with zero gold support: cannot be tuned; left at 0.0 with a note.
  - Precision-floor / recall-floor objectives that no threshold can reach:
    the script reports the best achievable value and returns a `SweepResult`
    with `feasible=False`. The CLI converts this to a non-zero exit.
  - No threshold beats argmax: result is still written (all-zeros) with a
    note so deployment doesn't silently ship without a thresholds.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from ner.bio import apply_threshold_gate, bio_ids_to_spans, softmax
from ner.constants import ID2LABEL, LABEL2ID, NUM_LABELS
from ner.eval.metrics import EvalReport, evaluate
from ner.schema import Entity, Record

if TYPE_CHECKING:
    from ner.infer.runtime import NERRuntime


@dataclass
class ProbCache:
    """Per-record cached softmax probabilities and tokenization metadata."""
    probs: list[np.ndarray]                    # each (n_tokens, n_labels) float32
    offsets: list[list[tuple[int, int]]]       # each token's char offset
    texts: list[str]                           # source text per record


@dataclass
class SweepResult:
    thresholds: dict[str, float]
    report: EvalReport
    objective: str
    feasible: bool
    notes: list[str] = field(default_factory=list)
    sweep_curve: list[dict] = field(default_factory=list)

    def to_json(self, gold_set_path: str | None, gold_records: int) -> dict:
        return {
            "labels": self.thresholds,
            "objective": self.objective,
            "feasible": self.feasible,
            "gold_set_path": gold_set_path,
            "gold_records": gold_records,
            "report": self.report.to_dict(),
            "notes": self.notes,
            "sweep_curve": self.sweep_curve,
        }


# ----------------------------- caching layer ------------------------------ #

def cache_probabilities(
    runtime: "NERRuntime",
    gold: list[Record],
) -> ProbCache:
    """Run the ONNX model once over the gold set, return cached softmax
    probabilities + offset mappings.

    Uses the runtime's existing `_encode` + `_session.run` to keep the
    tokenization path identical to production serving.
    """
    probs_list: list[np.ndarray] = []
    offsets_list: list[list[tuple[int, int]]] = []
    texts: list[str] = []

    for rec in gold:
        text = rec.text[: runtime.config.max_input_chars]
        input_ids, attention_mask, offsets = runtime._encode(text)
        logits = runtime._session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]  # shape (1, n_tokens, n_labels)
        p = softmax(logits[0].astype(np.float32), axis=-1)
        probs_list.append(p)
        offsets_list.append(list(offsets))
        texts.append(text)

    return ProbCache(probs=probs_list, offsets=offsets_list, texts=texts)


# --------------------------- decoding layer ------------------------------- #

def decode_with_thresholds(
    cache: ProbCache,
    thresholds: np.ndarray,
) -> list[list[Entity]]:
    """Gate per-token argmax by `thresholds` (shape (n_labels,)) and run the
    BIO span decoder. Returns one entity list per cached record."""
    out: list[list[Entity]] = []
    for probs, offsets, text in zip(cache.probs, cache.offsets, cache.texts):
        gated_ids = apply_threshold_gate(probs, thresholds)
        out.append(bio_ids_to_spans(gated_ids.tolist(), offsets, text))
    return out


# --------------------------- objective layer ------------------------------ #

ObjectiveFn = Callable[[EvalReport], float]


def _f1_micro(rep: EvalReport) -> float:
    return rep.micro.f1


def _f1_macro(rep: EvalReport) -> float:
    f1s = [m.f1 for m in rep.per_type.values() if m.support > 0]
    return sum(f1s) / len(f1s) if f1s else 0.0


def _f1_per_type(bucket: str) -> ObjectiveFn:
    def fn(rep: EvalReport) -> float:
        m = rep.per_type.get(bucket)
        return m.f1 if m else 0.0
    return fn


def _f1_at_precision_floor(floor: float) -> ObjectiveFn:
    """F1 subject to micro precision >= floor; -inf otherwise."""
    def fn(rep: EvalReport) -> float:
        if rep.micro.precision >= floor:
            return rep.micro.f1
        return -1.0  # infeasible; any feasible candidate dominates this
    return fn


def _f1_at_recall_floor(floor: float) -> ObjectiveFn:
    def fn(rep: EvalReport) -> float:
        if rep.micro.recall >= floor:
            return rep.micro.f1
        return -1.0
    return fn


def parse_objective(name: str, *, precision_floor: float = 0.0, recall_floor: float = 0.0) -> ObjectiveFn:
    if name == "f1_micro":
        return _f1_micro
    if name == "f1_macro":
        return _f1_macro
    if name.startswith("f1_per_type:"):
        bucket = name.split(":", 1)[1]
        return _f1_per_type(bucket)
    if name == "max_f1_at_precision_floor":
        return _f1_at_precision_floor(precision_floor)
    if name == "max_f1_at_recall_floor":
        return _f1_at_recall_floor(recall_floor)
    raise ValueError(f"Unknown objective: {name!r}")


# ----------------------------- sweep ------------------------------------- #

def _grid(step: float) -> np.ndarray:
    # 0.0, step, 2*step, ..., 1.0 - step
    n = int(round(1.0 / step))
    return np.linspace(0.0, 1.0 - step, n, dtype=np.float32)


def _support_per_label(gold: list[Record]) -> dict[int, int]:
    """Count gold entities per label id (B-* only, since I- support is implied)."""
    counts: dict[int, int] = {i: 0 for i in range(NUM_LABELS)}
    for rec in gold:
        for ent in rec.entities:
            type_key = f"NEG_{ent.type}" if ent.polarity == "NEG" else ent.type
            b_id = LABEL2ID[f"B-{type_key}"]
            counts[b_id] += 1
    return counts


def sweep(
    cache: ProbCache,
    gold: list[Record],
    objective_name: str,
    *,
    step: float = 0.01,
    precision_floor: float = 0.0,
    recall_floor: float = 0.0,
) -> SweepResult:
    objective = parse_objective(
        objective_name, precision_floor=precision_floor, recall_floor=recall_floor,
    )
    grid = _grid(step)
    notes: list[str] = []

    label_support = _support_per_label(gold)
    for lid, count in label_support.items():
        if lid == 0:
            continue
        if count == 0 and ID2LABEL[lid].startswith("B-"):
            notes.append(f"no gold support for {ID2LABEL[lid]}; left at 0.0")

    def evaluate_thresholds(thresholds: np.ndarray) -> EvalReport:
        preds = decode_with_thresholds(cache, thresholds)
        return evaluate(preds, gold)

    # Stage 1: global threshold sweep (one value applied to every non-O label).
    sweep_curve: list[dict] = []
    best_global = 0.0
    best_global_score = -float("inf")
    best_global_report: EvalReport | None = None
    for g in grid:
        thresholds = np.full(NUM_LABELS, float(g), dtype=np.float32)
        thresholds[0] = 0.0  # O is never gated
        rep = evaluate_thresholds(thresholds)
        score = objective(rep)
        sweep_curve.append({
            "stage": "global", "global": float(g),
            "f1": rep.micro.f1, "precision": rep.micro.precision, "recall": rep.micro.recall,
            "objective": score,
        })
        if score > best_global_score:
            best_global_score = score
            best_global = float(g)
            best_global_report = rep

    feasible = best_global_score > -1.0
    if not feasible:
        # Objective never satisfied at any global threshold; still try per-label
        # in case a label-specific configuration can hit the floor.
        notes.append(f"global sweep could not satisfy objective {objective_name!r}; falling back to per-label search at threshold 0.")

    thresholds = np.full(NUM_LABELS, best_global, dtype=np.float32)
    thresholds[0] = 0.0

    # Stage 2: per-label refinement.
    best_score = best_global_score
    best_report = best_global_report
    for lid in range(1, NUM_LABELS):
        baseline = thresholds[lid]
        local_best_t = baseline
        for t in grid:
            thresholds[lid] = float(t)
            rep = evaluate_thresholds(thresholds)
            score = objective(rep)
            sweep_curve.append({
                "stage": "per_label", "label": ID2LABEL[lid],
                "threshold": float(t),
                "f1": rep.micro.f1, "precision": rep.micro.precision, "recall": rep.micro.recall,
                "objective": score,
            })
            if score > best_score:
                best_score = score
                local_best_t = float(t)
                best_report = rep
        thresholds[lid] = local_best_t

    if best_score <= -1.0:
        feasible = False
    else:
        feasible = True

    if best_report is None:
        # Shouldn't happen unless gold is empty; fall back to a no-gate report.
        best_report = evaluate_thresholds(np.zeros(NUM_LABELS, dtype=np.float32))

    if all(thresholds[i] == 0.0 for i in range(NUM_LABELS)):
        notes.append("no threshold improved the objective; writing zeros (deployment is equivalent to argmax decoding)")

    thresholds_dict = {
        ID2LABEL[i]: float(thresholds[i])
        for i in range(NUM_LABELS) if ID2LABEL[i] != "O"
    }
    return SweepResult(
        thresholds=thresholds_dict,
        report=best_report,
        objective=objective_name,
        feasible=feasible,
        notes=notes,
        sweep_curve=sweep_curve,
    )


def thresholds_dict_to_array(d: dict[str, float]) -> np.ndarray:
    """Materialize a thresholds dict (label_name -> float) into a (NUM_LABELS,)
    numpy array suitable for `apply_threshold_gate`. Missing entries are 0.0."""
    arr = np.zeros(NUM_LABELS, dtype=np.float32)
    for label, t in d.items():
        if label in LABEL2ID:
            arr[LABEL2ID[label]] = float(t)
    arr[0] = 0.0  # O is never gated
    return arr


def write_thresholds_json(
    result: SweepResult,
    output_path: str | Path,
    *,
    gold_set_path: str | None,
    gold_records: int,
) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        result.to_json(gold_set_path, gold_records), indent=2,
    ))
    return p


def load_thresholds_json(path: str | Path) -> np.ndarray:
    obj = json.loads(Path(path).read_text())
    return thresholds_dict_to_array(obj.get("labels", {}))
