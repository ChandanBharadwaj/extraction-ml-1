"""Span-level precision / recall / F1 by entity type.

The TDD requires *exact character-offset spans*, so we score on exact match of
(type, start, end). Partial overlaps count as both a false positive and a false
negative, which is the strict standard for production NER.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ner.constants import ENTITY_TYPES
from ner.schema import Entity, Record


@dataclass(frozen=True)
class TypeMetrics:
    precision: float
    recall: float
    f1: float
    support: int
    pred_count: int


@dataclass(frozen=True)
class EvalReport:
    per_type: dict[str, TypeMetrics]
    micro: TypeMetrics

    def to_dict(self) -> dict:
        return {
            "per_type": {k: v.__dict__ for k, v in self.per_type.items()},
            "micro": self.micro.__dict__,
        }


def _score(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _key(e: Entity) -> tuple[str, int, int]:
    return (e.type, e.start, e.end)


def evaluate(predictions: list[list[Entity]], gold: list[Record]) -> EvalReport:
    """Compute per-type and micro-averaged span F1.

    Predictions are a list (one per record) of predicted entity lists, aligned
    positionally with `gold`.
    """
    if len(predictions) != len(gold):
        raise ValueError("predictions / gold length mismatch")

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for preds, ref in zip(predictions, gold):
        ref_set = {_key(e) for e in ref.entities}
        pred_set = {_key(e) for e in preds}
        for k in pred_set:
            if k in ref_set:
                tp[k[0]] += 1
            else:
                fp[k[0]] += 1
        for k in ref_set:
            if k not in pred_set:
                fn[k[0]] += 1

    per_type: dict[str, TypeMetrics] = {}
    total_tp = total_fp = total_fn = 0
    for et in ENTITY_TYPES:
        p, r, f = _score(tp[et], fp[et], fn[et])
        per_type[et] = TypeMetrics(
            precision=p, recall=r, f1=f,
            support=tp[et] + fn[et], pred_count=tp[et] + fp[et],
        )
        total_tp += tp[et]
        total_fp += fp[et]
        total_fn += fn[et]
    p, r, f = _score(total_tp, total_fp, total_fn)
    micro = TypeMetrics(
        precision=p, recall=r, f1=f,
        support=total_tp + total_fn, pred_count=total_tp + total_fp,
    )
    return EvalReport(per_type=per_type, micro=micro)
