"""Span-level precision / recall / F1 by entity type and polarity.

The TDD requires *exact character-offset spans*, so we score on exact match of
`(type, polarity, start, end)`. Partial overlaps count as both a false positive
and a false negative — the strict standard for production NER. A POS prediction
for a NEG gold span (or vice versa) is the same kind of miss as a wrong-type or
wrong-edge prediction: it's a polarity-mismatch span, counted as 1 FP + 1 FN.

`per_type` reporting splits negatable types so polarity quality is visible on
every eval pass — e.g. `COMMODITY(POS)` and `COMMODITY(NEG)` are separate rows.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ner.constants import ENTITY_TYPES, NEGATABLE_TYPES
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


def _key(e: Entity) -> tuple[str, str, int, int]:
    """Match key: (type, polarity, start, end). Polarity mismatch = miss."""
    return (e.type, e.polarity, e.start, e.end)


def _bucket(e: Entity) -> str:
    """Per-type reporting bucket. Negatable types split by polarity."""
    if e.type in NEGATABLE_TYPES:
        return f"{e.type}({e.polarity})"
    return e.type


def _bucket_keys() -> list[str]:
    out: list[str] = []
    for et in ENTITY_TYPES:
        if et in NEGATABLE_TYPES:
            out.append(f"{et}(POS)")
            out.append(f"{et}(NEG)")
        else:
            out.append(et)
    return out


def _bucket_for_key(key: tuple[str, str, int, int]) -> str:
    etype, polarity, _, _ = key
    if etype in NEGATABLE_TYPES:
        return f"{etype}({polarity})"
    return etype


def evaluate(predictions: list[list[Entity]], gold: list[Record]) -> EvalReport:
    """Compute per-type/polarity and micro-averaged span F1.

    Predictions are a list (one per record) of predicted entity lists, aligned
    positionally with `gold`.
    """
    if len(predictions) != len(gold):
        raise ValueError("predictions / gold length mismatch")

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for preds, ref in zip(predictions, gold):
        ref_set = {_key(e) for e in ref.entities}
        pred_set = {_key(e) for e in preds}
        for k in pred_set:
            if k in ref_set:
                tp[_bucket_for_key(k)] += 1
            else:
                fp[_bucket_for_key(k)] += 1
        for k in ref_set:
            if k not in pred_set:
                fn[_bucket_for_key(k)] += 1

    per_type: dict[str, TypeMetrics] = {}
    total_tp = total_fp = total_fn = 0
    for bucket in _bucket_keys():
        p, r, f = _score(tp[bucket], fp[bucket], fn[bucket])
        per_type[bucket] = TypeMetrics(
            precision=p, recall=r, f1=f,
            support=tp[bucket] + fn[bucket],
            pred_count=tp[bucket] + fp[bucket],
        )
        total_tp += tp[bucket]
        total_fp += fp[bucket]
        total_fn += fn[bucket]
    p, r, f = _score(total_tp, total_fp, total_fn)
    micro = TypeMetrics(
        precision=p, recall=r, f1=f,
        support=total_tp + total_fn, pred_count=total_tp + total_fp,
    )
    return EvalReport(per_type=per_type, micro=micro)
