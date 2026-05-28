"""Hugging Face Trainer-compatible metric callback that reports span F1.

We avoid seqeval here because seqeval scores at the token-tag level; the SLA is
*exact char-offset spans*, so we decode predictions back to char spans before
scoring against the gold validation records.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ner.bio import bio_ids_to_spans
from ner.eval.metrics import evaluate
from ner.schema import Record

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


class SpanF1Metric:
    """Stateful object holding the gold set and tokenizer for compute_metrics."""

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerFast",
        gold_records: list[Record],
    ):
        self.tokenizer = tokenizer
        self.gold = gold_records
        self._encoded = [
            tokenizer(
                r.text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=256,
            ) for r in gold_records
        ]

    def __call__(self, eval_pred) -> dict[str, float]:
        logits, _labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        decoded: list = []
        for i, rec in enumerate(self.gold):
            offsets = self._encoded[i]["offset_mapping"]
            spans = bio_ids_to_spans(preds[i].tolist(), offsets, rec.text)
            decoded.append(spans)
        report = evaluate(decoded, self.gold)
        out = {
            "f1": report.micro.f1,
            "precision": report.micro.precision,
            "recall": report.micro.recall,
        }
        for et, m in report.per_type.items():
            out[f"f1_{et}"] = m.f1
        return out
