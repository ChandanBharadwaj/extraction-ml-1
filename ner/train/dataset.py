"""Tokenize Records into model inputs with BIO labels per token.

This bridges char-offset spans to per-token labels via the fast tokenizer's
offset_mapping. Special tokens get label_id = -100 so CrossEntropyLoss skips them.

Lives in `ner.train` because it depends on a real tokenizer at training time.
At inference we re-tokenize with the same tokenizer but don't need labels.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ner.bio import char_spans_to_bio
from ner.constants import MAX_SEQ_LEN
from ner.schema import Record

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


def encode_record(
    record: Record,
    tokenizer: "PreTrainedTokenizerFast",
    *,
    max_length: int = MAX_SEQ_LEN,
) -> dict[str, Any]:
    enc = tokenizer(
        record.text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_special_tokens_mask=True,
    )
    labels = char_spans_to_bio(record.entities, enc["offset_mapping"])
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def encode_records(
    records: list[Record],
    tokenizer: "PreTrainedTokenizerFast",
    *,
    max_length: int = MAX_SEQ_LEN,
):
    """Yield encoded examples; intended to feed `datasets.Dataset.from_generator`."""
    for r in records:
        yield encode_record(r, tokenizer, max_length=max_length)
