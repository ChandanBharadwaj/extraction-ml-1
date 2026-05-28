"""BIO encode/decode against a fast-tokenizer offset_mapping.

Encoding (char spans -> per-token BIO ids):
    Given a list of `Entity` char spans and a token offset_mapping
    `[(c_start, c_end), ...]`, assign each token the label of the entity whose
    span it lies inside. The leftmost token of an entity gets B-, the rest I-.
    Special tokens (offset == (0,0)) get -100 so the loss ignores them.

Decoding (per-token argmax -> char spans):
    Scan tokens left-to-right; open a span on B-X or on I-X when no span is open;
    close it when the label changes or runs of I-X stop. Use offset_mapping to
    translate token boundaries back to character indices. Punctuation/whitespace
    that the tokenizer skipped (no token covers it) is naturally excluded.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from ner.constants import LABEL2ID
from ner.schema import Entity

LABEL_IGNORE: int = -100

OffsetMapping = Sequence[tuple[int, int]]


def _label_id(prefix: str, entity_type: str) -> int:
    return LABEL2ID[f"{prefix}-{entity_type}"]


def char_spans_to_bio(
    entities: Iterable[Entity],
    offset_mapping: OffsetMapping,
) -> list[int]:
    """Project gold char spans onto a tokenizer's offset_mapping.

    A token (c0, c1) is part of an entity (e0, e1) iff its character window
    overlaps the entity span. The first such token in the entity gets B-, the
    rest get I-. Tokens not in any entity get O. Special tokens (offset (0,0))
    get LABEL_IGNORE so they don't contribute to the loss.
    """
    n_tokens = len(offset_mapping)
    labels: list[int] = [LABEL2ID["O"]] * n_tokens

    # Mark special-token positions as ignore.
    for i, (c0, c1) in enumerate(offset_mapping):
        if c0 == 0 and c1 == 0:
            labels[i] = LABEL_IGNORE

    sorted_entities = sorted(entities, key=lambda e: e.start)
    for ent in sorted_entities:
        first = True
        for i, (c0, c1) in enumerate(offset_mapping):
            if c0 == 0 and c1 == 0:
                continue  # special token
            # Half-open overlap test.
            if c0 >= ent.end:
                break
            if c1 <= ent.start:
                continue
            labels[i] = _label_id("B" if first else "I", ent.type)
            first = False
    return labels


def bio_ids_to_spans(
    label_ids: Sequence[int],
    offset_mapping: OffsetMapping,
    source_text: str,
    *,
    strict: bool = False,
) -> list[Entity]:
    """Decode per-token label ids into character-level Entity spans.

    `strict=False` (default) is lenient about the model emitting an I- tag with
    no matching B- (treats it as the start of a new span), which is the standard
    "BIO2" decoding behavior used in production NER. Setting `strict=True`
    discards orphan I- runs (closer to seqeval's `IOB1` semantics).
    """
    from ner.constants import ID2LABEL

    spans: list[Entity] = []
    open_type: str | None = None
    open_char_start: int | None = None
    open_char_end: int | None = None

    def close() -> None:
        nonlocal open_type, open_char_start, open_char_end
        if open_type is not None and open_char_start is not None and open_char_end is not None:
            text = source_text[open_char_start:open_char_end]
            if text:
                spans.append(Entity(open_type, text, open_char_start, open_char_end))
        open_type = None
        open_char_start = None
        open_char_end = None

    for i, lid in enumerate(label_ids):
        if lid == LABEL_IGNORE:
            continue
        c0, c1 = offset_mapping[i]
        if c0 == 0 and c1 == 0:
            continue  # special token
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            close()
            continue
        prefix, etype = label.split("-", 1)
        if prefix == "B":
            close()
            open_type = etype
            open_char_start = c0
            open_char_end = c1
        elif prefix == "I":
            if open_type == etype:
                # Extend; tolerate tiny gaps (whitespace the tokenizer skipped).
                open_char_end = c1
            else:
                # Orphan I- tag.
                if strict:
                    close()
                else:
                    close()
                    open_type = etype
                    open_char_start = c0
                    open_char_end = c1
    close()
    return spans
