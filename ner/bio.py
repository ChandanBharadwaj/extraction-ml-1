"""BIO encode/decode against a fast-tokenizer offset_mapping.

Encoding (char spans -> per-token BIO ids):
    Given a list of `Entity` char spans and a token offset_mapping
    `[(c_start, c_end), ...]`, assign each token the label of the entity whose
    span it lies inside. The leftmost token of an entity gets B-, the rest I-.
    Special tokens (offset == (0,0)) get -100 so the loss ignores them.
    COMMODITY entities with polarity="NEG" use the NEG_COMMODITY labels.

Decoding (per-token argmax -> char spans):
    Scan tokens left-to-right; open a span on B-X or on I-X when no span is open;
    close it when the label changes or runs of I-X stop. Use offset_mapping to
    translate token boundaries back to character indices. Punctuation/whitespace
    that the tokenizer skipped (no token covers it) is naturally excluded.
    Decoded NEG_COMMODITY spans become Entity(type=COMMODITY, polarity=NEG).

Threshold gating:
    `apply_threshold_gate` is a pure-numpy helper used by the inference runtime
    (and the threshold tuner): a non-O argmax whose softmax probability falls
    below the per-label threshold is demoted to O. With thresholds == 0.0 it's
    a no-op, so absence of `thresholds.json` preserves classic argmax decoding.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

from ner.constants import LABEL2ID
from ner.schema import Entity

if TYPE_CHECKING:
    import numpy as np

LABEL_IGNORE: int = -100

OffsetMapping = Sequence[tuple[int, int]]

# Type prefix used to encode polarity=NEG on a COMMODITY label.
_NEG_PREFIX: str = "NEG_"


def _label_id_for_entity(prefix: str, ent: Entity) -> int:
    """Map (B/I, Entity) to a label id, honoring `ent.polarity`."""
    type_key = f"{_NEG_PREFIX}{ent.type}" if ent.polarity == "NEG" else ent.type
    return LABEL2ID[f"{prefix}-{type_key}"]


def _decode_label(label_name: str) -> tuple[str, str, str] | None:
    """Parse a label name like "B-NEG_COMMODITY" → ("B", "COMMODITY", "NEG").

    Returns None for "O" or unknown labels.
    """
    if label_name == "O" or "-" not in label_name:
        return None
    prefix, type_key = label_name.split("-", 1)
    if type_key.startswith(_NEG_PREFIX):
        return prefix, type_key[len(_NEG_PREFIX):], "NEG"
    return prefix, type_key, "POS"


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

    for i, (c0, c1) in enumerate(offset_mapping):
        if c0 == 0 and c1 == 0:
            labels[i] = LABEL_IGNORE

    sorted_entities = sorted(entities, key=lambda e: e.start)
    for ent in sorted_entities:
        first = True
        for i, (c0, c1) in enumerate(offset_mapping):
            if c0 == 0 and c1 == 0:
                continue
            if c0 >= ent.end:
                break
            if c1 <= ent.start:
                continue
            labels[i] = _label_id_for_entity("B" if first else "I", ent)
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
    no matching B- (treats it as the start of a new span), the standard "BIO2"
    decoding behavior. Setting `strict=True` discards orphan I- runs.

    NEG_COMMODITY-prefixed labels decode into Entity(polarity="NEG"). A
    polarity flip mid-span (B-COMMODITY followed by I-NEG_COMMODITY, or vice
    versa) closes the open span and opens a new one.
    """
    from ner.constants import ID2LABEL

    spans: list[Entity] = []
    open_type: str | None = None
    open_polarity: str = "POS"
    open_char_start: int | None = None
    open_char_end: int | None = None

    def close() -> None:
        nonlocal open_type, open_polarity, open_char_start, open_char_end
        if open_type is not None and open_char_start is not None and open_char_end is not None:
            text = source_text[open_char_start:open_char_end]
            if text:
                spans.append(Entity(
                    type=open_type, text=text,
                    start=open_char_start, end=open_char_end,
                    polarity=open_polarity,
                ))
        open_type = None
        open_polarity = "POS"
        open_char_start = None
        open_char_end = None

    for i, lid in enumerate(label_ids):
        if lid == LABEL_IGNORE:
            continue
        c0, c1 = offset_mapping[i]
        if c0 == 0 and c1 == 0:
            continue
        label_name = ID2LABEL.get(int(lid), "O")
        decoded = _decode_label(label_name)
        if decoded is None:
            close()
            continue
        prefix, etype, polarity = decoded
        if prefix == "B":
            close()
            open_type = etype
            open_polarity = polarity
            open_char_start = c0
            open_char_end = c1
        elif prefix == "I":
            if open_type == etype and open_polarity == polarity:
                open_char_end = c1
            else:
                if strict:
                    close()
                else:
                    close()
                    open_type = etype
                    open_polarity = polarity
                    open_char_start = c0
                    open_char_end = c1
    close()
    return spans


def softmax(logits: "np.ndarray", axis: int = -1) -> "np.ndarray":
    """Numerically stable softmax over the given axis (numpy only)."""
    import numpy as np

    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def apply_threshold_gate(
    probs: "np.ndarray",
    thresholds: "np.ndarray",
) -> "np.ndarray":
    """Return per-token label ids after applying per-label confidence gates.

    Args:
        probs: shape (n_tokens, n_labels) softmax probabilities.
        thresholds: shape (n_labels,) — minimum probability required for the
            argmax label to survive. Token positions where the argmax is "O"
            (label id 0) are never demoted regardless of threshold.

    Returns:
        int64 array of shape (n_tokens,). A non-O argmax whose softmax
        probability < thresholds[label_id] is demoted to label id 0 ("O").
        Threshold 0.0 on every label is a no-op (preserves argmax).
    """
    import numpy as np

    pred_ids = np.argmax(probs, axis=-1)
    max_probs = np.take_along_axis(probs, pred_ids[..., None], axis=-1).squeeze(-1)
    label_thresholds = thresholds[pred_ids]
    demote = (pred_ids != 0) & (max_probs < label_thresholds)
    pred_ids = np.where(demote, 0, pred_ids)
    return pred_ids.astype(np.int64)
