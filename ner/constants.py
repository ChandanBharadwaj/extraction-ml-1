"""Label space and global constants.

The BIO label list is the contract between training, ONNX export, and serving.
Order is fixed and must never change without re-exporting the model.
"""
from __future__ import annotations

ENTITY_TYPES: tuple[str, ...] = ("PERSON", "ORG", "ADDRESS", "COMMODITY")

# Order matters: index = class id. "O" must be 0.
# COMMODITY carries a polarity (POS/NEG); only its labels are duplicated for
# the negation case. Append-only — never reorder existing entries.
LABEL_LIST: tuple[str, ...] = (
    "O",
    "B-PERSON", "I-PERSON",
    "B-ORG", "I-ORG",
    "B-ADDRESS", "I-ADDRESS",
    "B-COMMODITY", "I-COMMODITY",
    "B-NEG_COMMODITY", "I-NEG_COMMODITY",
)

# Types that can be negated. Used by Entity validation; non-COMMODITY entities
# are always POS by domain convention.
NEGATABLE_TYPES: frozenset[str] = frozenset({"COMMODITY"})
POLARITIES: frozenset[str] = frozenset({"POS", "NEG"})

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(LABEL_LIST)}
NUM_LABELS: int = len(LABEL_LIST)

BASE_MODEL: str = "microsoft/deberta-v3-base"

MAX_INPUT_CHARS: int = 500
MAX_SEQ_LEN: int = 256  # token budget; 500 chars comfortably fits

LATENCY_SLA_MS: int = 1_000
