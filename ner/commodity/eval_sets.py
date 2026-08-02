"""Evaluation sets for open-vocabulary commodity extraction.

Two sets, with very different standing.

**The OOV probe** (built here). Commodity surfaces come from the HS nomenclature —
1,596 head forms and 2,575 long descriptions, 3.8% overlapping the repo's own
pool. Because the vocabulary is disjoint from anything in this repo, a model that
merely memorized the pool scores near zero, which is exactly the failure mode
worth catching early. Framing is synthetic: document skeletons are hand-written
here, so the probe measures *vocabulary generalization and span extent*, not real
document messiness. Treat its numbers as directional.

**Your held-out set** (loaded here, built by you). Real records, hand-corrected.
This is the only number worth a go/no-go. `load_jsonl_gold` validates every
record on load — unlike `ner.data.assembler.read_jsonl`, which skips validation
and will happily score against drifted offsets.

Offsets are computed in Python from the output buffer position as each field is
appended, never recovered by searching afterwards, so
`text[start:end] == entity.text` holds by construction.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from ner.schema import Entity, Record

RESOURCES = Path(__file__).resolve().parent / "resources"
HEADS_FILE = RESOURCES / "hs_heads.txt"
DESCRIPTIONS_FILE = RESOURCES / "hs_descriptions.txt"

# Document skeletons. `{c}` marks a commodity slot; everything else is framing.
# Deliberately vocabulary-free — no commodity words appear in any frame.
POS_FRAMES: tuple[str, ...] = (
    "Invoice {ref}: {qty} of {c} for {org}.",
    "Manifest {ref}: {qty} {c}; {qty} {c2}.",
    "STC {qty} {c}",
    "B/L {ref} | DESCRIPTION OF GOODS: {c} | NET {wt} KG",
    "Packing list line 1: {c} ({qty})",
    "Please confirm the booking for {qty} of {c}, ETA next Tuesday.",
    "{org} confirms shipment of {c} ex {port}.",
    "HS {hs} {c} | GROSS {wt} KG",
    "CARGO: {c} AND {c2} LOADED AT {port}",
    "Line item {n}: {c} — {qty} — {cur} {amt}",
    "we are shipping {c} this week, {qty} total",
    "DECLARED CARGO {c} SEAL {ref}",
)

NEG_FRAMES: tuple[str, ...] = (
    "{org} certifies this shipment contains no {c}.",
    "Manifest declares no {c} on board.",
    "NIL {c} DECLARED.",
    "This consignment is free of {c}.",
    "Carrier excludes {c} from the booking.",
    "B/L REMARK: NO {c} — CARGO IS {c2} IN BULK",
    "The cargo does not contain {c}, only {c2}.",
    "shipment lacks {c}; {c2} loaded instead",
    "PROHIBITED: {c}. ACCEPTED: {c2}.",
    "{org} confirms {c2} but no {c}.",
)

_ORGS = (
    "Nordwind Logistics GmbH", "Acme Trading Co.", "Oceanic Freight",
    "Delta Packaging", "Iberia Shipping S.L.", "Kaito Lines K.K.",
)
_PORTS = ("Rotterdam", "Jebel Ali", "Santos", "Busan", "Felixstowe", "Durban")
_CURRENCIES = ("USD", "EUR", "GBP", "JPY")
_QTY_UNITS = ("MT", "KG", "pallets", "cartons", "drums", "bags", "TEU", "crates")


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    """Knobs for probe generation."""

    n_records: int = 400
    seed: int = 20240802
    neg_ratio: float = 0.35
    long_description_ratio: float = 0.35
    upper_ratio: float = 0.2


def load_vocabulary() -> tuple[list[str], list[str]]:
    """Load (heads, descriptions). Regenerate with `scripts.build_hs_vocab`."""
    missing = [p for p in (HEADS_FILE, DESCRIPTIONS_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing HS vocabulary {[str(p) for p in missing]}; "
            "run `python -m scripts.build_hs_vocab`"
        )

    def _read(path: Path) -> list[str]:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]

    return _read(HEADS_FILE), _read(DESCRIPTIONS_FILE)


def split_vocabulary(
    values: list[str], holdout_ratio: float = 0.5, seed: int = 7
) -> tuple[list[str], list[str]]:
    """Partition a vocabulary so training and evaluation share no surface forms.

    Needed if a fine-tune ever happens: without a vocabulary-disjoint split, the
    eval measures memorization again, just with a bigger list.
    """
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - holdout_ratio))
    return sorted(shuffled[:cut]), sorted(shuffled[cut:])


class _Builder:
    """Assembles one record, tracking offsets as the buffer grows."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.pos = 0
        self.entities: list[Entity] = []

    def add(self, text: str) -> None:
        self.parts.append(text)
        self.pos += len(text)

    def add_commodity(self, surface: str, polarity: str) -> None:
        start = self.pos
        self.add(surface)
        self.entities.append(
            Entity("COMMODITY", surface, start, self.pos, polarity=polarity)
        )

    def build(self, meta: dict) -> Record:
        record = Record("".join(self.parts), self.entities, meta)
        record.validate()
        return record


_SLOT_RE = re.compile(r"\{(\w+)\}")


def _render(
    frame: str, rng: random.Random, commodities: list[str], polarity: str
) -> Record:
    builder = _Builder()
    cursor = 0
    commodity_slots = 0

    for match in _SLOT_RE.finditer(frame):
        builder.add(frame[cursor:match.start()])
        cursor = match.end()
        slot = match.group(1)

        if slot in {"c", "c2"}:
            surface = commodities[commodity_slots % len(commodities)]
            commodity_slots += 1
            # In contrast frames the second commodity is the asserted one.
            slot_polarity = polarity if slot == "c" else "POS"
            builder.add_commodity(surface, slot_polarity)
        elif slot == "org":
            builder.add(rng.choice(_ORGS))
        elif slot == "port":
            builder.add(rng.choice(_PORTS))
        elif slot == "qty":
            builder.add(f"{rng.randrange(1, 999)} {rng.choice(_QTY_UNITS)}")
        elif slot == "wt":
            builder.add(f"{rng.randrange(100, 99999):,}")
        elif slot == "ref":
            builder.add(f"{rng.choice('ABCDEFGH')}{rng.randrange(10000, 99999)}")
        elif slot == "hs":
            builder.add(f"{rng.randrange(1000, 9999)}.{rng.randrange(10, 99)}")
        elif slot == "cur":
            builder.add(rng.choice(_CURRENCIES))
        elif slot == "amt":
            builder.add(f"{rng.randrange(100, 99999):,}")
        elif slot == "n":
            builder.add(str(rng.randrange(1, 40)))
        else:  # pragma: no cover - guards frame typos
            raise ValueError(f"unknown slot {slot!r} in frame {frame!r}")

    builder.add(frame[cursor:])
    return builder.build({"frame": frame, "source": "hs_oov_probe"})


def _upper(record: Record) -> Record:
    """Uppercase a record. Length-preserving for ASCII, so offsets survive."""
    text = record.text.upper()
    if len(text) != len(record.text):
        return record
    return Record(
        text,
        [
            Entity(e.type, text[e.start:e.end], e.start, e.end, polarity=e.polarity)
            for e in record.entities
        ],
        dict(record.meta, cased="upper"),
    )


def build_oov_probe(
    config: ProbeConfig | None = None, vocabulary: list[str] | None = None
) -> list[Record]:
    """Generate the open-vocabulary probe.

    Every commodity surface is drawn from HS, so a model that learned the repo's
    589-string pool has seen ~4% of this vocabulary. Records mix short head forms
    with long HS descriptions, since span *extent* on descriptive text was the
    dominant error mode in earlier measurement.
    """
    config = config or ProbeConfig()
    rng = random.Random(config.seed)

    if vocabulary is None:
        heads, descriptions = load_vocabulary()
    else:
        heads, descriptions = vocabulary, vocabulary

    records: list[Record] = []
    for _ in range(config.n_records):
        negated = rng.random() < config.neg_ratio
        frames = NEG_FRAMES if negated else POS_FRAMES
        frame = rng.choice(frames)

        pool = descriptions if rng.random() < config.long_description_ratio else heads
        picks = rng.sample(pool, k=min(2, len(pool)))

        record = _render(frame, rng, picks, "NEG" if negated else "POS")
        if rng.random() < config.upper_ratio:
            record = _upper(record)
        records.append(record)

    return records


def load_jsonl_gold(path: str | Path) -> list[Record]:
    """Load hand-labeled records, validating every one.

    Raises `SpanError` on the first record whose offsets do not match its text —
    the failure mode that otherwise scores silently against garbage.
    """
    records: list[Record] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = Record.from_dict(json.loads(line))
                record.validate()
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            records.append(record)
    return records


def commodity_only(records: list[Record]) -> list[Record]:
    """Project records onto COMMODITY entities, leaving text and meta intact."""
    return [
        Record(r.text, [e for e in r.entities if e.type == "COMMODITY"], dict(r.meta))
        for r in records
    ]
