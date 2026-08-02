"""Dictionary matcher — included as a **memorization control**, not a candidate.

A gazetteer can only ever return commodities someone already wrote down, so it
cannot satisfy "extract all commodity goods from any text". It earns its place in
the benchmark for a different reason: run it with the repo's own 589-string pool
and it scores well on data built from that pool and near-zero on the HS probe.
That gap *is* the measurement of how much a score depends on shared vocabulary,
and it is the control that catches a model that has merely memorized its training
list.

Measured for reference: with the repo pool it reached 0.716 F1 on the repo's gold
set (78% of whose spans are in that pool) — and adding 7,710 HS terms to the same
matcher *lowered* it to 0.650, because dictionary recall buys precision collapse.

Takes its vocabulary as an argument; it hardcodes no commodity strings.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from ner.commodity.negation import PolarityTagger
from ner.schema import Entity

MIN_TERM_CHARS = 3


class GazetteerExtractor:
    """Longest-match, non-overlapping, word-boundary-aligned dictionary matcher."""

    def __init__(
        self,
        terms: Iterable[str],
        name: str = "gazetteer",
        tagger: PolarityTagger | None = None,
        min_chars: int = MIN_TERM_CHARS,
    ) -> None:
        cleaned = sorted(
            {t.strip().lower() for t in terms if len(t.strip()) >= min_chars},
            key=len,
            reverse=True,  # longest-first alternation => longest match wins
        )
        if not cleaned:
            raise ValueError("gazetteer needs at least one term")
        self._pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(t) for t in cleaned) + r")(?!\w)",
            re.IGNORECASE,
        )
        self.name = name
        self.n_terms = len(cleaned)
        self._tagger = tagger or PolarityTagger()

    def predict(self, text: str) -> list[Entity]:
        if not text:
            return []
        spans: list[Entity] = []
        taken: list[tuple[int, int]] = []
        for match in self._pattern.finditer(text):
            start, end = match.start(), match.end()
            if any(not (end <= a or start >= b) for a, b in taken):
                continue
            taken.append((start, end))
            spans.append(Entity("COMMODITY", text[start:end], start, end))
        return self._tagger.apply(text, spans)

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        return [self.predict(t) for t in texts]
