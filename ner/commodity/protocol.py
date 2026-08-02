"""The interface every commodity extractor must satisfy to be benchmarked.

Deliberately the same shape as `ner.infer.runtime.NERRuntime.predict` /
`.predict_batch`, so the repo's own model, a GLiNER wrapper, and a trivial
baseline are all drop-in substitutes for one another.

Contract (mirrors `NERRuntime`):

* `predict` returns entities whose offsets index into **the caller's text**, i.e.
  `text[e.start:e.end] == e.text` for every returned entity.
* `predict_batch` preserves order and length; a record with no commodities yields
  an empty list, never a dropped row.
* Neither raises on empty or whitespace-only input.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ner.schema import Entity

COMMODITY = "COMMODITY"


@runtime_checkable
class CommodityExtractor(Protocol):
    """Structural type for anything that can extract commodity spans."""

    name: str

    def predict(self, text: str) -> list[Entity]:
        ...

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        ...


class CommodityFilter:
    """Wrap a multi-type extractor and keep only its COMMODITY spans.

    Lets a 4-type model (the repo's existing `NERRuntime`) be scored on the same
    footing as a commodity-only one without modifying it.
    """

    def __init__(self, inner: CommodityExtractor, name: str | None = None) -> None:
        self._inner = inner
        self.name = name or f"{getattr(inner, 'name', type(inner).__name__)}[COMMODITY]"

    def predict(self, text: str) -> list[Entity]:
        return [e for e in self._inner.predict(text) if e.type == COMMODITY]

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        return [
            [e for e in ents if e.type == COMMODITY]
            for ents in self._inner.predict_batch(texts)
        ]


def default_predict_batch(
    extractor: CommodityExtractor, texts: list[str]
) -> list[list[Entity]]:
    """Fallback `predict_batch` for extractors that only implement `predict`."""
    return [extractor.predict(t) for t in texts]
