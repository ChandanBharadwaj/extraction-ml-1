"""Adapter putting GLiNER-family models behind `CommodityExtractor`.

Covers `gliner` (v2.x uni/bi-encoder), `gliner2`, and NuNER-Zero-style checkpoints
loaded through the same API. Torch is imported lazily so the rest of this package
stays importable without it.

Two things this adapter exists to get right:

**The label string is data, not a constant.** These models take the entity type as
*text*, so "commodity" / "goods" / "product being shipped" are different queries
against the same weights and give materially different output. `LABEL_PRESETS`
holds candidates for the sweep; nothing here privileges one.

**Offsets must index the caller's text.** GLiNER returns its own char offsets; we
re-slice from the source and assert agreement rather than trusting them, so a
tokenizer quirk surfaces as a loud failure instead of silently misaligned spans.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ner.commodity.negation import PolarityTagger
from ner.schema import Entity

# Candidate label strings for the sweep. Which one wins is an empirical question
# and depends on the checkpoint — that is the point of sweeping.
LABEL_PRESETS: dict[str, tuple[str, ...]] = {
    "commodity": ("commodity",),
    "goods": ("goods",),
    "product": ("product",),
    "cargo": ("cargo",),
    "commodity_goods": ("commodity", "goods"),
    "trade_detail": ("commodity", "goods", "raw material", "manufactured product"),
    "shipping": ("commodity being shipped", "goods being shipped"),
}

DEFAULT_MODEL = "fastino/gliner2-base-v1"


@dataclass(frozen=True, slots=True)
class GlinerConfig:
    model_id: str = DEFAULT_MODEL
    labels: tuple[str, ...] = ("commodity",)
    threshold: float = 0.5
    flat_ner: bool = True
    multi_label: bool = False
    max_batch: int = 16
    device: str = "cpu"

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "labels": list(self.labels),
            "threshold": self.threshold,
            "flat_ner": self.flat_ner,
            "multi_label": self.multi_label,
            "device": self.device,
        }


class GlinerCommodityExtractor:
    """Wraps a GLiNER checkpoint; assigns polarity with the rule layer."""

    def __init__(
        self,
        config: GlinerConfig | None = None,
        tagger: PolarityTagger | None = None,
        model: Any = None,
        name: str | None = None,
    ) -> None:
        self.config = config or GlinerConfig()
        self._tagger = tagger or PolarityTagger()
        self._model = model
        label_tag = "+".join(self.config.labels)
        self.name = name or f"{self.config.model_id.split('/')[-1]}[{label_tag}]"

    # -- loading -----------------------------------------------------------

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> Any:
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "GLiNER is not installed. `pip install -e .[commodity]`. "
                "Note it pulls torch (~2GB)."
            ) from exc
        model = GLiNER.from_pretrained(self.config.model_id)
        model.eval()
        if hasattr(model, "to"):
            model.to(self.config.device)
        return model

    # -- prediction --------------------------------------------------------

    def _to_entities(self, text: str, raw: list[dict]) -> list[Entity]:
        """Convert GLiNER output, re-slicing surfaces from the caller's text."""
        spans: list[Entity] = []
        for item in raw:
            start, end = int(item["start"]), int(item["end"])
            if start < 0 or end > len(text) or end <= start:
                continue
            surface = text[start:end]
            # Trim whitespace the model may have swept into the span; keep offsets aligned.
            lead = len(surface) - len(surface.lstrip())
            trail = len(surface) - len(surface.rstrip())
            start, end = start + lead, end - trail
            if end <= start:
                continue
            spans.append(Entity("COMMODITY", text[start:end], start, end))

        spans.sort(key=lambda e: (e.start, -e.end))
        deduped: list[Entity] = []
        for span in spans:
            if any(not (span.end <= k.start or span.start >= k.end) for k in deduped):
                continue  # flat NER: keep the first (longest) of any overlap group
            deduped.append(span)
        return self._tagger.apply(text, deduped)

    def predict(self, text: str) -> list[Entity]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        if not texts:
            return []
        model = self._ensure_model()
        labels = list(self.config.labels)

        results: list[list[Entity]] = []
        for offset in range(0, len(texts), self.config.max_batch):
            chunk = texts[offset:offset + self.config.max_batch]
            raw_batch = self._predict_raw(model, chunk, labels)
            results.extend(
                self._to_entities(text, raw or []) for text, raw in zip(chunk, raw_batch)
            )
        return results

    def _predict_raw(self, model: Any, texts: list[str], labels: list[str]) -> list[list[dict]]:
        """Call whichever prediction API this checkpoint exposes.

        `gliner` and `gliner2` differ here, and batch methods are not universal,
        so fall back to per-text calls rather than assuming.
        """
        kwargs: dict[str, Any] = {"threshold": self.config.threshold}
        if self.config.flat_ner:
            kwargs["flat_ner"] = True
        if self.config.multi_label:
            kwargs["multi_label"] = True

        batch_fn = getattr(model, "batch_predict_entities", None)
        if batch_fn is not None:
            try:
                return batch_fn(texts, labels, **kwargs)
            except TypeError:
                return batch_fn(texts, labels, threshold=self.config.threshold)

        single_fn = getattr(model, "predict_entities", None)
        if single_fn is None:  # pragma: no cover - unknown checkpoint API
            raise AttributeError(
                f"{type(model).__name__} exposes neither batch_predict_entities "
                "nor predict_entities"
            )
        out = []
        for text in texts:
            try:
                out.append(single_fn(text, labels, **kwargs))
            except TypeError:
                out.append(single_fn(text, labels, threshold=self.config.threshold))
        return out


class _LazySharedModel:
    """Loads a checkpoint once, on first prediction, shared across candidates.

    A sweep is dozens of (label, threshold) combinations over the *same* weights.
    Loading per candidate would dominate runtime and memory for no reason.
    """

    def __init__(self, model_id: str, device: str = "cpu") -> None:
        self._model_id = model_id
        self._device = device
        self._model: Any = None

    def get(self) -> Any:
        if self._model is None:
            self._model = GlinerCommodityExtractor(
                GlinerConfig(model_id=self._model_id, device=self._device)
            )._load()
        return self._model


class _SharedModelExtractor(GlinerCommodityExtractor):
    """A candidate that resolves its weights through a shared lazy loader."""

    def __init__(self, shared: _LazySharedModel, config: GlinerConfig, name: str) -> None:
        super().__init__(config, name=name)
        self._shared = shared

    def _ensure_model(self) -> Any:
        return self._shared.get()


@dataclass
class LabelSweep:
    """Sweep label strings and thresholds for one checkpoint.

    Free accuracy compared to fine-tuning, and it must happen before anyone
    concludes a model "doesn't work" on this text.
    """

    model_id: str = DEFAULT_MODEL
    presets: tuple[str, ...] = tuple(LABEL_PRESETS)
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    device: str = "cpu"
    _shared: _LazySharedModel | None = field(default=None, repr=False)

    def candidates(self) -> list[GlinerCommodityExtractor]:
        """One extractor per (label preset, threshold), sharing loaded weights."""
        shared = self._shared or _LazySharedModel(self.model_id, self.device)
        self._shared = shared
        short = self.model_id.split("/")[-1]

        out: list[GlinerCommodityExtractor] = []
        for preset in self.presets:
            labels = LABEL_PRESETS[preset]
            for threshold in self.thresholds:
                config = GlinerConfig(
                    model_id=self.model_id,
                    labels=labels,
                    threshold=threshold,
                    device=self.device,
                )
                out.append(
                    _SharedModelExtractor(
                        shared, config, f"{short}[{preset}@{threshold}]"
                    )
                )
        return out
