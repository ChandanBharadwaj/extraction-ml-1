"""Tests for the vocabulary-independent polarity layer.

The important tests here are `TestOpenVocabularyGeneralization`: every commodity
in them is deliberately absent from the repo's slot-fill pool. The polarity rule
must work on words it has never seen, because that is the whole premise —
negation cues are a closed class even when commodities are an open one.
"""
from __future__ import annotations

import pytest

from ner.commodity.negation import (
    FROZEN_COMPOUNDS,
    NEG_CUES,
    PolarityConfig,
    PolarityTagger,
)
from ner.eval.gold import GOLD_SEED
from ner.eval.metrics import evaluate
from ner.schema import Entity, Record


@pytest.fixture(scope="module")
def tagger() -> PolarityTagger:
    return PolarityTagger()


def _tag(tagger: PolarityTagger, text: str, surface: str) -> str:
    start = text.index(surface)
    ent = Entity("COMMODITY", surface, start, start + len(surface))
    return tagger.apply(text, [ent])[0].polarity


class TestGoldRegression:
    """The 72-record gold set is a regression fixture, not a benchmark.

    It shares vocabulary with the repo's pool, so it cannot measure open-vocabulary
    span extraction. It *can* pin polarity behavior, which is vocabulary-independent.
    """

    def test_perfect_spans_yield_perfect_polarity(self, tagger: PolarityTagger) -> None:
        gold = [
            Record(r.text, [e for e in r.entities if e.type == "COMMODITY"], {})
            for r in GOLD_SEED
        ]
        preds = [tagger.apply(r.text, list(r.entities)) for r in gold]
        assert evaluate(preds, gold).micro.f1 == 1.0

    @pytest.mark.parametrize("window", [40, 50, 60, 80])
    def test_not_knife_edge_on_window(self, window: int) -> None:
        """Result must not depend on an exact window size."""
        tagger = PolarityTagger(PolarityConfig(window_chars=window))
        gold = [
            Record(r.text, [e for e in r.entities if e.type == "COMMODITY"], {})
            for r in GOLD_SEED
        ]
        preds = [tagger.apply(r.text, list(r.entities)) for r in gold]
        assert evaluate(preds, gold).micro.f1 == 1.0


class TestOpenVocabularyGeneralization:
    """Commodities here appear nowhere in the repo's pool."""

    @pytest.mark.parametrize(
        "text,surface",
        [
            ("shipment contains no turbo-propellers", "turbo-propellers"),
            ("consignment is free of selenium", "selenium"),
            ("NIL PHOTOGRAPHIC FILM DECLARED", "PHOTOGRAPHIC FILM"),
            ("does not contain sewing machine needles", "sewing machine needles"),
            ("carrier excludes bulldozers from this booking", "bulldozers"),
            ("manifest declares no hearing aids on board", "hearing aids"),
            ("without diphosphorus pentoxide", "diphosphorus pentoxide"),
            ("cargo lacks umbrellas", "umbrellas"),
            ("prohibited: helium-3", "helium-3"),
            ("front-end shovel loaders not available", "front-end shovel loaders"),
        ],
    )
    def test_unseen_commodities_are_negated(
        self, tagger: PolarityTagger, text: str, surface: str
    ) -> None:
        assert _tag(tagger, text, surface) == "NEG"

    @pytest.mark.parametrize(
        "text,surface",
        [
            ("shipment of welding machines to Rotterdam", "welding machines"),
            ("STC 40 CRATES TRICYCLES", "TRICYCLES"),
            ("invoice covers transformers and switchgear", "transformers"),
            ("no bulldozers, only front-end shovel loaders", "front-end shovel loaders"),
            ("NO SELENIUM ON BOARD, PHOTOGRAPHIC FILM ONLY", "PHOTOGRAPHIC FILM"),
            ("contains no umbrellas; tricycles loaded in hold 2", "tricycles"),
        ],
    )
    def test_unseen_commodities_stay_positive(
        self, tagger: PolarityTagger, text: str, surface: str
    ) -> None:
        assert _tag(tagger, text, surface) == "POS"


class TestScopeTermination:
    def test_contrast_terminates_scope(self, tagger: PolarityTagger) -> None:
        text = "no special wood, but ordinary wood is acceptable"
        assert _tag(tagger, text, "special wood") == "NEG"
        assert _tag(tagger, text, "ordinary wood") == "POS"

    def test_clause_break_terminates_scope(self, tagger: PolarityTagger) -> None:
        assert _tag(tagger, "MARKS: NIL. CARGO: ARABICA COFFEE", "ARABICA COFFEE") == "POS"

    def test_dash_terminates_scope(self, tagger: PolarityTagger) -> None:
        text = "NO SCRAP METAL — CARGO IS CEMENT CLINKER"
        assert _tag(tagger, text, "CEMENT CLINKER") == "POS"

    def test_exception_clause_asserts_its_target(self, tagger: PolarityTagger) -> None:
        text = "No cargo other than skimmed milk powder is stowed"
        assert _tag(tagger, text, "skimmed milk powder") == "POS"

    def test_leading_only_does_not_rescue_the_denied_span(
        self, tagger: PolarityTagger
    ) -> None:
        """`, only` introduces the next span; it must not assert the previous one."""
        text = "does not contain wood, only plastic toys"
        assert _tag(tagger, text, "wood") == "NEG"
        assert _tag(tagger, text, "plastic toys") == "POS"


class TestHardNegatives:
    @pytest.mark.parametrize("compound", FROZEN_COMPOUNDS)
    def test_frozen_compounds_never_negate(
        self, tagger: PolarityTagger, compound: str
    ) -> None:
        """"sugar-free chocolate" asserts chocolate; it does not deny it."""
        text = f"shipment of {compound} widgets to the port"
        assert _tag(tagger, text, "widgets") == "POS"

    def test_cue_inside_span_is_ignored(self, tagger: PolarityTagger) -> None:
        """HS headings embed a literal NOT inside the commodity description."""
        text = "HS 0901.21 ROASTED COFFEE, NOT DECAFFEINATED | NET 12,000 KG"
        assert _tag(tagger, text, "ROASTED COFFEE, NOT DECAFFEINATED") == "POS"

    def test_distant_cue_is_out_of_scope(self, tagger: PolarityTagger) -> None:
        text = "no asbestos " + "x" * 80 + " selenium"
        assert _tag(tagger, text, "selenium") == "POS"


class TestApplySemantics:
    def test_non_commodity_types_are_untouched(self, tagger: PolarityTagger) -> None:
        text = "no goods from Acme Trading Co."
        ent = Entity("ORG", "Acme Trading Co.", text.index("Acme"), len(text) - 1)
        assert tagger.apply(text, [ent])[0].polarity == "POS"

    def test_offsets_and_surfaces_survive(self, tagger: PolarityTagger) -> None:
        text = "consignment contains no turbo-propellers at all"
        start = text.index("turbo-propellers")
        ent = Entity("COMMODITY", "turbo-propellers", start, start + len("turbo-propellers"))
        out = tagger.apply(text, [ent])[0]
        assert out.polarity == "NEG"
        assert text[out.start:out.end] == out.text

    def test_empty_input(self, tagger: PolarityTagger) -> None:
        assert tagger.apply("", []) == []


def test_cue_lexicons_are_deduplicated() -> None:
    for lexicon in (NEG_CUES, FROZEN_COMPOUNDS):
        lowered = [x.lower() for x in lexicon]
        assert len(lowered) == len(set(lowered))
