"""Tests for the benchmark harness, the OOV probe, and the memorization control."""
from __future__ import annotations

import json

import pytest

from ner.commodity.bench import (
    bootstrap_f1_ci,
    error_taxonomy,
    format_table,
    measure_latency,
    run_benchmark,
)
from ner.commodity.eval_sets import (
    ProbeConfig,
    build_oov_probe,
    commodity_only,
    load_jsonl_gold,
    load_vocabulary,
    split_vocabulary,
)
from ner.commodity.gazetteer import GazetteerExtractor
from ner.commodity.protocol import CommodityExtractor, CommodityFilter
from ner.eval.gold import GOLD_SEED
from ner.schema import Entity, Record, SpanError


class OracleExtractor:
    """Returns gold exactly — the F1 == 1.0 sanity anchor."""

    name = "oracle"

    def __init__(self, gold: list[Record]) -> None:
        self._by_text = {r.text: list(r.entities) for r in gold}

    def predict(self, text: str) -> list[Entity]:
        return list(self._by_text.get(text, []))

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        return [self.predict(t) for t in texts]


class SilentExtractor:
    name = "silent"

    def predict(self, text: str) -> list[Entity]:
        return []

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        return [[] for _ in texts]


class BrokenExtractor:
    """Violates the batch contract — the harness must catch it."""

    name = "broken"

    def predict(self, text: str) -> list[Entity]:
        return []

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        return [[]]


@pytest.fixture(scope="module")
def probe() -> list[Record]:
    return build_oov_probe(ProbeConfig(n_records=120, seed=99))


class TestProbe:
    def test_offsets_are_exact(self, probe: list[Record]) -> None:
        for record in probe:
            for entity in record.entities:
                assert record.text[entity.start:entity.end] == entity.text

    def test_every_record_validates(self, probe: list[Record]) -> None:
        for record in probe:
            record.validate()

    def test_deterministic(self) -> None:
        a = build_oov_probe(ProbeConfig(n_records=40, seed=5))
        b = build_oov_probe(ProbeConfig(n_records=40, seed=5))
        assert [r.text for r in a] == [r.text for r in b]

    def test_seed_changes_output(self) -> None:
        a = build_oov_probe(ProbeConfig(n_records=40, seed=5))
        b = build_oov_probe(ProbeConfig(n_records=40, seed=6))
        assert [r.text for r in a] != [r.text for r in b]

    def test_has_both_polarities(self, probe: list[Record]) -> None:
        polarities = {e.polarity for r in probe for e in r.entities}
        assert polarities == {"POS", "NEG"}

    def test_vocabulary_is_disjoint_from_repo_pool(self, probe: list[Record]) -> None:
        """The whole point: a pool-trained system must not recognize this."""
        from scripts.seedgen.commodities import COMMODITIES

        pool = {c.lower() for c in COMMODITIES}
        surfaces = [e.text.lower() for r in probe for e in r.entities]
        overlap = sum(1 for s in surfaces if s in pool) / len(surfaces)
        assert overlap < 0.10, f"probe shares {overlap:.1%} of surfaces with the pool"

    def test_includes_long_descriptive_spans(self, probe: list[Record]) -> None:
        """Span extent was the dominant error mode; the probe must exercise it."""
        lengths = [e.end - e.start for r in probe for e in r.entities]
        assert max(lengths) > 40


class TestVocabulary:
    def test_loads(self) -> None:
        heads, descriptions = load_vocabulary()
        assert len(heads) > 500
        assert len(descriptions) > 500

    def test_split_is_disjoint_and_stable(self) -> None:
        heads, _ = load_vocabulary()
        train, holdout = split_vocabulary(heads, holdout_ratio=0.5, seed=3)
        assert not set(train) & set(holdout)
        assert len(train) + len(holdout) == len(heads)
        assert split_vocabulary(heads, seed=3)[0] == train


class TestJsonlGold:
    def test_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "g.jsonl"
        record = Record("500 MT of cocoa beans", [Entity("COMMODITY", "cocoa beans", 10, 21)])
        path.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
        assert load_jsonl_gold(path)[0].entities[0].text == "cocoa beans"

    def test_rejects_drifted_offsets(self, tmp_path) -> None:
        """`ner.data.assembler.read_jsonl` would accept this silently."""
        path = tmp_path / "bad.jsonl"
        path.write_text(
            json.dumps(
                {
                    "text": "500 MT of cocoa beans",
                    "entities": [
                        {"type": "COMMODITY", "text": "cocoa beans",
                         "start": 0, "end": 11, "polarity": "POS"}
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="bad.jsonl:1"):
            load_jsonl_gold(path)

    def test_skips_blank_lines(self, tmp_path) -> None:
        path = tmp_path / "g.jsonl"
        record = Record("cocoa beans loaded", [Entity("COMMODITY", "cocoa beans", 0, 11)])
        path.write_text("\n" + json.dumps(record.to_dict()) + "\n\n", encoding="utf-8")
        assert len(load_jsonl_gold(path)) == 1


class TestErrorTaxonomy:
    def _record(self) -> Record:
        return Record(
            "cargo of raw cane sugar only",
            [Entity("COMMODITY", "raw cane sugar", 9, 23)],
        )

    def test_perfect_prediction_has_no_errors(self) -> None:
        ref = self._record()
        assert error_taxonomy([list(ref.entities)], [ref]).total == 0

    def test_boundary_error(self) -> None:
        ref = self._record()
        pred = [Entity("COMMODITY", "cane sugar", 13, 23)]
        tax = error_taxonomy([pred], [ref])
        assert (tax.boundary, tax.spurious, tax.missed) == (1, 0, 0)

    def test_spurious_and_missed(self) -> None:
        ref = self._record()
        pred = [Entity("COMMODITY", "cargo", 0, 5)]
        tax = error_taxonomy([pred], [ref])
        assert (tax.spurious, tax.missed, tax.boundary) == (1, 1, 0)

    def test_polarity_error_is_not_a_boundary_error(self) -> None:
        ref = self._record()
        pred = [Entity("COMMODITY", "raw cane sugar", 9, 23, polarity="NEG")]
        tax = error_taxonomy([pred], [ref])
        assert (tax.polarity, tax.boundary, tax.spurious) == (1, 0, 0)


class TestBenchmark:
    def test_oracle_scores_one(self, probe: list[Record]) -> None:
        result = run_benchmark(OracleExtractor(probe), probe, with_latency=False,
                              bootstrap_resamples=50)
        assert result.f1 == 1.0
        assert result.taxonomy.total == 0

    def test_silent_extractor_scores_zero(self, probe: list[Record]) -> None:
        result = run_benchmark(SilentExtractor(), probe, with_latency=False,
                              bootstrap_resamples=50)
        assert result.f1 == 0.0
        assert result.taxonomy.missed == sum(len(r.entities) for r in probe)

    def test_batch_contract_violation_raises(self, probe: list[Record]) -> None:
        with pytest.raises(ValueError, match="predict_batch returned"):
            run_benchmark(BrokenExtractor(), probe, with_latency=False, with_ci=False)

    def test_ci_brackets_the_estimate(self, probe: list[Record]) -> None:
        gaz = GazetteerExtractor(["cocoa beans", "selenium"], name="tiny")
        preds = gaz.predict_batch([r.text for r in probe])
        low, high = bootstrap_f1_ci(preds, probe, resamples=200)
        assert 0.0 <= low <= high <= 1.0

    def test_latency_is_measured(self, probe: list[Record]) -> None:
        stats = measure_latency(SilentExtractor(), [r.text for r in probe[:20]])
        assert stats.n_calls == 20
        assert stats.median_ms >= 0.0
        assert stats.p95_ms >= stats.median_ms

    def test_table_renders(self, probe: list[Record]) -> None:
        result = run_benchmark(SilentExtractor(), probe, with_latency=False,
                              bootstrap_resamples=50)
        table = format_table([result], budget_ms=100.0)
        assert "silent" in table and "error taxonomy" in table


class TestMemorizationControl:
    """Pins the property that motivated this whole package."""

    def test_pool_gazetteer_collapses_on_open_vocabulary(self, probe: list[Record]) -> None:
        from scripts.seedgen.commodities import COMMODITIES

        gaz = GazetteerExtractor(COMMODITIES, name="pool")
        on_pool_gold = run_benchmark(
            gaz, commodity_only(list(GOLD_SEED)), with_latency=False, with_ci=False
        ).f1
        on_open_vocab = run_benchmark(
            gaz, probe, with_latency=False, with_ci=False
        ).f1

        assert on_pool_gold > 0.7, "control should look strong on its own vocabulary"
        assert on_open_vocab < 0.15, "control must collapse on unseen vocabulary"
        assert on_pool_gold - on_open_vocab > 0.5


class TestGazetteer:
    def test_longest_match_wins(self) -> None:
        gaz = GazetteerExtractor(["sugar", "raw cane sugar"], name="g")
        [span] = gaz.predict("cargo of raw cane sugar")
        assert span.text == "raw cane sugar"

    def test_respects_word_boundaries(self) -> None:
        gaz = GazetteerExtractor(["tin"], name="g")
        assert gaz.predict("non-stick coatings") == []

    def test_offsets_are_exact(self) -> None:
        gaz = GazetteerExtractor(["cocoa beans"], name="g")
        text = "500 MT of cocoa beans loaded"
        [span] = gaz.predict(text)
        assert text[span.start:span.end] == span.text

    def test_applies_polarity(self) -> None:
        gaz = GazetteerExtractor(["cocoa beans"], name="g")
        [span] = gaz.predict("shipment contains no cocoa beans")
        assert span.polarity == "NEG"

    def test_empty_vocabulary_rejected(self) -> None:
        with pytest.raises(ValueError):
            GazetteerExtractor([])

    def test_empty_text(self) -> None:
        assert GazetteerExtractor(["tin"], name="g").predict("") == []


class TestProtocol:
    def test_gazetteer_satisfies_protocol(self) -> None:
        assert isinstance(GazetteerExtractor(["tin"], name="g"), CommodityExtractor)

    def test_filter_drops_other_types(self) -> None:
        class MultiType:
            name = "multi"

            def predict(self, text: str) -> list[Entity]:
                return [
                    Entity("COMMODITY", "tin", 0, 3),
                    Entity("ORG", "Acme", 4, 8),
                ]

            def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
                return [self.predict(t) for t in texts]

        filtered = CommodityFilter(MultiType())
        assert [e.type for e in filtered.predict("tin Acme")] == ["COMMODITY"]
        assert filtered.predict_batch(["tin Acme"])[0][0].type == "COMMODITY"


def test_entity_rejects_invalid_spans() -> None:
    with pytest.raises(SpanError):
        Entity("COMMODITY", "x", 5, 5)
