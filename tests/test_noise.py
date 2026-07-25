"""Noise transforms must preserve the invariant `text[start:end] == ent.text`
for every surviving entity. We brute-force this across many random seeds.
"""
from __future__ import annotations

import random

from ner.data.noise import NoiseConfig, apply_noise
from ner.schema import Entity, Record


def _record() -> Record:
    text = "Maria Gonzalez from Acme Trading Co. ordered refined copper cathode."
    ents = [
        Entity("PERSON", "Maria Gonzalez", 0, 14),
        Entity("ORG", "Acme Trading Co.", 20, 36),
        Entity("COMMODITY", "refined copper cathode", 45, 67),
    ]
    rec = Record(text=text, entities=ents)
    rec.validate()
    return rec


def test_noise_preserves_entity_substring_invariant():
    cfg = NoiseConfig(
        p_lowercase=1.0, p_drop_punct=1.0, drop_punct_rate=0.5,
        p_typo=1.0, typo_rate=0.05, p_truncate=0.0,
    )
    for seed in range(50):
        rec = _record()
        out = apply_noise(rec, cfg, random.Random(seed))
        out.validate()
        for ent in out.entities:
            assert out.text[ent.start:ent.end] == ent.text


def test_lowercase_offsets_unchanged():
    cfg = NoiseConfig(
        p_lowercase=1.0, p_drop_punct=0.0, p_typo=0.0, p_truncate=0.0,
    )
    rec = _record()
    out = apply_noise(rec, cfg, random.Random(0))
    assert out.text == rec.text.lower()
    for orig, noisy in zip(rec.entities, out.entities):
        assert (orig.start, orig.end) == (noisy.start, noisy.end)


def test_dropping_punctuation_between_adjacent_entities_does_not_merge_them():
    """Regression: previously, dropping the comma at `ent.end` made the
    decoder fall back to len(new_text), swallowing every following entity."""
    text = "ship A, B, C done"
    rec = Record(
        text=text,
        entities=[
            Entity("COMMODITY", "A", 5, 6),
            Entity("COMMODITY", "B", 8, 9),
            Entity("COMMODITY", "C", 11, 12),
        ],
    )
    rec.validate()
    cfg = NoiseConfig(
        p_lowercase=0.0, p_uppercase=0.0, p_ocr=0.0,
        p_drop_punct=1.0, drop_punct_rate=1.0,
        p_typo=0.0, p_truncate=0.0,
        preserve_entity_surface=True,
    )
    out = apply_noise(rec, cfg, random.Random(0))
    out.validate()
    assert [e.text for e in out.entities] == ["A", "B", "C"]


def test_preserve_entity_surface_blocks_typos_inside_entities():
    cfg = NoiseConfig(
        p_lowercase=0.0, p_uppercase=0.0, p_ocr=0.0, p_drop_punct=0.0,
        p_typo=1.0, typo_rate=1.0,  # every non-entity letter becomes random
        p_truncate=0.0, preserve_entity_surface=True,
    )
    rec = _record()
    out = apply_noise(rec, cfg, random.Random(1))
    # Entity surfaces unchanged.
    for orig, noisy in zip(rec.entities, out.entities):
        assert orig.text == noisy.text


def _neg_record() -> Record:
    """A record with a negation cue + a NEG_COMMODITY entity, mimicking
    what the slot-fill assembler produces."""
    text = "Acme does not contain wood, only steel sheet."
    # Negation cue "does not contain" starts at idx 5 (after "Acme ").
    cue_start = text.index("does not contain")
    cue_end = cue_start + len("does not contain")
    contrast_start = text.index(", only")
    contrast_end = contrast_start + len(", only")
    rec = Record(
        text=text,
        entities=[
            Entity("ORG", "Acme", 0, 4, polarity="POS"),
            Entity("COMMODITY", "wood", 22, 26, polarity="NEG"),
            Entity("COMMODITY", "steel sheet", 33, 44, polarity="POS"),
        ],
        meta={"preserve_spans": [(cue_start, cue_end), (contrast_start, contrast_end)]},
    )
    rec.validate()
    return rec


def test_noise_never_deletes_chars_inside_preserve_spans():
    cfg = NoiseConfig(
        p_lowercase=0.0, p_uppercase=0.0, p_ocr=0.0,
        p_drop_punct=1.0, drop_punct_rate=1.0,
        p_typo=0.0, p_truncate=0.0,
    )
    for seed in range(20):
        rec = _neg_record()
        out = apply_noise(rec, cfg, random.Random(seed))
        # Both cues must still be present verbatim in the noisy text.
        assert "does not contain" in out.text
        assert ", only" in out.text


def test_noise_never_typos_inside_preserve_spans():
    cfg = NoiseConfig(
        p_lowercase=0.0, p_uppercase=0.0, p_ocr=0.0, p_drop_punct=0.0,
        p_typo=1.0, typo_rate=1.0,
        p_truncate=0.0,
    )
    for seed in range(20):
        rec = _neg_record()
        out = apply_noise(rec, cfg, random.Random(seed))
        assert "does not contain" in out.text
        assert ", only" in out.text


def test_noise_preserves_polarity_on_entities_after_lowercasing():
    cfg = NoiseConfig(
        p_lowercase=1.0, p_drop_punct=0.0,
        p_typo=0.0, p_truncate=0.0,
    )
    rec = _neg_record()
    out = apply_noise(rec, cfg, random.Random(0))
    neg = [e for e in out.entities if e.polarity == "NEG"]
    assert len(neg) == 1
    assert neg[0].text == "wood"


def test_preserve_spans_shift_correctly_after_upstream_deletions():
    """If an earlier deletion shrinks the string, preserve_spans must still
    align with their cue text in the new buffer."""
    cfg = NoiseConfig(
        p_lowercase=0.0, p_uppercase=0.0, p_ocr=0.0,
        p_drop_punct=1.0, drop_punct_rate=1.0,
        p_typo=0.0, p_truncate=0.0,
    )
    for seed in range(20):
        rec = _neg_record()
        out = apply_noise(rec, cfg, random.Random(seed))
        spans = out.meta.get("preserve_spans", [])
        for a, b in spans:
            # Each shifted range still points at one of the original cues.
            assert out.text[a:b] in ("does not contain", ", only")


def test_uppercase_offsets_unchanged_and_surface_resliced():
    cfg = NoiseConfig(
        p_lowercase=0.0, p_uppercase=1.0, p_drop_punct=0.0,
        p_typo=0.0, p_ocr=0.0, p_truncate=0.0,
    )
    rec = _record()
    out = apply_noise(rec, cfg, random.Random(0))
    out.validate()
    assert out.text == rec.text.upper()
    for ent, orig in zip(out.entities, rec.entities):
        assert (ent.start, ent.end) == (orig.start, orig.end)
        assert ent.text == orig.text.upper()
        assert out.text[ent.start:ent.end] == ent.text


def test_casing_draw_is_exclusive():
    """p_lowercase=1.0 means lowercase always wins; uppercase never fires."""
    cfg = NoiseConfig(
        p_lowercase=1.0, p_uppercase=1.0, p_drop_punct=0.0,
        p_typo=0.0, p_ocr=0.0, p_truncate=0.0,
    )
    for seed in range(10):
        out = apply_noise(_record(), cfg, random.Random(seed))
        assert out.text == out.text.lower()


def test_uppercase_skips_length_changing_chars():
    text = "große lieferung of refined copper cathode"
    ents = [Entity("COMMODITY", "refined copper cathode", 19, 41)]
    rec = Record(text=text, entities=ents)
    rec.validate()
    cfg = NoiseConfig(
        p_lowercase=0.0, p_uppercase=1.0, p_drop_punct=0.0,
        p_typo=0.0, p_ocr=0.0, p_truncate=0.0,
    )
    out = apply_noise(rec, cfg, random.Random(0))
    out.validate()
    assert len(out.text) == len(text)
    assert "ß" in out.text  # ß kept, not expanded to SS
    assert out.entities[0].text == "REFINED COPPER CATHODE"


def test_ocr_confusions_fire_outside_entities_only():
    text = "PO 10500 for refined copper cathode lot 18"
    ents = [Entity("COMMODITY", "refined copper cathode", 13, 35)]
    rec = Record(text=text, entities=ents)
    rec.validate()
    cfg = NoiseConfig(
        p_lowercase=0.0, p_uppercase=0.0, p_drop_punct=0.0,
        p_typo=0.0, p_ocr=1.0, ocr_rate=1.0, p_truncate=0.0,
    )
    out = apply_noise(rec, cfg, random.Random(0))
    out.validate()
    # Entity surface untouched; confusable chars outside it all swapped.
    assert out.entities[0].text == "refined copper cathode"
    assert "10500" not in out.text  # 1/0/5 all confused
    assert len(out.text) == len(text)


def test_ocr_never_fires_inside_preserve_spans():
    text = "no 105 wood here"
    ents = [Entity("COMMODITY", "wood", 7, 11, polarity="NEG")]
    rec = Record(text=text, entities=ents, meta={"preserve_spans": [(0, 6)]})
    rec.validate()
    cfg = NoiseConfig(
        p_lowercase=0.0, p_uppercase=0.0, p_drop_punct=0.0,
        p_typo=0.0, p_ocr=1.0, ocr_rate=1.0, p_truncate=0.0,
    )
    out = apply_noise(rec, cfg, random.Random(0))
    assert out.text.startswith("no 105")  # preserve span covers "no 105"


def test_new_noise_is_deterministic_under_seed():
    cfg = NoiseConfig()
    a = apply_noise(_record(), cfg, random.Random(42))
    b = apply_noise(_record(), cfg, random.Random(42))
    assert a.text == b.text
    assert [(e.start, e.end, e.text) for e in a.entities] == [
        (e.start, e.end, e.text) for e in b.entities
    ]
