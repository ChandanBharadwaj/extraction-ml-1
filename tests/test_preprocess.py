"""Preprocessor invariants.

For span-level NER, preprocessing is dangerous unless the position map is
right. These tests exercise the cases that matter:

  - identity on already-clean text
  - idempotency (clean(clean(x)) == clean(x))
  - zero-width / control / NBSP / collapse / trim behavior
  - bidirectional projection: cleaned span -> original span -> original surface
  - training-side `apply_to_record` preserves the substring invariant on
    every reprojected Entity, including across whitespace collapse inside
    the entity surface
  - meta['preserve_spans'] (negation cues) are reprojected too
  - save/load round-trip
"""
from __future__ import annotations

import json

import pytest

from ner.preprocess import (
    PreprocessConfig,
    Preprocessor,
    PreprocessResult,
)
from ner.schema import Entity, Record


def test_clean_is_identity_on_already_clean_text():
    p = Preprocessor()
    text = "Maria Gonzalez from Acme Trading Co."
    r = p.clean(text)
    assert r.text == text
    assert r.char_map == tuple(range(len(text)))
    assert r.original_length == len(text)


def test_clean_is_idempotent():
    p = Preprocessor()
    text = "  Hi   there ​ friend you  "
    once = p.clean(text)
    twice = p.clean(once.text)
    assert twice.text == once.text


def test_zero_width_chars_stripped():
    p = Preprocessor()
    text = "hello​world‍!"  # zero-width space + ZWJ
    r = p.clean(text)
    assert r.text == "helloworld!"
    # Surviving chars map back to their original positions.
    assert r.text[0] == text[r.char_map[0]]
    assert r.text[5] == text[r.char_map[5]]  # 'w' came from original index 6


def test_control_chars_stripped_but_tab_newline_preserved():
    p = Preprocessor()
    # \x01 = strippable; \t and \n are kept (and \t normalized to space).
    text = "a\x01b\tc\nd"
    r = p.clean(text)
    # \t becomes space; \n is preserved (whitespace but not in normalize map).
    # Then collapse_whitespace runs and trims.
    assert "\x01" not in r.text
    assert r.text.startswith("a")
    assert r.text.endswith("d")


def test_nbsp_becomes_plain_space():
    p = Preprocessor()
    text = "Acme Trading"
    r = p.clean(text)
    assert r.text == "Acme Trading"
    assert " " in r.text and " " not in r.text


def test_collapse_runs_of_whitespace():
    p = Preprocessor()
    text = "Acme    Trading\t\tCo."
    r = p.clean(text)
    assert r.text == "Acme Trading Co."
    # Position map: each cleaned char has a valid index into the original.
    for i, ch in enumerate(r.text):
        assert text[r.char_map[i]] == ch or (
            ch == " " and text[r.char_map[i]] in (" ", "\t")
        )


def test_strip_leading_trailing_whitespace():
    p = Preprocessor()
    text = "   hello world   "
    r = p.clean(text)
    assert r.text == "hello world"
    assert r.char_map[0] == 3  # the 'h' was at original index 3
    assert r.char_map[-1] == 13  # the 'd' was at original index 13


def test_project_span_back_to_original():
    p = Preprocessor()
    text = "  Acme Trading   Co.  "
    r = p.clean(text)
    # cleaned = "Acme Trading Co."
    assert r.text == "Acme Trading Co."
    # Find "Trading" in cleaned, project back, slice from original.
    a = r.text.index("Trading")
    b = a + len("Trading")
    orig_a, orig_b = r.project_span(a, b)
    # The original substring covering "Trading" (whitespace surrounding it
    # was collapsed; the original could include NBSP and extra spaces).
    assert "Trading" in text[orig_a:orig_b]


def test_project_span_rejects_empty_or_out_of_range():
    p = Preprocessor()
    r = p.clean("hello")
    with pytest.raises(ValueError):
        r.project_span(2, 2)
    with pytest.raises(IndexError):
        r.project_span(0, 100)


def test_project_entity_uses_original_text_for_surface():
    p = Preprocessor()
    original = "  Acme   Trading  "
    r = p.clean(original)
    # "Acme Trading" sits at cleaned positions 0..12
    cleaned_ent = Entity("ORG", r.text[0:12], 0, 12)
    projected = r.project_entity(cleaned_ent, original)
    # Surface from original; substring invariant holds against original text.
    assert original[projected.start:projected.end] == projected.text


def test_apply_to_record_preserves_substring_invariant():
    p = Preprocessor()
    text = "  Maria  Gonzalez   from Acme Trading Co.  "
    # Entity offsets in the *raw* string.
    rec = Record(
        text=text,
        entities=[
            Entity("PERSON", text[2:17], 2, 17),  # "Maria  Gonzalez"
            Entity("ORG", text[25:41], 25, 41),    # "Acme Trading Co."
        ],
    )
    # The raw record violates the strict-surface rule slightly because of
    # the inter-name spaces, but it's a valid Record (text[s:e] == ent.text).
    rec.validate()
    cleaned = p.apply_to_record(rec)
    cleaned.validate()
    # Inside the entity, whitespace got collapsed.
    assert "Maria Gonzalez" in cleaned.text
    # No double-spaces survive anywhere.
    assert "  " not in cleaned.text
    # All cleaned entities still satisfy text[s:e] == ent.text.
    for e in cleaned.entities:
        assert cleaned.text[e.start:e.end] == e.text


def test_apply_to_record_reprojects_preserve_spans():
    p = Preprocessor()
    text = "  Manifest does not contain  wood  "
    rec = Record(
        text=text,
        entities=[Entity("COMMODITY", "wood", 29, 33, polarity="NEG")],
        meta={"preserve_spans": [(11, 27)]},  # "does not contain"
    )
    rec.validate()
    cleaned = p.apply_to_record(rec)
    cleaned.validate()
    spans = cleaned.meta["preserve_spans"]
    assert len(spans) == 1
    a, b = spans[0]
    # The cleaned preserve span still covers the cue surface.
    assert cleaned.text[a:b] == "does not contain"


def test_apply_to_record_drops_record_when_text_is_all_whitespace():
    p = Preprocessor()
    rec = Record(text="     \t   ")
    out = p.apply_to_record(rec)
    # Nothing survived.
    assert out.text == ""
    assert out.entities == []


def test_save_and_load_round_trip(tmp_path):
    cfg = PreprocessConfig(
        strip_zero_width=True,
        strip_control_chars=False,
        normalize_whitespace_chars=True,
        collapse_whitespace=False,
        strip_leading_trailing=True,
        nfc_normalize=False,
    )
    p = Preprocessor(cfg)
    path = tmp_path / "preprocess.json"
    p.save(path)
    loaded = Preprocessor.load(path)
    assert loaded.config == cfg
    # JSON is readable and contains the expected keys.
    d = json.loads(path.read_text())
    assert "strip_zero_width" in d and "collapse_whitespace" in d


def test_from_artifact_dir_falls_back_to_defaults(tmp_path):
    p = Preprocessor.from_artifact_dir(tmp_path)
    # No preprocess.json present -> default config.
    assert p.config == PreprocessConfig()


def test_disabled_config_is_full_noop():
    cfg = PreprocessConfig(
        strip_zero_width=False,
        strip_control_chars=False,
        normalize_whitespace_chars=False,
        collapse_whitespace=False,
        strip_leading_trailing=False,
        nfc_normalize=False,
    )
    p = Preprocessor(cfg)
    text = "  Hello​   world !  "
    r = p.clean(text)
    assert r.text == text
    assert r.char_map == tuple(range(len(text)))


def test_collapse_inside_entity_surface_is_recorded_in_entity_text():
    """When collapse touches whitespace inside an entity, the cleaned Entity's
    text is the collapsed form, not the original — that's the contract the
    runtime relies on when projecting back."""
    p = Preprocessor()
    text = "Acme    Trading Co. ordered"
    rec = Record(
        text=text,
        entities=[Entity("ORG", text[0:19], 0, 19)],  # "Acme    Trading Co."
    )
    rec.validate()
    cleaned = p.apply_to_record(rec)
    assert cleaned.entities[0].text == "Acme Trading Co."
    assert cleaned.text[cleaned.entities[0].start:cleaned.entities[0].end] == cleaned.entities[0].text
