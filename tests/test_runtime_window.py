"""Sliding-window inference over long inputs, without a real ONNX session.

The stub tokenizer maps each character to one token (offsets (i, i+1)) with
CLS/SEP sentinels, and the stub session labels tokens purely by token id:
id 7 -> B-COMMODITY, id 8 -> I-COMMODITY, everything else -> O. That makes
entity placement fully controllable by the id sequence, so we can pin the
windowing/merge behavior exactly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from ner.constants import LABEL2ID, NUM_LABELS
from ner.infer.runtime import NERRuntime, NERRuntimeConfig
from ner.preprocess import Preprocessor

CLS, SEP, PLAIN, B_TOK, I_TOK = 101, 102, 5, 7, 8


def _fake_runtime(session, id_for_char=None, **config_kwargs) -> NERRuntime:
    rt = NERRuntime.__new__(NERRuntime)
    rt.config = NERRuntimeConfig(onnx_path="", tokenizer_dir="", **config_kwargs)
    rt._session = session
    rt._tokenizer = None
    rt.thresholds = None
    rt.preprocessor = Preprocessor()

    id_for_char = id_for_char or {}

    def fake_encode_full(text: str):
        ids = [CLS] + [id_for_char.get(i, PLAIN) for i in range(len(text))] + [SEP]
        offsets = [(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)]
        return ids, offsets

    rt._encode_full = fake_encode_full  # type: ignore[method-assign]
    return rt


class _IdMappedSession:
    """Labels each token by its id; records every run call."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []  # (batch, seq_len) per run

    def run(self, _out_names, feed):
        ids = feed["input_ids"]
        self.calls.append(ids.shape)
        logits = np.zeros((*ids.shape, NUM_LABELS), dtype=np.float32)
        logits[..., LABEL2ID["O"]] = 1.0
        logits[ids == B_TOK] = 0.0
        logits[ids == B_TOK, LABEL2ID["B-COMMODITY"]] = 5.0
        logits[ids == I_TOK] = 0.0
        logits[ids == I_TOK, LABEL2ID["I-COMMODITY"]] = 5.0
        return [logits]


def test_short_input_takes_single_pass_path():
    session = _IdMappedSession()
    text = "a" * 100
    rt = _fake_runtime(session, id_for_char={10: B_TOK, 11: I_TOK, 12: I_TOK})
    ents = rt.predict(text)
    # One session call, one row, full sequence (100 content + CLS + SEP).
    assert session.calls == [(1, 102)]
    assert len(ents) == 1
    assert (ents[0].start, ents[0].end, ents[0].type) == (10, 13, "COMMODITY")


def test_entity_past_one_window_is_recovered():
    session = _IdMappedSession()
    # 600 content tokens >> 256-token window; entity at chars 400-402.
    text = "a" * 600
    rt = _fake_runtime(session, id_for_char={400: B_TOK, 401: I_TOK, 402: I_TOK})
    ents = rt.predict(text)
    assert len(ents) == 1
    assert (ents[0].start, ents[0].end) == (400, 403)
    assert ents[0].type == "COMMODITY"
    # Windowed path: exactly one batched call over full-length windows.
    assert len(session.calls) == 1
    n_windows, seq = session.calls[0]
    assert seq == 256
    assert n_windows >= 3


def test_windows_carry_cls_and_sep():
    session = _IdMappedSession()
    text = "a" * 600
    rt = _fake_runtime(session, id_for_char={})

    captured = {}
    real_run = session.run

    def spy_run(out_names, feed):
        captured["ids"] = feed["input_ids"].copy()
        return real_run(out_names, feed)

    session.run = spy_run
    rt.predict(text)
    ids = captured["ids"]
    assert (ids[:, 0] == CLS).all()
    assert (ids[:, -1] == SEP).all()


def test_merge_prefers_interior_window():
    """A token in the overlap region takes logits from the window where it
    sits more interior. The edge-sensitive session labels a token as
    B-COMMODITY only when it sits in the first 8 positions of its window;
    every such token is deep inside some other window, so after interior
    merge no entity survives."""

    class _EdgeSensitiveSession:
        def run(self, _out_names, feed):
            ids = feed["input_ids"]
            logits = np.zeros((*ids.shape, NUM_LABELS), dtype=np.float32)
            logits[..., LABEL2ID["O"]] = 1.0
            # Positions 1..8 (after CLS) get a spurious B-COMMODITY.
            logits[:, 1:9, :] = 0.0
            logits[:, 1:9, LABEL2ID["B-COMMODITY"]] = 5.0
            return [logits]

    text = "a" * 600
    rt = _fake_runtime(_EdgeSensitiveSession())
    ents = rt.predict(text)
    # The very first window's leading positions are legitimately most
    # interior there (no earlier window), so only entities from the global
    # start may survive; nothing from later windows' leading edges may.
    assert all(e.start < 9 for e in ents)


def test_input_beyond_char_cap_warns_and_truncates():
    session = _IdMappedSession()
    rt = _fake_runtime(session, max_input_chars=50)
    with pytest.warns(UserWarning, match="truncated"):
        ents = rt.predict("a" * 80)
    assert ents == []


def test_short_input_does_not_warn():
    session = _IdMappedSession()
    rt = _fake_runtime(session, max_input_chars=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rt.predict("a" * 20)


def test_predict_batch_mixes_short_and_long_inputs():
    session = _IdMappedSession()
    rt = _fake_runtime(
        session,
        id_for_char={10: B_TOK, 400: B_TOK, 401: I_TOK},
    )
    outs = rt.predict_batch(["a" * 100, "a" * 600, ""])
    assert len(outs) == 3
    # Short text: entity at 10; long text: entity at 400-401 (chars past its
    # own length aren't in text 1, so id_for_char hits both texts by index).
    assert [(e.start, e.end) for e in outs[0]] == [(10, 11)]
    assert (400, 402) in [(e.start, e.end) for e in outs[1]]
    assert outs[2] == []
