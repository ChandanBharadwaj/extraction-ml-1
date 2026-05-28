"""Lightweight checks that NERRuntime's preprocess wiring behaves correctly
at the edges, without standing up a real ONNX session.

The full preprocess <-> project_entity roundtrip is covered in
`test_preprocess.py`; here we only assert that the runtime uses those pieces
in the right places.
"""
from __future__ import annotations

import numpy as np

from ner.constants import LABEL2ID, NUM_LABELS
from ner.infer.runtime import NERRuntime, NERRuntimeConfig
from ner.preprocess import Preprocessor


def _fake_runtime() -> NERRuntime:
    """Build a runtime without invoking __init__ (no ORT session needed)."""
    rt = NERRuntime.__new__(NERRuntime)
    rt.config = NERRuntimeConfig(onnx_path="", tokenizer_dir="")
    rt._session = None
    rt._tokenizer = None
    rt.thresholds = None
    rt.preprocessor = Preprocessor()
    return rt


def test_predict_returns_empty_when_input_cleans_to_empty_string():
    """Pure whitespace + zero-width input should short-circuit before the
    model is ever called — useful both for correctness and as a denial-of-
    service guard against payloads designed to thrash the tokenizer."""
    rt = _fake_runtime()
    assert rt.predict("   ​​   ") == []


def test_predict_batch_handles_all_empty_inputs():
    """If every batch item cleans to empty, no session.run is invoked and
    each output slot is an empty list."""
    rt = _fake_runtime()
    outs = rt.predict_batch(["   ", "​​", ""])
    assert outs == [[], [], []]


def test_predict_batch_preserves_input_order_when_some_inputs_empty():
    """The output list must have the same length and ordering as the input
    list even when interior items are dropped because they cleaned to empty.
    We stub the session/tokenizer to skip the network so we can exercise
    the batching plumbing only."""
    rt = _fake_runtime()

    class _AllOSession:
        def run(self, _out_names, feed):
            ids = feed["input_ids"]
            logits = np.zeros((ids.shape[0], ids.shape[1], NUM_LABELS), dtype=np.float32)
            logits[..., LABEL2ID["O"]] = 1.0
            return [logits]

    rt._session = _AllOSession()

    # Replace _encode so we don't depend on a real tokenizer.
    def fake_encode(text: str):
        return (
            np.zeros((1, 3), dtype=np.int64),
            np.ones((1, 3), dtype=np.int64),
            [(0, 0), (0, len(text)), (0, 0)],  # CLS, one content token, SEP
        )

    rt._encode = fake_encode  # type: ignore[method-assign]

    outs = rt.predict_batch(["   ", "real text", "", "more text"])
    assert len(outs) == 4
    # All-O logits → no entities in any slot.
    assert outs == [[], [], [], []]
