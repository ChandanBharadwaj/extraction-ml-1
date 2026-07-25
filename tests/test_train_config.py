"""Torch-free unit tests for the training-side knobs (the Trainer itself
needs the [train] extra; these cover the pure-Python pieces)."""
from __future__ import annotations

import pytest

from ner.constants import LABEL2ID, NUM_LABELS
from ner.train.train import NEG_BOOST_WEIGHT, TrainConfig, class_weight_values


def test_class_weights_none_is_unweighted():
    assert class_weight_values("none") is None


def test_neg_boost_weights_only_the_neg_commodity_labels():
    w = class_weight_values("neg_boost")
    assert len(w) == NUM_LABELS
    boosted = {LABEL2ID["B-NEG_COMMODITY"], LABEL2ID["I-NEG_COMMODITY"]}
    for lid, value in enumerate(w):
        if lid in boosted:
            assert value == NEG_BOOST_WEIGHT
        else:
            assert value == 1.0


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError):
        class_weight_values("bogus")


def test_train_config_defaults_keep_legacy_behavior():
    cfg = TrainConfig(train_jsonl="x", output_dir="y")
    assert cfg.metric_for_best_model == "f1"
    assert cfg.class_weights == "none"
    assert cfg.gold_split == "all"
