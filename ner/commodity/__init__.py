"""Open-vocabulary COMMODITY extraction.

This package is deliberately independent of the repo's slot-fill pools. The goal
is to extract *any* commodity from *any* text, so nothing here may depend on a
hand-maintained list of commodity strings — a model or rule fitted to such a list
memorizes it instead of generalizing. See the module docstrings for the specific
measurements that motivated the split.

The two halves compose:

* a **span extractor** (open-vocabulary, model-backed) answers "where is a
  commodity mentioned?"
* a **polarity tagger** (rule-based, vocabulary-independent) answers "is that
  mention asserted or denied?"

Any object satisfying `CommodityExtractor` can be scored by `ner.commodity.bench`
against any gold set, so candidates are compared on identical footing.
"""
from ner.commodity.negation import PolarityConfig, PolarityTagger
from ner.commodity.protocol import CommodityExtractor, CommodityFilter

__all__ = [
    "CommodityExtractor",
    "CommodityFilter",
    "PolarityConfig",
    "PolarityTagger",
]
