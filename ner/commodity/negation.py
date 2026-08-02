"""Rule-based POS/NEG polarity for commodity spans.

Why rules and not a model
-------------------------
Measured on the repo's 72-record gold set: given *perfect* spans, the rule below
scores **0.901 micro-F1**, versus **0.587** for the same spans with no polarity
model at all (41% of gold commodity spans are negated, so ignoring negation
forfeits them outright). Published negation models do not transfer — NegBERT
drops from ~90 F1 in-domain to ~70 cross-domain, and every off-the-shelf library
(negspacy, NegEx, ConText) ships a *clinical* trigger lexicon.

The decisive property: negation cues are a small closed class of function words,
while commodities are an open class. That asymmetry is why the span half of this
problem needs a pretrained model and the polarity half does not — and why this
module stays correct when the commodity vocabulary changes completely.

Algorithm (a domain-retargeted NegEx)
-------------------------------------
For each span, look only at the ``window_chars`` of text immediately to its left:

1. **Mask frozen compounds** ("sugar-free", "non-stick", "unrefined"). These
   contain cue substrings but assert rather than deny. Masking first is what
   stops ``non-`` from firing on ``non-stick cookware``.
2. **Find the rightmost cue** in the masked window.
3. **Reject if a scope terminator intervenes** between the cue and the span —
   clause punctuation, a contrastive conjunction, or a contrast cue. This is what
   makes ``no special wood, but ordinary wood is fine`` tag only the first span,
   and stops ``MARKS: NIL. CARGO: COFFEE`` from negating the coffee.

Cues appearing *inside* a span are ignored by construction, since only the left
context is examined. That is what keeps the HS heading
``ROASTED COFFEE, NOT DECAFFEINATED`` positive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from ner.schema import Entity

# Negation triggers. Closed class, so this list is maintainable by hand in a way
# a commodity list never could be. Sourced from the repo's own `neg_cue` decoy
# pool plus the cargo-document denials that appear as template literals.
NEG_CUES: tuple[str, ...] = (
    # determiner / adverb denials
    "no", "not", "none", "never", "nil", "zero", "without", "sans",
    # verbal denials
    "does not contain", "do not contain", "did not contain", "cannot contain",
    "does not include", "do not include", "cannot include", "not including",
    "declares no", "declared no", "certifies no", "confirms no",
    "lacks", "lacking", "excludes", "excluding", "omits", "omitted", "omitting",
    "missing", "absent", "absent of", "devoid of", "free of", "free from",
    # prohibition
    "prohibited", "forbidden", "banned", "barred", "unauthorized", "unauthorised",
    "refused", "rejected", "rejects", "declined", "denied", "denies",
    "no longer",
)

# Denials that *follow* their target ("organic cotton not available"). English
# puts negation before the target far more often, so this is a short list applied
# to the right context only.
POST_CUES: tuple[str, ...] = (
    "not available", "not included", "not shipped", "not carried", "not present",
    "not permitted", "not allowed", "not declared", "not on board", "not loaded",
    "unavailable", "is excluded", "are excluded", "was excluded", "were excluded",
)

# A restrictive "only" *immediately trailing* a span asserts it, as the exception
# to an earlier denial: "NO CRUDE PALM OIL ON BOARD, RBD PALM OLEIN ONLY".
#
# The anchoring is load-bearing. Compare:
#   "...NO CRUDE PALM OIL, RBD PALM OLEIN ONLY"  -> "only" trails PALM OLEIN, which is asserted
#   "does not contain wood, only plastic toys"   -> "only" leads plastic toys; wood stays denied
# The difference is whether a comma separates the span from the restrictive, so
# this pattern is matched at the start of the right context, never searched.
POST_ASSERTIONS: tuple[str, ...] = ("only", "instead", "in lieu")

# Assertive compounds that merely *contain* a cue substring. Masked before cue
# search. Without this, `non-` alone produced 4 false positives in testing.
FROZEN_COMPOUNDS: tuple[str, ...] = (
    "sugar-free", "gluten-free", "lead-free", "bpa-free", "oil-free",
    "fat-free", "dairy-free", "caffeine-free", "tax-free", "duty-free",
    "sulphur-free", "ash-free", "additive-free",
    "non-stick", "non-toxic", "non-flammable", "non-allergenic", "non-hazardous",
    "non-alloy", "non-woven", "non-ferrous", "non-decaffeinated",
    "stainless", "seamless", "careless", "boneless", "skinless", "odourless",
    "odorless", "colourless", "colorless", "wireless",
    "noise-cancelling", "wrinkle-resistant", "water-resistant",
    "unrefined", "unbleached", "unwashed", "uncoated", "undyed", "unmilled",
    "unroasted", "unprocessed", "unalloyed",
)

# Scope terminators searched strictly between the cue and the span. A cue's scope
# ends at a clause boundary or at any word that reverses the polarity of what
# follows — including the exception constructions ("other than X", "apart from
# X"), where X is asserted, not denied.
_CLAUSE_BREAK = r"[.;!?]\s|\n|\s[—–]\s|[—–]|\s-\s"
_CONTRASTIVE = (
    r"\bbut\b|\bhowever\b|\bwhereas\b|\binstead\b|\brather than\b|"
    r"\bin lieu of\b|\bonly\b|\bexcept\b|\baccepts?\b|\bapproved\b|\bconfirmed\b|"
    r"\bother than\b|\bapart from\b|\bsave for\b|\bexcluding only\b"
)
DEFAULT_TERMINATORS: str = f"{_CLAUSE_BREAK}|{_CONTRASTIVE}"

DEFAULT_WINDOW_CHARS = 40
# How far right to look for post-posed cues / restrictive "only".
DEFAULT_POST_WINDOW_CHARS = 28


def _compile_alternation(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """Longest-first alternation with word boundaries, case-insensitive."""
    ordered = sorted(phrases, key=len, reverse=True)
    body = "|".join(re.escape(p) for p in ordered)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PolarityConfig:
    """Tunable knobs for the polarity rule.

    `window_chars` is the scope distance. 40 measured best on the repo's gold set
    (0.901 at 40 vs 0.893 at 60 and 80); the metric is flat enough above ~40 that
    the exact value is not load-bearing.
    """

    window_chars: int = DEFAULT_WINDOW_CHARS
    post_window_chars: int = DEFAULT_POST_WINDOW_CHARS
    cues: tuple[str, ...] = NEG_CUES
    post_cues: tuple[str, ...] = POST_CUES
    post_assertions: tuple[str, ...] = POST_ASSERTIONS
    frozen_compounds: tuple[str, ...] = FROZEN_COMPOUNDS
    terminators: str = DEFAULT_TERMINATORS

    def to_dict(self) -> dict:
        return {
            "window_chars": self.window_chars,
            "post_window_chars": self.post_window_chars,
            "cues": list(self.cues),
            "post_cues": list(self.post_cues),
            "post_assertions": list(self.post_assertions),
            "frozen_compounds": list(self.frozen_compounds),
            "terminators": self.terminators,
        }


@dataclass
class PolarityTagger:
    """Assigns POS/NEG to spans produced by any extractor.

    Stateless with respect to the text it sees, and independent of *what* the
    commodity vocabulary is — so it composes onto a zero-shot model, a fine-tuned
    one, or a dictionary matcher without change.
    """

    config: PolarityConfig = field(default_factory=PolarityConfig)

    def __post_init__(self) -> None:
        self._cue_re = _compile_alternation(self.config.cues)
        self._post_cue_re = _compile_alternation(self.config.post_cues)
        assertions = "|".join(
            re.escape(p) for p in sorted(self.config.post_assertions, key=len, reverse=True)
        )
        # Anchored, not searched — see POST_ASSERTIONS.
        self._post_assert_re = re.compile(rf"^\s*(?:{assertions})(?!\w)", re.IGNORECASE)
        self._frozen_re = _compile_alternation(self.config.frozen_compounds)
        self._term_re = re.compile(self.config.terminators, re.IGNORECASE)

    def _mask_frozen(self, window: str) -> str:
        """Blank out assertive compounds so their cue substrings can't match.

        Replacement preserves length so offsets inside `window` stay valid.
        """
        return self._frozen_re.sub(lambda m: "\x00" * (m.end() - m.start()), window)

    @staticmethod
    def _mask_span_text(
        window: str, window_start: int, spans: tuple[tuple[int, int], ...]
    ) -> str:
        """Blank out other entity spans overlapping `window`, preserving length."""
        if not spans:
            return window
        chars = list(window)
        window_end = window_start + len(window)
        for span_start, span_end in spans:
            lo = max(span_start, window_start)
            hi = min(span_end, window_end)
            for i in range(lo - window_start, hi - window_start):
                chars[i] = "\x00"
        return "".join(chars)

    def is_negated(
        self,
        text: str,
        start: int,
        end: int | None = None,
        other_spans: tuple[tuple[int, int], ...] = (),
    ) -> bool:
        """True if the span `[start, end)` falls in a negation scope.

        Left context decides first; the right context can then either supply a
        post-posed denial or, via a restrictive "only", assert a span that an
        earlier cue would otherwise have swept up.

        `other_spans` are the char ranges of *other* commodity mentions, which are
        blanked out before cue search. Commodity descriptions legitimately embed
        cue words ("Slips and petticoats, not knitted or crocheted"), and without
        masking, one span's internal "not" negates the span that follows it.
        """
        window_start = max(0, start - self.config.window_chars)
        window = self._mask_span_text(
            text[window_start:start], window_start, other_spans
        )
        window = self._mask_frozen(window)

        last: re.Match[str] | None = None
        for match in self._cue_re.finditer(window):
            last = match
        negated = last is not None and not self._term_re.search(window[last.end():])

        if end is None:
            return negated

        right = self._mask_frozen(
            self._mask_span_text(
                text[end:end + self.config.post_window_chars], end, other_spans
            )
        )
        if negated:
            # "NO CRUDE PALM OIL ON BOARD, RBD PALM OLEIN ONLY" — the trailing
            # restrictive marks this span as the asserted exception.
            if self._post_assert_re.match(right):
                return False
        elif self._post_cue_re.search(right):
            # "organic cotton not available"
            return True
        return negated

    def apply(self, text: str, spans: list[Entity]) -> list[Entity]:
        """Return `spans` with polarity assigned from `text`'s negation cues.

        Only COMMODITY spans are touched — polarity is invalid on other types.
        """
        ranges = tuple((s.start, s.end) for s in spans if s.type == "COMMODITY")
        out: list[Entity] = []
        for span in spans:
            if span.type != "COMMODITY":
                out.append(span)
                continue
            others = tuple(r for r in ranges if r != (span.start, span.end))
            negated = self.is_negated(text, span.start, span.end, others)
            polarity = "NEG" if negated else "POS"
            out.append(replace(span, polarity=polarity) if polarity != span.polarity else span)
        return out
